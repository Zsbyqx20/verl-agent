# Copyright 2025 the ReverseReasoningGenerator (RRG) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Trajectory (macro) reward manager for RRG: generative answer-recovery recall.

GiGPO's episode/macro advantage is normalized from `token_level_rewards` (this manager's
output), while the step/micro advantage comes from the env's per-step margins. So we do NOT
want the macro reward to be the accumulated sum of step margins (the EpisodeRewardManager
default) -- we replace it with the frozen reader's GENERATIVE ANSWER-RECOVERY recall: the
blind 8B reader assembles the gold-schema JSON answer from goal + the policy's reasonings,
and recall in [0,1] (field-level match vs gold) is written to that trajectory's last response
token (broadcast across its step-rows so episode_norm_reward sees one scalar per row).

Why recall, not the completion judge: E1-E4 (rrg-reward-probe-*) showed the completion judge
SATURATES (no within-group gradient) and is even anti-correlated on answer tasks, while
generative answer-recovery de-saturates on two independent datasets. So this is the validated
trajectory reward for answer-bearing RRG.

Rows are grouped by `traj_uid`; within a trajectory they are ordered by step parsed from
`anchor_obs` ("task_id:step_idx"). The task_id is the part BEFORE the last colon (RRG keys
look like "12306-11", no internal colon), used to look up gold + schema from a table built at
init from the task_root. The goal is parsed from the decoded prompt (build_text_obs writes
"# Task goal\\n..."); the generated reasonings are the decoded responses.
"""
from __future__ import annotations

import difflib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_GOAL_RE = re.compile(r"# Task goal\s*\n(.*?)\n\s*\n", re.DOTALL)


def _per_step_repeat(texts: list, lookback: int) -> list:
    """repeat_t = max difflib-ratio of texts[t] against any of the `lookback` strictly earlier
    texts in the SAME sequence (catches A->B->A cycles, not just adjacent duplicates); 0.0 at
    t=0. MUST stay identical to `per_step_repeat` in src/repetition_penalty_probe.py -- that is
    the offline-validated computation this reward term reuses."""
    out = [0.0]
    for t in range(1, len(texts)):
        best = 0.0
        for s in range(max(0, t - lookback), t):
            r = difflib.SequenceMatcher(None, texts[s], texts[t]).ratio()
            if r > best:
                best = r
        out.append(best)
    return out


def _build_gold_table(task_root: str) -> dict:
    """task_id ('env-tid') -> {'gold': ..., 'schema': ..., 'goal': ...} for answer-bearing tasks."""
    from agent_system.environments.env_package.rrg import answer_recovery as A
    import json
    root = Path(task_root)
    gold_csv = A.auto_gold_csv(root)
    schema_cache = json.loads((root / "params_cache.json").read_text(encoding="utf-8"))
    raw = json.loads((root / "tasks.json").read_text(encoding="utf-8"))
    table = {}
    for t in raw:
        env, tid = t["env_id"], t.get("task_id")
        key = f"{env}-{tid}"
        row = A._csv_row(gold_csv, env, tid)
        goal, _ = A.resolve_goal_and_lang(row, t.get("task"))
        gold = A.load_gold(gold_csv, env, tid)
        schema = schema_cache.get(f"{env}:{tid}")
        if goal and gold and schema:
            table[key] = {"gold": gold, "schema": schema, "goal": goal}
    return table


class RRGTrajectoryRewardManager:
    def __init__(self, tokenizer, num_examine, reader_url, reader_model,
                 concurrency: int = 64, data_kind: str = "amex",
                 train_task_root=None, val_task_root=None, is_val: bool = False,
                 answer_max_tokens: int = 2048, reader_key: str = "sk-dummy",
                 traj_reward_shaping: str = "none", shaping_power: float = 2.0,
                 correct_bonus_lambda: float = 0.5, recall_threshold: float = 0.9,
                 threshold_bonus: float = 0.5, answer_step_credit: bool = False,
                 step_credit_w: float = 1.0, step_credit_mode: str = "first_appearance",
                 step_credit_combine: str = "add", max_prefixes: int = 8,
                 clamp_negative: bool = True, repetition_penalty: bool = False,
                 repetition_penalty_w: float = 1.0, repetition_lookback: int = 15,
                 repetition_threshold: float = 0.97, processor=None, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.data_kind = data_kind
        self.answer_max_tokens = answer_max_tokens
        self.traj_reward_shaping = traj_reward_shaping
        self.shaping_power = shaping_power
        self.correct_bonus_lambda = correct_bonus_lambda
        self.recall_threshold = recall_threshold
        self.threshold_bonus = threshold_bonus
        # per-step answer-field credit (GiGPO MICRO channel); off by default = byte-identical
        self.answer_step_credit = bool(answer_step_credit)
        self.step_credit_w = float(step_credit_w)
        self.step_credit_mode = step_credit_mode      # first_appearance | delta
        self.step_credit_combine = step_credit_combine  # add | replace
        self.max_prefixes = int(max_prefixes)
        self.clamp_negative = bool(clamp_negative)
        # repetition penalty (GiGPO MICRO channel); off by default = byte-identical
        self.repetition_penalty = bool(repetition_penalty)
        self.repetition_penalty_w = float(repetition_penalty_w)
        self.repetition_lookback = int(repetition_lookback)
        self.repetition_threshold = float(repetition_threshold)
        from agent_system.environments.env_package.rrg.reward_client import RRGRewardClient
        self.reward_client = RRGRewardClient(
            base_url=reader_url, model_name=reader_model, concurrency=concurrency,
            api_key=reader_key)
        self._processor = processor
        self._self_judge_wg = None

        # Answer-bearing RRG: build the gold/schema table for the relevant split.
        self.gold_table = {}
        if data_kind == "rrg":
            root = val_task_root if (is_val and val_task_root) else train_task_root
            if not root:
                raise ValueError("rrg reward_manager: data_kind=rrg needs train_task_root/val_task_root")
            self.gold_table = _build_gold_table(root)

    def set_self_judge_wg(self, actor_rollout_wg):
        """Post-init injection: replace the HTTP reward client with a SelfJudgeClient
        that uses the policy's own vLLM engine for trajectory answer-recovery."""
        self._self_judge_wg = actor_rollout_wg
        from agent_system.environments.env_package.rrg.self_judge_client import SelfJudgeClient
        self.reward_client = SelfJudgeClient(
            tokenizer=self.tokenizer, processor=self._processor,
            actor_rollout_wg=actor_rollout_wg,
            config={"max_image_long": 768, "num_distractors": 4, "seed": 0})

    def _shape(self, recall: float, correct: bool) -> float:
        """Map raw answer-recovery recall (+correct) to the GiGPO macro reward (#3).
        Pushes the policy toward COMPLETE answers, not just easy-field recall. Raw
        recall/correct are still logged separately as the research metric."""
        s = self.traj_reward_shaping
        if s == "square":
            return float(recall) ** self.shaping_power
        if s == "correct_bonus":
            return float(recall) + self.correct_bonus_lambda * (1.0 if correct else 0.0)
        if s == "threshold":
            return float(recall) + (self.threshold_bonus if recall >= self.recall_threshold else 0.0)
        return float(recall)  # "none"

    def _step_credits(self, pr) -> list:
        """Per-step answer-field credit c_t from a score_prefix_recovery result.
        `first_appearance`: each FINAL-recovered gold field credits the earliest step it
        appears; sum_t c_t == recall_N exactly (credit per field = recall_N / |final fields|).
        `delta`: c_t = recall_t - recall_{t-1} (optionally clamped to >=0)."""
        recalls = pr["recalls"]
        n = len(recalls)
        c = [0.0] * n
        if n == 0:
            return c
        if self.step_credit_mode == "delta":
            prev = 0.0
            for t in range(n):
                d = recalls[t] - prev
                prev = recalls[t]
                c[t] = max(d, 0.0) if self.clamp_negative else d
            return c
        # first_appearance (default)
        mpaths = pr["matched_paths"]
        final = mpaths[-1] if mpaths else set()
        rec_n = recalls[-1]
        if not final or rec_n <= 0:
            return c
        per = rec_n / len(final)
        for pth in final:
            ft = next((t for t in range(n) if pth in mpaths[t]), n - 1)
            c[ft] += per
        return c

    def _repetition_penalties(self, texts: list) -> tuple:
        """Per-step (penalty_t, repeat_t) for one trajectory's reasoning-text sequence.
        penalty_t = -w * clip((repeat_t - threshold) / (1 - threshold), 0, 1): a ramp, always
        <=0, ZERO below threshold. Uses the same bounded-lookback difflib computation validated
        in src/repetition_penalty_probe.py -- see _per_step_repeat above."""
        raws = _per_step_repeat(texts, self.repetition_lookback)
        span = 1.0 - self.repetition_threshold
        pens = []
        for r in raws:
            frac = (r - self.repetition_threshold) / span if span > 0 else (1.0 if r >= self.repetition_threshold else 0.0)
            frac = min(max(frac, 0.0), 1.0)
            pens.append(-self.repetition_penalty_w * frac)
        return pens, raws

    def __call__(self, data, return_dict=False):
        if "rm_scores" in data.batch.keys():
            return {"reward_tensor": data.batch["rm_scores"]} if return_dict else data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        traj_uid = data.non_tensor_batch["traj_uid"]
        anchor = data.non_tensor_batch.get("anchor_obs", None)
        prompt_len = data.batch["prompts"].shape[-1]

        rows = []
        groups: dict = defaultdict(list)
        for i in range(len(data)):
            item = data[i]
            am = item.batch["attention_mask"]
            vpl = int(am[:prompt_len].sum())
            vrl = int(am[prompt_len:].sum())
            prompt_ids = item.batch["prompts"][-vpl:] if vpl > 0 else item.batch["prompts"]
            resp_ids = item.batch["responses"][:vrl]
            step, task_id = 0, ""
            if anchor is not None:
                a = str(anchor[i])
                # anchor = "task_id:step_idx"; RRG task_id has no internal colon.
                head, _, tail = a.rpartition(":")
                try:
                    step = int(tail)
                    task_id = head
                except ValueError:
                    step, task_id = 0, a
            rows.append({
                "vrl": vrl,
                "prompt": self.tokenizer.decode(prompt_ids, skip_special_tokens=True),
                "resp": self.tokenizer.decode(resp_ids, skip_special_tokens=True).strip(),
                "step": step,
                "task_id": task_id,
            })
            groups[traj_uid[i]].append(i)

        items, group_order, metas = [], [], []
        for uid, idxs in groups.items():
            idxs_sorted = sorted(idxs, key=lambda i: rows[i]["step"])
            m = _GOAL_RE.search(rows[idxs_sorted[0]]["prompt"])
            goal = m.group(1).strip() if m else ""
            reasonings = [rows[i]["resp"] for i in idxs_sorted]
            task_id = rows[idxs_sorted[0]]["task_id"]
            entry = self.gold_table.get(task_id)
            items.append({"goal": goal or (entry["goal"] if entry else ""),
                          "reasonings": reasonings,
                          "gold": entry["gold"] if entry else None,
                          "schema": entry["schema"] if entry else None})
            group_order.append(idxs_sorted)
            metas.append({"task_id": task_id, "n_steps": len(reasonings)})

        # Score only trajectories that resolved a gold+schema; others get recall 0.
        scorable = [j for j, it in enumerate(items) if it["gold"] and it["schema"]]
        results = [{"recall": 0.0, "correct": False, "answer": None,
                    "parse_error": "no gold/schema"} for _ in items]
        prefix_results = [None] * len(items)  # per-item score_prefix_recovery dict (step-credit mode)
        if scorable:
            sc_items = [items[j] for j in scorable]
            if self.answer_step_credit:
                # prefix-wise recovery: gives BOTH the macro recall (final prefix, reused below
                # so no extra reader call) and the per-step matched-field sets for micro credit.
                scored = self.reward_client.score_prefix_recovery(
                    sc_items, max_tokens=self.answer_max_tokens, max_prefixes=self.max_prefixes)
                for j, r in zip(scorable, scored):
                    prefix_results[j] = r
                    results[j] = {"recall": r["recall"], "correct": r["correct"],
                                  "answer": r["answer"], "parse_error": r["parse_error"]}
            else:
                scored = self.reward_client.score_traj_recovery(
                    sc_items, max_tokens=self.answer_max_tokens)
                for j, r in zip(scorable, scored):
                    results[j] = r

        correct_flags = []
        # Per-ROW metric columns (length == len(data)); the trainer assigns reward_extra_info
        # straight into non_tensor_batch, so each column MUST match the batch row count. We
        # broadcast each trajectory's scalar recall/correct to all of its step-rows.
        row_recall = np.zeros(len(data), dtype=np.float32)
        row_correct = np.zeros(len(data), dtype=np.float32)
        # String columns for the generation dump: what the blind reader reconstructed
        # (answer_pred) vs the gold answer. Broadcast per-trajectory to its step-rows.
        # These let us eyeball reward hacking / leakage in the dumped JSONL (is recall
        # earned by inference, or by restating the GT action / fluent hallucination).
        row_gold = np.array([""] * len(data), dtype=object)
        row_pred = np.array([""] * len(data), dtype=object)
        # Per-step answer-field credit goes into the GiGPO MICRO channel by mutating the
        # rollout's per-step `rewards` IN PLACE (the trainer builds step_rewards from it AFTER
        # this manager runs -- see the reordered block in ray_trainer). Diagnostic columns are
        # broadcast per-trajectory to its step-rows like the recall columns.
        do_credit = self.answer_step_credit and ("rewards" in data.non_tensor_batch)
        # Repetition penalty needs no reader call / gold lookup (pure text vs the trajectory's
        # own earlier steps) so it applies to every trajectory, not just answer-bearing ones.
        do_rep = self.repetition_penalty and ("rewards" in data.non_tensor_batch)
        step_rewards_arr = data.non_tensor_batch["rewards"] if (do_credit or do_rep) else None
        row_step_credit = np.zeros(len(data), dtype=np.float32)
        row_reader_calls = np.zeros(len(data), dtype=np.float32)
        row_temporal = np.zeros(len(data), dtype=np.float32)
        row_repeat_penalty = np.zeros(len(data), dtype=np.float32)
        row_repeat_high = np.zeros(len(data), dtype=np.float32)
        for idxs_sorted, res, it, pr in zip(group_order, results, items, prefix_results):
            recall = float(res["recall"])
            # row_recall logs RAW recall (the research metric); the reward written to the
            # tensor is the SHAPED value (#3). With shaping=none they are identical.
            reward = self._shape(recall, bool(res["correct"]))
            correct_flags.append(bool(res["correct"]))
            try:
                gold_str = json.dumps(it["gold"], ensure_ascii=False) if it["gold"] else ""
            except Exception:
                gold_str = str(it["gold"])
            pred = res.get("answer")
            try:
                pred_str = json.dumps(pred, ensure_ascii=False) if isinstance(pred, (dict, list)) else ("" if pred is None else str(pred))
            except Exception:
                pred_str = str(pred)
            credits = self._step_credits(pr) if (do_credit and pr is not None) else None
            rep_pens = rep_raws = None
            if do_rep:
                rep_pens, rep_raws = self._repetition_penalties([rows[i]["resp"] for i in idxs_sorted])
            any_late = False
            for t, i in enumerate(idxs_sorted):
                row_recall[i] = recall
                row_correct[i] = 1.0 if res["correct"] else 0.0
                row_gold[i] = gold_str
                row_pred[i] = pred_str
                vrl = rows[i]["vrl"]
                if vrl > 0:
                    reward_tensor[i, vrl - 1] = reward
                if credits is not None:
                    ct = credits[t] if t < len(credits) else 0.0
                    row_step_credit[i] = ct
                    base = float(step_rewards_arr[i]) if self.step_credit_combine == "add" else 0.0
                    step_rewards_arr[i] = float(base + self.step_credit_w * ct)
                    if ct > 0 and t > 0:
                        any_late = True
                if rep_pens is not None:
                    # Always ADDITIVE on top of whatever the credit block above left in place --
                    # this is a malus/veto (like the existing restatement/leakage penalty), not a
                    # shaping choice governed by step_credit_combine (that knob is credit-only).
                    row_repeat_penalty[i] = rep_pens[t]
                    row_repeat_high[i] = 1.0 if rep_raws[t] >= self.repetition_threshold else 0.0
                    step_rewards_arr[i] = float(step_rewards_arr[i]) + rep_pens[t]
            if credits is not None:
                ncalls = float(pr.get("n_reader_calls", 0)) if pr else 0.0
                for i in idxs_sorted:
                    row_reader_calls[i] = ncalls
                    row_temporal[i] = 1.0 if any_late else 0.0

        if self.num_examine and items:
            k = min(self.num_examine, len(items))
            for j in range(k):
                print(f"[rrg-traj] task={metas[j]['task_id']!r} steps={metas[j]['n_steps']} "
                      f"| recall={results[j]['recall']:.3f} correct={results[j]['correct']} "
                      f"| goal={items[j]['goal'][:60]!r}"
                      + (f" | err={results[j]['parse_error']}" if results[j]['parse_error'] else ""))

        if return_dict:
            # Per-row columns so the trainer's non_tensor_batch.update keeps length == len(data).
            extra = {"answer_recall": row_recall,
                     "answer_correct": row_correct,
                     "answer_gold": row_gold,
                     "answer_pred": row_pred}
            if self.answer_step_credit:
                extra["answer_step_credit"] = row_step_credit
                extra["answer_reader_calls"] = row_reader_calls
                extra["answer_temporal"] = row_temporal
            if self.repetition_penalty:
                extra["repetition_penalty"] = row_repeat_penalty
                extra["repeat_high"] = row_repeat_high
            return {"reward_tensor": reward_tensor, "reward_extra_info": extra}
        return reward_tensor
