"""RRG Reward Manager — two-judge flow over trajectories and step groups.

After rollout, this manager computes three per-item reward signals:
    * rrg_fact_rewards:   per-step ``useful_binary / (1 + fact_tokens/token_scale)``,
                          where useful is attributed by the trajectory-end judge.
    * rrg_reason_rewards: per-step rank reward in [-1, 1] from a sibling-rank
                          judge over the n rollouts at each source step.
    * rrg_final_rewards:  per-trajectory binary can-conclude signal.

The advantage estimator reads these via ``reward_extra_info`` and composes
token-level advantages using ``rrg_fact_mask`` and ``rrg_reason_mask``.
"""
from __future__ import annotations

import base64
import functools
import json
import os
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any, Callable, TypeVar

import numpy as np
import torch
from PIL import Image

from verl import DataProto

try:
    from openai import OpenAI
    from pydantic import BaseModel
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]
    BaseModel = object  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Judge output schemas
# ---------------------------------------------------------------------------

class FinalJudgment(BaseModel):  # type: ignore[misc]
    reasoning: str
    can_conclude: bool
    useful_fact_indices: list[int]


class RankJudgment(BaseModel):  # type: ignore[misc]
    reasoning: str
    order: list[int]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

FINAL_JUDGE_PROMPT = """\
You are the final validator for a GUI agent reasoning system.

You are given:
- A task goal.
- The final GUI screenshot at the moment the agent submits a finish action.
- A numbered observation bank: facts the agent recorded across the trajectory.
- The finish action the agent is about to take (JSON).

Your job has two parts, both based ONLY on the screenshot + the observation bank:
1. can_conclude: can you conclude that the given finish action is the correct
   next step for this task? Respond true if the evidence (screenshot + bank)
   supports the finish action, false otherwise.
2. useful_fact_indices: list the observation-bank indices that were GENUINELY
   NECESSARY to reach this conclusion. Be parsimonious — exclude indices that
   are redundant with the screenshot or with each other. If no observation
   was necessary, return an empty list.

Respond with JSON: {"reasoning": "...", "can_conclude": <true|false>, "useful_fact_indices": [<int>, ...]}.
Only include indices that exist in the observation bank.\
"""

RANK_JUDGE_PROMPT = """\
You are a reasoning-quality ranker for a GUI agent system.

You are given:
- A task goal.
- The BEFORE screenshot: the GUI state at this step, in which the agent must
  choose its next action. This is the SOLE source of evidence for whether a
  candidate reasoning is correct.
- The AFTER screenshot (when provided as a second image): the GUI state
  immediately after the ground-truth action is executed. Provided for
  fact-checking the action's effect ONLY — never as a justification source.
- The ground-truth next action (JSON).
- N candidate reasonings, each labeled by its 0-based index. Each reasoning
  is meant to explain why the ground-truth action — and not a plausible
  alternative — follows from the BEFORE screenshot.

Rank the candidates best → worst by how clearly and correctly each reasoning
explains the ground-truth action **using only what is visible in the BEFORE
screenshot**. The agent who wrote each reasoning had access to BEFORE only;
hindsight description of the AFTER state is cheating, not insight.

Use the AFTER screenshot only to:
  • Resolve ambiguous action targets (e.g., what a click coordinate hits).
  • Detect candidates whose claimed effect contradicts what actually
    happened (penalize these — they made an unfaithful prediction).

Do NOT reward candidates merely for matching the AFTER screenshot's
appearance. A reasoning that simply narrates the AFTER state without
defensible BEFORE-grounded justification ranks LOWER than one that grounds
its claim in BEFORE evidence.

Favor: BEFORE-grounded specificity, distinguishing right from plausible
wrong, predictions consistent with AFTER (when available).
Penalize: vagueness, unfaithful claims, hindsight description, restatements,
predictions contradicted by AFTER.

Respond with JSON: {"reasoning": "...", "order": [<best_idx>, ..., <worst_idx>]}.
``order`` MUST be a permutation of 0..N-1. Ties are not allowed — break ties by
BEFORE-grounded concreteness.\
"""


# ---------------------------------------------------------------------------
# LLM client plumbing
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def _get_client(base_url: str | None) -> Any:
    client = getattr(_thread_local, "client", None)
    if client is None:
        kwargs: dict[str, Any] = {}
        if base_url:
            kwargs["base_url"] = base_url
        kwargs["api_key"] = os.getenv("OPENAI_API_KEY", "dummy")
        client = OpenAI(**kwargs)  # type: ignore[operator]
        _thread_local.client = client
    return client


@functools.lru_cache(maxsize=None)
def _encode_image(image_path: str) -> str:
    with Image.open(image_path) as img:
        buf = BytesIO()
        fmt = img.format or "png"
        img.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{b64}"


_T = TypeVar("_T")


def _parse_json_response(content: str, schema: type[Any]) -> Any:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model output is not valid JSON: {exc}") from exc
    return schema.model_validate(payload)


def _llm_call_with_retry(fn: Callable[[], _T], max_retries: int) -> _T:
    delay = 1.0
    last_exc: BaseException = RuntimeError("max_retries must be >= 1")
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries:
                raise
            time.sleep(delay + random.uniform(0.0, 0.5))
            delay *= 2
    raise last_exc


def call_final_judge(
    task: str,
    finish_action: str,
    fact_bank: list[str],
    image_path: str,
    model: str,
    base_url: str | None,
    max_retries: int,
) -> FinalJudgment:
    bank_text = (
        "empty"
        if not fact_bank
        else "\n".join(f"{i}: {obs}" for i, obs in enumerate(fact_bank))
    )
    user_text = (
        f"Task Goal: {task}\n"
        f"Finish Action: {finish_action}\n"
        f"Observation Bank:\n{bank_text}"
    )

    def _call():
        client = _get_client(base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": FINAL_JUDGE_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": _encode_image(image_path)}},
                ]},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("FinalJudgment completion output is empty")
        return _parse_json_response(content, FinalJudgment)

    return _llm_call_with_retry(_call, max_retries)


def call_rank_judge(
    task: str,
    image_path: str,
    next_image_path: str | None,
    ground_truth_action: str,
    reasonings: list[str],
    model: str,
    base_url: str | None,
    max_retries: int,
) -> RankJudgment:
    labeled = "\n\n".join(f"[{i}]\n{text}" for i, text in enumerate(reasonings))
    has_after = bool(next_image_path) and os.path.isfile(next_image_path)
    image_note = (
        "Image 1 = BEFORE (the only correctness source). Image 2 = AFTER "
        "(action-effect fact-check only; never a justification source)."
        if has_after
        else "Image 1 = BEFORE. No AFTER screenshot is available for this step."
    )
    user_text = (
        f"Task Goal: {task}\n"
        f"Ground Truth Next Action: {ground_truth_action}\n"
        f"{image_note}\n"
        f"Candidate reasonings (N={len(reasonings)}):\n\n{labeled}"
    )

    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": user_text},
        {"type": "image_url", "image_url": {"url": _encode_image(image_path)}},
    ]
    if has_after:
        user_content.append(
            {"type": "image_url", "image_url": {"url": _encode_image(next_image_path)}}  # type: ignore[arg-type]
        )

    def _call():
        client = _get_client(base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": RANK_JUDGE_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("RankJudgment completion output is empty")
        return _parse_json_response(content, RankJudgment)

    return _llm_call_with_retry(_call, max_retries)


# ---------------------------------------------------------------------------
# Main Reward Manager
# ---------------------------------------------------------------------------

class RRGRewardManager:
    """Two-judge reward manager for RRG v2."""

    def __init__(
        self,
        tokenizer,
        num_examine: int = 0,
        config=None,
        envs=None,
        normalize_by_length: bool = False,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.envs = envs
        cfg = config or {}

        self.token_scale = float(cfg.get("token_scale", 32.0))
        # Penalty applied to written-but-not-useful fact tokens. Breaks the
        # zero-for-zero degeneracy where "write nothing" and "write useless
        # facts" both yield r_fact = 0; with this knob, useless writes are
        # strictly worse than empty writes.
        self.alpha_fact_penalty = float(cfg.get("alpha_fact_penalty", 0.2))
        # Absolute reward for silent steps (no facts written) inside
        # trajectories where J_final returns can_conclude=True. Without this,
        # correct restraint only shows up relative to noisier peers; with it,
        # the silent policy gets a small positive signal so r_final still has
        # somewhere to land via Ã_fact group-norm even when fact_mask is empty.
        self.silent_step_bonus = float(cfg.get("silent_step_bonus", 0.1))
        # Linear length penalty subtracted from r_reason: penalty grows with
        # reason_tokens / response_length, regardless of rank. Caps unbounded
        # reasoning growth that the rank judge alone can't suppress.
        self.reason_length_penalty = float(cfg.get("reason_length_penalty", 0.3))
        self.final_judge_model = cfg.get("final_judge_model", "doubao-seed-2-0-pro-260215")
        self.rank_judge_model = cfg.get("rank_judge_model", self.final_judge_model)
        self.judge_base_url = cfg.get("judge_base_url", None)
        self.max_retries = int(cfg.get("max_retries", 3))
        self.max_judge_workers = int(cfg.get("max_judge_workers", 16))

        self.debug_log = bool(cfg.get("debug_log", False))
        self.debug_log_samples = int(cfg.get("debug_log_samples", 3))
        self.debug_log_file = cfg.get("debug_log_file", None)
        self.log_judge_errors = bool(cfg.get("log_judge_errors", True))
        self.max_judge_error_logs = int(cfg.get("max_judge_error_logs", 20))
        self._judge_error_logs = 0

        self._pool: ThreadPoolExecutor | None = None

    def _get_pool(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.max_judge_workers)
        return self._pool

    def _log_debug(self, event: str, payload: dict, file_overrides: dict | None = None):
        # ``payload`` is what gets printed (kept truncated for terminal
        # readability). ``file_overrides`` keys overwrite into the JSONL
        # payload so the on-disk record preserves full untruncated content
        # for downstream analysis.
        if not self.debug_log:
            return
        print_message = {"event": event, **payload}
        print(f"[RRG][reward] {json.dumps(print_message, ensure_ascii=False, default=str)}")
        if self.debug_log_file:
            file_message = {"source": "reward", "event": event, **payload, **(file_overrides or {})}
            with open(self.debug_log_file, "a") as f:
                f.write(json.dumps(file_message, ensure_ascii=False, default=str) + "\n")

    def _log_error(self, event: str, payload: dict):
        if not self.log_judge_errors:
            return
        if self._judge_error_logs >= self.max_judge_error_logs:
            return
        self._judge_error_logs += 1
        message = {"event": event, **payload}
        print(f"[RRG][reward][error] {json.dumps(message, ensure_ascii=False, default=str)}")
        if self.debug_log_file:
            with open(self.debug_log_file, "a") as f:
                f.write(json.dumps({"source": "reward_error", **message}, ensure_ascii=False, default=str) + "\n")

    def __call__(self, data: DataProto, return_dict: bool = False):
        batch_size = len(data)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        # Per-batch judge operational counters; published via batch.meta_info for metrics.
        judge_stats = {
            "final_calls": 0,
            "final_failures": 0,
            "rank_calls": 0,
            "rank_failures": 0,
        }

        fact_rewards = np.zeros(batch_size, dtype=np.float32)
        reason_rewards = np.zeros(batch_size, dtype=np.float32)
        final_rewards = np.zeros(batch_size, dtype=np.float32)
        fact_tokens_arr = np.zeros(batch_size, dtype=np.int64)
        reason_tokens_arr = np.zeros(batch_size, dtype=np.int64)
        uids = data.non_tensor_batch.get("uid", None)
        step_group_uids = np.array(
            [str(u) for u in (uids if uids is not None else np.arange(batch_size))],
            dtype=object,
        )

        step_metadata = getattr(self.envs, "step_metadata", None)
        if step_metadata is None:
            if return_dict:
                return {
                    "reward_tensor": reward_tensor,
                    "reward_extra_info": {
                        "rrg_fact_rewards": fact_rewards.tolist(),
                        "rrg_reason_rewards": reason_rewards.tolist(),
                        "rrg_final_rewards": final_rewards.tolist(),
                        "rrg_step_group_uid": step_group_uids.tolist(),
                    },
                }
            return reward_tensor

        traj_uids = data.non_tensor_batch.get("traj_uid", np.arange(batch_size))

        # Unique trajectories preserving order — gather_rollout_data keeps env-slot
        # ordering, so the k-th unique traj_uid corresponds to env slot k.
        seen: set = set()
        unique_trajs_ordered: list = []
        for uid in traj_uids:
            if uid not in seen:
                unique_trajs_ordered.append(uid)
                seen.add(uid)

        # Map env_slot -> batch indices for the trajectory, in step order.
        env_slot_to_batch_idx: dict[int, np.ndarray] = {}
        for env_slot, traj_uid in enumerate(unique_trajs_ordered):
            traj_indices = np.where(traj_uids == traj_uid)[0]
            env_slot_to_batch_idx[env_slot] = traj_indices

        # Global fact indexing per trajectory.
        # traj_fact_content[env_slot] = list[str] (indexed by global_idx in write order)
        # step_fact_idxs[env_slot][step_t] = list[int] of global indices written at step t
        traj_fact_content: dict[int, list[str]] = {}
        step_fact_idxs: dict[int, dict[int, list[int]]] = {}
        for env_slot, _ in enumerate(unique_trajs_ordered):
            traj_meta = step_metadata.get(str(env_slot), [])
            contents: list[str] = []
            step_map: dict[int, list[int]] = defaultdict(list)
            counter = 0
            for t, meta in enumerate(traj_meta):
                for upd in meta.get("writing_updates", []):
                    if upd.get("action") != "add":
                        continue
                    content = upd.get("observation", "")
                    if not content:
                        continue
                    contents.append(content)
                    step_map[t].append(counter)
                    counter += 1
            traj_fact_content[env_slot] = contents
            step_fact_idxs[env_slot] = step_map

        pool = self._get_pool()

        # ------------------------------------------------------------------ #
        # Phase F — Final judge per trajectory                                #
        # ------------------------------------------------------------------ #
        all_final_futures: dict[Future, int] = {}
        for env_slot, _ in enumerate(unique_trajs_ordered):
            traj_meta = step_metadata.get(str(env_slot), [])
            if not traj_meta:
                continue
            last_meta = traj_meta[-1]
            image_path = last_meta.get("screenshot_path", "")
            raw_action = last_meta.get("ground_truth_action", "")
            finish_action_str = raw_action if isinstance(raw_action, str) else json.dumps(raw_action)
            fact_bank = (
                last_meta.get("final_fact_bank")
                or traj_fact_content.get(env_slot, [])
            )

            if not image_path or not os.path.isfile(image_path):
                continue  # default: can_conclude=True, useful=all (handled below)

            fut = pool.submit(
                call_final_judge,
                last_meta.get("task", ""),
                finish_action_str,
                list(fact_bank),
                image_path,
                self.final_judge_model,
                self.judge_base_url,
                self.max_retries,
            )
            all_final_futures[fut] = env_slot
            judge_stats["final_calls"] += 1

        final_results: dict[int, tuple[bool, set[int]]] = {}
        for fut in as_completed(all_final_futures):
            env_slot = all_final_futures[fut]
            try:
                judgment = fut.result()
                useful_set = {int(i) for i in judgment.useful_fact_indices}
                final_results[env_slot] = (bool(judgment.can_conclude), useful_set)
                if self.debug_log:
                    self._log_debug("final_judge", {
                        "env_slot": env_slot,
                        "can_conclude": judgment.can_conclude,
                        "useful": sorted(useful_set),
                        "reasoning": judgment.reasoning[:300],
                    }, file_overrides={"reasoning": judgment.reasoning})
            except Exception as exc:
                # Conservative default: can_conclude=True, no useful facts (avoids false positives)
                final_results[env_slot] = (True, set())
                judge_stats["final_failures"] += 1
                self._log_error("final_judge_failed", {
                    "env_slot": env_slot,
                    "error": repr(exc),
                })

        # ------------------------------------------------------------------ #
        # Phase R — Rank judge per (uid, step_t)                              #
        # ------------------------------------------------------------------ #
        # Group env_slots by uid (sibling rollouts)
        uid_to_slots: dict[str, list[int]] = defaultdict(list)
        env_slot_to_uid: dict[int, str] = {}
        for env_slot, _ in enumerate(unique_trajs_ordered):
            traj_indices = env_slot_to_batch_idx[env_slot]
            if len(traj_indices) == 0 or uids is None:
                continue
            uid_str = str(uids[traj_indices[0]])
            uid_to_slots[uid_str].append(env_slot)
            env_slot_to_uid[env_slot] = uid_str

        all_rank_futures: dict[Future, tuple[str, int, list[int]]] = {}
        for uid_str, sibling_slots in uid_to_slots.items():
            if len(sibling_slots) < 2:
                continue
            max_t = max(
                len(step_metadata.get(str(s), [])) for s in sibling_slots
            )
            for t in range(max_t):
                reasonings: list[str] = []
                valid_slots: list[int] = []
                for s in sibling_slots:
                    traj_meta = step_metadata.get(str(s), [])
                    if t < len(traj_meta):
                        reasonings.append(traj_meta[t].get("reasoning_text", "") or "")
                        valid_slots.append(s)
                if len(valid_slots) < 2:
                    continue
                first_meta = step_metadata[str(valid_slots[0])][t]
                image_path = first_meta.get("screenshot_path", "")
                if not image_path or not os.path.isfile(image_path):
                    continue
                # AFTER-action screenshot. Empty/missing on the final step;
                # call_rank_judge degrades gracefully to single-image mode.
                next_image_path = first_meta.get("next_screenshot_path", "") or None
                fut = pool.submit(
                    call_rank_judge,
                    first_meta.get("task", ""),
                    image_path,
                    next_image_path,
                    first_meta.get("ground_truth_action", "") if isinstance(first_meta.get("ground_truth_action", ""), str) else json.dumps(first_meta.get("ground_truth_action", "")),
                    reasonings,
                    self.rank_judge_model,
                    self.judge_base_url,
                    self.max_retries,
                )
                all_rank_futures[fut] = (uid_str, t, valid_slots)
                judge_stats["rank_calls"] += 1

        # rank_by_slot_step[(env_slot, step_t)] = rank in [0, n-1]; 0 = best.
        rank_by_slot_step: dict[tuple[int, int], float] = {}
        for fut in as_completed(all_rank_futures):
            uid_str, t, valid_slots = all_rank_futures[fut]
            n = len(valid_slots)
            middle = (n - 1) / 2.0
            rank_by_local: dict[int, float] = {}
            try:
                judgment = fut.result()
                seen_local: set[int] = set()
                pos = 0
                for raw_idx in judgment.order:
                    try:
                        local_idx = int(raw_idx)
                    except (TypeError, ValueError):
                        continue
                    if local_idx < 0 or local_idx >= n or local_idx in seen_local:
                        continue
                    rank_by_local[local_idx] = float(pos)
                    seen_local.add(local_idx)
                    pos += 1
                # Any unranked siblings get the middle rank (neutral advantage).
                for local_idx in range(n):
                    rank_by_local.setdefault(local_idx, middle)
                if self.debug_log:
                    self._log_debug("rank_judge", {
                        "uid": uid_str,
                        "step": t,
                        "n": n,
                        "order": judgment.order,
                        "reasoning": judgment.reasoning[:200],
                    }, file_overrides={"reasoning": judgment.reasoning})
            except Exception as exc:
                for local_idx in range(n):
                    rank_by_local[local_idx] = middle
                judge_stats["rank_failures"] += 1
                self._log_error("rank_judge_failed", {
                    "uid": uid_str,
                    "step": t,
                    "error": repr(exc),
                })
            for local_idx, env_slot in enumerate(valid_slots):
                rank_by_slot_step[(env_slot, t)] = rank_by_local[local_idx]

        # ------------------------------------------------------------------ #
        # Final assembly — per-step scalar rewards into batch-indexed arrays #
        # ------------------------------------------------------------------ #
        fact_mask_batch = data.batch.get("rrg_fact_mask", None)
        reason_mask_batch = data.batch.get("rrg_reason_mask", None)
        response_length = int(data.batch["responses"].shape[-1])

        debug_examples: list[dict] = []
        for env_slot, _traj_uid in enumerate(unique_trajs_ordered):
            traj_meta = step_metadata.get(str(env_slot), [])
            if not traj_meta:
                continue
            traj_indices = env_slot_to_batch_idx[env_slot]
            if len(traj_indices) == 0:
                continue

            uid_of_traj = env_slot_to_uid.get(env_slot, "")
            n_siblings = len(uid_to_slots.get(uid_of_traj, [env_slot])) if uid_of_traj else 1

            # Final reward for this trajectory.
            if env_slot in final_results:
                can_conclude, useful_indices = final_results[env_slot]
                final_called = True
            else:
                can_conclude, useful_indices = True, set()
                final_called = False
            r_final = 1.0 if can_conclude else 0.0

            step_fact_idx_map = step_fact_idxs.get(env_slot, {})

            for local_t, batch_idx in enumerate(traj_indices):
                if local_t >= len(traj_meta):
                    break

                global_idxs_at_t = step_fact_idx_map.get(local_t, [])
                if global_idxs_at_t:
                    if final_called:
                        useful_at_t = any(g in useful_indices for g in global_idxs_at_t)
                    else:
                        # Judge didn't run — give the benefit of the doubt.
                        useful_at_t = True
                    useful_binary = 1.0 if useful_at_t else 0.0
                else:
                    useful_binary = 0.0

                if fact_mask_batch is not None:
                    fact_tokens = int(fact_mask_batch[batch_idx].sum().item())
                else:
                    fact_tokens = 0
                if reason_mask_batch is not None:
                    reason_tokens = int(reason_mask_batch[batch_idx].sum().item())
                else:
                    reason_tokens = 0

                # r_fact branches:
                #   silent + can_conclude → +silent_step_bonus (correct restraint)
                #   silent + not concluded → 0
                #   useful write → +1 / (1 + fact_tokens / token_scale)
                #   useless write → −alpha_fact_penalty · (fact_tokens / token_scale)
                num_facts_at_t = len(global_idxs_at_t)
                scale_ratio = fact_tokens / max(1e-6, self.token_scale)
                if num_facts_at_t == 0:
                    r_fact = self.silent_step_bonus if can_conclude else 0.0
                elif useful_binary > 0:
                    r_fact = 1.0 / (1.0 + scale_ratio)
                else:
                    r_fact = -self.alpha_fact_penalty * scale_ratio

                # r_reason: rank-based reward minus a flat length penalty so
                # long reasoning is bounded for ALL ranks (not just the best).
                rank = rank_by_slot_step.get((env_slot, local_t))
                if rank is None or n_siblings < 2:
                    rank_reward = 0.0
                else:
                    denom = max(1, n_siblings - 1)
                    rank_reward = (n_siblings - 2.0 * rank - 1.0) / denom
                length_penalty = self.reason_length_penalty * (reason_tokens / max(1, response_length))
                r_reason = rank_reward - length_penalty

                fact_rewards[batch_idx] = r_fact
                reason_rewards[batch_idx] = r_reason
                final_rewards[batch_idx] = r_final
                fact_tokens_arr[batch_idx] = fact_tokens
                reason_tokens_arr[batch_idx] = reason_tokens
                if uids is not None:
                    step_group_uids[batch_idx] = f"{uids[batch_idx]}_{local_t}"

                # Set scalar reward at the last valid token for metric compatibility.
                combined = r_fact + r_reason + r_final
                prompt_length = data.batch["prompts"][batch_idx].shape[-1]
                valid_response_length = data.batch["attention_mask"][batch_idx][prompt_length:].sum().item()
                if valid_response_length > 0:
                    reward_tensor[batch_idx, int(valid_response_length) - 1] = torch.tensor(
                        combined,
                        dtype=reward_tensor.dtype,
                        device=reward_tensor.device,
                    )

            if self.debug_log and len(debug_examples) < self.debug_log_samples:
                debug_steps = []
                for local_t in range(min(len(traj_meta), self.debug_log_samples)):
                    batch_idx = int(traj_indices[local_t]) if local_t < len(traj_indices) else -1
                    debug_steps.append({
                        "step": local_t,
                        "num_facts": len(step_fact_idx_map.get(local_t, [])),
                        "fact_tokens": int(fact_tokens_arr[batch_idx]) if batch_idx >= 0 else 0,
                        "fact_reward": float(fact_rewards[batch_idx]) if batch_idx >= 0 else 0.0,
                        "reason_reward": float(reason_rewards[batch_idx]) if batch_idx >= 0 else 0.0,
                    })
                debug_examples.append({
                    "env_slot": env_slot,
                    "n_siblings": n_siblings,
                    "can_conclude": can_conclude,
                    "useful_count": len(useful_indices),
                    "total_facts": len(traj_fact_content.get(env_slot, [])),
                    "r_final": r_final,
                    "steps": debug_steps,
                })

        # Publish judge operational counters for compute_rrg_metrics.
        if hasattr(data, "meta_info") and isinstance(data.meta_info, dict):
            data.meta_info["rrg_judge_stats"] = dict(judge_stats)

        self._log_debug("batch_reward_summary", {
            "batch_size": batch_size,
            "num_trajectories": len(unique_trajs_ordered),
            "fact_mean": float(fact_rewards.mean()) if batch_size else 0.0,
            "fact_nonzero": int(np.count_nonzero(fact_rewards)),
            "reason_mean": float(reason_rewards.mean()) if batch_size else 0.0,
            "reason_abs_mean": float(np.abs(reason_rewards).mean()) if batch_size else 0.0,
            "final_mean": float(final_rewards.mean()) if batch_size else 0.0,
            "fact_tokens_mean": float(fact_tokens_arr.mean()) if batch_size else 0.0,
            "judge_stats": dict(judge_stats),
            "examples": debug_examples,
        })

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {
                    "rrg_fact_rewards": fact_rewards.tolist(),
                    "rrg_reason_rewards": reason_rewards.tolist(),
                    "rrg_final_rewards": final_rewards.tolist(),
                    "rrg_step_group_uid": step_group_uids.tolist(),
                    "rrg_fact_tokens": fact_tokens_arr.tolist(),
                    "rrg_reason_tokens": reason_tokens_arr.tolist(),
                },
            }
        return reward_tensor
