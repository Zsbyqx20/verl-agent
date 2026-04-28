"""RRG Reward Manager — three-judge flow over trajectories and step groups.

After rollout, this manager computes three per-item reward signals:
    * rrg_fact_rewards:   per-step sum of per-fact rewards from a step-level
                          grounding judge (J_step) plus an additive crucial
                          bonus from the stage-1 trajectory-end judge.
                          Per-fact: ungrounded → -r_pen, grounded ∧ trivial → 0,
                          grounded ∧ meaningful → +r_step. Crucial → +r_bonus.
    * rrg_reason_rewards: per-step rank reward in [-1, 1] from a sibling-rank
                          judge over the n rollouts at each source step.
    * rrg_final_rewards:  per-trajectory binary success signal. Computed via a
                          two-stage flow: stage 1 derives a predicted answer
                          from action reasonings + fact bank + last screenshot
                          (and emits is_finished + crucial_fact_indices); for
                          tasks with a ground-truth return value, stage 2
                          semantically compares predicted vs ground-truth and
                          returns a single ``matches`` boolean. Final value is
                          1.0 iff (is_finished AND matches), else 0.0.

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


class Stage1FinalJudgment(BaseModel):  # type: ignore[misc]
    # ``predicted_answer`` mirrors the ANSWER_TEMPLATE shape (dict / list /
    # nested) when one is provided, or is ``None`` for tasks without a
    # ground-truth return value. Stage 2 only runs in the former case.
    reasoning: str
    is_finished: bool
    predicted_answer: Any = None
    crucial_fact_indices: list[int]


class CompareJudgment(BaseModel):  # type: ignore[misc]
    reasoning: str
    matches: bool


class RankJudgment(BaseModel):  # type: ignore[misc]
    reasoning: str
    order: list[int]


class FactQuality(BaseModel):  # type: ignore[misc]
    grounded: bool
    meaningful: bool


class StepJudgment(BaseModel):  # type: ignore[misc]
    reasoning: str
    fact_quality: list[FactQuality]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

STAGE1_FINAL_JUDGE_PROMPT = """\
You are the stage-1 final validator for a GUI agent reasoning system.

You are given:
- A task goal.
- The agent's per-step action reasonings across the trajectory (numbered, in
  chronological order).
- A numbered observation bank: facts the agent recorded while doing the task.
- The final GUI screenshot at the moment the agent submits a finish action.
- Optionally, an ANSWER_TEMPLATE describing the expected return-value
  structure with leaf values redacted as the literal string "<fill in>". When
  an ANSWER_TEMPLATE is provided you must produce a ``predicted_answer`` with
  the EXACT SAME STRUCTURE (same keys, same list lengths) where each leaf is
  replaced by the concrete value supported by the trajectory evidence. When
  no ANSWER_TEMPLATE is provided, return ``predicted_answer`` as null.

Produce three things:
1. is_finished: based on action reasonings + observation bank + final
   screenshot, has the task been GENUINELY completed? Respond true only if
   the trajectory evidence supports it; respond false if the agent gave up,
   stopped early, or the final state does not satisfy the task goal.
2. predicted_answer: see ANSWER_TEMPLATE rules above. When ANSWER_TEMPLATE is
   provided, fill in the leaf values from the trajectory evidence (action
   reasonings, fact bank, final screenshot) — do NOT invent values absent
   from those inputs. Preserve the template's nested structure exactly.
3. crucial_fact_indices: list the observation-bank indices whose information
   was GENUINELY LOAD-BEARING for is_finished — facts that, if removed,
   would make the conclusion unsupportable from the final screenshot alone.
   Be strict and parsimonious: if a fact is interesting but not load-bearing
   for the conclusion, exclude it. If the final screenshot alone supports
   the conclusion, return an empty list. This is a sparse bonus signal, not a
   quality rating — quality is judged elsewhere.

Respond with JSON: {"reasoning": "...", "is_finished": <true|false>, "predicted_answer": <object|null>, "crucial_fact_indices": [<int>, ...]}.
Only include indices that exist in the observation bank.\
"""

STAGE2_COMPARE_PROMPT = """\
You are an answer comparator for a GUI agent task. You are given two
candidate answers for the same task: GROUND_TRUTH (correct, taken from the
task data) and PREDICTED (the agent's stage-1 output). Decide whether
PREDICTED matches GROUND_TRUTH SEMANTICALLY — not by surface form.

Treat answers as MATCHING if they convey the same underlying information:
- Equivalent phrasings of text fields ("$10" vs "10 dollars", "NYC" vs
  "New York City").
- Trivial formatting differences (case, whitespace, punctuation, leading
  articles, currency symbols).
- For list-typed fields that are inherently set-like (e.g., a set of tags
  with no obvious ordering), reordering of elements is acceptable.

Treat answers as NOT MATCHING if any field carries different factual
content:
- Different numeric values (counts, prices, IDs).
- Different proper nouns (names, titles) that are not aliases.
- Missing items in lists where order or count is meaningful, or where the
  list represents a ranked/ordered output.
- Predicted contains placeholder strings (e.g. literal "<fill in>") or is
  empty when ground truth is non-empty.

Return JSON: {"reasoning": "<one short sentence>", "matches": <true|false>}.\
"""

STEP_JUDGE_PROMPT = """\
You are a per-step fact validator for a GUI agent annotation system.

You are given:
- A task goal.
- The BEFORE screenshot at this step (the GUI state the agent observed when
  writing these facts).
- The existing observation bank (numbered facts already recorded in earlier
  steps), shown for redundancy detection only.
- A list of NEW facts the agent wrote at this step. You must rate each new
  fact independently.

For each NEW fact, decide two booleans:
1. grounded: is the fact accurately supported by the BEFORE screenshot? A
   fact is grounded if a careful reader looking only at the screenshot would
   agree it is true. Speculation, hallucination, or claims about UI elements
   not visible in the screenshot are NOT grounded.
2. meaningful: is the fact a non-trivial, non-redundant observation that
   would help a downstream reader understand the GUI state? Mark FALSE if:
     - the fact is implied by, or duplicates, any item already in the
       existing observation bank;
     - the fact is one of several near-duplicate slices of the same
       observation (atomic-but-redundant splitting — keep at most one per
       distinct observation, mark the rest not meaningful);
     - the fact is overly generic ("there is a button on the screen", "the
       screen has UI elements"), tautological, or restates the task goal
       without adding new information.

Order MUST match the order of NEW facts as given (0, 1, 2, ...). Output
exactly one entry per new fact.

Respond with JSON: {"reasoning": "...", "fact_quality": [{"grounded": <bool>, "meaningful": <bool>}, ...]}.\
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


def _redact_values(obj: Any) -> Any:
    """Recursively replace leaf values with the placeholder ``"<fill in>"``.

    Preserves dict / list structure exactly; non-container values become the
    placeholder string. Used to derive a stage-1 ANSWER_TEMPLATE from a
    ground-truth answer so the judge sees the schema but not the values.
    """
    if isinstance(obj, dict):
        return {k: _redact_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact_values(v) for v in obj]
    return "<fill in>"


def call_stage1_final_judge(
    task: str,
    action_reasonings: list[str],
    fact_bank: list[str],
    image_path: str,
    answer_template: Any,
    model: str,
    base_url: str | None,
    max_retries: int,
) -> Stage1FinalJudgment:
    bank_text = "empty" if not fact_bank else "\n".join(f"{i}: {obs}" for i, obs in enumerate(fact_bank))
    reasoning_text = "empty" if not action_reasonings else "\n".join(f"{i}: {r}" for i, r in enumerate(action_reasonings))
    if answer_template is not None:
        template_text = json.dumps(answer_template, ensure_ascii=False, indent=2)
        template_block = (
            "\n\nANSWER_TEMPLATE (fill the redacted leaf values; preserve structure):\n"
            f"{template_text}"
        )
    else:
        template_block = "\n\nNo ANSWER_TEMPLATE — return predicted_answer as null."
    user_text = (
        f"Task Goal: {task}\n"
        f"Action Reasonings:\n{reasoning_text}\n\n"
        f"Observation Bank:\n{bank_text}"
        f"{template_block}"
    )

    def _call():
        client = _get_client(base_url)
        response = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": STAGE1_FINAL_JUDGE_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_text},
                        {"type": "input_image", "image_url": _encode_image(image_path)},
                    ],
                },
            ],
            text_format=Stage1FinalJudgment,
        )
        result = response.output_parsed
        if result is None:
            raise ValueError("Stage1FinalJudgment output is None")
        return result

    return _llm_call_with_retry(_call, max_retries)


def call_compare_judge(
    task: str,
    gt_answer: Any,
    predicted_answer: Any,
    model: str,
    base_url: str | None,
    max_retries: int,
) -> CompareJudgment:
    gt_text = json.dumps(gt_answer, ensure_ascii=False, indent=2)
    pred_text = json.dumps(predicted_answer, ensure_ascii=False, indent=2)
    user_text = (
        f"Task Goal: {task}\n\n"
        f"GROUND_TRUTH:\n{gt_text}\n\n"
        f"PREDICTED:\n{pred_text}"
    )

    def _call():
        client = _get_client(base_url)
        response = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": STAGE2_COMPARE_PROMPT},
                {"role": "user", "content": user_text},
            ],
            text_format=CompareJudgment,
            temperature=0,
        )
        result = response.output_parsed
        if result is None:
            raise ValueError("CompareJudgment output is None")
        return result

    return _llm_call_with_retry(_call, max_retries)


def call_step_judge(
    task: str,
    image_path: str,
    existing_bank: list[str],
    new_facts: list[str],
    model: str,
    base_url: str | None,
    max_retries: int,
) -> StepJudgment:
    bank_text = "empty" if not existing_bank else "\n".join(f"{i}: {obs}" for i, obs in enumerate(existing_bank))
    new_text = "\n".join(f"[{i}] {fact}" for i, fact in enumerate(new_facts))
    user_text = f"Task Goal: {task}\nExisting Observation Bank (for redundancy check only):\n{bank_text}\n\nNEW facts written at this step (N={len(new_facts)}):\n{new_text}"

    def _call():
        client = _get_client(base_url)
        response = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": STEP_JUDGE_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_text},
                        {"type": "input_image", "image_url": _encode_image(image_path)},
                    ],
                },
            ],
            text_format=StepJudgment,
            temperature=0,
        )
        result = response.output_parsed
        if result is None:
            raise ValueError("StepJudgment output is None")
        return result

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
    image_note = "Image 1 = BEFORE (the only correctness source). Image 2 = AFTER (action-effect fact-check only; never a justification source)." if has_after else "Image 1 = BEFORE. No AFTER screenshot is available for this step."
    user_text = f"Task Goal: {task}\nGround Truth Next Action: {ground_truth_action}\n{image_note}\nCandidate reasonings (N={len(reasonings)}):\n\n{labeled}"

    user_content: list[dict[str, Any]] = [
        {"type": "input_text", "text": user_text},
        {"type": "input_image", "image_url": _encode_image(image_path)},
    ]
    if has_after:
        user_content.append(
            {"type": "input_image", "image_url": _encode_image(next_image_path)}  # type: ignore[arg-type]
        )

    def _call():
        client = _get_client(base_url)
        response = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": RANK_JUDGE_PROMPT},
                {"role": "user", "content": user_content},
            ],
            text_format=RankJudgment,
            temperature=0,
        )
        result = response.output_parsed
        if result is None:
            raise ValueError("RankJudgment output is None")
        return result

    return _llm_call_with_retry(_call, max_retries)


# ---------------------------------------------------------------------------
# Main Reward Manager
# ---------------------------------------------------------------------------


class RRGRewardManager:
    """Three-judge reward manager for RRG v3 (stage-1 + stage-2 final, plus J_step and J_rank)."""

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

        # Per-fact reward magnitudes (J_step verdict + crucial bonus from J_final).
        # ungrounded → -r_pen; grounded ∧ meaningful → +r_step;
        # crucial_fact_indices contribution is +r_bonus (additive, not gated).
        self.r_step = float(cfg.get("r_step", 0.3))
        self.r_pen = float(cfg.get("r_pen", 0.5))
        self.r_bonus = float(cfg.get("r_bonus", 0.7))
        # Linear length penalty subtracted from r_reason: penalty grows with
        # reason_tokens / response_length, regardless of rank. Caps unbounded
        # reasoning growth that the rank judge alone can't suppress.
        self.reason_length_penalty = float(cfg.get("reason_length_penalty", 0.3))
        self.w_reason = float(cfg.get("w_reason", 1.0))
        self.final_judge_model = cfg.get("final_judge_model", "doubao-seed-2-0-pro-260215")
        self.rank_judge_model = cfg.get("rank_judge_model", self.final_judge_model)
        self.step_judge_model = cfg.get("step_judge_model", self.final_judge_model)
        self.compare_judge_model = cfg.get("compare_judge_model", self.final_judge_model)
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
            # Stage-1 ``is_finished`` True verdicts (successes only — failure
            # branches do not bump this counter so the rate metric reflects
            # actual judge agreement).
            "final_is_finished": 0,
            "rank_calls": 0,
            "rank_failures": 0,
            "step_calls": 0,
            "step_failures": 0,
            # Stage-2 compare judge counters. ``compare_calls`` only fires
            # for trajectories with a ground-truth return value.
            "compare_calls": 0,
            "compare_failures": 0,
            "compare_matches": 0,
            # Fact-quality aggregates over all per-fact judgements in this batch.
            "fact_total": 0,
            "fact_grounded": 0,
            "fact_meaningful": 0,
            "fact_crucial": 0,
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
        # Phase F1 — Stage-1 final judge per trajectory                       #
        # ------------------------------------------------------------------ #
        # Stage 1 derives ``is_finished`` + ``predicted_answer`` (when a
        # ground-truth return value exists) + ``crucial_fact_indices`` from
        # task + ALL action reasonings + final fact bank + last screenshot.
        stage1_futures: dict[Future, int] = {}
        # env_slot -> ground-truth answer (None for tasks without a return
        # value; non-None values gate stage-2 compare). Tracked separately so
        # the post-stage-1 pass can decide whether to submit stage 2.
        slot_to_gt: dict[int, Any] = {}
        for env_slot, _ in enumerate(unique_trajs_ordered):
            traj_meta = step_metadata.get(str(env_slot), [])
            if not traj_meta:
                continue
            last_meta = traj_meta[-1]
            image_path = last_meta.get("screenshot_path", "")
            fact_bank = last_meta.get("final_fact_bank") or traj_fact_content.get(env_slot, [])

            # Detect return-value tasks. ``gt_answer`` is None when the task
            # has no return value; an empty container also counts as "no
            # ground truth" — both skip stage 2.
            gt_answer = last_meta.get("gt_answer", None)
            has_gt = gt_answer is not None and (not hasattr(gt_answer, "__len__") or len(gt_answer) > 0)
            slot_to_gt[env_slot] = gt_answer if has_gt else None
            answer_template = _redact_values(gt_answer) if has_gt else None

            action_reasonings = [m.get("reasoning_text", "") or "" for m in traj_meta]

            if not image_path or not os.path.isfile(image_path):
                continue  # charity default applied below (is_finished=True, matches=True)

            fut = pool.submit(
                call_stage1_final_judge,
                last_meta.get("task", ""),
                action_reasonings,
                list(fact_bank),
                image_path,
                answer_template,
                self.final_judge_model,
                self.judge_base_url,
                self.max_retries,
            )
            stage1_futures[fut] = env_slot
            judge_stats["final_calls"] += 1

        # Stage-1 results: env_slot -> (is_finished, predicted_answer, crucial_set).
        stage1_results: dict[int, tuple[bool, Any, set[int]]] = {}
        for fut in as_completed(stage1_futures):
            env_slot = stage1_futures[fut]
            try:
                judgment = fut.result()
                crucial_set = {int(i) for i in judgment.crucial_fact_indices}
                stage1_results[env_slot] = (
                    bool(judgment.is_finished),
                    judgment.predicted_answer,
                    crucial_set,
                )
                if bool(judgment.is_finished):
                    judge_stats["final_is_finished"] += 1
                if self.debug_log:
                    self._log_debug(
                        "stage1_final_judge",
                        {
                            "env_slot": env_slot,
                            "is_finished": judgment.is_finished,
                            "predicted_answer": judgment.predicted_answer,
                            "crucial": sorted(crucial_set),
                            "reasoning": judgment.reasoning[:300],
                        },
                        file_overrides={"reasoning": judgment.reasoning},
                    )
            except Exception as exc:
                # Charity default: is_finished=True, predicted={}, no crucial.
                stage1_results[env_slot] = (True, {}, set())
                judge_stats["final_failures"] += 1
                self._log_error(
                    "stage1_final_judge_failed",
                    {
                        "env_slot": env_slot,
                        "error": repr(exc),
                    },
                )

        # ------------------------------------------------------------------ #
        # Phase F2 — Stage-2 compare judge (return-value tasks only)          #
        # ------------------------------------------------------------------ #
        # Text-only call: compare predicted_answer (from stage 1) vs
        # gt_answer semantically; returns a single ``matches`` boolean.
        # No-return tasks skip this entirely; ``matches`` defaults to True.
        stage2_futures: dict[Future, int] = {}
        for env_slot, (_is_finished, predicted_answer, _crucial) in stage1_results.items():
            gt_answer = slot_to_gt.get(env_slot)
            if gt_answer is None:
                continue
            traj_meta = step_metadata.get(str(env_slot), [])
            task = traj_meta[-1].get("task", "") if traj_meta else ""
            fut = pool.submit(
                call_compare_judge,
                task,
                gt_answer,
                predicted_answer,
                self.compare_judge_model,
                self.judge_base_url,
                self.max_retries,
            )
            stage2_futures[fut] = env_slot
            judge_stats["compare_calls"] += 1

        stage2_matches: dict[int, bool] = {}
        for fut in as_completed(stage2_futures):
            env_slot = stage2_futures[fut]
            try:
                judgment = fut.result()
                matches = bool(judgment.matches)
                stage2_matches[env_slot] = matches
                if matches:
                    judge_stats["compare_matches"] += 1
                if self.debug_log:
                    self._log_debug(
                        "stage2_compare_judge",
                        {
                            "env_slot": env_slot,
                            "matches": matches,
                            "reasoning": judgment.reasoning[:300],
                        },
                        file_overrides={"reasoning": judgment.reasoning},
                    )
            except Exception as exc:
                # Charity default: matches=True (do not punish on judge error).
                stage2_matches[env_slot] = True
                judge_stats["compare_failures"] += 1
                self._log_error(
                    "stage2_compare_judge_failed",
                    {
                        "env_slot": env_slot,
                        "error": repr(exc),
                    },
                )

        # Combine stage-1 + stage-2 into final_results: env_slot -> (is_finished, matches, crucial_set).
        # No-return-value trajectories get matches=True (stage 2 was skipped);
        # missing slots fall back to (True, True, set()) at the consumer site.
        final_results: dict[int, tuple[bool, bool, set[int]]] = {}
        for env_slot, (is_finished, _pred, crucial_set) in stage1_results.items():
            matches = stage2_matches.get(env_slot, True)
            final_results[env_slot] = (is_finished, matches, crucial_set)

        # ------------------------------------------------------------------ #
        # Phase S — Step judge per (env_slot, step_t) where facts were written #
        # ------------------------------------------------------------------ #
        # One call per step that wrote at least one new fact. Returns a list of
        # FactQuality aligned positionally with the new facts at that step.
        all_step_futures: dict[Future, tuple[int, int, int]] = {}
        for env_slot, _ in enumerate(unique_trajs_ordered):
            traj_meta = step_metadata.get(str(env_slot), [])
            step_map = step_fact_idxs.get(env_slot, {})
            for t, meta in enumerate(traj_meta):
                if not step_map.get(t):
                    continue
                image_path = meta.get("screenshot_path", "")
                if not image_path or not os.path.isfile(image_path):
                    continue
                new_facts = [upd.get("observation", "") for upd in meta.get("writing_updates", []) if upd.get("action") == "add" and upd.get("observation", "")]
                if not new_facts:
                    continue
                existing_bank = list(meta.get("observations_before", []) or [])
                fut = pool.submit(
                    call_step_judge,
                    meta.get("task", ""),
                    image_path,
                    existing_bank,
                    new_facts,
                    self.step_judge_model,
                    self.judge_base_url,
                    self.max_retries,
                )
                all_step_futures[fut] = (env_slot, t, len(new_facts))
                judge_stats["step_calls"] += 1

        # step_results[(env_slot, step_t)] = list[FactQuality]; positional with
        # new_facts at that step. Failure default: all neutral (grounded=True,
        # meaningful=False) → r_per_fact = 0, no punishment on judge failure.
        step_results: dict[tuple[int, int], list[FactQuality]] = {}
        for fut in as_completed(all_step_futures):
            env_slot, t, n_facts = all_step_futures[fut]
            try:
                judgment = fut.result()
                qualities = list(judgment.fact_quality)
                # Pad/truncate to n_facts to keep positional alignment robust.
                if len(qualities) < n_facts:
                    qualities.extend(FactQuality(grounded=True, meaningful=False) for _ in range(n_facts - len(qualities)))
                elif len(qualities) > n_facts:
                    qualities = qualities[:n_facts]
                step_results[(env_slot, t)] = qualities
                if self.debug_log:
                    self._log_debug(
                        "step_judge",
                        {
                            "env_slot": env_slot,
                            "step": t,
                            "n_facts": n_facts,
                            "fact_quality": [{"grounded": q.grounded, "meaningful": q.meaningful} for q in qualities],
                            "reasoning": judgment.reasoning[:200],
                        },
                        file_overrides={"reasoning": judgment.reasoning},
                    )
            except Exception as exc:
                step_results[(env_slot, t)] = [FactQuality(grounded=True, meaningful=False) for _ in range(n_facts)]
                judge_stats["step_failures"] += 1
                self._log_error(
                    "step_judge_failed",
                    {
                        "env_slot": env_slot,
                        "step": t,
                        "error": repr(exc),
                    },
                )

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
        if self.w_reason != 0:
            for uid_str, sibling_slots in uid_to_slots.items():
                if len(sibling_slots) < 2:
                    continue
                max_t = max(len(step_metadata.get(str(s), [])) for s in sibling_slots)
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
                    self._log_debug(
                        "rank_judge",
                        {
                            "uid": uid_str,
                            "step": t,
                            "n": n,
                            "order": judgment.order,
                            "reasoning": judgment.reasoning[:200],
                        },
                        file_overrides={"reasoning": judgment.reasoning},
                    )
            except Exception as exc:
                for local_idx in range(n):
                    rank_by_local[local_idx] = middle
                judge_stats["rank_failures"] += 1
                self._log_error(
                    "rank_judge_failed",
                    {
                        "uid": uid_str,
                        "step": t,
                        "error": repr(exc),
                    },
                )
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

            # Final reward for this trajectory: 1.0 iff stage-1 is_finished
            # AND stage-2 matches (matches defaults to True for no-return tasks).
            if env_slot in final_results:
                is_finished, matches, crucial_indices = final_results[env_slot]
            else:
                is_finished, matches, crucial_indices = True, True, set()
            r_final = 1.0 if (is_finished and matches) else 0.0

            step_fact_idx_map = step_fact_idxs.get(env_slot, {})

            for local_t, batch_idx in enumerate(traj_indices):
                if local_t >= len(traj_meta):
                    break

                global_idxs_at_t = step_fact_idx_map.get(local_t, [])

                if fact_mask_batch is not None:
                    fact_tokens = int(fact_mask_batch[batch_idx].sum().item())
                else:
                    fact_tokens = 0
                if reason_mask_batch is not None:
                    reason_tokens = int(reason_mask_batch[batch_idx].sum().item())
                else:
                    reason_tokens = 0

                # Per-fact reward aggregation (silent step → 0):
                #   ungrounded                             → -r_pen
                #   grounded ∧ ¬meaningful                 →  0
                #   grounded ∧ meaningful                  → +r_step
                #   + crucial_fact_indices contains it     → +r_bonus (additive)
                # r_fact[t] = Σ over facts written at step t.
                qualities = step_results.get((env_slot, local_t), [])
                per_fact_rewards: list[float] = []
                for fact_pos, global_idx in enumerate(global_idxs_at_t):
                    if fact_pos < len(qualities):
                        q = qualities[fact_pos]
                        grounded = bool(q.grounded)
                        meaningful = bool(q.meaningful)
                    else:
                        # Step judge didn't run / fewer entries than facts:
                        # neutral default — no reward, no penalty.
                        grounded, meaningful = True, False
                    is_crucial = int(global_idx) in crucial_indices
                    if not grounded:
                        per_fact = -self.r_pen
                    elif meaningful:
                        per_fact = self.r_step
                    else:
                        per_fact = 0.0
                    if is_crucial:
                        per_fact += self.r_bonus
                    per_fact_rewards.append(per_fact)
                    judge_stats["fact_total"] += 1
                    if grounded:
                        judge_stats["fact_grounded"] += 1
                    if grounded and meaningful:
                        judge_stats["fact_meaningful"] += 1
                    if is_crucial:
                        judge_stats["fact_crucial"] += 1
                r_fact = float(sum(per_fact_rewards))

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
                    qualities = step_results.get((env_slot, local_t), [])
                    step_globals = step_fact_idx_map.get(local_t, [])
                    fact_breakdown = [
                        {
                            "global_idx": int(g),
                            "grounded": bool(qualities[i].grounded) if i < len(qualities) else None,
                            "meaningful": bool(qualities[i].meaningful) if i < len(qualities) else None,
                            "crucial": bool(int(g) in crucial_indices),
                        }
                        for i, g in enumerate(step_globals)
                    ]
                    debug_steps.append(
                        {
                            "step": local_t,
                            "num_facts": len(step_globals),
                            "fact_tokens": int(fact_tokens_arr[batch_idx]) if batch_idx >= 0 else 0,
                            "fact_reward": float(fact_rewards[batch_idx]) if batch_idx >= 0 else 0.0,
                            "reason_reward": float(reason_rewards[batch_idx]) if batch_idx >= 0 else 0.0,
                            "facts": fact_breakdown,
                        }
                    )
                debug_examples.append(
                    {
                        "env_slot": env_slot,
                        "n_siblings": n_siblings,
                        "is_finished": is_finished,
                        "matches": matches,
                        "crucial_count": len(crucial_indices),
                        "total_facts": len(traj_fact_content.get(env_slot, [])),
                        "r_final": r_final,
                        "steps": debug_steps,
                    }
                )

        # Publish judge operational counters for compute_rrg_metrics.
        if hasattr(data, "meta_info") and isinstance(data.meta_info, dict):
            data.meta_info["rrg_judge_stats"] = dict(judge_stats)

        self._log_debug(
            "batch_reward_summary",
            {
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
            },
        )

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
