"""RRG Reward Manager — computes multi-signal rewards using frozen LLM judges.

After rollout completes, reads step metadata from the environment manager,
calls J_cite and J_fact judges in parallel, and returns per-step citation and
writing rewards via ``reward_extra_info``.
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
from dataclasses import dataclass, field
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

class CitationJudgment(BaseModel):  # type: ignore[misc]
    reasoning: str
    necessary_indices: list[int]


class FactJudgment(BaseModel):  # type: ignore[misc]
    correctness: float
    atomicity: float
    granularity: float
    reasoning: str


class FinalJudgment(BaseModel):  # type: ignore[misc]
    reasoning: str
    score: float


# ---------------------------------------------------------------------------
# Judge prompts (from rrg/reward_calc_batch.py)
# ---------------------------------------------------------------------------

CITE_JUDGE_PROMPT = """\
You are a citation quality judge for a GUI agent reasoning system.

The agent maintains a list of observations (a fact bank) across a multi-step task. \
At each step, the agent may cite prior observations that support its decision.

Your job: given the task goal, the current GUI screenshot, the ground-truth action, \
and the current observation bank, determine which observations were GENUINELY NECESSARY \
for making this specific decision. There are several principles:
1. If I can made the correct action only relying on the task goal and the screenshot: \
no observation is necessary in the current step.
2. If many observations are necessary and one contains all information of others, \
only take that one as necessary in the current step.

Respond with JSON: {"reasoning": "...", "necessary_indices": [...]}.
Only include indices that exist in the observation bank.\
"""

FACT_JUDGE_PROMPT = """\
You are a fact quality judge for a GUI agent reasoning system.

The agent observes the current GUI screenshot, writes observations about what it sees \
right now, then reasons and takes an action. Observations should describe the CURRENT \
visible state — not predict what will happen after the action.

You will be given the prior observation bank (facts accumulated from earlier steps). \
Use it as ground-truth context when evaluating the new observation — for example, \
to resolve cross-step references such as item counts or sequential progress that \
cannot be verified from the current screenshot alone.

You must evaluate a single newly written observation on three dimensions:

1. correctness (0.0–1.0): Is the observation factually accurate given the current \
screenshot and the prior observation bank? 1.0 = completely accurate; use prior \
observations to resolve references that are not directly visible on screen.

2. atomicity (0.0–1.0): Does the observation express a coherent, self-contained unit \
of information? 1.0 = well-formed unit; lower if it mixes unrelated ideas. \
Note: bundling closely related attributes of a single entity is acceptable and should NOT be penalized — \
those are attributes of the same object, not independent ideas.

3. granularity (0.0–1.0): Is the observation appropriately scoped? \
1.0 = well scoped; lower if it is too vague to be useful, or too fine-grained/fragmented \
(captures irrelevant micro-details unlikely to help later steps).

Respond with JSON: {"reasoning": "...", "correctness": ..., "atomicity": ..., "granularity": ...}\
"""

FINAL_JUDGE_PROMPT = """\
You are a final answer validator for a GUI agent reasoning system.

The agent replays a multi-step GUI task while building an observation bank — \
a list of facts recorded across all steps. At the end of the task, the agent \
submits an explicit answer (such as a specific value, count, text, or selection).

Your job: given the task goal, the final GUI screenshot, the agent's accumulated \
observation bank, and the correct answer, score how well the combination of the \
observation bank and the current screenshot supports the correct answer.

There are several principles:
1. Joint sufficiency: evaluate the observation bank and the current screenshot \
together. Facts missing from the bank but clearly visible in the screenshot still \
count — the agent could read them from the screen when answering.
2. Relevance over completeness: the bank need not contain every observed detail. \
Only the facts (from the bank or the screenshot) necessary to produce the correct \
answer matter for the score.
3. Correctness anchor: the provided answer IS the correct answer. Your task is to \
score how well the available evidence supports it, not to re-verify it independently.

Scoring guide: 1.0 = all required information is clearly available; 0.0 ~ 1.0 = partial \
information is available but key details are missing or ambiguous; 0.0 = the \
required information is entirely absent.

Respond with JSON: {"reasoning": "...", "score": <float 0.0–1.0>}.\
"""


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class FactVersionRecord:
    obs_index: int
    step_written: int
    version: int
    action: str
    content: str
    judgment: FactJudgment | None = None
    validated_uses: list[tuple[int, int]] = field(default_factory=list)
    T: float = 0.0
    quality_score: float = 0.0
    R_fact: float = 0.0


# ---------------------------------------------------------------------------
# Pure reward computation (ported from rrg/reward_calc_batch.py)
# ---------------------------------------------------------------------------

def compute_cite_step_reward(
    cited: set[int],
    necessary: set[int],
    alpha_prec: float,
    alpha_rec: float,
    alpha_rep: float,
) -> tuple[float, float, float, float]:
    """Returns (P, Q, U, R_cite_step)."""
    redundant = cited - necessary
    P = len(cited & necessary) / max(1, len(cited))
    Q = len(cited & necessary) / max(1, len(necessary))
    U = len(redundant) / max(1, len(cited))
    if not cited and not necessary:
        R = alpha_prec + alpha_rec
    else:
        R = alpha_prec * P + alpha_rec * Q - alpha_rep * U
    return P, Q, U, R


def compute_fact_quality(
    judgment: FactJudgment,
    lambda_val: float,
    lambda_atom: float,
    lambda_gran: float,
) -> float:
    return judgment.correctness * (
        lambda_val
        + lambda_atom * judgment.atomicity
        + lambda_gran * judgment.granularity
    )


def compute_T_version(
    validated_uses: list[tuple[int, int]],
    step_written: int,
    total_steps: int,
    gamma: float,
) -> float:
    span = max(1, total_steps - 1)
    total = 0.0
    for step_u, m in validated_uses:
        exponent = (step_u - step_written - 1) / span
        total += (gamma ** exponent) / (1 + m)
    return total


def compute_R_fact_version(
    version: FactVersionRecord,
    previous: FactVersionRecord | None,
    lambda_use: float,
    update_bonus_scale: float,
) -> float:
    if version.T <= 0.0:
        return 0.0
    if version.action == "add" or previous is None:
        return lambda_use * version.T + version.quality_score
    # Update operation
    changed = version.content.strip() != previous.content.strip()
    revision_gain = max(0.0, version.quality_score - previous.quality_score) if changed else 0.0
    return lambda_use * version.T + update_bonus_scale * revision_gain


def _reconstruct_obs_bank(observations_before: list[str], writing_updates: list[dict]) -> list[str]:
    """Apply writing_updates to observations_before and return the resulting bank."""
    bank = list(observations_before)
    for upd in writing_updates:
        action = upd.get("action", "add")
        obs_index = upd.get("observation_index")
        content = upd.get("observation", "")
        if not isinstance(obs_index, int):
            continue
        if action == "add":
            while len(bank) <= obs_index:
                bank.append("")
            bank[obs_index] = content
        elif action == "update" and obs_index < len(bank):
            bank[obs_index] = content
    return bank


# ---------------------------------------------------------------------------
# LLM Judge callers
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


def call_cite_judge(
    task: str,
    image_path: str,
    action: dict | str,
    observations: list[str],
    model: str,
    base_url: str | None,
    max_retries: int,
) -> CitationJudgment:
    obs_text = (
        "empty"
        if not observations
        else "\n".join(f"{i}: {obs}" for i, obs in enumerate(observations))
    )
    action_str = json.dumps(action) if isinstance(action, dict) else action
    user_text = (
        f"Task Goal: {task}\n"
        f"Ground Truth Action: {action_str}\n"
        f"Observation Bank:\n{obs_text}"
    )

    def _call():
        client = _get_client(base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CITE_JUDGE_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": _encode_image(image_path)}},
                ]},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("CitationJudgment completion output is empty")
        return _parse_json_response(content, CitationJudgment)

    return _llm_call_with_retry(_call, max_retries)


def call_fact_judge(
    task: str,
    fact_content: str,
    image_path: str,
    observations_before: list[str],
    model: str,
    base_url: str | None,
    max_retries: int,
) -> FactJudgment:
    obs_text = (
        "empty"
        if not observations_before
        else "\n".join(f"{i}: {obs}" for i, obs in enumerate(observations_before))
    )
    user_text = (
        f"Task Goal: {task}\n"
        f"Prior Observation Bank:\n{obs_text}\n"
        f'Newly written observation: "{fact_content}"'
    )

    def _call():
        client = _get_client(base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": FACT_JUDGE_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": _encode_image(image_path)}},
                ]},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("FactJudgment completion output is empty")
        return _parse_json_response(content, FactJudgment)

    return _llm_call_with_retry(_call, max_retries)


def call_final_judge(
    task: str,
    final_answer: str,
    observation_bank: list[str],
    image_path: str,
    model: str,
    base_url: str | None,
    max_retries: int,
) -> FinalJudgment:
    bank_text = (
        "empty"
        if not observation_bank
        else "\n".join(f"{i}: {obs}" for i, obs in enumerate(observation_bank) if obs)
    )
    user_text = (
        f"Task Goal: {task}\n"
        f"Correct Answer: {final_answer}\n"
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


# ---------------------------------------------------------------------------
# Main Reward Manager
# ---------------------------------------------------------------------------

class RRGRewardManager:
    """Multi-signal reward manager for RRG with frozen LLM judges."""

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

        # Hyperparameters
        self.alpha_prec = cfg.get("alpha_prec", 0.4)
        self.alpha_rec = cfg.get("alpha_rec", 0.4)
        self.alpha_rep = cfg.get("alpha_rep", 0.2)
        self.lambda_use = cfg.get("lambda_use", 0.4)
        self.lambda_val = cfg.get("lambda_val", 0.2)
        self.lambda_atom = cfg.get("lambda_atom", 0.2)
        self.lambda_gran = cfg.get("lambda_gran", 0.2)
        self.gamma_discount = cfg.get("gamma_discount", 0.9)
        self.update_bonus_scale = cfg.get("update_bonus_scale", 0.5)
        self.beta_cite = cfg.get("beta_cite", 0.5)
        self.beta_write = cfg.get("beta_write", 0.5)

        # Judge config
        self.judge_model = cfg.get("judge_model", "doubao-seed-2-0-pro-260215")
        self.judge_base_url = cfg.get("judge_base_url", None)
        self.max_retries = cfg.get("max_retries", 3)
        self.max_judge_workers = cfg.get("max_judge_workers", 16)
        self.debug_log = bool(cfg.get("debug_log", False))
        self.debug_log_samples = int(cfg.get("debug_log_samples", 3))
        self.debug_log_file = cfg.get("debug_log_file", None)
        self.log_judge_errors = bool(cfg.get("log_judge_errors", True))
        self.max_judge_error_logs = int(cfg.get("max_judge_error_logs", 20))
        self._judge_error_logs = 0

        # Persistent thread pool — shared across all __call__ invocations and all
        # trajectories within a batch, enabling cross-trajectory parallelism.
        self._pool: ThreadPoolExecutor | None = None

    def _get_pool(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.max_judge_workers)
        return self._pool

    def _log_debug(self, event: str, payload: dict):
        if not self.debug_log:
            return
        message = {"event": event, **payload}
        print(f"[RRG][reward] {json.dumps(message, ensure_ascii=False, default=str)}")
        if self.debug_log_file:
            with open(self.debug_log_file, "a") as f:
                f.write(json.dumps({"source": "reward", **message}, ensure_ascii=False, default=str) + "\n")

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
        """Compute RRG rewards using accumulated step metadata from envs.

        Returns a reward_tensor (for compatibility) plus per-step citation and
        writing rewards in ``reward_extra_info`` which flows to the advantage
        estimator via ``batch.non_tensor_batch``.

        Judge calls are batched across all trajectories before collecting results,
        maximising utilisation of the shared thread pool.
        """
        batch_size = len(data)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        # Collect step metadata from the environment manager
        step_metadata = getattr(self.envs, "step_metadata", None)
        if step_metadata is None:
            if return_dict:
                return {
                    "reward_tensor": reward_tensor,
                    "reward_extra_info": {
                        "rrg_cite_rewards": [0.0] * batch_size,
                        "rrg_write_rewards": [0.0] * batch_size,
                    },
                }
            return reward_tensor

        # Group batch items by trajectory.  step_metadata is keyed by env-slot
        # index ("0", "1", ...).  gather_rollout_data preserves env-slot order,
        # so the first unique traj_uid encountered belongs to env slot 0, etc.
        traj_uids = data.non_tensor_batch.get("traj_uid", np.arange(batch_size))

        seen: set = set()
        unique_trajs_ordered = []
        for uid in traj_uids:
            if uid not in seen:
                unique_trajs_ordered.append(uid)
                seen.add(uid)

        cite_rewards = np.zeros(batch_size, dtype=np.float32)
        write_rewards = np.zeros(batch_size, dtype=np.float32)
        final_rewards = np.zeros(batch_size, dtype=np.float32)
        uids = data.non_tensor_batch.get("uid", None)
        step_group_uids = np.array(
            [str(u) for u in (uids if uids is not None else np.arange(batch_size))],
            dtype=object,
        )
        reward_debug_examples = []
        total_meta_steps = 0
        total_writes = 0

        pool = self._get_pool()

        # ------------------------------------------------------------------ #
        # Phase A — Submit ALL cite judge futures across all trajectories     #
        # ------------------------------------------------------------------ #
        # future -> (env_slot, step_t, cited_set)
        all_cite_futures: dict[Future, tuple[int, int, set[int]]] = {}
        # missing_cite[env_slot][t] = (image_path, cited_indices) for steps with no usable image
        missing_cite: dict[int, dict[int, tuple[str, list]]] = {}

        for env_slot, traj_uid in enumerate(unique_trajs_ordered):
            traj_meta = step_metadata.get(str(env_slot), [])
            if not traj_meta:
                continue
            total_meta_steps += len(traj_meta)
            total_writes += sum(len(meta.get("writing_updates", [])) for meta in traj_meta)

            for t, meta in enumerate(traj_meta):
                image_path = meta.get("screenshot_path", "")
                cited_indices = meta.get("cited_indices", [])
                if not image_path or not os.path.isfile(image_path):
                    missing_cite.setdefault(env_slot, {})[t] = (image_path, cited_indices)
                    continue
                fut = pool.submit(
                    call_cite_judge,
                    meta.get("task", ""),
                    image_path,
                    meta.get("ground_truth_action", ""),
                    meta.get("observations_before", []),
                    self.judge_model, self.judge_base_url, self.max_retries,
                )
                all_cite_futures[fut] = (env_slot, t, set(cited_indices))

        # ------------------------------------------------------------------ #
        # Phase B — Collect all cite results                                  #
        # ------------------------------------------------------------------ #
        # necessary_sets_by_slot[env_slot][t] = set of necessary obs indices
        necessary_sets_by_slot: dict[int, dict[int, set[int]]] = defaultdict(dict)
        cite_rewards_by_slot: dict[int, dict[int, float]] = defaultdict(dict)
        cite_debug_by_slot: dict[int, dict[int, dict]] = defaultdict(dict)

        # Backfill missing-image steps with zero reward and empty necessary sets
        for env_slot, steps in missing_cite.items():
            for t, (image_path, cited_indices) in steps.items():
                necessary_sets_by_slot[env_slot][t] = set()
                cite_rewards_by_slot[env_slot][t] = 0.0
                if self.debug_log:
                    cite_debug_by_slot[env_slot][t] = {
                        "error": "missing_image",
                        "image_path": image_path,
                        "cited": cited_indices,
                    }

        for fut in as_completed(all_cite_futures):
            env_slot, t, cited_set = all_cite_futures[fut]
            traj_meta_for_err = step_metadata.get(str(env_slot), [])
            try:
                judgment = fut.result()
                necessary_set = set(judgment.necessary_indices)
                P, Q, U, R = compute_cite_step_reward(
                    cited_set, necessary_set,
                    self.alpha_prec, self.alpha_rec, self.alpha_rep,
                )
                necessary_sets_by_slot[env_slot][t] = necessary_set
                cite_rewards_by_slot[env_slot][t] = R
                if self.debug_log:
                    cite_debug_by_slot[env_slot][t] = {
                        "cited": sorted(cited_set),
                        "necessary": sorted(necessary_set),
                        "precision": P,
                        "recall": Q,
                        "overcite": U,
                        "reward": R,
                    }
            except Exception as exc:
                necessary_sets_by_slot[env_slot][t] = set()
                cite_rewards_by_slot[env_slot][t] = 0.0
                if self.debug_log:
                    cite_debug_by_slot[env_slot][t] = {
                        "error": repr(exc),
                        "cited": sorted(cited_set),
                    }
                self._log_error("cite_judge_failed", {
                    "env_slot": env_slot,
                    "step_index": t,
                    "error": repr(exc),
                    "image_path": traj_meta_for_err[t].get("screenshot_path", "") if t < len(traj_meta_for_err) else "",
                })

        # ------------------------------------------------------------------ #
        # Phase C — Submit ALL fact judge futures across all trajectories     #
        # (filtered to obs_indices that can have T > 0)                       #
        # ------------------------------------------------------------------ #
        # future -> (env_slot, step_t, content, upd)
        all_fact_futures: dict[Future, tuple[int, int, str, dict]] = {}
        # fact_debug_skip_by_slot[env_slot][t] = list of skipped/errored entries
        fact_debug_skip_by_slot: dict[int, dict[int, list]] = defaultdict(lambda: defaultdict(list))

        for env_slot, traj_uid in enumerate(unique_trajs_ordered):
            traj_meta = step_metadata.get(str(env_slot), [])
            if not traj_meta:
                continue
            necessary_sets = necessary_sets_by_slot[env_slot]

            # Only obs_indices that are both cited and necessary at some step can have T > 0
            relevant_obs_indices: set[int] = set()
            for t, necessary_set in necessary_sets.items():
                if t < len(traj_meta):
                    cited_at_t = set(traj_meta[t].get("cited_indices", []))
                    relevant_obs_indices.update(cited_at_t & necessary_set)

            for t, meta in enumerate(traj_meta):
                task = meta.get("task", "")
                image_path = meta.get("screenshot_path", "")
                observations_before = meta.get("observations_before", [])

                for upd in meta.get("writing_updates", []):
                    obs_index = upd.get("observation_index")
                    content = upd.get("observation", "")
                    if obs_index not in relevant_obs_indices:
                        if self.debug_log:
                            fact_debug_skip_by_slot[env_slot][t].append({
                                "action": upd.get("action", "add"),
                                "observation_index": obs_index,
                                "content": content[:200],
                                "skipped": "not_in_relevant_obs_indices",
                            })
                        continue
                    if not content or not image_path or not os.path.isfile(image_path):
                        if self.debug_log:
                            fact_debug_skip_by_slot[env_slot][t].append({
                                "action": upd.get("action", "add"),
                                "observation_index": obs_index,
                                "content": content[:200],
                                "error": "empty_content_or_missing_image",
                                "image_path": image_path,
                            })
                        continue
                    fut = pool.submit(
                        call_fact_judge,
                        task, content, image_path, observations_before,
                        self.judge_model, self.judge_base_url, self.max_retries,
                    )
                    all_fact_futures[fut] = (env_slot, t, content, upd)

        # ------------------------------------------------------------------ #
        # Phase D — Collect all fact results                                  #
        # ------------------------------------------------------------------ #
        # fact_versions_by_slot[env_slot][obs_index] = list[FactVersionRecord] (unsorted)
        fact_versions_by_slot: dict[int, dict[int, list[FactVersionRecord]]] = defaultdict(lambda: defaultdict(list))
        fact_debug_by_slot: dict[int, dict[int, list]] = defaultdict(lambda: defaultdict(list))

        for fut in as_completed(all_fact_futures):
            env_slot, t, content, upd = all_fact_futures[fut]
            raw_obs_index = upd.get("observation_index")
            if not isinstance(raw_obs_index, int):
                continue
            obs_index: int = raw_obs_index
            action = upd.get("action", "add")
            traj_meta_for_err = step_metadata.get(str(env_slot), [])
            try:
                judgment = fut.result()
                quality = compute_fact_quality(
                    judgment,
                    self.lambda_val, self.lambda_atom, self.lambda_gran,
                )
                version = FactVersionRecord(
                    obs_index=obs_index,
                    step_written=t,
                    version=-1,
                    action=action,
                    content=content,
                    judgment=judgment,
                    quality_score=quality,
                )
                fact_versions_by_slot[env_slot][obs_index].append(version)
                if self.debug_log:
                    fact_debug_by_slot[env_slot][t].append({
                        "action": action,
                        "observation_index": obs_index,
                        "content": content[:300],
                        "correctness": judgment.correctness,
                        "atomicity": judgment.atomicity,
                        "granularity": judgment.granularity,
                        "quality": quality,
                        "judge_reasoning": judgment.reasoning[:300],
                    })
            except Exception as exc:
                if self.debug_log:
                    fact_debug_by_slot[env_slot][t].append({
                        "action": action,
                        "observation_index": obs_index,
                        "content": content[:300],
                        "error": repr(exc),
                    })
                self._log_error("fact_judge_failed", {
                    "env_slot": env_slot,
                    "step_index": t,
                    "error": repr(exc),
                    "content": content[:300],
                    "image_path": traj_meta_for_err[t].get("screenshot_path", "") if t < len(traj_meta_for_err) else "",
                })

        # ------------------------------------------------------------------ #
        # Phase E — Submit and collect final judge futures                    #
        # Called only for trajectories whose last action is "terminate" with  #
        # an explicit answer (content != None).                               #
        # ------------------------------------------------------------------ #
        # future -> env_slot
        all_final_futures: dict[Future, int] = {}

        for env_slot, traj_uid in enumerate(unique_trajs_ordered):
            traj_meta = step_metadata.get(str(env_slot), [])
            if not traj_meta:
                continue
            last_meta = traj_meta[-1]
            image_path = last_meta.get("screenshot_path", "")

            raw_action = last_meta.get("ground_truth_action", "")
            try:
                action_dict = json.loads(raw_action) if isinstance(raw_action, str) else (raw_action or {})
            except (json.JSONDecodeError, TypeError):
                action_dict = {}

            if action_dict.get("action") != "terminate" or action_dict.get("content") is None:
                continue  # no explicit answer — R_final defaults to 1.0

            final_answer = json.dumps(action_dict["content"]) if isinstance(action_dict["content"], dict) else str(action_dict["content"])
            obs_bank = _reconstruct_obs_bank(
                last_meta.get("observations_before", []),
                last_meta.get("writing_updates", []),
            )

            if not obs_bank or not image_path or not os.path.isfile(image_path):
                continue  # cannot evaluate — keep R_final = 1.0

            fut = pool.submit(
                call_final_judge,
                last_meta.get("task", ""),
                final_answer,
                obs_bank,
                image_path,
                self.judge_model, self.judge_base_url, self.max_retries,
            )
            all_final_futures[fut] = env_slot

        # Collect: score in [0.0, 1.0]; trajectories without an explicit answer default to 1.0
        final_score_by_slot: dict[int, float] = {}
        for fut in as_completed(all_final_futures):
            env_slot = all_final_futures[fut]
            try:
                judgment = fut.result()
                final_score_by_slot[env_slot] = float(judgment.score)
                if self.debug_log:
                    last_meta = step_metadata.get(str(env_slot), [{}])[-1]
                    self._log_debug("final_judge", {
                        "env_slot": env_slot,
                        "score": judgment.score,
                        "reasoning": judgment.reasoning[:300],
                        "image_path": last_meta.get("screenshot_path", ""),
                    })
            except Exception as exc:
                final_score_by_slot[env_slot] = 1.0  # default to 1.0 on error
                self._log_error("final_judge_failed", {
                    "env_slot": env_slot,
                    "error": repr(exc),
                    "image_path": step_metadata.get(str(env_slot), [{}])[-1].get("screenshot_path", ""),
                })

        # ------------------------------------------------------------------ #
        # Final loop — per-trajectory reward computation (pure math, no LLM) #
        # ------------------------------------------------------------------ #
        for env_slot, traj_uid in enumerate(unique_trajs_ordered):
            traj_mask = traj_uids == traj_uid
            traj_indices = np.where(traj_mask)[0]

            traj_meta = step_metadata.get(str(env_slot), [])
            if not traj_meta:
                continue

            total_steps = len(traj_meta)
            necessary_sets = necessary_sets_by_slot[env_slot]
            fact_versions = fact_versions_by_slot[env_slot]

            # Sort each obs slot's versions by step_written and assign version index
            for obs_index, versions in fact_versions.items():
                versions.sort(key=lambda v: v.step_written)
                for j, v in enumerate(versions):
                    v.version = j

            # Pass 2: mark validated uses using citation necessary sets
            for t, necessary_set in necessary_sets.items():
                if not necessary_set or t >= len(traj_meta):
                    continue
                cited_at_t = set(traj_meta[t].get("cited_indices", []))
                for obs_index in cited_at_t & necessary_set:
                    versions = fact_versions.get(obs_index)
                    if not versions:
                        continue
                    active = next((v for v in reversed(versions) if v.step_written < t), None)
                    if active is None:
                        continue
                    m = sum(1 for (u, _) in active.validated_uses if u < t)
                    active.validated_uses.append((t, m))

            # Pass 3: compute T(f) and R_fact per version; accumulate to step rewards
            step_write_rewards = [0.0] * total_steps
            write_debug: dict[int, dict] = {}

            if self.debug_log:
                for t in range(total_steps):
                    facts_list = (
                        fact_debug_skip_by_slot[env_slot].get(t, [])
                        + fact_debug_by_slot[env_slot].get(t, [])
                    )
                    if facts_list:
                        write_debug[t] = {"facts": list(facts_list)}

            for obs_index, versions in fact_versions.items():
                previous: FactVersionRecord | None = None
                for version in versions:
                    version.T = compute_T_version(
                        version.validated_uses, version.step_written, total_steps, self.gamma_discount
                    )
                    version.R_fact = compute_R_fact_version(
                        version, previous, self.lambda_use, self.update_bonus_scale
                    )
                    step_write_rewards[version.step_written] += version.R_fact
                    if self.debug_log and version.step_written < total_steps:
                        for entry in write_debug.get(version.step_written, {}).get("facts", []):
                            if (
                                entry.get("observation_index") == obs_index
                                and entry.get("action") == version.action
                                and not entry.get("error")
                                and not entry.get("skipped")
                            ):
                                entry["T"] = version.T
                                entry["R_fact"] = version.R_fact
                                entry["validated_uses"] = version.validated_uses
                                entry["quality"] = version.quality_score
                                break
                    previous = version

            if self.debug_log:
                for t, detail in write_debug.items():
                    detail["reward"] = step_write_rewards[t]

            step_cite_rewards = [cite_rewards_by_slot[env_slot].get(t, 0.0) for t in range(total_steps)]

            if self.debug_log and len(reward_debug_examples) < self.debug_log_samples:
                debug_steps = []
                for local_step in range(min(total_steps, self.debug_log_samples)):
                    meta = traj_meta[local_step]
                    debug_steps.append({
                        "step_index": meta.get("step_index", local_step),
                        "obs_before": len(meta.get("observations_before", [])),
                        "cited": meta.get("cited_indices", []),
                        "num_writes": len(meta.get("writing_updates", [])),
                        "cite_reward": step_cite_rewards[local_step],
                        "write_reward": step_write_rewards[local_step],
                        "cite_detail": cite_debug_by_slot[env_slot].get(local_step, {}),
                        "write_detail": write_debug.get(local_step, {}),
                    })
                reward_debug_examples.append({
                    "env_slot": env_slot,
                    "traj_uid": traj_uid,
                    "batch_items": traj_indices.tolist(),
                    "num_steps": total_steps,
                    "steps": debug_steps,
                })

            # R_final: judge score in [0,1] for explicit-answer tasks; 1.0 otherwise
            r_final = final_score_by_slot.get(env_slot, 1.0)
            for batch_idx in traj_indices:
                final_rewards[batch_idx] = r_final

            # Assign cite/write rewards and stamp per-step group UIDs
            for local_idx, batch_idx in enumerate(traj_indices):
                if local_idx < len(step_cite_rewards):
                    cite_rewards[batch_idx] = step_cite_rewards[local_idx]
                if local_idx < len(step_write_rewards):
                    write_rewards[batch_idx] = step_write_rewards[local_idx]
                if uids is not None:
                    step_group_uids[batch_idx] = f"{uids[batch_idx]}_{local_idx}"

            # Set scalar reward at last valid token for each item (metric compatibility)
            for local_idx, batch_idx in enumerate(traj_indices):
                combined = (
                    self.beta_cite * cite_rewards[batch_idx]
                    + self.beta_write * write_rewards[batch_idx]
                )
                prompt_length = data.batch["prompts"][batch_idx].shape[-1]
                valid_response_length = data.batch["attention_mask"][batch_idx][prompt_length:].sum().item()
                if valid_response_length > 0:
                    reward_tensor[batch_idx, int(valid_response_length) - 1] = torch.tensor(
                        combined,
                        dtype=reward_tensor.dtype,
                        device=reward_tensor.device,
                    )

        self._log_debug("batch_reward_summary", {
            "batch_size": batch_size,
            "unique_trajs": len(unique_trajs_ordered),
            "metadata_steps": total_meta_steps,
            "total_writing_updates": total_writes,
            "cite_mean": float(cite_rewards.mean()) if batch_size else 0.0,
            "cite_min": float(cite_rewards.min()) if batch_size else 0.0,
            "cite_max": float(cite_rewards.max()) if batch_size else 0.0,
            "cite_nonzero": int(np.count_nonzero(cite_rewards)),
            "write_mean": float(write_rewards.mean()) if batch_size else 0.0,
            "write_min": float(write_rewards.min()) if batch_size else 0.0,
            "write_max": float(write_rewards.max()) if batch_size else 0.0,
            "write_nonzero": int(np.count_nonzero(write_rewards)),
            "examples": reward_debug_examples,
        })

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {
                    "rrg_cite_rewards": cite_rewards.tolist(),
                    "rrg_write_rewards": write_rewards.tolist(),
                    "rrg_final_rewards": final_rewards.tolist(),
                    "rrg_step_group_uid": step_group_uids.tolist(),
                },
            }
        return reward_tensor
