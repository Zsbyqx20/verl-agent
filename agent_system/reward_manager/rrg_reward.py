"""RRG Reward Manager — computes multi-signal rewards using frozen LLM judges.

After rollout completes, reads step metadata from the environment manager,
calls J_cite and J_fact judges in parallel, and returns per-step citation and
writing rewards via ``reward_extra_info``.
"""
from __future__ import annotations

import base64
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from io import BytesIO

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
    return (
        lambda_val * judgment.correctness
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


# ---------------------------------------------------------------------------
# LLM Judge callers
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def _get_client(base_url: str | None) -> OpenAI:
    client = getattr(_thread_local, "client", None)
    if client is None:
        kwargs = {}
        if base_url:
            kwargs["base_url"] = base_url
        kwargs["api_key"] = os.getenv("OPENAI_API_KEY", "dummy")
        client = OpenAI(**kwargs)
        _thread_local.client = client
    return client


def _encode_image(image_path: str) -> str:
    with Image.open(image_path) as img:
        buf = BytesIO()
        fmt = img.format or "png"
        img.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{b64}"


def _parse_json_response(content: str, schema: type[BaseModel]) -> BaseModel:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model output is not valid JSON: {exc}") from exc
    return schema.model_validate(payload)


def _llm_call_with_retry(fn, max_retries: int, *args, **kwargs):
    delay = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if attempt >= max_retries:
                raise
            time.sleep(delay + random.uniform(0.0, 0.5))
            delay *= 2


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
            top_p=0.8,
            temperature=0.7,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("FactJudgment completion output is empty")
        return _parse_json_response(content, FactJudgment)

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
        """
        batch_size = len(data)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        # Collect step metadata from the environment manager
        step_metadata = getattr(self.envs, "step_metadata", None)
        if step_metadata is None:
            # Fallback: no metadata, return zero rewards
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
        # so the first unique traj_uid encountered belongs to env slot 0, the
        # second to env slot 1, etc.
        traj_uids = data.non_tensor_batch.get("traj_uid", np.arange(batch_size))

        # Build ordered list of unique traj_uids (first-seen order = env-slot order)
        seen: set = set()
        unique_trajs_ordered = []
        for uid in traj_uids:
            if uid not in seen:
                unique_trajs_ordered.append(uid)
                seen.add(uid)

        # Per-item citation and writing rewards
        cite_rewards = np.zeros(batch_size, dtype=np.float32)
        write_rewards = np.zeros(batch_size, dtype=np.float32)
        reward_debug_examples = []
        total_meta_steps = 0
        total_writes = 0

        # Compute rewards per trajectory, mapping env-slot index → metadata
        for env_slot, traj_uid in enumerate(unique_trajs_ordered):
            traj_mask = traj_uids == traj_uid
            traj_indices = np.where(traj_mask)[0]

            traj_meta = step_metadata.get(str(env_slot), [])
            if not traj_meta:
                continue

            total_steps = len(traj_meta)
            total_meta_steps += total_steps
            total_writes += sum(len(meta.get("writing_updates", [])) for meta in traj_meta)

            # Phase 1: Compute citation rewards (call J_cite per step)
            step_cite_rewards, cite_debug = self._compute_citation_rewards(traj_meta, total_steps)

            # Phase 2: Compute writing rewards (call J_fact per written fact)
            step_write_rewards, write_debug = self._compute_writing_rewards(traj_meta, total_steps)

            if self.debug_log and len(reward_debug_examples) < self.debug_log_samples:
                debug_steps = []
                for local_step in range(min(len(traj_meta), self.debug_log_samples)):
                    meta = traj_meta[local_step]
                    debug_steps.append({
                        "step_index": meta.get("step_index", local_step),
                        "obs_before": len(meta.get("observations_before", [])),
                        "cited": meta.get("cited_indices", []),
                        "num_writes": len(meta.get("writing_updates", [])),
                        "cite_reward": step_cite_rewards[local_step] if local_step < len(step_cite_rewards) else 0.0,
                        "write_reward": step_write_rewards[local_step] if local_step < len(step_write_rewards) else 0.0,
                        "cite_detail": cite_debug.get(local_step, {}),
                        "write_detail": write_debug.get(local_step, {}),
                    })
                reward_debug_examples.append({
                    "env_slot": env_slot,
                    "traj_uid": traj_uid,
                    "batch_items": traj_indices.tolist(),
                    "num_steps": total_steps,
                    "steps": debug_steps,
                })

            # Assign to batch indices
            for local_idx, batch_idx in enumerate(traj_indices):
                if local_idx < len(step_cite_rewards):
                    cite_rewards[batch_idx] = step_cite_rewards[local_idx]
                if local_idx < len(step_write_rewards):
                    write_rewards[batch_idx] = step_write_rewards[local_idx]

            # Set scalar reward at last valid token for each item (for metric compatibility)
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
                },
            }
        return reward_tensor

    def _compute_citation_rewards(
        self, traj_meta: list[dict], total_steps: int
    ) -> tuple[list[float], dict[int, dict]]:
        """Call J_cite for each step and compute R_cite_step."""
        step_rewards = []
        debug_details: dict[int, dict] = {}

        with ThreadPoolExecutor(max_workers=self.max_judge_workers) as pool:
            futures = {}
            for t, meta in enumerate(traj_meta):
                task = meta.get("task", "")
                image_path = meta.get("screenshot_path", "")
                action = meta.get("ground_truth_action", "")
                observations_before = meta.get("observations_before", [])
                cited_indices = meta.get("cited_indices", [])

                if not image_path or not os.path.isfile(image_path):
                    step_rewards.append(0.0)
                    debug_details[t] = {
                        "error": "missing_image",
                        "image_path": image_path,
                        "cited": cited_indices,
                    }
                    continue

                fut = pool.submit(
                    call_cite_judge,
                    task, image_path, action, observations_before,
                    self.judge_model, self.judge_base_url, self.max_retries,
                )
                futures[fut] = (t, set(cited_indices))

            # Collect results
            results: dict[int, float] = {}
            for fut in as_completed(futures):
                t, cited_set = futures[fut]
                try:
                    judgment = fut.result()
                    necessary_set = set(judgment.necessary_indices)
                    _, _, _, R = compute_cite_step_reward(
                        cited_set, necessary_set,
                        self.alpha_prec, self.alpha_rec, self.alpha_rep,
                    )
                    results[t] = R
                    if self.debug_log:
                        P, Q, U, _ = compute_cite_step_reward(
                            cited_set, necessary_set,
                            self.alpha_prec, self.alpha_rec, self.alpha_rep,
                        )
                        debug_details[t] = {
                            "cited": sorted(cited_set),
                            "necessary": sorted(necessary_set),
                            "precision": P,
                            "recall": Q,
                            "overcite": U,
                            "reward": R,
                        }
                except Exception as exc:
                    results[t] = 0.0
                    debug_details[t] = {
                        "error": repr(exc),
                        "cited": sorted(cited_set),
                    }
                    self._log_error("cite_judge_failed", {
                        "step_index": t,
                        "error": repr(exc),
                        "image_path": traj_meta[t].get("screenshot_path", ""),
                    })

        return [results.get(t, 0.0) for t in range(len(traj_meta))], debug_details

    def _compute_writing_rewards(
        self, traj_meta: list[dict], total_steps: int
    ) -> tuple[list[float], dict[int, dict]]:
        """Call J_fact for each written fact and compute R_write_step.

        For the initial implementation, we compute quality scores for each
        written fact but skip the future-use (T) computation — that requires
        knowing citation judgments for all future steps, which adds complexity.
        This gives a quality-only signal; T computation will be added in phase 2.
        """
        step_rewards = [0.0] * len(traj_meta)
        debug_details: dict[int, dict] = {}

        with ThreadPoolExecutor(max_workers=self.max_judge_workers) as pool:
            futures = {}
            for t, meta in enumerate(traj_meta):
                task = meta.get("task", "")
                image_path = meta.get("screenshot_path", "")
                observations_before = meta.get("observations_before", [])
                writing_updates = meta.get("writing_updates", [])

                for upd in writing_updates:
                    content = upd.get("observation", "")
                    if not content or not image_path or not os.path.isfile(image_path):
                        if self.debug_log:
                            debug_details.setdefault(t, {"facts": []})["facts"].append({
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
                    futures[fut] = (t, content, upd)

            for fut in as_completed(futures):
                t, content, upd = futures[fut]
                try:
                    judgment = fut.result()
                    quality = compute_fact_quality(
                        judgment,
                        self.lambda_val, self.lambda_atom, self.lambda_gran,
                    )
                    step_rewards[t] += quality
                    if self.debug_log:
                        debug_details.setdefault(t, {"facts": []})["facts"].append({
                            "action": upd.get("action", ""),
                            "observation_index": upd.get("observation_index", None),
                            "content": content[:300],
                            "correctness": judgment.correctness,
                            "atomicity": judgment.atomicity,
                            "granularity": judgment.granularity,
                            "quality": quality,
                            "judge_reasoning": judgment.reasoning[:300],
                        })
                except Exception as exc:
                    if self.debug_log:
                        debug_details.setdefault(t, {"facts": []})["facts"].append({
                            "action": upd.get("action", ""),
                            "observation_index": upd.get("observation_index", None),
                            "content": content[:300],
                            "error": repr(exc),
                        })
                    self._log_error("fact_judge_failed", {
                        "step_index": t,
                        "error": repr(exc),
                        "content": content[:300],
                        "image_path": traj_meta[t].get("screenshot_path", ""),
                    })

        if self.debug_log:
            for t, detail in debug_details.items():
                detail["reward"] = step_rewards[t]
        return step_rewards, debug_details
