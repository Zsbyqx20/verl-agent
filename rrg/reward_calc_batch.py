from __future__ import annotations

import argparse
import base64
import json
import os
import random
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import orjson
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Hyperparameters (copied verbatim from reward_calc.py)
# ---------------------------------------------------------------------------
ALPHA_PREC = 0.4
ALPHA_REC = 0.4
ALPHA_REP = 0.2
ALPHA_NULL = ALPHA_PREC + ALPHA_REC

LAMBDA_USE = 0.4
LAMBDA_VAL = 0.2
LAMBDA_ATOM = 0.2
LAMBDA_GRAN = 0.2
UPDATE_BONUS_SCALE = 0.5

BETA_CITE = 0.5
BETA_WRITE = 0.5

GAMMA = 0.9

# ---------------------------------------------------------------------------
# File name constants
# ---------------------------------------------------------------------------
CHECKPOINT_NAME = "checkpoint.json"
MANIFEST_NAME = "manifest.json"
RESULTS_NAME = "results.jsonl"
FAILURES_NAME = "failures.jsonl"

DEFAULT_MODEL = "doubao-seed-2-0-pro-260215"
EndpointType = Literal["responses", "completions"]

console = Console()
_thread_local = threading.local()

# ---------------------------------------------------------------------------
# Judge output schemas (copied verbatim from reward_calc.py)
# ---------------------------------------------------------------------------


class CitationJudgment(BaseModel):
    reasoning: str
    necessary_indices: list[int]


class FactJudgment(BaseModel):
    correctness: float
    atomicity: float
    granularity: float
    reasoning: str

# ---------------------------------------------------------------------------
# Judge prompts (copied verbatim from reward_calc.py)
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
# Internal data structures (copied verbatim from reward_calc.py)
# ---------------------------------------------------------------------------


@dataclass
class FactVersionRecord:
    obs_index: int
    step_written: int
    version: int
    action: str
    content: str
    judgment: FactJudgment
    validated_uses: list[tuple[int, int]] = field(default_factory=list)
    T: float = 0.0
    R_fact: float = 0.0
    quality_score: float = 0.0
    revision_gain: float = 0.0


@dataclass
class FactRecord:
    obs_index: int
    step_written: int
    latest_step_written: int
    content: str
    judgment: FactJudgment
    versions: list[FactVersionRecord] = field(default_factory=list)
    validated_uses: list[tuple[int, int]] = field(default_factory=list)
    T: float = 0.0
    R_fact: float = 0.0


@dataclass
class StepReward:
    step_index: int
    cited: list[int]
    necessary: list[int]
    redundant: list[int]
    P: float
    Q: float
    U: float
    R_cite_step: float
    new_fact_indices: list[int]

# ---------------------------------------------------------------------------
# Pure reward computation (copied verbatim from reward_calc.py)
# ---------------------------------------------------------------------------


def compute_cite_step_reward(cited: set[int], necessary: set[int], redundant: set[int]) -> tuple[float, float, float, float]:
    P = len(cited & necessary) / max(1, len(cited))
    Q = len(cited & necessary) / max(1, len(necessary))
    U = len(redundant) / max(1, len(cited))
    if not cited and not necessary:
        R = ALPHA_NULL
    else:
        R = ALPHA_PREC * P + ALPHA_REC * Q - ALPHA_REP * U
    return P, Q, U, R


def compute_T(fact: FactRecord, total_steps: int) -> float:
    total = 0.0
    span = max(1, total_steps - 1)
    for step_u, m in fact.validated_uses:
        exponent = (step_u - fact.step_written - 1) / span
        total += (GAMMA ** exponent) / (1 + m)
    return total


def compute_R_fact(fact: FactRecord) -> float:
    return (
        LAMBDA_USE * fact.T
        + LAMBDA_VAL * fact.judgment.correctness
        + LAMBDA_ATOM * fact.judgment.atomicity
        + LAMBDA_GRAN * fact.judgment.granularity
    )


def compute_fact_quality(judgment: FactJudgment) -> float:
    return (
        LAMBDA_VAL * judgment.correctness
        + LAMBDA_ATOM * judgment.atomicity
        + LAMBDA_GRAN * judgment.granularity
    )


def compute_R_fact_version(version: FactVersionRecord, previous: FactVersionRecord | None) -> float:
    if version.T <= 0.0:
        version.revision_gain = 0.0
        return 0.0
    if version.action == "add" or previous is None:
        quality_term = version.quality_score
    else:
        changed = version.content.strip() != previous.content.strip()
        version.revision_gain = max(0.0, version.quality_score - previous.quality_score) if changed else 0.0
        quality_term = UPDATE_BONUS_SCALE * version.revision_gain
    return LAMBDA_USE * version.T + quality_term


def normalize_observation_updates(obs_before: list[str], obs_updates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    current_obs_count = len(obs_before)
    for upd in obs_updates:
        normalized_update = dict(upd)
        action = normalized_update.get("action")
        if action == "add":
            obs_index = normalized_update.get("observation_index")
            if not isinstance(obs_index, int):
                obs_index = current_obs_count
                normalized_update["observation_index"] = obs_index
            current_obs_count = max(current_obs_count, obs_index + 1)
        elif action == "update":
            obs_index = normalized_update.get("observation_index")
            if not isinstance(obs_index, int):
                raise ValueError("When action is update, observation_index must be an integer.")
        normalized.append(normalized_update)
    return normalized

# ---------------------------------------------------------------------------
# Utilities (adapted from belief_state_extract_batch.py)
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def encode_image(image_path: str) -> str:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img_format = img.format or "png"
        img.save(buffered, format=img_format)
        img_bytes = buffered.getvalue()
    base64_data = base64.b64encode(img_bytes).decode("utf-8")
    mime_type = f"image/{img_format.lower()}"
    return f"data:{mime_type};base64,{base64_data}"


def get_client() -> OpenAI:
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"))
        _thread_local.client = client
    return client


def parse_json_response(content: str, schema: type[BaseModel]) -> BaseModel:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model output is not valid JSON: {exc}") from exc
    return schema.model_validate(payload)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    temp_path.replace(path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("ab") as f:
        f.write(orjson.dumps(record))
        f.write(b"\n")

# ---------------------------------------------------------------------------
# Thread-safe judge callers
# ---------------------------------------------------------------------------


def call_cite_judge(
    task: str,
    image_path: str,
    action: dict,
    observations: list[str],
    model: str,
    endpoint: EndpointType,
) -> CitationJudgment:
    obs_text = (
        "empty"
        if not observations
        else "\n".join(f"{i}: {obs}" for i, obs in enumerate(observations))
    )
    user_text = (
        f"Task Goal: {task}\n"
        f"Ground Truth Action: {orjson.dumps(action).decode()}\n"
        f"Observation Bank:\n{obs_text}"
    )
    image_url = encode_image(image_path)
    client = get_client()
    if endpoint == "responses":
        response = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": CITE_JUDGE_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_text},
                        {"type": "input_image", "image_url": image_url},
                    ],
                },  # type: ignore
            ],
            text_format=CitationJudgment,
        )
        result = response.output_parsed
        if result is None:
            raise ValueError("CitationJudgment output is None")
        return result

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CITE_JUDGE_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("CitationJudgment completion output is empty")
    return parse_json_response(content, CitationJudgment)  # type: ignore[return-value]


def call_fact_judge(
    task: str,
    fact_content: str,
    image_path: str,
    observations_before: list[str],
    model: str,
    endpoint: EndpointType,
) -> FactJudgment:
    obs_text = (
        "empty"
        if not observations_before
        else "\n".join(f"{i}: {obs}" for i, obs in enumerate(observations_before))
    )
    user_text = (
        f"Task Goal: {task}\n"
        f"Prior Observation Bank:\n{obs_text}\n"
        f"Newly written observation: \"{fact_content}\""
    )
    image_url = encode_image(image_path)
    client = get_client()
    if endpoint == "responses":
        response = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": FACT_JUDGE_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_text},
                        {"type": "input_image", "image_url": image_url},
                    ],
                },  # type: ignore
            ],
            text_format=FactJudgment,
            top_p=0.8,
            temperature=0.7
        )
        result = response.output_parsed
        if result is None:
            raise ValueError("FactJudgment output is None")
        return result

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": FACT_JUDGE_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        response_format={"type": "json_object"},
        top_p=0.8,
        temperature=0.7
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("FactJudgment completion output is empty")
    return parse_json_response(content, FactJudgment)  # type: ignore[return-value]

# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------


def llm_call_with_retry(fn, max_retries: int, *args, **kwargs):
    delay = 1.0
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries:
                raise
            time.sleep(delay + random.uniform(0.0, 0.5))
            delay *= 2
    raise last_exc  # type: ignore[misc]

# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def fact_version_to_dict(f: FactVersionRecord) -> dict[str, Any]:
    return {
        "obs_index": f.obs_index,
        "step_written": f.step_written,
        "version": f.version,
        "action": f.action,
        "content": f.content,
        "judgment": f.judgment.model_dump(),
        "validated_uses": f.validated_uses,
        "T": f.T,
        "R_fact": f.R_fact,
        "quality_score": f.quality_score,
        "revision_gain": f.revision_gain,
    }


def fact_record_to_dict(f: FactRecord) -> dict[str, Any]:
    return {
        "obs_index": f.obs_index,
        "step_written": f.step_written,
        "latest_step_written": f.latest_step_written,
        "content": f.content,
        "judgment": f.judgment.model_dump(),
        "validated_uses": f.validated_uses,
        "T": f.T,
        "R_fact": f.R_fact,
        "version_count": len(f.versions),
        "versions": [fact_version_to_dict(v) for v in f.versions],
    }


def step_reward_to_dict(sr: StepReward) -> dict[str, Any]:
    return {
        "step_index": sr.step_index,
        "cited": sr.cited,
        "necessary": sr.necessary,
        "redundant": sr.redundant,
        "P": sr.P,
        "Q": sr.Q,
        "U": sr.U,
        "R_cite_step": sr.R_cite_step,
        "new_fact_indices": sr.new_fact_indices,
    }

# ---------------------------------------------------------------------------
# Step-level Pass 1 (cite + fact judges, run concurrently within the step)
# ---------------------------------------------------------------------------


def run_step_pass1(
    task: str,
    step: dict[str, Any],
    max_retries: int,
    model: str,
    endpoint: EndpointType,
) -> tuple[StepReward, list[FactVersionRecord]]:
    """Run J_cite and all J_fact calls for one step.

    All LLM inputs are frozen in the JSONL (observations_before, cited_indices,
    obs_updates, screenshot, action), so this function is independent of other
    steps and safe to run concurrently.
    """
    t: int = step["step_index"]
    image_path: str = step["image"]
    action: dict = step["action"]
    obs_before: list[str] = step["observations_before"]
    cited_indices: list[int] = step["output"]["observation_citation"] or []
    obs_updates = normalize_observation_updates(
        obs_before,
        step["output"]["observation_update"] or [],
    )
    writes = [u for u in obs_updates if u["action"] in {"add", "update"}]

    # Submit cite judge + all fact judges to an ephemeral pool.
    # Using an ephemeral pool (not the shared step_executor) avoids deadlock
    # when run_step_pass1 is itself executing inside step_executor.
    with ThreadPoolExecutor(max_workers=1 + len(writes)) as pool:
        cite_future = pool.submit(
            llm_call_with_retry, call_cite_judge, max_retries,
            task, image_path, action, obs_before, model, endpoint,
        )
        fact_futures: list[tuple[int, str, str, Future]] = []
        for upd in writes:
            obs_index = upd["observation_index"]
            content: str = upd["observation"]
            fact_futures.append((
                obs_index,
                upd["action"],
                content,
                pool.submit(
                    llm_call_with_retry, call_fact_judge, max_retries,
                    task, content, image_path, obs_before, model, endpoint,
                ),
            ))
        # Gather — futures complete inside the `with` block
        cite_j: CitationJudgment = cite_future.result()
        step_facts: list[FactVersionRecord] = []
        new_fact_indices: list[int] = []
        for obs_index, action_name, content, ff in fact_futures:
            fact_j: FactJudgment = ff.result()
            step_facts.append(FactVersionRecord(
                obs_index=obs_index,
                step_written=t,
                version=-1,
                action=action_name,
                content=content,
                judgment=fact_j,
                quality_score=compute_fact_quality(fact_j),
            ))
            if action_name == "add":
                new_fact_indices.append(obs_index)

    cited_set = set(cited_indices)
    necessary_set = set(cite_j.necessary_indices)
    redundant_set = cited_set - necessary_set
    P, Q, U, R_cite_step = compute_cite_step_reward(cited_set, necessary_set, redundant_set)

    step_reward = StepReward(
        step_index=t,
        cited=cited_indices,
        necessary=list(necessary_set),
        redundant=list(redundant_set),
        P=P, Q=Q, U=U,
        R_cite_step=R_cite_step,
        new_fact_indices=new_fact_indices,
    )
    return step_reward, step_facts

# ---------------------------------------------------------------------------
# CheckpointStore
# ---------------------------------------------------------------------------


class CheckpointStore:
    def __init__(
        self,
        run_dir: Path,
        checkpoint_data: dict[str, Any],
        input_path: Path,
    ) -> None:
        self.run_dir = run_dir
        self.checkpoint_path = run_dir / CHECKPOINT_NAME
        self.results_path = run_dir / RESULTS_NAME
        self.failures_path = run_dir / FAILURES_NAME
        self.manifest_path = run_dir / MANIFEST_NAME
        self.input_path = input_path
        self.data = checkpoint_data
        self.lock = threading.Lock()

    def _find_task_state_locked(self, task_index: int) -> dict[str, Any]:
        for task_state in self.data["tasks"]:
            if task_state["task_index"] == task_index:
                return task_state
        raise KeyError(f"Unknown task_index: {task_index}")

    def save_manifest(self, ended_at: str | None = None) -> None:
        with self.lock:
            counts = self.snapshot_counts_locked()
            manifest = {
                "run_id": self.data["run_id"],
                "input_file": self.data["input_file"],
                "model": self.data["model"],
                "endpoint": self.data.get("endpoint", "responses"),
                "workers": self.data["workers"],
                "step_workers": self.data["step_workers"],
                "max_retries": self.data["max_retries"],
                "started_at": self.data["started_at"],
                "ended_at": ended_at,
                "checkpoint_path": str(self.checkpoint_path),
                "results_path": str(self.results_path),
                "failures_path": str(self.failures_path),
                "task_count": self.data["task_count"],
                "counts": counts,
            }
            atomic_write_json(self.manifest_path, manifest)

    def save_checkpoint_locked(self) -> None:
        self.data["saved_at"] = now_iso()
        atomic_write_json(self.checkpoint_path, self.data)

    def snapshot_counts_locked(self) -> dict[str, int]:
        counts = {"pending": 0, "running": 0, "completed": 0, "failed": 0}
        for task_state in self.data["tasks"]:
            status = task_state["status"]
            counts[status] = counts.get(status, 0) + 1
        return counts

    def snapshot_counts(self) -> dict[str, int]:
        with self.lock:
            return self.snapshot_counts_locked()

    def normalize_incomplete_tasks(self) -> None:
        with self.lock:
            for task_state in self.data["tasks"]:
                if task_state["status"] == "running":
                    task_state["status"] = "pending"
            self.save_checkpoint_locked()

    def get_runnable_task_indices(self) -> list[int]:
        with self.lock:
            return [
                task_state["task_index"]
                for task_state in self.data["tasks"]
                if task_state["status"] in {"pending", "running"}
            ]

    def get_task_state_copy(self, task_index: int) -> dict[str, Any]:
        with self.lock:
            task_state = self._find_task_state_locked(task_index)
            return orjson.loads(orjson.dumps(task_state))

    def mark_task_running(self, task_index: int) -> None:
        with self.lock:
            task_state = self._find_task_state_locked(task_index)
            if task_state["started_at"] is None:
                task_state["started_at"] = now_iso()
            task_state["status"] = "running"
            self.save_checkpoint_locked()

    def mark_task_completed(
        self,
        task_index: int,
        step_rewards: list[StepReward],
        facts: dict[int, FactRecord],
        R_cite: float,
        R_write: float,
        R_traj: float,
    ) -> None:
        with self.lock:
            task_state = self._find_task_state_locked(task_index)
            task_state["status"] = "completed"
            task_state["completed_at"] = now_iso()
            task_state["step_rewards"] = [step_reward_to_dict(sr) for sr in step_rewards]
            task_state["facts"] = {str(k): fact_record_to_dict(v) for k, v in facts.items()}
            task_state["R_cite"] = R_cite
            task_state["R_write"] = R_write
            task_state["R_traj"] = R_traj
            if not task_state["result_written"]:
                append_jsonl(self.results_path, self._build_result_record_locked(task_index))
                task_state["result_written"] = True
            self.save_checkpoint_locked()

    def mark_task_failed(self, task_index: int, error_record: dict[str, Any]) -> None:
        with self.lock:
            task_state = self._find_task_state_locked(task_index)
            task_state["status"] = "failed"
            task_state["completed_at"] = now_iso()
            task_state["last_error"] = error_record
            if not task_state["failure_written"]:
                append_jsonl(self.failures_path, self._build_failure_record_locked(task_index))
                task_state["failure_written"] = True
            self.save_checkpoint_locked()

    def _build_result_record_locked(self, task_index: int) -> dict[str, Any]:
        task_state = self._find_task_state_locked(task_index)
        return {
            "run_id": self.data["run_id"],
            "source_file": self.data["input_file"],
            "task_index": task_state["task_index"],
            "task_id": task_state["task_id"],
            "task": task_state["task"],
            "status": "completed",
            "total_steps": task_state["total_steps"],
            "hyperparameters": {
                "ALPHA_PREC": ALPHA_PREC, "ALPHA_REC": ALPHA_REC,
                "ALPHA_REP": ALPHA_REP, "ALPHA_NULL": ALPHA_NULL,
                "LAMBDA_USE": LAMBDA_USE, "LAMBDA_VAL": LAMBDA_VAL,
                "LAMBDA_ATOM": LAMBDA_ATOM, "LAMBDA_GRAN": LAMBDA_GRAN,
                "UPDATE_BONUS_SCALE": UPDATE_BONUS_SCALE,
                "BETA_CITE": BETA_CITE, "BETA_WRITE": BETA_WRITE,
                "GAMMA": GAMMA,
            },
            "step_rewards": task_state["step_rewards"],
            "facts": task_state["facts"],
            "R_cite": task_state["R_cite"],
            "R_write": task_state["R_write"],
            "R_traj": task_state["R_traj"],
            "started_at": task_state["started_at"],
            "completed_at": task_state["completed_at"],
        }

    def _build_failure_record_locked(self, task_index: int) -> dict[str, Any]:
        task_state = self._find_task_state_locked(task_index)
        return {
            "run_id": self.data["run_id"],
            "source_file": self.data["input_file"],
            "task_index": task_state["task_index"],
            "task_id": task_state["task_id"],
            "task": task_state["task"],
            "status": "failed",
            "total_steps": task_state["total_steps"],
            "last_error": task_state["last_error"],
            "started_at": task_state["started_at"],
            "completed_at": task_state["completed_at"],
        }

# ---------------------------------------------------------------------------
# Per-task orchestrator
# ---------------------------------------------------------------------------


def task_label(task_state: dict[str, Any]) -> str:
    task_id = task_state.get("task_id")
    return f"Task {task_id}" if task_id is not None else f"Task #{task_state['task_index']}"


def run_task(
    trajectory: dict[str, Any],
    task_state: dict[str, Any],
    store: CheckpointStore,
    step_executor: ThreadPoolExecutor,
    max_retries: int,
    model: str,
    endpoint: EndpointType,
    progress: Progress,
    progress_task_id: TaskID,
) -> dict[str, Any]:
    label = task_label(task_state)
    task_index: int = task_state["task_index"]
    task: str = trajectory["task"]
    steps: list[dict] = trajectory["completed_steps"]
    total: int = len(steps)

    # Build a dict keyed by step_index to handle any gaps safely.
    steps_by_index = {s["step_index"]: s for s in steps}

    store.mark_task_running(task_index)

    try:
        # --- Pass 1: dispatch all steps concurrently to the shared step_executor ---
        # Steps are independent for reward evaluation: all inputs (observations_before,
        # cited_indices, obs_updates, screenshot) are frozen in the JSONL.
        step_futures: dict[Future, int] = {
            step_executor.submit(run_step_pass1, task, step, max_retries, model, endpoint): step["step_index"]
            for step in steps
        }
        step_results: dict[int, tuple[StepReward, list[FactVersionRecord]]] = {}
        for future in as_completed(step_futures):
            t = step_futures[future]
            step_reward, step_facts = future.result()  # raises on LLM failure
            step_results[t] = (step_reward, step_facts)
            progress.update(
                progress_task_id,
                advance=1,
                description=f"{label} step {len(step_results)}/{total}",
            )

        # Merge in step-index order
        step_rewards: list[StepReward] = []
        fact_versions: dict[int, list[FactVersionRecord]] = {}
        cite_judgments: dict[int, CitationJudgment] = {}
        for t in sorted(step_results):
            sr, sf = step_results[t]
            step_rewards.append(sr)
            for version in sf:
                versions = fact_versions.setdefault(version.obs_index, [])
                version.version = len(versions)
                versions.append(version)
            cite_judgments[t] = CitationJudgment(
                reasoning="",
                necessary_indices=sr.necessary,
            )

        # --- Pass 2: compute T(f) for each versioned fact (pure math, no LLM) ---
        for t, cite_j in cite_judgments.items():
            necessary_set = set(cite_j.necessary_indices)
            step = steps_by_index[t]
            cited_at_step = set(step["output"]["observation_citation"] or [])
            validated_at_step = cited_at_step & necessary_set
            for obs_index in validated_at_step:
                versions = fact_versions.get(obs_index)
                if not versions:
                    continue
                active_version = None
                for version in versions:
                    if version.step_written < t:
                        active_version = version
                    else:
                        break
                if active_version is None:
                    continue
                m = sum(1 for (u, _) in active_version.validated_uses if u < t)
                active_version.validated_uses.append((t, m))

        facts: dict[int, FactRecord] = {}
        for obs_index, versions in fact_versions.items():
            previous_version: FactVersionRecord | None = None
            for version in versions:
                version.T = compute_T(version, total)  # type: ignore
                version.R_fact = compute_R_fact_version(version, previous_version)
                previous_version = version
            latest = versions[-1]
            aggregated_uses = [use for version in versions for use in version.validated_uses]
            facts[obs_index] = FactRecord(
                obs_index=obs_index,
                step_written=versions[0].step_written,
                latest_step_written=latest.step_written,
                content=latest.content,
                judgment=latest.judgment,
                versions=versions,
                validated_uses=aggregated_uses,
                T=sum(version.T for version in versions),
                R_fact=sum(version.R_fact for version in versions),
            )

        # --- Aggregation (divide by total for length-invariance) ---
        R_cite = sum(sr.R_cite_step for sr in step_rewards) / total
        R_write = sum(f.R_fact for f in facts.values()) / total
        R_traj = BETA_CITE * R_cite + BETA_WRITE * R_write

        store.mark_task_completed(task_index, step_rewards, facts, R_cite, R_write, R_traj)
        progress.update(progress_task_id, description=f"{label} [green](completed)[/green]")
        return {"task_index": task_index, "status": "completed"}

    except Exception as exc:
        error_record = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "failed_at": now_iso(),
        }
        store.mark_task_failed(task_index, error_record)
        progress.update(progress_task_id, description=f"{label} [red](failed)[/red]")
        return {"task_index": task_index, "status": "failed"}

# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def load_trajectories(input_path: Path) -> list[dict[str, Any]]:
    trajectories: list[dict[str, Any]] = []
    with input_path.open("rb") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = orjson.loads(line)
            except orjson.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_number}: {e}") from e
            for key in ("task", "completed_steps", "task_index"):
                if key not in record:
                    raise ValueError(
                        f"Record on line {line_number} missing required key '{key}'"
                    )
            trajectories.append(record)
    return trajectories


def build_trajectory_map(trajectories: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    by_task_index: dict[int, dict[str, Any]] = {}
    for trajectory in trajectories:
        task_index = trajectory["task_index"]
        if task_index in by_task_index:
            raise ValueError(f"Duplicate task_index in input: {task_index}")
        by_task_index[task_index] = trajectory
    return by_task_index


def build_task_state(trajectory: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_index": trajectory["task_index"],
        "task_id": trajectory.get("task_id"),
        "task": trajectory["task"],
        "status": "pending",
        "total_steps": trajectory.get("total_steps", len(trajectory["completed_steps"])),
        "step_rewards": [],
        "facts": {},
        "R_cite": None,
        "R_write": None,
        "R_traj": None,
        "last_error": None,
        "started_at": None,
        "completed_at": None,
        "result_written": False,
        "failure_written": False,
    }

# ---------------------------------------------------------------------------
# Store initialisation
# ---------------------------------------------------------------------------


def create_run_dir(run_dir_arg: str | None) -> Path:
    if run_dir_arg:
        run_dir = Path(run_dir_arg)
    else:
        run_id = datetime.now().astimezone().strftime("reward_calc_%Y%m%d_%H%M%S")
        run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def create_checkpoint_data(
    input_path: Path,
    run_dir: Path,
    trajectories: list[dict[str, Any]],
    workers: int,
    step_workers: int,
    max_retries: int,
    model: str,
    endpoint: EndpointType,
) -> dict[str, Any]:
    return {
        "run_id": run_dir.name,
        "input_file": str(input_path),
        "model": model,
        "endpoint": endpoint,
        "workers": workers,
        "step_workers": step_workers,
        "max_retries": max_retries,
        "started_at": now_iso(),
        "saved_at": None,
        "task_count": len(trajectories),
        "tasks": [build_task_state(t) for t in trajectories],
    }


def load_or_create_store(
    input_path: Path,
    run_dir: Path,
    workers: int,
    step_workers: int,
    max_retries: int,
    model: str | None,
    endpoint: EndpointType | None,
    resume: bool,
    limit: int | None,
) -> tuple[CheckpointStore, list[dict[str, Any]]]:
    trajectories = load_trajectories(input_path)
    if limit is not None:
        trajectories = trajectories[:limit]
    checkpoint_path = run_dir / CHECKPOINT_NAME

    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint_data = orjson.loads(checkpoint_path.read_bytes())
        if Path(checkpoint_data["input_file"]) != input_path:
            raise ValueError(
                f"Input file mismatch for resume: checkpoint uses "
                f"{checkpoint_data['input_file']}, but CLI provided {input_path}"
            )
        if checkpoint_data["task_count"] != len(trajectories):
            raise ValueError(
                f"Task count mismatch: checkpoint has {checkpoint_data['task_count']} tasks "
                f"but {len(trajectories)} were loaded (check --limit)."
            )
        checkpoint_model = checkpoint_data.get("model", DEFAULT_MODEL)
        checkpoint_endpoint = checkpoint_data.get("endpoint", "responses")
        if model is None:
            model = checkpoint_model
        elif checkpoint_model != model:
            raise ValueError(
                f"Model mismatch for resume: checkpoint uses {checkpoint_model}, "
                f"but CLI provided {model}"
            )
        if endpoint is None:
            endpoint = checkpoint_endpoint
        elif checkpoint_endpoint != endpoint:
            raise ValueError(
                f"Endpoint mismatch for resume: checkpoint uses {checkpoint_endpoint}, "
                f"but CLI provided {endpoint}"
            )
    else:
        if checkpoint_path.exists():
            raise FileExistsError(
                f"Checkpoint already exists in {run_dir}. "
                "Use --resume or choose a new --run-dir."
            )
        model = model or DEFAULT_MODEL
        endpoint = endpoint or "responses"
        checkpoint_data = create_checkpoint_data(
            input_path, run_dir, trajectories, workers, step_workers, max_retries, model, endpoint
        )

    assert model is not None
    assert endpoint is not None

    store = CheckpointStore(run_dir=run_dir, checkpoint_data=checkpoint_data, input_path=input_path)
    if resume:
        store.normalize_incomplete_tasks()
    else:
        store.save_manifest()
        with store.lock:
            store.save_checkpoint_locked()
    return store, trajectories

# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------


def overall_description(counts: dict[str, int]) -> str:
    return (
        "[bold cyan]Batch progress[/bold cyan] "
        f"pending={counts['pending']} "
        f"running={counts['running']} "
        f"completed={counts['completed']} "
        f"failed={counts['failed']}"
    )


def submit_task(
    executor: ThreadPoolExecutor,
    step_executor: ThreadPoolExecutor,
    progress: Progress,
    trajectory: dict[str, Any],
    task_state: dict[str, Any],
    store: CheckpointStore,
    max_retries: int,
    model: str,
    endpoint: EndpointType,
) -> tuple[Future[dict[str, Any]], TaskID]:
    task_bar_id = progress.add_task(
        description=f"{task_label(task_state)} (0/{task_state['total_steps']} steps)",
        total=task_state["total_steps"],
        completed=0,
    )
    future = executor.submit(
        run_task,
        trajectory,
        task_state,
        store,
        step_executor,
        max_retries,
        model,
        endpoint,
        progress,
        task_bar_id,
    )
    return future, task_bar_id

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch reward calculation runner")
    parser.add_argument("--input", required=True, help="Path to results.jsonl from belief_state_extract_batch")
    parser.add_argument("--run-dir", help="Output directory for artifacts (auto-generated if omitted)")
    parser.add_argument("--model", help=f"Model name to query (default: {DEFAULT_MODEL})")
    parser.add_argument("--endpoint", choices=("responses", "completions"), help="API endpoint to use (default: responses)")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent task workers (default: 8)")
    parser.add_argument("--step-workers", type=int, default=16, help="Shared thread pool size for step-level LLM calls (default: 16)")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per LLM call (default: 3)")
    parser.add_argument("--limit", type=int, help="Only process the first N trajectories")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing checkpoint (requires --run-dir)")
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    run_dir = create_run_dir(args.run_dir)
    store, trajectories = load_or_create_store(
        input_path=input_path,
        run_dir=run_dir,
        workers=args.workers,
        step_workers=args.step_workers,
        max_retries=args.max_retries,
        model=args.model,
        endpoint=args.endpoint,
        resume=args.resume,
        limit=args.limit,
    )
    selected_model: str = store.data["model"]
    selected_endpoint: EndpointType = store.data.get("endpoint", "responses")
    trajectories_by_task_index = build_trajectory_map(trajectories)

    runnable_indices = store.get_runnable_task_indices()
    initial_counts = store.snapshot_counts()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )

    with progress:
        overall_task_id = progress.add_task(
            description=overall_description(initial_counts),
            total=len(trajectories),
            completed=initial_counts["completed"] + initial_counts["failed"],
        )

        if not runnable_indices:
            progress.update(overall_task_id, description=overall_description(initial_counts))
        else:
            next_pointer = 0
            running: dict[Future[dict[str, Any]], TaskID] = {}

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                with ThreadPoolExecutor(max_workers=args.step_workers) as step_executor:
                    # Initial fill
                    while next_pointer < len(runnable_indices) and len(running) < args.workers:
                        task_index = runnable_indices[next_pointer]
                        task_state = store.get_task_state_copy(task_index)
                        future, task_bar_id = submit_task(
                            executor=executor,
                            step_executor=step_executor,
                            progress=progress,
                            trajectory=trajectories_by_task_index[task_index],
                            task_state=task_state,
                            store=store,
                            max_retries=args.max_retries,
                            model=selected_model,
                            endpoint=selected_endpoint,
                        )
                        running[future] = task_bar_id
                        next_pointer += 1
                        progress.update(overall_task_id, description=overall_description(store.snapshot_counts()))

                    # Sliding window
                    while running:
                        completed_futures, _ = wait(running.keys(), return_when=FIRST_COMPLETED)
                        for future in completed_futures:
                            task_bar_id = running.pop(future)
                            future.result()
                            progress.update(task_bar_id, visible=False)
                            counts = store.snapshot_counts()
                            progress.update(
                                overall_task_id,
                                completed=counts["completed"] + counts["failed"],
                                description=overall_description(counts),
                            )

                            if next_pointer < len(runnable_indices):
                                task_index = runnable_indices[next_pointer]
                                task_state = store.get_task_state_copy(task_index)
                                next_future, next_task_bar_id = submit_task(
                                    executor=executor,
                                    step_executor=step_executor,
                                    progress=progress,
                                    trajectory=trajectories_by_task_index[task_index],
                                    task_state=task_state,
                                    store=store,
                                    max_retries=args.max_retries,
                                    model=selected_model,
                                    endpoint=selected_endpoint,
                                )
                                running[next_future] = next_task_bar_id
                                next_pointer += 1
                                progress.update(
                                    overall_task_id,
                                    description=overall_description(store.snapshot_counts()),
                                )

    store.save_manifest(ended_at=now_iso())
    final_counts = store.snapshot_counts()
    console.print(
        f"[green]Finished.[/green] completed={final_counts['completed']} "
        f"failed={final_counts['failed']} run_dir={run_dir}"
    )


if __name__ == "__main__":
    main()
