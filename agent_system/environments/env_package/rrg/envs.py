# Copyright 2025 the ReverseReasoningGenerator (RRG) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""In-process vectorized REPLAY env for reverse-reasoning RL (AMEX navigation toy).

Unlike the live gym envs (sokoban/webshop), RRG trajectories are FIXED: the action is given
and the policy only generates the reasoning. So this env does NOT simulate anything — it
replays recorded `(marked_screenshot, GT action, goal)` frames step by step. Because every
rollout of a task replays the IDENTICAL frame sequence, all G group-mates share the same
state at every step → GiGPO's anchor-state grouping is exact (anchor = (task_id, step_idx)).

No Ray actors: replay is just indexing into preloaded frames, so a plain in-process vector is
faster and simpler. Reward is NOT computed here (the env never sees the generated reasoning) —
RRGEnvironmentManager.step computes the per-step action-recovery margin from the policy output.

Contract mirrors SokobanMultiProcessEnv: reset()->(obs_list, info_list);
step(actions)->(obs_list, reward_list, done_list, info_list); attrs mode/num_processes/group_n/env_num.
"""
from __future__ import annotations

import ast
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _parse_action(a) -> dict:
    return a if isinstance(a, dict) else ast.literal_eval(a)


def _load_rrg_episodes(task_root: str, num_episodes: int | None) -> List[Dict[str, Any]]:
    """Load native RRG tasks (answer-bearing) from a task_root directory.

    Layout: task_root/tasks.json (per-task `steps` with action + image rel-path),
    task_root/rrg-source.csv (gold answer in 'Validate 备注', goal+lang), and
    task_root/params_cache.json (required-output JSON Schema keyed 'env:tid'). Each
    frame additionally carries `goal, gold, schema` (identical across a task's frames)
    so the trajectory reward manager can score generative answer-recovery.

    Unlike AMEX, RRG tasks.json leaves per-step `reasoning` EMPTY (gold reasoning lives in
    the SFT jsonl, not needed here -- the policy generates the reasoning). So we keep every
    step that has a parseable action and an on-disk screenshot; we do NOT filter on reasoning.
    Only tasks with both a gold answer and a schema are kept (answer-recovery needs both).
    """
    from agent_system.environments.env_package.rrg import answer_recovery as A

    root = Path(task_root)
    gold_csv = A.auto_gold_csv(root)
    schema_cache = json.loads((root / "params_cache.json").read_text(encoding="utf-8"))
    raw = json.loads((root / "tasks.json").read_text(encoding="utf-8"))
    raw.sort(key=lambda t: (str(t["env_id"]), str(t.get("task_id"))))  # deterministic

    episodes = []
    for t in raw:
        env, tid = t["env_id"], t.get("task_id")
        key = f"{env}-{tid}"
        row = A._csv_row(gold_csv, env, tid)
        goal, lang = A.resolve_goal_and_lang(row, t.get("task"))
        gold = A.load_gold(gold_csv, env, tid)
        schema = schema_cache.get(f"{env}:{tid}")
        if not (goal and gold and schema):
            continue
        frames = []
        for s in t["steps"]:
            try:
                action = _parse_action(s["action"])
            except (ValueError, SyntaxError):
                continue
            img = root / s["image"]
            if not img.is_file():
                continue
            frames.append({
                "task_id": key,
                "step_idx": len(frames),          # contiguous 0-based replay index
                "goal": goal,
                "action": action,
                "image_path": str(img),
                "lang": lang,
                "gold": gold,
                "schema": schema,
            })
        if frames:
            for fr in frames:
                fr["num_steps"] = len(frames)
            episodes.append({"task_id": key, "frames": frames,
                             "gold": gold, "schema": schema, "goal": goal})

    episodes.sort(key=lambda e: e["task_id"])  # deterministic order
    if num_episodes is not None and num_episodes > 0:
        episodes = episodes[:num_episodes]
    return episodes


def _load_episodes(data_jsonl: str, image_root: str, num_episodes: int | None) -> List[Dict[str, Any]]:
    """Group recorded steps into ordered episodes. Each frame carries everything the manager
    needs to render the prompt and score the step."""
    rows_by_task: Dict[str, list] = defaultdict(list)
    with open(data_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows_by_task[str(r["task_id"])].append(r)

    episodes = []
    for task_id, rows in rows_by_task.items():
        rows.sort(key=lambda r: int(r["step"]))
        frames = []
        for r in rows:
            reasoning = (r.get("reasoning") or "").strip()
            if not reasoning:
                continue  # keep only steps that have a gold reasoning (a learnable step)
            step = int(r["step"])
            img = Path(image_root) / f"amex-{task_id}" / f"step_{step:03d}.png"
            if not img.is_file():
                continue
            frames.append({
                "task_id": task_id,
                "step_idx": step,
                "goal": r["goal"],
                "action": _parse_action(r["action"]),
                "image_path": str(img),
                "lang": r.get("lang", "en"),
                "gold_reasoning": reasoning,
            })
        if frames:
            for fr in frames:
                fr["num_steps"] = len(frames)
            episodes.append({"task_id": task_id, "frames": frames})

    episodes.sort(key=lambda e: e["task_id"])  # deterministic order
    if num_episodes is not None and num_episodes > 0:
        episodes = episodes[:num_episodes]
    return episodes


class RRGReplayVecEnv:
    def __init__(self, seed=0, env_num=1, group_n=1, is_train=True, env_kwargs=None):
        env_kwargs = env_kwargs or {}
        self.mode = "rgb_array"  # always multimodal (marked screenshots)
        self.is_train = is_train
        self.env_num = env_num
        self.group_n = group_n
        self.num_processes = env_num * group_n
        self._rng = np.random.RandomState(seed)

        data_kind = env_kwargs.get("data_kind", "amex")
        if data_kind == "rrg":
            self.episodes = _load_rrg_episodes(
                env_kwargs["task_root"], env_kwargs.get("num_episodes"))
        else:
            self.episodes = _load_episodes(
                env_kwargs["data_jsonl"], env_kwargs["image_root"], env_kwargs.get("num_episodes"))
        if not self.episodes:
            raise RuntimeError(f"RRGReplayVecEnv: no episodes loaded (data_kind={data_kind})")

        self._assign: List[int] = []   # episode index per slot
        self._ptr: List[int] = []      # frame pointer per slot

    # ----- helpers ----- #
    def _frame(self, slot: int) -> Dict[str, Any]:
        ep = self.episodes[self._assign[slot]]
        frames = ep["frames"]
        ptr = min(self._ptr[slot], len(frames) - 1)
        return frames[ptr]

    def _select_episodes(self) -> List[int]:
        n = self.env_num
        if self.is_train:
            idx = self._rng.randint(0, len(self.episodes), size=n).tolist()
        else:
            # deterministic, wrap if fewer episodes than env_num
            idx = [(i % len(self.episodes)) for i in range(n)]
        # repeat group_n consecutively so each block of G slots = one task (the uid group)
        return np.repeat(np.array(idx), self.group_n).tolist()

    # ----- gym-like API ----- #
    def reset(self):
        self._assign = self._select_episodes()
        self._ptr = [0] * self.num_processes
        obs_list = [self._frame(s) for s in range(self.num_processes)]
        info_list = [{} for _ in range(self.num_processes)]
        return obs_list, info_list

    def step(self, actions):
        assert len(actions) == self.num_processes
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for s in range(self.num_processes):
            n_steps = self.episodes[self._assign[s]]["frames"][0]["num_steps"]
            self._ptr[s] += 1
            done = self._ptr[s] >= n_steps
            obs_list.append(self._frame(s))           # clamped to last frame when done
            reward_list.append(0.0)                    # margin reward added by the manager
            done_list.append(done)
            info_list.append({"won": bool(done)})      # toy: recorded trajs are gold-successful
        return obs_list, np.array(reward_list, dtype=np.float32), np.array(done_list, dtype=bool), info_list

    def render(self, mode=None, env_idx=None):
        if env_idx is not None:
            return self._frame(env_idx)["image_path"]
        return [self._frame(s)["image_path"] for s in range(self.num_processes)]

    def close(self):
        pass


def build_rrg_envs(seed=0, env_num=1, group_n=1, is_train=True, env_kwargs=None):
    return RRGReplayVecEnv(seed=seed, env_num=env_num, group_n=group_n,
                           is_train=is_train, env_kwargs=env_kwargs)
