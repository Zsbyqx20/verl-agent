from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from PIL import Image


class RRGReplayEnv:
    """Trajectory replay environment for the Reverse Reasoning Generator.

    Loads pre-recorded successful GUI trajectories from a JSON file and
    replays them step-by-step. The model generates reasoning (not actions);
    the environment simply advances to the next pre-recorded step.

    Expected JSON format — a list of trajectory dicts::

        [{
            "task": "...",
            "env_id": "task_031",
            "task_id": "031",
            "steps": [
                {"action": {"action": "click", "coordinate": [x, y]},
                 "reasoning": "",
                 "image": "/abs/path/to/screenshot.png"},
                ...
                {"action": {"action": "terminate", "status": "success",
                            "content": {...}}, ...}
            ]
        }]
    """

    def __init__(
        self,
        trajectory_data_path: str,
        seed: int,
        env_num: int,
        group_n: int,
        is_train: bool = True,
    ):
        self.batch_size = env_num * group_n
        self.group_n = group_n
        self.env_num = env_num
        self.is_train = is_train
        self._rng = np.random.RandomState(seed)

        # Load trajectory data
        with open(trajectory_data_path) as f:
            self._all_trajectories: list[dict[str, Any]] = json.load(f)
        assert len(self._all_trajectories) > 0, "No trajectories found"

        # Per-env state
        self._current_trajs: list[dict[str, Any] | None] = [None] * self.batch_size
        self._step_idx: list[int] = [0] * self.batch_size
        self._done: list[bool] = [False] * self.batch_size

    def _sample_trajectory(self) -> dict[str, Any]:
        idx = self._rng.randint(0, len(self._all_trajectories))
        return self._all_trajectories[idx]

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load a screenshot as an RGB numpy array."""
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Screenshot not found: {image_path}")
        img = Image.open(image_path).convert("RGB")
        return np.array(img)

    def reset(self, kwargs: list[dict] | None = None) -> tuple[list[str], list[np.ndarray], list[dict]]:
        """Reset all environments, loading new trajectories.

        For grouped rollouts, the same source trajectory is shared across
        ``group_n`` consecutive env slots so that different rollouts operate
        on the same trajectory (enabling group-based normalization).

        Returns:
            text_obs: list of text observations (task + ground-truth action info)
            image_obs: list of screenshot numpy arrays
            infos: list of info dicts
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []

        for base_idx in range(self.env_num):
            traj = self._sample_trajectory()
            for g in range(self.group_n):
                i = base_idx * self.group_n + g
                self._current_trajs[i] = traj
                self._step_idx[i] = 0
                self._done[i] = False

                step_data = traj["steps"][0]
                before_image = self._load_image(step_data["image"])
                action_str = json.dumps(step_data["action"])

                has_after = len(traj["steps"]) > 1
                if has_after:
                    after_image = self._load_image(traj["steps"][1]["image"])
                    images = [before_image, after_image]
                else:
                    images = [before_image]

                text_obs_list.append(traj["task"])
                image_obs_list.append(images)
                info_list.append({
                    "won": False,
                    "task": traj["task"],
                    "task_id": traj.get("task_id", ""),
                    "env_id": traj.get("env_id", ""),
                    "step_index": 0,
                    "total_steps": len(traj["steps"]),
                    "ground_truth_action": action_str,
                    "screenshot_path": step_data["image"],
                    "has_after_image": has_after,
                    "after_screenshot_path": traj["steps"][1]["image"] if has_after else "",
                    "step_data": step_data,
                })

        return text_obs_list, image_obs_list, info_list

    def step(self, text_actions: list[str]) -> tuple[list[str], list[list[np.ndarray]], list[float], list[bool], list[dict]]:
        """Advance to the next pre-recorded step.

        Args:
            text_actions: Model's reasoning outputs (one per env). These are
                stored in infos for downstream reward computation but do not
                affect the trajectory replay.

        Returns:
            text_obs, image_obs, rewards, dones, infos
        """
        text_obs_list = []
        image_obs_list = []
        rewards = []
        dones = []
        info_list = []

        for i in range(self.batch_size):
            traj = self._current_trajs[i]
            assert traj is not None

            if self._done[i]:
                last_step = traj["steps"][-1]
                image = self._load_image(last_step["image"])
                text_obs_list.append(traj["task"])
                image_obs_list.append([image])
                rewards.append(0.0)
                dones.append(True)

                terminate_action = last_step.get("action", {})
                has_answer = (
                    terminate_action.get("action") == "terminate"
                    and terminate_action.get("content") is not None
                )

                info_list.append({
                    "won": True,
                    "model_output": text_actions[i],
                    "step_index": len(traj["steps"]) - 1,
                    "total_steps": len(traj["steps"]),
                    "step_data": last_step,
                    "screenshot_path": last_step["image"],
                    "ground_truth_action": json.dumps(last_step["action"]),
                    "has_after_image": False,
                    "after_screenshot_path": "",
                    "has_answer": has_answer,
                    "terminate_content": terminate_action.get("content") if has_answer else None,
                })
                continue

            # Record model output for reward computation
            current_step_data = traj["steps"][self._step_idx[i]]

            # Advance step
            self._step_idx[i] += 1
            next_step_idx = self._step_idx[i]

            if next_step_idx >= len(traj["steps"]):
                # Trajectory complete
                self._done[i] = True
                # Reuse last screenshot (no after-action image at end of trajectory)
                last_step = traj["steps"][-1]
                image = self._load_image(last_step["image"])
                text_obs_list.append(traj["task"])
                image_obs_list.append([image])
                rewards.append(0.0)
                dones.append(True)

                # Check if trajectory has an explicit answer
                terminate_action = last_step.get("action", {})
                has_answer = (
                    terminate_action.get("action") == "terminate"
                    and terminate_action.get("content") is not None
                )

                info_list.append({
                    "won": True,  # all trajectories are successful
                    "model_output": text_actions[i],
                    "step_index": next_step_idx - 1,
                    "total_steps": len(traj["steps"]),
                    "step_data": current_step_data,
                    "screenshot_path": current_step_data["image"],
                    "ground_truth_action": json.dumps(current_step_data["action"]),
                    "has_after_image": False,
                    "after_screenshot_path": "",
                    "has_answer": has_answer,
                    "terminate_content": terminate_action.get("content") if has_answer else None,
                })
            else:
                # Serve next step
                next_step = traj["steps"][next_step_idx]
                before_image = self._load_image(next_step["image"])
                action_str = json.dumps(next_step["action"])

                # Include after-action screenshot if the step after next exists
                has_after = next_step_idx + 1 < len(traj["steps"])
                if has_after:
                    after_step = traj["steps"][next_step_idx + 1]
                    after_image = self._load_image(after_step["image"])
                    images = [before_image, after_image]
                    after_screenshot_path = after_step["image"]
                else:
                    images = [before_image]
                    after_screenshot_path = ""

                text_obs_list.append(traj["task"])
                image_obs_list.append(images)
                rewards.append(0.0)
                dones.append(False)

                info_list.append({
                    "won": False,
                    "model_output": text_actions[i],
                    "step_index": next_step_idx,
                    "total_steps": len(traj["steps"]),
                    "step_data": current_step_data,
                    "screenshot_path": current_step_data["image"],
                    # current step's action (not next step's) — for judge evaluation
                    "ground_truth_action": json.dumps(current_step_data["action"]),
                    "next_screenshot_path": next_step["image"],
                    # next step's action — for constructing next prompt in env manager
                    "next_ground_truth_action": action_str,
                    "has_after_image": has_after,
                    "after_screenshot_path": after_screenshot_path,
                })

        return (
            text_obs_list,
            image_obs_list,
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            info_list,
        )

    def close(self):
        pass


def build_rrg_envs(
    trajectory_data_path: str,
    seed: int,
    env_num: int,
    group_n: int,
    is_train: bool = True,
    **kwargs,
) -> RRGReplayEnv:
    """Factory function to create an RRG replay environment."""
    return RRGReplayEnv(
        trajectory_data_path=trajectory_data_path,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
    )
