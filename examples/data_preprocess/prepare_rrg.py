"""Prepare parquet dataset for RRG training from trajectory JSON files.

Usage:
    python -m examples.data_preprocess.prepare_rrg \
        --input /path/to/traces.json \
        --local_dir ~/data/verl-agent/rrg \
        --train_ratio 0.9
"""

import argparse
import json
import os
from collections import Counter

import pandas as pd
from PIL import Image


def _parse_size_arg(size_arg: str | None) -> tuple[int, int] | None:
    if size_arg is None:
        return None
    normalized = size_arg.lower().replace(" ", "")
    try:
        width_str, height_str = normalized.split("x", maxsplit=1)
        return int(width_str), int(height_str)
    except ValueError as exc:
        raise ValueError(f"Invalid --target_image_size={size_arg!r}, expected WIDTHxHEIGHT") from exc


def _inspect_trajectory_images(traj: dict) -> tuple[bool, tuple[int, int] | None, str]:
    image_sizes: set[tuple[int, int]] = set()
    for step in traj.get("steps", []):
        image_path = step.get("image")
        if not image_path or not os.path.isfile(image_path):
            return False, None, "missing_image"
        try:
            with Image.open(image_path) as image:
                image_sizes.add(image.size)
        except Exception:
            return False, None, "unreadable_image"

    if not image_sizes:
        return False, None, "empty_trajectory"
    if len(image_sizes) > 1:
        return False, None, "mixed_sizes_within_trajectory"
    return True, next(iter(image_sizes)), "ok"


def main():
    parser = argparse.ArgumentParser(description="Prepare RRG training data")
    parser.add_argument("--input", required=True, help="Path to trajectory JSON file")
    parser.add_argument("--local_dir", default="~/data/verl-agent/rrg")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--train_data_size", type=int, default=None, help="Override train set size")
    parser.add_argument("--val_data_size", type=int, default=None, help="Override val set size")
    parser.add_argument(
        "--target_image_size",
        default=None,
        help="Keep only trajectories whose screenshots are all this size, e.g. 1170x2532. "
        "If omitted, the dominant valid size is selected automatically.",
    )
    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    with open(args.input) as f:
        trajectories = json.load(f)

    size_counter: Counter[tuple[int, int]] = Counter()
    invalid_reason_counter: Counter[str] = Counter()
    inspected: list[tuple[dict, tuple[int, int]]] = []
    for traj in trajectories:
        is_valid, image_size, reason = _inspect_trajectory_images(traj)
        if not is_valid or image_size is None:
            invalid_reason_counter[reason] += 1
            continue
        size_counter[image_size] += 1
        inspected.append((traj, image_size))

    if not inspected:
        raise ValueError("No valid trajectories found after checking image existence and per-trajectory size consistency")

    target_image_size = _parse_size_arg(args.target_image_size)
    if target_image_size is None:
        target_image_size = size_counter.most_common(1)[0][0]

    filtered_trajectories = [traj for traj, image_size in inspected if image_size == target_image_size]
    filtered_out_by_size = len(inspected) - len(filtered_trajectories)

    total = len(filtered_trajectories)
    if total == 0:
        raise ValueError(f"No trajectories remain after filtering for target image size {target_image_size}")

    if args.train_data_size and args.val_data_size:
        n_train = args.train_data_size
        n_val = args.val_data_size
    else:
        n_train = int(total * args.train_ratio)
        n_val = total - n_train

    if n_train + n_val > total:
        raise ValueError(
            f"Requested train+val size {n_train + n_val} exceeds filtered trajectory count {total}"
        )

    train_trajs = filtered_trajectories[:n_train]
    val_trajs = filtered_trajectories[n_train : n_train + n_val]

    print(f"Input trajectories: {len(trajectories)}")
    print(f"Valid trajectories before size filter: {len(inspected)}")
    print(f"Chosen target image size: {target_image_size[0]}x{target_image_size[1]}")
    print(f"Filtered out by image size: {filtered_out_by_size}")
    if invalid_reason_counter:
        print(f"Filtered out by invalid data: {dict(invalid_reason_counter)}")
    print(f"Final trajectories: {total}, train: {len(train_trajs)}, val: {len(val_trajs)}")

    def build_records(trajs, split):
        records = []
        for idx, traj in enumerate(trajs):
            records.append({
                "data_source": "rrg",
                "prompt": [{"role": "user", "content": "<image>"}],
                "ability": "agent",
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "task_id": traj.get("task_id", ""),
                    "env_id": traj.get("env_id", ""),
                },
            })
        return records

    train_df = pd.DataFrame(build_records(train_trajs, "train"))
    val_df = pd.DataFrame(build_records(val_trajs, "test"))

    train_path = os.path.join(local_dir, "train.parquet")
    val_path = os.path.join(local_dir, "test.parquet")
    cleaned_path = os.path.join(local_dir, "traces_clean.json")
    train_traces_path = os.path.join(local_dir, "train_traces.json")
    val_traces_path = os.path.join(local_dir, "test_traces.json")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    with open(cleaned_path, "w") as f:
        json.dump(filtered_trajectories, f)
    with open(train_traces_path, "w") as f:
        json.dump(train_trajs, f)
    with open(val_traces_path, "w") as f:
        json.dump(val_trajs, f)

    print(f"Saved train ({len(train_df)} rows) to {train_path}")
    print(f"Saved val ({len(val_df)} rows) to {val_path}")
    print(f"Saved cleaned trajectories ({len(filtered_trajectories)}) to {cleaned_path}")
    print(f"Saved train trajectories ({len(train_trajs)}) to {train_traces_path}")
    print(f"Saved val trajectories ({len(val_trajs)}) to {val_traces_path}")


if __name__ == "__main__":
    main()
