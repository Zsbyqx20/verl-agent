"""Prepare parquet dataset for RRG training from trajectory JSON files.

Supports both JSON array and JSONL input formats. Trajectories using the
``completed_steps`` field name are automatically normalised to ``steps``.
Images that do not match the target resolution are resized and saved to
``{local_dir}/resized_images/``; the output traces JSON references the
resized copies so that ``RRGReplayEnv`` receives consistently-sized images.

Usage:
    python -m examples.data_preprocess.prepare_rrg \
        --input /path/to/traces.json \
        --local_dir ~/data/verl-agent/rrg \
        --train_ratio 0.9
"""

import argparse
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm


def _parse_size_arg(size_arg: str | None) -> tuple[int, int] | None:
    if size_arg is None:
        return None
    normalized = size_arg.lower().replace(" ", "")
    try:
        width_str, height_str = normalized.split("x", maxsplit=1)
        return int(width_str), int(height_str)
    except ValueError as exc:
        raise ValueError(f"Invalid --target_image_size={size_arg!r}, expected WIDTHxHEIGHT") from exc


def _load_trajectories(input_path: str) -> list[dict]:
    """Load trajectories from a JSON array or JSONL file."""
    with open(input_path) as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def _iter_unique_image_paths(trajectories: list[dict]) -> list[str]:
    seen: set[str] = set()
    unique_paths: list[str] = []
    for traj in trajectories:
        for step in traj.get("steps", []):
            image_path = step.get("image")
            if image_path and image_path not in seen:
                seen.add(image_path)
                unique_paths.append(image_path)
    return unique_paths


def _normalize_trajectory(traj: dict) -> dict:
    """Rename ``completed_steps`` → ``steps`` if the latter is absent."""
    if "completed_steps" in traj and "steps" not in traj:
        traj = dict(traj)
        traj["steps"] = traj.pop("completed_steps")
    return traj


def _validate_trajectory(traj: dict) -> tuple[bool, str]:
    """Return (True, 'ok') when all step images exist and are readable."""
    steps = traj.get("steps", [])
    if not steps:
        return False, "empty_trajectory"
    for step in steps:
        image_path = step.get("image")
        if not image_path or not os.path.isfile(image_path):
            return False, "missing_image"
        try:
            with Image.open(image_path):
                pass
        except Exception:
            return False, "unreadable_image"
    return True, "ok"


def _validate_trajectory_with_image_info(
    traj: dict, image_info: dict[str, tuple[bool, tuple[int, int] | None]]
) -> tuple[bool, str]:
    """Return (True, 'ok') using precomputed image existence/readability metadata."""
    steps = traj.get("steps", [])
    if not steps:
        return False, "empty_trajectory"
    for step in steps:
        image_path = step.get("image")
        if not image_path:
            return False, "missing_image"
        ok, _ = image_info.get(image_path, (False, None))
        if not ok:
            return False, "missing_image" if not os.path.isfile(image_path) else "unreadable_image"
    return True, "ok"


def _inspect_image(image_path: str) -> tuple[str, bool, tuple[int, int] | None]:
    if not image_path or not os.path.isfile(image_path):
        return image_path, False, None
    try:
        with Image.open(image_path) as img:
            return image_path, True, img.size
    except Exception:
        return image_path, False, None


def _scan_images(image_paths: list[str], workers: int) -> dict[str, tuple[bool, tuple[int, int] | None]]:
    results: dict[str, tuple[bool, tuple[int, int] | None]] = {}
    if workers <= 1:
        iterator = map(_inspect_image, image_paths)
        for image_path, ok, size in tqdm(iterator, total=len(image_paths), desc="Inspecting images"):
            results[image_path] = (ok, size)
        return results

    with ProcessPoolExecutor(max_workers=workers) as executor:
        iterator = executor.map(_inspect_image, image_paths, chunksize=32)
        for image_path, ok, size in tqdm(iterator, total=len(image_paths), desc="Inspecting images"):
            results[image_path] = (ok, size)
    return results


def _collect_image_sizes(
    trajectories: list[dict], image_info: dict[str, tuple[bool, tuple[int, int] | None]]
) -> Counter:
    size_counter: Counter[tuple[int, int]] = Counter()
    for traj in trajectories:
        for step in traj.get("steps", []):
            ok, size = image_info[step["image"]]
            if ok and size is not None:
                size_counter[size] += 1
    return size_counter


def _ensure_size(image_path: str, target_size: tuple[int, int], resized_dir: str, cache: dict[str, str]) -> str:
    """Return a path to an image guaranteed to be ``target_size``.

    If the image is already the right size the original path is returned unchanged.
    Otherwise a resized copy is saved under ``resized_dir`` (keyed by a hash of
    the original path to avoid collisions) and that path is returned.
    """
    if image_path in cache:
        return cache[image_path]
    with Image.open(image_path) as img:
        if img.size == target_size:
            cache[image_path] = image_path
            return image_path
        path_hash = hashlib.sha1(image_path.encode()).hexdigest()[:16]
        ext = os.path.splitext(image_path)[1] or ".png"
        out_path = os.path.join(resized_dir, f"{path_hash}{ext}")
        if not os.path.exists(out_path):
            img.convert("RGB").resize(target_size, Image.Resampling.LANCZOS).save(out_path)
    cache[image_path] = out_path
    return out_path


def _apply_resize(traj: dict, target_size: tuple[int, int], resized_dir: str, cache: dict[str, str]) -> dict:
    """Return a copy of ``traj`` with all step ``image`` paths at ``target_size``."""
    steps = []
    for step in traj["steps"]:
        step = dict(step)
        step["image"] = _ensure_size(step["image"], target_size, resized_dir, cache)
        steps.append(step)
    return {**traj, "steps": steps}


def _resize_one_image(job: tuple[str, tuple[int, int], str]) -> tuple[str, str]:
    image_path, target_size, resized_dir = job
    path_hash = hashlib.sha1(image_path.encode()).hexdigest()[:16]
    ext = os.path.splitext(image_path)[1] or ".png"
    out_path = os.path.join(resized_dir, f"{path_hash}{ext}")
    if not os.path.exists(out_path):
        with Image.open(image_path) as img:
            img.convert("RGB").resize(target_size, Image.Resampling.LANCZOS).save(out_path)
    return image_path, out_path


def _resize_images(
    image_info: dict[str, tuple[bool, tuple[int, int] | None]],
    target_size: tuple[int, int],
    resized_dir: str,
    workers: int,
) -> dict[str, str]:
    cache: dict[str, str] = {}
    resize_jobs: list[tuple[str, tuple[int, int], str]] = []
    for image_path, (ok, size) in image_info.items():
        if not ok or size is None:
            continue
        if size == target_size:
            cache[image_path] = image_path
        else:
            resize_jobs.append((image_path, target_size, resized_dir))

    if not resize_jobs:
        return cache

    if workers <= 1:
        iterator = map(_resize_one_image, resize_jobs)
        for image_path, out_path in tqdm(iterator, total=len(resize_jobs), desc="Resizing images"):
            cache[image_path] = out_path
        return cache

    with ProcessPoolExecutor(max_workers=workers) as executor:
        iterator = executor.map(_resize_one_image, resize_jobs, chunksize=8)
        for image_path, out_path in tqdm(iterator, total=len(resize_jobs), desc="Resizing images"):
            cache[image_path] = out_path
    return cache


def main():
    parser = argparse.ArgumentParser(description="Prepare RRG training data")
    parser.add_argument("--input", required=True, help="Path to trajectory JSON array or JSONL file")
    parser.add_argument("--local_dir", default="~/data/verl-agent/rrg")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--train_data_size", type=int, default=None, help="Override train set size")
    parser.add_argument("--val_data_size", type=int, default=None, help="Override val set size")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Number of worker processes for image inspection and resizing. Use 1 to disable multiprocessing.",
    )
    parser.add_argument(
        "--target_image_size",
        default=None,
        help="Target resolution for all screenshots, e.g. 1170x2532. "
        "If omitted, the dominant size across all valid trajectories is used. "
        "Images at a different size are resized and saved to {local_dir}/resized_images/.",
    )
    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    raw_trajectories = _load_trajectories(args.input)
    trajectories = [_normalize_trajectory(t) for t in raw_trajectories]
    unique_image_paths = _iter_unique_image_paths(trajectories)
    image_info = _scan_images(unique_image_paths, workers=max(1, args.workers))

    valid_trajectories: list[dict] = []
    invalid_reason_counter: Counter[str] = Counter()
    for traj in tqdm(trajectories, desc="Validating trajectories"):
        ok, reason = _validate_trajectory_with_image_info(traj, image_info)
        if ok:
            valid_trajectories.append(traj)
        else:
            invalid_reason_counter[reason] += 1

    if not valid_trajectories:
        raise ValueError("No valid trajectories found after checking image existence and readability")

    size_counter = _collect_image_sizes(valid_trajectories, image_info)
    target_image_size = _parse_size_arg(args.target_image_size)
    if target_image_size is None:
        target_image_size = size_counter.most_common(1)[0][0]

    resized_dir = os.path.join(local_dir, "resized_images")
    os.makedirs(resized_dir, exist_ok=True)
    resize_cache = _resize_images(image_info, target_image_size, resized_dir, workers=max(1, args.workers))
    resized_trajectories = [
        _apply_resize(t, target_image_size, resized_dir, resize_cache)
        for t in tqdm(valid_trajectories, desc="Rewriting trajectories")
    ]
    resized_count = sum(1 for orig, new in resize_cache.items() if orig != new)

    total = len(resized_trajectories)
    if args.train_data_size and args.val_data_size:
        n_train = args.train_data_size
        n_val = args.val_data_size
    else:
        n_train = int(total * args.train_ratio)
        n_val = total - n_train

    if n_train + n_val > total:
        raise ValueError(f"Requested train+val size {n_train + n_val} exceeds trajectory count {total}")

    train_trajs = resized_trajectories[:n_train]
    val_trajs = resized_trajectories[n_train : n_train + n_val]

    print(f"Input trajectories:   {len(raw_trajectories)}")
    print(f"Unique images:        {len(unique_image_paths)}")
    print(f"Valid trajectories:   {len(valid_trajectories)}")
    if invalid_reason_counter:
        print(f"Filtered out:         {dict(invalid_reason_counter)}")
    print(f"Target image size:    {target_image_size[0]}x{target_image_size[1]}")
    print(f"Images resized:       {resized_count}")
    print(f"Workers:              {max(1, args.workers)}")
    print(f"Final: {total} total, {len(train_trajs)} train, {len(val_trajs)} val")

    def build_records(trajs, split):
        return [
            {
                "data_source": "rrg",
                "prompt": [{"role": "user", "content": "<image>"}],
                "ability": "agent",
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "task_id": traj.get("task_id", ""),
                    "env_id": traj.get("env_id", ""),
                },
            }
            for idx, traj in enumerate(trajs)
        ]

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
        json.dump(resized_trajectories, f)
    with open(train_traces_path, "w") as f:
        json.dump(train_trajs, f)
    with open(val_traces_path, "w") as f:
        json.dump(val_trajs, f)

    print(f"Saved train ({len(train_df)} rows) → {train_path}")
    print(f"Saved val   ({len(val_df)} rows) → {val_path}")
    print(f"Saved cleaned trajectories ({len(resized_trajectories)}) → {cleaned_path}")
    print(f"Saved train trajectories ({len(train_trajs)}) → {train_traces_path}")
    print(f"Saved val trajectories   ({len(val_trajs)}) → {val_traces_path}")


if __name__ == "__main__":
    main()
