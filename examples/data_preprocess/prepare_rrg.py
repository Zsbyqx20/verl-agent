# Copyright 2025 the ReverseReasoningGenerator (RRG) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Emit a DUMMY parquet for the RRG replay env.

Exactly like the upstream `examples/data_preprocess/prepare.py` (which loads geometry3k only to
indicate modality + data size), but self-contained — no network/HF download. The replay env
(`env_name=rrg`) ignores the parquet content entirely and serves real recorded frames; the
parquet only sizes gen_batch (== train/val_data_size) and signals the visual modality.

Run:
    python3 -m examples.data_preprocess.prepare_rrg --train_data_size 5 --val_data_size 5
"""
import argparse
import io
import os

import numpy as np
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage


def _dummy_png_bytes() -> bytes:
    im = PILImage.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _rows(n: int, split: str):
    out = []
    for i in range(n):
        out.append({
            "data_source": "rrg",
            "prompt": [{"role": "user", "content": "<image>"}],
            "images": [{"bytes": _dummy_png_bytes(), "path": None}],
            "ability": "agent",
            "extra_info": {"split": split, "index": i},
        })
    return out


FEATURES = Features({
    "data_source": Value("string"),
    "prompt": [{"role": Value("string"), "content": Value("string")}],
    "images": Sequence(Image()),
    "ability": Value("string"),
    "extra_info": {"split": Value("string"), "index": Value("int64")},
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_dir", default="~/data/verl-agent/rrg")
    ap.add_argument("--train_data_size", default=5, type=int)
    ap.add_argument("--val_data_size", default=5, type=int)
    # IMPORTANT: parquet ROW COUNT sets steps-per-epoch (len(dataloader) = rows / batch_size,
    # ray_trainer.py:623). It is INDEPENDENT of *_data_size (= batch_size = tasks/step). The env
    # re-samples real episodes each batch (num_episodes=null), so content is dummy but the row
    # count is the training-step budget. Defaults below keep rows == batch (1 step/epoch) unless
    # --train_rows/--val_rows are given. For the full set use --train_rows = ceil(pool / batch).
    ap.add_argument("--train_rows", default=None, type=int,
                    help="train parquet rows = steps_per_epoch * batch (default: train_data_size)")
    ap.add_argument("--val_rows", default=None, type=int,
                    help="val parquet rows (default: val_data_size)")
    args = ap.parse_args()

    train_rows = args.train_rows if args.train_rows is not None else args.train_data_size
    val_rows = args.val_rows if args.val_rows is not None else args.val_data_size

    out_dir = os.path.expanduser(args.local_dir)
    os.makedirs(out_dir, exist_ok=True)
    for split, n, fname in [("train", train_rows, "train.parquet"),
                            ("test", val_rows, "test.parquet")]:
        ds = Dataset.from_list(_rows(n, split), features=FEATURES)
        path = os.path.join(out_dir, fname)
        ds.to_parquet(path)
        print(f"wrote {n} dummy rows -> {path}  (steps this split = {n} / batch_size)")


if __name__ == "__main__":
    main()
