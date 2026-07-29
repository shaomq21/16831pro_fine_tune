#!/usr/bin/env python3
"""Rewrite TFRecord shards keeping only TensorFlow-readable records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tensorflow as tf

_TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS))

from rlds_mask_state import DEFAULT_OUT_ROOT, TFRECORD_PREFIX


def rewrite_shard(path: Path) -> tuple[int, int]:
    if not path.exists() or path.stat().st_size == 0:
        return 0, 0
    records = []
    try:
        for raw in tf.data.TFRecordDataset(str(path)):
            records.append(raw.numpy())
    except tf.errors.DataLossError:
        pass
    if not records:
        path.unlink(missing_ok=True)
        return 0, 0
    tmp = path.with_suffix(path.suffix + ".rewrite")
    with tf.io.TFRecordWriter(str(tmp)) as w:
        for rec in records:
            w.write(rec)
    before = path.stat().st_size
    tmp.replace(path)
    return len(records), before - path.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser(description="Keep only TF-readable TFRecord records")
    ap.add_argument("--data_mix", default="libero_spatial_no_noops")
    ap.add_argument("--out_root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--num_shards", type=int, default=16)
    args = ap.parse_args()

    prefix = TFRECORD_PREFIX.get(args.data_mix, args.data_mix.replace("_no_noops", ""))
    ver = Path(args.out_root) / args.data_mix / "1.0.0"
    total = 0
    for sid in range(args.num_shards):
        path = ver / f"{prefix}-train.tfrecord-{sid:05d}-of-{args.num_shards:05d}"
        n, removed = rewrite_shard(path)
        if n:
            print(f"shard {sid:2d}: {n} TF-readable records, dropped {removed / 1024 / 1024:.1f} MB")
        total += n
    print(f"Total TF-readable records: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
