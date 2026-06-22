#!/usr/bin/env python3
"""Audit which global episode indices are present in masked TFRecords vs expected."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import tensorflow_datasets as tfds

TFRECORD_PREFIX = {
    "libero_goal_no_noops": "libero_goal",
    "libero_object_no_noops": "libero_object",
    "libero_spatial_no_noops": "libero_spatial",
    "libero_90_no_noops": "libero_90",
    "libero_10_no_noops": "libero_10",
}


def _count_records_per_shard(masked_root: Path, data_mix: str, n_shards: int = 16) -> list[int]:
    prefix = TFRECORD_PREFIX.get(data_mix, data_mix.replace("_no_noops", ""))
    ver_dir = masked_root / data_mix / "1.0.0"
    counts = []
    for sid in range(n_shards):
        path = ver_dir / f"{prefix}-train.tfrecord-{sid:05d}-of-{n_shards:05d}"
        n = 0
        if path.exists():
            with open(path, "rb") as f:
                while True:
                    h = f.read(12)
                    if len(h) < 12:
                        break
                    (ln,) = struct.unpack("<Q", h[:8])
                    if f.read(ln) is None or len(f.read(4)) < 4:
                        break
                    n += 1
        counts.append(n)
    return counts


def _done_from_workers(out_root: Path, data_mix: str, num_workers: int) -> set[int]:
    import json

    done: set[int] = set()
    for w in range(num_workers):
        path = out_root / f".rlds_mask_progress_{data_mix}_w{w}.json"
        if not path.exists():
            path = out_root / f".rlds_resume_{data_mix}_w{w}.json"
            if path.exists():
                with open(path) as f:
                    completed = int(json.load(f).get("completed", 0))
            else:
                completed = 0
        else:
            with open(path) as f:
                completed = int(json.load(f).get("completed_in_run", 0))
        for i in range(completed):
            done.add(w + i * num_workers)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_mix", default="libero_spatial_no_noops")
    ap.add_argument("--masked_root", default="/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/datasets/masked_libero_rlds")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--total_episodes", type=int, default=432)
    args = ap.parse_args()

    masked_root = Path(args.masked_root)
    out_root = masked_root

    counts = _count_records_per_shard(masked_root, args.data_mix)
    total_in_tfrecord = sum(counts)
    print(f"TFRecord episodes (complete records): {total_in_tfrecord}")
    print(f"Per-shard: {counts}")

    # Episodes that should exist if contiguous 0..N-1
    # Global ep i goes to shard i % 16, position i // 16 in shard
    present: set[int] = set()
    for sid, cnt in enumerate(counts):
        for pos in range(cnt):
            present.add(sid + pos * 16)

    done_claimed = _done_from_workers(out_root, args.data_mix, args.num_workers)
    # single-worker extras
    import json

    sp = out_root / f".rlds_mask_progress_{args.data_mix}.json"
    if sp.exists():
        with open(sp) as f:
            p = json.load(f)
        rf = int(p.get("resume_from", 0))
        cr = int(p.get("completed_in_run", 0))
        done_claimed.update(rf + i for i in range(cr))

    print(f"\nResume claims done: {len(done_claimed)} unique global episodes")
    print(f"TFRecord implies present: {len(present)} unique global episodes (if contiguous from 0)")

  # gaps in present set up to max present
    if present:
        mx = max(present)
        missing_in_tf = [i for i in range(mx + 1) if i not in present]
        print(f"\nMissing in TFRecord (0..{mx}): {len(missing_in_tf)} episodes")
        if missing_in_tf:
            print(f"  first 30 gaps: {missing_in_tf[:30]}")

    claimed_not_in_tf = sorted(done_claimed - present)
    in_tf_not_claimed = sorted(present - done_claimed)
    if claimed_not_in_tf:
        print(f"\nClaimed done but maybe not in TFRecord: {claimed_not_in_tf[:20]}...")
    if in_tf_not_claimed:
        print(f"In TFRecord but not in resume claim: {in_tf_not_claimed[:20]}...")


if __name__ == "__main__":
    main()
