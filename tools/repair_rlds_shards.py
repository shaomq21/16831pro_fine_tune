#!/usr/bin/env python3
"""Truncate corrupt TFRecord tail garbage (from old watchdog kill mid-write).

After truncate, re-sync resume from TFRecord:
  python tools/init_multi_gpu_resume.py --data_mix ... --num_workers 8 --out_root ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS))

from rlds_mask_state import DEFAULT_OUT_ROOT, TFRECORD_PREFIX, truncate_shard_to_valid_prefix


def main() -> int:
    ap = argparse.ArgumentParser(description="Truncate corrupt TFRecord shard tails")
    ap.add_argument("--data_mix", default="libero_spatial_no_noops")
    ap.add_argument("--out_root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--num_shards", type=int, default=16)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    prefix = TFRECORD_PREFIX.get(args.data_mix, args.data_mix.replace("_no_noops", ""))
    ver = Path(args.out_root) / args.data_mix / "1.0.0"
    total_removed = 0
    for sid in range(args.num_shards):
        path = ver / f"{prefix}-train.tfrecord-{sid:05d}-of-{args.num_shards:05d}"
        if not path.exists():
            continue
        size_before = path.stat().st_size
        if args.dry_run:
            from rlds_mask_state import scan_tfrecord_prefix

            n, end = scan_tfrecord_prefix(path)
            removed = size_before - end
            print(
                f"shard {sid:2d}: {n} valid records, "
                f"would remove {removed / 1024 / 1024:.1f} MB "
                f"({size_before / 1024 / 1024:.1f} -> {end / 1024 / 1024:.1f} MB)"
            )
            total_removed += removed
        else:
            kept, removed = truncate_shard_to_valid_prefix(path)
            if removed:
                print(
                    f"shard {sid:2d}: kept {kept} records, "
                    f"removed {removed / 1024 / 1024:.1f} MB"
                )
            total_removed += removed
    print(f"Total garbage removed: {total_removed / 1024 / 1024:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
