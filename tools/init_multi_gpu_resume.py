#!/usr/bin/env python3
"""Initialize per-worker resume state for multi-GPU RLDS masking.

Uses masked TFRecord files as ground truth for which episodes are done.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS_DIR))

from rlds_mask_state import (
    DEFAULT_OUT_ROOT,
    done_episodes_from_resume_files,
    done_episodes_from_tfrecords,
    worker_done_episodes,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_mix", required=True)
    ap.add_argument("--num_workers", type=int, required=True)
    ap.add_argument("--out_root", default=DEFAULT_OUT_ROOT)
    ap.add_argument(
        "--last_episode",
        type=int,
        default=None,
        help="Override: next global episode index (ignores TFRecord scan)",
    )
    args = ap.parse_args()

    out_root = Path(args.out_root)

    if args.last_episode is not None:
        last_done = max(args.last_episode - 1, -1)
        done = set(range(last_done + 1))
        print(f"Using --last_episode={args.last_episode} -> {len(done)} episodes done")
    else:
        tf_done = done_episodes_from_tfrecords(out_root, args.data_mix)
        resume_done = done_episodes_from_resume_files(out_root, args.data_mix, args.num_workers)
        if tf_done:
            done = tf_done
            print(f"From TFRecords: {len(done)} readable episodes")
            ghost = resume_done - tf_done
            if ghost:
                print(
                    f"  Dropping {len(ghost)} resume-only episodes "
                    f"(marked done but not readable on disk)"
                )
        elif resume_done:
            done = resume_done
            print(f"From resume files (no TFRecords yet): {len(done)} episodes")
        else:
            done = set()
            print("No prior progress found")

    for w in range(args.num_workers):
        lane_done = sorted(worker_done_episodes(done, w, args.num_workers))
        path = out_root / f".rlds_resume_{args.data_mix}_w{w}.json"
        with open(path, "w") as f:
            json.dump({"completed": len(lane_done), "done_episodes": lane_done}, f)
        nxt = w
        while nxt in done:
            nxt += args.num_workers
        print(f"  worker {w}: completed={len(lane_done)}, next ep ~{nxt} -> {path}")

    if args.num_workers == 1:
        single_path = out_root / f".rlds_resume_{args.data_mix}.json"
        last_episode = 0
        for ep in range(10_000):
            if ep not in done:
                last_episode = ep
                break
        else:
            last_episode = max(done) + 1 if done else 0
        with open(single_path, "w") as f:
            json.dump({"last_episode": last_episode}, f)
        print(f"  single-GPU resume: last_episode={last_episode} -> {single_path}")


if __name__ == "__main__":
    main()
