#!/usr/bin/env python3
"""Export raw+masked debug frames from in-progress RLDS mask runs.

Polls worker progress JSON files and exports newly completed episodes by pairing
source RLDS with masked TFRecord output (no masker reload needed).

Usage:
  python tools/rlds_mask_live_debug.py --data_mix libero_spatial_no_noops
  python tools/rlds_mask_live_debug.py --data_mix libero_spatial_no_noops --watch 60
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from debug_image_prune import DEFAULT_MAX_DEBUG_IMAGES, prune_debug_images

_REPO_ROOT = _SCRIPT_DIR.parent
DEFAULT_SRC = os.environ.get(
    "RLDS_DATA_ROOT",
    "/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/datasets/modified_libero_rlds",
)
DEFAULT_MASKED = os.environ.get(
    "RLDS_OUT_ROOT",
    "/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/datasets/masked_libero_rlds",
)

TFRECORD_PREFIX = {
    "libero_goal_no_noops": "libero_goal",
    "libero_object_no_noops": "libero_object",
    "libero_spatial_no_noops": "libero_spatial",
    "libero_90_no_noops": "libero_90",
    "libero_10_no_noops": "libero_10",
}


def _episode_steps_from_decoded(decoded: dict) -> list[dict]:
    steps = []
    for step in decoded["steps"]:
        sd = {}
        for k, v in step.items():
            if k == "observation":
                sd[k] = {ok: _np_value(ov) for ok, ov in v.items()}
            else:
                sd[k] = _np_value(v)
        steps.append(sd)
    return steps


def _read_tfrecord_at_index(path: Path, record_index: int, max_record_bytes: int = 64 * 1024 * 1024) -> bytes | None:
    import struct

    with open(path, "rb") as f:
        for i in range(record_index + 1):
            header = f.read(12)
            if len(header) < 12:
                return None
            (length,) = struct.unpack("<Q", header[:8])
            if length <= 0 or length > max_record_bytes:
                return None
            payload = f.read(length)
            if len(payload) < length:
                return None
            if len(f.read(4)) < 4:
                return None
            if i == record_index:
                return payload
    return None


def _masked_episode_steps_at(masked_root: Path, data_mix: str, ep_idx: int, n_shards: int = 16) -> list[dict] | None:
    prefix = TFRECORD_PREFIX.get(data_mix, data_mix.replace("_no_noops", ""))
    ver_dir = masked_root / data_mix / "1.0.0"
    shard_id = ep_idx % n_shards
    record_in_shard = ep_idx // n_shards
    shard_path = ver_dir / f"{prefix}-train.tfrecord-{shard_id:05d}-of-{n_shards:05d}"
    if not shard_path.exists():
        return None
    rec = _read_tfrecord_at_index(shard_path, record_in_shard)
    if rec is None:
        return None
    try:
        builder = tfds.builder(data_mix, data_dir=str(masked_root))
        decoded = builder.info.features.deserialize_example(rec)
    except Exception:
        return None
    return _episode_steps_from_decoded(decoded)


def _safe_lang(lang: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", lang.strip().lower())[:80]


def _frame_indices(n_steps: int, n_frames: int) -> list[int]:
    if n_steps <= 0:
        return []
    if n_frames <= 1:
        return [0]
    return sorted(
        {min(n_steps - 1, max(0, int(round(i * (n_steps - 1) / (n_frames - 1))))) for i in range(n_frames)}
    )


def _np_value(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return x


def _decode_str(s) -> str:
    if hasattr(s, "decode"):
        return s.decode("utf-8")
    return str(s)


def _max_completed_episode(masked_root: Path, data_mix: str, num_workers: int) -> int:
    best = -1
    for w in range(num_workers):
        path = masked_root / f".rlds_mask_progress_{data_mix}_w{w}.json"
        payload = None
        if path.exists():
            try:
                with open(path) as f:
                    payload = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        if payload is None:
            continue
        completed = int(payload.get("completed_in_run", 0))
        global_ep = int(payload.get("global_episode", 0))
        phase = payload.get("phase", "")
        if phase == "writing" and completed > 0:
            best = max(best, global_ep)
        elif completed > 0:
            # During masking, last fully written episode is one behind current.
            best = max(best, global_ep - num_workers if phase == "masking" else global_ep)
    return best


def _load_episode_steps(data_root: Path, data_mix: str, ep_idx: int) -> list[dict] | None:
    builder = tfds.builder(data_mix, data_dir=str(data_root))
    ds = builder.as_dataset(split="train", shuffle_files=False)
    for i, episode in enumerate(ds.skip(ep_idx).take(1)):
        if i != 0:
            return None
        steps = []
        for step in episode["steps"].as_numpy_iterator():
            obs = step["observation"]
            steps.append(
                {
                    "observation": {k: _np_value(v) for k, v in obs.items()},
                    "language_instruction": _np_value(step.get("language_instruction", b"")),
                }
            )
        return steps
    return None


def _export_episodes(
    src_root: Path,
    masked_root: Path,
    data_mix: str,
    out_root: Path,
    start_ep: int,
    max_ep: int,
    n_frames: int,
    max_images: int,
) -> int:
    new_count = 0
    for ep_idx in range(start_ep, max_ep + 1):
        masked_steps = _masked_episode_steps_at(masked_root, data_mix, ep_idx)
        if not masked_steps:
            continue
        src_steps = _load_episode_steps(src_root, data_mix, ep_idx)
        if not src_steps:
            continue
        lang = _decode_str(masked_steps[0].get("language_instruction", b"")).lower().strip()
        if not lang:
            continue
        task_dir = out_root / _safe_lang(lang)
        task_dir.mkdir(parents=True, exist_ok=True)
        n = min(len(src_steps), len(masked_steps))
        for fi in _frame_indices(n, n_frames):
            raw_arr = src_steps[fi]["observation"].get("image")
            masked_arr = masked_steps[fi]["observation"].get("image")
            if raw_arr is None or masked_arr is None:
                continue
            Image.fromarray(np.asarray(raw_arr)).convert("RGB").save(
                task_dir / f"ep{ep_idx:05d}_frame{fi:03d}_raw.png"
            )
            Image.fromarray(np.asarray(masked_arr)).convert("RGB").save(
                task_dir / f"ep{ep_idx:05d}_frame{fi:03d}_masked.png"
            )
        new_count += 1
        print(f"  exported ep {ep_idx}", flush=True)
        state_path = out_root / ".exported_up_to.json"
        with open(state_path, "w") as f:
            json.dump({"exported_up_to": ep_idx, "max_seen": max_ep}, f, indent=1)
    removed = prune_debug_images(out_root, max_images)
    if removed:
        print(f"  pruned {removed} old debug image(s), keeping {max_images}", flush=True)
    return new_count


def _run_once(args) -> int:
    tf.config.set_visible_devices([], "GPU")
    src_root = Path(args.src_root)
    masked_root = Path(args.masked_root)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    state_path = out_root / ".exported_up_to.json"
    exported_up_to = -1
    if state_path.exists():
        try:
            with open(state_path) as f:
                exported_up_to = int(json.load(f).get("exported_up_to", -1))
        except (json.JSONDecodeError, OSError, ValueError):
            exported_up_to = -1

    max_ep = _max_completed_episode(masked_root, args.data_mix, args.num_workers)
    if args.max_episode is not None:
        max_ep = min(max_ep, args.max_episode)
    if max_ep < 0:
        print("No completed episodes yet.", flush=True)
        return 0

    start = max(exported_up_to + 1, args.start_episode)
    if start > max_ep:
        print(f"Up to date (exported through ep {exported_up_to}, latest done ~ep {max_ep})", flush=True)
        return 0

    new_count = _export_episodes(
        src_root, masked_root, args.data_mix, out_root, start, max_ep, args.num_frames, args.max_debug_images
    )
    if new_count > 0 or start <= max_ep:
        exported_up_to = max_ep

    with open(state_path, "w") as f:
        json.dump({"exported_up_to": exported_up_to, "max_seen": max_ep}, f, indent=1)
    print(f"DONE: {new_count} episodes -> {out_root.resolve()} (through ep {exported_up_to})", flush=True)
    return new_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_mix", default="libero_spatial_no_noops")
    ap.add_argument("--src_root", default=DEFAULT_SRC)
    ap.add_argument("--masked_root", default=DEFAULT_MASKED)
    ap.add_argument(
        "--out_dir",
        default="debug_masked_validation/libero_spatial_current_run/live",
    )
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--num_frames", type=int, default=1)
    ap.add_argument("--max_debug_images", type=int, default=DEFAULT_MAX_DEBUG_IMAGES)
    ap.add_argument("--start_episode", type=int, default=0)
    ap.add_argument("--max_episode", type=int, default=None)
    ap.add_argument("--watch", type=int, default=0, help="Poll every N seconds (0 = run once)")
    args = ap.parse_args()

    if args.watch <= 0:
        _run_once(args)
        return

    print(f"Watching every {args.watch}s -> {args.out_dir}", flush=True)
    while True:
        _run_once(args)
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
