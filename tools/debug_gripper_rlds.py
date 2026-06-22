#!/usr/bin/env python3
"""Export raw + masked debug frames (no click overlays, gripper white dots only)."""

from __future__ import annotations

import argparse
import re
import struct
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "openvla-oft"))

from gripper_project import gripper_pixels_from_obs  # noqa: E402

_STORAGE = "/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/datasets"
DEFAULT_SRC_ROOT = f"{_STORAGE}/modified_libero_rlds"
DEFAULT_MASKED_ROOT = f"{_STORAGE}/masked_libero_rlds"

TFRECORD_PREFIX = {
    "libero_goal_no_noops": "libero_goal",
    "libero_object_no_noops": "libero_object",
    "libero_spatial_no_noops": "libero_spatial",
    "libero_90_no_noops": "libero_90",
    "libero_10_no_noops": "libero_10",
}


def _read_complete_tfrecords(path: Path) -> list[bytes]:
    records: list[bytes] = []
    with open(path, "rb") as f:
        while True:
            header = f.read(12)
            if len(header) < 12:
                break
            (length,) = struct.unpack("<Q", header[:8])
            payload = f.read(length)
            if len(payload) < length:
                break
            if len(f.read(4)) < 4:
                break
            records.append(payload)
    return records


def _decode_str(s) -> str:
    if hasattr(s, "decode"):
        return s.decode("utf-8")
    return str(s)


def _safe_name(lang: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", lang.strip().lower())[:80]


def _frame_indices(n_steps: int, n_frames: int) -> list[int]:
    if n_steps <= 0:
        return []
    if n_frames <= 1:
        return [0]
    return sorted({min(n_steps - 1, max(0, int(round(i * (n_steps - 1) / (n_frames - 1))))) for i in range(n_frames)})


def _np_value(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return x


def _episode_steps(decoded: dict) -> list[dict]:
    steps = []
    for s in decoded["steps"]:
        sd = {}
        for k, v in s.items():
            if k == "observation":
                sd[k] = {ok: _np_value(ov) for ok, ov in v.items()}
            else:
                sd[k] = _np_value(v)
        steps.append(sd)
    return steps


def _iter_episodes_from_tfrecords(data_root: Path, data_mix: str):
    prefix = TFRECORD_PREFIX.get(data_mix, data_mix.replace("_no_noops", ""))
    ver_dir = data_root / data_mix / "1.0.0"
    builder = tfds.builder(data_mix, data_dir=str(data_root))
    features = builder.info.features
    global_idx = 0
    for shard_path in sorted(ver_dir.glob(f"{prefix}-train.tfrecord-*")):
        for rec in _read_complete_tfrecords(shard_path):
            decoded = features.deserialize_example(rec)
            steps = _episode_steps(decoded)
            yield global_idx, steps
            global_idx += 1


def _strip_click_overlay(img: np.ndarray) -> np.ndarray:
    """Remove baked-in red/green crosshair + coordinate text; keep mask tints and white gripper dots."""
    arr = np.asarray(img).copy()
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    # Saturated red/green strokes (crosshair + text), not light mask tints (~255,120,120) / (~120,255,120)
    red_overlay = (r > 130) & (r - g > 70) & (r - b > 70) & (g < 110)
    green_overlay = (g > 130) & (g - r > 70) & (g - b > 70) & (r < 110)
    arr[red_overlay | green_overlay] = 0
    return arr


def _load_source_episode(src_root: Path, data_mix: str, ep_idx: int) -> list[dict] | None:
    builder = tfds.builder(data_mix, data_dir=str(src_root))
    ds = builder.as_dataset(split="train", shuffle_files=False)
    for i, episode in enumerate(ds.skip(ep_idx).take(1)):
        if i == 0:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_mix", default="libero_spatial_no_noops")
    ap.add_argument("--src_root", default=DEFAULT_SRC_ROOT)
    ap.add_argument("--masked_root", default=DEFAULT_MASKED_ROOT)
    ap.add_argument("--out_dir", default="debug_masked_validation/libero_spatial_gripper")
    ap.add_argument("--patched_before", type=int, default=159, help="Ep index < this = patched RLDS")
    ap.add_argument("--num_frames", type=int, default=6)
    ap.add_argument("--max_tasks", type=int, default=10)
    ap.add_argument("--max_episodes", type=int, default=None, help="Stop after this many masked episodes")
    ap.add_argument(
        "--no_strip_overlay",
        action="store_true",
        help="Save masked RLDS images as-is (current run has no click overlay)",
    )
    args = ap.parse_args()

    tf.config.set_visible_devices([], "GPU")
    src_root = Path(args.src_root)
    masked_root = Path(args.masked_root)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    patched_by_lang: dict[str, tuple[int, list]] = {}
    new_by_lang: dict[str, tuple[int, list]] = {}
    src_cache: dict[int, list] = {}

    for ep_idx, masked_steps in _iter_episodes_from_tfrecords(masked_root, args.data_mix):
        if args.max_episodes is not None and ep_idx >= args.max_episodes:
            break
        if not masked_steps:
            continue
        lang = _decode_str(masked_steps[0].get("language_instruction", b"")).lower().strip()
        if not lang:
            continue
        if ep_idx < args.patched_before:
            if lang not in patched_by_lang:
                patched_by_lang[lang] = (ep_idx, masked_steps)
        elif lang not in new_by_lang:
            new_by_lang[lang] = (ep_idx, masked_steps)
        if len(patched_by_lang) >= args.max_tasks and len(new_by_lang) >= min(args.max_tasks, 3):
            break

    print(f"patched tasks: {len(patched_by_lang)}, new-gen: {len(new_by_lang)}", flush=True)

    summary_lines = [
        f"data_mix={args.data_mix}",
        f"patched_before={args.patched_before}",
        f"patched_tasks={len(patched_by_lang)}",
        f"new_gen_tasks={len(new_by_lang)}",
        "raw = source RLDS; masked = masked RLDS output (as stored in TFRecord)",
        f"strip_overlay={not args.no_strip_overlay}",
        "",
    ]

    for tag, by_lang in [("patched", patched_by_lang), ("generated", new_by_lang)]:
        if not by_lang:
            summary_lines.append(f"[{tag}] no episodes")
            continue
        for lang in sorted(by_lang):
            ep_idx, masked_steps = by_lang[lang]
            if ep_idx not in src_cache:
                src_cache[ep_idx] = _load_source_episode(src_root, args.data_mix, ep_idx)
            src_steps = src_cache[ep_idx]
            if src_steps is None:
                continue

            task_dir = out_root / tag / _safe_name(lang)
            task_dir.mkdir(parents=True, exist_ok=True)
            n = min(len(masked_steps), len(src_steps))
            indices = _frame_indices(n, args.num_frames)
            frame_info = []

            for fi in indices:
                raw_arr = src_steps[fi]["observation"].get("image")
                masked_arr = masked_steps[fi]["observation"].get("image")
                proprio = masked_steps[fi]["observation"].get("state")
                joint = masked_steps[fi]["observation"].get("joint_state")
                if raw_arr is None or masked_arr is None:
                    continue
                raw_path = task_dir / f"ep{ep_idx:04d}_frame{fi:03d}_raw.png"
                masked_path = task_dir / f"ep{ep_idx:04d}_frame{fi:03d}_masked.png"
                Image.fromarray(np.asarray(raw_arr)).convert("RGB").save(raw_path)
                arr = np.asarray(masked_arr)
                if not args.no_strip_overlay:
                    arr = _strip_click_overlay(arr)
                Image.fromarray(arr).convert("RGB").save(masked_path)
                pts = gripper_pixels_from_obs(proprio, joint_state=joint) if proprio is not None else []
                frame_info.append(f"  ep{ep_idx} f{fi}: gripper={pts}")

            summary_lines.append(f"[{tag}] {lang} (ep {ep_idx}, len={n})")
            summary_lines.extend(frame_info)
            print(f"  {tag}: ep{ep_idx} {lang[:55]} ({len(indices)} frames)", flush=True)

    with open(out_root / "summary.txt", "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"DONE -> {out_root.resolve()}")


if __name__ == "__main__":
    main()
