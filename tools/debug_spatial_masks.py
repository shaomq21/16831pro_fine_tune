#!/usr/bin/env python3
"""
Generate mask debug images per task (libero_spatial, libero_90 STUDY_SCENE4 books, etc.).

Reads RLDS episodes, one episode per unique language instruction,
saves multiple frames (default: evenly spaced through episode) with SAM3 temporal masks.

Usage:
  # libero_spatial (all 10 tasks), 8 frames per episode
  python tools/debug_spatial_masks.py --suite libero_spatial

  # custom frame sampling (fractions of episode length)
  python tools/debug_spatial_masks.py --suite libero_spatial \\
      --frame_fracs 0,0.15,0.3,0.45,0.6,0.75,0.9

  # libero_90 STUDY_SCENE4 book tasks only
  python tools/debug_spatial_masks.py --suite libero_90_study4

  # custom mix + language filter
  python tools/debug_spatial_masks.py --data_mix libero_90_no_noops --lang_filter "pick up the book"
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_OPENVLA_ROOT = _REPO_ROOT / "openvla-oft"
sys.path.insert(0, str(_OPENVLA_ROOT))

from mask_processor import EpisodeMaskTracker, GroundedSAMConfig, GroundedSAMMasker
from mask_spatial import (
    LIBERO_90_STUDY_SCENE4_TASKS,
    get_libero_spatial_task_points,
    mask_role_description,
    primitive_skill_for_lang,
)
from sam3_backend import DEFAULT_SAM3_CKPT

_STORAGE = "/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/datasets"
_DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
_DINO_CKPT = "groundingdino_swint_ogc.pth"
_SAM_CKPT = "sam_vit_b_01ec64.pth"
_SAM_TYPE = "vit_b"

SUITE_PRESETS = {
    "libero_spatial": {
        "data_mix": "libero_spatial_no_noops",
        "data_root": f"{_STORAGE}/modified_libero_rlds",
        "out_dir": "debug_masked_validation/libero_spatial_sam3_multi",
        "lang_allowlist": None,
        "sam_backend": "sam3",
        "use_tracker": True,
    },
    "libero_goal": {
        "data_mix": "libero_goal_no_noops",
        "data_root": f"{_STORAGE}/masked_libero_rlds",
        "out_dir": "debug_masked_validation/libero_goal_grounded_sam",
        "lang_allowlist": None,
        "sam_backend": "sam1",
        "sam_type": _SAM_TYPE,
        "sam_checkpoint": _SAM_CKPT,
        "use_tracker": False,
    },
    "libero_90_study4": {
        "data_mix": "libero_90_no_noops",
        "out_dir": "debug_masked_validation/libero_90_study_scene4",
        "lang_allowlist": set(LIBERO_90_STUDY_SCENE4_TASKS),
    },
}


def _decode_str(s):
    if hasattr(s, "decode"):
        return s.decode("utf-8")
    return str(s)


def _safe_name(lang: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", lang.strip().lower())[:100]


def _parse_frame_fracs(s: str) -> list[float]:
    fracs = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        f = float(part)
        fracs.append(max(0.0, min(1.0, f)))
    return sorted(set(fracs))


def _frame_indices(n_steps: int, fracs: list[float]) -> list[int]:
    if n_steps <= 0:
        return []
    idxs = []
    for f in fracs:
        if n_steps == 1:
            idxs.append(0)
        else:
            idxs.append(min(n_steps - 1, max(0, int(round(f * (n_steps - 1))))))
    # preserve order, dedupe
    seen = set()
    out = []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--suite",
        type=str,
        choices=list(SUITE_PRESETS.keys()),
        default=None,
        help="Preset: libero_spatial | libero_goal | libero_90_study4",
    )
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--data_mix", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--lang_filter", type=str, default=None, help="Substring filter on language instruction")
    ap.add_argument("--sam3_ckpt", type=str, default=DEFAULT_SAM3_CKPT)
    ap.add_argument("--sam_backend", type=str, default=None, choices=["sam1", "sam3"])
    ap.add_argument("--sam_type", type=str, default=None)
    ap.add_argument("--sam_ckpt", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--mid_frac", type=float, default=0.45, help="Deprecated; use --frame_fracs")
    ap.add_argument(
        "--frame_fracs",
        type=str,
        default="0,0.12,0.25,0.38,0.5,0.62,0.75,0.88",
        help="Comma-separated episode fractions to save (0=start, 1=end)",
    )
    ap.add_argument("--num_frames", type=int, default=None, help="If set, evenly sample this many frames (overrides --frame_fracs)")
    ap.add_argument("--max_tasks", type=int, default=30)
    ap.add_argument("--overlay", action="store_true", help="Save raw frames with red/green click points")
    args = ap.parse_args()

    preset = SUITE_PRESETS.get(args.suite or "", {})
    data_mix = args.data_mix or preset.get("data_mix", "libero_spatial_no_noops")
    data_root = args.data_root or preset.get("data_root") or str(_OPENVLA_ROOT / "datasets/modified_libero_rlds")
    out_root = Path(args.out_dir or preset.get("out_dir") or f"debug_masked_validation/{data_mix}")
    lang_allowlist = preset.get("lang_allowlist")
    sam_backend = args.sam_backend or preset.get("sam_backend", "sam3")
    use_tracker = preset.get("use_tracker", sam_backend == "sam3")
    if args.suite == "libero_goal":
        use_tracker = False

    tf.config.set_visible_devices([], "GPU")
    out_root.mkdir(parents=True, exist_ok=True)

    sam_type = args.sam_type or preset.get("sam_type", _SAM_TYPE)
    sam_ckpt_name = args.sam_ckpt or preset.get("sam_checkpoint", _SAM_CKPT)
    cfg = GroundedSAMConfig(
        dino_config_path=str(_OPENVLA_ROOT / _DINO_CONFIG),
        dino_checkpoint_path=str(_OPENVLA_ROOT / _DINO_CKPT),
        sam_backend=sam_backend,
        sam_type=sam_type,
        sam_checkpoint_path=str(_OPENVLA_ROOT / sam_ckpt_name),
        sam3_checkpoint_path=args.sam3_ckpt,
        device=args.device,
    )
    print(f"Loading masker ({sam_backend}) for {data_mix}...", flush=True)
    masker = GroundedSAMMasker(cfg)

    builder = tfds.builder(data_mix, data_dir=data_root)
    ds = builder.as_dataset(split="train", shuffle_files=False)

    seen_lang = set()
    task_episodes = {}

    for episode in ds:
        steps = list(episode["steps"].as_numpy_iterator())
        if not steps:
            continue
        lang = _decode_str(steps[0].get("language_instruction", b"")).lower().strip()
        if not lang or lang in seen_lang:
            continue
        if lang_allowlist is not None and lang not in lang_allowlist:
            continue
        if args.lang_filter and args.lang_filter.lower() not in lang:
            continue
        seen_lang.add(lang)
        task_episodes[lang] = steps
        if len(seen_lang) >= args.max_tasks:
            break

    print(f"Found {len(task_episodes)} tasks", flush=True)
    if lang_allowlist:
        missing = lang_allowlist - seen_lang
        if missing:
            print(f"WARNING: missing allowlisted tasks: {sorted(missing)}", flush=True)

    for lang, steps in sorted(task_episodes.items()):
        task_dir = out_root / _safe_name(lang)
        task_dir.mkdir(parents=True, exist_ok=True)

        n = len(steps)
        if args.num_frames is not None and args.num_frames > 0:
            if args.num_frames == 1:
                fracs = [0.0]
            else:
                fracs = [i / (args.num_frames - 1) for i in range(args.num_frames)]
        else:
            fracs = _parse_frame_fracs(args.frame_fracs)
            if not fracs:
                fracs = [0.0, args.mid_frac]
        indices = _frame_indices(n, fracs)
        tracker = EpisodeMaskTracker(masker) if use_tracker else None

        saved_frames = []
        save_set = set(indices)
        max_fi = max(indices) if indices else 0

        for fi in range(max_fi + 1):
            img_arr = steps[fi]["observation"]["image"]
            proprio = steps[fi]["observation"].get("state")
            joint = steps[fi]["observation"].get("joint_state")
            img = Image.fromarray(img_arr).convert("RGB")
            if tracker is not None:
                masked = tracker.mask_image_from_lang(
                    img, lang, proprio_state=proprio, joint_state=joint
                )
            else:
                masked = masker.mask_image_from_lang(
                    img, lang, proprio_state=proprio, joint_state=joint
                )
            if fi not in save_set:
                continue
            img.save(task_dir / f"frame{fi:03d}_raw.png")
            masked.save(task_dir / f"frame{fi:03d}_masked.png")
            saved_frames.append(fi)

        if args.overlay:
            from PIL import ImageDraw
            pts = get_libero_spatial_task_points(lang)
            if pts:
                ov = Image.open(task_dir / "frame000_raw.png").convert("RGB")
                draw = ImageDraw.Draw(ov)
                W, H = ov.size
                for key, color in [("red", (255, 0, 0)), ("green", (0, 255, 0))]:
                    if key not in pts:
                        continue
                    px, py = pts[key]
                    cx, cy = int(px * W), int(py * H)
                    r = 6
                    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=2)
                    draw.line([cx - 10, cy, cx + 10, cy], fill=color, width=2)
                    draw.line([cx, cy - 10, cx, cy + 10], fill=color, width=2)
                ov.save(task_dir / "frame000_overlay.png")

        red_desc, green_desc = mask_role_description(lang)
        with open(task_dir / "task.txt", "w") as f:
            f.write(f"{lang}\n")
            f.write(f"episode_len: {n}\n")
            f.write(f"saved_frames: {saved_frames}\n")
            f.write(f"frame_fracs: {[round(fi / max(n - 1, 1), 3) for fi in saved_frames]}\n")
            f.write(f"primitive_skill: {primitive_skill_for_lang(lang)}\n")
            f.write(f"red: {red_desc}\n")
            f.write(f"green: {green_desc}\n")
            f.write(f"backend: {sam_backend}\n")
            if sam_backend == "sam1":
                f.write("perception: grounded-dino + sam1 (per-frame, original libero_goal)\n")
            else:
                f.write("perception: sam3 (click points + temporal tracking)\n")

        print(f"  saved {task_dir.name} ({len(saved_frames)} frames: {saved_frames})", flush=True)

    print(f"DONE -> {out_root}")


if __name__ == "__main__":
    main()
