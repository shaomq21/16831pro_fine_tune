#!/usr/bin/env python3
"""Quick tracking validation: one sequential pass, save a few key frames."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds
from PIL import Image

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "openvla-oft"))

from mask_processor import GroundedSAMConfig, GroundedSAMMasker, EpisodeMaskTracker
from sam3_backend import DEFAULT_SAM3_CKPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--save_frames", type=str, default="0,55,109")
    ap.add_argument("--max_frame", type=int, default=110)
    ap.add_argument("--out", type=str, default="rlds_mask_debug/libero_spatial_no_noops/preview/track_val")
    ap.add_argument("--data_root", type=str,
        default="/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/datasets/modified_libero_rlds")
    args = ap.parse_args()

    save_set = {int(x) for x in args.save_frames.split(",")}
    out = _REPO / args.out / f"ep{args.ep:03d}"
    out.mkdir(parents=True, exist_ok=True)

    op = _REPO / "openvla-oft"
    cfg = GroundedSAMConfig(
        dino_config_path=str(op / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
        dino_checkpoint_path=str(op / "groundingdino_swint_ogc.pth"),
        sam_checkpoint_path=str(op / "sam_vit_b_01ec64.pth"),
        sam_backend="sam3",
        sam3_checkpoint_path=DEFAULT_SAM3_CKPT,
        device="cuda:0",
    )
    tracker = EpisodeMaskTracker(GroundedSAMMasker(cfg))

    builder = tfds.builder("libero_spatial_no_noops", data_dir=args.data_root)
    ep = next(iter(builder.as_dataset(split="train", shuffle_files=False).skip(args.ep).take(1)))
    steps = list(ep["steps"].as_numpy_iterator())
    lang = steps[0]["language_instruction"].decode().lower()
    n = min(args.max_frame, len(steps))

    log = []
    for fi in range(n):
        img = Image.fromarray(steps[fi]["observation"]["image"]).convert("RGB")
        masked = tracker.mask_image_from_lang(img, lang)
        if fi in save_set:
            img.save(out / f"frame{fi:03d}_raw.png")
            masked.save(out / f"frame{fi:03d}_masked.png")
            entry = {
                "frame": fi,
                "center": tracker.prev_red_center,
                "box": tracker.prev_red_box.tolist() if tracker.prev_red_box is not None else None,
            }
            log.append(entry)
            print(f"saved frame {fi}: center={tracker.prev_red_center}")

    meta = {"episode": args.ep, "lang": lang, "frames": log}
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("done ->", out)


if __name__ == "__main__":
    main()
