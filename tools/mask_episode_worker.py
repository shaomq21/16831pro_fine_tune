#!/usr/bin/env python3
"""Persistent mask worker for eval (matches RLDS training backends).

Keeps GroundedSAM / SAM3 loaded and (for sam3) EpisodeMaskTracker state across frames.

Protocol (stdin lines):
  READY wait       -> prints READY
  RESET            -> clear episode tracker; prints OK
  MASK <in> <out> <alpha> <lang...>
                   -> write masked RGB to <out>; prints OK
  QUIT             -> exit

Run with vla-preprocess python (has groundingdino + ultralytics SAM3).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_tools = Path(__file__).resolve().parent
_oft = _tools.parent / "openvla-oft"
sys.path.insert(0, str(_oft))

from PIL import Image  # noqa: E402

from mask_processor import (  # noqa: E402
    EpisodeMaskTracker,
    GroundedSAMConfig,
    GroundedSAMMasker,
)
from sam3_backend import DEFAULT_SAM3_CKPT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam_backend", default="sam1", choices=["sam1", "sam3"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dino_config", default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    ap.add_argument("--dino_ckpt", default="groundingdino_swint_ogc.pth")
    ap.add_argument("--sam_ckpt", default="sam_vit_b_01ec64.pth")
    ap.add_argument("--sam_type", default="vit_b")
    ap.add_argument("--sam3_ckpt", default=DEFAULT_SAM3_CKPT)
    ap.add_argument("--fast", action="store_true", help="SAM3 fast_mode (skip video clip)")
    args = ap.parse_args()

    # Resolve ckpt paths relative to openvla-oft cwd expectation
    os.chdir(_oft)

    cfg = GroundedSAMConfig(
        dino_config_path=args.dino_config,
        dino_checkpoint_path=args.dino_ckpt,
        sam_checkpoint_path=args.sam_ckpt,
        sam_type=args.sam_type,
        sam_backend=args.sam_backend,
        sam3_checkpoint_path=args.sam3_ckpt,
        device=args.device,
        fast_mode=args.fast,
    )
    print(f"Loading masker backend={args.sam_backend} device={args.device}...", flush=True)
    masker = GroundedSAMMasker(cfg)
    tracker = EpisodeMaskTracker(masker) if args.sam_backend == "sam3" else None
    print("READY", flush=True)

    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        if line == "QUIT":
            print("OK", flush=True)
            break
        if line == "RESET":
            if tracker is not None:
                tracker.reset()
            print("OK", flush=True)
            continue
        if line.startswith("MASK "):
            # MASK <in> <out> <alpha> <lang...>
            parts = line.split(" ", 4)
            if len(parts) < 5:
                print("ERROR bad MASK args", flush=True)
                continue
            _, in_path, out_path, alpha_s, lang = parts
            try:
                alpha = float(alpha_s)
                img = Image.open(in_path).convert("RGB")
                if tracker is not None:
                    out = tracker.mask_image_from_lang(img, lang, alpha=alpha)
                else:
                    out = masker.mask_image_from_lang(img, lang, alpha=alpha)
                out.convert("RGB").save(out_path)
                print("OK", flush=True)
            except Exception as e:
                print(f"ERROR {type(e).__name__}: {e}", flush=True)
            continue
        print(f"ERROR unknown cmd: {line[:80]}", flush=True)


if __name__ == "__main__":
    main()
