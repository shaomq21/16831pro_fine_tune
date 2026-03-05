#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

# mask_processor 在 openvla-oft 下，保证子进程能导入
_tools_dir = Path(__file__).resolve().parent
_openvla_root = _tools_dir.parent / "openvla-oft"
if _openvla_root.is_dir():
    sys.path.insert(0, str(_openvla_root))

from mask_processor import GroundedSAMMasker, GroundedSAMConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_in", required=True)
    ap.add_argument("--image_out", required=True)
    ap.add_argument("--lang", required=True)

    ap.add_argument("--dino_config", required=True)
    ap.add_argument("--dino_ckpt", required=True)
    ap.add_argument("--sam_ckpt", required=True)
    ap.add_argument("--sam_type", default="vit_h")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alpha", type=float, default=0.35, help="0=black bg + original objects, 0.35=red/green overlay")
    ap.add_argument("--shift_plate_mask_x", type=int, default=None, help="For 'put bowl on plate': shift green plate box by N pixels (right=positive); mask will be beside plate.")
    ap.add_argument("--hide_green", action="store_true", help="Do not draw green (plate/destination) mask; only red mask shown.")
    ap.add_argument("--green_mask_out", default=None, help="If set, save green (destination) mask as binary PNG (255=green region, 0=else). Used e.g. for push-stove mask eval.")
    ap.add_argument("--red_mask_out", default=None, help="If set, save red (source/plate) mask as binary PNG (255=red region, 0=else).")

    args = ap.parse_args()

    cfg = GroundedSAMConfig(
        dino_config_path=args.dino_config,
        dino_checkpoint_path=args.dino_ckpt,
        sam_checkpoint_path=args.sam_ckpt,
        sam_type=args.sam_type,
        device=args.device,
    )
    masker = GroundedSAMMasker(cfg)

    img = Image.open(args.image_in).convert("RGB")
    need_masks = bool(getattr(args, "green_mask_out", None) or getattr(args, "red_mask_out", None))
    out = masker.mask_image_from_lang(
        img,
        args.lang,
        alpha=args.alpha,
        shift_green_plate_pixels=args.shift_plate_mask_x,
        draw_green=not getattr(args, "hide_green", False),
        return_masks=need_masks,
    )
    if need_masks:
        out_pil, red_mask, green_mask = out
        out = out_pil
        green_path = getattr(args, "green_mask_out", None)
        if green_path:
            os.makedirs(os.path.dirname(green_path) or ".", exist_ok=True)
            green_uint8 = np.where(green_mask, 255, 0).astype(np.uint8)
            Image.fromarray(green_uint8, mode="L").save(green_path)
            print(green_path, flush=True)
        red_path = getattr(args, "red_mask_out", None)
        if red_path:
            os.makedirs(os.path.dirname(red_path) or ".", exist_ok=True)
            red_uint8 = np.where(red_mask, 255, 0).astype(np.uint8)
            Image.fromarray(red_uint8, mode="L").save(red_path)
            print(red_path, flush=True)

    os.makedirs(os.path.dirname(args.image_out) or ".", exist_ok=True)
    out.save(args.image_out)
    print(args.image_out, flush=True)

if __name__ == "__main__":
    main()
