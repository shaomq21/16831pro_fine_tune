#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
from PIL import Image

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
    out = masker.mask_image_from_lang(img, args.lang)
    os.makedirs(os.path.dirname(args.image_out), exist_ok=True)
    out.save(args.image_out)
    print(args.image_out, flush=True)

if __name__ == "__main__":
    main()
