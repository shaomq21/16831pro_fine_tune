#!/usr/bin/env python3
"""
Download OpenVLA 7B from Hugging Face Hub to a local directory for use as base model.
Usage:
  python scripts/download_openvla_7b.py [--output-dir DIR]
  Default output: openvla-oft/checkpoints/openvla-7b
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "openvla/openvla-7b"


def main():
    parser = argparse.ArgumentParser(description="Download OpenVLA 7B as base model")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Local directory to save the model (default: <repo_root>/checkpoints/openvla-7b)",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        # Default: project root is parent of 'scripts', so checkpoints/openvla-7b
        repo_root = Path(__file__).resolve().parent.parent
        args.output_dir = repo_root / "checkpoints" / "openvla-7b"

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID} to {args.output_dir} ...")
    path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(args.output_dir),
    )
    print(f"Done. Base model saved at: {path}")
    print(f"Use as base model: set vla_path='{path}' in your finetune config.")


if __name__ == "__main__":
    main()
