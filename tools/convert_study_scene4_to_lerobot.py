#!/usr/bin/env python3
"""Convert libero_90 STUDY_SCENE4 book tasks (tasks_info.txt 82-85) to LeRobot format."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import tensorflow_datasets as tfds

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_OPENVLA_ROOT = _REPO_ROOT / "openvla-oft"
sys.path.insert(0, str(_OPENVLA_ROOT))

from mask_spatial import LIBERO_90_STUDY_SCENE4_TASKS  # noqa: E402

DEFAULT_DATA_DIR = (
    "/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/datasets/modified_libero_rlds"
)
DEFAULT_LEROBOT_HOME = (
    "/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/lerobot_datasets"
)
DEFAULT_REPO_ID = "local/libero_90_study_scene4"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--lerobot_home",
        type=str,
        default=DEFAULT_LEROBOT_HOME,
        help=f"LeRobot dataset root on external disk (default: {DEFAULT_LEROBOT_HOME})",
    )
    args = parser.parse_args()

    try:
        from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    if args.lerobot_home:
        import os

        os.environ["HF_LEROBOT_HOME"] = args.lerobot_home
        from lerobot.datasets import lerobot_dataset as ld

        ld.HF_LEROBOT_HOME = Path(args.lerobot_home)
        output_path = Path(args.lerobot_home) / args.repo_id
    else:
        output_path = HF_LEROBOT_HOME / args.repo_id

    allow = frozenset(LIBERO_90_STUDY_SCENE4_TASKS)
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="panda",
        fps=10,
        features={
            "observation.images.image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.image2": {
                "dtype": "image",
                "shape": (128, 128, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=8,
        image_writer_processes=4,
    )

    raw = tfds.load("libero_90_no_noops", data_dir=args.data_dir, split="train")
    n_episodes = 0
    n_frames = 0
    for episode in raw:
        steps = list(episode["steps"].as_numpy_iterator())
        if not steps:
            continue
        lang0 = steps[0]["language_instruction"].decode()
        if lang0 not in allow:
            continue
        for step in steps:
            dataset.add_frame(
                {
                    "observation.images.image": step["observation"]["image"],
                    "observation.images.image2": step["observation"]["wrist_image"],
                    "observation.state": step["observation"]["state"],
                    "action": step["action"],
                    "task": step["language_instruction"].decode(),
                }
            )
            n_frames += 1
        dataset.save_episode()
        n_episodes += 1

    print(f"Saved {n_episodes} episodes, {n_frames} frames -> {output_path}")
    print(f"Tasks: {len(allow)}")


if __name__ == "__main__":
    main()
