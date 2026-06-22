#!/usr/bin/env python3
"""Add gripper white dots to already-masked RLDS images using RLDS proprio state."""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_OPENVLA_ROOT = _REPO_ROOT / "openvla-oft"
sys.path.insert(0, str(_OPENVLA_ROOT))

from gripper_project import draw_gripper_dots_on_rgb  # noqa: E402

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


def _np_value(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return x


def _patch_episode_dict(episode: dict) -> dict:
    steps = []
    for step in episode["steps"]:
        step = {k: _np_value(v) if not isinstance(v, dict) else v for k, v in step.items()}
        obs = dict(step["observation"])
        for ok, ov in list(obs.items()):
            obs[ok] = _np_value(ov)
        img = obs.get("image")
        state = obs.get("state")
        joint = obs.get("joint_state")
        if img is not None and state is not None:
            arr = np.asarray(img)
            st = np.asarray(state)
            js = np.asarray(joint) if joint is not None else None
            obs["image"] = draw_gripper_dots_on_rgb(arr, st, joint_state=js)
        step["observation"] = obs
        steps.append(step)
    meta = episode.get("episode_metadata", {})
    if isinstance(meta, dict):
        meta = {k: _np_value(v) for k, v in meta.items()}
    return {"steps": steps, "episode_metadata": meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_mix", default="libero_spatial_no_noops")
    ap.add_argument(
        "--data_root",
        default="/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/datasets/masked_libero_rlds",
    )
    ap.add_argument("--num_shards", type=int, default=16)
    args = ap.parse_args()

    tf.config.set_visible_devices([], "GPU")
    data_root = Path(args.data_root)
    out_dir = data_root / args.data_mix / "1.0.0"
    prefix = TFRECORD_PREFIX.get(args.data_mix, args.data_mix.replace("_no_noops", ""))

    src_root = Path("/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/datasets/modified_libero_rlds")
    builder = tfds.builder(args.data_mix, data_dir=str(src_root))
    features = builder.info.features
    from tensorflow_datasets.core import example_serializer

    serializer = example_serializer.ExampleSerializer(features.get_serialized_info())

    tmp_dir = out_dir / "_gripper_patch_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    shard_counts = [0] * args.num_shards
    total_eps = 0

    shard_paths = sorted(out_dir.glob(f"{prefix}-train.tfrecord-*"))
    for shard_id, shard_path in enumerate(shard_paths):
        records = _read_complete_tfrecords(shard_path)
        print(f"{shard_path.name}: {len(records)} complete records", flush=True)
        writer = tf.io.TFRecordWriter(
            str(tmp_dir / f"{prefix}-train.tfrecord-{shard_id:05d}-of-{args.num_shards:05d}")
        )
        for rec in tqdm(records, desc=f"shard {shard_id:02d}"):
            decoded = features.deserialize_example(rec)
            steps_list = []
            for s in decoded["steps"]:
                sd = {}
                for k, v in s.items():
                    if k == "observation":
                        sd[k] = {ok: _np_value(ov) for ok, ov in v.items()}
                    else:
                        sd[k] = _np_value(v)
                steps_list.append(sd)
            meta = decoded.get("episode_metadata", {})
            if isinstance(meta, dict):
                meta = {k: _np_value(v) for k, v in meta.items()}
            patched = _patch_episode_dict({"steps": steps_list, "episode_metadata": meta})
            encoded = features.encode_example(patched)
            serialized = encoded if isinstance(encoded, bytes) else serializer.serialize_example(encoded)
            writer.write(serialized)
            shard_counts[shard_id] += 1
            total_eps += 1
        writer.close()

    for i in range(args.num_shards):
        src = tmp_dir / f"{prefix}-train.tfrecord-{i:05d}-of-{args.num_shards:05d}"
        dst = out_dir / src.name
        os.replace(src, dst)
    tmp_dir.rmdir()

    info_path = out_dir / "dataset_info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
    else:
        with open(src_root / args.data_mix / "1.0.0" / "dataset_info.json") as f:
            info = json.load(f)
    for s in info.get("splits", []):
        if s.get("name") == "train":
            s["shardLengths"] = [str(c) for c in shard_counts]
            break
    with open(info_path, "w") as f:
        json.dump(info, f, indent=1)

    resume_path = data_root / f".rlds_resume_{args.data_mix}.json"
    with open(resume_path, "w") as f:
        json.dump({"last_episode": total_eps}, f)

    print(f"Patched {total_eps} episodes -> {out_dir}")


if __name__ == "__main__":
    main()
