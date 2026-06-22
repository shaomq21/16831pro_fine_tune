"""Shared helpers for RLDS mask resume / TFRecord coverage."""

from __future__ import annotations

import json
import struct
from pathlib import Path

TFRECORD_PREFIX = {
    "libero_goal_no_noops": "libero_goal",
    "libero_object_no_noops": "libero_object",
    "libero_spatial_no_noops": "libero_spatial",
    "libero_90_no_noops": "libero_90",
    "libero_10_no_noops": "libero_10",
}

DEFAULT_OUT_ROOT = "/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/datasets/masked_libero_rlds"


def count_tfrecord_examples(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    n = 0
    with open(path, "rb") as f:
        while True:
            header = f.read(12)
            if len(header) < 12:
                break
            (length,) = struct.unpack("<Q", header[:8])
            if length <= 0 or length > 500_000_000:
                break
            f.seek(length, 1)
            if len(f.read(4)) < 4:
                break
            n += 1
    return n


def shard_record_counts(out_root: Path, data_mix: str, n_shards: int = 16) -> list[int]:
    prefix = TFRECORD_PREFIX.get(data_mix, data_mix.replace("_no_noops", ""))
    ver_dir = out_root / data_mix / "1.0.0"
    counts = []
    for sid in range(n_shards):
        path = ver_dir / f"{prefix}-train.tfrecord-{sid:05d}-of-{n_shards:05d}"
        counts.append(count_tfrecord_examples(path))
    return counts


def done_episodes_from_tfrecords(
    out_root: Path,
    data_mix: str,
    n_shards: int = 16,
    total_episodes: int | None = None,
) -> set[int]:
    """Global episode indices present in masked TFRecord shards."""
    if total_episodes is None:
        total_episodes = _total_episodes_from_info(out_root, data_mix)
    done: set[int] = set()
    for sid, cnt in enumerate(shard_record_counts(out_root, data_mix, n_shards)):
        for pos in range(cnt):
            ep = sid + pos * n_shards
            if ep < total_episodes:
                done.add(ep)
    return done


def _total_episodes_from_info(out_root: Path, data_mix: str) -> int:
    info_path = out_root / data_mix / "1.0.0" / "dataset_info.json"
    if info_path.exists():
        try:
            with open(info_path) as f:
                info = json.load(f)
            for split in info.get("splits", []):
                if split.get("name") == "train" and "shardLengths" in split:
                    return sum(int(x) for x in split["shardLengths"])
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    return 432


def worker_done_episodes(done: set[int], worker_id: int, num_workers: int) -> set[int]:
    return {ep for ep in done if ep % num_workers == worker_id}
