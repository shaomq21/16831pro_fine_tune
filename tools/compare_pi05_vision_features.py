#!/usr/bin/env python3
"""Pairwise cosine similarity of pi05 vision backbone features across STUDY_SCENE4 tasks."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

DEFAULT_STORAGE = "/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune"


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def task_mean_features(path: Path) -> np.ndarray:
    arr = np.load(path)
    return arr.mean(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature_dir",
        type=str,
        default=f"{DEFAULT_STORAGE}/runs/pi05_study_scene4_analysis/vision_features_finetuned",
    )
    args = parser.parse_args()

    d = Path(args.feature_dir)
    meta_path = d / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}; run extract_pi05_vision_features.py first")

    with open(meta_path) as f:
        meta = json.load(f)

    suffix = meta.get("feature_suffix", "vision_features")
    feature_label = meta.get("feature_description", meta.get("feature_type", "unknown"))

    keys = []
    feats = {}
    for task, info in meta["tasks"].items():
        key = info["key"]
        fpath = d / f"{key}_{suffix}.npy"
        if not fpath.exists():
            print(f"SKIP missing: {fpath}")
            continue
        keys.append(key)
        feats[key] = task_mean_features(fpath)

    print(f"\nFeature similarity ({feature_label})")
    print(f"Checkpoint: {meta['checkpoint']}\n")

    # Pairwise matrix
    n = len(keys)
    matrix = np.zeros((n, n))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            matrix[i, j] = cosine_sim(feats[ki], feats[kj])

    header = " " * 28 + "  ".join(f"{k[:22]:>22}" for k in keys)
    print(header)
    for i, ki in enumerate(keys):
        row = f"{ki[:26]:26}  " + "  ".join(f"{matrix[i, j]:22.4f}" for j in range(n))
        print(row)

    print("\n--- Pairwise details ---")
    pairs = []
    for a, b in combinations(keys, 2):
        sim = cosine_sim(feats[a], feats[b])
        pairs.append((a, b, sim))
        label = "highly similar" if sim > 0.99 else "quite similar" if sim > 0.9 else "moderate" if sim > 0.7 else "different"
        print(f"  {a} <-> {b}: cos={sim:.4f} ({label})")

    out = {
        "keys": keys,
        "cosine_matrix": matrix.tolist(),
        "pairs": [{"a": a, "b": b, "cosine": s} for a, b, s in pairs],
    }
    with open(d / "similarity.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {d / 'similarity.json'}")


if __name__ == "__main__":
    main()
