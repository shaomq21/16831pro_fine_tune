#!/usr/bin/env python3
"""
Compare similarity between two hidden states produced by extract_hidden_states.py.
"""

import argparse
import numpy as np
from pathlib import Path


def compare_hidden_states(hs1: np.ndarray, hs2: np.ndarray) -> dict:
    """Compute various similarity metrics between two hidden states."""
    h1 = hs1.astype(np.float32).flatten()
    h2 = hs2.astype(np.float32).flatten()
    assert h1.shape == h2.shape, f"Shape mismatch: {hs1.shape} vs {hs2.shape}"

    # Cosine similarity (1 = identical, 0 = orthogonal, -1 = opposite)
    cos_sim = np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-8)

    # L2 distance (0 = identical)
    l2_dist = np.linalg.norm(h1 - h2)

    # Normalized L2 distance (for cross-scale comparison)
    l2_norm = l2_dist / (np.linalg.norm(h1) + np.linalg.norm(h2) + 1e-8)

    # Pearson correlation
    h1_centered = h1 - h1.mean()
    h2_centered = h2 - h2.mean()
    pearson = np.dot(h1_centered, h2_centered) / (
        np.sqrt(np.dot(h1_centered, h1_centered)) * np.sqrt(np.dot(h2_centered, h2_centered)) + 1e-8
    )

    # Per-token cosine similarity (if 3D: batch, seq, dim)
    if hs1.ndim == 3:
        per_token_cos = []
        for i in range(hs1.shape[1]):
            t1 = hs1[0, i].flatten()
            t2 = hs2[0, i].flatten()
            c = np.dot(t1, t2) / (np.linalg.norm(t1) * np.linalg.norm(t2) + 1e-8)
            per_token_cos.append(c)
        per_token_mean = np.mean(per_token_cos)
        per_token_min = np.min(per_token_cos)
        per_token_max = np.max(per_token_cos)
    else:
        per_token_mean = per_token_min = per_token_max = float("nan")

    return {
        "cosine_similarity": float(cos_sim),
        "l2_distance": float(l2_dist),
        "l2_normalized": float(l2_norm),
        "pearson_correlation": float(pearson),
        "per_token_cos_mean": float(per_token_mean) if not np.isnan(per_token_mean) else None,
        "per_token_cos_min": float(per_token_min) if not np.isnan(per_token_min) else None,
        "per_token_cos_max": float(per_token_max) if not np.isnan(per_token_max) else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare similarity between two hidden states")
    parser.add_argument(
        "--dir",
        type=str,
        default="./output_hidden_states",
        help="Directory containing prompt_1_hidden_states.npy and prompt_2_hidden_states.npy",
    )
    args = parser.parse_args()

    d = Path(args.dir)
    f1 = d / "prompt_1_hidden_states.npy"
    f2 = d / "prompt_2_hidden_states.npy"
    if not f1.exists():
        raise FileNotFoundError(f"Not found: {f1}")
    if not f2.exists():
        raise FileNotFoundError(f"Not found: {f2}")

    hs1 = np.load(f1)
    hs2 = np.load(f2)
    print(f"Prompt 1 hidden states shape: {hs1.shape}")
    print(f"Prompt 2 hidden states shape: {hs2.shape}")

    metrics = compare_hidden_states(hs1, hs2)

    print("\n--- Similarity Metrics ---")
    print(f"  Cosine similarity:    {metrics['cosine_similarity']:.6f}")
    print(f"  Pearson correlation:  {metrics['pearson_correlation']:.6f}")
    print(f"  L2 distance:          {metrics['l2_distance']:.4f}")
    print(f"  Normalized L2 dist:   {metrics['l2_normalized']:.6f}")
    if metrics["per_token_cos_mean"] is not None:
        print(f"  Per-token cos (mean): {metrics['per_token_cos_mean']:.6f}")
        print(f"  Per-token cos (min):  {metrics['per_token_cos_min']:.6f}")
        print(f"  Per-token cos (max):  {metrics['per_token_cos_max']:.6f}")

    cos = metrics["cosine_similarity"]
    if cos > 0.99:
        conclusion = "highly similar"
    elif cos > 0.9:
        conclusion = "quite similar"
    elif cos > 0.7:
        conclusion = "moderately similar"
    else:
        conclusion = "quite different"
    print(f"\nConclusion: The two hidden states are {conclusion} (cosine similarity: {cos:.4f})")


if __name__ == "__main__":
    main()
