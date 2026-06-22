#!/usr/bin/env python3
"""
Debug masks from LIBERO sim segmentation (ground truth via BDDL obj_of_interest).

Requires conda env with LIBERO + MuJoCo EGL (e.g. ``subopt``):
  export MUJOCO_GL=egl
  conda activate subopt

  # init frame only
  python tools/debug_sim_masks.py --suite libero_spatial

  # multiple frames via RLDS action replay
  python tools/debug_sim_masks.py --suite libero_spatial --replay_rlds \\
      --frame_fracs 0,0.12,0.25,0.38,0.5,0.62,0.75,0.88
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_OPENVLA_ROOT = _REPO_ROOT / "openvla-oft"
sys.path.insert(0, str(_OPENVLA_ROOT))

from libero_sim_mask import LiberoSimMasker  # noqa: E402

SUITE_DATA_MIX = {
    "libero_spatial": "libero_spatial_no_noops",
    "libero_goal": "libero_goal_no_noops",
    "libero_object": "libero_object_no_noops",
    "libero_10": "libero_10_no_noops",
    "libero_90": "libero_90_no_noops",
}


def _parse_frame_fracs(s: str) -> list[float]:
    return sorted(set(max(0.0, min(1.0, float(p.strip()))) for p in s.split(",") if p.strip()))


def _frame_indices(n_steps: int, fracs: list[float]) -> list[int]:
    if n_steps <= 0:
        return []
    out, seen = [], set()
    for f in fracs:
        i = 0 if n_steps == 1 else min(n_steps - 1, max(0, int(round(f * (n_steps - 1)))))
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _safe_name(lang: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", lang.strip().lower())[:100]


def _load_rlds_episodes(data_mix: str, data_root: str, max_tasks: int):
    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")
    ds = tfds.builder(data_mix, data_dir=data_root).as_dataset(split="train", shuffle_files=False)
    episodes = {}
    for ep in ds:
        steps = list(ep["steps"].as_numpy_iterator())
        if not steps:
            continue
        lang = steps[0]["language_instruction"]
        if hasattr(lang, "decode"):
            lang = lang.decode("utf-8")
        lang = lang.lower().strip()
        if lang in episodes:
            continue
        actions = [s["action"] for s in steps]
        episodes[lang] = {"steps": steps, "actions": actions}
        if len(episodes) >= max_tasks:
            break
    return episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--init_idx", type=int, default=0)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--max_tasks", type=int, default=50)
    ap.add_argument("--replay_rlds", action="store_true", help="Replay RLDS actions for mid-episode frames")
    ap.add_argument("--episodes_pkl", default=None, help="Pickle/JSON episode actions (for subopt env)")
    ap.add_argument("--data_root", default=str(_OPENVLA_ROOT / "datasets/modified_libero_rlds"))
    ap.add_argument("--data_mix", default=None)
    ap.add_argument(
        "--frame_fracs",
        default="0,0.12,0.25,0.38,0.5,0.62,0.75,0.88",
        help="Episode fractions when --replay_rlds",
    )
    ap.add_argument("--num_frames", type=int, default=None)
    args = ap.parse_args()

    out_root = Path(args.out_dir or f"debug_masked_validation/{args.suite}_sim_multi")
    out_root.mkdir(parents=True, exist_ok=True)

    masker = LiberoSimMasker(args.suite)
    masker._ensure_libero()

    if args.replay_rlds:
        if args.episodes_pkl:
            path = args.episodes_pkl
            if path.endswith(".json"):
                import json
                with open(path) as f:
                    raw = json.load(f)
                episodes = {k: {"actions": v["actions"], "steps": None} for k, v in raw.items()}
            else:
                import pickle
                with open(path, "rb") as f:
                    episodes = pickle.load(f)
            print(f"Loaded {len(episodes)} episodes from {path}", flush=True)
        else:
            data_mix = args.data_mix or SUITE_DATA_MIX.get(args.suite, f"{args.suite}_no_noops")
            episodes = _load_rlds_episodes(data_mix, args.data_root, args.max_tasks)
            print(f"Replay sim masks: {len(episodes)} tasks from {data_mix}", flush=True)
        for lang, ep in sorted(episodes.items()):
            actions = ep["actions"]
            steps = ep.get("steps")
            n = len(actions) if steps is None else len(steps)
            if args.num_frames and args.num_frames > 0:
                fracs = [0.0] if args.num_frames == 1 else [i / (args.num_frames - 1) for i in range(args.num_frames)]
            else:
                fracs = _parse_frame_fracs(args.frame_fracs)
            indices = _frame_indices(n, fracs)
            task_dir = out_root / _safe_name(lang)
            task_dir.mkdir(parents=True, exist_ok=True)
            saved = []
            for fi in indices:
                masked, raw, meta = masker.mask_at_rlds_step(
                    lang, actions, fi, init_idx=args.init_idx
                )
                raw.save(task_dir / f"frame{fi:03d}_raw.png")
                masked.save(task_dir / f"frame{fi:03d}_masked.png")
                saved.append(fi)
            with open(task_dir / "task.txt", "w") as f:
                f.write(f"{meta.language}\n")
                f.write(f"episode_len: {n}\n")
                f.write(f"saved_frames: {saved}\n")
                f.write(f"red: {meta.red_object}\n")
                f.write(f"green: {meta.green_object}\n")
                f.write(f"init_idx: {args.init_idx}\n")
            print(f"  {meta.task_name}: frames {saved}", flush=True)
    else:
        n = min(args.max_tasks, masker._bench.n_tasks)
        print(f"Init-only sim masks for {args.suite} ({n} tasks)", flush=True)
        for task_id in range(n):
            masked, raw, meta = masker.mask_at_init(
                task_id=task_id, init_idx=args.init_idx, return_raw=True
            )
            task_dir = out_root / meta.task_name
            task_dir.mkdir(parents=True, exist_ok=True)
            raw.save(task_dir / "frame000_raw.png")
            masked.save(task_dir / "frame000_masked.png")
            with open(task_dir / "task.txt", "w") as f:
                f.write(f"{meta.language}\n")
                f.write(f"red: {meta.red_object}\n")
                f.write(f"green: {meta.green_object}\n")
            print(f"  saved {meta.task_name}", flush=True)

    masker.close()
    print(f"DONE -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
