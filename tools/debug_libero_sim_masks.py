#!/usr/bin/env python3
"""
Generate sim-based mask debug images for all LIBERO task suites.

Default: masks + gripper finger dots from SegmentationRenderEnv (BDDL obj_of_interest).
Only use Grounded-SAM / Roboflow when PERCEPTION_MODE=real_perception.

Requires: conda env with LIBERO + MuJoCo EGL (e.g. ``subopt``):
  export MUJOCO_GL=egl
  conda activate subopt
  python tools/debug_libero_sim_masks.py

  # specific suites
  python tools/debug_libero_sim_masks.py --suites libero_spatial libero_goal

  # more mid-episode frames (RLDS action replay)
  python tools/debug_libero_sim_masks.py --replay_rlds --frame_fracs 0,0.25,0.5,0.75
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "openvla-oft"))

from libero_sim_mask import LiberoSimMasker, gripper_finger_pixels  # noqa: E402

ALL_SUITES = ("libero_spatial", "libero_goal", "libero_object", "libero_90")
SUITE_DATA_MIX = {s: f"{s}_no_noops" for s in ALL_SUITES}


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


def export_rlds_episodes(data_mix: str, data_root: str, out_json: Path) -> dict:
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
        episodes[lang] = {"n": len(steps), "actions": [s["action"].tolist() for s in steps]}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(episodes, f)
    return episodes


def load_episodes_json(path: Path) -> dict:
    with open(path) as f:
        raw = json.load(f)
    return {k: {"actions": v["actions"], "steps": None} for k, v in raw.items()}


def generate_suite(
    suite: str,
    out_root: Path,
    *,
    init_idx: int = 0,
    replay_rlds: bool = False,
    episodes: dict | None = None,
    frame_fracs: list[float],
    init_only: bool = False,
) -> tuple[int, int]:
    masker = LiberoSimMasker(suite)
    masker._ensure_libero()
    n_tasks = masker._bench.n_tasks
    ok, fail = 0, 0

    suite_dir = out_root / suite
    suite_dir.mkdir(parents=True, exist_ok=True)

    for task_id in range(n_tasks):
        task = masker._bench.get_task(task_id)
        task_dir = suite_dir / task.name
        task_dir.mkdir(parents=True, exist_ok=True)

        if replay_rlds and episodes and task.language.lower().strip() in episodes:
            lang = task.language.lower().strip()
            ep = episodes[lang]
            actions = ep["actions"]
            n = len(actions)
            indices = [0] if init_only else _frame_indices(n, frame_fracs)
            saved = []
            try:
                for fi in indices:
                    masked, raw, meta = masker.mask_at_rlds_step(
                        lang, actions, fi, init_idx=init_idx
                    )
                    raw.save(task_dir / f"frame{fi:03d}_raw.png")
                    masked.save(task_dir / f"frame{fi:03d}_masked.png")
                    saved.append(fi)
                ok += 1
            except Exception as e:
                fail += 1
                with open(task_dir / "error.txt", "w") as f:
                    f.write(str(e))
                print(f"  FAIL {task.name}: {e}", flush=True)
                continue
            meta_line = f"gripper: {meta.gripper_pixels}\n"
        else:
            try:
                masked, raw, meta = masker.mask_at_init(
                    task_id=task_id, init_idx=init_idx, return_raw=True
                )
                raw.save(task_dir / "frame000_raw.png")
                masked.save(task_dir / "frame000_masked.png")
                saved = [0]
                ok += 1
                meta_line = f"gripper: {meta.gripper_pixels}\n"
            except Exception as e:
                fail += 1
                with open(task_dir / "error.txt", "w") as f:
                    f.write(str(e))
                print(f"  FAIL {task.name}: {e}", flush=True)
                continue

        with open(task_dir / "task.txt", "w") as f:
            f.write(f"{task.language}\n")
            f.write(f"saved_frames: {saved}\n")
            f.write(f"red: {meta.red_object}\n")
            f.write(f"green: {meta.green_object}\n")
            f.write(meta_line)
            f.write(f"init_idx: {init_idx}\n")
            f.write("perception: sim (BDDL obj_of_interest + finger tips)\n")

        print(f"  OK {task.name} frames={saved}", flush=True)

    masker.close()
    return ok, fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="+", default=list(ALL_SUITES))
    ap.add_argument("--out_dir", default="debug_masked_validation/libero_all_sim")
    ap.add_argument("--init_idx", type=int, default=0)
    ap.add_argument("--replay_rlds", action="store_true")
    ap.add_argument("--data_root", default=str(_REPO_ROOT / "openvla-oft/datasets/modified_libero_rlds"))
    ap.add_argument("--frame_fracs", default="0,0.25,0.5,0.75")
    ap.add_argument("--init_only", action="store_true", help="Only frame 0 (faster for libero_90)")
    ap.add_argument("--export_rlds_json", action="store_true", help="Export RLDS actions to JSON (needs tensorflow)")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    fracs = _parse_frame_fracs(args.frame_fracs)

    total_ok, total_fail = 0, 0
    for suite in args.suites:
        print(f"\n=== {suite} ===", flush=True)
        episodes = None
        ep_json = out_root / f"{suite}_episodes.json"
        if args.replay_rlds:
            if args.export_rlds_json or not ep_json.exists():
                data_mix = SUITE_DATA_MIX.get(suite, f"{suite}_no_noops")
                print(f"Exporting RLDS episodes -> {ep_json}", flush=True)
                try:
                    episodes = export_rlds_episodes(data_mix, args.data_root, ep_json)
                except Exception as e:
                    print(f"WARNING: RLDS export failed ({e}); init-only for {suite}", flush=True)
            else:
                episodes = load_episodes_json(ep_json)
                print(f"Loaded {len(episodes)} episodes from {ep_json}", flush=True)

        use_init_only = args.init_only or (suite == "libero_90" and not args.replay_rlds)
        ok, fail = generate_suite(
            suite,
            out_root,
            init_idx=args.init_idx,
            replay_rlds=args.replay_rlds and episodes is not None,
            episodes=episodes,
            frame_fracs=fracs,
            init_only=use_init_only,
        )
        total_ok += ok
        total_fail += fail
        print(f"{suite}: {ok} ok, {fail} fail", flush=True)

    print(f"\nDONE -> {out_root}  ({total_ok} ok, {total_fail} fail)", flush=True)


if __name__ == "__main__":
    main()
