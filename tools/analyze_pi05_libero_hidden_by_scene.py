#!/usr/bin/env python3
"""Analyze pi05_libero hidden similarity within same-scene LIBERO task groups.

Uses pretrained pi05_libero (no finetuning). Extracts:
  - vlm_prefix_l18      (front patch hidden, mean over patches)
  - vlm_prefix_l18_lang (language token hidden, masked mean)

Grouping modes (--grouping):
  coarse_scene   - group by BDDL scene name / problem folder (default)
  identical_init - group by exact (:init ...) block in BDDL (same object placements)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from compare_pi05_vision_features import cosine_sim  # noqa: E402
from extract_pi05_vision_features import (  # noqa: E402
    _frame_to_batch,
    _masked_mean,
    extract_vlm_prefix_l18_front_patches,
    extract_vlm_prefix_l18_language_tokens,
    load_pi05_policy,
)

DEFAULT_STORAGE = "/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune"

SUITE_DATASETS = {
    "libero_spatial": "lerobot/libero_spatial_image",
    "libero_object": "lerobot/libero_object_image",
    "libero_goal": "lerobot/libero_goal_image",
    "libero_10": "lerobot/libero_10_image",
}

FEATURE_TYPES = ("vlm_prefix_l18", "vlm_prefix_l18_lang")


def _short_name(task: str) -> str:
    return (
        task.replace("pick up the ", "")
        .replace(" and place it ", "_to_")
        .replace(" ", "_")
        .replace("the_", "")
    )[:80]


def extract_init_block(bddl_text: str) -> str:
    match = re.search(r"\(:init\s+(.*?)\n\s*\)", bddl_text, re.S)
    return match.group(1).strip() if match else ""


def init_fingerprint(bddl_text: str) -> tuple[str, ...]:
    lines = [ln.strip() for ln in extract_init_block(bddl_text).split("\n") if ln.strip()]
    return tuple(sorted(lines))


def get_identical_init_groups(suite_name: str) -> dict[str, list[dict]]:
    """Group tasks whose BDDL (:init ...) predicates are identical."""
    from libero.libero import benchmark, get_libero_path

    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    by_fp: dict[tuple[str, ...], list[dict]] = {}
    bddl_root = Path(get_libero_path("bddl_files"))

    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        bddl_path = bddl_root / task.problem_folder / task.bddl_file
        fp = init_fingerprint(bddl_path.read_text() if bddl_path.exists() else "")
        by_fp.setdefault(fp, []).append(
            {
                "task_id": task_id,
                "language": task.language,
                "bddl_file": task.bddl_file,
                "short_name": _short_name(task.language),
                "init_block": extract_init_block(bddl_path.read_text() if bddl_path.exists() else ""),
            }
        )

    groups: dict[str, list[dict]] = {}
    for idx, (fp, tasks) in enumerate(sorted(by_fp.items(), key=lambda x: -len(x[1]))):
        group_id = f"identical_init_{idx:02d}_n{len(tasks)}"
        groups[group_id] = tasks
    return groups


def get_scene_groups(suite_name: str) -> dict[str, list[dict]]:
    """Return {scene_name: [{task_id, language, bddl_file}, ...]}."""
    from libero.libero import benchmark, get_libero_path

    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    groups: dict[str, list[dict]] = {}
    bddl_root = Path(get_libero_path("bddl_files"))

    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        bddl_path = bddl_root / task.problem_folder / task.bddl_file
        text = bddl_path.read_text() if bddl_path.exists() else ""

        scene = None
        for pat in [
            r"\(define\s+\(problem\s+(\S+)\)",
            r":scene\s+(\S+)",
            r"(\w+_SCENE\d+)",
        ]:
            mm = re.search(pat, text, re.I)
            if mm:
                scene = mm.group(1)
                break
        if scene is None:
            sm = re.match(r"([A-Z_]+SCENE\d+)_", task.bddl_file)
            scene = sm.group(1) if sm else task.problem_folder

        groups.setdefault(scene, []).append(
            {
                "task_id": task_id,
                "language": task.language,
                "bddl_file": task.bddl_file,
                "short_name": _short_name(task.language),
            }
        )
    return groups


def _normalize_frame(frame: dict) -> dict:
    frame = dict(frame)
    task = frame.get("task", "")
    if isinstance(task, bytes):
        frame["task"] = task.decode()
    if "observation.images.wrist_image" in frame and "observation.images.image2" not in frame:
        frame["observation.images.image2"] = frame["observation.images.wrist_image"]
    return frame


def dataset_root(lerobot_home: Path, repo_id: str) -> Path:
    """One root per HF repo to avoid meta/data collisions."""
    return lerobot_home / "hub" / repo_id.replace("/", "--")


def sample_frames_by_task_index(
    lerobot_home: Path,
    repo_id: str,
    task_ids: set[int],
    max_per_task: int,
    *,
    first_frame_only: bool = False,
) -> dict[int, list[dict]]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = dataset_root(lerobot_home, repo_id)
    ds = LeRobotDataset(repo_id, root=root)
    by_task: dict[int, list[dict]] = {tid: [] for tid in task_ids}
    for i in range(len(ds)):
        frame = _normalize_frame(dict(ds[i]))
        task_id = int(frame["task_index"])
        if task_id not in by_task:
            continue
        if first_frame_only:
            if int(frame.get("frame_index", -1)) != 0:
                continue
            if by_task[task_id]:
                continue
            by_task[task_id].append(frame)
            if all(by_task[tid] for tid in task_ids):
                break
            continue
        if len(by_task[task_id]) >= max_per_task:
            if all(len(v) >= max_per_task for v in by_task.values()):
                break
            continue
        by_task[task_id].append(frame)
    return by_task


def extract_feature_vector(
    feature_type: str,
    policy,
    preprocessor,
    frame: dict,
) -> np.ndarray:
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    batch = preprocessor(_frame_to_batch(frame))
    images, img_masks = policy._preprocess_images(batch)
    token = batch[OBS_LANGUAGE_TOKENS]
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    model = policy.model

    if feature_type == "vlm_prefix_l18":
        feats = extract_vlm_prefix_l18_front_patches(model, images, img_masks, token, masks)
        vec = feats.mean(dim=1)
    elif feature_type == "vlm_prefix_l18_lang":
        feats = extract_vlm_prefix_l18_language_tokens(model, images, img_masks, token, masks)
        vec = _masked_mean(feats, masks)
    else:
        raise ValueError(feature_type)
    return vec.cpu().float().numpy()[0]


def summarize_group(feats: dict[str, np.ndarray]) -> dict:
    keys = sorted(feats.keys())
    if len(keys) < 2:
        return {
            "n_tasks": len(keys),
            "mean_offdiag_cosine": None,
            "pairs": [],
            "cosine_matrix": None,
            "keys": keys,
        }

    n = len(keys)
    matrix = np.zeros((n, n))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            matrix[i, j] = cosine_sim(feats[ki], feats[kj])

    pairs = []
    offdiag = []
    for a, b in combinations(keys, 2):
        s = cosine_sim(feats[a], feats[b])
        pairs.append({"a": a, "b": b, "cosine": s})
        offdiag.append(s)

    return {
        "n_tasks": len(keys),
        "mean_offdiag_cosine": float(np.mean(offdiag)),
        "min_pair_cosine": float(min(offdiag)),
        "max_pair_cosine": float(max(offdiag)),
        "pairs": pairs,
        "cosine_matrix": matrix.tolist(),
        "keys": keys,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=f"{DEFAULT_STORAGE}/ckpts/pi05/pi05_libero",
    )
    parser.add_argument(
        "--lerobot_home",
        type=str,
        default=f"{DEFAULT_STORAGE}/lerobot_datasets",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"{DEFAULT_STORAGE}/runs/pi05_libero_pretrained_hidden_by_scene",
    )
    parser.add_argument(
        "--suites",
        type=str,
        default="libero_spatial,libero_object,libero_goal,libero_10",
    )
    parser.add_argument("--max_per_task", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--grouping",
        type=str,
        default="coarse_scene",
        choices=("coarse_scene", "identical_init"),
        help="coarse_scene: BDDL scene name; identical_init: exact (:init) match",
    )
    parser.add_argument(
        "--first_frame_only",
        action="store_true",
        help="Use only frame_index==0 (first frame of first demo episode per task)",
    )
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    suites = [s.strip() for s in args.suites.split(",") if s.strip()]

    print(f"Loading pi05 from {args.checkpoint}")
    policy, _, preprocessor = load_pi05_policy(args.checkpoint, args.device)

    report = {
        "checkpoint": args.checkpoint,
        "max_per_task": args.max_per_task,
        "grouping": args.grouping,
        "first_frame_only": args.first_frame_only,
        "feature_types": list(FEATURE_TYPES),
        "suites": {},
        "pooled": {ft: {"all_pair_cosines": [], "mean": None} for ft in FEATURE_TYPES},
    }

    get_groups = get_identical_init_groups if args.grouping == "identical_init" else get_scene_groups

    for suite in suites:
        repo_id = SUITE_DATASETS[suite]
        print(f"\n===== {suite} ({repo_id}) grouping={args.grouping} =====")
        scene_groups = get_groups(suite)
        suite_entry = {"dataset": repo_id, "scene_groups": {}}

        all_task_ids = {t["task_id"] for tasks in scene_groups.values() for t in tasks}
        print(f"  loading dataset, {len(all_task_ids)} tasks across {len(scene_groups)} scene groups")
        frames_by_task = sample_frames_by_task_index(
            Path(args.lerobot_home),
            repo_id,
            all_task_ids,
            args.max_per_task,
            first_frame_only=args.first_frame_only,
        )

        for scene_name, tasks in sorted(scene_groups.items()):
            print(f"  scene: {scene_name} ({len(tasks)} tasks)")
            group_entry = {
                "scene": scene_name,
                "tasks": tasks,
                "feature_types": {},
            }

            if len(tasks) < 2:
                print("    skip similarity (<2 tasks in group)")
                group_entry["skipped"] = "fewer than 2 tasks"
                suite_entry["scene_groups"][scene_name] = group_entry
                continue

            for feature_type in FEATURE_TYPES:
                task_feats: dict[str, np.ndarray] = {}
                for t in tasks:
                    tid = t["task_id"]
                    frames = frames_by_task.get(tid, [])
                    if not frames:
                        print(f"    WARNING: no frames for task_id={tid} ({t['short_name']})")
                        continue
                    vecs = [
                        extract_feature_vector(feature_type, policy, preprocessor, fr)
                        for fr in frames
                    ]
                    task_feats[t["short_name"]] = np.stack(vecs, axis=0).mean(axis=0)

                summary = summarize_group(task_feats)
                group_entry["feature_types"][feature_type] = summary
                if summary["mean_offdiag_cosine"] is not None:
                    for pair in summary["pairs"]:
                        report["pooled"][feature_type]["all_pair_cosines"].append(
                            {
                                "suite": suite,
                                "group": scene_name,
                                "a": pair["a"],
                                "b": pair["b"],
                                "cosine": pair["cosine"],
                            }
                        )
                    print(
                        f"    {feature_type}: mean_offdiag={summary['mean_offdiag_cosine']:.4f} "
                        f"range=[{summary['min_pair_cosine']:.4f}, {summary['max_pair_cosine']:.4f}]"
                    )

            suite_entry["scene_groups"][scene_name] = group_entry

        report["suites"][suite] = suite_entry

    for ft in FEATURE_TYPES:
        vals = [x["cosine"] for x in report["pooled"][ft]["all_pair_cosines"]]
        report["pooled"][ft]["mean"] = float(np.mean(vals)) if vals else None
        report["pooled"][ft]["n_pairs"] = len(vals)
        if vals:
            print(
                f"\nPOOLED {ft}: mean={report['pooled'][ft]['mean']:.4f} "
                f"over {len(vals)} pairs"
            )

    suffix_parts = ["identical_init" if args.grouping == "identical_init" else "by_scene"]
    if args.first_frame_only:
        suffix_parts.append("first_frame")
    suffix = "_".join(suffix_parts)
    report_path = out_root / f"hidden_similarity_{suffix}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Human-readable summary
    summary_lines = [
        f"# pi05_libero hidden similarity ({args.grouping}"
        f"{', first frame only' if args.first_frame_only else ''})",
        f"checkpoint: {args.checkpoint}",
        "",
    ]
    for ft in FEATURE_TYPES:
        p = report["pooled"][ft]
        if p["mean"] is not None:
            summary_lines.append(
                f"**Pooled {ft}**: mean={p['mean']:.4f} over {p['n_pairs']} task pairs"
            )
    summary_lines.append("")
    for suite, sdata in report["suites"].items():
        summary_lines.append(f"## {suite}")
        for scene, gdata in sdata["scene_groups"].items():
            summary_lines.append(f"### {scene} ({gdata.get('tasks') and len(gdata['tasks'])} tasks)")
            if gdata.get("skipped"):
                summary_lines.append(f"- skipped: {gdata['skipped']}")
                continue
            for ft in FEATURE_TYPES:
                s = gdata["feature_types"].get(ft, {})
                if s.get("mean_offdiag_cosine") is None:
                    continue
                summary_lines.append(
                    f"- **{ft}**: mean_offdiag={s['mean_offdiag_cosine']:.4f}, "
                    f"min={s['min_pair_cosine']:.4f}, max={s['max_pair_cosine']:.4f}"
                )
        summary_lines.append("")

    summary_path = out_root / f"summary_{suffix}.md"
    summary_path.write_text("\n".join(summary_lines))
    print(f"\nSaved {report_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
