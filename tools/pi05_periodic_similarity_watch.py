#!/usr/bin/env python3
"""Watch pi05 finetune checkpoints and periodically extract/compare vision similarity."""

from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
DEFAULT_STORAGE = "/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune"

FEATURE_TYPES = ("vision_tower", "vlm_prefix_l18")


def _step_from_ckpt(name: str) -> int:
    try:
        return int(name)
    except ValueError:
        return -1


def _is_ready(ckpt_dir: Path) -> bool:
    pretrained = ckpt_dir / "pretrained_model" / "model.safetensors"
    return pretrained.is_file()


def _mean_offdiag(matrix: list[list[float]]) -> float:
    n = len(matrix)
    if n < 2:
        return 1.0
    vals = []
    for i in range(n):
        for j in range(n):
            if i != j:
                vals.append(matrix[i][j])
    return float(sum(vals) / len(vals))


def _load_state(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return set(json.loads(path.read_text()))


def _save_state(path: Path, done: set[str]) -> None:
    path.write_text(json.dumps(sorted(done), indent=2) + "\n")


def _get_wandb_run_id(log_dir: Path) -> str | None:
    paths = glob.glob(str(log_dir / "wandb/latest-run/run-*"))
    if len(paths) != 1:
        return None
    match = re.search(r"run-([^\.]+)\.wandb", paths[0].split("/")[-1])
    return match.group(1) if match else None


class _WandbSimilarityLogger:
    def __init__(
        self,
        *,
        enabled: bool,
        log_dir: Path,
        project: str,
        entity: str | None,
        job_name: str,
    ) -> None:
        self.enabled = enabled
        self.log_dir = log_dir
        self.project = project
        self.entity = entity or None
        self.job_name = job_name
        self._wandb = None
        self._run_id: str | None = None

    def _ensure_init(self) -> bool:
        if not self.enabled or self._wandb is not None:
            return self._wandb is not None

        run_id = _get_wandb_run_id(self.log_dir)
        if run_id is None:
            return False

        import wandb

        wandb.init(
            id=run_id,
            project=self.project,
            entity=self.entity,
            name=self.job_name,
            dir=str(self.log_dir),
            resume="must",
            reinit=True,
        )
        self._wandb = wandb
        self._run_id = run_id
        print(f"[similarity_watch] wandb joined run id={run_id} url={wandb.run.get_url()}")
        return True

    def log_step(self, step: int, entries: list[dict]) -> None:
        if not self.enabled:
            return
        if not self._ensure_init():
            return

        payload = {}
        for entry in entries:
            ft = entry["feature_type"]
            payload[f"sim/{ft}/mean_offdiag_cosine"] = entry["mean_offdiag_cosine"]
        self._wandb.log(payload, step=step)
        print(f"[similarity_watch] wandb logged step={step} {payload}")


def _list_numeric_ckpts(ckpt_root: Path) -> list[Path]:
    return sorted(
        (d for d in ckpt_root.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )


def _prune_checkpoints(ckpt_root: Path, keep: int) -> None:
    import shutil

    ckpts = _list_numeric_ckpts(ckpt_root)
    if len(ckpts) <= keep:
        return
    for ckpt_dir in ckpts[:-keep]:
        print(f"[similarity_watch] prune checkpoint {ckpt_dir} (keep latest {keep})")
        shutil.rmtree(ckpt_dir)


def _append_timeline(timeline_path: Path, entry: dict) -> None:
    rows: list[dict] = []
    if timeline_path.is_file():
        rows = json.loads(timeline_path.read_text())
    rows.append(entry)
    timeline_path.write_text(json.dumps(rows, indent=2) + "\n")


def _run_feature_type(
    *,
    feature_type: str,
    python: str,
    extract_py: Path,
    compare_py: Path,
    pretrained: Path,
    feat_dir: Path,
    lerobot_home: str,
    repo_id: str,
    device: str,
    max_per_task: int,
) -> dict:
    out_dir = feat_dir / feature_type
    subprocess.run(
        [
            python,
            str(extract_py),
            "--checkpoint",
            str(pretrained),
            "--lerobot_home",
            lerobot_home,
            "--repo_id",
            repo_id,
            "--output_dir",
            str(out_dir),
            "--device",
            device,
            "--max_per_task",
            str(max_per_task),
            "--feature_type",
            feature_type,
        ],
        check=True,
    )
    subprocess.run([python, str(compare_py), "--feature_dir", str(out_dir)], check=True)
    sim_path = out_dir / "similarity.json"
    sim = json.loads(sim_path.read_text()) if sim_path.is_file() else {}
    return {
        "feature_type": feature_type,
        "feature_dir": str(out_dir),
        "mean_offdiag_cosine": _mean_offdiag(sim.get("cosine_matrix", [])),
        "pairs": sim.get("pairs", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_output_dir", type=str, required=True)
    parser.add_argument(
        "--analysis_dir",
        type=str,
        default=f"{DEFAULT_STORAGE}/runs/pi05_study_scene4_analysis/periodic",
    )
    parser.add_argument("--lerobot_home", type=str, default=f"{DEFAULT_STORAGE}/lerobot_datasets")
    parser.add_argument("--repo_id", type=str, default="local/libero_90_study_scene4")
    parser.add_argument("--poll_sec", type=int, default=120)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_per_task", type=int, default=16)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument(
        "--feature_types",
        type=str,
        default=",".join(FEATURE_TYPES),
        help="Comma-separated: vision_tower,vlm_prefix_l18",
    )
    parser.add_argument(
        "--prune_checkpoints",
        action="store_true",
        help="Delete old checkpoint dirs after feature extraction",
    )
    parser.add_argument(
        "--keep_checkpoints",
        type=int,
        default=2,
        help="When pruning, keep this many newest numeric checkpoint dirs (default: 2)",
    )
    parser.add_argument("--wandb_enable", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="pi05_study_scene4")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_job_name", type=str, default="pi05_study_scene4")
    parser.add_argument(
        "--wandb_log_dir",
        type=str,
        default="",
        help="Training output dir containing wandb/ (defaults to --train_output_dir)",
    )
    args = parser.parse_args()

    feature_types = [x.strip() for x in args.feature_types.split(",") if x.strip()]

    train_out = Path(args.train_output_dir)
    ckpt_root = train_out / "checkpoints"
    analysis = Path(args.analysis_dir)
    analysis.mkdir(parents=True, exist_ok=True)

    state_path = analysis / ".processed_checkpoints.json"
    timeline_path = analysis / "similarity_timeline.json"
    done = _load_state(state_path)

    extract_py = _SCRIPT_DIR / "extract_pi05_vision_features.py"
    compare_py = _SCRIPT_DIR / "compare_pi05_vision_features.py"
    wandb_log_dir = Path(args.wandb_log_dir) if args.wandb_log_dir else train_out
    wandb_logger = _WandbSimilarityLogger(
        enabled=args.wandb_enable,
        log_dir=wandb_log_dir,
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        job_name=args.wandb_job_name,
    )

    print(f"[similarity_watch] watching {ckpt_root} every {args.poll_sec}s")
    print(f"[similarity_watch] feature_types={feature_types}")
    print(f"[similarity_watch] results -> {analysis}")

    while True:
        if ckpt_root.is_dir():
            for ckpt_dir in sorted(ckpt_root.iterdir()):
                if not ckpt_dir.is_dir() or not ckpt_dir.name.isdigit():
                    continue
                key = ckpt_dir.name
                if key in done or not _is_ready(ckpt_dir):
                    continue

                step = _step_from_ckpt(key)
                pretrained = ckpt_dir / "pretrained_model"
                feat_dir = analysis / f"step_{key}"
                print(f"[similarity_watch] step={step} checkpoint={pretrained}")

                entries = []
                for feature_type in feature_types:
                    print(f"[similarity_watch] extracting {feature_type}")
                    entry = _run_feature_type(
                        feature_type=feature_type,
                        python=args.python,
                        extract_py=extract_py,
                        compare_py=compare_py,
                        pretrained=pretrained,
                        feat_dir=feat_dir,
                        lerobot_home=args.lerobot_home,
                        repo_id=args.repo_id,
                        device=args.device,
                        max_per_task=args.max_per_task,
                    )
                    entries.append(entry)
                    print(
                        f"[similarity_watch] {feature_type} mean_offdiag_cosine="
                        f"{entry['mean_offdiag_cosine']:.4f}"
                    )

                _append_timeline(
                    timeline_path,
                    {
                        "step": step,
                        "checkpoint": str(pretrained),
                        "feature_dir": str(feat_dir),
                        "features": entries,
                    },
                )
                wandb_logger.log_step(step, entries)
                done.add(key)
                _save_state(state_path, done)

                if args.prune_checkpoints:
                    _prune_checkpoints(ckpt_root, args.keep_checkpoints)

        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
