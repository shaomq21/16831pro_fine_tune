#!/usr/bin/env python3
"""Live progress bar for tools/rlds_mask.py (episode + step level).

Reads .rlds_mask_progress_<data_mix>.json when the worker writes it (new runs).
Falls back to parsing main.log for SAM3 activity on older / already-running jobs.

Usage:
  python tools/monitor_rlds_mask_progress.py --data_mix libero_spatial_no_noops
  python tools/monitor_rlds_mask_progress.py --data_mix libero_spatial_no_noops --log logs/rlds_mask_libero_spatial_no_noops/main.log
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = os.environ.get(
    "RLDS_DATA_ROOT",
    "/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/datasets/modified_libero_rlds",
)
DEFAULT_OUT_ROOT = os.environ.get(
    "RLDS_OUT_ROOT",
    "/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/datasets/masked_libero_rlds",
)


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _parse_log_progress(log_path: Path, resume_from: int, total_to_process: int) -> dict:
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    resume_m = None
    for m in re.finditer(r"Resuming from episode (\d+)", text):
        resume_m = m
    if resume_m is None:
        resume_from_log = resume_from
        chunk = text
    else:
        resume_from_log = int(resume_m.group(1))
        chunk = text[resume_m.start() :]

    completed = 0
    for m in re.finditer(r"RLDS mask:[^\n]*\|\s*(\d+)/(\d+)", chunk):
        completed = max(completed, int(m.group(1)))
        total_to_process = int(m.group(2))

    last_tqdm = chunk.rfind("RLDS mask:")
    sam3_tail = chunk[last_tqdm:] if last_tqdm >= 0 else chunk
    sam3_in_ep = len(re.findall(r"Results saved to", sam3_tail))

    global_ep = resume_from_log + completed
    ep_steps_guess = max(sam3_in_ep // 2, 1) if sam3_in_ep else 100
    ep_frac = min(sam3_in_ep / (ep_steps_guess * 2.5), 0.98) if sam3_in_ep else 0.0
    overall = (completed + ep_frac) / max(total_to_process, 1)

    return {
        "resume_from": resume_from_log,
        "total_to_process": total_to_process,
        "completed_in_run": completed,
        "global_episode": global_ep,
        "episode_steps_total": ep_steps_guess,
        "episode_step": int(ep_frac * ep_steps_guess),
        "episode_progress": ep_frac,
        "overall_progress": min(overall, 1.0),
        "language": "",
        "phase": "masking (log estimate)",
        "source": "log",
        "sam3_in_ep": sam3_in_ep,
    }


def _load_progress_file(out_root: Path, data_mix: str, worker_id: int, num_workers: int = 1) -> dict | None:
    if num_workers > 1:
        path = out_root / f".rlds_mask_progress_{data_mix}_w{worker_id}.json"
    else:
        path = out_root / f".rlds_mask_progress_{data_mix}.json"
    data = _read_json(path)
    if not data:
        return None
    data["source"] = "progress_file"
    data["_path"] = str(path)
    updated = data.get("updated_at")
    if updated:
        try:
            ts = datetime.fromisoformat(updated.replace("Z", "+00:00"))
            data["_age_sec"] = (datetime.now(timezone.utc) - ts).total_seconds()
        except ValueError:
            data["_age_sec"] = None
    return data


def _resolve_totals(out_root: Path, data_mix: str, worker_id: int) -> tuple[int, int]:
    resume_path = out_root / f".rlds_resume_{data_mix}.json"
    if worker_id:
        resume_path = out_root / f".rlds_resume_{data_mix}_w{worker_id}.json"
    resume_from = 0
    st = _read_json(resume_path)
    if st and "last_episode" in st:
        resume_from = int(st["last_episode"])
    total_eps = 432
    info = _read_json(Path(DEFAULT_DATA_ROOT) / data_mix / "1.0.0" / "dataset_info.json")
    if info:
        for split in info.get("splits", []):
            if split.get("name") == "train" and "shardLengths" in split:
                total_eps = sum(int(x) for x in split["shardLengths"])
                break
    total_to_process = total_eps - resume_from
    return resume_from, max(total_to_process, 1)


def _format_eta(seconds: float) -> str:
    if seconds <= 0 or seconds > 86400 * 7:
        return "?"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _detect_num_workers(out_root: Path, data_mix: str) -> int:
    n = 1
    while (out_root / f".rlds_mask_progress_{data_mix}_w{n}.json").exists() or (
        out_root / f".rlds_resume_{data_mix}_w{n}.json"
    ).exists():
        n += 1
    return max(n, 1)


def _running_worker_count(data_mix: str) -> int:
    import subprocess

    try:
        out = subprocess.check_output(
            ["pgrep", "-f", f"tools/rlds_mask.py.*{data_mix}"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return len([ln for ln in out.splitlines() if ln.strip()])
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0


def _done_from_tfrecords(out_root: Path, data_mix: str, total_eps: int) -> set[int]:
    sys.path.insert(0, str(_REPO_ROOT / "tools"))
    from rlds_mask_state import done_episodes_from_tfrecords

    return done_episodes_from_tfrecords(out_root, data_mix, total_episodes=total_eps)


def _resume_done_count(out_root: Path, data_mix: str, num_workers: int) -> int:
    sys.path.insert(0, str(_REPO_ROOT / "tools"))
    from rlds_mask_state import count_resume_done_episodes

    return count_resume_done_episodes(out_root, data_mix, num_workers)


def _aggregate_multi_progress(out_root: Path, data_mix: str, num_workers: int, total_eps: int) -> dict:
    done_eps = _done_from_tfrecords(out_root, data_mix, total_eps)
    resume_n = _resume_done_count(out_root, data_mix, num_workers)
    in_progress_frac = 0.0
    active: list[str] = []
    for w in range(num_workers):
        prog = _load_progress_file(out_root, data_mix, w, num_workers)
        if prog and prog.get("_age_sec", 999) < 600:
            ge = int(prog.get("global_episode", 0))
            frac = float(prog.get("episode_progress", 0))
            if ge not in done_eps:
                in_progress_frac += frac
            ep_step = int(prog.get("episode_step") or 0)
            ep_total = int(prog.get("episode_steps_total") or 0)
            active.append(f"w{w}:ep{ge} {ep_step}/{ep_total}")
    total_contrib = len(done_eps) + in_progress_frac
    mismatch = ""
    if resume_n > len(done_eps):
        mismatch = f" resume={resume_n}!"
    return {
        "total_episodes": total_eps,
        "overall_progress": min(total_contrib / max(total_eps, 1), 1.0),
        "overall_n": total_contrib,
        "done_n": len(done_eps),
        "resume_n": resume_n,
        "mismatch": mismatch,
        "phase": "masking",
        "active_workers": "; ".join(active[:4]) + ("..." if len(active) > 4 else ""),
        "source": f"tfrecord+live x{num_workers}",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Live RLDS mask progress bar")
    ap.add_argument("--data_mix", default="libero_spatial_no_noops")
    ap.add_argument("--out_root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--log", default=None, help="Worker log (default: logs/rlds_mask_<data_mix>/main.log)")
    ap.add_argument("--worker_id", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=0, help="0=auto-detect multi-worker")
    ap.add_argument("--refresh", type=float, default=1.0, help="Update interval seconds")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    log_path = Path(args.log) if args.log else _REPO_ROOT / f"logs/rlds_mask_{args.data_mix}" / "main.log"
    num_workers = args.num_workers or _detect_num_workers(out_root, args.data_mix)
    running = _running_worker_count(args.data_mix)
    single_prog = _load_progress_file(out_root, args.data_mix, 0, 1)
    if running == 1 and single_prog and single_prog.get("_age_sec", 999) < 180:
        num_workers = 1
    elif num_workers == 1 and (out_root / f".rlds_resume_{args.data_mix}_w0.json").exists():
        num_workers = max(_detect_num_workers(out_root, args.data_mix), 1)

    total_eps = 432
    info = _read_json(Path(DEFAULT_DATA_ROOT) / args.data_mix / "1.0.0" / "dataset_info.json")
    if info:
        for split in info.get("splits", []):
            if split.get("name") == "train" and "shardLengths" in split:
                total_eps = sum(int(x) for x in split["shardLengths"])
                break

    resume_from, total_to_process = _resolve_totals(out_root, args.data_mix, 0 if num_workers == 1 else args.worker_id)
    if num_workers > 1:
        total_to_process = total_eps

    bar = tqdm(
        total=total_to_process,
        desc=f"RLDS mask x{num_workers}" if num_workers > 1 else "RLDS mask",
        unit="ep",
        dynamic_ncols=True,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {postfix} [{elapsed}<{remaining}]",
    )

    last_overall_n = 0.0
    rate_window: list[tuple[float, float]] = []

    try:
        while True:
            if num_workers > 1:
                prog = _aggregate_multi_progress(out_root, args.data_mix, num_workers, total_eps)
                overall_n = float(prog["overall_n"])
                total = total_eps
                ep_bar = prog.get("active_workers", "")
                phase = prog.get("phase", "?")
            else:
                prog = _load_progress_file(out_root, args.data_mix, args.worker_id, num_workers)
                if prog is None or prog.get("_age_sec", 999) > 120:
                    prog = _parse_log_progress(log_path, resume_from, total_to_process)

                total = int(prog.get("total_to_process") or total_to_process)
                if num_workers == 1:
                    total = int(prog.get("total_episodes") or total_eps)
                ep_step = int(prog.get("episode_step") or 0)
                ep_total = int(prog.get("episode_steps_total") or 0)
                global_ep = int(prog.get("global_episode") or (resume_from + int(prog.get("completed_in_run") or 0)))
                if num_workers == 1:
                    rf = int(prog.get("resume_from") or resume_from)
                    cr = int(prog.get("completed_in_run") or 0)
                    overall_n = min(rf + cr + float(prog.get("episode_progress") or 0), float(total))
                else:
                    overall_frac = float(prog.get("overall_progress") or 0.0)
                    overall_n = min(overall_frac * total, float(total))
                ep_bar = f"ep{global_ep} step {ep_step}/{ep_total}" if ep_total else f"ep{global_ep}"
                if prog.get("sam3_in_ep"):
                    ep_bar += f" sam3~{prog['sam3_in_ep']}"
                phase = prog.get("phase") or "?"

            if bar.total != total:
                bar.total = total
                bar.refresh()

            now = time.time()
            rate_window.append((now, overall_n))
            rate_window = [(t, v) for t, v in rate_window if now - t <= 120]
            if len(rate_window) >= 2 and rate_window[-1][1] > rate_window[0][1]:
                dt = rate_window[-1][0] - rate_window[0][0]
                dv = rate_window[-1][1] - rate_window[0][1]
                ep_per_sec = dv / dt if dt > 0 else 0
                remaining = (total - overall_n) / ep_per_sec if ep_per_sec > 0 else 0
                eta = _format_eta(remaining)
                rate_str = f"{ep_per_sec * 3600:.1f} ep/h"
            else:
                eta = "?"
                rate_str = "? ep/h"

            bar.n = overall_n
            done_tag = ""
            if num_workers > 1 and "done_n" in prog:
                dn = int(prog["done_n"])
                mm = prog.get("mismatch") or ""
                done_tag = f"tf={dn}{mm} | "
            bar.set_postfix_str(
                f"{done_tag}{overall_n:.1f}/{total} ep | {ep_bar} | {phase} | ETA {eta} | {rate_str}",
                refresh=False,
            )
            if overall_n != last_overall_n:
                bar.refresh()
                last_overall_n = overall_n

            tf_done = int(prog.get("done_n") or 0) if num_workers > 1 else overall_n
            if num_workers > 1 and tf_done >= total and running == 0:
                bar.n = total
                bar.set_postfix_str("done (TFRecord verified)", refresh=True)
                break
            if num_workers == 1 and (phase == "done" or overall_n >= total):
                bar.n = total
                bar.set_postfix_str("done", refresh=True)
                break

            time.sleep(max(0.2, args.refresh))
    except KeyboardInterrupt:
        bar.close()
        print("\n(stopped monitor)")
        return 0

    bar.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
