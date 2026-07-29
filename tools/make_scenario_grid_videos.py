#!/usr/bin/env python3
"""Build per-scenario grid videos of raw|masked side-by-side rollouts.

Scenarios: origin (baseline), lang-l1, lang-l2, bg-0/1/2, color-0/1.
Prefer success=True clips. lang-l1/l2 reuse baseline pool (masked-lang unchanged)
but pick different successful trials when possible.
"""
from __future__ import annotations

import argparse
import math
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

SCEN_FILE = {
    "origin": "baseline",
    "lang_l1": "baseline",  # visual same; different trial pick
    "lang_l2": "baseline",
    "bg_0": "background_0",
    "bg_1": "background_1",
    "bg_2": "background_2",
    "color_0": "color0",
    "color_1": "color1",
}

PAT = re.compile(
    r"success=(True|False)--task=(.+?)--sidebyside_.+?"
    r"-(baseline|background_[012]|color[01])_trial(\d+)\.mp4$"
)


def index_videos(roots: list[Path]) -> dict[str, dict[str, list[tuple]]]:
    """scen_tag -> task -> list of (success, trial, mtime, path)."""
    by: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for root in roots:
        if not root.is_dir():
            continue
        for p in root.glob("*sidebyside*.mp4"):
            m = PAT.search(p.name)
            if not m:
                continue
            suc = m.group(1) == "True"
            task, scen, trial = m.group(2), m.group(3), int(m.group(4))
            by[scen][task].append((suc, trial, p.stat().st_mtime, p))
    return by


def pick_clip(
    cands: list[tuple],
    *,
    prefer_success: bool = True,
    trial_bias: int | None = None,
) -> tuple | None:
    if not cands:
        return None
    items = list(cands)
    # Prefer success, then trial_bias match, then higher trial, then newer
    def key(t):
        suc, trial, mtime, _ = t
        bias = 0 if (trial_bias is None or trial == trial_bias) else 1
        return (
            0 if (prefer_success and suc) else 1,
            0 if suc else 1,
            bias,
            -trial,
            -mtime,
        )

    items.sort(key=key)
    return items[0]


def short_label(task: str, success: bool, max_len: int = 36) -> str:
    t = task.replace("_", " ")
    if len(t) > max_len:
        t = t[: max_len - 1] + "…"
    mark = "OK" if success else "FAIL"
    return f"[{mark}] {t}"


def probe_duration(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nk=1:nw=1",
            str(path),
        ],
        text=True,
    ).strip()
    try:
        return float(out)
    except ValueError:
        return 8.0


def build_grid(
    clips: list[tuple[Path, str, bool]],
    out_path: Path,
    *,
    cols: int,
    tile_w: int,
    tile_h: int,
    title: str,
    fps: int = 10,
    max_dur: float = 10.0,
) -> None:
    """clips: (path, label, success)."""
    n = len(clips)
    rows = math.ceil(n / cols)
    # pad empty black tiles
    while len(clips) < rows * cols:
        clips.append((None, "", False))  # type: ignore

    # Fixed target duration; short clips are padded (tpad), long ones trimmed.
    dur = float(max_dur)

    with tempfile.TemporaryDirectory(prefix="scen_grid_") as td:
        td_path = Path(td)
        tile_paths = []
        for i, (src, label, _suc) in enumerate(clips):
            tp = td_path / f"tile_{i:02d}.mp4"
            if src is None:
                # black placeholder
                subprocess.check_call(
                    [
                        "ffmpeg",
                        "-y",
                        "-f",
                        "lavfi",
                        "-i",
                        f"color=c=black:s={tile_w}x{tile_h}:d={dur}:r={fps}",
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-an",
                        str(tp),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                # scale + pad + drawtext; loop source so short episodes fill `dur`
                lab = (
                    label.replace("\\", "\\\\")
                    .replace(":", "\\:")
                    .replace("'", "\\'")
                    .replace("%", "\\%")
                )
                vf = (
                    f"scale={tile_w}:{tile_h}:force_original_aspect_ratio=decrease,"
                    f"pad={tile_w}:{tile_h}:(ow-iw)/2:(oh-ih)/2:black,"
                    f"drawtext=text='{lab}':fontsize=14:fontcolor=white:"
                    f"borderw=2:bordercolor=black:x=6:y=6,"
                    f"fps={fps},setpts=PTS-STARTPTS"
                )
                subprocess.check_call(
                    [
                        "ffmpeg",
                        "-y",
                        "-stream_loop",
                        "-1",
                        "-i",
                        str(src),
                        "-vf",
                        vf,
                        "-t",
                        str(dur),
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-an",
                        str(tp),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            tile_paths.append(tp)

        # xstack
        inputs = []
        for tp in tile_paths:
            inputs.extend(["-i", str(tp)])
        layout_parts = []
        for i in range(rows * cols):
            r, c = divmod(i, cols) if False else (i // cols, i % cols)
            layout_parts.append(f"{c * tile_w}_{r * tile_h}")
        layout = "|".join(layout_parts)
        # build filter: [0][1]... xstack
        n_in = len(tile_paths)
        labels = "".join(f"[{i}:v]" for i in range(n_in))
        title_esc = (
            title.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        )
        filt = (
            f"{labels}xstack=inputs={n_in}:layout={layout}[g];"
            f"[g]drawtext=text='{title_esc}':fontsize=28:fontcolor=yellow:"
            f"borderw=3:bordercolor=black:x=(w-text_w)/2:y=8[v]"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            *inputs,
            "-filter_complex",
            filt,
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rollout_dirs",
        type=str,
        default=(
            "/home/fan-test/maggie/subopt_proj/third_party/16831pro_fine_tune/"
            "openvla-oft/rollouts/2026_07_26,"
            "/home/fan-test/maggie/subopt_proj/third_party/16831pro_fine_tune/"
            "openvla-oft/rollouts/2026_07_27"
        ),
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=(
            "/home/fan-test/maggie/subopt_proj/third_party/16831pro_fine_tune/"
            "openvla-oft/rollouts/scenario_grids"
        ),
    )
    ap.add_argument("--max_per_video", type=int, default=16)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--tile_w", type=int, default=480)
    ap.add_argument("--tile_h", type=int, default=240)
    ap.add_argument("--max_dur", type=float, default=10.0)
    args = ap.parse_args()

    roots = [Path(p.strip()) for p in args.rollout_dirs.split(",") if p.strip()]
    by = index_videos(roots)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for scen_name, file_tag in SCEN_FILE.items():
        pool = by.get(file_tag, {})
        tasks = sorted(pool.keys())
        if not tasks:
            print(f"[skip] {scen_name}: no clips for {file_tag}")
            continue

        trial_bias = None
        if scen_name == "lang_l1":
            trial_bias = 1
        elif scen_name == "lang_l2":
            trial_bias = 2

        selected = []
        n_ok = 0
        for task in tasks:
            hit = pick_clip(pool[task], trial_bias=trial_bias)
            if hit is None:
                continue
            suc, trial, _, path = hit
            if suc:
                n_ok += 1
            selected.append((path, short_label(task, suc), suc, task, trial))

        # split into pages if needed
        pages = []
        mpv = args.max_per_video
        if len(selected) <= mpv:
            pages = [selected]
        else:
            # prefer 2 roughly equal pages
            mid = (len(selected) + 1) // 2
            # but respect max_per_video
            chunk = min(mpv, max(mid, (len(selected) + 1) // 2))
            for i in range(0, len(selected), chunk):
                pages.append(selected[i : i + chunk])

        note = ""
        if scen_name.startswith("lang_"):
            note = " (masked-lang unchanged; clips from origin pool)"

        for pi, page in enumerate(pages):
            part = "" if len(pages) == 1 else f"_part{pi+1}of{len(pages)}"
            out = out_dir / f"grid_{scen_name}{part}.mp4"
            title = (
                f"{scen_name}{note} | {n_ok}/{len(selected)} success preferred "
                f"| page {pi+1}/{len(pages)}"
            )
            print(f"building {out.name}  n={len(page)} ...")
            build_grid(
                [(p, lab, s) for p, lab, s, _, _ in page],
                out,
                cols=args.cols,
                tile_w=args.tile_w,
                tile_h=args.tile_h,
                title=title,
                max_dur=args.max_dur,
            )
            sz = out.stat().st_size / (1024 * 1024)
            print(f"  -> {out} ({sz:.1f} MB)")
            manifest.append(
                {
                    "scenario": scen_name,
                    "part": pi + 1,
                    "n_parts": len(pages),
                    "n_tiles": len(page),
                    "n_success_in_scenario": n_ok,
                    "n_tasks": len(selected),
                    "path": str(out),
                    "tasks": [
                        {"task": t, "success": s, "trial": tr, "src": str(p)}
                        for p, _, s, t, tr in page
                    ],
                }
            )

    import json

    man_path = out_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {man_path}")
    print(f"total videos: {len(manifest)}")


if __name__ == "__main__":
    main()
