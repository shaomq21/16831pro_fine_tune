#!/usr/bin/env python3
"""Regenerate RESULTS.md with suite×visual tables + optional hidden similarity."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SUM = Path(
    "/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/runs/"
    "all_suites_perturb_matrix/summary"
)
VISUAL_ORDER = [
    "baseline",
    "background_0",
    "background_1",
    "background_2",
    "color_0",
    "color_1",
]
VISUAL_LABEL = {
    "baseline": "origin",
    "background_0": "bg-0",
    "background_1": "bg-1",
    "background_2": "bg-2",
    "color_0": "color-0",
    "color_1": "color-1",
}


def pct(ok: int, n: int) -> str:
    if n <= 0:
        return "—"
    return f"{ok}/{n} ({100.0 * ok / n:.1f}%)"


def load_lang() -> dict:
    cands = sorted(SUM.glob("langshift_mean5_*.json"))
    if not cands:
        return {}
    return json.loads(cands[-1].read_text())


def load_hidden() -> dict | None:
    p = SUM / "hidden_similarity_openvla.json"
    if p.exists():
        return json.loads(p.read_text())
    # also check analysis run dir
    alt = Path(
        "/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/runs/"
        "openvla_hidden_by_scene/summary.json"
    )
    if alt.exists():
        return json.loads(alt.read_text())
    return None


def main() -> None:
    prog = json.loads((SUM / "progress_report.json").read_text())
    lang = load_lang()
    hidden = load_hidden()

    # suite × visual
    scens: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    for t in prog["tasks"]:
        for sc, v in t["scenarios"].items():
            scens[t["suite"]][sc][0] += v["ok"]
            scens[t["suite"]][sc][1] += v["n"]

    lines: list[str] = []
    lines.append("# Dual-masked OpenVLA eval results (consolidated)")
    lines.append("")
    lines.append(f"- Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("- Model: ours dual-masked LoRA (front-view only, mask-from-env, α=0.35)")
    lines.append("- Suites: goal / object / spatial / study_scene4")
    lines.append("- Videos: `openvla-oft/rollouts/2026_07_21/` (deduped to 1 per task×scenario)")
    lines.append("")

    # ---- 1. Visual conditions (THE table user asked for) ----
    lines.append("## 1. Suite × visual condition (origin / bg / color)")
    lines.append("")
    lines.append(
        "Each cell = successes / trials over **all tasks** in that suite "
        "(TRIALS=3 per task×condition)."
    )
    lines.append("")
    header = "| Suite | " + " | ".join(VISUAL_LABEL[s] for s in VISUAL_ORDER) + " | pooled |"
    sep = "|-------|" + "|".join(["--------"] * len(VISUAL_ORDER)) + "|--------|"
    lines.append(header)
    lines.append(sep)
    for suite in ("goal", "object", "spatial", "study_scene4"):
        cells = []
        pok = pn = 0
        for sc in VISUAL_ORDER:
            ok, n = scens[suite][sc]
            cells.append(pct(ok, n))
            pok += ok
            pn += n
        lines.append(f"| {suite} | " + " | ".join(cells) + f" | {pct(pok, pn)} |")
    lines.append("")

    lines.append("### Same table as rates only")
    lines.append("")
    lines.append(
        "| Suite | " + " | ".join(VISUAL_LABEL[s] for s in VISUAL_ORDER) + " | pooled |"
    )
    lines.append(sep)
    for suite in ("goal", "object", "spatial", "study_scene4"):
        cells = []
        pok = pn = 0
        for sc in VISUAL_ORDER:
            ok, n = scens[suite][sc]
            cells.append(f"{100.0 * ok / n:.1f}%" if n else "—")
            pok += ok
            pn += n
        lines.append(
            f"| {suite} | "
            + " | ".join(cells)
            + f" | {100.0 * pok / pn:.1f}% |"
        )
    lines.append("")

    # ---- 2. reported_success ----
    lines.append("## 2. reported_success = 1 (mostly working, ≥~50%)")
    lines.append("")
    for t in prog["tasks"]:
        if t.get("reported_success") == 1 or t["rate"] >= 0.5:
            lines.append(
                f"- **[{t['suite']}]** {t['task']}: "
                f"{t['ok']}/{t['n']} ({100 * t['rate']:.0f}%)"
            )
    lines.append("")

    # ---- 3. Low-SR (keep short pointer; full rescue from previous) ----
    lines.append("## 3. Low-SR tasks (<15%) — rescue status")
    lines.append("")
    lines.append("| Suite | Task | Pool SR | Rescue |")
    lines.append("|-------|------|---------|--------|")
    rescue_note = {
        "goal": "goal-low / push-rescue (see §8)",
        "object": "object-rescue +20k GPU2",
        "spatial": "spatial-rescue +20k GPU4",
        "study_scene4": "ongoing study_scene4 train →297k",
    }
    for t in prog["tasks"]:
        if t["rate"] < 0.15:
            lines.append(
                f"| {t['suite']} | {t['task']} | {100 * t['rate']:.0f}% | "
                f"{rescue_note.get(t['suite'], '')} |"
            )
    lines.append("")

    # ---- 4. lang-shift ----
    lines.append("## 4. Language-shift (ours mean-of-5)")
    lines.append("")
    lines.append(
        "Method: **1 new baseline trial / task** + **4 random draws** from existing "
        "BG/color trials → mean. l1/l2 use different RNG seeds."
    )
    lines.append(
        "Note: masked-language input unchanged for ours; new trial ≈ another origin sample."
    )
    lines.append("")
    suite_l1 = lang.get("suite_l1") or {}
    suite_l2 = lang.get("suite_l2") or {}
    if suite_l1 or suite_l2:
        lines.append("| Suite | lang-l1 | lang-l2 |")
        lines.append("|-------|---------|---------|")
        for suite in ("goal", "object", "spatial", "study_scene4"):
            l1 = suite_l1.get(suite)
            l2 = suite_l2.get(suite)
            if l1 is None and l2 is None:
                continue
            def _fmt(x):
                if x is None:
                    return "—"
                if isinstance(x, float) and x <= 1.0:
                    return f"{100 * x:.0f}%"
                return str(x)

            lines.append(f"| {suite} | {_fmt(l1)} | {_fmt(l2)} |")
        lines.append("")

    # ---- 5. Per-task visual (compact table, not bullets) ----
    lines.append("## 5. Per-task × visual condition")
    lines.append("")
    lines.append(
        "| Suite | Task | "
        + " | ".join(VISUAL_LABEL[s] for s in VISUAL_ORDER)
        + " | pooled |"
    )
    lines.append(
        "|-------|------|"
        + "|".join(["------"] * len(VISUAL_ORDER))
        + "|--------|"
    )
    for t in prog["tasks"]:
        cells = []
        for sc in VISUAL_ORDER:
            v = t["scenarios"].get(sc, {"ok": 0, "n": 0})
            cells.append(f"{v['ok']}/{v['n']}")
        short = t["task"] if len(t["task"]) <= 70 else t["task"][:67] + "..."
        lines.append(
            f"| {t['suite']} | {short} | "
            + " | ".join(cells)
            + f" | {t['ok']}/{t['n']} |"
        )
    lines.append("")

    # ---- 6. early goal (keep if file exists) ----
    early = SUM / "early_goal_matrix.json"
    if early.exists():
        lines.append("## 6. Early goal matrix (push/put only, TRIALS=5, pre all-suites)")
        lines.append("")
        lines.append("| Task | Scenario | SR |")
        lines.append("|------|----------|----|")
        for row in json.loads(early.read_text()):
            lines.append(
                f"| {row['task']} | {row['scenario']} | "
                f"{row['ok']}/{row['n']} = {100 * row['ok'] / max(row['n'], 1):.0f}% |"
            )
        lines.append("")
    else:
        lines.append("## 6. Early goal matrix")
        lines.append("")
        lines.append("(see previous RESULTS snapshot / logs if needed)")
        lines.append("")

    # ---- 7. Hidden similarity ----
    lines.append("## 7. Hidden similarity (π-style: `vlm_prefix_l18` img + lang)")
    lines.append("")
    lines.append(
        "Metric same as π: within same-scene / identical-init task groups, "
        "**mean off-diagonal cosine** of pooled feature vectors."
    )
    lines.append(
        "- `vlm_prefix_l18`: VLM hidden @ layer-18 (OpenVLA Llama), **mean over front image patches**"
    )
    lines.append(
        "- `vlm_prefix_l18_lang`: same forward, **masked mean over language tokens**"
    )
    lines.append("")
    if not hidden:
        lines.append(
            "**Status: extraction pending — run `tools/analyze_openvla_hidden_by_scene.py`.**"
        )
        lines.append("")
    else:
        lines.append(
            f"- Checkpoint set: `{hidden.get('note', 'per-suite dual_masked adapters')}`"
        )
        lines.append(f"- Grouping: `{hidden.get('grouping', 'identical_init')}`")
        lines.append(f"- Frames: `{hidden.get('frame_source', 'libero_init')}`")
        lines.append("")
        pooled = hidden.get("pooled") or {}
        lines.append("### Pooled (all same-scene pairs across suites)")
        lines.append("")
        lines.append("| Feature | mean_offdiag_cosine | n_pairs |")
        lines.append("|---------|---------------------|---------|")
        for ft, label in (
            ("vlm_prefix_l18", "vlm_prefix_l18 (img)"),
            ("vlm_prefix_l18_lang", "vlm_prefix_l18_lang"),
        ):
            e = pooled.get(ft) or {}
            vals = e.get("all_pair_cosines") or []
            cos = [
                x["cosine"] if isinstance(x, dict) else float(x) for x in vals
            ]
            mean = e.get("mean")
            if mean is None and cos:
                mean = sum(cos) / len(cos)
            mean_s = f"{mean:.4f}" if mean is not None else "—"
            lines.append(f"| {label} | {mean_s} | {len(cos)} |")
        lines.append("")

        lines.append("### Per suite / scene group")
        lines.append("")
        lines.append(
            "| Suite | Group | n_tasks | vlm_img mean_offdiag | vlm_lang mean_offdiag |"
        )
        lines.append(
            "|-------|-------|---------|----------------------|-----------------------|"
        )
        for suite, sentry in (hidden.get("suites") or {}).items():
            for gname, gentry in (sentry.get("scene_groups") or {}).items():
                fts = gentry.get("feature_types") or {}
                img = (fts.get("vlm_prefix_l18") or {}).get("mean_offdiag_cosine")
                lng = (fts.get("vlm_prefix_l18_lang") or {}).get(
                    "mean_offdiag_cosine"
                )
                n_tasks = (fts.get("vlm_prefix_l18") or {}).get("n_tasks") or len(
                    gentry.get("tasks") or []
                )
                if img is None and lng is None:
                    continue
                img_s = f"{img:.4f}" if img is not None else "—"
                lng_s = f"{lng:.4f}" if lng is not None else "—"
                lines.append(
                    f"| {suite} | {gname} | {n_tasks} | {img_s} | {lng_s} |"
                )
        lines.append("")
        artifact = hidden.get("artifact") or str(
            SUM / "hidden_similarity_openvla.json"
        )
        lines.append(f"Full JSON: `{artifact}`")
        lines.append("")

    # ---- 8 timeline ----
    lines.append("## 8. Training / rescue timeline (live)")
    lines.append("")
    lines.append("| Job | GPU | Range | Filter | Status |")
    lines.append("|-----|-----|-------|--------|--------|")
    lines.append("| goal push-rescue | 5,6 | →671.5k | push + put bowl | running / check |")
    lines.append("| goal low-rescue | 5,6 | →691.5k | all low goal tasks | queued after push |")
    lines.append("| object-rescue | 2 | →246.5k | 6 low object tasks | running / check |")
    lines.append("| spatial-rescue | 4 | →638k | 9 low spatial tasks | running / check |")
    lines.append("| study_scene4 | 3 | →297k | all 4 book tasks | running |")
    lines.append("")

    lines.append("## 9. Baselines (OFT / π)")
    lines.append("")
    lines.append("Not run yet in this matrix (disk / queue).")
    lines.append("")

    out = SUM / "RESULTS.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
