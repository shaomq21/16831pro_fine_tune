#!/usr/bin/env python3
"""Grounding-claim metrics for dual-masked OpenVLA (same-scene).

Reports (π-style + claim-side):
  1) vlm_prefix_l18_lang          — mean_offdiag (instruction collapse)
  2) vlm_prefix_l18               — full patch mean_offdiag (baseline; diluted by black)
  3) vlm_prefix_l18_img_nonblack  — mean over non-black patches only
  4) vlm_prefix_l18_img_rg        — mean over red∪green interest patches only
  5) action_token (last layer)    — natural mean_offdiag across tasks
  6) Intervention A (lang swap):  same img_i, lang_i vs lang_j → action-token cos/L2
  7) Intervention B (img swap):   same lang_i, img_i vs img_j → action-token cos/L2

Claim reading: lang collapsed + ROI-img separates + Δ_act(B) ≳ Δ_act(A)
⇒ actions driven by grounded vision more than instruction wording.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_OFT = _REPO_ROOT / "openvla-oft"
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_OFT))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from analyze_openvla_hidden_by_scene import (  # noqa: E402
    LAYER_INDEX,
    SUITE_CFG,
    grab_masked_init_image_with_state,
    load_openvla,
    study_scene4_group,
    _short_name,
)
from analyze_pi05_libero_hidden_by_scene import (  # noqa: E402
    get_identical_init_groups,
    get_scene_groups,
    summarize_group,
)
from compare_pi05_vision_features import cosine_sim  # noqa: E402

DEFAULT_STORAGE = "/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune"


def patch_grid_masks(
    image_rgb: np.ndarray,
    num_patches: int,
    *,
    black_thr: float = 12.0,
    red_r: float = 80.0,
    green_g: float = 80.0,
) -> dict[str, np.ndarray]:
    """Map dual-masked RGB → per-patch bool masks (length = num_patches).

    OpenVLA fused backbone reports num_patches=256 for 224px → treat as 16×16.
    If fused stacks two towers (512), use first half spatial then tile (rare).
    """
    img = np.asarray(image_rgb, dtype=np.float32)
    h, w = img.shape[:2]
    side = int(round(num_patches**0.5))
    if side * side != num_patches:
        # fused: often 2 * 16*16; use spatial grid of side for first tower, repeat
        side = int(round((num_patches // 2) ** 0.5)) if num_patches % 2 == 0 else side
        spatial = side * side
    else:
        spatial = num_patches

    ph, pw = h / side, w / side
    nonblack = np.zeros(spatial, dtype=bool)
    rg = np.zeros(spatial, dtype=bool)
    for idx in range(spatial):
        r, c = divmod(idx, side)
        y0, y1 = int(r * ph), int((r + 1) * ph)
        x0, x1 = int(c * pw), int((c + 1) * pw)
        patch = img[y0:y1, x0:x1]
        mean = patch.reshape(-1, 3).mean(axis=0)
        if mean.mean() > black_thr:
            nonblack[idx] = True
        # red-ish / green-ish (dual-mask palette)
        if mean[0] > red_r and mean[0] > mean[1] + 20 and mean[0] > mean[2] + 20:
            rg[idx] = True
        if mean[1] > green_g and mean[1] > mean[0] + 20 and mean[1] > mean[2] + 20:
            rg[idx] = True

    def _fit(m: np.ndarray) -> np.ndarray:
        if num_patches == spatial:
            return m
        # repeat spatial mask for fused double-length patch tokens
        reps = num_patches // spatial
        return np.tile(m, reps)[:num_patches]

    return {
        "nonblack": _fit(nonblack),
        "rg": _fit(rg),
        "all": np.ones(num_patches, dtype=bool),
    }


@torch.inference_mode()
def forward_bundle(
    vla,
    processor,
    image_rgb: np.ndarray,
    task_label: str,
    *,
    layer_index: int = LAYER_INDEX,
) -> dict[str, Any]:
    """One multimodal forward → lang / img pools / action-token vectors."""
    from prismatic.vla.constants import ACTION_DIM, IGNORE_INDEX, NUM_ACTIONS_CHUNK

    device = next(vla.parameters()).device
    pil = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).convert("RGB")
    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    inputs = processor(prompt, pil)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    pixel_values = inputs["pixel_values"].to(device=device, dtype=torch.bfloat16)

    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.tensor([[29871]], dtype=input_ids.dtype, device=device)),
            dim=1,
        )
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones((1, 1), dtype=attention_mask.dtype, device=device),
            ),
            dim=1,
        )

    labels = torch.full_like(input_ids, IGNORE_INDEX)
    input_ids, attention_mask = vla._prepare_input_for_action_prediction(
        input_ids, attention_mask
    )
    labels = vla._prepare_labels_for_action_prediction(labels, input_ids)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        input_embeddings = vla.get_input_embeddings()(input_ids)
        all_actions_mask = vla._process_action_masks(labels)
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            1, -1, input_embeddings.shape[2]
        )
        patches = vla._process_vision_features(
            pixel_values, language_embeddings, use_film=False
        )
        input_embeddings = input_embeddings * ~all_actions_mask.unsqueeze(-1)
        mm_emb, mm_attn = vla._build_multimodal_attention(
            input_embeddings, patches, attention_mask
        )
        lm_out = vla.language_model(
            inputs_embeds=mm_emb,
            attention_mask=mm_attn,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    npatch = int(patches.shape[1])
    n_action = ACTION_DIM * NUM_ACTIONS_CHUNK
    li = min(layer_index, len(lm_out.hidden_states) - 1)
    hs = lm_out.hidden_states[li].float()[0]  # [seq, D]
    hsl = lm_out.hidden_states[-1].float()[0]

    img_tok = hs[1 : 1 + npatch]  # [P, D]
    lang_tok = hs[1 + npatch : -(n_action + 1)]
    act_tok = hsl[-(n_action + 1) : -1]  # exclude STOP

    masks = patch_grid_masks(image_rgb, npatch)

    def _masked_mean(tokens: torch.Tensor, m: np.ndarray) -> np.ndarray:
        idx = np.where(m)[0]
        if len(idx) == 0:
            return tokens.mean(0).cpu().numpy().astype(np.float32)
        return tokens[idx].mean(0).cpu().numpy().astype(np.float32)

    return {
        "vlm_prefix_l18": img_tok.mean(0).cpu().numpy().astype(np.float32),
        "vlm_prefix_l18_img_nonblack": _masked_mean(img_tok, masks["nonblack"]),
        "vlm_prefix_l18_img_rg": _masked_mean(img_tok, masks["rg"]),
        "vlm_prefix_l18_lang": lang_tok.mean(0).cpu().numpy().astype(np.float32),
        "action_token": act_tok.reshape(-1).cpu().numpy().astype(np.float32),
        "action_token_mean": act_tok.mean(0).cpu().numpy().astype(np.float32),
        "meta": {
            "layer_index": li,
            "num_patches": npatch,
            "n_nonblack": int(masks["nonblack"].sum()),
            "n_rg": int(masks["rg"].sum()),
            "n_lang": int(lang_tok.shape[0]),
            "n_action": int(act_tok.shape[0]),
        },
    }


def _pair_stats(feats: dict[str, np.ndarray]) -> dict:
    s = summarize_group(feats)
    if s["mean_offdiag_cosine"] is None:
        return s
    # add mean L2 offdiag
    keys = s["keys"]
    l2s = []
    for a, b in combinations(keys, 2):
        l2s.append(float(np.linalg.norm(feats[a] - feats[b])))
    s["mean_offdiag_l2"] = float(np.mean(l2s)) if l2s else None
    return s


def _delta_stats(pairs: list[dict]) -> dict:
    if not pairs:
        return {"n": 0, "mean_cosine": None, "mean_l2": None, "mean_one_minus_cos": None}
    cos = [p["cosine"] for p in pairs]
    l2 = [p["l2"] for p in pairs]
    return {
        "n": len(pairs),
        "mean_cosine": float(np.mean(cos)),
        "mean_l2": float(np.mean(l2)),
        "mean_one_minus_cos": float(np.mean([1.0 - c for c in cos])),
        "min_cosine": float(min(cos)),
        "max_cosine": float(max(cos)),
        "pairs_sample": pairs[:8],
    }


def analyze_group(
    vla,
    processor,
    task_suite,
    tasks: list[dict],
    *,
    layer_index: int,
    mask_alpha: float,
    resolution: int,
) -> dict[str, Any]:
    # Cache images
    images: dict[str, np.ndarray] = {}
    langs: dict[str, str] = {}
    for t in tasks:
        key = t.get("short_name") or _short_name(t["language"])
        langs[key] = t["language"]
        images[key] = grab_masked_init_image_with_state(
            task_suite, int(t["task_id"]), resolution=resolution, mask_alpha=mask_alpha
        )

    keys = sorted(langs.keys())
    # All (img_key, lang_key) forwards for interventions
    cache: dict[tuple[str, str], dict] = {}
    meta_sample = None
    for ik in keys:
        for lk in keys:
            print(f"      forward img={ik[:28]} lang={lk[:28]}")
            bundle = forward_bundle(
                vla, processor, images[ik], langs[lk], layer_index=layer_index
            )
            meta_sample = bundle["meta"]
            cache[(ik, lk)] = bundle

    feat_names = [
        "vlm_prefix_l18",
        "vlm_prefix_l18_img_nonblack",
        "vlm_prefix_l18_img_rg",
        "vlm_prefix_l18_lang",
        "action_token",
        "action_token_mean",
    ]
    # Natural: matching (i,i)
    natural = {ft: {} for ft in feat_names}
    for k in keys:
        b = cache[(k, k)]
        for ft in feat_names:
            natural[ft][k] = b[ft]

    natural_summary = {ft: _pair_stats(natural[ft]) for ft in feat_names}

    # Intervention A: lang swap — (img_i, lang_i) vs (img_i, lang_j)
    pairs_a = []
    for i in keys:
        for j in keys:
            if i == j:
                continue
            a = cache[(i, i)]["action_token"]
            b = cache[(i, j)]["action_token"]
            pairs_a.append(
                {
                    "img": i,
                    "lang_ref": i,
                    "lang_swap": j,
                    "cosine": cosine_sim(a, b),
                    "l2": float(np.linalg.norm(a - b)),
                }
            )

    # Intervention B: img swap — (img_i, lang_i) vs (img_j, lang_i)
    pairs_b = []
    for i in keys:
        for j in keys:
            if i == j:
                continue
            a = cache[(i, i)]["action_token"]
            b = cache[(j, i)]["action_token"]
            pairs_b.append(
                {
                    "lang": i,
                    "img_ref": i,
                    "img_swap": j,
                    "cosine": cosine_sim(a, b),
                    "l2": float(np.linalg.norm(a - b)),
                }
            )

    return {
        "tasks": tasks,
        "keys": keys,
        "extract_meta": meta_sample,
        "natural": natural_summary,
        "intervention_A_lang_swap": _delta_stats(pairs_a),
        "intervention_B_img_swap": _delta_stats(pairs_b),
        "claim_ratios": {
            "lang_collapsed_mean_offdiag": natural_summary["vlm_prefix_l18_lang"].get(
                "mean_offdiag_cosine"
            ),
            "img_full_mean_offdiag": natural_summary["vlm_prefix_l18"].get(
                "mean_offdiag_cosine"
            ),
            "img_nonblack_mean_offdiag": natural_summary[
                "vlm_prefix_l18_img_nonblack"
            ].get("mean_offdiag_cosine"),
            "img_rg_mean_offdiag": natural_summary["vlm_prefix_l18_img_rg"].get(
                "mean_offdiag_cosine"
            ),
            "action_natural_mean_offdiag": natural_summary["action_token"].get(
                "mean_offdiag_cosine"
            ),
            "A_lang_swap_mean_one_minus_cos": _delta_stats(pairs_a)[
                "mean_one_minus_cos"
            ],
            "B_img_swap_mean_one_minus_cos": _delta_stats(pairs_b)[
                "mean_one_minus_cos"
            ],
            "B_minus_A_one_minus_cos": (
                None
                if not pairs_a or not pairs_b
                else float(
                    _delta_stats(pairs_b)["mean_one_minus_cos"]
                    - _delta_stats(pairs_a)["mean_one_minus_cos"]
                )
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suites", type=str, default="goal,object,spatial,study_scene4"
    )
    parser.add_argument(
        "--grouping",
        type=str,
        default="coarse_scene",
        choices=("coarse_scene", "identical_init"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"{DEFAULT_STORAGE}/runs/openvla_grounding_claim",
    )
    parser.add_argument(
        "--summary_copy",
        type=str,
        default=(
            f"{DEFAULT_STORAGE}/runs/all_suites_perturb_matrix/summary/"
            "grounding_claim_openvla.json"
        ),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--layer_index", type=int, default=LAYER_INDEX)
    parser.add_argument("--mask_alpha", type=float, default=0.35)
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    suites = [s.strip() for s in args.suites.split(",") if s.strip()]

    from libero.libero import benchmark

    get_groups = (
        get_identical_init_groups
        if args.grouping == "identical_init"
        else get_scene_groups
    )

    report: dict[str, Any] = {
        "note": (
            "OpenVLA dual-masked grounding claim: lang collapse + ROI-img + "
            "action-token interventions A(lang)/B(img)"
        ),
        "grouping": args.grouping,
        "layer_index": args.layer_index,
        "suites": {},
    }

    for suite_key in suites:
        scfg = SUITE_CFG[suite_key]
        print(f"\n===== {suite_key} =====")
        vla, processor, _ = load_openvla(scfg["ckpt"], scfg["unnorm_key"], args.device)

        if suite_key == "study_scene4":
            scene_groups = study_scene4_group()
            task_suite = benchmark.get_benchmark_dict()["libero_90"]()
        else:
            scene_groups = get_groups(scfg["libero"])
            task_suite = benchmark.get_benchmark_dict()[scfg["libero"]]()

        suite_entry: dict[str, Any] = {"checkpoint": scfg["ckpt"], "scene_groups": {}}
        for scene_name, tasks in sorted(scene_groups.items()):
            if len(tasks) < 2:
                suite_entry["scene_groups"][scene_name] = {
                    "skipped": "fewer than 2 tasks"
                }
                continue
            print(f"  group {scene_name}: {len(tasks)} tasks")
            suite_entry["scene_groups"][scene_name] = analyze_group(
                vla,
                processor,
                task_suite,
                tasks,
                layer_index=args.layer_index,
                mask_alpha=args.mask_alpha,
                resolution=args.resolution,
            )
            cr = suite_entry["scene_groups"][scene_name]["claim_ratios"]
            print(
                f"    lang={cr['lang_collapsed_mean_offdiag']:.4f} "
                f"img_full={cr['img_full_mean_offdiag']:.4f} "
                f"img_nb={cr['img_nonblack_mean_offdiag']:.4f} "
                f"img_rg={cr['img_rg_mean_offdiag']:.4f}"
            )
            print(
                f"    act_nat={cr['action_natural_mean_offdiag']:.4f} "
                f"A(1-cos)={cr['A_lang_swap_mean_one_minus_cos']:.4f} "
                f"B(1-cos)={cr['B_img_swap_mean_one_minus_cos']:.4f} "
                f"B-A={cr['B_minus_A_one_minus_cos']:.4f}"
            )

        report["suites"][suite_key] = suite_entry
        del vla
        torch.cuda.empty_cache()

    out_json = out_root / "summary.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_json}")
    if args.summary_copy:
        Path(args.summary_copy).write_text(json.dumps(report, indent=2))
        print(f"Copied {args.summary_copy}")


if __name__ == "__main__":
    main()
