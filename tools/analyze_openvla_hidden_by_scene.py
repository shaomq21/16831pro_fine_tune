#!/usr/bin/env python3
"""Analyze dual-masked OpenVLA hidden similarity (π-style metrics).

Same feature names / pooling as π0.5 tooling:
  - vlm_prefix_l18      : LM hidden @ layer 18, mean over front image patch tokens
  - vlm_prefix_l18_lang : same forward, masked mean over language tokens

Grouping: identical_init (exact BDDL :init) within each suite — same visual layout,
different language goals. study_scene4 = the 4 book tasks as one group.

Frames: LIBERO init state #0, agentview, dual-mask black-bg (matches train/eval).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
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

from compare_pi05_vision_features import cosine_sim  # noqa: E402
from analyze_pi05_libero_hidden_by_scene import (  # noqa: E402
    get_identical_init_groups,
    get_scene_groups,
    summarize_group,
)

DEFAULT_STORAGE = "/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune"
FEATURE_TYPES = ("vlm_prefix_l18", "vlm_prefix_l18_lang")
LAYER_INDEX = 18  # match π naming; OpenVLA Llama-7B has 32 layers

SUITE_CFG = {
    "goal": {
        "libero": "libero_goal",
        "ckpt": (
            f"{DEFAULT_STORAGE}/runs/openvla_adapters/"
            "openvla-7b+dual_masked_goal+b4+lr-0.0005+lora-r32+dropout-0.0"
            "+lora-attn-only--suite_goal_oft_lr"
        ),
        "unnorm_key": "simu_libero_goal_no_noops",
    },
    "object": {
        "libero": "libero_object",
        "ckpt": (
            f"{DEFAULT_STORAGE}/runs/openvla_adapters/"
            "openvla-7b+dual_masked_object+b4+lr-0.0005+lora-r32+dropout-0.0"
            "+lora-attn-only--suite_object_oft_lr"
        ),
        "unnorm_key": "simu_libero_object_no_noops",
    },
    "spatial": {
        "libero": "libero_spatial",
        "ckpt": (
            f"{DEFAULT_STORAGE}/runs/openvla_adapters/"
            "openvla-7b+dual_masked_spatial+b4+lr-0.0005+lora-r32+dropout-0.0"
            "+lora-attn-only--suite_spatial_oft_lr"
        ),
        "unnorm_key": "simu_libero_spatial_no_noops",
    },
    "study_scene4": {
        "libero": "libero_90",
        "ckpt": (
            f"{DEFAULT_STORAGE}/runs/openvla_adapters/"
            "openvla-7b+dual_masked_study_scene4+b2+lr-0.0005+lora-r32+dropout-0.0"
            "+lora-attn-only--suite_study_scene4_oft_lr"
        ),
        "unnorm_key": "simu_libero_90_study_scene4_no_noops",
        "lang_allowlist": True,
    },
}


def _short_name(task: str) -> str:
    return (
        task.replace("pick up the ", "")
        .replace(" and place it ", "_to_")
        .replace(" ", "_")
        .replace("the_", "")
    )[:80]


@dataclass
class LoadCfg:
    pretrained_checkpoint: str
    unnorm_key: str
    base_vla_path: str | None = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    num_images_in_input: int = 1
    use_film: bool = False
    use_proprio: bool = False
    center_crop: bool = False


def load_openvla(ckpt: str, unnorm_key: str, device: str):
    from experiments.robot.openvla_utils import get_processor, get_vla

    cfg = LoadCfg(pretrained_checkpoint=ckpt, unnorm_key=unnorm_key)
    vla = get_vla(cfg)
    processor = get_processor(cfg)
    vla.eval()
    # Prefer explicit device if model isn't already mapped
    try:
        vla = vla.to(device)
    except Exception:
        pass
    return vla, processor, cfg


def num_patches(vla) -> int:
    return int(
        vla.vision_backbone.get_num_patches()
        * vla.vision_backbone.get_num_images_in_input()
    )


@torch.inference_mode()
def extract_openvla_vlm_features(
    vla,
    processor,
    image_rgb: np.ndarray,
    task_label: str,
    *,
    layer_index: int = LAYER_INDEX,
) -> dict[str, np.ndarray]:
    """Return {vlm_prefix_l18, vlm_prefix_l18_lang} vectors [D].

    Mirrors OpenVLA-OFT ``predict_action`` multimodal prep (fake action labels),
    then reads LM ``hidden_states[layer_index]``. Patch tokens sit first in the
    multimodal sequence (after optional BOS), matching finetune.py layout:
    ``hidden[:, :num_patches]`` = vision, rest = language (+ action placeholders).
    """
    from prismatic.vla.constants import IGNORE_INDEX

    device = next(vla.parameters()).device
    pil = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).convert("RGB")
    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    inputs = processor(prompt, pil)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    pixel_values = inputs["pixel_values"].to(device=device, dtype=torch.bfloat16)

    # Same special empty-token + action-token prep as predict_action
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (
                input_ids,
                torch.tensor([[29871]], dtype=input_ids.dtype, device=device),
            ),
            dim=1,
        )
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=device),
            ),
            dim=1,
        )

    labels = input_ids.clone()
    labels[:] = IGNORE_INDEX
    input_ids, attention_mask = vla._prepare_input_for_action_prediction(
        input_ids, attention_mask
    )
    labels = vla._prepare_labels_for_action_prediction(labels, input_ids)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        input_embeddings = vla.get_input_embeddings()(input_ids)
        all_actions_mask = vla._process_action_masks(labels)
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )
        projected_patch_embeddings = vla._process_vision_features(
            pixel_values, language_embeddings, use_film=False
        )
        # Zero action-token embeddings (same as forward without noisy actions)
        input_embeddings = input_embeddings * ~all_actions_mask.unsqueeze(-1)
        multimodal_embeddings, multimodal_attention_mask = vla._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )
        lm_out = vla.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

    n_hs = len(lm_out.hidden_states)
    li = min(layer_index, n_hs - 1)
    hs = lm_out.hidden_states[li].float()  # [B, seq, D]

    # Multimodal layout from _build_multimodal_attention: [BOS?] + patches + language...
    # finetune uses num_patches starting at index 0 of LM hidden — verify via shape.
    npatch = int(projected_patch_embeddings.shape[1])
    # After _build_multimodal_attention, patches are inserted after first token:
    #   cat([emb[:, :1], patches, emb[:, 1:]])
    # so patch slice is [1 : 1+npatch] if BOS present. Match training code which
    # uses hidden[:, :num_patches] on the *labels-aligned* view where patches
    # replace the leading region. Inspect build:
    # Prefer training convention: first npatch tokens after any leading special.
    # Empirically OpenVLA-OFT training uses num_patches as vision length at the
    # start of the multimodal LM sequence *including* how labels are built:
    # labels = cat([labels[:, :1], patch_labels, labels[:, 1:]])
    # So position 0 is BOS/label0, positions 1:1+P are patches.
    img_tokens = hs[:, 1 : 1 + npatch, :]
    # Language = non-action tokens after patches. Exclude trailing action placeholders.
    lang_start = 1 + npatch
    # multimodal attention mask marks valid tokens
    mm_mask = multimodal_attention_mask
    if mm_mask.dtype != torch.bool:
        mm_mask = mm_mask.bool()
    lang_tokens = hs[:, lang_start:, :]
    lang_mask = mm_mask[:, lang_start:]
    # Also drop action-token positions if we can map them; approximate: keep
    # tokens before the last ACTION_DIM*NUM_ACTIONS_CHUNK positions when present.
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

    n_action = ACTION_DIM * NUM_ACTIONS_CHUNK
    if lang_tokens.shape[1] > n_action + 1:
        lang_tokens = lang_tokens[:, : -(n_action + 1), :]
        lang_mask = lang_mask[:, : -(n_action + 1)]

    img_vec = img_tokens.mean(dim=1)[0].cpu().numpy()
    m = lang_mask.to(dtype=lang_tokens.dtype).unsqueeze(-1)
    lang_vec = ((lang_tokens * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0))[0]
    lang_vec = lang_vec.cpu().numpy()
    return {
        "vlm_prefix_l18": img_vec.astype(np.float32),
        "vlm_prefix_l18_lang": lang_vec.astype(np.float32),
        "_meta": {
            "layer_index_used": li,
            "n_hidden_states": n_hs,
            "num_patches": npatch,
            "seq_len": int(hs.shape[1]),
            "lang_len": int(lang_tokens.shape[1]),
            "img_slice": f"1:{1 + npatch}",
        },
    }


def grab_masked_init_image(task, resolution: int = 224, mask_alpha: float = 0.35) -> np.ndarray:
    """Reset LIBERO task to init state 0; return dual-masked agentview RGB."""
    from experiments.robot.libero.libero_utils import (
        get_libero_env,
        get_libero_image,
        mask_image_from_libero_seg,
    )

    env, _ = get_libero_env(
        task, "openvla", resolution=resolution, use_segmentation_env=True
    )
    try:
        # Prefer official init states when available
        init_states = None
        try:
            # caller may pass suite+id; here we just reset then set if possible
            pass
        except Exception:
            pass
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        # wait a few no-op steps for objects to settle (common in eval)
        for _ in range(10):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        rgb = get_libero_image(obs)
        seg = obs.get("agentview_segmentation_instance")
        if seg is None:
            return rgb
        return mask_image_from_libero_seg(rgb, seg, env, alpha=mask_alpha)
    finally:
        env.close()


def grab_masked_init_image_with_state(
    task_suite,
    task_id: int,
    resolution: int = 256,
    mask_alpha: float = 0.35,
) -> np.ndarray:
    """LIBERO init0 dual-masked agentview (same path as eval mask_rgb_from_obs).

    Important: SegmentationRenderEnv only fills ``segmentation_id_mapping`` in
    ``reset()``. ``set_init_state`` alone leaves mapping empty → all-black masks.
    """
    from experiments.robot.libero.libero_utils import get_libero_env
    from libero_sim_mask import mask_rgb_from_obs

    task = task_suite.get_task(task_id)
    env, _ = get_libero_env(
        task, "openvla", resolution=resolution, use_segmentation_env=True
    )
    try:
        env.reset()  # populate segmentation_id_mapping
        init_states = task_suite.get_task_init_states(task_id)
        obs = env.set_init_state(init_states[0])
        for _ in range(10):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        masked, red_name, green_name = mask_rgb_from_obs(
            obs["agentview_image"],
            obs["agentview_segmentation_instance"],
            env,
            alpha=mask_alpha,
            sim=getattr(env, "sim", None),
        )
        masked = np.asarray(masked, dtype=np.uint8)
        if masked.mean() < 1.0:
            raise RuntimeError(
                f"dual-mask still black for task_id={task_id} "
                f"red={red_name} green={green_name} "
                f"mapping_n={len(getattr(env, 'segmentation_id_mapping', {}) or {})}"
            )
        return masked
    finally:
        env.close()


def study_scene4_group() -> dict[str, list[dict]]:
    from libero.libero import benchmark
    from mask_spatial import LIBERO_90_STUDY_SCENE4_TASKS

    allow = {t.lower() for t in LIBERO_90_STUDY_SCENE4_TASKS}
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    tasks = []
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        if task.language.lower() not in allow:
            continue
        tasks.append(
            {
                "task_id": task_id,
                "language": task.language,
                "bddl_file": task.bddl_file,
                "short_name": _short_name(task.language),
            }
        )
    return {"study_scene4_books": tasks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suites",
        type=str,
        default="goal,object,spatial,study_scene4",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"{DEFAULT_STORAGE}/runs/openvla_hidden_by_scene",
    )
    parser.add_argument(
        "--summary_copy",
        type=str,
        default=(
            f"{DEFAULT_STORAGE}/runs/all_suites_perturb_matrix/summary/"
            "hidden_similarity_openvla.json"
        ),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--grouping",
        type=str,
        default="coarse_scene",
        choices=("identical_init", "coarse_scene"),
        help="Match π analyze_pi05 default: coarse_scene (same BDDL scene folder)",
    )
    parser.add_argument("--layer_index", type=int, default=LAYER_INDEX)
    parser.add_argument("--mask_alpha", type=float, default=0.35)
    parser.add_argument("--resolution", type=int, default=224)
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    suites = [s.strip() for s in args.suites.split(",") if s.strip()]

    report: dict[str, Any] = {
        "note": "dual_masked OpenVLA per-suite adapters; π-style vlm img+lang",
        "grouping": args.grouping,
        "frame_source": "libero_init0_dual_masked",
        "layer_index": args.layer_index,
        "feature_types": list(FEATURE_TYPES),
        "suites": {},
        "pooled": {ft: {"all_pair_cosines": [], "mean": None} for ft in FEATURE_TYPES},
        "artifact": str(out_root / "summary.json"),
    }

    from libero.libero import benchmark

    get_groups = (
        get_identical_init_groups
        if args.grouping == "identical_init"
        else get_scene_groups
    )

    for suite_key in suites:
        scfg = SUITE_CFG[suite_key]
        print(f"\n===== {suite_key} =====")
        print(f"Loading {scfg['ckpt']}")
        vla, processor, _ = load_openvla(scfg["ckpt"], scfg["unnorm_key"], args.device)

        if suite_key == "study_scene4":
            scene_groups = study_scene4_group()
            task_suite = benchmark.get_benchmark_dict()["libero_90"]()
        else:
            scene_groups = get_groups(scfg["libero"])
            task_suite = benchmark.get_benchmark_dict()[scfg["libero"]]()

        suite_entry: dict[str, Any] = {"checkpoint": scfg["ckpt"], "scene_groups": {}}

        for scene_name, tasks in sorted(scene_groups.items()):
            print(f"  group {scene_name}: {len(tasks)} tasks")
            group_entry: dict[str, Any] = {
                "scene": scene_name,
                "tasks": tasks,
                "feature_types": {},
            }
            if len(tasks) < 2:
                group_entry["skipped"] = "fewer than 2 tasks"
                suite_entry["scene_groups"][scene_name] = group_entry
                continue

            feats_by_type: dict[str, dict[str, np.ndarray]] = {
                ft: {} for ft in FEATURE_TYPES
            }
            meta_sample = None
            for tinfo in tasks:
                tid = int(tinfo["task_id"])
                lang = tinfo["language"]
                key = tinfo.get("short_name") or _short_name(lang)
                print(f"    frame+feat: {key}")
                img = grab_masked_init_image_with_state(
                    task_suite,
                    tid,
                    resolution=args.resolution,
                    mask_alpha=args.mask_alpha,
                )
                vecs = extract_openvla_vlm_features(
                    vla,
                    processor,
                    img,
                    lang,
                    layer_index=args.layer_index,
                )
                meta_sample = vecs.pop("_meta", None)
                for ft in FEATURE_TYPES:
                    feats_by_type[ft][key] = vecs[ft]

            if meta_sample:
                group_entry["extract_meta"] = meta_sample

            for ft in FEATURE_TYPES:
                summary = summarize_group(feats_by_type[ft])
                group_entry["feature_types"][ft] = summary
                if summary["mean_offdiag_cosine"] is not None:
                    for pair in summary["pairs"]:
                        report["pooled"][ft]["all_pair_cosines"].append(
                            {
                                "suite": suite_key,
                                "group": scene_name,
                                "a": pair["a"],
                                "b": pair["b"],
                                "cosine": pair["cosine"],
                            }
                        )
                    print(
                        f"    {ft}: mean_offdiag={summary['mean_offdiag_cosine']:.4f} "
                        f"range=[{summary['min_pair_cosine']:.4f}, "
                        f"{summary['max_pair_cosine']:.4f}]"
                    )

            suite_entry["scene_groups"][scene_name] = group_entry

        report["suites"][suite_key] = suite_entry
        # free GPU before next suite
        del vla
        torch.cuda.empty_cache()

    for ft in FEATURE_TYPES:
        vals = [x["cosine"] for x in report["pooled"][ft]["all_pair_cosines"]]
        report["pooled"][ft]["mean"] = float(np.mean(vals)) if vals else None
        report["pooled"][ft]["n_pairs"] = len(vals)
        print(
            f"POOLED {ft}: mean={report['pooled'][ft]['mean']} n={len(vals)}"
        )

    out_json = out_root / "summary.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_json}")

    if args.summary_copy:
        cp = Path(args.summary_copy)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(report, indent=2))
        print(f"Copied to {cp}")


if __name__ == "__main__":
    main()
