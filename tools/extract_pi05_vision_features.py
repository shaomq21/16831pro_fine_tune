#!/usr/bin/env python3
"""
Extract pi0.5 visual features for STUDY_SCENE4 task comparison.

Feature types:
  vision_tower        - SigLIP output (before projector/VLM), mean over patches
  vlm_prefix_l18      - PaliGemma VLM after 18 transformer layers + final RMSNorm
                        (prefix_output), front-view image patch token positions only
  vlm_prefix_l18_lang - same prefix forward, language token positions only
                        (masked mean over valid instruction tokens)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_OPENVLA_ROOT = _REPO_ROOT / "openvla-oft"
sys.path.insert(0, str(_OPENVLA_ROOT))

from mask_spatial import LIBERO_90_STUDY_SCENE4_TASKS  # noqa: E402

DEFAULT_STORAGE = "/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune"
DEFAULT_LEROT_HOME = f"{DEFAULT_STORAGE}/lerobot_datasets"
DEFAULT_OUTPUT_ROOT = f"{DEFAULT_STORAGE}/runs/pi05_study_scene4_analysis"

FEATURE_SPECS = {
    "vision_tower": {
        "suffix": "vision_features",
        "description": "vision_tower.last_hidden_state (mean over patches)",
    },
    "vlm_prefix_l18": {
        "suffix": "vlm_l18_front_features",
        "description": (
            "paligemma.language_model layer-18 final RMSNorm prefix_output, "
            "front-view image patch token positions only (mean over patches)"
        ),
    },
    "vlm_prefix_l18_lang": {
        "suffix": "vlm_l18_lang_features",
        "description": (
            "paligemma.language_model layer-18 final RMSNorm prefix_output, "
            "language token positions only (masked mean over valid tokens)"
        ),
    },
}


def _short_name(task: str) -> str:
    return (
        task.replace("pick up the ", "")
        .replace(" and place it ", "_to_")
        .replace(" ", "_")
        .replace("the_", "")
    )[:80]


def _to_tensor(v):
    if torch.is_tensor(v):
        return v
    if hasattr(v, "numpy"):
        v = v.numpy()
    return torch.as_tensor(v)


def _frame_to_batch(frame: dict) -> dict:
    batch: dict = {}
    for k, v in frame.items():
        if k == "task":
            task = v.decode() if isinstance(v, bytes) else v
            batch["task"] = [task]
            continue
        t = _to_tensor(v)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        elif k.startswith("observation.images") and t.ndim == 3:
            t = t.unsqueeze(0)
        elif k == "observation.state" and t.ndim == 1:
            t = t.unsqueeze(0)
        elif k == "action" and t.ndim == 1:
            t = t.unsqueeze(0)
        batch[k] = t
    return batch


@torch.inference_mode()
def extract_vision_backbone_features(pi05_model, pixel_values: torch.Tensor) -> torch.Tensor:
    vt = pi05_model.paligemma_with_expert.paligemma.model.vision_tower
    out = vt(pixel_values.to(dtype=torch.float32))
    return out.last_hidden_state


@torch.inference_mode()
def extract_vlm_prefix_l18_front_patches(
    pi05_model,
    images: list[torch.Tensor],
    img_masks: list[torch.Tensor],
    token: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Return front-view patch hidden states [B, num_patches, hidden_dim]."""
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    front_emb = pi05_model.paligemma_with_expert.embed_image(images[0])
    num_front = front_emb.shape[1]

    prefix_embs, prefix_pad_masks, prefix_att_masks = pi05_model.embed_prefix(
        images, img_masks, token, masks
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = pi05_model._prepare_attention_masks_4d(prefix_att_2d_masks)

    lm = pi05_model.paligemma_with_expert.paligemma.language_model
    lm.config._attn_implementation = "eager"  # noqa: SLF001

    (prefix_out, _), _ = pi05_model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=False,
    )
    return prefix_out[:, :num_front, :]


@torch.inference_mode()
def extract_vlm_prefix_l18_language_tokens(
    pi05_model,
    images: list[torch.Tensor],
    img_masks: list[torch.Tensor],
    token: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Return language-token hidden states [B, num_lang_tokens, hidden_dim] after prefix LM."""
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    num_img_tokens = sum(
        pi05_model.paligemma_with_expert.embed_image(img).shape[1] for img in images
    )

    prefix_embs, prefix_pad_masks, prefix_att_masks = pi05_model.embed_prefix(
        images, img_masks, token, masks
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = pi05_model._prepare_attention_masks_4d(prefix_att_2d_masks)

    lm = pi05_model.paligemma_with_expert.paligemma.language_model
    lm.config._attn_implementation = "eager"  # noqa: SLF001

    (prefix_out, _), _ = pi05_model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=False,
    )
    return prefix_out[:, num_img_tokens:, :]


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean pool over sequence dim with bool/float mask [B, L]. Returns [B, D]."""
    m = mask.to(dtype=tokens.dtype).unsqueeze(-1)
    summed = (tokens * m).sum(dim=1)
    denom = m.sum(dim=1).clamp(min=1.0)
    return summed / denom


def load_pi05_policy(checkpoint_dir: str, device: str):
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    policy = PI05Policy.from_pretrained(checkpoint_dir, local_files_only=True)
    policy.eval()
    policy.to(device)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint_dir,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, policy.config, preprocessor


def sample_frames_by_task(lerobot_home: str, repo_id: str, max_per_task: int):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=Path(lerobot_home) / repo_id)
    by_task: dict[str, list[dict]] = {t: [] for t in LIBERO_90_STUDY_SCENE4_TASKS}
    for i in range(len(ds)):
        frame = dict(ds[i])
        task = frame.get("task", "")
        if isinstance(task, bytes):
            task = task.decode()
        if task not in by_task:
            continue
        if len(by_task[task]) >= max_per_task:
            continue
        frame["task"] = task
        by_task[task].append(frame)
        if all(len(v) >= max_per_task for v in by_task.values()):
            break
    return by_task


def _prepare_front_view_tensor(frame: dict, device: str) -> torch.Tensor:
    img = frame.get("observation.images.image", frame.get("image"))
    if img is None:
        raise KeyError("missing front-view image")
    if torch.is_tensor(img):
        pv = img
    else:
        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        pv = torch.from_numpy(np.array(Image.fromarray(arr))).permute(2, 0, 1).float() / 255.0
    pv = pv.unsqueeze(0).to(device)
    if pv.shape[-1] != 224:
        pv = torch.nn.functional.interpolate(pv, size=(224, 224), mode="bilinear", align_corners=False)
    return pv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--lerobot_home", type=str, default=DEFAULT_LEROT_HOME)
    parser.add_argument("--repo_id", type=str, default="local/libero_90_study_scene4")
    parser.add_argument("--output_dir", type=str, default=f"{DEFAULT_OUTPUT_ROOT}/vision_features")
    parser.add_argument("--max_per_task", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--feature_type",
        type=str,
        default="vision_tower",
        choices=sorted(FEATURE_SPECS),
    )
    args = parser.parse_args()

    spec = FEATURE_SPECS[args.feature_type]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    policy, _, preprocessor = load_pi05_policy(args.checkpoint, args.device)
    pi05_model = policy.model
    by_task = sample_frames_by_task(args.lerobot_home, args.repo_id, args.max_per_task)

    results = {}
    for task in LIBERO_90_STUDY_SCENE4_TASKS:
        frames = by_task.get(task, [])
        if not frames:
            print(f"WARNING: no frames for task: {task}")
            continue
        feats = []
        for frame in frames:
            if args.feature_type == "vision_tower":
                pv = _prepare_front_view_tensor(frame, args.device)
                patch_feats = extract_vision_backbone_features(pi05_model, pv)
                pooled = patch_feats.mean(dim=1).cpu().float().numpy()[0]
            else:
                batch = preprocessor(_frame_to_batch(frame))
                images, img_masks = policy._preprocess_images(batch)
                from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

                token = batch[OBS_LANGUAGE_TOKENS]
                masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
                if args.feature_type == "vlm_prefix_l18":
                    patch_feats = extract_vlm_prefix_l18_front_patches(
                        pi05_model, images, img_masks, token, masks
                    )
                    pooled = patch_feats.mean(dim=1).cpu().float().numpy()[0]
                elif args.feature_type == "vlm_prefix_l18_lang":
                    lang_feats = extract_vlm_prefix_l18_language_tokens(
                        pi05_model, images, img_masks, token, masks
                    )
                    pooled = _masked_mean(lang_feats, masks).cpu().float().numpy()[0]
                else:
                    raise ValueError(f"unsupported feature_type: {args.feature_type}")
            feats.append(pooled)

        arr = np.stack(feats, axis=0)
        key = _short_name(task)
        np.save(out_dir / f"{key}_{spec['suffix']}.npy", arr)
        results[task] = {"key": key, "n_samples": len(feats), "shape": list(arr.shape)}
        print(f"  {key}: {arr.shape}")

    with open(out_dir / "meta.json", "w") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "feature_type": args.feature_type,
                "feature_suffix": spec["suffix"],
                "tasks": results,
                "feature_description": spec["description"],
            },
            f,
            indent=2,
        )
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
