"""
Quick GRPO-lite-style online RL for continuous OpenVLA-OFT under color perturbation.

Inspired by SimpleVLA-RL / grpo_lite_rl:
  - binary success reward
  - batch-mean baseline advantage
  - Gaussian action exploration (continuous analogue of temperature sampling)
  - train a fresh vision LoRA + action head; never merge into the SFT checkpoint

Default task: libero_spatial id=2 (black bowl from table center) — origin SR high,
color SR ~33% in post-rescue matrix → room to improve with sparse success signal.
"""

import json
import logging
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import draccus
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from libero.libero import benchmark
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image

sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    mask_image_from_libero_seg,
    quat2axisangle,
)
from experiments.robot.libero.run_libero_color_perturb_eval import (
    _apply_bowl_perturbation,
    _apply_color_perturbation,
)
from experiments.robot.openvla_utils import (
    DEVICE,
    get_action_head,
    get_processor,
    get_proprio_projector,
    normalize_proprio,
    prepare_images_for_vla,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    ACTION_TOKEN_BEGIN_IDX,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    STOP_INDEX,
)
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType
from prismatic.vla.datasets.datasets import language_mask_processor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rl_lora_color_quick")

TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 150,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclass
class RLConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    base_vla_path: Optional[str] = None
    rl_lora_path: Optional[str] = None  # if set in eval mode, load this adapter (unmerged)

    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    num_images_in_input: int = 1
    use_proprio: bool = True
    center_crop: bool = False
    num_open_loop_steps: int = 8
    lora_rank: int = 8
    unnorm_key: str = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    task_suite_name: str = "libero_spatial"
    task_id: int = 2
    num_steps_wait: int = 10
    env_img_res: int = 256

    use_mask_for_policy: bool = True
    use_mask_from_env: bool = True
    mask_alpha: float = 0.35
    perturb_mode: str = "colors"  # colors | bowl
    color_variants: str = "0,1"

    # RL / GRPO-lite: same (init,color) group, relative advantage across samples
    mode: str = "train"  # train | eval
    num_iters: int = 8
    num_groups_per_iter: int = 2
    group_size: int = 4
    eval_trials_per_variant: int = 5
    action_noise_std: float = 0.06
    lr_lora: float = 2e-5
    lr_action_head: float = 4e-5
    max_update_chunks: int = 10
    bc_steps_per_iter: int = 2
    max_init_pool: int = 12
    save_dir: str = ""
    local_log_dir: str = "./experiments/logs/rl_lora_color_quick"
    seed: int = 7
    skip_baseline_eval: bool = False
    # legacy flags (kept for shell compatibility)
    rollouts_per_iter: int = 8
    greedy_frac: float = 0.0
    min_success_to_update: int = 1
    success_buffer_size: int = 64


def _parse_variants(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def _normalize_actions_np(actions: np.ndarray, norm_stats: Dict[str, Any]) -> np.ndarray:
    """Unnormalized -> normalized [-1,1] using dataset stats (inverse of _unnormalize_actions)."""
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = np.array(norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool)))
        high, low = np.array(norm_stats["max"]), np.array(norm_stats["min"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = np.array(norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool)))
        high, low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
    else:
        raise ValueError("Unsupported normalization type")
    out = actions.copy()
    out[..., mask] = 2 * (actions[..., mask] - low[mask]) / (high[mask] - low[mask] + 1e-8) - 1
    return np.clip(out, -1.0, 1.0)


def _unnormalize_actions_np(normalized: np.ndarray, norm_stats: Dict[str, Any]) -> np.ndarray:
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = np.array(norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool)))
        high, low = np.array(norm_stats["max"]), np.array(norm_stats["min"])
    else:
        mask = np.array(norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool)))
        high, low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
    out = normalized.copy()
    out[..., mask] = 0.5 * (normalized[..., mask] + 1) * (high[mask] - low[mask] + 1e-8) + low[mask]
    return out


def _discover_attn_lora_targets(vision_backbone) -> str:
    suffixes = set()
    for name, mod in vision_backbone.named_modules():
        if isinstance(mod, torch.nn.Linear) and ".attn." in name:
            suffixes.add(name.split(".")[-1])
    if not suffixes:
        raise ValueError("No attn Linear modules found for LoRA")
    return r"^.*\.attn\.(" + "|".join(re.escape(s) for s in sorted(suffixes)) + r")$"


def attach_rl_lora(vla, rank: int = 8) -> None:
    """Attach a fresh trainable vision LoRA on top of (already-merged) SFT weights. Do not merge."""
    bb = vla.vision_backbone
    if isinstance(bb, PeftModel):
        logger.info("Vision backbone already PeftModel — keeping unmerged for RL")
        for n, p in bb.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
        return
    target = _discover_attn_lora_targets(bb)
    cfg = LoraConfig(
        r=rank,
        lora_alpha=min(rank, 16),
        lora_dropout=0.0,
        target_modules=target,
        init_lora_weights="gaussian",
    )
    vla.vision_backbone = get_peft_model(bb, cfg)
    try:
        vla.vision_backbone.enable_input_require_grads()
    except Exception:
        pass
    vla.vision_backbone.print_trainable_parameters()


def load_rl_lora_unmerged(vla, adapter_dir: str) -> None:
    bb = vla.vision_backbone
    if isinstance(bb, PeftModel):
        bb.load_adapter(adapter_dir, adapter_name="rl")
        bb.set_adapter("rl")
    else:
        vla.vision_backbone = PeftModel.from_pretrained(bb, adapter_dir, is_trainable=False)
    logger.info("Loaded RL LoRA (unmerged) from %s", adapter_dir)


def freeze_non_lora(vla, action_head, proprio_projector) -> None:
    for p in vla.parameters():
        p.requires_grad = False
    for n, p in vla.vision_backbone.named_parameters():
        if "lora_" in n:
            p.requires_grad = True
    if action_head is not None:
        for p in action_head.parameters():
            p.requires_grad = True
    if proprio_projector is not None:
        for p in proprio_projector.parameters():
            p.requires_grad = True


def forward_normalized_actions(
    vla,
    processor,
    obs: Dict[str, Any],
    task_label: str,
    cfg: RLConfig,
    action_head,
    proprio_projector,
) -> torch.Tensor:
    """Normalized action chunk prediction. Shape (1, NUM_ACTIONS_CHUNK, ACTION_DIM)."""
    all_images = [obs["full_image"]]
    if cfg.num_images_in_input > 1:
        all_images.extend([obs[k] for k in obs.keys() if "wrist" in k])
    all_images = prepare_images_for_vla(all_images, cfg)
    primary = all_images.pop(0)
    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    inputs = processor(prompt, primary)
    pixel_values = inputs["pixel_values"].to(DEVICE, dtype=torch.bfloat16)
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.tensor([[29871]], device=DEVICE, dtype=input_ids.dtype)),
            dim=1,
        )
        attention_mask = torch.cat(
            (attention_mask, torch.ones((1, 1), device=DEVICE, dtype=attention_mask.dtype)),
            dim=1,
        )

    labels = input_ids.clone()
    labels[:] = IGNORE_INDEX
    NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1
    input_ids, attention_mask = vla._prepare_input_for_action_prediction(input_ids, attention_mask)
    labels = vla._prepare_labels_for_action_prediction(labels, input_ids)

    input_embeddings = vla.get_input_embeddings()(input_ids)
    all_actions_mask = vla._process_action_masks(labels)
    language_embeddings = input_embeddings[~all_actions_mask].reshape(
        input_embeddings.shape[0], -1, input_embeddings.shape[2]
    )
    projected_patch_embeddings = vla._process_vision_features(pixel_values, language_embeddings, False)

    proprio = obs.get("state")
    if cfg.use_proprio and proprio_projector is not None and proprio is not None:
        proprio_t = torch.as_tensor(proprio, device=projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype)
        projected_patch_embeddings = vla._process_proprio_features(
            projected_patch_embeddings, proprio_t, proprio_projector
        )

    NUM_PATCHES = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()
    if cfg.use_proprio and proprio_projector is not None:
        NUM_PATCHES += 1

    all_actions_mask_u = all_actions_mask.unsqueeze(-1)
    input_embeddings = input_embeddings * ~all_actions_mask_u
    multimodal_embeddings, multimodal_attention_mask = vla._build_multimodal_attention(
        input_embeddings, projected_patch_embeddings, attention_mask
    )
    lm_out = vla.language_model(
        input_ids=None,
        attention_mask=multimodal_attention_mask,
        inputs_embeds=multimodal_embeddings,
        output_hidden_states=True,
        return_dict=True,
    )
    last_hidden = lm_out.hidden_states[-1]
    actions_hidden = last_hidden[
        :,
        NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
        :,
    ]
    pred = action_head.predict_action(actions_hidden.to(torch.bfloat16))
    return pred  # (1, chunk, dim)


def prepare_obs(obs, resize_size, cfg, color_variant: int, env, task_description_raw: str):
    img = get_libero_image(obs)
    wrist = get_libero_wrist_image(obs)
    if cfg.perturb_mode == "colors":
        img = _apply_color_perturbation(img, color_variant)
        wrist = _apply_color_perturbation(wrist, color_variant)
    elif cfg.perturb_mode == "bowl":
        img = _apply_bowl_perturbation(img, color_variant)
        wrist = _apply_bowl_perturbation(wrist, color_variant)

    policy_img = img
    if cfg.use_mask_for_policy and cfg.use_mask_from_env:
        img_np = np.asarray(img)
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        seg_key = "agentview_segmentation_instance"
        if seg_key in obs:
            try:
                policy_img = mask_image_from_libero_seg(img_np, obs[seg_key], env, alpha=cfg.mask_alpha)
            except Exception as e:
                logger.warning("mask_from_env failed: %s", e)
                policy_img = np.zeros_like(img_np)
        else:
            policy_img = np.zeros_like(img_np)

    observation = {
        "full_image": resize_image_for_policy(policy_img, resize_size),
        "wrist_image": resize_image_for_policy(wrist, resize_size),
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }
    return observation, img


def process_action(action, model_family):
    action = normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = invert_gripper_action(action)
    return action


@torch.no_grad()
def predict_chunk_numpy(
    vla, processor, obs, task_label, cfg, action_head, proprio_projector, noise_std: float = 0.0
) -> Tuple[List[np.ndarray], np.ndarray, Dict[str, Any]]:
    """Return (unnorm action list, normalized noisy chunk, stored obs snapshot)."""
    # normalize proprio in-place copy
    obs = dict(obs)
    if cfg.use_proprio:
        proprio_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
        obs["state"] = normalize_proprio(obs["state"], proprio_stats)

    pred = forward_normalized_actions(
        vla, processor, obs, task_label, cfg, action_head, proprio_projector
    )
    mu = pred.float().cpu().numpy()[0]  # (chunk, dim)
    if noise_std > 0:
        noise = np.random.randn(*mu.shape).astype(np.float32) * noise_std
        norm_exec = np.clip(mu + noise, -1.0, 1.0)
    else:
        norm_exec = mu
    action_stats = vla.norm_stats[cfg.unnorm_key]["action"]
    unnorm = _unnormalize_actions_np(norm_exec, action_stats)
    actions = [unnorm[i] for i in range(len(unnorm))]
    # store lightweight obs for later update
    snap = {
        "full_image": np.asarray(obs["full_image"]).copy(),
        "wrist_image": np.asarray(obs["wrist_image"]).copy() if "wrist_image" in obs else None,
        "state": np.asarray(obs["state"]).copy(),
        "task_label": task_label,
        "target_norm": norm_exec.astype(np.float32),
        "noise_std": float(noise_std),
    }
    return actions, norm_exec, snap


def run_episode(
    cfg: RLConfig,
    env,
    raw_task_description: str,
    vla,
    resize_size,
    processor,
    action_head,
    proprio_projector,
    initial_state,
    color_variant: int,
    explore: bool,
) -> Tuple[bool, List[Dict[str, Any]]]:
    env.reset()
    obs = env.set_init_state(initial_state) if initial_state is not None else env.get_observation()
    action_queue: deque = deque(maxlen=cfg.num_open_loop_steps)
    transitions: List[Dict[str, Any]] = []
    t = 0
    max_steps = TASK_MAX_STEPS.get(cfg.task_suite_name, 150)
    task_description = (
        language_mask_processor(raw_task_description) if cfg.use_mask_for_policy else raw_task_description
    )
    success = False
    noise_std = cfg.action_noise_std if explore else 0.0

    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            observation, _ = prepare_obs(obs, resize_size, cfg, color_variant, env, raw_task_description)

            if len(action_queue) == 0:
                actions, _, snap = predict_chunk_numpy(
                    vla, processor, observation, task_description, cfg, action_head, proprio_projector, noise_std
                )
                transitions.append(snap)
                action_queue.extend(actions)

            action = process_action(action_queue.popleft(), cfg.model_family)
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1
    except Exception as e:
        logger.warning("Episode error: %s", e)

    return success, transitions


def save_rl_checkpoint(save_dir: str, vla, action_head, proprio_projector, meta: Dict[str, Any]) -> None:
    os.makedirs(save_dir, exist_ok=True)
    adapter_dir = os.path.join(save_dir, "lora_adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    bb = vla.vision_backbone
    if isinstance(bb, PeftModel):
        bb.save_pretrained(adapter_dir)
        logger.info("Saved unmerged RL LoRA -> %s", adapter_dir)
    else:
        logger.warning("vision_backbone is not PeftModel; nothing to save for LoRA")
    if action_head is not None:
        torch.save(action_head.state_dict(), os.path.join(save_dir, "action_head--rl_checkpoint.pt"))
    if proprio_projector is not None:
        torch.save(proprio_projector.state_dict(), os.path.join(save_dir, "proprio_projector--rl_checkpoint.pt"))
    with open(os.path.join(save_dir, "rl_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def resolve_unnorm_key(cfg: RLConfig, model) -> None:
    preset = str(cfg.unnorm_key or "").strip()
    if preset and preset in model.norm_stats:
        cfg.unnorm_key = preset
        return
    unnorm_key = cfg.task_suite_name
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    for candidate in (
        f"simu_{unnorm_key}",
        f"simu_{cfg.task_suite_name}_no_noops",
        f"sam_{unnorm_key}",
        f"sam_{cfg.task_suite_name}_no_noops",
    ):
        if candidate in model.norm_stats:
            unnorm_key = candidate
            break
    assert unnorm_key in model.norm_stats, f"unnorm_key not in {list(model.norm_stats)[:20]}"
    cfg.unnorm_key = unnorm_key


def eval_sr(
    cfg: RLConfig,
    task_suite,
    vla,
    resize_size,
    processor,
    action_head,
    proprio_projector,
    variants: List[int],
    trials: int,
    tag: str,
) -> Dict[str, Any]:
    task = task_suite.get_task(cfg.task_id)
    initial_states = task_suite.get_task_init_states(cfg.task_id)
    env, task_description = get_libero_env(
        task,
        cfg.model_family,
        resolution=cfg.env_img_res,
        use_segmentation_env=(cfg.use_mask_for_policy and cfg.use_mask_from_env),
    )
    results = {"tag": tag, "per_variant": {}, "successes": 0, "episodes": 0}
    for cv in variants:
        ok = 0
        for ep in range(trials):
            init = initial_states[ep % len(initial_states)]
            success, _ = run_episode(
                cfg, env, task_description, vla, resize_size, processor, action_head, proprio_projector,
                init, cv, explore=False,
            )
            ok += int(success)
            results["episodes"] += 1
            results["successes"] += int(success)
            logger.info("[%s] color%d trial%d success=%s", tag, cv, ep, success)
        results["per_variant"][str(cv)] = {"successes": ok, "trials": trials, "sr": ok / max(trials, 1)}
    results["sr"] = results["successes"] / max(results["episodes"], 1)
    logger.info("[%s] overall SR=%.3f (%d/%d)", tag, results["sr"], results["successes"], results["episodes"])
    return results


def rl_update_groups(
    cfg: RLConfig,
    vla,
    processor,
    action_head,
    proprio_projector,
    optimizer,
    groups: List[List[Tuple[float, List[Dict[str, Any]]]]],
) -> Dict[str, float]:
    """
    True GRPO-lite over groups: each group = G rollouts of the SAME (init, color).
    advantage = r - mean_group(r). Only reinforce positive-advantage samples.
    Skip groups with zero reward variance (all success or all fail).
    """
    pos: List[Tuple[float, Dict[str, Any]]] = []
    group_srs = []
    n_used_groups = 0
    for group in groups:
        rewards = np.array([r for r, _ in group], dtype=np.float32)
        group_srs.append(float(rewards.mean()))
        if rewards.std() < 1e-6:
            continue  # no relative signal
        baseline = float(rewards.mean())
        n_used_groups += 1
        for r, traj in group:
            adv = float(r - baseline)
            if adv <= 1e-6:
                continue
            for tr in traj:
                pos.append((adv, tr))

    if not pos:
        return {
            "loss": 0.0,
            "group_sr": float(np.mean(group_srs) if group_srs else 0.0),
            "n_pos": 0,
            "skipped": 1.0,
            "n_groups_used": float(n_used_groups),
        }

    vla.train()
    action_head.train()
    if proprio_projector is not None:
        proprio_projector.train()

    total_loss = 0.0
    n_steps = 0
    for _ in range(max(1, cfg.bc_steps_per_iter)):
        sample = pos
        if len(sample) > cfg.max_update_chunks:
            idx = np.random.choice(len(sample), size=cfg.max_update_chunks, replace=False)
            sample = [sample[i] for i in idx]
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        k = 0
        with torch.enable_grad():
            for w, tr in sample:
                obs = {"full_image": tr["full_image"], "state": tr["state"]}
                if tr.get("wrist_image") is not None and cfg.num_images_in_input > 1:
                    obs["wrist_image"] = tr["wrist_image"]
                pred = forward_normalized_actions(
                    vla, processor, obs, tr["task_label"], cfg, action_head, proprio_projector
                )
                target = torch.as_tensor(tr["target_norm"], device=pred.device, dtype=pred.dtype).unsqueeze(0)
                loss = F.l1_loss(pred, target) * float(w)
                loss.backward()
                step_loss += float(loss.detach().item())
                k += 1
        torch.nn.utils.clip_grad_norm_(
            [p for p in list(vla.parameters()) + list(action_head.parameters()) if p.requires_grad],
            1.0,
        )
        optimizer.step()
        total_loss += step_loss / max(k, 1)
        n_steps += 1

    vla.eval()
    action_head.eval()
    if proprio_projector is not None:
        proprio_projector.eval()
    return {
        "loss": total_loss / max(n_steps, 1),
        "group_sr": float(np.mean(group_srs) if group_srs else 0.0),
        "n_pos": float(min(len(pos), cfg.max_update_chunks) * n_steps),
        "skipped": 0.0,
        "n_groups_used": float(n_used_groups),
    }


@draccus.wrap()
def main(cfg: RLConfig) -> None:
    assert cfg.pretrained_checkpoint, "pretrained_checkpoint required"
    set_seed_everywhere(cfg.seed)
    variants = _parse_variants(cfg.color_variants)
    if not cfg.save_dir:
        cfg.save_dir = str(
            Path("/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/runs/rl_lora_color_quick")
            / f"{cfg.task_suite_name}_task{cfg.task_id}_{DATE_TIME}"
        )
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    log_path = os.path.join(cfg.local_log_dir, f"rl_{DATE_TIME}.log")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    logger.info("cfg=%s", cfg)

    # Build a minimal namespace cfg object expected by get_model / get_action_head
    class _C:
        pass

    mcfg = _C()
    for k, v in cfg.__dict__.items():
        setattr(mcfg, k, v)
    mcfg.num_diffusion_steps_train = 50
    mcfg.num_diffusion_steps_inference = 50
    mcfg.use_film = False

    logger.info("Loading model from %s", cfg.pretrained_checkpoint)
    vla = get_model(mcfg)
    proprio_projector = get_proprio_projector(mcfg, vla.llm_dim, proprio_dim=8) if cfg.use_proprio else None
    action_head = get_action_head(mcfg, vla.llm_dim)
    processor = get_processor(mcfg)
    resolve_unnorm_key(cfg, vla)
    mcfg.unnorm_key = cfg.unnorm_key
    resize_size = get_image_resize_size(mcfg)

    if cfg.mode == "eval":
        assert cfg.rl_lora_path, "eval mode needs --rl_lora_path"
        load_rl_lora_unmerged(vla, cfg.rl_lora_path)
        ah_path = os.path.join(os.path.dirname(cfg.rl_lora_path.rstrip("/")), "action_head--rl_checkpoint.pt")
        if os.path.dirname(cfg.rl_lora_path.rstrip("/")).endswith("lora_adapter"):
            ah_path = os.path.join(Path(cfg.rl_lora_path).parent, "action_head--rl_checkpoint.pt")
        parent = str(Path(cfg.rl_lora_path).parent) if Path(cfg.rl_lora_path).name == "lora_adapter" else cfg.rl_lora_path
        ah_path = os.path.join(parent, "action_head--rl_checkpoint.pt")
        pp_path = os.path.join(parent, "proprio_projector--rl_checkpoint.pt")
        if os.path.isfile(ah_path):
            action_head.load_state_dict(torch.load(ah_path, map_location=DEVICE, weights_only=False))
            logger.info("Loaded RL action_head from %s", ah_path)
        if proprio_projector is not None and os.path.isfile(pp_path):
            proprio_projector.load_state_dict(torch.load(pp_path, map_location=DEVICE, weights_only=False))
        vla.eval()
        action_head.eval()
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[cfg.task_suite_name]()
        res = eval_sr(
            cfg, task_suite, vla, resize_size, processor, action_head, proprio_projector,
            variants, cfg.eval_trials_per_variant, tag="rl_eval",
        )
        out = os.path.join(cfg.save_dir, "eval_results.json")
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        logger.info("Wrote %s", out)
        return

    # TRAIN: attach fresh RL LoRA (do not merge)
    attach_rl_lora(vla, rank=cfg.lora_rank)
    freeze_non_lora(vla, action_head, proprio_projector)
    try:
        vla.language_model.gradient_checkpointing_enable()
    except Exception as e:
        logger.warning("gradient_checkpointing_enable failed: %s", e)

    trainable = [p for p in vla.parameters() if p.requires_grad]
    trainable += [p for p in action_head.parameters() if p.requires_grad]
    if proprio_projector is not None:
        trainable += [p for p in proprio_projector.parameters() if p.requires_grad]
    # param groups
    lora_params = [p for n, p in vla.named_parameters() if p.requires_grad and "lora_" in n]
    other_params = [p for p in trainable if id(p) not in {id(x) for x in lora_params}]
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": cfg.lr_lora},
            {"params": other_params, "lr": cfg.lr_action_head},
        ],
        weight_decay=0.0,
    )
    logger.info("Trainable tensors: lora=%d other=%d", len(lora_params), len(other_params))

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    task = task_suite.get_task(cfg.task_id)
    initial_states = task_suite.get_task_init_states(cfg.task_id)
    env, task_description = get_libero_env(
        task,
        cfg.model_family,
        resolution=cfg.env_img_res,
        use_segmentation_env=(cfg.use_mask_for_policy and cfg.use_mask_from_env),
    )
    logger.info("Task %d: %s", cfg.task_id, task_description)

    history: Dict[str, Any] = {"baseline_eval": None, "iters": [], "final_eval": None}

    vla.eval()
    action_head.eval()
    if not cfg.skip_baseline_eval:
        history["baseline_eval"] = eval_sr(
            cfg, task_suite, vla, resize_size, processor, action_head, proprio_projector,
            variants, cfg.eval_trials_per_variant, tag="baseline",
        )
        with open(os.path.join(cfg.save_dir, "baseline_eval.json"), "w") as f:
            json.dump(history["baseline_eval"], f, indent=2)

    t0 = time.time()
    init_pool = list(initial_states[: max(1, min(cfg.max_init_pool, len(initial_states)))])
    for it in range(cfg.num_iters):
        groups = []
        succ_all = 0
        n_all = 0
        for g in range(cfg.num_groups_per_iter):
            init = init_pool[(it * cfg.num_groups_per_iter + g) % len(init_pool)]
            cv = variants[g % len(variants)]
            group = []
            for s in range(cfg.group_size):
                # Always explore within a GRPO group so samples differ
                success, traj = run_episode(
                    cfg, env, task_description, vla, resize_size, processor, action_head, proprio_projector,
                    init, cv, explore=True,
                )
                group.append((float(success), traj))
                succ_all += int(success)
                n_all += 1
                logger.info(
                    "iter %d group %d sample %d color%d success=%s chunks=%d",
                    it, g, s, cv, success, len(traj),
                )
            groups.append(group)
        metrics = rl_update_groups(
            cfg, vla, processor, action_head, proprio_projector, optimizer, groups
        )
        row = {
            "iter": it,
            "train_sr": succ_all / max(n_all, 1),
            **metrics,
            "elapsed_s": time.time() - t0,
        }
        history["iters"].append(row)
        logger.info("iter %d metrics=%s", it, row)
        with open(os.path.join(cfg.save_dir, "train_history.json"), "w") as f:
            json.dump(history, f, indent=2)
        if (it + 1) % 2 == 0 or it + 1 == cfg.num_iters:
            save_rl_checkpoint(
                cfg.save_dir,
                vla,
                action_head,
                proprio_projector,
                meta={"iter": it, "cfg": {k: str(v) if isinstance(v, Path) else v for k, v in cfg.__dict__.items()}},
            )

    history["final_eval"] = eval_sr(
        cfg, task_suite, vla, resize_size, processor, action_head, proprio_projector,
        variants, cfg.eval_trials_per_variant, tag="after_rl",
    )
    save_rl_checkpoint(
        cfg.save_dir,
        vla,
        action_head,
        proprio_projector,
        meta={"iter": cfg.num_iters - 1, "final": True},
    )
    with open(os.path.join(cfg.save_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    summary = {
        "task": task_description,
        "task_id": cfg.task_id,
        "suite": cfg.task_suite_name,
        "baseline_sr": None if not history["baseline_eval"] else history["baseline_eval"]["sr"],
        "final_sr": history["final_eval"]["sr"],
        "save_dir": cfg.save_dir,
        "merged": False,
    }
    with open(os.path.join(cfg.save_dir, "SUMMARY.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("DONE summary=%s", summary)


if __name__ == "__main__":
    main()
