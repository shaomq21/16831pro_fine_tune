"""
run_libero_color_perturb_eval.py

Color perturbation eval: two tasks only.
Each task: color detection unchanged, but each detected color maps to 2 output colors (2 variants).
- "push the plate" → perturb_colors: white/red detection; variant 0: white→yellow, red→light_blue; variant 1: white→cyan, red→light_green.
- "put the bowl" → perturb_bowl: dark_grey detection; variant 0: →red; variant 1: →blue.
Runs 2 variants × num_trials_per_task episodes per task. Video naming: push/put + model_label + "-color0" or "-color1".
Run on openvla_oft_goal and openvla_7b via run_all_color_perturb.sh.
"""
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

from PIL import Image, ImageDraw, ImageFont
import json
import logging
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import draccus
import numpy as np
import tqdm

try:
    import pandas as pd
    import openpyxl  # noqa: F401
    _HAS_EXCEL = True
except ImportError:
    _HAS_EXCEL = False
from libero.libero import benchmark

import wandb
from prismatic.vla.datasets.datasets import (
    mask_image_via_other_env,
    language_mask_processor,
)

sys.path.append("../..")
import imageio
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    mask_image_from_libero_seg,
    quat2axisangle,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import os as _os
import torch

_EVAL_FORCE_CPU = _os.environ.get("EVAL_FORCE_CPU", "0") == "1"
_old_torch_load = torch.load


def _torch_load_safe(*args, **kwargs):
    if _EVAL_FORCE_CPU:
        kwargs.setdefault("map_location", "cpu")
        kwargs.setdefault("weights_only", False)
    return _old_torch_load(*args, **kwargs)


torch.load = _torch_load_safe


class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Only put task for color perturb eval (set to both tasks to run push + put)
COLOR_TASKS = [
    "put the bowl on the plate",
]


def _task_short_name(task_description: str) -> str:
    """Return short name for video: 'push' or 'put'."""
    d = (task_description or "").strip().lower()
    if "push" in d:
        return "push"
    if "put" in d:
        return "put"
    return "task"


def _task_description_for_policy(raw_description: str, cfg) -> str:
    """For OFT/7B only: push plate->flat shaped object on the right; put bowl->gray bowl."""
    if getattr(cfg, "use_mask_for_policy", True):
        return raw_description  # current model: no modification
    d = (raw_description or "").strip().lower()
    out = raw_description
    if "push" in d and "plate" in d:
        out = out.replace("the plate", "the flat shaped object on the right").replace(
            "The plate", "The flat shaped object on the right"
        )
    if "put" in d and "bowl" in d:
        out = out.replace("the plate", "the flat shaped object on the right")
    return out


TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 150,
    "libero_10": 520,
    "libero_90": 400,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    base_vla_path: Optional[str] = None

    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 1
    use_proprio: bool = True

    center_crop: bool = True
    num_open_loop_steps: int = 8
    lora_rank: int = 32
    unnorm_key: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    task_suite_name: str = TaskSuite.LIBERO_GOAL
    num_steps_wait: int = 10
    num_trials_per_task: int = 3
    initial_states_path: str = "DEFAULT"
    env_img_res: int = 256

    run_id_note: Optional[str] = None
    model_label: str = "openvla_oft_goal"
    local_log_dir: str = "./experiments/info"

    loadinfo: bool = False
    perturb_colors: bool = False
    perturb_bowl: bool = False
    use_mask_for_policy: bool = False   # True only for your trained model; openvla_oft_goal and 7b use raw image
    use_mask_from_env: bool = True      # When use_mask_for_policy: get mask from LIBERO seg vs Grounded-SAM

    use_wandb: bool = False
    wandb_entity: str = "maggiesh-carnegie-mellon-university"
    wandb_project: str = "validation"

    seed: int = 7


def _resolve_base_vla_path(cfg: GenerateConfig) -> None:
    base = getattr(cfg, "base_vla_path", None)
    if not base or not isinstance(base, str) or not base.strip():
        return
    base = base.strip()
    if (base.startswith("/") or base.startswith(".")) and not os.path.isdir(base):
        openvla_root = Path(__file__).resolve().parent.parent.parent.parent
        fallback = openvla_root / "checkpoints" / "openvla-7b"
        if fallback.is_dir():
            cfg.base_vla_path = str(fallback)
            logger.info("base_vla_path %s not found; using %s", base, cfg.base_vla_path)


def validate_config(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"
    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting center_crop==True when model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    assert cfg.task_suite_name in [s.value for s in TaskSuite], f"Invalid task_suite_name: {cfg.task_suite_name}"
    if cfg.loadinfo and not _HAS_EXCEL:
        raise ImportError("loadinfo=True requires pandas and openpyxl. pip install pandas openpyxl")


def initialize_model(cfg: GenerateConfig):
    model = get_model(cfg)
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        unnorm_key = cfg.task_suite_name
        if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
            unnorm_key = f"{unnorm_key}_no_noops"
        assert unnorm_key in model.norm_stats, f"unnorm_key {unnorm_key} not found!"
        cfg.unnorm_key = unnorm_key
    return model, action_head, proprio_projector, noisy_action_projector, processor


def setup_logging(cfg: GenerateConfig):
    run_id = f"COLOR-PERTURB-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info("Logging to %s", local_log_filepath)
    if cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)
    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    initial_states = task_suite.get_task_init_states(task_id)
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    log_message("Using default initial states", log_file)
    return initial_states, None


def prepare_observation(obs, resize_size):
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }
    return observation, img, wrist_img


def process_action(action, model_family):
    action = normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = invert_gripper_action(action)
    return action


def _apply_bowl_perturbation(
    img: Union[np.ndarray, Image.Image], variant: int = 0
) -> Union[np.ndarray, Image.Image]:
    """Bowl: dark grey → variant 0=red, variant 1=blue. Color detection unchanged."""
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return img
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mean_rgb = (r + g + b) / 3.0
    spread = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    dark_grey_mask = (spread < 40) & (mean_rgb >= 50) & (mean_rgb <= 140)
    if variant == 0:
        arr[dark_grey_mask, 0], arr[dark_grey_mask, 1], arr[dark_grey_mask, 2] = 255, 40, 40
    else:
        arr[dark_grey_mask, 0], arr[dark_grey_mask, 1], arr[dark_grey_mask, 2] = 40, 40, 255
    out = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(out) if isinstance(img, Image.Image) else out


def _apply_color_perturbation(
    img: Union[np.ndarray, Image.Image], variant: int = 0
) -> Union[np.ndarray, Image.Image]:
    """Plate: white→yellow/cyan only. Variant 0: white→yellow; variant 1: white→cyan. Red region disabled."""
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return img
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    white_mask = (r > 200) & (g > 200) & (b > 200)
    if variant == 0:
        arr[white_mask, 0], arr[white_mask, 1], arr[white_mask, 2] = 255, 255, 0
    else:
        arr[white_mask, 0], arr[white_mask, 1], arr[white_mask, 2] = 0, 255, 255
    out = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(out) if isinstance(img, Image.Image) else out


def _draw_step_on_image(img: Union[np.ndarray, Image.Image], step: int) -> Image.Image:
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
    except OSError:
        font = ImageFont.load_default()
    text = f"Step {step}"
    x, y = 10, 10
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        draw.text((x + dx, y + dy), text, fill="black", font=font)
    draw.text((x, y), text, fill="white", font=font)
    return img


def _save_loadinfo_excel(rows: List[Dict[str, Any]], output_path: str, log_file=None) -> None:
    if not rows:
        return
    if not _HAS_EXCEL:
        log_message("loadinfo: pandas/openpyxl not installed", log_file)
        return
    try:
        df = pd.DataFrame(rows)
        df.to_excel(output_path, index=False, engine="openpyxl")
        log_message(f"loadinfo: Saved Excel to {output_path}", log_file)
    except Exception as e:
        log_message(f"loadinfo: Failed to save Excel: {e}", log_file)


def _save_sidebyside_video(
    raw_images: List,
    masked_images: List,
    idx: int,
    success: bool,
    task_description: str,
    suffix: str,
    log_file=None,
    fps: int = 30,
    model_label: Optional[str] = None,
) -> str:
    """Save video with raw (left) and masked (right) side by side."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    extra = f"--{suffix}" if suffix else ""
    model_tag = model_label if model_label else "color_perturb"
    mp4_path = f"{rollout_dir}/{DATE_TIME}--{model_tag}--episode={idx}--success={success}--task={processed}{extra}.mp4"
    writer = imageio.get_writer(mp4_path, fps=fps)
    for raw, mask in zip(raw_images, masked_images):
        raw_np = np.asarray(raw)
        mask_np = np.asarray(mask)
        if raw_np.shape != mask_np.shape:
            mask_np = np.asarray(
                Image.fromarray(mask_np).resize((raw_np.shape[1], raw_np.shape[0]), Image.Resampling.LANCZOS)
            )
        sidebyside = np.concatenate([raw_np, mask_np], axis=1)
        writer.append_data(sidebyside)
    writer.close()
    logger.info("Saved side-by-side video: %s", mp4_path)
    if log_file:
        log_file.write(f"Saved side-by-side video: {mp4_path}\n")
        log_file.flush()
    return mp4_path


def run_episode(
    cfg: GenerateConfig,
    env,
    raw_task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
    loadinfo_output_dir: Optional[str] = None,
    episode_idx: int = 0,
    color_variant: int = 0,
):
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    t = 0
    replay_images = []
    replay_masked_images = []
    last_masked = None  # Reuse for video when not querying policy (mask subprocess is slow; only compute every 8 steps)
    max_steps = TASK_MAX_STEPS.get(cfg.task_suite_name, 150)
    loadinfo_rows: List[Dict[str, Any]] = []
    loadinfo_ep_dir: Optional[str] = None
    if cfg.loadinfo and loadinfo_output_dir:
        safe_name = raw_task_description.replace(" ", "_").replace("/", "_")
        loadinfo_ep_dir = os.path.join(loadinfo_output_dir, safe_name, f"episode_{episode_idx}")
        os.makedirs(loadinfo_ep_dir, exist_ok=True)

    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                # Record frame during wait so video has at least num_steps_wait frames
                observation, img, wrist_img = prepare_observation(obs, resize_size)
                if getattr(cfg, "perturb_colors", False):
                    img = _apply_color_perturbation(img, color_variant)
                    wrist_img = _apply_color_perturbation(wrist_img, color_variant)
                if getattr(cfg, "perturb_bowl", False):
                    img = _apply_bowl_perturbation(img, color_variant)
                    wrist_img = _apply_bowl_perturbation(wrist_img, color_variant)
                replay_images.append(img)
                img_np = np.asarray(img) if not isinstance(img, np.ndarray) else img
                if img_np.ndim == 2:
                    img_np = np.stack([img_np] * 3, axis=-1)
                replay_masked_images.append(np.asarray(img_np))
                obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # Every step: prepare frame and record for video
            # Use raw task description when policy doesn't use mask (openvla_oft_goal, openvla_7b)
            task_description = (
                language_mask_processor(raw_task_description)
                if getattr(cfg, "use_mask_for_policy", False)
                else raw_task_description
            )
            observation, img, wrist_img = prepare_observation(obs, resize_size)
            if getattr(cfg, "perturb_colors", False):
                img = _apply_color_perturbation(img, color_variant)
                wrist_img = _apply_color_perturbation(wrist_img, color_variant)
            if getattr(cfg, "perturb_bowl", False):
                img = _apply_bowl_perturbation(img, color_variant)
                wrist_img = _apply_bowl_perturbation(wrist_img, color_variant)
            replay_images.append(img)

            # Perturbed images for policy (both full and wrist when num_images_in_input > 1)
            img_perturbed = img
            wrist_img_perturbed = wrist_img

            # Only compute mask when we need to query policy (every num_open_loop_steps); reuse last_masked for other steps
            if getattr(cfg, "use_mask_for_policy", False):
                if len(action_queue) == 0:
                    if getattr(cfg, "use_mask_from_env", False):
                        img_np = np.asarray(img_perturbed) if not isinstance(img_perturbed, np.ndarray) else img_perturbed
                        if img_np.ndim == 2:
                            img_np = np.stack([img_np] * 3, axis=-1)
                        seg_key = "agentview_segmentation_instance"
                        if seg_key in obs:
                            masked = mask_image_from_libero_seg(img_np, obs[seg_key], env, alpha=0.5)
                        else:
                            masked = img_np
                        last_masked = np.asarray(masked)
                        replay_masked_images.append(last_masked.copy())
                    else:
                        if isinstance(img_perturbed, np.ndarray):
                            img_pil = Image.fromarray(img_perturbed)
                        else:
                            img_pil = img_perturbed
                        out_dir = "/home/ubuntu/16831pro_fine_tune/debug_masked_validation"
                        os.makedirs(out_dir, exist_ok=True)
                        out_path = os.path.join(out_dir, f"{raw_task_description.replace(' ', '_')}.png")
                        masked = mask_image_via_other_env(img_pil.convert("RGB"), raw_task_description, out_path)
                        last_masked = np.asarray(masked)
                        replay_masked_images.append(last_masked.copy())
                    observation["full_image"] = resize_image_for_policy(last_masked, resize_size)
                    observation["wrist_image"] = resize_image_for_policy(wrist_img_perturbed, resize_size)
                else:
                    replay_masked_images.append(last_masked.copy())
            else:
                img_np = np.asarray(img_perturbed) if not isinstance(img_perturbed, np.ndarray) else img_perturbed
                if img_np.ndim == 2:
                    img_np = np.stack([img_np] * 3, axis=-1)
                replay_masked_images.append(np.asarray(img_np))
                if len(action_queue) == 0:
                    observation["full_image"] = resize_image_for_policy(img_perturbed, resize_size)
                    observation["wrist_image"] = resize_image_for_policy(wrist_img_perturbed, resize_size)

            # Query policy only when action queue is empty (every 8 steps)
            if len(action_queue) == 0:
                actions = get_action(
                    cfg, model, observation, task_description,
                    processor=processor, action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)

                if cfg.loadinfo and loadinfo_ep_dir:
                    if getattr(cfg, "use_mask_for_policy", False) and last_masked is not None:
                        img_for_step = np.asarray(last_masked)
                    else:
                        img_for_step = np.asarray(img) if not isinstance(img, np.ndarray) else img
                    if img_for_step.ndim == 2:
                        img_for_step = np.stack([img_for_step] * 3, axis=-1)
                    masked_with_step = _draw_step_on_image(img_for_step, t)
                    mask_path = os.path.join(loadinfo_ep_dir, f"mask_step_{t:04d}.png")
                    masked_with_step.save(mask_path)
                    row: Dict[str, Any] = {"step": t}
                    for i, a in enumerate(actions):
                        arr = np.asarray(a)
                        for j in range(arr.size):
                            row[f"action_chunk_{i}_{j}"] = float(arr.flat[j])
                    proprio = observation.get("state")
                    if proprio is not None:
                        for j in range(proprio.size):
                            row[f"proprio_{j}"] = float(proprio.flat[j])
                    loadinfo_rows.append(row)

            action = action_queue.popleft()
            action = process_action(action, cfg.model_family)
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    if cfg.loadinfo and loadinfo_ep_dir and loadinfo_rows:
        excel_path = os.path.join(loadinfo_ep_dir, "action_chunk_proprio.xlsx")
        _save_loadinfo_excel(loadinfo_rows, excel_path, log_file)

    return success, replay_images, replay_masked_images


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
):
    task = task_suite.get_task(task_id)
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)
    env, task_description = get_libero_env(
        task, cfg.model_family, resolution=cfg.env_img_res,
        use_segmentation_env=(getattr(cfg, "use_mask_for_policy", False) and getattr(cfg, "use_mask_from_env", False)),
    )
    raw_task_description = _task_description_for_policy(task_description, cfg)
    task_short = _task_short_name(task_description)

    task_episodes, task_successes = 0, 0
    for color_variant in range(2):  # 2 color conversions per task: variant 0 and variant 1
        color_suffix = f"{task_short}_{cfg.model_label}-color{color_variant}"
        log_message(f"\nTask: {task_description} | Color variant: {color_variant}", log_file)

        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            log_message(f"\nTask: {task_description} | Color variant {color_variant} | Trial {episode_idx + 1}", log_file)

            if cfg.initial_states_path == "DEFAULT":
                initial_state = initial_states[episode_idx]
            else:
                initial_states_task_key = task_description.replace(" ", "_")
                episode_key = f"demo_{episode_idx}"
                if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                    log_message(f"Skipping task {task_id} episode {episode_idx} (failed expert demo)", log_file)
                    continue
                initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

            log_message(f"Starting episode {task_episodes + 1}...", log_file)
            loadinfo_output_dir = None
            if cfg.loadinfo:
                loadinfo_output_dir = os.path.join(cfg.local_log_dir, "loadinfo_output")
                os.makedirs(loadinfo_output_dir, exist_ok=True)

            success, replay_images, replay_masked_images = run_episode(
                cfg, env, raw_task_description, model, resize_size,
                processor, action_head, proprio_projector, noisy_action_projector,
                initial_state, log_file, loadinfo_output_dir=loadinfo_output_dir, episode_idx=episode_idx,
                color_variant=color_variant,
            )

            task_episodes += 1
            total_episodes += 1
            if success:
                task_successes += 1
                total_successes += 1

            # Only save side-by-side video (raw left, masked right)
            suffix_side = f"sidebyside_{color_suffix}"
            if cfg.num_trials_per_task > 1:
                suffix_side += f"_trial{episode_idx + 1}"
            _save_sidebyside_video(
                replay_images, replay_masked_images, total_episodes, success=success,
                task_description=task_description, suffix=suffix_side, log_file=log_file,
                model_label=cfg.model_label,
            )

            log_message(f"Success: {success}", log_file)
            log_message(f"# episodes so far: {total_episodes}", log_file)
            log_message(f"# successes: {total_successes} ({100.0 * total_successes / total_episodes:.1f}%)", log_file)

    task_sr = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_sr = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    log_message(f"Task success rate: {task_sr}", log_file)
    log_message(f"Total success rate: {total_sr}", log_file)
    if cfg.use_wandb:
        wandb.log({f"success_rate/{task_description}": task_sr, f"num_episodes/{task_description}": task_episodes})
    return total_episodes, total_successes


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    _resolve_base_vla_path(cfg)
    validate_config(cfg)

    if getattr(cfg, "load_in_8bit", False) or getattr(cfg, "load_in_4bit", False):
        from accelerate import big_modeling as _acc_bm
        import transformers.modeling_utils as _tf_mu
        _orig = _acc_bm.dispatch_model
        def _patched(m, *a, force_hooks=False, **kw):
            return _orig(m, *a, force_hooks=True, **kw)
        _acc_bm.dispatch_model = _tf_mu.dispatch_model = _patched

    set_seed_everywhere(cfg.seed)
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    task_ids_to_run = []
    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        desc = (task.language or "").strip().lower()
        if desc in [t.strip().lower() for t in COLOR_TASKS]:
            task_ids_to_run.append((task_id, task.language))

    if not task_ids_to_run:
        log_message("No matching tasks for COLOR_TASKS.", log_file)
        log_file.close()
        return 0.0

    log_message(f"Color perturb tasks: {[d for _, d in task_ids_to_run]}", log_file)
    log_message(f"2 color variants × {cfg.num_trials_per_task} trials per task", log_file)
    log_file.flush()

    total_episodes, total_successes = 0, 0
    for task_id, task_description in task_ids_to_run:
        desc_lower = (task_description or "").strip().lower()
        if "push" in desc_lower:
            cfg.perturb_colors = True
            cfg.perturb_bowl = False
        else:
            cfg.perturb_colors = False
            cfg.perturb_bowl = True

        total_episodes, total_successes = run_task(
            cfg, task_suite, task_id, model, resize_size,
            processor, action_head, proprio_projector, noisy_action_projector,
            total_episodes, total_successes, log_file,
        )

    final_sr = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_sr:.4f} ({100 * final_sr:.1f}%)", log_file)
    if cfg.use_wandb:
        wandb.log({"success_rate/total": final_sr, "num_episodes/total": total_episodes})
        wandb.save(local_log_filepath)
    log_file.close()
    return final_sr


if __name__ == "__main__":
    eval_libero()
