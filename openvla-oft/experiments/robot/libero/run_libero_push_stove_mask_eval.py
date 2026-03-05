"""
run_libero_push_stove_mask_eval.py

Test "push the plate to the front of the stove" with a synthetic "no stove" setup:
- Policy sees: full mask_processor output (black bg + red plate overlay + green stove + gripper white dots).
- Raw (left) video: same frame with the stove rectangle set to WHITE.

Uses get_policy_image_and_green_mask_via_other_env (mask_one) so plate red is the same overlay as
mask_processor and gripper two points are included. Optional green_rect_shift_x/y for position jitter;
use --jitter_shifts "0,0 10,0 -10,0 0,10 0,-10 5,5" for multiple jitter experiments (1 trial each).
"""
import os
import tempfile

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

from PIL import Image, ImageDraw, ImageFont
import json
import logging
from collections import deque
from dataclasses import dataclass, replace as dc_replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
    get_policy_image_and_green_mask_via_other_env,
    language_mask_processor,
)

import sys
sys.path.append("../..")
import imageio
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
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


TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 150,
    "libero_10": 520,
    "libero_90": 400,
}

TASK_NAME = "push the plate to the front of the stove"

# Policy: plate = red, stove rect = green; raw video: white at stove rect.
RED_RGB = np.array([255, 0, 0], dtype=np.uint8)
GREEN_RGB = np.array([0, 255, 0], dtype=np.uint8)
WHITE_RGB = np.array([255, 255, 255], dtype=np.uint8)
BLACK_RGB = np.array([0, 0, 0], dtype=np.uint8)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


_DEFAULT_CURRENT_CHECKPOINT = "/home/ubuntu/runs/openvla_adapters/openvla-7b+libero_goal_no_noops+b8+lr-0.0001+lora-r8+dropout-0.0+lora-attn-only--13500_chkpt"


@dataclass
class GenerateConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = _DEFAULT_CURRENT_CHECKPOINT
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
    load_in_8bit: bool = True
    load_in_4bit: bool = False

    task_suite_name: str = TaskSuite.LIBERO_GOAL
    num_steps_wait: int = 10
    num_trials_per_task: int = 3
    initial_states_path: str = "DEFAULT"
    env_img_res: int = 256

    run_id_note: Optional[str] = None
    model_label: str = "current"
    local_log_dir: str = "./experiments/info"

    loadinfo: bool = False
    use_mask_for_policy: bool = True
    use_mask_from_env: bool = False

    # Position jitter for white/green rectangle (stove): shift in pixels. (0,0)=no jitter.
    green_rect_shift_x_px: int = 0
    green_rect_shift_y_px: int = 0
    # If set, run multiple jitter experiments (one trial each). E.g. "0,0 10,0 -10,0 0,10 0,-10 5,5"
    jitter_shifts: str = ""
    # Expand stove rectangle by this many pixels on each side (default 0; increase if stove peeks out).
    green_rect_margin_px: int = 0
    # Recompute mask every N steps (0=only first frame). Default 8 = align with action inference (num_open_loop_steps).
    update_mask_every_n_steps: int = 8

    use_wandb: bool = False
    wandb_entity: str = "maggiesh-carnegie-mellon-university"
    wandb_project: str = "validation"

    seed: int = 7


def _resolve_base_vla_path(cfg: GenerateConfig) -> None:
    """Ensure base_vla_path is set so adapter-only checkpoints (no config.json, have lora_adapter/) can load."""
    openvla_root = Path(__file__).resolve().parent.parent.parent.parent
    fallback = openvla_root / "checkpoints" / "openvla-7b"
    base = getattr(cfg, "base_vla_path", None)
    if base and isinstance(base, str) and base.strip():
        base = base.strip()
        if (base.startswith("/") or base.startswith(".")) and not os.path.isdir(base) and fallback.is_dir():
            cfg.base_vla_path = str(fallback)
            logger.info("base_vla_path %s not found; using %s", base, cfg.base_vla_path)
        return
    if fallback.is_dir():
        cfg.base_vla_path = str(fallback)
        logger.info("base_vla_path defaulted to %s (required for adapter-only checkpoints)", cfg.base_vla_path)


def _normalize_checkpoint_path(ckpt: Any) -> str:
    def _use_default_if_invalid(p: str) -> str:
        p = (p or "").strip()
        return p if p and p.lower() != "none" else _DEFAULT_CURRENT_CHECKPOINT

    env_ckpt = os.environ.get("CURRENT_CHECKPOINT", _DEFAULT_CURRENT_CHECKPOINT)
    if ckpt is None:
        return _use_default_if_invalid(env_ckpt)
    s = str(ckpt).strip()
    if not s or s.lower() == "none":
        return _use_default_if_invalid(env_ckpt)
    return s


def validate_config(cfg: GenerateConfig) -> None:
    cfg.pretrained_checkpoint = _normalize_checkpoint_path(cfg.pretrained_checkpoint)
    ckpt = cfg.pretrained_checkpoint
    assert ckpt and ckpt.strip() and ckpt.lower() != "none", (
        "pretrained_checkpoint must be a non-empty path. "
        "Set --pretrained_checkpoint or env CURRENT_CHECKPOINT."
    )
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    assert cfg.task_suite_name in [s.value for s in TaskSuite], f"Invalid task_suite_name: {cfg.task_suite_name}"


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
    run_id = f"PUSH-STOVE-MASK-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
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
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    extra = f"--{suffix}" if suffix else ""
    model_tag = model_label if model_label else "push_stove_mask"
    mp4_path = f"{rollout_dir}/{DATE_TIME}--{model_tag}--episode={idx}--success={success}--task={processed}{extra}.mp4"
    n_raw, n_mask = len(raw_images), len(masked_images)
    if n_raw != n_mask and log_file:
        log_file.write(f"Frame count mismatch: raw={n_raw} masked={n_mask}, using min\n")
        log_file.flush()
    n_frames = max(1, min(n_raw, n_mask))
    raw_list = list(raw_images[:n_frames]) if raw_images else [np.zeros((256, 256, 3), dtype=np.uint8)]
    mask_list = list(masked_images[:n_frames]) if masked_images else [np.zeros((256, 256, 3), dtype=np.uint8)]
    if not raw_list:
        raw_list = [np.zeros((256, 256, 3), dtype=np.uint8)]
    if not mask_list:
        mask_list = [np.zeros((256, 256, 3), dtype=np.uint8)]
    if len(raw_list) == 1:
        raw_list = raw_list * (fps * 3)
        mask_list = mask_list * (fps * 3)
        if log_file:
            log_file.write("Video had only 1 frame; duplicated for 3s playback.\n")
            log_file.flush()
    writer = imageio.get_writer(mp4_path, fps=fps)
    for raw, mask in zip(raw_list, mask_list):
        raw_np = np.asarray(raw)
        mask_np = np.asarray(mask)
        if raw_np.shape != mask_np.shape:
            mask_np = np.asarray(
                Image.fromarray(mask_np).resize((raw_np.shape[1], raw_np.shape[0]), Image.Resampling.LANCZOS)
            )
        sidebyside = np.concatenate([raw_np, mask_np], axis=1)
        writer.append_data(np.clip(sidebyside.astype(np.uint8), 0, 255))
    writer.close()
    logger.info("Saved side-by-side video: %s (%s frames)", mp4_path, len(raw_list))
    if log_file:
        log_file.write(f"Saved side-by-side video: {mp4_path}\n")
        log_file.flush()
    return mp4_path


def _ensure_mask_shape(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize mask to (h, w) if needed; return bool (H,W)."""
    if mask.shape[:2] == (h, w):
        return mask.astype(bool) if mask.dtype != bool else mask
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8)).resize(
            (w, h), Image.Resampling.NEAREST
        )
    ).astype(bool)


def _mask_to_rectangular(mask: np.ndarray) -> np.ndarray:
    """Return a rectangular (axis-aligned) mask covering the same region as mask. Shape (H,W) bool."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return np.zeros_like(mask, dtype=bool)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    out = np.zeros(mask.shape, dtype=bool)
    out[rmin : rmax + 1, cmin : cmax + 1] = True
    return out


def _rect_bounds(mask: np.ndarray):
    """Return (rmin, rmax, cmin, cmax) for the axis-aligned box of mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    rmin, rmax = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    cmin, cmax = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    return (rmin, rmax, cmin, cmax)


def _expand_rect_bounds(h: int, w: int, rmin: int, rmax: int, cmin: int, cmax: int, margin_px: int):
    """Expand rect by margin_px on each side, clipped to image. Returns (rmin, rmax, cmin, cmax)."""
    rmin = max(0, rmin - margin_px)
    rmax = min(h - 1, rmax + margin_px)
    cmin = max(0, cmin - margin_px)
    cmax = min(w - 1, cmax + margin_px)
    return (rmin, rmax, cmin, cmax)


def _shift_rect_bounds(h: int, w: int, rmin: int, rmax: int, cmin: int, cmax: int, dx: int, dy: int):
    """Return (rmin_n, rmax_n, cmin_n, cmax_n) shifted by (dx, dy) and clipped to [0,h)x[0,w)."""
    rmin_n = max(0, min(h, rmin + dy))
    rmax_n = max(0, min(h, rmax + dy))
    cmin_n = max(0, min(w, cmin + dx))
    cmax_n = max(0, min(w, cmax + dx))
    if rmin_n >= rmax_n or cmin_n >= cmax_n:
        return (0, 0, 0, 0)
    return (rmin_n, rmax_n, cmin_n, cmax_n)


def _apply_region_mask(img_np: np.ndarray, mask: np.ndarray, color: np.ndarray) -> np.ndarray:
    """Return a copy of img_np with mask region set to color. img_np (H,W,3), mask (H,W) bool."""
    out = np.array(img_np, dtype=np.uint8, copy=True)
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=-1)
    mask = _ensure_mask_shape(mask, out.shape[0], out.shape[1])
    out[mask] = color
    return out


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
    policy_image_pil: Optional[Image.Image] = None
    stove_rect_mask_np: Optional[np.ndarray] = None
    stove_rect_bounds: Optional[tuple] = None  # (rmin, rmax, cmin, cmax) for first-frame green mask
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
                obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            task_description = language_mask_processor(raw_task_description)
            observation, img, wrist_img = prepare_observation(obs, resize_size)
            img_np = np.asarray(img) if not isinstance(img, np.ndarray) else img.copy()
            if img_np.ndim == 2:
                img_np = np.stack([img_np] * 3, axis=-1)
            policy_frame = img_np

            if cfg.use_mask_for_policy:
                h, w = img_np.shape[0], img_np.shape[1]
                steps_since_wait = t - cfg.num_steps_wait
                update_every = getattr(cfg, "update_mask_every_n_steps", 0)
                should_update_mask = (
                    policy_image_pil is None
                    or (update_every > 0 and steps_since_wait >= 0 and steps_since_wait % update_every == 0)
                )
                if should_update_mask:
                    img_pil = Image.fromarray(img_np)
                    tmp_dir = tempfile.mkdtemp(prefix="push_stove_mask_")
                    policy_out = os.path.join(tmp_dir, "policy_mask_processor.png")
                    green_mask_path = os.path.join(tmp_dir, "stove_green_mask.png")
                    policy_image_pil, _ = get_policy_image_and_green_mask_via_other_env(
                        img_pil.convert("RGB"), raw_task_description, policy_out, green_mask_path
                    )
                    if policy_image_pil.size != (w, h):
                        policy_image_pil = policy_image_pil.resize((w, h), Image.Resampling.LANCZOS)
                    # Green rectangle: only set from first frame so it doesn't suddenly change size (e.g. SAM drift).
                    if stove_rect_mask_np is None:
                        stove_shape_mask = _ensure_mask_shape(
                            np.array(Image.open(green_mask_path).convert("L")) > 0, h, w
                        )
                        stove_rect_mask_np = _mask_to_rectangular(stove_shape_mask)
                        stove_rect_bounds = _rect_bounds(stove_rect_mask_np)
                        margin_px = max(0, getattr(cfg, "green_rect_margin_px", 0))
                        if stove_rect_bounds and margin_px > 0:
                            rmin, rmax, cmin, cmax = _expand_rect_bounds(h, w, *stove_rect_bounds, margin_px)
                            stove_rect_mask_np = np.zeros((h, w), dtype=bool)
                            stove_rect_mask_np[rmin : rmax + 1, cmin : cmax + 1] = True
                            stove_rect_bounds = (rmin, rmax, cmin, cmax)

                if policy_image_pil is not None and stove_rect_mask_np is not None:
                    policy_frame = np.array(policy_image_pil)
                    if policy_frame.ndim == 2:
                        policy_frame = np.stack([policy_frame] * 3, axis=-1)
                    shift_x = getattr(cfg, "green_rect_shift_x_px", 0)
                    shift_y = getattr(cfg, "green_rect_shift_y_px", 0)
                    if stove_rect_bounds and (shift_x != 0 or shift_y != 0):
                        rmin, rmax, cmin, cmax = stove_rect_bounds
                        rmin_s, rmax_s, cmin_s, cmax_s = _shift_rect_bounds(h, w, rmin, rmax, cmin, cmax, shift_x, shift_y)
                        if rmax_s > rmin_s and cmax_s > cmin_s:
                            policy_frame[rmin_s:rmax_s, cmin_s:cmax_s] = GREEN_RGB
                        rect_for_raw = np.zeros((h, w), dtype=bool)
                        rect_for_raw[rmin_s:rmax_s, cmin_s:cmax_s] = True
                        raw_frame = _apply_region_mask(img_np, rect_for_raw, WHITE_RGB)
                    else:
                        policy_frame = _apply_region_mask(policy_frame, stove_rect_mask_np, GREEN_RGB)
                        raw_frame = _apply_region_mask(img_np, stove_rect_mask_np, WHITE_RGB)
                else:
                    raw_frame = img_np
                    policy_frame = np.array(policy_image_pil) if policy_image_pil is not None else img_np
                    if policy_frame.ndim == 2:
                        policy_frame = np.stack([policy_frame] * 3, axis=-1)

                replay_images.append(raw_frame)
                replay_masked_images.append(policy_frame)
                observation["full_image"] = resize_image_for_policy(policy_frame, resize_size)
                observation["wrist_image"] = resize_image_for_policy(wrist_img, resize_size)
            else:
                replay_images.append(img_np)
                replay_masked_images.append(img_np)
                if len(action_queue) == 0:
                    observation["full_image"] = resize_image_for_policy(img_np, resize_size)
                    observation["wrist_image"] = resize_image_for_policy(wrist_img, resize_size)

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
                    img_for_step = policy_frame if (cfg.use_mask_for_policy and stove_rect_mask_np is not None) else img_np
                    if img_for_step.ndim == 2:
                        img_for_step = np.stack([img_for_step] * 3, axis=-1)
                    masked_with_step = _draw_step_on_image(img_for_step, t)
                    mask_path = os.path.join(loadinfo_ep_dir, f"mask_step_{t:04d}.png")
                    Image.fromarray(masked_with_step).save(mask_path)
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
        use_segmentation_env=False,
    )
    raw_task_description = task_description

    task_episodes, task_successes = 0, 0
    suffix = f"push_stove_mask_{cfg.model_label}"

    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description} | Trial {episode_idx + 1}", log_file)

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
        )

        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        suffix_side = f"sidebyside_{suffix}"
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
        if desc == TASK_NAME.strip().lower():
            task_ids_to_run.append((task_id, task.language))
            break

    if not task_ids_to_run:
        log_message(f"No task matching '{TASK_NAME}'. Available tasks may differ.", log_file)
        log_file.close()
        return 0.0

    def _parse_jitter_shifts(s: str) -> List[Tuple[int, int]]:
        if not (s or s.strip()):
            return []
        out = []
        for part in s.strip().split():
            part = part.strip()
            if not part:
                continue
            a, _, b = part.partition(",")
            try:
                out.append((int(a.strip()), int(b.strip())))
            except ValueError:
                continue
        return out

    jitter_list = _parse_jitter_shifts(getattr(cfg, "jitter_shifts", "") or "")
    if jitter_list:
        log_message("Policy: mask_processor full image (red plate + green stove + gripper); raw: white at stove rect", log_file)
        log_message(f"Jitter experiments: {len(jitter_list)} shifts, 1 trial each: {jitter_list}", log_file)
    else:
        log_message("Policy: mask_processor full image (red plate + green stove + gripper); raw: white at stove rect", log_file)
        log_message(f"{cfg.num_trials_per_task} trials", log_file)
    log_file.flush()

    total_episodes, total_successes = 0, 0
    for task_id, task_description in task_ids_to_run:
        if jitter_list:
            for jidx, (dx, dy) in enumerate(jitter_list):
                cfg_jitter = dc_replace(
                    cfg,
                    green_rect_shift_x_px=dx,
                    green_rect_shift_y_px=dy,
                    num_trials_per_task=1,
                    run_id_note=(cfg.run_id_note or "") + f"jitter_{dx}_{dy}",
                )
                log_message(f"\n--- Jitter experiment {jidx + 1}/{len(jitter_list)}: shift=({dx}, {dy}) ---", log_file)
                total_episodes, total_successes = run_task(
                    cfg_jitter, task_suite, task_id, model, resize_size,
                    processor, action_head, proprio_projector, noisy_action_projector,
                    total_episodes, total_successes, log_file,
                )
        else:
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
