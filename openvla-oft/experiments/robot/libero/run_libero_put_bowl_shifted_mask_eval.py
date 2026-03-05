"""
run_libero_put_bowl_shifted_mask_eval.py

Test "put the bowl on the plate" with mask at original plate location and (optionally) no-plate env.

When use_no_plate_env=True (default):
  - Test environment is a new LIBERO scene WITHOUT the plate (put_the_bowl_on_the_plate_no_plate.bddl).
  - Mask: green is generated at the ORIGINAL plate location (from a reference env-with-plate image).
  - Policy sees: scene with no plate + green mask at where the plate would be.
  - Success: bowl placed in the plate_region (custom check), since env has no plate object.

When use_no_plate_env=False:
  - Normal env with plate; mask at original plate (shift_plate_pixels=0) or shifted if list given.

Only runs "put the bowl on the plate" task (use_mask_for_policy=True).
"""
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

from PIL import Image, ImageDraw, ImageFont
import json
import logging
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
    mask_image_via_other_env_shifted_plate,
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

TASK_NAME = "put the bowl on the plate"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Default current-model checkpoint (override with --pretrained_checkpoint or env CURRENT_CHECKPOINT)
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
    load_in_8bit: bool = True   # default True to avoid OOM on ~14GB GPU (mask subprocess + policy)
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
    use_no_plate_env: bool = True   # When True: eval in env without plate, mask at original plate location
    dest_mask_white: bool = True    # When True: destination (plate) region in mask is drawn white; else green
    shift_plate_pixels: int = 0     # when use_no_plate_env=True this is forced to 0 (no shift)
    shift_plate_pixels_list: str = ""  # comma-separated; ignored when use_no_plate_env=True

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


def _normalize_checkpoint_path(ckpt: Any) -> str:
    """Ensure we have a valid local path; never pass None or 'None' to from_pretrained."""
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
    run_id = f"PUT-BOWL-SHIFTED-MASK-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
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


# Plate region in table frame (from put_the_bowl_on_the_plate.bddl main_table_plate_region)
# (:ranges ((0.04 -0.03 0.06 -0.01))) -> x in [0.04, 0.06], y in [-0.03, -0.01]
PLATE_REGION_X_MIN, PLATE_REGION_X_MAX = 0.04, 0.06
PLATE_REGION_Y_MIN, PLATE_REGION_Y_MAX = -0.03, -0.01


def _extract_green_mask_from_masked_image(masked_rgb: np.ndarray) -> np.ndarray:
    """Extract boolean mask of green (destination) pixels from a masked image (red/green overlay)."""
    r, g, b = masked_rgb[:, :, 0], masked_rgb[:, :, 1], masked_rgb[:, :, 2]
    green_mask = (g > 200) & (r < 100) & (b < 100)
    return green_mask.astype(np.uint8)


def _overlay_dest_mask_on_image(
    img: np.ndarray,
    dest_mask: np.ndarray,
    alpha: float = 0.35,
    color: tuple = (255, 255, 255),
) -> np.ndarray:
    """Overlay color at dest_mask pixels on img (e.g. white for destination region). img and dest_mask same H,W."""
    out = np.array(img, dtype=np.float64)
    rgb = np.array([float(color[0]), float(color[1]), float(color[2])])
    mask_f = dest_mask.astype(np.float64)
    if mask_f.ndim == 2:
        mask_f = np.stack([mask_f] * 3, axis=-1)
    for c in range(3):
        out[:, :, c] = (1 - alpha * mask_f[:, :, c]) * out[:, :, c] + alpha * mask_f[:, :, c] * rgb[c]
    return np.clip(out, 0, 255).astype(np.uint8)


def _check_bowl_in_plate_region(env) -> bool:
    """Check if akita_black_bowl_1 is inside the plate_region (for no-plate env success)."""
    try:
        body_id = env.obj_body_id.get("akita_black_bowl_1")
        if body_id is None:
            return False
        pos = env.sim.data.body_xpos[body_id]
        x, y = pos[0], pos[1]
        return (
            PLATE_REGION_X_MIN <= x <= PLATE_REGION_X_MAX
            and PLATE_REGION_Y_MIN <= y <= PLATE_REGION_Y_MAX
        )
    except Exception:
        return False


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
    model_tag = model_label if model_label else "shifted_mask"
    mp4_path = f"{rollout_dir}/{DATE_TIME}--{model_tag}--episode={idx}--success={success}--task={processed}{extra}.mp4"
    # Align lengths (should match; if not, use min)
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
    # If only one frame, duplicate so video is viewable (e.g. 3 sec at 30fps)
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
    green_mask_ref: Optional[np.ndarray] = None,
    use_no_plate_env: bool = False,
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
    last_masked = None
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
            replay_images.append(img)

            img_perturbed = img
            wrist_img_perturbed = wrist_img

            if cfg.use_mask_for_policy:
                if len(action_queue) == 0:
                    if isinstance(img_perturbed, np.ndarray):
                        img_pil = Image.fromarray(img_perturbed)
                    else:
                        img_pil = img_perturbed
                    out_dir = "/home/ubuntu/16831pro_fine_tune/debug_masked_validation"
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f"put_the_bowl_on_the_plate_shifted_trial{episode_idx}.png")
                    if green_mask_ref is not None:
                        # No-plate env: current image has no plate; get red (bowl) from mask, overlay ref green
                        masked = mask_image_via_other_env_shifted_plate(
                            img_pil.convert("RGB"),
                            raw_task_description,
                            out_path,
                            shift_pixels=0,
                            hide_plate_mask=False,
                        )
                        masked_arr = np.asarray(masked)
                        # Ensure green mask shape matches current image (do not mutate shared ref)
                        gm = green_mask_ref
                        gh, gw = gm.shape[:2]
                        if masked_arr.shape[0] != gh or masked_arr.shape[1] != gw:
                            from PIL import Image as PILImage
                            gm_img = PILImage.fromarray((gm * 255).astype(np.uint8))
                            gm_img = gm_img.resize((masked_arr.shape[1], masked_arr.shape[0]), PILImage.NEAREST)
                            gm = (np.asarray(gm_img) > 127).astype(np.uint8)
                        dest_color = (255, 255, 255) if getattr(cfg, "dest_mask_white", True) else (0, 255, 0)
                        last_masked = _overlay_dest_mask_on_image(masked_arr, gm, color=dest_color)
                    else:
                        # With-plate env: green at original (shift=0) or shifted position
                        masked = mask_image_via_other_env_shifted_plate(
                            img_pil.convert("RGB"),
                            raw_task_description,
                            out_path,
                            shift_pixels=cfg.shift_plate_pixels,
                            hide_plate_mask=False,
                        )
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
                    if cfg.use_mask_for_policy and last_masked is not None:
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
            if use_no_plate_env:
                if _check_bowl_in_plate_region(env):
                    success = True
                    break
            elif done:
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
    use_no_plate = getattr(cfg, "use_no_plate_env", False)
    green_mask_ref = None

    if use_no_plate:
        # Create reference env WITH plate to get green mask at original plate location
        ref_env, _ = get_libero_env(
            task, cfg.model_family, resolution=cfg.env_img_res,
            use_segmentation_env=False,
        )
        ref_env.reset()
        ref_init = initial_states[0] if initial_states is not None else None
        if ref_init is not None:
            ref_env.set_init_state(ref_init)
        ref_obs = ref_env.get_observation()
        ref_img = get_libero_image(ref_obs)
        if isinstance(ref_img, np.ndarray):
            ref_pil = Image.fromarray(ref_img)
        else:
            ref_pil = ref_img
        out_dir = "/home/ubuntu/16831pro_fine_tune/debug_masked_validation"
        os.makedirs(out_dir, exist_ok=True)
        ref_mask_path = os.path.join(out_dir, "put_the_bowl_on_the_plate_ref_plate_mask.png")
        ref_masked = mask_image_via_other_env_shifted_plate(
            ref_pil.convert("RGB"),
            task.language,
            ref_mask_path,
            shift_pixels=0,
            hide_plate_mask=False,
        )
        green_mask_ref = _extract_green_mask_from_masked_image(np.asarray(ref_masked))
        if hasattr(ref_env, "close"):
            ref_env.close()
        del ref_env
        log_message("Precomputed green mask at original plate location (from ref env with plate)", log_file)

    env, task_description = get_libero_env(
        task, cfg.model_family, resolution=cfg.env_img_res,
        use_segmentation_env=False,
        bddl_file_override="put_the_bowl_on_the_plate_no_plate.bddl" if use_no_plate else None,
    )
    raw_task_description = task_description

    task_episodes, task_successes = 0, 0
    suffix = f"put_{cfg.model_label}_noplate" if use_no_plate else f"put_{cfg.model_label}_shifted{cfg.shift_plate_pixels}"

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
            green_mask_ref=green_mask_ref,
            use_no_plate_env=use_no_plate,
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

    use_no_plate = getattr(cfg, "use_no_plate_env", True)
    shift_list = [] if use_no_plate else []
    if not use_no_plate:
        list_str = (getattr(cfg, "shift_plate_pixels_list", "") or "").strip() or os.environ.get("SHIFT_PLATE_PIXELS_LIST", "").strip()
        if list_str:
            shift_list = [int(x.strip()) for x in list_str.split(",") if x.strip()]

    log_message(f"Put bowl mask eval: {TASK_NAME}", log_file)
    if use_no_plate:
        log_message("Mode: no-plate env, green mask at original plate location", log_file)
        log_message(f"{cfg.num_trials_per_task} trials", log_file)
    elif shift_list:
        log_message(f"Green mask positions (one episode each): {shift_list} pixels", log_file)
    else:
        log_message(f"Green mask shifted by {cfg.shift_plate_pixels} pixels", log_file)
        log_message(f"{cfg.num_trials_per_task} trials", log_file)
    log_file.flush()

    total_episodes, total_successes = 0, 0
    for task_id, task_description in task_ids_to_run:
        if shift_list:
            for shift_px in shift_list:
                cfg.shift_plate_pixels = shift_px
                cfg.num_trials_per_task = 1
                total_episodes, total_successes = run_task(
                    cfg, task_suite, task_id, model, resize_size,
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
