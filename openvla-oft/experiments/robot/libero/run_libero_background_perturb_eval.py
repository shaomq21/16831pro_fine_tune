"""
run_libero_background_perturb_eval.py

Eval with background-only perturbation: elliptical color patches.
Two tasks: push plate to stove, put bowl on plate.
Schedule: background(3 variants) + baseline, 4 per task.
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

from PIL import Image
import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import draccus
import numpy as np
import tqdm

from libero.libero import benchmark
from prismatic.vla.datasets.datasets import language_mask_processor

# Append project root for experiments.robot imports (run from openvla-oft root)
_openvla_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_openvla_root))
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
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_EVAL_FORCE_CPU = os.environ.get("EVAL_FORCE_CPU", "0") == "1"
_old_torch_load = torch.load


def _torch_load_safe(*args, **kwargs):
    if _EVAL_FORCE_CPU:
        kwargs.setdefault("map_location", "cpu")
        kwargs.setdefault("weights_only", False)
    return _old_torch_load(*args, **kwargs)


torch.load = _torch_load_safe


# Only these two tasks
GENERALIZATION_TASKS = [
    "push the plate to the front of the stove",
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


def _task_description_for_policy(raw_description: str, cfg: "BackgroundPerturbConfig") -> str:
    """For OFT/7B (no mask): rewrite push 'plate'->'flat shaped object on the right'; put 'bowl'->'gray bowl'."""
    if getattr(cfg, "use_mask_for_policy", True):
        return raw_description
    d = (raw_description or "").strip().lower()
    out = raw_description
    if "push" in d and "plate" in d:
        out = out.replace("the plate", "the flat shaped object on the right")
    if "put" in d and "bowl" in d:
        out = out.replace("the bowl", "the gray bowl")
    return out

# Task suite for LIBERO_GOAL (contains these tasks)
TASK_SUITE_NAME = "libero_goal"
# Match run_libero_eval: libero_goal longest demo ~270 steps
TASK_MAX_STEPS = 300


class PerturbType(str, Enum):
    NONE = "none"
    BACKGROUND = "background"
    HUE = "hue"
    CHANNEL_SHUFFLE = "channel_shuffle"
    POSTERIZE = "posterize"
    INVERT = "invert"
    MASK_BG_BLACK = "mask_bg_black"  # black bg, only target objects (Grounded-SAM, alpha=0)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _apply_background_perturbation(
    img: Union[np.ndarray, Image.Image], variant: int, rng: np.random.Generator
) -> Union[np.ndarray, Image.Image]:
    """
    Background-only: soft elliptical color patches (光圈). Mild intensity.
    variant 0: warm (soft pink/peach), 1: cool (soft blue/lavender), 2: mixed (soft green/cyan).
    """
    arr = np.asarray(img).astype(np.float32).copy()
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return img

    h, w = arr.shape[:2]
    blend_alpha = 0.15  # mild overlay

    if variant == 0:
        colors = [
            [255, 230, 240],  # soft pink
            [255, 245, 230],  # peach
        ]
    elif variant == 1:
        colors = [
            [235, 240, 255],  # soft blue
            [245, 238, 255],  # lavender
        ]
    else:
        colors = [
            [235, 252, 245],  # mint
            [238, 250, 255],  # soft cyan
        ]

    # 4–6 soft elliptical patches (光圈)
    n_patches = 4 + rng.integers(0, 3)
    for _ in range(n_patches):
        cx = rng.integers(w // 4, 3 * w // 4)
        cy = rng.integers(h // 4, 3 * h // 4)
        rx = rng.integers(w // 6, w // 3)
        ry = rng.integers(h // 6, h // 3)
        color = np.array(colors[rng.integers(0, len(colors))], dtype=np.float32)

        yy, xx = np.ogrid[:h, :w]
        dist = ((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2
        mask = np.exp(-dist * 0.5)
        mask = np.clip(mask, 0, 1)
        for c in range(3):
            arr[:, :, c] = (1 - blend_alpha * mask) * arr[:, :, c] + blend_alpha * mask * color[c]

    out = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(out) if isinstance(img, Image.Image) else out


def _apply_hue_rotation(
    img: Union[np.ndarray, Image.Image], variant: int, rng: np.random.Generator
) -> Union[np.ndarray, Image.Image]:
    """Hue shift: variant 0 = +120°, variant 1 = +240°. Preserves geometry."""
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return img
    pil = Image.fromarray(arr) if not isinstance(img, Image.Image) else img
    pil = pil.convert("HSV")
    arr_hsv = np.array(pil)
    # PIL H is 0-255, 360° = 256. +120° ≈ 85, +240° ≈ 170
    shift = 85 if variant == 0 else 170
    arr_hsv[:, :, 0] = (arr_hsv[:, :, 0].astype(np.int32) + shift) % 256
    arr_hsv = np.clip(arr_hsv, 0, 255).astype(np.uint8)
    out_pil = Image.fromarray(arr_hsv, mode="HSV").convert("RGB")
    return out_pil if isinstance(img, Image.Image) else np.asarray(out_pil)


def _apply_channel_shuffle(
    img: Union[np.ndarray, Image.Image], variant: int, rng: np.random.Generator
) -> Union[np.ndarray, Image.Image]:
    """Permute RGB channels. variant 0 = BGR, 1 = GBR, 2 = BRG."""
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return img
    perms = [[2, 1, 0], [1, 2, 0], [2, 0, 1]]  # BGR, GBR, BRG
    perm = perms[variant % 3]
    out = arr[:, :, perm].copy()
    return Image.fromarray(out) if isinstance(img, Image.Image) else out


def _apply_posterize(
    img: Union[np.ndarray, Image.Image], variant: int, rng: np.random.Generator
) -> Union[np.ndarray, Image.Image]:
    """Reduce bits per channel. variant 0 = 3 bits, 1 = 2 bits."""
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return img
    n_levels = 8 if variant == 0 else 4  # 3 bits or 2 bits
    step = 255.0 / (n_levels - 1)
    arr = np.round(arr / step) * step
    out = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(out) if isinstance(img, Image.Image) else out


def _apply_invert(
    img: Union[np.ndarray, Image.Image], variant: int, rng: np.random.Generator
) -> Union[np.ndarray, Image.Image]:
    """Negative/invert image. Preserves geometry."""
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return img
    out = 255 - arr
    return Image.fromarray(out) if isinstance(img, Image.Image) else out


def _apply_perturbation(
    img: Union[np.ndarray, Image.Image],
    perturb_type: PerturbType,
    perturb_variant: int,
    rng: np.random.Generator,
) -> Union[np.ndarray, Image.Image]:
    """Dispatch to the appropriate perturbation function."""
    pil = Image.fromarray(np.asarray(img)) if not isinstance(img, Image.Image) else img
    if perturb_type == PerturbType.NONE:
        return img
    if perturb_type == PerturbType.BACKGROUND:
        return _apply_background_perturbation(pil, perturb_variant, rng)
    if perturb_type == PerturbType.HUE:
        return _apply_hue_rotation(pil, perturb_variant, rng)
    if perturb_type == PerturbType.CHANNEL_SHUFFLE:
        return _apply_channel_shuffle(pil, perturb_variant, rng)
    if perturb_type == PerturbType.POSTERIZE:
        return _apply_posterize(pil, perturb_variant, rng)
    if perturb_type == PerturbType.INVERT:
        return _apply_invert(pil, perturb_variant, rng)
    return img


def _apply_bowl_perturbation(img: Union[np.ndarray, Image.Image]) -> Union[np.ndarray, Image.Image]:
    """图里定位深灰/灰碗像素，重设为亮红色。"""
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return img
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mean_rgb = (r + g + b) / 3.0
    spread = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    dark_grey_mask = (spread < 40) & (mean_rgb >= 50) & (mean_rgb <= 140)
    arr[dark_grey_mask, 0] = 255
    arr[dark_grey_mask, 1] = 40
    arr[dark_grey_mask, 2] = 40
    out = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(out) if isinstance(img, Image.Image) else out


@dataclass
class BackgroundPerturbConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    base_vla_path: Optional[str] = None

    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 2  # OFT/libero: pass full + wrist; both get perturbation
    use_proprio: bool = True

    center_crop: bool = True
    num_open_loop_steps: int = 8
    lora_rank: int = 32
    unnorm_key: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    task_suite_name: str = TASK_SUITE_NAME
    num_steps_wait: int = 10
    num_trials_per_task: int = 3  # Episodes per (task, background) combination
    initial_states_path: str = "DEFAULT"
    env_img_res: int = 256

    run_id_note: Optional[str] = None
    model_label: str = "openvla_oft"  # For video naming: e.g. "openvla_7b", "openvla_oft_goal", "current"
    local_log_dir: str = "./experiments/info"
    use_mask_for_policy: bool = True  # If False (e.g. openvla_7b, oft_goal), policy sees raw image; only new model uses mask
    use_mask_from_env: bool = False  # False = use mask_processor (Grounded-SAM)
    dino_config_path: str = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    dino_ckpt_path: str = "groundingdino_swint_ogc.pth"
    sam_ckpt_path: str = "sam_vit_b_01ec64.pth"
    sam_type: str = "vit_b"
    mask_device: str = "cpu"  # Use CPU for mask to save GPU memory for VLA; subprocess path also runs on CPU
    perturb_bowl: bool = False  # If True: 深灰/灰碗像素→亮红
    run_baseline: bool = True  # If False (OFT/7B): skip baseline, only run perturb conditions
    use_wandb: bool = False
    wandb_entity: str = "maggiesh-carnegie-mellon-university"
    wandb_project: str = "validation"
    seed: int = 7


def _resolve_base_vla_path(cfg: BackgroundPerturbConfig) -> None:
    base = getattr(cfg, "base_vla_path", None)
    if not base or not isinstance(base, str) or not base.strip():
        return
    base = base.strip()
    if (base.startswith("/") or base.startswith(".")) and not os.path.isdir(base):
        root = Path(__file__).resolve().parent.parent.parent.parent
        fallback = root / "checkpoints" / "openvla-7b"
        if fallback.is_dir():
            cfg.base_vla_path = str(fallback)
            logger.info("base_vla_path %s not found; using %s", base, cfg.base_vla_path)


def _validate_config(cfg: BackgroundPerturbConfig) -> None:
    assert cfg.pretrained_checkpoint, "pretrained_checkpoint must be set!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit!"


def _initialize_model(cfg: BackgroundPerturbConfig):
    model = get_model(cfg)
    # Base openvla/openvla-7b uses built-in discrete action prediction; no external action_head/proprio_projector
    is_base_hf_model = cfg.pretrained_checkpoint == "openvla/openvla-7b"
    proprio_projector = None
    if cfg.use_proprio and not is_base_hf_model:
        proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)

    action_head = None
    if (cfg.use_l1_regression or cfg.use_diffusion) and not is_base_hf_model:
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


def _prepare_observation(obs, resize_size):
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_resized = resize_image_for_policy(wrist_img, resize_size)
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }
    return observation, img, wrist_img


def _process_action(action, model_family: str):
    action = normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = invert_gripper_action(action)
    return action


_mask_processor_masker = None
_use_mask_subprocess = False

# Mask subprocess config (same as datasets.py, but env forces CPU)
_VLA_PREPROCESS_PY = "/home/ubuntu/miniconda3/envs/vla-preprocess/bin/python"
_MASK_ONE_SCRIPT = Path(__file__).resolve().parents[3].parent / "tools" / "mask_one.py"
_DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
_DINO_CKPT = "groundingdino_swint_ogc.pth"
_SAM_CKPT = "sam_vit_b_01ec64.pth"
_SAM_TYPE = "vit_b"


def _mask_via_subprocess_cpu(img_pil: Image.Image, lang: str, alpha: float = 0.35) -> Image.Image:
    """Run mask_one.py in subprocess with CUDA hidden; mask uses CPU only. alpha=0 => black bg, objects in original color."""
    import subprocess
    import tempfile
    import shutil
    # Free GPU cache in main process before mask (leaves more room for VLA)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["NVIDIA_VISIBLE_DEVICES"] = ""
    for k in ["WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
        env.pop(k, None)
    tmp_dir = tempfile.mkdtemp(prefix="bg_perturb_mask_tmp_")
    try:
        in_path = os.path.join(tmp_dir, "in.png")
        out_path = os.path.join(tmp_dir, "out.png")
        img_pil.save(in_path)
        with open(in_path, "rb") as f:
            os.fsync(f.fileno())
        import shlex
        lang_safe = shlex.quote(lang)
        cmd_str = (
            'CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES="" '
            f'"{_VLA_PREPROCESS_PY}" -u "{_MASK_ONE_SCRIPT}" '
            f'--image_in "{in_path}" --image_out "{out_path}" --lang {lang_safe} '
            f'--alpha {alpha} '
            f'--dino_config "{_DINO_CONFIG}" --dino_ckpt "{_DINO_CKPT}" '
            f'--sam_ckpt "{_SAM_CKPT}" --sam_type "{_SAM_TYPE}" --device cpu'
        )
        r = subprocess.run(cmd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        if r.returncode != 0:
            raise RuntimeError(f"mask_one.py failed: {r.stderr}")
        return Image.open(out_path).convert("RGB")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _get_mask_bg_black(
    img_pil: Image.Image, lang: str, cfg: BackgroundPerturbConfig
) -> Image.Image:
    """Black background, only target objects in original color (alpha=0)."""
    masker = _get_mask_processor_masker(cfg)
    img_rgb = img_pil.convert("RGB") if hasattr(img_pil, "convert") else Image.fromarray(np.asarray(img_pil)).convert("RGB")
    if masker is not None:
        out = masker.mask_image_from_lang(img_rgb, lang, alpha=0.0)
    else:
        out = _mask_via_subprocess_cpu(img_rgb, lang, alpha=0.0)
    return out.convert("RGB") if hasattr(out, "convert") else Image.fromarray(np.asarray(out)).convert("RGB")


def _get_mask_processor_masker(cfg: BackgroundPerturbConfig):
    """Lazy-load GroundedSAMMasker. Falls back to subprocess if groundingdino not in current env."""
    global _mask_processor_masker, _use_mask_subprocess
    if _mask_processor_masker is not None:
        return _mask_processor_masker
    if _use_mask_subprocess:
        return None
    try:
        from mask_processor import GroundedSAMMasker, GroundedSAMConfig
        mask_cfg = GroundedSAMConfig(
            dino_config_path=cfg.dino_config_path,
            dino_checkpoint_path=cfg.dino_ckpt_path,
            sam_checkpoint_path=cfg.sam_ckpt_path,
            sam_type=cfg.sam_type,
            device=cfg.mask_device,
        )
        _mask_processor_masker = GroundedSAMMasker(mask_cfg)
        return _mask_processor_masker
    except ModuleNotFoundError as e:
        logger.info("mask_processor in-process not available (%s), using subprocess (vla-preprocess)", e)
        _use_mask_subprocess = True
        return None


def _save_sidebyside_video(
    raw_images: List[np.ndarray],
    masked_images: List[np.ndarray],
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
    model_tag = model_label if model_label else "bg_perturb"
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
    cfg: BackgroundPerturbConfig,
    env,
    raw_task_description: str,
    model,
    resize_size,
    processor,
    action_head,
    proprio_projector,
    noisy_action_projector,
    initial_state,
    log_file,
    perturb_type: PerturbType,
    perturb_variant: int,
    rng: np.random.Generator,
) -> Tuple[bool, List, List]:
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    t = 0
    replay_images = []
    replay_masked_images = []
    last_masked = None  # Reuse for video when not calling policy (mask subprocess is slow)
    last_masked_full = None  # For MASK_BG_BLACK reuse
    last_masked_wrist = None
    max_steps = TASK_MAX_STEPS
    success = False

    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # Record one frame per step for full video
            # Use raw task description when policy doesn't use mask (openvla_oft_goal, openvla_7b);
            # they were trained on raw language, not "red masked"/"green masked"
            task_description = (
                language_mask_processor(raw_task_description)
                if cfg.use_mask_for_policy
                else raw_task_description
            )
            observation, img, wrist_img = _prepare_observation(obs, resize_size)

            img_np = np.asarray(img) if not isinstance(img, np.ndarray) else img
            if img_np.ndim == 2:
                img_np = np.stack([img_np] * 3, axis=-1)
            img_pil = Image.fromarray(img_np)
            wrist_img_np = np.asarray(wrist_img) if not isinstance(wrist_img, np.ndarray) else wrist_img
            if wrist_img_np.ndim == 2:
                wrist_img_np = np.stack([wrist_img_np] * 3, axis=-1)
            wrist_img_pil = Image.fromarray(wrist_img_np)

            # Perturbation: apply to BOTH full_image and wrist_image
            if perturb_type == PerturbType.MASK_BG_BLACK:
                if len(action_queue) == 0:
                    img_pil = _get_mask_bg_black(img_pil, raw_task_description, cfg)
                    wrist_img_pil = _get_mask_bg_black(wrist_img_pil, raw_task_description, cfg)
                    last_masked_full = img_pil
                    last_masked_wrist = wrist_img_pil
                else:
                    img_pil = last_masked_full
                    wrist_img_pil = last_masked_wrist
            elif perturb_type != PerturbType.NONE:
                img_pil = _apply_perturbation(img_pil, perturb_type, perturb_variant, rng)
                wrist_img_pil = _apply_perturbation(wrist_img_pil, perturb_type, perturb_variant, rng)
            # Bowl perturbation: 深灰/灰碗 → 亮红 - apply to BOTH images
            if getattr(cfg, "perturb_bowl", False):
                img_pil = _apply_bowl_perturbation(img_pil)
                wrist_img_pil = _apply_bowl_perturbation(wrist_img_pil)

            img_for_replay = np.asarray(img_pil)
            replay_images.append(img_for_replay)
            wrist_img_perturbed = np.asarray(wrist_img_pil)

            # Only run mask when we need to query policy (every num_open_loop_steps); reuse last_masked for video to avoid slow subprocess every step
            if cfg.use_mask_for_policy:
                if len(action_queue) == 0:
                    # Need new actions: run mask (slow), use for policy and video
                    if cfg.use_mask_from_env:
                        from experiments.robot.libero.libero_utils import mask_image_from_libero_seg
                        seg_key = "agentview_segmentation_instance"
                        if seg_key in obs:
                            try:
                                masked = mask_image_from_libero_seg(
                                    img_for_replay, obs[seg_key], env, alpha=0.5
                                )
                            except (TypeError, AttributeError):
                                masked = img_for_replay
                        else:
                            masked = img_for_replay
                    else:
                        masker = _get_mask_processor_masker(cfg)
                        img_pil_rgb = Image.fromarray(img_for_replay).convert("RGB")
                        if masker is not None:
                            masked_pil = masker.mask_image_from_lang(
                                img_pil_rgb,
                                raw_task_description,
                                alpha=0.35,
                            )
                            masked = np.asarray(masked_pil)
                        else:
                            masked_pil = _mask_via_subprocess_cpu(
                                img_pil_rgb, raw_task_description
                            )
                            masked = np.asarray(masked_pil)
                    last_masked = np.asarray(masked)
                    replay_masked_images.append(last_masked.copy())
                    observation["full_image"] = resize_image_for_policy(masked, resize_size)
                    observation["wrist_image"] = resize_image_for_policy(wrist_img_perturbed, resize_size)
                else:
                    replay_masked_images.append(last_masked.copy())
            else:
                replay_masked_images.append(img_for_replay.copy())
                observation["full_image"] = resize_image_for_policy(img_for_replay, resize_size)
                observation["wrist_image"] = resize_image_for_policy(wrist_img_perturbed, resize_size)

            # Query policy only when action queue is empty
            if len(action_queue) == 0:
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)

            action = action_queue.popleft()
            action = _process_action(action, cfg.model_family)
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                # Record final frame so video has at least 2 frames (often only 1 step runs)
                observation_fin, img_fin, _ = _prepare_observation(obs, resize_size)
                img_fin_np = np.asarray(img_fin) if not isinstance(img_fin, np.ndarray) else img_fin
                if img_fin_np.ndim == 2:
                    img_fin_np = np.stack([img_fin_np] * 3, axis=-1)
                img_fin_pil = Image.fromarray(img_fin_np)
                if perturb_type == PerturbType.MASK_BG_BLACK:
                    img_fin_pil = _get_mask_bg_black(img_fin_pil, raw_task_description, cfg)
                elif perturb_type != PerturbType.NONE:
                    img_fin_pil = _apply_perturbation(img_fin_pil, perturb_type, perturb_variant, rng)
                if getattr(cfg, "perturb_bowl", False):
                    img_fin_pil = _apply_bowl_perturbation(img_fin_pil)
                img_fin_replay = np.asarray(img_fin_pil)
                replay_images.append(img_fin_replay)
                if cfg.use_mask_for_policy and last_masked is not None:
                    replay_masked_images.append(last_masked.copy())
                else:
                    replay_masked_images.append(img_fin_replay.copy())
                break
            t += 1

    except Exception as e:
        logger.exception("Episode error: %s", e)

    return success, replay_images, replay_masked_images


def run_background_perturb_eval(cfg: BackgroundPerturbConfig) -> float:
    _resolve_base_vla_path(cfg)
    _validate_config(cfg)

    if getattr(cfg, "load_in_8bit", False) or getattr(cfg, "load_in_4bit", False):
        from accelerate import big_modeling as _acc_bm
        import transformers.modeling_utils as _tf_mu
        _orig = _acc_bm.dispatch_model
        def _patched(m, *a, force_hooks=False, **kw):
            return _orig(m, *a, force_hooks=True, **kw)
        _acc_bm.dispatch_model = _tf_mu.dispatch_model = _patched

    set_seed_everywhere(cfg.seed)
    model, action_head, proprio_projector, noisy_action_projector, processor = _initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)

    run_id = f"BG-PERTURB-{TASK_SUITE_NAME}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    log_path = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(log_path, "w")
    logger.info("Logging to %s", log_path)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    task_ids_to_run = []
    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        desc = (task.language or "").strip().lower()
        if desc in [t.strip().lower() for t in GENERALIZATION_TASKS]:
            task_ids_to_run.append((task_id, task.language))

    if not task_ids_to_run:
        log_file.write("No matching tasks found. Check GENERALIZATION_TASKS.\n")
        log_file.close()
        return 0.0

    num_trials = cfg.num_trials_per_task
    log_file.write(f"Running {len(task_ids_to_run)} tasks: {[d for _, d in task_ids_to_run]}\n")
    sched_desc = "baseline + " if cfg.run_baseline else ""
    log_file.write(f"Perturbations: {sched_desc}background(3); {num_trials} per (task, perturb)\n")
    log_file.flush()

    # Baseline (optional) + periphery solid (3 warm/cool variants)
    schedule = []
    if cfg.run_baseline:
        schedule.append((PerturbType.NONE, 0))
    for v in range(3):
        schedule.append((PerturbType.BACKGROUND, v))

    total_episodes = 0
    total_successes = 0
    rng = np.random.default_rng(cfg.seed)
    task_short = _task_short_name(task_ids_to_run[0][1])  # will overwrite per task

    for task_id, task_description in task_ids_to_run:
        task_short = _task_short_name(task_description)
        task = task_suite.get_task(task_id)
        env, _ = get_libero_env(
            task, cfg.model_family, resolution=cfg.env_img_res,
            use_segmentation_env=cfg.use_mask_from_env,
        )
        raw_desc = _task_description_for_policy(task_description, cfg)
        initial_states = task_suite.get_task_init_states(task_id)
        # initial_states can be a numpy array; avoid "truth value of array is ambiguous"
        n_inits = len(initial_states) if initial_states is not None else 0

        for sched_idx, (ptype, pvar) in enumerate(schedule):
            if ptype == PerturbType.NONE:
                label = "baseline"
                perturb_part = "baseline"
            else:
                label = f"{ptype.value}_{pvar}"
                perturb_part = f"{ptype.value}_{pvar}"
            name_part = f"{task_short}_{cfg.model_label}-{perturb_part}"

            for trial in range(num_trials):
                log_file.write(f"\n--- Task: {task_description} | Perturb: {label} | Trial: {trial + 1}/{num_trials} ---\n")
                log_file.flush()

                initial_state = initial_states[trial % n_inits] if n_inits else None
                if initial_state is None and hasattr(task_suite, "get_task_init_states"):
                    inits = task_suite.get_task_init_states(task_id)
                    initial_state = inits[0] if inits is not None and len(inits) > 0 else None

                success, repl_raw, repl_masked = run_episode(
                    cfg, env, raw_desc, model, resize_size,
                    processor, action_head, proprio_projector, noisy_action_projector,
                    initial_state, log_file, ptype, pvar, rng,
                )

                total_episodes += 1
                if success:
                    total_successes += 1

                # Only save side-by-side video (raw left, masked right)
                suffix_side = f"sidebyside_{name_part}"
                if num_trials > 1:
                    suffix_side += f"_trial{trial+1}"

                _save_sidebyside_video(
                    repl_raw, repl_masked, total_episodes, success,
                    task_description, suffix_side, log_file=log_file,
                    model_label=cfg.model_label,
                )

                log_file.write(f"Success: {success} | Total: {total_successes}/{total_episodes}\n")
                log_file.flush()

    rate = total_successes / total_episodes if total_episodes > 0 else 0.0
    log_file.write(f"\n=== Final: {total_successes}/{total_episodes} = {rate:.2%} ===\n")
    log_file.close()
    return rate


if __name__ == "__main__":
    @draccus.wrap()
    def main(cfg: BackgroundPerturbConfig):
        run_background_perturb_eval(cfg)

    main()
