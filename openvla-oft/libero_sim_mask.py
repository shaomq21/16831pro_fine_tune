"""
Ground-truth red/green masks from LIBERO SegmentationRenderEnv.

Uses BDDL ``obj_of_interest`` (source=red, dest=green) and MuJoCo instance
segmentation — no SAM / click points needed.

Requires: LIBERO + robosuite + MUJOCO_GL=egl (headless). Use conda env ``subopt``
or any env with ``pip install -e LIBERO`` and working MuJoCo EGL.

Usage (eval / debug):
    masker = LiberoSimMasker("libero_spatial")
    masked_pil, meta = masker.mask_at_init(task_id=0, init_idx=0)

Offline RLDS replay (if demo HDF5 available):
    masks = masker.replay_hdf5_demo(hdf5_path, task_suite="libero_spatial")
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw

# Headless render default (override before import if needed)
os.environ.setdefault("MUJOCO_GL", "egl")


@dataclass
class SimMaskMeta:
    task_name: str
    language: str
    red_object: Optional[str]
    green_object: Optional[str]
    suite: str
    init_idx: int = 0
    step_idx: int = 0
    gripper_pixels: Optional[List[Tuple[int, int]]] = None


def _squeeze_seg(seg: np.ndarray) -> np.ndarray:
    seg = np.asarray(seg)
    if seg.ndim == 3 and seg.shape[-1] == 1:
        seg = seg[..., 0]
    return seg


def flip_agentview(img: np.ndarray) -> np.ndarray:
    """Match OpenVLA / RLDS preprocessing (180° rotation)."""
    return img[::-1, ::-1].copy()


def instance_masks_from_seg(env, seg: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-object bool masks (H,W) in camera frame."""
    seg = _squeeze_seg(seg)
    out: Dict[str, np.ndarray] = {}
    mapping = getattr(env, "segmentation_id_mapping", {}) or {}
    for seg_id, name in mapping.items():
        out[name] = np.squeeze(seg == (seg_id + 1))
    return out


FINGER_BODY_NAMES = ("gripper0_finger_joint1_tip", "gripper0_finger_joint2_tip")


def draw_white_dots(
    img_arr: np.ndarray,
    centers_xy: List[Tuple[int, int]],
    radius: int = 4,
) -> np.ndarray:
    if not centers_xy:
        return img_arr
    img_pil = Image.fromarray(img_arr, mode="RGB")
    draw = ImageDraw.Draw(img_pil)
    for cx, cy in centers_xy:
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill=(255, 255, 255),
            outline=(255, 255, 255),
        )
    return np.array(img_pil)


def gripper_finger_pixels(
    sim,
    camera: str = "agentview",
    height: int = 256,
    width: int = 256,
) -> List[Tuple[int, int]]:
    """Project both gripper finger tips to RLDS-oriented agentview pixels (x, y)."""
    from robosuite.utils import camera_utils as cu

    T = cu.get_camera_transform_matrix(sim, camera, height, width)
    pts: List[Tuple[int, int]] = []
    for bname in FINGER_BODY_NAMES:
        try:
            bid = sim.model.body_name2id(bname)
        except Exception:
            continue
        pos = sim.data.body_xpos[bid]
        # robosuite returns (row, col) in raw agentview render space.
        # flip_agentview mirrors columns; rows stay aligned with projection row index.
        row, col = cu.project_points_from_world_to_camera(pos[None], T, height, width)[0]
        x = width - 1 - int(round(col))
        y = int(round(row))
        if 0 <= x < width and 0 <= y < height:
            pts.append((x, y))
    return pts


_CABINET_REGION_SUFFIXES = (
    "_top_region",
    "_middle_region",
    "_bottom_region",
    "_top_side",
)


def cabinet_region_band(obj_name: str) -> Optional[str]:
    """Return drawer band: top | middle | bottom | top_side."""
    if not obj_name:
        return None
    if obj_name.endswith("_top_side"):
        return "top_side"
    if obj_name.endswith("_top_region"):
        return "top"
    if obj_name.endswith("_middle_region"):
        return "middle"
    if obj_name.endswith("_bottom_region"):
        return "bottom"
    return None


def cabinet_base_name(obj_name: str) -> str:
    """Strip cabinet drawer / stove region suffix for instance-seg lookup."""
    base = obj_name
    for suf in _CABINET_REGION_SUFFIXES + ("_cook_region",):
        if base.endswith(suf):
            return base[: -len(suf)]
    return base


def refine_cabinet_drawer_mask(full_mask: np.ndarray, band: str) -> np.ndarray:
    """
    Instance seg merges all drawers into one cabinet blob; split vertically
    into top / middle / bottom bands (agentview, RLDS-flipped coords).
    """
    full_mask = np.asarray(full_mask, dtype=bool)
    if not full_mask.any() or band in (None, "top_side"):
        return full_mask
    ys = np.where(full_mask.any(axis=1))[0]
    if len(ys) == 0:
        return full_mask
    y0, y1 = int(ys[0]), int(ys[-1])
    h = y1 - y0 + 1
    edges = [y0, y0 + h // 3, y0 + (2 * h) // 3, y1 + 1]
    bands = {"top": (edges[0], edges[1]), "middle": (edges[1], edges[2]), "bottom": (edges[2], edges[3])}
    lo, hi = bands.get(band, (y0, y1 + 1))
    out = np.zeros_like(full_mask)
    out[lo:hi] = full_mask[lo:hi]
    return out


def _parsed_problem(env) -> dict:
    inner = getattr(env, "env", env)
    return getattr(inner, "parsed_problem", {}) or {}


def _goal_cabinet_regions(env) -> Dict[str, str]:
    """Map cabinet base instance -> goal region name (e.g. ..._bottom_region)."""
    regions: Dict[str, str] = {}
    for state in _parsed_problem(env).get("goal_state", []):
        if not state:
            continue
        pred = str(state[0]).lower()
        for token in state[1:]:
            if not isinstance(token, str) or cabinet_region_band(token) is None:
                continue
            base = cabinet_base_name(token)
            if pred in ("open", "close", "in", "on"):
                regions[base] = token
    return regions


def _region_from_language(lang: str) -> Optional[str]:
    lang = (lang or "").lower()
    if "bottom drawer" in lang or "bottom layer" in lang:
        return "bottom"
    if "middle drawer" in lang or "middle layer" in lang:
        return "middle"
    if "top drawer" in lang or "top layer" in lang:
        return "top"
    return None


def _apply_cabinet_region_hint(
    obj_name: Optional[str],
    *,
    goal_regions: Dict[str, str],
    lang: str,
) -> Optional[str]:
    if not obj_name:
        return obj_name
    if cabinet_region_band(obj_name):
        return obj_name
    base = cabinet_base_name(obj_name)
    if base in goal_regions:
        return goal_regions[base]
    band = _region_from_language(lang)
    if band and (base.endswith("_cabinet_1") or "cabinet" in base):
        return f"{base}_{band}_region"
    return obj_name


def resolve_seg_instance(obj_name: str, available: Dict[str, np.ndarray]) -> Optional[str]:
    """Map BDDL obj_of_interest name to an instance-segmentation key."""
    if obj_name in available:
        return obj_name
    keys = list(available.keys())
    if "stove_front" in obj_name or obj_name.endswith("stove_front_region"):
        for k in keys:
            if "stove" in k.lower():
                return k
    base = cabinet_base_name(obj_name)
    if base in available:
        return base
    for k in sorted(keys, key=len, reverse=True):
        if obj_name.startswith(k + "_") or obj_name == k:
            return k
    return None


def _instance_mask(
    obj_name: Optional[str],
    masks: Dict[str, np.ndarray],
) -> np.ndarray:
    if not obj_name:
        return None
    seg_key = resolve_seg_instance(obj_name, masks)
    if not seg_key or seg_key not in masks:
        raise KeyError(f"Object {obj_name!r} not in segmentation (have {list(masks)})")
    m = flip_agentview(np.squeeze(masks[seg_key]))
    band = cabinet_region_band(obj_name)
    if band and band != "top_side":
        m = refine_cabinet_drawer_mask(m, band)
    return m


def _interest_pair(env) -> Tuple[Optional[str], Optional[str]]:
    """Return (red_instance, green_instance) from BDDL obj_of_interest + goal regions."""
    objs = list(getattr(env, "obj_of_interest", []))
    lang = ""
    if hasattr(env, "language_instruction"):
        lang = str(env.language_instruction or "").lower()
    elif hasattr(env, "get_language_instruction"):
        lang = str(env.get_language_instruction() or "").lower()

    goal_regions = _goal_cabinet_regions(env)
    hint = lambda name: _apply_cabinet_region_hint(name, goal_regions=goal_regions, lang=lang)

    if len(objs) >= 2:
        return hint(objs[0]), hint(objs[1])
    if len(objs) == 1:
        if lang.startswith("open ") or lang.startswith("turn on ") or lang.startswith("close "):
            return None, hint(objs[0])
        return hint(objs[0]), None
    return None, None


def compose_black_bg_mask(
    rgb: np.ndarray,
    red_mask: np.ndarray,
    green_mask: np.ndarray,
    *,
    alpha: float = 0.35,
    draw_green: bool = True,
) -> np.ndarray:
    """Same visual style as ``GroundedSAMMasker.mask_image_from_lang``."""
    out = np.zeros_like(rgb, dtype=np.uint8)
    light_red = np.array([255, 120, 120], dtype=np.float32)
    light_green = np.array([120, 255, 120], dtype=np.float32)
    red_mask = np.asarray(red_mask, dtype=bool) & (~np.asarray(green_mask, dtype=bool))
    if red_mask.any():
        tinted = (1.0 - alpha) * rgb[red_mask] + alpha * light_red
        out[red_mask] = np.clip(tinted, 0, 255).astype(np.uint8)
    if draw_green and np.asarray(green_mask, dtype=bool).any():
        tinted = (1.0 - alpha) * rgb[green_mask] + alpha * light_green
        out[green_mask] = np.clip(tinted, 0, 255).astype(np.uint8)
    return out


def mask_rgb_from_obs(
    rgb_camera: np.ndarray,
    seg_camera: np.ndarray,
    env,
    *,
    alpha: float = 0.35,
    draw_green: bool = True,
    draw_gripper: bool = True,
    sim=None,
) -> Tuple[np.ndarray, Optional[str], Optional[str]]:
    """
    Build masked RGB (flipped, RLDS orientation) from sim obs.

    Returns (masked_rgb, red_obj_name, green_obj_name).
    """
    rgb = flip_agentview(rgb_camera)
    masks = instance_masks_from_seg(env, seg_camera)
    red_name, green_name = _interest_pair(env)

    red_m = np.zeros(rgb.shape[:2], dtype=bool)
    green_m = np.zeros(rgb.shape[:2], dtype=bool)

    if red_name:
        red_m = _instance_mask(red_name, masks)
    if green_name:
        green_m = _instance_mask(green_name, masks)

    masked = compose_black_bg_mask(rgb, red_m, green_m, alpha=alpha, draw_green=draw_green)
    if draw_gripper:
        sim_obj = sim if sim is not None else getattr(env, "sim", None)
        if sim_obj is not None:
            H, W = rgb.shape[:2]
            dots = gripper_finger_pixels(sim_obj, height=H, width=W)
            masked = draw_white_dots(masked, dots, radius=max(2, min(5, min(H, W) // 70)))
    return masked, red_name, green_name


def lang_to_task_name(lang: str) -> str:
    return re.sub(r"\s+", "_", lang.strip().lower())


def resolve_suite_for_lang(lang: str, suite_hint: Optional[str] = None) -> str:
    if suite_hint:
        return suite_hint
    name = lang_to_task_name(lang)
    repo = Path(__file__).resolve().parents[1] / "LIBERO/libero/libero/bddl_files"
    if not repo.exists():
        repo = Path(__file__).resolve().parents[2] / "LIBERO/libero/libero/bddl_files"
    for suite_dir in sorted(repo.iterdir()):
        if suite_dir.is_dir() and (suite_dir / f"{name}.bddl").exists():
            return suite_dir.name
    raise FileNotFoundError(f"No BDDL for language: {lang!r}")


class LiberoSimMasker:
    """Lazy LIBERO env cache; one SegmentationRenderEnv per task."""

    def __init__(self, suite: str, resolution: int = 256):
        self.suite = suite
        self.resolution = resolution
        self._bench = None
        self._envs: Dict[int, object] = {}
        self._task_name_to_id: Dict[str, int] = {}

    def _ensure_libero(self):
        if self._bench is not None:
            return
        from libero.libero.benchmark import get_benchmark

        self._bench = get_benchmark(self.suite)()
        for i, task in enumerate(self._bench.tasks):
            self._task_name_to_id[task.name] = i

    def task_id_from_lang(self, lang: str) -> int:
        self._ensure_libero()
        key = lang_to_task_name(lang)
        if key not in self._task_name_to_id:
            raise KeyError(f"Task {key!r} not in suite {self.suite}")
        return self._task_name_to_id[key]

    def _load_init_states(self, task_id: int):
        import os
        import torch
        from libero.libero import get_libero_path

        task = self._bench.get_task(task_id)
        init_path = os.path.join(
            get_libero_path("init_states"),
            task.problem_folder,
            task.init_states_file,
        )
        return torch.load(init_path, weights_only=False)

    def _get_env(self, task_id: int):
        if task_id in self._envs:
            return self._envs[task_id]
        from libero.libero.envs import SegmentationRenderEnv

        bddl_path = self._bench.get_task_bddl_file_path(task_id)
        env = SegmentationRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=self.resolution,
            camera_widths=self.resolution,
            camera_segmentations="instance",
        )
        env.reset()
        self._envs[task_id] = env
        return env

    def mask_at_init(
        self,
        *,
        task_id: Optional[int] = None,
        lang: Optional[str] = None,
        init_idx: int = 0,
        alpha: float = 0.35,
        return_raw: bool = False,
    ) -> Union[Image.Image, Tuple[Image.Image, SimMaskMeta], Tuple[Image.Image, Image.Image, SimMaskMeta]]:
        self._ensure_libero()
        if task_id is None:
            if lang is None:
                raise ValueError("Need task_id or lang")
            task_id = self.task_id_from_lang(lang)
        task = self._bench.get_task(task_id)
        env = self._get_env(task_id)
        init_states = self._load_init_states(task_id)
        if init_idx < 0 or init_idx >= len(init_states):
            raise IndexError(f"init_idx {init_idx} out of range [0, {len(init_states)})")
        obs = env.set_init_state(init_states[init_idx])
        masked, red_name, green_name = mask_rgb_from_obs(
            obs["agentview_image"],
            obs["agentview_segmentation_instance"],
            env,
            alpha=alpha,
            sim=env.sim,
        )
        meta = SimMaskMeta(
            task_name=task.name,
            language=task.language,
            red_object=red_name,
            green_object=green_name,
            suite=self.suite,
            init_idx=init_idx,
            gripper_pixels=gripper_finger_pixels(env.sim, height=self.resolution, width=self.resolution),
        )
        masked_pil = Image.fromarray(masked)
        if not return_raw:
            return masked_pil, meta
        raw_pil = Image.fromarray(flip_agentview(obs["agentview_image"]))
        return masked_pil, raw_pil, meta

    def mask_from_live_obs(
        self,
        task_id: int,
        obs: dict,
        *,
        alpha: float = 0.35,
    ) -> Image.Image:
        """Use during eval when SegmentationRenderEnv is already stepping."""
        env = self._get_env(task_id)
        masked, _, _ = mask_rgb_from_obs(
            obs["agentview_image"],
            obs["agentview_segmentation_instance"],
            env,
            alpha=alpha,
        )
        return Image.fromarray(masked)

    def obs_at_rlds_step(
        self,
        lang: str,
        actions: np.ndarray,
        step_idx: int,
        *,
        init_idx: int = 0,
        num_steps_wait: int = 10,
    ):
        """Replay sim to the observation aligned with RLDS ``steps[step_idx]``."""
        self._ensure_libero()
        task_id = self.task_id_from_lang(lang)
        env = self._get_env(task_id)
        init_states = self._load_init_states(task_id)
        obs = env.set_init_state(init_states[init_idx % len(init_states)])
        dummy = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        for t in range(max(0, step_idx)):
            act = dummy if t < num_steps_wait else actions[t - num_steps_wait]
            obs, _, _, _ = env.step(act)
        return obs, env

    def mask_at_rlds_step(
        self,
        lang: str,
        actions: np.ndarray,
        step_idx: int,
        *,
        init_idx: int = 0,
        num_steps_wait: int = 10,
        alpha: float = 0.35,
    ) -> Tuple[Image.Image, Image.Image, SimMaskMeta]:
        """Masked + raw PIL at RLDS step index (sim replay)."""
        self._ensure_libero()
        task_id = self.task_id_from_lang(lang)
        task = self._bench.get_task(task_id)
        obs, env = self.obs_at_rlds_step(
            lang, actions, step_idx, init_idx=init_idx, num_steps_wait=num_steps_wait
        )
        masked, red_name, green_name = mask_rgb_from_obs(
            obs["agentview_image"],
            obs["agentview_segmentation_instance"],
            env,
            alpha=alpha,
            sim=env.sim,
        )
        meta = SimMaskMeta(
            task_name=task.name,
            language=task.language,
            red_object=red_name,
            green_object=green_name,
            suite=self.suite,
            init_idx=init_idx,
            step_idx=step_idx,
            gripper_pixels=gripper_finger_pixels(env.sim, height=self.resolution, width=self.resolution),
        )
        raw = Image.fromarray(flip_agentview(obs["agentview_image"]))
        return Image.fromarray(masked), raw, meta

    def replay_rlds_episode(
        self,
        lang: str,
        actions: np.ndarray,
        init_idx: int = 0,
        num_steps_wait: int = 10,
        alpha: float = 0.35,
    ) -> List[Image.Image]:
        """
        Replay RLDS action sequence in sim and return masked frames.

        Note: RLDS does not store demo index; default init_idx=0. For exact
        replay, use ``replay_hdf5_demo`` with the source HDF5 file.
        """
        self._ensure_libero()
        task_id = self.task_id_from_lang(lang)
        env = self._get_env(task_id)
        init_states = self._load_init_states(task_id)
        obs = env.set_init_state(init_states[init_idx % len(init_states)])

        dummy = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        frames: List[Image.Image] = []
        t = 0
        for action in actions:
            if t < num_steps_wait:
                obs, _, _, _ = env.step(dummy)
            else:
                obs, _, _, _ = env.step(np.asarray(action, dtype=np.float64))
            masked, _, _ = mask_rgb_from_obs(
                obs["agentview_image"],
                obs["agentview_segmentation_instance"],
                env,
                alpha=alpha,
            )
            frames.append(Image.fromarray(masked))
            t += 1
        return frames

    def replay_hdf5_demo(
        self,
        hdf5_path: str,
        *,
        demo_key: str = "demo_0",
        alpha: float = 0.35,
        cap_index: int = 0,
    ) -> List[Image.Image]:
        """
        Perfect replay using full MuJoCo states stored in LIBERO demo HDF5.
        """
        import h5py

        self._ensure_libero()
        frames: List[Image.Image] = []
        with h5py.File(hdf5_path, "r") as f:
            # infer task from filename
            stem = Path(hdf5_path).stem.replace("_demo", "")
            if stem not in self._task_name_to_id:
                raise KeyError(f"Cannot map HDF5 {hdf5_path} to suite {self.suite}")
            task_id = self._task_name_to_id[stem]
            env = self._get_env(task_id)
            states = f[f"data/{demo_key}/states"][()]
            actions = f[f"data/{demo_key}/actions"][()]
            model_xml = f[f"data/{demo_key}"].attrs.get("model_file", b"")
            if model_xml:
                try:
                    from libero.libero import envs as _  # noqa: F401
                    import libero.libero.utils.utils as libero_utils

                    xml = libero_utils.postprocess_model_xml(
                        model_xml.decode() if isinstance(model_xml, bytes) else model_xml,
                        {},
                    )
                    env.reset_from_xml_string(xml)
                    env.sim.reset()
                except Exception:
                    pass
            env.sim.set_state_from_flattened(states[0])
            env.sim.forward()
            env._update_observables(force=True)
            obs = env.regenerate_obs_from_state(states[0])
            for j, action in enumerate(actions):
                if j < cap_index:
                    if j + 1 < len(states):
                        obs = env.regenerate_obs_from_state(states[j + 1])
                    continue
                masked, _, _ = mask_rgb_from_obs(
                    obs["agentview_image"],
                    obs["agentview_segmentation_instance"],
                    env,
                    alpha=alpha,
                )
                frames.append(Image.fromarray(masked))
                if j + 1 < len(states):
                    obs = env.regenerate_obs_from_state(states[j + 1])
                else:
                    obs, _, _, _ = env.step(np.asarray(action, dtype=np.float64))
        return frames

    def close(self):
        for env in self._envs.values():
            try:
                env.close()
            except Exception:
                pass
        self._envs.clear()
