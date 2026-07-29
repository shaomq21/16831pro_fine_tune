"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os

import imageio
import numpy as np
import tensorflow as tf
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SegmentationRenderEnv

from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)


def get_libero_env(task, model_family, resolution=256, use_segmentation_env=False, bddl_file_override=None):
    """Initializes and returns the LIBERO environment, along with the task description.

    Args:
        task: LIBERO task (has .language, .problem_folder, .bddl_file).
        model_family: Model family string (e.g. "openvla").
        resolution: Camera height/width.
        use_segmentation_env: If True, use SegmentationRenderEnv so obs include
            agentview_segmentation_instance; enables mask-from-env (no Grounded-SAM).
        bddl_file_override: If set, use this BDDL filename instead of task.bddl_file
            (still under task.problem_folder). E.g. "put_the_bowl_on_the_plate_no_plate.bddl".
    """
    task_description = task.language
    bddl_file = (bddl_file_override or task.bddl_file).strip() if bddl_file_override else task.bddl_file
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    if use_segmentation_env:
        env = SegmentationRenderEnv(camera_segmentations="instance", **env_args)
    else:
        env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs):
    """Extracts third-person image from observations and preprocesses it."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_libero_wrist_image(obs):
    """Extracts wrist camera image from observations and preprocesses it."""
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def mask_image_from_libero_seg(rgb_np, seg_obs, env, alpha=0.35):
    """Black-bg red/green mask matching training (libero_sim_mask / dual_masked RLDS).

    ``rgb_np`` must already be in RLDS/OpenVLA orientation (``get_libero_image`` flip).
    Maps BDDL ``obj_of_interest`` (incl. goal regions like stove_front_region) onto
    instance-seg keys via ``resolve_seg_instance``.
    """
    from libero_sim_mask import (
        _interest_pair,
        _instance_mask,
        compose_black_bg_mask,
        draw_white_dots,
        gripper_finger_pixels,
        instance_masks_from_seg,
    )

    rgb = np.asarray(rgb_np, dtype=np.uint8)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    h, w = rgb.shape[:2]
    red_m = np.zeros((h, w), dtype=bool)
    green_m = np.zeros((h, w), dtype=bool)

    masks = instance_masks_from_seg(env, seg_obs)
    red_name, green_name = _interest_pair(env)
    if red_name:
        try:
            red_m = np.asarray(_instance_mask(red_name, masks), dtype=bool)
        except KeyError:
            red_m = np.zeros((h, w), dtype=bool)
    if green_name:
        try:
            green_m = np.asarray(_instance_mask(green_name, masks), dtype=bool)
        except KeyError:
            green_m = np.zeros((h, w), dtype=bool)

    if not red_m.any() and not green_m.any():
        # Do not silently return raw — still black bg so video failure is obvious.
        return np.zeros_like(rgb)

    out = compose_black_bg_mask(rgb, red_m, green_m, alpha=float(alpha), draw_green=True)
    sim_obj = getattr(env, "sim", None)
    if sim_obj is not None:
        dots = gripper_finger_pixels(sim_obj, height=h, width=w)
        out = draw_white_dots(out, dots, radius=max(2, min(5, min(h, w) // 70)))
    return out


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None, suffix=None, fps=30, model_label=None, rollout_dir=None, video_basename=None):
    """Saves an MP4 replay of an episode. Same fps for raw and masked so they stay in sync.
    If model_label is provided, it is used in the filename (e.g. openvla_7b, openvla_oft_goal).
    If video_basename is set, saves as {rollout_dir}/{video_basename}.mp4 (suffix ignored)."""
    rollout_dir = rollout_dir or f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    extra = f"--{suffix}" if suffix else ""
    model_tag = (model_label if model_label else "openvla_oft")
    if video_basename:
        mp4_path = os.path.join(rollout_dir, f"{video_basename}.mp4")
    else:
        mp4_path = f"{rollout_dir}/{DATE_TIME}--{model_tag}--episode={idx}--success={success}--task={processed_task_description}{extra}.mp4"
    # Normalize every frame to uint8 HWC so all frames are written (avoid "only first frame" bug)
    frames = []
    for img in rollout_images:
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        frames.append(np.clip(arr.astype(np.uint8), 0, 255))
    if not frames:
        frames = [np.zeros((256, 256, 3), dtype=np.uint8)]
    n_original = len(frames)
    # If only one frame, duplicate so the video is playable (e.g. 3 sec at 30fps)
    if n_original == 1:
        frames = frames * (fps * 3)
    video_writer = imageio.get_writer(mp4_path, fps=fps)
    for f in frames:
        video_writer.append_data(f)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path} ({n_original} frames -> {len(frames)} written)")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
