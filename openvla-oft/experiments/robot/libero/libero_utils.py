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


def get_libero_env(task, model_family, resolution=256, use_segmentation_env=False):
    """Initializes and returns the LIBERO environment, along with the task description.

    Args:
        task: LIBERO task (has .language, .problem_folder, .bddl_file).
        model_family: Model family string (e.g. "openvla").
        resolution: Camera height/width.
        use_segmentation_env: If True, use SegmentationRenderEnv so obs include
            agentview_segmentation_instance; enables mask-from-env (no Grounded-SAM).
    """
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
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


def mask_image_from_libero_seg(rgb_np, seg_obs, env, alpha=0.5):
    """Paint red/green overlay from LIBERO instance segmentation to match Grounded-SAM mask style.

    Args:
        rgb_np: RGB image (H,W,3) in same geometry as get_libero_image(obs), i.e. already flipped.
        seg_obs: Raw segmentation from obs["agentview_segmentation_instance"] (camera frame).
        env: SegmentationRenderEnv (must have get_segmentation_instances, obj_of_interest).
        alpha: Blend strength for overlay (0=no tint, 1=full color).

    Returns:
        RGB image (H,W,3) uint8 with red tint on first obj_of_interest, green on second.
    """
    # get_segmentation_instances expects raw camera-frame seg
    seg_dict = env.get_segmentation_instances(seg_obs.copy())
    obj_of_interest = getattr(env, "obj_of_interest", [])
    if not obj_of_interest:
        return rgb_np

    out = np.array(rgb_np, dtype=np.float64)
    red = np.array([255, 0, 0], dtype=np.float64)
    green = np.array([0, 255, 0], dtype=np.float64)

    # RGB from get_libero_image is flipped; seg_dict masks are in camera frame → flip masks to align
    for i, obj_name in enumerate(obj_of_interest[:2]):
        mask = seg_dict.get(obj_name)
        if mask is None:
            continue
        mask_f = (mask[::-1, ::-1] > 0).astype(np.float64)
        if mask_f.sum() == 0:
            continue
        color = red if i == 0 else green
        for c in range(3):
            out[:, :, c] = (1 - alpha * mask_f) * out[:, :, c] + alpha * mask_f * color[c]

    return np.clip(out, 0, 255).astype(np.uint8)


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None, suffix=None, fps=30):
    """Saves an MP4 replay of an episode. Same fps for raw and masked so they stay in sync."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    extra = f"--{suffix}" if suffix else ""
    mp4_path = f"{rollout_dir}/{DATE_TIME}--openvla_oft--episode={idx}--success={success}--task={processed_task_description}{extra}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=fps)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
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
