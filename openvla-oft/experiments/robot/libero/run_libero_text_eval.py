"""
run_libero_text_eval.py

Evaluate on two LIBERO goal tasks with *rephrased* language (no mask):
- "push the plate to the front of the stove" -> "push the right flat-shaped object to the front of the leftmost flat-shaped object"
- "put the bowl on the plate" -> "put the bowl on the right flat-shaped object"

Runs on OpenVLA 7B and/or OFT goal; video naming includes model (openvla_7b / openvla_oft_goal) and full task text.
Run from openvla-oft root. Use run_all_text_eval.sh to test both models.
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
import draccus
import tqdm
from libero.libero import benchmark

import wandb

sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


TASK_SUITE_NAME = "libero_goal"
TASK_MAX_STEPS = 300  # libero_goal

# Original task language (from env) -> new language for policy (rephrased, no mask)
# Value can be str or list of str; if list, the task is run once per variant.
TASK_LANGUAGE_MAP = {
    "push the plate to the front of the stove": "push the right flat-shaped object to the front of the leftmost flat-shaped object",
    "put the bowl on the plate": "put the bowl on the right flat-shaped object",
}

# Push task only: two separate text variants for ablation (use with --task_subset push --use_push_stove_plate_variants).
# 1) stove -> a flat cuboid;  2) plate -> plate on the right.
TASK_LANGUAGE_MAP_PUSH_STOVE_PLATE = {
    "push the plate to the front of the stove": [
        "push the plate to the front of a flat cuboid",
        "push the plate on the right to the front of the stove",
    ],
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 8
    lora_rank: int = 32
    unnorm_key: Union[str, Path] = ""
    load_in_8bit: bool = True   # Default True to avoid OOM on 14GB GPU; set False for bf16
    load_in_4bit: bool = False
    base_vla_path: Optional[str] = None

    task_suite_name: str = TASK_SUITE_NAME
    num_steps_wait: int = 10
    num_trials_per_task: int = 10
    initial_states_path: str = "DEFAULT"
    env_img_res: int = 256

    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"
    seed: int = 7

    # For video naming: e.g. "openvla_7b", "openvla_oft_goal"
    model_label: str = "openvla_oft_goal"

    # Restrict to tasks whose raw description contains this substring (e.g. "push", "put"). None = all mapped tasks.
    task_subset: Optional[str] = None
    # If True, for push task use TASK_LANGUAGE_MAP_PUSH_STOVE_PLATE (stove->flat cuboid, plate->plate on the right).
    use_push_stove_plate_variants: bool = False
    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint, "pretrained_checkpoint must be set!"
    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expect center_crop=True when checkpoint was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"


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
        check_unnorm_key(cfg, model)
    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    unnorm_key = cfg.task_suite_name
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA norm_stats!"
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    run_id = f"TEXT-EVAL-{cfg.task_suite_name}-{cfg.model_label}-{DATE_TIME}"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info("Logging to %s", local_log_filepath)
    if cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)
    return log_file, local_log_filepath, run_id


def log_message(msg: str, log_file=None):
    logger.info(msg)
    if log_file:
        log_file.write(msg + "\n")
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
    return observation, img


def process_action(action, model_family):
    action = normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = invert_gripper_action(action)
    return action


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
):
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(
            f"WARNING: num_open_loop_steps ({cfg.num_open_loop_steps}) != NUM_ACTIONS_CHUNK ({NUM_ACTIONS_CHUNK})"
        )
    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS
    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)
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
            action = process_action(action, cfg.model_family)
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1
    except Exception as e:
        log_message(f"Episode error: {e}", log_file)
    return success, replay_images


def _task_to_slug(text: str, max_len: int = 120) -> str:
    """Full task description for filename (no truncation up to max_len)."""
    s = text.lower().replace(" ", "_").replace("\n", "_").replace(".", "_").replace(",", "_")
    return s[:max_len] if len(s) > max_len else s


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    raw_task_description: str,
    task_description: str,
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
    env, _ = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res, use_segmentation_env=False)

    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask (policy text): {task_description}", log_file)

        if cfg.initial_states_path == "DEFAULT":
            initial_state = initial_states[episode_idx]
        else:
            initial_states_task_key = raw_task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} (failed expert demo)", log_file)
                continue
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)
        success, replay_images = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
        )

        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Full video naming: model_label + task type (push/put) + full task text
        suffix = _task_to_slug(task_description)
        save_rollout_video(
            replay_images,
            total_episodes,
            success=success,
            task_description=task_description,
            log_file=log_file,
            suffix=suffix,
            model_label=cfg.model_label,
        )

        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)
    if cfg.use_wandb:
        wandb.log(
            {f"success_rate/{task_description}": task_success_rate, f"num_episodes/{task_description}": task_episodes}
        )
    # Close env and free GPU/GL so next task can create a new env without framebuffer/OOM issues
    try:
        env.close()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return total_episodes, total_successes


@draccus.wrap()
def eval_libero_text(cfg: GenerateConfig) -> float:
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    # Choose map: push-only variants (stove/plate rewrites) or default
    lang_map = TASK_LANGUAGE_MAP_PUSH_STOVE_PLATE if cfg.use_push_stove_plate_variants else TASK_LANGUAGE_MAP
    task_subset_lower = (cfg.task_subset or "").strip().lower()

    task_ids_to_run: List[tuple] = []
    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        raw_desc = (task.language or "").strip()
        raw_lower = raw_desc.lower()
        if cfg.task_subset and task_subset_lower and task_subset_lower not in raw_lower:
            continue
        new_desc_or_list = next((v for k, v in lang_map.items() if k.lower() == raw_lower), None)
        if new_desc_or_list is None:
            continue
        new_desc_list: List[str] = [new_desc_or_list] if isinstance(new_desc_or_list, str) else list(new_desc_or_list)
        for new_desc in new_desc_list:
            task_ids_to_run.append((task_id, raw_desc, new_desc))

    if not task_ids_to_run:
        log_message("No matching tasks found. Check TASK_LANGUAGE_MAP.", log_file)
        log_file.close()
        return 0.0

    log_message(f"Running {len(task_ids_to_run)} tasks with rephrased language (no mask): {[t[2] for t in task_ids_to_run]}", log_file)
    log_message(f"Model label for video naming: {cfg.model_label}", log_file)

    total_episodes, total_successes = 0, 0
    for task_id, raw_task_description, task_description in task_ids_to_run:
        total_episodes, total_successes = run_task(
            cfg,
            task_suite,
            task_id,
            raw_task_description,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            log_file,
        )

    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)
    if cfg.use_wandb:
        wandb.log({"success_rate/total": final_success_rate, "num_episodes/total": total_episodes})
        wandb.save(local_log_filepath)
    log_file.close()
    return final_success_rate


if __name__ == "__main__":
    eval_libero_text()
