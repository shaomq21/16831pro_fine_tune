#!/usr/bin/env bash
# Quick masked LIBERO eval: few tasks, 1 trial each, current suite checkpoint.
set -u

SUITE="${SUITE:?set SUITE=spatial|object|goal}"
MAX_TASKS="${MAX_TASKS:-3}"
TASK_IDS="${TASK_IDS:-}"   # optional override, e.g. TASK_IDS=5 for goal
GPU="${GPU:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFT="${REPO_ROOT}/openvla-oft"
cd "${OFT}"

STORAGE_ROOT="${STORAGE_ROOT:-/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/conda_envs/simplevla/bin/python}"
BASE_VLA="${OFT}/checkpoints/openvla-7b"

case "${SUITE}" in
  spatial) TASK_SUITE=libero_spatial ;;
  object)  TASK_SUITE=libero_object ;;
  goal)    TASK_SUITE=libero_goal ;;
  study_scene4) TASK_SUITE=libero_90 ;;
  *) echo "Unknown SUITE=${SUITE}"; exit 1 ;;
esac

CKPT_DIR="${STORAGE_ROOT}/runs/openvla_adapters/openvla-7b+dual_masked_${SUITE}+b4+lr-0.0005+lora-r32+dropout-0.0+lora-attn-only--suite_${SUITE}_oft_lr"
LOG="${OFT}/logs/eval_quick_${SUITE}.log"

if [[ ! -s "${CKPT_DIR}/action_head--latest_checkpoint.pt" ]]; then
  echo "Missing checkpoint: ${CKPT_DIR}"
  exit 1
fi

# Goal: first 3 tasks are in SKIP list; default to task 5 (push plate)
if [[ "${SUITE}" == "goal" && -z "${TASK_IDS}" ]]; then
  TASK_IDS="5"
  MAX_TASKS=""
fi

EXTRA_TASK_ARGS=()
if [[ -n "${TASK_IDS}" ]]; then
  EXTRA_TASK_ARGS+=(--task_ids "${TASK_IDS}")
else
  EXTRA_TASK_ARGS+=(--max_tasks "${MAX_TASKS}")
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_ROOT}/LIBERO:${OFT}:${PYTHONPATH:-}"

echo "===== $(date -Iseconds) eval suite=${SUITE} ckpt=${CKPT_DIR} gpu=${GPU} =====" | tee "${LOG}"
echo "  task_suite=${TASK_SUITE} trials=1 ${EXTRA_TASK_ARGS[*]}" | tee -a "${LOG}"

"${PYTHON}" experiments/robot/libero/run_libero_eval_mask.py \
  --pretrained_checkpoint "${CKPT_DIR}" \
  --base_vla_path "${BASE_VLA}" \
  --task_suite_name "${TASK_SUITE}" \
  --num_trials_per_task 1 \
  "${EXTRA_TASK_ARGS[@]}" \
  --num_images_in_input 1 \
  --use_proprio True \
  --use_l1_regression True \
  --lora_rank 32 \
  --center_crop False \
  --use_mask_from_env True \
  --mask_alpha 0.35 \
  --local_log_dir "${OFT}/experiments/logs/eval_quick" \
  --run_id_note "suite_${SUITE}_r32_ckpt" \
  2>&1 | tee -a "${LOG}"
