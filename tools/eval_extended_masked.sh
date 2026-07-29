#!/usr/bin/env bash
# Extended masked eval: all tasks in suite, 1 trial each, save one masked MP4 per task.
set -u

SUITE="${SUITE:?set SUITE=spatial|goal}"
GPU="${GPU:-1}"
MAX_TASKS="${MAX_TASKS:-10}"
SKIP_FILTERED="${SKIP_FILTERED:-0}"   # 0 = run all tasks including previously skipped goal tasks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFT="${REPO_ROOT}/openvla-oft"
cd "${OFT}"

STORAGE_ROOT="${STORAGE_ROOT:-/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/conda_envs/simplevla/bin/python}"
BASE_VLA="${OFT}/checkpoints/openvla-7b"

case "${SUITE}" in
  spatial) TASK_SUITE=libero_spatial ;;
  goal)    TASK_SUITE=libero_goal ;;
  *) echo "Unknown SUITE=${SUITE} (use spatial or goal)"; exit 1 ;;
esac

CKPT_DIR="${STORAGE_ROOT}/runs/openvla_adapters/openvla-7b+dual_masked_${SUITE}+b4+lr-0.0005+lora-r32+dropout-0.0+lora-attn-only--suite_${SUITE}_oft_lr"
VIDEO_DIR="${OFT}/experiments/logs/eval_videos/${SUITE}_r32"
LOG="${OFT}/logs/eval_extended_${SUITE}.log"
mkdir -p "${VIDEO_DIR}" "${OFT}/logs"

if [[ ! -s "${CKPT_DIR}/action_head--latest_checkpoint.pt" ]]; then
  echo "Missing checkpoint: ${CKPT_DIR}"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_ROOT}/LIBERO:${OFT}:${PYTHONPATH:-}"

SKIP_FLAG="--skip_filtered_tasks True"
if [[ "${SKIP_FILTERED}" == "0" ]]; then
  SKIP_FLAG="--skip_filtered_tasks False"
fi

echo "===== $(date -Iseconds) extended eval suite=${SUITE} ckpt=${CKPT_DIR} =====" | tee "${LOG}"
echo "  tasks=0..$((MAX_TASKS-1)) trials=1 videos=${VIDEO_DIR} skip_filtered=${SKIP_FILTERED}" | tee -a "${LOG}"

"${PYTHON}" experiments/robot/libero/run_libero_eval_mask.py \
  --pretrained_checkpoint "${CKPT_DIR}" \
  --base_vla_path "${BASE_VLA}" \
  --task_suite_name "${TASK_SUITE}" \
  --num_trials_per_task 1 \
  --max_tasks "${MAX_TASKS}" \
  ${SKIP_FLAG} \
  --save_video_mode masked \
  --rollout_video_dir "${VIDEO_DIR}" \
  --num_images_in_input 1 \
  --use_proprio True \
  --use_l1_regression True \
  --lora_rank 32 \
  --center_crop False \
  --use_mask_from_env True \
  --mask_alpha 0.35 \
  --local_log_dir "${OFT}/experiments/logs/eval_extended" \
  --run_id_note "suite_${SUITE}_r32_extended" \
  2>&1 | tee -a "${LOG}"

echo "===== Videos in ${VIDEO_DIR} =====" | tee -a "${LOG}"
ls -lh "${VIDEO_DIR}"/*.mp4 2>/dev/null | tee -a "${LOG}" || true
