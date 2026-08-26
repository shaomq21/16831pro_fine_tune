#!/usr/bin/env bash
# Fast color-conditioned RL on continuous dual-masked OFT (unmerged).
# Default recipe (anti-collapse): success-only groups, action_head-only, small lr, rollback.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFT="${REPO_ROOT}/openvla-oft"
cd "${OFT}"

STORAGE_ROOT="${STORAGE_ROOT:-/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/conda_envs/simplevla/bin/python}"
BASE_VLA="${OFT}/checkpoints/openvla-7b"

SUITE="${SUITE:-spatial}"          # spatial | goal | object
GPU="${GPU:-1}"
NUM_ITERS="${NUM_ITERS:-8}"
ROLLOUTS="${ROLLOUTS:-4}"
EVAL_TRIALS="${EVAL_TRIALS:-4}"
NOTE="${NOTE:-$(date +%Y%m%d_%H%M%S)}"

case "${SUITE}" in
  spatial) TASK_SUITE=libero_spatial; CKPT_SUITE=spatial; TASK_ID="${TASK_ID:-2}" ;;
  goal)    TASK_SUITE=libero_goal;    CKPT_SUITE=goal;    TASK_ID="${TASK_ID:-5}" ;;
  object)  TASK_SUITE=libero_object;  CKPT_SUITE=object;  TASK_ID="${TASK_ID:-0}" ;;
  *) echo "Unknown SUITE=${SUITE}"; exit 1 ;;
esac

# Task-dependent default OOD
if [[ -z "${PERTURB_MODE:-}" ]]; then
  case "${TASK_ID}" in
    1|3|4|8) PERTURB_MODE=bowl ;;
    *) PERTURB_MODE=colors ;;
  esac
fi

CKPT_DIR="${CKPT_DIR:-${STORAGE_ROOT}/runs/openvla_adapters/openvla-7b+dual_masked_${CKPT_SUITE}+b4+lr-0.0005+lora-r32+dropout-0.0+lora-attn-only--suite_${CKPT_SUITE}_oft_lr}"
SAVE_DIR="${SAVE_DIR:-${STORAGE_ROOT}/runs/rl_lora_color_quick/${TASK_SUITE}_task${TASK_ID}_${NOTE}}"
LOG_DIR="${OFT}/logs"
mkdir -p "${SAVE_DIR}" "${LOG_DIR}"
LOG="${LOG_DIR}/rl_lora_color_${SUITE}_task${TASK_ID}_${NOTE}.log"

export CUDA_VISIBLE_DEVICES="${GPU}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_ROOT}/LIBERO:${OFT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "===== $(date -Iseconds) RL LoRA color quick =====" | tee "${LOG}"
echo "suite=${TASK_SUITE} task_id=${TASK_ID} gpu=${GPU} ckpt=${CKPT_DIR}" | tee -a "${LOG}"
echo "perturb=${PERTURB_MODE} vision_lora=${TRAIN_VISION_LORA:-False} save=${SAVE_DIR}" | tee -a "${LOG}"

"${PYTHON}" experiments/robot/libero/run_rl_lora_color_quick.py \
  --pretrained_checkpoint "${CKPT_DIR}" \
  --base_vla_path "${BASE_VLA}" \
  --task_suite_name "${TASK_SUITE}" \
  --task_id "${TASK_ID}" \
  --mode train \
  --num_iters "${NUM_ITERS}" \
  --num_groups_per_iter "${NUM_GROUPS:-3}" \
  --group_size "${GROUP_SIZE:-4}" \
  --eval_trials_per_variant "${EVAL_TRIALS}" \
  --skip_baseline_eval "${SKIP_BASELINE_EVAL:-True}" \
  --lora_rank 8 \
  --train_vision_lora "${TRAIN_VISION_LORA:-False}" \
  --train_proprio "${TRAIN_PROPRIO:-True}" \
  --only_success_trajs "${ONLY_SUCCESS_TRAJS:-True}" \
  --prefer_greedy_bc "${PREFER_GREEDY_BC:-True}" \
  --skip_update_above_sr "${SKIP_UPDATE_ABOVE_SR:-0.85}" \
  --min_success_to_update "${MIN_SUCCESS:-1}" \
  --rollback_on_collapse True \
  --action_noise_std "${NOISE:-0.03}" \
  --lr_lora "${LR_LORA:-1e-5}" \
  --lr_action_head "${LR_AH:-1e-5}" \
  --max_update_chunks "${MAX_CHUNKS:-6}" \
  --bc_steps_per_iter "${BC_STEPS:-1}" \
  --greedy_frac "${GREEDY_FRAC:-0.5}" \
  --max_init_pool 12 \
  --num_images_in_input 1 \
  --use_proprio True \
  --use_l1_regression True \
  --use_mask_for_policy True \
  --use_mask_from_env True \
  --mask_alpha 0.35 \
  --center_crop False \
  --perturb_mode "${PERTURB_MODE}" \
  --color_variants "0,1" \
  --save_dir "${SAVE_DIR}" \
  --local_log_dir "${OFT}/experiments/logs/rl_lora_color_quick" \
  2>&1 | tee -a "${LOG}"

echo "===== $(date -Iseconds) DONE rc=${PIPESTATUS[0]} save=${SAVE_DIR} =====" | tee -a "${LOG}"
if [[ -f "${SAVE_DIR}/SUMMARY.json" ]]; then
  echo "SUMMARY:" | tee -a "${LOG}"
  cat "${SAVE_DIR}/SUMMARY.json" | tee -a "${LOG}"
fi
