#!/usr/bin/env bash
# Finetune ONE masked LIBERO suite (official OFT lr + lora-r32, single image).
# PHASE=1: lr=5e-4, decay@100k, grad_accum=1, lora attn-only
# PHASE=2: lr=5e-5, second decay@+100k, grad_accum=2
# PHASE=3: merge phase1/2 ckpt, all-linear LoRA, grad_accum=8 (global batch 64 @ 2 GPU)
set -u

SUITE="${SUITE:?set SUITE=spatial|object|goal|study_scene4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || -z "${NUM_GPUS:-}" ]]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/suite_gpu_layout.sh"
fi
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFT="${REPO_ROOT}/openvla-oft"
cd "${OFT}"

STORAGE_ROOT="${STORAGE_ROOT:-/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune}"
ADAPTER_RUN_ROOT="${STORAGE_ROOT}/runs/openvla_adapters"
BASE_VLA_PATH="${OFT}/checkpoints/openvla-7b"

PYTHON="${PYTHON:-${STORAGE_ROOT}/conda_envs/simplevla/bin/python}"
TORCHRUN="${TORCHRUN:-${STORAGE_ROOT}/conda_envs/simplevla/bin/torchrun}"

DATA_ROOT="${OFT}/datasets/dual_masked_libero_rlds"
DATASET_NAME="dual_masked_${SUITE}"

PHASE="${PHASE:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LORA_RANK="${LORA_RANK:-32}"
LR_SCHEDULE_RESET="${LR_SCHEDULE_RESET:-False}"
MAX_STEPS="${MAX_STEPS:-650000}"
SAVE_FREQ="${SAVE_FREQ:-1500}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-10000}"
KEEP_LAST_N="${KEEP_LAST_N:-2}"

LORA_TARGET="${LORA_TARGET:-attn-only}"
USE_MERGED_BASE="${USE_MERGED_BASE:-False}"
if [[ "${PHASE}" == "3" ]]; then
  LEARNING_RATE="${LEARNING_RATE:-5e-4}"
  NUM_STEPS_BEFORE_DECAY="${NUM_STEPS_BEFORE_DECAY:-100000}"
  GRAD_ACCUM="${GRAD_ACCUM:-8}"
  LR_SCHEDULE_RESET="${LR_SCHEDULE_RESET:-True}"
  LORA_TARGET="${LORA_TARGET:-all-linear}"
  RUN_ID_NOTE="${RUN_ID_NOTE:-suite_${SUITE}_oft_lr_p3}"
elif [[ "${PHASE}" == "2" ]]; then
  LEARNING_RATE="${LEARNING_RATE:-5e-5}"
  NUM_STEPS_BEFORE_DECAY="${NUM_STEPS_BEFORE_DECAY:-100000}"
  GRAD_ACCUM="${GRAD_ACCUM:-2}"
  LR_SCHEDULE_RESET="${LR_SCHEDULE_RESET:-True}"
  RUN_ID_NOTE="${RUN_ID_NOTE:-suite_${SUITE}_oft_lr_p2}"
else
  LEARNING_RATE="${LEARNING_RATE:-5e-4}"
  NUM_STEPS_BEFORE_DECAY="${NUM_STEPS_BEFORE_DECAY:-100000}"
  GRAD_ACCUM="${GRAD_ACCUM:-1}"
  RUN_ID_NOTE="${RUN_ID_NOTE:-suite_${SUITE}_oft_lr}"
fi

EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM))
LEARNING_RATE_STR=$("${PYTHON}" -c "print(float('${LEARNING_RATE}'))")

# Resume dirs use phase-1 run_id note (b4+lr-0.0005+...--suite_*_oft_lr)
PHASE1_NOTE="suite_${SUITE}_oft_lr"
PHASE1_PREFIX="openvla-7b+dual_masked_${SUITE}+b4+lr-0.0005+lora-r${LORA_RANK}+dropout-0.0+lora-attn-only--${PHASE1_NOTE}"

find_run_dir () {
  local note="$1"
  local d
  for d in "${ADAPTER_RUN_ROOT}"/openvla-7b+dual_masked_"${SUITE}"+b*+lr-*+lora-r"${LORA_RANK}"+dropout-0.0+lora-attn-only--"${note}"; do
    if [[ -d "${d}" ]]; then
      echo "${d}"
      return 0
    fi
  done
  return 1
}

LATEST_RUN_DIR=""
if LATEST_RUN_DIR="$(find_run_dir "${PHASE1_NOTE}")"; then
  :
elif [[ "${PHASE}" == "2" ]] && LATEST_RUN_DIR="$(find_run_dir "${RUN_ID_NOTE}")"; then
  :
fi

TRAIN_LOG="${OFT}/logs/finetune_suite_${SUITE}.log"
mkdir -p "${ADAPTER_RUN_ROOT}"

NUM_GPUS="${NUM_GPUS:?set NUM_GPUS}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

WANDB_ENTITY="${WANDB_ENTITY:-maggiesh-carnegie-mellon-university}"
WANDB_PROJECT="${WANDB_PROJECT:-openvla_gripper_proprio_fast}"
export WANDB_MODE="${WANDB_MODE:-online}"

latest_saved_step () {
  grep -oE 'Saving Model Checkpoint for Step [0-9]+' "${TRAIN_LOG}" 2>/dev/null | tail -1 | grep -oE '[0-9]+' || true
}

has_merged_full_model () {
  local d="$1"
  [[ -f "${d}/model.safetensors" ]] && return 0
  compgen -G "${d}/model-*.safetensors" > /dev/null
}

# Phase 3+: train on top of the merged full backbone saved in the run dir.
if [[ "${USE_MERGED_BASE:-False}" == "True" ]]; then
  if [[ -z "${LATEST_RUN_DIR}" ]] || ! has_merged_full_model "${LATEST_RUN_DIR}"; then
    echo "ERROR: USE_MERGED_BASE=True but merged full model weights not found in ${LATEST_RUN_DIR:-<none>}"
    echo "  expected model.safetensors or model-*.safetensors (run merge_lora_weights_and_save.py first)"
    exit 1
  fi
  BASE_VLA_PATH="${LATEST_RUN_DIR}"
fi

RESUME_OK=0
if [[ -n "${LATEST_RUN_DIR}" ]]; then
  if [[ -s "${LATEST_RUN_DIR}/action_head--latest_checkpoint.pt" ]] \
    || [[ -s "${LATEST_RUN_DIR}/lora_adapter/adapter_config.json" ]] \
    || { [[ "${USE_MERGED_BASE:-False}" == "True" ]] && has_merged_full_model "${LATEST_RUN_DIR}"; }; then
    RESUME_OK=1
  fi
fi

if [[ "${RESUME_OK}" == "1" ]]; then
  VLA_PATH="${LATEST_RUN_DIR}"
  RESUME_STEP=$(latest_saved_step)
  RESUME_FLAG=True
  echo "  mode=resume step=${RESUME_STEP:-unknown} phase=${PHASE} lora=${LORA_TARGET} merged_base=${USE_MERGED_BASE:-False}"
else
  VLA_PATH="${BASE_VLA_PATH}"
  RESUME_FLAG=False
  RESUME_STEP=""
  echo "  mode=fresh from base phase=${PHASE} lora=${LORA_TARGET}"
fi

echo "===== $(date) suite=${SUITE} finetune phase=${PHASE} ====="
echo "  data=${DATA_ROOT} mix=${DATASET_NAME}"
echo "  run_dir=${LATEST_RUN_DIR:-<new>}"
echo "  vla_path=${VLA_PATH}"
echo "  gpus=${NUM_GPUS} (${CUDA_VISIBLE_DEVICES}) batch/gpu=${BATCH_SIZE} grad_accum=${GRAD_ACCUM} effective_batch=$((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))"
echo "  lr=${LEARNING_RATE} decay_10x@${NUM_STEPS_BEFORE_DECAY} lr_schedule_reset=${LR_SCHEDULE_RESET} lora_rank=${LORA_RANK} lora_target=${LORA_TARGET}"
echo "  base_vla_path=${BASE_VLA_PATH} use_merged_base=${USE_MERGED_BASE:-False}"
echo "  image_aug=False num_images=1 | save_freq=${SAVE_FREQ} max_steps=${MAX_STEPS}"
echo "  wandb=${WANDB_ENTITY}/${WANDB_PROJECT} note=${RUN_ID_NOTE}"
df -h "${STORAGE_ROOT}" | tail -1

"${TORCHRUN}" --standalone --nnodes 1 --nproc-per-node "${NUM_GPUS}" vla-scripts/finetune.py \
  --vla_path "${VLA_PATH}" \
  --base_vla_path "${BASE_VLA_PATH}" \
  --data_root_dir "${DATA_ROOT}" \
  --dataset_name "${DATASET_NAME}" \
  --run_root_dir "${ADAPTER_RUN_ROOT}" \
  --use_lora True \
  --lora_rank "${LORA_RANK}" \
  --lora_target_modules "${LORA_TARGET}" \
  --merge_lora_during_training False \
  --batch_size "${BATCH_SIZE}" \
  --grad_accumulation_steps "${GRAD_ACCUM}" \
  --shuffle_buffer_size "${SHUFFLE_BUFFER_SIZE}" \
  --learning_rate "${LEARNING_RATE}" \
  --num_steps_before_decay "${NUM_STEPS_BEFORE_DECAY}" \
  --lr_schedule_reset "${LR_SCHEDULE_RESET}" \
  --max_steps "${MAX_STEPS}" \
  --image_aug False \
  --num_images_in_input 1 \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}" \
  --run_id_note "${RUN_ID_NOTE}" \
  --save_latest_checkpoint_only True \
  --save_freq "${SAVE_FREQ}" \
  --resume "${RESUME_FLAG}" \
  $(if [[ -n "${RESUME_STEP}" ]]; then echo --resume_step "${RESUME_STEP}"; fi) \
  --use_proprio True \
  --use_l1_regression True
