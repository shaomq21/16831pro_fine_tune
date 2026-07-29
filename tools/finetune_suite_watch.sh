#!/usr/bin/env bash
# Watcher for a SINGLE masked suite run (spatial|object|goal|study_scene4).
set -u

SUITE="${SUITE:?set SUITE=spatial|object|goal|study_scene4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFT="${REPO_ROOT}/openvla-oft"
LOG="${OFT}/logs/finetune_suite_${SUITE}_watch.log"
PIDFILE="${OFT}/logs/finetune_suite_${SUITE}_watch.pid"
PHASE_FILE="${OFT}/logs/finetune_suite_${SUITE}_phase.env"
GPU_FILE="${OFT}/logs/finetune_suite_${SUITE}_gpu.env"

STORAGE_ROOT="${STORAGE_ROOT:-/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune}"

# Multi-GPU defaults per suite (override with finetune_suite_${SUITE}_gpu.env)
if [[ -f "${GPU_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${GPU_FILE}"
else
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/suite_gpu_layout.sh"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES for this suite}"
export NUM_GPUS="${NUM_GPUS:?set NUM_GPUS for this suite}"

POLL_SECS="${POLL_SECS:-60}"
RESTART_SLEEP="${RESTART_SLEEP:-30}"

mkdir -p "${OFT}/logs"
echo $$ > "${PIDFILE}"

reload_phase_config () {
  PHASE=1
  BATCH_SIZE=4
  GRAD_ACCUM=1
  LEARNING_RATE=5e-4
  NUM_STEPS_BEFORE_DECAY=100000
  LR_SCHEDULE_RESET=False
  LORA_RANK=32
  SHUFFLE_BUFFER_SIZE=10000
  SAVE_FREQ=1500
  MAX_STEPS=650000
  LORA_TARGET=attn-only
  USE_MERGED_BASE=False
  if [[ -f "${GPU_FILE}" ]]; then
    # shellcheck source=/dev/null
    source "${GPU_FILE}"
  fi
  if [[ -f "${PHASE_FILE}" ]]; then
    # shellcheck source=/dev/null
    source "${PHASE_FILE}"
  fi
  export PHASE="${PHASE:-1}"
  export BATCH_SIZE="${BATCH_SIZE:-4}"
  export GRAD_ACCUM="${GRAD_ACCUM:-1}"
  export LEARNING_RATE="${LEARNING_RATE:-5e-4}"
  export NUM_STEPS_BEFORE_DECAY="${NUM_STEPS_BEFORE_DECAY:-100000}"
  export LR_SCHEDULE_RESET="${LR_SCHEDULE_RESET:-False}"
  export LORA_RANK="${LORA_RANK:-32}"
  export SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-10000}"
  export SAVE_FREQ="${SAVE_FREQ:-1500}"
  export MAX_STEPS="${MAX_STEPS:-650000}"
  export LORA_TARGET="${LORA_TARGET:-attn-only}"
  export USE_MERGED_BASE="${USE_MERGED_BASE:-False}"
  if [[ "${PHASE}" == "3" ]]; then
    export LEARNING_RATE="${LEARNING_RATE:-5e-4}"
    export GRAD_ACCUM="${GRAD_ACCUM:-8}"
    export LR_SCHEDULE_RESET="${LR_SCHEDULE_RESET:-True}"
    export LORA_TARGET="${LORA_TARGET:-all-linear}"
    export RUN_ID_NOTE="${RUN_ID_NOTE:-suite_${SUITE}_oft_lr_p3}"
  elif [[ "${PHASE}" == "2" ]]; then
    export LEARNING_RATE="${LEARNING_RATE:-5e-5}"
    export GRAD_ACCUM="${GRAD_ACCUM:-2}"
    export LR_SCHEDULE_RESET="${LR_SCHEDULE_RESET:-True}"
    export RUN_ID_NOTE="${RUN_ID_NOTE:-suite_${SUITE}_oft_lr_p2}"
  else
    export RUN_ID_NOTE="${RUN_ID_NOTE:-suite_${SUITE}_oft_lr}"
  fi
}

reload_phase_config

log () { echo "===== $(date -Iseconds) [${SUITE}] $*" | tee -a "${LOG}"; }

matches_this_suite () {
  pgrep -af "vla-scripts/finetune.py" 2>/dev/null | grep -q "dual_masked_${SUITE}\b"
}

launch () {
  reload_phase_config
  df -h "${STORAGE_ROOT}" | tail -1 | tee -a "${LOG}"
  log "Launching finetune_suite.sh phase=${PHASE} on GPUs ${CUDA_VISIBLE_DEVICES} (nproc=${NUM_GPUS}) batch=${BATCH_SIZE} lr=${LEARNING_RATE} grad_accum=${GRAD_ACCUM} lora=${LORA_TARGET} merged_base=${USE_MERGED_BASE} max_steps=${MAX_STEPS}"
  SUITE="${SUITE}" PHASE="${PHASE}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" NUM_GPUS="${NUM_GPUS}" \
    BATCH_SIZE="${BATCH_SIZE}" LORA_TARGET="${LORA_TARGET}" USE_MERGED_BASE="${USE_MERGED_BASE}" \
    LEARNING_RATE="${LEARNING_RATE}" GRAD_ACCUM="${GRAD_ACCUM}" LR_SCHEDULE_RESET="${LR_SCHEDULE_RESET}" \
    NUM_STEPS_BEFORE_DECAY="${NUM_STEPS_BEFORE_DECAY}" MAX_STEPS="${MAX_STEPS}" RUN_ID_NOTE="${RUN_ID_NOTE}" \
    bash "${SCRIPT_DIR}/finetune_suite.sh" >> "${OFT}/logs/finetune_suite_${SUITE}.log" 2>&1
  local code=$?
  log "finetune_suite.sh exited with code ${code}"
  return "${code}"
}

log "Watcher started pid=$$ gpus=${CUDA_VISIBLE_DEVICES} nproc=${NUM_GPUS} phase=${PHASE} lr=${LEARNING_RATE} grad_accum=${GRAD_ACCUM} poll=${POLL_SECS}s"

while true; do
  if matches_this_suite; then
    sleep "${POLL_SECS}"
    continue
  fi
  log "Suite ${SUITE} finetune not running — (re)starting..."
  launch || true
  log "Sleeping ${RESTART_SLEEP}s before next check..."
  sleep "${RESTART_SLEEP}"
done
