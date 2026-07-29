#!/usr/bin/env bash
# After each suite finishes its phase-1/2 target step:
#   1) merge attn-only LoRA into backbone (offline)
#   2) backup old lora_adapter/
#   3) switch to phase 3: all-linear LoRA, grad_accum=8 (global batch 64 @ 2 GPU), lr=5e-4
#
# Per-suite default target steps (override with TARGET_STEP_<SUITE> env):
#   spatial/goal: 500000 (after phase-2 plateau)
#   object/study_scene4: 150000 (after phase-1 warm-up)
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFT="${REPO_ROOT}/openvla-oft"
LOG="${OFT}/logs/suite_phase3_trigger.log"

STORAGE_ROOT="${STORAGE_ROOT:-/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune}"
ADAPTER_RUN_ROOT="${STORAGE_ROOT}/runs/openvla_adapters"
BASE_VLA="${OFT}/checkpoints/openvla-7b"
PYTHON="${PYTHON:-${STORAGE_ROOT}/conda_envs/simplevla/bin/python}"

POLL_SECS="${POLL_SECS:-120}"
EXTRA_PHASE3_STEPS="${EXTRA_PHASE3_STEPS:-150000}"
SUITES_TO_SWITCH="${SUITES_TO_SWITCH:-spatial goal object study_scene4}"

log () { echo "===== $(date -Iseconds) $*" | tee -a "${LOG}"; }

target_step_for_suite () {
  local suite="$1"
  local var="TARGET_STEP_${suite^^}"
  if [[ -n "${!var:-}" ]]; then
    echo "${!var}"
    return
  fi
  case "${suite}" in
    spatial|goal) echo "500000" ;;
    object|study_scene4) echo "150000" ;;
    *) echo "150000" ;;
  esac
}

latest_step () {
  local suite="$1"
  grep -oE 'Saving Model Checkpoint for Step [0-9]+' "${OFT}/logs/finetune_suite_${suite}.log" 2>/dev/null \
    | tail -1 | grep -oE '[0-9]+' || echo "0"
}

find_run_dir () {
  local suite="$1"
  local d
  for d in "${ADAPTER_RUN_ROOT}"/openvla-7b+dual_masked_"${suite}"+b*+lr-*+lora-r32+dropout-0.0+lora-attn-only--suite_"${suite}"_oft_lr*; do
    if [[ -d "${d}" ]]; then
      echo "${d}"
      return 0
    fi
  done
  return 1
}

stop_suite () {
  local suite="$1"
  pkill -f "vla-scripts/finetune.py.*dual_masked_${suite}\b" 2>/dev/null || true
}

merge_suite () {
  local suite="$1"
  local ckpt_dir="$2"
  if [[ ! -d "${ckpt_dir}/lora_adapter" ]]; then
    log "${suite}: no lora_adapter — skip merge (may already be merged)"
    return 0
  fi
  log "${suite}: merging attn-only LoRA into backbone at ${ckpt_dir}"
  "${PYTHON}" "${OFT}/vla-scripts/merge_lora_weights_and_save.py" \
    --base_checkpoint "${BASE_VLA}" \
    --lora_finetuned_checkpoint_dir "${ckpt_dir}" >> "${LOG}" 2>&1
  if [[ -d "${ckpt_dir}/lora_adapter" ]]; then
    local backup="${ckpt_dir}/lora_adapter.phase12_attn_only.$(date +%Y%m%d_%H%M%S)"
    mv "${ckpt_dir}/lora_adapter" "${backup}"
    log "${suite}: backed up old adapter -> ${backup}"
  fi
}

write_phase3_env () {
  local suite="$1"
  local step="$2"
  local max_steps=$((step + EXTRA_PHASE3_STEPS))
  cat > "${OFT}/logs/finetune_suite_${suite}_phase.env" <<EOF
PHASE=3
LEARNING_RATE=5e-4
NUM_STEPS_BEFORE_DECAY=100000
GRAD_ACCUM=8
LR_SCHEDULE_RESET=True
LORA_TARGET=all-linear
USE_MERGED_BASE=True
RUN_ID_NOTE=suite_${suite}_oft_lr_p3
MAX_STEPS=${max_steps}
EOF
}

suite_ready () {
  local suite="$1"
  local step target
  step=$(latest_step "${suite}")
  target=$(target_step_for_suite "${suite}")
  (( step >= target ))
}

log "Phase-3 trigger watching suites: ${SUITES_TO_SWITCH}"
log "Extra phase-3 steps after switch: ${EXTRA_PHASE3_STEPS}"

pending=(${SUITES_TO_SWITCH})
while ((${#pending[@]} > 0)); do
  still=()
  for suite in "${pending[@]}"; do
    step=$(latest_step "${suite}")
    target=$(target_step_for_suite "${suite}")
    if (( step >= target )); then
      log "${suite}: ready (ckpt=${step} >= ${target})"
      run_dir=""
      if ! run_dir="$(find_run_dir "${suite}")"; then
        log "${suite}: ERROR — run dir not found under ${ADAPTER_RUN_ROOT}"
        continue
      fi
      log "${suite}: stopping current training..."
      stop_suite "${suite}"
      sleep 5
      merge_suite "${suite}" "${run_dir}"
      write_phase3_env "${suite}" "${step}"
      log "${suite}: wrote ${OFT}/logs/finetune_suite_${suite}_phase.env (max_steps=$((step + EXTRA_PHASE3_STEPS)))"
    else
      log "${suite}: waiting ckpt=${step} (need ${target})"
      still+=("${suite}")
    fi
  done
  pending=("${still[@]}")
  if ((${#pending[@]} > 0)); then
    sleep "${POLL_SECS}"
  fi
done

log "All suites switched to phase 3. Watchers will relaunch with all-linear + grad_accum=8 + merged base."
