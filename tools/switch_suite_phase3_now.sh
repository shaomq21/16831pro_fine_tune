#!/usr/bin/env bash
# Immediately switch all suites to phase 3 (skip phase-2 wait).
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFT="${REPO_ROOT}/openvla-oft"
LOG="${OFT}/logs/suite_phase3_switch_now.log"

STORAGE_ROOT="${STORAGE_ROOT:-/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune}"
ADAPTER_RUN_ROOT="${STORAGE_ROOT}/runs/openvla_adapters"
BASE_VLA="${OFT}/checkpoints/openvla-7b"
PYTHON="${PYTHON:-${STORAGE_ROOT}/conda_envs/simplevla/bin/python}"
MERGE_GPU="${MERGE_GPU:-7}"
# Always merge LoRA into a full HF checkpoint before phase-3 unless explicitly skipped.
SKIP_DISK_MERGE="${SKIP_DISK_MERGE:-0}"

EXTRA_PHASE3_STEPS="${EXTRA_PHASE3_STEPS:-150000}"
SUITES="${SUITES:-spatial goal object study_scene4}"

log () { echo "===== $(date -Iseconds) $*" | tee -a "${LOG}"; }

latest_step () {
  local suite="$1"
  grep -oE 'Saving Model Checkpoint for Step [0-9]+' "${OFT}/logs/finetune_suite_${suite}.log" 2>/dev/null \
    | tail -1 | grep -oE '[0-9]+' || echo "0"
}

find_run_dir () {
  local suite="$1"
  local best="" best_mtime=0 d mt
  for d in "${ADAPTER_RUN_ROOT}"/openvla-7b+dual_masked_"${suite}"+b*+lr-*+lora-r32+dropout-0.0+lora-attn-only--suite_"${suite}"_oft_lr*; do
    [[ -d "${d}" ]] || continue
    if [[ -s "${d}/action_head--latest_checkpoint.pt" ]]; then
      mt=$(stat -c %Y "${d}/action_head--latest_checkpoint.pt" 2>/dev/null || echo 0)
    elif [[ -d "${d}/lora_adapter" ]]; then
      mt=$(stat -c %Y "${d}/lora_adapter" 2>/dev/null || echo 0)
    else
      continue
    fi
    if (( mt > best_mtime )); then
      best="${d}"
      best_mtime="${mt}"
    fi
  done
  if [[ -n "${best}" ]]; then
    echo "${best}"
    return 0
  fi
  return 1
}

grad_accum_for_suite () {
  local suite="$1"
  if [[ "${suite}" == "study_scene4" ]]; then
    echo "32"
  else
    echo "8"
  fi
}

stop_suite () {
  local suite="$1"
  pkill -f "vla-scripts/finetune.py.*dual_masked_${suite}\b" 2>/dev/null || true
}

stop_watchers () {
  pkill -f "finetune_suite_watch.sh" 2>/dev/null || true
}

merge_suite () {
  local suite="$1"
  local ckpt_dir="$2"
  local base_for_merge="${3:-${BASE_VLA}}"
  if [[ "${SKIP_DISK_MERGE}" == "1" ]]; then
    log "${suite}: SKIP_DISK_MERGE=1 — refusing to leave phase-3 without a merged full ckpt"
    return 1
  fi
  if [[ ! -f "${ckpt_dir}/lora_adapter/adapter_config.json" ]]; then
    if compgen -G "${ckpt_dir}/model-*.safetensors" > /dev/null || [[ -f "${ckpt_dir}/model.safetensors" ]]; then
      log "${suite}: already has merged full model and no adapter — ok"
      return 0
    fi
    log "${suite}: no lora_adapter and no merged full model — cannot proceed"
    return 1
  fi
  log "${suite}: merging LoRA into full model at ${ckpt_dir} (base=${base_for_merge}, GPU ${MERGE_GPU})"
  CUDA_VISIBLE_DEVICES="${MERGE_GPU}" "${PYTHON}" "${OFT}/vla-scripts/merge_lora_weights_and_save.py" \
    --base_checkpoint "${base_for_merge}" \
    --lora_finetuned_checkpoint_dir "${ckpt_dir}" >> "${LOG}" 2>&1
  if [[ ! -f "${ckpt_dir}/model.safetensors" ]] && ! compgen -G "${ckpt_dir}/model-*.safetensors" > /dev/null; then
    log "${suite}: ERROR — merge did not produce model*.safetensors"
    return 1
  fi
  if [[ -d "${ckpt_dir}/lora_adapter" ]]; then
    local backup="${ckpt_dir}/lora_adapter.merged_backup.$(date +%Y%m%d_%H%M%S)"
    mv "${ckpt_dir}/lora_adapter" "${backup}"
    log "${suite}: backed up adapter -> ${backup}"
  fi
}

write_phase3_env () {
  local suite="$1"
  local step="$2"
  local max_steps=$((step + EXTRA_PHASE3_STEPS))
  local grad_accum
  grad_accum=$(grad_accum_for_suite "${suite}")
  cat > "${OFT}/logs/finetune_suite_${suite}_phase.env" <<EOF
PHASE=3
LEARNING_RATE=5e-4
NUM_STEPS_BEFORE_DECAY=100000
GRAD_ACCUM=${grad_accum}
LR_SCHEDULE_RESET=True
LORA_TARGET=all-linear
USE_MERGED_BASE=True
RUN_ID_NOTE=suite_${suite}_oft_lr_p3
MAX_STEPS=${max_steps}
EOF
  log "${suite}: phase.env written (from_step=${step}, max_steps=${max_steps}, grad_accum=${grad_accum}, USE_MERGED_BASE=True)"
}

log "Immediate phase-3 switch for: ${SUITES}"
log "Extra steps after switch: ${EXTRA_PHASE3_STEPS}"

pkill -f "trigger_suite_phase3.sh" 2>/dev/null || true
log "Stopped background phase-3 trigger (if any)"

log "Stopping all suite training + watchers..."
for suite in ${SUITES}; do
  stop_suite "${suite}"
done
stop_watchers
sleep 5

for suite in ${SUITES}; do
  step=$(latest_step "${suite}")
  run_dir=""
  if ! run_dir="$(find_run_dir "${suite}")"; then
    log "${suite}: ERROR — no run dir with checkpoints"
    continue
  fi
  log "${suite}: ckpt_step=${step} run_dir=${run_dir}"
  merge_suite "${suite}" "${run_dir}"
  write_phase3_env "${suite}" "${step}"
done

log "Relaunching watchers..."
bash "${SCRIPT_DIR}/launch_suite_watchers.sh" >> "${LOG}" 2>&1

log "Done. Phase 3 should start for all suites with all-linear LoRA + grad_accum=8."
