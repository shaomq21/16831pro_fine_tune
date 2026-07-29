#!/usr/bin/env bash
# Bake each suite's current LoRA into a full HF checkpoint, then relaunch phase-3
# training from that merged base (USE_MERGED_BASE=True).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFT="${REPO_ROOT}/openvla-oft"
LOG="${OFT}/logs/fix_suite_merged_base.log"
mkdir -p "${OFT}/logs"

STORAGE_ROOT="${STORAGE_ROOT:-/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune}"
ADAPTER_RUN_ROOT="${STORAGE_ROOT}/runs/openvla_adapters"
BASE_VLA="${OFT}/checkpoints/openvla-7b"
PYTHON="${PYTHON:-${STORAGE_ROOT}/conda_envs/simplevla/bin/python}"
MERGE_GPU="${MERGE_GPU:-1}"
SUITES="${SUITES:-spatial goal object study_scene4}"

log () { echo "===== $(date -Iseconds) $*" | tee -a "${LOG}"; }

find_run_dir () {
  local suite="$1" best="" best_mtime=0 d mt
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
  [[ -n "${best}" ]] || return 1
  echo "${best}"
}

latest_step () {
  local suite="$1"
  grep -oE 'Saving Model Checkpoint for Step [0-9]+' "${OFT}/logs/finetune_suite_${suite}.log" 2>/dev/null \
    | tail -1 | grep -oE '[0-9]+' || echo "0"
}

grad_accum_for_suite () {
  case "$1" in
    study_scene4) echo "32" ;;
    *) echo "8" ;;
  esac
}

has_merged () {
  local d="$1"
  [[ -f "${d}/model.safetensors" ]] && return 0
  compgen -G "${d}/model-*.safetensors" > /dev/null
}

stop_all () {
  log "Stopping suite training + watchers..."
  pkill -9 -f 'finetune_suite_watch.sh' 2>/dev/null || true
  pkill -9 -f 'vla-scripts/finetune.py.*dual_masked_' 2>/dev/null || true
  sleep 5
}

write_phase3_env () {
  local suite="$1" step="$2" max_steps grad_accum
  # Keep existing MAX_STEPS if already set higher.
  max_steps=""
  if [[ -f "${OFT}/logs/finetune_suite_${suite}_phase.env" ]]; then
    # shellcheck source=/dev/null
    source "${OFT}/logs/finetune_suite_${suite}_phase.env"
    max_steps="${MAX_STEPS:-}"
  fi
  if [[ -z "${max_steps}" ]]; then
    max_steps=$((step + 150000))
  fi
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
  log "${suite}: phase.env -> USE_MERGED_BASE=True step=${step} max_steps=${max_steps} grad_accum=${grad_accum}"
}

merge_one () {
  local suite="$1" run_dir step
  run_dir="$(find_run_dir "${suite}")"
  step="$(latest_step "${suite}")"
  log "${suite}: run_dir=${run_dir} ckpt_step=${step}"

  if has_merged "${run_dir}" && [[ ! -f "${run_dir}/lora_adapter/adapter_config.json" ]]; then
    log "${suite}: merged full model already present and adapter cleared — skip merge"
    write_phase3_env "${suite}" "${step}"
    return 0
  fi

  if [[ ! -f "${run_dir}/lora_adapter/adapter_config.json" ]]; then
    log "${suite}: ERROR — no lora_adapter to merge and no full model"
    return 1
  fi

  df -h "${STORAGE_ROOT}" | tee -a "${LOG}"
  log "${suite}: merging current LoRA into full model on GPU ${MERGE_GPU}..."
  CUDA_VISIBLE_DEVICES="${MERGE_GPU}" "${PYTHON}" "${OFT}/vla-scripts/merge_lora_weights_and_save.py" \
    --base_checkpoint "${BASE_VLA}" \
    --lora_finetuned_checkpoint_dir "${run_dir}" >> "${LOG}" 2>&1

  if ! has_merged "${run_dir}"; then
    log "${suite}: ERROR — merge did not write model*.safetensors"
    return 1
  fi

  local backup="${run_dir}/lora_adapter.pre_merged_base.$(date +%Y%m%d_%H%M%S)"
  mv "${run_dir}/lora_adapter" "${backup}"
  log "${suite}: adapter backed up -> ${backup}"
  write_phase3_env "${suite}" "${step}"
}

log "Fix suites onto merged full base, then relaunch watchers"
stop_all

failed=0
for suite in ${SUITES}; do
  if ! merge_one "${suite}"; then
    failed=1
  fi
done

if (( failed )); then
  log "ERROR: one or more merges failed — not relaunching"
  exit 1
fi

log "Relaunching watchers with USE_MERGED_BASE=True..."
bash "${SCRIPT_DIR}/launch_suite_watchers.sh" | tee -a "${LOG}"
log "Done."
