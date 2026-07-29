#!/usr/bin/env bash
# After spatial+goal save phase-1 checkpoints, restart them (and optionally object) in phase 2:
#   lr 5e-5 -> 5e-6 after 100k steps, grad_accum=2 (effective batch x2).
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFT="${REPO_ROOT}/openvla-oft"
LOG="${OFT}/logs/suite_phase2_trigger.log"

TARGET_STEP="${TARGET_STEP:-405000}"
POLL_SECS="${POLL_SECS:-60}"
SUITES_TO_SWITCH="${SUITES_TO_SWITCH:-spatial goal}"

log () { echo "===== $(date -Iseconds) $*" | tee -a "${LOG}"; }

latest_step () {
  local suite="$1"
  grep -oE 'Saving Model Checkpoint for Step [0-9]+' "${OFT}/logs/finetune_suite_${suite}.log" 2>/dev/null \
    | tail -1 | grep -oE '[0-9]+' || echo "0"
}

write_phase2_env () {
  local suite="$1"
  cat > "${OFT}/logs/finetune_suite_${suite}_phase.env" <<EOF
PHASE=2
LEARNING_RATE=5e-5
NUM_STEPS_BEFORE_DECAY=100000
GRAD_ACCUM=2
LR_SCHEDULE_RESET=True
RUN_ID_NOTE=suite_${suite}_oft_lr_p2
MAX_STEPS=650000
EOF
}

all_ready () {
  local suite
  for suite in ${SUITES_TO_SWITCH}; do
    local step
    step=$(latest_step "${suite}")
    if (( step < TARGET_STEP )); then
      return 1
    fi
  done
  return 0
}

stop_suite () {
  local suite="$1"
  pkill -f "vla-scripts/finetune.py.*dual_masked_${suite}\b" 2>/dev/null || true
}

log "Phase-2 trigger watching target_step>=${TARGET_STEP} for: ${SUITES_TO_SWITCH}"

while ! all_ready; do
  for suite in ${SUITES_TO_SWITCH}; do
    log "${suite}: latest_ckpt_step=$(latest_step "${suite}") (need ${TARGET_STEP})"
  done
  sleep "${POLL_SECS}"
done

log "Checkpoint threshold reached — switching to phase 2"
for suite in ${SUITES_TO_SWITCH}; do
  log "${suite}: stopping phase-1 run..."
  stop_suite "${suite}"
  write_phase2_env "${suite}"
  log "${suite}: wrote ${OFT}/logs/finetune_suite_${suite}_phase.env"
done

sleep 5
log "Watchers will auto-relaunch with phase=2 (grad_accum=2, lr=5e-5, decay@+100k)"
