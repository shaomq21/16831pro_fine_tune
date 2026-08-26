#!/usr/bin/env bash
# Wait until dual_masked_goal full shards finish uploading, then launch mask RL LoRA
# (continuous OFT; no baseline re-eval). Primary: goal task8 bowl color (known 40% SR).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORAGE="${STORAGE_ROOT:-/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune}"
CKPT="${CKPT_DIR:-${STORAGE}/runs/openvla_adapters/openvla-7b+dual_masked_goal+b4+lr-0.0005+lora-r32+dropout-0.0+lora-attn-only--suite_goal_oft_lr}"
LOG="${ROOT}/openvla-oft/logs/watch_mask_goal_ckpt_then_rl.log"
POLL_SEC="${POLL_SEC:-60}"

mkdir -p "$(dirname "${LOG}")"
echo "===== $(date -Iseconds) watching ${CKPT} =====" | tee -a "${LOG}"

ready() {
  local missing=0
  for i in 1 2 3 4; do
    local f="${CKPT}/model-0000${i}-of-00004.safetensors"
    local part="${f}.filepart"
    if [[ -f "${part}" ]]; then
      return 1
    fi
    if [[ ! -f "${f}" ]]; then
      missing=1
    fi
  done
  # Also accept adapter-only load path: lora + action_head (no merged shards)
  # Prefer full shards when user is uploading them; if FORCE_ADAPTER_ONLY=1 skip shards.
  if [[ "${FORCE_ADAPTER_ONLY:-0}" == "1" ]]; then
    [[ -f "${CKPT}/lora_adapter/adapter_model.safetensors" && -f "${CKPT}/action_head--latest_checkpoint.pt" ]]
    return $?
  fi
  [[ ${missing} -eq 0 ]]
}

while ! ready; do
  sz=$(stat -c%s "${CKPT}/model-00001-of-00004.safetensors.filepart" 2>/dev/null || echo 0)
  echo "$(date -Iseconds) waiting... shard1_filepart=${sz} bytes" | tee -a "${LOG}"
  sleep "${POLL_SEC}"
done

echo "===== $(date -Iseconds) ckpt ready; launching RL =====" | tee -a "${LOG}"

# Task 8 first (color baseline 40%). Skip baseline eval inside train via SKIP_INIT_EVAL if supported.
cd "${ROOT}"
SUITE=goal TASK_ID=8 GPU="${GPU:-0}" NOTE=mask_bowl8_color \
  NUM_ITERS="${NUM_ITERS:-8}" EVAL_TRIALS="${EVAL_TRIALS:-4}" \
  nohup bash tools/run_rl_lora_color_quick.sh >> "${LOG}" 2>&1 &
echo "rl_bowl8_pid=$!" | tee -a "${LOG}"

# Optional second task on another GPU
if [[ "${LAUNCH_TASK5:-1}" == "1" ]]; then
  sleep 5
  SUITE=goal TASK_ID=5 GPU="${GPU5:-1}" NOTE=mask_plate5_color \
    NUM_ITERS="${NUM_ITERS:-8}" EVAL_TRIALS="${EVAL_TRIALS:-4}" \
    nohup bash tools/run_rl_lora_color_quick.sh >> "${LOG}.task5" 2>&1 &
  echo "rl_plate5_pid=$!" | tee -a "${LOG}"
fi

echo "===== launched; see ${LOG} =====" | tee -a "${LOG}"
