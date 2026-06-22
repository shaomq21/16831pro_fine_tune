#!/usr/bin/env bash
# Use any idle GPU(s) for libero_spatial RLDS masking (no need to wait for all 8).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs/rlds_mask_libero_spatial_no_noops"
MONITOR_LOG="$LOG_DIR/gpu_monitor.log"
DATA_MIX="libero_spatial_no_noops"
MAX_GPUS=8
IDLE_MINUTES=2
POLL_SEC=60
# GPU idle if util < this AND free mem >= this (avoid cards with loaded models)
UTIL_IDLE=10
MIN_FREE_MIB=8192

STORAGE_ROOT="/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/datasets"
DATA_ROOT="${STORAGE_ROOT}/modified_libero_rlds"
OUT_ROOT="${STORAGE_ROOT}/masked_libero_rlds"

mkdir -p "$LOG_DIR"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MONITOR_LOG"
}

# Valid worker counts (16 shards must divide evenly)
pick_num_workers() {
  local n="$1"
  for w in 8 4 2 1; do
    if (( n >= w )); then
      echo "$w"
      return
    fi
  done
  echo 0
}

# Print space-separated idle GPU indices (low to high)
idle_gpu_indices() {
  local out=()
  while IFS=',' read -r idx util mem total; do
    idx="${idx// /}"
    util="${util//\%/}"
    util="${util// /}"
    mem="${mem// MiB/}"
    mem="${mem// /}"
    total="${total// MiB/}"
    total="${total// /}"
    [[ "$idx" =~ ^[0-9]+$ ]] || continue
    [[ "$util" =~ ^[0-9]+$ ]] || continue
    [[ "$mem" =~ ^[0-9]+$ ]] || continue
    [[ "$total" =~ ^[0-9]+$ ]] || continue
    free=$(( total - mem ))
    if (( util < UTIL_IDLE && free >= MIN_FREE_MIB )); then
      out+=("$idx")
    fi
  done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -n "$MAX_GPUS")
  echo "${out[*]}"
}

rlds_running() {
  pgrep -f "tools/rlds_mask.py.*${DATA_MIX}" >/dev/null 2>&1 || \
  pgrep -f "rlds_mask_multi_gpu.sh ${DATA_MIX}" >/dev/null 2>&1
}

start_rlds() {
  local gpu_csv="$1"
  local num_workers="$2"
  log "Starting RLDS: ${DATA_MIX}, workers=${num_workers}, GPUs=${gpu_csv}"
  log "  input:  ${DATA_ROOT}/${DATA_MIX}"
  log "  output: ${OUT_ROOT}/${DATA_MIX}"
  rm -f "${OUT_ROOT}/${DATA_MIX}/1.0.0/"*.tfrecord
  rm -f "${OUT_ROOT}/.rlds_resume_${DATA_MIX}_w"*.json
  rm -f "${OUT_ROOT}/.rlds_shard_counts_${DATA_MIX}_w"*.json
  RLDS_DATA_ROOT="$DATA_ROOT" RLDS_OUT_ROOT="$OUT_ROOT" \
    nohup bash "$REPO_ROOT/tools/rlds_mask_multi_gpu.sh" "$DATA_MIX" "$num_workers" "$gpu_csv" \
    >> "$LOG_DIR/main.log" 2>&1 &
  log "RLDS launcher PID=$!"
}

idle_since=""
log "Monitor started (use any idle GPU, util<${UTIL_IDLE}%, free>=${MIN_FREE_MIB}MiB, wait=${IDLE_MINUTES}m, out=${OUT_ROOT})"

while true; do
  if rlds_running; then
    log "RLDS running; monitor exiting."
    exit 0
  fi

  read -ra idle_gpus <<< "$(idle_gpu_indices)"
  n_idle=${#idle_gpus[@]}

  if (( n_idle == 0 )); then
    if [[ -n "$idle_since" ]]; then
      log "No idle GPUs; reset timer (was since $idle_since)"
    fi
    idle_since=""
  else
    num_workers=$(pick_num_workers "$n_idle")
    # Use lowest-index idle GPUs
    selected=()
    for (( i=0; i<num_workers; i++ )); do
      selected+=("${idle_gpus[$i]}")
    done
    gpu_csv=$(IFS=,; echo "${selected[*]}")

    if [[ -z "$idle_since" ]]; then
      idle_since="$(date '+%Y-%m-%d %H:%M:%S')"
      log "Idle GPUs [${idle_gpus[*]}] -> plan ${num_workers} worker(s) on [${gpu_csv}]; timer started"
    else
      idle_epoch=$(date -d "$idle_since" +%s)
      now_epoch=$(date +%s)
      idle_min=$(( (now_epoch - idle_epoch) / 60 ))
      log "Idle ${idle_min}/${IDLE_MINUTES}m | available=[${idle_gpus[*]}] -> use ${num_workers} on [${gpu_csv}]"
      if (( idle_min >= IDLE_MINUTES )); then
        start_rlds "$gpu_csv" "$num_workers"
        exit 0
      fi
    fi
  fi
  sleep "$POLL_SEC"
done
