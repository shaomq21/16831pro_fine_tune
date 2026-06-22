#!/usr/bin/env bash
# Watch RLDS mask workers; auto-restart on crash/stale progress; upgrade to 4 GPUs ASAP.
#
# Prefers GPUs 0,1,2,4 when all healthy. Falls back to 2 or 1 GPU while waiting.
#
# Usage:
#   nohup bash tools/watch_rlds_mask.sh >> logs/rlds_mask_<mix>/watchdog.log 2>&1 &

set -euo pipefail

DATA_MIX="${DATA_MIX:-libero_spatial_no_noops}"
TARGET_GPUS="${TARGET_GPUS:-0,1,2,4}"
TARGET_NUM_WORKERS="${TARGET_NUM_WORKERS:-4}"
POLL_SEC="${POLL_SEC:-30}"
STALE_SEC="${STALE_SEC:-900}"
TOTAL_EPISODES="${TOTAL_EPISODES:-432}"
NUM_SHARDS=16

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${RLDS_DATA_ROOT:-/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/datasets/modified_libero_rlds}"
OUT_ROOT="${RLDS_OUT_ROOT:-/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/datasets/masked_libero_rlds}"
PYTHON="${RLDS_PYTHON:-/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/conda_envs/vla-preprocess/bin/python}"
LOG_DIR="$REPO_ROOT/logs/rlds_mask_${DATA_MIX}"
WATCH_LOG="$LOG_DIR/watchdog.log"
STATE_FILE="$LOG_DIR/watchdog_state.env"

mkdir -p "$LOG_DIR"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$WATCH_LOG"
}

gpu_cuda_ok() {
  local g="$1"
  CUDA_VISIBLE_DEVICES="$g" "$PYTHON" -c "import torch; torch.zeros(1, device='cuda')" >/dev/null 2>&1
}

target_gpus_all_ok() {
  local g
  IFS=',' read -ra want <<< "$TARGET_GPUS"
  for g in "${want[@]}"; do
    gpu_cuda_ok "$g" || return 1
  done
  return 0
}

healthy_gpus_ordered() {
  local seen="" g ok=()
  for g in ${TARGET_GPUS//,/ } 0 1 2 3 4 5 6 7; do
    [[ " $seen " == *" $g "* ]] && continue
    seen="$seen $g"
    gpu_cuda_ok "$g" && ok+=("$g")
  done
  echo "${ok[@]}"
}

pick_worker_count() {
  local n="$1"
  for w in 16 8 4 2 1; do
    if (( n >= w && NUM_SHARDS % w == 0 )); then
      echo "$w"
      return
    fi
  done
  echo 0
}

count_workers() {
  pgrep -f "tools/rlds_mask.py.*${DATA_MIX}" 2>/dev/null | wc -l
}

count_done_episodes() {
  "$PYTHON" - <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__import__("os").environ["REPO_ROOT"]) / "tools"))
from rlds_mask_state import done_episodes_from_tfrecords

out = Path(__import__("os").environ["OUT_ROOT"])
mix = __import__("os").environ["DATA_MIX"]
print(len(done_episodes_from_tfrecords(out, mix)))
PY
}

progress_age_sec() {
  local w="$1"
  local path
  if [[ "${NUM_WORKERS:-1}" == "1" ]]; then
    path="$OUT_ROOT/.rlds_mask_progress_${DATA_MIX}.json"
  else
    path="$OUT_ROOT/.rlds_mask_progress_${DATA_MIX}_w${w}.json"
  fi
  [[ -f "$path" ]] || { echo 999999; return; }
  "$PYTHON" - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path
p = Path("$path")
d = json.loads(p.read_text())
t = datetime.fromisoformat(d["updated_at"].replace("Z", "+00:00"))
print(int((datetime.now(timezone.utc) - t).total_seconds()))
PY
}

load_state() {
  NUM_WORKERS=1
  GPU_IDS="0"
  if [[ -f "$STATE_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE"
  fi
}

save_state() {
  echo "NUM_WORKERS=$NUM_WORKERS" > "$STATE_FILE"
  echo "GPU_IDS=$GPU_IDS" >> "$STATE_FILE"
}

workers_healthy() {
  load_state
  local proc stale=0
  proc="$(count_workers)"
  if (( proc < NUM_WORKERS )); then
    log "UNHEALTHY: ${proc}/${NUM_WORKERS} worker process(es)"
    return 1
  fi
  if (( NUM_WORKERS == 1 )); then
    age="$(progress_age_sec 0)"
    if (( age > STALE_SEC )); then
      log "UNHEALTHY: progress stale ${age}s"
      return 1
    fi
    return 0
  fi
  for (( w=0; w<NUM_WORKERS; w++ )); do
    age="$(progress_age_sec "$w")"
    if (( age > STALE_SEC )); then
      log "UNHEALTHY: w${w} stale ${age}s"
      stale=1
    fi
  done
  (( stale == 0 ))
}

should_upgrade_to_target() {
  load_state
  target_gpus_all_ok || return 1
  if (( NUM_WORKERS < TARGET_NUM_WORKERS )); then
    return 0
  fi
  [[ "$GPU_IDS" != "$TARGET_GPUS" ]]
}

launch_workers() {
  local nw="$1"
  shift
  local -a gpus=("$@")
  GPU_IDS=$(IFS=,; echo "${gpus[*]}")
  NUM_WORKERS="$nw"
  save_state

  log "Launching ${NUM_WORKERS} worker(s) on GPU(s) ${GPU_IDS} (resume, no data wipe)"

  "$PYTHON" "$REPO_ROOT/tools/init_multi_gpu_resume.py" \
    --data_mix "$DATA_MIX" --num_workers "$NUM_WORKERS" --out_root "$OUT_ROOT"

  if (( NUM_WORKERS == 1 )); then
    CUDA_VISIBLE_DEVICES="${gpus[0]}" nohup "$PYTHON" "$REPO_ROOT/tools/rlds_mask.py" \
      --data_mix "$DATA_MIX" \
      --data_root "$DATA_ROOT" \
      --out_root "$OUT_ROOT" \
      --debug_dir "rlds_mask_debug/${DATA_MIX}" \
      --debug_every_episodes 1 --debug_frames 1 --max_debug_images 20 \
      --no_mask_wrist --num_workers 1 --worker_id 0 --device cuda:0 --fast --resume \
      > "$LOG_DIR/worker_0.log" 2>&1 &
    log "Single-GPU worker PID=$! on GPU ${gpus[0]}"
  else
    RESUME=1 BACKGROUND=1 bash "$REPO_ROOT/tools/rlds_mask_multi_gpu.sh" \
      "$DATA_MIX" "$NUM_WORKERS" "$GPU_IDS"
    log "Multi-GPU launch on ${GPU_IDS}"
  fi
}

restart_workers() {
  local reason="${1:-restart}"
  pkill -f "tools/rlds_mask.py.*${DATA_MIX}" 2>/dev/null || true
  sleep 5

  if target_gpus_all_ok; then
    log "${reason}: all target GPUs ${TARGET_GPUS} healthy -> ${TARGET_NUM_WORKERS} workers"
    IFS=',' read -ra gpus <<< "$TARGET_GPUS"
    launch_workers "$TARGET_NUM_WORKERS" "${gpus[@]}"
    return 0
  fi

  read -ra ok_gpus <<< "$(healthy_gpus_ordered)"
  local n_ok=${#ok_gpus[@]}
  if (( n_ok == 0 )); then
    log "ERROR: no healthy GPU found"
    return 1
  fi

  local nw
  nw="$(pick_worker_count "$n_ok")"
  if (( nw == 0 )); then
    log "ERROR: cannot pick worker count for ${n_ok} GPU(s)"
    return 1
  fi

  local selected=()
  for (( i=0; i<nw; i++ )); do
    selected+=("${ok_gpus[$i]}")
  done
  log "${reason}: fallback ${nw} worker(s) on ${selected[*]} (waiting for ${TARGET_GPUS})"
  launch_workers "$nw" "${selected[@]}"
}

export OUT_ROOT DATA_MIX REPO_ROOT
log "Watchdog started: target=${TARGET_GPUS} x${TARGET_NUM_WORKERS} poll=${POLL_SEC}s stale=${STALE_SEC}s"

while true; do
  load_state
  done_n="$(count_done_episodes)"
  if (( done_n >= TOTAL_EPISODES )); then
    log "All ${TOTAL_EPISODES} episodes done — exiting"
    exit 0
  fi

  if should_upgrade_to_target; then
    log "UPGRADE: target GPUs ${TARGET_GPUS} ready — switching from ${NUM_WORKERS}x[${GPU_IDS}]"
    restart_workers "upgrade" || log "Upgrade failed; retry in ${POLL_SEC}s"
  elif workers_healthy; then
    if (( NUM_WORKERS < TARGET_NUM_WORKERS )); then
      log "OK: ${done_n}/${TOTAL_EPISODES} ep | ${NUM_WORKERS}w [${GPU_IDS}] | waiting for ${TARGET_GPUS}"
    else
      log "OK: ${done_n}/${TOTAL_EPISODES} ep | ${NUM_WORKERS}w [${GPU_IDS}]"
    fi
  else
    restart_workers "recover" || log "Recover failed; retry in ${POLL_SEC}s"
  fi

  sleep "$POLL_SEC"
done
