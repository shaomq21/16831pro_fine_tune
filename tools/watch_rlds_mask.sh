#!/usr/bin/env bash
# Monitor RLDS mask workers. NEVER kills running processes.
#
# - Logs TFRecord progress every POLL_SEC
# - AUTO_LAUNCH=1 (default): when ALL workers have stopped and TFRecords incomplete,
#   automatically starts a fresh 8-GPU fleet (resume from TFRecord truth)
# - Partial fleet (e.g. 2/8 still masking) is left alone — never killed
#
# Usage:
#   TARGET_GPUS=0,1,2,3,4,5,6,7 TARGET_NUM_WORKERS=8 \
#     nohup bash tools/watch_rlds_mask.sh >> logs/rlds_mask_<mix>/watchdog.log 2>&1 &

set -euo pipefail

DATA_MIX="${DATA_MIX:-libero_spatial_no_noops}"
TARGET_GPUS="${TARGET_GPUS:-0,1,2,3,4,5,6,7}"
TARGET_NUM_WORKERS="${TARGET_NUM_WORKERS:-8}"
POLL_SEC="${POLL_SEC:-60}"
AUTO_LAUNCH="${AUTO_LAUNCH:-1}"
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

count_workers() {
  pgrep -f "tools/rlds_mask.py.*${DATA_MIX}" 2>/dev/null | wc -l
}

count_tfrecord_done() {
  TOTAL_EPISODES="$TOTAL_EPISODES" "$PYTHON" - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__import__("os").environ["REPO_ROOT"]) / "tools"))
from rlds_mask_state import done_episodes_from_tfrecords

out = Path(__import__("os").environ["OUT_ROOT"])
mix = __import__("os").environ["DATA_MIX"]
total = int(__import__("os").environ.get("TOTAL_EPISODES", "432"))
print(len(done_episodes_from_tfrecords(out, mix, total_episodes=total)))
PY
}

count_resume_done() {
  local nw="${TARGET_NUM_WORKERS}"
  NUM_WORKERS="$nw" "$PYTHON" - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__import__("os").environ["REPO_ROOT"]) / "tools"))
from rlds_mask_state import done_episodes_from_resume_files

out = Path(__import__("os").environ["OUT_ROOT"])
mix = __import__("os").environ["DATA_MIX"]
nw = int(__import__("os").environ.get("NUM_WORKERS", "8"))
print(len(done_episodes_from_resume_files(out, mix, nw)))
PY
}

save_state() {
  echo "NUM_WORKERS=$TARGET_NUM_WORKERS" > "$STATE_FILE"
  echo "GPU_IDS=$TARGET_GPUS" >> "$STATE_FILE"
}

launch_workers() {
  if ! target_gpus_all_ok; then
    log "LAUNCH skipped: target GPUs ${TARGET_GPUS} not all healthy"
    return 1
  fi

  save_state
  log "LAUNCH: starting ${TARGET_NUM_WORKERS} workers on ${TARGET_GPUS} (resume, no kill)"

  "$PYTHON" "$REPO_ROOT/tools/init_multi_gpu_resume.py" \
    --data_mix "$DATA_MIX" --num_workers "$TARGET_NUM_WORKERS" --out_root "$OUT_ROOT"

  RESUME=1 BACKGROUND=1 bash "$REPO_ROOT/tools/rlds_mask_multi_gpu.sh" \
    "$DATA_MIX" "$TARGET_NUM_WORKERS" "$TARGET_GPUS"
  log "Multi-GPU launch on ${TARGET_GPUS}"
}

maybe_launch() {
  local proc tf_done
  proc="$(count_workers)"
  tf_done="$(count_tfrecord_done)"

  if (( proc > 0 )); then
    return 0
  fi
  if (( tf_done >= TOTAL_EPISODES )); then
    return 0
  fi
  if [[ "$AUTO_LAUNCH" != "1" ]]; then
    log "STOPPED: 0 workers, TFRecord ${tf_done}/${TOTAL_EPISODES} (AUTO_LAUNCH=0, not restarting)"
    return 0
  fi
  log "AUTO-RESTART: all workers stopped, TFRecord ${tf_done}/${TOTAL_EPISODES} — relaunching"
  launch_workers || log "Launch failed; will retry in ${POLL_SEC}s"
}

export OUT_ROOT DATA_MIX REPO_ROOT
log "Watchdog started: monitor + auto-restart on full stop (never kills) | target=${TARGET_GPUS} x${TARGET_NUM_WORKERS} auto_launch=${AUTO_LAUNCH} poll=${POLL_SEC}s"

while true; do
  tf_done="$(count_tfrecord_done)"
  resume_done="$(count_resume_done)"
  proc="$(count_workers)"

  if (( tf_done >= TOTAL_EPISODES )); then
    log "DONE: ${tf_done}/${TOTAL_EPISODES} episodes in TFRecord — exiting"
    exit 0
  fi

  if (( resume_done > tf_done )); then
    log "WARN: resume=${resume_done} > TFRecord=${tf_done} (resume ahead of disk)"
  fi

  if (( proc == 0 )); then
    log "IDLE: TFRecord ${tf_done}/${TOTAL_EPISODES} | 0 workers"
    maybe_launch
  elif (( proc < TARGET_NUM_WORKERS )); then
    log "OK: TFRecord ${tf_done}/${TOTAL_EPISODES} | ${proc}/${TARGET_NUM_WORKERS}w active (others finished — normal)"
  else
    log "OK: TFRecord ${tf_done}/${TOTAL_EPISODES} | ${proc}/${TARGET_NUM_WORKERS}w active"
  fi

  sleep "$POLL_SEC"
done
