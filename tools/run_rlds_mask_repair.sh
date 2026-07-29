#!/usr/bin/env bash
# Repair re-run: sync resume from TFRecord, launch 8-GPU mask, start watchdog (auto-restart on full stop).
#
# Usage:
#   bash tools/run_rlds_mask_repair.sh
#   bash tools/run_rlds_mask_repair.sh libero_spatial_no_noops

set -euo pipefail

DATA_MIX="${1:-libero_spatial_no_noops}"
TARGET_GPUS="${TARGET_GPUS:-0,1,2,3,4,5,6,7}"
TARGET_NUM_WORKERS="${TARGET_NUM_WORKERS:-8}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs/rlds_mask_${DATA_MIX}"
OUT_ROOT="${RLDS_OUT_ROOT:-/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/datasets/masked_libero_rlds}"
PYTHON="${RLDS_PYTHON:-/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/conda_envs/vla-preprocess/bin/python}"

mkdir -p "$LOG_DIR"

echo "=== RLDS mask repair: ${DATA_MIX} ==="

if pgrep -f "tools/rlds_mask.py.*${DATA_MIX}" >/dev/null 2>&1; then
  n="$(pgrep -f "tools/rlds_mask.py.*${DATA_MIX}" | wc -l)"
  echo "Workers already running (${n}); skip launch, ensure watchdog only."
else
  "$PYTHON" "$REPO_ROOT/tools/init_multi_gpu_resume.py" \
    --data_mix "$DATA_MIX" --num_workers "$TARGET_NUM_WORKERS" --out_root "$OUT_ROOT"
  DEBUG_EVERY_EPISODES=0 RESUME=1 BACKGROUND=1 \
    bash "$REPO_ROOT/tools/rlds_mask_multi_gpu.sh" "$DATA_MIX" "$TARGET_NUM_WORKERS" "$TARGET_GPUS"
fi

# Single watchdog instance
pkill -f "watch_rlds_mask.sh" 2>/dev/null || true
sleep 1
TARGET_GPUS="$TARGET_GPUS" TARGET_NUM_WORKERS="$TARGET_NUM_WORKERS" AUTO_LAUNCH=1 \
  nohup bash "$REPO_ROOT/tools/watch_rlds_mask.sh" >> "$LOG_DIR/watchdog.log" 2>&1 &
echo "Watchdog PID=$! (auto-restart when all workers stop, never kills)"

echo ""
echo "Monitor:"
echo "  python $REPO_ROOT/tools/monitor_rlds_mask_progress.py --data_mix $DATA_MIX --num_workers $TARGET_NUM_WORKERS"
echo "Watchdog log:"
echo "  tail -f $LOG_DIR/watchdog.log"
