#!/usr/bin/env bash
# Launch rlds_mask.py on multiple GPUs in parallel.
# Usage:
#   BACKGROUND=1 bash tools/rlds_mask_multi_gpu.sh libero_spatial_no_noops 4 2,4,1,0
#   bash tools/rlds_mask_multi_gpu.sh libero_spatial_no_noops 8 0,1,2,3,4,5,6,7

set -euo pipefail

DATA_MIX="${1:-libero_spatial_no_noops}"
NUM_GPUS="${2:-4}"
GPU_IDS="${3:-}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${RLDS_DATA_ROOT:-/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/datasets/modified_libero_rlds}"
OUT_ROOT="${RLDS_OUT_ROOT:-/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/datasets/masked_libero_rlds}"
LOG_DIR="$REPO_ROOT/logs/rlds_mask_${DATA_MIX}"
PYTHON="${RLDS_PYTHON:-/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/conda_envs/vla-preprocess/bin/python}"
export GRIPPER_SIM_PYTHON="${GRIPPER_SIM_PYTHON:-/home/fan-test/miniconda3/envs/subopt/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
FAST="${FAST:-1}"
RESUME="${RESUME:-1}"
BACKGROUND="${BACKGROUND:-0}"
DEBUG_EVERY_EPISODES="${DEBUG_EVERY_EPISODES:-1}"
DEBUG_FRAMES="${DEBUG_FRAMES:-1}"
MAX_DEBUG_IMAGES="${MAX_DEBUG_IMAGES:-20}"

mkdir -p "$LOG_DIR"

if [[ -n "$GPU_IDS" ]]; then
  IFS=',' read -ra GPUS <<< "$GPU_IDS"
  NUM_GPUS="${#GPUS[@]}"
else
  GPUS=()
  for (( i=0; i<NUM_GPUS; i++ )); do
    GPUS+=("$i")
  done
fi

if (( NUM_GPUS == 0 )); then
  echo "No GPUs specified"
  exit 1
fi

EXTRA_ARGS=()
if [[ "$FAST" == "1" ]]; then
  EXTRA_ARGS+=(--fast)
fi
if [[ "$RESUME" == "1" ]]; then
  EXTRA_ARGS+=(--resume)
  "$PYTHON" "$REPO_ROOT/tools/init_multi_gpu_resume.py" \
    --data_mix "$DATA_MIX" --num_workers "$NUM_GPUS" --out_root "$OUT_ROOT"
fi

echo "Launching $NUM_GPUS workers for $DATA_MIX on GPUs: ${GPUS[*]}"
echo "  data:  $DATA_ROOT"
echo "  out:   $OUT_ROOT"
echo "  fast=$FAST resume=$RESUME debug_every_episodes=$DEBUG_EVERY_EPISODES debug_frames=$DEBUG_FRAMES"

PIDS=()
for (( i=0; i<NUM_GPUS; i++ )); do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PYTHON" "$REPO_ROOT/tools/rlds_mask.py" \
    --data_mix "$DATA_MIX" \
    --data_root "$DATA_ROOT" \
    --out_root "$OUT_ROOT" \
    --debug_dir "rlds_mask_debug/${DATA_MIX}" \
    --debug_every_episodes "$DEBUG_EVERY_EPISODES" \
    --debug_frames "$DEBUG_FRAMES" \
    --max_debug_images "$MAX_DEBUG_IMAGES" \
    --no_mask_wrist \
    --num_workers "$NUM_GPUS" \
    --worker_id "$i" \
    --device cuda:0 \
    "${EXTRA_ARGS[@]}" \
    > "$LOG_DIR/worker_${i}.log" 2>&1 &
  PIDS+=($!)
  echo "  worker $i -> GPU ${GPUS[$i]} (PID ${PIDS[-1]}) log=$LOG_DIR/worker_${i}.log"
done

echo "${PIDS[*]}" > "$LOG_DIR/worker_pids.txt"

if [[ "$BACKGROUND" == "1" ]]; then
  echo "Workers running in background. Monitor:"
  echo "  python $REPO_ROOT/tools/monitor_rlds_mask_progress.py --data_mix $DATA_MIX --num_workers $NUM_GPUS"
  echo "After all finish, run finalize:"
  echo "  $PYTHON $REPO_ROOT/tools/rlds_mask.py --data_mix $DATA_MIX --data_root $DATA_ROOT --out_root $OUT_ROOT --num_workers $NUM_GPUS --finalize"
  exit 0
fi

echo "Waiting for workers..."
FAIL=0
for (( i=0; i<NUM_GPUS; i++ )); do
  if ! wait "${PIDS[$i]}"; then
    echo "Worker $i FAILED — see $LOG_DIR/worker_${i}.log"
    FAIL=1
  else
    echo "Worker $i done."
  fi
done

if [[ "$FAIL" -ne 0 ]]; then
  echo "Some workers failed; not running --finalize"
  exit 1
fi

"$PYTHON" "$REPO_ROOT/tools/rlds_mask.py" \
  --data_mix "$DATA_MIX" \
  --data_root "$DATA_ROOT" \
  --out_root "$OUT_ROOT" \
  --num_workers "$NUM_GPUS" \
  --finalize

echo "All done: $OUT_ROOT/$DATA_MIX/1.0.0/"
