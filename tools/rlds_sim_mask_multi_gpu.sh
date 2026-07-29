#!/usr/bin/env bash
# Launch rlds_sim_mask.py on multiple workers (CPU MuJoCo; GPU ids for process isolation only).
#
# Usage:
#   BACKGROUND=1 bash tools/rlds_sim_mask_multi_gpu.sh libero_spatial_no_noops 4 4,5,6,7
#   bash tools/rlds_sim_mask_multi_gpu.sh libero_goal_no_noops 4 4,5,6,7

set -euo pipefail

DATA_MIX="${1:-libero_spatial_no_noops}"
NUM_WORKERS="${2:-4}"
GPU_IDS="${3:-4,5,6,7}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${RLDS_DATA_ROOT:-/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/datasets/modified_libero_rlds}"
OUT_ROOT="${RLDS_SIM_OUT_ROOT:-$REPO_ROOT/openvla-oft/datasets/simu_masked_libero_rlds}"
LOG_DIR="$REPO_ROOT/logs/rlds_sim_mask_${DATA_MIX}"
PYTHON="${RLDS_SIM_PYTHON:-/home/fan-test/miniconda3/envs/subopt/bin/python}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES=""
RESUME="${RESUME:-1}"
BACKGROUND="${BACKGROUND:-0}"
PERTURB_PROB="${PERTURB_PROB:-0.3}"
PERTURB_STRENGTH="${PERTURB_STRENGTH:-2}"

mkdir -p "$LOG_DIR"

IFS=',' read -ra GPUS <<< "$GPU_IDS"
if (( ${#GPUS[@]} != NUM_WORKERS )); then
  echo "GPU count (${#GPUS[@]}) must match NUM_WORKERS ($NUM_WORKERS)"
  exit 1
fi

EXTRA_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  EXTRA_ARGS+=(--resume)
fi

echo "Launching $NUM_WORKERS sim-mask workers for $DATA_MIX"
echo "  data:  $DATA_ROOT"
echo "  out:   $OUT_ROOT"
echo "  python: $PYTHON"
echo "  perturb_prob=$PERTURB_PROB perturb_strength=$PERTURB_STRENGTH resume=$RESUME"

PIDS=()
for (( i=0; i<NUM_WORKERS; i++ )); do
  # Pin worker to GPU slot for EGL isolation; sim mask itself is CPU-only.
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PYTHON" "$REPO_ROOT/tools/rlds_sim_mask.py" \
    --data_mix "$DATA_MIX" \
    --data_root "$DATA_ROOT" \
    --out_root "$OUT_ROOT" \
    --perturb_prob "$PERTURB_PROB" \
    --perturb_strength "$PERTURB_STRENGTH" \
    --num_workers "$NUM_WORKERS" \
    --worker_id "$i" \
    --seed $((42 + i)) \
    "${EXTRA_ARGS[@]}" \
    > "$LOG_DIR/worker_${i}.log" 2>&1 &
  PIDS+=($!)
  echo "  worker $i -> GPU ${GPUS[$i]} (PID ${PIDS[-1]}) log=$LOG_DIR/worker_${i}.log"
done

echo "${PIDS[*]}" > "$LOG_DIR/worker_pids.txt"

if [[ "$BACKGROUND" == "1" ]]; then
  echo "Workers running in background."
  echo "After all finish, finalize:"
  echo "  $PYTHON $REPO_ROOT/tools/rlds_sim_mask.py --data_mix $DATA_MIX --data_root $DATA_ROOT --out_root $OUT_ROOT --num_workers $NUM_WORKERS --finalize"
  exit 0
fi

echo "Waiting for workers..."
FAIL=0
for (( i=0; i<NUM_WORKERS; i++ )); do
  if ! wait "${PIDS[$i]}"; then
    echo "Worker $i FAILED — see $LOG_DIR/worker_${i}.log"
    FAIL=1
  else
    echo "Worker $i done."
  fi
done

if [[ "$FAIL" -ne 0 ]]; then
  exit 1
fi

"$PYTHON" "$REPO_ROOT/tools/rlds_sim_mask.py" \
  --data_mix "$DATA_MIX" \
  --data_root "$DATA_ROOT" \
  --out_root "$OUT_ROOT" \
  --num_workers "$NUM_WORKERS" \
  --finalize

echo "All done: $OUT_ROOT/$DATA_MIX/1.0.0/"
