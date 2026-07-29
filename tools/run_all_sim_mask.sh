#!/usr/bin/env bash
# Sequentially collect sim-masked RLDS for LIBERO suites + libero_90 STUDY_SCENE4 books.
#
# Usage:
#   BACKGROUND=1 bash tools/run_all_sim_mask.sh
#   GPU_IDS=4,5,6,7 bash tools/run_all_sim_mask.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DATA_ROOT="${RLDS_DATA_ROOT:-/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/datasets/modified_libero_rlds}"
OUT_ROOT="${RLDS_SIM_OUT_ROOT:-$REPO_ROOT/openvla-oft/datasets/simu_masked_libero_rlds}"
LOG_ROOT="$REPO_ROOT/logs/run_all_sim_mask"
PYTHON="${RLDS_SIM_PYTHON:-/home/fan-test/miniconda3/envs/subopt/bin/python}"

SUITES=(
  libero_spatial_no_noops
  libero_goal_no_noops
  libero_object_no_noops
  libero_10_no_noops
  libero_90_study_scene4_no_noops
)
if [[ -n "${SUITES_OVERRIDE:-}" ]]; then
  read -ra SUITES <<< "$SUITES_OVERRIDE"
fi

mkdir -p "$LOG_ROOT"

_run_suite() {
  local mix="$1"
  local src="$DATA_ROOT/$mix/1.0.0"
  if [[ ! -d "$src" ]]; then
    echo "SKIP $mix — source not found: $src"
    return 0
  fi
  echo "========== $mix =========="
  BACKGROUND=0 RESUME=1 GPU_IDS="$GPU_IDS" NUM_WORKERS="$NUM_WORKERS" \
    RLDS_DATA_ROOT="$DATA_ROOT" RLDS_SIM_OUT_ROOT="$OUT_ROOT" RLDS_SIM_PYTHON="$PYTHON" \
    bash "$REPO_ROOT/tools/rlds_sim_mask_multi_gpu.sh" "$mix" "$NUM_WORKERS" "$GPU_IDS" \
    2>&1 | tee "$LOG_ROOT/${mix}.log"
}

if [[ "${BACKGROUND:-0}" == "1" ]]; then
  nohup bash "$0" > "$LOG_ROOT/master.log" 2>&1 &
  echo "Started all-suite sim mask collection (PID $!). Log: $LOG_ROOT/master.log"
  exit 0
fi

for mix in "${SUITES[@]}"; do
  _run_suite "$mix"
done

echo "All four suites done -> $OUT_ROOT"
