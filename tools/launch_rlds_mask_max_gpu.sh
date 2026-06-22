#!/usr/bin/env bash
# Pick max idle GPUs and launch RLDS masking workers.
# Usage: BACKGROUND=1 bash tools/launch_rlds_mask_max_gpu.sh [data_mix]
set -euo pipefail

DATA_MIX="${1:-libero_spatial_no_noops}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${RLDS_DATA_ROOT:-/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/datasets/modified_libero_rlds}"
OUT_ROOT="${RLDS_OUT_ROOT:-/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/datasets/masked_libero_rlds}"
PYTHON="${RLDS_PYTHON:-/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/conda_envs/vla-preprocess/bin/python}"
MAX_GPUS="${MAX_GPUS:-8}"
UTIL_IDLE="${UTIL_IDLE:-15}"
MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"
NUM_SHARDS=16

pick_num_workers() {
  local n="$1"
  for w in 16 8 4 2 1; do
    if (( n >= w && NUM_SHARDS % w == 0 )); then
      echo "$w"
      return
    fi
  done
  echo 0
}

idle_gpu_indices() {
  local out=()
  while IFS=',' read -r idx util mem total; do
    idx="${idx// /}"
    util="${util//\%/}"
    util="${util// /}"
    util="${util//\[N\/A\]/0}"
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

echo "Scanning idle GPUs (util<${UTIL_IDLE}%, free>=${MIN_FREE_MIB}MiB)..."
read -ra idle_gpus <<< "$(idle_gpu_indices)"
n_idle=${#idle_gpus[@]}

if (( n_idle == 0 )); then
  echo "No idle GPUs found."
  exit 1
fi

num_workers=$(pick_num_workers "$n_idle")
if (( num_workers == 0 )); then
  echo "Could not pick a valid worker count for ${n_idle} idle GPU(s)."
  exit 1
fi

selected=()
for (( i=0; i<num_workers; i++ )); do
  selected+=("${idle_gpus[$i]}")
done
gpu_csv=$(IFS=,; echo "${selected[*]}")

echo "Idle GPUs: [${idle_gpus[*]}]"
echo "Using ${num_workers} workers on GPUs: [${gpu_csv}]"

# Stop stale workers (if any)
if pgrep -f "tools/rlds_mask.py.*${DATA_MIX}" >/dev/null 2>&1; then
  echo "Stopping existing rlds_mask workers..."
  pkill -f "tools/rlds_mask.py.*${DATA_MIX}" || true
  sleep 3
fi

RESUME=1 BACKGROUND="${BACKGROUND:-1}" bash "$REPO_ROOT/tools/rlds_mask_multi_gpu.sh" \
  "$DATA_MIX" "$num_workers" "$gpu_csv"
