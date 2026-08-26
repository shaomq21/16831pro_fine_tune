#!/usr/bin/env bash
# Launch stable RL on libero_goal tasks. Default skips already-strong #4/#5/#8.
# #1/#2 retries use greedy-BC + skip-update-when-high-SR.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
mkdir -p openvla-oft/logs

SKIP_DONE="${SKIP_DONE:-1}"
NOTE_PREFIX="${NOTE_PREFIX:-allg}"
NUM_ITERS="${NUM_ITERS:-6}"

# tid:gpu:mode
JOBS=(
  "0:0:colors"
  "1:1:bowl"
  "2:2:colors"
  "3:3:bowl"
  "6:4:colors"
  "7:5:colors"
  "9:6:colors"
)
if [[ "${RUN_ALL:-0}" == "1" || "${SKIP_DONE}" != "1" ]]; then
  JOBS+=("4:3:bowl" "5:7:colors" "8:4:bowl")
fi

pkill -f 'run_rl_lora_color_quick.py' 2>/dev/null || true
sleep 2

for spec in "${JOBS[@]}"; do
  IFS=':' read -r tid gpu mode <<<"$spec"
  note="${NOTE_PREFIX}_t${tid}"
  export SUITE=goal TASK_ID="$tid" GPU="$gpu" NOTE="$note" PERTURB_MODE="$mode"
  export NUM_ITERS NUM_GROUPS=3 GROUP_SIZE=4 EVAL_TRIALS=4
  export SKIP_BASELINE_EVAL=True PREFER_GREEDY_BC=True GREEDY_FRAC=0.5
  export SKIP_UPDATE_ABOVE_SR=0.85 NOISE=0.03 LR_AH=1e-5
  # Stronger protection for failed #1/#2 and typically-saturated wine/stove
  if [[ "$tid" == "1" || "$tid" == "2" || "$tid" == "7" || "$tid" == "9" ]]; then
    export NOISE=0.02 SKIP_UPDATE_ABOVE_SR=0.80 LR_AH=5e-6 GREEDY_FRAC=0.5
  fi
  echo "LAUNCH task=$tid gpu=$gpu mode=$mode note=$note noise=$NOISE skip_above=$SKIP_UPDATE_ABOVE_SR"
  nohup bash tools/run_rl_lora_color_quick.sh \
    > "openvla-oft/logs/rl_lora_color_goal_task${tid}_${note}.launch.log" 2>&1 &
  echo "  pid=$!"
  sleep 3
done

echo "===== launched ${#JOBS[@]} jobs ====="
sleep 5
pgrep -af 'run_rl_lora_color_quick.py' | rg 'task_id' | head -20 || true
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -8
