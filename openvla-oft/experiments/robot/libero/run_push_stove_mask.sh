#!/usr/bin/env bash
# Push the plate to the front of the stove - "no stove" mask eval.
# Policy sees: stove region replaced by solid GREEN mask.
# Raw (left) video: stove region set to WHITE (no stove in recording).
# Run from openvla-oft root.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENVLA_OFT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${OPENVLA_OFT_ROOT}"

export PYTHONPATH="/home/ubuntu/16831pro_fine_tune/LIBERO:${PYTHONPATH:-}"

CURRENT_CHECKPOINT="${CURRENT_CHECKPOINT:-/home/ubuntu/runs/openvla_adapters/openvla-7b+libero_goal_no_noops+b8+lr-0.0001+lora-r8+dropout-0.0+lora-attn-only--13500_chkpt}"
BASE_VLA_PATH="${BASE_VLA_PATH:-${OPENVLA_OFT_ROOT}/checkpoints/openvla-7b}"
EXTRA_8BIT=""
[[ "${LOAD_8BIT}" != "0" ]] && EXTRA_8BIT="--load_in_8bit True"

echo "========== Push stove mask eval (green at stove, raw video white at stove) =========="
python experiments/robot/libero/run_libero_push_stove_mask_eval.py \
  --pretrained_checkpoint "${CURRENT_CHECKPOINT}" \
  --base_vla_path "${BASE_VLA_PATH}" \
  --model_label "current" \
  --use_mask_for_policy True \
  --num_trials_per_task 3 ${EXTRA_8BIT}

echo "========== Push stove mask eval finished =========="
