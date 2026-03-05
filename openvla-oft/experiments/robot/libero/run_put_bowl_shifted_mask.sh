#!/usr/bin/env bash
# Put bowl on plate - shifted plate mask eval.
# Mask: green region BESIDE the plate (same shape), not on the plate.
# Tests if current model follows the mask or the real scene.
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

echo "========== Put bowl on plate - shifted plate mask (current model) =========="
python experiments/robot/libero/run_libero_put_bowl_shifted_mask_eval.py \
  --pretrained_checkpoint "${CURRENT_CHECKPOINT}" \
  --base_vla_path "${BASE_VLA_PATH}" \
  --model_label "current" \
  --use_mask_for_policy True \
  --shift_plate_pixels 80 \
  --num_trials_per_task 3 ${EXTRA_8BIT}

echo "========== Shifted plate mask eval finished =========="
