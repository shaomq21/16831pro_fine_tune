#!/usr/bin/env bash
# Color perturbation eval: push (plate color) + put (bowl color).
# Each task: 2 color variants (each detected color → 2 possible output colors), 3 trials per variant.
# Video naming: push/put + model_label + "-color0" or "-color1".
# Current (mask) / OpenVLA OFT goal (no mask) / OpenVLA 7B (no mask). Run from openvla-oft root.
#
# Optional env: CURRENT_CHECKPOINT, BASE_VLA_PATH. Default 8-bit; set LOAD_8BIT=0 to disable.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENVLA_OFT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${OPENVLA_OFT_ROOT}"

export PYTHONPATH="/home/ubuntu/16831pro_fine_tune/LIBERO:${PYTHONPATH:-}"

# Paths (override with env if needed)
CURRENT_CHECKPOINT="${CURRENT_CHECKPOINT:-/home/ubuntu/runs/openvla_adapters/openvla-7b+libero_goal_no_noops+b8+lr-0.0001+lora-r8+dropout-0.0+lora-attn-only--13500_chkpt}"
BASE_VLA_PATH="${BASE_VLA_PATH:-${OPENVLA_OFT_ROOT}/checkpoints/openvla-7b}"
EXTRA_8BIT=""
[[ "${LOAD_8BIT}" != "0" ]] && EXTRA_8BIT="--load_in_8bit True"

# echo "========== 1/3 Current model (with mask - pass masked images to policy) =========="
# python experiments/robot/libero/run_libero_color_perturb_eval.py \
#   --pretrained_checkpoint "${CURRENT_CHECKPOINT}" \
#   --base_vla_path "${BASE_VLA_PATH}" \
#   --model_label "current" \
#   --use_mask_for_policy True \
#   --use_mask_from_env False \
#   --num_trials_per_task 3 ${EXTRA_8BIT}

echo "========== 2/3 OpenVLA OFT goal (libero_goal) - Color perturb, no mask =========="
python experiments/robot/libero/run_libero_color_perturb_eval.py \
  --pretrained_checkpoint "moojink/openvla-7b-oft-finetuned-libero-goal" \
  --base_vla_path "${BASE_VLA_PATH}" \
  --model_label "openvla_oft_goal" \
  --use_mask_for_policy False \
  --num_images_in_input 2 \
  --num_trials_per_task 1 ${EXTRA_8BIT}

echo "========== 3/3 OpenVLA 7B base - Color perturb, no mask =========="
python experiments/robot/libero/run_libero_color_perturb_eval.py \
  --pretrained_checkpoint "openvla/openvla-7b" \
  --base_vla_path "${BASE_VLA_PATH}" \
  --model_label "openvla_7b" \
  --use_mask_for_policy False \
  --num_trials_per_task 3 ${EXTRA_8BIT}

echo "========== Color perturb experiments finished =========="
