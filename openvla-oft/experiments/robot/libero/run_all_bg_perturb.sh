#!/usr/bin/env bash
# Run all background-perturb experiments. Only your (current) model uses mask; openvla_oft_goal and 7b use raw image.
# Current (mask) / OpenVLA OFT goal (no mask) / OpenVLA 7B (no mask). Each: 2 tasks × 4 conditions × 3 trials = 24 episodes. Run from openvla-oft root.
#
# Optional env: CURRENT_CHECKPOINT, BASE_VLA_PATH. Default: 8-bit for all three models (14GB GPU); set LOAD_8BIT=0 to disable.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENVLA_OFT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${OPENVLA_OFT_ROOT}"

# Required for 'libero' module (LIBERO repo path)
export PYTHONPATH="/home/ubuntu/16831pro_fine_tune/LIBERO:${PYTHONPATH:-}"

# Paths (override with env if needed)
CURRENT_CHECKPOINT="${CURRENT_CHECKPOINT:-/home/ubuntu/runs/openvla_adapters/openvla-7b+libero_goal_no_noops+b8+lr-0.0001+lora-r8+dropout-0.0+lora-attn-only--13500_chkpt}"
BASE_VLA_PATH="${BASE_VLA_PATH:-${OPENVLA_OFT_ROOT}/checkpoints/openvla-7b}"
# Default 8-bit for all three models to avoid CUDA OOM on 14GB GPU; set LOAD_8BIT=0 to use bf16
EXTRA_8BIT=""
[[ "${LOAD_8BIT}" != "0" ]] && EXTRA_8BIT="--load_in_8bit True"

echo "========== 1/3 Current model (with mask) =========="
# python experiments/robot/libero/run_libero_background_perturb_eval.py \
#   --pretrained_checkpoint "${CURRENT_CHECKPOINT}" \
#   --base_vla_path "${BASE_VLA_PATH}" \
#   --model_label "current" \
#   --use_mask_for_policy True \
#   --num_trials_per_task 3 ${EXTRA_8BIT}

# # OpenVLA OFT goal: skip baseline (already verified), only perturb
# echo "========== 2/3 OpenVLA OFT goal (no mask) =========="
# python experiments/robot/libero/run_libero_background_perturb_eval.py \
#   --pretrained_checkpoint "moojink/openvla-7b-oft-finetuned-libero-goal" \
#   --base_vla_path "${BASE_VLA_PATH}" \
#   --model_label "openvla_oft_goal" \
#   --use_mask_for_policy False \
#   --run_baseline False \
#   --num_images_in_input 2 \
#   --num_trials_per_task 1 ${EXTRA_8BIT}

echo "========== 3/3 OpenVLA 7B base (no mask) =========="
# Base 7B expects num_images_in_input=1 (single image); OFT uses 2 (full+wrist)
python experiments/robot/libero/run_libero_background_perturb_eval.py \
  --pretrained_checkpoint "openvla/openvla-7b" \
  --model_label "openvla_7b" \
  --use_mask_for_policy False \
  --run_baseline False \
  --num_images_in_input 1 \
  --num_trials_per_task 3 ${EXTRA_8BIT}

echo "========== All experiments finished =========="

