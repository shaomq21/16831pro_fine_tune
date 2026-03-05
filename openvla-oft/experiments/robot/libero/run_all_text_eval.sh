#!/usr/bin/env bash
# Text eval: two tasks with rephrased language (no mask).
# - push: "push the right flat-shaped object to the front of the leftmost flat-shaped object"
# - put:  "put the bowl on the right flat-shaped object"
# Run on OpenVLA 7B and OFT goal; video naming: model (openvla_7b / openvla_oft_goal) + full task text.
# Run from openvla-oft root.
#
# Optional env: BASE_VLA_PATH, LOAD_8BIT=0 to disable 8-bit.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENVLA_OFT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${OPENVLA_OFT_ROOT}"

export PYTHONPATH="/home/ubuntu/16831pro_fine_tune/LIBERO:${PYTHONPATH:-}"

BASE_VLA_PATH="${BASE_VLA_PATH:-${OPENVLA_OFT_ROOT}/checkpoints/openvla-7b}"
EXTRA_8BIT=""
[[ "${LOAD_8BIT}" != "0" ]] && EXTRA_8BIT="--load_in_8bit True"

echo "========== 1/2 OpenVLA OFT goal - text eval (no mask) =========="
python experiments/robot/libero/run_libero_text_eval.py \
  --pretrained_checkpoint "moojink/openvla-7b-oft-finetuned-libero-goal" \
  --base_vla_path "${BASE_VLA_PATH}" \
  --model_label "openvla_oft_goal" \
  --num_trials_per_task 10 ${EXTRA_8BIT}

echo "========== 2/2 OpenVLA 7B - text eval (no mask) =========="
python experiments/robot/libero/run_libero_text_eval.py \
  --pretrained_checkpoint "openvla/openvla-7b" \
  --base_vla_path "${BASE_VLA_PATH}" \
  --model_label "openvla_7b" \
  --num_trials_per_task 10 ${EXTRA_8BIT}

echo "========== Text eval (push/put rephrased) finished =========="
