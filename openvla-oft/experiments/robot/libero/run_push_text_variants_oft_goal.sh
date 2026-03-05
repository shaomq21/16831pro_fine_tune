#!/usr/bin/env bash
# Run OFT-goal on push task only with two text variants (no mask):
# 1) "push the plate to the front of a flat cuboid" (stove -> a flat cuboid)
# 2) "push the plate on the right to the front of the stove" (plate -> plate on the right)
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

echo "========== OFT-goal: push task with stove->flat cuboid & plate->plate on the right =========="
python experiments/robot/libero/run_libero_text_eval.py \
  --pretrained_checkpoint "moojink/openvla-7b-oft-finetuned-libero-goal" \
  --base_vla_path "${BASE_VLA_PATH}" \
  --model_label "openvla_oft_goal" \
  --task_subset "push" \
  --use_push_stove_plate_variants True \
  --num_trials_per_task 10 ${EXTRA_8BIT}

echo "========== Push text variants (OFT-goal) finished =========="
