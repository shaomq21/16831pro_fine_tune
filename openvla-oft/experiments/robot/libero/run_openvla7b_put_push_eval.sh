#!/usr/bin/env bash
# Eval OpenVLA 7B (base) on put + push only. Uses original LIBERO task prompt (no rewrites).
# Tasks: push_the_plate_to_the_front_of_the_stove (id=5), put_the_bowl_on_the_plate (id=8).
# Run from openvla-oft root.
#
# Optional env: LOAD_8BIT=1 (default) for 8-bit to save GPU memory.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENVLA_OFT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${OPENVLA_OFT_ROOT}"

export PYTHONPATH="/home/ubuntu/16831pro_fine_tune/LIBERO:${PYTHONPATH:-}"

EXTRA=""
[[ "${LOAD_8BIT}" != "0" ]] && EXTRA="--load_in_8bit True"

echo "========== OpenVLA 7B eval: put + push only, original prompt =========="
python experiments/robot/libero/run_libero_eval.py \
  --task_suite_name libero_goal \
  --task_ids "5,8" \
  --pretrained_checkpoint "openvla/openvla-7b" \
  --num_images_in_input 1 \
  --num_trials_per_task 10 \
  ${EXTRA}

echo "========== Done =========="
