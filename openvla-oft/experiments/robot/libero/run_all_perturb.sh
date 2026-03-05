#!/usr/bin/env bash
# Run all perturbation evals: background perturb (3 models) + color perturb (2 models).
# Run from openvla-oft root. Optional env: CURRENT_CHECKPOINT, BASE_VLA_PATH, LOAD_8BIT=1

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENVLA_OFT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${OPENVLA_OFT_ROOT}"

export PYTHONPATH="/home/ubuntu/16831pro_fine_tune/LIBERO:${PYTHONPATH:-}"

echo "########################################"
echo "# Part 1/2: Background perturbation    #"
echo "# (current, openvla_oft_goal, openvla_7b)#"
echo "########################################"
"${SCRIPT_DIR}/run_all_bg_perturb.sh"

echo ""
echo "########################################"
echo "# Part 2/2: Color perturbation         #"
echo "# (openvla_oft_goal, openvla_7b)       #"
echo "########################################"
"${SCRIPT_DIR}/run_all_color_perturb.sh"

echo ""
echo "========== All perturbation experiments finished =========="
