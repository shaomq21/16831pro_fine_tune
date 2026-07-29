#!/usr/bin/env bash
# Goal-suite perturb matrix:
#   origin | lang-l1 | lang-l2 | l1+bg-0/1/2 | l1+color-1/2 (color0/1)
# Models: ours (masked), openvla-oft. Pi launched separately.
# Videos: side-by-side raw|masked. Default TRIALS=5 (paper-friendly vs speed).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFT="${REPO_ROOT}/openvla-oft"
cd "${OFT}"

STORAGE_ROOT="${STORAGE_ROOT:-/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/conda_envs/simplevla/bin/python}"
BASE_VLA="${OFT}/checkpoints/openvla-7b"
OUR_CKPT="${STORAGE_ROOT}/runs/openvla_adapters/openvla-7b+dual_masked_goal+b4+lr-0.0005+lora-r32+dropout-0.0+lora-attn-only--suite_goal_oft_lr"
OFT_CKPT="${OFT_CKPT:-moojink/openvla-7b-oft-finetuned-libero-goal}"

OUT_ROOT="${OUT_ROOT:-${STORAGE_ROOT}/runs/goal_perturb_matrix}"
LOG_DIR="${OUT_ROOT}/logs"
VIDEO_NOTE="${VIDEO_NOTE:-matrix_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_DIR}" "${OUT_ROOT}/summary"

export PYTHONPATH="${REPO_ROOT}/LIBERO:${OFT}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${STORAGE_ROOT}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export VLA_PREPROCESS_PY="${VLA_PREPROCESS_PY:-${STORAGE_ROOT}/conda_envs/vla-preprocess/bin/python}"

TRIALS="${TRIALS:-5}"
LOAD_8BIT="${LOAD_8BIT:-0}"
# Always pass explicitly: text_eval defaults load_in_8bit=True (needs bitsandbytes).
if [[ "${LOAD_8BIT}" == "1" ]]; then
  EXTRA_8BIT=(--load_in_8bit True)
else
  EXTRA_8BIT=(--load_in_8bit False)
fi

# Parallel: ours on GPU_OURS, oft on GPU_OFT
MODELS="${MODELS:-ours,oft}"
GPU_OURS="${GPU_OURS:-0}"
GPU_OFT="${GPU_OFT:-1}"

run_one() {
  local gpu="$1"; shift
  local tag="$1"; shift
  local logfile="${LOG_DIR}/${tag}.log"
  echo "===== $(date -Iseconds) START ${tag} GPU=${gpu} =====" | tee -a "${logfile}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "$@" 2>&1 | tee -a "${logfile}"
  local rc=${PIPESTATUS[0]}
  echo "===== $(date -Iseconds) END ${tag} rc=${rc} =====" | tee -a "${logfile}"
  return 0  # continue matrix even if one job fails
}

run_ours() {
  # Masked lang unchanged for l1/l2 — one bg+color pass covers origin & visual conditions.
  run_one "${GPU_OURS}" "ours_bg_origin_l1visual" \
    experiments/robot/libero/run_libero_background_perturb_eval.py \
    --pretrained_checkpoint "${OUR_CKPT}" \
    --base_vla_path "${BASE_VLA}" \
    --model_label "ours_masked" \
    --use_mask_for_policy True \
    --use_mask_from_env True \
    --run_baseline True \
    --num_images_in_input 1 \
    --num_trials_per_task "${TRIALS}" \
    --use_proprio True \
    --use_l1_regression True \
    --lora_rank 32 \
    --center_crop False \
    --local_log_dir "${LOG_DIR}" \
    --run_id_note "ours_bg_${VIDEO_NOTE}" \
    "${EXTRA_8BIT[@]}"

  run_one "${GPU_OURS}" "ours_color_l1visual" \
    experiments/robot/libero/run_libero_color_perturb_eval.py \
    --pretrained_checkpoint "${OUR_CKPT}" \
    --base_vla_path "${BASE_VLA}" \
    --model_label "ours_masked" \
    --use_mask_for_policy True \
    --use_mask_from_env True \
    --num_images_in_input 1 \
    --num_trials_per_task "${TRIALS}" \
    --use_proprio True \
    --use_l1_regression True \
    --lora_rank 32 \
    --center_crop False \
    --local_log_dir "${LOG_DIR}" \
    --run_id_note "ours_color_${VIDEO_NOTE}" \
    "${EXTRA_8BIT[@]}"
}

run_oft() {
  # origin only (raw lang, no visual perturb)
  run_one "${GPU_OFT}" "oft_origin" \
    experiments/robot/libero/run_libero_background_perturb_eval.py \
    --pretrained_checkpoint "${OFT_CKPT}" \
    --base_vla_path "${BASE_VLA}" \
    --model_label "openvla_oft_origin" \
    --use_mask_for_policy False \
    --lang_mode origin \
    --run_baseline True \
    --run_background False \
    --num_images_in_input 2 \
    --num_trials_per_task "${TRIALS}" \
    --local_log_dir "${LOG_DIR}" \
    --run_id_note "oft_origin_${VIDEO_NOTE}" \
    "${EXTRA_8BIT[@]}"

  # lang-l1 only (no visual) via text map; also l1+bg + l1+color
  run_one "${GPU_OFT}" "oft_text_l1" \
    experiments/robot/libero/run_libero_text_eval.py \
    --pretrained_checkpoint "${OFT_CKPT}" \
    --base_vla_path "${BASE_VLA}" \
    --model_label "openvla_oft_l1" \
    --num_images_in_input 2 \
    --num_trials_per_task "${TRIALS}" \
    --local_log_dir "${LOG_DIR}" \
    --run_id_note "oft_l1_${VIDEO_NOTE}" \
    "${EXTRA_8BIT[@]}"

  run_one "${GPU_OFT}" "oft_text_l2" \
    experiments/robot/libero/run_libero_text_eval.py \
    --pretrained_checkpoint "${OFT_CKPT}" \
    --base_vla_path "${BASE_VLA}" \
    --model_label "openvla_oft_l2" \
    --use_push_stove_plate_variants True \
    --task_subset push \
    --num_images_in_input 2 \
    --num_trials_per_task "${TRIALS}" \
    --local_log_dir "${LOG_DIR}" \
    --run_id_note "oft_l2_${VIDEO_NOTE}" \
    "${EXTRA_8BIT[@]}"

  run_one "${GPU_OFT}" "oft_bg_under_l1" \
    experiments/robot/libero/run_libero_background_perturb_eval.py \
    --pretrained_checkpoint "${OFT_CKPT}" \
    --base_vla_path "${BASE_VLA}" \
    --model_label "openvla_oft_l1" \
    --use_mask_for_policy False \
    --lang_mode l1 \
    --run_baseline False \
    --num_images_in_input 2 \
    --num_trials_per_task "${TRIALS}" \
    --local_log_dir "${LOG_DIR}" \
    --run_id_note "oft_bg_l1_${VIDEO_NOTE}" \
    "${EXTRA_8BIT[@]}"

  run_one "${GPU_OFT}" "oft_color_under_l1" \
    experiments/robot/libero/run_libero_color_perturb_eval.py \
    --pretrained_checkpoint "${OFT_CKPT}" \
    --base_vla_path "${BASE_VLA}" \
    --model_label "openvla_oft_l1" \
    --use_mask_for_policy False \
    --lang_mode l1 \
    --num_images_in_input 2 \
    --num_trials_per_task "${TRIALS}" \
    --local_log_dir "${LOG_DIR}" \
    --run_id_note "oft_color_l1_${VIDEO_NOTE}" \
    "${EXTRA_8BIT[@]}"
}

PIDS=()
if [[ ",${MODELS}," == *",ours,"* ]]; then
  run_ours &
  PIDS+=($!)
fi
if [[ ",${MODELS}," == *",oft,"* ]]; then
  run_oft &
  PIDS+=($!)
fi

ec=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || ec=1
done

echo "===== Matrix finished $(date -Iseconds) ec=${ec} ====="
echo "Logs: ${LOG_DIR}"
rg -n 'Final:|Overall success|Success rate|=== Final' "${LOG_DIR}"/*.{log,txt} 2>/dev/null \
  | tee "${OUT_ROOT}/summary/success_snip.txt" || true
exit 0
