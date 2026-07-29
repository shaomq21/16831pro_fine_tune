#!/usr/bin/env bash
# Re-eval after rescue finetune: only low-SR / rescued tasks.
# Suites in parallel on GPUS (default 2,3,4,5). Conditions: baseline+bg+color, TRIALS each.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFT="${REPO_ROOT}/openvla-oft"
cd "${OFT}"

STORAGE_ROOT="${STORAGE_ROOT:-/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/conda_envs/simplevla/bin/python}"
BASE_VLA="${OFT}/checkpoints/openvla-7b"

OUT_ROOT="${OUT_ROOT:-${STORAGE_ROOT}/runs/rescue_reeval}"
LOG_DIR="${OUT_ROOT}/logs"
VIDEO_NOTE="${VIDEO_NOTE:-rescue_reeval_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_DIR}" "${OUT_ROOT}/summary"

export PYTHONPATH="${REPO_ROOT}/LIBERO:${OFT}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM=false

TRIALS="${TRIALS:-3}"
GPUS="${GPUS:-2,3,4,5}"
SKIP_COLOR="${SKIP_COLOR:-0}"
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"

GOAL_TASKS="push the plate to the front of the stove|open the middle drawer of the cabinet|open the top drawer and put the bowl inside|put the bowl on top of the cabinet|put the cream cheese in the bowl|put the wine bottle on the rack|put the wine bottle on top of the cabinet|turn on the stove"
OBJECT_TASKS="pick up the alphabet soup and place it in the basket|pick up the chocolate pudding and place it in the basket|pick up the cream cheese and place it in the basket|pick up the ketchup and place it in the basket|pick up the milk and place it in the basket|pick up the tomato sauce and place it in the basket"
SPATIAL_TASKS="pick up the black bowl from table center and place it on the plate|pick up the black bowl next to the cookie box and place it on the plate|pick up the black bowl next to the plate and place it on the plate|pick up the black bowl next to the ramekin and place it on the plate|pick up the black bowl on the ramekin and place it on the plate|pick up the black bowl on the stove and place it on the plate|pick up the black bowl on the wooden cabinet and place it on the plate|pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate|pick up the black bowl on the cookie box and place it on the plate"
STUDY_TASKS="pick up the book on the left and place it on top of the shelf|pick up the book in the middle and place it on the cabinet shelf|pick up the book on the right and place it on the cabinet shelf|pick up the book on the right and place it under the cabinet shelf"

ckpt_for() {
  case "$1" in
    study_scene4)
      echo "${STORAGE_ROOT}/runs/openvla_adapters/openvla-7b+dual_masked_study_scene4+b2+lr-0.0005+lora-r32+dropout-0.0+lora-attn-only--suite_study_scene4_oft_lr"
      ;;
    *)
      echo "${STORAGE_ROOT}/runs/openvla_adapters/openvla-7b+dual_masked_$1+b4+lr-0.0005+lora-r32+dropout-0.0+lora-attn-only--suite_$1_oft_lr"
      ;;
  esac
}

MASTER="${LOG_DIR}/launcher_${VIDEO_NOTE}.log"
echo "===== $(date -Iseconds) START rescue reeval trials=${TRIALS} gpus=${GPUS} =====" | tee "${MASTER}"

run_suite() {
  local suite="$1" gpu="$2" tasks="$3" tsuite="$4"
  local ckpt unorm=()
  ckpt="$(ckpt_for "${suite}")"
  if [[ "${suite}" == "study_scene4" ]]; then
    unorm=(--unnorm_key simu_libero_90_study_scene4_no_noops)
  fi
  local tag="rescue_${suite}"
  local slog="${LOG_DIR}/${tag}_${VIDEO_NOTE}.log"
  echo "===== $(date -Iseconds) START ${tag} GPU=${gpu} =====" | tee -a "${slog}" "${MASTER}"
  if [[ ! -s "${ckpt}/action_head--latest_checkpoint.pt" ]]; then
    echo "MISSING ckpt ${ckpt}" | tee -a "${slog}" "${MASTER}"
    return 1
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" experiments/robot/libero/run_libero_background_perturb_eval.py \
    --pretrained_checkpoint "${ckpt}" \
    --base_vla_path "${BASE_VLA}" \
    --task_suite_name "${tsuite}" \
    --tasks "${tasks}" \
    --model_label "ours_masked" \
    --use_mask_for_policy True \
    --use_mask_from_env True \
    --run_baseline True \
    --run_background True \
    --num_images_in_input 1 \
    --num_trials_per_task "${TRIALS}" \
    --use_proprio True \
    --use_l1_regression True \
    --lora_rank 32 \
    --center_crop False \
    --local_log_dir "${LOG_DIR}" \
    --run_id_note "${tag}_bg_${VIDEO_NOTE}" \
    --load_in_8bit False \
    "${unorm[@]}" \
    2>&1 | tee -a "${slog}"
  local bg_rc=${PIPESTATUS[0]}

  local color_rc=0
  if [[ "${SKIP_COLOR}" != "1" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" experiments/robot/libero/run_libero_color_perturb_eval.py \
      --pretrained_checkpoint "${ckpt}" \
      --base_vla_path "${BASE_VLA}" \
      --task_suite_name "${tsuite}" \
      --tasks "${tasks}" \
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
      --run_id_note "${tag}_color_${VIDEO_NOTE}" \
      --load_in_8bit False \
      "${unorm[@]}" \
      2>&1 | tee -a "${slog}"
    color_rc=${PIPESTATUS[0]}
  fi
  echo "===== $(date -Iseconds) END ${tag} bg_rc=${bg_rc} color_rc=${color_rc} =====" | tee -a "${slog}" "${MASTER}"
  return $(( bg_rc != 0 || color_rc != 0 ))
}

pids=()
run_suite goal "${GPU_ARR[0]}" "${GOAL_TASKS}" libero_goal &
pids+=($!)
run_suite object "${GPU_ARR[1]}" "${OBJECT_TASKS}" libero_object &
pids+=($!)
run_suite spatial "${GPU_ARR[2]}" "${SPATIAL_TASKS}" libero_spatial &
pids+=($!)
run_suite study_scene4 "${GPU_ARR[3]}" "${STUDY_TASKS}" libero_90 &
pids+=($!)

echo "launched pids=${pids[*]}" | tee -a "${MASTER}"
ec=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then ec=1; fi
done
echo "===== $(date -Iseconds) rescue reeval finished ec=${ec} =====" | tee -a "${MASTER}"
echo "Logs: ${LOG_DIR}" | tee -a "${MASTER}"
exit "${ec}"
