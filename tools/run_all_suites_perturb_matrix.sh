#!/usr/bin/env bash
# Multi-suite masked perturb matrix for ours ckpts:
#   suites: goal | object | spatial | study_scene4
#   conditions: baseline + bg-0/1/2 + color-0/1 (TRIALS each)
#   tasks: all suite tasks (study_scene4 = 4 book tasks)
#
# Env:
#   SUITES="object,spatial,study_scene4,goal"   # default skips goal if SKIP_GOAL=1
#   TRIALS=3  GPUS="2,4,7"  MODELS=ours
#   SKIP_COLOR=0  SKIP_GOAL=0
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFT="${REPO_ROOT}/openvla-oft"
cd "${OFT}"

STORAGE_ROOT="${STORAGE_ROOT:-/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/conda_envs/simplevla/bin/python}"
BASE_VLA="${OFT}/checkpoints/openvla-7b"

OUT_ROOT="${OUT_ROOT:-${STORAGE_ROOT}/runs/all_suites_perturb_matrix}"
LOG_DIR="${OUT_ROOT}/logs"
VIDEO_NOTE="${VIDEO_NOTE:-allsuites_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_DIR}" "${OUT_ROOT}/summary"

export PYTHONPATH="${REPO_ROOT}/LIBERO:${OFT}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM=false

TRIALS="${TRIALS:-3}"
SKIP_COLOR="${SKIP_COLOR:-0}"
SKIP_GOAL="${SKIP_GOAL:-0}"
LOAD_8BIT="${LOAD_8BIT:-0}"
if [[ "${LOAD_8BIT}" == "1" ]]; then
  EXTRA_8BIT=(--load_in_8bit True)
else
  EXTRA_8BIT=(--load_in_8bit False)
fi

# Default: object,spatial,study_scene4 first (goal already partially done)
DEFAULT_SUITES="object,spatial,study_scene4"
if [[ "${SKIP_GOAL}" != "1" ]]; then
  DEFAULT_SUITES="${DEFAULT_SUITES},goal"
fi
SUITES="${SUITES:-${DEFAULT_SUITES}}"
GPUS="${GPUS:-2,4,7}"
IFS=',' read -r -a SUITE_ARR <<< "${SUITES}"
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"

ckpt_for_suite() {
  local suite="$1"
  case "${suite}" in
    study_scene4)
      echo "${STORAGE_ROOT}/runs/openvla_adapters/openvla-7b+dual_masked_study_scene4+b2+lr-0.0005+lora-r32+dropout-0.0+lora-attn-only--suite_study_scene4_oft_lr"
      ;;
    *)
      echo "${STORAGE_ROOT}/runs/openvla_adapters/openvla-7b+dual_masked_${suite}+b4+lr-0.0005+lora-r32+dropout-0.0+lora-attn-only--suite_${suite}_oft_lr"
      ;;
  esac
}

task_suite_for() {
  case "$1" in
    goal) echo libero_goal ;;
    object) echo libero_object ;;
    spatial) echo libero_spatial ;;
    study_scene4) echo libero_90 ;;
    *) echo "unknown suite $1" >&2; return 1 ;;
  esac
}

tasks_for() {
  case "$1" in
    study_scene4)
      echo "pick up the book in the middle and place it on the cabinet shelf|pick up the book on the left and place it on top of the shelf|pick up the book on the right and place it on the cabinet shelf|pick up the book on the right and place it under the cabinet shelf"
      ;;
    *)
      echo "all"
      ;;
  esac
}

unnorm_extra_for() {
  case "$1" in
    study_scene4) echo --unnorm_key simu_libero_90_study_scene4_no_noops ;;
    *) echo "" ;;
  esac
}

MASTER_LOG="${LOG_DIR}/matrix_launcher_${VIDEO_NOTE}.log"
echo "===== $(date -Iseconds) START all-suites matrix suites=${SUITES} gpus=${GPUS} trials=${TRIALS} =====" | tee "${MASTER_LOG}"

i=0
pids=()
for suite in "${SUITE_ARR[@]}"; do
  suite="$(echo "${suite}" | xargs)"
  [[ -z "${suite}" ]] && continue
  gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
  ckpt="$(ckpt_for_suite "${suite}")"
  tsuite="$(task_suite_for "${suite}")"
  tasks="$(tasks_for "${suite}")"
  unorm_args="$(unnorm_extra_for "${suite}")"
  tag="ours_${suite}"
  slog="${LOG_DIR}/${tag}_${VIDEO_NOTE}.log"

  if [[ ! -s "${ckpt}/action_head--latest_checkpoint.pt" ]]; then
    echo "SKIP ${suite}: missing ckpt ${ckpt}" | tee -a "${MASTER_LOG}"
    continue
  fi

  (
    echo "===== $(date -Iseconds) START ${tag} GPU=${gpu} ckpt=${ckpt} =====" | tee -a "${slog}"
    # shellcheck disable=SC2086
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
      ${unorm_args} \
      "${EXTRA_8BIT[@]}" \
      2>&1 | tee -a "${slog}"
    bg_rc=${PIPESTATUS[0]}

    if [[ "${SKIP_COLOR}" != "1" ]]; then
      # shellcheck disable=SC2086
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
        ${unorm_args} \
        "${EXTRA_8BIT[@]}" \
        2>&1 | tee -a "${slog}"
      color_rc=${PIPESTATUS[0]}
    else
      color_rc=0
    fi
    echo "===== $(date -Iseconds) END ${tag} bg_rc=${bg_rc} color_rc=${color_rc} =====" | tee -a "${slog}"
  ) &
  pids+=($!)
  echo "launched ${tag} on GPU ${gpu} pid=${pids[$((${#pids[@]} - 1))]}" | tee -a "${MASTER_LOG}"
  i=$((i + 1))
done

ec=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then ec=1; fi
done
echo "===== $(date -Iseconds) Matrix finished ec=${ec} =====" | tee -a "${MASTER_LOG}"
echo "Logs: ${LOG_DIR}" | tee -a "${MASTER_LOG}"
exit "${ec}"
