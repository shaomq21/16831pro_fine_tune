#!/usr/bin/env bash
# Re-eval dual-masked adapters with *generated* masks matching training:
#   goal    → Grounded-DINO + SAM1  (--sam_backend sam1)
#   spatial → SAM3 + temporal tracker (--sam_backend sam3)
# object / study_scene4 were sim-only in training → skip.
#
# Env:
#   SUITES=goal,spatial  TRIALS=3
#   VLA_GPUS=7,0         MASK_GPUS=1,1   # physical GPUs for VLA / mask worker
#   SKIP_COLOR=1         RUN_BACKGROUND=1
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFT="${REPO_ROOT}/openvla-oft"
cd "${OFT}"

STORAGE_ROOT="${STORAGE_ROOT:-/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/conda_envs/simplevla/bin/python}"
export VLA_PREPROCESS_PY="${VLA_PREPROCESS_PY:-${STORAGE_ROOT}/conda_envs/vla-preprocess/bin/python}"
BASE_VLA="${OFT}/checkpoints/openvla-7b"

# Prefer home FS (docker volume nearly full)
OUT_ROOT="${OUT_ROOT:-${OFT}/runs/sam_mask_reeval}"
LOG_DIR="${OUT_ROOT}/logs"
NOTE="${NOTE:-sam_reeval_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_DIR}" "${OUT_ROOT}/summary"

export PYTHONPATH="${REPO_ROOT}/LIBERO:${OFT}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM=false

TRIALS="${TRIALS:-3}"
SKIP_COLOR="${SKIP_COLOR:-1}"
RUN_BACKGROUND="${RUN_BACKGROUND:-True}"
SUITES="${SUITES:-goal,spatial}"
# Separate physical GPUs: VLA vs mask worker
# goal/sam1 → cpu mask (slow but no GPU contention); spatial/sam3 → MASK_GPU
VLA_GPUS="${VLA_GPUS:-7,0}"
MASK_GPUS="${MASK_GPUS:-cpu,1}"

IFS=',' read -r -a SUITE_ARR <<< "${SUITES}"
IFS=',' read -r -a VLA_GPU_ARR <<< "${VLA_GPUS}"
IFS=',' read -r -a MASK_GPU_ARR <<< "${MASK_GPUS}"

ckpt_for_suite() {
  echo "${STORAGE_ROOT}/runs/openvla_adapters/openvla-7b+dual_masked_${1}+b4+lr-0.0005+lora-r32+dropout-0.0+lora-attn-only--suite_${1}_oft_lr"
}

task_suite_for() {
  case "$1" in
    goal) echo libero_goal ;;
    spatial) echo libero_spatial ;;
    *) echo "unsupported suite $1 (only goal/spatial have non-simu training masks)" >&2; return 1 ;;
  esac
}

sam_backend_for() {
  case "$1" in
    goal) echo sam1 ;;
    spatial) echo sam3 ;;
    *) echo sam1 ;;
  esac
}

MASTER_LOG="${LOG_DIR}/launcher_${NOTE}.log"
echo "===== $(date -Iseconds) START sam-mask reeval suites=${SUITES} trials=${TRIALS} =====" | tee "${MASTER_LOG}"
echo "  VLA_GPUS=${VLA_GPUS} MASK_GPUS=${MASK_GPUS} SKIP_COLOR=${SKIP_COLOR}" | tee -a "${MASTER_LOG}"

i=0
pids=()
for suite in "${SUITE_ARR[@]}"; do
  suite="$(echo "${suite}" | xargs)"
  [[ -z "${suite}" ]] && continue
  vla_gpu="${VLA_GPU_ARR[$((i % ${#VLA_GPU_ARR[@]}))]}"
  mask_gpu="${MASK_GPU_ARR[$((i % ${#MASK_GPU_ARR[@]}))]}"
  ckpt="$(ckpt_for_suite "${suite}")"
  tsuite="$(task_suite_for "${suite}")" || continue
  backend="$(sam_backend_for "${suite}")"
  tag="ours_${suite}_${backend}"
  slog="${LOG_DIR}/${tag}_${NOTE}.log"

  if [[ ! -s "${ckpt}/action_head--latest_checkpoint.pt" ]]; then
    echo "SKIP ${suite}: missing ckpt ${ckpt}" | tee -a "${MASTER_LOG}"
    continue
  fi

  (
    EXTRA_MASK_DEVICE=(--mask_device cpu)
    if [[ -n "${mask_gpu}" && "${mask_gpu}" != "cpu" && "${mask_gpu}" != "-" ]]; then
      export MASK_GPU="${mask_gpu}"
      EXTRA_MASK_DEVICE=(--mask_device cuda)
    else
      unset MASK_GPU
    fi
    if [[ "${backend}" == "sam3" ]]; then
      EXTRA_MASK_DEVICE=(--mask_device cuda)
      if [[ -z "${MASK_GPU:-}" ]]; then
        export MASK_GPU="${mask_gpu:-1}"
      fi
    fi
    echo "===== $(date -Iseconds) START ${tag} VLA_GPU=${vla_gpu} MASK_GPU=${MASK_GPU:-cpu} backend=${backend} =====" | tee -a "${slog}"
    CUDA_VISIBLE_DEVICES="${vla_gpu}" "${PYTHON}" experiments/robot/libero/run_libero_background_perturb_eval.py \
      --pretrained_checkpoint "${ckpt}" \
      --base_vla_path "${BASE_VLA}" \
      --task_suite_name "${tsuite}" \
      --tasks all \
      --model_label "ours_masked_${backend}" \
      --use_mask_for_policy True \
      --use_mask_from_env False \
      --sam_backend "${backend}" \
      --mask_alpha 0.35 \
      "${EXTRA_MASK_DEVICE[@]}" \
      --run_baseline True \
      --run_background "${RUN_BACKGROUND}" \
      --num_images_in_input 1 \
      --num_trials_per_task "${TRIALS}" \
      --use_proprio True \
      --use_l1_regression True \
      --lora_rank 32 \
      --center_crop False \
      --local_log_dir "${LOG_DIR}" \
      --run_id_note "${tag}_bg_${NOTE}" \
      --load_in_8bit False \
      2>&1 | tee -a "${slog}"
    bg_rc=${PIPESTATUS[0]}
    echo "===== $(date -Iseconds) END ${tag} bg_rc=${bg_rc} =====" | tee -a "${slog}"
    exit "${bg_rc}"
  ) &
  pids+=($!)
  echo "launched ${tag} VLA=${vla_gpu} MASK=${mask_gpu:-cpu} pid=${pids[$((${#pids[@]} - 1))]}" | tee -a "${MASTER_LOG}"
  i=$((i + 1))
done

ec=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then ec=1; fi
done
echo "===== $(date -Iseconds) sam-mask reeval finished ec=${ec} =====" | tee -a "${MASTER_LOG}"
echo "Logs: ${LOG_DIR}" | tee -a "${MASTER_LOG}"
exit "${ec}"
