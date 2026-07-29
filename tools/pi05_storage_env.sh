#!/usr/bin/env bash
# Source this before any pi0.5 download / train / feature-extract command.
# Keeps checkpoints and HF cache on the large external volume, not $HOME or /.

# Usage: source tools/pi05_storage_env.sh

STORAGE_ROOT="${STORAGE_ROOT:-/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune}"

export STORAGE_ROOT
export PIP_CACHE_DIR="${STORAGE_ROOT}/pip_cache"
export HF_HOME="${STORAGE_ROOT}/hf_cache"
export HF_HUB_CACHE="${STORAGE_ROOT}/hf_cache/hub"
export HUGGINGFACE_HUB_CACHE="${STORAGE_ROOT}/hf_cache/hub"
export TRANSFORMERS_CACHE="${STORAGE_ROOT}/hf_cache"
export HF_LEROBOT_HOME="${STORAGE_ROOT}/lerobot_datasets"
export HF_DATASETS_CACHE="${STORAGE_ROOT}/hf_cache/datasets"

# pi0.5 checkpoint layout on external disk
export PI05_CKPT_ROOT="${STORAGE_ROOT}/ckpts/pi05"
export PI05_PRETRAINED="${PI05_PRETRAINED:-${PI05_CKPT_ROOT}/pi05_libero}"
export PI05_PALIGEMMA_TOKENIZER="${PI05_CKPT_ROOT}/paligemma-3b-pt-224"
export PI05_FINETUNE_ROOT="${STORAGE_ROOT}/runs/pi05_study_scene4_finetune"
export PI05_ANALYSIS_ROOT="${STORAGE_ROOT}/runs/pi05_study_scene4_analysis"
export PI05_LEROBOT_DATASET="${HF_LEROBOT_HOME}/local/libero_90_study_scene4"

mkdir -p \
  "${PIP_CACHE_DIR}" \
  "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" \
  "${HF_LEROBOT_HOME}" \
  "${PI05_CKPT_ROOT}" \
  "${PI05_ANALYSIS_ROOT}" \
  "${STORAGE_ROOT}/datasets" \
  "${STORAGE_ROOT}/runs"

# Redirect ~/.cache/huggingface -> external (only if not already a symlink)
_MAIN_HF="${HOME}/.cache/huggingface"
if [[ -L "${_MAIN_HF}" ]]; then
  :
elif [[ ! -e "${_MAIN_HF}" ]]; then
  mkdir -p "$(dirname "${_MAIN_HF}")"
  ln -sfn "${HF_HOME}" "${_MAIN_HF}"
elif [[ -d "${_MAIN_HF}" ]]; then
  echo "[pi05_storage_env] NOTE: ${_MAIN_HF} exists on main disk ($(du -sh "${_MAIN_HF}" 2>/dev/null | cut -f1))."
  echo "  HF ops use ${HF_HOME} via env vars; old ~/.cache/huggingface is NOT auto-migrated."
fi

pi05_storage_env() {
  echo "STORAGE_ROOT=${STORAGE_ROOT}"
  echo "PI05_CKPT_ROOT=${PI05_CKPT_ROOT}"
  echo "PI05_PRETRAINED=${PI05_PRETRAINED}"
  echo "HF_HUB_CACHE=${HF_HUB_CACHE}"
}
