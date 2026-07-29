#!/usr/bin/env bash
# Download lerobot/pi05_libero to external disk: $PI05_CKPT_ROOT/pi05_libero
#
# Usage:
#   bash tools/download_pi05_libero.sh
#   HF_TOKEN=hf_xxx bash tools/download_pi05_libero.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=pi05_storage_env.sh
source "${SCRIPT_DIR}/pi05_storage_env.sh"

PYTHON="${PYTHON:-/home/fan-test/miniconda3/envs/wav_new/bin/python}"

# Patch transformers for pi05 (idempotent)
OPENPI_ROOT="${SCRIPT_DIR}/../third_party/openpi"
if [[ -d "${OPENPI_ROOT}/src/openpi/models_pytorch/transformers_replace" ]]; then
  TRANSFORMERS_DIR="$("$PYTHON" -c "import transformers, pathlib; print(pathlib.Path(transformers.__file__).parent)")"
  cp -r "${OPENPI_ROOT}/src/openpi/models_pytorch/transformers_replace/"* "${TRANSFORMERS_DIR}/" 2>/dev/null || true
fi

DEST="${PI05_PRETRAINED}"
mkdir -p "${DEST}"

if [[ -f "${DEST}/model.safetensors" ]]; then
  echo "pi05_libero ready at ${DEST} ($(du -sh "${DEST}/model.safetensors" | cut -f1))"
  ls -lh "${DEST}/"
  exit 0
fi

# Recover from prior partial downloads on external disk (avoid re-downloading 14G)
_LEGACY_MODEL="${HF_HOME}/models--lerobot--pi05_libero/blobs/21b8711787c4a75861b02cff6aa81675a3a943d32b435a68262ac4461e476ba4"
_HUB_SNAP=$(find "${HF_HUB_CACHE}/models--lerobot--pi05_libero/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)

if [[ -f "${_LEGACY_MODEL}" && -n "${_HUB_SNAP}" && -f "${_HUB_SNAP}/config.json" ]]; then
  echo "Assembling pi05_libero from existing external cache -> ${DEST}"
  for f in config.json policy_preprocessor.json policy_postprocessor.json README.md; do
    if [[ -e "${_HUB_SNAP}/${f}" ]]; then
      cp -L "${_HUB_SNAP}/${f}" "${DEST}/${f}"
    fi
  done
  cp -a "${_LEGACY_MODEL}" "${DEST}/model.safetensors"
  echo "Assembled $(du -sh "${DEST}" | cut -f1) at ${DEST}"
  ls -lh "${DEST}/"
  exit 0
fi

echo "Downloading lerobot/pi05_libero -> ${DEST}"
echo "  HF_HUB_CACHE=${HF_HUB_CACHE}"

"$PYTHON" - <<PY
from huggingface_hub import snapshot_download
from pathlib import Path
dest = Path("${DEST}")
path = snapshot_download(
    "lerobot/pi05_libero",
    local_dir=str(dest),
    local_dir_use_symlinks=False,
)
print("Downloaded to", path)
PY

echo "Done: ${DEST}"
ls -lh "${DEST}/"
