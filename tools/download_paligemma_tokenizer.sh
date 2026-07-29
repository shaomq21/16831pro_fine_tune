#!/usr/bin/env bash
# Download google/paligemma-3b-pt-224 tokenizer to external disk (required for lerobot pi05 train).
#
# Prerequisite: accept license at https://huggingface.co/google/paligemma-3b-pt-224
# Then either:
#   export HF_TOKEN=hf_xxx
#   huggingface-cli login
#
# Usage:
#   source tools/pi05_storage_env.sh && bash tools/download_paligemma_tokenizer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=pi05_storage_env.sh
source "${SCRIPT_DIR}/pi05_storage_env.sh"

TOKENIZER_DIR="${PI05_CKPT_ROOT}/paligemma-3b-pt-224"
PYTHON="${PYTHON:-/home/fan-test/miniconda3/envs/wav_new/bin/python}"

mkdir -p "${TOKENIZER_DIR}"

if [[ -f "${TOKENIZER_DIR}/tokenizer_config.json" ]]; then
  echo "PaliGemma tokenizer ready at ${TOKENIZER_DIR}"
  exit 0
fi

if [[ -z "${HF_TOKEN:-}" ]] && [[ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  if [[ -f "${HOME}/.huggingface/token" ]]; then
    export HF_TOKEN="$(tr -d '[:space:]' < "${HOME}/.huggingface/token")"
  fi
fi

if [[ -z "${HF_TOKEN:-}" ]] && [[ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  cat <<EOF
ERROR: PaliGemma tokenizer not found and no HF token set.

LeRobot pi05 training needs the gated model google/paligemma-3b-pt-224.

1) Open https://huggingface.co/google/paligemma-3b-pt-224 and click "Agree and access"
   (403 = not approved yet; 401 = no token)
2) Create a read token: https://huggingface.co/settings/tokens
3) Run:
     export HF_TOKEN=hf_xxx
     bash tools/download_paligemma_tokenizer.sh

Tokenizer will be saved to: ${TOKENIZER_DIR}
EOF
  exit 1
fi

echo "Downloading google/paligemma-3b-pt-224 -> ${TOKENIZER_DIR}"
"$PYTHON" - <<PY
import os
from huggingface_hub import snapshot_download
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
path = snapshot_download(
    "google/paligemma-3b-pt-224",
    local_dir="${TOKENIZER_DIR}",
    local_dir_use_symlinks=False,
    token=token,
    allow_patterns=[
        "tokenizer*",
        "spiece.model",
        "special_tokens_map.json",
        "config.json",
        "preprocessor_config.json",
    ],
)
print("Downloaded to", path)
PY

if [[ ! -f "${TOKENIZER_DIR}/tokenizer_config.json" ]]; then
  echo "ERROR: download finished but tokenizer_config.json missing in ${TOKENIZER_DIR}"
  exit 1
fi
echo "PaliGemma tokenizer ready at ${TOKENIZER_DIR}"
