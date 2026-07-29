#!/usr/bin/env bash
# Launch (or relaunch) all suite watchers with multi-GPU defaults.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFT="${REPO_ROOT}/openvla-oft"
mkdir -p "${OFT}/logs"

start_watcher () {
  local suite="$1"
  unset CUDA_VISIBLE_DEVICES NUM_GPUS
  SUITE="${suite}"
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/suite_gpu_layout.sh"
  local pidfile="${OFT}/logs/finetune_suite_${suite}_watch.pid"
  if [[ -f "${pidfile}" ]]; then
    local oldpid
    oldpid=$(cat "${pidfile}" 2>/dev/null || true)
    if [[ -n "${oldpid}" ]] && kill -0 "${oldpid}" 2>/dev/null; then
      echo "Stopping old ${suite} watcher pid=${oldpid}"
      kill "${oldpid}" 2>/dev/null || true
      sleep 2
    fi
  fi
  echo "Starting ${suite} watcher: GPUs=${CUDA_VISIBLE_DEVICES} nproc=${NUM_GPUS}"
  nohup env SUITE="${suite}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" NUM_GPUS="${NUM_GPUS}" \
    bash "${SCRIPT_DIR}/finetune_suite_watch.sh" >> "${OFT}/logs/finetune_suite_${suite}_watch.log" 2>&1 &
}

for suite in spatial goal object study_scene4; do
  start_watcher "${suite}"
done

echo "All suite watchers launched. PIDs:"
pgrep -af "finetune_suite_watch.sh" || true
