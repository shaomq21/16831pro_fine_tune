#!/usr/bin/env bash
# Copy the perturb-matrix RESULTS.md + scenario-grid rollout videos into docs/
# so they can be committed / pushed to GitHub.
#
#   bash tools/sync_eval_docs.sh
#   PUSH=1 bash tools/sync_eval_docs.sh   # also git add + commit + push
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_RESULTS="${SRC_RESULTS:-${REPO_ROOT}/openvla-oft/runs/all_suites_perturb_matrix/summary/RESULTS.md}"
SRC_GRIDS="${SRC_GRIDS:-${REPO_ROOT}/openvla-oft/rollouts/scenario_grids}"
DEST_DOCS="${REPO_ROOT}/docs"
DEST_GRIDS="${DEST_DOCS}/scenario_grids"

if [[ ! -f "${SRC_RESULTS}" ]]; then
  echo "missing RESULTS.md: ${SRC_RESULTS}" >&2
  exit 1
fi
if [[ ! -d "${SRC_GRIDS}" ]]; then
  echo "missing scenario grids: ${SRC_GRIDS}" >&2
  exit 1
fi

mkdir -p "${DEST_GRIDS}"
cp -f "${SRC_RESULTS}" "${DEST_DOCS}/RESULTS.md"
rsync -a --delete --include='*.mp4' --include='manifest.json' --exclude='*' \
  "${SRC_GRIDS}/" "${DEST_GRIDS}/"

HEADER=$'> GitHub copy of `openvla-oft/runs/all_suites_perturb_matrix/summary/RESULTS.md`.\n> Rollout matrix videos: [`scenario_grids/`](./scenario_grids/).\n'
if ! grep -q 'GitHub copy of' "${DEST_DOCS}/RESULTS.md"; then
  tmp="$(mktemp)"
  printf '%s\n' "${HEADER}" | cat - "${DEST_DOCS}/RESULTS.md" > "${tmp}"
  mv "${tmp}" "${DEST_DOCS}/RESULTS.md"
fi

echo "synced:"
echo "  ${DEST_DOCS}/RESULTS.md"
echo "  ${DEST_GRIDS}/ ($(find "${DEST_GRIDS}" -name '*.mp4' | wc -l) mp4)"

if [[ "${PUSH:-0}" == "1" ]]; then
  cd "${REPO_ROOT}"
  git add docs/RESULTS.md docs/scenario_grids
  if git diff --cached --quiet; then
    echo "docs already up to date; nothing to commit"
  else
    git commit -m "$(cat <<'EOF'
Sync eval RESULTS.md and rollout grid videos into docs.

EOF
)"
  fi
  git push origin HEAD
fi
