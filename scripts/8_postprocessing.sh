#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VISER_DIR="$REPO_ROOT/vis_scripts/viser_m"
VIS_SCRIPT="$VISER_DIR/vis.sh"
DATA_ROOT="$REPO_ROOT/data"

if [[ $# -lt 1 ]]; then
  echo "Usage: sh prep_rl.sh <split_or_path> [hmr_type]" >&2
  exit 1
fi

SPLIT_INPUT="${1%/}"
HMR_TYPE="${2:-gv}"
LOG_DIR="${LOG_DIR:-/tmp/vis_megasam_logs}"
mkdir -p "$LOG_DIR"

declare -a CANDIDATES=(
  "$SPLIT_INPUT"
  "${SPLIT_INPUT}_img"
  "$REPO_ROOT/$SPLIT_INPUT"
  "$REPO_ROOT/${SPLIT_INPUT}_img"
  "$DATA_ROOT/$SPLIT_INPUT"
  "$DATA_ROOT/${SPLIT_INPUT}_img"
)

DATA_PATH=""
for candidate in "${CANDIDATES[@]}"; do
  [[ -z "$candidate" ]] && continue
  if [[ -d "$candidate" ]]; then
    DATA_PATH="$(cd "$candidate" && pwd)"
    break
  fi
done

if [[ -z "$DATA_PATH" ]]; then
  echo "Could not locate data directory for '$SPLIT_INPUT'." >&2
  exit 1
fi

if [[ ! -x "$VIS_SCRIPT" ]]; then
  echo "vis.sh not found or not executable at: $VIS_SCRIPT" >&2
  exit 1
fi

pushd "$VISER_DIR" >/dev/null

shopt -s nullglob
seq_dirs=("$DATA_PATH"/*/)
shopt -u nullglob

if (( ${#seq_dirs[@]} == 0 )); then
  echo "No sequence folders found under $DATA_PATH" >&2
  popd >/dev/null
  exit 1
fi

for seq_dir in "${seq_dirs[@]}"; do
  seq_name="$(basename "${seq_dir%/}")"
  results_file="$REPO_ROOT/results/output/scene/${seq_name}_${HMR_TYPE}_sgd_cvd_hr.npz"
  [[ -f "$results_file" ]] || continue

  logfile="${LOG_DIR}/${seq_name}.log"
  HMR_TYPE="$HMR_TYPE" SAVE_MODE=on bash "$VIS_SCRIPT" "$seq_name" >"$logfile" 2>&1
done

popd >/dev/null
