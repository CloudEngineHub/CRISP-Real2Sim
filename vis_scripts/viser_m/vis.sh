#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <sequence_name>" >&2
  exit 1
fi

SEQ="$1"

HMR_TYPE="${HMR_TYPE:-gv}"
ROOT_DIR="../../results/output/scene/${SEQ}_${HMR_TYPE}_sgd_cvd_hr.npz"

# default OFF
SAVE_MODE="${SAVE_MODE:-off}"

case "${SAVE_MODE,,}" in
  on|true|1|yes|y)
    SAVE_FLAG="--save_mode"
    ;;
  off|false|0|no|n|"")
    SAVE_FLAG="--no-save_mode"
    ;;
  *)
    echo "Invalid SAVE_MODE='$SAVE_MODE' (use on/off or true/false)" >&2
    exit 2
    ;;
esac

python visualizer_megasam.py --data "$ROOT_DIR" --hmr_type "$HMR_TYPE" "$SAVE_FLAG"
