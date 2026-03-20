#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    cat <<'EOF' >&2
Usage:
  bash bridge_crisp_sequence.sh <SEQ_NAME> [DATE_TAG] [METHOD] [HMR_TYPE] [bridge args...]

Example:
  bash bridge_crisp_sequence.sh 40_indoor_walk_big_circle bridge0318 ours gv

Extra args are forwarded to scripts/bridge_crisp_to_motiontracking.py.
EOF
    exit 1
fi

SEQ_NAME="$1"
DATE_TAG="${2:-bridge}"
METHOD="${3:-ours}"
HMR_TYPE="${4:-gv}"

if [[ $# -ge 4 ]]; then
    shift 4
else
    shift $#
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/poselib:$SCRIPT_DIR/isaac_utils:$SCRIPT_DIR/smpllib${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

cd "$SCRIPT_DIR"
exec "$PYTHON_BIN" scripts/bridge_crisp_to_motiontracking.py \
    "$SEQ_NAME" \
    --date "$DATE_TAG" \
    --method "$METHOD" \
    --hmr-type "$HMR_TYPE" \
    "$@"
