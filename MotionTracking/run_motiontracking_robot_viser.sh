#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/poselib:$SCRIPT_DIR/isaac_utils${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    CONDA_PREFIX="$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)"
    export CONDA_PREFIX
fi

if [[ -d "$CONDA_PREFIX/lib" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

cd "$SCRIPT_DIR"
exec "$PYTHON_BIN" scripts/visualize_viser_robot.py "$@"
