#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <sequence_name> [detailed_planes] [static_camera] [segment_mode] [sq_loss_threshold] [save_mode] [save_clustering]" >&2
    exit 1
fi

HMR_TYPE="${HMR_TYPE:-gv}"
ROOT_DIR="../../results/output/scene/${1}_${HMR_TYPE}_sgd_cvd_hr.npz"


python visualizer_megasam.py --data "$ROOT_DIR" --hmr_type "${HMR_TYPE}" 