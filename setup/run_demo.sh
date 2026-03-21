#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${1:-crisp}"
INPUT_ROOT="${2:---demo}"

bash "$ROOT/setup_crisp.sh" "$ENV_NAME"
bash "$ROOT/fetch_crisp_assets.sh"

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

bash "$ROOT/run_crisp_video.sh" "$INPUT_ROOT"
