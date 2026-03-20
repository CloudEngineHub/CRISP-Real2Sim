#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

ENV_NAME="${1:-crisp}"
INPUT_ROOT="${2:---demo}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required but was not found on PATH." >&2
    exit 1
fi

CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -z "$CONDA_BASE" || ! -d "$CONDA_BASE" ]]; then
    echo "Failed to resolve conda base directory." >&2
    exit 1
fi
ENV_PREFIX="$CONDA_BASE/envs/$ENV_NAME"

bash "$REPO_ROOT/setup_crisp_video_env.sh" "$ENV_NAME" --with-assets

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    echo "Failed to resolve python in env: $ENV_PREFIX" >&2
    exit 1
fi

export PATH="$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

bash "$REPO_ROOT/run_crisp_video.sh" "$INPUT_ROOT"
