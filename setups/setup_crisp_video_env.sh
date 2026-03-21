#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="crisp"
FETCH_ASSETS=0

for arg in "$@"; do
    case "$arg" in
        --with-assets) FETCH_ASSETS=1 ;;
        *) ENV_NAME="$arg" ;;
    esac
done

bash "$ROOT/setups/setup_crisp.sh" "$ENV_NAME"

if [[ "$FETCH_ASSETS" == "1" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
    bash "$ROOT/setups/fetch_crisp_assets.sh"
fi
