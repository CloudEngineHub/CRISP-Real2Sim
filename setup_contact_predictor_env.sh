#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-crisp_contact}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CP_DIR="$SCRIPT_DIR/prep/Contact-Predictor"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required but was not found on PATH." >&2
    exit 1
fi

if [[ ! -d "$CP_DIR" ]]; then
    echo "Contact-Predictor not found at $CP_DIR" >&2
    exit 1
fi

echo "[1/4] Creating conda env '$ENV_NAME'"
CONDA_SOLVER=classic conda create -n "$ENV_NAME" python=3.10 -y

CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -z "$CONDA_BASE" || ! -d "$CONDA_BASE" ]]; then
    echo "Failed to resolve conda base directory." >&2
    exit 1
fi
ENV_PREFIX="${CONDA_BASE}/envs/${ENV_NAME}"
PYTHON_BIN="$ENV_PREFIX/bin/python"
PIP_BIN="$ENV_PREFIX/bin/pip"

echo "[2/4] Installing PyTorch"
"$PIP_BIN" install \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

echo "[3/4] Installing Contact-Predictor dependencies"
TMP_REQ="$(mktemp)"
trap 'rm -f "$TMP_REQ"' EXIT
rg -v '^git\\+https://github.com/facebookresearch/pytorch3d.git@stable$' \
    "$CP_DIR/requirements.txt" > "$TMP_REQ"
"$PIP_BIN" install -r "$TMP_REQ"
"$PIP_BIN" install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"

echo "[4/4] Done"
cat <<EOF

Environment ready.

Next:
  conda activate $ENV_NAME
  cd "$CP_DIR"
  bash fetch_data.sh hcontact-wScene
  cd "$SCRIPT_DIR"
  bash scripts/0_interactvlm.sh /abs/path/to/data/demo/wall-kicking stairs

EOF
