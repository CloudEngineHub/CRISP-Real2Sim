#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-crisp_contact}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CP_DIR="$ROOT/prep/Contact-Predictor"

echo "[1/3] conda create -n $ENV_NAME python=3.10"
conda create -n "$ENV_NAME" python=3.10 -y

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "[2/3] install pytorch"
python -m pip install \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

echo "[3/3] install contact predictor packages"
TMP_REQ="$(mktemp)"
trap 'rm -f "$TMP_REQ"' EXIT
rg -v '^git\\+https://github.com/facebookresearch/pytorch3d.git@stable$' \
    "$CP_DIR/requirements.txt" > "$TMP_REQ"
python -m pip install -r "$TMP_REQ"
python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"

cat <<EOF

Ready:
  conda activate $ENV_NAME
  cd $CP_DIR

EOF
