#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-crisp_nksr}"
SOURCE_ENV="${SOURCE_ENV:-crisp}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NKSR_DIR="$ROOT/prep/NKSR"

if [[ ! -d "$NKSR_DIR" ]]; then
    echo "NKSR repo not found at: $NKSR_DIR" >&2
    echo "Clone it first, for example:" >&2
    echo "  git clone https://github.com/nv-tlabs/NKSR.git $NKSR_DIR" >&2
    exit 1
fi

echo "[1/3] clone conda env: $SOURCE_ENV -> $ENV_NAME"
conda create --name "$ENV_NAME" --clone "$SOURCE_ENV" -y

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "[2/3] install NKSR runtime deps"
python -m pip install pykdtree python-pycg ninja pyntcloud

echo "[3/3] build NKSR"
cd "$NKSR_DIR"
python -m pip install --no-build-isolation package/

cat <<EOF

Ready:
  conda activate $ENV_NAME
  cd $ROOT/vis_scripts/viser_m
  bash run_nksr.sh <SEQ_NAME>

EOF
