#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-crisp_rl}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MT_DIR="$ROOT/MotionTracking"
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/motiontracking"
ISAAC_ARCHIVE="$CACHE_DIR/IsaacGym_Preview_4_Package.tar.gz"
ISAAC_ROOT="$CACHE_DIR/isaacgym_preview4"

echo "[1/6] conda create -n $ENV_NAME python=3.8"
conda create -n "$ENV_NAME" python=3.8 -y

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "[2/6] install pytorch"
python -m pip install \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    "xformers>=0.0.27" \
    --index-url https://download.pytorch.org/whl/cu124
python -m pip install \
    torch-scatter \
    -f https://data.pyg.org/whl/torch-2.4.1+cu124.html

echo "[3/6] install runtime packages"
python -m pip install \
    hydra-core==1.3.2 \
    pytorch-lightning==2.0.0 \
    termcolor==2.4.0 \
    scikit-image==0.21.0 \
    opencv-python==4.13.0.92 \
    rtree==1.2.0 \
    trimesh==4.11.3 \
    protobuf==3.20.3 \
    vector_quantize_pytorch==1.14.8 \
    librosa==0.11.0 \
    wandb==0.24.2 \
    tensorboard==2.14.0 \
    ninja==1.11.1.4 \
    imageio==2.35.1 \
    pyyaml==6.0.3 \
    joblib==1.4.2 \
    rich==14.3.3 \
    tqdm==4.67.3 \
    typer==0.12.5 \
    typed-argument-parser==1.10.1 \
    matplotlib==3.6.3 \
    scipy==1.10.1 \
    transformers==4.46.3 \
    "viser==1.0.24"

echo "[4/6] install isaac gym"
mkdir -p "$CACHE_DIR"
if [[ ! -f "$ISAAC_ARCHIVE" ]]; then
    wget -q -O "$ISAAC_ARCHIVE" https://developer.nvidia.com/isaac-gym-preview-4
fi
if [[ ! -d "$ISAAC_ROOT/isaacgym/python" ]]; then
    mkdir -p "$ISAAC_ROOT"
    tar -xzf "$ISAAC_ARCHIVE" -C "$ISAAC_ROOT"
fi
python -m pip install --no-deps "$ISAAC_ROOT/isaacgym/python"

ISAAC_PKG="$(
python - <<'PY'
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("isaacgym")
if spec is None or spec.origin is None:
    raise SystemExit("isaacgym package path not found")
print(Path(spec.origin).resolve().parent)
PY
)"
mkdir -p "$ISAAC_PKG/_bindings"
rm -rf "$ISAAC_PKG/_bindings/src"
cp -R "$ISAAC_ROOT/isaacgym/python/isaacgym/_bindings/src" "$ISAAC_PKG/_bindings/"

echo "[5/6] install smplsim"
python -m pip install "git+https://github.com/ZhengyiLuo/SMPLSim.git@b5c08720503ad5fff64050c4d289c42d947fcf8d"
python -m pip install --no-build-isolation "git+https://github.com/mattloper/chumpy.git"

echo "[6/6] install local helpers"
python -m pip install --no-deps "$MT_DIR/isaac_utils"

cat <<EOF

Ready:
  conda activate $ENV_NAME
  cd $ROOT
  bash setups/validate_motiontracking_viser_env.sh
  cd $MT_DIR

EOF
