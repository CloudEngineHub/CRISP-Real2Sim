#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-crisp_rl}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
MT_DIR="$REPO_ROOT/MotionTracking"
CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/motiontracking-viser-env"
ISAACGYM_ARCHIVE="$CACHE_ROOT/IsaacGym_Preview_4_Package.tar.gz"
ISAACGYM_ROOT="$CACHE_ROOT/isaacgym_preview4"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required but was not found on PATH." >&2
    exit 1
fi

mkdir -p "$CACHE_ROOT"

CONDA_BASE="$(
    python - <<'PY'
from conda.base.context import context
print(context.root_prefix)
PY
)"
ENV_PREFIX="${CONDA_BASE}/envs/${ENV_NAME}"

if [[ -x "${ENV_PREFIX}/bin/python" ]]; then
    echo "[1/8] Reusing existing conda env '$ENV_NAME'"
else
    echo "[1/8] Creating conda env '$ENV_NAME' with Python 3.8"
    CONDA_SOLVER=classic conda create -n "$ENV_NAME" python=3.8 -y
fi

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    ENV_PREFIX="$(
        conda run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)' 2>/dev/null | tail -n 1
    )"
fi

PYTHON_BIN="$ENV_PREFIX/bin/python"
PIP_BIN="$ENV_PREFIX/bin/pip"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Failed to resolve python for conda env '$ENV_NAME'." >&2
    exit 1
fi

echo "[2/8] Installing CUDA 12.4 PyTorch stack"
"$PIP_BIN" install \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    "xformers>=0.0.27" \
    --index-url https://download.pytorch.org/whl/cu124
"$PIP_BIN" install \
    torch-scatter \
    -f https://data.pyg.org/whl/torch-2.4.1+cu124.html

echo "[3/8] Installing MotionTracking runtime dependencies"
"$PIP_BIN" install \
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
    transformers==4.46.3

echo "[4/8] Downloading and installing Isaac Gym Preview 4"
if [[ ! -f "$ISAACGYM_ARCHIVE" ]]; then
    wget -q -O "$ISAACGYM_ARCHIVE" https://developer.nvidia.com/isaac-gym-preview-4
fi
if [[ ! -d "$ISAACGYM_ROOT/isaacgym/python" ]]; then
    mkdir -p "$ISAACGYM_ROOT"
    tar -xzf "$ISAACGYM_ARCHIVE" -C "$ISAACGYM_ROOT"
fi
"$PIP_BIN" install --no-deps "$ISAACGYM_ROOT/isaacgym/python"

ISAACGYM_PKG_DIR="$("$PYTHON_BIN" - <<'PY'
from pathlib import Path
import isaacgym
print(Path(isaacgym.__file__).resolve().parent)
PY
)"
ISAACGYM_SRC_DIR="$ISAACGYM_ROOT/isaacgym/python/isaacgym/_bindings/src"
if [[ -d "$ISAACGYM_SRC_DIR" ]]; then
    mkdir -p "$ISAACGYM_PKG_DIR/_bindings"
    rm -rf "$ISAACGYM_PKG_DIR/_bindings/src"
    cp -R "$ISAACGYM_SRC_DIR" "$ISAACGYM_PKG_DIR/_bindings/"
fi

echo "[5/8] Installing SMPLSim"
"$PIP_BIN" install \
    "git+https://github.com/ZhengyiLuo/SMPLSim.git@b5c08720503ad5fff64050c4d289c42d947fcf8d"
"$PIP_BIN" install --no-build-isolation \
    "git+https://github.com/mattloper/chumpy.git"

echo "[6/8] Installing repo-local helpers"
"$PIP_BIN" install --no-deps "$MT_DIR/isaac_utils"

echo "[7/8] Installing robot-viser visualization dependencies"
"$PIP_BIN" install "viser==1.0.24"

echo "[8/8] Validating MotionTracking + robot-viser"
CONDA_PREFIX="$ENV_PREFIX" \
    bash "$REPO_ROOT/validate_motiontracking_viser_env.sh" "$PYTHON_BIN"

cat <<EOF

Environment ready.

Important:
- This setup intentionally does NOT run \`pip install MotionTracking\`, because the current
  \`MotionTracking/setup.py\` forces \`torch==2.0.1\` and breaks the Isaac Gym + cu124 stack.
- It also does NOT install \`MotionTracking/poselib\` as a wheel, because the current
  packaging omits subpackages. The wrappers source it through \`PYTHONPATH\`.

Usage:
  conda activate $ENV_NAME
  cd "$MT_DIR"
  bash run_crisp_test.sh --smoke-only
  bash run_motiontracking_robot_viser.sh /abs/path/to/record_dir --port 8080

The wrappers will also set:
  PYTHONPATH += MotionTracking, MotionTracking/poselib, MotionTracking/isaac_utils
  LD_LIBRARY_PATH += \$CONDA_PREFIX/lib

EOF
