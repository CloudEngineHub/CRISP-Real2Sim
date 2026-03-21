#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-crisp}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[1/7] conda create -n $ENV_NAME python=3.10"
conda create -n "$ENV_NAME" python=3.10 -y

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

export CUDA_HOME="$CONDA_PREFIX"
export CUB_HOME="$CONDA_PREFIX/include"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MAX_JOBS="${MAX_JOBS:-4}"

echo "[2/7] install cuda toolkit"
conda install -y \
    -c nvidia/label/cuda-12.4.1 \
    -c defaults \
    cuda-toolkit=12.4.1 \
    cuda-compiler=12.4.1 \
    cuda-command-line-tools=12.4.1 \
    cuda-nvcc=12.4.131

echo "[3/7] install build tools"
conda install -y -c conda-forge make ninja cmake

echo "[4/7] install python packages"
python -m pip install -r "$ROOT/requirements-crisp-video.txt"
python -m pip install --no-deps nerfstudio==0.3.3
python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
python -m pip install --no-build-isolation "git+https://github.com/mattloper/chumpy.git"

echo "[5/7] install local packages"
SAM2_BUILD_ALLOW_ERRORS=1 python -m pip install --no-deps -e "$ROOT/prep/AutoMask"
python -m pip install --no-deps -e "$ROOT/prep/HMR"
python -m pip install --no-deps -e "$ROOT/prep/UFM/UniCeption"
python -m pip install --no-deps -e "$ROOT/prep/UFM"
python -m pip install --no-deps -e "$ROOT/vis_scripts/viser_m"
python -m pip install numpy==1.26.4 opencv-python==4.6.0.66

echo "[6/7] build mogesam extensions"
(
    cd "$ROOT/prep/MogeSAM/third_party/pointops2"
    LIBRARY_PATH="$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}" python setup.py install
)
(
    cd "$ROOT/prep/MogeSAM/third_party/megasam/base"
    LIBRARY_PATH="$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}" python setup.py install
)

echo "[7/7] install geocalib"
if [[ ! -d "$ROOT/prep/GeoCalib/.git" ]]; then
    git clone https://github.com/hongsukchoi/GeoCalib.git "$ROOT/prep/GeoCalib"
fi
python -m pip install -e "$ROOT/prep/GeoCalib"

cat <<EOF

Ready:
  conda activate $ENV_NAME
  bash setups/validate_crisp_video_env.sh

EOF
