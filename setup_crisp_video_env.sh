#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/crisp-video-env"
TORCH_HUB_DIR="${TORCH_HOME:-$HOME/.cache/torch}/hub"
TORCH_HUB_CHECKPOINTS="$TORCH_HUB_DIR/checkpoints"
PIP_CONSTRAINTS="$CACHE_ROOT/pip-constraints.txt"
PIP_REQUIREMENTS="$REPO_ROOT/requirements-crisp-video.txt"
VALIDATE_SCRIPT="$REPO_ROOT/validate_crisp_video_env.sh"

usage() {
    cat <<'EOF'
Usage:
  bash setup_crisp_video_env.sh [env_name]
  bash setup_crisp_video_env.sh [env_name] --with-assets

Options:
  --with-assets     Also download demo checkpoints and torch.hub caches.
                    By default this script installs the core CRISP environment only.
EOF
}

ENV_NAME="crisp"
FETCH_ASSETS=0
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-assets)
            FETCH_ASSETS=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ ${#POSITIONAL_ARGS[@]} -gt 1 ]]; then
    echo "Too many positional arguments." >&2
    usage >&2
    exit 2
fi

if [[ ${#POSITIONAL_ARGS[@]} -eq 1 ]]; then
    ENV_NAME="${POSITIONAL_ARGS[0]}"
fi

TOTAL_STEPS=8
if (( FETCH_ASSETS == 1 )); then
    TOTAL_STEPS=10
fi

STEP_INDEX=1
log_step() {
    echo "[$STEP_INDEX/$TOTAL_STEPS] $1"
    STEP_INDEX=$((STEP_INDEX + 1))
}

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required but was not found on PATH." >&2
    exit 1
fi

if [[ ! -f "$VALIDATE_SCRIPT" ]]; then
    echo "Validation script not found: $VALIDATE_SCRIPT" >&2
    exit 1
fi

mkdir -p "$CACHE_ROOT" "$TORCH_HUB_DIR" "$TORCH_HUB_CHECKPOINTS"

CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -z "$CONDA_BASE" || ! -d "$CONDA_BASE" ]]; then
    echo "Failed to resolve conda base directory." >&2
    exit 1
fi
ENV_PREFIX="$CONDA_BASE/envs/$ENV_NAME"

log_step "Creating conda env '$ENV_NAME' with Python 3.10"
if [[ -x "$ENV_PREFIX/bin/python" ]]; then
    echo "Conda env '$ENV_NAME' already exists. Reusing it."
else
    CONDA_SOLVER=classic conda create -n "$ENV_NAME" python=3.10 -y
fi

PYTHON_BIN="$ENV_PREFIX/bin/python"
PIP_BIN="$ENV_PREFIX/bin/pip"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Failed to resolve python for conda env '$ENV_NAME'." >&2
    exit 1
fi

have_python_module() {
    "$PYTHON_BIN" -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)' "$1"
}

activate_build_env() {
    export CUDA_HOME="$ENV_PREFIX"
    export CUB_HOME="$ENV_PREFIX/include"
    export PATH="$ENV_PREFIX/bin:$PATH"
    export LD_LIBRARY_PATH="$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export MAX_JOBS="${MAX_JOBS:-4}"
    unset CC || true
    unset CXX || true
}

HOST_GXX="$(command -v g++ || true)"
HOST_GXX_MAJOR=""
if [[ -n "$HOST_GXX" ]]; then
    HOST_GXX_MAJOR="$("$HOST_GXX" -dumpfullversion -dumpversion | awk -F. '{print $1}')"
fi

cat > "$PIP_CONSTRAINTS" <<'EOF'
accelerate==1.12.0
av==16.1.0
easydict==1.13
einops==0.8.1
gdown==5.2.1
h5py==3.15.1
hydra_zen==0.16.0
huggingface_hub==0.36.0
kornia==0.8.2
nerfstudio==0.3.3
nerfacc==0.5.2
numpy==1.26.4
opencv-python==4.6.0.66
opencv-python-headless==4.11.0.86
open3d==0.19.0
plotly==6.5.2
protobuf==3.20.3
pycolmap==3.13.0
pyequilib==0.5.8
pymeshlab==2025.7
pyliblzfse==0.4.1
pytorch-lightning==2.6.0
rawpy==0.25.1
rich==14.2.0
scikit-image==0.25.2
scikit-learn==1.7.2
scipy==1.15.3
smplx==0.1.28
tensorboard==2.20.0
toolz==1.1.0
tyro==1.0.5
typed-argument-parser==1.11.0
ultralytics==8.4.6
ultralytics-thop==2.0.18
wandb==0.24.0
wis3d==1.0.1
yacs==0.1.8
EOF

log_step "Installing CUDA 12.4 compiler toolchain"
if [[ -x "$ENV_PREFIX/bin/nvcc" ]]; then
    echo "CUDA 12.4 toolchain already present in '$ENV_NAME'."
else
    CONDA_SOLVER=classic conda install -n "$ENV_NAME" -y \
        --override-channels \
        -c nvidia/label/cuda-12.4.1 \
        -c defaults \
        cuda-toolkit=12.4.1 \
        cuda-compiler=12.4.1 \
        cuda-command-line-tools=12.4.1 \
        cuda-nvcc=12.4.131
fi

if [[ ! -x "$ENV_PREFIX/bin/cmake" || ! -x "$ENV_PREFIX/bin/ninja" ]]; then
    CONDA_SOLVER=classic conda install -n "$ENV_NAME" -y -c conda-forge make ninja cmake
fi

if [[ -n "$HOST_GXX_MAJOR" && "$HOST_GXX_MAJOR" -gt 13 ]]; then
    cat >&2 <<EOF
Warning: detected host g++ major version $HOST_GXX_MAJOR at $HOST_GXX.
CUDA 12.4 officially supports GCC up to 13 for extension builds.
If pytorch3d or CUDA extension compilation fails on your machine, provide a GCC 13
toolchain and rerun this script.
EOF
fi

activate_build_env

log_step "Installing pinned Python dependencies"
activate_build_env
"$PIP_BIN" install -r "$PIP_REQUIREMENTS" -c "$PIP_CONSTRAINTS"

log_step "Installing special build dependencies"
activate_build_env
if have_python_module nerfstudio; then
    echo "nerfstudio already installed. Skipping."
else
    "$PIP_BIN" install --no-deps nerfstudio==0.3.3
fi
if have_python_module pytorch3d; then
    echo "pytorch3d already installed. Skipping."
else
    "$PIP_BIN" install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
fi
if have_python_module chumpy; then
    echo "chumpy already installed. Skipping."
else
    "$PIP_BIN" install --no-build-isolation "git+https://github.com/mattloper/chumpy.git"
fi

log_step "Installing local helper packages"
activate_build_env
SAM2_BUILD_ALLOW_ERRORS=1 "$PIP_BIN" install --no-deps -e "$REPO_ROOT/prep/AutoMask"
"$PIP_BIN" install --no-deps -e "$REPO_ROOT/prep/HMR"
"$PIP_BIN" install --no-deps -e "$REPO_ROOT/prep/UFM/UniCeption"
"$PIP_BIN" install --no-deps -e "$REPO_ROOT/prep/UFM"
"$PIP_BIN" install --no-deps -e "$REPO_ROOT/vis_scripts/viser_m"
"$PIP_BIN" install numpy==1.26.4 opencv-python==4.6.0.66

log_step "Building CUDA extensions used by MogeSAM"
activate_build_env
(cd "$REPO_ROOT/prep/MogeSAM/third_party/pointops2" && \
    LIBRARY_PATH="$ENV_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}" \
    "$PYTHON_BIN" setup.py install)
(cd "$REPO_ROOT/prep/MogeSAM/third_party/megasam/base" && \
    LIBRARY_PATH="$ENV_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}" \
    "$PYTHON_BIN" setup.py install)

log_step "Installing GeoCalib"
if [[ ! -d "$REPO_ROOT/prep/GeoCalib/.git" ]]; then
    git clone https://github.com/hongsukchoi/GeoCalib.git "$REPO_ROOT/prep/GeoCalib"
fi
"$PIP_BIN" install -e "$REPO_ROOT/prep/GeoCalib"

log_step "Validating core CRISP environment"
bash "$VALIDATE_SCRIPT" "$PYTHON_BIN"

if (( FETCH_ASSETS == 1 )); then
    log_step "Ensuring checkpoints used by the demo pipeline exist"
    if [[ ! -f "$REPO_ROOT/prep/AutoMask/checkpoints/sam2_hiera_large.pt" ]]; then
        (cd "$REPO_ROOT/prep/AutoMask/checkpoints" && sh ./download_ckpts.sh)
    fi
    if [[ ! -f "$REPO_ROOT/prep/MogeSAM/checkpoints/tapip3d_final.pth" ]]; then
        mkdir -p "$REPO_ROOT/prep/MogeSAM/checkpoints"
        wget -q -O "$REPO_ROOT/prep/MogeSAM/checkpoints/tapip3d_final.pth" \
            https://huggingface.co/zbww/tapip3d/resolve/main/tapip3d_final.pth
    fi
    if [[ ! -f "$REPO_ROOT/prep/MogeSAM/third_party/megasam/Depth-Anything/checkpoints/depth_anything_vitl14.pth" ]]; then
        mkdir -p "$REPO_ROOT/prep/MogeSAM/third_party/megasam/Depth-Anything/checkpoints"
        curl -L \
            -o "$REPO_ROOT/prep/MogeSAM/third_party/megasam/Depth-Anything/checkpoints/depth_anything_vitl14.pth" \
            "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth"
    fi
    if [[ ! -f "$REPO_ROOT/prep/MogeSAM/third_party/megasam/cvd_opt/raft-things.pth" ]]; then
        mkdir -p "$REPO_ROOT/prep/MogeSAM/third_party/megasam/cvd_opt"
        "$PYTHON_BIN" -m gdown --folder --fuzzy \
            'https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT' \
            -O "$REPO_ROOT/prep/MogeSAM/third_party/megasam/cvd_opt/" \
            --remaining-ok --continue
        if [[ -f "$REPO_ROOT/prep/MogeSAM/third_party/megasam/cvd_opt/models/raft-things.pth" ]]; then
            mv -f \
                "$REPO_ROOT/prep/MogeSAM/third_party/megasam/cvd_opt/models/raft-things.pth" \
                "$REPO_ROOT/prep/MogeSAM/third_party/megasam/cvd_opt/raft-things.pth"
        fi
    fi
    if [[ ! -f "$REPO_ROOT/prep/HMR/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt" ]]; then
        mkdir -p "$REPO_ROOT/prep/HMR/inputs" "$REPO_ROOT/prep/HMR/outputs"
        "$PYTHON_BIN" -m gdown --folder \
            "https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD" \
            -O "$REPO_ROOT/prep/HMR/inputs/" \
            --remaining-ok --continue
    fi

    log_step "Prefetching torch.hub repositories used by MogeSAM"
    if [[ ! -d "$TORCH_HUB_DIR/facebookresearch_co-tracker_main/.git" ]]; then
        rm -rf "$TORCH_HUB_DIR/facebookresearch_co-tracker_main"
        git clone --depth 1 https://github.com/facebookresearch/co-tracker.git \
            "$TORCH_HUB_DIR/facebookresearch_co-tracker_main"
    fi
    if [[ ! -d "$TORCH_HUB_DIR/facebookresearch_dinov2_main/.git" ]]; then
        rm -rf "$TORCH_HUB_DIR/facebookresearch_dinov2_main"
        git clone --depth 1 https://github.com/facebookresearch/dinov2.git \
            "$TORCH_HUB_DIR/facebookresearch_dinov2_main"
    fi
    if [[ ! -f "$TORCH_HUB_CHECKPOINTS/scaled_offline.pth" ]]; then
        wget -q -O "$TORCH_HUB_CHECKPOINTS/scaled_offline.pth" \
            https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
    fi
fi

cat <<EOF

Environment ready.

Validated scope:
  - Core CRISP video pipeline environment
  - Excludes Contact-Predictor
  - Excludes MotionTracking

Validation command:
  conda activate $ENV_NAME
  bash "$VALIDATE_SCRIPT"

Notes:
- By default this script installs environment dependencies only. It does not
  download demo checkpoints or torch.hub caches.
- Pass \`--with-assets\` if you want this script to also fetch the demo assets
  needed by \`run_crisp_video.sh --demo\`.
- The wrapper adds \`runtime_shims/sitecustomize.py\` via PYTHONPATH so cached
  torch.hub repos are used when GitHub branch probing returns transient 5xx.
- CUDA extension builds use the CUDA 12.4 toolchain installed into the conda env.
- For a clean clone, make sure the official SMPL / SMPL-X assets are available
  for the HMR stage. Follow the README asset section if your checkout does not
  already contain the required body-model files.
- Demo videos can be downloaded with the README \`gdown --folder ... -O data\` step.

EOF
