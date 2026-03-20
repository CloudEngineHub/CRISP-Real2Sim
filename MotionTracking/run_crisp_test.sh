#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/poselib:$SCRIPT_DIR/isaac_utils${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [[ "${1:-}" == "--smoke-only" ]]; then
    "$PYTHON_BIN" - <<'PY'
import isaacgym
import torch
from poselib.core.rotation3d import quat_mul
from isaac_utils import torch_utils
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES

print("MotionTracking environment smoke test passed.")
print(f"torch={torch.__version__}")
print(f"smpl_bones={len(SMPL_BONE_ORDER_NAMES)}")
PY
    exit 0
fi

DATE="${DATE:-0921}"
SCENE="${SCENE:-MPH1Library}"
METHOD="${METHOD:-ours}"

DEFAULT_MOTION="$SCRIPT_DIR/motion_data/${DATE}/${SCENE}_${METHOD}.npy"
DEFAULT_CHECKPOINT="$SCRIPT_DIR/results/${DATE}/${DATE}_${SCENE}_${METHOD}/lightning_logs/version_0/last.ckpt"
ALT_MOTION="$REPO_ROOT/motion_data/${DATE}/${SCENE}_${METHOD}.npy"
ALT_CHECKPOINT="$REPO_ROOT/results/${DATE}/${DATE}_${SCENE}_${METHOD}/lightning_logs/version_0/last.ckpt"

MOTION_FILE="${MOTION_FILE:-$DEFAULT_MOTION}"
CHECKPOINT="${CHECKPOINT:-$DEFAULT_CHECKPOINT}"

if [[ ! -f "$MOTION_FILE" && -f "$ALT_MOTION" ]]; then
    MOTION_FILE="$ALT_MOTION"
fi
if [[ ! -f "$CHECKPOINT" && -f "$ALT_CHECKPOINT" ]]; then
    CHECKPOINT="$ALT_CHECKPOINT"
fi

CONFIG_PATH="${CONFIG_PATH:-}"
if [[ -z "$CONFIG_PATH" && -f "$(dirname "$CHECKPOINT")/config.yaml" ]]; then
    CONFIG_PATH="$(dirname "$CHECKPOINT")/config.yaml"
fi
if [[ -z "$CONFIG_PATH" && -f "$(dirname "$(dirname "$CHECKPOINT")")/config.yaml" ]]; then
    CONFIG_PATH="$(dirname "$(dirname "$CHECKPOINT")")/config.yaml"
fi

missing=0
if [[ ! -f "$MOTION_FILE" ]]; then
    echo "Missing motion file: $MOTION_FILE" >&2
    missing=1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Missing checkpoint: $CHECKPOINT" >&2
    missing=1
fi
if [[ "$missing" -ne 0 ]]; then
    cat <<EOF >&2

The MotionTracking environment is ready, but the crisp_test assets are not present.

Provide them explicitly:
  export MOTION_FILE=/abs/path/to/MPH1Library_ours.npy
  export CHECKPOINT=/abs/path/to/last.ckpt
  bash run_crisp_test.sh

Or use the default naming convention:
  DATE=$DATE
  SCENE=$SCENE
  METHOD=$METHOD

EOF
    exit 1
fi

export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

ARGS=(
    motion_tracking/eval_agent.py
    "critic_units=[1024,512]"
    "exp=mimic"
    "backbone=isaacgym_smpl_pulse_shape"
    "+opt=[mimic/target_pose_transformer_with_target_time_final,mimic/global_tracking]"
    "motion_file=${MOTION_FILE}"
    "early_reward_term=null"
    "headless=True"
    "num_future_steps=10"
    "checkpoint=${CHECKPOINT}"
    "max_episode_length=1000"
    "num_envs=1"
    "init_start_prob=1"
    "reset_track_steps_min=1000000"
    "reset_track_steps_max=1000001"
    "provide_future_states=True"
    "enable_height_termination=False"
    "fix_pd_offsets=False"
    "obs_relative_to_surface=True"
    "ref_height_adjust=0.0"
    "scene_file=${DATE}/${SCENE}/${METHOD}/${METHOD}"
)

if [[ -n "$CONFIG_PATH" ]]; then
    ARGS+=("config_path=${CONFIG_PATH}")
fi

cd "$SCRIPT_DIR"
exec "$PYTHON_BIN" "${ARGS[@]}"
