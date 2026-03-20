#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    cat <<'EOF' >&2
Usage:
  bash run_bridged_train.sh <DATE> <SCENE> <METHOD> [hydra overrides...]

Example:
  bash run_bridged_train.sh bridge0318 40_indoor_walk_big_circle ours

Set PRINT_ONLY=1 to print the command without running it.
EOF
    exit 1
fi

DATE="$1"
SCENE="$2"
METHOD="$3"
shift 3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SMPL_RUNTIME_DIR="$(bash "$SCRIPT_DIR/prepare_motiontracking_smpl_runtime.sh")"

export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/poselib:$SCRIPT_DIR/isaac_utils:$SCRIPT_DIR/smpllib${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

MOTION_FILE="motion_data/${DATE}/${SCENE}_${METHOD}.npy"
SCENE_FILE="${DATE}/${SCENE}/${METHOD}/${METHOD}"
SCENE_URDF="motion_tracking/data/assets/urdf/${SCENE_FILE}.urdf"

if [[ ! -f "$SCRIPT_DIR/$MOTION_FILE" ]]; then
    echo "Missing bridged motion file: $SCRIPT_DIR/$MOTION_FILE" >&2
    exit 1
fi

if [[ ! -f "$SCRIPT_DIR/$SCENE_URDF" ]]; then
    echo "Missing bridged scene URDF: $SCRIPT_DIR/$SCENE_URDF" >&2
    exit 1
fi

ARGS=(
    motion_tracking/train_agent.py
    "exp=mimic"
    "backbone=isaacgym_smpl_pulse_shape"
    "+opt=[mimic/target_pose_transformer_with_target_time_final,mimic/global_tracking,mimic/early_termination_tracking_err,mimic/dynamic_sampling]"
    "dynamic_weight_max=10000"
    "motion_file=${MOTION_FILE}"
    "ngpu=1"
    "eval_metrics_every=500000"
    "training_max_steps=100000000000"
    "name=bridge_${DATE}_${SCENE}_${METHOD}"
    "num_future_steps=10"
    "critic_units=[1024,512]"
    "init_start_prob=0.1"
    "num_mini_epochs=1"
    "fix_pd_offsets=False"
    "obs_relative_to_surface=True"
    "num_envs=1024"
    "batch_size=4096"
    "provide_future_states=True"
    "headless=True"
    "ref_height_adjust=0.0"
    "scene_file=${SCENE_FILE}"
    "asset.dynamic_shape.smpl_data_dir=${SMPL_RUNTIME_DIR}"
)
ARGS+=("$@")

cd "$SCRIPT_DIR"
if [[ "${PRINT_ONLY:-0}" == "1" ]]; then
    printf '%q ' "$PYTHON_BIN" "${ARGS[@]}"
    echo
    exit 0
fi

exec "$PYTHON_BIN" "${ARGS[@]}"
