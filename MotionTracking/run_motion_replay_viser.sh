#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    cat <<'EOF' >&2
Usage:
  bash run_motion_replay_viser.sh <DATE> <SCENE> <METHOD> [hydra overrides...]

Example:
  MT_VISER_PORT=8082 bash run_motion_replay_viser.sh stairs0319 56_outdoor_stairs_up_down ours

Notes:
  - This is a kinematic replay. It uses `sync_motion=True` and does not load a checkpoint.
  - Set MT_VISER_PORT to avoid conflicts with an existing training viewer.
  - Set PRINT_ONLY=1 to print the command without running it.
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
VISER_PORT="${MT_VISER_PORT:-8080}"

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
    motion_tracking/eval_agent.py
    "exp=mimic"
    "backbone=isaacgym_smpl_pulse_shape"
    "+opt=[mimic/target_pose_transformer_with_target_time_final,mimic/global_tracking,mimic/early_termination_tracking_err,mimic/dynamic_sampling]"
    "motion_file=${MOTION_FILE}"
    "early_reward_term=null"
    "headless=False"
    "visualize_markers=False"
    "num_future_steps=10"
    "max_episode_length=1000"
    "num_envs=1"
    "num_games=1"
    "max_eval_steps=512"
    "init_start_prob=1"
    "reset_track_steps_min=1000000"
    "reset_track_steps_max=1000001"
    "provide_future_states=True"
    "enable_height_termination=False"
    "fix_pd_offsets=False"
    "obs_relative_to_surface=True"
    "ref_height_adjust=0.0"
    "sync_motion=True"
    "scene_file=${SCENE_FILE}"
    "asset.dynamic_shape.smpl_data_dir=${SMPL_RUNTIME_DIR}"
)
ARGS+=("$@")

cd "$SCRIPT_DIR"
if [[ "${PRINT_ONLY:-0}" == "1" ]]; then
    printf 'MT_VISER_PORT=%q ' "$VISER_PORT"
    printf '%q ' "$PYTHON_BIN" "${ARGS[@]}"
    echo
    exit 0
fi

XVFB_PID=""
if [[ -z "${DISPLAY:-}" ]]; then
    export DISPLAY=:99
    Xvfb "$DISPLAY" -screen 0 1280x720x24 >/tmp/mt_motion_replay_xvfb.log 2>&1 &
    XVFB_PID=$!
    trap 'if [[ -n "$XVFB_PID" ]]; then kill "$XVFB_PID" 2>/dev/null || true; fi' EXIT
fi

export MT_VISER_PORT="$VISER_PORT"
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

echo "[Motion Replay] http://localhost:${VISER_PORT}"
exec "$PYTHON_BIN" "${ARGS[@]}"
