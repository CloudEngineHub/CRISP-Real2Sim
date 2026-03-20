#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    cat <<'EOF' >&2
Usage:
  bash run_motion_replay_kinematic.sh <DATE> <SCENE> <METHOD> [replay args...]

Example:
  MT_VISER_PORT=8082 bash run_motion_replay_kinematic.sh stairs0319 56_outdoor_stairs_up_down ours

Notes:
  - This is a pure kinematic replay in Viser. It does not use Isaac Gym.
  - Pass `--once` after the first 3 args to play a clip once.
EOF
    exit 1
fi

DATE="$1"
SCENE="$2"
METHOD="$3"
shift 3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
VISER_PORT="${MT_VISER_PORT:-8080}"

export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/poselib:$SCRIPT_DIR/isaac_utils:$SCRIPT_DIR/smpllib${PYTHONPATH:+:$PYTHONPATH}"

MOTION_FILE="$SCRIPT_DIR/motion_data/${DATE}/${SCENE}_${METHOD}.npy"
SCENE_URDF="$SCRIPT_DIR/motion_tracking/data/assets/urdf/${DATE}/${SCENE}/${METHOD}/${METHOD}.urdf"

if [[ ! -f "$MOTION_FILE" ]]; then
    echo "Missing bridged motion file: $MOTION_FILE" >&2
    exit 1
fi

ARGS=(
    "$SCRIPT_DIR/scripts/replay_motion_viser.py"
    "$MOTION_FILE"
    "--port" "$VISER_PORT"
)

if [[ -f "$SCENE_URDF" ]]; then
    ARGS+=("--scene-urdf" "$SCENE_URDF")
fi

ARGS+=("$@")

if [[ "${PRINT_ONLY:-0}" == "1" ]]; then
    printf 'MT_VISER_PORT=%q ' "$VISER_PORT"
    printf '%q ' "$PYTHON_BIN" "${ARGS[@]}"
    echo
    exit 0
fi

exec "$PYTHON_BIN" "${ARGS[@]}"
