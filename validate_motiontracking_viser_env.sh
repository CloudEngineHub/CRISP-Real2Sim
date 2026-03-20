#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${1:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
MT_DIR="$REPO_ROOT/MotionTracking"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    CONDA_PREFIX="$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)"
    export CONDA_PREFIX
fi

export PYTHONPATH="$MT_DIR:$MT_DIR/poselib:$MT_DIR/isaac_utils${PYTHONPATH:+:$PYTHONPATH}"
if [[ -d "$CONDA_PREFIX/lib" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

echo "[1/4] MotionTracking smoke test"
PYTHON_BIN="$PYTHON_BIN" bash "$MT_DIR/run_crisp_test.sh" --smoke-only

echo "[2/4] robot-viser import and server smoke test"
"$PYTHON_BIN" - <<'PY'
import tempfile
from pathlib import Path

import numpy as np
from tensorboard import summary  # noqa: F401

from motion_tracking.utils.robot_viser import RobotMjcfViser
from motion_tracking.utils.viser_visualizer import ViserHelper


MJCF_TEXT = """<mujoco model="mini_robot">
  <worldbody>
    <body name="pelvis" pos="0 0 0">
      <geom type="capsule" fromto="0 0 0 0 0 0.4" size="0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


with tempfile.TemporaryDirectory(prefix="motiontracking-viser-") as tmp_dir:
    mjcf_path = Path(tmp_dir) / "mini_robot.xml"
    mjcf_path.write_text(MJCF_TEXT)

    viser = ViserHelper(port=18082)
    if not viser.ok():
        raise RuntimeError("Viser server failed to start.")

    robot = RobotMjcfViser(viser, str(mjcf_path), None)
    if len(robot._geom_specs) == 0:
        raise RuntimeError("RobotMjcfViser did not parse any robot geometry.")

    robot.update(
        np.zeros((1, 3), dtype=np.float32),
        np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    )
    print("viser_server_ok=True")
    print(f"robot_geom_bodies={len(robot._geom_specs)}")

    if viser.server is not None:
        viser.server.stop()
PY

echo "[3/4] Visualizer CLI smoke test"
(
    cd "$MT_DIR"
    "$PYTHON_BIN" scripts/visualize_viser_robot.py --help >/dev/null
)

echo "[4/4] Synthetic playback smoke test"
if ! command -v timeout >/dev/null 2>&1; then
    echo "timeout command is required for visualization validation." >&2
    exit 1
fi

TMP_DIR="$(mktemp -d /tmp/motiontracking-viser-smoke.XXXXXX)"
export TMP_DIR
LOG_FILE="$TMP_DIR/visualizer.log"
trap 'rm -rf "$TMP_DIR"' EXIT

cat >"$TMP_DIR/mini_robot.xml" <<'XML'
<mujoco model="mini_robot">
  <worldbody>
    <body name="pelvis" pos="0 0 0">
      <geom type="capsule" fromto="0 0 0 0 0 0.4" size="0.05"/>
    </body>
  </worldbody>
</mujoco>
XML

"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
import numpy as np
import yaml

tmp_dir = Path(os.environ["TMP_DIR"])
pos = np.zeros((2, 1, 3), dtype=np.float32)
rot = np.zeros((2, 1, 4), dtype=np.float32)
rot[..., 3] = 1.0
np.savez(tmp_dir / "rigid_bodies_0.npz", pos=pos, rot=rot)

with (tmp_dir / "robot_vis_info.yaml").open("w") as f:
    yaml.safe_dump({"body_names": [], "dt": 1.0 / 60.0, "asset_xml": "mini_robot.xml"}, f)
PY

set +e
(
    cd "$MT_DIR"
    timeout 5s "$PYTHON_BIN" scripts/visualize_viser_robot.py "$TMP_DIR" --port 18081
) >"$LOG_FILE" 2>&1
RC=$?
set -e

if [[ "$RC" -ne 124 ]]; then
    cat "$LOG_FILE" >&2
    echo "Synthetic robot-viser playback smoke test failed (exit=$RC)." >&2
    exit 1
fi

if ! rg -q "HTTP.*localhost:18081|Websocket.*localhost:18081" "$LOG_FILE"; then
    cat "$LOG_FILE" >&2
    echo "Viser server did not announce the expected robot visualization endpoint." >&2
    exit 1
fi

echo "MotionTracking + robot-viser environment validation passed."
