#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

import numpy as np

from motion_tracking.utils.robot_viser import RobotMjcfViser
from motion_tracking.utils.viser_visualizer import (
    ViserHelper,
    add_ground_grid,
    load_static_urdf,
)
from poselib.skeleton.skeleton3d import SkeletonMotion


def _default_mjcf_path() -> Path:
    candidates = [
        Path("/tmp/smpl/smpl_humanoid_custom_merged.xml"),
        Path("/tmp/smpl/smpl_humanoid_custom.xml"),
    ]
    candidates.extend(sorted(Path("/tmp/smpl").glob("smpl_humanoid_*.xml")))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Could not find a generated SMPL MJCF in /tmp/smpl. "
        "Run a MotionTracking train/eval job once, or pass --mjcf-path explicitly."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Kinematic motion replay with Viser.")
    ap.add_argument("motion_file", type=str, help="Path to bridged motion .npy")
    ap.add_argument("--scene-urdf", type=str, default=None, help="Optional static scene URDF")
    ap.add_argument("--mjcf-path", type=str, default=None, help="Robot MJCF path; defaults to generated /tmp/smpl asset")
    ap.add_argument("--port", type=int, default=8080, help="Viser port")
    ap.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    ap.add_argument("--once", action="store_true", help="Play once instead of looping")
    args = ap.parse_args()

    motion_path = Path(args.motion_file)
    if not motion_path.is_file():
        raise FileNotFoundError(f"Motion file not found: {motion_path}")

    motion_data = np.load(motion_path, allow_pickle=True).item()
    motion = SkeletonMotion.from_dict(motion_data)

    body_pos = motion.global_translation.detach().cpu().numpy().astype(np.float32)
    body_rot = motion.global_rotation.detach().cpu().numpy().astype(np.float32)
    body_names = list(motion.skeleton_tree.node_names)
    fps = float(motion.fps)
    dt = 1.0 / max(fps * args.speed, 1e-6)

    mjcf_path = Path(args.mjcf_path) if args.mjcf_path is not None else _default_mjcf_path()
    if not mjcf_path.is_file():
        raise FileNotFoundError(f"MJCF file not found: {mjcf_path}")

    viser = ViserHelper(port=args.port)
    if not viser.ok():
        print("[Replay] Viser not available.", file=sys.stderr)
        return 1

    add_ground_grid(viser)
    if args.scene_urdf is not None:
        scene_path = Path(args.scene_urdf)
        if scene_path.is_file():
            load_static_urdf(viser, str(scene_path), prefix="/scene")
        else:
            print(f"[Replay] Scene URDF not found: {scene_path}")

    robot = RobotMjcfViser(viser, str(mjcf_path), body_names)

    root0 = body_pos[0, 0]
    viser.set_camera(
        root0 + np.array([0.0, -2.5, 1.5], dtype=np.float32),
        root0 + np.array([0.0, 0.0, 0.6], dtype=np.float32),
    )

    print(f"[Replay] motion={motion_path}")
    print(f"[Replay] mjcf={mjcf_path}")
    print(f"[Replay] frames={body_pos.shape[0]} fps={fps}")
    print(f"[Replay] http://localhost:{args.port}")

    try:
        while True:
            for idx in range(body_pos.shape[0]):
                robot.update(body_pos[idx], body_rot[idx])
                time.sleep(dt)
            if args.once:
                break
    except KeyboardInterrupt:
        print("\n[Replay] Stopped by user.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
