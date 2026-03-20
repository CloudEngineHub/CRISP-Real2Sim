#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MT_DIR = SCRIPT_DIR.parent
REPO_ROOT = MT_DIR.parent

for extra_path in (MT_DIR, MT_DIR / "poselib", MT_DIR / "isaac_utils", MT_DIR / "smpllib", SCRIPT_DIR):
    extra_str = str(extra_path)
    if extra_str not in sys.path:
        sys.path.insert(0, extra_str)

from place_scene_motion_utils import get_smpl_robot  # noqa: E402
from poselib.skeleton.skeleton3d import SkeletonMotion  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize CRISP post_scene outputs into the directory layout that MotionTracking training expects."
    )
    parser.add_argument("seq_name", help="Sequence name under results/output/post_scene/<seq>/<hmr_type>.")
    parser.add_argument("--date", default="bridge", help="MotionTracking date tag, e.g. 0921 or bridge0318.")
    parser.add_argument("--method", default="ours", help="Method tag used by MotionTracking run_train/run_test.")
    parser.add_argument("--hmr-type", default="gv", help="Subdirectory under the CRISP output, default: gv.")
    parser.add_argument(
        "--post-scene-root",
        type=Path,
        default=REPO_ROOT / "results/output/post_scene",
        help="Root like results/output/post_scene.",
    )
    parser.add_argument(
        "--motiontracking-root",
        type=Path,
        default=MT_DIR,
        help="MotionTracking repo root. Defaults to the local MotionTracking checkout.",
    )
    parser.add_argument(
        "--materialize",
        choices=("copy", "symlink"),
        default="copy",
        help="How to place source assets inside MotionTracking. Default: copy.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing bridged files.")
    return parser.parse_args()


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def materialize_file(src: Path, dst: Path, mode: str, force: bool) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"Missing source file: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if force:
            remove_path(dst)
        else:
            return

    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def validate_motion_file(motion_file: Path) -> tuple[int, int]:
    motion = SkeletonMotion.from_file(str(motion_file))
    num_frames = int(motion.tensor.shape[0])
    num_joints = int(getattr(motion, "num_joints", motion.tensor.shape[1]))
    if num_frames <= 0 or num_joints <= 0:
        raise RuntimeError(f"Invalid MotionTracking motion file: {motion_file}")
    return num_frames, num_joints


def iter_mesh_refs(urdf_file: Path) -> Iterable[Path]:
    root = ET.parse(urdf_file).getroot()
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if not filename:
            continue
        yield urdf_file.parent / filename


def validate_scene_files(urdf_file: Path) -> int:
    missing = [mesh_path for mesh_path in iter_mesh_refs(urdf_file) if not mesh_path.is_file()]
    if missing:
        joined = "\n".join(str(path) for path in missing[:20])
        raise FileNotFoundError(f"URDF references missing mesh files:\n{joined}")
    return sum(1 for _ in iter_mesh_refs(urdf_file))


def build_temp_smpl_root() -> tempfile.TemporaryDirectory:
    temp_dir = tempfile.TemporaryDirectory(prefix="motiontracking-bridge-")
    smpl_root = Path(temp_dir.name) / "data" / "smpl"
    smpl_root.mkdir(parents=True, exist_ok=True)

    src_root = REPO_ROOT / "prep" / "data" / "smpl"
    if not src_root.is_dir():
        raise FileNotFoundError(f"Missing SMPL source directory: {src_root}")

    for name in ("J_regressor_extra.npy", "SMPL_MALE.pkl", "SMPL_NEUTRAL.pkl"):
        src = src_root / name
        if src.is_file():
            os.symlink(src, smpl_root / name)

    neutral_src = smpl_root / "SMPL_NEUTRAL.pkl"
    male_src = smpl_root / "SMPL_MALE.pkl"
    female_dst = smpl_root / "SMPL_FEMALE.pkl"
    if not female_dst.exists():
        fallback = neutral_src if neutral_src.exists() else male_src
        if fallback.exists():
            os.symlink(fallback, female_dst)

    return temp_dir


def main() -> None:
    args = parse_args()

    post_scene_root = args.post_scene_root.resolve()
    mt_root = args.motiontracking_root.resolve()

    src_seq_root = post_scene_root / args.seq_name / args.hmr_type
    if not src_seq_root.is_dir():
        raise FileNotFoundError(f"Missing CRISP post_scene sequence: {src_seq_root}")

    motion_src = src_seq_root / "hmr" / "human_motion.npz"
    if not motion_src.is_file():
        raise FileNotFoundError(
            f"Missing aligned motion file: {motion_src}\n"
            "Run the CRISP post-scene step first; raw scene output alone is not enough for MotionTracking training."
        )

    scene_src_root = src_seq_root / "scene_mesh_sqs"
    scene_urdf_src = scene_src_root / "scene_mesh_sqs.urdf"
    scene_obj_src = scene_src_root / "scene_mesh_sqs.obj"
    if not scene_urdf_src.is_file():
        raise FileNotFoundError(f"Missing scene URDF: {scene_urdf_src}")
    if not scene_obj_src.is_file():
        raise FileNotFoundError(f"Missing scene OBJ: {scene_obj_src}")

    motion_dst_dir = mt_root / "motion_data" / args.date
    motion_npz_dst = motion_dst_dir / f"{args.seq_name}_{args.method}.npz"
    motion_npy_dst = motion_dst_dir / f"{args.seq_name}_{args.method}.npy"

    scene_dst_dir = (
        mt_root
        / "motion_tracking/data/assets/urdf"
        / args.date
        / args.seq_name
        / args.method
    )
    scene_urdf_dst = scene_dst_dir / f"{args.method}.urdf"
    scene_obj_dst = scene_dst_dir / f"{args.method}.obj"

    materialize_file(motion_src, motion_npz_dst, args.materialize, args.force)
    materialize_file(scene_urdf_src, scene_urdf_dst, args.materialize, args.force)
    materialize_file(scene_obj_src, scene_obj_dst, args.materialize, args.force)

    for suffix in ("sqs_params.npy", "sqs_params.npz"):
        src = scene_src_root / suffix
        if src.is_file():
            materialize_file(src, scene_dst_dir / suffix, args.materialize, args.force)

    pieces_dir = scene_src_root / "pieces"
    if pieces_dir.is_dir():
        for part_src in sorted(pieces_dir.glob("*.obj")):
            materialize_file(part_src, scene_dst_dir / part_src.name, args.materialize, args.force)

    if args.force or not motion_npy_dst.exists():
        prev_cwd = Path.cwd()
        temp_smpl_root = build_temp_smpl_root()
        try:
            # The current LocalRobot helper resolves SMPL assets from ./data/smpl.
            os.chdir(temp_smpl_root.name)
            with np.load(motion_src, allow_pickle=True) as motion_data:
                get_smpl_robot(
                    motion_data,
                    str(motion_npy_dst),
                    [0.0, 0.0, 0.0],
                    0.0,
                    need_rotate=False,
                )
        finally:
            os.chdir(prev_cwd)
            temp_smpl_root.cleanup()

    num_frames, num_joints = validate_motion_file(motion_npy_dst)
    num_mesh_refs = validate_scene_files(scene_urdf_dst)

    scene_file_arg = f"{args.date}/{args.seq_name}/{args.method}/{args.method}"

    print("Bridge completed.")
    print(f"source_motion_npz={motion_src}")
    print(f"bridged_motion_npz={motion_npz_dst}")
    print(f"bridged_motion_npy={motion_npy_dst}")
    print(f"bridged_scene_urdf={scene_urdf_dst}")
    print(f"bridged_scene_obj={scene_obj_dst}")
    print(f"motion_frames={num_frames}")
    print(f"motion_joints={num_joints}")
    print(f"scene_mesh_refs={num_mesh_refs}")
    print(f"scene_file_arg={scene_file_arg}")
    print()
    print("Training command:")
    print(
        f"  cd {mt_root}\n"
        f"  bash run_bridged_train.sh {args.date} {args.seq_name} {args.method}"
    )
    print()
    print("Evaluation command:")
    print(
        f"  cd {mt_root}\n"
        f"  bash run_bridged_eval.sh {args.date} {args.seq_name} {args.method} /abs/path/to/last.ckpt"
    )


if __name__ == "__main__":
    main()
