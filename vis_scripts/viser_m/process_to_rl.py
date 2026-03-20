#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = (THIS_DIR / ".." / "..").resolve()

DEFAULT_INPUT_ROOT = REPO_ROOT / "results" / "output" / "post_scene"
DEFAULT_MOTION_ROOT = REPO_ROOT / "MotionTracking" / "motion_data"


def _parse_xyz(raw: str) -> Tuple[float, float, float]:
    tokens = raw.replace(",", " ").split()
    if len(tokens) != 3:
        raise argparse.ArgumentTypeError("`--xyz-transl` must contain exactly 3 numbers, e.g. '0 0 0'.")
    try:
        x, y, z = (float(tokens[0]), float(tokens[1]), float(tokens[2]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid `--xyz-transl`: {raw}") from exc
    return x, y, z


def _load_get_smpl_robot() -> Callable:
    mt_root = REPO_ROOT / "MotionTracking"
    mt_scripts = mt_root / "scripts"

    for p in (str(mt_root), str(mt_scripts)):
        if p not in sys.path:
            sys.path.insert(0, p)

    from place_scene_motion_utils import get_smpl_robot  # type: ignore

    return get_smpl_robot


def discover_sequences(input_root: Path, hmr_type: str) -> List[str]:
    seqs: List[str] = []
    if not input_root.exists():
        return seqs
    for candidate in sorted(input_root.iterdir()):
        if not candidate.is_dir():
            continue
        motion_npz = candidate / hmr_type / "hmr" / "human_motion.npz"
        if motion_npz.exists():
            seqs.append(candidate.name)
    return seqs


def process_sequence(
    seq_name: str,
    input_root: Path,
    motion_root: Path,
    date: str,
    method: str,
    hmr_type: str,
    xyz_transl: Tuple[float, float, float],
    xy_rotate_deg: float,
    need_rotate: bool,
    get_smpl_robot: Callable,
) -> bool:
    src_npz = input_root / seq_name / hmr_type / "hmr" / "human_motion.npz"
    if not src_npz.exists():
        print(f"[WARN] Missing input for {seq_name}: {src_npz}")
        return False

    out_dir = motion_root / date
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npy = out_dir / f"{seq_name}_{method}.npy"

    motion_data = np.load(src_npz, allow_pickle=True)
    try:
        get_smpl_robot(
            motion_data=motion_data,
            outputpath=str(out_npy),
            xyz_transl=list(xyz_transl),
            xy_rotatate_deg=float(xy_rotate_deg),
            need_rotate=bool(need_rotate),
        )
    finally:
        if hasattr(motion_data, "close"):
            motion_data.close()

    print(f"[OK] {seq_name}: {src_npz} -> {out_npy}")
    return True


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seq-names",
        nargs="+",
        help="Sequence names to process. If omitted, auto-discovers under input-root.",
    )
    parser.add_argument("--hmr-type", default="gv", help="Name of the HMR subfolder to use.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT, help="Root containing post_scene outputs.")
    parser.add_argument("--motion-root", type=Path, default=DEFAULT_MOTION_ROOT, help="Where to save MotionTracking .npy.")
    parser.add_argument("--date", default="0921", help="Subfolder under motion-root (e.g. 0921).")
    parser.add_argument("--method", default="ours", help="Suffix used in output file name: <seq>_<method>.npy.")
    parser.add_argument(
        "--xyz-transl",
        type=_parse_xyz,
        default=(0.0, 0.0, 0.0),
        help="Extra xyz translation passed to get_smpl_robot, e.g. '0 0 0'.",
    )
    parser.add_argument("--xy-rotate-deg", type=float, default=0.0, help="Extra xy-plane rotation in degrees.")
    parser.add_argument("--need-rotate", action="store_true", help="Apply legacy pre-rotation in get_smpl_robot.")
    parser.add_argument("--strict", action="store_true", help="Exit with non-zero status if any sequence fails.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)

    input_root = args.input_root if args.input_root.is_absolute() else (REPO_ROOT / args.input_root)
    motion_root = args.motion_root if args.motion_root.is_absolute() else (REPO_ROOT / args.motion_root)

    seq_names = args.seq_names if args.seq_names else discover_sequences(input_root, args.hmr_type)
    if not seq_names:
        print(f"[WARN] No sequences found under {input_root} for hmr_type={args.hmr_type}")
        return

    get_smpl_robot = _load_get_smpl_robot()

    ok = 0
    fail = 0
    for seq_name in seq_names:
        success = process_sequence(
            seq_name=seq_name,
            input_root=input_root,
            motion_root=motion_root,
            date=args.date,
            method=args.method,
            hmr_type=args.hmr_type,
            xyz_transl=args.xyz_transl,
            xy_rotate_deg=args.xy_rotate_deg,
            need_rotate=args.need_rotate,
            get_smpl_robot=get_smpl_robot,
        )
        ok += int(success)
        fail += int(not success)

    print(f"[DONE] success={ok}, failed={fail}, output_root={motion_root / args.date}")
    if args.strict and fail > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
