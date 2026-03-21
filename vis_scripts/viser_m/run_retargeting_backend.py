#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pickle
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


THIS_FILE = Path(__file__).resolve()
CRISP_ROOT = THIS_FILE.parents[2]
FAR_ROOT = CRISP_ROOT.parent
GMR_ROOT = FAR_ROOT / "GMR"
HOLOSOMA_RT_ROOT = FAR_ROOT / "holosoma" / "src" / "holosoma_retargeting"

DEFAULT_POST_SCENE_ROOT = CRISP_ROOT / "results" / "output" / "post_scene"
DEFAULT_HMR_INIT_ROOT = CRISP_ROOT / "results" / "init" / "hmr"
DEFAULT_OUTPUT_ROOT = CRISP_ROOT / "results" / "output" / "retargeting"
DEFAULT_GMR_BODY_MODELS = GMR_ROOT / "assets" / "body_models"
DEFAULT_CRISP_SMPLX_MODELS = CRISP_ROOT / "prep" / "data" / "smplx" / "models" / "smplx"

HOLOSOMA_ROBOT_URDFS = {
    "g1": HOLOSOMA_RT_ROOT / "models" / "g1" / "g1_29dof.urdf",
    "t1": HOLOSOMA_RT_ROOT / "models" / "t1" / "t1_23dof.urdf",
}

COMMON_TO_GMR_ROBOT = {
    "g1": "unitree_g1",
    "t1": "booster_t1",
}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _run(cmd: Sequence[str], *, cwd: Path | None = None, dry_run: bool = False) -> None:
    rendered = shlex.join(str(part) for part in cmd)
    if cwd is not None:
        _log(f"[cmd] (cwd={cwd}) {rendered}")
    else:
        _log(f"[cmd] {rendered}")
    if dry_run:
        return
    subprocess.run([str(part) for part in cmd], cwd=str(cwd) if cwd else None, check=True)


def _ensure_file(path: Path, desc: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {desc}: {path}")
    return path


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_holosoma_input(
    seq_name: str,
    hmr_type: str,
    post_scene_root: Path,
    output_root: Path,
) -> tuple[Path, Path]:
    hmr_dir = post_scene_root / seq_name / hmr_type / "hmr"
    direct_npz = hmr_dir / f"{seq_name}.npz"
    if direct_npz.exists():
        return hmr_dir, direct_npz

    fallback_npz = hmr_dir / "hps_track_smplx.npz"
    if fallback_npz.exists():
        staging_dir = _ensure_dir(output_root / "_inputs" / "holosoma" / seq_name / hmr_type)
        staged_npz = staging_dir / f"{seq_name}.npz"
        if staged_npz.exists() or staged_npz.is_symlink():
            staged_npz.unlink()
        staged_npz.symlink_to(fallback_npz)
        return staging_dir, staged_npz

    raise FileNotFoundError(
        f"Could not find holosoma-ready SMPL-X joints for sequence '{seq_name}'. "
        f"Checked: {direct_npz} and {fallback_npz}"
    )


def _resolve_gmr_input(seq_name: str, hmr_init_root: Path) -> Path:
    return _ensure_file(hmr_init_root / seq_name / "hmr4d_results.pt", "GMR GVHMR prediction")


def _ensure_gmr_body_models(
    gmr_body_models_root: Path,
    crisp_smplx_models_root: Path,
    *,
    dry_run: bool = False,
) -> Path:
    smplx_dir = gmr_body_models_root / "smplx"
    needed = ("SMPLX_NEUTRAL.pkl", "SMPLX_MALE.pkl", "SMPLX_FEMALE.pkl")

    if all((smplx_dir / name).exists() for name in needed):
        return smplx_dir

    if not crisp_smplx_models_root.exists():
        raise FileNotFoundError(
            "GMR needs SMPL-X body models, but none were found under "
            f"{gmr_body_models_root} or {crisp_smplx_models_root}"
        )

    if dry_run:
        _log(f"[dry-run] Would link {smplx_dir} -> {crisp_smplx_models_root}")
        return smplx_dir

    _ensure_dir(gmr_body_models_root)
    if smplx_dir.exists() or smplx_dir.is_symlink():
        if smplx_dir.is_symlink() or smplx_dir.is_file():
            smplx_dir.unlink()
        else:
            raise RuntimeError(
                f"Refusing to replace existing directory at {smplx_dir}. "
                "Move it aside or populate it with SMPL-X models directly."
            )
    smplx_dir.symlink_to(crisp_smplx_models_root)
    return smplx_dir


def _resolve_holosoma_robot(robot: str) -> tuple[str, Path]:
    if robot not in HOLOSOMA_ROBOT_URDFS:
        supported = ", ".join(sorted(HOLOSOMA_ROBOT_URDFS))
        raise ValueError(f"Holosoma backend only supports: {supported}. Got: {robot}")
    urdf = _ensure_file(HOLOSOMA_ROBOT_URDFS[robot], "holosoma robot URDF")
    return robot, urdf


def _resolve_gmr_robot(robot: str, explicit_gmr_robot: str | None) -> str:
    if explicit_gmr_robot:
        return explicit_gmr_robot
    return COMMON_TO_GMR_ROBOT.get(robot, robot)


def _run_holosoma(args: argparse.Namespace) -> Path:
    robot_id, robot_urdf = _resolve_holosoma_robot(args.robot)
    data_path, motion_npz = _resolve_holosoma_input(args.seq_name, args.hmr_type, args.post_scene_root, args.output_root)
    out_dir = _ensure_dir(args.output_root / "holosoma" / args.seq_name / robot_id)
    out_npz = out_dir / f"{args.seq_name}.npz"

    _log(f"[holosoma] input:  {motion_npz}")
    _log(f"[holosoma] output: {out_npz}")

    cmd = [
        "conda",
        "run",
        "-n",
        args.holosoma_env,
        "python",
        str(HOLOSOMA_RT_ROOT / "examples" / "robot_retarget.py"),
        "--robot",
        robot_id,
        "--task-type",
        "robot_only",
        "--task-name",
        args.seq_name,
        "--data_format",
        "smplx",
        "--data_path",
        str(data_path),
        "--save_dir",
        str(out_dir),
        "--robot-config.robot-urdf-file",
        str(robot_urdf),
    ]
    _run(cmd, cwd=HOLOSOMA_RT_ROOT, dry_run=args.dry_run)

    if not args.dry_run:
        _ensure_file(out_npz, "holosoma retargeted output")
    return out_npz


def _run_gmr(args: argparse.Namespace) -> Path:
    gmr_robot = _resolve_gmr_robot(args.robot, args.gmr_robot)
    gvhmr_pred = _resolve_gmr_input(args.seq_name, args.hmr_init_root)
    _ensure_gmr_body_models(args.gmr_body_models_root, args.crisp_smplx_models_root, dry_run=args.dry_run)

    out_dir = _ensure_dir(args.output_root / "gmr" / args.seq_name / gmr_robot)
    raw_pkl = out_dir / f"{args.seq_name}_{gmr_robot}.pkl"
    qpos_npz = out_dir / f"{args.seq_name}_{gmr_robot}_qpos.npz"

    _log(f"[gmr] input:       {gvhmr_pred}")
    _log(f"[gmr] raw output:  {raw_pkl}")
    _log(f"[gmr] qpos output: {qpos_npz}")

    cmd = [
        "conda",
        "run",
        "-n",
        args.gmr_env,
        "python",
        str(THIS_FILE),
        "--_internal-gmr-run",
        "--_gmr-input",
        str(gvhmr_pred),
        "--_gmr-raw-pkl",
        str(raw_pkl),
        "--_gmr-qpos-npz",
        str(qpos_npz),
        "--_gmr-robot-id",
        gmr_robot,
        "--_gmr-tgt-fps",
        str(args.gmr_tgt_fps),
    ]
    _run(cmd, cwd=CRISP_ROOT, dry_run=args.dry_run)

    if not args.dry_run:
        _ensure_file(raw_pkl, "GMR raw pickle output")
        _ensure_file(qpos_npz, "GMR normalized qpos output")
    return qpos_npz


def _run_gmr_internal(args: argparse.Namespace) -> None:
    gmr_input = _ensure_file(args._gmr_input, "internal GMR input")
    raw_pkl = args._gmr_raw_pkl
    qpos_npz = args._gmr_qpos_npz
    gmr_robot = args._gmr_robot_id
    tgt_fps = args._gmr_tgt_fps

    import torch

    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.utils.smpl import get_gvhmr_data_offline_fast, load_gvhmr_pred_file

    smplx_root = _ensure_file(DEFAULT_GMR_BODY_MODELS / "smplx" / "SMPLX_NEUTRAL.pkl", "GMR SMPL-X model").parent.parent

    smplx_data, body_model, smplx_output, actual_human_height = load_gvhmr_pred_file(str(gmr_input), smplx_root)
    smplx_frames, aligned_fps = get_gvhmr_data_offline_fast(
        smplx_data,
        body_model,
        smplx_output,
        tgt_fps=tgt_fps,
    )

    retargeter = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=gmr_robot,
    )

    qpos_list: list[np.ndarray] = []
    num_frames = len(smplx_frames)
    for idx, frame in enumerate(smplx_frames):
        qpos = np.asarray(retargeter.retarget(frame), dtype=np.float32)
        qpos_list.append(qpos)
        if idx == 0 or (idx + 1) == num_frames or (idx + 1) % 50 == 0:
            _log(f"[gmr-internal] processed {idx + 1}/{num_frames} frames")

    qpos_arr = np.asarray(qpos_list, dtype=np.float32)
    root_pos = qpos_arr[:, :3]
    root_rot_wxyz = qpos_arr[:, 3:7]
    dof_pos = qpos_arr[:, 7:]
    root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]

    _ensure_dir(raw_pkl.parent)
    with raw_pkl.open("wb") as f:
        pickle.dump(
            {
                "fps": aligned_fps,
                "root_pos": root_pos,
                "root_rot": root_rot_xyzw,
                "dof_pos": dof_pos,
                "local_body_pos": None,
                "link_body_list": None,
            },
            f,
        )

    np.savez_compressed(
        qpos_npz,
        qpos=qpos_arr,
        fps=np.float32(aligned_fps),
        backend="gmr",
        robot=gmr_robot,
        gvhmr_pred_file=str(gmr_input),
        raw_pickle=str(raw_pkl),
    )

    _log(f"[gmr-internal] saved raw pickle: {raw_pkl}")
    _log(f"[gmr-internal] saved qpos npz:  {qpos_npz}")

    # Avoid torch shutdown warnings keeping references alive longer than needed.
    del smplx_output
    del body_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run retargeting from CRISP-Real2Sim outputs using either holosoma or GMR. "
            "Holosoma consumes post-scene SMPL-X joints, while GMR consumes GVHMR predictions."
        )
    )
    parser.add_argument("--backend", choices=("holosoma", "gmr"), help="Retargeting backend to run.")
    parser.add_argument("--seq-name", help="Sequence name under CRISP-Real2Sim results.")
    parser.add_argument("--hmr-type", default="gv", help="Subfolder under post_scene/<seq>/ (default: gv).")
    parser.add_argument(
        "--robot",
        default="g1",
        help=(
            "Robot identifier. "
            "For holosoma, supported values are g1 and t1. "
            "For GMR, g1/t1 are mapped to unitree_g1/booster_t1 unless --gmr-robot is set."
        ),
    )
    parser.add_argument("--gmr-robot", default=None, help="Explicit GMR robot id, e.g. fourier_gr3.")
    parser.add_argument("--holosoma-env", default="retgt", help="Conda environment for holosoma retargeting.")
    parser.add_argument("--gmr-env", default="gmr", help="Conda environment for GMR retargeting.")
    parser.add_argument("--gmr-tgt-fps", type=int, default=30, help="Target FPS for offline GMR retargeting.")
    parser.add_argument("--post-scene-root", type=Path, default=DEFAULT_POST_SCENE_ROOT)
    parser.add_argument("--hmr-init-root", type=Path, default=DEFAULT_HMR_INIT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gmr-body-models-root", type=Path, default=DEFAULT_GMR_BODY_MODELS)
    parser.add_argument("--crisp-smplx-models-root", type=Path, default=DEFAULT_CRISP_SMPLX_MODELS)
    parser.add_argument("--dry-run", action="store_true", help="Print commands and inferred paths without running.")

    parser.add_argument("--_internal-gmr-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_gmr-input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_gmr-raw-pkl", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_gmr-qpos-npz", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_gmr-robot-id", help=argparse.SUPPRESS)
    parser.add_argument("--_gmr-tgt-fps", type=int, default=30, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args._internal_gmr_run:
        required = {
            "_gmr_input": args._gmr_input,
            "_gmr_raw_pkl": args._gmr_raw_pkl,
            "_gmr_qpos_npz": args._gmr_qpos_npz,
            "_gmr_robot_id": args._gmr_robot_id,
        }
        missing = [key for key, value in required.items() if value in (None, "")]
        if missing:
            raise SystemExit(f"Missing internal GMR args: {', '.join(missing)}")
        _run_gmr_internal(args)
        return

    if not args.backend or not args.seq_name:
        parser.error("--backend and --seq-name are required")

    args.post_scene_root = args.post_scene_root.resolve()
    args.hmr_init_root = args.hmr_init_root.resolve()
    args.output_root = args.output_root.resolve()
    args.gmr_body_models_root = args.gmr_body_models_root.resolve()
    args.crisp_smplx_models_root = args.crisp_smplx_models_root.resolve()

    if args.backend == "holosoma":
        output = _run_holosoma(args)
    else:
        output = _run_gmr(args)

    _log(f"[done] backend={args.backend} seq={args.seq_name} output={output}")


if __name__ == "__main__":
    main()
