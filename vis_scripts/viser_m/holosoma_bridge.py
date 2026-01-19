#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np


def load_obj_z_stats(obj_path: Path) -> tuple[float, float, float, float]:
    """
    Parse an .obj and return:
      min_z, max_z, mesh_height=(max_z - min_z), height_offset=(-min_z)
    """
    min_z = None
    max_z = None
    with obj_path.open("r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    z = float(parts[3])
                    min_z = z if min_z is None else min(min_z, z)
                    max_z = z if max_z is None else max(max_z, z)

    if min_z is None or max_z is None:
        raise RuntimeError(f"No vertices found in: {obj_path}")

    mesh_height = float(max_z - min_z)
    height_offset = float(-min_z)
    return float(min_z), float(max_z), mesh_height, height_offset


def pick_joints_array(npz: np.lib.npyio.NpzFile, src_path: Path) -> np.ndarray:
    """Try common keys for joints and normalize to (T, J, 3)."""
    keys = list(npz.keys())
    candidates = [
        "joints", "J", "j3d", "joints3d", "joints_world", "global_joint_positions",
        "pred_joints", "post_scene", "joint_pos",
    ]

    arr = None
    chosen_key = None
    for k in candidates:
        if k in npz:
            arr = npz[k]
            chosen_key = k
            break

    if arr is None:
        raise KeyError(
            f"Could not find joints array in {src_path}.\n"
            f"Available keys: {keys}\n"
            f"Add your key name to `candidates` in pick_joints_array()."
        )

    arr = np.asarray(arr)

    # (T, J, 3)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr
    # (T, 3, J)
    if arr.ndim == 3 and arr.shape[1] == 3:
        return np.transpose(arr, (0, 2, 1))
    # (T, J*3)
    if arr.ndim == 2 and arr.shape[-1] % 3 == 0:
        J = arr.shape[-1] // 3
        return arr.reshape(arr.shape[0], J, 3)

    raise ValueError(f"Unsupported joints shape {arr.shape} from key '{chosen_key}' in {src_path}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", type=str, required=True)

    # Root of CRISP outputs (change default to your real path pattern)
    parser.add_argument(
        "--crisp_big_path",
        type=str,
        default="../../results/output/scene",
        help="Root folder containing per-seq CRISP outputs. Default: <repo_root>/crisp_big (best-effort).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output dir. Default: <this_script_dir>/../../results/output/rl_scene",
    )

    parser.add_argument(
        "--num_joints",
        type=int,
        default=22,
        help="How many joints to keep (default 22).",
    )

    args = parser.parse_args()

    # Best-effort repo root guess: scripts/.. or current file's parent/..
    script_dir = Path(__file__).resolve().parent
    repo_root = (script_dir / "..").resolve()

    crisp_big_path = Path(args.crisp_big_path).expanduser().resolve() if args.crisp_big_path else (repo_root / "crisp_big")
    crisp_seq_dir = crisp_big_path / args.seq_name / Path('gv')

    joint_file = crisp_seq_dir / Path('hmr/hps_track_smplx.npz')#/ args.joint_rel
    print(joint_file)
    mesh_file = crisp_seq_dir / Path('scene_mesh_sqs/scene_mesh_sqs.obj') #/ args.mesh_rel

    if not crisp_seq_dir.exists():
        raise FileNotFoundError(f"Sequence folder not found: {crisp_seq_dir}")
    if not joint_file.exists():
        raise FileNotFoundError(f"Joint file not found: {joint_file}")
    if not mesh_file.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_file}")

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else (script_dir / "../../results/output/rl_scene")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.seq_name}.npz"

    min_z, max_z, mesh_height, height_offset = load_obj_z_stats(mesh_file)

    data = np.load(joint_file, allow_pickle=True)

    joints = pick_joints_array(data, joint_file)
    joints = joints[:, : args.num_joints, :]  # (T, J, 3)

    np.savez_compressed(
        out_file,
        global_joint_positions=joints.astype(np.float32),
        height_offset=np.float32(height_offset),  # add to z so min-z -> 0
        mesh_height=np.float32(mesh_height),      # max-min
        mesh_min_z=np.float32(min_z),
        mesh_max_z=np.float32(max_z),
        seq_name=str(args.seq_name),
        src_joint_file=str(joint_file),
        src_mesh_file=str(mesh_file),
        crisp_seq_dir=str(crisp_seq_dir),
    )

    print(f"Saved: {out_file}")
    print(f"seq: {args.seq_name}")
    print(f"joints: {joint_file}")
    print(f"mesh:   {mesh_file}")
    print(f"joints shape: {joints.shape}")
    print(f"mesh min_z={min_z:.6f} max_z={max_z:.6f} range={mesh_height:.6f} offset={height_offset:.6f}")


if __name__ == "__main__":
    main()
