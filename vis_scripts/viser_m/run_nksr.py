#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_input_npz(seq_name: str, hmr_type: str) -> Path:
    return _repo_root() / "results" / "output" / "scene" / seq_name / hmr_type / "nksr_input" / "pointcloud_world.npz"


def _default_output_dir(seq_name: str, hmr_type: str) -> Path:
    return _repo_root() / "results" / "output" / "scene" / seq_name / hmr_type / "nksr"


def _load_points(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path) as f:
        points = np.asarray(f["points"], dtype=np.float32)
        normals = np.asarray(f["normals"], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape [N, 3], got {points.shape}")
    if normals.shape != points.shape:
        raise ValueError(f"Expected normals with shape {points.shape}, got {normals.shape}")
    return points, normals


def _maybe_subsample(points: np.ndarray, normals: np.ndarray, max_input_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_input_points <= 0 or points.shape[0] <= max_input_points:
        return points, normals
    rng = np.random.default_rng(0)
    keep_idx = rng.choice(points.shape[0], size=max_input_points, replace=False)
    keep_idx.sort()
    return points[keep_idx], normals[keep_idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NKSR on CRISP pointcloud_world.npz and export a mesh.")
    parser.add_argument("--sequence-name", type=str, help="Sequence name under results/output/scene/<SEQ>/...")
    parser.add_argument("--hmr-type", type=str, default="gv")
    parser.add_argument("--input-npz", type=Path, help="Explicit NKSR input npz. Overrides --sequence-name.")
    parser.add_argument("--output-dir", type=Path, help="Directory for exported mesh files.")
    parser.add_argument("--detail-level", type=float, default=0.1, help="NKSR detail level in [0, 1].")
    parser.add_argument("--voxel-size", type=float, default=None, help="Explicit NKSR voxel size. Overrides detail level.")
    parser.add_argument("--chunk-size", type=float, default=-1.0, help="Chunk size for large scenes. Use >0 to enable chunked reconstruction.")
    parser.add_argument("--mise-iter", type=int, default=1)
    parser.add_argument("--max-input-points", type=int, default=-1, help="Optional random subsample for quick tests.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.input_npz is None and not args.sequence_name:
        raise SystemExit("Provide either --sequence-name or --input-npz.")

    input_npz = args.input_npz or _default_input_npz(args.sequence_name, args.hmr_type)
    if not input_npz.is_file():
        raise SystemExit(f"NKSR input not found: {input_npz}")

    output_dir = args.output_dir or _default_output_dir(args.sequence_name, args.hmr_type)
    output_dir.mkdir(parents=True, exist_ok=True)

    points_np, normals_np = _load_points(input_npz)
    points_np, normals_np = _maybe_subsample(points_np, normals_np, args.max_input_points)

    device = torch.device(args.device)

    import nksr

    input_xyz = torch.from_numpy(points_np).to(device=device, dtype=torch.float32)
    input_normal = torch.from_numpy(normals_np).to(device=device, dtype=torch.float32)

    reconstructor = nksr.Reconstructor(device)
    if args.chunk_size > 0:
        reconstructor.chunk_tmp_device = torch.device("cpu")

    reconstruct_kwargs = {
        "normal": input_normal,
        "detail_level": args.detail_level,
        "chunk_size": args.chunk_size,
    }
    if args.voxel_size is not None:
        reconstruct_kwargs["voxel_size"] = args.voxel_size
        reconstruct_kwargs["detail_level"] = None

    field = reconstructor.reconstruct(input_xyz, **reconstruct_kwargs)
    mesh = field.extract_dual_mesh(mise_iter=args.mise_iter)

    vertices = mesh.v.detach().cpu().numpy()
    faces = mesh.f.detach().cpu().numpy()
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    obj_path = output_dir / "scene_mesh_nksr.obj"
    ply_path = output_dir / "scene_mesh_nksr.ply"
    meta_path = output_dir / "scene_mesh_nksr.json"

    tri_mesh.export(obj_path)
    tri_mesh.export(ply_path)

    meta = {
        "input_npz": str(input_npz),
        "output_dir": str(output_dir),
        "device": str(device),
        "input_points": int(points_np.shape[0]),
        "detail_level": args.detail_level,
        "voxel_size": args.voxel_size,
        "chunk_size": args.chunk_size,
        "mise_iter": args.mise_iter,
        "vertices": int(vertices.shape[0]),
        "faces": int(faces.shape[0]),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
