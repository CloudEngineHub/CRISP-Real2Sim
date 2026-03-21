#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray


@dataclass
class PieceSpec:
    piece_id: int
    piece_name: str
    walkable: bool
    center: np.ndarray
    solid_center: np.ndarray
    normal: np.ndarray
    piece_axis: np.ndarray
    tangent_u: np.ndarray
    tangent_v: np.ndarray
    half_size: np.ndarray
    core_half_size: np.ndarray
    solid_half_size: np.ndarray
    top_area: float
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    source_num_vertices: int
    source_num_faces: int
    support_face_count: int
    support_vertex_count: int
    support_up_dot: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a merged mesh-aligned reward heatmap from piece meshes. "
            "The exported heat arrays are aligned with the provided merged OBJ if one is given."
        )
    )
    parser.add_argument("--pieces-dir", type=Path, required=True, help="Directory containing part_*.obj meshes.")
    parser.add_argument(
        "--merged-obj",
        type=Path,
        default=None,
        help="Optional merged OBJ. If provided, heat arrays are aligned to this mesh instead of the local concat order.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for OBJ/NPY/NPZ/JSON/PLY/PNG.")
    parser.add_argument(
        "--piece-params-json",
        type=Path,
        default=None,
        help="Optional JSON with explicit piece slab parameters overriding auto-estimated walkable blocks.",
    )
    parser.add_argument(
        "--sqs-params",
        type=Path,
        default=None,
        help="Optional rotated SQ params (.npy or .npz). If provided, use SQ local z axes instead of mesh PCA axes.",
    )
    parser.add_argument(
        "--world-rotation-npy",
        type=Path,
        default=None,
        help="Optional world rotation / geo-calib matrix saved alongside the sequence. Stored in metadata exports.",
    )
    parser.add_argument("--slab-thickness", type=float, default=0.05, help="Full slab thickness in meters.")
    parser.add_argument(
        "--piece-normal-thresh",
        type=float,
        default=0.8,
        help="Minimum face normal alignment with the piece's thin-slab axis.",
    )
    parser.add_argument(
        "--world-z-veto-deg",
        type=float,
        default=80.0,
        help=(
            "Reject only near-vertical faces with respect to world +Z. "
            "Faces steeper than this angle are removed; other cases follow the piece normal."
        ),
    )
    parser.add_argument(
        "--world-up-max-angle-deg",
        type=float,
        default=45.0,
        help="Only faces whose normal is within this angle of world +Z are treated as valid support faces.",
    )
    parser.add_argument(
        "--top-band",
        type=float,
        default=0.03,
        help="Only faces within this vertical distance from the local piece max-z are treated as top faces.",
    )
    parser.add_argument(
        "--min-top-area",
        type=float,
        default=0.05,
        help="Minimum summed top-face area required to mark a piece as walkable.",
    )
    parser.add_argument(
        "--inset-scale",
        type=float,
        default=0.85,
        help="Shrink estimated support patch extents to stay away from edges.",
    )
    parser.add_argument(
        "--gaussian-scale-xy",
        type=float,
        default=0.55,
        help="XY Gaussian sigma as a fraction of the inferred half-size.",
    )
    parser.add_argument(
        "--gaussian-scale-z",
        type=float,
        default=0.5,
        help="Z Gaussian sigma as a fraction of the slab half-thickness.",
    )
    parser.add_argument(
        "--heat-shape",
        type=str,
        default="edge_aware",
        choices=("edge_aware", "gaussian"),
        help="Heat profile inside each reward block.",
    )
    parser.add_argument(
        "--flat-ratio-xy",
        type=float,
        default=0.68,
        help="Fraction of half-size kept near-uniform before edge decay begins in the tangent plane.",
    )
    parser.add_argument(
        "--flat-ratio-z",
        type=float,
        default=0.4,
        help="Fraction of half-thickness kept near-uniform before vertical decay begins.",
    )
    parser.add_argument(
        "--superellipse-power",
        type=float,
        default=10.0,
        help="Higher values make the XY heat footprint more rectangular.",
    )
    parser.add_argument(
        "--edge-drop-power",
        type=float,
        default=6.0,
        help="Higher values delay the drop until closer to the slab edge, then decay more sharply.",
    )
    parser.add_argument("--chunk-size", type=int, default=50000, help="Chunk size for heat evaluation.")
    parser.add_argument(
        "--core-ratio-xy",
        type=float,
        default=0.78,
        help="Safe landing core half-size as a fraction of the support half-size in the tangent plane.",
    )
    parser.add_argument(
        "--core-ratio-z",
        type=float,
        default=0.6,
        help="Safe landing core half-thickness as a fraction of the support slab half-thickness.",
    )
    parser.add_argument(
        "--solid-padding-xy",
        type=float,
        default=0.01,
        help="Extra xy padding added to each piece's solid OBB for penetration checks.",
    )
    parser.add_argument(
        "--solid-padding-z",
        type=float,
        default=0.005,
        help="Extra z padding added to each piece's solid OBB for penetration checks.",
    )
    parser.add_argument(
        "--atlas-resolution",
        type=int,
        default=512,
        help="Square resolution used for the exported 2.5D atlas heat/height lookup tables.",
    )
    parser.add_argument(
        "--atlas-padding",
        type=float,
        default=0.05,
        help="Extra world-space xy padding added around the scene bounds when baking the atlas.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="red_gray",
        help="Matplotlib colormap for PLY/PNG export. Use 'red_gray' for gray non-walkable + red walkable.",
    )
    return parser.parse_args()


def load_obj_mesh(path: Path) -> Mesh:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                idxs = [int(token.split("/")[0]) - 1 for token in line.split()[1:]]
                if len(idxs) < 3:
                    continue
                for i in range(1, len(idxs) - 1):
                    faces.append([idxs[0], idxs[i], idxs[i + 1]])
    if not vertices or not faces:
        raise ValueError(f"OBJ mesh is empty or invalid: {path}")
    return Mesh(np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32))


def write_obj_mesh(path: Path, mesh: Mesh) -> None:
    with path.open("w", encoding="utf-8") as f:
        for v in mesh.vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in mesh.faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def concat_meshes(meshes: list[Mesh]) -> Mesh:
    vertices = []
    faces = []
    vertex_offset = 0
    for mesh in meshes:
        vertices.append(mesh.vertices)
        faces.append(mesh.faces + vertex_offset)
        vertex_offset += mesh.vertices.shape[0]
    return Mesh(np.concatenate(vertices, axis=0), np.concatenate(faces, axis=0))


def normalize(vec: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-8:
        if fallback is None:
            raise ValueError("Cannot normalize a near-zero vector without a fallback.")
        return fallback.astype(np.float32)
    return (vec / norm).astype(np.float32)


def build_tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(reference, normal))) > 0.95:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    tangent_u = normalize(np.cross(normal, reference))
    tangent_v = normalize(np.cross(normal, tangent_u))
    return tangent_u, tangent_v


def face_geometry(mesh: Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tri_vertices = mesh.vertices[mesh.faces]
    edge0 = tri_vertices[:, 1] - tri_vertices[:, 0]
    edge1 = tri_vertices[:, 2] - tri_vertices[:, 0]
    face_normals_raw = np.cross(edge0, edge1)
    face_double_area = np.linalg.norm(face_normals_raw, axis=1)
    face_areas = 0.5 * face_double_area
    safe_normals = np.divide(
        face_normals_raw,
        face_double_area[:, None],
        out=np.zeros_like(face_normals_raw),
        where=face_double_area[:, None] > 1.0e-12,
    )
    face_centers = tri_vertices.mean(axis=1)
    return face_centers.astype(np.float32), safe_normals.astype(np.float32), face_areas.astype(np.float32)


def euler_zyx_to_matrix(euler_zyx: np.ndarray) -> np.ndarray:
    rz, ry, rx = [float(x) for x in euler_zyx]
    cz, sz = np.cos(rz), np.sin(rz)
    cy, sy = np.cos(ry), np.sin(ry)
    cx, sx = np.cos(rx), np.sin(rx)
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    return (rz_m @ ry_m @ rx_m).astype(np.float32)


def load_sqs_payload(path: Path | None) -> dict[str, np.ndarray] | None:
    if path is None or not path.exists():
        return None
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as payload:
            return {key: np.asarray(payload[key]) for key in payload.files}
    params = np.asarray(np.load(path, allow_pickle=True), dtype=np.float32)
    return {"params": params}


def estimate_piece_axis(vertices: np.ndarray) -> np.ndarray:
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(centered.shape[0], 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmin(eigvals))].astype(np.float32)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if float(np.dot(axis, world_up)) < 0.0:
        axis = -axis
    return normalize(axis, fallback=world_up)


def quantile_interval(values: np.ndarray, low: float = 0.05, high: float = 0.95) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    lo = float(np.quantile(values, low))
    hi = float(np.quantile(values, high))
    if hi < lo:
        hi = lo
    return lo, hi


def infer_piece_spec(
    mesh: Mesh,
    piece_id: int,
    piece_name: str,
    args: argparse.Namespace,
    sqs_row: np.ndarray | None = None,
) -> PieceSpec:
    face_centers, face_normals, face_areas = face_geometry(mesh)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    sqs_center_override = None
    sqs_half_axes_override = None
    if sqs_row is not None and sqs_row.shape[0] >= 11:
        sqs_rotation = euler_zyx_to_matrix(np.asarray(sqs_row[5:8], dtype=np.float32))
        piece_axis = normalize(sqs_rotation[:, 2], fallback=world_up)
        sqs_center_override = np.asarray(sqs_row[8:11], dtype=np.float32)
        sqs_half_axes_override = np.asarray(sqs_row[2:5], dtype=np.float32)
    else:
        piece_axis = estimate_piece_axis(mesh.vertices)

    world_up_dot = float(np.cos(np.deg2rad(args.world_up_max_angle_deg)))
    cap_mask = np.abs(face_normals @ piece_axis) > args.piece_normal_thresh
    support_mask = cap_mask & ((face_normals @ world_up) > world_up_dot)
    top_area = float(face_areas[support_mask].sum())

    walkable = bool(np.any(support_mask) and top_area >= args.min_top_area)

    if walkable:
        weighted_normal = (face_normals[support_mask] * face_areas[support_mask, None]).sum(axis=0)
        normal = normalize(weighted_normal, fallback=world_up)
        if normal[2] < 0.0:
            normal = -normal
        tangent_u, tangent_v = build_tangent_basis(normal)

        top_tri_vertices = mesh.vertices[mesh.faces[support_mask]].reshape(-1, 3)
        center = top_tri_vertices.mean(axis=0).astype(np.float32)

        delta = top_tri_vertices - center[None, :]
        u = delta @ tangent_u
        v = delta @ tangent_v
        w = delta @ normal

        u_lo, u_hi = quantile_interval(u)
        v_lo, v_hi = quantile_interval(v)
        w_center = float(np.quantile(w, 0.5))
        center = center + ((u_lo + u_hi) * 0.5) * tangent_u + ((v_lo + v_hi) * 0.5) * tangent_v + w_center * normal

        half_u = max((u_hi - u_lo) * 0.5 * args.inset_scale, 0.02)
        half_v = max((v_hi - v_lo) * 0.5 * args.inset_scale, 0.02)
        half_w = max(args.slab_thickness * 0.5, 1.0e-3)
        half_size = np.array([half_u, half_v, half_w], dtype=np.float32)
    else:
        normal = piece_axis
        tangent_u, tangent_v = build_tangent_basis(normal)
        center = mesh.vertices.mean(axis=0).astype(np.float32)
        half_size = np.array([0.0, 0.0, max(args.slab_thickness * 0.5, 1.0e-3)], dtype=np.float32)

    core_half_size = np.array(
        [
            max(half_size[0] * args.core_ratio_xy, 0.005),
            max(half_size[1] * args.core_ratio_xy, 0.005),
            max(half_size[2] * args.core_ratio_z, 0.001),
        ],
        dtype=np.float32,
    )

    solid_basis = np.stack([tangent_u, tangent_v, normal], axis=0).astype(np.float32)
    solid_local = mesh.vertices @ solid_basis.T
    solid_lo = solid_local.min(axis=0)
    solid_hi = solid_local.max(axis=0)
    solid_center_local = 0.5 * (solid_lo + solid_hi)
    solid_half_size = 0.5 * (solid_hi - solid_lo)
    solid_half_size = solid_half_size + np.array(
        [args.solid_padding_xy, args.solid_padding_xy, args.solid_padding_z],
        dtype=np.float32,
    )
    solid_center = (solid_center_local @ solid_basis).astype(np.float32)
    if sqs_center_override is not None:
        solid_center = sqs_center_override.astype(np.float32, copy=False)
    if sqs_half_axes_override is not None:
        solid_half_size = np.maximum(
            sqs_half_axes_override.astype(np.float32, copy=False)
            + np.array([args.solid_padding_xy, args.solid_padding_xy, args.solid_padding_z], dtype=np.float32),
            np.array([1.0e-3, 1.0e-3, 1.0e-3], dtype=np.float32),
        )
    aabb_min = mesh.vertices.min(axis=0).astype(np.float32)
    aabb_max = mesh.vertices.max(axis=0).astype(np.float32)

    return PieceSpec(
        piece_id=piece_id,
        piece_name=piece_name,
        walkable=walkable,
        center=center,
        solid_center=solid_center,
        normal=normal,
        piece_axis=piece_axis,
        tangent_u=tangent_u,
        tangent_v=tangent_v,
        half_size=half_size,
        core_half_size=core_half_size,
        solid_half_size=solid_half_size.astype(np.float32),
        top_area=top_area,
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        source_num_vertices=int(mesh.vertices.shape[0]),
        source_num_faces=int(mesh.faces.shape[0]),
        support_face_count=int(np.count_nonzero(support_mask)),
        support_vertex_count=int(np.unique(mesh.faces[support_mask]).size if np.any(support_mask) else 0),
        support_up_dot=float(face_normals[support_mask, 2].mean()) if np.any(support_mask) else 0.0,
    )


def load_piece_overrides(path: Path) -> dict[str, PieceSpec]:
    data = json.loads(path.read_text(encoding="utf-8"))
    piece_specs = data.get("piece_specs", [])
    overrides: dict[str, PieceSpec] = {}
    for item in piece_specs:
        piece_name = str(item["piece_name"])
        overrides[piece_name] = PieceSpec(
            piece_id=int(item.get("piece_id", -1)),
            piece_name=piece_name,
            walkable=bool(item["walkable"]),
            center=np.asarray(item["center"], dtype=np.float32),
            solid_center=np.asarray(item.get("solid_center", item["center"]), dtype=np.float32),
            normal=normalize(np.asarray(item["normal"], dtype=np.float32), fallback=np.array([0.0, 0.0, 1.0], dtype=np.float32)),
            piece_axis=normalize(np.asarray(item.get("piece_axis", item["normal"]), dtype=np.float32), fallback=np.array([0.0, 0.0, 1.0], dtype=np.float32)),
            tangent_u=normalize(np.asarray(item["tangent_u"], dtype=np.float32), fallback=np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            tangent_v=normalize(np.asarray(item["tangent_v"], dtype=np.float32), fallback=np.array([0.0, 1.0, 0.0], dtype=np.float32)),
            half_size=np.asarray(item["half_size"], dtype=np.float32),
            core_half_size=np.asarray(item.get("core_half_size", item["half_size"]), dtype=np.float32),
            solid_half_size=np.asarray(item.get("solid_half_size", item["half_size"]), dtype=np.float32),
            top_area=float(item.get("top_area", 0.0)),
            aabb_min=np.asarray(item.get("aabb_min", item["center"]), dtype=np.float32),
            aabb_max=np.asarray(item.get("aabb_max", item["center"]), dtype=np.float32),
            source_num_vertices=int(item.get("source_num_vertices", 0)),
            source_num_faces=int(item.get("source_num_faces", 0)),
            support_face_count=int(item.get("support_face_count", 0)),
            support_vertex_count=int(item.get("support_vertex_count", 0)),
            support_up_dot=float(item.get("support_up_dot", 0.0)),
        )
    return overrides


def mesh_alignment_summary(reference: Mesh, other: Mesh) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "same_num_vertices": bool(reference.vertices.shape[0] == other.vertices.shape[0]),
        "same_num_faces": bool(reference.faces.shape[0] == other.faces.shape[0]),
        "same_vertex_order": False,
        "same_face_order": False,
        "vertex_max_abs_diff": None,
        "face_max_abs_diff": None,
    }
    if summary["same_num_vertices"]:
        vertex_diff = np.abs(reference.vertices - other.vertices)
        summary["vertex_max_abs_diff"] = float(vertex_diff.max()) if vertex_diff.size else 0.0
        summary["same_vertex_order"] = bool(np.allclose(reference.vertices, other.vertices, atol=1.0e-6))
    if summary["same_num_faces"]:
        face_diff = np.abs(reference.faces - other.faces)
        summary["face_max_abs_diff"] = float(face_diff.max()) if face_diff.size else 0.0
        summary["same_face_order"] = bool(np.array_equal(reference.faces, other.faces))
    summary["topology_matches_exactly"] = bool(summary["same_vertex_order"] and summary["same_face_order"])
    return summary


def evaluate_heat(points: np.ndarray, piece_specs: list[PieceSpec], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    walkable_specs = [spec for spec in piece_specs if spec.walkable]
    if not walkable_specs:
        return np.zeros(points.shape[0], dtype=np.float32), -np.ones(points.shape[0], dtype=np.int32)

    centers = np.stack([spec.center for spec in walkable_specs], axis=0)
    tangent_u = np.stack([spec.tangent_u for spec in walkable_specs], axis=0)
    tangent_v = np.stack([spec.tangent_v for spec in walkable_specs], axis=0)
    normals = np.stack([spec.normal for spec in walkable_specs], axis=0)
    half_size = np.stack([spec.half_size for spec in walkable_specs], axis=0)

    sigma_u = np.maximum(half_size[:, 0] * args.gaussian_scale_xy, 0.01)
    sigma_v = np.maximum(half_size[:, 1] * args.gaussian_scale_xy, 0.01)
    sigma_w = np.maximum(half_size[:, 2] * args.gaussian_scale_z, 0.005)
    flat_u = np.maximum(half_size[:, 0] * args.flat_ratio_xy, 0.01)
    flat_v = np.maximum(half_size[:, 1] * args.flat_ratio_xy, 0.01)
    flat_w = np.maximum(half_size[:, 2] * args.flat_ratio_z, 0.002)
    edge_u = np.maximum(half_size[:, 0] - flat_u, 0.005)
    edge_v = np.maximum(half_size[:, 1] - flat_v, 0.005)
    edge_w = np.maximum(half_size[:, 2] - flat_w, 0.002)

    heat = np.zeros(points.shape[0], dtype=np.float32)
    best_idx = -np.ones(points.shape[0], dtype=np.int32)

    for start in range(0, points.shape[0], args.chunk_size):
        end = min(start + args.chunk_size, points.shape[0])
        chunk = points[start:end]
        delta = chunk[:, None, :] - centers[None, :, :]
        local_u = np.einsum("cpk,pk->cp", delta, tangent_u)
        local_v = np.einsum("cpk,pk->cp", delta, tangent_v)
        local_w = np.einsum("cpk,pk->cp", delta, normals)

        if args.heat_shape == "gaussian":
            scaled = (
                (local_u / sigma_u[None, :]) ** 2
                + (local_v / sigma_v[None, :]) ** 2
                + (local_w / sigma_w[None, :]) ** 2
            )
            chunk_heat = np.exp(-0.5 * scaled).astype(np.float32, copy=False)
        else:
            abs_u = np.abs(local_u)
            abs_v = np.abs(local_v)
            abs_w = np.abs(local_w)

            tx = np.clip((abs_u - flat_u[None, :]) / edge_u[None, :], 0.0, 1.0)
            ty = np.clip((abs_v - flat_v[None, :]) / edge_v[None, :], 0.0, 1.0)
            tz = np.clip((abs_w - flat_w[None, :]) / edge_w[None, :], 0.0, 1.0)

            p = max(float(args.superellipse_power), 1.0)
            t_xy = (np.power(tx, p) + np.power(ty, p)) ** (1.0 / p)
            heat_xy = 1.0 - np.power(np.clip(t_xy, 0.0, 1.0), args.edge_drop_power)
            heat_z = 1.0 - np.power(tz, args.edge_drop_power)

            inside_outer = (
                (abs_u <= half_size[None, :, 0])
                & (abs_v <= half_size[None, :, 1])
                & (abs_w <= half_size[None, :, 2])
            )
            chunk_heat = (heat_xy * heat_z * inside_outer.astype(np.float32)).astype(np.float32, copy=False)

        best_local = np.argmax(chunk_heat, axis=1)
        best_heat = chunk_heat[np.arange(chunk_heat.shape[0]), best_local]
        heat[start:end] = best_heat
        best_idx[start:end] = np.where(best_heat > 1.0e-8, best_local, -1).astype(np.int32)

    walkable_piece_indices = [i for i, spec in enumerate(piece_specs) if spec.walkable]
    if walkable_piece_indices:
        remapped = np.asarray(walkable_piece_indices, dtype=np.int32)
        valid_mask = best_idx >= 0
        best_idx[valid_mask] = remapped[best_idx[valid_mask]]

    return heat, best_idx


def edge_aware_heat_from_local(
    local_u: np.ndarray,
    local_v: np.ndarray,
    local_w: np.ndarray,
    half_size: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    if args.heat_shape == "gaussian":
        sigma_u = max(float(half_size[0]) * args.gaussian_scale_xy, 0.01)
        sigma_v = max(float(half_size[1]) * args.gaussian_scale_xy, 0.01)
        sigma_w = max(float(half_size[2]) * args.gaussian_scale_z, 0.005)
        scaled = (local_u / sigma_u) ** 2 + (local_v / sigma_v) ** 2 + (local_w / sigma_w) ** 2
        return np.exp(-0.5 * scaled).astype(np.float32, copy=False)

    abs_u = np.abs(local_u)
    abs_v = np.abs(local_v)
    abs_w = np.abs(local_w)

    flat_u = max(float(half_size[0]) * args.flat_ratio_xy, 0.01)
    flat_v = max(float(half_size[1]) * args.flat_ratio_xy, 0.01)
    flat_w = max(float(half_size[2]) * args.flat_ratio_z, 0.002)
    edge_u = max(float(half_size[0]) - flat_u, 0.005)
    edge_v = max(float(half_size[1]) - flat_v, 0.005)
    edge_w = max(float(half_size[2]) - flat_w, 0.002)

    tx = np.clip((abs_u - flat_u) / edge_u, 0.0, 1.0)
    ty = np.clip((abs_v - flat_v) / edge_v, 0.0, 1.0)
    tz = np.clip((abs_w - flat_w) / edge_w, 0.0, 1.0)

    p = max(float(args.superellipse_power), 1.0)
    t_xy = (np.power(tx, p) + np.power(ty, p)) ** (1.0 / p)
    heat_xy = 1.0 - np.power(np.clip(t_xy, 0.0, 1.0), args.edge_drop_power)
    heat_z = 1.0 - np.power(tz, args.edge_drop_power)
    inside_outer = (
        (abs_u <= float(half_size[0]))
        & (abs_v <= float(half_size[1]))
        & (abs_w <= float(half_size[2]))
    )
    return (heat_xy * heat_z * inside_outer.astype(np.float32)).astype(np.float32, copy=False)


def compute_piece_surface_heat(
    mesh: Mesh,
    spec: PieceSpec,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    face_centers, face_normals, _ = face_geometry(mesh)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    world_up_dot = float(np.cos(np.deg2rad(args.world_up_max_angle_deg)))
    support_mask = (
        (np.abs(face_normals @ spec.piece_axis) > args.piece_normal_thresh)
        & ((face_normals @ world_up) > world_up_dot)
    )

    face_heat = np.zeros(mesh.faces.shape[0], dtype=np.float32)
    if spec.walkable and np.any(support_mask):
        delta = face_centers[support_mask] - spec.center[None, :]
        local_u = delta @ spec.tangent_u
        local_v = delta @ spec.tangent_v
        local_w = delta @ spec.normal
        face_heat[support_mask] = edge_aware_heat_from_local(local_u, local_v, local_w, spec.half_size, args)

    vertex_heat = np.zeros(mesh.vertices.shape[0], dtype=np.float32)
    if np.any(support_mask):
        support_faces = mesh.faces[support_mask].reshape(-1)
        support_heat = np.repeat(face_heat[support_mask], 3)
        np.maximum.at(vertex_heat, support_faces, support_heat)

    return face_heat, vertex_heat, support_mask


def evaluate_surface_heat(
    mesh: Mesh,
    piece_specs: list[PieceSpec],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    walkable_specs = [spec for spec in piece_specs if spec.walkable]
    num_faces = mesh.faces.shape[0]
    num_vertices = mesh.vertices.shape[0]
    if not walkable_specs:
        return (
            np.zeros(num_faces, dtype=np.float32),
            -np.ones(num_faces, dtype=np.int32),
            np.zeros(num_vertices, dtype=np.float32),
            -np.ones(num_vertices, dtype=np.int32),
        )

    face_centers, face_normals, _ = face_geometry(mesh)
    centers = np.stack([spec.center for spec in walkable_specs], axis=0)
    tangent_u = np.stack([spec.tangent_u for spec in walkable_specs], axis=0)
    tangent_v = np.stack([spec.tangent_v for spec in walkable_specs], axis=0)
    normals = np.stack([spec.normal for spec in walkable_specs], axis=0)
    piece_axes = np.stack([spec.piece_axis for spec in walkable_specs], axis=0)
    half_size = np.stack([spec.half_size for spec in walkable_specs], axis=0)

    sigma_u = np.maximum(half_size[:, 0] * args.gaussian_scale_xy, 0.01)
    sigma_v = np.maximum(half_size[:, 1] * args.gaussian_scale_xy, 0.01)
    sigma_w = np.maximum(half_size[:, 2] * args.gaussian_scale_z, 0.005)
    flat_u = np.maximum(half_size[:, 0] * args.flat_ratio_xy, 0.01)
    flat_v = np.maximum(half_size[:, 1] * args.flat_ratio_xy, 0.01)
    flat_w = np.maximum(half_size[:, 2] * args.flat_ratio_z, 0.002)
    edge_u = np.maximum(half_size[:, 0] - flat_u, 0.005)
    edge_v = np.maximum(half_size[:, 1] - flat_v, 0.005)
    edge_w = np.maximum(half_size[:, 2] - flat_w, 0.002)

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    world_up_dot = float(np.cos(np.deg2rad(args.world_up_max_angle_deg)))

    face_heat = np.zeros(num_faces, dtype=np.float32)
    best_piece_f = -np.ones(num_faces, dtype=np.int32)

    for start in range(0, num_faces, args.chunk_size):
        end = min(start + args.chunk_size, num_faces)
        chunk_centers = face_centers[start:end]
        chunk_normals = face_normals[start:end]
        delta = chunk_centers[:, None, :] - centers[None, :, :]
        local_u = np.einsum("cpk,pk->cp", delta, tangent_u)
        local_v = np.einsum("cpk,pk->cp", delta, tangent_v)
        local_w = np.einsum("cpk,pk->cp", delta, normals)

        if args.heat_shape == "gaussian":
            scaled = (
                (local_u / sigma_u[None, :]) ** 2
                + (local_v / sigma_v[None, :]) ** 2
                + (local_w / sigma_w[None, :]) ** 2
            )
            chunk_heat = np.exp(-0.5 * scaled).astype(np.float32, copy=False)
        else:
            abs_u = np.abs(local_u)
            abs_v = np.abs(local_v)
            abs_w = np.abs(local_w)

            tx = np.clip((abs_u - flat_u[None, :]) / edge_u[None, :], 0.0, 1.0)
            ty = np.clip((abs_v - flat_v[None, :]) / edge_v[None, :], 0.0, 1.0)
            tz = np.clip((abs_w - flat_w[None, :]) / edge_w[None, :], 0.0, 1.0)

            p = max(float(args.superellipse_power), 1.0)
            t_xy = (np.power(tx, p) + np.power(ty, p)) ** (1.0 / p)
            heat_xy = 1.0 - np.power(np.clip(t_xy, 0.0, 1.0), args.edge_drop_power)
            heat_z = 1.0 - np.power(tz, args.edge_drop_power)
            inside_outer = (
                (abs_u <= half_size[None, :, 0])
                & (abs_v <= half_size[None, :, 1])
                & (abs_w <= half_size[None, :, 2])
            )
            chunk_heat = (heat_xy * heat_z * inside_outer.astype(np.float32)).astype(np.float32, copy=False)

        normal_gate = (
            (np.abs(chunk_normals @ piece_axes.T) > args.piece_normal_thresh)
            & ((chunk_normals @ world_up)[:, None] > world_up_dot)
        )
        chunk_heat = chunk_heat * normal_gate.astype(np.float32)

        best_local = np.argmax(chunk_heat, axis=1)
        best_heat = chunk_heat[np.arange(chunk_heat.shape[0]), best_local]
        face_heat[start:end] = best_heat
        best_piece_f[start:end] = np.where(best_heat > 1.0e-8, best_local, -1).astype(np.int32)

    walkable_piece_ids = np.asarray([spec.piece_id for spec in walkable_specs], dtype=np.int32)
    valid_face_mask = best_piece_f >= 0
    best_piece_f[valid_face_mask] = walkable_piece_ids[best_piece_f[valid_face_mask]]

    vertex_heat = np.zeros(num_vertices, dtype=np.float32)
    best_piece_v = -np.ones(num_vertices, dtype=np.int32)
    for face_idx, face in enumerate(mesh.faces):
        value = float(face_heat[face_idx])
        if value <= 1.0e-8:
            continue
        piece_id = int(best_piece_f[face_idx])
        for vertex_idx in face:
            if value > float(vertex_heat[vertex_idx]):
                vertex_heat[vertex_idx] = value
                best_piece_v[vertex_idx] = piece_id

    return face_heat, best_piece_f, vertex_heat, best_piece_v


def bake_atlas(
    mesh: Mesh,
    piece_specs: list[PieceSpec],
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    walkable_specs = [spec for spec in piece_specs if spec.walkable]
    resolution = int(args.atlas_resolution)
    padding = float(args.atlas_padding)

    xy_min = mesh.vertices[:, :2].min(axis=0).astype(np.float32) - padding
    xy_max = mesh.vertices[:, :2].max(axis=0).astype(np.float32) + padding
    extent = np.maximum(xy_max - xy_min, 1.0e-3)
    cell_size = extent / max(resolution - 1, 1)

    xs = np.linspace(float(xy_min[0]), float(xy_max[0]), num=resolution, dtype=np.float32)
    ys = np.linspace(float(xy_min[1]), float(xy_max[1]), num=resolution, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")

    heat = np.zeros((resolution, resolution), dtype=np.float32)
    valid = np.zeros((resolution, resolution), dtype=np.uint8)
    top_z = np.full((resolution, resolution), np.nan, dtype=np.float32)
    piece_id = np.full((resolution, resolution), -1, dtype=np.int16)
    best_z = np.full((resolution, resolution), -np.inf, dtype=np.float32)

    for spec in walkable_specs:
        if abs(float(spec.normal[2])) < 1.0e-4:
            continue

        dx = grid_x - float(spec.center[0])
        dy = grid_y - float(spec.center[1])
        plane_z = float(spec.center[2]) - (
            float(spec.normal[0]) * dx + float(spec.normal[1]) * dy
        ) / float(spec.normal[2])
        points = np.stack([grid_x, grid_y, plane_z], axis=-1)
        delta = points - spec.center.reshape(1, 1, 3)
        local_u = np.einsum("hwk,k->hw", delta, spec.tangent_u)
        local_v = np.einsum("hwk,k->hw", delta, spec.tangent_v)
        local_w = np.einsum("hwk,k->hw", delta, spec.normal)
        piece_heat = edge_aware_heat_from_local(local_u, local_v, local_w, spec.half_size, args)
        piece_valid = piece_heat > 1.0e-8

        prefer = piece_valid & ((plane_z > best_z + 1.0e-5) | ((np.abs(plane_z - best_z) <= 1.0e-5) & (piece_heat > heat)))
        heat[prefer] = piece_heat[prefer]
        valid[prefer] = 1
        top_z[prefer] = plane_z[prefer]
        piece_id[prefer] = int(spec.piece_id)
        best_z[prefer] = plane_z[prefer]

    return {
        "heat": heat.astype(np.float16),
        "valid": valid,
        "top_z": top_z.astype(np.float16),
        "piece_id": piece_id,
        "origin_xy": xy_min.astype(np.float32),
        "cell_size_xy": cell_size.astype(np.float32),
        "scene_extent_xy": extent.astype(np.float32),
    }


def build_piece_metadata(piece_specs: list[PieceSpec], mesh: Mesh) -> dict[str, np.ndarray]:
    num_pieces = len(piece_specs)
    rot_w2p = np.stack(
        [np.stack([spec.tangent_u, spec.tangent_v, spec.normal], axis=0) for spec in piece_specs],
        axis=0,
    ).astype(np.float32)
    metadata = {
        "piece_ids": np.asarray([spec.piece_id for spec in piece_specs], dtype=np.int32),
        "walkable": np.asarray([spec.walkable for spec in piece_specs], dtype=np.bool_),
        "piece_centers_w": np.stack([spec.center for spec in piece_specs], axis=0).astype(np.float32),
        "piece_solid_centers_w": np.stack([spec.solid_center for spec in piece_specs], axis=0).astype(np.float32),
        "piece_axis_world": np.stack([spec.piece_axis for spec in piece_specs], axis=0).astype(np.float32),
        "piece_rot_w2p": rot_w2p,
        "support_half_size": np.stack([spec.half_size for spec in piece_specs], axis=0).astype(np.float32),
        "core_half_size": np.stack([spec.core_half_size for spec in piece_specs], axis=0).astype(np.float32),
        "solid_half_size": np.stack([spec.solid_half_size for spec in piece_specs], axis=0).astype(np.float32),
        "aabb_min_w": np.stack([spec.aabb_min for spec in piece_specs], axis=0).astype(np.float32),
        "aabb_max_w": np.stack([spec.aabb_max for spec in piece_specs], axis=0).astype(np.float32),
        "top_area": np.asarray([spec.top_area for spec in piece_specs], dtype=np.float32),
        "support_face_count": np.asarray([spec.support_face_count for spec in piece_specs], dtype=np.int32),
        "support_vertex_count": np.asarray([spec.support_vertex_count for spec in piece_specs], dtype=np.int32),
        "support_up_dot": np.asarray([spec.support_up_dot for spec in piece_specs], dtype=np.float32),
        "piece_name_utf8": np.asarray([spec.piece_name for spec in piece_specs], dtype=f"<U{max(len(spec.piece_name) for spec in piece_specs)}"),
        "scene_origin_w": mesh.vertices.min(axis=0).astype(np.float32),
        "scene_extent_xy": (mesh.vertices[:, :2].max(axis=0) - mesh.vertices[:, :2].min(axis=0)).astype(np.float32),
        "num_pieces": np.asarray([num_pieces], dtype=np.int32),
    }
    return metadata


def build_sqs_params(piece_specs: list[PieceSpec], world_rotation: np.ndarray | None) -> dict[str, np.ndarray]:
    axis_basis = []
    for spec in piece_specs:
        axis_u, axis_v = build_tangent_basis(spec.piece_axis)
        axis_basis.append(np.stack([axis_u, axis_v, spec.piece_axis], axis=0))
    rot_w2p = np.stack(axis_basis, axis=0).astype(np.float32)
    rot_p2w = np.transpose(rot_w2p, (0, 2, 1)).astype(np.float32)
    payload: dict[str, np.ndarray] = {
        "piece_ids": np.asarray([spec.piece_id for spec in piece_specs], dtype=np.int32),
        "walkable": np.asarray([spec.walkable for spec in piece_specs], dtype=np.bool_),
        "piece_name_utf8": np.asarray([spec.piece_name for spec in piece_specs], dtype=f"<U{max(len(spec.piece_name) for spec in piece_specs)}"),
        "piece_centers_w": np.stack([spec.solid_center for spec in piece_specs], axis=0).astype(np.float32),
        "piece_rot_w2p": rot_w2p,
        "piece_rot_p2w": rot_p2w,
        "piece_half_axes": np.stack([spec.solid_half_size for spec in piece_specs], axis=0).astype(np.float32),
        "piece_axis_world": np.stack([spec.piece_axis for spec in piece_specs], axis=0).astype(np.float32),
        "support_normal_world": np.stack([spec.normal for spec in piece_specs], axis=0).astype(np.float32),
        "support_half_axes": np.stack([spec.half_size for spec in piece_specs], axis=0).astype(np.float32),
    }
    if world_rotation is not None:
        payload["world_rotation"] = world_rotation.astype(np.float32, copy=False)
    return payload


def merge_sqs_export_payload(
    source_payload: dict[str, np.ndarray] | None,
    piece_specs: list[PieceSpec],
    world_rotation: np.ndarray | None,
) -> dict[str, np.ndarray]:
    if source_payload is None:
        return build_sqs_params(piece_specs, world_rotation)

    payload = {key: np.asarray(value) for key, value in source_payload.items()}
    payload["piece_ids"] = np.asarray([spec.piece_id for spec in piece_specs], dtype=np.int32)
    payload["walkable"] = np.asarray([spec.walkable for spec in piece_specs], dtype=np.bool_)
    payload["piece_name_utf8"] = np.asarray(
        [spec.piece_name for spec in piece_specs],
        dtype=f"<U{max(len(spec.piece_name) for spec in piece_specs)}",
    )
    payload["piece_axis_world"] = np.stack([spec.piece_axis for spec in piece_specs], axis=0).astype(np.float32)
    payload["support_normal_world"] = np.stack([spec.normal for spec in piece_specs], axis=0).astype(np.float32)
    payload["support_half_axes"] = np.stack([spec.half_size for spec in piece_specs], axis=0).astype(np.float32)
    if world_rotation is not None and "world_rotation" not in payload:
        payload["world_rotation"] = world_rotation.astype(np.float32, copy=False)
    return payload


def get_colormap(cmap_name: str):
    if cmap_name == "red_gray":
        return LinearSegmentedColormap.from_list(
            "red_gray",
            [
                (0.0, (0.45, 0.45, 0.45)),
                (1.0e-6, (0.45, 0.45, 0.45)),
                (0.04, (1.0, 0.88, 0.25)),
                (0.35, (0.96, 0.45, 0.12)),
                (0.8, (0.82, 0.05, 0.05)),
                (1.0, (0.75, 0.0, 0.0)),
            ],
        )
    try:
        return plt.get_cmap(cmap_name)
    except ValueError:
        return plt.get_cmap("viridis")


def heat_to_rgb(heat: np.ndarray, cmap_name: str) -> np.ndarray:
    cmap = get_colormap(cmap_name)
    rgba = cmap(np.clip(heat, 0.0, 1.0))
    rgb = np.round(rgba[:, :3] * 255.0).astype(np.uint8)
    return rgb


def write_colored_ply(path: Path, mesh: Mesh, vertex_rgb: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {mesh.vertices.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {mesh.faces.shape[0]}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v, color in zip(mesh.vertices, vertex_rgb):
            f.write(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f} {int(color[0])} {int(color[1])} {int(color[2])}\n")
        for face in mesh.faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def set_axes_equal(ax: Any, vertices: np.ndarray) -> None:
    min_xyz = vertices.min(axis=0)
    max_xyz = vertices.max(axis=0)
    center = (min_xyz + max_xyz) * 0.5
    radius = float(np.max(max_xyz - min_xyz) * 0.55)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def render_preview(mesh: Mesh, face_heat: np.ndarray, out_path: Path, cmap_name: str) -> None:
    tri_vertices = mesh.vertices[mesh.faces]
    face_rgb = heat_to_rgb(face_heat, cmap_name).astype(np.float32) / 255.0

    fig = plt.figure(figsize=(16, 5), constrained_layout=True)
    views = [
        ("Iso", 24, -55),
        ("Top", 88, -90),
        ("Side", 12, 10),
    ]
    for plot_idx, (title, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, plot_idx, projection="3d")
        collection = Poly3DCollection(tri_vertices, linewidths=0.0)
        collection.set_facecolor(face_rgb)
        collection.set_edgecolor(face_rgb)
        ax.add_collection3d(collection)
        set_axes_equal(ax, mesh.vertices)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(title)

    sm = plt.cm.ScalarMappable(cmap=get_colormap(cmap_name))
    sm.set_array(np.linspace(0.0, 1.0, num=100, dtype=np.float32))
    fig.colorbar(sm, ax=fig.axes, shrink=0.6, pad=0.02, label="landing_heat")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def piece_spec_to_jsonable(spec: PieceSpec) -> dict[str, Any]:
    return {
        "piece_id": int(spec.piece_id),
        "piece_name": spec.piece_name,
        "walkable": bool(spec.walkable),
        "center": [float(x) for x in spec.center.tolist()],
        "solid_center": [float(x) for x in spec.solid_center.tolist()],
        "normal": [float(x) for x in spec.normal.tolist()],
        "piece_axis": [float(x) for x in spec.piece_axis.tolist()],
        "tangent_u": [float(x) for x in spec.tangent_u.tolist()],
        "tangent_v": [float(x) for x in spec.tangent_v.tolist()],
        "half_size": [float(x) for x in spec.half_size.tolist()],
        "core_half_size": [float(x) for x in spec.core_half_size.tolist()],
        "solid_half_size": [float(x) for x in spec.solid_half_size.tolist()],
        "top_area": float(spec.top_area),
        "aabb_min": [float(x) for x in spec.aabb_min.tolist()],
        "aabb_max": [float(x) for x in spec.aabb_max.tolist()],
        "source_num_vertices": int(spec.source_num_vertices),
        "source_num_faces": int(spec.source_num_faces),
        "support_face_count": int(spec.support_face_count),
        "support_vertex_count": int(spec.support_vertex_count),
        "support_up_dot": float(spec.support_up_dot),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    world_rotation = None
    if args.world_rotation_npy is not None and args.world_rotation_npy.exists():
        world_rotation = np.load(args.world_rotation_npy).astype(np.float32)
    sqs_payload = load_sqs_payload(args.sqs_params)

    piece_paths = sorted(args.pieces_dir.glob("part_*.obj"))
    if not piece_paths:
        raise FileNotFoundError(f"No part_*.obj files found in {args.pieces_dir}")

    piece_meshes = [load_obj_mesh(path) for path in piece_paths]
    concat_mesh = concat_meshes(piece_meshes)

    if args.merged_obj is not None:
        eval_mesh = load_obj_mesh(args.merged_obj)
        alignment = mesh_alignment_summary(concat_mesh, eval_mesh)
    else:
        eval_mesh = concat_mesh
        alignment = mesh_alignment_summary(concat_mesh, eval_mesh)

    sqs_params_rows = None
    if sqs_payload is not None and "params" in sqs_payload:
        sqs_params_rows = np.asarray(sqs_payload["params"], dtype=np.float32)
        if sqs_params_rows.ndim != 2 or sqs_params_rows.shape[1] < 11:
            raise ValueError(f"Expected SQ params with shape (N, 11+), got {sqs_params_rows.shape}")
        if sqs_params_rows.shape[0] != len(piece_paths):
            print(
                f"[WARN] SQ param count {sqs_params_rows.shape[0]} does not match piece count {len(piece_paths)}; "
                "falling back to mesh-estimated axes."
            )
            sqs_params_rows = None

    piece_specs = [
        infer_piece_spec(
            mesh,
            idx,
            path.name,
            args,
            sqs_row=sqs_params_rows[idx] if sqs_params_rows is not None else None,
        )
        for idx, (mesh, path) in enumerate(zip(piece_meshes, piece_paths))
    ]
    if args.piece_params_json is not None:
        overrides = load_piece_overrides(args.piece_params_json)
        piece_specs = [
            overrides.get(spec.piece_name, spec)
            if overrides.get(spec.piece_name, spec).piece_id >= 0
            else PieceSpec(
                piece_id=spec.piece_id,
                piece_name=overrides[spec.piece_name].piece_name,
                walkable=overrides[spec.piece_name].walkable,
                center=overrides[spec.piece_name].center,
                solid_center=overrides[spec.piece_name].solid_center,
                normal=overrides[spec.piece_name].normal,
                piece_axis=overrides[spec.piece_name].piece_axis,
                tangent_u=overrides[spec.piece_name].tangent_u,
                tangent_v=overrides[spec.piece_name].tangent_v,
                half_size=overrides[spec.piece_name].half_size,
                core_half_size=overrides[spec.piece_name].core_half_size,
                solid_half_size=overrides[spec.piece_name].solid_half_size,
                top_area=overrides[spec.piece_name].top_area,
                aabb_min=overrides[spec.piece_name].aabb_min,
                aabb_max=overrides[spec.piece_name].aabb_max,
                source_num_vertices=overrides[spec.piece_name].source_num_vertices,
                source_num_faces=overrides[spec.piece_name].source_num_faces,
            )
            for spec in piece_specs
        ]

    heat_f, best_piece_f, heat_v, best_piece_v = evaluate_surface_heat(eval_mesh, piece_specs, args)
    direct_support_face_count = int(np.count_nonzero(heat_f > 1.0e-8))
    direct_face_alignment_used = bool(alignment["topology_matches_exactly"])

    out_obj = args.output_dir / "merged_mesh.obj"
    out_npz = args.output_dir / "heatmap_aligned.npz"
    out_heat_v = args.output_dir / "heat_vertices.npy"
    out_heat_f = args.output_dir / "heat_faces.npy"
    out_best_piece_v = args.output_dir / "best_piece_vertices.npy"
    out_best_piece_f = args.output_dir / "best_piece_faces.npy"
    out_ply = args.output_dir / "merged_mesh_heat.ply"
    out_png = args.output_dir / "heat_preview.png"
    out_meta = args.output_dir / "metadata.json"
    out_obj_metadata = args.output_dir / "obj_metadata.npz"
    out_atlas = args.output_dir / "atlas.npz"
    out_sqs_params = args.output_dir / "sqs_params.npz"
    out_world_rotation_npy = args.output_dir / "world_rotation.npy"
    out_world_rotation_txt = args.output_dir / "world_rotation.txt"

    write_obj_mesh(out_obj, eval_mesh)
    np.save(out_heat_v, heat_v)
    np.save(out_heat_f, heat_f)
    np.save(out_best_piece_v, best_piece_v)
    np.save(out_best_piece_f, best_piece_f)
    piece_metadata = build_piece_metadata(piece_specs, eval_mesh)
    sqs_params = merge_sqs_export_payload(sqs_payload, piece_specs, world_rotation)
    atlas = bake_atlas(eval_mesh, piece_specs, args)
    np.savez_compressed(out_obj_metadata, **piece_metadata)
    np.savez_compressed(out_atlas, **atlas)
    np.savez_compressed(out_sqs_params, **sqs_params)
    if world_rotation is not None:
        np.save(out_world_rotation_npy, world_rotation)
        np.savetxt(out_world_rotation_txt, world_rotation, fmt="%.8f")
    np.savez_compressed(
        out_npz,
        vertices=eval_mesh.vertices,
        faces=eval_mesh.faces,
        heat_v=heat_v,
        heat_f=heat_f,
        best_piece_v=best_piece_v,
        best_piece_f=best_piece_f,
    )

    vertex_rgb = heat_to_rgb(heat_v, args.cmap)
    write_colored_ply(out_ply, eval_mesh, vertex_rgb)
    render_preview(eval_mesh, heat_f, out_png, args.cmap)

    metadata = {
        "schema_version": 1,
        "source_piece_dir": str(args.pieces_dir.resolve()),
        "source_merged_obj": str(args.merged_obj.resolve()) if args.merged_obj is not None else None,
        "generated_merged_obj": str(out_obj.resolve()),
        "generated_heat_vertices": str(out_heat_v.resolve()),
        "generated_heat_faces": str(out_heat_f.resolve()),
        "generated_bundle_npz": str(out_npz.resolve()),
        "generated_obj_metadata_npz": str(out_obj_metadata.resolve()),
        "generated_atlas_npz": str(out_atlas.resolve()),
        "generated_sqs_params_npz": str(out_sqs_params.resolve()),
        "generated_colored_ply": str(out_ply.resolve()),
        "generated_preview_png": str(out_png.resolve()),
        "source_sqs_params": str(args.sqs_params.resolve()) if args.sqs_params is not None and args.sqs_params.exists() else None,
        "source_world_rotation_npy": str(args.world_rotation_npy.resolve()) if args.world_rotation_npy is not None else None,
        "generated_world_rotation_npy": str(out_world_rotation_npy.resolve()) if world_rotation is not None else None,
        "heat_definition": {
            "name": "landing_heat",
            "range": [0.0, 1.0],
            "meaning": "Higher means closer to a selected support-cap face center after geo-calibrated piece-normal filtering.",
            "aligned_with": "provided merged OBJ if source_merged_obj is set, otherwise deterministic piece concatenation order",
        },
        "reward_block_defaults": {
            "slab_thickness": float(args.slab_thickness),
            "piece_normal_thresh": float(args.piece_normal_thresh),
            "world_z_veto_deg": float(args.world_z_veto_deg),
            "world_up_max_angle_deg": float(args.world_up_max_angle_deg),
            "top_band": float(args.top_band),
            "min_top_area": float(args.min_top_area),
            "inset_scale": float(args.inset_scale),
            "heat_shape": str(args.heat_shape),
            "gaussian_scale_xy": float(args.gaussian_scale_xy),
            "gaussian_scale_z": float(args.gaussian_scale_z),
            "flat_ratio_xy": float(args.flat_ratio_xy),
            "flat_ratio_z": float(args.flat_ratio_z),
            "superellipse_power": float(args.superellipse_power),
            "edge_drop_power": float(args.edge_drop_power),
            "core_ratio_xy": float(args.core_ratio_xy),
            "core_ratio_z": float(args.core_ratio_z),
            "solid_padding_xy": float(args.solid_padding_xy),
            "solid_padding_z": float(args.solid_padding_z),
            "atlas_resolution": int(args.atlas_resolution),
            "atlas_padding": float(args.atlas_padding),
        },
        "merged_alignment": {
            **alignment,
            "direct_face_alignment_used": bool(direct_face_alignment_used),
        },
        "piece_specs": [piece_spec_to_jsonable(spec) for spec in piece_specs],
        "training_assets": {
            "obj_metadata_keys": sorted(piece_metadata.keys()),
            "atlas_keys": sorted(atlas.keys()),
            "sqs_params_keys": sorted(sqs_params.keys()),
        },
        "summary": {
            "num_pieces": len(piece_specs),
            "num_walkable_pieces": int(sum(spec.walkable for spec in piece_specs)),
            "used_sqs_axis": bool(sqs_params_rows is not None),
            "num_vertices": int(eval_mesh.vertices.shape[0]),
            "num_faces": int(eval_mesh.faces.shape[0]),
            "num_support_faces": int(direct_support_face_count),
            "heat_v_min": float(heat_v.min()) if heat_v.size else 0.0,
            "heat_v_max": float(heat_v.max()) if heat_v.size else 0.0,
            "heat_f_min": float(heat_f.min()) if heat_f.size else 0.0,
            "heat_f_max": float(heat_f.max()) if heat_f.size else 0.0,
            "atlas_valid_ratio": float(atlas["valid"].mean()) if atlas["valid"].size else 0.0,
        },
    }
    out_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata["summary"], indent=2))
    print(json.dumps({"merged_alignment": alignment}, indent=2))
    print(json.dumps({"outputs": {
        "merged_mesh_obj": str(out_obj),
        "heat_vertices_npy": str(out_heat_v),
        "heat_faces_npy": str(out_heat_f),
        "obj_metadata_npz": str(out_obj_metadata),
        "atlas_npz": str(out_atlas),
        "sqs_params_npz": str(out_sqs_params),
        "metadata_json": str(out_meta),
        "colored_ply": str(out_ply),
        "preview_png": str(out_png),
    }}, indent=2))


if __name__ == "__main__":
    main()
