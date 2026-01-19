#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rotate scene + HMR using gravity alignment, then compute ONE shared z-translation (shape [1,3])
so that the rotated scene min z becomes 0, and apply it to BOTH scene and HMR.

ADDED (per your request):
- Only the *penetration-avoidance optimization* (optimize transl_delta) is integrated.
- It does NOT use any extra "T" logic; it uses YOUR existing R_align + shared_translation output.
- It does NOT dump hmr_after_opt; we still dump ONLY "hmr_after_shared" (but it includes the optimized translation if enabled).

Run:
  python read_ours.py --seq-names far_robot

Optional:
  --no-optimize-penetration   Disable penetration optimization
  --opt-...                   Tune optimization hyperparameters
"""
from geocalib import GeoCalib
import argparse
import copy
import shutil
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from pytorch3d import transforms
from sklearn.neighbors import KDTree

warnings.filterwarnings("ignore")

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = (THIS_DIR / ".." / "..").resolve()
DEFAULT_INPUT_ROOT = REPO_ROOT / "results/output/scene"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results/output/post_scene"
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
for p in (THIS_DIR, THIS_DIR.parent, REPO_ROOT):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from smpl import SMPL  # noqa: E402


# -----------------------------------------------------------------------------
# Camera / alignment helpers (ROTATION ONLY; NO camera translation returned)
# -----------------------------------------------------------------------------
def _find_first_image(scene_name: str, data_root: Path) -> Optional[np.ndarray]:
    """Find and load the first image for a scene under a *_img split."""
    if not data_root.exists():
        return None
    for split_dir in data_root.iterdir():
        if not split_dir.is_dir() or not split_dir.name.endswith("_img"):
            continue
        candidate_dir = split_dir / scene_name
        if not candidate_dir.is_dir():
            continue
        for ext in ("*.jpg", "*.png", "*.jpeg"):
            images = sorted(candidate_dir.glob(ext))
            if not images:
                continue
            img_path = images[0]
            try:
                from PIL import Image

                return np.array(Image.open(img_path).convert("RGB"))
            except Exception:
                try:
                    import imageio.v2 as imageio

                    return imageio.imread(img_path)
                except Exception:
                    return None
    return None


def _load_frame_for_rotation(
    scene_name: str,
    hmr_type: str,
    repo_root: Path,
    camera_npz: Optional[Path],
    data_root: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one RGB frame and matching cam2world for calibration.
    Prefer default MegaSAM export to match visualization scripts.
    """
    scene_npz = repo_root / "results/output/scene" / f"{scene_name}_{hmr_type}_sgd_cvd_hr.npz"

    candidates: List[Path] = []
    if camera_npz is not None and camera_npz != scene_npz:
        candidates.append(camera_npz)
    candidates.append(scene_npz)

    for cand in candidates:
        if not cand.exists():
            continue
        if cand.suffix == ".npz":
            data = np.load(cand, allow_pickle=True)
            images = data.get("images")
            cams = data.get("cam_c2w")
            if images is None or cams is None:
                continue
            idx = 0
            if "valid_frame_indices" in data and len(data["valid_frame_indices"]) > 0:
                idx = int(np.array(data["valid_frame_indices"]).ravel()[0])
            return images[idx], cams[idx]
        if cand.suffix == ".npy":
            camera_payload = np.load(cand, allow_pickle=True).item()
            cams = camera_payload.get("cam_c2w")
            if cams is None:
                continue
            image = _find_first_image(scene_name, data_root)
            if image is None:
                raise FileNotFoundError(f"Could not find an image for {scene_name} under {data_root}.")
            return image, cams[0]

    searched = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not locate camera/image data for {scene_name}. "
        f"Expected MegaSAM output at {scene_npz} (or pass --camera-npz). Tried:\n{searched}"
    )


def get_calibration_roll_pitch(image: np.ndarray, device: str) -> Tuple[float, float]:
    """Get roll and pitch calibration from an image using GeoCalib."""
    from geocalib.utils import print_calibration
    

    model = GeoCalib().to(device)
    input_image = torch.tensor(image, dtype=torch.float32).to(device).permute(2, 0, 1)
    result = model.calibrate(input_image)

    gravity = result["gravity"]
    roll_rad, pitch_rad = gravity.rp.unbind(-1)
    roll_rad = float(roll_rad.item())
    pitch_rad = float(pitch_rad.item())
    print_calibration(result)
    return roll_rad, pitch_rad


def _is_likely_w2c(T: np.ndarray) -> bool:
    """Best-effort: decide whether T is world->cam (w2c) rather than cam->world (c2w)."""
    if T.shape != (4, 4):
        return False
    R = T[:3, :3]
    t = T[:3, 3]
    c0 = np.linalg.norm(t)
    c1 = np.linalg.norm(-R.T @ t)
    return (c1 + 1e-6) < 0.5 * (c0 + 1e-6)


def ensure_cam_c2w(cam: np.ndarray) -> np.ndarray:
    """Return cam2world; if input seems to be world2cam, invert it."""
    cam = cam.astype(np.float32)
    if cam.shape == (3, 4):
        cam4 = np.eye(4, dtype=np.float32)
        cam4[:3, :4] = cam
        cam = cam4
    if cam.shape != (4, 4):
        raise ValueError(f"Expected (4,4) or (3,4) camera matrix, got {cam.shape}")
    if _is_likely_w2c(cam):
        return np.linalg.inv(cam).astype(np.float32)
    return cam


def get_world_alignment_from_image(
    rgb: np.ndarray,
    cam_raw: np.ndarray,
    device: str,
    is_megasam: bool = True,
) -> np.ndarray:
    """
    Returns only:
      world_rotation: (3,3) alignment rotation

    IMPORTANT: NO camera translation is returned/used.
    """
    img = rgb
    if is_megasam:
        img = img.astype(np.float32) / 255.0

    roll, pitch = get_calibration_roll_pitch(img, device)

    pitch_rotm = np.array(
        [[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]],
        dtype=np.float32,
    )
    roll_rotm = np.array(
        [[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]],
        dtype=np.float32,
    )

    cam_c2w = ensure_cam_c2w(cam_raw)
    camR = cam_c2w[:3, :3].astype(np.float32)

    world_rotation = pitch_rotm @ roll_rotm @ camR

    # y-up -> z-up
    yup_to_zup = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
    world_rotation = yup_to_zup @ world_rotation

    return world_rotation.astype(np.float32)


def compute_world_alignment(
    scene_name: str,
    hmr_type: str,
    repo_root: Path,
    camera_npz: Optional[Path],
    data_root: Path,
    is_megasam: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      T_align: (4,4) with rotation filled (translation filled later by shared ground shift)
      world_rotation: (3,3)
    """
    rgb_img, cam_raw = _load_frame_for_rotation(scene_name, hmr_type, repo_root, camera_npz, data_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    world_rotation = get_world_alignment_from_image(rgb_img, cam_raw, device, is_megasam)

    T_align = np.eye(4, dtype=np.float32)
    T_align[:3, :3] = world_rotation
    return T_align, world_rotation


# -----------------------------------------------------------------------------
# Scene geometry (ONE PASS, trimesh)
# -----------------------------------------------------------------------------
def _load_trimesh_obj(path: Path) -> trimesh.Trimesh:
    """Load .obj once. If it is a Scene, concatenate geometries."""
    m = trimesh.load(path, process=False)
    if isinstance(m, trimesh.Scene):
        if len(m.geometry) == 0:
            return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64), process=False)
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    assert isinstance(m, trimesh.Trimesh)
    return m


def _rotate_vertices_rowvec(V: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Row-vector convention: v' = v @ R^T"""
    return V @ R.T


def rotate_scene_geometry_once_and_ground_trimesh(
    input_root: Path,
    output_root: Path,
    R_align: np.ndarray,  # (3,3)
) -> Tuple[Optional[Path], np.ndarray, Optional[trimesh.Trimesh]]:
    """
    STRICT: rotate in-memory, compute min_z on rotated verts, then translate in-memory, then export.
    Each mesh file is read once in its own processing.

    Returns:
      scene_mesh_written (prefer sqs mesh path if exists)
      shared_translation (3,)
      scene_mesh_for_sdf (trimesh.Trimesh)  # rotated+grounded final mesh (for SDF), if available
    """
    output_root.mkdir(parents=True, exist_ok=True)

    main: List[Tuple[str, trimesh.Trimesh]] = []
    min_z: Optional[float] = None

    scene_mesh_for_sdf: Optional[trimesh.Trimesh] = None

    sqs_src = input_root / "scene_mesh_sqs" / "scene_mesh_sqs.obj"
    if sqs_src.exists():
        m = _load_trimesh_obj(sqs_src)
        if m.vertices.size > 0:
            V = _rotate_vertices_rowvec(m.vertices.view(np.ndarray), R_align)
            m.vertices = V
            zmin = float(V[:, 2].min())
            min_z = zmin if min_z is None else min(min_z, zmin)
        main.append(("sqs", m))

    if min_z is None:
        print("[WARN] No scene mesh found; using zero translation.")
        shared_translation = np.zeros(3, dtype=np.float32)
    else:
        shared_translation = np.array([0.0, 0.0, -min_z], dtype=np.float32)
        print(f"[OK] Shared scene/human translation (z-shift): {shared_translation}")

    scene_mesh_written = None

    for tag, m in main:
        if m.vertices.size > 0:
            m.vertices = m.vertices + shared_translation[None, :]
        if tag == "sqs":
            # keep a copy for SDF (already rotated+grounded in-memory)
            scene_mesh_for_sdf = m.copy()

            dst_dir = output_root / "scene_mesh_sqs"
            dst_dir.mkdir(parents=True, exist_ok=True)
            out_path = dst_dir / "scene_mesh_sqs.obj"
            m.export(out_path)
            scene_mesh_written = out_path

            urdf_src = input_root / "scene_mesh_sqs" / "scene_mesh_sqs.urdf"
            if urdf_src.exists():
                shutil.copy2(urdf_src, dst_dir / urdf_src.name)

    # Pieces
    pieces_src = input_root / "scene_mesh_sqs" / "pieces"
    if pieces_src.exists():
        pieces_dst = output_root / "scene_mesh_sqs" / "pieces"
        pieces_dst.mkdir(parents=True, exist_ok=True)
        for mesh_path in sorted(pieces_src.glob("*.obj")):
            pm = _load_trimesh_obj(mesh_path)
            if pm.vertices.size > 0:
                Vp = _rotate_vertices_rowvec(pm.vertices.view(np.ndarray), R_align)
                pm.vertices = Vp + shared_translation[None, :]
            pm.export(pieces_dst / mesh_path.name)

    return scene_mesh_written, shared_translation, scene_mesh_for_sdf


# -----------------------------------------------------------------------------
# Penetration optimization (ONLY the optimize part you asked for)
# -----------------------------------------------------------------------------
class SurfacePointCloud:
    """
    Approximate SDF from a triangle mesh by sampling surface points + normals
    and using KDTree nearest neighbors with a normal-based inside vote.

    Sign convention:
      + outside, - inside
    """

    def __init__(self, points: np.ndarray, normals: np.ndarray):
        assert points.ndim == 2 and points.shape[1] == 3
        assert normals.ndim == 2 and normals.shape[1] == 3
        self.points = points.astype(np.float32, copy=False)
        self.normals = normals.astype(np.float32, copy=False)
        self.kd_tree = KDTree(self.points)

    def get_sdf(self, query_points: np.ndarray, sample_count: int = 11, return_gradients: bool = False):
        # query_points: (M,3)
        distances, indices = self.kd_tree.query(query_points, k=sample_count)
        distances = distances.astype(np.float32, copy=False)

        closest_points = self.points[indices]  # (M,k,3)
        closest_normals = self.normals[indices]  # (M,k,3)

        direction = query_points[:, None, :] - closest_points  # (M,k,3)
        inside_votes = np.einsum("mki,mki->mk", direction, closest_normals) < 0
        inside = (inside_votes.sum(axis=1) > (sample_count * 0.5))

        sdf = distances[:, 0].copy()
        sdf[inside] *= -1.0

        if not return_gradients:
            return sdf

        # gradient: default toward nearest surface point; near surface use normal for stability
        grad = direction[:, 0, :].copy()  # (M,3)
        grad[inside] *= -1.0

        near_surface = np.abs(sdf) < 0.0075
        grad[near_surface] = closest_normals[near_surface, 0, :]

        gn = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
        grad = grad / gn
        return sdf, grad


def build_surface_pointcloud_from_trimesh(
    mesh: trimesh.Trimesh,
    sample_point_count: int = 10000000,
) -> SurfacePointCloud:
    """
    Sample points on mesh surface + face normals to build an approximate SDF query structure.
    """
    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        # empty mesh
        pts = np.zeros((1, 3), dtype=np.float32)
        nrm = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        return SurfacePointCloud(pts, nrm)

    pts, face_idx = trimesh.sample.sample_surface(mesh, sample_point_count)
    face_idx = face_idx.astype(np.int64, copy=False)
    normals = mesh.face_normals[face_idx]
    return SurfacePointCloud(pts, normals)


class DifferentiableSDF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, surface_pc: SurfacePointCloud, points_tensor: torch.Tensor):
        # points_tensor: (M,3) on GPU/CPU; SDF computed on CPU numpy
        pts_np = points_tensor.detach().cpu().numpy()
        sdf, grad = surface_pc.get_sdf(pts_np, return_gradients=True)
        ctx.save_for_backward(torch.from_numpy(grad).to(points_tensor.device, dtype=points_tensor.dtype))
        return torch.from_numpy(sdf).to(points_tensor.device, dtype=points_tensor.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (grad,) = ctx.saved_tensors
        # chain rule: dL/dx = dL/dsdf * dsdf/dx  (approx by grad)
        return None, grad_output.unsqueeze(-1) * grad


def _fps_torch(x: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Simple farthest point sampling on a single point set.
    x: (N,3)
    return indices: (npoint,)
    """
    device = x.device
    N = x.shape[0]
    if npoint >= N:
        return torch.arange(N, device=device, dtype=torch.long)

    centroids = torch.zeros(npoint, dtype=torch.long, device=device)
    distance = torch.ones(N, device=device) * 1e10
    farthest = torch.randint(0, N, (1,), device=device)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = x[farthest, :].view(1, 3)
        dist = torch.sum((x - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def optimize_translation_to_avoid_penetration(
    surface_pc: SurfacePointCloud,
    smpl_model: SMPL,
    poses_rotm: torch.Tensor,          # (T,23,3,3)
    global_orient_rotm: torch.Tensor,  # (T,1,3,3)
    betas: torch.Tensor,               # (T,10) or (1,10)
    transl_orig: torch.Tensor,         # (T,3) (already in aligned/world coords)
    sample_n_verts: int = 256,
    num_iters: int = 200,
    lr: float = 1e-2,
    margin: float = 0.01,
    w_pen: float = 1.0,
    w_reg: float = 0.1,
    w_smooth: float = 1.0,
    z_init: float = 0.0,
    frame_stride: int = 1,
) -> torch.Tensor:
    """
    Optimize per-frame translation delta to reduce penetration into scene.

    Returns:
      transl_delta: (T,3) tensor (same device as transl_orig)
    """
    device = transl_orig.device
    T = transl_orig.shape[0]

    transl_delta = torch.zeros((T, 3), device=device, dtype=transl_orig.dtype, requires_grad=True)
    if z_init != 0.0:
        transl_delta.data[:, 2] = float(z_init)

    optimizer = torch.optim.Adam([transl_delta], lr=float(lr))

    # pre-pick FPS vertex indices from frame 0 for consistency
    with torch.no_grad():
        out0 = smpl_model(
            body_pose=poses_rotm[:1],
            global_orient=global_orient_rotm[:1],
            betas=betas[:1] if betas.ndim == 2 and betas.shape[0] == T else betas,
            transl=transl_orig[:1],
            pose2rot=False,
            default_smpl=True,
        )
        v0 = out0.vertices[0]  # (N,3)
        vidx = _fps_torch(v0, int(sample_n_verts)).detach()

    for _ in range(int(num_iters)):
        optimizer.zero_grad(set_to_none=True)

        # optional frame subsampling for speed (deterministic stride)
        if frame_stride <= 1:
            frame_idx = None
            poses_use = poses_rotm
            go_use = global_orient_rotm
            betas_use = betas
            transl_use = transl_orig + transl_delta
        else:
            idx = torch.arange(0, T, int(frame_stride), device=device)
            frame_idx = idx
            poses_use = poses_rotm[idx]
            go_use = global_orient_rotm[idx]
            if betas.ndim == 2 and betas.shape[0] == T:
                betas_use = betas[idx]
            else:
                betas_use = betas
            transl_use = (transl_orig + transl_delta)[idx]

        smpl_out = smpl_model(
            body_pose=poses_use,
            global_orient=go_use,
            betas=betas_use,
            transl=transl_use,
            pose2rot=False,
            default_smpl=True,
        )
        verts = smpl_out.vertices  # (Tb,N,3)

        verts_sub = verts[:, vidx, :]          # (Tb,K,3)
        verts_flat = verts_sub.reshape(-1, 3)  # (Tb*K,3)

        sdf_vals = DifferentiableSDF.apply(surface_pc, verts_flat)  # (Tb*K,)
        penetration_loss = torch.clamp(margin - sdf_vals, min=0.0).mean()

        reg_loss = (transl_delta ** 2).mean()

        if T <= 1:
            smooth_loss = torch.zeros((), device=device, dtype=transl_orig.dtype)
        else:
            # smoothness always applied on FULL sequence (not subsampled) for stable motion
            smooth_loss = (torch.diff(transl_orig + transl_delta, dim=0) ** 2).mean()

        total = float(w_pen) * penetration_loss + float(w_reg) * reg_loss + float(w_smooth) * smooth_loss
        total.backward()
        optimizer.step()

    return transl_delta.detach()


# -----------------------------------------------------------------------------
# HMR / SMPL helpers
# -----------------------------------------------------------------------------
def process_smpl_data(smpl_data_path: Path):
    payload = np.load(smpl_data_path, allow_pickle=True).item()
    poses = payload["body_pose"].cpu()
    trans = payload["transl"].cpu()
    global_orient = payload["global_orient"].cpu()
    betas = payload["betas"].cpu()
    cams = payload.get("pred_cam")
    return poses, trans, global_orient, betas, cams, payload


def plot_translation_and_velocity(transl: np.ndarray, dt: float = 1 / 30, save_path: Optional[Path] = None):
    T = transl.shape[0]
    time = np.arange(T) * dt
    velocity = np.gradient(transl, dt, axis=0)

    labels = ["x", "y", "z"]
    colors = ["r", "g", "b"]

    fig, axs = plt.subplots(3, 2, figsize=(12, 9))
    for i in range(3):
        axs[i, 0].plot(time, transl[:, i], color=colors[i])
        axs[i, 0].set_ylabel(f"{labels[i]} (m)")
        axs[i, 0].set_title(f"Translation {labels[i]}")
        axs[i, 0].grid(True)

        axs[i, 1].plot(time, velocity[:, i], color=colors[i], linestyle="--")
        axs[i, 1].set_ylabel(f"v{labels[i]} (m/s)")
        axs[i, 1].set_title(f"Velocity v{labels[i]}")
        axs[i, 1].grid(True)

    for ax in axs[-1, :]:
        ax.set_xlabel("Time (s)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def dump_hmr_visualization(
    smpl_model_cpu: SMPL,
    body_pose: torch.Tensor,
    global_orient: torch.Tensor,
    betas: torch.Tensor,
    transl: torch.Tensor,
    save_dir: Path,
    prefix: str,
    stride: int = 10,
):
    """Dump SMPL meshes + translation plot (CPU model)."""
    save_dir.mkdir(parents=True, exist_ok=True)

    body_pose_cpu = body_pose.detach().cpu()
    global_orient_cpu = global_orient.detach().cpu()
    betas_cpu = betas.detach().cpu()
    transl_cpu = transl.detach().cpu()

    with torch.no_grad():
        out = smpl_model_cpu(
            body_pose=body_pose_cpu,
            global_orient=global_orient_cpu,
            betas=betas_cpu,
            transl=transl_cpu,
            pose2rot=False,
            default_smpl=True,
        )

    verts = out.vertices.detach().cpu().numpy()
    for i in range(0, verts.shape[0], max(1, int(stride))):
        trimesh.Trimesh(vertices=verts[i], faces=smpl_model_cpu.faces, process=False).export(
            save_dir / f"{prefix}_{i:04d}.obj"
        )

    plot_translation_and_velocity(transl_cpu.numpy(), save_path=save_dir / f"{prefix}_transl_plot.png")


def transform_saved_joint_npz_like_scene(
    src_npz: Path,
    dst_npz: Path,
    R_align: np.ndarray,  # (3,3)
    t_shared: np.ndarray,  # (3,)
) -> bool:
    """
    Load an NPZ containing `global_joint_positions` and apply the SAME transform as scene:
        x' = x @ R_align.T + t_shared
    Supports shapes:
      - (J, 3)
      - (T, J, 3)
      - (N, 3)  (generic)
    Writes out an NPZ with transformed `global_joint_positions` (and preserves other keys if present).
    Returns True if transformed+saved, False if src missing or key missing.
    """
    if not src_npz.exists():
        return False

    data = np.load(src_npz, allow_pickle=True)
    if "global_joint_positions" not in data:
        return False

    joints = data["global_joint_positions"].astype(np.float32, copy=False)
    R = R_align.astype(np.float32, copy=False)
    t = t_shared.astype(np.float32, copy=False)

    # apply row-vector convention: x' = x @ R.T + t
    if joints.ndim == 2 and joints.shape[-1] == 3:
        joints_T = joints @ R.T + t[None, :]
    elif joints.ndim == 3 and joints.shape[-1] == 3:
        # (T,J,3)
        joints_T = joints @ R.T + t[None, None, :]
    else:
        raise ValueError(f"Unsupported global_joint_positions shape: {joints.shape}")

    # Preserve any other keys in the npz
    out_kwargs = {k: data[k] for k in data.files if k != "global_joint_positions"}
    out_kwargs["global_joint_positions"] = joints_T.astype(np.float32, copy=False)

    dst_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst_npz, **out_kwargs)
    return True


# -----------------------------------------------------------------------------
# Transform correctness (GV-style offset correction)
# -----------------------------------------------------------------------------
def correct_transl_after_rigid_transform_from_original(
    smpl_model_cpu: SMPL,
    body_pose_rotm: torch.Tensor,  # (T, 23, 3, 3)
    betas: torch.Tensor,  # (T,10) or (1,10)
    global_orient_rotm_orig: torch.Tensor,  # (T,1,3,3) ORIGINAL
    transl_orig: torch.Tensor,  # (T,3) ORIGINAL
    R_align: np.ndarray,  # (3,3)
    t_shared: np.ndarray,  # (3,)
    global_orient_rotm_T: torch.Tensor,  # (T,1,3,3) already R-applied
    transl_T: torch.Tensor,  # (T,3) already (R,t)-applied
) -> Tuple[torch.Tensor, float]:
    """
    GV-style correctness check:
      verts_target = R_align * verts_orig + t_shared
      verts_params_T = SMPL(body_pose, betas, global_orient_T, transl_T)
    Then compute offset = mean(verts_target - verts_params_T), add to transl_T.

    Returns:
      transl_T_corrected (T,3)
      mean_alignment_error (float) after correction
    """
    device = transl_T.device
    dtype = transl_T.dtype

    R = torch.from_numpy(R_align).to(device="cpu", dtype=torch.float32)[None, :, :]  # (1,3,3)
    t = torch.from_numpy(t_shared).to(device="cpu", dtype=torch.float32)[None, :]  # (1,3)

    body_pose_cpu = body_pose_rotm.detach().cpu()
    betas_cpu = betas.detach().cpu()
    go_orig_cpu = global_orient_rotm_orig.detach().cpu()
    tr_orig_cpu = transl_orig.detach().cpu()

    go_T_cpu = global_orient_rotm_T.detach().cpu()
    tr_T_cpu = transl_T.detach().cpu()

    with torch.no_grad():
        out_orig = smpl_model_cpu(
            body_pose=body_pose_cpu,
            global_orient=go_orig_cpu,
            betas=betas_cpu,
            transl=tr_orig_cpu,
            pose2rot=False,
            default_smpl=True,
        )
        verts_orig = out_orig.vertices  # (T,N,3)

    verts_target = torch.einsum("bij,bnj->bni", R.expand(verts_orig.shape[0], -1, -1), verts_orig) + t.expand(
        verts_orig.shape[0], -1
    )[:, None, :]

    with torch.no_grad():
        out_T = smpl_model_cpu(
            body_pose=body_pose_cpu,
            global_orient=go_T_cpu,
            betas=betas_cpu,
            transl=tr_T_cpu,
            pose2rot=False,
            default_smpl=True,
        )
        verts_T = out_T.vertices

    offset = (verts_target - verts_T).mean(dim=1)  # (T,3)
    tr_T_corrected = tr_T_cpu + offset

    with torch.no_grad():
        out_corr = smpl_model_cpu(
            body_pose=body_pose_cpu,
            global_orient=go_T_cpu,
            betas=betas_cpu,
            transl=tr_T_corrected,
            pose2rot=False,
            default_smpl=True,
        )
        verts_corr = out_corr.vertices

    err = torch.norm(verts_target - verts_corr, dim=-1).mean().item()
    return tr_T_corrected.to(device=device, dtype=dtype), float(err)


# -----------------------------------------------------------------------------
# Export helpers
# -----------------------------------------------------------------------------
def export_to_targets(
    seq_name: str,
    output_root: Path,
    export_motion_root: Optional[Path],
    export_urdf_root: Optional[Path],
    export_tag: Optional[str],
) -> None:
    motion_src = output_root / f"{seq_name}_ours.npz"
    if export_motion_root and motion_src.exists():
        motion_dst_root = export_motion_root / export_tag if export_tag else export_motion_root
        motion_dst_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(motion_src, motion_dst_root / motion_src.name)

    if export_urdf_root:
        src_sqs = output_root / "scene_mesh_sqs"
        urdf_src = src_sqs / "scene_mesh_sqs.urdf"
        mesh_src = src_sqs / "scene_mesh_sqs.obj"
        pieces_src = src_sqs / "pieces"

        target_dir = export_urdf_root / (export_tag if export_tag else "") / seq_name / "ours"
        target_dir.mkdir(parents=True, exist_ok=True)
        if urdf_src.exists():
            shutil.copy2(urdf_src, target_dir / "ours.urdf")
        if mesh_src.exists():
            shutil.copy2(mesh_src, target_dir / "mesh.obj")
        if pieces_src.exists():
            for item in pieces_src.iterdir():
                dst_item = target_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dst_item, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dst_item)


def discover_sequences(input_root: Path, hmr_type: str) -> List[str]:
    seqs: List[str] = []
    for candidate in sorted(input_root.iterdir()):
        if candidate.is_dir() and (candidate / hmr_type).is_dir():
            seqs.append(candidate.name)
    return seqs


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def process_sequence(
    seq_name: str,
    input_root: Path,
    output_root: Path,
    hmr_type: str,
    camera_npz: Optional[Path],
    data_root: Path,
    is_megasam: bool,
    export_motion_root: Optional[Path],
    export_urdf_root: Optional[Path],
    export_tag: Optional[str],
    debug_stride: int,
    correct_transl: bool,
    optimize_penetration: bool,
    opt_sdf_samples: int,
    opt_verts_samples: int,
    opt_iters: int,
    opt_lr: float,
    opt_margin: float,
    opt_w_pen: float,
    opt_w_reg: float,
    opt_w_smooth: float,
    opt_z_init: float,
    opt_frame_stride: int,
) -> None:
    repo_root = REPO_ROOT
    seq_input_root = input_root / seq_name / hmr_type
    if not seq_input_root.exists():
        print(f"[WARN] Input path missing for {seq_name}: {seq_input_root}")
        return

    seq_output_root = output_root / seq_name / hmr_type
    seq_output_root.mkdir(parents=True, exist_ok=True)
    vis_dir = seq_output_root / "saved_obj"

    if camera_npz is not None:
        print("[WARN] Ignoring --camera-npz; using default *_sgd_cvd_hr camera export.")

    # 1) rotation
    T_align, world_rotation = compute_world_alignment(
        scene_name=seq_name,
        hmr_type=hmr_type,
        repo_root=repo_root,
        camera_npz=None,
        data_root=data_root,
        is_megasam=is_megasam,
    )

    # 2) scene rotate+ground (shared translation)
    _scene_mesh_path, shared_translation, scene_mesh_for_sdf = rotate_scene_geometry_once_and_ground_trimesh(
        input_root=seq_input_root,
        output_root=seq_output_root,
        R_align=world_rotation,
    )

    joints_src = seq_input_root / "hmr" / f"{seq_name}.npz"
    joints_dst = seq_output_root / "hmr" / f"{seq_name}.npz"

    did = transform_saved_joint_npz_like_scene(
        src_npz=joints_src,
        dst_npz=joints_dst,
        R_align=world_rotation,
        t_shared=shared_translation,
    )
    if did:
        print(f"[OK] Transformed joint npz -> {joints_dst}")
    else:
        print(f"[INFO] No joint npz transformed (missing file or key): {joints_src}")

    shared_translation_1x3 = shared_translation[None, :]
    T_align[:3, 3] = shared_translation

    np.save(seq_output_root / "world_rotation.npy", world_rotation.astype(np.float32))
    np.savetxt(seq_output_root / "world_rotation.txt", world_rotation, fmt="%.8f")

    # 3) HMR load
    smpl_data_path = seq_input_root / "hmr" / "hps_track.npy"
    poses, trans, global_orient, betas, _pred_cam, hmr_payload = process_smpl_data(smpl_data_path)

    # unify dtype/device
    device = global_orient.device
    dtype = global_orient.dtype
    poses = poses.to(device=device, dtype=dtype)
    trans = trans.to(device=device, dtype=dtype)
    betas = betas.to(device=device, dtype=dtype)
    global_orient = global_orient.to(device=device, dtype=dtype)

    world_rot_torch = torch.from_numpy(world_rotation).to(device=device, dtype=dtype)
    translation_offset = torch.from_numpy(shared_translation_1x3).to(device=device, dtype=dtype)

    # 4) SINGLE transform for HMR (parameter-space)
    global_orient_rot = torch.einsum("ij,t...jk->t...ik", world_rot_torch, global_orient)
    trans_rot = trans @ world_rot_torch.T
    trans_shared = trans_rot + translation_offset

    # 4.5) Optional GV-style correction so that SMPL(params_T) matches (R,t)*SMPL(params_orig)
    if correct_transl:
        smpl_model_cpu = SMPL().to("cpu")
        trans_corrected, mean_err = correct_transl_after_rigid_transform_from_original(
            smpl_model_cpu=smpl_model_cpu,
            body_pose_rotm=poses,
            betas=betas,
            global_orient_rotm_orig=global_orient,
            transl_orig=trans,
            R_align=world_rotation,
            t_shared=shared_translation,
            global_orient_rotm_T=global_orient_rot,
            transl_T=trans_shared,
        )
        print(f"[ALIGN] mean vertex error after transl correction: {mean_err:.6f}")
        trans_shared = trans_corrected  # corrected transl in world coords

    # 4.6) Penetration optimization (ONLY this part was added)
    if optimize_penetration and (scene_mesh_for_sdf is not None) and (scene_mesh_for_sdf.vertices.size > 0):
        print("[OPT] Building surface pointcloud (approx SDF) ...")
        surface_pc = build_surface_pointcloud_from_trimesh(scene_mesh_for_sdf, sample_point_count=int(opt_sdf_samples))

        # SMPL on GPU/CPU (match transl device)
        smpl_model_opt = SMPL().to(device)

        print("[OPT] Optimizing translation delta to avoid penetration ...")
        transl_delta = optimize_translation_to_avoid_penetration(
            surface_pc=surface_pc,
            smpl_model=smpl_model_opt,
            poses_rotm=poses,
            global_orient_rotm=global_orient_rot,
            betas=betas,
            transl_orig=trans_shared,
            sample_n_verts=int(opt_verts_samples),
            num_iters=int(opt_iters),
            lr=float(opt_lr),
            margin=float(opt_margin),
            w_pen=float(opt_w_pen),
            w_reg=float(opt_w_reg),
            w_smooth=float(opt_w_smooth),
            z_init=float(opt_z_init),
            frame_stride=int(opt_frame_stride),
        )
        trans_hmr_final = trans_shared + transl_delta 
        z_offset = 0.2
        trans_hmr_final[:, 2] += z_offset # <-- apply optimized delta to final transl
        print("[OPT] Done. Applied transl_delta to trans_shared.")
    elif optimize_penetration:
        print("[OPT] Skipped (no scene mesh available for SDF).")

    # 5) save ONLY hmr_after_shared visualization (no hmr_after_opt)
    smpl_model_cpu = SMPL().to("cpu")
    dump_hmr_visualization(
        smpl_model_cpu=smpl_model_cpu,
        body_pose=poses,
        global_orient=global_orient_rot,
        betas=betas,
        transl=trans_hmr_final,
        save_dir=vis_dir,
        prefix="hmr_after_shared",
        stride=debug_stride,
    )

    # 6) save motion
    poses_axis_angle = torch.cat(
        [
            transforms.matrix_to_axis_angle(global_orient_rot.squeeze(1)),
            transforms.matrix_to_axis_angle(poses).reshape(len(poses), -1),
        ],
        dim=-1,
    ).detach().cpu().numpy()

    np.savez(
        seq_output_root / f"{seq_name}_ours.npz",
        trans=trans_hmr_final.detach().cpu().numpy(),
        poses=poses_axis_angle,
        betas=betas.detach().cpu().numpy(),
        T_align=T_align,
        T_align_new=T_align,
        gender="neutral",
        mocap_framerate=30,
        world_rotation=world_rotation,
        shared_translation=shared_translation_1x3.astype(np.float32),
    )

    # 7) write back hps_track (updated)
    if torch.is_tensor(hmr_payload.get("global_orient")):
        hmr_payload["global_orient"] = global_orient_rot.to(hmr_payload["global_orient"].device)
    hmr_payload["transl"] = trans_hmr_final.to(hmr_payload["transl"].device)

    # pred_cam: rotate + add ONLY shared translation (no camera translation)
    # NOTE: we intentionally DO NOT add per-frame optimization delta into pred_cam.
    if isinstance(hmr_payload.get("pred_cam"), list):
        pred_cam_list = hmr_payload["pred_cam"]
        if len(pred_cam_list) >= 1 and torch.is_tensor(pred_cam_list[0]):
            pred_cam_list[0] = torch.einsum(
                "ij,tjk->tik",
                world_rot_torch.to(pred_cam_list[0].device, dtype=pred_cam_list[0].dtype),
                pred_cam_list[0],
            )
        if len(pred_cam_list) >= 2 and torch.is_tensor(pred_cam_list[1]):
            R_dev = world_rot_torch.to(pred_cam_list[1].device, dtype=pred_cam_list[1].dtype)
            pred_cam_list[1] = pred_cam_list[1] @ R_dev.T
            base_shift = translation_offset.to(pred_cam_list[1].device, dtype=pred_cam_list[1].dtype)
            pred_cam_list[1] = pred_cam_list[1] + base_shift
        hmr_payload["pred_cam"] = pred_cam_list

    hmr_dst = seq_output_root / "hmr" / "hps_track.npy"
    hmr_dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(hmr_dst, hmr_payload)

    export_to_targets(seq_name, seq_output_root, export_motion_root, export_urdf_root, export_tag)
    print(f"[OK] Finished {seq_name} -> {seq_output_root}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seq-names",
        nargs="+",
        help="Sequence names to process. If omitted, auto-discovers under input-root.",
    )
    parser.add_argument("--hmr-type", default="gv", help="Name of the HMR subfolder to use.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT, help="Root containing raw scene outputs.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Where to save rotated+shifted outputs.")
    parser.add_argument("--camera-npz", type=Path, default=None, help="Optional explicit camera NPZ (ignored by default pipeline).")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Dataset root used to locate reference images.")
    parser.add_argument("--no-megasam", action="store_true", help="Disable MegaSAM-specific image normalization.")
    parser.add_argument("--debug-stride", type=int, default=10, help="Stride for dumping visualization OBJ meshes.")
    parser.add_argument("--export-motion-root", type=Path, default=None, help="Destination root for motion npz export.")
    parser.add_argument("--export-urdf-root", type=Path, default=None, help="Destination root for URDF/mesh export.")
    parser.add_argument("--export-tag", default=None, help="Optional tag subdirectory when exporting assets.")
    parser.add_argument(
        "--no-correct-transl",
        action="store_true",
        help="Disable GV-style translation correction (verts(params_T) matching (R,t)*verts(params_orig)).",
    )

    # penetration optimization knobs (enabled by default)
    parser.add_argument(
        "--no-optimize-penetration",
        action="store_true",
        help="Disable penetration-avoidance translation optimization.",
    )
    parser.add_argument("--opt-sdf-samples", type=int, default=300_000, help="Surface samples for approximate SDF (KDTree).")
    parser.add_argument("--opt-verts-samples", type=int, default=256, help="Number of body vertices sampled for SDF loss.")
    parser.add_argument("--opt-iters", type=int, default=20, help="Optimization iterations.")
    parser.add_argument("--opt-lr", type=float, default=1e-2, help="Optimizer learning rate.")
    parser.add_argument("--opt-margin", type=float, default=0.01, help="Penetration margin (meters).")
    parser.add_argument("--opt-w-pen", type=float, default=1.0, help="Weight for penetration loss.")
    parser.add_argument("--opt-w-reg", type=float, default=0.1, help="Weight for translation regularization.")
    parser.add_argument("--opt-w-smooth", type=float, default=1.0, help="Weight for temporal smoothness.")
    parser.add_argument("--opt-z-init", type=float, default=0.0, help="Initial z offset for transl_delta.")
    parser.add_argument("--opt-frame-stride", type=int, default=1, help="Use every k-th frame in penetration term (speed).")

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    repo_root = REPO_ROOT

    input_root = args.input_root if args.input_root.is_absolute() else (repo_root / args.input_root)
    output_root = args.output_root if args.output_root.is_absolute() else (repo_root / args.output_root)
    data_root = args.data_root if args.data_root.is_absolute() else (repo_root / args.data_root)
    camera_npz = (
        args.camera_npz
        if args.camera_npz and args.camera_npz.is_absolute()
        else (repo_root / args.camera_npz if args.camera_npz else None)
    )
    export_motion_root = (
        args.export_motion_root
        if args.export_motion_root is None or args.export_motion_root.is_absolute()
        else repo_root / args.export_motion_root
    )
    export_urdf_root = (
        args.export_urdf_root
        if args.export_urdf_root is None or args.export_urdf_root.is_absolute()
        else repo_root / args.export_urdf_root
    )

    seq_names = args.seq_names if args.seq_names else discover_sequences(input_root, args.hmr_type)
    if not seq_names:
        print(f"[WARN] No sequences found under {input_root} for hmr_type={args.hmr_type}")
        return

    is_megasam = not args.no_megasam
    correct_transl = not args.no_correct_transl
    optimize_penetration = not args.no_optimize_penetration

    for seq_name in seq_names:
        process_sequence(
            seq_name=seq_name,
            input_root=input_root,
            output_root=output_root,
            hmr_type=args.hmr_type,
            camera_npz=camera_npz,
            data_root=data_root,
            is_megasam=is_megasam,
            export_motion_root=export_motion_root,
            export_urdf_root=export_urdf_root,
            export_tag=args.export_tag,
            debug_stride=args.debug_stride,
            correct_transl=correct_transl,
            optimize_penetration=optimize_penetration,
            opt_sdf_samples=args.opt_sdf_samples,
            opt_verts_samples=args.opt_verts_samples,
            opt_iters=args.opt_iters,
            opt_lr=args.opt_lr,
            opt_margin=args.opt_margin,
            opt_w_pen=args.opt_w_pen,
            opt_w_reg=args.opt_w_reg,
            opt_w_smooth=args.opt_w_smooth,
            opt_z_init=args.opt_z_init,
            opt_frame_stride=args.opt_frame_stride,
        )


if __name__ == "__main__":
    main()
