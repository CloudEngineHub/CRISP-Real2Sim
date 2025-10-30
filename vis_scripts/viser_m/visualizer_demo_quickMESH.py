import time
import sys
import argparse
from pathlib import Path
import trimesh
import numpy as onp
import tyro
import numpy as np
import pickle
import torch
import smplx

import os
import re
import trimesh
import argparse
import glob
from tqdm.auto import tqdm
# from sqs_utils.superquadric import *
from pytorch3d.transforms import euler_angles_to_matrix
import os, shutil
import copy
import glob
import viser
import viser.extras
import viser.transforms as tf
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans        # fast & GPU‑friendly via torch tensors
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # For colormap
import smplx
from smpl import SMPL, BodyModelSMPLH, BodyModelSMPLX
import torch
import os
import cv2
import numpy as np
import argparse
from scipy.ndimage import distance_transform_edt
import vdbfusion
import torch.nn.functional as F
from mesh_to_sdf import sample_sdf_near_surface
from scipy.ndimage import gaussian_filter, sobel
import json
from read_emdb_utils import save_rotated
from optim_utils import *
import numpy as np, trimesh, xml.etree.ElementTree as ET
from pathlib import Path
import scipy
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import trimesh
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import trimesh
from utils import *
from toy_exp.vis_normal import Vis
from pathlib import Path
import shutil
import numpy as np

import torch
import torch.nn.functional as F
from typing import Dict, Any

from metrics_calculations import *



import os, cv2, torch, numpy as np
from pathlib import Path



def compute_similarity_transform_vertices_first(target_verts, source_verts):
    """
    用第一帧的顶点做相似变换 (scale s, rotation R, translation t) 的 Procrustes 对齐。
    target_verts: (T,V,3) 或 (V,3)；GT 顶点（目标系）
    source_verts: (T,V,3) 或 (V,3)；预测顶点（待对齐）
    返回: s (0-dim torch.Tensor), R (3,3 torch.Tensor), t (3,) torch.Tensor
    """
    def to_cpu_f32_np_or_t(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float().cpu()
        elif isinstance(x, torch.Tensor):
            return x.detach().float().cpu()
        else:
            raise TypeError(f"Unsupported type: {type(x)}")

    A = to_cpu_f32_np_or_t(target_verts)
    B = to_cpu_f32_np_or_t(source_verts)

    if A.dim() == 3:  # (T,V,3) -> 取第一帧
        A = A[0]
    if B.dim() == 3:
        B = B[0]
    assert A.shape == B.shape and A.dim() == 2 and A.size(1) == 3, f"bad shapes: {A.shape} vs {B.shape}"

    # 去中心
    muA = A.mean(dim=0, keepdim=True)
    muB = B.mean(dim=0, keepdim=True)
    Ac = A - muA
    Bc = B - muB

    # SVD 求旋转
    H = Bc.t().mm(Ac)                     # (3,3)
    U, S, Vh = torch.linalg.svd(H)        # H ≈ U @ diag(S) @ Vh
    R = Vh.t().mm(U.t())                  # R = V * U^T
    if torch.det(R) < 0:                  # 处理反射
        Vh[-1, :] *= -1
        R = Vh.t().mm(U.t())

    # 尺度（相似变换）
    varB = (Bc ** 2).sum()
    s = S.sum() / (varB + 1e-12)

    # 平移
    t = (muA.squeeze(0) - s * (R.mm(muB.squeeze(0))))

    return s, R, t


def clip_gt_data(start_end_index, masks, poses_body, poses_root, betas, trans):
    """
    Clip ground truth data based on the available frame range.
    
    Args:
        start_end_index: tuple of (start_idx, end_idx)
        masks: boolean mask array
        poses_body: body poses array
        poses_root: root poses array
        betas: shape parameters array
        trans: translation array
        
    Returns:
        Clipped versions of all input arrays
    """
    start_idx, end_idx = start_end_index
    
    # Ensure we don't go out of bounds
    n_frames = len(masks)
    start_idx = max(0, start_idx)
    end_idx = min(n_frames - 1, end_idx)
    
    # Clip all arrays to the valid range
    # Adding 1 to end_idx because Python slicing is exclusive at the end
    clip_slice = slice(start_idx, end_idx + 1)
    
    masks_clipped = masks[clip_slice]
    poses_body_clipped = poses_body[clip_slice]
    poses_root_clipped = poses_root[clip_slice]
    betas_clipped = betas[clip_slice]
    trans_clipped = trans[clip_slice]
    
    return (masks_clipped, poses_body_clipped, poses_root_clipped, 
            betas_clipped, trans_clipped)

def find_start_end(folder_path):
    """
    Read all images in the folder and extract the smallest and largest frame indices.
    
    Args:
        folder_path: Path to folder containing image files
        
    Returns:
        tuple: (start_index, end_index) representing the range of frame indices
    """
    import re
    
    # Get all image files (common formats)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not image_files:
        raise ValueError(f"No image files found in {folder_path}")
    
    # Extract frame indices from filenames
    indices = []
    for img_path in image_files:
        basename = os.path.basename(img_path)
        # Extract all numbers from the filename
        numbers = re.findall(r'\d+', basename)
        if numbers:
            # Usually the frame index is the first or most prominent number
            frame_idx = int(numbers[-1] if numbers else numbers[0])
            indices.append(frame_idx)
    
    if not indices:
        raise ValueError(f"Could not extract frame indices from files in {folder_path}")
    
    start_idx = min(indices)
    end_idx = max(indices)
    
    return start_idx, end_idx


device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
smpl_template = smplx.create("/data3/zihanwa3/_Robotics/_vision/tram/data/smpl/SMPL_NEUTRAL.pkl", 
                        model_type="smpl", 
                        gender="neutral",
                        num_betas=10, 
                        # batch_size=T, 
                        ext="pkl").to(device)
tt = lambda x: torch.from_numpy(x).float().to(device)
def find_gt_file(tgt_name, gt_data_dir):
    """Find the corresponding ground truth file for a target name"""
    pattern = os.path.join(gt_data_dir, "**", f"*{tgt_name}_data.pkl")
    matches = glob.glob(pattern, recursive=True)
    if matches:
        return matches[0]
    else:
        print(f"Warning: No ground truth file found for {tgt_name}")
        return None

# Add this function after your existing helper functions
def read_gt_EMDB2_simplified(file_path):
    """Simplified GT reading without clipping logic"""
    ann = pickle.load(open(file_path, 'rb'))
    
    masks = ann['good_frames_mask']
    poses_body = ann["smpl"]["poses_body"]
    poses_root = ann["smpl"]["poses_root"]
    betas = np.repeat(ann["smpl"]["betas"].reshape((1, -1)), repeats=ann["n_frames"], axis=0)
    trans = ann["smpl"]["trans"]
    
    # Forward SMPL
    gt = smpl_template(body_pose=tt(poses_body), 
                       global_orient=tt(poses_root), 
                       betas=tt(betas), 
                       transl=tt(trans),
                       pose2rot=True, 
                       default_smpl=True)
    
    gt_vert = gt.vertices.cpu().numpy()
    gt_j3d = gt.joints[:,:24].cpu().numpy()
    
    return gt_vert, gt_j3d, masks

def align_meshes(source_j3d, target_j3d, source_verts, align_type='first'):
    """
    Align source mesh to target using joint positions
    align_type: 'first' for first-frame alignment, 'global' for all-frames alignment
    """
    if align_type == 'first':
        # First frame alignment
        aligned_j3d = first_align_joints(target_j3d, source_j3d, debug=False)
        # Get transformation
        s, R, t = compute_alignment_transform(target_j3d, source_j3d, align_type='first')
    else:  # global
        aligned_j3d = global_align_joints(target_j3d, source_j3d)
        s, R, t = compute_alignment_transform(target_j3d, source_j3d, align_type='global')
    
    # Apply same transform to vertices
    aligned_verts = (source_verts - source_j3d[:, 0:1, :]) @ R.T * s + target_j3d[:, 0:1, :] + t
    
    return aligned_verts, aligned_j3d

def compute_alignment_transform(target_j3d, source_j3d, align_type='first'):
    """Compute alignment transformation parameters"""
    if align_type == 'first':
        # Align using first frame
        target_0 = target_j3d[0]
        source_0 = source_j3d[0]
    else:
        # Use all frames
        target_0 = target_j3d.reshape(-1, 3)
        source_0 = source_j3d.reshape(-1, 3)
    
    # Center
    target_centered = target_0 - target_0.mean(axis=0)
    source_centered = source_0 - source_0.mean(axis=0)
    
    # Compute optimal rotation via SVD
    H = source_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    
    # Scale
    s = np.trace(S) / np.sum(source_centered ** 2)
    
    # Translation
    t = target_0.mean(axis=0) - s * (source_0 @ R.T).mean(axis=0)
    
    return s, R, t

def _sanitize_trimesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    finite = np.isfinite(V).all(axis=1)
    if not finite.all():
        map_idx = -np.ones(len(V), dtype=np.int64)
        map_idx[finite] = np.arange(finite.sum())
        V2 = V[finite]
        F2 = F[np.all(finite[F], axis=1)]
        F2 = map_idx[F2]
        mesh = trimesh.Trimesh(V2, F2, process=False)
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    return mesh

def export_scene_as_urdf(parts, transforms, out_dir):
    """
    parts       : list[trimesh.Trimesh]               原始子网格（局部坐标）
    transforms  : list[np.ndarray, shape (4,4)]       每个子网格到世界的齐次矩阵
    out_dir     : Path | str                          
    """
    out_dir = Path(out_dir)
    scene_type = out_dir.name
    urdf_path = out_dir

    
    # Create fresh output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    out_dir = out_dir / 'pieces'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ─── URDF 根 ──────────────────────────────────────────────────────────────
    robot = ET.Element("robot", name="scene")

    # 只有一个 link，名字随意；这里叫 "scene_link"
    scene_link = ET.SubElement(robot, "link", name="scene_link")

    # ─── 逐个导出网格并在 link 下添加 visual / collision 对 ────────────────
    for i, (mesh_local, T) in enumerate(zip(parts, transforms)):
        name = f"part_{i:03d}"
        mesh_file = out_dir / f"{name}.obj"
        mesh_local.export(mesh_file)

        # 位姿拆分
        xyz = T[:3, 3]
        rpy = to_rpy(T[:3, :3])

        # 可视 & 碰撞；两段完全一样，只是 tag 不同
        for tag in ("visual", "collision"):
            sec = ET.SubElement(scene_link, tag)
            ET.SubElement(
                sec,
                "origin",
                xyz=" ".join(f"{v:.6f}" for v in xyz),
                rpy=" ".join(f"{v:.6f}" for v in rpy),
            )
            geom = ET.SubElement(sec, "geometry")
            ET.SubElement(
                geom,
                "mesh",
                filename=str(mesh_file.relative_to(out_dir)),
            )

    # ─── 写盘 ────────────────────────────────────────────────────────────────
    ET.ElementTree(robot).write(
        urdf_path / f"{scene_type}.urdf",
        encoding="utf-8",
        xml_declaration=True,
    )

def save_custom_mesh(mesh_parts, tgt_folder):
    scene = trimesh.Scene()
    keep_T = []
    for p in mesh_parts:
        scene.add_geometry(p)
    for name, mesh in scene.geometry.items():
        T_world, _ = scene.graph.get(name)     # 4×4
        mesh_world = mesh.copy()
        mesh_world.apply_transform(T_world)    # 变到世界坐标系，做接触测试
        keep_T.append(T_world)

    export_scene_as_urdf(mesh_parts, keep_T, tgt_folder)


def convert_results_to_params_direct(
    results: dict,
    eps1: float = -2.398,     # log(0.1) roundness default
    eps2: float = -2.398,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Convert refined SQ results to the 11-parameter vector
        [eps1, eps2,
         sx,  sy,  sz,
         rz,  ry,  rx,     # ZYX order
         tx,  ty,  tz]     # (N,11) tensor
    """
    S_items = results['S_items']          # log half-axes, torch(3,)
    R_items = results['R_items']          # 3×3 rotations (body→world) **after** re-orthonorm.
    T_items = results['T_items']          # 3-vec centres

    # Optional per-SQ epsilon values (already in log space)
    eps_items = results.get('eps_items', None)

    params = []
    for i, (S_log, R_bw, T) in enumerate(zip(S_items, R_items, T_items)):
        # ----- epsilons -----------------------------------------------
        if eps_items is not None and i < len(eps_items):
            eps1_i, eps2_i = eps_items[i]
        else:
            eps1_i, eps2_i = eps1, eps2          # global default

        # ----- scale ---------------------------------------------------
        sx, sy, sz = torch.exp(S_log).tolist()   # back to linear half-axes

        # ----- rotation  (body→world matrix → Euler ZYX) ---------------
        R_wb = R_bw.T                            # pytorch3d expects world→body on batch dim
        rz, ry, rx = matrix_to_euler_angles(
            R_wb.unsqueeze(0), convention='ZYX'
        ).squeeze(0).tolist()

        # ----- translation --------------------------------------------
        tx, ty, tz = T.tolist()

        params.append([
            eps1_i, eps2_i,
            sx, sy, sz,
            rz, ry, rx,
            tx, ty, tz
        ])

    if not params:
        return torch.zeros((0, 11), device=device)

    return torch.tensor(params, dtype=torch.float32, device=device)

import h5py
import numpy as np
import torch
import smplx

def load_dict_from_hdf5(h5file, path="/"):
    """
    Recursively load a nested dictionary from an HDF5 file.
    """
    result = {}
    for key in h5file[path].keys():
        key_path = f"{path}{key}"
        if isinstance(h5file[key_path], h5py.Group):
            result[key] = load_dict_from_hdf5(h5file, key_path + "/")
        else:
            result[key] = h5file[key_path][:]
    # attributes
    for attr_key, attr_value in h5file.attrs.items():
        if attr_key.startswith(path):
            result[attr_key[len(path):]] = attr_value
    return result

def undo_gravity_calibration(calibrated_keypoints_path, world_scale_factor=1.0):
    """
    Undo the gravity calibration transformation applied to keypoints.
    
    Args:
        calibrated_keypoints_path: Path to the gravity calibrated keypoints h5 file
        world_scale_factor: The scale factor that was applied during calibration
        
    Returns:
        Dictionary with original (uncalibrated) data
    """
    # Load the calibrated data
    with h5py.File(calibrated_keypoints_path, 'r') as f:
        calibrated_data = load_dict_from_hdf5(f)
    
    # Get the world rotation matrix that was applied
    world_rotation = calibrated_data['world_rotation']  # (3, 3)
    
    # Compute the inverse transformation
    # The calibration applied: new_point = (old_point @ world_rotation.T) * world_scale_factor
    # To undo: old_point = (new_point / world_scale_factor) @ world_rotation
    world_rotation_inv = world_rotation.T  # Since world_rotation is orthogonal, inverse = transpose
    
    # Initialize output dictionary
    uncalibrated_data = {}
    
    # Process root_orient (global orientation)
    if 'root_orient' in calibrated_data:
        uncalibrated_data['root_orient'] = {}
        for person_id, calibrated_orient in calibrated_data['root_orient'].items():
            # Calibration applied: new_orient = world_rotation @ old_orient
            # To undo: old_orient = world_rotation.T @ new_orient
            uncalibrated_orient = world_rotation_inv @ calibrated_orient  # (3, 3) @ (T, 1, 3, 3)
            uncalibrated_data['root_orient'][person_id] = uncalibrated_orient.astype(np.float32)
    
    # Process joints
    if 'joints' in calibrated_data:
        uncalibrated_data['joints'] = {}
        for person_id, calibrated_joints in calibrated_data['joints'].items():
            # Calibration applied: new_joints = (old_joints @ world_rotation.T) * world_scale_factor
            # To undo: old_joints = (new_joints / world_scale_factor) @ world_rotation
            uncalibrated_joints = (calibrated_joints / world_scale_factor) @ world_rotation
            uncalibrated_data['joints'][person_id] = uncalibrated_joints.astype(np.float32)
    
    # Keep joint names as is
    uncalibrated_data['joint_names'] = calibrated_data.get('joint_names', [])
    
    return uncalibrated_data, world_rotation

def export_superquadrics(params, *,
                         out_dir="meshes",
                         stem="sq",
                         combine=True,
                         lat_res=64, lon_res=128,
                         filetype="obj"):
    """
    params   : (N,11) array -- one SQ per row
    combine  : True  → single file   (stem_scene.obj/ply)
               False → one file per SQ (stem_00.obj, …)
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    meshes  = []
    for p in params:
        sq = superquadric(p[0:2], p[2:5], p[5:8], p[8:11])
        V, F = sq.get_mesh(lat_res=lat_res, lon_res=lon_res)
        meshes.append(trimesh.Trimesh(vertices=V, faces=F, process=False))

    if combine:
        scene  = trimesh.util.concatenate(meshes)
        return scene 
        fname  = out_dir / f"{stem}_scene.{filetype}"
        scene.export(fname)
        print(f"wrote {fname}  ({len(scene.vertices)} verts, {len(scene.faces)} faces)")


def load_contact_sequence(interact_contact_path, start_frame=0, end_frame=-1):
    """加载一段连续帧的contact数据"""
    contact_sequence = []
    for i in range(start_frame, end_frame + 1):
        contact_data = np.load(os.path.join(interact_contact_path, f'{i:05d}.npz'))['pred_contact_3d_smplh']
        contact_sequence.append(contact_data > 1e-7)  # 转换为布尔mask
    return np.array(contact_sequence)  # shape: [num_frames, 6890]



def clean_and_make(dir_path: str | Path) -> Path:
    """
    Remove *everything* that may already be in `dir_path` and then
    recreate the (now-empty) directory.  Returns the directory as a Path.
    """
    p = Path(dir_path)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    """
    将 3×3 旋转矩阵转 roll-pitch-yaw (XYZ 固定轴) 顺序。
    返回值单位：rad
    """
    # 这里用的是 ROS 的惯用定义：R = Rz(yaw) * Ry(pitch) * Rx(roll)
    sy = np.hypot(R[0, 0], R[1, 0])
    singular = sy < 1e-6
    if not singular:
        roll  = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = np.arctan2(R[1, 0], R[0, 0])
    else:           # pitch ≈ ±90°
        roll  = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = 0.0
    return roll, pitch, yaw


def get_grouped_contact_ids_smpl():
    """Get contact IDs for SMPL model (converted from SMPL-X)"""
    # Load grouped SMPL-X contact IDs
    from smpl_utils import load_contact_ids_with_mode
    CONTACT_IDS_SMPL_LIST = load_contact_ids_with_mode(mode='grouped')
    
    leg_ids, hand_ids, gluteus_ids, back_ids, thigh_ids = CONTACT_IDS_SMPL_LIST
    
    # Load SMPL-X to SMPL mapping
    PROJ_ROOT = '/data3/zihanwa3/_Robotics/_vision/GVHMR/'
    smplx2smpl_map = torch.load(
        f"{PROJ_ROOT}/hmr4d/utils/body_model/smplx2smpl_sparse.pt",
        map_location="cpu"
    )
    
    if smplx2smpl_map.is_sparse:
        smplx2smpl_map = smplx2smpl_map.to_dense()
    
    x2s = torch.argmax(smplx2smpl_map, dim=0).cpu().numpy()
    
    def convert_indices_smplx_to_smpl(smplx_indices, x2s_lookup):
        return np.unique(x2s_lookup[smplx_indices.cpu().numpy()]).tolist()
    
    # Convert to SMPL indices
    contact_ids_dict = {
        'leg': convert_indices_smplx_to_smpl(leg_ids, x2s),
        'hand': convert_indices_smplx_to_smpl(hand_ids, x2s),
        'gluteus': convert_indices_smplx_to_smpl(gluteus_ids, x2s),
        'back': convert_indices_smplx_to_smpl(back_ids, x2s),
        'thigh': convert_indices_smplx_to_smpl(thigh_ids, x2s)
    }
    
    # Also return combined array for compatibility
    all_ids = np.concatenate(list(contact_ids_dict.values()))
    
    return contact_ids_dict, all_ids


def to_cpu_float32(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().cpu()
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu()
    raise TypeError(f"Unsupported type: {type(x)}")

def _ensure_T_match(gt, pred):
    T = min(len(gt), len(pred))
    return gt[:T], pred[:T], T

@torch.no_grad()
def apply_alignment_to_verts(pred_vert, s, R, t):
    """
    pred_vert: (T, V, 3)
    s: () or (1,) or (T,)
    R: (3,3) or (T,3,3)
    t: (3,)  or (T,3)
    returns: (T, V, 3)  (CPU float32)
    """
    def to_cpu_f32(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x.detach().float().cpu() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)

    pred_vert = to_cpu_f32(pred_vert)  # (T,V,3)
    s = to_cpu_f32(s)
    R = to_cpu_f32(R)
    t = to_cpu_f32(t)

    assert pred_vert.dim() == 3 and pred_vert.size(-1) == 3, f"pred_vert shape {pred_vert.shape} must be (T,V,3)"
    T, V, _ = pred_vert.shape

    # ---- batch R 到 (T,3,3)
    if R.dim() == 2:              # (3,3) -> (T,3,3)
        Rb = R.unsqueeze(0).expand(T, -1, -1)
    elif R.dim() == 3 and R.size(0) == T:
        Rb = R
    else:
        raise ValueError(f"Bad R shape {R.shape}; expect (3,3) or (T,3,3) with T={T}")

    # 旋转： (T,V,3) @ (T,3,3)^T -> (T,V,3)
    pred_rot = torch.einsum("tvi,tij->tvj", pred_vert, Rb.transpose(1, 2))

    # ---- 处理 s（标量 / 长度1 / 按帧）
    if s.dim() == 0 or s.numel() == 1:     # 标量或 [1]，用广播
        scale = s.reshape(1, 1, 1)         # (1,1,1) 广播到 (T,V,3)
    elif s.dim() == 1 and s.size(0) == T:  # (T,) -> (T,1,1)
        scale = s.view(T, 1, 1)
    else:
        raise ValueError(f"Bad s shape {s.shape}; expect scalar/[1]/(T,) with T={T}")

    # ---- 处理 t（(3,) / (T,3)）
    if t.dim() == 1 and t.size(0) == 3:        # (3,) -> (1,1,3) 广播
        tb = t.view(1, 1, 3)
    elif t.dim() == 2 and t.size(0) == T and t.size(1) == 3:
        tb = t.view(T, 1, 3)
    else:
        raise ValueError(f"Bad t shape {t.shape}; expect (3,) or (T,3) with T={T}")

    pred_aligned = pred_rot * scale + tb
    return pred_aligned



def axis_angle_to_matrix_(rotvecs):
    """
    Convert axis-angle to rotation matrix.
    Input: rotvecs [T, 21, 3]
    Output: rotmats [T, 21, 3, 3]
    """
    theta = torch.norm(rotvecs, dim=-1, keepdim=True)  # [T, 21, 1]
    axis = rotvecs / (theta + 1e-8)  # Avoid div-by-zero
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]  # [T, 21]

    cos = torch.cos(theta)[..., 0]  # [T, 21]
    sin = torch.sin(theta)[..., 0]
    one_minus_cos = 1 - cos

    rot = torch.zeros(rotvecs.shape[:-1] + (3, 3), device=rotvecs.device)

    rot[..., 0, 0] = cos + x * x * one_minus_cos
    rot[..., 0, 1] = x * y * one_minus_cos - z * sin
    rot[..., 0, 2] = x * z * one_minus_cos + y * sin
    rot[..., 1, 0] = y * x * one_minus_cos + z * sin
    rot[..., 1, 1] = cos + y * y * one_minus_cos
    rot[..., 1, 2] = y * z * one_minus_cos - x * sin
    rot[..., 2, 0] = z * x * one_minus_cos - y * sin
    rot[..., 2, 1] = z * y * one_minus_cos + x * sin
    rot[..., 2, 2] = cos + z * z * one_minus_cos

    return rot



import numpy as np
import torch

def read_gt_EMDB2(file_path, raw_clips_path):
    """
    Read ground truth EMDB2 data and clip it based on available frames.
    
    Args:
        file_path: Path to the ground truth pickle file
        raw_clips_path: Path to the folder containing raw video frames
        
    Returns:
        gt_ori, gt_j3d, gt_vert, valid_frames_mask, clipped_length, start_idx
    """
    ann = pickle.load(open(file_path, 'rb'))
    
    # Get frame range from actual image files
    start_end_index = find_start_end(raw_clips_path)
    start_idx, end_idx = start_end_index
    seq_len = end_idx - start_idx + 1
    
    # Get masks and data
    masks = ann['good_frames_mask']
    gender = ann['gender']
    poses_body = ann["smpl"]["poses_body"]
    poses_root = ann["smpl"]["poses_root"]
    betas = np.repeat(ann["smpl"]["betas"].reshape((1, -1)), repeats=ann["n_frames"], axis=0)
    trans = ann["smpl"]["trans"]
    
    # Clip all data to match available frames
    masks_clipped, poses_body_clipped, poses_root_clipped, betas_clipped, trans_clipped = clip_gt_data(
        start_end_index, masks, poses_body, poses_root, betas, trans
    )
    ## everythign clipped by masks_clipped
    
    # Forward SMPL with clipped data (ALL frames, not just valid ones)
    gt = smpl_template(body_pose=tt(poses_body_clipped), 
                       global_orient=tt(poses_root_clipped), 
                       betas=tt(betas_clipped), 
                       transl=tt(trans_clipped),
                       pose2rot=True, 
                       default_smpl=True)
    
    # Get vertices and joints for ALL clipped frames
    gt_vert = gt.vertices
    gt_j3d = gt.joints[:,:24]
    gt_ori = axis_angle_to_matrix(tt(poses_root_clipped))
    
    # Return clipped mask that corresponds to the clipped GT data
    return gt_ori, gt_j3d, gt_vert, masks_clipped, seq_len, start_idx
def tsdf_to_voxelGrid(vdb_volume: vdbfusion.VDBVolume,
                      trunc_multiple: float = 1.0,
                      cube_res: int = 256):
    """
    Convert ``vdbfusion.VDBVolume`` → (sdf_flat, voxelGrid).
    
    Parameters
    ----------
    trunc_multiple : float
        Truncation distance as multiples of the voxel size (default 2×).
    cube_res : int or None
        • None   → keep native (Nx,Ny,Nz).  
        • int    → resample to that *cubic* resolution (e.g. 100) to
                    replicate the original CSV convention.

    Returns
    -------
    sdf_flat : (N,) np.ndarray   – Fortran‑order flattening (X fastest).
    voxelGrid : dict             – ready for ``_marching_primitives``.
    """
    # ────── 1. Extract TSDF & bounding box ────────────────────────────
    grid              = vdb_volume.tsdf
    min_ijk, max_ijk  = grid.evalActiveVoxelBoundingBox()   # (i,j,k)

    Nx = max_ijk[0] - min_ijk[0] + 1
    Ny = max_ijk[1] - min_ijk[1] + 1
    Nz = max_ijk[2] - min_ijk[2] + 1
    tsdf_np = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    grid.copyToArray(tsdf_np, ijk=min_ijk)

    # ────── 2. World‑space AABB (metres) ─────────────────────────────
    T     = grid.transform
    xmin, ymin, zmin = T.indexToWorld(min_ijk)
    xmax, ymax, zmax = T.indexToWorld(max_ijk)

    # ────── 3. Optional resample to a cubic grid ─────────────────────
    if cube_res is not None:
        # scale factors along each axis
        zoom_xyz = (cube_res / Nx, cube_res / Ny, cube_res / Nz)
        tsdf_np  = scipy.ndimage.zoom(tsdf_np, zoom_xyz, order=1)  # trilinear
        Nx, Ny, Nz = cube_res, cube_res, cube_res

    # ────── 4. Coordinate lin‑spaces & point cloud ───────────────────
    x_lin = np.linspace(xmin, xmax, Nx, dtype=float)
    y_lin = np.linspace(ymin, ymax, Ny, dtype=float)
    z_lin = np.linspace(zmin, zmax, Nz, dtype=float)

    X, Y, Z   = np.meshgrid(x_lin, y_lin, z_lin, indexing='ij')  # (Nx,Ny,Nz)
    pts_flat  = np.stack((X, Y, Z), axis=3).reshape(-1, 3, order='F').T

    # ────── 5. Pack outputs ──────────────────────────────────────────────
    voxel_size = float(vdb_volume.voxel_size)
    truncation = trunc_multiple * voxel_size

    sdf_flat = tsdf_np.flatten(order='F')      # (Nx*Ny*Nz,)

    voxelGrid = {
        'size':        np.array([Nx, Ny, Nz], dtype=int),
        'range':       np.array([xmin, xmax, ymin, ymax, zmin, zmax],
                                dtype=float),
        'x':           x_lin,
        'y':           y_lin,
        'z':           z_lin,
        'points':      pts_flat,
        'interval':    voxel_size,
        'truncation':  truncation,
        'disp_range':  [-np.inf, truncation],
        'visualizeArclength': 0.01 *
                              np.linalg.norm([xmax - xmin,
                                              ymax - ymin,
                                              zmax - zmin]),
    }

    return sdf_flat, voxelGrid





def filter_bg_points(bg_pos: np.ndarray,
                     bg_col: np.ndarray,
                     contact_verts: np.ndarray,
                     dist_thr: float = 1):
    """
    bg_pos        : (N, 3)  background XYZ points   (float32/float64)
    bg_col        : (N, 3)  corresponding RGB       (same length as bg_pos)
    contact_verts : (T, M, 3) or (M, 3)

    Keeps a bg point only if it lies within `dist_thr` (meters) of
    at least one contact vertex.  Returns the filtered bg_pos/bg_col.
    """
    # Flatten contact verts to (K, 3) where K = T*M   (or M if already (M,3))
    contact_flat = contact_verts.reshape(-1, 3)

    # Pair-wise squared distances → (N, K)
    diff2 = ((bg_pos[:, None, :] - contact_flat[None, :, :]) ** 2).sum(axis=-1)

    # For each bg point, take its nearest contact-vertex distance
    min_dist2 = np.min(diff2, axis=1)

    keep_mask = min_dist2 <= dist_thr ** 2
    return bg_pos[keep_mask], bg_col[keep_mask]

import h5py
import numpy as np
import torch
from pathlib import Path

def load_world_rotation_from_h5(calibrated_path):
    """
    Load only the world rotation matrix from the calibrated h5 file.
    """
    with h5py.File(calibrated_path, 'r') as f:
        world_rotation = f['world_rotation'][:]  # (3, 3)
    return world_rotation




# Direct code to add to your existing pipeline
def apply_undo_inline(pred_rotmat, pred_trans, pred_shape, tgt_name, smpl_template):
    """
    Direct inline code to UNDO gravity calibration.
    Everything on CPU.
    """
    # Load world rotation
    calibrated_path = f"/data3/zihanwa3/_Robotics/_baselines/VideoMimic/real2sim/_videomimic_data/output_calib_mesh/megahunter_megasam_reconstruction_results_{tgt_name}_cam01_frame_0_-1_subsample_1/gravity_calibrated_keypoints.h5"
    
    with h5py.File(calibrated_path, 'r') as f:
        world_rotation = f['world_rotation'][:]  # (3, 3)
    
    world_rot_tensor = torch.from_numpy(world_rotation).float()
    
    # Make sure everything is on CPU
    pred_trans = pred_trans.cpu()
    pred_rotmat = pred_rotmat.cpu()
    pred_shape = pred_shape.cpu()
    
    # UNDO calibration on translation
    if pred_trans.dim() == 3:
        pred_trans = pred_trans.squeeze(1)  # Ensure (T, 3)
    uncalibrated_trans = pred_trans @ world_rot_tensor  # (new / scale) @ world_rotation
    
    # UNDO calibration on global orientation
    uncalibrated_rotmat = pred_rotmat.clone()
    world_rot_inv = world_rot_tensor.T
    for t in range(pred_rotmat.shape[0]):
        uncalibrated_rotmat[t, 0] = world_rot_inv @ pred_rotmat[t, 0]
    
    # Use with SMPL (on CPU)
    pred = smpl_template(
        body_pose=uncalibrated_rotmat[:, 1:],
        global_orient=uncalibrated_rotmat[:, [0]],
        betas=pred_shape,
        transl=uncalibrated_trans,
        pose2rot=False,
        default_smpl=True
    )
    
    return pred


import torch
def to_cuda(data):
    """Move data in the batch to cuda(), carefully handle data that is not tensor"""
    if isinstance(data, torch.Tensor):
        return data.cuda()
    elif isinstance(data, dict):
        return {k: to_cuda(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_cuda(v) for v in data]
    else:
        return data
def axis_angle_to_matrix(angle_axis):
    # angle_axis: [N, 3]
    theta = torch.norm(angle_axis, dim=1, keepdim=True)  # [N, 1]
    axis = angle_axis / (theta + 1e-6)  # [N, 3]
    
    K = torch.zeros(angle_axis.shape[0], 3, 3, device=angle_axis.device)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]
    
    I = torch.eye(3, device=angle_axis.device).unsqueeze(0)
    R = I + torch.sin(theta).unsqueeze(-1) * K + (1 - torch.cos(theta)).unsqueeze(-1) * torch.bmm(K, K)
    return R  # [N, 3, 3]


def get_intrinsics_matrix(camera_params):
    """
    Returns a 3x3 camera intrinsics matrix from a dictionary input.
    
    Expected keys in `camera_params`:
        - 'img_focal': horizontal focal length (fx)
        - 'img_center': numpy array or list with [cx, cy]
        - 'spec_focal': vertical focal length (fy)
    """
    fx = camera_params['img_focal']
    fy = camera_params['spec_focal']
    cx, cy = camera_params['img_center']
    
    K = np.array([
        [fx, 0, cx],
        [0, fx, cy],
        [0, 0, 1]
    ])
    return K 

def _extract_mesh_vertex_colors(mesh: trimesh.Trimesh, force_gray=False, default_rgb=(180,180,180)):
    V = len(mesh.vertices)
    if force_gray or V == 0:
        return np.full((V,3), np.array(default_rgb, np.uint8), dtype=np.uint8)
    # 顶点色优先
    try:
        vc = np.asarray(mesh.visual.vertex_colors)
        if vc is not None and vc.size >= V*3:
            vc = vc.reshape(-1,4)[:, :3]
            if vc.max() <= 1.0:
                vc = (vc * 255.0).clip(0,255).astype(np.uint8)
            else:
                vc = vc.astype(np.uint8)
            return vc
    except Exception:
        pass
    # 兜底灰色
    return np.full((V,3), np.array(default_rgb, np.uint8), dtype=np.uint8)

def _prep_scene_mesh_points(mesh: trimesh.Trimesh, n_samples: int, force_gray=False):
    """
    返回 world 坐标点与颜色（不做任何变换）
    n_samples=0 → 用顶点；>0 → surface 均匀采样。
    """
    if n_samples <= 0:
        Vw = np.asarray(mesh.vertices, dtype=np.float64)
        C  = _extract_mesh_vertex_colors(mesh, force_gray=force_gray)
        return Vw, C

    # surface 均匀采样（颜色用灰色或顶点色重心插值不足稳定，这里按需简化为灰色）
    P, _ = trimesh.sample.sample_surface_even(mesh, int(n_samples))
    P = P.astype(np.float64)
    if force_gray:
        C = np.full((len(P),3), 180, np.uint8)
    else:
        # 简洁起见：直接灰色；如需纹理/顶点色插值，可扩展
        C = np.full((len(P),3), 180, np.uint8)
    return P, C


def matrix_to_axis_angle(R):
    # R: [N, 3, 3]
    cos_theta = ((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]) - 1) / 2
    theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))  # [N]
    
    rx = R[:, 2, 1] - R[:, 1, 2]
    ry = R[:, 0, 2] - R[:, 2, 0]
    rz = R[:, 1, 0] - R[:, 0, 1]
    r = torch.stack([rx, ry, rz], dim=1)  # [N, 3]

    # Normalize rotation axis
    r_norm = torch.norm(r, dim=1, keepdim=True) + 1e-6
    axis = r / r_norm

    return axis * theta.unsqueeze(1)  # [N, 3]
def build_4x4_poses(world_cam_R, world_cam_T):
    N = len(world_cam_R)
    all_poses = []

    for i in range(N):
        # Create a 4x4 identity
        pose = np.eye(4)

        # Fill in the top-left 3x3 with the rotation
        pose[:3, :3] = world_cam_R[i]

        # Fill in the top-right 3x1 with the translation
        pose[:3, 3] = world_cam_T[i]

        all_poses.append(pose)

    # Stack into [N, 4, 4]
    return np.stack(all_poses, axis=0)

def _world2cam_from_Twc(T_wc_np: np.ndarray):
    """Twc(相机到世界) -> PyTorch3D 需要的 R,T:  world->cam"""
    R_wc = T_wc_np[:3, :3]
    t_wc = T_wc_np[:3, 3]
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc
    return R_cw, t_cw



def _mesh_to_textured(vertices_np: np.ndarray, faces_np: np.ndarray,
                      colors_u8: np.ndarray | None, rgb_fallback=(180,180,180),
                      device="cpu") -> Meshes:
    V = torch.tensor(vertices_np, dtype=torch.float32, device=device)
    F = torch.tensor(faces_np.astype(np.int64), dtype=torch.int64, device=device)
    if colors_u8 is None or colors_u8.shape[0] != vertices_np.shape[0]:
        col = torch.tensor(rgb_fallback, dtype=torch.float32, device=device).view(1,1,3)/255.0
        C = col.repeat(1, V.shape[0], 1)   # (1,V,3)
    else:
        C = torch.tensor(colors_u8, dtype=torch.float32, device=device)[None, ...] / 255.0  # (1,V,3)
    tex = TexturesVertex(verts_features=C)
    return Meshes(verts=[V], faces=[F], textures=tex)

def _extract_trimesh_vertex_colors_u8(mesh: trimesh.Trimesh) -> np.ndarray | None:
    try:
        vc = np.asarray(mesh.visual.vertex_colors)
        if vc is None or vc.size == 0: return None
        vc = vc.reshape(-1, vc.shape[-1])
        if vc.shape[1] >= 3:
            vc = vc[:, :3]
            if vc.max() <= 1.0: vc = (vc * 255.0).clip(0,255).astype(np.uint8)
            else: vc = vc.astype(np.uint8)
            return vc
    except Exception:
        pass
    return None
# ===== 工具：给网格做“法线+棋盘”顶点色 =====
def _shaded_checker_colors(V_t, F_t, base_rgb=(180,180,180), freq=25.0):
    tmp_mesh = Meshes(verts=[V_t], faces=[F_t])
    N = tmp_mesh.verts_normals_packed()              # (V,3)

    ldir = torch.tensor([0.4, 0.8, 1.0], device=V_t.device)
    ldir = ldir / (ldir.norm() + 1e-8)
    lambert = (N @ ldir).clamp(min=0.0).unsqueeze(1) # (V,1)

    s = torch.sin(V_t[:, 0] * freq) * torch.sin(V_t[:, 2] * freq)
    checker = (s > 0).float().unsqueeze(1)

    base = torch.tensor(base_rgb, dtype=torch.float32, device=V_t.device) / 255.0
    base = base.view(1, 3).expand_as(V_t)

    ambient = 0.25; diffuse = 0.75
    tile_lo, tile_hi = 0.75, 1.15
    color = base * (ambient + diffuse * lambert) * (tile_lo + (tile_hi - tile_lo) * checker)
    return color.clamp(0.0, 1.0)

from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, RasterizationSettings,
    SoftPhongShader, TexturesVertex, PointLights
)
from pytorch3d.renderer import PointLights, SoftPhongShader
def _build_renderer(H, W, device):
    rast = RasterizationSettings(
        image_size=(H, W),
        faces_per_pixel=1,
        cull_backfaces=False,
        blur_radius=0.0,
    )
    lights = PointLights(device=device, location=[[0.6, 1.2, 1.8]])  # 更有对比的漫反射
    return MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=rast),
        shader=SoftPhongShader(lights=lights, device=device),
    )

import numpy as np
import cv2

def project_points(
    Xw: np.ndarray,
    K: np.ndarray,
    T_world_camera: np.ndarray,
    H: int, W: int,
    splat_radius: int = 1,
    z_eps: float = 1e-4
):
    assert Xw.ndim == 2 and Xw.shape[1] == 3
    img  = np.full((H, W, 3), 255, np.uint8)
    zbuf = np.full((H, W), np.inf, np.float64)

    R_wc = T_world_camera[:3, :3]
    t_wc = T_world_camera[:3, 3]
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc

    Xc = (Xw @ R_cw.T) + t_cw
    Z  = Xc[:, 2]
    valid = Z > z_eps
    if not np.any(valid):
        return img, zbuf, np.array([], np.int32), np.array([], np.int32), np.array([], np.float64), np.array([], np.int64)

    valid_idx = np.nonzero(valid)[0]   # <<< 关键：保留原始下标
    Xc = Xc[valid]; Z = Z[valid]

    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    u = fx * (Xc[:, 0] / Z) + cx
    v = fy * (Xc[:, 1] / Z) + cy

    u_i = np.round(u).astype(np.int32)
    v_i = np.round(v).astype(np.int32)

    inside = (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)
    keep_idx = valid_idx[inside]       # <<< 关键：和 colors 对齐用

    return img, zbuf, u_i[inside], v_i[inside], Z[inside], keep_idx

def draw_points_no_splat(
    img: np.ndarray,          # (H,W,3) uint8
    zbuf: np.ndarray,         # (H,W) float64, 先全设为 +inf
    u_i: np.ndarray,          # (M,)
    v_i: np.ndarray,          # (M,)
    Z: np.ndarray,            # (M,)
    colors: np.ndarray,       # (M,3) uint8
    H: int, W: int
) -> np.ndarray:
    # 只在单个像素写入；如已有更近深度则覆盖
    for uu, vv, zz, col in zip(u_i, v_i, Z, colors):
        if zz <= 0:
            continue
        if 0 <= uu < W and 0 <= vv < H and zz < zbuf[vv, uu]:
            zbuf[vv, uu] = zz
            img[vv, uu]  = col
    return img


# ===== paste this near your imports =====
def ensure_headless_client(server, timeout_sec: float = 30.0):
    """
    启动 headless Chromium 连接 Viser，并返回第一个 ClientHandle。
    若已有人连上（你自己或别的 headless），则直接返回那个 client。
    """
    import time
    from contextlib import suppress

    url = f"http://{server.get_host()}:{server.get_port()}"
    # 若已经有客户端，就不再拉起浏览器
    clients = server.get_clients()
    if clients:
        return list(clients.values())[0], None  # (client, browser_handles=None)

    # 尝试用 Playwright 启动 headless Chromium
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        raise RuntimeError(
            "需要 headless 客户端，但未安装 Playwright。请先执行：\n"
            "  pip install playwright\n"
            "  python -m playwright install chromium\n"
            f"原始错误：{e}"
        )

    pw = sync_playwright().start()
    browser = pw.chromium.launch(
        headless=True,
        args=[
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--no-sandbox",
            "--disable-setuid-sandbox",
        ],
    )
    page = browser.new_page(viewport={"width": 1280, "height": 800, "deviceScaleFactor": 1.0})
    page.goto(url, wait_until="load")

    # 等待 client 出现
    t0 = time.time()
    while True:
        clients = server.get_clients()
        if clients:
            client = list(clients.values())[0]
            return client, (pw, browser, page)  # 返回浏览器句柄，便于收尾
        if time.time() - t0 > timeout_sec:
            # 失败则清理并报错
            with suppress(Exception):
                page.close(); browser.close(); pw.stop()
            raise TimeoutError(f"在 {timeout_sec:.0f}s 内没有 headless 客户端连上 {url}")
        time.sleep(0.05)

def close_headless_browser(browser_handles):
    """渲染完毕后关闭 headless 浏览器（如果是我们启动的）。"""
    from contextlib import suppress
    if browser_handles is None:
        return
    pw, browser, page = browser_handles
    with suppress(Exception):
        page.close()
        browser.close()
        pw.stop()
# ===== end paste =====


def _concat_3cols(chunks, want_dtype=None, is_color=False):
    arrs = []
    for idx, c in enumerate(chunks):
        if c is None:
            continue
        c = np.asarray(c)
        # Allow flat length-3k vectors
        if c.ndim == 1:
            if c.size % 3 != 0:
                raise ValueError(f"Chunk {idx} is 1-D with size {c.size}, not divisible by 3.")
            c = c.reshape(-1, 3)
        # Allow extra leading dims (…,3)
        if c.shape[-1] != 3:
            raise ValueError(f"Chunk {idx} has shape {c.shape}, expected (..., 3).")
        c = c.reshape(-1, 3)
        arrs.append(c)

    if not arrs:
        return np.empty((0, 3), dtype=(want_dtype or np.float64))

    out = np.concatenate(arrs, axis=0)

    if is_color:
        # If float colors in [0,1], scale to [0,255]; if already 0–255, leave as-is
        if np.issubdtype(out.dtype, np.floating):
            maxv = float(np.nanmax(out)) if out.size else 1.0
            out = (out * 255.0 if maxv <= 1.0 else out)
        out = np.clip(out, 0, 255).astype(np.uint8, copy=False)
    elif want_dtype is not None:
        out = out.astype(want_dtype, copy=False)

    return out

device_p3d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from pytorch3d.renderer import PerspectiveCameras


TEX_SIZE = 1024
def _load_npz_to_dict(path: Path, *, allow_pickle: bool = False) -> dict:
    """Utility: load .npz into a writable dict."""
    with np.load(path, allow_pickle=allow_pickle) as f:
        return {k: f[k] for k in f}

def _make_checker_map(size=1024, tiles=18, device="cpu"):
    """
    生成棋盘纹理 (H, W, 3) float32 in [0,1]   ← 注意不再有 batch 维度
    """
    H = W = int(size)
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    tile_h = max(1, H // tiles)
    tile_w = max(1, W // tiles)
    c = ((yy // tile_h) + (xx // tile_w)) % 2
    c = c.float().unsqueeze(-1).repeat(1, 1, 3)  # (H, W, 3)

    col0 = torch.tensor([0.85, 0.85, 0.85], device=device).view(1, 1, 3)
    col1 = torch.tensor([0.65, 0.65, 0.65], device=device).view(1, 1, 3)
    img = col0 * (1 - c) + col1 * c
    return img  # (H, W, 3)


def colorize_scene_by_multiframe_contacts(
    nksr_mesh: trimesh.Trimesh,
    contact_groups_pts: dict[str, np.ndarray],   # {group_name: (K,3) or (T,*,3)}
    colors_rgb: dict[str, list[int]],            # {group_name: [R,G,B]}
    radius: float = 0.03,                        # meters; ~3cm good default
    base_gray: int = 178,                        # background color ~0.7*255
    return_labels: bool = False
):
    """
    Assign each scene vertex the color of the *closest* contact group within `radius`,
    aggregating contacts across ALL frames. Unlabeled vertices get neutral gray.

    contact_groups_pts: For each group, pass either:
        - (K,3) points aggregated across all frames, OR
        - (T,Ni,3) per-frame arrays (we'll reshape to (K,3)).

    colors_rgb: maps each group to an [R,G,B] in 0..255.
    """
    mesh = nksr_mesh.copy()
    V = np.asarray(mesh.vertices, dtype=np.float64)
    M = V.shape[0]

    # Prepare KD-trees per group
    group_names = list(contact_groups_pts.keys())
    G = len(group_names)
    trees = []
    for g in group_names:
        pts = np.asarray(contact_groups_pts[g])
        pts = pts.reshape(-1, 3)  # handle (T,*,3) or (K,3)
        pts = pts[np.isfinite(pts).all(axis=1)]  # drop NaNs/infs
        if pts.size == 0:
            trees.append(None)
        else:
            trees.append(cKDTree(pts))

    # Distances per group to nearest contact point (∞ if none within radius)
    dists = np.full((G, M), np.inf, dtype=np.float32)
    for gi, tree in enumerate(trees):
        if tree is None:
            continue
        # query nearest contact point; cap at radius
        di, _ = tree.query(V, k=1, distance_upper_bound=radius)
        dists[gi] = di

    # Choose the closest group (smallest distance) per vertex
    best_d = dists.min(axis=0)
    best_g = dists.argmin(axis=0)
    contact_mask = np.isfinite(best_d)

    # Build vertex colors (uint8)
    vc = np.empty((M, 4), dtype=np.uint8)
    vc[:] = [base_gray, base_gray, base_gray, 255]  # background gray

    for gi, g in enumerate(group_names):
        color = np.array(colors_rgb[g], dtype=np.uint8)
        gi_mask = contact_mask & (best_g == gi)
        vc[gi_mask, :3] = color

    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=vc)

    if return_labels:
        labels = np.full(M, -1, dtype=np.int32)
        labels[contact_mask] = best_g[contact_mask]
        return mesh, labels, group_names
    return mesh

def _camera_visibility_debug(Vw_np, K, T_wc, H, W):
    """返回：在前方的点数、投影落在图内的点数"""
    Vw = Vw_np.astype(np.float64)
    R_wc = T_wc[:3, :3]
    t_wc = T_wc[:3, 3]
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc
    Xc = (Vw @ R_cw.T) + t_cw              # (N,3)
    Z = Xc[:, 2]
    front = Z > 1e-4
    if not np.any(front):
        return 0, 0
    Xc = Xc[front]
    Z = Z[front]
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    u = fx * (Xc[:,0] / Z) + cx
    v = fy * (Xc[:,1] / Z) + cy
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return int(front.sum()), int(inside.sum())



def _make_solid_map(rgb01, size=1024, device="cpu"):
    """
    纯色贴图 (H, W, 3) float32 in [0,1]       ← 注意不再有 batch 维度
    """
    h = w = int(size)
    col = torch.tensor(rgb01, dtype=torch.float32, device=device).view(1, 1, 3)
    img = col.expand(h, w, 3)  # (H, W, 3)
    return img


def matrix_to_axis_angle_pytorch3d(rotation_matrices):
    """
    Convert rotation matrices to axis-angle using PyTorch3D (if available).
    
    Args:
        rotation_matrices: torch.Tensor of shape [..., 3, 3]
    
    Returns:
        axis_angles: torch.Tensor of shape [..., 3]
    """
    try:
        from pytorch3d.transforms import matrix_to_axis_angle
        return matrix_to_axis_angle(rotation_matrices)
    except ImportError:
        print("pytorch3d not available, using manual implementation")
        return matrix_to_axis_angle_manual(rotation_matrices)

import torch


def to_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

def make_affine(s, R, t):
    A = np.eye(4, dtype=np.float32)
    A[:3, :3] = float(s) * to_np(R)
    A[:3, 3]  = to_np(t)
    return A
from pytorch3d.renderer import PerspectiveCameras
# === UV 工具 ===
def _planar_uv_from_bbox(V_t: torch.Tensor, axes=(0, 2), eps: float = 1e-6) -> torch.Tensor:
    """
    按 bbox 做平面投影 UV（默认 XZ→UV）。V_t: (V,3) torch
    返回 (V,2) in [0,1]
    """
    x = V_t[:, axes[0]]
    y = V_t[:, axes[1]]
    x0, x1 = x.min(), x.max()
    y0, y1 = y.min(), y.max()
    u = (x - x0) / (x1 - x0 + eps)
    v = (y - y0) / (y1 - y0 + eps)
    # 让棋盘更密一点：重复几次
    uv = torch.stack([u, v], dim=-1) * 4.0   # ×4 意味着多铺几块
    return uv % 1.0



def _twc_to_cam_pose_world(T_wc: np.ndarray):
    """
    输入: T_wc (4x4, camera->world)
    输出: cam_pos_world(3,), cam_quat_wxyz(4,)
    """
    R_wc = np.asarray(T_wc[:3, :3], dtype=float)
    t_wc = np.asarray(T_wc[:3, 3],  dtype=float)

    # 显式从旋转矩阵生成四元数（w,x,y,z）
    q_wxyz = viser.transforms.SO3.from_matrix(R_wc).wxyz
    q_wxyz = np.asarray(q_wxyz, dtype=float).reshape(4,)   # 确保是 (4,)

    return t_wc.astype(float), q_wxyz

def _camera_from_K_Twc_p3d(K_np, Twc_np, H, W, device, flip='yz'):
    # K: OpenCV 像素内参；Twc: cam->world
    K   = torch.as_tensor(K_np, dtype=torch.float32, device=device)
    Twc = torch.as_tensor(Twc_np, dtype=torch.float32, device=device)

    R_wc = Twc[:3, :3]           # cam->world
    t_wc = Twc[:3, 3]
    R_cw = R_wc.t()              # world->cam
    t_cw = -R_cw @ t_wc

    # ------- 可切换的坐标翻转 -------
    flips = {
        'none': torch.diag(torch.tensor([ 1.,  1.,  1.], device=device)),
        'y':    torch.diag(torch.tensor([ 1., -1.,  1.], device=device)),
        'z':    torch.diag(torch.tensor([ 1.,  1., -1.], device=device)),
        'yz':   torch.diag(torch.tensor([ 1., -1., -1.], device=device)),  # 你当前用的
    }
    R_cv2p3d = flips.get(flip, flips['yz'])
    R_p3d = R_cv2p3d @ R_cw
    t_p3d = (R_cv2p3d @ t_cw).unsqueeze(0)

    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    return PerspectiveCameras(
        R=R_p3d.unsqueeze(0), T=t_p3d,
        focal_length=torch.tensor([[fx, fy]], device=device),
        principal_point=torch.tensor([[cx, cy]], device=device),
        image_size=torch.tensor([[H, W]], device=device),
        in_ndc=False, device=device
    )

def main(
   data: Path = "./demo_tmp/NULL.npz",
   downsample_factor: int = 1,
   max_frames: int = 800,
   share: bool = False,
   conf_threshold: float = 0.3,
   foreground_conf_threshold: float = 0.,
   point_size: float = 0.01,
   camera_frustum_scale: float = 0.02,
   no_mask: bool = False,
   xyzw: bool = True,
   axes_scale: float = 0.25,
   bg_downsample_factor: int = 1,
   init_conf: bool = False,
   cam_thickness: float = 1.5,
   save_mode: bool = False, 
   transfer_data: bool = False, 
   moge_base_path: str = '/data3/zihanwa3/_Robotics/_vision/TAPIP3D/_raw_mega_priors',
   contact_threshold: float = 5  # ADDED: threshold for contact detection
) -> None:
   from pathlib import Path  
   tgt_name = str(data).split('_sgd')[0].split('/')[-1]   
   tgt_name = "_".join(tgt_name.split("_")[:-1]) 
   moge_data = os.path.join(moge_base_path, f'{tgt_name}.npz')
   tgt_folder = os.path.join('/data3/zihanwa3/_Robotics/_geo/differentiable-blocksworld/post_results', tgt_name)
   #if Path(tgt_folder).exists():
   #  shutil.rmtree(Path(tgt_folder))

   if 'door' in str(data):
       extra_obj = True
   
   data = _load_npz_to_dict(data)
   moge_data = np.load(moge_data)
   data["depths"] = moge_data["depths"]
   data["images"] = moge_data["images"]
   data['cam_c2w'] = moge_data['cam_c2w']
   data['intrinsic'] = moge_data['intrinsic']

   gt_data_dir = '/data3/zihanwa3/_Robotics/_data'
   raw_clips_dir = '/data3/zihanwa3/_Robotics/_data/_EMDB_2'

   gt_hmr = find_gt_file(tgt_name, gt_data_dir)
   raw_clips_path = os.path.join(raw_clips_dir, tgt_name)
   

   import random

   def generate_four_digit():
       return random.randint(1000, 9999)
   server = viser.ViserServer(port=generate_four_digit())
   if share:
       server.request_share_url()

   ratio_block_scene =1 
   do_mesh = True
   use_world = True
   server.scene.set_up_direction('-z')
   if no_mask or not do_mesh:             
       init_conf = True    
       fg_conf_thre = conf_threshold 
   print("Loading frames!")
   
   key_R, key_T =  'world_cam_R', 'world_cam_T' 

   num_frames = len(data['depths'])

   npz_cam_data = data
   device='cuda'
   base_folder = '/data3/zihanwa3/_Robotics/_vision/tram/megasamra'
   base   = Path(base_folder)
   candid = sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith(tgt_name))

   camera = np.load(candid[0] / "camera.npy", allow_pickle=True).item()
   fx= fy = img_focal = camera['img_focal']
   pred_cam ={}
   pred_cam[key_R] = npz_cam_data['cam_c2w'][:, :3, :3]
   pred_cam[key_T] = (npz_cam_data['cam_c2w'][:, :3, 3] )* npz_cam_data['scale']
   world_cam_R = torch.tensor(pred_cam[key_R]).to(device)
   world_cam_T = torch.tensor(pred_cam[key_T]).to(device)

   from smpl_utils import (
       process_tram_smpl,
       process_gv_smpl,
       load_contact_ids_from_file,
       load_contact_ids_with_mode, 
       filter_vertices_by_contact,
       vis_hmr,
       analyze_motion,
       filter_stable_contacts_simple,
       inspect_contact_points
   )
   

   device='cuda'
   smpl = SMPL().to(device)
   interact_contact_path = os.path.join('/data3/zihanwa3/_Robotics/_data/_contact', tgt_name) 

   # ADDED: Process multiple HMR types
   hmr_results = {}
   hmr_colors = {
       'gv':  [205, 232, 200],    
       'tram': [5, 232, 200],    
       'videomicmic': [229, 80, 80],  # Red
       'GT': [255,165,0],

       'postrl_vmm': [160, 32, 240], 
       'postrl_tram': [255, 255, 255], 
       'postrl_ours': [80, 229, 80],

   }
   hmr_types = [ 'GT', 'videomicmic', 'postrl_vmm', 'postrl_ours']# , 'gv'] # , 'postrl' tram 
   hmr_types = ['videomimic']# , 'gv'] # , 'postrl' tram 

   # nksr_mesh_path = os.path.join('outest_true', tgt_name, 'scene_nksr_coacd_mesh_nksr.obj')
   nksr_mesh_path = f'/data3/zihanwa3/_Robotics/_baselines/VideoMimic/real2sim/_videomimic_data/output_calib_mesh/megahunter_megasam_reconstruction_results_{tgt_name}_cam01_frame_0_-1_subsample_1/background_mesh.obj'


   smpl_results = process_gv_smpl(
    tgt_name=tgt_name,
    world_cam_R=world_cam_R,
    world_cam_T=world_cam_T,
    max_frames=max_frames,
    smpl_model=smpl,
    device='cuda'
)
   faces = smpl_results['faces']# .cpu().numpy()

   if True: 
    calibrated_path = f"/data3/zihanwa3/_Robotics/_baselines/VideoMimic/real2sim/_videomimic_data/output_calib_mesh/megahunter_megasam_reconstruction_results_{tgt_name}_cam01_frame_0_-1_subsample_1/gravity_calibrated_keypoints.h5"
    calibrated_path = calibrated_path.format(tgt_name=tgt_name)
    
    # Undo the calibration
    uncalibrated_data, world_rotation = undo_gravity_calibration(
        calibrated_path, 
        world_scale_factor=1.0  # Use the same scale factor that was used during calibration
    )

   for hmr_type in hmr_types:
       print(f"Processing HMR type: {hmr_type}")
      
       if hmr_type == 'videomimic':


          vmm_folder = '/data3/zihanwa3/_Robotics/_baselines/VideoMimic/real2sim/__scene_vanila'
          file_path = os.path.join(vmm_folder, f'sloper4d_seq_{tgt_name}', "hps_combined_track_0.npy") 

          pred_smpl = np.load(file_path, allow_pickle=True).item()

          pred_rotmat = torch.tensor(pred_smpl['pred_rotmat'])    # T, 24, 3, 3
          pred_shape = torch.tensor(pred_smpl['pred_shape'])      # T, 10
          pred_trans = torch.tensor(pred_smpl['pred_trans'])      # T, 1, 3

          mean_shape = pred_shape.mean(dim=0, keepdim=True)
          pred_shape = mean_shape.repeat(len(pred_shape), 1)

          pred = smpl_template(body_pose=pred_rotmat[:,1:], 
                                  global_orient=pred_rotmat[:,[0]], 
                                  betas=pred_shape, 
                                  transl=pred_trans.squeeze(),
                                  pose2rot=False, 
                                  default_smpl=True)
          pred_vert = pred.vertices.numpy()
          pred_j3d = pred.joints[:, :24].numpy()
       else:
           print(f"Unknown HMR type: {hmr_type}, skipping...")
           continue
      

       ALIGN = False 
       if ALIGN and hmr_type != 'GT':
          
          print(gt_j3d.shape, pred_j3d.shape)
          # (308, 24, 3) (307, 24, 3)
          if isinstance(gt_j3d, np.ndarray):
              gt_j3d = torch.from_numpy(gt_j3d).float()#.to(pred_j3d.device)
          if isinstance(pred_j3d, np.ndarray):
              pred_j3d = torch.from_numpy(pred_j3d).float()#to(pred_j3d.device)

          if isinstance(pred_vert, np.ndarray):
              pred_vert = torch.from_numpy(pred_vert).float()#to(pred_j3d.device)

          wa_j3d, s_first, R_first, t_first = global_align_joints(gt_j3d[:len(pred_j3d)], pred_j3d)#.cpu().numpy()
          print(s_first.shape, R_first.shape, t_first.shape, s_first)
          # 应用到所有帧的顶点 (T,V,3)
          
  
          s_first_, R_first_, t_first_ = s_first, R_first, t_first 


          # t_first = torch.from_numpy(t_first).float()
          print(f'applied to {hmr_type}!!!')
          pred_vert = apply_alignment_to_verts(pred_vert, s_first, R_first, t_first).cpu().numpy()


          # pred_vert = w_j3d
      
       hmr_results[hmr_type] = {
           'pred_vert': pred_vert,
           'num_frames': num_frames, 
           'color': hmr_colors.get(hmr_type, [200, 200, 200])
       }

   # Use first HMR result for global parameters
   first_hmr = list(hmr_results.values())[0]
   num_frames =  first_hmr['num_frames']                               # first_hmr['num_frames']

   save_dir = '/data3/zihanwa3/_Robotics/_geo/differentiable-blocksworld/post_results'
   save_dir = os.path.join(save_dir, tgt_name)
   os.makedirs(save_dir, exist_ok=True)
   save_dir = os.path.join(save_dir, 'hmr')
   os.makedirs(save_dir, exist_ok=True)
   hmr_dir = os.path.join(save_dir, 'hmr')
   os.makedirs(hmr_dir, exist_ok=True)

   loader = viser.extras.Record3dLoader_Customized_Megasam(
       data,
       npz_cam_data, 
       conf_threshold=1.0,
       foreground_conf_threshold=foreground_conf_threshold,
       no_mask=no_mask,
       xyzw=xyzw,
       init_conf=init_conf,
   )


   if True:
        CONTACT_IDS_SMPL_LIST = load_contact_ids_with_mode(mode='grouped')
        
        leg_ids, hand_ids, gluteus_ids, back_ids, thigh_ids = CONTACT_IDS_SMPL_LIST
        leg_ids = leg_ids.cpu().numpy()
        hand_ids = hand_ids.cpu().numpy()
        gluteus_ids = gluteus_ids.cpu().numpy()
        back_ids = back_ids.cpu().numpy()
        thigh_ids = thigh_ids.cpu().numpy()
        PROJ_ROOT = '/data3/zihanwa3/_Robotics/_vision/GVHMR/'
        smplx2smpl_map = torch.load(
            f"{PROJ_ROOT}/hmr4d/utils/body_model/smplx2smpl_sparse.pt",
            map_location="cpu"
        )

        if smplx2smpl_map.is_sparse:
            smplx2smpl_map = smplx2smpl_map.to_dense()

        # lookup: for each smplx vertex i, x2s[i] = closest smpl vertex
        x2s = torch.argmax(smplx2smpl_map, dim=0).cpu().numpy()   # shape (10475,)

        def convert_indices_smplx_to_smpl(smplx_indices, x2s_lookup):
            """Return SMPL indices (list of ints) corresponding to SMPL-X indices."""
            return np.unique(x2s_lookup[smplx_indices]).tolist()

        # === convert all groups to SMPL indices ===
        leg_ids_smpl     = convert_indices_smplx_to_smpl(leg_ids,     x2s)
        hand_ids_smpl    = convert_indices_smplx_to_smpl(hand_ids,    x2s)
        gluteus_ids_smpl = convert_indices_smplx_to_smpl(gluteus_ids, x2s)
        back_ids_smpl    = convert_indices_smplx_to_smpl(back_ids,    x2s)
        thigh_ids_smpl   = convert_indices_smplx_to_smpl(thigh_ids,   x2s)
        vis_inverval = 45

   # ADDED: Single NKSR mesh handle
   nksr_mesh_handle = None
   
   # ADDED: Multiple HMR mesh handles
   hmr_mesh_handles = {}

   # Add playback UI.
   with server.gui.add_folder("Playback"):
       gui_timestep = server.gui.add_slider(
           "Timestep",
           min=0,
           max=num_frames - 1,
           step=1,
           initial_value=0,
           disabled=True,
       )
       gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
       gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
       gui_playing = server.gui.add_checkbox("Playing", True)
       gui_framerate = server.gui.add_slider(
           "FPS", min=1, max=60, step=0.1, initial_value=loader.fps
       )
       gui_show_all_frames = server.gui.add_checkbox("Show all frames", False)
       gui_stride = server.gui.add_slider(
           "Stride",
           min=1,
           max=num_frames,
           step=1,
           initial_value=1,
           disabled=True,
       )

   # ADDED: Simplified display controls
   with server.gui.add_folder("Display"):
       gui_show_nksr = server.gui.add_checkbox("Show NKSR Mesh", True)
       gui_show_sqs = server.gui.add_checkbox("Show SQS Mesh", True)  # ADD THIS LINE
       gui_show_points = server.gui.add_checkbox("Show Points", True)
       
       # ADDED: HMR model controls
       gui_hmr_controls = {}
       for hmr_type in hmr_types:
           gui_hmr_controls[hmr_type] = server.gui.add_checkbox(f"Show HMR {hmr_type.upper()}", True)


   def _apply_hmr_visibility():
        if gui_show_all_frames.value:
            stride = max(1, int(gui_stride.value))
            with server.atomic():
                # 显示所有 HMR（按 stride 采样）
                for ht, handles in hmr_mesh_handles.items():
                    show_ht = gui_hmr_controls[ht].value
                    for i, h in enumerate(handles):
                        h.visible = show_ht and (i % stride == 0)
        else:
            cur = int(gui_timestep.value)
            with server.atomic():
                # 仅显示当前帧的 HMR
                for ht, handles in hmr_mesh_handles.items():
                    show_ht = gui_hmr_controls[ht].value
                    for i, h in enumerate(handles):
                        h.visible = show_ht and (i == cur)

   # Frame step buttons.
   @gui_next_frame.on_click
   def _(_) -> None:
       gui_timestep.value = (gui_timestep.value + 1) % num_frames

   @gui_prev_frame.on_click
   def _(_) -> None:
       gui_timestep.value = (gui_timestep.value - 1) % num_frames

   # Disable frame controls when we're playing.
   @gui_playing.on_update
   def _(_) -> None:
       gui_timestep.disabled = gui_playing.value or gui_show_all_frames.value
       gui_next_frame.disabled = gui_playing.value or gui_show_all_frames.value
       gui_prev_frame.disabled = gui_playing.value or gui_show_all_frames.value

   # ADDED: NKSR visibility callback
   @gui_show_nksr.on_update
   def _(_):
       if nksr_mesh_handle:
           nksr_mesh_handle.visible = gui_show_nksr.value

   @gui_show_sqs.on_update
   def _(_):
       if sqs_mesh_handle:
           sqs_mesh_handle.visible = gui_show_sqs.value
            
   # ADDED: HMR visibility callbacks
   for hmr_type in hmr_types:
       @gui_hmr_controls[hmr_type].on_update
       def _(_, ht=hmr_type):
           if ht in hmr_mesh_handles:
               current_frame = gui_timestep.value
               for i, handle in enumerate(hmr_mesh_handles[ht]):
                   handle.visible = gui_hmr_controls[ht].value and (i == current_frame)
   
   # ADDED: Point visibility callback that also controls contact mesh
   @gui_show_points.on_update  
   def _(_):
       current_timestep = gui_timestep.value
       # Update contact mesh visibility based on show_points
       for i, contact_handle in enumerate(contact_mesh_handles):
           if contact_handle is not None:
               contact_handle.visible = gui_show_points.value and (i == current_timestep)
   @gui_show_all_frames.on_update
   def _(_):
        gui_stride.disabled = not gui_show_all_frames.value

        if gui_show_all_frames.value:
            stride = max(1, int(gui_stride.value))
            with server.atomic():
                for i, node in enumerate(frame_nodes):
                    node.visible = (i % stride == 0)
            # 禁用逐帧控件
            gui_playing.disabled = True
            gui_timestep.disabled = True
            gui_next_frame.disabled = True
            gui_prev_frame.disabled = True
        else:
            cur = int(gui_timestep.value)
            with server.atomic():
                for i, node in enumerate(frame_nodes):
                    node.visible = (i == cur)
            gui_playing.disabled = False
            gui_timestep.disabled = gui_playing.value
            gui_next_frame.disabled = gui_playing.value
            gui_prev_frame.disabled = gui_playing.value

        _apply_hmr_visibility()

   # MODIFIED: Simplified timestep update with contact mesh visibility
   @gui_timestep.on_update
   def _(_) -> None:
       current_timestep = gui_timestep.value
       if not gui_show_all_frames.value:
           with server.atomic():
               # Update frame nodes visibility
               for i, frame_node in enumerate(frame_nodes):
                   frame_node.visible = (i == current_timestep)
                   
               # Update HMR mesh visibility
               for hmr_type, handles in hmr_mesh_handles.items():
                   show_hmr = gui_hmr_controls[hmr_type].value
                   for i, handle in enumerate(handles):
                       handle.visible = show_hmr and (i == current_timestep)
               
               # ADDED: Update contact mesh visibility
               for i, contact_handle in enumerate(contact_mesh_handles):
                   if contact_handle is not None:
                       # Show contact mesh for current frame if points are shown
                       contact_handle.visible = (i == current_timestep) and gui_show_points.value
                       
       server.flush()

   @gui_stride.on_update
   def _(_) -> None:
       if gui_show_all_frames.value:
           stride = gui_stride.value
           with server.atomic():
               for i, frame_node in enumerate(frame_nodes):
                   frame_node.visible = (i % stride == 0)

   # Load in frames.
   server.scene.add_frame(
       "/frames",
       wxyz=tf.SO3.exp(onp.array([onp.pi / 2.0, 0.0, 0.0])).wxyz,
       position=(0, 0, 0),
       show_axes=False,
   )

   server.scene.add_frame(
       "/frames",
       wxyz=tf.SO3.exp(onp.array([0.0, 0.0, 0.0])).wxyz,
       position=(0, 0, 0),
       show_axes=False,
   )

   frame_nodes: list[viser.FrameHandle] = []
   
   # ADDED: Store contact mesh handles for each frame
   contact_mesh_handles = []

   # ADDED: Load NKSR mesh if provided
   nksr_mesh_original = None  # Store original NKSR mesh
   if False:
    nksr_mesh_original = trimesh.load(nksr_mesh_path)
    nksr_mesh_handle = server.scene.add_mesh_trimesh(
        name="/nksr_mesh",
        mesh=nksr_mesh_original,
    )
    nksr_mesh_handle.visible = gui_show_nksr.value
    print(f"Loaded NKSR mesh with {len(nksr_mesh_original.vertices)} vertices")

   sqs_mesh_original = None  # Store original SQS mesh
   if True:
    # sqs_mesh_path = os.path.join('/data3/zihanwa3/_Robotics/_vision/mega-sam/post_results', tgt_name, 'scene_mesh_nksr', 'scene_nksr_coacd_mesh_nksr.obj')

    sqs_mesh_path = f'/data3/zihanwa3/_Robotics/_baselines/VideoMimic/real2sim/_videomimic_data/output_calib_mesh/megahunter_megasam_reconstruction_results_{tgt_name}_cam01_frame_0_-1_subsample_1/background_mesh.obj'

    sqs_mesh_original = trimesh.load(sqs_mesh_path)

    # 假设你已经有 s_first, R_first, t_first

    calibrated_path = f"/data3/zihanwa3/_Robotics/_baselines/VideoMimic/real2sim/_videomimic_data/output_calib_mesh/megahunter_megasam_reconstruction_results_{tgt_name}_cam01_frame_0_-1_subsample_1/gravity_calibrated_keypoints.h5"

    with h5py.File(calibrated_path, 'r') as f:
        world_rotation = f['world_rotation'][:]  # (3, 3)

    world_rot_tensor = torch.from_numpy(world_rotation).float()

    # 撤销gravity calibration
    # 1. 撤销translation变换
    uncalibrated_trans = pred_trans.squeeze() @ world_rot_tensor

    # 2. 撤销global orientation变换  
    uncalibrated_rotmat = pred_rotmat.clone()
    world_rot_inv = world_rot_tensor.T  # 逆矩阵
    A_inv = np.eye(4, dtype=np.float64)
    A_inv[:3, :3] = world_rotation.T
    A = A_inv
    scene_mesh_contact_tf = sqs_mesh_original.copy()
    scene_mesh_contact_tf.apply_transform(A)
    new_dir = '/data3/zihanwa3/_Robotics/_geo/differentiable-blocksworld/__vmm_aligned'
    os.makedirs(new_dir, exist_ok=True)
    out_path = os.path.join(new_dir, f'{tgt_name}_aligned.obj')
    obj_str = scene_mesh_contact_tf.export(file_type='obj')
    with open(out_path, 'w') as f:
        f.write(obj_str)

    # B. 直接导出到文件（不需要上面的字符串保存）
    # scene_mesh_contact_tf.export(out_path)

    print('Saved to:', out_path)

        

   if True:
    device_p3d = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sqs_mesh_handle = server.scene.add_mesh_trimesh(
        name="/sqs_mesh",
        mesh=scene_mesh_contact_tf,
    )
    sqs_mesh_handle.visible = gui_show_sqs.value
   interval = 100
   for hmr_type, hmr_result in hmr_results.items():
        # hmr_data = hmr_result['data']
        color = hmr_result['color']
        pred_vert = hmr_result['pred_vert']# .cpu().numpy()
        hmr_data = hmr_result
        
        handles = []
        for i in range(0, len(pred_vert), interval):
            handle = server.scene.add_mesh_simple(
                name=f"/hmr/{hmr_type}/frame_{i}",
                vertices=pred_vert[i],
                faces=faces,
                color=color,
                flat_shading=False,
                wireframe=False,
            )
            handle.visible = gui_hmr_controls[hmr_type].value and (i == 0)
            handles.append(handle)
        
        hmr_mesh_handles[hmr_type] = handles
 

if __name__ == "__main__":
   tyro.cli(main)