import time
import sys
import argparse
from pathlib import Path
import trimesh
import numpy as onp
import tyro
from tqdm.auto import tqdm
from sqs_utils.superquadric import *
from pytorch3d.transforms import euler_angles_to_matrix
import os, shutil
import copy
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

from pytorch3d.structures import Meshes

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
    out_dir     : Path | str                          保存 *.obj 和 scene.urdf 的目录
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


def contact_ids():
    body_segments_dir = '/data3/zihanwa3/_Robotics/_data/_PROX/body_segments'
    contact_verts_ids = []
    contact_body_parts = ['L_Leg', 'R_Leg', 'L_Hand', 'R_Hand', 'gluteus', 'back', 'thighs']
    for part in contact_body_parts:
        with open(os.path.join(body_segments_dir, part + '.json'), 'r') as f:
            data = json.load(f)
            contact_verts_ids.append(list(set(data["verts_ind"])))
    contact_verts_ids = np.concatenate(contact_verts_ids)
    return contact_verts_ids
def load_contact_ids(device="cuda"):
    ids_np = contact_ids()                 # numpy array from your helper
    ids    = torch.as_tensor(ids_np, dtype=torch.long, device=device)
    return ids  
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

    # ────── 5. Pack outputs ──────────────────────────────────────────
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

def _load_npz_to_dict(path: Path, *, allow_pickle: bool = False) -> dict:
    """Utility: load .npz into a writable dict."""
    with np.load(path, allow_pickle=allow_pickle) as f:
        return {k: f[k] for k in f}


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
    moge_base_path: str = '/data3/zihanwa3/_Robotics/_vision/TAPIP3D/_raw_mega_priors'
) -> None:
    from pathlib import Path  
    tgt_name = str(data).split('_sgd')[0].split('/')[-1]   
    tgt_name = "_".join(tgt_name.split("_")[:-1]) 
    moge_data = os.path.join(moge_base_path, f'{tgt_name}.npz')
    tgt_folder = os.path.join('/data3/zihanwa3/_Robotics/_geo/differentiable-blocksworld/post_results', tgt_name)
    if Path(tgt_folder).exists():
      shutil.rmtree(Path(tgt_folder))

    if 'door' in str(data):
        extra_obj = True
    
    data = _load_npz_to_dict(data)
    moge_data = np.load(moge_data)
    data["depths"] = moge_data["depths"]
    data["images"] = moge_data["images"]
    data['cam_c2w'] = moge_data['cam_c2w']
    data['intrinsic'] = moge_data['intrinsic']

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
        'gv': [229, 80, 80],      # Red
        'tram': [80, 229, 80],    # Green  
        'baseline': [80, 80, 229], # Blue
    }
    hmr_types = ['tram']
        
    nksr_mesh_path = os.path.join('outest_true', tgt_name, 'scene_nksr_coacd_mesh_nksr.obj')
    for hmr_type in hmr_types:
        print(f"Processing HMR type: {hmr_type}")
        
        if hmr_type == 'tram':
            smpl_results = process_tram_smpl(
                tgt_name=tgt_name,
                world_cam_R=world_cam_R,
                world_cam_T=world_cam_T,
                max_frames=max_frames,
                smpl_model=smpl,
                device='cuda'
            )
        elif hmr_type == 'gv':
            smpl_results = process_gv_smpl(
                tgt_name=tgt_name,
                world_cam_R=world_cam_R,
                world_cam_T=world_cam_T,
                max_frames=max_frames,
                smpl_model=smpl,
                use_world=use_world,
                device='cuda'
            )
        else:
            print(f"Unknown HMR type: {hmr_type}, skipping...")
            continue
            
        hmr_results[hmr_type] = {
            'data': smpl_results,
            'color': hmr_colors.get(hmr_type, [200, 200, 200])
        }

    # Use first HMR result for global parameters
    first_hmr = list(hmr_results.values())[0]['data']
    num_frames = first_hmr['num_frames']
    faces = first_hmr['faces']

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
        gui_show_points = server.gui.add_checkbox("Show Points", True)
        
        # ADDED: HMR model controls
        gui_hmr_controls = {}
        for hmr_type in hmr_types:
            gui_hmr_controls[hmr_type] = server.gui.add_checkbox(f"Show HMR {hmr_type.upper()}", True)

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

    # ADDED: HMR visibility callbacks
    for hmr_type in hmr_types:
        @gui_hmr_controls[hmr_type].on_update
        def _(_, ht=hmr_type):
            if ht in hmr_mesh_handles:
                current_frame = gui_timestep.value
                for i, handle in enumerate(hmr_mesh_handles[ht]):
                    handle.visible = gui_hmr_controls[ht].value and (i == current_frame)

    @gui_show_all_frames.on_update
    def _(_) -> None:
        gui_stride.disabled = not gui_show_all_frames.value

        if gui_show_all_frames.value:
            stride = gui_stride.value
            with server.atomic():
                for i, frame_node in enumerate(frame_nodes):
                    visible = (i % stride == 0)
                    frame_node.visible = visible

            gui_playing.disabled = True
            gui_timestep.disabled = True
            gui_next_frame.disabled = True
            gui_prev_frame.disabled = True

        else:
            current_timestep = gui_timestep.value
            with server.atomic():
                for i, frame_node in enumerate(frame_nodes):
                    frame_node.visible = (i == current_timestep)

            gui_playing.disabled = False
            gui_timestep.disabled = gui_playing.value
            gui_next_frame.disabled = gui_playing.value
            gui_prev_frame.disabled = gui_playing.value

    # MODIFIED: Simplified timestep update
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

    # ADDED: Load NKSR mesh if provided
    if nksr_mesh_path and os.path.exists(nksr_mesh_path):
        try:
            nksr_mesh = trimesh.load(nksr_mesh_path)
            nksr_mesh_handle = server.scene.add_mesh_trimesh(
                name="/nksr_mesh",
                mesh=nksr_mesh,
            )
            nksr_mesh_handle.visible = gui_show_nksr.value
            print(f"Loaded NKSR mesh with {len(nksr_mesh.vertices)} vertices")
        except Exception as e:
            print(f"Failed to load NKSR mesh: {e}")
    interval = 10
    # ADDED: Create HMR mesh handles for all types
    for hmr_type, hmr_result in hmr_results.items():
        hmr_data = hmr_result['data']
        color = hmr_result['color']
        pred_vert = hmr_data['pred_vert'].cpu().numpy()[::interval]
        
        handles = []
        for i in range(len(pred_vert)):
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
        print(f"Created {len(handles)} mesh handles for HMR {hmr_type}")



    try:
        single_image = bool(int(tgt_name[0]))
        single_image = False
    except (ValueError, IndexError):
        single_image = True
        if 'cam' in tgt_name:
            single_image=False


    frame_indices = []
    
    for i in tqdm(range(0, num_frames, interval)):
        frame = loader.get_frame(i)
        frame_indices.append(i)
        
        if single_image == True:
            seg_network = Vis()
            parent_folder = f'/data3/zihanwa3/_Robotics/_geo/differentiable-blocksworld/datasets/megasam/{tgt_name}'
        else:
            output_pts, real_normal, extras = frame.get_point_cloud(downsample_factor, bg_downsample_factor=1)
            position, color, bg_position, bg_color, po_all, clr_all, obj_pos, obj_clr = output_pts
            depth, rotation, translation, K, points_bg_map, _ = extras

        frame_nodes.append(server.scene.add_frame(f"/frames/t{i}", show_axes=False))

        # Add camera frustum
        fov = 2 * onp.arctan2(frame.rgb.shape[0] / 2, frame.K[0, 0])
        aspect = frame.rgb.shape[1] / frame.rgb.shape[0]
        norm_i = i / (num_frames - 1) if num_frames > 1 else 0
        color_rgba = cm.viridis(norm_i)
        color_rgb = color_rgba[:3]

        server.scene.add_camera_frustum(
            f"/frames/t{i}/frustum",
            fov=fov,
            aspect=aspect,
            scale=camera_frustum_scale,
            image=frame.rgb[::downsample_factor, ::downsample_factor],
            wxyz=tf.SO3.from_matrix(frame.T_world_camera[:3, :3]).wxyz,
            position=frame.T_world_camera[:3, 3],
            color=color_rgb,
            thickness=cam_thickness,
        )

        server.scene.add_frame(
            f"/frames/t{i}/frustum/axes",
            axes_length=camera_frustum_scale * axes_scale * 10,
            axes_radius=camera_frustum_scale * axes_scale,
        )

    for i, frame_node in enumerate(frame_nodes):
        if gui_show_all_frames.value:
            frame_node.visible = (i % gui_stride.value == 0)
        else:
            frame_node.visible = i == gui_timestep.value

    prev_timestep = gui_timestep.value

    print('Loading complete!')
    print(f"Loaded {len(hmr_mesh_handles)} HMR models: {list(hmr_mesh_handles.keys())}")
    if nksr_mesh_handle:
        print("Loaded NKSR mesh")

    if save_mode:
        server.stop()
    else:
        while True:
            if gui_playing.value and not gui_show_all_frames.value:
                gui_timestep.value = (gui_timestep.value + 1) % len(frame_nodes)

            time.sleep(1.0 / gui_framerate.value)

if __name__ == "__main__":
    tyro.cli(main)