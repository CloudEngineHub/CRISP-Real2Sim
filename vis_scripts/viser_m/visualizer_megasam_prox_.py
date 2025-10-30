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
from prox_utils import *
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

    # 只有一个 link，名字随意；这里叫 “scene_link”
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


def load_contact_sequence(interact_contact_path, start_frame=0, end_frame=-1, thre=0.5):
    """加载一段连续帧的contact数据"""
    contact_sequence = []
    for i in range(start_frame, end_frame + 1):
        contact_data = np.load(os.path.join(interact_contact_path, f'{i:05d}.npz'))['pred_contact_3d_smplh']
        contact_sequence.append(contact_data > thre)  # 转换为布尔mask
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
    hmr_type: str = 'gv',
    moge_base_path: str = '/data3/zihanwa3/_Robotics/_vision/TAPIP3D/_raw_mega_priors' # '/data3/zihanwa3/_Robotics/_vision/TAPIP3D/outputs'
) -> None:
    from pathlib import Path  # <-- Import Path here if not already imported
    tgt_name = str(data).split('_sgd')[0].split('/')[-1]   # gives 'MPH112_00169_01_tram'
    tgt_name = "_".join(tgt_name.split("_")[:-1]) 
    moge_data = os.path.join(moge_base_path, f'{tgt_name}.npz')
    # compare_npz_shapes(data, moge_data)
    tgt_folder = os.path.join('/data3/zihanwa3/_Robotics/_geo/differentiable-blocksworld/post_results', tgt_name)
    # if Path(tgt_folder).exists():
    #   shutil.rmtree(Path(tgt_folder))


    if 'door' in str(data):
        extra_obj = True
    
    # data = np.load(data)
    data = _load_npz_to_dict(data)
    moge_data = np.load(moge_data)
    data["depths"] = moge_data["depths"]
    data["images"] = moge_data["images"]
    data['cam_c2w'] = moge_data['cam_c2w']
    data['intrinsic'] = moge_data['intrinsic']
    print(f"scale is {data['scale']}")
    # data['scale'] = 1 3.184327021929371
    # print(moge_data['cam_c2w'][0], data['cam_c2w'][0])


    import random

    def generate_four_digit():
        return random.randint(1000, 9999)
    server = viser.ViserServer(port=generate_four_digit())
    # serializer = server.get_scene_serializer()
    if share:
        server.request_share_url()

    ratio_block_scene =1 
    do_mesh = True
    use_world = True
    server.scene.set_up_direction('-z')
    if no_mask or not do_mesh:             # not using dynamic / static mask
        init_conf = True    # must use init_conf map, to avoid depth cleaning
        fg_conf_thre = conf_threshold # now fg_conf_thre is the same as conf_thre
    print("Loading frames!")
    
    key_R, key_T =  'world_cam_R', 'world_cam_T' 

    num_frames = len(data['depths'])# min(max_frames, len(data['depths'])) 
    # num_frames = max_frames

    npz_cam_data = data# pred_cam
    device='cuda'
    base_folder = '/data3/zihanwa3/_Robotics/_vision/tram/megasamra'# 449_resize/camera.npy
    base   = Path(base_folder)
    candid = sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith(tgt_name))

    if not candid:
        raise FileNotFoundError(f"No sub-directory in {base} starts with “{tgt_name}”.")
    if len(candid) > 1:
        print(f"[warn] multiple matches, picking {candid[0].name!r}")

    camera = np.load(candid[0] / "camera.npy", allow_pickle=True).item()
    fx= fy = img_focal = camera['img_focal']
    pred_cam ={}
    pred_cam[key_R] = npz_cam_data['cam_c2w'][:, :3, :3]
    pred_cam[key_T] = (npz_cam_data['cam_c2w'][:, :3, 3] )* npz_cam_data['scale']
    world_cam_R = torch.tensor(pred_cam[key_R]).to(device)#[:num_frames]
    world_cam_T = torch.tensor(pred_cam[key_T]).to(device)#[:num_frames]

    from smpl_utils import (
        process_tram_smpl,
        process_gv_smpl,
        load_contact_ids_from_file,
        load_contact_ids_with_mode, 
        filter_vertices_by_contact,
        vis_hmr,
        analyze_motion,
        filter_stable_contacts_simple,
        inspect_contact_points, 
        pick_best_frames_per_segment
    )
    

    device='cuda'
    smpl = SMPL().to(device)
    interact_contact_path = os.path.join('/data3/zihanwa3/_Robotics/_data/_contact', tgt_name) 


    
    if hmr_type == 'tram':
        smpl_results = process_tram_smpl(
            tgt_name=tgt_name,
            world_cam_R=world_cam_R,
            world_cam_T=world_cam_T,
            max_frames=max_frames,
            smpl_model=smpl,
            device='cuda'
        )
        
        num_frames = smpl_results['num_frames']
        global_orient_world = smpl_results['global_orient_world']
        transl_world = smpl_results['transl_world']
        pred_vert = smpl_results['pred_vert'].cpu().numpy()
        pred_j3dg = smpl_results['pred_j3dg']
        body_pose = smpl_results['body_pose']
        pred_shapes = smpl_results['pred_shapes']
        faces = smpl_results['faces']
        # CONTACT_IDS_SMPL = load_contact_ids_from_file()
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

        pred_contact_vert_leg     = pred_vert[:, leg_ids_smpl, :]
        pred_contact_vert_hand    = pred_vert[:, hand_ids_smpl, :]
        pred_contact_vert_gluteus = pred_vert[:, gluteus_ids_smpl, :]
        pred_contact_vert_back    = pred_vert[:, back_ids_smpl, :]
        pred_contact_vert_thigh   = pred_vert[:, thigh_ids_smpl, :]


        pred_contact_vert_list = [
            pred_contact_vert_leg,
            pred_contact_vert_hand, 
            pred_contact_vert_gluteus,
            pred_contact_vert_back,
            pred_contact_vert_thigh
        ]

        contact_colors_rgb = [
            [0, 255,   0],  # leg - green
            [255,   0,   0],  # hand - red
            [255, 255,   0],  # gluteus - yellow
            [0,   0,   0],  # back - black
            [255,   0, 255]   # thigh - magenta
        ]



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
        
        num_frames = smpl_results['num_frames']
        global_orient_world = smpl_results['global_orient_world']
        transl_world = smpl_results['transl_world']
        pred_vert = smpl_results['pred_vert'].cpu().numpy()
        pred_j3dg = smpl_results['pred_j3dg']
        body_pose = smpl_results['body_pose']
        pred_shapes = smpl_results['pred_shapes']
        faces = smpl_results['faces']

    
    save_dir = '/data3/zihanwa3/_Robotics/_geo/differentiable-blocksworld/post_results'
    save_dir = os.path.join(save_dir, tgt_name)
    os.makedirs(save_dir, exist_ok=True)
    save_dir = os.path.join(save_dir, hmr_type)
    os.makedirs(save_dir, exist_ok=True)
    hmr_dir = os.path.join(save_dir, 'hmr')
    os.makedirs(hmr_dir, exist_ok=True)

    
    results = {'pred_cam': [world_cam_R, world_cam_T], # cam 
                'body_pose': body_pose, # smpl 
                'global_orient': global_orient_world, # smpl
                'betas': pred_shapes, # smpl 
                'transl': transl_world, # smpl 
                'pose2rot': False, 
                'default_smpl': True
              }

    org_vis = os.path.join(hmr_dir, 'org_vis')
    os.makedirs(org_vis, exist_ok=True)
    
    vis_hmr(results, org_vis, device, every=20)
    np.save(os.path.join(hmr_dir ,'hps_track.npy'), results)
    human_transl_np = transl_world.detach().cpu().numpy()


    loader = viser.extras.Record3dLoader_Customized_Megasam(
        data,
        npz_cam_data, 
        conf_threshold=1.0,
        foreground_conf_threshold=foreground_conf_threshold,
        no_mask=no_mask,
        xyzw=xyzw,
        init_conf=init_conf,
    )

    sq_point_handles: list[viser.PointHandle] = []       # 新建列表
    sq_mesh_handles:  list[viser.MeshHandle]  = []       # 复用/替换你上面已有的


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


        gui_next_sqs = server.gui.add_button("Next SQs", disabled=True)
        gui_prev_sqs = server.gui.add_button("Prev SQs", disabled=True)
        gui_sqs_pt = server.gui.add_checkbox("SQs2pt", True)

        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=loader.fps
        )
        gui_framerate_options = server.gui.add_button_group(
            "FPS options", ("10", "20", "30", "60")
        )
        gui_show_all_frames = server.gui.add_checkbox("Show all frames", False)
        gui_show_human_mesh = server.gui.add_checkbox("Show human mesh", False)
        gui_stride = server.gui.add_slider(
            "Stride",
            min=1,
            max=num_frames,
            step=1,
            initial_value=1,
            disabled=True,
        )
        gui_depth_scale = server.gui.add_slider(
            "Stride",
            min=0,
            max=20,
            step=0.1,
            initial_value=1,
            disabled=True,
        )





    # Add GUI controls
    with server.gui.add_folder("Layers"):
        gui_show_contact = server.gui.add_checkbox("Show Contact", True)
        gui_show_sqs = server.gui.add_checkbox("SQs", True)
        gui_show_scene_raw = server.gui.add_checkbox("Scene mesh (raw)", True)
        gui_show_scene_coacd_contact = server.gui.add_checkbox("Scene mesh (contact-COACD)", True)
        gui_show_scene_coacd = server.gui.add_checkbox("Scene mesh (COACD)", True)
        gui_show_points = server.gui.add_checkbox("All points", True)
        gui_show_humans = server.gui.add_checkbox("HMR", True)
        
        # Add point cloud display options
        gui_pc_display_mode = server.gui.add_dropdown(
            "Point Cloud Display",
            options=["Filtered", "Original", "Mono"],
            initial_value="Filtered"
        )

        # ── SQ / PointCloud 显示模式 ──────────────────────────────────────────
        gui_sq_mode = server.gui.add_dropdown(
            "SQ Display Mode",
            options=["SQs + Points", "Points Only"],
            initial_value="SQs + Points",
        )


    # Visibility callbacks
    @gui_show_sqs.on_update
    def _(_):
        with server.atomic():
            for s in sq_mesh_handles:
                s.visible = gui_show_sqs.value


    # ── 统一切换 SQ 网格和“按 SQ 划分的点云” ─────────────────────────────
    @gui_sq_mode.on_update
    def _(_):
        show_sq = (gui_sq_mode.value == "SQs + Points")
        with server.atomic():
            # SQ mesh（红色几何体）
            for mh in sq_mesh_handles:
                mh.visible = show_sq and gui_show_sqs.value   # 仍受原“SQs”总开关控制
            # 属于每个 SQ 的点云（青色）
            for ph in sq_point_handles:
                ph.visible = gui_show_points.value            # 点云始终跟 All points 开关


    @gui_show_scene_raw.on_update
    def _(_):
        if scene_mesh_raw_handle is not None:
            scene_mesh_raw_handle.visible = gui_show_scene_raw.value

    @gui_show_scene_coacd.on_update
    def _(_):
        if scene_mesh_coacd_handle is not None:
            scene_mesh_coacd_handle.visible = gui_show_scene_coacd.value

    @gui_show_scene_coacd_contact.on_update
    def _(_):
        if scene_mesh_coacd_contact_handle is not None:
            scene_mesh_coacd_contact_handle.visible = gui_show_scene_coacd_contact.value

    @gui_show_humans.on_update
    def _(_):
        with server.atomic():
            for h in human_mesh_handles:
                h.visible = gui_show_humans.value

    @gui_show_contact.on_update
    def _(_):
        with server.atomic():
            for h in contact_vertices_handles:
                h.visible = gui_show_contact.value

    @gui_show_points.on_update
    def _(_):
        stride = gui_stride.value
        with server.atomic():
            # 1️⃣ 帧点云
            for i, frame_node in enumerate(frame_nodes):
                frame_node.visible = gui_show_points.value
            # 2️⃣ SQ 点云
            for ph in sq_point_handles:
                ph.visible = gui_show_points.value


    # Point cloud display mode callback
    @gui_pc_display_mode.on_update
    def _(_):
        mode = gui_pc_display_mode.value
        with server.atomic():
            # Update visibility of point cloud handles based on selected mode
            for i, handles in enumerate(point_cloud_handles_dict['filtered']):
                handles.visible = (mode == "Filtered") and gui_show_points.value and (
                    (i == gui_timestep.value and not gui_show_all_frames.value) or 
                    (i % gui_stride.value == 0 and gui_show_all_frames.value)
                )
            
            for i, handles in enumerate(point_cloud_handles_dict['original']):
                handles.visible = (mode == "Original") and gui_show_points.value and (
                    (i == gui_timestep.value and not gui_show_all_frames.value) or 
                    (i % gui_stride.value == 0 and gui_show_all_frames.value)
                )
            
            for i, handles in enumerate(point_cloud_handles_dict['mono']):
                if handles is not None:
                    handles.visible = (mode == "Mono") and gui_show_points.value and (
                        (i == gui_timestep.value and not gui_show_all_frames.value) or 
                        (i % gui_stride.value == 0 and gui_show_all_frames.value)
                    )






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

    # Toggle frame visibility when the timestep slider changes.
# Also update the @gui_timestep.on_update callback to handle the frame index mapping:

    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        if not gui_show_all_frames.value:
            with server.atomic():
                # Map timestep to actual frame index in our data
                current_handle_idx = current_timestep // interval if current_timestep % interval == 0 else -1
                prev_handle_idx = prev_timestep // interval if prev_timestep % interval == 0 else -1
                
                # Update frame visibility
                if current_handle_idx >= 0 and current_handle_idx < len(frame_nodes):
                    frame_nodes[current_handle_idx].visible = True
                if prev_handle_idx >= 0 and prev_handle_idx < len(frame_nodes):
                    frame_nodes[prev_handle_idx].visible = False
                
                # Update point cloud visibility based on selected mode
                mode = gui_pc_display_mode.value.lower()
                if gui_show_points.value:
                    # Hide all point clouds for previous frame
                    if prev_handle_idx >= 0:
                        for pc_type in ['filtered', 'original', 'mono']:
                            if prev_handle_idx < len(point_cloud_handles_dict[pc_type]):
                                handle = point_cloud_handles_dict[pc_type][prev_handle_idx]
                                if handle is not None:
                                    handle.visible = False
                    
                    # Show the appropriate point cloud for current frame
                    if current_handle_idx >= 0:
                        if mode == "filtered" and current_handle_idx < len(point_cloud_handles_dict['filtered']):
                            point_cloud_handles_dict['filtered'][current_handle_idx].visible = True
                        elif mode == "original" and current_handle_idx < len(point_cloud_handles_dict['original']):
                            point_cloud_handles_dict['original'][current_handle_idx].visible = True
                        elif mode == "mono" and current_handle_idx < len(point_cloud_handles_dict['mono']):
                            if point_cloud_handles_dict['mono'][current_handle_idx] is not None:
                                point_cloud_handles_dict['mono'][current_handle_idx].visible = True
                                
        prev_timestep = current_timestep
        server.flush()

    @gui_show_all_frames.on_update
    def _(_) -> None:
        gui_stride.disabled = not gui_show_all_frames.value  # Enable/disable stride slider

        if gui_show_all_frames.value:
            # Show frames with stride
            stride = gui_stride.value
            mode = gui_pc_display_mode.value.lower()  # "filtered", "original", or "mono"

            with server.atomic():
                for idx, frame_node in enumerate(frame_nodes):
                    # Determine if this frame should be visible based on its actual index in frame_indices
                    actual_frame = frame_indices[idx] if idx < len(frame_indices) else idx * interval
                    visible = (actual_frame % stride == 0)
                    frame_node.visible = visible

                    # Handle point cloud visibility for the selected mode
                    for pc_type in ['filtered', 'original', 'mono']:
                        if idx < len(point_cloud_handles_dict[pc_type]):
                            handle = point_cloud_handles_dict[pc_type][idx]
                            if handle is not None:
                                handle.visible = gui_show_points.value and (pc_type == mode) and visible

            # Disable playback controls
            gui_playing.disabled = True
            gui_timestep.disabled = True
            gui_next_frame.disabled = True
            gui_prev_frame.disabled = True

        else:
            current_timestep = gui_timestep.value
            mode = gui_pc_display_mode.value.lower()

            with server.atomic():
                # Find which handle index corresponds to current timestep
                handle_idx = -1
                for idx, frame_idx in enumerate(frame_indices):
                    if frame_idx == current_timestep:
                        handle_idx = idx
                        break
                
                # Update frame visibility
                for idx, frame_node in enumerate(frame_nodes):
                    frame_node.visible = (idx == handle_idx)

                # Update point cloud visibility
                for pc_type in ['filtered', 'original', 'mono']:
                    for idx, handle in enumerate(point_cloud_handles_dict[pc_type]):
                        if handle is not None:
                            handle.visible = (
                                gui_show_points.value and 
                                (pc_type == mode) and 
                                (idx == handle_idx)
                            )

            # Re-enable playback controls
            gui_playing.disabled = False
            gui_timestep.disabled = gui_playing.value
            gui_next_frame.disabled = gui_playing.value
            gui_prev_frame.disabled = gui_playing.value

    def sqs_params_2_mesh(sqs, lat_res=32, lon_res=64, combine=True, device='cuda'):
        """
        Convert superquadric parameters to a TriMesh using the provided utility functions.
        
        Args:
            sqs: numpy array of shape (N, 11) where each row contains:
                [eps1, eps2, scale_x, scale_y, scale_z, euler_z, euler_y, euler_x, tx, ty, tz]
            lat_res: latitude resolution for mesh generation (default: 32)
            lon_res: longitude resolution for mesh generation (default: 64)
            combine: if True, combine all superquadrics into a single mesh
                    if False, return a list of individual meshes
            device: torch device to use for computation
        
        Returns:
            trimesh.Trimesh object (if combine=True) or list of trimesh.Trimesh objects
        """
        
        if isinstance(sqs, np.ndarray):
            sqs = torch.from_numpy(sqs).float()
        
        if sqs.shape[0] == 0:
            # Return empty mesh if no superquadrics
            return trimesh.Trimesh(vertices=np.array([]), faces=np.array([]))
        
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        sqs = sqs.to(device)
        
        meshes = []
        
        # Get base icosphere for mesh generation
        base_mesh = get_icosphere(level=2)  # Higher level for better resolution
        base_verts, base_faces = base_mesh.get_mesh_verts_faces(0)
        base_verts = base_verts.to(device)
        base_faces = base_faces.to(device)
        
        # Convert vertices to spherical coordinates
        eta = torch.asin(base_verts[..., 1].clamp(-1, 1))
        omega = torch.atan2(base_verts[..., 0], base_verts[..., 2])
        
        for i in range(sqs.shape[0]):
            try:
                # Extract parameters
                eps1 = sqs[i, 0].clamp(0.1, 2.0)  # Clamp to valid range
                eps2 = sqs[i, 1].clamp(0.1, 2.0)
                scale = sqs[i, 2:5]
                euler = sqs[i, 5:8]
                translation = sqs[i, 8:11]
                
                # Generate superquadric vertices using parametric equation
                verts = parametric_sq(eta, omega, eps1.unsqueeze(0), eps2.unsqueeze(0))
                verts = verts.squeeze(0)  # Remove batch dimension
                
                # Apply scale
                verts = verts * scale
                
                # Apply rotation (Euler ZYX convention)
                R = euler_angles_to_matrix(euler.unsqueeze(0), convention="ZYX").squeeze(0)
                verts = verts @ R.T
                
                # Apply translation
                verts = verts + translation
                
                # Convert to numpy for trimesh
                verts_np = verts.cpu().numpy()
                faces_np = base_faces.cpu().numpy()
                
                # Create trimesh object
                mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
                
                # Fix normals to point outward
                mesh.fix_normals()
                
                # Validate mesh
                if len(mesh.vertices) > 0 and len(mesh.faces) > 0:
                    meshes.append(mesh)
                
            except Exception as e:
                print(f"Warning: Failed to create superquadric {i}: {e}")
                continue
        
        if len(meshes) == 0:
            # Return empty mesh if no valid superquadrics were created
            return trimesh.Trimesh(vertices=np.array([]), faces=np.array([]))
        
        if combine:
            # Combine all meshes into a single scene mesh
            combined_mesh = trimesh.util.concatenate(meshes)
            # Clean up the combined mesh
            combined_mesh.remove_duplicate_faces()
            combined_mesh.remove_degenerate_faces()
            return combined_mesh
        else:
            # Return list of individual meshes
            return meshes




    # Update frame visibility when the stride changes.
    @gui_stride.on_update
    def _(_) -> None:
        if gui_show_all_frames.value:
            # Update frame visibility based on new stride
            stride = gui_stride.value
            with server.atomic():
                for i, frame_node in enumerate(frame_nodes):
                    frame_node.visible = (i % stride == 0)



    # Load in frames.ss
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
    
    R_ROOT = tf.SO3.exp(onp.array([onp.pi / 2.0, 0.0, 0.0])).as_matrix()  # 3×3
    T_ROOT4 = np.eye(4, dtype=float)                                      # 4×4 齐次
    #T_ROOT4[:3, :3] = R_ROOT

    frame_nodes: list[viser.FrameHandle] = []
    all_positions = []
    all_colors = []

    mono_pcs = []
    bg_org_positions=[]
    bg_org_colors = []

    bg_positions = []
    bg_colors = []
    positions=[]
    colors=[]


    vdb_volume = vdbfusion.VDBVolume(voxel_size=0.05,
                        sdf_trunc=0.15, space_carving=True)
    point_cloud_handles_dict = {
        'filtered': [],
        'original': [],
        'mono': []
    }

    if num_frames > 200:
        stttride = 3
    elif num_frames >=100:
        stttride = 2
        #  // 2
    stttride= (num_frames // 90) + 1
    

    #stttride = 1
    inspect_everything = True

    moge_mesh=False
    TTTTTTEST=False
    show_moge_priors = False
    mono_cleaner = False

    human_mesh_handles: list[viser.MeshHandle] = []  
    contact_vertices_handles: list[viser.MeshHandle] = []  
    num_projections = 10
    source_idx = 0


    rgbs = []
    normals = []
    mono_normals = []
    existing_sqs = []
    exisitng_points = []
    sqs_mesh_handles: list[viser.MeshHandle] = []  
    mono_pc = None
    depths = []
    pointclouds_hmr_body = []
    rotations = []
    translations = []
    mono_pc_clrs = []
    per_frame_sq = []
    per_sq_one_list = []

    points_bg_map_nksr = []
    points_normal_nksr = []
    
    all_frame_mode = False
    results = {
        'S_items': [],
        'R_items': [],
        'T_items': [],
        'pts_items': []
    }

    try:
        single_image = (int(tgt_name[0]))
        single_image = False


        
        
    except (ValueError, IndexError):

        single_image = True
        if 'cam' in tgt_name:
            single_image=False

    # single_image = False
    # single_image = bool(int(tgt_name[0]))
    if 'qitao' in tgt_name:
      single_image = False 

    #[depth, rotation, translation, K]
    print(single_image, 'singleimg')
    debug = False 
    try: 
        if not transfer_data:
            contacted_masks = load_contact_sequence(interact_contact_path, end_frame=num_frames-1, thre=0.4)
            contacted_masks = filter_stable_contacts_simple(contacted_masks[:, 0, :])
            inspect_contact_points(contacted_masks)
            static_frames, static_segments = analyze_motion(pred_vert, visualize=True)
            best_frames, counts = pick_best_frames_per_segment(
                contacted_masks=contacted_masks,
                static_segments=static_segments,   # 例如 [(257,284), (342,416), (716,742)]
                return_vertices=False              # 如需接触顶点，加 True
            )



            print(f"[ALL] best frame = {best_frames}")
    except:
        asfasfa =2
        contact_points = None 
    if TTTTTTEST == False:
        interval = 7 # 7# 30#stttride=  (num_frames // 90) + 1
        frame_indices = []  # Track which frames we're processing
        if debug:
          num_frames = interval+1

        times_list = list(range(0, num_frames, interval))
        for i in tqdm(times_list):
        # for i in tqdm(range(0, 1)):
            frame = loader.get_frame(i)
            frame_indices.append(i)
            mono_pc = None
            
            if single_image == True:
                seg_network = Vis()
                parent_folder = f'/data3/zihanwa3/_Robotics/_geo/differentiable-blocksworld/datasets/megasam/{tgt_name}'
                if len(mono_normals)==0:
                    mono_pc, mono_normal, rgb, colors__, extras = frame.get_mono_data(i, seg_network, 
                    parent_folder, single_image=single_image)
                    
                    depth, rotation, translation, K, points_bg_map, _ = extras
                    distance_filtering=True
                    # real_normal
                    if distance_filtering:
                      points_bg_map_, _, depth = filter_bg_points_by_human_distance(points_bg_map, points_bg_map, human_transl_np, depth, max_dist=2.7)
                    else:
                      _, _, depth, _ = filter_bg_points_by_human_distance(points_bg_map, points_bg_map, human_transl_np, depth, max_dist=1)

                    bg_position, bg_color = mono_pc, colors__
                    _, T_world_camera = frame.get_sdf(bg_position, downsample_factor, bg_downsample_factor, vdb_volume=vdb_volume)
                    vdb_volume.integrate(points_bg_map_.astype(np.float64), T_world_camera)

                    bg_color=bg_color.reshape(-1, 3)
                    mono_normals.append(mono_normal)
                    rgbs.append(rgb)
                    bg_positions.append(mono_pc)  # Store mono point cloud
                    bg_colors.append(rgb)
                    depths.append(depth)
                    bg_org_positions = bg_positions
                    bg_org_colors = bg_colors
                    rotations.append(rotation)
                    translations.append(translation)
                    all_positions.append(points_bg_map)



            else:

                # Get original point cloud data
                # human_transl_np
                output_pts, real_normal, extras = frame.get_point_cloud(downsample_factor, bg_downsample_factor=1)
                position, color, bg_position, bg_color, po_all, clr_all, obj_pos, obj_clr = output_pts
                
                # Store original background positions and colors
                bg_org_positions.append(bg_position.copy())
                bg_org_colors.append(bg_color.copy())
                '''if mono_cleaner:
                    output_pts, real_normal, extras = frame.get_filtered_point_cloud(downsample_factor, bg_downsample_factor=1, mono_normal=mono_normal)
                else:
                    _, _, extras = frame.get_filtered_point_cloud(downsample_factor, bg_downsample_factor=1)
                
                '''
                depth, rotation, translation, K, points_bg_map, _ = extras
                distance_filtering=True
                if distance_filtering:
                  points_bg_map_, _, depth, real_normal_ = filter_bg_points_by_human_distance(points_bg_map, points_bg_map, human_transl_np, depth, real_normal, max_dist=2.7)
                else:
                  _, _, depth, _ = filter_bg_points_by_human_distance(points_bg_map, points_bg_map, human_transl_np, depth, max_dist=1)
                points_bg_map_nksr.append(points_bg_map_)
                points_normal_nksr.append(real_normal_)

                
                position, color, bg_position, bg_color, _, clr_all, obj_pos, obj_clr = output_pts
                if do_mesh:
                    faces, vertices, vertex_colors, vertex_uvs, tri = frame.get_meshes(po_all, downsample_factor, bg_downsample_factor)
                    _, T_world_camera = frame.get_sdf(bg_position, downsample_factor, bg_downsample_factor, vdb_volume=vdb_volume)
                vdb_volume.integrate(points_bg_map_.astype(np.float64), T_world_camera)

                mono_normals.append(real_normal)
                depths.append(depth)
                rotations.append(rotation)
                translations.append(translation)
                bg_positions.append(bg_position)
                bg_colors.append(bg_color)
                positions.append(position)
                colors.append(color)
                all_positions.append(points_bg_map)
                all_colors.append(clr_all)

                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]         # NOT K[0,1]
                cy = K[1, 2]         # NOT K[1,-1]

            inferring_scene=False
            if inferring_scene:
                for pred_contact_vert_g, contact_rgb in zip(pred_contact_vert_list, contact_colors_rgb):
                    pred_contact_vert_t = pred_contact_vert_g[i]  # shape: (num_contact_verts, 3)
                    
                    # Find bg_position points within small range of contact vertices
                    threshold = 0.05  # adjust this threshold as needed
                    
                    # Compute distances between all bg_position points and contact vertices
                    distances = np.linalg.norm(bg_position[:, None, :] - pred_contact_vert_t[None, :, :], axis=2)
                    
                    # Find bg_position points that are within threshold of any contact vertex
                    min_distances = np.min(distances, axis=1)  # shape: (N,)
                    nearby_mask = min_distances < threshold
                    
                    if np.any(nearby_mask):
                        # Get the nearby bg_position points
                        nearby_points = bg_position[nearby_mask]
                        
                        # Create colors array for these points (all same color for this contact group)
                        num_nearby = nearby_points.shape[0]
                        colors_contact = np.tile(contact_rgb, (num_nearby, 1))  # shape: (num_nearby, 3)
                        
                        contact_vertices = server.scene.add_point_cloud(
                            name=f"/frames/t{i}/point_cloud_contact_{len(contact_vertices_handles)}",
                            points=nearby_points,
                            colors=colors_contact,
                            point_size=0.02,
                            point_shape="rounded",
                        )
                        contact_vertices_handles.append(contact_vertices)
                        pointclouds_hmr_body.append(pred_contact_vert_t)
                        smplllll=False
                        if smplllll:
                            contact_vertices = server.scene.add_point_cloud(
                                name=f"/frames/t{i}/point_cloud_contact_{len(contact_vertices_handles)}",
                                points=pred_contact_vert_t,
                                colors=np.tile(contact_rgb, (len(pred_contact_vert_t), 1)),
                                point_size=0.02,
                                point_shape="rounded",
                            )
                            contact_vertices_handles.append(contact_vertices)


            if not transfer_data:
                '''
                leg_ids_smpl     = convert_indices_smplx_to_smpl(leg_ids,     x2s)
                hand_ids_smpl    = convert_indices_smplx_to_smpl(hand_ids,    x2s)
                gluteus_ids_smpl = convert_indices_smplx_to_smpl(gluteus_ids, x2s)
                back_ids_smpl    = convert_indices_smplx_to_smpl(back_ids,    x2s)
                thigh_ids_smpl   = convert_indices_smplx_to_smpl(thigh_ids,   x2s)

                pred_contact_vert_leg     = pred_vert[:, leg_ids_smpl, :]
                pred_contact_vert_hand    = pred_vert[:, hand_ids_smpl, :]
                pred_contact_vert_gluteus = pred_vert[:, gluteus_ids_smpl, :]
                pred_contact_vert_back    = pred_vert[:, back_ids_smpl, :]
                pred_contact_vert_thigh   = pred_vert[:, thigh_ids_smpl, :]


                pred_contact_vert_list = [
                    pred_contact_vert_leg,
                    pred_contact_vert_hand, 
                    pred_contact_vert_gluteus,
                    pred_contact_vert_back,
                    pred_contact_vert_thigh
                ]

                contact_colors_rgb = [
                    [0, 255,   0],  # leg - green
                    [255,   0,   0],  # hand - red
                    [255, 255,   0],  # gluteus - yellow
                    [0,   0,   0],  # back - black
                    [255,   0, 255]   # thigh - magenta
                ]
                '''
                # interact_contact_ti = np.load(os.path.join(interact_contact_path, f'{i:05d}.npz'))['pred_contact_3d_smplh']
                # contacted_mask = interact_contact_ti > 1e-7
                frame_nodes.append(server.scene.add_frame(f"/frames/t{i}", show_axes=False))

                # best_frames = best_frames[276, 382, 724] 
                # best_frames = list(range(-56, 0))
                # if single_image:
                #  best_frames = [535]
                #  = None 
                for contact_fram in best_frames:
                    contacted_mask = contacted_masks[contact_fram][None, ...]
                    # print(contacted_mask.shape,  pred_vert[[contact_fram]].shape)
                    human_scene_contacing_pc = pred_vert[[contact_fram]][contacted_mask]
                    ### all [1. 6890, ]
                    ### 1. save to 
                    contact_pc_base = '/data3/zihanwa3/_Robotics/_geo/differentiable-blocksworld/cache_contact'
                    os.makedirs(contact_pc_base, exist_ok=True)
                    contact_pc_filtered = server.scene.add_point_cloud(
                        name=f"/frames/t{contact_fram}/point_cloud_filtered_contact",
                        points=human_scene_contacing_pc,
                        colors=[0, 255,   0],
                        point_size=0.028,
                        point_shape="rounded",
                    )
                    contact_points = human_scene_contacing_pc
                    save_point_cloud_ply(
                      path=f"/data3/zihanwa3/_Robotics/_vision/mega-sam/post_results/{tgt_name}/contact_points.ply",
                      points=contact_points,   
                      ascii=False           # 大点云推荐 False（二进制）
                      ) 
                      

                    contact_vertices_handles.append(contact_pc_filtered)

                # point_cloud_handles_dict['filtered'].append(contact_pc_filtered)
            #except:
            #   continue
            # if i  // interval in [0, 10, 40]:
            human_mesh_handle = server.scene.add_mesh_simple(
                name=f"/frames/t{i}/human_mesh",
                vertices=pred_vert[[i]],
                faces=smpl.faces,
                flat_shading=False,
                wireframe=False,
                color=[229, 80, 80],
            )
            
            human_mesh_handles.append(human_mesh_handle)
            human_mesh_handle.visible=True

            
            pc_filtered = server.scene.add_point_cloud(
                name=f"/frames/t{i}/point_cloud_filtered",
                points=bg_position,
                colors=bg_color,
                point_size=point_size,
                point_shape="rounded",
            )
            if single_image:
              break
              save_point_cloud_ply(
                path=f"/data3/zihanwa3/_Robotics/_vision/mega-sam/post_results/{tgt_name}/points.ply",
                points=bg_position,
                colors=bg_color,      
                ascii=False           # 大点云推荐 False（二进制）
               ) 

            point_cloud_handles_dict['filtered'].append(pc_filtered)
            
            '''pc_original = server.scene.add_point_cloud(
                name=f"/frames/t{i}/point_cloud_original",
                points=bg_org_positions[-1],
                colors=bg_org_colors[-1],
                point_size=point_size,
                point_shape="rounded",
            )
            point_cloud_handles_dict['original'].append(pc_original)
            pc_original.visible = False  #

            if mono_pc is not None:
                pc_mono = server.scene.add_point_cloud(
                    name=f"/frames/t{i}/point_cloud_mono",
                    points=mono_pc,
                    colors = colors__,
                    point_size=point_size,
                    point_shape="rounded",
                )
                point_cloud_handles_dict['mono'].append(pc_mono)
                pc_mono.visible = False 
            else:
                point_cloud_handles_dict['mono'].append(None)'''
            norm_i = i / (num_frames - 1) if 90 > 1 else 0  # Normalize index to [0, 1]
            color_rgba = cm.viridis(norm_i)  # Get RGBA color from colormap
            color_rgb = color_rgba[:3]  # Use RGB components

            fov = 2 * onp.arctan2(frame.rgb.shape[0] / 2, frame.K[0, 0])
            aspect = frame.rgb.shape[1] / frame.rgb.shape[0]
            server.scene.add_camera_frustum(
                f"/frames/t{i}/frustum",
                fov=fov,
                aspect=aspect,
                scale=camera_frustum_scale,
                image=frame.rgb[::downsample_factor, ::downsample_factor],
                wxyz=tf.SO3.from_matrix(frame.T_world_camera[:3, :3]).wxyz,
                position=frame.T_world_camera[:3, 3],
                color=color_rgb,  # Set the color for the frustum
                thickness=cam_thickness,
            )

            # Add some axes.
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

        # Add background frame.

        
        bg_positions = onp.concatenate(bg_positions, axis=0)
        bg_colors = onp.concatenate(bg_colors, axis=0)

        depthmaps = np.array(depths)
        R_cam = np.array(rotations)
        T_cam = np.array(translations)

        depthmaps = torch.tensor(np.array(depths), device=device, dtype=torch.float32)
        R_cam = torch.tensor(np.array(rotations), device=device, dtype=torch.float32)
        T_cam = torch.tensor(np.array(translations), device=device, dtype=torch.float32)
        mono_normals = torch.tensor(np.array(mono_normals), device=device, dtype=torch.float32)
        pointclouds = torch.tensor(np.array(all_positions), device=device, dtype=torch.float32)



        '''if not transfer_data:
            hmr_tensors = [torch.tensor(pc, device=device, dtype=torch.float32) for pc in pointclouds_hmr_body]
            pointclouds_hmr_body = torch.cat(hmr_tensors, dim=0)
            pointclouds = torch.cat([pointclouds, pointclouds_hmr_body], dim=0)'''



        def save_bg_normals(tgt_name: str,
                            bg_positions,
                            normals,
                            out_dir: str = "cache",
                            fmt: str = "pt") -> str:
            """
            """
            out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

            # 保证是 float32 Tensor（无论传进来是 list/np/torch）
            bg_positions = torch.as_tensor(bg_positions, dtype=torch.float32)
            normals      = torch.as_tensor(normals,      dtype=torch.float32)

            if fmt == "pt":
                path = out / f"{tgt_name}.pt"
                torch.save({"bg_positions": bg_positions.cpu(),
                            "normals": normals.cpu()}, path)
            elif fmt == "npz":
                path = out / f"{tgt_name}.npz"
                np.savez_compressed(path,
                                    bg_positions=bg_positions.cpu().numpy(),
                                    normals=normals.cpu().numpy())
            else:
                raise ValueError("fmt must be 'pt' or 'npz'")
            return str(path)

        try:
          normals_tensor_nksr = torch.from_numpy(np.concatenate(points_normal_nksr, axis=0))
          pointclouds_tensor_nksr = torch.from_numpy(np.concatenate(points_bg_map_nksr, axis=0))

          save_bg_normals(tgt_name, pointclouds_tensor_nksr, normals_tensor_nksr, fmt="pt")
        except:
          fasfas =2 


        # Handle mono_normals conversion properly
        if isinstance(mono_normals, list):
            if isinstance(mono_normals[0], torch.Tensor):
                mono_normals = torch.stack(mono_normals).to(device)
            else:
                mono_normals = torch.tensor(np.array(mono_normals), device=device, dtype=torch.float32)
        else:
            mono_normals = mono_normals.to(device)
        

        root = Path('/data3/zihanwa3/_Robotics/_data/_flows_covisibility')
        data_dir = root / Path(tgt_name)
        debug_dir = Path('./')


        all_frame_mode = False
        print(len(mono_normals))


        try:
          a = contact_points
        except: 
          contact_points = None 

        results = interval_flow_segmentation_pipeline_with_vis(
            mono_normals=mono_normals,
            depthmaps=depthmaps,
            pointclouds=pointclouds, ### 
            data_dir=data_dir,
            frame_indices=frame_indices,
            interval=interval,
            device=device,
            save_debug=False,
            debug_dir=debug_dir,
            contact_points=contact_points,
            stat_cam=single_image
        )
        #### in results:: S is log(), 
  
        ###


        ### before there are exp to use 
        # convert_results_to_params_direct
        ### 
        max_iter = 1 if transfer_data else int(4e0)
      
        
        params = refine_sq_with_chamfer(
            results,
            lr=0,
            mesh_level=3,
            max_iter=max_iter,
            device="cuda"
        )
            

        params_np = params.detach().cpu().numpy()              
        per_sq_one_list = sqs_params_2_mesh(params, combine=False)
        per_sq_one_list = [
            _sanitize_trimesh(m) for m in per_sq_one_list
            if len(m.vertices) and np.isfinite(m.vertices).all()
        ]
        # --- save as .npy (single array) -----------------------

        
        np.save(f"_sqs_params/{tgt_name}.npy", params_np)



    optim = True
    coacd = True
    filtering = True
    contact_only = False
    tri_mesh_handle=None

    load_from_sc = False 
    scene_dir = os.path.join(save_dir, 'scene')
    os.makedirs(scene_dir, exist_ok=True)


    if TTTTTTEST==False:
        if coacd:
            import coacd
            vert, tri = vdb_volume.extract_triangle_mesh()
            vertices, faces = vert, tri
            tri = trimesh.Trimesh(
            vertices=vertices,   
            faces=faces, 
            process=False
            )

            mesh = tri
            obj_data = trimesh.exchange.obj.export_obj(tri)
            tri_mesh_handle = server.scene.add_mesh_trimesh(
                name=f"/frames/{i}/scene_mesh_",
                mesh=tri
            )
            with open(os.path.join(scene_dir, "scene_mesh.obj"), "w") as f:
                f.write(obj_data)

            tri_mesh_handle.visible=True

        if 1==0:
          nksr_mesh_path = os.path.join(f'out/{tgt_name}/scene_mesh_nksr.obj')
          nksr_mesh = trimesh.load(nksr_mesh_path)
          scene_mesh_raw_handle   = tri_mesh_handle   
          scene_mesh_coacd_handle = server.scene.add_mesh_trimesh(
              name="/frames/0/scene_mesh_coacd",
              mesh=nksr_mesh,        # or the concatenated COACD mesh
          )

        per_sq_pt_one_list = results["pts_items"]

        for idx, (pts, sq_mesh) in enumerate(zip(per_sq_pt_one_list, per_sq_one_list)):
            # ---- ① 点云也要旋转 ----
            pts_vis   = pts.detach().cpu().numpy()
            # pts_vis  = (R_ROOT @ pts_vis.T).T        # (N,3)

            pc = server.scene.add_point_cloud(
                name=f"/sq_pairs/pc_{idx}",
                points=pts_vis,
                colors=[229, 80, 80],
                point_size=0.05,
                point_shape="rounded",
            )
            pc.visible = (idx == 0)                      # 先只亮第 0 个
            sq_point_handles.append(pc)

            # ---- ② 网格同理 ----
            sq_mesh_vis = sq_mesh.copy()
            sq_mesh_vis.apply_transform(T_ROOT4)    # ← 一步到位（4×4 齐次）

            mh = server.scene.add_mesh_trimesh(
                name=f"/sq_pairs/sq_{idx}",
                mesh=sq_mesh_vis,
            )


            mh.visible = (idx == 0)
            sq_mesh_handles.append(mh)
                
        
        scene_mesh_contact = trimesh.util.concatenate(per_sq_one_list)



        with server.gui.add_folder("SQ Playback"):
            gui_sq_timestep = server.gui.add_slider(
                "SQ Index", min=0, max=len(sq_mesh_handles)-1, step=1,
                initial_value=0, disabled=True,
            )
            gui_sq_next  = server.gui.add_button("Next SQ")
            gui_sq_prev  = server.gui.add_button("Prev SQ")
            gui_sq_play  = server.gui.add_checkbox("SQ Playing", True)
            gui_sq_fps   = server.gui.add_slider(
                "SQ-FPS", min=0.5, max=30, step=0.5, initial_value=5.0,
        )
  

        prev_sq_idx = 0        # 保持当前可见索引

        @gui_sq_next.on_click
        def _(_) -> None:
            gui_sq_timestep.value = (gui_sq_timestep.value + 1) % len(sq_mesh_handles)

        @gui_sq_prev.on_click
        def _(_) -> None:
            gui_sq_timestep.value = (gui_sq_timestep.value - 1) % len(sq_mesh_handles)

        @gui_sq_play.on_update
        def _(_) -> None:
            gui_sq_timestep.disabled = gui_sq_play.value    # 播放时禁用手动拖

        @gui_sq_timestep.on_update
        def _(_) -> None:
            nonlocal prev_sq_idx
            cur = gui_sq_timestep.value
            with server.atomic():
                # 关掉旧的
                sq_point_handles[prev_sq_idx].visible = False
                sq_mesh_handles [prev_sq_idx].visible = False
                # 打开新的
                sq_point_handles[cur].visible = True
                sq_mesh_handles [cur].visible = True
            prev_sq_idx = cur
            server.flush()



        scene_mesh_coacd_contact_handle = server.scene.add_mesh_trimesh(
            name="/frames/0/scene_mesh_coacd_contact",
            mesh=scene_mesh_contact,        # or the concatenated COACD mesh
        )

        

        os.makedirs(tgt_folder, exist_ok=True)
        tgt_folder = os.path.join(tgt_folder, hmr_type)
        BIGBIG_folder = tgt_folder
        os.makedirs(tgt_folder, exist_ok=True)
        tgt_tgt_folder = tgt_folder

        tgt_folder = os.path.join(tgt_folder, 'scene_mesh_sqs')
        os.makedirs(tgt_folder, exist_ok=True)
        scene_mesh_contact.export(os.path.join(tgt_folder, 'scene_mesh_sqs.obj'))
        save_custom_mesh(per_sq_one_list, tgt_folder)
        
        if transfer_data:
          '''try:
            mesh = coacd.Mesh(mesh.vertices, mesh.faces)
            result = coacd.run_coacd(mesh,
                              merge=False,
                              threshold=0.02, # threshold=0.01, 
                              preprocess_resolution=150,
                              # preprocess_resolution=80, #  50 by d
                              # max_ch_vertex=256, # max_ch_vertex=512  default = 256
                              resolution=4000, #resolution=200000
                              max_convex_hull=100)
            mesh_parts = []
            for vs, fs in result:
                mesh_parts.append(trimesh.Trimesh(vs, fs))
            scene = trimesh.Scene()
            np.random.seed(0)
            for p in mesh_parts:
                # p.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(np.uint8)
                scene.add_geometry(p)
            scene_mesh = scene_mesh = trimesh.util.concatenate(mesh_parts)
          except:
            sfasjfaf =5 '''
            
          # scene_mesh.export('')

          dst_path = tgt_folder.replace('_geo/differentiable-blocksworld', '_vision/mega-sam')
          BIGBIG_folder_tgt = BIGBIG_folder.replace('_geo/differentiable-blocksworld', '_vision/mega-sam')
          # if Path(BIGBIG_folder_tgt).exists():
          #    shutil.rmtree(BIGBIG_folder_tgt)
          shutil.copytree(BIGBIG_folder, BIGBIG_folder_tgt, dirs_exist_ok=True)
        scene_mesh_coacd_contact_handle.visible = True


 
        print('done!')

    prev_timestep = gui_timestep.value

    if save_mode:
        server.stop()
    else:
        while True:
            if gui_playing.value and not gui_show_all_frames.value:
                gui_timestep.value = (gui_timestep.value + 1) % len(frame_nodes)

            if gui_sq_play.value and len(sq_mesh_handles):
                gui_sq_timestep.value = (gui_sq_timestep.value + 1) % len(sq_mesh_handles)

            time.sleep(1.0 / min(gui_framerate.value, gui_sq_fps.value))
if __name__ == "__main__":

    tyro.cli(main)

        