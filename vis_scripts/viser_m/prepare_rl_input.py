import time
import sys
import argparse
from pathlib import Path
import trimesh
import numpy as onp
import tyro
from tqdm.auto import tqdm
import os, shutil
import copy
import viser
import viser.extras
import viser.transforms as tf
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
from optim import build_sdf_volume, GPUSDF, sdf_to_pointcloud, optim_alm
from utils import make_gaussian_spheres_urdf
import json
from read_emdb_utils import save_rotated
import os
import shutil
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree

# ------------------------------------------------------------------
# 1) Copy the motion .npz and the entire scene folder to the robot repo
# ------------------------------------------------------------------
def copy_assets(org_motion_file: str,
                org_scene_dir: str,
                org_urdf_dir: str,
                out_motion_file: str,
                out_scene_dir: str,
                out_urdf_dir: str) -> None:
    """
    Copy a single motion file and a directory of OBJ assets to the
    specified output locations.

    - org_motion_file : path to *.npz motion file to copy
    - org_scene_dir   : directory that holds the OBJ scene meshes
    - out_motion_file : destination *.npz path inside robot_folder
    - out_scene_dir   : destination directory for the OBJ meshes
    """
    # Make sure the motion-data directory exists
    Path(out_motion_file).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(org_motion_file, out_motion_file)

    # --- scene meshes ---
    if os.path.isdir(out_scene_dir):
        shutil.rmtree(out_scene_dir)                # 先删
    shutil.copytree(org_scene_dir, out_scene_dir)   # 再拷 OBJ

    # --- URDF ---
    Path(out_urdf_dir).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(org_urdf_dir, out_urdf_dir)        # 最后拷，文件就留下来了

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


def write_urdf_for_objs(
    obj_folder: str,
    urdf_path: str,
    mesh_subdir: str = "primitive_set",
    copy_meshes: bool = True,
) -> None:
    """
    Build a *single* URDF that lists every OBJ in `obj_folder`.

    Parameters
    ----------
    obj_folder : str
        Directory containing *.obj meshes.
    urdf_path : str
        Where to save the merged URDF (e.g. ".../scene.urdf").
    mesh_subdir : str, optional
        Sub-folder (created next to the URDF) that will hold / reference the meshes.
    scale : (sx, sy, sz), optional
        <mesh scale="sx sy sz">  (defaults to 1,1,1).
    origins : dict[str, (x, y, z)], optional
        Optional per-mesh translation.  Keys are mesh basenames (without path);
        any mesh not present defaults to (0,0,0).
    sdf_resolution : int | None, optional
        If given, adds   <sdf resolution="..."/>   inside the link.
    copy_meshes : bool, optional
        Copy the OBJ files into `mesh_subdir`.  If False, they are referenced in-place.
    """
    obj_folder = Path(obj_folder).expanduser().resolve()
    urdf_path  = Path(urdf_path).expanduser().resolve()
    urdf_path.parent.mkdir(parents=True, exist_ok=True)

    mesh_dir = urdf_path.parent / mesh_subdir
    mesh_dir.mkdir(parents=True, exist_ok=True)
    robot = Element("robot", name="merged_object")
    link  = SubElement(robot, "link", name="object")
    for obj_path in sorted(obj_folder.glob("*.obj")):
        # decide how the mesh should be referenced
        if copy_meshes:
            dst_path = mesh_dir / obj_path.name
            if not dst_path.exists():
                shutil.copy2(obj_path, dst_path)
            mesh_ref = f"{mesh_subdir}/{obj_path.name}"
        else:
            mesh_ref = str(obj_path.relative_to(urdf_path.parent))

        # visual + collision blocks (always at origin, no scale)
        for tag in ("visual", "collision"):
            sec    = SubElement(link, tag)
            SubElement(sec, "origin", xyz="0 0 0", rpy="0 0 0")
            geom   = SubElement(sec, "geometry")
            SubElement(geom, "mesh", filename=mesh_ref)

    # ------------------------------------------------------------------ #
    ElementTree(robot).write(
        urdf_path,
        encoding="utf-8",
        xml_declaration=True,
        short_empty_elements=False,
    )
    print(f"✓ wrote merged URDF with {len(list(obj_folder.glob('*.obj')))} meshes → {urdf_path}")



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
    hmr_type: str = '', 
) -> None:
    from pathlib import Path  # <-- Import Path here if not already imported
    tgt_name = str(data).split('_sgd')[0].split('/')[-1]   # gives 'MPH112_00169_01_tram'
    tgt_name = "_".join(tgt_name.split("_")[:-1]) 


    data = np.load(data)
    import random
    save_dir = './post_results'
    save_dir = os.path.join(save_dir, tgt_name)
    os.makedirs(save_dir, exist_ok=True)
    save_dir = os.path.join(save_dir, hmr_type)


    print(save_dir)


    hmr_dir = os.path.join(save_dir, 'hmr')
    os.makedirs(hmr_dir, exist_ok=True)

    scene_dir = os.path.join(save_dir, 'scene')
    os.makedirs(scene_dir, exist_ok=True)

    acd_scene_dir = str(Path(save_dir) / "scene_acd")

    org_motion_file = f'{hmr_dir}/roted.npz'
    org_scene_dir = f'{acd_scene_dir}/roted'
    org_urdf_file = f'{acd_scene_dir}/scene.urdf'

    robot_folder = '/data3/zihanwa3/_Robotics/humanoid/hybrid-imitation-parkour'
    scene_folder = 'parkour_anim/data/assets/urdf'
    scene_folder = os.path.join(scene_folder, tgt_name)
    os.makedirs(scene_folder, exist_ok=True)

    
    out_motion_file = os.path.join(robot_folder, 'motion_data', 'prox')
    os.makedirs(out_motion_file, exist_ok=True)
    out_motion_file =  os.path.join(out_motion_file, f'{tgt_name}.npz')
    out_scene_dir = os.path.join(robot_folder, scene_folder)
    out_urdf_dir = os.path.join(robot_folder, scene_folder, f'{tgt_name}.urdf')

    copy_assets(org_motion_file, org_scene_dir, org_urdf_file, out_motion_file, out_scene_dir, out_urdf_dir)
    #write_urdf_for_objs(obj_folder=out_scene_dir, urdf_path=os.path.join(out_scene_dir, f'{tgt_name}_new.urdf'))

if __name__ == "__main__":

    tyro.cli(main)

