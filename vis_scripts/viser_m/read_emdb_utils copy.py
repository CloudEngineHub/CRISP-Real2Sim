from smpl import SMPL
import numpy as np

import os
import trimesh

import open3d as o3d
from scipy.spatial.transform import Rotation as sRot

import torch
import trimesh
import smplx
# rotation_utils.py
import torch
import os
from pathlib import Path
import numpy as np
import trimesh

def vis_hmr(data, save_path, device, every=100):
  smpl = SMPL().to(device)
  pred = smpl(**data)
  pred_vert = pred.vertices.cpu().numpy()
  for i in range(0, len(pred_vert), every):
      mesh = trimesh.Trimesh(vertices=pred_vert[i], faces=smpl.faces, process=False)
      mesh.export(os.path.join(save_path, f"human_beforerot_{i:04d}.obj"))


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    axis_angle: (..., 3)  axis * angle  (rad)
    returns   : (..., 3, 3) rotation matrices
    """
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)          # (...,1)
    axis  = axis_angle / (angle.clamp_min(1e-8))                         # normalise

    # Rodrigues’ formula
    x, y, z = axis.unbind(-1)                                            # each (...,)
    zeros   = torch.zeros_like(x)
    K = torch.stack([ zeros, -z,    y,
                      z,    zeros, -x,
                     -y,     x,   zeros ], dim=-1).reshape(axis.shape[:-1] + (3, 3))

    I = torch.eye(3, dtype=axis.dtype, device=axis.device).expand_as(K)
    sin, cos = torch.sin(angle)[..., None], torch.cos(angle)[..., None]
    R = I + sin*K + (1.-cos) * (K @ K)
    return R


def matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """
    R: (..., 3, 3) rotation matrices
    returns (..., 3) axis * angle  (rad)
    """
    # numeric safeguard
    trace = R.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)              # (...,)
    cos_theta = ((trace - 1.) / 2.).clamp(-1.+1e-6, 1.-1e-6)
    theta = torch.acos(cos_theta)                                       # (...,)

    # sin(theta) might be tiny → use first-order expansion
    sin_theta = torch.sin(theta)
    mask = sin_theta.abs() < 1e-6

    rx = (R[..., 2, 1] - R[..., 1, 2]) / (2.*sin_theta)
    ry = (R[..., 0, 2] - R[..., 2, 0]) / (2.*sin_theta)
    rz = (R[..., 1, 0] - R[..., 0, 1]) / (2.*sin_theta)
    axis = torch.stack((rx, ry, rz), dim=-1)

    # first-order fallback for very small angles
    if mask.any():
        axis[mask] = torch.stack(
            [(R[..., 2, 1]-R[..., 1, 2]),
             (R[..., 0, 2]-R[..., 2, 0]),
             (R[..., 1, 0]-R[..., 0, 1])], dim=-1)[mask] * 0.5
    return axis * theta.unsqueeze(-1)



def estimate_rigid_transform(pcd1_np, pcd2_np):
    assert pcd1_np.shape == pcd2_np.shape, "Shape mismatch"
    
    # 构建 Open3D 点云对象
    src = o3d.geometry.PointCloud()
    tgt = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(pcd1_np)
    tgt.points = o3d.utility.Vector3dVector(pcd2_np)

    # 用对应点配准（point-to-point）
    reg = o3d.pipelines.registration.registration_icp(
        src, tgt, max_correspondence_distance=1.0,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    
    return reg.transformation  # shape (4, 4)


def transform_objs(src_dir,
                   R: np.ndarray = None,
                   transl: np.ndarray = None,
                   out_subdir: str = "roted"):
    """
    Apply a rigid transform (R, transl) to every OBJ mesh in `src_dir`.

    Parameters
    ----------
    src_dir : str | Path
        Directory that contains the original OBJ files.
    R : (3,3) array_like, optional
        Rotation matrix.  Defaults to identity if omitted.
    transl : (3,) array_like, optional
        Translation vector (x, y, z).  Defaults to zero if omitted.
    out_subdir : str
        Folder created inside `src_dir` where transformed files are written.
    """
    src_dir  = Path(src_dir)
    out_dir  = src_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ­­­­­­------------------------------------------------------------------ #
    # Build the 4 × 4 homogeneous matrix                                     #
    # ­­­­­­------------------------------------------------------------------ #
    if R is None:
        R = np.eye(3)
    if transl is None:
        transl = np.zeros(3)
    #R_extra = sRot.from_euler('xyz', [-np.pi / 2, 0, 0], degrees=False).as_matrix()

    # --- original data -----------------------------------------------------------
    R      = np.asarray(R,      dtype=float).reshape(3, 3)
    transl = np.asarray(transl, dtype=float).reshape(3)

    # ----------------------------------------------------------------------------- 
    # OPTION A – apply the extra rotation **FIRST** (world-frame prepend)
    R = R#R_extra @ R          # first R_extra, then original R
    transl = np.asarray(transl, dtype=float).reshape(3)

    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3]  = transl

    # ­­­­­­------------------------------------------------------------------ #
    # Process all OBJ files                                                   #
    # ­­­­­­------------------------------------------------------------------ #
    for obj_path in sorted(src_dir.glob("*.obj")):
        mesh = trimesh.load(obj_path, process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump(concatenate=True)       # Scene → single mesh

        mesh.apply_transform(T)                      # in-place

        out_path = out_dir / obj_path.name
        mesh.export(out_path)
        print(f"✔ saved {out_path}")



#data_path = "parkour_video_data/emdb_6_org"

#save_path = f"{data_path}/saved_obj"

def save_rotated(scene_path, hmr_path, primtive_parent, hmr_type=None):
    
    #os.makedirs(save_path,exist_ok=True)
    # manual_transform = sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False).as_matrix()
  

    required_files = [
        f'{scene_path}/scene_mesh.obj',
        f'{scene_path}/scene_mesh_coacd.obj',
        f'{hmr_path}/{hmr_type}.npy',
        "data/smpl/SMPL_NEUTRAL.pkl"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"ERROR: Required file not found: {file_path}")
            return None, None
            
    device='cpu'
    transform = sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False).as_matrix()

    scene = o3d.io.read_triangle_mesh(f'{scene_path}/scene_mesh.obj')
    scene.compute_vertex_normals()
    print('oh')
    scene.rotate(transform, center=(0, 0, 0))
    print('ofsfsah')

    pcd = scene.sample_points_poisson_disk(20000) 

    points = np.asarray(pcd.points)
    z_vals = points[:, 2]
    
    z_min = np.percentile(z_vals, 0)
    z_max = np.percentile(z_vals, 40)
    mask = (z_vals >= z_min) & (z_vals <= z_max)
    selected_points = points[mask]
    print('ofsfsaofsfsahofsfsahh')

    # 用筛选后的点再建一个新的PointCloud
    selected_pcd = o3d.geometry.PointCloud()
    selected_pcd.points = o3d.utility.Vector3dVector(selected_points)


    # 平面拟合（使用RANSAC找地面）
    plane_model, inliers = selected_pcd.segment_plane(distance_threshold=0.01,
                                            ransac_n=3,
                                            num_iterations=1000)
    [a, b, c, d] = plane_model
    ground_normal = np.array([a, b, c])
    ground_normal /= np.linalg.norm(ground_normal)

    # 计算旋转矩阵，将 normal 对齐到 z 轴负方向
    target = np.array([0, 0, 1])
    rot_axis = np.cross(ground_normal, target)
    rot_angle = np.arccos(np.clip(np.dot(ground_normal, target), -1.0, 1.0))

    if np.linalg.norm(rot_axis) < 1e-6:
        R_align = np.eye(3)
    else:
        rot_axis = rot_axis / np.linalg.norm(rot_axis)
        R_align = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * rot_angle)

    scene.rotate(R_align, center=(0, 0, 0))  # center可以指定旋转中心，一般用 (0,0,0)
    # import pdb;pdb.set_trace()
    
    z_transl = -np.min(np.asarray(scene.vertices)[:,2])
    scene.translate([0, 0, z_transl])
    # import pdb;pdb.set_trace()
    o3d.io.write_triangle_mesh(f"{scene_path}/scene_afterrot.obj", scene)

    scene_acd = o3d.io.read_triangle_mesh(f'{scene_path}/scene_mesh_coacd.obj')
    scene_acd.compute_vertex_normals()
    scene_acd.rotate(transform, center=(0, 0, 0))
    scene_acd.rotate(R_align, center=(0, 0, 0))
    scene_acd.translate([0, 0, z_transl])
    o3d.io.write_triangle_mesh(f"{scene_path}/scene_coacd_afterrot.obj", scene_acd)



    T_align = np.eye(4)
    T_align[:3,:3] = R_align @ transform
    T_align[:3,3] = [0, 0, z_transl]

    # import pdb;pdb.set_trace()

    transform_objs(primtive_parent,
                   R=R_align @ transform,
                   transl=np.array([0, 0, z_transl]),
                   out_subdir= "roted")
    

    smpl = SMPL().to(device)
    hsfm_pkl = f'{hmr_path}/{hmr_type}.npy'
    pred_smpl = np.load(hsfm_pkl, allow_pickle=True).item()
    pred_cams = pred_smpl['pred_cam']#.cuda()[:num_frames]
    body_pose = pred_smpl['body_pose'].cpu()#.cuda()[:num_frames]

    pred_shapes = pred_smpl['betas'].cpu()#.cuda()[:num_frames]
    transl_world = pred_smpl['transl'].cpu()# .squeeze(1)#.cuda()[:num_frames]
    global_orient_world = pred_smpl['global_orient'].cpu() # axis_angle_to_matrix(pred_smpl['global_orient'].cpu()).unsqueeze(1)#.cuda()[:num_frames]

    '''else:

        pred_cams = pred_smpl['pred_cam']#.cuda()[:num_frames]
        body_pose = pred_smpl['body_pose'].cpu()#.cuda()[:num_frames]
        pred_shapes = pred_smpl['betas'].cpu()#.cuda()[:num_frames]
        transl_world = pred_smpl['transl'].cpu()#.cuda()[:num_frames]
        global_orient_world = pred_smpl['global_orient'].cpu()#.cuda()[:num_frames]
    '''

    pred = smpl(body_pose=body_pose, 
                global_orient=global_orient_world, 
                betas=pred_shapes, 
                transl=transl_world,
                pose2rot=False, 
                default_smpl=True)
    


    pred_vert = pred.vertices.cpu().numpy()

    #for i in range(len(pred_vert)):
    #    mesh = trimesh.Trimesh(vertices=pred_vert[i], faces=smpl.faces)
    #    mesh.export(f"{save_path}/human_beforerot_{i:04d}.obj")



    pred_j3dg = pred.joints[:, :24].cpu().numpy()

    pose_aa = np.concatenate([matrix_to_axis_angle(global_orient_world.squeeze(1)), matrix_to_axis_angle(body_pose).flatten(1)],axis=-1)

    root_trans = transl_world.numpy()

    transform = R_align @ transform

    # 构造新的 transform
    transform = sRot.from_matrix(transform)

    new_root = (transform * sRot.from_rotvec(pose_aa[:, :3])).as_rotvec()
    pose_aa[:, :3] = new_root
    root_trans = root_trans.dot(transform.as_matrix().T)

    trans = root_trans
    
    gender = "neutral"
    mocap_framerate = 30
    betas = pred_shapes
    poses = pose_aa

    model = smplx.create("data/smpl/SMPL_NEUTRAL.pkl", model_type="smpl",
                            gender="neutral",
                            num_betas=10,
                            batch_size=1, 
                            ext="pkl")

    # amass_data = np.load("door_amass.npz")
    # import pdb;pdb.set_trace()
    # betas = torch.zeros(10)[None]
    transl = torch.from_numpy(trans)
    poses = torch.from_numpy(poses)
    output_temp = model(betas=betas, transl = transl, global_orient = poses[:,:3], body_pose = poses[:,3:],
                    return_verts=True)

    vertices_temp = output_temp.vertices.detach().cpu().numpy()

    trans_offset = pred_vert @ transform.as_matrix().T - vertices_temp
    # import pdb;pdb.set_trace()
    transl = transl + np.mean(trans_offset,axis=1) + np.array([0,0,z_transl])

    output = model(betas=betas, transl = transl, global_orient = poses[:,:3], body_pose = poses[:,3:],
                    return_verts=True)

    vertices = output.vertices.detach().cpu().numpy()

    #seq_name = data_path.split("/")[1]
    # import pdb;pdb.set_trace()
    np.savez(f"{hmr_path}/roted_{hmr_type}.npz", trans=transl.numpy()[:], gender=gender, mocap_framerate=mocap_framerate, betas=betas.numpy()[:], poses=poses.numpy()[:], T_align = T_align)
    every = 100
    save_path=os.path.join(hmr_path, 'rot_vis')
    os.makedirs(save_path, exist_ok=True)


    for i in range(0, len(vertices), every):
        mesh = trimesh.Trimesh(vertices=vertices[i], faces=smpl.faces, process=False)
        mesh.export(os.path.join(save_path, f"human_beforerot_{i:04d}.obj"))


    return R_align, np.array([0, 0, z_transl])