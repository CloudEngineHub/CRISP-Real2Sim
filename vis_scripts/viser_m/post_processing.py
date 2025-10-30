

import time
import sys
import argparse
from pathlib import Path
import trimesh
import numpy as onp
import tyro
from tqdm.auto import tqdm
import smplx
from smpl import SMPL, BodyModelSMPLH, BodyModelSMPLX
import torch
import os
import cv2
import numpy as np
import argparse
from scipy.ndimage import distance_transform_edt

import torch.nn.functional as F
from smpl import SMPL
import numpy as np
from pytorch3d import transforms
import os
import trimesh

import open3d as o3d
from scipy.spatial.transform import Rotation as sRot

import torch
import trimesh
import smplx

def main(data: Path = "./demo_tmp/NULL.npz"):
    tgt_name = str(data).split('_sgd')[0].split('/')[-1]
    save = True
    save_dir = '/data3/zihanwa3/_Robotics/_vision/mega-sam/post_results'
    save_dir = os.path.join(save_dir, tgt_name)

    hmr_dir = os.path.join(save_dir, 'hmr')
    scene_dir = os.path.join(save_dir, 'scene')

    post_hmr_dir = os.path.join(save_dir, 'post_hmr')
    post_scene_dir = os.path.join(save_dir, 'post_scene')
    os.makedirs(post_hmr_dir ,exist_ok=True)
    os.makedirs(post_scene_dir ,exist_ok=True)



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


    device='cpu'
    #data_path = "parkour_video_data/emdb_6_org"
    # save_path = f"{data_path}/saved_obj"
    post_scene_dir


    # manual_transform = sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False).as_matrix()
    transform = sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False).as_matrix()

    scene = o3d.io.read_triangle_mesh(f'{save_dir}/scene/scene_mesh_coacd.obj')
    scene.compute_vertex_normals()
    scene.rotate(transform, center=(0, 0, 0))
    pcd = scene.sample_points_poisson_disk(20000)  # 采样稀疏一点更快
    points = np.asarray(pcd.points)
    z_vals = points[:, 2]
    # 只保留靠近底部的点，比如 z 在底部10%~30%的范围
    z_min = np.percentile(z_vals, 0)
    z_max = np.percentile(z_vals, 40)
    mask = (z_vals >= z_min) & (z_vals <= z_max)
    selected_points = points[mask]

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
    z_transl = -np.min(np.asarray(scene.vertices)[:,2])
    scene.translate([0, 0, z_transl])

    input_dir = f'{save_dir}/scene_acd'
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.obj'):
            subfile_path = os.path.join(input_dir, filename)
            subscene = o3d.io.read_triangle_mesh(subfile_path)

            subscene.rotate(transform, center=(0, 0, 0))
            subscene.rotate(R_align, center=(0, 0, 0))
            subscene.translate([0, 0, z_transl])

            tgt_name = os.path.splitext(filename)[0]
            out_path = subfile_path#os.path.join(output_dir, f"{tgt_name}_afterrot.obj")
            o3d.io.write_triangle_mesh(out_path, subscene)



    # import pdb;pdb.set_trace()
    o3d.io.write_triangle_mesh(f"{post_scene_dir}/{tgt_name}_afterrot.obj", scene)

    T_align = np.eye(4)
    T_align[:3,:3] = R_align @ transform
    T_align[:3,3] = [0, 0, z_transl]

    # import pdb;pdb.set_trace()



    smpl = SMPL().to(device)
    hsfm_pkl = f'{save_dir}/hmr/hps_track.npy'

    pred_smpl = np.load(hsfm_pkl, allow_pickle=True).item()


    if "org" in hsfm_pkl or "gvhmr" in hsfm_pkl:
        pred_cams = pred_smpl['pred_cam']#.cuda()[:num_frames]
        body_pose = pred_smpl['body_pose'].cpu()#.cuda()[:num_frames]

        pred_shapes = pred_smpl['betas'].cpu()#.cuda()[:num_frames]
        transl_world = pred_smpl['transl'].cpu().squeeze(1)#.cuda()[:num_frames]
        global_orient_world =  transforms.axis_angle_to_matrix(pred_smpl['global_orient'].cpu()).unsqueeze(1)#.cuda()[:num_frames]

    else:

        pred_cams = pred_smpl['pred_cam']#.cuda()[:num_frames]
        body_pose = pred_smpl['body_pose'].cpu()#.cuda()[:num_frames]

        pred_shapes = pred_smpl['betas'].cpu()#.cuda()[:num_frames]
        transl_world = pred_smpl['transl'].cpu()#.cuda()[:num_frames]
        global_orient_world = pred_smpl['global_orient'].cpu()#.cuda()[:num_frames]

    pred = smpl(body_pose=body_pose, 
                global_orient=global_orient_world, 
                betas=pred_shapes, 
                transl=transl_world,
                pose2rot=False, 
                default_smpl=True)
    pred_vert = pred.vertices.cpu().numpy()

    '''for i in range(len(pred_vert)):
        mesh = trimesh.Trimesh(vertices=pred_vert[i], faces=smpl.faces)
        mesh.export(f"{save_path}/human_beforerot_{i:04d}.obj")'''



    pred_j3dg = pred.joints[:, :24].cpu().numpy()

    pose_aa = np.concatenate([transforms.matrix_to_axis_angle(global_orient_world.squeeze(1)), transforms.matrix_to_axis_angle(body_pose).flatten(1)],axis=-1)

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

    model = smplx.create("/data3/zihanwa3/_Robotics/humanoid/hybrid-imitation-parkour/data/smpl/SMPL_NEUTRAL.pkl", model_type="smpl",
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
    transl = transl + np.mean(trans_offset,axis=1) + np.array([0,0,z_transl+0.1])

    output = model(betas=betas, transl = transl, global_orient = poses[:,:3], body_pose = poses[:,3:],
                    return_verts=True)

    vertices = output.vertices.detach().cpu().numpy()

    # seq_name = data_path.split("/")[1]
    # import pdb;pdb.set_trace()
    np.savez(f"{post_scene_dir}/{tgt_name}.npz", trans=transl.numpy()[:], gender=gender, mocap_framerate=mocap_framerate, betas=betas.numpy()[:], poses=poses.numpy()[:], T_align = T_align)

    #for i in range(len(vertices)):
    #    mesh = trimesh.Trimesh(vertices=vertices[i], faces=model.faces)
    #    mesh.export(f"{post_hmr_dir}/human_afterrot_{i:04d}.obj")





if __name__ == "__main__":

    tyro.cli(main)

