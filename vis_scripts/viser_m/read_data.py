import time
import sys
import argparse
from pathlib import Path

import numpy as onp
from tqdm.auto import tqdm

import matplotlib.cm as cm  # For colormap
import smplx
from smpl import SMPL
import torch
import os
import cv2
import numpy as np
import argparse


device='cuda'
smpl = SMPL().to(device)
hsfm_pkl = 'hps_track.npy'
num_frames = 40 # min(max_frames, loader.num_frames()) 

'''
# load cam, mesh and smpl:
      results = {'pred_cam': [world_cam_R, world_cam_T], # cam 
                  'body_pose': pred_rotmats[:, 1:, :, :], # smpl 
                  'global_orient': global_orient_world, # smpl
                  'betas': pred_shapes, # smpl 
                  'transl': transl_world, # smpl 
                  'vertices': pred_vert, # mesh 
                  'faces': smpl.faces # mesh
                }

  load scene in world:
'''
pred_smpl = np.load(hsfm_pkl, allow_pickle=True).item()
scene = np.load('scene_T_HW_XYZRGB.npy') ### in a shape of [T, H*W, XYZRGB]
print(pred_smpl.keys(), scene.shape)

pred_cams = pred_smpl['pred_cam']#.cuda()[:num_frames]
body_pose = pred_smpl['body_pose']#.cuda()[:num_frames]
pred_shapes = pred_smpl['betas']#.cuda()[:num_frames]
transl_world = pred_smpl['transl']#.cuda()[:num_frames]
global_orient_world = pred_smpl['global_orient']#.cuda()[:num_frames]
pred = smpl(body_pose=body_pose, 
            global_orient=global_orient_world, 
            betas=pred_shapes, 
            transl=transl_world,
            pose2rot=False, 
            default_smpl=True)
pred_vert = pred.vertices
pred_j3dg = pred.joints[:, :24]