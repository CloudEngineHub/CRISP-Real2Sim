import os
import numpy as np
import torch
from torch.nn import functional as F
import contextlib

from smplx import SMPL as _SMPL
from smplx import SMPLLayer as _SMPLLayer
from smplx.body_models import SMPLOutput
from smplx.lbs import vertices2joints

from constants import JOINT_MAP, JOINT_NAMES


# SMPL data path
SMPL_DATA_PATH = "/data3/zihanwa3/_Robotics/_vision/tram/data/smpl/"

SMPL_MODEL_PATH = os.path.join(SMPL_DATA_PATH, "SMPL_NEUTRAL.pkl")
SMPL_MEAN_PARAMS = os.path.join(SMPL_DATA_PATH, "smpl_mean_params.npz")
SMPL_KINTREE_PATH = os.path.join(SMPL_DATA_PATH, "kintree_table.pkl")
JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(SMPL_DATA_PATH, 'J_regressor_extra.npy')
JOINT_REGRESSOR_H36M = os.path.join(SMPL_DATA_PATH, 'J_regressor_h36m.npy')


class SMPL(_SMPL):

    def __init__(self, create_default=False, *args, **kwargs):
        kwargs["model_path"] = "/data3/zihanwa3/_Robotics/_vision/tram/data/smpl"

        # remove the verbosity for the 10-shapes beta parameters
        with contextlib.redirect_stdout(None):
            super(SMPL, self).__init__(
                create_body_pose=create_default,
                create_betas=create_default,
                create_global_orient=create_default,
                create_transl=create_default,
                *args, 
                **kwargs
            )

        # SPIN 49(25 OP + 24) joints
        joints = [JOINT_MAP[i] for i in JOINT_NAMES]
        J_regressor_extra = np.load(JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)
        
        
    def forward(self, default_smpl=False, *args, **kwargs):
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        if default_smpl:
            return smpl_output

        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]

        output = SMPLOutput(vertices=smpl_output.vertices,
                            global_orient=smpl_output.global_orient,
                            body_pose=smpl_output.body_pose,
                            betas=smpl_output.betas,
                            full_pose=smpl_output.full_pose,
                            joints=joints)

        return output


    def query(self, hmr_output, default_smpl=False):
        pred_rotmat = hmr_output['pred_rotmat']
        pred_shape = hmr_output['pred_shape']

        smpl_out = self(global_orient=pred_rotmat[:, [0]],
                        body_pose = pred_rotmat[:, 1:],
                        betas = pred_shape,
                        default_smpl = default_smpl,
                        pose2rot=False)
        return smpl_out


import torch
import torch.nn as nn
import smplx

kwargs_disable_member_var = {
    "create_body_pose": False,
    "create_betas": False,
    "create_global_orient": False,
    "create_transl": False,
    "create_left_hand_pose": False,
    "create_right_hand_pose": False,
    "create_expression": False,
    "create_jaw_pose": False,
    "create_leye_pose": False,
    "create_reye_pose": False,
}


class BodyModelSMPLX(nn.Module):
    """Support Batch inference"""

    def __init__(self, model_path, **kwargs):
        super().__init__()
        # enable flexible batchsize, handle missing variable at forward()
        kwargs.update(kwargs_disable_member_var)
        self.bm = smplx.create(model_path=model_path, **kwargs)
        self.faces = self.bm.faces
        self.hand_pose_dim = self.bm.num_pca_comps if self.bm.use_pca else 3 * self.bm.NUM_HAND_JOINTS

        # For fast computing of skeleton under beta
        shapedirs = self.bm.shapedirs  # (V, 3, 10)
        J_regressor = self.bm.J_regressor[:22, :]  # (22, V)
        v_template = self.bm.v_template  # (V, 3)
        J_template = J_regressor @ v_template  # (22, 3)
        J_shapedirs = torch.einsum("jv, vcd -> jcd", J_regressor, shapedirs)  # (22, 3, 10)
        self.register_buffer("J_template", J_template, False)
        self.register_buffer("J_shapedirs", J_shapedirs, False)

    def forward(
        self,
        betas=None,
        global_orient=None,
        transl=None,
        body_pose=None,
        left_hand_pose=None,
        right_hand_pose=None,
        expression=None,
        jaw_pose=None,
        leye_pose=None,
        reye_pose=None,
        **kwargs
    ):

        device, dtype = self.bm.shapedirs.device, self.bm.shapedirs.dtype

        model_vars = [
            betas,
            global_orient,
            body_pose,
            transl,
            expression,
            left_hand_pose,
            right_hand_pose,
            jaw_pose,
            leye_pose,
            reye_pose,
        ]
        batch_size = 1
        for var in model_vars:
            if var is None:
                continue
            batch_size = max(batch_size, len(var))

        if global_orient is None:
            global_orient = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        if body_pose is None:
            body_pose = (
                torch.zeros(3 * self.bm.NUM_BODY_JOINTS, device=device, dtype=dtype)[None]
                .expand(batch_size, -1)
                .contiguous()
            )
        if left_hand_pose is None:
            left_hand_pose = (
                torch.zeros(self.hand_pose_dim, device=device, dtype=dtype)[None].expand(batch_size, -1).contiguous()
            )
        if right_hand_pose is None:
            right_hand_pose = (
                torch.zeros(self.hand_pose_dim, device=device, dtype=dtype)[None].expand(batch_size, -1).contiguous()
            )
        if jaw_pose is None:
            jaw_pose = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        if leye_pose is None:
            leye_pose = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        if reye_pose is None:
            reye_pose = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        if expression is None:
            expression = torch.zeros([batch_size, self.bm.num_expression_coeffs], dtype=dtype, device=device)
        if betas is None:
            betas = torch.zeros([batch_size, self.bm.num_betas], dtype=dtype, device=device)
        if transl is None:
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)

        bm_out = self.bm(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            transl=transl,
            expression=expression,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            **kwargs
        )

        return bm_out

    def get_skeleton(self, betas):
        """betas: (*, 10) -> skeleton_beta: (*, 22, 3)"""
        skeleton_beta = self.J_template + torch.einsum("...d, jcd -> ...jc", betas, self.J_shapedirs)  # (22, 3)
        return skeleton_beta

    def forward_bfc(self, **kwargs):
        """Wrap (B, F, C) to (B*F, C) and unwrap (B*F, C) to (B, F, C)"""
        for k in kwargs:
            assert len(kwargs[k].shape) == 3
        B, F = kwargs["body_pose"].shape[:2]
        smplx_out = self.forward(**{k: v.reshape(B * F, -1) for k, v in kwargs.items()})
        smplx_out.vertices = smplx_out.vertices.reshape(B, F, -1, 3)
        smplx_out.joints = smplx_out.joints.reshape(B, F, -1, 3)
        return smplx_out


import torch
import torch.nn as nn
import smplx

kwargs_disable_member_var = {
    "create_body_pose": False,
    "create_betas": False,
    "create_global_orient": False,
    "create_transl": False,
    "create_left_hand_pose": False,
    "create_right_hand_pose": False,
}


class BodyModelSMPLH(nn.Module):
    """Support Batch inference"""

    def __init__(self, model_path, **kwargs):
        super().__init__()
        # enable flexible batchsize, handle missing variable at forward()
        kwargs.update(kwargs_disable_member_var)
        self.bm = smplx.create(model_path=model_path, **kwargs)
        self.faces = self.bm.faces
        self.is_smpl = kwargs.get("model_type", "smpl") == "smpl"
        if not self.is_smpl:
            self.hand_pose_dim = self.bm.num_pca_comps if self.bm.use_pca else 3 * self.bm.NUM_HAND_JOINTS

        # For fast computing of skeleton under beta
        shapedirs = self.bm.shapedirs  # (V, 3, 10)
        J_regressor = self.bm.J_regressor[:22, :]  # (22, V)
        v_template = self.bm.v_template  # (V, 3)
        J_template = J_regressor @ v_template  # (22, 3)
        J_shapedirs = torch.einsum("jv, vcd -> jcd", J_regressor, shapedirs)  # (22, 3, 10)
        self.register_buffer("J_template", J_template, False)
        self.register_buffer("J_shapedirs", J_shapedirs, False)

    def forward(
        self,
        betas=None,
        global_orient=None,
        transl=None,
        body_pose=None,
        left_hand_pose=None,
        right_hand_pose=None,
        **kwargs
    ):

        device, dtype = self.bm.shapedirs.device, self.bm.shapedirs.dtype

        model_vars = [betas, global_orient, body_pose, transl, left_hand_pose, right_hand_pose]
        batch_size = 1
        for var in model_vars:
            if var is None:
                continue
            batch_size = max(batch_size, len(var))

        if global_orient is None:
            global_orient = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        if body_pose is None:
            body_pose = (
                torch.zeros(3 * self.bm.NUM_BODY_JOINTS, device=device, dtype=dtype)[None]
                .expand(batch_size, -1)
                .contiguous()
            )
        if not self.is_smpl:
            if left_hand_pose is None:
                left_hand_pose = (
                    torch.zeros(self.hand_pose_dim, device=device, dtype=dtype)[None]
                    .expand(batch_size, -1)
                    .contiguous()
                )
            if right_hand_pose is None:
                right_hand_pose = (
                    torch.zeros(self.hand_pose_dim, device=device, dtype=dtype)[None]
                    .expand(batch_size, -1)
                    .contiguous()
                )
        if betas is None:
            betas = torch.zeros([batch_size, self.bm.num_betas], dtype=dtype, device=device)
        if transl is None:
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)

        bm_out = self.bm(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            transl=transl,
            **kwargs
        )

        return bm_out

    def get_skeleton(self, betas):
        """betas: (*, 10) -> skeleton_beta: (*, 22, 3)"""
        skeleton_beta = self.J_template + torch.einsum("...d, jcd -> ...jc", betas, self.J_shapedirs)  # (22, 3)
        return skeleton_beta
