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
import json
from read_emdb_utils import save_rotated

import numpy as np, trimesh, xml.etree.ElementTree as ET
from pathlib import Path
import scipy
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import trimesh
from superquadrics import superquadric
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import trimesh



class DifferentiableBlocksWorld(nn.Module):
    name = 'dbw'

    def __init__(self, img_size, priors=None, **kwargs):
        super().__init__()
        self._init_kwargs = deepcopy(kwargs)
        self._init_kwargs['img_size'] = img_size
        self._init_renderer(img_size, **kwargs.get('renderer', {}))
        self._init_rend_optim(**kwargs.get('rend_optim', {}))
        self.use_ndc = kwargs.get("use_ndc", False)

        
        if priors is None:
          self._init_blocks(**kwargs.get('mesh', {}))
        else:
          self._init_blocks_w_dense(priors, **kwargs.get('mesh', {}))
          # _init_blocks_w_dense _init_blocks_w_normal _init_blockssss
          # self._init_blocks_w_dense(priors, **kwargs.get('mesh', {}))

        self._init_loss(**kwargs.get('loss', {}))
        self.cur_epoch = 0
        if priors is None:
            self.texture_free=False
        else:
            self.texture_free=True

    def _init_blocks_w_dense(
        self,
        priors,  # legacy 7-tuple OR dict/npz-like
        ratio_block_scene: float = 1.0,
        opacity_init: float = 1.0,
        *,
        frame_idx: int = 0,
        downsample_factor: int = 32,
        estimate_normals: bool = False,
        K: Optional[Union[np.ndarray, torch.Tensor]] = None,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        **kwargs,
    ):
        """
        Unified initializer that supports:
          • Legacy 'priors' 7-tuple: (plane_instances, masks_avg_normals, masks_areas,
                                      depth_areas, geo, plane_colors, cam_int)
          • Minimal npz-like dict w/ keys: images, depth/depths, init_conf, enlarged_dynamic_mask,
                                          obj_masks, uncertainty.

        Only the 'images' + 'depth'(/'depths') fields are strictly required in the new path.
        All others are optional and will be gracefully defaulted.

        Args:
            frame_idx: Which frame in the sequence to use to seed blocks.
            downsample_factor: Pixel stride for block sampling.
            estimate_normals: If True and no normals provided, estimate from depth (slow-ish).
            K or (fx,fy,cx,cy): Camera intrinsics override. If omitted, attempt self.renderer.cameras[0].
        """
        import math
        import numpy as np
        import torch
        import torch.nn.functional as F
        from torch import nn

        device = getattr(self, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # ------------------------------------------------------------------
        # Helper: safe fetch from npz/dict/Namespace
        # ------------------------------------------------------------------
        def _get(d, *names, default=None):
            for n in names:
                if isinstance(d, (dict,)):
                    if n in d:
                        return d[n]
                else:
                    if hasattr(d, n):
                        return getattr(d, n)
            return default

        # ------------------------------------------------------------------
        # Distinguish legacy vs new format
        # ------------------------------------------------------------------
        legacy_tuple = (
            isinstance(priors, (tuple, list))
            and len(priors) == 7
            and not isinstance(priors, dict)
        )

        if legacy_tuple:
            # -------------------- LEGACY PATH (unchanged logic) --------------------
            (plane_instances,
            masks_avg_normals,
            masks_areas,
            depth_areas,
            geo,
            plane_colors,
            cam_int) = priors

            # Expect geo = (depthmap_batch, normalmap_batch, image_batch)
            depthmap = torch.as_tensor(geo[0], dtype=torch.float32, device=device)[0][0]
            normalmap = torch.as_tensor(geo[1], dtype=torch.float32, device=device)[0]
            input_image = torch.as_tensor(geo[2], dtype=torch.float32, device=device)[0]

            # cam_int = (K_int, R_cam_int, T_cam_int)
            if isinstance(cam_int, (tuple, list)) and len(cam_int) >= 1:
                K_int = torch.as_tensor(cam_int[0], dtype=torch.float32, device=device)
            else:
                raise ValueError("Legacy priors missing camera intrinsics.")

            fx, fy, cx, cy = K_int[0, 0], K_int[1, 1], K_int[0, 2], K_int[1, 2]

            # Convert formats to common path below
            # Ensure image in [0,1]
            if input_image.max() > 1:
                input_image = input_image / 255.0
            if input_image.dim() == 3 and input_image.shape[0] == 3:
                input_image = input_image.permute(1, 2, 0)

            # [3,H,W] -> [H,W,3] for normalmap
            if normalmap.dim() == 3 and normalmap.shape[0] == 3:
                normalmap = normalmap.permute(1, 2, 0)

            # Colors for planes (legacy)
            plane_colors = torch.as_tensor(plane_colors, dtype=torch.float32, device=device) / 255.0

        else:
            # -------------------- NEW MINIMAL DATA PATH --------------------
            data = priors  # rename for clarity

            # Required ------------------------------------------------------
            images = _get(data, 'images')
            if images is None:
                raise ValueError("New data path requires key 'images'.")

            depth = _get(data, 'depth', 'depths')
            if depth is None:
                raise ValueError("New data path requires key 'depth' or 'depths'.")

            # Convert to tensors -------------------------------------------
            images = torch.as_tensor(images, device=device)
            depth = torch.as_tensor(depth, device=device)

            # Squeeze depth's trailing channel if needed
            if depth.dim() == 4 and depth.shape[-1] == 1:
                depth = depth[..., 0]

            # Force float32
            images = images.float()
            depth = depth.float()

            # Normalize image if uint8 range
            if images.max() > 1.0:
                images = images / 255.0

            # Ensure images are (N,H,W,3)
            if images.dim() == 4 and images.shape[1] == 3 and images.shape[-1] != 3:
                # assume (N,3,H,W)
                images = images.permute(0, 2, 3, 1)

            if images.dim() != 4 or images.shape[-1] != 3:
                raise ValueError(f"Images expected shape (N,H,W,3); got {tuple(images.shape)}.")

            N, H, W, _ = images.shape

            # Pick frame
            if not (0 <= frame_idx < N):
                raise ValueError(f"frame_idx {frame_idx} out of range 0..{N-1}.")
            input_image = images[frame_idx]  # (H,W,3)
            depthmap = depth[frame_idx]      # (H,W)

            # Optional keys ------------------------------------------------
            # (We don't *need* them to build blocks, but we store for later.)
            self.images = images.detach().cpu().numpy()  # keep numpy copy to match your loader
            self.depths = depth.detach().cpu().numpy()

            self.init_conf_data = _get(data, 'init_conf', default=[])
            self.masks = _get(data, 'enlarged_dynamic_mask', default=[])
            self.obj_masks = _get(data, 'obj_masks', default=[])
            self.confidences = np.array(_get(data, 'uncertainty', default=[]))

            # Safe mask reshape/resize
            import skimage.transform
            if isinstance(self.masks, (np.ndarray, torch.Tensor)) and self.masks.size != 0:
                m_np = np.asarray(self.masks)
                # Accept shapes (N,H,W) or (H,W); pick frame
                if m_np.ndim == 3 and m_np.shape[0] == N:
                    m_np = m_np[frame_idx]
                elif m_np.ndim == 4 and m_np.shape[-1] == 1:
                    m_np = m_np[..., 0]
                # resize to (H,W) nearest-neighbor
                m_np = skimage.transform.resize(m_np, (H, W), order=0, preserve_range=True, anti_aliasing=False)
                m_np = (m_np > 0.5).astype(np.bool_)
            else:
                # default all-valid mask
                m_np = np.ones((H, W), dtype=bool)
            self.masks = m_np

            # We do not have plane colors / normals; set placeholders
            plane_colors = None
            normalmap = None

            # Intrinsics ---------------------------------------------------
            if K is None and (fx is None or fy is None or cx is None or cy is None):
                # Try to infer from renderer's first camera (if available)
                try:
                    cam = self.renderer.cameras
                    # Many PyTorch3D cams: .focal_length, .principal_point, in NDC or pix
                    # We'll grab [0]
                    if hasattr(cam, 'focal_length'):
                        f = cam.focal_length[0]  # (2,) f_x, f_y
                        fx, fy = float(f[0]), float(f[1])
                    if hasattr(cam, 'principal_point'):
                        p = cam.principal_point[0]
                        cx, cy = float(p[0]), float(p[1])
                except Exception:
                    pass

            if K is not None:
                K_t = torch.as_tensor(K, dtype=torch.float32, device=device)
                fx, fy, cx, cy = K_t[0, 0], K_t[1, 1], K_t[0, 2], K_t[1, 2]
            else:
                # if any remain None, fallback
                if fx is None or fy is None:
                    fx = fy = float(max(H, W))
                if cx is None:
                    cx = float(W / 2.0)
                if cy is None:
                    cy = float(H / 2.0)

            K_int = torch.tensor([[fx, 0.0, cx],
                                  [0.0, fy, cy],
                                  [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)

        # ------------------------------------------------------------------
        # Common attributes used downstream
        # ------------------------------------------------------------------
        self.S_world = kwargs.pop('S_world', 1.0)
        self.ratio_block_scene = ratio_block_scene
        self.scale_min = kwargs.pop('scale_min', 0.0)
        self.txt_size = kwargs.pop('txt_size', 256)
        self.txt_bkg_upscale = kwargs.pop('txt_bkg_upscale', 1)

        # World transform (kept trivial unless supplied)
        elev, azim, roll = kwargs.pop('R_world', [0.0, 0.0, 0.0])
        R_world = (
            elev_to_rotation_matrix(elev)
            @ azim_to_rotation_matrix(azim)
            @ roll_to_rotation_matrix(roll)
        )[None]
        self.register_buffer('R_world', R_world.to(device=device, dtype=torch.float32))
        self.register_buffer('T_world', torch.tensor([0.0, 0.0, 0.0], device=device)[None])

        # UV data for icosphere -----------------------------------------------------
        faces_uvs, verts_uvs = get_icosphere_uvs(level=1, fix_continuity=True, fix_poles=True)
        p_left = abs(int(np.floor(verts_uvs.min(0)[0][0].item() * self.txt_size)))
        p_right = int(np.ceil((verts_uvs.max(0)[0][0].item() - 1) * self.txt_size))
        verts_u = (verts_uvs[..., 0] * self.txt_size + p_left) / (self.txt_size + p_left + p_right)
        verts_uvs = torch.stack([verts_u, verts_uvs[..., 1]], dim=-1)
        self.txt_padding = (p_left, p_right)
        self.BNF = len(faces_uvs)

        # ------------------------------------------------------------------
        # Depth + normal prep
        # ------------------------------------------------------------------
        depthmap = torch.as_tensor(depthmap, dtype=torch.float32, device=device)

        if normalmap is None:
            if estimate_normals:
                normalmap = _estimate_normals_from_depth(depthmap, fx, fy, cx, cy, device=device)
            else:
                # broadcast fallback constant normal pointing -Z in camera coords
                normalmap = torch.zeros((depthmap.shape[0], depthmap.shape[1], 3),
                                        dtype=torch.float32, device=device)
                normalmap[..., 2] = -1.0
        else:
            normalmap = torch.as_tensor(normalmap, dtype=torch.float32, device=device)
            if normalmap.dim() == 3 and normalmap.shape[0] == 3:
                normalmap = normalmap.permute(1, 2, 0)

        # Ensure input_image (H,W,3) float in [0,1]
        input_image = torch.as_tensor(input_image, dtype=torch.float32, device=device)
        if input_image.dim() == 3 and input_image.shape[0] == 3 and input_image.shape[-1] != 3:
            input_image = input_image.permute(1, 2, 0)
        if input_image.max() > 1.0:
            input_image = input_image / 255.0
        input_image = torch.clamp(input_image, 0.0, 1.0)

        H, W = depthmap.shape[-2], depthmap.shape[-1]

        # ------------------------------------------------------------------
        # Build per-pixel blocks (downsampled)
        # ------------------------------------------------------------------
        valid_mask = depthmap > 0
        valid_yx = torch.nonzero(valid_mask, as_tuple=False)

        # stride filter
        valid_yx = valid_yx[(valid_yx[:, 0] % downsample_factor == 0)
                            & (valid_yx[:, 1] % downsample_factor == 0)]

        if valid_yx.numel() == 0:
            raise ValueError("No valid depth pixels found to seed blocks.")

        R_list, T_list, S_list, color_list = [], [], [], []
        half_w = max(1, downsample_factor // 2)

        for yx in valid_yx:
            y = int(yx[0].item())
            x = int(yx[1].item())

            d = depthmap[y, x]
            if not torch.isfinite(d) or d <= 0:
                continue

            # Unproject OpenCV -> camera coords
            x_cam_cv = (x - cx) * d / fx
            y_cam_cv = (y - cy) * d / fy
            z_cam_cv = d

            # Convert to PyTorch3D coords (flip X,Y)
            pos = torch.tensor([-x_cam_cv, -y_cam_cv, z_cam_cv], dtype=torch.float32, device=device)
            T_list.append(pos)

            # normal
            n_cam = normalmap[y, x, :]
            n_cam = F.normalize(n_cam, dim=0)

            # Build orthonormal frame: z = normal
            z_axis = n_cam
            # Choose a helper axis not parallel to n_cam
            if torch.abs(z_axis[1]) < 0.7:
                helper = torch.tensor([0.0, 1.0, 0.0], device=device)
            else:
                helper = torch.tensor([1.0, 0.0, 0.0], device=device)
            x_axis = helper - torch.dot(helper, z_axis) * z_axis
            x_axis = F.normalize(x_axis, dim=0)
            y_axis = torch.cross(z_axis, x_axis)
            y_axis = F.normalize(y_axis, dim=0)
            R_block_to_cam = torch.stack([x_axis, y_axis, z_axis], dim=1)
            R_list.append(R_block_to_cam)

            # scale heuristic
            scale = float(d) / float((fx + fy) / 2.0) * 32.0
            if self.scale_min > 0:
                scale = max(scale, float(self.scale_min))
            log_scale = math.log(scale / ratio_block_scene)
            S_list.append(torch.full((3,), log_scale, device=device, dtype=torch.float32))

            # average color window
            y0 = max(0, y - half_w)
            y1 = min(H, y + half_w + 1)
            x0 = max(0, x - half_w)
            x1 = min(W, x + half_w + 1)
            win = input_image[y0:y1, x0:x1, :]
            win_depth = depthmap[y0:y1, x0:x1]
            val_mask = win_depth > 0
            if val_mask.any():
                pix_col = win[val_mask].mean(dim=0)
            else:
                pix_col = input_image[y, x]
            pix_col = torch.clamp(pix_col, 0.0, 1.0)
            color_list.append(pix_col)

        if len(R_list) == 0:
            raise ValueError("All valid pixels filtered out; no blocks created.")

        # ------------------------------------------------------------------
        # Register & parameterize blocks
        # ------------------------------------------------------------------
        self.n_blocks = len(R_list)

        # plane_colors: re-use per-block color (new path) or carry forward legacy if present
        if plane_colors is None:
            plane_colors = torch.stack(color_list)  # (NB,3)
        self.register_buffer('plane_colors', plane_colors.to(device=device, dtype=torch.float32))

        self.register_buffer('block_faces_uvs', faces_uvs)
        self.register_buffer('block_verts_uvs', verts_uvs)

        # one unit icosphere per block
        unit_sphere = get_icosphere(level=1)
        self.blocks = join_meshes_as_batch(
            [unit_sphere.scale_verts(ratio_block_scene) for _ in range(self.n_blocks)]
        )

        # trainable params
        self.R_6d = nn.Parameter(matrix_to_rotation_6d(torch.stack(R_list)))
        self.T = nn.Parameter(torch.stack(T_list))
        self.S = nn.Parameter(torch.stack(S_list))

        # superquadric shape exponents
        self.sq_eps = nn.Parameter(torch.full((self.n_blocks, 2), -2.398, device=device))
        verts = self.blocks.verts_padded() / ratio_block_scene
        self.register_buffer('sq_eta', torch.asin(verts[..., 1]))
        self.register_buffer('sq_omega', torch.atan2(verts[..., 0], verts[..., 2]))

        # per-block textures (flat color)
        TS = self.txt_size
        maps = self.plane_colors[:, None, None].expand(-1, TS, TS, -1)
        self.register_buffer('textures', maps.float())

        # Opacity parameter (alpha_logit) if not already present
        if not hasattr(self, 'alpha_logit'):
            # logit(opacity_init)
            opacity_init = float(np.clip(opacity_init, 1e-6, 1.0 - 1e-6))
            alpha0 = math.log(opacity_init / (1.0 - opacity_init))
            self.alpha_logit = nn.Parameter(
                torch.full((self.n_blocks,), alpha0, dtype=torch.float32, device=device)
            )

        # Track that our textures are already linear [0,1]
        self.texture_free = True
    def build_blocks(self, filter_transparent=False, world_coord=False, as_scene=False, synthetic_colors=False):
        coarse_learning = self.training and self.is_live('coarse_learning')
        S, R, T = self.S.exp() + self.scale_min, rotation_6d_to_matrix(self.R_6d), self.T
        if self.opacity_noise and coarse_learning:
            alpha_logit = self.alpha_logit + self.opacity_noise * torch.randn_like(self.alpha_logit)
        else:
            alpha_logit = self.alpha_logit


        self._alpha = torch.sigmoid(alpha_logit)
        self._alpha_full = self._alpha.clone()  # this tensor won't be filtered / altered based on opacities
        maps = self.textures# torch.sigmoid(self.textures)
        if synthetic_colors:
            values = torch.linspace(0, 1, self.n_blocks + 1)[1:]
            colors = torch.from_numpy(get_fancy_cmap()(values.cpu().numpy())).float().to(maps.device)
            maps = colors[:, None, None].expand(-1, self.txt_size, self.txt_size, -1)

        # build_blocks 中替换掉那行
        if self.texture_free:                    # plane_colors 本身已是 [0,1]
            maps = self.textures                 # 直接用，不再 sigmoid
        else:
            maps = torch.sigmoid(self.textures)  # 旧分支保持不变
 
        verts = (self.get_blocks_verts() * S[:, None]) @ R + T[:, None].cuda()
        faces = self.blocks.faces_padded().cuda()
        self._blocks_maps, self._blocks_SRT = maps, (S, R, T)

        # Filter blocks based on opacities
        if filter_transparent or self.kill_blocks:
            if filter_transparent:
                mask = torch.sigmoid(self.alpha_logit) > 0.5
            else:
                mask = torch.sigmoid(self.alpha_logit) > 0.01
            self._alpha_full = self._alpha_full * mask
            NB = sum(mask).item()
            mask=mask.cuda()
            if NB == 0:
                verts, faces, maps = [], [], []
            else:
                verts, faces, maps, self._alpha = verts[mask], faces[mask], maps[mask], self._alpha[mask]
        else:
            NB = self.n_blocks

        # Regularization
        if len(maps) > 0 and coarse_learning:
            if self.is_live('decimate_txt'):
                sub_maps = F.avg_pool2d(maps.permute(0, 3, 1, 2), self.decim_factor, stride=self.decim_factor)
                maps = F.interpolate(sub_maps, scale_factor=self.decim_factor).permute(0, 2, 3, 1)

        # Build textures and meshes object
        verts_uvs = self.block_verts_uvs[None].expand(self.n_blocks, -1, -1)[:NB] if NB != 0 else []
        faces_uvs = self.block_faces_uvs[None].expand(self.n_blocks, -1, -1)[:NB] if NB != 0 else []
        if len(maps) > 0:
            p_left, p_right = self.txt_padding
            maps = F.pad(maps.permute(0, 3, 1, 2), pad=(p_left, p_right, 0, 0), mode='circular').permute(0, 2, 3, 1)
        txt = TexturesUV(maps, faces_uvs, verts_uvs, align_corners=True)
        if (world_coord or as_scene) and len(verts) > 0:
            verts = (verts.float() * self.S_world) @ self.R_world + self.T_world[:, None]
        blocks = Meshes(verts, faces, textures=txt)
        return join_meshes_as_scene(blocks) if (as_scene and len(blocks) > 0) else blocks

