"""
vis_normal.py  –  lazily loads Metric3D so the dataset can be pickled by
                  DataLoader workers (spawn/posix).

Only this file was changed; your training code stays the same.
"""
from dataclasses import dataclass, field
from typing import Type, Union
from pathlib import Path
import copy

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from neuralplane.frontend.monocular.basis import MonocularPredictorConfig
from neuralplane.frontend.monocular.snu import SNUConfig
from neuralplane.frontend.monocular.stable_normal import StableNormalConfig
from neuralplane.frontend.segment.sam_tools import setup_sam
from neuralplane.frontend.segment.mask_generator import infer_masks
from neuralplane.frontend.monocular.tools import (
    to_world_space,
    remove_small_isolated_areas,
    merge_normal_clusters,
)
import cv2
import torch
from moge.model.v1 import MoGeModel
# from moge.model.v2 import MoGeModel # Let's try MoGe-2
import math
from pathlib import Path
from typing import Union, Iterable, Tuple, Dict, List
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm

def merge_normal_clusters(pred, sorted_topk, centers):
    """
    Merge the normal clusters based on the distance between the centers of the clusters.
    """
    new_pred = copy.deepcopy(pred)
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)

    num_clusters = len(sorted_topk)
    flag = np.zeros(num_clusters)
    new_num_clusters = num_clusters

    for i in range(num_clusters):
        if flag[i] == 1:
            continue
        for j in range(i + 1, num_clusters):
            if flag[j] == 1:
                continue

            if np.dot(centers[sorted_topk[i]], centers[sorted_topk[j]]).sum() > 0.95:
                new_pred[pred == sorted_topk[j]] = sorted_topk[i]
                new_num_clusters -= 1
                flag[j] = 1

    if new_num_clusters != num_clusters:
        count_values = np.bincount(new_pred)
        topk = np.argpartition(count_values,-new_num_clusters)[-new_num_clusters:]
        sorted_topk_idx = np.argsort(count_values[topk])
        sorted_topk = topk[sorted_topk_idx][::-1]

    return new_pred, sorted_topk, new_num_clusters
# ---------------------------------------------------------------------
# helper ---------------------------------------------------------------
# ---------------------------------------------------------------------
def _pad_and_stack(list_of_arrays, pad_value=0):
    """Pads a list of (M_i,…,*) arrays to the same M=max(M_i) and stacks them.
    Returns tensor (N,max_M,…) and a boolean valid_mask (N,max_M)."""
    max_len = max(arr.shape[0] for arr in list_of_arrays) if list_of_arrays else 0
    if max_len == 0:
        return torch.empty(0), torch.empty(0, dtype=torch.bool)
    out   = []
    valid = []
    for arr in list_of_arrays:
        pad_size = (max_len - arr.shape[0],) + arr.shape[1:]
        pad_arr  = F.pad(
            torch.as_tensor(arr),
            (0, 0) * (arr.ndim - 1) + (0, pad_size[0]),  # pad on plane dimension
            value=pad_value,
        )
        out.append(pad_arr)
        valid.append(
            torch.arange(max_len) < arr.shape[0]
        )  # True where real, False where padded
    return torch.stack(out), torch.stack(valid)

def fov_x_from_cam_info(cam_info=None) -> float:
    """
    Estimate horizontal FoV (degrees) from camera intrinsics.

    Args:
        cam_info (dict): Must contain
            - 'focal_length_x'  (fx) in pixels
            - 'camera_center_x' (cx) in pixels
              (If you know the true image width, pass it instead.)

    Returns:
        float: Horizontal FoV in degrees.
    """
    if cam_info:
        fx = cam_info['focal_length_x']
        cx = cam_info['camera_center_x']

        # Treat 2*cx as the full image width (sensor is centered)
        width_px = 2.0 * cx

        # FoV_x = 2 * arctan( width / (2*fx) )
        fov_x_rad = 2.0 * math.atan(width_px / (2.0 * fx))
        return math.degrees(fov_x_rad)
    else:
        return None 


_NORMAL_MODEL = None

# ──────────────────────────────────────────────────────────────
# 1.  global lazy-loader (safe for multiprocessing “spawn” mode)
# ──────────────────────────────────────────────────────────────

def _load_metric3d(device: torch.device, arch: str = "metric3d_vit_large"):
    """
    Load the Metric3D network only once *inside* each process.

    Returns the cached model if it’s already loaded.
    """
    global _NORMAL_MODEL
    model_type = True
    if _NORMAL_MODEL is None:
      if model_type:
        try:
          model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
          mogegggg = 'v2'
        except:
          
          model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
          mogegggg = 'v1'
          
        model.eval().to(device)
        for p in model.parameters():  # we never back-prop through it
            p.requires_grad = False
        _NORMAL_MODEL = model
        return _NORMAL_MODEL, mogegggg

        """
        `output` has keys "points", "depth", "mask", "normal" (optional) and "intrinsics",
        The maps are in the same size as the input image. 
        {
            "points": (H, W, 3),    # point map in OpenCV camera coordinate system (x right, y down, z forward). For MoGe-2, the point map is in metric scale.
            "depth": (H, W),        # depth map
            "normal": (H, W, 3)     # normal map in OpenCV camera coordinate system. (available for MoGe-2-normal)
            "mask": (H, W),         # a binary mask for valid pixels. 
            "intrinsics": (3, 3),   # normalized camera intrinsics
        }
        """
      else:
        model = torch.hub.load(
            "yvanyin/metric3d", arch, pretrain=True, trust_repo=True
        )
        model.eval().to(device)
        for p in model.parameters():  # we never back-prop through it
            p.requires_grad = False
        _NORMAL_MODEL = model
    return _NORMAL_MODEL, mogegggg


# ──────────────────────────────────────────────────────────────
# 2.  main visualiser class
# ──────────────────────────────────────────────────────────────
class Vis(nn.Module):
    def __init__(self, pre_moge=None, device: str | torch.device = "cuda"):
        super().__init__()

        # ── hyper-parameters ───────────────────────────────────
        self.n_init_normal_clusters = 8
        self.n_normal_clusters = 6
        self.num_sam_prompts = 256
        self.min_plane_size = 50  # absolute pixel count
        self.input_size = (616, 1064)  # Metric3Dv2-ViT resolution


        self.fov_x = fov_x_from_cam_info(pre_moge)
        
        if isinstance(device, str):
            device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.device = device

        # SAM once (it *is* picklable)
        self.sam_model = setup_sam(device=self.device)

        # Metric3D loader placeholders
        self._normal_arch = "metric3d_vit_large"
        self._normal_model = None  # filled lazily in `_get_normal_model`

    # ──────────────────────────────────────────────────────────
    # helper: fetch the Metric3D model (lazy, per process)
    # ──────────────────────────────────────────────────────────
    def _get_normal_model(self):
        if self._normal_model is None:
            self._normal_model, mogegggg = _load_metric3d(self.device, self._normal_arch)
        return self._normal_model, mogegggg

    # ──────────────────────────────────────────────────────────
    # helper: resize / pad just like the official demo script
    # ──────────────────────────────────────────────────────────
    def _prepare_input(self, rgb_np: np.ndarray):
        h, w = rgb_np.shape[:2]
        in_h, in_w = self.input_size
        scale = min(in_h / h, in_w / w)

        # resize
        rgb_rs = cv2.resize(
            rgb_np, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR
        )

        # constant-colour pad to 616×1064
        mean_bgr = [123.675, 116.28, 103.53]  # BGR order
        pad_h = in_h - rgb_rs.shape[0]
        pad_w = in_w - rgb_rs.shape[1]
        top, bot = pad_h // 2, pad_h - pad_h // 2
        left, rgt = pad_w // 2, pad_w - pad_w // 2
        rgb_pad = cv2.copyMakeBorder(
            rgb_rs, top, bot, left, rgt, cv2.BORDER_CONSTANT, value=mean_bgr
        )

        # normalise (ImageNet stats)
        mean = np.asarray(mean_bgr, dtype=np.float32)
        std = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)
        rgb_ts = torch.from_numpy(rgb_pad.transpose(2, 0, 1)).float()
        rgb_ts = (rgb_ts - torch.from_numpy(mean)[:, None, None]) / torch.from_numpy(
            std
        )[:, None, None]
        rgb_ts = rgb_ts.unsqueeze(0).to(self.device)  # (1,3,616,1064)

        pad = [top, bot, left, rgt]
        return rgb_ts, pad, scale

    # ──────────────────────────────────────────────────────────
    # uint8 helper
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def to_uint8(arr: np.ndarray, *, assume_normal: bool = False) -> np.ndarray:
        out = arr.astype(np.float32)
        if assume_normal:
            out = (out + 1.0) * 0.5 * 255.0
        elif out.max() <= 1.0:
            out *= 255.0
        return np.clip(out, 0, 255).astype(np.uint8)


    def select_planes(self, plane_instances, masks_avg_normals, masks_areas, masks_depths, 
                    trim=None, strategy='closest', min_area_ratio=0.01):
        """
        Select K planes based on different strategies.
        
        Args:
            plane_instances: List of boolean masks for each plane
            masks_avg_normals: Array of average normals for each plane
            masks_areas: Array of areas for each plane
            masks_depths: Array of average depths for each plane
            trim (int, optional): Number of planes to keep. If None, keep all.
            strategy (str): Selection strategy. Options:
                - 'closest': Select K planes with smallest average depth
                - 'largest': Select K planes with largest area
                - 'frontal': Select K planes most perpendicular to camera (frontal facing)
                - 'mixed': Score based on combination of depth, area, and orientation
            min_area_ratio (float): Minimum area ratio relative to image size to consider
        
        Returns:
            Tuple of (selected_indices, selection_info)
        """
        n_planes = len(plane_instances)
        
        if trim is None or trim >= n_planes:
            return list(range(n_planes)), {"strategy": "all", "n_selected": n_planes}
        
        # Calculate selection scores based on strategy
        if strategy == 'closest':
            # Lower depth = higher score
            scores = -masks_depths
            
        elif strategy == 'largest':
            # Larger area = higher score
            scores = masks_areas.astype(float)
            
        elif strategy == 'frontal':
            # Planes facing camera (normal aligned with -Z) get higher scores
            # Assuming camera looks along +Z, frontal planes have normal ~ [0,0,-1]
            camera_dir = np.array([0, 0, -1])
            frontal_scores = np.array([
                np.dot(normal, camera_dir) for normal in masks_avg_normals
            ])
            scores = frontal_scores
            
        elif strategy == 'mixed':
            # Combine multiple factors
            # Normalize each factor to [0, 1]
            depth_scores = 1.0 - (masks_depths - masks_depths.min()) / (masks_depths.ptp() + 1e-8)
            area_scores = masks_areas / masks_areas.max()
            
            # Frontal facing score
            camera_dir = np.array([0, 0, -1])
            frontal_scores = np.array([
                max(0, np.dot(normal, camera_dir)) for normal in masks_avg_normals
            ])
            
            # Weighted combination
            scores = (
                0.4 * depth_scores +      # Prefer closer planes
                0.3 * area_scores +       # Prefer larger planes
                0.3 * frontal_scores      # Prefer frontal-facing planes
            )
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Filter out very small planes if requested
        if min_area_ratio > 0:
            total_pixels = plane_instances[0].size  # Assuming all masks have same size
            min_area = total_pixels * min_area_ratio
            valid_mask = masks_areas >= min_area
            
            # Set scores of invalid planes to -inf
            scores = np.where(valid_mask, scores, -np.inf)
        
        # Select top K planes
        selected_indices = np.argsort(scores)[::-1][:trim]
        selected_indices = sorted(selected_indices)  # Keep original order
        
        # Prepare selection info
        selection_info = {
            "strategy": strategy,
            "n_original": n_planes,
            "n_selected": len(selected_indices),
            "selected_indices": selected_indices,
            "scores": scores[selected_indices],
            "depth_range": [masks_depths[selected_indices].min(), 
                        masks_depths[selected_indices].max()],
            "area_range": [masks_areas[selected_indices].min(), 
                        masks_areas[selected_indices].max()],
        }
        
        return selected_indices, selection_info


    # ──────────────────────────────────────────────────────────
    # normal clustering
    # ──────────────────────────────────────────────────────────
    def _normals_cluster(self, normals: np.ndarray):
        kmeans = KMeans(
            n_clusters=self.n_init_normal_clusters, random_state=0, n_init=1
        ).fit(normals.reshape(-1, 3))
        pred = kmeans.labels_
        centers = kmeans.cluster_centers_

        # pick the n largest clusters
        counts = np.bincount(pred)
        topk = np.argpartition(counts, -self.n_normal_clusters)[-self.n_normal_clusters :]
        topk = topk[np.argsort(counts[topk])[::-1]]  # sort desc by size

        pred, topk, n_valid = merge_normal_clusters(pred, topk, centers)

        masks = []
        for idx in range(n_valid):
            m = pred == topk[idx]
            m_clean = remove_small_isolated_areas(
                (m > 0).reshape(*self.img_shape) * 255, min_size=self.min_plane_size
            ).reshape(-1)
            m[m_clean == 0] = 0

            num_lbl, lbl_img = cv2.connectedComponents(
                (m * 255).reshape(self.img_shape).astype(np.uint8)
            )
            for lab in range(1, num_lbl):
                masks.append(lbl_img == lab)
        return masks

    # ──────────────────────────────────────────────────────────
    # main call
    # ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def forward(
        self,
        image_path: Union[str, Path, Iterable[Union[str, Path]]],
        *,
        optim: bool = False,
        trim=None,
        strategy: str = "closest",
    ) -> Dict[str, torch.Tensor]:
        """Batched plane‑detection forward pass.

        Returns a dict **of tensors**.  Every tensor has batch‑dim **N** as axis‑0.
        """
        # --------------------------------------------------------------
        # 0 . normalise paths                                           
        # --------------------------------------------------------------
        if isinstance(image_path, (str, Path)):
            paths: List[Path] = [Path(image_path)]
        else:
            paths = [Path(p) for p in image_path]

        device = getattr(self, "device", "cuda")

        # per‑image collectors ----------------------------------------
        all_plane_masks, all_avg_normals, all_areas, all_depths, all_colors = (
            [],
            [],
            [],
            [],
            [],
        )
        masked_depths, masked_normals, masked_imgs, Ks = [], [], [], []

        # --------------------------------------------------------------
        # 1 . run pipeline one image at a time                          
        # --------------------------------------------------------------
        for pth in tqdm(paths):
            # -- read image -------------------------------------------
            
            input_image_org = cv2.cvtColor(cv2.imread(str(pth)), cv2.COLOR_BGR2RGB)
            input_image = (
                torch.tensor(input_image_org / 255.0, dtype=torch.float32, device=device)
                .permute(2, 0, 1)
            )
            
            self.img_shape = input_image.shape[1:]  # (H, W)

            # -- depth/normal inference -------------------------------
            mogemmmodel, mogegggg = self._get_normal_model()
            
            output = mogemmmodel.infer(input_image, fov_x=self.fov_x)
            output = {
                k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v)
                for k, v in output.items()
            }
            depth = output["depth"]               # (H,W)
            intrinsics = output["intrinsics"]     # (3,3)
            normals = output.get("normal")         # (H,W,3)

            sam_img = input_image.permute(1, 2, 0).detach().cpu().numpy()
            ## 
            if mogegggg == 'v1':
              normals = np.repeat(depth[..., None], 3, axis=-1)

            # -- plane segmentation ----------------------------------
            normal_clusters = self._normals_cluster(normals)
            keypts = torch.rand(self.num_sam_prompts, 2, device=device) * 2 - 1
            sam_out = infer_masks(
                self.sam_model,
                sam_img,
                keypoints=keypts,
                device=self.device,
                num_pts_active=0,
            )["masks"]
            masks = sorted(sam_out["masks"].cpu().numpy(), key=lambda m: m.sum())



            seg_mask = np.zeros(self.img_shape, np.uint8)
            count = 0
            for m in masks:
                for n_mask in normal_clusters:
                    inter = m & n_mask
                    if inter.sum() < self.min_plane_size:
                        continue
                    count += 1
                    seg_mask[inter] = count

            # gather per‑plane stats ----------------------------------
            plane_instances, masks_avg_normals, masks_areas = [], [], []
            for i in range(1, count + 1):
                mask_pi = seg_mask == i
                if mask_pi.sum() < self.min_plane_size:
                    continue
                plane_instances.append(mask_pi)
                masks_areas.append(mask_pi.sum())
                avg_n = normals[mask_pi].mean(0)
                avg_n /= np.linalg.norm(avg_n) + 1e-8
                masks_avg_normals.append(avg_n)

            masks_avg_normals = np.stack(masks_avg_normals) if masks_avg_normals else np.zeros((0, 3))
            masks_areas = np.asarray(masks_areas)
            masks_depths = np.asarray([depth[m].mean() for m in plane_instances]) if plane_instances else np.zeros(0)

            # ------------------ selection (trim/strategy) ------------
            selected_indices, _ = self.select_planes(
                plane_instances,
                masks_avg_normals,
                masks_areas,
                masks_depths,
                trim=trim,
                strategy=strategy,
            )
            plane_instances = [plane_instances[i] for i in selected_indices]
            masks_avg_normals = masks_avg_normals[selected_indices] if masks_avg_normals.size else masks_avg_normals
            masks_areas = masks_areas[selected_indices] if masks_areas.size else masks_areas
            masks_depths = masks_depths[selected_indices] if masks_depths.size else masks_depths

            # ------------------ masking for downstream ---------------
            big_keep_mask = np.ones_like(np.stack(plane_instances, axis=0).any(axis=0) if plane_instances else np.zeros_like(depth, dtype=bool))
            masked_depth = depth.copy()
            masked_normal = normals.copy()
            masked_img = input_image_org.copy()
            masked_depth[~big_keep_mask] = 0.0
            masked_normal[~big_keep_mask] = 0.0
            masked_img[~big_keep_mask] = 0.0

            # ------------------ plane colours (fixed palette) --------
            from neuralplane.utils.disp import (
                overlay_masks,
                ColorPalette,
                create_consistent_plane_visualization,
            )
            n_planes = len(plane_instances)
            palette = ColorPalette(num_of_colors=n_planes)
            _, plane_colors = create_consistent_plane_visualization(
                input_image_org, plane_instances, n_planes
            )

            H, W = self.img_shape
            plane_masks_ts = (
                torch.stack([torch.as_tensor(m, dtype=torch.bool, device=device) for m in plane_instances])
                if plane_instances
                else torch.zeros((0, H, W), dtype=torch.bool, device=device)
            )

            all_plane_masks.append(plane_masks_ts)
            all_avg_normals.append(torch.as_tensor(masks_avg_normals, dtype=torch.float32, device=device))
            all_areas.append(torch.as_tensor(masks_areas, dtype=torch.float32, device=device))
            all_depths.append(torch.as_tensor(masks_depths, dtype=torch.float32, device=device))
            all_colors.append(torch.as_tensor(plane_colors, dtype=torch.float32, device=device))

            masked_depths.append(torch.as_tensor(masked_depth, dtype=torch.float32, device=device))  # (H,W)
            masked_normals.append(torch.as_tensor(masked_normal, dtype=torch.float32, device=device).permute(2, 0, 1))
            masked_imgs.append(torch.as_tensor(masked_img, dtype=torch.float32, device=device).permute(2, 0, 1))
            Ks.append(torch.as_tensor(intrinsics, dtype=torch.float32, device=device))

        # -------------------------------------------------------------
        # 2 . batch‑ify all lists                                        
        # -------------------------------------------------------------
        plane_masks, plane_valid = _pad_and_stack(all_plane_masks, pad_value=False)
        avg_normals, _           = _pad_and_stack(all_avg_normals, pad_value=0.0)
        areas, _                 = _pad_and_stack(all_areas,        pad_value=0.0)
        depths, _                = _pad_and_stack(all_depths,       pad_value=0.0)
        plane_colors, _          = _pad_and_stack(all_colors,       pad_value=0.0)

        masked_depth_ts = torch.stack(masked_depths).unsqueeze(1)  # (N,1,H,W)
        masked_norm_ts  = torch.stack(masked_normals)              # (N,3,H,W)
        masked_img_ts   = torch.stack(masked_imgs)                 # (N,3,H,W)
        intrinsics_ts   = torch.stack(Ks)                          # (N,3,3)

        # -------------------------------------------------------------
        # 3 .  pack & return                                           
        # -------------------------------------------------------------
        # plane_instances, normals, masks_areas, depth_areas, depthmap, plane_colors, K_unit
        return [
            plane_masks, 
            avg_normals, 
            areas,
            depths,
            [
                masked_depth_ts,
                masked_norm_ts,
                masked_img_ts
            ],
            plane_colors,
            intrinsics_ts


        ]
        return {
            "plane_masks": plane_masks,      # bool  (N,max_planes,H,W)
            "plane_valid": plane_valid,      # bool  (N,max_planes)
            "avg_normals": avg_normals,      # float (N,max_planes,3)
            "areas": areas,                  # float (N,max_planes)
            "depths": depths,                # float (N,max_planes)
            "plane_colors": plane_colors,    # float (N,max_planes,3)
            "masked_depth": masked_depth_ts, # float (N,1,H,W)
            "masked_normals": masked_norm_ts,# float (N,3,H,W)
            "masked_image": masked_img_ts,   # float (N,3,H,W)
            "intrinsics": intrinsics_ts,     # float (N,3,3)
        }

# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    vis_model = Vis()
    vis_model(image_path="stairs.png")