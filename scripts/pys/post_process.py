
import argparse
import os
from pathlib import Path
from smpl import SMPL
import imageio as iio
import numpy as np
import torch
import cv2
from scipy.ndimage import center_of_mass
from chamferdist import ChamferDistance
'''  np.savez(
      ,
      images=np.uint8(img_data_pt.cpu().numpy().transpose(0, 2, 3, 1) * 255.0),
      depths=np.clip(np.float16(1.0 / disp_data_opt), 1e-3, 1e2),
      intrinsic=K_o.detach().cpu().numpy(),
      cam_c2w=cam_c2w.detach().cpu().numpy(),
  )
  
'''
def shift_mask(mask, x_shift, y_shift):
    h, w = mask.shape
    shifted_mask = np.zeros_like(mask)

    # Calculate source and destination coordinates
    src_x_start = max(0, -x_shift)
    src_x_end = min(w, w - x_shift)  # original
    dst_x_start = max(0, x_shift)
    dst_x_end = min(w, w + x_shift)

    src_y_start = max(0, -y_shift)
    src_y_end = min(h, h - y_shift)
    dst_y_start = max(0, y_shift)
    dst_y_end = min(h, h + y_shift)

    # Copy shifted region
    shifted_mask[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
        mask[src_y_start:src_y_end, src_x_start:src_x_end]

    return shifted_mask

def unproject_depth_to_3d(vanila_depth, valid_mask, intrinsics):
    """
    Unprojects valid depth pixels into 3D camera coordinates.
    Inputs:
        vanila_depth: (H, W) tensor of depths
        valid_mask: (H, W) boolean indicating valid depth pixels
        intrinsics: [fx, fy, cx, cy]
    Returns:
        source_cloud: (N, 3) 3D points in camera coordinates
    """
    fx, fy, cx, cy = intrinsics
    H, W = vanila_depth.shape
    
    # Create pixel coordinate grids
    vs = torch.arange(H, device=vanila_depth.device)
    us = torch.arange(W, device=vanila_depth.device)
    v_grid, u_grid = torch.meshgrid(vs, us, indexing='xy')  # shape [H, W]
    
    # Flatten
    u_flat = u_grid.reshape(-1)  # [H*W]
    v_flat = v_grid.reshape(-1)  # [H*W]
    depth_flat = vanila_depth.reshape(-1)  # [H*W]
    valid_flat = valid_mask.reshape(-1)  # [H*W]
    
    # Keep only valid
    u_valid = u_flat[valid_flat]
    v_valid = v_flat[valid_flat]
    d_valid = depth_flat[valid_flat]
    
    # Unproject using pinhole camera model
    X = (u_valid - cx) / fx * d_valid
    Y = (v_valid - cy) / fy * d_valid
    Z = d_valid
    
    # Stack into (N, 3)
    source_cloud = torch.stack([X, Y, Z], dim=-1)  # shape [N, 3]
    return source_cloud

def calculate_centroid(point_cloud: torch.Tensor) -> torch.Tensor:
    """
    Calculate the centroid of a point cloud.
    Args:
        point_cloud (torch.Tensor): A tensor of shape [N, 3] representing the point cloud.
    Returns:
        torch.Tensor: A tensor of shape [3] representing the centroid.
    """
    return torch.mean(point_cloud, dim=0)

def get_mask_corner(mask):
  ys, xs = np.where(mask != 0)

  # Top-left: min x, min y
  top_left = (ys.min(), xs.min())

  # Top-right: min y, max x
  top_right = (ys.min(), xs.max())

  # Bottom-left: max y, min x
  bottom_left = (ys.max(), xs.min())

  # Bottom-right: max y, max x
  bottom_right = (ys.max(), xs.max())

  return top_left, top_right, bottom_left, bottom_right
def overlay_masks(mask1, mask2):
    h, w = mask1.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    # Red: valid_mask
    vis[mask1 > 0] = [0, 0, 255]
    # Green: best_mask
    vis[mask2 > 0] = [0, 255, 0]
    # Yellow overlap: both masks
    vis[(mask1 > 0) & (mask2 > 0)] = [0, 255, 255]
    return vis
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--output_dir", type=str, default="outputs_cvd", help="outputs direcotry"
  )
  parser.add_argument(
      "--input_dir", type=str, default="outputs_cvd", help="outputs direcotry"
  )
  parser.add_argument(
      "--method", type=str, default="median", help="outputs direcotry"
  )
  parser.add_argument(
      "--hmr_type", type=str, default="gv", help="outputs direcotry"
  )

  parser.add_argument("--scene_name", type=str, help="scene name")

  args = parser.parse_args()
  root_path = '/data3/zihanwa3/_Robotics/_data/toy_exp_msk'
  output_dir = args.output_dir
  scene_name = args.scene_name 
  print(output_dir, scene_name)

  # saved_path="%s/%s_sgd_cvd_hr.npz" % (output_dir, scene_name)
  saved_path="%s/%s.npz" % (output_dir, scene_name)

  vanila_data = np.load(saved_path)
  images = vanila_data['images']
  depths = vanila_data['depths']
  cam_c2w = vanila_data['cam_c2w']
  intrinsic = vanila_data['intrinsic']
  # uncertainty = vanila_data['uncertainty']

  scales = []
  shifts = []
  masks = []
  mask_shifts = []
  gt_masks = []
  mono_disp_list = []

  valid_indices = []
  shifts = []
  mono_disp_list = []


  seq_folder = '../../results/init/hmr'
  mask_folder = f'../../results/init/dyn_mask/{scene_name}/person'
  cam_folder = f'../../results/init/vslam/{scene_name}/camera.npy'



  for i in range(0, len(images)):
      depth_path = f'{seq_folder}/depth_out/mesh_depth_{i}.npy'
      
      # Check if depth file exists
      if not os.path.exists(depth_path):
          print(f"Warning: Depth file not found for frame {i}, skipping...")
          continue
          
      tgt_depth = np.load(depth_path)
      vanila_depth = depths[i]
      
      # Resize target depth to match vanilla depth dimensions
      tgt_depth = cv2.resize(
          tgt_depth, 
          vanila_depth.shape[::-1], 
          interpolation=cv2.INTER_NEAREST
      )
      
      # Check if depths are valid before processing
      if np.isnan(tgt_depth).any() or np.isnan(vanila_depth).any():
          print(f"Warning: NaN values detected in frame {i}, skipping...")
          continue
      
      mono_disp_list.append(vanila_depth)
      
      # Load mask
      mask_path = f'{root_path}/{scene_name}/person/dyn_mask_{i}.npz'
      if not os.path.exists(mask_path):
          print(f"Warning: Mask file not found for frame {i}, skipping...")
          continue
          
      mask = np.load(mask_path)['dyn_mask'][0]
      target_size = tgt_depth.shape[::-1]
      mask = cv2.resize(mask.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)
      
      # Create valid mask - only consider positive depth values
      valid_mask = (tgt_depth > 0.0) & (vanila_depth > 0.0) & np.isfinite(tgt_depth) & np.isfinite(vanila_depth)
      # valid_mask = (tgt_depth > 0.0) & (vanila_depth > 0.0) & np.isfinite(tgt_depth) & np.isfinite(vanila_depth)
      
      # Check if we have enough valid pixels
      if valid_mask.sum() < 20:  # Minimum threshold of valid pixels
          print(f"Warning: Not enough valid depth pixels in frame {i} ({valid_mask.sum()} pixels), skipping...")
          continue
      
      valid_dus_depth = tgt_depth[valid_mask]
      valid_mons_depth = vanila_depth[valid_mask]
      
      gt_disp = valid_dus_depth
      da_disp = valid_mons_depth
      
      # Calculate scale robustly
      with np.errstate(divide='ignore', invalid='ignore'):
          scale_candidates = gt_disp / da_disp
          # Filter out invalid scale values
          valid_scales = scale_candidates[np.isfinite(scale_candidates) & (scale_candidates > 0)]
          
          if len(valid_scales) > 0:
              scale = np.median(valid_scales)
              shift = 0
              
              # Sanity check on scale value
              if 0.1 < scale < 20.0:  # Reasonable scale range
                  scales.append(scale)
                  shifts.append(shift)
                  valid_indices.append(i)
              else:
                  print(f"Warning: Unreasonable scale {scale:.3f} for frame {i}, skipping...")
          else:
              print(f"Warning: No valid scale could be computed for frame {i}, skipping...")

  # Check if we have any valid frames
  if len(scales) == 0:
      raise ValueError("No valid frames found! Check your depth data.")

  # Compute alignment parameters from valid frames only
  ss_product = np.array(scales)
  print(f"Valid scales from {len(scales)} frames: {ss_product}")
  med_idx = np.argmin(np.abs(ss_product - np.median(ss_product)))
  align_scale = scales[med_idx]
  align_shift = shifts[med_idx]

  print(f"Alignment scale: {align_scale:.3f}, shift: {align_shift:.3f}")

  # Process video masks if available
  video_dir = os.path.join('/data3/zihanwa3/_Robotics/_vision/mega-sam/_visuals', scene_name)
  os.makedirs(video_dir, exist_ok=True)

  if len(masks) > 0 and len(gt_masks) > 0:
      try:
          gt_mask_video = torch.tensor(gt_masks).float()
          mask_video = torch.tensor(masks)
          if gt_mask_video.ndim == 3:
              gt_mask_video = gt_mask_video.unsqueeze(1)
          if mask_video.ndim == 3:
              mask_video = mask_video.unsqueeze(1)
          
          T, _, H, W = mask_video.shape
          gt_mask_video = F.interpolate(gt_mask_video, size=(H, W), mode='nearest').bool()
          
          combined_video = torch.cat((gt_mask_video, mask_video), dim=-1).squeeze(1)
          iio.mimwrite(
              os.path.join(video_dir, "masks.mp4"),
              (combined_video.numpy() * 255).astype(np.uint8),
              fps=15,
          )
      except Exception as e:
          print(f"Warning: Could not create mask video: {e}")

  # Process all frames with the computed alignment
  aligns = (align_scale, align_shift)
  depths_aligned = []
  masks_aligned = []
  obj_masks_aligned = []
  valid_images = []
  valid_cam_c2w = []
  valid_uncertainty = []

  for idx, i in enumerate(range(len(mono_disp_list))):
      if i not in valid_indices:
          continue
          
      mono_disp = mono_disp_list[idx]
      scale, shift = aligns[0], aligns[1]
      
      # Apply alignment and clip to reasonable range
      depth = np.clip(
          (scale * mono_disp + shift),
          1e-9,
          1e4,
      )
      
      # Final validity check
      if np.isnan(depth).any() or np.isinf(depth).any():
          print(f"Warning: Invalid depth values after alignment for frame {i}, skipping...")
          continue
      
      depths_aligned.append(depth)
      valid_images.append(images[i])
      valid_cam_c2w.append(cam_c2w[i])
      if 'uncertainty' in locals() and uncertainty is not None:
          valid_uncertainty.append(uncertainty[i])
      
      # Load masks
      mask_path = f'{root_path}/{scene_name}/person/dyn_mask_{i}.npz'
      if os.path.exists(mask_path):
          mask = np.load(mask_path)['dyn_mask'][0]
          masks_aligned.append(mask)
      else:
          masks_aligned.append(np.zeros_like(depth, dtype=bool))
      
      # Try to load object masks
      try:
          obj_path = f'{root_path}/{scene_name}/door/dyn_mask_{i}.npz'
          obj_mask = np.load(obj_path)['dyn_mask'][0]
          obj_masks_aligned.append(obj_mask)
      except:
          pass

  # Create output directory
  output_dir = 'output'
  os.makedirs(seq_folder, exist_ok=True)
  os.makedirs(output_dir, exist_ok=True)

  # Save only valid data
  print(f"Saving {len(depths_aligned)} valid frames out of {len(images)} total frames")

  save_dict = {
      "scale": align_scale,
      "images": np.array(valid_images),
      "depths": np.array(depths_aligned),
      "intrinsic": intrinsic,
      "cam_c2w": np.array(valid_cam_c2w),
      "mask_shifts": mask_shifts,
      "enlarged_dynamic_mask": np.array(masks_aligned),
      "valid_frame_indices": np.array(valid_indices),  # Track which frames were valid
  }

  # Add optional fields if they exist
  if len(obj_masks_aligned) > 0:
      save_dict["obj_masks"] = np.array(obj_masks_aligned)

  if 'uncertainty' in locals() and len(valid_uncertainty) > 0:
      save_dict["uncertainty"] = np.array(valid_uncertainty)

  # Save the processed data
  output_path = "%s/%s_%s_sgd_cvd_hr.npz" % (output_dir, scene_name, args.hmr_type)
  np.savez(output_path, **save_dict)

  print(f"Successfully saved to {output_path}")
  print(f"Valid frames: {len(valid_indices)}/{len(images)}")
  print(f"Alignment scale: {align_scale:.3f}")



  
