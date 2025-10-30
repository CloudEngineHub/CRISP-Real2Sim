import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
import torch
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import DBSCAN
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
from collections import defaultdict
import re

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import hsv_to_rgb
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

from PIL import Image
import torch
import open3d as o3d
import open3d.core as o3c
import numpy as np
from math import ceil
from sklearn.cluster import DBSCAN
from math import ceil
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import colorsys
from sklearn.cluster import KMeans
from pathlib import Path
import copy
from math import ceil
import colorsys
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from typing import Tuple
import numpy as np
from scipy.spatial import cKDTree

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
from copy import copy
from typing import Dict, List

_PRIM_KEYS = ["S_items", "R_items", "T_items", "pts_items"]
PART_ORDER = ["leg", "hand", "gluteus", "back", "thigh"]

import numpy as np
import torch
from typing import Dict, List, Union

ArrayLike = Union[np.ndarray, torch.Tensor]

def _to_tensor(x: ArrayLike, device: torch.device, dtype: torch.dtype, non_blocking=False) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype, non_blocking=non_blocking)
    return torch.as_tensor(x, dtype=dtype, device=device)

def scene_results_to_torch(
    merged_results: Dict[str, List[ArrayLike]],
    *,
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.float32,
    S_from: str = "half",      # "half"=given half-sizes -> store log(half); "full"=full sizes; "raw"=already log(half)
    expect_stacked: bool = False,  # True if S/R/T came stacked as arrays; otherwise lists are fine
    non_blocking: bool = False
) -> Dict[str, List[torch.Tensor]]:
    """
    Convert a (possibly NumPy) scene/merged_results dict back to torch Tensors on `device`.

    Inputs convention:
      S_from:
        - "half": S_items are half-sizes -> we store log(half) to be consistent with your pipeline
        - "full": S_items are full sizes -> convert to log(half) via log(full/2)
        - "raw" : S_items already are log(half) -> store as-is
      expect_stacked:
        - If True, allows S_items: (N,3), R_items: (N,3,3), T_items: (N,3) and will split to lists.
        - If False, keeps per-item lists.

    Returns (all lists of torch tensors on `device`):
      { 'S_items': [3], 'R_items':[3,3], 'T_items':[3], 'pts_items':[Ni,3], ... }
    """
    device = torch.device(device)
    out: Dict[str, List[torch.Tensor]] = {}

    def split_or_iter(a, axis=0):
        if expect_stacked and isinstance(a, (np.ndarray, torch.Tensor)):
            # split stacked arrays into list along axis 0
            if isinstance(a, np.ndarray):
                a = torch.from_numpy(a)
            return [a[i] for i in range(a.shape[axis])]
        return a  # assume already a list

    # --- S_items ---
    S_in = merged_results.get("S_items", [])
    S_seq = split_or_iter(S_in)
    S_items = []
    for S in S_seq:
        St = _to_tensor(S, device, dtype, non_blocking)
        if S_from == "half":
            Slog = torch.log(St.clamp_min(1e-12))
        elif S_from == "full":
            Slog = torch.log((St * 0.5).clamp_min(1e-12))
        elif S_from == "raw":
            Slog = St
        else:
            raise ValueError("S_from must be 'half', 'full', or 'raw'")
        S_items.append(Slog)
    out["S_items"] = S_items

    # --- R_items ---
    R_in = merged_results.get("R_items", [])
    R_seq = split_or_iter(R_in)
    R_items = []
    for R in R_seq:
        Rt = _to_tensor(R, device, dtype, non_blocking)
        R_items.append(Rt)
    out["R_items"] = R_items

    # --- T_items ---
    T_in = merged_results.get("T_items", [])
    T_seq = split_or_iter(T_in)
    T_items = []
    for T in T_seq:
        Tt = _to_tensor(T, device, dtype, non_blocking)
        T_items.append(Tt)
    out["T_items"] = T_items

    # --- pts_items (variable-length, keep lists) ---
    P_in = merged_results.get("pts_items", [])
    P_items = []
    for P in P_in:
        Pt = _to_tensor(P, device, dtype, non_blocking)
        P_items.append(Pt)
    out["pts_items"] = P_items

    # pass through extras, converting arrays/tensors → tensors on device
    for k, v in merged_results.items():
        if k in ("S_items", "R_items", "T_items", "pts_items"):
            continue
        if isinstance(v, list):
            out[k] = [_to_tensor(x, device, dtype, non_blocking) if isinstance(x, (np.ndarray, torch.Tensor)) else x
                      for x in v]
        elif isinstance(v, (np.ndarray, torch.Tensor)):
            out[k] = _to_tensor(v, device, dtype, non_blocking)
        else:
            out[k] = v

    return out

def _as_list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return v
    # if a single tensor/array sneaks in, wrap it
    return [v]

def merge_primitives_dicts(a: Dict, b: Dict, keys: List[str] = None) -> Dict:
    """
    Robustly merge two primitive dicts:
      {'S_items': [..], 'R_items': [..], 'T_items': [..], 'pts_items': [..]}
    Other keys are passed through from `a` (and added from `b` if missing).
    """
    if keys is None:
        keys = _PRIM_KEYS

    out = {}

    # First copy everything from a (shallow is fine since we'll replace list keys)
    for k, v in (a or {}).items():
        out[k] = v

    # Add any extra non-primitive keys from b that aren't present
    for k, v in (b or {}).items():
        if k not in keys and k not in out:
            out[k] = v

    # Now merge list-valued primitive keys
    for k in keys:
        la = _as_list((a or {}).get(k))
        lb = _as_list((b or {}).get(k))
        # Force creation of a brand-new list object
        out[k] = list(la) + list(lb)

    # Sanity check: lengths should match across S/R/T
    nS, nR, nT = len(out["S_items"]), len(out["R_items"]), len(out["T_items"])
    if not (nS == nR == nT):
        raise ValueError(f"Merged lengths mismatch: |S|={nS}, |R|={nR}, |T|={nT}")

    return out

def build_global_segments_single_view(all_frame_segments):
    """
    Build global_segments for a single fixed image.
    
    Args:
        all_frame_segments: List with one element containing the segments from the single frame
                           Format: [{ seg_id: {properties}, ... }]
    
    Returns:
        global_segments: Dict mapping global_id to list of (frame_idx, local_seg_id) tuples
    """
    global_segments = {}
    
    # Since we have only one frame (index 0)
    frame_idx = 0
    frame_segments = all_frame_segments[0]  # Get the single frame's segments
    
    # Each local segment becomes its own global segment
    global_id = 0
    for local_seg_id in frame_segments.keys():
        global_segments[global_id] = [(frame_idx, local_seg_id)]
        global_id += 1
    
    return global_segments

from copy import deepcopy
from typing import Dict, List, Tuple, Optional

def attach_contacts_and_return(
    results: Dict[str, List],
    results_extra: Optional[Dict[str, List]] = None,
    drop_unconnected: bool = False,
    in_place: bool = False,
) -> Tuple[Dict[str, List], Dict]:
    """
    Snap contact primitives to scene primitives, update results, and return:
      - updated_results
      - report dict with details

    Args
    ----
    results: dict with keys ['S_items','R_items','T_items', 'pts_items', ...]
    results_extra: dict for contact primitives that were merged into results
    drop_unconnected: if True, remove any contacts that failed to attach
    in_place: if True, modify `results` directly; else work on a deep copy

    Returns
    -------
    updated_results, report
      report = {
        'n_scene': int,
        'n_contact': int,
        'n_attached': int,
        'attached': List[{'idx': int, 'mode': str}],
        'failed':   List[int],
      }
    """
    if results_extra is None or len(results_extra.get('S_items', [])) == 0:
        # nothing to attach
        return (results if in_place else deepcopy(results)), {
            'n_scene': len(results.get('S_items', [])),
            'n_contact': 0,
            'n_attached': 0,
            'attached': [],
            'failed': [],
        }

    out = results if in_place else deepcopy(results)

    # 1) Build scene list (all non-contact primitives are scene)
    total = len(out['S_items'])
    n_contact = len(results_extra['S_items'])
    start_idx = total - n_contact

    scene = []
    for S, R, T in zip(out['S_items'][:start_idx], out['R_items'][:start_idx], out['T_items'][:start_idx]):
        scene.append({
            'S': torch.exp(S).detach().cpu().numpy(),  # S_items are log-half-sizes in your code
            'R': R.detach().cpu().numpy(),
            'T': T.detach().cpu().numpy(),
        })

    # 2) Build contact list (the tail of results)
    contact_idxs = list(range(start_idx, total))
    contact = []
    for i in contact_idxs:
        S = out['S_items'][i]; R = out['R_items'][i]; T = out['T_items'][i]
        contact.append({
            'S': torch.exp(S).detach().cpu().numpy(),
            'R': R.detach().cpu().numpy(),
            'T': T.detach().cpu().numpy(),
            'idx': i
        })

    # 3) Snap each contact to the scene
    attached_meta = []
    failed = []
    for C in contact:
        C_snap, info = snap_contact_to_scene(C, scene)
        if info and info.get('connected', False):
            i = C['idx']
            # write back snapped R/T/S (store S as log again to be consistent)
            out['R_items'][i] = torch.from_numpy(C_snap['R'])
            out['T_items'][i] = torch.from_numpy(C_snap['T'])
            out['S_items'][i] = torch.log(torch.from_numpy(C_snap['S']))
            attached_meta.append({'idx': i, 'mode': info.get('mode', 'unknown')})
        else:
            failed.append(C['idx'])

    # 4) Optionally drop unconnected contacts (keep lists aligned across keys)
    if drop_unconnected and failed:
        keep_mask = [True]*total
        for i in failed:
            keep_mask[i] = False
        # apply to all primitive-aligned lists
        for k in ['S_items','R_items','T_items','pts_items']:
            if k in out:
                out[k] = [v for v, keep in zip(out[k], keep_mask) if keep]

    report = {
        'n_scene': len(scene),
        'n_contact': n_contact,
        'n_attached': len(attached_meta),
        'attached': attached_meta,
        'failed': failed,
    }
    return out, report


def visualize_single_view_results(
    seg_map: np.ndarray,
    seg_props: Dict[int, Dict],
    global_segments: Dict[int, List[Tuple[int, int]]],
    primitives: Dict[str, List],
    depth_image: torch.Tensor,
    normal_image: torch.Tensor,
    save_dir: Path,
    frame_idx: int = 0,
    rgb_image: Optional[np.ndarray] = None,
    show_3d: bool = True
) -> None:
    """
    Comprehensive visualization for single view segmentation and primitive fitting.
    
    Args:
        seg_map: (H, W) segmentation map
        seg_props: Dictionary of segment properties
        global_segments: Global segment groupings
        primitives: Fitted primitives (S_items, R_items, T_items, pts_items)
        depth_image: (H, W) depth map
        normal_image: (H, W, 3) normal map
        save_dir: Directory to save visualizations
        frame_idx: Frame index for labeling
        rgb_image: Optional RGB image for overlay
        show_3d: Whether to generate 3D visualizations
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.patches import Rectangle
    import matplotlib.patches as mpatches
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    H, W = seg_map.shape
    
    # Helper function to generate distinct colors
    def get_colors(n):
        if n <= 20:
            return cm.get_cmap('tab20')(np.linspace(0, 1, n))
        else:
            return cm.get_cmap('hsv')(np.linspace(0, 0.9, n))
    
    # 1. Create main visualization figure
    fig = plt.figure(figsize=(20, 12))
    
    # 2. Original RGB/Depth view
    ax1 = plt.subplot(2, 4, 1)
    if rgb_image is not None:
        ax1.imshow(rgb_image)
        ax1.set_title('RGB Image')
    else:
        depth_vis = depth_image.cpu().numpy()
        depth_vis = np.clip(depth_vis, 0, np.percentile(depth_vis[depth_vis > 0], 95))
        ax1.imshow(depth_vis, cmap='viridis')
        ax1.set_title('Depth Map')
    ax1.axis('off')
    
    # 3. Normal visualization
    ax2 = plt.subplot(2, 4, 2)
    normal_vis = (normal_image.cpu().numpy() + 1) / 2  # Convert from [-1,1] to [0,1]
    normal_vis = np.clip(normal_vis, 0, 1)
    ax2.imshow(normal_vis)
    ax2.set_title('Surface Normals')
    ax2.axis('off')
    
    # 4. Per-pixel segmentation
    ax3 = plt.subplot(2, 4, 3)
    unique_segments = np.unique(seg_map)
    valid_segments = unique_segments[unique_segments >= 0]
    
    if len(valid_segments) > 0:
        colors = get_colors(len(valid_segments))
        seg_colored = np.zeros((*seg_map.shape, 3))
        for idx, seg_id in enumerate(valid_segments):
            mask = seg_map == seg_id
            seg_colored[mask] = colors[idx][:3]
    else:
        seg_colored = np.zeros((*seg_map.shape, 3))
    
    ax3.imshow(seg_colored)
    ax3.set_title(f'Segmentation ({len(valid_segments)} segments)')
    ax3.axis('off')
    
    # 5. Segment properties overlay
    ax4 = plt.subplot(2, 4, 4)
    ax4.imshow(seg_colored)
    
    # Draw segment centroids and IDs
    for seg_id, props in seg_props.items():
        # Get 2D centroid from pixels
        pixels = np.array(props['pixels'])
        if len(pixels) > 0:
            centroid_2d = pixels.mean(axis=0)
            ax4.plot(centroid_2d[1], centroid_2d[0], 'w*', markersize=8, markeredgecolor='k')
            ax4.text(centroid_2d[1], centroid_2d[0]-5, str(seg_id), 
                    color='white', fontsize=8, ha='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    ax4.set_title('Segment IDs & Centroids')
    ax4.axis('off')
    
    # 6. Primitive visualization (if available)
    ax5 = plt.subplot(2, 4, 5)
    if 'S_items' in primitives and len(primitives['S_items']) > 0:
        # Create a primitive overlay visualization
        prim_overlay = np.zeros((*seg_map.shape, 3))
        prim_colors = get_colors(len(primitives['S_items']))
        
        # Map primitives back to segments
        for prim_idx, (S, R, T) in enumerate(zip(primitives['S_items'], 
                                                  primitives['R_items'], 
                                                  primitives['T_items'])):
            # Find which segments belong to this primitive
            # This is simplified - you might need to track the mapping
            color = prim_colors[prim_idx][:3]
            
            # For visualization, we'll color segments that were merged into this primitive
            # This requires tracking which global segment produced which primitive
            if prim_idx < len(valid_segments):
                mask = seg_map == valid_segments[prim_idx]
                prim_overlay[mask] = color
        
        ax5.imshow(prim_overlay)
        ax5.set_title(f'Fitted Primitives ({len(primitives["S_items"])} boxes)')
    else:
        ax5.imshow(seg_colored)
        ax5.set_title('No Primitives Fitted')
    ax5.axis('off')
    
    # 7. Statistics panel
    ax6 = plt.subplot(2, 4, 6)
    ax6.axis('off')
    
    stats_text = f"Frame: {frame_idx}\n"
    stats_text += f"Resolution: {W}×{H}\n"
    stats_text += f"Segments: {len(valid_segments)}\n"
    stats_text += f"Primitives: {len(primitives.get('S_items', []))}\n\n"
    
    # Segment statistics
    stats_text += "Segment Sizes:\n"
    for seg_id in valid_segments[:5]:  # Show first 5
        if seg_id in seg_props:
            n_pixels = len(seg_props[seg_id]['pixels'])
            stats_text += f"  Seg {seg_id}: {n_pixels} pixels\n"
    
    if len(valid_segments) > 5:
        stats_text += f"  ... and {len(valid_segments)-5} more\n"
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 8. Normal clustering visualization
    ax7 = plt.subplot(2, 4, 7)
    
    # Show normal clusters by coloring segments based on their average normal
    normal_cluster_vis = np.zeros((*seg_map.shape, 3))
    for seg_id, props in seg_props.items():
        mask = seg_map == seg_id
        # Convert normal to RGB color (shift from [-1,1] to [0,1])
        normal_color = (props['avg_normal'].cpu().numpy() + 1) / 2
        normal_cluster_vis[mask] = normal_color
    
    ax7.imshow(normal_cluster_vis)
    ax7.set_title('Normal-based Coloring')
    ax7.axis('off')
    
    # 9. Boundary visualization
    ax8 = plt.subplot(2, 4, 8)
    
    # Create boundary image
    from scipy import ndimage
    boundary_img = np.zeros((H, W))
    for seg_id in valid_segments:
        mask = seg_map == seg_id
        eroded = ndimage.binary_erosion(mask)
        boundary = mask & ~eroded
        boundary_img[boundary] = 1
    
    ax8.imshow(seg_colored)
    ax8.contour(boundary_img, colors='white', linewidths=1)
    ax8.set_title('Segment Boundaries')
    ax8.axis('off')
    
    plt.suptitle(f'Single View Segmentation Results - Frame {frame_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save main figure
    main_save_path = save_dir / 'single_view_complete.png'
    plt.savefig(main_save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved main visualization to {main_save_path}")
    
    # Additional individual visualizations
    
    # 10. Save individual segmentation map with legend
    fig2, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(seg_colored)
    ax.set_title(f'Segmentation Map - {len(valid_segments)} Segments')
    ax.axis('off')
    
    # Create legend
    if len(valid_segments) <= 20:
        patches = []
        colors = get_colors(len(valid_segments))
        for idx, seg_id in enumerate(valid_segments):
            if seg_id in seg_props:
                n_pixels = len(seg_props[seg_id]['pixels'])
                label = f'Seg {seg_id} ({n_pixels} px)'
                patches.append(mpatches.Patch(color=colors[idx][:3], label=label))
        
        ax.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5), 
                 fontsize=8, title='Segments')
    
    seg_save_path = save_dir / 'segmentation_with_legend.png'
    plt.savefig(seg_save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved segmentation with legend to {seg_save_path}")

def filter_bg_points_by_human_distance(
    bg_position: np.ndarray,
    bg_color: np.ndarray,
    human_transl_np: np.ndarray,
    depth,
    real_normal: np.ndarray | None = None,
    max_dist: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    """
    tree = cKDTree(human_transl_np)
    distances, _ = tree.query(bg_position, k=1)          # nearest-neighbour distance
    mask = distances <= max_dist
    depth[~mask] = 0
    if real_normal is None: 
      return bg_position[mask], bg_color[mask], depth# , real_normal[mask]
    else:
      return bg_position[mask], bg_color[mask], depth, real_normal[mask]
def build_global_segments_single_view(all_frame_segments):
    """
    Build global_segments for a single fixed image.
    
    Args:
        all_frame_segments: List with one element containing the segments from the single frame
                           Format: [{ seg_id: {properties}, ... }]
    
    Returns:
        global_segments: Dict mapping global_id to list of (frame_idx, local_seg_id) tuples
    """
    global_segments = {}
    
    # Since we have only one frame (index 0)
    frame_idx = 0
    frame_segments = all_frame_segments[0]  # Get the single frame's segments
    
    # Each local segment becomes its own global segment
    global_id = 0
    for local_seg_id in frame_segments.keys():
        global_segments[global_id] = [(frame_idx, local_seg_id)]
        global_id += 1
    
    return global_segments

def visualize_single_view_results(
    seg_map: np.ndarray,
    seg_props: Dict[int, Dict],
    global_segments: Dict[int, List[Tuple[int, int]]],
    primitives: Dict[str, List],
    depth_image: torch.Tensor,
    normal_image: torch.Tensor,
    save_dir: Path,
    frame_idx: int = 0,
    rgb_image: Optional[np.ndarray] = None,
    show_3d: bool = True
) -> None:
    """
    Comprehensive visualization for single view segmentation and primitive fitting.
    
    Args:
        seg_map: (H, W) segmentation map
        seg_props: Dictionary of segment properties
        global_segments: Global segment groupings
        primitives: Fitted primitives (S_items, R_items, T_items, pts_items)
        depth_image: (H, W) depth map
        normal_image: (H, W, 3) normal map
        save_dir: Directory to save visualizations
        frame_idx: Frame index for labeling
        rgb_image: Optional RGB image for overlay
        show_3d: Whether to generate 3D visualizations
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.patches import Rectangle
    import matplotlib.patches as mpatches
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    H, W = seg_map.shape
    
    # Helper function to generate distinct colors
    def get_colors(n):
        if n <= 20:
            return cm.get_cmap('tab20')(np.linspace(0, 1, n))
        else:
            return cm.get_cmap('hsv')(np.linspace(0, 0.9, n))
    
    # 1. Create main visualization figure
    fig = plt.figure(figsize=(20, 12))
    
    # 2. Original RGB/Depth view
    ax1 = plt.subplot(2, 4, 1)
    if rgb_image is not None:
        ax1.imshow(rgb_image)
        ax1.set_title('RGB Image')
    else:
        depth_vis = depth_image.cpu().numpy()
        depth_vis = np.clip(depth_vis, 0, np.percentile(depth_vis[depth_vis > 0], 95))
        ax1.imshow(depth_vis, cmap='viridis')
        ax1.set_title('Depth Map')
    ax1.axis('off')
    
    # 3. Normal visualization
    ax2 = plt.subplot(2, 4, 2)
    normal_vis = (normal_image.cpu().numpy() + 1) / 2  # Convert from [-1,1] to [0,1]
    normal_vis = np.clip(normal_vis, 0, 1)
    ax2.imshow(normal_vis)
    ax2.set_title('Surface Normals')
    ax2.axis('off')
    
    # 4. Per-pixel segmentation
    ax3 = plt.subplot(2, 4, 3)
    unique_segments = np.unique(seg_map)
    valid_segments = unique_segments[unique_segments >= 0]
    
    if len(valid_segments) > 0:
        colors = get_colors(len(valid_segments))
        seg_colored = np.zeros((*seg_map.shape, 3))
        for idx, seg_id in enumerate(valid_segments):
            mask = seg_map == seg_id
            seg_colored[mask] = colors[idx][:3]
    else:
        seg_colored = np.zeros((*seg_map.shape, 3))
    
    ax3.imshow(seg_colored)
    ax3.set_title(f'Segmentation ({len(valid_segments)} segments)')
    ax3.axis('off')
    
    # 5. Segment properties overlay
    ax4 = plt.subplot(2, 4, 4)
    ax4.imshow(seg_colored)
    
    # Draw segment centroids and IDs
    for seg_id, props in seg_props.items():
        # Get 2D centroid from pixels
        pixels = np.array(props['pixels'])
        if len(pixels) > 0:
            centroid_2d = pixels.mean(axis=0)
            ax4.plot(centroid_2d[1], centroid_2d[0], 'w*', markersize=8, markeredgecolor='k')
            ax4.text(centroid_2d[1], centroid_2d[0]-5, str(seg_id), 
                    color='white', fontsize=8, ha='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    ax4.set_title('Segment IDs & Centroids')
    ax4.axis('off')
    
    # 6. Primitive visualization (if available)
    ax5 = plt.subplot(2, 4, 5)
    if 'S_items' in primitives and len(primitives['S_items']) > 0:
        # Create a primitive overlay visualization
        prim_overlay = np.zeros((*seg_map.shape, 3))
        prim_colors = get_colors(len(primitives['S_items']))
        
        # Map primitives back to segments
        for prim_idx, (S, R, T) in enumerate(zip(primitives['S_items'], 
                                                  primitives['R_items'], 
                                                  primitives['T_items'])):
            # Find which segments belong to this primitive
            # This is simplified - you might need to track the mapping
            color = prim_colors[prim_idx][:3]
            
            # For visualization, we'll color segments that were merged into this primitive
            # This requires tracking which global segment produced which primitive
            if prim_idx < len(valid_segments):
                mask = seg_map == valid_segments[prim_idx]
                prim_overlay[mask] = color
        
        ax5.imshow(prim_overlay)
        ax5.set_title(f'Fitted Primitives ({len(primitives["S_items"])} boxes)')
    else:
        ax5.imshow(seg_colored)
        ax5.set_title('No Primitives Fitted')
    ax5.axis('off')
    
    # 7. Statistics panel
    ax6 = plt.subplot(2, 4, 6)
    ax6.axis('off')
    
    stats_text = f"Frame: {frame_idx}\n"
    stats_text += f"Resolution: {W}×{H}\n"
    stats_text += f"Segments: {len(valid_segments)}\n"
    stats_text += f"Primitives: {len(primitives.get('S_items', []))}\n\n"
    
    # Segment statistics
    stats_text += "Segment Sizes:\n"
    for seg_id in valid_segments[:5]:  # Show first 5
        if seg_id in seg_props:
            n_pixels = len(seg_props[seg_id]['pixels'])
            stats_text += f"  Seg {seg_id}: {n_pixels} pixels\n"
    
    if len(valid_segments) > 5:
        stats_text += f"  ... and {len(valid_segments)-5} more\n"
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 8. Normal clustering visualization
    ax7 = plt.subplot(2, 4, 7)
    
    # Show normal clusters by coloring segments based on their average normal
    normal_cluster_vis = np.zeros((*seg_map.shape, 3))
    for seg_id, props in seg_props.items():
        mask = seg_map == seg_id
        # Convert normal to RGB color (shift from [-1,1] to [0,1])
        normal_color = (props['avg_normal'].cpu().numpy() + 1) / 2
        normal_cluster_vis[mask] = normal_color
    
    ax7.imshow(normal_cluster_vis)
    ax7.set_title('Normal-based Coloring')
    ax7.axis('off')
    
    # 9. Boundary visualization
    ax8 = plt.subplot(2, 4, 8)
    
    # Create boundary image
    from scipy import ndimage
    boundary_img = np.zeros((H, W))
    for seg_id in valid_segments:
        mask = seg_map == seg_id
        eroded = ndimage.binary_erosion(mask)
        boundary = mask & ~eroded
        boundary_img[boundary] = 1
    
    ax8.imshow(seg_colored)
    ax8.contour(boundary_img, colors='white', linewidths=1)
    ax8.set_title('Segment Boundaries')
    ax8.axis('off')
    
    plt.suptitle(f'Single View Segmentation Results - Frame {frame_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save main figure
    main_save_path = save_dir / 'single_view_complete.png'
    plt.savefig(main_save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved main visualization to {main_save_path}")
    
    # Additional individual visualizations
    
    # 10. Save individual segmentation map with legend
    fig2, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(seg_colored)
    ax.set_title(f'Segmentation Map - {len(valid_segments)} Segments')
    ax.axis('off')
    
    # Create legend
    if len(valid_segments) <= 20:
        patches = []
        colors = get_colors(len(valid_segments))
        for idx, seg_id in enumerate(valid_segments):
            if seg_id in seg_props:
                n_pixels = len(seg_props[seg_id]['pixels'])
                label = f'Seg {seg_id} ({n_pixels} px)'
                patches.append(mpatches.Patch(color=colors[idx][:3], label=label))
        
        ax.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5), 
                 fontsize=8, title='Segments')
    
    seg_save_path = save_dir / 'segmentation_with_legend.png'
    plt.savefig(seg_save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved segmentation with legend to {seg_save_path}")




def build_global_segments_greedy(
    all_frame_segments: List[Dict[int, Dict]],
    correspondence_pairs: List[Tuple[int, int, int, int]]
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Union-find over *all* correspondence pairs (frame_i, seg_i, frame_j, seg_j).
    Returns {global_id: [(frame_idx, local_seg_id), …]}.
    """
    parent = {}
    def key(fi, sid): return f"{fi}_{sid}"
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    # initialise every local segment
    for fi, segs in enumerate(all_frame_segments):
        for sid in segs:
            find(key(fi, sid))

    # greedy unions
    for fi, sid_i, fj, sid_j in correspondence_pairs:
        union(key(fi, sid_i), key(fj, sid_j))

    # collect & renumber
    groups = defaultdict(list)
    for fi, segs in enumerate(all_frame_segments):
        for sid in segs:
            groups[find(key(fi, sid))].append((fi, sid))

    return {gid: members for gid, members in enumerate(groups.values())}

def visualize_per_frame_segments(
    all_seg_maps: List[np.ndarray],
    frame_indices: List[int],
    save_dir: Path,
    max_cols: int = 4
) -> None:
    """
    Visualize segmentation maps for each frame.
    Each segment gets a unique color within its frame.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    num_frames = len(all_seg_maps)
    num_rows = (num_frames + max_cols - 1) // max_cols
    
    fig, axes = plt.subplots(num_rows, max_cols, figsize=(4*max_cols, 4*num_rows))
    if num_frames == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    for idx, (seg_map, frame_idx) in enumerate(zip(all_seg_maps, frame_indices)):
        ax = axes_flat[idx]
        
        # Create colored visualization
        unique_segments = np.unique(seg_map)
        valid_segments = unique_segments[unique_segments >= 0]
        
        # Create color map
        if len(valid_segments) > 0:
            colors = cm.get_cmap('tab20')(np.linspace(0, 1, len(valid_segments)))
            
            # Create RGB image
            rgb_image = np.zeros((*seg_map.shape, 3))
            for seg_idx, seg_id in enumerate(valid_segments):
                mask = seg_map == seg_id
                rgb_image[mask] = colors[seg_idx][:3]
        else:
            rgb_image = np.zeros((*seg_map.shape, 3))
        
        ax.imshow(rgb_image)
        ax.set_title(f'Frame {frame_idx}\n{len(valid_segments)} segments')
        ax.axis('off')
        fig.savefig('_per_frame.png')
    
    # Hide unused subplots
    for idx in range(num_frames, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.tight_layout()
    save_path = save_dir / 'per_frame_segments.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved per-frame segments visualization to {save_path}")


def visualize_merged_segments(
    all_seg_maps: List[np.ndarray],
    global_segments: Dict[int, List[Tuple[int, int]]],
    frame_indices: List[int],
    save_dir: Path,
    max_cols: int = 4
) -> None:
    """
    Visualize merged segments across frames.
    Segments that are merged across frames share the same color.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    num_frames = len(all_seg_maps)
    num_rows = (num_frames + max_cols - 1) // max_cols
    
    fig, axes = plt.subplots(num_rows, max_cols, figsize=(4*max_cols, 4*num_rows))
    if num_frames == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes_flat = axes.flatten()
    
    # Assign colors to global segments
    num_global_segments = len(global_segments)
    if num_global_segments > 0:
        # Use a colormap with enough distinct colors
        if num_global_segments <= 20:
            global_colors = cm.get_cmap('tab20')(np.linspace(0, 1, num_global_segments))
        else:
            global_colors = cm.get_cmap('hsv')(np.linspace(0, 0.9, num_global_segments))
    
    # Create merged visualization for each frame
    for frame_idx_pos, (seg_map, frame_idx) in enumerate(zip(all_seg_maps, frame_indices)):
        ax = axes_flat[frame_idx_pos]
        
        # Create RGB image
        rgb_image = np.zeros((*seg_map.shape, 3))
        
        # Fill in colors based on global segment membership
        segments_in_frame = 0
        for global_id, members in global_segments.items():
            # Check if this global segment appears in current frame
            for member_frame_idx, local_seg_id in members:
                if member_frame_idx == frame_idx_pos:  # Use position index, not frame number
                    mask = seg_map == local_seg_id
                    if mask.any():
                        rgb_image[mask] = global_colors[global_id][:3]
                        segments_in_frame += 1
        
        ax.imshow(rgb_image)
        ax.set_title(f'Frame {frame_idx}\n{segments_in_frame} global segments')
        ax.axis('off')
        fig.savefig('_global_frame.png')
    
    # Hide unused subplots
    for idx in range(num_frames, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.tight_layout()
    save_path = save_dir / 'merged_segments.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved merged segments visualization to {save_path}")

def estimate_normals_knn(
    points: torch.Tensor,
    k: int = 20,
    align_to_direction: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Estimate normals for a point cloud using k-nearest neighbors and PCA.
    
    Args:
        points: Point cloud tensor of shape [N, 3]
        k: Number of nearest neighbors to use for normal estimation
        align_to_direction: Optional direction vector to align normals consistently
                          (e.g., viewpoint for consistent orientation)
    
    Returns:
        normals: Estimated normals of shape [N, 3]
    """
    N = points.shape[0]
    device = points.device
    
    # Compute pairwise distances
    # Using broadcasting: ||pi - pj||^2 = ||pi||^2 + ||pj||^2 - 2*pi·pj
    points_norm = (points ** 2).sum(dim=1, keepdim=True)  # [N, 1]
    distances = points_norm + points_norm.T - 2 * torch.mm(points, points.T)  # [N, N]
    distances = torch.clamp(distances, min=0.0)  # Handle numerical errors
    
    # Find k-nearest neighbors (including the point itself)
    _, indices = torch.topk(distances, k=min(k, N), dim=1, largest=False)  # [N, k]
    
    # Initialize normals
    normals = torch.zeros_like(points)
    
    for i in range(N):
        # Get neighborhood points
        neighbors = points[indices[i]]  # [k, 3]
        
        # Center the points
        centroid = neighbors.mean(dim=0, keepdim=True)  # [1, 3]
        centered = neighbors - centroid  # [k, 3]
        
        # Compute covariance matrix
        cov = torch.mm(centered.T, centered) / k  # [3, 3]
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        
        # Normal is the eigenvector with smallest eigenvalue
        normal = eigenvectors[:, 0]  # [3]
        
        # Store the normal
        normals[i] = normal
    
    # Normalize the normals
    normals = F.normalize(normals, p=2, dim=1)
    
    # Align normals consistently if direction is provided
    if align_to_direction is not None:
        align_to_direction = F.normalize(align_to_direction, p=2, dim=-1)
        if align_to_direction.dim() == 1:
            align_to_direction = align_to_direction.unsqueeze(0).expand_as(normals)
        
        # Flip normals that point away from the alignment direction
        dots = (normals * align_to_direction).sum(dim=1, keepdim=True)
        normals = torch.where(dots < 0, -normals, normals)
    
    return normals

def save_segment_statistics(
    all_seg_maps: List[np.ndarray],
    all_frame_segments: List[Dict[int, Dict]],
    global_segments: Dict[int, List[Tuple[int, int]]],
    all_frame_correspondences: List[List[Tuple[int, int]]],
    frame_indices: List[int],
    save_dir: Path
) -> None:
    """
    Save detailed statistics about segments for debugging.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    stats_file = save_dir / 'segment_statistics.txt'
    
    with open(stats_file, 'w') as f:
        f.write("=== SEGMENT STATISTICS ===\n\n")
        
        # Per-frame statistics
        f.write("PER-FRAME SEGMENTS:\n")
        for idx, (seg_map, frame_idx) in enumerate(zip(all_seg_maps, frame_indices)):
            segments = all_frame_segments[idx]
            f.write(f"\nFrame {frame_idx} (index {idx}):\n")
            f.write(f"  Total segments: {len(segments)}\n")
            
            for seg_id, props in segments.items():
                num_pixels = len(props['pixels'])
                avg_normal = props['avg_normal'].cpu().numpy()
                centroid = props['centroid'].cpu().numpy()
                f.write(f"    Segment {seg_id}: {num_pixels} pixels\n")
                f.write(f"      Normal: [{avg_normal[0]:.3f}, {avg_normal[1]:.3f}, {avg_normal[2]:.3f}]\n")
                f.write(f"      Centroid: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]\n")
        
        # Correspondence statistics
        f.write("\n\nCORRESPONDENCES:\n")
        for idx, correspondences in enumerate(all_frame_correspondences):
            if idx < len(frame_indices) - 1:
                f.write(f"\nFrame {frame_indices[idx]} -> Frame {frame_indices[idx+1]}:\n")
                f.write(f"  Found {len(correspondences)} matches\n")
                for seg1, seg2 in correspondences:
                    f.write(f"    Segment {seg1} -> Segment {seg2}\n")
        
        # Global segment statistics
        f.write("\n\nGLOBAL SEGMENTS:\n")
        f.write(f"Total global segments: {len(global_segments)}\n\n")
        
        # Sort by number of frames (descending)
        sorted_global = sorted(global_segments.items(), 
                             key=lambda x: len(x[1]), reverse=True)
        
        for global_id, members in sorted_global:
            f.write(f"\nGlobal Segment {global_id}:\n")
            f.write(f"  Appears in {len(members)} frames\n")
            f.write(f"  Members:\n")
            for frame_idx, local_seg_id in members:
                actual_frame = frame_indices[frame_idx]
                f.write(f"    Frame {actual_frame} (idx {frame_idx}), Local segment {local_seg_id}\n")
        
        # Find segments that appear in multiple frames
        multi_frame_segments = [gid for gid, members in global_segments.items() if len(members) >= 2]
        f.write(f"\n\nSegments appearing in 2+ frames: {len(multi_frame_segments)}\n")
        
    print(f"Saved segment statistics to {stats_file}")


def debug_flow_matching(
    seg_props1: Dict[int, Dict],
    seg_props2: Dict[int, Dict],
    flow_forward: np.ndarray,
    covis_mask: np.ndarray,
    frame1_idx: int,
    frame2_idx: int,
    save_dir: Path
) -> None:
    """
    Debug flow matching between two specific frames.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    H, W = flow_forward.shape[:2]
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Show flow magnitude
    flow_mag = np.sqrt(flow_forward[..., 0]**2 + flow_forward[..., 1]**2)
    im1 = axes[0, 0].imshow(flow_mag, cmap='viridis')
    axes[0, 0].set_title(f'Flow Magnitude\nFrame {frame1_idx} -> {frame2_idx}')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Show covisibility mask
    im2 = axes[0, 1].imshow(covis_mask, cmap='gray')
    axes[0, 1].set_title('Covisibility Mask')
    
    # Debug text
    debug_text = f"Frame {frame1_idx} -> {frame2_idx}\n\n"
    debug_text += f"Segments in frame 1: {len(seg_props1)}\n"
    debug_text += f"Segments in frame 2: {len(seg_props2)}\n\n"
    
    # Check each segment pair
    for seg1_id, props1 in seg_props1.items():
        for seg2_id, props2 in seg_props2.items():
            # Normal similarity
            normal_sim = torch.dot(props1['avg_normal'], props2['avg_normal']).item()
            
            # Flow-based overlap
            overlap_count = 0
            valid_flow_pixels = 0
            
            for y, x in props1['boundary_pixels'][:20]:  # Sample first 20 pixels
                if covis_mask[y, x]:
                    print(flow_forward.shape, y, x)
                    flow = flow_forward[y, x]
                    
                    y_new = int(round(y + flow[1]))
                    x_new = int(round(x + flow[0]))
                    
                    if 0 <= y_new < H and 0 <= x_new < W:
                        valid_flow_pixels += 1
                        if (y_new, x_new) in props2['pixels']:
                            overlap_count += 1
            
            if valid_flow_pixels > 0:
                overlap_ratio = overlap_count / valid_flow_pixels
                if normal_sim > 0.9 or overlap_ratio > 0.1:
                    debug_text += f"Seg {seg1_id} -> Seg {seg2_id}: "
                    debug_text += f"normal_sim={normal_sim:.3f}, "
                    debug_text += f"overlap={overlap_count}/{valid_flow_pixels} "
                    debug_text += f"({overlap_ratio:.2f})\n"
    
    # Show debug text
    axes[1, 0].text(0.05, 0.95, debug_text, transform=axes[1, 0].transAxes,
                    verticalalignment='top', fontsize=8, family='monospace')
    axes[1, 0].axis('off')
    
    # Hide last subplot
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    save_path = save_dir / f'flow_debug_{frame1_idx}_{frame2_idx}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved flow debug to {save_path}")

def load_flow_and_covis_data(
    data_dir: Path,
    frame_indices: List[int],
    interval: int = 7
) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], np.ndarray]]:
    """
    Load flow and covisibility data for given frame indices.
    
    Returns
    -------
    flow_forward : dict mapping (frame_i, frame_j) -> flow array
    flow_backward : dict mapping (frame_j, frame_i) -> flow array  
    covis_masks : dict mapping (frame_i, frame_j) -> covisibility mask
    """
    flow_forward = {}
    flow_backward = {}
    covis_masks = {}
    
    # Load covisibility masks between interval frames
    for i in range(len(frame_indices) - 1):
        frame_i = frame_indices[i]
        frame_j = frame_indices[i + 1]
        
        # Try to find covisibility file
        covis_file = data_dir / f"covis_{frame_i:05d}_{frame_j:05d}.npy"
        if covis_file.exists():
            covis_masks[(frame_i, frame_j)] = np.load(covis_file)
    
    # Load flow data - for now, we'll use direct flow if available
    # In practice, you might need to accumulate flow over intervals
    for i in range(len(frame_indices) - 1):
        frame_i = frame_indices[i]
        frame_j = frame_indices[i + 1]
        
        # For interval=7, we might need to accumulate flow
        # For now, let's check if direct flow exists
        flow_file_forward = data_dir / f"flow_{frame_i:05d}_{frame_j:05d}.npy"
        flow_file_backward = data_dir / f"flow_{frame_j:05d}_{frame_i:05d}.npy"
        
        if flow_file_forward.exists():
            flow_forward[(frame_i, frame_j)] = np.load(flow_file_forward)
        if flow_file_backward.exists():
            flow_backward[(frame_j, frame_i)] = np.load(flow_file_backward)
    
    return flow_forward, flow_backward, covis_masks


def flow_to_color(flow, max_magnitude=None):
    """
    Convert optical flow to HSV color representation.
    
    Parameters:
    -----------
    flow : np.ndarray, shape (H, W, 2)
        Optical flow with u, v components
    max_magnitude : float, optional
        Maximum magnitude for normalization. If None, uses flow max.
        
    Returns:
    --------
    color_image : np.ndarray, shape (H, W, 3)
        RGB color representation of flow
    """
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    
    # Convert to polar coordinates
    magnitude = np.sqrt(fx**2 + fy**2)
    angle = np.arctan2(fy, fx)
    
    # Normalize angle to [0, 1] for hue
    hue = (angle + np.pi) / (2 * np.pi)
    
    # Normalize magnitude for saturation
    if max_magnitude is None:
        max_magnitude = np.max(magnitude)
    
    saturation = np.clip(magnitude / max_magnitude, 0, 1)
    value = np.ones_like(hue)
    
    # Create HSV image
    hsv = np.stack([hue, saturation, value], axis=-1)
    
    # Convert to RGB
    rgb = hsv_to_rgb(hsv)
    
    return rgb


def visualize_flow(flow, title="Optical Flow", save_path=None, figsize=(12, 8), 
                  max_magnitude=None, show_colorbar=True, show_arrows=False, 
                  arrow_step=20):
    """
    Visualize optical flow as HSV color image.
    
    Parameters:
    -----------
    flow : np.ndarray, shape (H, W, 2)
        Optical flow with u, v components
    title : str
        Title for the plot
    save_path : str or Path, optional
        Path to save the visualization
    figsize : tuple
        Figure size
    max_magnitude : float, optional
        Maximum magnitude for color scaling
    show_colorbar : bool
        Whether to show magnitude colorbar
    show_arrows : bool
        Whether to overlay flow arrows
    arrow_step : int
        Step size for arrow sampling
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # HSV color representation
    color_image = flow_to_color(flow, max_magnitude)
    ax1.imshow(color_image)
    ax1.set_title(f"{title} - HSV Representation")
    ax1.axis('off')
    
    # Magnitude representation
    magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
    im = ax2.imshow(magnitude, cmap='viridis')
    ax2.set_title(f"{title} - Magnitude")
    ax2.axis('off')
    
    if show_colorbar:
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    # Overlay arrows if requested
    if show_arrows:
        h, w = flow.shape[:2]
        y, x = np.mgrid[0:h:arrow_step, 0:w:arrow_step]
        u = flow[::arrow_step, ::arrow_step, 0]
        v = flow[::arrow_step, ::arrow_step, 1]
        
        ax1.quiver(x, y, u, v, angles='xy', scale_units='xy', 
                  scale=1, color='white', width=0.002, alpha=0.7)
        ax2.quiver(x, y, u, v, angles='xy', scale_units='xy', 
                  scale=1, color='white', width=0.002, alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Flow visualization saved to: {save_path}")
    
    plt.show()


def visualize_covisibility(covis_mask, title="Covisibility Mask", save_path=None, 
                          figsize=(10, 6), cmap='viridis'):
    """
    Visualize covisibility mask.
    
    Parameters:
    -----------
    covis_mask : np.ndarray, shape (H, W)
        Binary or continuous covisibility mask
    title : str
        Title for the plot
    save_path : str or Path, optional
        Path to save the visualization
    figsize : tuple
        Figure size
    cmap : str
        Colormap for visualization
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    im = ax.imshow(covis_mask, cmap=cmap)
    ax.set_title(title)
    ax.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Covisibility')
    
    # Add statistics text
    stats_text = f"Min: {covis_mask.min():.3f}\n"
    stats_text += f"Max: {covis_mask.max():.3f}\n"
    stats_text += f"Mean: {covis_mask.mean():.3f}\n"
    stats_text += f"Shape: {covis_mask.shape}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='white', alpha=0.8), fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Covisibility visualization saved to: {save_path}")
    
    plt.show()


def create_flow_colorwheel(size=256, save_path=None):
    """
    Create and save a flow color wheel for reference.
    
    Parameters:
    -----------
    size : int
        Size of the color wheel
    save_path : str or Path, optional
        Path to save the color wheel
    """
    # Create coordinate grids
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Create flow field (circular)
    flow = np.stack([X, Y], axis=-1)
    
    # Mask for circular region
    radius = np.sqrt(X**2 + Y**2)
    mask = radius <= 1
    
    # Convert to color
    color_wheel = flow_to_color(flow, max_magnitude=1)
    color_wheel[~mask] = 1  # White background outside circle
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(color_wheel)
    ax.set_title("Flow Color Wheel\n(Hue = Direction, Saturation = Magnitude)", 
                fontsize=14)
    ax.axis('off')
    
    # Add direction labels
    ax.text(size//2, 10, "↑", ha='center', va='center', fontsize=20, fontweight='bold')
    ax.text(size//2, size-10, "↓", ha='center', va='center', fontsize=20, fontweight='bold')
    ax.text(10, size//2, "←", ha='center', va='center', fontsize=20, fontweight='bold')
    ax.text(size-10, size//2, "→", ha='center', va='center', fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Flow color wheel saved to: {save_path}")
    
    plt.show()


def batch_visualize_flow_data(flow_forward, flow_backward, covis_masks, 
                             output_dir="./flow_visualizations"):
    """
    Batch visualize all flow and covisibility data.
    
    Parameters:
    -----------
    flow_forward : dict
        Forward flow data
    flow_backward : dict  
        Backward flow data
    covis_masks : dict
        Covisibility masks
    output_dir : str or Path
        Output directory for visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Saving visualizations to: {output_dir}")
    
    # Create color wheel reference
    create_flow_colorwheel(save_path=output_dir / "flow_colorwheel.png")
    
    # Visualize forward flows
    print("\nProcessing forward flows...")
    for (frame_i, frame_j), flow in flow_forward.items():
        title = f"Forward Flow: {frame_i:05d} → {frame_j:05d}"
        save_path = output_dir / f"flow_forward_{frame_i:05d}_{frame_j:05d}.png"
        visualize_flow(flow, title=title, save_path=save_path, 
                      show_arrows=True, arrow_step=30)
        plt.close('all')  # Close to save memory
    
    # Visualize backward flows
    print("\nProcessing backward flows...")
    for (frame_j, frame_i), flow in flow_backward.items():
        title = f"Backward Flow: {frame_j:05d} → {frame_i:05d}"
        save_path = output_dir / f"flow_backward_{frame_j:05d}_{frame_i:05d}.png"
        visualize_flow(flow, title=title, save_path=save_path, 
                      show_arrows=True, arrow_step=30)
        plt.close('all')
    
    # Visualize covisibility masks
    print("\nProcessing covisibility masks...")
    for (frame_i, frame_j), mask in covis_masks.items():
        title = f"Covisibility: {frame_i:05d} ↔ {frame_j:05d}"
        save_path = output_dir / f"covis_{frame_i:05d}_{frame_j:05d}.png"
        visualize_covisibility(mask, title=title, save_path=save_path)
        plt.close('all')
    
    print(f"\nAll visualizations saved to: {output_dir}")


def generate_sample_data():
    """Generate sample flow and covisibility data for testing."""
    print("Generating sample data...")
    
    # Create sample directory
    sample_dir = Path("./sample_flow_data")
    sample_dir.mkdir(exist_ok=True)
    
    h, w = 384, 512
    
    # Generate sample flow data
    for i in range(3):
        frame_i = i * 7
        frame_j = (i + 1) * 7
        
        # Create synthetic flow field
        x = np.linspace(-2, 2, w)
        y = np.linspace(-1.5, 1.5, h)
        X, Y = np.meshgrid(x, y)
        
        # Circular flow pattern
        u = -Y * np.exp(-(X**2 + Y**2)) + np.random.normal(0, 0.1, (h, w))
        v = X * np.exp(-(X**2 + Y**2)) + np.random.normal(0, 0.1, (h, w))
        
        flow = np.stack([u, v], axis=-1)
        
        # Save flow data
        np.save(sample_dir / f"flow_{frame_i:05d}_{frame_j:05d}.npy", flow)
        np.save(sample_dir / f"flow_{frame_j:05d}_{frame_i:05d}.npy", -flow)  # Reverse flow
        
        # Generate covisibility mask
        center_x, center_y = w//2, h//2
        covis_mask = np.zeros((h, w))
        
        # Create elliptical visibility region
        for y in range(h):
            for x in range(w):
                dx = (x - center_x) / (w * 0.3)
                dy = (y - center_y) / (h * 0.3)
                if dx**2 + dy**2 <= 1:
                    covis_mask[y, x] = np.exp(-(dx**2 + dy**2))
        
        # Add some noise
        covis_mask += np.random.uniform(0, 0.1, (h, w))
        covis_mask = np.clip(covis_mask, 0, 1)
        
        np.save(sample_dir / f"covis_{frame_i:05d}_{frame_j:05d}.npy", covis_mask)
    
    print(f"Sample data generated in: {sample_dir}")
    return sample_dir


def _generate_distinct_colors(n_colors):
    """Generate visually distinct colors for visualization."""
    if n_colors == 0:
        return np.array([])
    
    hues = np.linspace(0, 1, n_colors, endpoint=False)
    colors = []
    for hue in hues:
        # Use HSV to RGB conversion for better color distribution
        rgb = plt.cm.hsv(hue)[:3]
        colors.append(rgb)
    return np.array(colors)




def merge_normal_clusters(pred, topk, centres):
    """Merge similar normal clusters based on dot product similarity."""
    n_valid = len(topk)
    if n_valid <= 1:
        return pred, topk, n_valid
    
    # Compute pairwise similarities between cluster centers
    similarities = np.dot(centres[topk], centres[topk].T)
    
    # Merge clusters with high similarity (dot product > 0.95)
    merge_threshold = 0.95
    merged = np.arange(n_valid)
    
    for i in range(n_valid):
        for j in range(i + 1, n_valid):
            if similarities[i, j] > merge_threshold:
                # Merge cluster j into cluster i
                merged[merged == merged[j]] = merged[i]
    
    # Create mapping from old labels to new merged labels
    unique_merged = np.unique(merged)
    merge_map = {topk[old_idx]: topk[merged[old_idx]] for old_idx in range(n_valid)}
    
    # Apply merging to predictions
    new_pred = pred.copy()
    for old_label, new_label in merge_map.items():
        if old_label != new_label:
            new_pred[pred == old_label] = new_label
    
    # Get final unique clusters
    final_topk = topk[np.unique(merged)]
    n_valid = len(final_topk)
    
    return new_pred, final_topk, n_valid


def cluster_normals_single_frame(
    normals: torch.Tensor,
    valid_mask: torch.Tensor,
    n_clusters: int = 12,
    n_init_normal_clusters: int = 10,
    return_labels_map: bool = True
):
    """
    Cluster normals for a single frame.
    
    Parameters
    ----------
    normals : (H, W, 3) torch.Tensor
        Normal vectors
    valid_mask : (H, W) torch.Tensor
        Boolean mask of valid pixels
    n_clusters : int
        Target number of clusters
    n_init_normal_clusters : int
        Initial number of clusters for KMeans
    return_labels_map : bool
        If True, returns full (H,W) label map; if False, returns only valid pixel labels
    
    Returns
    -------
    labels : (H, W) int32 array if return_labels_map=True, else 1D array of valid pixel labels
        Cluster labels, -1 for invalid pixels
    n_valid_clusters : int
        Number of valid clusters found
    """
    H, W = normals.shape[:2]
    device = normals.device
    
    # Get valid normals
    valid_normals = normals[valid_mask].cpu().numpy()
    
    if return_labels_map:
        labels = np.full((H, W), -1, dtype=np.int32)
    
    if len(valid_normals) == 0:
        if return_labels_map:
            return labels, 0
        else:
            return np.array([], dtype=np.int32), 0
    
    # Check normal validity (unit vectors)
    mag = np.linalg.norm(valid_normals, axis=1)
    finite = np.isfinite(valid_normals).all(axis=1)
    unit = np.abs(mag - 1.0) < 1e-2
    normal_valid = finite & unit
    
    if not normal_valid.any():
        if return_labels_map:
            return labels, 0
        else:
            return np.full(len(valid_normals), -1, dtype=np.int32), 0
    
    # Cluster only valid normals
    valid_normals_clean = valid_normals[normal_valid]
    
    # Perform KMeans clustering
    kmeans = KMeans(
        n_clusters=min(n_init_normal_clusters, len(valid_normals_clean)),
        n_init=1,
        random_state=0
    ).fit(valid_normals_clean)
    
    pred = kmeans.labels_
    centres = kmeans.cluster_centers_
    counts = np.bincount(pred)
    
    # Select top-k clusters by size
    n_actual_clusters = min(n_clusters, len(counts))
    topk = np.argpartition(counts, -n_actual_clusters)[-n_actual_clusters:]
    topk = topk[np.argsort(counts[topk])[::-1]]
    
    # Merge similar clusters
    pred, topk, n_valid = merge_normal_clusters(pred, topk, centres)
    
    # Create relabeling map
    mapping = {old: new for new, old in enumerate(topk[:n_valid])}
    relabel = np.full(pred.shape, -1, dtype=np.int32)
    for old, new in mapping.items():
        relabel[pred == old] = new
    
    # Create output labels
    if return_labels_map:
        # Map back to full image
        valid_indices = np.where(valid_mask.cpu().numpy())
        valid_idx_clean = np.where(normal_valid)[0]
        
        # First map to valid pixels
        valid_labels = np.full(len(valid_normals), -1, dtype=np.int32)
        valid_labels[valid_idx_clean] = relabel
        
        # Then map to full image
        labels[valid_indices] = valid_labels
        return labels, n_valid
    else:
        # Return only valid pixel labels
        valid_labels = np.full(len(valid_normals), -1, dtype=np.int32)
        valid_labels[normal_valid] = relabel
        return valid_labels, n_valid


def segment_single_frame_normals(
    normal_image: torch.Tensor,
    depth_image: torch.Tensor,
    eps_spatial: float = 3.0,
    min_samples: int = 30,
    min_points: int = 500,
    n_normal_clusters: int = 6,
    n_init_normal_clusters: int = 10,
    temporal_weight: float = 0.0,  # Not used for single frame
    min_final_points: int = 1400,
    device: str = 'cuda'
) -> Tuple[np.ndarray, int]:
    """
    Segment a single frame based on normal similarity using the adapted clustering approach.
    
    Parameters
    ----------
    normal_image : (H, W, 3) torch.Tensor
        Normal vectors for the frame
    depth_image : (H, W) torch.Tensor
        Depth values for the frame
    eps_spatial : float
        DBSCAN spatial epsilon parameter
    min_samples : int
        DBSCAN minimum samples parameter
    min_points : int
        Minimum points for a valid frame
    n_normal_clusters : int
        Target number of normal clusters
    n_init_normal_clusters : int
        Initial number of clusters for KMeans
    temporal_weight : float
        Not used for single frame (kept for compatibility)
    min_final_points : int
        Minimum points for a valid segment
    device : str
        Device to use for computation
    
    Returns
    -------
    seg_map : (H, W) np.ndarray
        Segment IDs, -1 for background/invalid
    num_segments : int
        Number of segments found
    """
    H, W = normal_image.shape[:2]
    
    # Create mask of valid pixels
    valid_mask = (depth_image > 0) & torch.isfinite(normal_image).all(dim=-1)
    
    if valid_mask.sum() < min_points:
        return np.full((H, W), -1, dtype=np.int32), 0
    
    # Step 1: Cluster normals
    cluster_labels, n_valid_clusters = cluster_normals_single_frame(
        normal_image,
        valid_mask,
        n_clusters=n_normal_clusters,
        n_init_normal_clusters=n_init_normal_clusters,
        return_labels_map=True
    )
    
    if n_valid_clusters == 0:
        return np.full((H, W), -1, dtype=np.int32), 0
    
    # Step 2: Apply DBSCAN within each normal cluster for spatial segmentation
    seg_map = np.full((H, W), -1, dtype=np.int32)
    global_seg_id = 0
    
    for cluster_id in range(n_valid_clusters):
        # Get pixels belonging to this normal cluster
        cluster_mask = cluster_labels == cluster_id
        y_coords, x_coords = np.where(cluster_mask)
        

        
        # Create spatial features (no temporal component for single frame)
        spatial_features = np.column_stack([y_coords, x_coords])
        
        # Apply DBSCAN for spatial segmentation
        dbscan = DBSCAN(eps=eps_spatial, min_samples=1, metric='chebyshev')
        # dbscan = DBSCAN(eps=eps_spatial, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(spatial_features)
        
        # Assign global segment IDs
        for local_seg_id in np.unique(dbscan_labels):

            local_mask = dbscan_labels == local_seg_id
            seg_indices = (y_coords[local_mask], x_coords[local_mask])
            
            # Check minimum segment size
            if len(seg_indices[0]) >= min_final_points:
                seg_map[seg_indices] = global_seg_id
                global_seg_id += 1
    
    return seg_map, global_seg_id
def compute_segment_properties(
    seg_map: np.ndarray,
    normals: torch.Tensor,
    depth: torch.Tensor,
    pointclouds: torch.Tensor,
    device: str = 'cuda'
) -> Dict[int, Dict]:
    """
    Compute properties for each segment in a frame.
    
    Returns dict mapping segment_id to properties:
        - avg_normal: average normal in world space
        - centroid: 3D centroid in world space
        - pixels: list of (y, x) pixel coordinates
        - boundary_pixels: pixels on segment boundary
    """
    H, W = seg_map.shape
    segment_props = {}
    
    
    for seg_id in np.unique(seg_map):
        
        mask = seg_map == seg_id
        y_idx, x_idx = np.where(mask)
        

        
        # Convert indices to torch tensors
        y_idx_t = torch.tensor(y_idx, device=device, dtype=torch.long)
        x_idx_t = torch.tensor(x_idx, device=device, dtype=torch.long)
        
        # Get normals for this segment (already in world space)
        seg_normals = normals[y_idx_t, x_idx_t]
        avg_normal = F.normalize(seg_normals.mean(dim=0), dim=0)
        
        d = depth[y_idx_t, x_idx_t]
        valid = d > 0
        

        
        pts_world = pointclouds[y_idx_t, x_idx_t]
        centroid = pts_world.mean(dim=0)
        
        # Find boundary pixels
        kernel = np.ones((3, 3), dtype=bool)
        from scipy import ndimage
        eroded = ndimage.binary_erosion(mask, kernel)
        boundary = mask & ~eroded
        boundary_y, boundary_x = np.where(boundary)
        
        segment_props[seg_id] = {
            'avg_normal': avg_normal,
            'centroid': centroid,
            'pixels': list(zip(y_idx, x_idx)),
            'boundary_pixels': list(zip(boundary_y, boundary_x)),
            'world_points': pts_world
        }
    
    return segment_props


def find_segment_correspondences(
    props1: Dict[int, Dict],
    props2: Dict[int, Dict],
    flow_forward: np.ndarray,
    covis_mask: np.ndarray,
    normal_threshold: float = 0.95,  # Cosine similarity
    overlap_threshold: float = 0.3
) -> List[Tuple[int, int]]:
    """
    Find segment correspondences between two frames using flow.
    
    Returns list of (seg_id1, seg_id2) pairs that should be merged.
    """
    H, W = flow_forward.shape[:2]
    correspondences = []
    
    for seg1_id, props1_seg in props1.items():
        # Check normal similarity with all segments in frame 2
        best_match = None
        best_score = 0
        
        for seg2_id, props2_seg in props2.items():
            # Check normal similarity
            normal_sim = torch.dot(props1_seg['avg_normal'], props2_seg['avg_normal']).item()
            if normal_sim < normal_threshold:
                continue
            
            # Warp segment 1 pixels using flow
            warped_pixels = []
            valid_count = 0
            
            for y, x in props1_seg['boundary_pixels']:
                if covis_mask[y, x]:
                    flow = flow_forward[y, x]
                    y_new = int(round(y + flow[1]))
                    x_new = int(round(x + flow[0]))
                    
                    if 0 <= y_new < H and 0 <= x_new < W:
                        warped_pixels.append((y_new, x_new))
                        valid_count += 1
            
            if valid_count == 0:
                continue
            
            # Check overlap with segment 2
            overlap_count = 0
            for y, x in warped_pixels:
                if (y, x) in props2_seg['pixels']:
                    overlap_count += 1
            
            overlap_ratio = overlap_count / len(props2_seg['boundary_pixels']) if props2_seg['boundary_pixels'] else 0
            
            # Combined score
            score = normal_sim * overlap_ratio
            if score > best_score and overlap_ratio > overlap_threshold:
                best_score = score
                best_match = seg2_id
        
        if best_match is not None:
            correspondences.append((seg1_id, seg2_id))
    
    return correspondences


def merge_segments_across_frames(
    all_frame_segments: List[Dict[int, Dict]],
    all_frame_correspondences: List[List[Tuple[int, int]]],
    num_frames: int
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Merge segments across all frames using union-find.
    
    Returns dict mapping global_segment_id to list of (frame_idx, local_segment_id) pairs.
    """
    # Build union-find structure
    parent = {}
    
    def make_key(frame_idx, seg_id):
        return f"{frame_idx}_{seg_id}"
    
    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Initialize all segments
    for frame_idx in range(num_frames):
        for seg_id in all_frame_segments[frame_idx].keys():
            key = make_key(frame_idx, seg_id)
            find(key)  # Initialize
    
    # Apply correspondences
    for frame_idx, correspondences in enumerate(all_frame_correspondences):
        for seg1_id, seg2_id in correspondences:
            key1 = make_key(frame_idx, seg1_id)
            key2 = make_key(frame_idx + 1, seg2_id)
            union(key1, key2)
    
    # Group segments by root
    global_segments = defaultdict(list)
    for frame_idx in range(num_frames):
        for seg_id in all_frame_segments[frame_idx].keys():
            key = make_key(frame_idx, seg_id)
            root = find(key)
            global_segments[root].append((frame_idx, seg_id))
    
    # Renumber global segments
    final_segments = {}
    for i, (root, members) in enumerate(global_segments.items()):
        final_segments[i] = members
    
    return final_segments




def load_and_resize_flow(
    flow_path: Path,
    target_height: int,
    target_width: int
) -> np.ndarray:
    """
    Load flow from file and resize to target dimensions.
    
    Args:
        flow_path: Path to flow .npy file
        target_height: Target height to resize to
        target_width: Target width to resize to
    
    Returns:
        flow: (H, W, 2) array with resized flow
    """
    # Load flow - expecting shape (2, H_orig, W_orig)
    flow_raw = np.load(flow_path)
    
    if flow_raw.shape[0] == 2:
        # Convert from (2, H, W) to (H, W, 2)
        flow = np.transpose(flow_raw, (1, 2, 0))
    else:
        flow = flow_raw
    
    orig_h, orig_w = flow.shape[:2]
    
    # Scale factors for flow values
    scale_x = target_width / orig_w
    scale_y = target_height / orig_h
    
    # Resize flow using cv2
    flow_resized = cv2.resize(flow, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    
    # Scale flow values to account for resolution change
    flow_resized[..., 0] *= scale_x  # u component
    flow_resized[..., 1] *= scale_y  # v component
    
    return flow_resized


def load_flow_and_covis_data_fixed(
    data_dir: Path,
    frame_indices: List[int],
    target_height: int,
    target_width: int,
    interval: int = 7
) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], np.ndarray]]:
    """
    Load flow and covisibility data for given frame indices with proper resizing.
    
    Args:
        data_dir: Directory containing flow and covis files
        frame_indices: List of frame indices
        target_height: Target height (should match depth/normal maps)
        target_width: Target width (should match depth/normal maps)
        interval: Frame interval
    
    Returns:
        flow_forward: dict mapping (frame_i, frame_j) -> flow array (H, W, 2)
        flow_backward: dict mapping (frame_j, frame_i) -> flow array (H, W, 2)
        covis_masks: dict mapping (frame_i, frame_j) -> covisibility mask (H, W)
    """
    flow_forward = {}
    flow_backward = {}
    covis_masks = {}
    
    for i in range(len(frame_indices) - 1):
        frame_i = frame_indices[i]
        frame_j = frame_indices[i + 1]
        
        # Load and resize forward flow
        flow_file_forward = data_dir / f"flow_{frame_i:05d}_{frame_j:05d}.npy"
        if flow_file_forward.exists():
            flow_forward[(frame_i, frame_j)] = load_and_resize_flow(
                flow_file_forward, target_height, target_width
            )
            print(f"  Loaded forward flow {frame_i}->{frame_j}, shape: {flow_forward[(frame_i, frame_j)].shape}")
        
        # Load and resize backward flow
        flow_file_backward = data_dir / f"flow_{frame_j:05d}_{frame_i:05d}.npy"
        if flow_file_backward.exists():
            flow_backward[(frame_j, frame_i)] = load_and_resize_flow(
                flow_file_backward, target_height, target_width
            )
        
        # Load and resize covisibility mask
        covis_file = data_dir / f"covis_{frame_i:05d}_{frame_j:05d}.npy"
        if covis_file.exists():
            covis_raw = np.load(covis_file)
            # Resize covis mask if needed
            if covis_raw.shape != (target_height, target_width):
                covis_resized = cv2.resize(
                    covis_raw.astype(np.float32), 
                    (target_width, target_height), 
                    interpolation=cv2.INTER_LINEAR
                )
                covis_masks[(frame_i, frame_j)] = covis_resized
            else:
                covis_masks[(frame_i, frame_j)] = covis_raw
            print(f"  Loaded covis mask {frame_i}->{frame_j}, shape: {covis_masks[(frame_i, frame_j)].shape}")
    
    return flow_forward, flow_backward, covis_masks


def robust_plane_ransac(P, n_hint=None, n_iters=500, thresh=0.02):
    """Return n (unit), centre, inlier_mask."""
    best_inliers = None
    best_n = None
    best_c = None
    N = P.shape[0]
    for _ in range(n_iters):
        # try until we sample a non-degenerate triplet
        for _ in range(10):
            idx = torch.randint(0, N, (3,), device=P.device)
            v1, v2, v3 = P[idx]
            n = torch.cross(v2-v1, v3-v1)
            if torch.linalg.norm(n) > 1e-6:
                n = F.normalize(n, dim=0)
                break
        else:  # could not find good triplet – give up
            continue

        # prefer orientation consistent with the hint
        if n_hint is not None and torch.dot(n, n_hint) < 0:
            n = -n

        dists = torch.abs((P @ n) - torch.dot(v1, n))
        inliers = dists < thresh
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers, best_n = inliers, n
            best_c         = P[inliers].mean(0)

            # early exit if we already agree with the hint and have many inliers
            if n_hint is not None and torch.dot(best_n, n_hint) > 0.99 and inliers.sum() > 0.8*N:
                break

    return best_n, best_c, best_inliers

# -------- helpers (scoped) --------
def make_local_frame(n: torch.Tensor):
    n = F.normalize(n, dim=0)
    helper = torch.tensor([1., 0., 0.], device=n.device, dtype=n.dtype)
    if torch.abs(torch.dot(n, helper)) > 0.9:
        helper = torch.tensor([0., 1., 0.], device=n.device, dtype=n.dtype)
    u = torch.linalg.cross(n, helper); u = F.normalize(u, dim=0)
    v = torch.linalg.cross(n, u);      v = F.normalize(v, dim=0)
    return u, v, n

def convex_hull_monotone_chain(xy2d: torch.Tensor) -> torch.Tensor:
    idx = torch.argsort(xy2d[:, 0] + 1e-9 * xy2d[:, 1])
    pts = xy2d[idx].tolist()
    def cross(o, a, b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return torch.tensor(hull, dtype=xy2d.dtype)

def polygon_area(xy2d: torch.Tensor) -> torch.Tensor:
    # xy2d: (K,2) closed polygon (not necessarily closed in memory)
    x = xy2d[:, 0]; y = xy2d[:, 1]
    x2 = torch.roll(x, -1); y2 = torch.roll(y, -1)
    return 0.5 * (x*y2 - y*x2).abs().sum()

def plane_fit_least_squares(P: torch.Tensor):
    """Total least-squares plane fit using ALL points: returns (n, c)."""
    c = P.mean(0)
    Q = P - c
    # smallest singular vector of covariance is the normal
    _, _, Vt = torch.linalg.svd(Q, full_matrices=False)
    n = F.normalize(Vt[-1], dim=0)
    return n, c


def min_area_rect(xy_cpu: torch.Tensor):
    """Return (theta, half2d[2], mid2d[2]) of min-area rectangle over hull."""
    hull = convex_hull_monotone_chain(xy_cpu)
    if hull.shape[0] < 3:
        # fallback: centered 2D PCA
        xy0 = xy_cpu - xy_cpu.mean(0, keepdim=True)
        C = (xy0.T @ xy0) / max(xy0.shape[0]-1, 1)
        _, evecs = torch.linalg.eigh(C)
        x2d = evecs[:, 1]
        theta = torch.atan2(x2d[1], x2d[0])
        R2 = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                            [torch.sin(theta),  torch.cos(theta)]], dtype=xy_cpu.dtype)
        q = xy_cpu @ R2
        mn, mx = q.min(0).values, q.max(0).values
        half = (mx - mn) / 2
        mid  = (mx + mn) / 2
        area_hull = polygon_area(convex_hull_monotone_chain(xy_cpu))
        area_rect = (2*half[0])*(2*half[1])
        rect_ratio = (area_hull / (area_rect + 1e-9)).clamp(max=1.0)
        return theta, half, mid, rect_ratio

    edges = hull.roll(-1, dims=0) - hull
    angs = torch.atan2(edges[:, 1], edges[:, 0])
    best_area = float('inf'); best = None
    for theta in angs:
        cth, sth = torch.cos(theta), torch.sin(theta)
        R2 = torch.tensor([[cth, -sth], [sth, cth]], dtype=xy_cpu.dtype)
        q  = xy_cpu @ R2
        mn, mx = q.min(0).values, q.max(0).values
        half = (mx - mn) / 2
        mid  = (mx + mn) / 2
        area = (2*half[0])*(2*half[1])
        if area < best_area:
            best_area = area; best = (theta, half, mid)
    theta, half, mid = best
    area_hull = polygon_area(hull)
    rect_ratio = (area_hull / (best_area + 1e-9)).clamp(max=1.0)
    return theta, half, mid, rect_ratio

def extent(u: torch.Tensor):
    q = torch.quantile(u, torch.tensor([0.00, 0.99], device=u.device, dtype=u.dtype))
    return (q[1] - q[0]) / 2, (q[1] + q[0]) / 2

def fit_one_plane_box(P: torch.Tensor, n: torch.Tensor, c: torch.Tensor, gid):
    """Fit a single plane-aligned box to points P given plane (n,c). Returns (R_bw, centre, half_sz, rect_ratio)."""
    P0 = P - c
    Pp = P0 - (P0 @ n).unsqueeze(1) * n

    # Optional FPS just for orientation robustness
    Porient = Pp
    '''if Pp.shape[0] > min_fps_points:
        n_samples = max(min_fps_points, int(Pp.shape[0] * fps_ratio))
        try:
            from pytorch3d.ops import sample_farthest_points
            Porient, _ = sample_farthest_points(Pp.unsqueeze(0), K=n_samples, random_start_point=True)
            Porient = Porient.squeeze(0)
            print(f"[{gid}] FPS: {Pp.shape[0]} -> {n_samples} points")
        except Exception:
            idx = torch.randperm(Pp.shape[0], device=Pp.device)[:n_samples]
            Porient = Pp[idx]
    '''
    device= Porient.device
    u0, v0, _ = make_local_frame(n)
    UV = torch.stack([u0, v0], dim=1)                  # (3,2)
    xy_cpu = (Porient @ UV).detach().cpu()             # (M,2)

    theta, half2d, mid2d, rect_ratio = min_area_rect(xy_cpu)

    # Lift 2D axes to 3D
    cth = torch.cos(theta).to(device=device); sth = torch.sin(theta).to(device=device)
    x_axis = (UV.to(device) @ torch.stack([cth, sth]).to(dtype=P.dtype, device=device))
    x_axis = F.normalize(x_axis - (x_axis @ n) * n, dim=0)
    y_axis = F.normalize(torch.linalg.cross(n, x_axis), dim=0)

    # Extents from quantiles in those axes (robust to outliers)
    u_proj = (Pp @ x_axis); v_proj = (Pp @ y_axis)
    u_half, u_mid = extent(u_proj)
    v_half, v_mid = extent(v_proj)

    # Thickness from normal distances
    dist_n = (P0 @ n).abs()
    z_half = torch.quantile(dist_n, torch.tensor(0.95, device=device, dtype=P.dtype)).clamp(min=0.02)

    centre  = c + u_mid * x_axis + v_mid * y_axis
    half_sz = torch.stack([u_half, v_half, z_half]).clamp(min=0.02)
    R_bw = torch.stack([x_axis, y_axis, n], dim=1)
    if torch.det(R_bw) < 0: R_bw[:, 0] = -R_bw[:, 0]
    return R_bw, centre, half_sz, rect_ratio

def split_plane_clusters(P: torch.Tensor, n: torch.Tensor, c: torch.Tensor, gid,
                            tau: float = 0.70, max_splits: int = 3, min_pts: int = 150):
    """
    Greedy 1D median split along principal in-plane axis until each cluster is 'rectangular enough'.
    Returns list of (R_bw, centre, half_sz).
    """
    out = []
    # Work list of index tensors
    P0 = P - c
    Pp = P0 - (P0 @ n).unsqueeze(1) * n
    u0, v0, _ = make_local_frame(n)
    UV = torch.stack([u0, v0], dim=1)
    xy = Pp @ UV

    # clusters start with all points
    clusters = [torch.arange(P.shape[0], device=P.device)]
    attempts = 0
    while attempts < max_splits and len(clusters) > 0:
        idx = clusters.pop(0)
        if idx.numel() < min_pts:
            R_bw, centre, half_sz, _ = fit_one_plane_box(P[idx], n, c, gid)
            out.append((R_bw, centre, half_sz))
            continue

        # rectangularity of this cluster
        theta, half2d, mid2d, rr = min_area_rect((xy[idx]).detach().cpu())
        if rr >= tau or idx.numel() < 2 * min_pts:
            R_bw, centre, half_sz, _ = fit_one_plane_box(P[idx], n, c, gid)
            out.append((R_bw, centre, half_sz))
        else:
            # split along principal axis
            xyc = xy[idx] - xy[idx].mean(0, keepdim=True)
            C = (xyc.T @ xyc) / max(xyc.shape[0]-1, 1)
            _, evecs = torch.linalg.eigh(C)
            main = evecs[:, 1].to(device=P.device, dtype=P.dtype)
            proj = (xy[idx] @ main)
            med = torch.median(proj)
            left  = idx[proj <= med]
            right = idx[proj >  med]
            # ensure both sides large enough; if not, bail out
            if left.numel() < min_pts or right.numel() < min_pts:
                R_bw, centre, half_sz, _ = fit_one_plane_box(P[idx], n, c, gid)
                out.append((R_bw, centre, half_sz))
            else:
                clusters.append(left)
                clusters.append(right)
                attempts += 1
    return out
def process_global_segments(
    global_segments: Dict,
    all_frame_segments: List,
    min_frames: int = 2,
    device: str = 'cuda',
    fps_ratio: float = 0.3,
    min_fps_points: int = 100,
    use_plane_constraint: bool = True, 
    stat_cam: bool = False
) -> Dict[str, List[torch.Tensor]]:
    """
    Process global segments with compact 3D bounding box estimation.

    """
    S_items, R_items, T_items, pts_items = [], [], [], []



    # -------- main loop --------
    pending: Dict[str, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
    candidate_area: Dict[str, float] = {}
    cache_all_points: Dict[str, torch.Tensor] = {}
    cache_avg_normal: Dict[str, torch.Tensor] = {}
 
    for gid, frame_segs in global_segments.items():

        pts_list, normal_list = [], []
        for fi, lid in frame_segs:
            seg = all_frame_segments[fi][lid]
            world_points = seg['world_points'].to(device)
            avg_normal   = seg['avg_normal'].to(device)
            pts_list.append(world_points); normal_list.append(avg_normal)

        P_all = torch.cat(pts_list, 0)
        n_avg = F.normalize(torch.stack(normal_list).mean(0), dim=0)

        cache_all_points[gid] = P_all
        cache_avg_normal[gid] = n_avg
        pending[gid] = []
        rect_ratio_threshold = 0.7

        if use_plane_constraint:
            n, c, inliers = robust_plane_ransac(P_all, n_avg)
            P = P_all[inliers] if inliers.sum() > 50 else P_all

            # single rectangle candidate (for area ranking)
            R_bw1, centre1, half_sz1, rect_ratio = fit_one_plane_box(P, n, c, gid)
            candidate_area[gid] = float((half_sz1[0] * 2) * (half_sz1[1] * 2))

            if rect_ratio < rect_ratio_threshold and P.shape[0] >= 300:
                pieces = split_plane_clusters(P, n, c, gid,
                                              tau=rect_ratio_threshold,
                                              max_splits=3, min_pts=150)
                for R_bw, centre, half_sz in pieces:
                    half_sz = half_sz.clamp(min=0.02)
                    pending[gid].append((R_bw, centre, half_sz))
            else:
                half_sz1 = half_sz1.clamp(min=0.02)
                pending[gid].append((R_bw1, centre1, half_sz1))
        else:
            # No plane constraint: fit plane to ALL points
            n_ls, c_ls = plane_fit_least_squares(P_all)
            R_bw, centre, half_sz, _ = fit_one_plane_box(P_all, n_ls, c_ls, gid)
            half_sz = half_sz.clamp(min=0.02)
            pending[gid].append((R_bw, centre, half_sz))
            candidate_area[gid] = float((half_sz[0] * 2) * (half_sz[1] * 2))

    if len(pending) == 0:
        return dict(S_items=[], R_items=[], T_items=[], pts_items=[])

    # -------- select ground by largest 2D area; refit using ALL its points --------
    '''if not stat_cam:
      gid_ground = max(candidate_area.keys(), key=lambda g: candidate_area[g])
      P_ground_full = cache_all_points[gid_ground]

      # Plane from ALL ground points
      n_g, c_g = plane_fit_least_squares(P_ground_full)
      R_bw_g, centre_g, half_sz_g, _ = fit_one_plane_box(P_ground_full, n_g, c_g, gid_ground)

      # Force fixed thickness (Z half-size)
      half_sz_g = half_sz_g.clone()
      half_sz_g[2] = 0.05#max(ground_fixed_thickness * 0.5, 0.01)
      half_sz_g = half_sz_g.clamp(min=0.02)

      # Replace any pending boxes for this gid with the single ground box
      pending[gid_ground] = [(R_bw_g, centre_g, half_sz_g)]'''

    # -------- materialize outputs --------
    for gid, boxes in pending.items():
        P_all = cache_all_points[gid]
        for (R_bw, centre, half_sz) in boxes:
            S_items.append(torch.log(half_sz))
            R_items.append(R_bw.t().contiguous())  # world→body
            T_items.append(centre)
            pts_items.append(P_all.cpu())

    return dict(S_items=S_items, R_items=R_items, T_items=T_items, pts_items=pts_items)


def find_segment_correspondences_improved(
    seg_map1: np.ndarray,
    seg_map2: np.ndarray,
    props1: Dict[int, Dict],
    props2: Dict[int, Dict],
    flow_forward: np.ndarray,
    covis_mask: np.ndarray,
    normal_threshold: float = 0.9,
    min_overlap_pixels: int = 50,
    overlap_ratio_threshold: float = 0.1
) -> List[Tuple[int, int]]:
    """
    Find segment correspondences between two frames using flow warping.
    
    The logic is:
    1. For each segment in frame 1
    2. Warp all its pixels to frame 2 using optical flow
    3. Check which segments in frame 2 the warped pixels land on
    4. If enough pixels land on a segment in frame 2 AND normals are similar, mark as correspondence
    
    Args:
        seg_map1: (H, W) segmentation map for frame 1
        seg_map2: (H, W) segmentation map for frame 2
        props1: Segment properties for frame 1
        props2: Segment properties for frame 2
        flow_forward: (H, W, 2) optical flow from frame 1 to frame 2
        covis_mask: (H, W) covisibility mask
        normal_threshold: Minimum cosine similarity for normals
        min_overlap_pixels: Minimum number of pixels that must overlap
        overlap_ratio_threshold: Minimum ratio of warped pixels that must overlap
    
    Returns:
        List of (seg_id1, seg_id2) correspondence pairs
    """
    H, W = flow_forward.shape[:2]
    correspondences = []
    correspondence_scores = {}
    
    # For each segment in frame 1
    for seg1_id, props1_seg in props1.items():
        # Get all pixels for this segment
        seg1_pixels = props1_seg['pixels']
        
        # Track where warped pixels land in frame 2
        seg2_overlap_counts = {}
        valid_warped_count = 0
        
        # Warp each pixel using flow
        for y, x in seg1_pixels:
            # Check if this pixel is covisible
            if not covis_mask[y, x]:
                continue
            
            # Get flow at this pixel
            flow = flow_forward[y, x]
            
            # Warp to frame 2
            x_new = x + flow[0]
            y_new = y + flow[1]
            
            # Round to nearest pixel
            x_new = int(round(x_new))
            y_new = int(round(y_new))
            
            # Check if warped position is within bounds
            if 0 <= x_new < W and 0 <= y_new < H:
                valid_warped_count += 1
                
                # Check which segment this lands on in frame 2
                seg2_id = seg_map2[y_new, x_new]
                
                if seg2_id >= 0:  # Valid segment
                    if seg2_id not in seg2_overlap_counts:
                        seg2_overlap_counts[seg2_id] = 0
                    seg2_overlap_counts[seg2_id] += 1
        
        # No valid warped pixels
        if valid_warped_count == 0:
            continue
        
        # Check each potential match in frame 2
        for seg2_id, overlap_count in seg2_overlap_counts.items():
            if seg2_id not in props2:
                continue
            
            # Check if enough pixels overlap
            if overlap_count < min_overlap_pixels:
                continue
            
            # Calculate overlap ratio
            overlap_ratio = overlap_count / valid_warped_count
            if overlap_ratio < overlap_ratio_threshold:
                continue
            
            normal_sim = torch.dot(
                props1_seg['avg_normal'], 
                props2[seg2_id]['avg_normal']
            ).item()
            
            if normal_sim < normal_threshold:
                continue
            
            # Calculate combined score
            score = normal_sim * overlap_ratio * (overlap_count / 100.0)
            
            # Store correspondence with score
            key = (seg1_id, seg2_id)
            if key not in correspondence_scores or correspondence_scores[key] < score:
                correspondence_scores[key] = score
    
    # Extract final correspondences (avoiding duplicates)
    seg1_matched = set()
    seg2_matched = set()
    
    # Sort by score and take best matches
    sorted_matches = sorted(correspondence_scores.items(), key=lambda x: x[1], reverse=True)
    
    for (seg1_id, seg2_id), score in sorted_matches:
        # Each segment can only match once
        if seg1_id not in seg1_matched and seg2_id not in seg2_matched:
            correspondences.append((seg1_id, seg2_id))
            seg1_matched.add(seg1_id)
            seg2_matched.add(seg2_id)
            print(f"    Match: Seg {seg1_id} -> Seg {seg2_id}, score: {score:.3f}")
    
    return correspondences

import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from typing import Dict, List, Tuple, Optional

# --- 1) 仅用 3D 点生成 1–2 个“薄盒”接触体 ------------------------------------

def build_contact_planes_from_points_3d(
    contact_points_3d: np.ndarray | torch.Tensor,
    max_planes: int = 2,
    ransac_thresh: float = 0.02,
    min_inliers: int = 80,
    fixed_thickness: float = 0.02,   # 盒子厚度（总厚度=2*half_z）
    device: str = "cuda",
) -> Dict[str, List[torch.Tensor]]:
    """
    输入:
      contact_points_3d: (N,3) 世界坐标
    输出:
      primitives dict: S_items(log half), R_items(world→body), T_items(world), pts_items
    依赖: robust_plane_ransac, fit_one_plane_box, estimate_normals_knn (你已有)
    """
    if isinstance(contact_points_3d, torch.Tensor):
        P_all = contact_points_3d.to(device=device, dtype=torch.float32)
    else:
        P_all = torch.from_numpy(np.asarray(contact_points_3d)).to(device=device, dtype=torch.float32)
    if P_all.numel() == 0:
        return dict(S_items=[], R_items=[], T_items=[], pts_items=[])

    # 估一个 hint 法向（可选）
    try:
        n_est = estimate_normals_knn(P_all, k=min(20, max(4, P_all.shape[0]//50)))
        n_hint = F.normalize(n_est.mean(0), dim=0)
    except Exception:
        n_hint = None

    used = torch.zeros(P_all.shape[0], dtype=torch.bool, device=device)
    S_items, R_items, T_items, pts_items = [], [], [], []

    for _ in range(max_planes):
        free_idx = (~used).nonzero(as_tuple=False).squeeze(-1)
        if free_idx.numel() < min_inliers:
            break
        Q = P_all[free_idx]

        n, c, inl = robust_plane_ransac(Q, n_hint=n_hint, n_iters=500, thresh=ransac_thresh)
        if inl is None or inl.sum().item() < min_inliers:
            break

        inl_full = free_idx[inl]
        # 在平面内做最小包围矩形 → OBB
        R_bw, centre, half_sz, rect_ratio = fit_one_plane_box(Q[inl], n, c, gid=0)

        # 锁定法向厚度为一个很薄的值
        half_sz = half_sz.clone()
        half_sz[2] = max(fixed_thickness * 0.5, 0.005)

        S_items.append(torch.log(half_sz))
        R_items.append(R_bw.t().contiguous())   # world→body
        T_items.append(centre)
        pts_items.append(P_all.detach().cpu())

        used[inl_full] = True
        n_hint = n  # 下一轮的 hint

    return dict(S_items=S_items, R_items=R_items, T_items=T_items, pts_items=pts_items)

# --- 2) 盒子↔场景吸附（仅做“贴面 + 轻微对齐”），最后合并并返回 --------------------

def _axis_world_from_Rwb(R_wb: torch.Tensor, axis: str = 'z') -> torch.Tensor:
    """从 world→body 矩阵取 body 轴在世界坐标下的向量。（R_bw = R_wb^T）"""
    if isinstance(R_wb, np.ndarray):
        R_wb = torch.from_numpy(R_wb)
    R_bw = R_wb.t()
    a = {'x': 0, 'y': 1, 'z': 2}[axis]
    return R_bw[:, a]  # (3,)

def _closest_face_on_box(T: np.ndarray | torch.Tensor,
                         S_half: np.ndarray | torch.Tensor,
                         n_face: torch.Tensor,
                         R_wb: np.ndarray | torch.Tensor) -> Tuple[torch.Tensor, float]:
    """
    给定盒子(world→body=R_wb, 中心T, 半尺寸S_half)与一个候选面法向 n_face(世界系)，
    计算该面所在的世界系平面点 P = T + d * n_face 与 d（有符号距离, n_face 指向外侧）。
    """
    if isinstance(T, np.ndarray):     T = torch.from_numpy(T)
    if isinstance(S_half, np.ndarray): S_half = torch.from_numpy(S_half)
    if isinstance(R_wb, np.ndarray):  R_wb = torch.from_numpy(R_wb)

    # box 的 world 轴
    R_bw = R_wb.t()
    axes = [R_bw[:,0], R_bw[:,1], R_bw[:,2]]
    # 在 n_face 方向上，正负两个面中的那个与 n_face 同向（外法向）
    # 距离为对应半尺寸在该轴方向上的投影之和，但对正交盒其实就是 S_half·|axis·n_face| 的符号组合。
    # 我们直接选与 n_face 夹角最大的那个轴作为“主支撑”来近似确定 d。
    dots = torch.tensor([torch.dot(a, n_face).abs() for a in axes])
    i = int(torch.argmax(dots))
    sign = torch.sign(torch.dot(axes[i], n_face))  # 选与 n_face 同向的那一面
    d = sign * S_half[i]
    P = T + d * axes[i]  # 面心近似（足够用于贴合）
    # 返回在 n_face 方向上的有符号距离：近似用 d
    return P, float(d)

def snap_contact_box_to_scene(
    C: Dict, scene_boxes: List[Dict],
    max_dist: float = 0.25,
    normal_dot_thresh: float = 0.5,
    contact_axis: str = 'z',
    surface_epsilon: float = 0.005
):
    """
    把接触盒 C 贴到最近、法向相容的场景盒某一面上。
    C/scene 里字段：
      'S' (half, 世界尺度), 'R' (world→body), 'T' (世界中心)
    返回: (C_snapped, info)
    """
    Cn = _axis_world_from_Rwb(C['R'], contact_axis)          # 接触面法向（世界系）
    Cn = F.normalize(Cn, dim=0)

    T_c = torch.as_tensor(C['T'], dtype=torch.float32)
    S_c = torch.as_tensor(C['S'], dtype=torch.float32)

    best = None
    for sb in scene_boxes:
        S_s = torch.as_tensor(sb['S'], dtype=torch.float32)
        R_s = torch.as_tensor(sb['R'], dtype=torch.float32)
        T_s = torch.as_tensor(sb['T'], dtype=torch.float32)

        # 场景盒 6 个外法向（世界系）
        nxs = [_axis_world_from_Rwb(R_s, a) for a in ['x','y','z']]
        face_normals = nxs + [-n for n in nxs]

        for n_s in face_normals:
            n_s = F.normalize(n_s, dim=0)
            if torch.dot(n_s, Cn) <= normal_dot_thresh:
                continue

            # 找该面中心近似位置
            P_face, _ = _closest_face_on_box(T_s, S_s, n_s, R_s)

            # 估算“沿法向”的距离
            d = torch.dot((P_face - T_c), n_s).item()
            dist_abs = abs(d)
            if dist_abs > max_dist:
                continue

            score = dist_abs  # 越小越好
            if best is None or score < best[0]:
                best = (score, n_s, P_face, d, sb)

    if best is None:
        return C, {'connected': False}

    _, n_s, P_face, d, sb = best

    # 仅沿法向平移，使接触盒表面贴到场景面上，留一点点间隙
    T_new = T_c + (d - S_c[2] + surface_epsilon) * n_s   # 这里默认接触轴是 z → 用 S_c[2]
    # 若你的接触轴不是 z，上面这行把 2 改为 {'x':0,'y':1,'z':2}[contact_axis]

    C_snapped = dict(S=C['S'].copy(), R=C['R'].copy(), T=T_new.cpu().numpy())
    return C_snapped, {'connected': True, 'mode': 'face_snap'}

def make_contacts_then_snap_and_merge_3d(
    scene_results: Dict[str, List],      # 场景 primitives（S=log(half), R=world→body, T=world）
    contact_points_3d: np.ndarray | torch.Tensor,
    *,
    # 平面/薄盒
    max_planes: int = 2,
    ransac_thresh: float = 0.02,
    min_inliers: int = 80,
    fixed_thickness: float = 0.02,
    # 吸附
    max_dist: float = 2.5,
    normal_dot_thresh: float = 0.5,
    contact_axis: str = 'z',
    surface_epsilon: float = 0.005,
    drop_unconnected: bool = True,
    in_place: bool = False,
    device: str = "cuda",
):
    """
    纯 3D 版“一条龙”：
      contact_points_3d → 1–2 薄盒 → 吸附到 scene → 合并 & 返回
    """
    # 1) 生成接触薄盒
    results_extra = build_contact_planes_from_points_3d(
        contact_points_3d,
        max_planes=max_planes,
        ransac_thresh=ransac_thresh,
        min_inliers=min_inliers,
        fixed_thickness=fixed_thickness,
        device=device,
    )

    # 2) 和 scene 合并一个视图（方便在尾部迭代吸附）
    merged = merge_primitives_dicts(scene_results, results_extra)

    # 3) 准备场景盒（前半部分）
    total = len(merged['S_items'])
    n_contact = len(results_extra['S_items'])
    start_idx = total - n_contact

    scene = []
    for S, R, T in zip(merged['S_items'][:start_idx], merged['R_items'][:start_idx], merged['T_items'][:start_idx]):
        scene.append({'S': torch.exp(S).cpu().numpy(),
                      'R': R.cpu().numpy(),
                      'T': T.cpu().numpy()})

    # 4) 逐个接触盒吸附
    out = merged if in_place else deepcopy(merged)
    attached, failed = [], []
    for i in range(start_idx, total):
        C = {'S': torch.exp(out['S_items'][i]).cpu().numpy(),
             'R': out['R_items'][i].cpu().numpy(),
             'T': out['T_items'][i].cpu().numpy(),
             'idx': i}
        C_snap, info = snap_contact_box_to_scene(
            C, scene,
            max_dist=max_dist,
            normal_dot_thresh=normal_dot_thresh,
            contact_axis=contact_axis,
            surface_epsilon=surface_epsilon
        )
        if info.get('connected', False):
            out['T_items'][i] = torch.from_numpy(C_snap['T']).to(out['T_items'][i].device)
            # S/R 不变（只是平移）
            attached.append({'idx': i, 'mode': info.get('mode', 'face_snap')})
        else:
            failed.append(i)

    # 5) 可选：丢弃未吸附成功的接触盒
    if drop_unconnected and failed:
        keep = [True]*total
        for i in failed: keep[i] = False
        for k in ['S_items','R_items','T_items','pts_items']:
            if k in out:
                out[k] = [v for v, m in zip(out[k], keep) if m]

    report = {
        'n_scene': start_idx,
        'n_contact_planes': n_contact,
        'n_attached': len(attached),
        'attached': attached,
        'failed': failed,
    }
    return out, report


def process_global_points(
    world_points: torch.Tensor,
    min_frames: int = 2,
    device: str = 'cuda',
    fps_ratio: float = 0.3,
    min_fps_points: int = 100,
    use_plane_constraint: bool = True
) -> Dict[str, List[torch.Tensor]]:
    """
    Process global segments with compact 3D bounding box estimation.

    """
    S_items, R_items, T_items, pts_items = [], [], [], []

    # for gid, frame_segs in global_segments.items():
    gid = 0
    if True: 
        pts_list, normal_list = [], []
        # for fi, lid in frame_segs:
        # seg = all_frame_segments[fi][lid]
        world_points = world_points.to(device)
        avg_normal = estimate_normals_knn(world_points, k=20).to(device)
        avg_normal = avg_normal.mean(dim=0)
        avg_normal = avg_normal / avg_normal.norm()  # normalize to unit length


        assert world_points.dim() == 2 and world_points.shape[1] == 3, f"world_points must be (N,3), got {world_points.shape}"
        assert avg_normal.dim() == 1 and avg_normal.shape[0] == 3,     f"avg_normal must be (3,), got {avg_normal.shape}"
        pts_list.append(world_points); normal_list.append(avg_normal)

        P_all = torch.cat(pts_list, 0)
        assert P_all.dim() == 2 and P_all.shape[1] == 3, f"Concatenated points must be (N, 3), got {P_all.shape}"
        n_avg = F.normalize(torch.stack(normal_list).mean(0), dim=0)

        if use_plane_constraint:

            n, c, inliers = robust_plane_ransac(P_all, n_avg)
            planarity_ratio = (inliers.float().mean()).item()
            print(f"[{gid}] Using plane-constrained bbox (planarity: {planarity_ratio:.2f})")
            P = P_all[inliers] if inliers.sum() > 50 else P_all

            # first try single rectangle
            R_bw1, centre1, half_sz1, rect_ratio = fit_one_plane_box(P, n, c, gid)

            if rect_ratio < 0.70 and P.shape[0] >= 300:
                print(f"[{gid}] Rectangularity {rect_ratio:.2f} too low → splitting into sub-planes")
                pieces = split_plane_clusters(P, n, c, gid, tau=0.70, max_splits=3, min_pts=150)
                for R_bw, centre, half_sz in pieces:
                    half_sz = half_sz.clamp(min=0.02)
                    S_items.append(torch.log(half_sz))
                    R_items.append(R_bw.t().contiguous())
                    T_items.append(centre)
                    pts_items.append(P_all.cpu())
                    print(f"[{gid}] Sub-box dims: {(half_sz * 2).detach().cpu().numpy()}")
                # continue  # go to next gid

            # single box path
            half_sz = half_sz1.clamp(min=0.02)
            R_bw, centre = R_bw1, centre1


        # clamp & store
        half_sz = half_sz.clamp(min=0.02)
        S_items.append(torch.log(half_sz))
        R_items.append(R_bw.t().contiguous())  # world→body
        T_items.append(centre)
        pts_items.append(P_all.cpu())

        # checks
        assert S_items[-1].shape == (3,), f"S_item must be (3,), got {S_items[-1].shape}"
        assert R_items[-1].shape == (3, 3), f"R_item must be (3, 3), got {R_items[-1].shape}"
        assert T_items[-1].shape == (3,), f"T_item must be (3,), got {T_items[-1].shape}"
        print(f"[{gid}] Bbox dimensions: {(half_sz * 2).cpu().numpy()}")

    return dict(S_items=S_items, R_items=R_items, T_items=T_items, pts_items=pts_items)
def interval_flow_segmentation_pipeline_with_vis(
    mono_normals: torch.Tensor,  # (T, H, W, 3)
    depthmaps: torch.Tensor,     # (T, H, W)
    pointclouds: torch.Tensor,
    data_dir: Path,
    frame_indices: List[int],
    interval: int = 7,
    device: str = 'cuda',
    save_debug: bool = True,
    debug_dir: Path = Path('debug_segments'), 
    contact_points: Optional[np.ndarray] = None,
    stat_cam=False, 
) -> Dict[str, List]:
    """
    Complete pipeline with fixed flow loading and correspondence finding.
    """
    num_frames = len(frame_indices)
    H, W = depthmaps.shape[1:3]
    
    print(f"Processing {num_frames} frames with resolution {H}x{W}")
    
    # Step 1: Segment each frame independently
    print("\nStep 1: Segmenting individual frames...")
    all_frame_segments = []
    all_seg_maps = []
    eps_spatial = 2.0
    if stat_cam:
      # eps_spatial = 4.0
      frame_indices = [0]
      num_frames = 1
    for i in range(num_frames):
        seg_map, num_segs = segment_single_frame_normals(
            mono_normals[i],
            depthmaps[i],
            eps_spatial=eps_spatial,
            n_normal_clusters=8,
            min_samples=10,
            min_points=22,
            device=device
        )
        
        seg_props = compute_segment_properties(
            seg_map,
            mono_normals[i],
            depthmaps[i],
            pointclouds[i],
            device=device
        )
        
        all_frame_segments.append(seg_props)
        all_seg_maps.append(seg_map)
        print(f"  Frame {frame_indices[i]}: {len(seg_props)} segments")

    
    if stat_cam:
        global_segments = build_global_segments_single_view(all_frame_segments)
        

    else:
        # Step 2: Load flow data with proper resizing
        print("\nStep 2: Loading and resizing flow data...")
        flow_forward, flow_backward, covis_masks = load_flow_and_covis_data_fixed(
            data_dir, frame_indices, H, W, interval
        )
        
        # Optional: visualize flow
        if save_debug:
            debug_dir.mkdir(parents=True, exist_ok=True)
            # You can uncomment this if you want flow visualization
            # batch_visualize_flow_data(flow_forward, flow_backward, covis_masks, debug_dir / 'flow_vis')
        
        # Step 3: Find correspondences with improved method
        print("\nStep 3: Finding greedy segment correspondences across ALL frames …")
        correspondence_pairs = []            # (fi, seg_i, fj, seg_j)

        # map real frame-number → position index 0…num_frames-1
        frame_to_pos = {f: i for i, f in enumerate(frame_indices)}

        for (fi, fj), flow in flow_forward.items():
            if (fi, fj) not in covis_masks:          # need covis mask too
                continue
            i_pos, j_pos = frame_to_pos[fi], frame_to_pos[fj]

            corr = find_segment_correspondences_improved(
                all_seg_maps[i_pos], all_seg_maps[j_pos],
                all_frame_segments[i_pos], all_frame_segments[j_pos],
                flow, covis_masks[(fi, fj)],
                normal_threshold=0.85,
                min_overlap_pixels=20,
                overlap_ratio_threshold=0.01
            )
            print(f"  {fi}->{fj}: {len(corr)} matches")
            for seg_i, seg_j in corr:
                correspondence_pairs.append((i_pos, seg_i, j_pos, seg_j))

        print(f"\nStep 4: Union-fusing {len(correspondence_pairs)} correspondences …")
        global_segments = build_global_segments_greedy(
            all_frame_segments, correspondence_pairs
        )
        print(f"  ↳ {len(global_segments)} global segments after greedy fusion")
        print(f"  Found {len(global_segments)} global segments")
        
        # Save debug visualizations if requested
    if save_debug:
      print("\nSaving debug visualizations...")

      visualize_per_frame_segments(all_seg_maps, frame_indices, debug_dir)
      visualize_merged_segments(all_seg_maps, global_segments, frame_indices, debug_dir)
      save_segment_statistics(
          all_seg_maps,
          all_frame_segments,
          global_segments,
          correspondence_pairs,
          frame_indices,
          debug_dir
      )
    # Step 5: Process global segments into primitives
    print("\nStep 5: Fitting primitives to global segments...")
    results = process_global_segments(
        global_segments,
        all_frame_segments,
        min_frames=2,
        device=device
    )

    results_extra = {}
    if contact_points is not None and len(contact_points) > 0:
        results_extra = process_global_points(
            torch.from_numpy(contact_points),
            min_frames=2,
            device=device
        )

    # results = merge_primitives_dicts(results, results_extra)
    results, rep = make_contacts_then_snap_and_merge_3d(
        scene_results=results,
        contact_points_3d=contact_points,   # (N,3) 世界坐标
        max_planes=2,
        fixed_thickness=0.02,
        drop_unconnected=True,
        device='cuda',
    )
    print(rep)


    # process_global_points
    
    print(f"\n✓ Generated {len(results['S_items'])} primitives from {len(global_segments)} global segments")
    
    return results


# Modified functions for scene-grounded contact points across all timesteps

def collect_all_stable_contact_points_enhanced(
    *,
    contacted_masks: np.ndarray,                    # [T, V] bool
    static_segments: List[Tuple[int, int]],
    per_part: Dict[str, Dict],                      # Contains per-part data
    pred_vert: np.ndarray,                          # [T, V, 3]
    part_ids_list: List,
    body_part_params: Optional[Dict[str, Dict]] = None,
    min_contact_quality: float = 0.3,               
    temporal_sampling: str = "all_stable"           # Changed default to get all stable frames
) -> Tuple[List[np.ndarray], Dict[str, List[np.ndarray]], Optional[np.ndarray]]:
    """
    Collect ALL stable contact points across time intervals for scene grounding.
    Modified to collect points from all stable contact frames, not just best frames.
    
    Returns:
      per_segment_points: list of (N_seg,3) arrays; contact points for each segment
      per_part_points: dict mapping part names to list of (N_part_seg,3) arrays per segment  
      all_contact_points: concatenated (∑N,3) array across all segments and parts
    """
    
    T, V = pred_vert.shape[:2]
    
    per_segment_points = []
    per_part_points = {part_name: [] for part_name in PART_ORDER}
    all_collected_points = []
    
    print(f"Collecting ALL stable contact points across {len(static_segments)} segments...")
    print(f"Temporal sampling strategy: {temporal_sampling}")
    
    for seg_idx, (s, e) in enumerate(static_segments):
        print(f"\nSegment {seg_idx}: frames {s}-{e} (duration: {e-s})")
        segment_all_points = []
        
        # For each body part, collect points from ALL stable contact frames in this segment
        for part_idx, part_name in enumerate(PART_ORDER):
            part_data = per_part[part_name]
            
            # Use the intersection mask (contact AND low velocity) for this part
            intersection_mask = part_data.get("intersection_mask", part_data["has_contact_mask"])
            part_ids = np.array(part_ids_list[part_idx])
            
            # Collect from ALL frames where this part has stable contact in this segment
            part_segment_points = []
            frames_used = 0
            
            for frame_idx in range(s, e):
                # Check if this frame has stable contact for this part
                if intersection_mask[frame_idx]:
                    # Get contact mask for this part at this frame
                    if "mask_partspace" in part_data:
                        part_contact_mask = part_data["mask_partspace"][frame_idx]
                        if part_contact_mask.shape[0] != len(part_ids):
                            continue
                        contacting_vertices = pred_vert[frame_idx][part_ids][part_contact_mask]
                    else:
                        full_contact_mask = part_data["mask"][frame_idx]
                        part_contact_mask = full_contact_mask[part_ids]
                        contacting_vertices = pred_vert[frame_idx][part_ids][part_contact_mask]
                    
                    if contacting_vertices.shape[0] > 0:
                        part_segment_points.append(contacting_vertices)
                        frames_used += 1
            
            # Combine all frames for this part in this segment
            if part_segment_points:
                part_all_points = np.concatenate(part_segment_points, axis=0)
                per_part_points[part_name].append(part_all_points)
                segment_all_points.append(part_all_points)
                print(f"  {part_name}: {part_all_points.shape[0]} points from {frames_used} frames")
            else:
                per_part_points[part_name].append(np.empty((0, 3)))
                print(f"  {part_name}: no contact points")
        
        # Combine all parts for this segment
        if segment_all_points:
            segment_combined = np.concatenate(segment_all_points, axis=0)
            per_segment_points.append(segment_combined)
            all_collected_points.append(segment_combined)
            print(f"Segment {seg_idx} total: {segment_combined.shape[0]} contact points")
        else:
            per_segment_points.append(np.empty((0, 3)))
            print(f"Segment {seg_idx}: no contact points")
    
    # Concatenate all segments
    if all_collected_points:
        all_contact_points = np.concatenate(all_collected_points, axis=0)
        print(f"\nFINAL: {all_contact_points.shape[0]} total contact points across all segments and timesteps")
    else:
        all_contact_points = None
        print(f"\nFINAL: No contact points found")
    
    return per_segment_points, per_part_points, all_contact_points


def select_frames_and_collect_contacts_enhanced(
    *,
    # outputs from analyze_contacts_5parts(...)
    contacted_masks: np.ndarray,                    # [T, V] bool
    static_segments: List[Tuple[int, int]],
    best_frames_global: List[Optional[int]],
    counts_global: np.ndarray,                      # [T]
    per_part: Dict[str, Dict],                      # Contains per-part best_frames and masks
    # geometry
    pred_vert: np.ndarray,                          # [T, V, 3]
    # policy
    part_ids_list: List, 
    policy: str = "all_stable",                     # Changed default to collect all stable frames
    body_part_params: Optional[Dict[str, Dict]] = None,
    enforce_lowvel_run: bool = True,
    vel_threshold_global: Optional[float] = None,
    min_run_global: Optional[int] = None,
) -> Tuple[List[Optional[int]], List[np.ndarray], Optional[np.ndarray]]:
    """
    Modified to collect ALL stable contact points across all timesteps, not just best frames.
    This enables proper scene grounding by having more comprehensive contact data.
    
    Returns:
      selected_frames: one frame index (or None) per segment (kept for compatibility)
      per_frame_pcs:   list of (Ni,3) arrays; each is contact points for that segment
      all_points:      concatenated (∑Ni,3) array across all parts and timesteps
    """
    T, V = pred_vert.shape[:2]
    
    # Use the enhanced collection function
    per_segment_points, per_part_points, all_points = collect_all_stable_contact_points_enhanced(
        contacted_masks=contacted_masks,
        static_segments=static_segments,
        per_part=per_part,
        pred_vert=pred_vert,
        part_ids_list=part_ids_list,
        body_part_params=body_part_params,
        temporal_sampling="all_stable"  # Collect from all stable frames
    )
    
    # For compatibility, still return selected frames (can be the best frames)
    selected_frames = best_frames_global[:len(static_segments)]
    
    print(f"\nCollected contact points from ALL stable timesteps for scene grounding")
    print(f"Total segments: {len(per_segment_points)}")
    if all_points is not None:
        print(f"Total contact points: {all_points.shape[0]}")
    
    return selected_frames, per_segment_points, all_points


def collect_all_timestep_contact_points(
    contacted_masks: np.ndarray,                    # [T, V] bool
    static_segments: List[Tuple[int, int]],
    per_part: Dict[str, Dict],                      # Contains per-part data
    pred_vert: np.ndarray,                          # [T, V, 3]
    part_ids_list: List,
    body_part_params: Optional[Dict[str, Dict]] = None
) -> Dict[str, np.ndarray]:
    """
    Collect ALL contact points across ALL timesteps for each body part.
    
    Returns:
        Dict mapping part_name to (N_part, 3) array of all contact points across time
    """
    T, V = pred_vert.shape[:2]
    all_part_contacts = {}
    
    print(f"\nCollecting contact points across {T} timesteps...")
    
    for part_idx, part_name in enumerate(PART_ORDER):
        part_data = per_part[part_name]
        part_ids = np.array(part_ids_list[part_idx])
        
        # Use intersection mask (contact AND low velocity)
        intersection_mask = part_data.get("intersection_mask", part_data["has_contact_mask"])
        
        part_contact_points = []
        frames_with_contact = 0
        
        # Collect from ALL timesteps where contact occurs
        for t in range(T):
            if intersection_mask[t]:
                # Get contact mask for this part at this timestep
                if "mask_partspace" in part_data:
                    part_contact_mask = part_data["mask_partspace"][t]
                    if part_contact_mask.shape[0] == len(part_ids):
                        contacting_vertices = pred_vert[t][part_ids][part_contact_mask]
                        if contacting_vertices.shape[0] > 0:
                            part_contact_points.append(contacting_vertices)
                            frames_with_contact += 1
                else:
                    full_contact_mask = part_data["mask"][t]
                    part_contact_mask = full_contact_mask[part_ids]
                    contacting_vertices = pred_vert[t][part_ids][part_contact_mask]
                    if contacting_vertices.shape[0] > 0:
                        part_contact_points.append(contacting_vertices)
                        frames_with_contact += 1
        
        # Concatenate all timesteps for this part
        if part_contact_points:
            all_points = np.concatenate(part_contact_points, axis=0)
            all_part_contacts[part_name] = all_points
            print(f"  {part_name}: {all_points.shape[0]} points from {frames_with_contact} timesteps")
        else:
            all_part_contacts[part_name] = np.empty((0, 3))
            print(f"  {part_name}: no contact points")
    
    return all_part_contacts

def align_contact_planes_to_scene(
    contact_planes: Dict[str, List[Dict]],
    scene_primitives: Dict[str, List],
    normal_threshold: float = 0.95,
    distance_threshold: float = 0.05,
    overlap_threshold: float = 0.1
) -> Tuple[List[Dict], Dict]:
    """
    Align contact planes with scene primitives. Merge coplanar or connect adjacent planes.
    
    Returns:
        aligned_primitives: List of aligned contact plane primitives
        alignment_info: Dict with alignment statistics
    """

    
    aligned_prims = []
    alignment_info = {
        'merged': 0,
        'connected': 0,
        'independent': 0,
        'details': []
    }
    
    # Convert scene primitives to plane representation
    scene_planes = []
    for i, (S, R, T) in enumerate(zip(
        scene_primitives['S_items'],
        scene_primitives['R_items'], 
        scene_primitives['T_items']
    )):
        # R is world→body, so R.T is body→world
        R_bw = R.t() if isinstance(R, torch.Tensor) else R.T
        normal = R_bw[:, 2]  # z-axis of box is normal
        normal = normal / np.linalg.norm(normal)
        
        S_exp = torch.exp(S) if isinstance(S, torch.Tensor) else np.exp(S)
        
        scene_planes.append({
            'normal': normal,
            'center': T,
            'half_size': S_exp,
            'R_bw': R_bw,
            'idx': i
        })
    
    # Process each contact plane
    for part_name, planes in contact_planes.items():
        for contact_plane in planes:
            contact_n = contact_plane['normal']
            contact_T = contact_plane['T']
            contact_d = contact_plane['plane_d']
            
            best_match = None
            best_score = 0
            best_type = 'independent'
            
            # Check alignment with each scene plane
            for scene_plane in scene_planes:
                scene_n = scene_plane['normal']
                scene_T = scene_plane['center']
                
                # Check normal similarity
                if isinstance(scene_n, torch.Tensor):
                    scene_n = scene_n.cpu().numpy()
                if isinstance(scene_T, torch.Tensor):
                    scene_T = scene_T.cpu().numpy()
                
                normal_dot = np.abs(np.dot(contact_n, scene_n))
                
                if normal_dot < normal_threshold:
                    continue
                
                # Check if planes are coplanar
                scene_d = np.dot(scene_n, scene_T)
                plane_distance = abs(contact_d - scene_d)
                
                if plane_distance < distance_threshold:
                    # Coplanar - merge them
                    score = normal_dot * (1.0 / (plane_distance + 0.001))
                    if score > best_score:
                        best_score = score
                        best_match = scene_plane
                        best_type = 'merged'
                elif plane_distance < 2:  # Adjacent planes
                    # Check for edge connection
                    score = normal_dot * 0.5  # Lower score for connections
                    if score > best_score:
                        best_score = score
                        best_match = scene_plane
                        best_type = 'connected'
            
            # Apply alignment
            if best_type == 'merged' and best_match is not None:
                # Align to scene plane exactly
                aligned_plane = contact_plane.copy()
                aligned_plane['T'] = best_match['center']
                aligned_plane['normal'] = best_match['normal']
                aligned_prims.append(aligned_plane)
                alignment_info['merged'] += 1
                alignment_info['details'].append({
                    'part': part_name,
                    'type': 'merged',
                    'scene_idx': best_match['idx']
                })
                
            elif best_type == 'connected' and best_match is not None:
                # Keep orientation but snap position to be adjacent
                aligned_plane = contact_plane.copy()
                # Project contact center onto scene plane and offset by thickness
                to_contact = contact_T - best_match['center']
                dist_along_normal = np.dot(to_contact, best_match['normal'])
                projected = contact_T - dist_along_normal * best_match['normal']
                
                # Offset by combined half thicknesses
                offset = (contact_plane['S'][2] + best_match['half_size'][2]) * best_match['normal']
                aligned_plane['T'] = projected + offset
                
                aligned_prims.append(aligned_plane)
                alignment_info['connected'] += 1
                alignment_info['details'].append({
                    'part': part_name,
                    'type': 'connected',
                    'scene_idx': best_match['idx']
                })
                
            else:
                # Keep as independent
                aligned_prims.append(contact_plane)
                alignment_info['independent'] += 1
                alignment_info['details'].append({
                    'part': part_name,
                    'type': 'independent'
                })
    
    print(f"\nAlignment results: {alignment_info['merged']} merged, "
          f"{alignment_info['connected']} connected, {alignment_info['independent']} independent")
    
    return aligned_prims, alignment_info




def fit_contact_planes_per_part(
    all_part_contacts: Dict[str, np.ndarray],
    max_planes_per_part: int = 2,
    ransac_thresh: float = 0.02,
    min_inliers: int = 50,
    fixed_thickness: float = 0.02,
    device: str = "cuda"
) -> Dict[str, List[Dict]]:
    """
    Fit 1-2 planes to each body part's contact points across all timesteps.
    
    Returns:
        Dict mapping part_name to list of plane primitives
    """

    part_planes = {}
    
    for part_name, points_np in all_part_contacts.items():
        if points_np.shape[0] < min_inliers:
            part_planes[part_name] = []
            continue
        
        points = torch.from_numpy(points_np).to(device=device, dtype=torch.float32)
        
        # Estimate normal hint
        try:
            n_est = estimate_normals_knn(points, k=min(20, max(4, points.shape[0]//50)))
            n_hint = F.normalize(n_est.mean(0), dim=0)
        except:
            n_hint = None
        
        planes = []
        used = torch.zeros(points.shape[0], dtype=torch.bool, device=device)
        
        for plane_idx in range(max_planes_per_part):
            free_idx = (~used).nonzero(as_tuple=False).squeeze(-1)
            if free_idx.numel() < min_inliers:
                break
            
            Q = points[free_idx]
            
            # RANSAC plane fitting
            n, c, inl = robust_plane_ransac(Q, n_hint=n_hint, n_iters=500, thresh=ransac_thresh)
            if inl is None or inl.sum().item() < min_inliers:
                break
            
            inl_full = free_idx[inl]
            
            # Fit oriented bounding box in the plane
            R_bw, centre, half_sz, rect_ratio = fit_one_plane_box(Q[inl], n, c, gid=0)
            
            # Set thin thickness for contact planes
            half_sz = half_sz.clone()
            half_sz[2] = max(fixed_thickness * 0.5, 0.005)
            
            plane_prim = {
                'S': half_sz.cpu().numpy(),
                'R': R_bw.t().contiguous().cpu().numpy(),  # world→body
                'T': centre.cpu().numpy(),
                'normal': n.cpu().numpy(),
                'plane_d': torch.dot(n, c).cpu().item(),
                'part': part_name,
                'n_points': inl.sum().item()
            }
            planes.append(plane_prim)
            
            used[inl_full] = True
            n_hint = n  # Use for next plane
        
        part_planes[part_name] = planes
        print(f"  {part_name}: fitted {len(planes)} contact plane(s)")
    
    return part_planes

import numpy as np
import torch
from typing import Dict, List, Union, Optional

ArrayLike = Union[np.ndarray, torch.Tensor]

def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.asarray(x)

def scene_results_to_numpy(
    scene_results: Dict[str, List[ArrayLike]],
    *,
    S_mode: str = "half",   # "half" = exp(log_half), "raw" = leave as-is, "full" = 2*exp(log_half)
    stack: bool = False,    # if True, stack S/R/T into single arrays; pts stays list (variable lengths)
    copy: bool = True
) -> Dict[str, Union[List[np.ndarray], np.ndarray]]:
    """
    Convert a scene_results dict with keys like ['S_items','R_items','T_items','pts_items']
    from torch.Tensors to NumPy arrays.

    Args
    ----
    scene_results : dict
        { 'S_items': [log_half(3,)], 'R_items': [3x3], 'T_items': [3,], 'pts_items': [Ni x 3], ... }
    S_mode : str
        - "half": returns half-sizes = exp(log_half)              (default; matches your viz)
        - "raw" : returns stored values as-is (log_half)
        - "full": returns full sizes = 2 * exp(log_half)
    stack : bool
        If True, returns:
          S_items -> (N,3), R_items -> (N,3,3), T_items -> (N,3)
        If False, keeps them as lists of arrays.
        pts_items is always kept a list (varying lengths).
    copy : bool
        If True, returns copies; otherwise returns views when possible.

    Returns
    -------
    out : dict
    """
    keys = ["S_items", "R_items", "T_items", "pts_items"]
    out: Dict[str, Union[List[np.ndarray], np.ndarray]] = {}

    # helper to maybe copy
    def maybe_copy(a: np.ndarray) -> np.ndarray:
        return a.copy() if copy else a

    # Convert S_items with the chosen mode
    S_list = []
    for S in scene_results.get("S_items", []):
        S_np = _to_numpy(S).astype(np.float32, copy=False)
        if S_mode == "half":
            S_np = np.exp(S_np)
        elif S_mode == "full":
            S_np = 2.0 * np.exp(S_np)
        # else "raw": keep log-half as-is
        S_list.append(maybe_copy(S_np))
    out["S_items"] = np.stack(S_list, axis=0) if (stack and len(S_list) > 0) else S_list

    # R_items (world→body), T_items (centres)
    R_list = [maybe_copy(_to_numpy(R).astype(np.float32, copy=False))
              for R in scene_results.get("R_items", [])]
    T_list = [maybe_copy(_to_numpy(T).astype(np.float32, copy=False))
              for T in scene_results.get("T_items", [])]

    out["R_items"] = np.stack(R_list, axis=0) if (stack and len(R_list) > 0) else R_list
    out["T_items"] = np.stack(T_list, axis=0) if (stack and len(T_list) > 0) else T_list

    # pts_items are typically variable-length; keep them as a list
    P_list = []
    for P in scene_results.get("pts_items", []):
        P_np = _to_numpy(P).astype(np.float32, copy=False)
        P_list.append(maybe_copy(P_np))
    out["pts_items"] = P_list

    # pass through any extra non-primitive keys (convert tensors to numpy)
    for k, v in scene_results.items():
        if k in keys:
            continue
        if isinstance(v, list):
            out[k] = [maybe_copy(_to_numpy(x)) if isinstance(x, (np.ndarray, torch.Tensor)) else x for x in v]
        elif isinstance(v, (np.ndarray, torch.Tensor)):
            out[k] = maybe_copy(_to_numpy(v))
        else:
            out[k] = v

    return out


def convert_aligned_planes_to_primitives(
    aligned_planes: List[Dict]
) -> Dict[str, List]:
    """
    Convert aligned contact planes back to primitive format.
    """
    S_items = []
    R_items = []
    T_items = []
    pts_items = []
    
    for plane in aligned_planes:
        S_items.append(torch.log(torch.tensor(plane['S'], dtype=torch.float32)))
        R_items.append(torch.tensor(plane['R'], dtype=torch.float32))
        T_items.append(torch.tensor(plane['T'], dtype=torch.float32))
        pts_items.append(torch.zeros((100, 3)))  # Dummy points for compatibility
    
    return {
        'S_items': S_items,
        'R_items': R_items,
        'T_items': T_items,
        'pts_items': pts_items
    }


def process_contact_planes_with_scene_alignment(
    contacted_masks: np.ndarray,
    static_segments: List[Tuple[int, int]],
    per_part: Dict[str, Dict],
    pred_vert: np.ndarray,
    part_ids_list: List,
    scene_results: Dict[str, List],
    body_part_params: Optional[Dict[str, Dict]] = None,
    device: str = 'cuda'
) -> Tuple[Dict[str, List], Dict]:
    """
    Main pipeline: collect all timestep contacts → fit planes per part → align with scene.
    
    Returns:
        merged_results: Scene primitives + aligned contact planes
        processing_info: Statistics about the processing
    """
    # Step 1: Collect ALL timestep contact points per body part
    print("\n=== Step 1: Collecting contact points across all timesteps ===")
    all_part_contacts = collect_all_timestep_contact_points(
        contacted_masks, static_segments, per_part, 
        pred_vert, part_ids_list, body_part_params
    )
    
    # Step 2: Fit planes to each part's aggregated contact points
    print("\n=== Step 2: Fitting planes to aggregated contact points ===")
    contact_planes = fit_contact_planes_per_part(
        all_part_contacts,
        max_planes_per_part=2,
        ransac_thresh=0.02,
        min_inliers=50,
        fixed_thickness=0.02,
        device=device
    )

    scene_results = scene_results_to_numpy(scene_results)
    
    # Step 3: Align contact planes with scene primitives
    print("\n=== Step 3: Aligning contact planes with scene ===")
    aligned_planes, alignment_info = align_contact_planes_to_scene(
        contact_planes,
        scene_results,
        normal_threshold=0.95,
        distance_threshold=0.05,
        overlap_threshold=0.1
    )
    
    # Step 4: Convert to primitive format and merge
    contact_primitives = convert_aligned_planes_to_primitives(aligned_planes)
    
    merged_results = merge_primitives_dicts(scene_results, contact_primitives)
    
    processing_info = {
        'n_contact_planes': len(aligned_planes),
        'alignment_info': alignment_info,
        'parts_processed': list(contact_planes.keys()),
        'total_scene_primitives': len(scene_results['S_items']),
        'total_merged_primitives': len(merged_results['S_items'])
    }
    
    print(f"\n=== Final: {processing_info['total_merged_primitives']} total primitives ===")
    merged_results = scene_results_to_torch(
    merged_results, device=device, S_from="half", expect_stacked=True
    )


    return merged_results, processing_info