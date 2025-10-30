import os, csv
import numpy as np
from typing import Dict, List, Tuple, Optional

PART_ORDER = ["leg", "hand", "gluteus", "back", "thigh"]


def _plot_5x3_panel_png(
    path_png: str,
    per_part: Dict[str, Dict],             
    part_ids_list: List[np.ndarray],
    static_segments: List[Tuple[int,int]],
    body_part_params: Dict[str, Dict],
):
    import matplotlib.pyplot as plt

    nrows, ncols = 5, 3
    # infer T
    any_part = per_part[PART_ORDER[0]]
    T = int(len(any_part["velocity"]))
    x = np.arange(T)

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.6 * nrows), sharex=True)

    # column headers
    axes[0, 0].set_title("Stable contact (highlight ≥ run)")
    axes[0, 1].set_title("Velocity (≤ thr, highlight ≥ run)")
    axes[0, 2].set_title("AND (contact ∧ low-vel) + selections")

    for r, pname in enumerate(PART_ORDER):
        cfg          = body_part_params.get(pname, {})
        vthr         = float(cfg.get("vel_threshold", 0.01))
        part_min_run = int(cfg.get("min_consecutive_frames", 30))

        counts_k  = per_part[pname]["counts"]       # [T]
        part_size = max(1, len(part_ids_list[r]))
        # majority-of-part for panel-1
        has_contact = (counts_k >= 0.5 * part_size) # [T] bool
        contact_long = _enforce_min_run(has_contact, part_min_run)

        vel = per_part[pname]["velocity"]           # [T] float
        low_mask = _low_velocity_mask(vel, vthr, part_min_run)  # run-length enforced
        inter_mask = has_contact & low_mask
        inter_long = _enforce_min_run(inter_mask, part_min_run)

        best_k = per_part[pname]["best_frames"]     # list[Optional[int]]

        # --- col 0: contact
        ax = axes[r, 0]
        ax.plot(x, has_contact.astype(int), drawstyle='steps-post', linewidth=1.2)
        for s, e in _mask_to_runs(contact_long):
            ax.axvspan(s, e-1, alpha=0.25)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, linestyle=':', linewidth=0.6)
        ax.set_ylabel(pname)

        # --- col 1: velocity
        ax = axes[r, 1]
        ax.plot(x, vel, linewidth=1.2)
        ax.axhline(vthr, linestyle='--', linewidth=1.0)
        for s, e in _mask_to_runs(low_mask):
            ax.axvspan(s, e-1, alpha=0.20)
        ax.grid(True, linestyle=':', linewidth=0.6)

        # --- col 2: AND + selections
        ax = axes[r, 2]
        ax.plot(x, inter_mask.astype(int), drawstyle='steps-post', linewidth=1.2)
        for s, e in _mask_to_runs(inter_long):
            ax.axvspan(s, e-1, alpha=0.30)
        for s, e in static_segments:
            ax.axvline(s, linestyle=':', linewidth=0.7)
            ax.axvline(e-1, linestyle=':', linewidth=0.7)
        for bf in best_k:
            if bf is not None:
                ax.axvline(int(bf), linewidth=1.6)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, linestyle=':', linewidth=0.6)

    for c in range(ncols):
        axes[-1, c].set_xlabel("frame")

    plt.tight_layout()
    fig.savefig(path_png, dpi=160)
    plt.close(fig)


def _mask_to_runs(mask: np.ndarray) -> List[Tuple[int,int]]:
    if mask.size == 0: return []
    d = np.diff(mask.astype(np.int8))
    starts = np.where(d == 1)[0] + 1
    ends   = np.where(d == -1)[0] + 1
    if mask[0]:  starts = np.r_[0, starts]
    if mask[-1]: ends   = np.r_[ends, mask.size]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]

def find_contact_and_velocity_segments(per_part: Dict[str, Dict], min_duration: int = 15) -> List[Tuple[int, int]]:
    """
    Find segments where ANY part has both contact AND low velocity (the blue regions in column 3).
    This replaces the static segmentation with segments based on actual contact+velocity windows.
    """
    # Get the intersection masks from all parts
    intersection_masks = []
    for part_name in PART_ORDER:
        if part_name in per_part:
            intersection_masks.append(per_part[part_name]["intersection_mask"])
    
    if not intersection_masks:
        return [(0, len(per_part[PART_ORDER[0]]["intersection_mask"]))]
    
    # Union of all intersection masks - where ANY part has contact AND low velocity
    combined_mask = np.any(np.stack(intersection_masks, axis=0), axis=0)  # [T]
    
    # Find continuous segments
    segments = []
    runs = _mask_to_runs(combined_mask)
    
    for start, end in runs:
        if (end - start) >= min_duration:
            segments.append((start, end))
    
    print(f"Found {len(segments)} contact+velocity segments:")
    for i, (s, e) in enumerate(segments):
        print(f"  Segment {i}: frames {s}-{e} (duration: {e-s})")
    
    return segments

def _enforce_min_run(mask: np.ndarray, min_len: int) -> np.ndarray:
    if min_len <= 1: return mask.copy()
    out = np.zeros_like(mask, dtype=bool)
    for s, e in _mask_to_runs(mask):
        if (e - s) >= min_len:
            out[s:e] = True
    return out

def _low_velocity_mask(vel: np.ndarray, thr: float, min_run: int) -> np.ndarray:
    low = vel <= thr
    return _enforce_min_run(low, min_run)

def _center_velocity(track_xyz: np.ndarray, mode: str = "median") -> np.ndarray:
    if track_xyz.size == 0: return np.zeros((track_xyz.shape[0],), dtype=float)
    ctr = np.median(track_xyz, axis=1) if mode == "median" else track_xyz.mean(axis=1)
    d = np.diff(ctr, axis=0)
    sp = np.linalg.norm(d, axis=1) if d.size else np.zeros((0,), dtype=float)
    out = np.zeros((ctr.shape[0],), dtype=float)
    if sp.size > 0:
        out[0] = sp[0]
        out[1:] = sp
    return out

def _detect_static_frames_from_vertices(pred_vert: Optional[np.ndarray],
                                        vel_threshold: float = 0.01,
                                        acc_threshold: float = 0.005,
                                        window_size: int = 5) -> Tuple[np.ndarray, List[Tuple[int,int]]]:
    if pred_vert is None or pred_vert.size == 0:
        return np.zeros((0,), dtype=bool), []
    T = pred_vert.shape[0]
    center = pred_vert.mean(axis=1)
    vel = np.linalg.norm(np.diff(center, axis=0), axis=1)
    if window_size > 1 and vel.size >= window_size:
        k = window_size; kernel = np.ones(k)/k
        vel = np.convolve(vel, kernel, mode='same')
    acc = np.linalg.norm(np.diff(np.diff(center, axis=0), axis=0), axis=1) if T>=3 else np.array([])
    vflag = np.zeros((T,), dtype=bool)
    if vel.size > 0:
        vflag[0] = vel[0] < vel_threshold
        vflag[1:] = vel < vel_threshold
    aflag = np.ones((T,), dtype=bool)
    if acc.size > 0:
        aflag[0] = True
        aflag[1] = acc[0] < acc_threshold
        aflag[2:] = acc < acc_threshold
    static = vflag & aflag
    segs = []
    s=None
    for i, f in enumerate(static):
        if f and s is None: s=i
        elif (not f) and s is not None: segs.append((s,i)); s=None
    if s is not None: segs.append((s,T))
    return static, segs

def filter_stable_contacts_simple(contact_masks: np.ndarray,
                                  min_consecutive_frames: int = 60,
                                  relax_last_N: int = 0,
                                  relax_min_run_last: int = 0) -> np.ndarray:
    T, V = contact_masks.shape
    out = np.zeros_like(contact_masks, dtype=bool)
    if T == 0 or V == 0: return out
    if min_consecutive_frames > 1 and T >= min_consecutive_frames:
        for i in range(T - min_consecutive_frames + 1):
            window = contact_masks[i:i + min_consecutive_frames]
            stable = np.all(window, axis=0)
            if stable.any():
                out[i:i + min_consecutive_frames] |= stable[None, :]
    else:
        out |= contact_masks
    if relax_last_N > 0 and relax_min_run_last > 0:
        tail_start = max(0, T - relax_last_N)
        tail = contact_masks[tail_start:]
        if tail.size > 0:
            run_len = np.zeros_like(tail, dtype=int)
            run_len[0] = tail[0].astype(int)
            for t in range(1, tail.shape[0]):
                run_len[t] = (run_len[t-1] + 1) * tail[t]
            keep_tail = np.zeros_like(tail, dtype=bool)
            rows, cols = np.where(run_len >= relax_min_run_last)
            for r, c in zip(rows, cols):
                length = run_len[r, c]
                start_r = r - length + 1
                keep_tail[start_r:r+1, c] = True
            out[tail_start:] |= keep_tail
    out &= contact_masks
    return out


# ------------------------------- debug saving -------------------------------

def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def _save_npz_csv_series(path_stub: str, series: Dict[str, np.ndarray]):
    np.savez_compressed(path_stub + ".npz", **series)
    # also CSVs for quick eyeballing
    for k, arr in series.items():
        csv_path = f"{path_stub}__{k}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            if arr.ndim == 1:
                w.writerow([k])
                for v in arr:
                    w.writerow([v])
            else:
                # write header
                w.writerow([f"{k}[{i}]" for i in range(arr.shape[1])])
                for row in arr:
                    w.writerow(list(row))
# ========================= ROBUST VELOCITY HELPERS =========================
import numpy as np

def _kabsch_rotation(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Rotation-only best-fit from A->B (Nx3 each), both already centered.
    Returns 3x3 R with det=+1.
    """
    if A.ndim != 2 or B.ndim != 2 or A.shape != B.shape or A.shape[1] != 3:
        return np.eye(3, dtype=float)
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R

def _hampel_filter_1d(x: np.ndarray, window_size: int = 5, n_sigma: float = 3.0) -> np.ndarray:
    """Hampel outlier filter on 1D array; returns filtered copy."""
    if x.size == 0: return x.copy()
    y = x.copy()
    n = len(x)
    k = int(max(1, window_size))
    for i in range(n):
        lo = max(0, i - k)
        hi = min(n, i + k + 1)
        s = x[lo:hi]
        med = float(np.median(s))
        mad = float(np.median(np.abs(s - med))) + 1e-12
        thr = n_sigma * 1.4826 * mad
        if abs(x[i] - med) > thr:
            y[i] = med
    return y

def _smooth_1d_zero_phase(x: np.ndarray, win: int = 7) -> np.ndarray:
    """Zero-phase moving average (forward + backward) to avoid lag."""
    if x.size == 0 or win <= 1: return x.copy()
    k = np.ones(int(win), dtype=float) / float(win)
    y = np.convolve(x, k, mode='same')
    y = np.convolve(y[::-1], k, mode='same')[::-1]
    return y

def _mad_adaptive_threshold(x: np.ndarray, base: float = 0.01, alpha: float = 0.5) -> float:
    """Adaptive threshold blending base with series MAD."""
    if x.size == 0: return float(base)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)) * 1.4826)
    return float((1 - alpha) * base + alpha * mad)

def _hysteresis_mask_from_series(x: np.ndarray, thr: float, open_len: int, close_len: int) -> np.ndarray:
    """
    Hysteresis run-length gating for x <= thr.
    - Needs 'open_len' consecutive frames to switch ON.
    - Switches OFF if we see 'close_len' consecutive frames > thr.
    """
    T = int(x.shape[0])
    on = False
    out = np.zeros((T,), dtype=bool)
    below_run = 0
    above_run = 0
    for t in range(T):
        if x[t] <= thr:
            below_run += 1
            above_run = 0
        else:
            above_run += 1
            below_run = 0
        if not on and below_run >= max(1, open_len):
            on = True
        if on and above_run >= max(1, close_len):
            on = False
        out[t] = on
    return out

def _robust_part_velocity(track_xyz: np.ndarray, smooth_win: int = 7,
                          hampel_ws: int = 5, hampel_k: float = 3.0) -> np.ndarray:
    """
    De-rotate part by aligning to previous frame via Kabsch (on that part),
    then use de-rotated centroid shift as velocity; Hampel + smoothing.
    track_xyz: [T, Lk, 3]
    """
    T = track_xyz.shape[0]
    if T == 0: return np.zeros((0,), dtype=float)
    centers = np.zeros((T, 3), dtype=float)
    # t=0
    P0 = track_xyz[0]
    c0 = P0.mean(axis=0, keepdims=True)
    P0c = P0 - c0
    centers[0] = c0.squeeze()
    for t in range(1, T):
        Pt = track_xyz[t]
        ct = Pt.mean(axis=0, keepdims=True)
        Ptc = Pt - ct
        Pprev = track_xyz[t-1]
        cp = Pprev.mean(axis=0, keepdims=True)
        Ppc = Pprev - cp
        R = _kabsch_rotation(Ptc, Ppc)
        centers[t] = (R @ ct.T).T.squeeze()
    d = np.diff(centers, axis=0)
    sp = np.linalg.norm(d, axis=1) if d.size else np.zeros((0,), dtype=float)
    v = np.zeros((T,), dtype=float)
    if sp.size > 0:
        v[0] = sp[0]
        v[1:] = sp
    v = _hampel_filter_1d(v, window_size=hampel_ws, n_sigma=hampel_k)
    v = _smooth_1d_zero_phase(v, win=smooth_win)
    return v

def _robust_global_velocity_from_pred_vert(pred_vert: np.ndarray,
                                           part_ids_list: list,
                                           smooth_win: int = 7,
                                           hampel_ws: int = 5, hampel_k: float = 3.0) -> np.ndarray:
    """
    Global robust velocity using torso anchors = back ∪ gluteus (by PART_ORDER).
    pred_vert: [T, V, 3]
    part_ids_list: [leg, hand, gluteus, back, thigh]
    """
    T = pred_vert.shape[0]
    if T == 0: return np.zeros((0,), dtype=float)

    # pick anchors: back + gluteus
    try:
        idx_back     = PART_ORDER.index("back")
        idx_gluteus  = PART_ORDER.index("gluteus")
    except ValueError:
        # fallback: all verts
        anchor_ids = np.arange(pred_vert.shape[1], dtype=int)
    else:
        anchor_ids = np.unique(np.concatenate([part_ids_list[idx_back], part_ids_list[idx_gluteus]]).ravel()).astype(int)

    centers = np.zeros((T, 3), dtype=float)
    # t=0
    A0 = pred_vert[0, anchor_ids]
    c0 = A0.mean(axis=0, keepdims=True)
    centers[0] = c0.squeeze()

    for t in range(1, T):
        At = pred_vert[t, anchor_ids]
        ct = At.mean(axis=0, keepdims=True)
        Atc = At - ct

        Aprev = pred_vert[t-1, anchor_ids]
        cp = Aprev.mean(axis=0, keepdims=True)
        Apc = Aprev - cp

        R = _kabsch_rotation(Atc, Apc)
        centers[t] = (R @ ct.T).T.squeeze()

    d = np.diff(centers, axis=0)
    sp = np.linalg.norm(d, axis=1) if d.size else np.zeros((0,), dtype=float)
    v = np.zeros((T,), dtype=float)
    if sp.size > 0:
        v[0] = sp[0]
        v[1:] = sp
    v = _hampel_filter_1d(v, window_size=hampel_ws, n_sigma=hampel_k)
    v = _smooth_1d_zero_phase(v, win=smooth_win)
    return v
# ======================= END ROBUST VELOCITY HELPERS =======================

def _plot_mask_and_velocity_png(path_png: str, velocity: np.ndarray,
                                has_contact: np.ndarray, low_mask: np.ndarray,
                                inter_mask: np.ndarray):
    try:
        import matplotlib.pyplot as plt
        T = velocity.shape[0]
        x = np.arange(T)
        plt.figure(figsize=(10, 3))
        plt.plot(x, velocity, label="velocity")
        # Shade masks
        for name, m in [("contact", has_contact), ("low-vel", low_mask), ("∧", inter_mask)]:
            if m.any():
                segs = _mask_to_runs(m)
                for s,e in segs:
                    plt.axvspan(s, e-1, alpha=0.15, label=name)
        plt.xlabel("frame"); plt.ylabel("speed")
        plt.title(os.path.basename(path_png))
        plt.tight_layout()
        plt.savefig(path_png, dpi=160)
        plt.close()
    except Exception:
        # plotting optional; ignore if headless/no matplotlib
        pass


def apply_temporal_smoothing_to_vertices(pred_vert: np.ndarray, 
                                       window_size: int = 7,
                                       method: str = "savgol") -> np.ndarray:
    """
    Apply temporal smoothing to vertex positions to reduce HMR jitter.
    
    Args:
        pred_vert: [T, V, 3] vertex positions
        window_size: smoothing window size (odd number recommended)
        method: "savgol", "gaussian", or "median"
    """
    from scipy import ndimage
    from scipy.signal import savgol_filter
    
    T, V, _ = pred_vert.shape
    smoothed = pred_vert.copy()
    
    if method == "savgol" and T >= window_size:
        # Savitzky-Golay filter - preserves features while smoothing
        poly_order = min(3, window_size - 1)
        for v in range(V):
            for dim in range(3):
                smoothed[:, v, dim] = savgol_filter(
                    pred_vert[:, v, dim], 
                    window_length=window_size, 
                    polyorder=poly_order,
                    mode='nearest'
                )
    elif method == "gaussian":
        # Gaussian smoothing
        sigma = window_size / 4.0
        for v in range(V):
            for dim in range(3):
                smoothed[:, v, dim] = ndimage.gaussian_filter1d(
                    pred_vert[:, v, dim], 
                    sigma=sigma, 
                    mode='nearest'
                )
    elif method == "median":
        # Median filter - good for removing outliers
        for v in range(V):
            for dim in range(3):
                smoothed[:, v, dim] = ndimage.median_filter(
                    pred_vert[:, v, dim], 
                    size=window_size, 
                    mode='nearest'
                )
    
    return smoothed


def extract_contact_points_from_single_frame(
    frame_idx: int,
    pred_vert: np.ndarray,                    # [T, V, 3]
    part_ids_list: List[np.ndarray],          # List of 5 arrays with vertex indices
    interact_contact_path: str,               # Path to contact prediction files
    contact_threshold: float = 0.5,           # Threshold for contact confidence
    use_all_parts: bool = True,               # If True, use all parts; if False, use only high-confidence parts
    min_confidence_per_part: float = 0.3,     # Minimum avg confidence to include a part
) -> Tuple[int, np.ndarray, Dict[str, np.ndarray]]:
    """
    Extract contact points from a single specified frame for all body parts.
    
    Args:
        frame_idx: The frame index to extract contacts from
        pred_vert: [T, V, 3] vertex positions
        part_ids_list: List of vertex indices for each body part
        interact_contact_path: Path to contact prediction npz files
        contact_threshold: Threshold for considering a vertex as in contact
        use_all_parts: Whether to use all parts or filter by confidence
        min_confidence_per_part: Minimum average confidence to include a part
    
    Returns:
        frame_idx: The input frame index (for reference)
        all_contact_points: [N, 3] array of all contact points from this frame
        per_part_contacts: Dict mapping part names to their contact points
    """
    T, V, _ = pred_vert.shape
    
    # Validate frame index
    if frame_idx < 0 or frame_idx >= T:
        print(f"Warning: frame_idx {frame_idx} out of bounds [0, {T-1}]. Clamping.")
        frame_idx = np.clip(frame_idx, 0, T-1)
    
    print(f"Extracting contact points from frame {frame_idx}")
    
    # Load contact predictions for this frame
    contact_scores = np.zeros(V, dtype=float)
    contact_file = os.path.join(interact_contact_path, f"{frame_idx:05d}.npz")
    
    if os.path.exists(contact_file):
        data = np.load(contact_file, allow_pickle=True)
        scores = np.squeeze(data["pred_contact_3d_smplh"])
        if scores.ndim == 1 and scores.shape[0] >= V:
            contact_scores = scores[:V]
    else:
        print(f"Warning: Contact file {contact_file} not found. Using zero scores.")
    
    # Extract contact points for each body part
    all_contact_points_list = []
    per_part_contacts = {}
    
    for part_idx, part_ids in enumerate(part_ids_list):
        part_name = PART_ORDER[part_idx] if part_idx < len(PART_ORDER) else f"part_{part_idx}"
        
        # Get contact scores for this part
        part_scores = contact_scores[part_ids]
        
        # Check average confidence for this part
        avg_confidence = part_scores.mean()
        
        if not use_all_parts and avg_confidence < min_confidence_per_part:
            print(f"  {part_name}: Skipping (avg confidence {avg_confidence:.3f} < {min_confidence_per_part})")
            per_part_contacts[part_name] = np.array([])
            continue
        
        # Find vertices in contact
        contact_mask = part_scores > contact_threshold
        num_contacts = contact_mask.sum()
        
        if num_contacts > 0:
            # Get the 3D positions of contacting vertices
            contacting_vertex_ids = part_ids[contact_mask]
            contact_points = pred_vert[frame_idx, contacting_vertex_ids, :]
            
            all_contact_points_list.append(contact_points)
            per_part_contacts[part_name] = contact_points
            
            print(f"  {part_name}: {num_contacts} contact points (avg confidence: {avg_confidence:.3f})")
        else:
            per_part_contacts[part_name] = np.array([])
            print(f"  {part_name}: No contact points above threshold")
    
    # Concatenate all contact points
    if all_contact_points_list:
        all_contact_points = np.concatenate(all_contact_points_list, axis=0)
        print(f"Total: {all_contact_points.shape[0]} contact points from frame {frame_idx}")
    else:
        all_contact_points = np.array([])
        print(f"No contact points found in frame {frame_idx}")
    
    return frame_idx, all_contact_points, per_part_contacts


def extract_contact_points_from_frame_simple(
    frame_idx: int,
    pred_vert: np.ndarray,                    # [T, V, 3]
    part_ids_list: List[np.ndarray],          # List of 5 arrays with vertex indices
    num_points_per_part: int = 50,            # Fixed number of points to sample per part
    use_velocity: bool = False,               # If True, prefer low-velocity vertices
    velocity_window: int = 5,                 # Window for velocity calculation
) -> Tuple[int, np.ndarray]:
    """
    Simplified version: Extract fixed number of points from each body part at specified frame.
    No contact prediction needed - just samples vertices uniformly or based on velocity.
    
    Returns:
        frame_idx: The input frame index
        contact_points: [N, 3] array of sampled points
    """
    T, V, _ = pred_vert.shape
    frame_idx = np.clip(frame_idx, 0, T-1)
    
    all_points = []
    
    for part_idx, part_ids in enumerate(part_ids_list):
        part_name = PART_ORDER[part_idx] if part_idx < len(PART_ORDER) else f"part_{part_idx}"
        
        # Determine how many points to sample
        n_sample = min(num_points_per_part, len(part_ids))
        
        if use_velocity and frame_idx > 0:
            # Sample based on low velocity
            start_frame = max(0, frame_idx - velocity_window)
            part_track = pred_vert[start_frame:frame_idx+1, part_ids, :]
            
            # Compute velocity for each vertex
            velocities = np.zeros(len(part_ids))
            for v_idx in range(len(part_ids)):
                if part_track.shape[0] > 1:
                    vel = np.linalg.norm(part_track[-1, v_idx] - part_track[-2, v_idx])
                    velocities[v_idx] = vel
            
            # Select vertices with lowest velocities
            selected_indices = np.argsort(velocities)[:n_sample]
        else:
            # Random sampling
            selected_indices = np.random.choice(len(part_ids), size=n_sample, replace=False)
        
        selected_vertex_ids = part_ids[selected_indices]
        points = pred_vert[frame_idx, selected_vertex_ids, :]
        all_points.append(points)
        
        print(f"  {part_name}: sampled {len(points)} points")
    
    contact_points = np.concatenate(all_points, axis=0)
    print(f"Total: {contact_points.shape[0]} points from frame {frame_idx}")
    
    return frame_idx, contact_points



def robust_velocity_with_outlier_detection(track_xyz: np.ndarray, 
                                         smooth_win: int = 9,
                                         outlier_threshold: float = 3.0,
                                         min_velocity: float = 0.001) -> np.ndarray:
    """
    Enhanced velocity calculation with outlier detection and smoothing.
    """
    T = track_xyz.shape[0]
    if T == 0: return np.zeros((0,), dtype=float)
    
    # 1. Apply spatial smoothing to each frame (reduce pose jitter)
    smoothed_track = np.zeros_like(track_xyz)
    for t in range(T):
        # Smooth each frame's vertex positions (spatial coherence)
        smoothed_track[t] = track_xyz[t]  # Start with original
    
    # 2. Apply temporal smoothing
    if T >= smooth_win:
        from scipy.signal import savgol_filter
        poly_order = min(3, smooth_win - 1)
        for i in range(track_xyz.shape[1]):
            for dim in range(3):
                smoothed_track[:, i, dim] = savgol_filter(
                    track_xyz[:, i, dim], 
                    window_length=smooth_win,
                    polyorder=poly_order,
                    mode='nearest'
                )
    
    # 3. Compute robust velocity using smoothed positions
    centers = smoothed_track.mean(axis=1)  # [T, 3]
    
    # 4. Calculate velocity with outlier detection
    velocities = []
    for t in range(T):
        if t == 0:
            v = np.linalg.norm(centers[1] - centers[0]) if T > 1 else 0.0
        else:
            v = np.linalg.norm(centers[t] - centers[t-1])
        velocities.append(v)
    
    velocities = np.array(velocities)
    
    # 5. Remove velocity outliers using MAD (Median Absolute Deviation)
    median_vel = np.median(velocities)
    mad = np.median(np.abs(velocities - median_vel))
    threshold = outlier_threshold * 1.4826 * mad  # Convert MAD to std equivalent
    
    # Replace outliers with median
    outlier_mask = np.abs(velocities - median_vel) > threshold
    velocities[outlier_mask] = median_vel
    
    # 6. Apply final smoothing to velocities
    if len(velocities) >= 5:
        from scipy.signal import medfilt
        velocities = medfilt(velocities, kernel_size=5)
    
    # 7. Enforce minimum velocity to avoid numerical issues
    velocities = np.maximum(velocities, min_velocity)
    
    return velocities

def stabilize_contact_scores(contact_scores: np.ndarray, 
                           temporal_smooth: int = 3,
                           spatial_smooth: bool = True) -> np.ndarray:
    """
    Stabilize contact scores to reduce HMR-induced noise.
    
    Args:
        contact_scores: [T, V] contact probability scores
        temporal_smooth: temporal smoothing window
        spatial_smooth: whether to apply spatial smoothing
    """
    from scipy import ndimage
    
    T, V = contact_scores.shape
    stabilized = contact_scores.copy()
    
    # 1. Temporal smoothing - reduce frame-to-frame jitter
    if temporal_smooth > 1:
        for v in range(V):
            stabilized[:, v] = ndimage.uniform_filter1d(
                contact_scores[:, v], 
                size=temporal_smooth, 
                mode='nearest'
            )
    
    # 2. Spatial smoothing - neighboring vertices should have similar contact probabilities
    if spatial_smooth:
        for t in range(T):
            # Simple smoothing - can be enhanced with mesh topology
            stabilized[t] = ndimage.uniform_filter1d(
                stabilized[t], 
                size=3,  # smooth over 3 neighboring vertices
                mode='nearest'
            )
    
    return stabilized

def analyze_contacts_5parts(
    interact_contact_path: str,
    num_frames: int,
    part_ids_list: List[np.ndarray],
    pred_contact_vert_list: List[np.ndarray],
    body_part_params: Dict[str, Dict],
    total_verts: int = 6890,
    pred_vert_global: Optional[np.ndarray] = None,
    min_static_duration: int = 15,
    debug_dir: Optional[str] = None,
    # NEW STABILITY PARAMETERS
    enable_vertex_smoothing: bool = True,
    vertex_smooth_window: int = 7,
    enable_contact_smoothing: bool = True,
    contact_smooth_window: int = 3,
    velocity_outlier_threshold: float = 3.0,
    velocity_smooth_window: int = 9,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int,int]], List[Optional[int]], np.ndarray, Dict[str,Dict]]:
    """
    Enhanced version with HMR stability improvements.
    """
    assert len(part_ids_list) == 5 and len(pred_contact_vert_list) == 5, "expect 5 parts"
    T = int(num_frames)
    
    print(f"Starting STABILIZED analysis for T={T} frames")
    print(f"Vertex smoothing: {enable_vertex_smoothing} (window={vertex_smooth_window})")
    print(f"Contact smoothing: {enable_contact_smoothing} (window={contact_smooth_window})")
    
    # STEP 0: Apply vertex position smoothing if enabled
    smoothed_pred_vert_global = pred_vert_global
    smoothed_pred_contact_vert_list = pred_contact_vert_list
    
    if enable_vertex_smoothing and pred_vert_global is not None:
        print("Applying vertex position smoothing...")
        smoothed_pred_vert_global = apply_temporal_smoothing_to_vertices(
            pred_vert_global, 
            window_size=vertex_smooth_window,
            method="savgol"
        )
        
        # Also smooth the per-part vertex lists
        smoothed_pred_contact_vert_list = []
        for k in range(5):
            smoothed_track = apply_temporal_smoothing_to_vertices(
                pred_contact_vert_list[k],
                window_size=vertex_smooth_window,
                method="savgol"
            )
            smoothed_pred_contact_vert_list.append(smoothed_track)
    
    # Load per-frame scores and create masks
    per_part_masks = []
    per_part_masks_fullspace = []
    
    for k in range(5):
        part_size = len(part_ids_list[k])
        per_part_masks.append(np.zeros((T, part_size), dtype=bool))
        per_part_masks_fullspace.append(np.zeros((T, total_verts), dtype=bool))

    all_contact_scores = np.zeros((T, total_verts), dtype=float)
    
    for t in range(T):
        pth = os.path.join(interact_contact_path, f"{t:05d}.npz")
        if not os.path.exists(pth): 
            continue
        arr = np.load(pth, allow_pickle=True)
        scores = np.squeeze(arr["pred_contact_3d_smplh"])
        print(arr.keys())
        if scores.ndim != 1 or scores.shape[0] < total_verts:
            continue
        all_contact_scores[t] = scores
    
    # STEP 1: Apply contact score smoothing if enabled
    if enable_contact_smoothing:
        print("Applying contact score smoothing...")
        all_contact_scores = stabilize_contact_scores(
            all_contact_scores,
            temporal_smooth=contact_smooth_window,
            spatial_smooth=True
        )
    
    # Convert to per-part masks using smoothed scores
    for t in range(T):
        scores = all_contact_scores[t]
        for k, pname in enumerate(PART_ORDER):
            ids = part_ids_list[k]
            thr = float(body_part_params.get(pname, {}).get("contact_threshold", 0.4))
            
            part_mask = scores[ids] > thr
            per_part_masks[k][t] = part_mask
            
            full_mask = np.zeros((total_verts,), dtype=bool)
            full_mask[ids] = part_mask
            per_part_masks_fullspace[k][t] = full_mask

    # Apply temporal stability (same as before)
    per_part_masks_stable = []
    per_part_masks_stable_fullspace = []
    
    for k, pname in enumerate(PART_ORDER):
        cfg = body_part_params.get(pname, {})
        stable_part = filter_stable_contacts_simple(
            per_part_masks[k],
            min_consecutive_frames=int(cfg.get("min_consecutive_frames", 15)),
            relax_last_N=int(cfg.get("relax_last_N", 0)),
            relax_min_run_last=int(cfg.get("relax_min_run_last", 0)),
        )
        per_part_masks_stable.append(stable_part)
        
        stable_full = np.zeros((T, total_verts), dtype=bool)
        for t in range(T):
            stable_full[t][part_ids_list[k]] = stable_part[t]
        per_part_masks_stable_fullspace.append(stable_full)

    # Global mask
    contacted_masks = np.any(np.stack(per_part_masks_stable_fullspace, axis=0), axis=0)
    has_any_contact = (contacted_masks.sum(axis=1) > 0)

    # STEP 2: Enhanced velocity calculation with stabilization
    print("Computing stabilized velocities...")
    per_vels = []
    for k in range(5):
        stabilized_vel = robust_velocity_with_outlier_detection(
            smoothed_pred_contact_vert_list[k],
            smooth_win=velocity_smooth_window,
            outlier_threshold=velocity_outlier_threshold,
            min_velocity=0.001
        )
        per_vels.append(stabilized_vel)
    # Compute intersection masks FIRST (before segmentation)
    per_counts = [m.sum(axis=1) for m in per_part_masks_stable]
    per_debug = {}
    
    for k, pname in enumerate(PART_ORDER):
        cfg = body_part_params.get(pname, {})
        vel_thr = float(cfg.get("vel_threshold", 0.01))
        part_min_run = int(cfg.get("min_consecutive_frames", 15))

        counts_k = per_counts[k]
        part_size = max(1, len(part_ids_list[k]))
        has_contact_k = (counts_k >= 0.5 * part_size)

        low_mask_k = _low_velocity_mask(per_vels[k], vel_thr, part_min_run)
        inter_mask_k = has_contact_k & low_mask_k

        per_debug[pname] = {
            "velocity": per_vels[k],
            "has_contact_mask": has_contact_k,
            "low_velocity_mask": low_mask_k,
            "intersection_mask": inter_mask_k,
            "counts": counts_k,
        }

    # Find segments based on intersection masks (the blue regions!)
    static_segments = find_contact_and_velocity_segments(per_debug, min_duration=min_static_duration)
    
    # Dummy static_frames for compatibility
    static_frames = np.zeros((T,), dtype=bool)
    for s, e in static_segments:
        static_frames[s:e] = True
    
    # Find best frames within the actual segments
    per_best_frames: List[List[Optional[int]]] = []
    
    for k, pname in enumerate(PART_ORDER):
        inter_mask_k = per_debug[pname]["intersection_mask"]
        has_contact_k = per_debug[pname]["has_contact_mask"]
        
        best_k: List[Optional[int]] = []
        for (s, e) in static_segments:
            idx = np.arange(s, e)
            cand = idx[inter_mask_k[s:e]]
            if cand.size == 0:
                best_k.append(None)
            else:
                v = per_vels[k][cand]
                order = np.lexsort((-cand, v))
                chosen = int(cand[order[0]])
                if not has_contact_k[chosen]:
                    best_k.append(None)
                else:
                    best_k.append(chosen)
        
        per_best_frames.append(best_k)

    # Global best frames
    weights = np.array([float(body_part_params.get(p, {}).get("weight", 1.0)) for p in PART_ORDER], dtype=float).reshape(5,1)
    counts_stack = np.stack(per_counts, axis=0)
    counts_global = (weights * counts_stack).sum(axis=0)
    
    best_frames_global: List[Optional[int]] = []
    for (s, e) in static_segments:
        idx = np.arange(s, e)
        cand = idx[has_any_contact[s:e]]
        if cand.size == 0:
            best_frames_global.append(None)
        else:
            seg_counts = counts_global[cand]
            best_frames_global.append(int(cand[int(np.argmax(seg_counts))]))

    # Package results - DEBUG VERSION
    per_part: Dict[str, Dict] = {}
    for k, pname in enumerate(PART_ORDER):
        print(f"\nProcessing best_vertices for {pname}:")
        print(f"  part_ids_list[{k}] length: {len(part_ids_list[k])}")
        print(f"  per_part_masks_stable[{k}] shape: {per_part_masks_stable[k].shape}")
        
        best_vertices = []
        for seg_idx, frame_idx in enumerate(per_best_frames[k]):
            if frame_idx is None:
                best_vertices.append(None)
                print(f"  Segment {seg_idx}: No frame selected")
            else:
                print(f"  Segment {seg_idx}: frame {frame_idx}")
                # Debug the shapes here
                part_contact_mask = per_part_masks_stable[k][frame_idx]
                print(f"    part_contact_mask shape: {part_contact_mask.shape}")
                print(f"    part_ids_list[{k}] length: {len(part_ids_list[k])}")
                print(f"    part_contact_mask dtype: {part_contact_mask.dtype}")
                
                if part_contact_mask.shape[0] != len(part_ids_list[k]):
                    print(f"    ERROR: Shape mismatch! {part_contact_mask.shape[0]} vs {len(part_ids_list[k])}")
                    best_vertices.append(None)
                else:
                    # Convert to numpy array if needed
                    part_ids_array = np.array(part_ids_list[k]) if not isinstance(part_ids_list[k], np.ndarray) else part_ids_list[k]
                    # This is where the error occurs
                    actual_vertex_indices = part_ids_array[part_contact_mask]
                    print(f"    Found {actual_vertex_indices.shape[0]} contacting vertices")
                    best_vertices.append(actual_vertex_indices)
        
        per_part[pname] = {
            "mask": per_part_masks_stable_fullspace[k],
            "mask_partspace": per_part_masks_stable[k],
            "counts": per_counts[k],
            "velocity": per_vels[k],
            "best_frames": per_best_frames[k],
            "best_vertices": best_vertices,
            "has_contact_mask": per_debug[pname]["has_contact_mask"],
            "low_velocity_mask": per_debug[pname]["low_velocity_mask"],
            "intersection_mask": per_debug[pname]["intersection_mask"],
        }

    # Debug visualization
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        per_part_viz = {}
        for k, pname in enumerate(PART_ORDER):
            per_part_viz[pname] = {
                "mask": per_part_masks_stable_fullspace[k],
                "counts": per_part_masks_stable_fullspace[k].sum(axis=1),
                "velocity": per_vels[k],
                "best_frames": per_best_frames[k],
                "best_vertices": per_part[pname]["best_vertices"],
                "has_contact_mask": per_debug[pname]["has_contact_mask"],
                "low_velocity_mask": per_debug[pname]["low_velocity_mask"],
                "intersection_mask": per_debug[pname]["intersection_mask"],
            }
        
        out_all = os.path.join(debug_dir, "ALL_parts__5x3_contact_vel_overlap.png")
        _plot_5x3_panel_png(out_all, per_part_viz, part_ids_list, static_segments, body_part_params)

    return contacted_masks, static_frames, static_segments, best_frames_global, counts_global, per_part

def _global_center_velocity_from_pred_vert(pred_vert: np.ndarray, smooth: int = 1) -> np.ndarray:
    """pred_vert: [T, V, 3] → length-T speed of geometric center (optionally smoothed)."""
    if pred_vert is None or pred_vert.size == 0:
        return np.zeros((0,), dtype=float)
    ctr = pred_vert.mean(axis=1)                 # [T, 3]
    d   = np.diff(ctr, axis=0)                   # [T-1, 3]
    sp  = np.linalg.norm(d, axis=1) if d.size else np.zeros((0,), dtype=float)
    v   = np.zeros((ctr.shape[0],), dtype=float) # [T]
    if sp.size > 0:
        v[0]   = sp[0]
        v[1:]  = sp
    if smooth > 1 and v.size >= smooth:
        k = np.ones(smooth, dtype=float) / smooth
        v = np.convolve(v, k, mode='same')
    return v


def select_last_window_lowest_velocity_contacts(
    pred_vert: np.ndarray,                    # [T, V, 3]
    part_ids_list: List[np.ndarray],          # List of 5 arrays with vertex indices
    window_size: int = 30,                    # Size of temporal window to consider
    velocity_percentile: float = 10.0,        # Percentile of lowest velocity vertices to select
    min_points_per_part: int = 5,             # Minimum contact points per body part
    smoothing_window: int = 5,                # Velocity smoothing window
) -> Tuple[List[int], np.ndarray]:
    """
    Blindly select contact points from the last temporal window based on lowest velocity.
    
    Returns:
        selected_frames: List of frame indices (one per body part)
        contact_points: [N, 3] array of all selected contact points
    """
    T, V, _ = pred_vert.shape
    
    # Define the last window
    window_start = max(0, T - window_size)
    window_end = T
    window_frames = list(range(window_start, window_end))
    
    print(f"Blind selection: Using last window frames {window_start}-{window_end}")
    
    all_contact_points = []
    selected_frames = []
    
    for part_idx, part_ids in enumerate(part_ids_list):
        part_name = PART_ORDER[part_idx] if part_idx < len(PART_ORDER) else f"part_{part_idx}"
        
        # Extract this part's vertices for the window
        part_verts_window = pred_vert[window_start:window_end, part_ids, :]  # [W, L, 3]
        
        # Compute per-vertex velocities across the window
        vertex_velocities = np.zeros((len(part_ids), len(window_frames)))
        
        for v_idx in range(len(part_ids)):
            vertex_track = part_verts_window[:, v_idx, :]  # [W, 3]
            
            # Compute frame-to-frame velocity for this vertex
            for t in range(1, len(window_frames)):
                vel = np.linalg.norm(vertex_track[t] - vertex_track[t-1])
                vertex_velocities[v_idx, t] = vel
            
            # Handle first frame
            if len(window_frames) > 1:
                vertex_velocities[v_idx, 0] = vertex_velocities[v_idx, 1]
            
            # Apply smoothing
            if smoothing_window > 1 and len(window_frames) >= smoothing_window:
                from scipy.ndimage import uniform_filter1d
                vertex_velocities[v_idx] = uniform_filter1d(
                    vertex_velocities[v_idx], 
                    size=smoothing_window, 
                    mode='nearest'
                )
        
        # Find frame with overall lowest velocity for this part
        mean_velocities_per_frame = vertex_velocities.mean(axis=0)  # [W]
        best_frame_idx = np.argmin(mean_velocities_per_frame)
        best_frame_global = window_start + best_frame_idx
        
        selected_frames.append(best_frame_global)
        
        # At the best frame, select vertices with lowest velocities
        frame_velocities = vertex_velocities[:, best_frame_idx]  # [L]
        
        # Determine number of vertices to select
        num_to_select = max(
            min_points_per_part,
            int(np.ceil(len(part_ids) * velocity_percentile / 100.0))
        )
        num_to_select = min(num_to_select, len(part_ids))
        
        # Select vertices with lowest velocities
        lowest_vel_indices = np.argsort(frame_velocities)[:num_to_select]
        
        # Get the actual vertex IDs and their 3D positions
        selected_vertex_ids = part_ids[lowest_vel_indices]
        contact_points_part = pred_vert[best_frame_global, selected_vertex_ids, :]
        
        all_contact_points.append(contact_points_part)
        
        print(f"  {part_name}: frame {best_frame_global}, "
              f"selected {len(contact_points_part)} vertices with lowest velocity")
    
    # Concatenate all contact points
    if all_contact_points:
        contact_points = np.concatenate(all_contact_points, axis=0)
        print(f"Total blind selection: {contact_points.shape[0]} contact points")
    else:
        contact_points = np.array([])
    
    return selected_frames, contact_points


def hybrid_contact_selection(
    # Existing analyze_contacts_5parts outputs
    contacted_masks: np.ndarray,
    static_segments: List[Tuple[int, int]],
    per_part: Dict[str, Dict],
    best_frames_global: List[Optional[int]],
    counts_global: np.ndarray,
    # Geometry
    pred_vert: np.ndarray,
    part_ids_list: List[np.ndarray],
    # Control parameters
    use_blind_fallback: bool = True,
    blind_window_size: int = 30,
    blind_velocity_percentile: float = 10.0,
    confidence_threshold: float = 0.3,  # Min avg contact confidence to trust analysis
) -> Tuple[List[Optional[int]], np.ndarray]:
    """
    Hybrid approach: Use sophisticated analysis when confident, 
    fall back to blind last-window selection when not.
    """
    
    # Check if we have good contact detection results
    has_good_contacts = False
    
    if static_segments and len(static_segments) > 0:
        # Check if we have reasonable contact detections
        total_contact_frames = 0
        for seg in static_segments:
            total_contact_frames += (seg[1] - seg[0])
        
        # Calculate average contact confidence
        avg_contacts = contacted_masks.sum() / (contacted_masks.shape[0] * contacted_masks.shape[1])
        
        if total_contact_frames > 10 and avg_contacts > confidence_threshold:
            has_good_contacts = True
            print(f"Using sophisticated contact analysis (avg confidence: {avg_contacts:.3f})")
    
    if has_good_contacts:
        # Use your existing sophisticated method
        selected_frames, _, all_points = select_frames_and_collect_contacts(
            contacted_masks=contacted_masks,
            static_segments=static_segments,
            part_ids_list=part_ids_list,
            best_frames_global=best_frames_global,
            counts_global=counts_global,
            per_part=per_part,
            pred_vert=pred_vert,
            policy="counts",
        )
        
        if all_points is not None and all_points.shape[0] > 0:
            return selected_frames, all_points
    
    # Fall back to blind selection
    if use_blind_fallback:
        print("Falling back to blind last-window velocity-based selection")
        selected_frames, contact_points = select_last_window_lowest_velocity_contacts(
            pred_vert=pred_vert,
            part_ids_list=part_ids_list,
            window_size=blind_window_size,
            velocity_percentile=blind_velocity_percentile,
        )
        return selected_frames, contact_points
    
    # No contacts found
    return [], np.array([])

def select_frames_and_collect_contacts(
    *,
    # outputs from analyze_contacts_5parts(...)
    contacted_masks: np.ndarray,                    # [T, V] bool - NOT USED, we use per_part data
    static_segments: List[Tuple[int, int]],
    best_frames_global: List[Optional[int]],
    counts_global: np.ndarray,                      # [T]
    per_part: Dict[str, Dict],                      # Contains per-part best_frames and masks
    # geometry
    pred_vert: np.ndarray,                          # [T, V, 3]
    # policy
    part_ids_list: List, 
    policy: str = "counts",                         # "counts" | "velocity"
    body_part_params: Optional[Dict[str, Dict]] = None,
    enforce_lowvel_run: bool = True,
    vel_threshold_global: Optional[float] = None,
    min_run_global: Optional[int] = None,
) -> Tuple[List[Optional[int]], List[np.ndarray], Optional[np.ndarray]]:
    """
    Extract contact points from each body part's selected frames and concatenate them.
    
    Returns:
      selected_frames: one frame index (or None) per final window (from per-part selections)
      per_frame_pcs:   list of (Ni,3) arrays; each is contact points for that window
      all_points:      concatenated (∑Ni,3) array across all parts and windows
    """
    T, V = pred_vert.shape[:2]
    
    # Collect all contact points from all parts across all their selected frames
    all_contact_points = []
    selected_frames_per_segment = []
    
    print(f"Processing {len(static_segments)} static segments...")
    
    for segment_idx, (s, e) in enumerate(static_segments):
        print(f"\nSegment {segment_idx}: frames {s}-{e}")
        segment_contact_points = []
        segment_selected_frames = []
        
        # For each body part, get its selected frame in this segment
        for part_idx, part_name in enumerate(PART_ORDER):
            part_data = per_part[part_name]
            part_best_frames = part_data["best_frames"]  # List[Optional[int]] - one per segment
            part_mask = part_data["mask"]  # [T, V] - stable contact mask for this part
            part_ids = part_ids_list[part_idx]  # vertex indices for this part
            
            if segment_idx < len(part_best_frames):
                selected_frame = part_best_frames[segment_idx]
                
                if selected_frame is not None:
                    print(f"  {part_name}: selected frame {selected_frame}")
                    
                    # Get contact mask for this part at the selected frame
                    part_contact_mask_full = part_mask[selected_frame]  # [V] - full vertex mask
                    part_contact_mask = part_contact_mask_full[part_ids]  # [len(part_ids)] - mask for this part's vertices
                    
                    # Extract 3D coordinates of contacting vertices for this part
                    contacting_part_vertices = pred_vert[selected_frame][part_ids][part_contact_mask]  # [N_part_contacts, 3]
                    
                    if contacting_part_vertices.shape[0] > 0:
                        segment_contact_points.append(contacting_part_vertices)
                        segment_selected_frames.append(selected_frame)
                        print(f"    Found {contacting_part_vertices.shape[0]} contact points")
                    else:
                        print(f"    No contact points found")
                else:
                    print(f"  {part_name}: no frame selected")
        
        # Concatenate all parts' contact points for this segment
        if segment_contact_points:
            segment_all_points = np.concatenate(segment_contact_points, axis=0)
            all_contact_points.append(segment_all_points)
            # For selected_frames, we can use any of the selected frames (they should be from same segment)
            # or use the global best frame for this segment if available
            if segment_idx < len(best_frames_global) and best_frames_global[segment_idx] is not None:
                selected_frames_per_segment.append(best_frames_global[segment_idx])
            elif segment_selected_frames:
                selected_frames_per_segment.append(segment_selected_frames[0])  # Use first selected frame
            else:
                selected_frames_per_segment.append(None)
            print(f"  Segment total: {segment_all_points.shape[0]} contact points")
        else:
            selected_frames_per_segment.append(None)
            print(f"  Segment total: 0 contact points")
    
    # Concatenate all segments
    if all_contact_points:
        all_points = np.concatenate(all_contact_points, axis=0)
        print(f"\nTotal contact points across all segments: {all_points.shape[0]}")
    else:
        all_points = None
        print(f"\nNo contact points found across any segments")
    
    return selected_frames_per_segment, all_contact_points, all_points
def load_contact_points_for_parts(
    contact_dir,
    part_ids_list,
    pred_vertices_world,
    thresholds=None,
    max_points_per_part=8000,
    max_total_points=32000,
    camera_rotations: Optional[np.ndarray] = None,
    camera_translations: Optional[np.ndarray] = None,
    scales = None,
    apply_scale: bool = True,
):
    """Aggregate per-part contact points from stored prediction files.

    Args:
        contact_dir: Directory containing ``*.npz`` contact prediction files.
        part_ids_list: Iterable of numpy arrays with SMPL vertex indices per body part.
        pred_vertices_world: Array of shape ``[T, V, 3]`` with vertices in world space.
        thresholds: Optional iterable of score thresholds per body part (default 0.5).
        max_points_per_part: Cap for retained points per part to keep visualization light-weight.
        max_total_points: Cap for concatenated contact cloud.

    Returns:
        Tuple ``(per_part_points, all_points)`` where both entries are ``float32`` arrays.
    """
    pred_vertices_world = np.asarray(pred_vertices_world)
    if pred_vertices_world.ndim != 3 or pred_vertices_world.shape[2] != 3:
        raise ValueError("pred_vertices_world must have shape [T, V, 3]")

    num_parts = len(part_ids_list)
    if thresholds is None:
        thresholds = [0.5] * num_parts
    elif len(thresholds) != num_parts:
        raise ValueError("thresholds length does not match number of parts")

    if not os.path.isdir(contact_dir):
        empty = np.empty((0, 3), dtype=np.float32)
        return [empty.copy() for _ in range(num_parts)], empty

    try:
        frame_files = sorted(
            f for f in os.listdir(contact_dir)
            if f.endswith('.npz')
        )
    except FileNotFoundError:
        frame_files = []

    if not frame_files:
        empty = np.empty((0, 3), dtype=np.float32)
        return [empty.copy() for _ in range(num_parts)], empty

    rng = np.random.default_rng(42)

    def _downsample(points, limit):
        if limit is None or limit <= 0 or points.shape[0] <= limit:
            return points
        idx = rng.choice(points.shape[0], limit, replace=False)
        return points[idx]

    per_part_accum = [[] for _ in range(num_parts)]
    total_frames = pred_vertices_world.shape[0]

    rotations_np = None
    translations_np = None
    scales_np = None
    if apply_scale and camera_rotations is not None and camera_translations is not None:
        rotations_np = np.asarray(camera_rotations, dtype=np.float32)
        translations_np = np.asarray(camera_translations, dtype=np.float32)
        if rotations_np.shape[0] != total_frames or translations_np.shape[0] != total_frames:
            raise ValueError("Camera extrinsics must have one entry per frame")
        if scales is None:
            scales_np = np.ones((total_frames,), dtype=np.float32)
        else:
            scales_np = np.asarray(scales, dtype=np.float32).reshape(-1)
            if scales_np.size == 1 and total_frames > 1:
                scales_np = np.full((total_frames,), float(scales_np[0]), dtype=np.float32)
            elif scales_np.size != total_frames:
                raise ValueError("scales must broadcast to the number of frames")
    else:
        rotations_np = None
        translations_np = None
        scales_np = None

    for filename in frame_files:
        stem = filename[:-4]
        if not stem.isdigit():
            continue
        frame_idx = int(stem)
        if frame_idx < 0 or frame_idx >= total_frames:
            continue

        npz_path = os.path.join(contact_dir, filename)
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                if "pred_contact_3d_raw_smplh" not in data:
                    continue
                #print(data.files) 
                
                scores = np.asarray(data["pred_contact_3d_raw_smplh"]).squeeze()
        except OSError:
            continue

        if scores.ndim != 1:
            continue

        for part_idx, (ids_raw, thr) in enumerate(zip(part_ids_list, thresholds)):
            if len(ids_raw) == 0:
                continue
            ids = np.asarray(ids_raw, dtype=np.int64)
            valid = (ids >= 0) & (ids < scores.shape[0])
            if not valid.all():
                ids = ids[valid]
            if ids.size == 0:
                continue

            mask = scores[ids] >= float(thr)
            print(scores[ids], 'thr!!!', not np.any(mask))
            if not np.any(mask):
                continue

            points = pred_vertices_world[frame_idx, ids[mask], :]
            if points.size == 0:
                continue

            points = np.asarray(points, dtype=np.float32)
            if rotations_np is not None:
                rot = rotations_np[frame_idx]
                trans = translations_np[frame_idx]
                scale_val = float(scales_np[frame_idx]) if scales_np is not None else 1.0
                pts_cam = (rot.T @ (points - trans).T).T
                pts_cam *= scale_val
                points = (rot @ pts_cam.T).T + trans
            per_part_accum[part_idx].append(_downsample(points, max_points_per_part))

    per_part_points = []
    for part_list in per_part_accum:
        if part_list:
            per_part_points.append(np.concatenate(part_list, axis=0).astype(np.float32, copy=False))
        else:
            per_part_points.append(np.empty((0, 3), dtype=np.float32))

    non_empty = [arr for arr in per_part_points if arr.size]
    if non_empty:
        all_points = np.concatenate(non_empty, axis=0).astype(np.float32, copy=False)
        all_points = _downsample(all_points, max_total_points)
    else:
        all_points = np.empty((0, 3), dtype=np.float32)

    return per_part_points, all_points

