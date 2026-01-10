#!/usr/bin/env python3
"""
Bridge post_scene SMPL outputs into Holosoma-ready SMPLH data.

This script reads the *_ours.npz files produced under results/output/post_scene,
reconstructs SMPL joints, maps them to the SMPLH joint ordering used by Holosoma,
and writes InterMimic-style .pt files so the existing retargeter can be run
without touching Holosoma code. A matching height_dict.pkl is also generated
for scale estimation. __release/prep/UFM
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import smplx
import torch
from smplx import joint_names as smplx_joint_names
import torch
import copy
from typing import Dict, Any

import torch


def process_gv_smpl(
    tgt_name: str,
    world_cam_R: torch.Tensor,
    world_cam_T: torch.Tensor,
    max_frames: int,
    smpl_model=None,  # legacy.
    use_world: bool = True,
    device: str = "cuda",
    # ---- 新增的可选项（你也可以删掉这些参数，写死也行）----
    human_up_axis: int = 1,      # 常见是Y-up=1；如果你是Z-up改成2
    human_n_joints: int = 22,    # 常用只取body 22个；想要全关节就设为None
    human_data_on_cpu: bool = True,  # True: 方便torch.save / numpy；False: 保持device
) -> Dict[str, Any]:
    """
    Process SMPL data from GV (GVHMR) format.
    同时返回 smplx_out 与 human_data:
      human_data["global_joint_positions"]
      human_data["height"]
    """
    from smpl import BodyModelSMPLX, BodyModelSMPLH

    # --------- helper: slice & human_data ----------
    def _slice_smpl_params(params: Dict[str, torch.Tensor], T: int) -> Dict[str, torch.Tensor]:
        """只切那些看起来是按帧(batch=T)存的张量 (ndim>=2 and shape[0] >= T)"""
        out = {}
        for k, v in params.items():
            if torch.is_tensor(v) and v.ndim >= 2 and v.shape[0] >= T:
                out[k] = v[:T]
            else:
                out[k] = v
        return out

    @torch.no_grad()
    def _make_human_data(
        smplx_out,
        smplx_model: torch.nn.Module,
        up_axis: int = 1,
        n_joints: int = 22,
        to_cpu: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        返回:
          global_joint_positions: (T, n_joints, 3)   (world或cam，取决于 smplx_out 的坐标系)
          height: scalar (canonical, 由betas决定，零姿态下计算)
        """
        joints = smplx_out.joints  # (T, J, 3)
        if n_joints is not None:
            joints = joints[:, :n_joints, :]

        # --- canonical height（推荐）：用第一帧 betas，在零姿态/零平移下算身高 ---
        betas0 = smplx_out.betas[:1].detach()
        canon_out = smplx_model(betas=betas0)  # 其它不传 => wrapper里会补零
        canon_verts = canon_out.vertices[0]  # (V, 3)
        height = canon_verts[:, up_axis].max() - canon_verts[:, up_axis].min()  # scalar

        # 如果你想要“每帧高度”（会受姿态影响），用下面这段替换上面的 canonical height：
        # verts = smplx_out.vertices  # (T, V, 3)
        # up = verts[..., up_axis]   # (T, V)
        # height = up.max(dim=1).values - up.min(dim=1).values  # (T,)

        if to_cpu:
            return {
                "global_joint_positions": joints.detach().cpu().float(),
                "height": height.detach().cpu().float(),
            }
        else:
            return {
                "global_joint_positions": joints.detach(),
                "height": height.detach(),
            }

    # --------- load pred ----------
    hmr4d_path = HMR_RESULTS_ROOT / tgt_name / "hmr4d_results.pt"
    pred = torch.load(hmr4d_path)

    # 取一个合理的 num_frames：同时受 max_frames / 相机帧数 / pred帧数约束
    pred_params = pred["smpl_params_incam"]
    pred_len = None
    for key in ["transl", "global_orient", "body_pose"]:
        if key in pred_params and torch.is_tensor(pred_params[key]) and pred_params[key].ndim >= 2:
            pred_len = pred_params[key].shape[0]
            break
    if pred_len is None:
        pred_len = len(world_cam_R)

    num_frames = min(max_frames, len(world_cam_R), len(world_cam_T), pred_len)

    world_cam_R = world_cam_R[:num_frames].to(device)
    world_cam_T = world_cam_T[:num_frames].to(device)

    # 切 pred 参数到 num_frames（很重要，否则 world_cam_R 的 batch 对不上）
    pred_params = _slice_smpl_params(pred_params, num_frames)

    # --------- load mapping & models ----------
    smplx2smpl = torch.load(GVHMR_UTILS_DIR / "body_model" / "smplx2smpl_sparse.pt").to(device)

    bm_kwargs = {
        "model_type": "smplx",
        "gender": "neutral",
        "num_pca_comps": 12,
        "flat_hand_mean": False,
    }
    modelggg = BodyModelSMPLX(model_path=str(GVHMR_BODY_MODELS_DIR), **bm_kwargs).to(device)

    # SMPL for faces
    bm_kwargs_smpl = {
        "model_path": str(GVHMR_BODY_MODELS_DIR),
        "model_type": "smpl",
        "gender": "neutral",
        "num_betas": 10,
        "create_body_pose": False,
        "create_betas": False,
        "create_global_orient": False,
        "create_transl": False,
    }
    model_ = BodyModelSMPLH(**bm_kwargs_smpl)
    faces = model_.faces

    # --------- SMPL-X forward in camera space ----------
    smplx_out_cam = modelggg(**to_cuda(pred_params))
    pred_c_verts = convert_smplx_to_smpl(smplx_out_cam.vertices, smplx2smpl)

    pred_shapes = smplx_out_cam.betas[:, :10]
    global_orient_cam = smplx_out_cam.global_orient

    # Process body pose: axis-angle -> rotmat -> pad two joints
    rotvecs = smplx_out_cam.body_pose.view(-1, 21, 3)
    rotmats = axis_angle_to_matrix_batch(rotvecs)
    T = rotmats.shape[0]
    identity_rot = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).repeat(T, 2, 1, 1)
    body_pose = torch.cat([rotmats, identity_rot], dim=1)

    transl_cam = pred_params["transl"].unsqueeze(1).to(device)  # (T,1,3)

    # --------- transform to world if needed ----------
    if use_world:
        # Convert global orient to world space
        global_orient_world = axis_angle_to_matrix(global_orient_cam)           # (T,3,3)
        global_orient_world = torch.matmul(world_cam_R, global_orient_world)   # (T,3,3)
        global_orient_world = matrix_to_axis_angle(global_orient_world)        # (T,3)

        # Transform SMPL vertices and translation to world
        transl_world = torch.einsum("bij, bnj->bni", world_cam_R, transl_cam) + world_cam_T.unsqueeze(1)  # (T,1,3)
        pred_vert = torch.einsum("bij,bnj->bni", world_cam_R, pred_c_verts) + world_cam_T[:, None]        # (T,V,3)

        # Adjust translation to align vertices properly
        pred_smpl_world = copy.deepcopy(pred_params)
        pred_smpl_world["transl"] = transl_world.squeeze(1)       # (T,3)
        pred_smpl_world["global_orient"] = global_orient_world    # (T,3)

        smplx_out_temp = modelggg(**to_cuda(pred_smpl_world))
        pred_verts_temp = convert_smplx_to_smpl(smplx_out_temp.vertices, smplx2smpl)

        transl_offset = pred_vert - pred_verts_temp
        transl_world = transl_world + transl_offset[:, 0:1, :]

        # Verify alignment
        pred_smpl_world["transl"] = transl_world.squeeze(1)
        smplx_out_temp = modelggg(**to_cuda(pred_smpl_world))
        pred_verts_temp = convert_smplx_to_smpl(smplx_out_temp.vertices, smplx2smpl)
        transl_offset = pred_vert - pred_verts_temp
        print(f"Alignment error: {torch.norm(transl_offset, dim=1).mean():.6f}")

        # 最终 world 下的 smplx 输出（非常关键：用这个去做 human_data）
        smplx_out_ret = smplx_out_temp

        transl_world = transl_world.squeeze(1)  # (T,3)
        global_orient_world_ret = axis_angle_to_matrix(global_orient_world).unsqueeze(1)  # 兼容你原来的返回格式
    else:
        pred_vert = pred_c_verts
        transl_world = transl_cam.squeeze(1)
        global_orient_world_ret = global_orient_cam
        smplx_out_ret = smplx_out_cam

    # --------- NEW: build human_data ----------
    human_data = _make_human_data(
        smplx_out=smplx_out_ret,
        smplx_model=modelggg,
        up_axis=human_up_axis,
        n_joints=human_n_joints,
        to_cpu=human_data_on_cpu,
    )

    # --------- original return + NEW keys ----------
    return {
        "num_frames": num_frames,
        "global_orient_world": global_orient_world_ret,
        "transl_world": transl_world,
        "pred_vert": pred_vert,
        "pred_j3dg": None,  # 维持原逻辑不改
        "body_pose": body_pose,
        "pred_shapes": pred_shapes,
        "faces": faces,

        # ✅ 新增：直接返回 SMPL-X 输出 + human_data
        "smplx_out": smplx_out_ret,
        "human_data": human_data,
    }


@torch.no_grad()
def build_human_data_from_smplx_out(smplx_out, up_axis=1, keep_first_n_joints=22):
    """
    smplx_out.joints: (B, J, 3) or (B, F, J, 3)
    smplx_out.vertices: (B, V, 3) or (B, F, V, 3)
    """
    joints = smplx_out.joints
    if keep_first_n_joints is not None:
        joints = joints[..., :keep_first_n_joints, :]  # keeps first N joints

    # Height: best computed from vertices (more reliable than joints)
    verts = smplx_out.vertices
    up = verts[..., up_axis]  # (B,V) or (B,F,V)
    height = up.max(dim=-1).values - up.min(dim=-1).values  # (B,) or (B,F)

    human_data = {
        "global_joint_positions": joints.detach().cpu().float(),
        "height": height.detach().cpu().float(),
    }
    return human_data

bm = BodyModelSMPLX(model_path=MODEL_DIR, **YOUR_KWARGS).to("cuda")
import torch

human_data = torch.load("human_data.pt", map_location="cpu")

human_joints = human_data["global_joint_positions"]
human_height = human_data["height"]   # <-- remove the colon, it’s a syntax error

print(human_joints.shape, human_height.shape)



# Example inputs (replace with your real tensors)
betas = torch.zeros(1, bm.bm.num_betas, device="cuda")
global_orient = torch.zeros(1, 3, device="cuda")
body_pose = torch.zeros(1, 3 * bm.bm.NUM_BODY_JOINTS, device="cuda")
transl = torch.zeros(1, 3, device="cuda")

with torch.no_grad():
    smplx_out = bm(
        betas=betas,
        global_orient=global_orient,
        body_pose=body_pose,
        transl=transl,
        # left_hand_pose=..., right_hand_pose=..., expression=..., etc as needed
    )
# SMPLH joint order expected by Holosoma (copied from holosoma_retargeting/config_types/data_type.py)
SMPLH_DEMO_JOINTS: List[str] = [
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Index1",
    "L_Index2",
    "L_Index3",
    "L_Middle1",
    "L_Middle2",
    "L_Middle3",
    "L_Pinky1",
    "L_Pinky2",
    "L_Pinky3",
    "L_Ring1",
    "L_Ring2",
    "L_Ring3",
    "L_Thumb1",
    "L_Thumb2",
    "L_Thumb3",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Index1",
    "R_Index2",
    "R_Index3",
    "R_Middle1",
    "R_Middle2",
    "R_Middle3",
    "R_Pinky1",
    "R_Pinky2",
    "R_Pinky3",
    "R_Ring1",
    "R_Ring2",
    "R_Ring3",
    "R_Thumb1",
    "R_Thumb2",
    "R_Thumb3",
]

# Special cases where the SMPLH demo name does not directly match smplx joint names
SPECIAL_NAME_MAP = {
    "torso": "spine1",
    "spine": "spine2",
    "chest": "spine3",
    "l_toe": "left_foot",
    "r_toe": "right_foot",
    "l_thorax": "left_collar",
    "r_thorax": "right_collar",
}


def _demo_to_smplx_name(demo_name: str) -> str:
    """Convert SMPLH demo joint name to the smplx joint_names style."""
    key = demo_name.lower()
    key = key.replace("l_", "left_").replace("r_", "right_")
    return SPECIAL_NAME_MAP.get(key, key)


def _load_height_dict(height_path: Path) -> Dict[str, float]:
    if height_path.exists():
        with height_path.open("rb") as f:
            return pickle.load(f)
    return {}


def _save_height_dict(height_path: Path, values: Dict[str, float]) -> None:
    height_path.parent.mkdir(parents=True, exist_ok=True)
    with height_path.open("wb") as f:
        pickle.dump(values, f)


def _remap_joints(
    joints: np.ndarray, smplx_name_to_idx: Dict[str, int]
) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """Remap smplx joint array to SMPLH demo ordering, returning missing entries."""
    out = np.zeros((joints.shape[0], len(SMPLH_DEMO_JOINTS), 3), dtype=np.float32)
    missing: List[Tuple[str, str]] = []

    for i, demo_name in enumerate(SMPLH_DEMO_JOINTS):
        smplx_name = _demo_to_smplx_name(demo_name)
        idx = smplx_name_to_idx.get(smplx_name)
        if idx is not None and idx < joints.shape[1]:
            out[:, i] = joints[:, idx]
            continue

        # Fallback to a parent joint if the fine joint is unavailable.
        if demo_name.startswith("L_"):
            parent_name = "left_wrist"
        elif demo_name.startswith("R_"):
            parent_name = "right_wrist"
        else:
            parent_name = "pelvis"
        parent_idx = smplx_name_to_idx.get(parent_name)
        if parent_idx is not None and parent_idx < joints.shape[1]:
            out[:, i] = joints[:, parent_idx]
        missing.append((demo_name, smplx_name))

    return out, missing


def _build_intermimic_tensor(human_joints: np.ndarray, object_pose: np.ndarray) -> torch.Tensor:
    """
    Create a minimal tensor matching load_intermimic_data expectations:
    - human joints stored from column 162 (52 * 3 positions)
    - object pose stored from column 318 in order [x, y, z, qx, qy, qz, qw]
    """
    t, j, _ = human_joints.shape
    payload = torch.zeros((t, 325), dtype=torch.float32)
    payload[:, 162 : 162 + j * 3] = torch.from_numpy(human_joints.reshape(t, -1))
    payload[:, 318:325] = torch.from_numpy(object_pose)
    return payload


def _default_object_pose(num_frames: int) -> np.ndarray:
    """Identity object pose repeated over time in [x, y, z, qx, qy, qz, qw] order."""
    pose = np.zeros((num_frames, 7), dtype=np.float32)
    pose[:, -1] = 1.0  # qw
    return pose


def process_sequence(
    npz_path: Path,
    model: smplx.SMPL,
    output_dir: Path,
    height_map: Dict[str, float],
) -> None:
    data = np.load(npz_path)
    poses = torch.from_numpy(data["poses"]).float()
    transl = torch.from_numpy(data["trans"]).float()
    betas = torch.from_numpy(data["betas"]).float()

    out = model(
        betas=betas,
        transl=transl,
        global_orient=poses[:, :3],
        body_pose=poses[:, 3:],
        return_verts=False,
    )
    smplx_name_to_idx = {name: idx for idx, name in enumerate(smplx_joint_names.SMPLH_JOINT_NAMES)}
    joints_np = out.joints.detach().cpu().numpy().astype(np.float32)
    remapped, missing = _remap_joints(joints_np, smplx_name_to_idx)
    if missing:
        print(f"[WARN] {npz_path.name}: missing joints (filled with parents): {missing}")

    height = float(remapped[..., 2].max() - remapped[..., 2].min())
    subject_id = npz_path.stem.split("_")[0]
    height_map[subject_id] = height

    object_pose = _default_object_pose(remapped.shape[0])
    intermimic_tensor = _build_intermimic_tensor(remapped, object_pose)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(intermimic_tensor, output_dir / f"{npz_path.stem}.pt")

    # Optional debug NPZ for inspection or alternative loaders
    np.savez(
        output_dir / f"{npz_path.stem}_smplh_joints.npz",
        global_joint_positions=remapped,
        height=height,
        mocap_framerate=data.get("mocap_framerate", 30),
    )


def discover_npz_files(input_root: Path, hmr_type: str) -> List[Path]:
    return sorted(input_root.glob(f"*/{hmr_type}/*_ours.npz"))


def parse_args() -> argparse.Namespace:
    repo_root = Path("../..") #  / Path(__file__).resolve().parent
    default_input = repo_root / "results" / "output" / "post_scene"
    default_output = repo_root / "holosoma" / "demo_data" / "ours_omniretarget"
    default_height_dict = repo_root / "holosoma" / "demo_data" / "height_dict.pkl"
    default_model = Path("/home/ANT.AMAZON.COM/zzzihanw/FAR/CRISP-Real2Sim/prep/data/smplx/models/smplx/SMPLX_NEUTRAL.pkl")

    parser = argparse.ArgumentParser(description="Convert post_scene outputs into Holosoma-ready SMPLH PT files.")
    parser.add_argument("--input-root", type=Path, default=default_input, help="Root of post_scene outputs.")
    parser.add_argument("--hmr-type", type=str, default="gv", help="Subfolder name under each sequence.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output,
        help="Where to write Holosoma-compatible files (pt + debug npz).",
    )
    parser.add_argument(
        "--height-dict",
        type=Path,
        default=default_height_dict,
        help="Path to height_dict.pkl used by Holosoma (will be created/updated).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=default_model,
        help="Path to the SMPL model file (SMPL_NEUTRAL.pkl).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root
    out = process_gv_smpl(...)

    smplx_out = out["smplx_out"]
    human_data = out["human_data"]

    human_joints = human_data["global_joint_positions"]
    human_height = human_data["height"]

    height_dict = _load_height_dict(args.height_dict)

    _save_height_dict(args.height_dict, height_dict)
    print(f"[OK] Wrote {len(npz_files)} sequences to {args.output_root}")
    print(f"[OK] Updated height dict at {args.height_dict}")


if __name__ == "__main__":
    main()
