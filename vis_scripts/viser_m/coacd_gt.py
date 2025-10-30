from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import torch
import nksr
import coacd
import shutil
import tyro
# ==== 示例 ====
import vdbfusion
import os 

def load_bg_normals(
    tgt_name: str,
    device: str = "cuda",
    out_dir: str = "cache",
    fmt: str = "pt",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    读取 {out_dir}/{tgt_name}.pt 或 .npz，返回 (bg_positions, normals)，均为 device 上的 float32 Tensor。
    """
    dev = torch.device(device)
    path = Path(out_dir) / f"{tgt_name}.{fmt}"
    if not path.exists():
        raise FileNotFoundError(f"Cache not found: {path}")

    if fmt == "pt":
        data = torch.load(path, map_location="cpu")
        bg_positions = torch.as_tensor(data["bg_positions"], dtype=torch.float32).to(dev)
        normals      = torch.as_tensor(data["normals"],      dtype=torch.float32).to(dev)
    elif fmt == "npz":
        data = np.load(path, allow_pickle=False)
        bg_positions = torch.from_numpy(data["bg_positions"]).to(dev).float()
        normals      = torch.from_numpy(data["normals"]).to(dev).float()
    else:
        raise ValueError("fmt must be 'pt' or 'npz'")
    return bg_positions, normals


def _voxel_downsample(
    P: torch.Tensor,
    N: torch.Tensor,
    voxel_size: float,
    method: str = "first",
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    体素下采样：
      - method='first'  : 每个体素取首个点（最快，推荐）
      - method='random' : 每个体素随机取一个点
      - method='mean'   : 每个体素对位置/法线做平均（法线会再单位化）
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")

    # 量化到体素网格坐标
    q = torch.floor(P / voxel_size).to(torch.int64)

    # 找唯一体素
    uniq, inv, counts = torch.unique(q, dim=0, return_inverse=True, return_counts=True)

    if method == "first":
        order = torch.argsort(inv)              # 将同一体素的点相邻
        inv_sorted = inv[order]
        change = torch.ones_like(inv_sorted, dtype=torch.bool)
        change[1:] = inv_sorted[1:] != inv_sorted[:-1]
        sel = order[change]                     # 每段的第一个
        P_ds = P[sel]
        N_ds = N[sel]

    elif method == "random":
        order = torch.argsort(inv)
        # 体素在有序序列中的起始位置
        starts = torch.zeros_like(counts)
        starts[1:] = torch.cumsum(counts, dim=0)[:-1]
        if seed is not None:
            gen = torch.Generator(device=P.device).manual_seed(seed)
            offs = torch.floor(torch.rand(counts.size(0), generator=gen, device=P.device) * counts.to(P.device)).to(torch.long)
        else:
            offs = torch.floor(torch.rand(counts.size(0), device=P.device) * counts.to(P.device)).to(torch.long)
        sel = order[starts + offs]
        P_ds = P[sel]
        N_ds = N[sel]

    elif method == "mean":
        num = uniq.size(0)
        P_acc = torch.zeros((num, 3), device=P.device, dtype=P.dtype)
        N_acc = torch.zeros((num, 3), device=N.device, dtype=N.dtype)
        for d in range(3):
            P_acc[:, d].index_add_(0, inv, P[:, d])
            N_acc[:, d].index_add_(0, inv, N[:, d])
        cnt = counts.to(P.dtype).unsqueeze(1)
        P_ds = P_acc / cnt
        N_ds = N_acc / cnt
        # 法线单位化
        N_ds = N_ds / torch.clamp(torch.linalg.norm(N_ds, dim=1, keepdim=True), min=1e-12)
    else:
        raise ValueError("method must be one of {'first','random','mean'}")

    return P_ds.contiguous(), N_ds.contiguous()


def prepare_points_normals(
    points: torch.Tensor,
    normals: torch.Tensor,
    device: str = "cuda",
    voxel_size: Optional[float] = None,
    voxel_method: str = "first",
    max_points: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将 (F,H,W,3) 或 (N,3) 的点/法线整理为 (N,3)，过滤非有限值并单位化法线，
    可选：体素下采样 + 最大点数限制。
    """
    dev = torch.device(device)

    P = torch.as_tensor(points,  dtype=torch.float32, device=dev)
    N = torch.as_tensor(normals, dtype=torch.float32, device=dev)

    # 形状整理：支持 (..., 3)
    if P.ndim == 4 and P.shape[-1] == 3:
        P = P.reshape(-1, 3)
    elif P.ndim == 2 and P.shape[1] == 3:
        pass
    else:
        raise ValueError(f"points shape must be (...,3), got {tuple(P.shape)}")

    if N.ndim == 4 and N.shape[-1] == 3:
        N = N.reshape(-1, 3)
    elif N.ndim == 2 and N.shape[1] == 3:
        pass
    else:
        raise ValueError(f"normals shape must be (...,3), got {tuple(N.shape)}")

    if P.shape[0] != N.shape[0]:
        raise ValueError(f"N points != N normals: {P.shape[0]} vs {N.shape[0]}")

    # 过滤非有限值
    ok = torch.isfinite(P).all(dim=1) & torch.isfinite(N).all(dim=1)

    # 去掉零法线并单位化
    nlen = torch.linalg.norm(N, dim=1)
    ok = ok & (nlen > 0)
    P = P[ok]
    N = N[ok]
    nlen = nlen[ok].unsqueeze(1)
    N = N / torch.clamp(nlen, min=1e-12)

    # 体素下采样（可选）
    if voxel_size is not None:
        P, N = _voxel_downsample(P, N, voxel_size=float(voxel_size), method=voxel_method, seed=seed)

    # 限制最大点数（可选）
    if max_points is not None and P.size(0) > max_points:
        if seed is not None:
            g = torch.Generator(device=P.device).manual_seed(seed)
            idx = torch.randperm(P.size(0), device=P.device, generator=g)[:max_points]
        else:
            idx = torch.randperm(P.size(0), device=P.device)[:max_points]
        P = P[idx].contiguous()
        N = N[idx].contiguous()

    return P.contiguous(), N.contiguous()


def baseline_nksr(
    points: torch.Tensor,
    normals: torch.Tensor,
    device: str = "cuda",
    detail_level: float = 1.0,
    mise_iter: int = 1,
):
    dev = torch.device(device)
    pts = torch.as_tensor(points,  dtype=torch.float32, device=dev).contiguous()
    nor = torch.as_tensor(normals, dtype=torch.float32, device=dev).contiguous()

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points shape must be (N,3), got {tuple(pts.shape)}")
    if nor.ndim != 2 or nor.shape[1] != 3 or nor.shape[0] != pts.shape[0]:
        raise ValueError(f"normals shape must be (N,3) and match N, got {tuple(nor.shape)} vs {pts.shape[0]}")

    reconstructor = nksr.Reconstructor(str(dev))
    field = reconstructor.reconstruct(pts, nor, detail_level=detail_level)
    mesh = field.extract_dual_mesh(mise_iter=mise_iter)
    return mesh

from pathlib import Path
import numpy as np
import torch
import trimesh

def _as_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _pick_attr(obj, names):
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            return v() if callable(v) else v
    return None

def to_trimesh(mesh) -> trimesh.Trimesh:
    """
    将 NKSR/常见三角网格对象转换为 trimesh.Trimesh。
    识别属性名：vertices|verts|v, faces|tris|triangles|f, vertex_normals|normals|vn
    """
    V = _pick_attr(mesh, ["vertices", "verts", "v"])
    F = _pick_attr(mesh, ["faces", "tris", "triangles", "f"])
    VN = _pick_attr(mesh, ["vertex_normals", "normals", "vn"])

    # 兼容 Open3D TriangleMesh
    if V is None or F is None:
        try:
            import open3d as o3d
            if isinstance(mesh, o3d.geometry.TriangleMesh):
                V = np.asarray(mesh.vertices)
                F = np.asarray(mesh.triangles)
                VN = np.asarray(mesh.vertex_normals) if len(mesh.vertex_normals) else None
        except Exception:
            pass

    if V is None or F is None:
        raise ValueError("Cannot locate vertices/faces on mesh. "
                         "Tried attributes: vertices|verts|v and faces|tris|triangles|f.")

    V = _as_numpy(V).astype(np.float64, copy=False)  # trimesh 用 float64 更稳
    F = _as_numpy(F).astype(np.int64, copy=False)
    if VN is not None:
        VN = _as_numpy(VN)
        if VN.shape != V.shape:
            VN = None  # 形状不匹配就忽略

    return trimesh.Trimesh(vertices=V, faces=F, vertex_normals=VN, process=False)

def export_trimesh(mesh, out_path: str):
    """
    导出为 .ply/.obj/.glb（根据后缀自动选择）。
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tm = to_trimesh(mesh)
    tm.export(out.as_posix())
    print(f"Saved mesh -> {out}")

def thicken_along_normals(mesh_like,
                          thickness: float = 0.05,
                          inward: bool = True,
                          close_open_boundaries: bool = True) -> trimesh.Trimesh:
    """
    Offset by ±normal and stitch boundary walls. inward=True => reverse along normals.
    """
    m = to_trimesh(mesh_like).copy()
    m.remove_duplicate_faces()
    m.remove_unreferenced_vertices()
    m.fix_normals()

    n_verts = len(m.vertices)
    V_out = m.vertices.copy()
    dir_sign = -1.0 if inward else 1.0
    V_in  = V_out + dir_sign * thickness * m.vertex_normals

    F_out = m.faces.copy()
    F_in  = m.faces[:, ::-1] + n_verts

    verts = np.vstack([V_out, V_in])
    faces = [F_out, F_in]

    if close_open_boundaries:
        # Get boundary edges - different approach
        # Use edges and check which edges appear only once in faces
        edges = m.edges
        edges_sorted = np.sort(edges, axis=1)
        
        # Count occurrences of each edge
        unique_edges, inverse, counts = np.unique(
            edges_sorted, axis=0, return_inverse=True, return_counts=True
        )
        
        # Boundary edges appear only once
        boundary_edges = unique_edges[counts == 1]
        
        if len(boundary_edges) > 0:
            walls = []
            for u, v in boundary_edges:
                u_in, v_in = u + n_verts, v + n_verts
                walls.append([u, v, v_in])
                walls.append([u, v_in, u_in])
            faces.append(np.asarray(walls, dtype=np.int64))

    thick = trimesh.Trimesh(vertices=verts, faces=np.vstack(faces), process=True)
    thick.remove_duplicate_faces()
    thick.remove_degenerate_faces()
    thick.fix_normals()
    return thick

def main(
  tgt_name: str, 
  thick: bool = False,
):
    # input_file = '/data3/zihanwa3/_Robotics/_data/_PROX/scenes/BasementSittingBooth.ply'
    tgt_name = tgt_name.split('_')[0]
    input_base = '/data3/zihanwa3/_Robotics/_data/_PROX/scenes'
    input_file = os.path.join(input_base, f'{tgt_name}.ply')
    

    mesh = trimesh.load(input_file, force="mesh")
    mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    merge = True
    input_file = input_file.replace('scenes', 'scenes_convex')
    os.makedirs(input_file, exist_ok=True)

    result = coacd.run_coacd(mesh,
                                merge=merge,
                                threshold=0.01, # threshold=0.01, 
                                preprocess_resolution=150,
                                max_convex_hull=300, 
                                # preprocess_resolution=80, #  50 by d
                                # max_ch_vertex=256, # max_ch_vertex=512  default = 256
                                resolution=200000, #resolution=200000
                                )


    scene = trimesh.Scene()
    mesh_parts = []

    for vs, fs in result:
        mesh_parts.append(trimesh.Trimesh(vs, fs))
    np.random.seed(0)
    for p in mesh_parts:
        #p.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(np.uint8)
        scene.add_geometry(p)
    if merge: 
        prefix = 'merge'
    else:
        prefix = ''
    nksr_coacd_mesh = input_file.replace('.ply', '.obj')
    print(f'saved to {nksr_coacd_mesh}')
    scene.export(nksr_coacd_mesh)

if __name__ == "__main__":
  tyro.cli(main)

      