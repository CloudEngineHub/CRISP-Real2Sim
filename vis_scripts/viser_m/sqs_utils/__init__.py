from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import time
import yaml

from numpy.random import seed as np_seed
from numpy.random import get_state as np_get_state
from numpy.random import set_state as np_set_state
from random import seed as rand_seed
from random import getstate as rand_get_state
from random import setstate as rand_set_state
import torch
from torch import manual_seed as torch_seed
from torch import get_rng_state as torch_get_state
from torch import set_rng_state as torch_set_state

import numpy as np
from pathlib import Path

def _to_numpy(a):
    try:
        import torch
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(a)

def _fix_shape(arr):
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"array must be 2D, got shape {arr.shape}")
    if arr.shape[1] == 3:
        return arr
    if arr.shape[0] == 3:
        return arr.T
    raise ValueError(f"array must be (N,3) or (3,N), got {arr.shape}")

def _prep_colors(colors, N):
    if colors is None:
        return None
    colors = _to_numpy(colors)
    colors = _fix_shape(colors)

    # 若是单个颜色，广播到 N
    if colors.shape[0] == 1:
        colors = np.repeat(colors, N, axis=0)

    if colors.dtype.kind in "fc":  # float -> 假设 0..1，转换到 0..255
        colors = np.clip(colors, 0.0, 1.0) * 255.0
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    return colors

def _prep_normals(normals, N):
    if normals is None:
        return None
    normals = _to_numpy(normals)
    normals = _fix_shape(normals).astype(np.float32)
    if normals.shape[0] != N:
        raise ValueError(f"normals length {normals.shape[0]} != points length {N}")
    # 归一化（避免除零）
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    mask = norm > 1e-12
    normals[mask] = normals[mask] / norm[mask]
    return normals

def save_point_cloud_ply(path, points, colors=None, normals=None, ascii=False):
    """
    保存点云到 PLY 文件（默认二进制 little-endian）。
    points: (N,3) 或 (3,N)，numpy / torch；将保存为 float32
    colors: (N,3)，uint8 或 float[0..1]；将保存为 uint8
    normals: (N,3)，float；将保存为 float32（可选）
    ascii: True 则保存为 ASCII；默认 False（二进制）
    """
    points = _to_numpy(points)
    points = _fix_shape(points).astype(np.float32)
    N = points.shape[0]

    colors = _prep_colors(colors, N) if colors is not None else None
    normals = _prep_normals(normals, N) if normals is not None else None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 组装 PLY header
    props = ["property float x", "property float y", "property float z"]
    if normals is not None:
        props += ["property float nx", "property float ny", "property float nz"]
    if colors is not None:
        props += ["property uchar red", "property uchar green", "property uchar blue"]

    fmt = "ascii 1.0" if ascii else "binary_little_endian 1.0"
    header = ["ply", f"format {fmt}", f"element vertex {N}"] + props + ["end_header"]
    header_bytes = ("\n".join(header) + "\n").encode("ascii")

    with open(path, "wb") as f:
        f.write(header_bytes)
        if ascii:
            # ASCII 写入
            for i in range(N):
                row = [f"{points[i,0]:.6f}", f"{points[i,1]:.6f}", f"{points[i,2]:.6f}"]
                if normals is not None:
                    row += [f"{normals[i,0]:.6f}", f"{normals[i,1]:.6f}", f"{normals[i,2]:.6f}"]
                if colors is not None:
                    r, g, b = colors[i].tolist()
                    row += [str(int(r)), str(int(g)), str(int(b))]
                line = " ".join(row) + "\n"
                f.write(line.encode("ascii"))
        else:
            # 二进制写入（构造结构化 dtype，然后一次性写入）
            dtype_fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
            if normals is not None:
                dtype_fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
            if colors is not None:
                dtype_fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
            ply_dtype = np.dtype(dtype_fields)
            arr = np.empty(N, dtype=ply_dtype)
            arr["x"], arr["y"], arr["z"] = points[:,0], points[:,1], points[:,2]
            if normals is not None:
                arr["nx"], arr["ny"], arr["nz"] = normals[:,0], normals[:,1], normals[:,2]
            if colors is not None:
                arr["red"], arr["green"], arr["blue"] = colors[:,0], colors[:,1], colors[:,2]
            arr.tofile(f)

    return str(path)


def path_exists(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError('{} does not exist'.format(path.absolute()))
    return path


def path_mkdir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_files_from(dir_path, valid_extensions=None, recursive=False, sort=False):
    path = path_exists(dir_path)
    if recursive:
        files = [f.absolute() for f in path.glob('**/*') if f.is_file()]
    else:
        files = [f.absolute() for f in path.glob('*') if f.is_file()]

    if valid_extensions is not None:
        valid_extensions = [valid_extensions] if isinstance(valid_extensions, str) else valid_extensions
        valid_extensions = ['.{}'.format(ext) if not ext.startswith('.') else ext for ext in valid_extensions]
        files = list(filter(lambda f: f.suffix in valid_extensions, files))

    return sorted(files) if sort else files


def load_yaml(path, default_path=None):
    path = path_exists(path)
    with open(path, mode='r') as fp:
        cfg_s = yaml.load(fp, Loader=yaml.FullLoader)

    if default_path is not None:
        default_path = path_exists(default_path)
        with open(default_path, mode='r') as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)
    else:
        # try current dir default
        default_path = path.parent / 'default.yml'
        if default_path.exists():
            with open(default_path, mode='r') as fp:
                cfg = yaml.load(fp, Loader=yaml.FullLoader)
        else:
            cfg = {}

    update_recursive(cfg, cfg_s)
    return cfg


def dump_yaml(cfg, path):
    with open(path, mode='w') as f:
        return yaml.safe_dump(cfg, f)


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.
    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used
    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


@contextmanager
def timer(name, unit='s'):
    start = time.time()
    yield
    delta = time.time() - start
    if unit == 's':
        pass
    elif unit == 'min':
        delta /= 60
    else:
        raise NotImplementedError
    print('{}: {:.2f}{}'.format(name, delta, unit))


class use_seed:
    def __init__(self, seed=None):
        if seed is not None:
            assert isinstance(seed, int) and seed >= 0
        self.seed = seed

    def __enter__(self):
        if self.seed is not None:
            self.rand_state = rand_get_state()
            self.np_state = np_get_state()
            self.torch_state = torch_get_state()
            self.torch_cudnn_deterministic = torch.backends.cudnn.deterministic
            rand_seed(self.seed)
            np_seed(self.seed)
            torch_seed(self.seed)
            torch.backends.cudnn.deterministic = True
        return self

    def __exit__(self, typ, val, _traceback):
        if self.seed is not None:
            rand_set_state(self.rand_state)
            np_set_state(self.np_state)
            torch_set_state(self.torch_state)
            torch.backends.cudnn.deterministic = self.torch_cudnn_deterministic

    def __call__(self, f):
        @wraps(f)
        def wrapper(*args, **kw):
            seed = self.seed if self.seed is not None else kw.pop('seed', None)
            with use_seed(seed):
                return f(*args, **kw)

        return wrapper
