"""
Minimal replacement for the NeuralPlane-based `Vis` helper.

This module keeps the callable contract used inside the visualizers but
implements everything with lightweight NumPy / Torch primitives so we no
longer need Nerfstudio installed just to test the UI plumbing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple, Union

import cv2
import numpy as np
import torch

TensorTuple = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
PathLike = Union[str, Path]


@dataclass
class SimpleVis:
    """
    Drop-in replacement for `toy_exp.vis_normal.Vis`.

    • Keeps the `__call__(img_paths, optim=None)` signature.
    • Returns tensors with the shapes that `_record3d_customized_megasam`
      expects, but they encode trivial heuristics (flat normals, linear
      depth ramp, RGB straight from the source image).
    • Generates empty plane instances so the down-stream logic gracefully
      skips the NeuralPlane-specific refinement.
    """

    default_fov_deg: float = 60.0
    device: Union[str, torch.device] = "cpu"

    def __post_init__(self) -> None:
        if isinstance(self.device, str):
            self.device = torch.device(
                self.device if torch.cuda.is_available() else "cpu"
            )

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        img_paths: Union[PathLike, Sequence[PathLike]],
        optim: object = None,  # kept for API compatibility
    ) -> Tuple[
        torch.Tensor,  # plane_instances
        torch.Tensor,  # masks_avg_normals
        torch.Tensor,  # masks_areas
        torch.Tensor,  # masks_depths
        TensorTuple,   # (depth, normals, image)
        torch.Tensor,  # plane_colors
        torch.Tensor,  # camera intrinsics
    ]:
        rgb_np = self._load_image(img_paths)
        geo = self._make_geo_tensors(rgb_np)

        H, W = rgb_np.shape[:2]
        plane_instances = torch.zeros(
            (1, 0, H, W), dtype=torch.bool, device=self.device
        )
        masks_avg_normals = torch.zeros((1, 0, 3), dtype=torch.float32, device=self.device)
        masks_areas = torch.zeros((1, 0), dtype=torch.float32, device=self.device)
        masks_depths = torch.zeros((1, 0), dtype=torch.float32, device=self.device)
        plane_colors = torch.zeros((1, 0, 3), dtype=torch.float32, device=self.device)

        cam_intr = torch.from_numpy(self._make_intrinsics(H, W)).to(
            device=self.device, dtype=torch.float32
        )[None, ...]

        return (
            plane_instances,
            masks_avg_normals,
            masks_areas,
            masks_depths,
            geo,
            plane_colors,
            cam_intr,
        )

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ensure_path_list(
        img_paths: Union[PathLike, Sequence[PathLike]]
    ) -> Sequence[PathLike]:
        if isinstance(img_paths, (str, Path)):
            return [img_paths]
        if isinstance(img_paths, Sequence) and img_paths:
            return img_paths
        raise ValueError("img_paths must be a path or a non-empty sequence of paths.")

    def _load_image(self, img_paths: Union[PathLike, Sequence[PathLike]]) -> np.ndarray:
        path_list = self._ensure_path_list(img_paths)
        for candidate in path_list:
            candidate = Path(candidate)
            if candidate.is_file():
                img = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        raise FileNotFoundError(f"Could not read any image from {path_list}")

    def _make_geo_tensors(self, rgb_np: np.ndarray) -> TensorTuple:
        H, W = rgb_np.shape[:2]

        # Depth ramp goes from 0.5 m near plane to 2.5 m far plane.
        depth_vals = np.linspace(0.5, 2.5, H, dtype=np.float32)
        depth = torch.from_numpy(depth_vals[:, None]).repeat(1, W)[None, None, ...]

        normal = torch.zeros((1, 3, H, W), dtype=torch.float32)
        normal[:, 2, :, :] = 1.0  # facing +Z

        image = torch.from_numpy(rgb_np.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0

        depth = depth.to(self.device)
        normal = normal.to(self.device)
        image = image.to(self.device)
        return depth, normal, image

    def _make_intrinsics(self, H: int, W: int) -> np.ndarray:
        f = 0.5 * W / np.tan(np.deg2rad(self.default_fov_deg) / 2.0)
        cx, cy = W * 0.5, H * 0.5
        K = np.array(
            [
                [f, 0.0, cx],
                [0.0, f, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return K
