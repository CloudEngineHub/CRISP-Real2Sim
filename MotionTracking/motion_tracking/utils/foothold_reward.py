from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch import Tensor


class FootholdRewardHelper:
    def __init__(self, metadata_path: str | Path, device: torch.device):
        self.metadata_path = Path(metadata_path)
        self.device = device

        data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        defaults = data.get("reward_block_defaults", {})
        piece_specs = [spec for spec in data.get("piece_specs", []) if spec.get("walkable", False)]
        if not piece_specs:
            raise ValueError(f"No walkable piece specs found in foothold metadata: {self.metadata_path}")

        self.flat_ratio_xy = float(defaults.get("flat_ratio_xy", 0.92))
        self.flat_ratio_z = float(defaults.get("flat_ratio_z", 0.75))
        self.superellipse_power = float(defaults.get("superellipse_power", 24.0))
        self.edge_drop_power = float(defaults.get("edge_drop_power", 24.0))

        self.centers = torch.tensor([spec["center"] for spec in piece_specs], device=device, dtype=torch.float32)
        self.tangent_u = torch.tensor([spec["tangent_u"] for spec in piece_specs], device=device, dtype=torch.float32)
        self.tangent_v = torch.tensor([spec["tangent_v"] for spec in piece_specs], device=device, dtype=torch.float32)
        self.normals = torch.tensor([spec["normal"] for spec in piece_specs], device=device, dtype=torch.float32)
        self.outer_half_size = torch.tensor([spec["half_size"] for spec in piece_specs], device=device, dtype=torch.float32)
        self.piece_indices = torch.tensor(
            [idx for idx, spec in enumerate(data.get("piece_specs", [])) if spec.get("walkable", False)],
            device=device,
            dtype=torch.long,
        )

        self.core_half_size = self.outer_half_size.clone()
        self.core_half_size[:, 0] *= self.flat_ratio_xy
        self.core_half_size[:, 1] *= self.flat_ratio_xy
        self.core_half_size[:, 2] *= self.flat_ratio_z

        self.edge_u = torch.clamp(self.outer_half_size[:, 0] - self.core_half_size[:, 0], min=0.005)
        self.edge_v = torch.clamp(self.outer_half_size[:, 1] - self.core_half_size[:, 1], min=0.005)
        self.edge_w = torch.clamp(self.outer_half_size[:, 2] - self.core_half_size[:, 2], min=0.002)

    def evaluate_points(self, points: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            points: [num_envs, num_points, 3]
        Returns:
            Dict of per-point scores/masks, shape [num_envs, num_points]
        """
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f"Expected points to have shape [N, K, 3], got {tuple(points.shape)}")

        delta = points.unsqueeze(2) - self.centers.view(1, 1, -1, 3)
        local_u = torch.einsum("enkd,kd->enk", delta, self.tangent_u)
        local_v = torch.einsum("enkd,kd->enk", delta, self.tangent_v)
        local_w = torch.einsum("enkd,kd->enk", delta, self.normals)

        abs_u = local_u.abs()
        abs_v = local_v.abs()
        abs_w = local_w.abs()

        inside_outer = (
            (abs_u <= self.outer_half_size.view(1, 1, -1, 3)[..., 0])
            & (abs_v <= self.outer_half_size.view(1, 1, -1, 3)[..., 1])
            & (abs_w <= self.outer_half_size.view(1, 1, -1, 3)[..., 2])
        )
        inside_core = (
            (abs_u <= self.core_half_size.view(1, 1, -1, 3)[..., 0])
            & (abs_v <= self.core_half_size.view(1, 1, -1, 3)[..., 1])
            & (abs_w <= self.core_half_size.view(1, 1, -1, 3)[..., 2])
        )

        tx = torch.clamp((abs_u - self.core_half_size.view(1, 1, -1, 3)[..., 0]) / self.edge_u.view(1, 1, -1), 0.0, 1.0)
        ty = torch.clamp((abs_v - self.core_half_size.view(1, 1, -1, 3)[..., 1]) / self.edge_v.view(1, 1, -1), 0.0, 1.0)
        tz = torch.clamp((abs_w - self.core_half_size.view(1, 1, -1, 3)[..., 2]) / self.edge_w.view(1, 1, -1), 0.0, 1.0)

        p = max(self.superellipse_power, 1.0)
        t_xy = (torch.pow(tx, p) + torch.pow(ty, p)).pow(1.0 / p)
        heat_xy = 1.0 - torch.pow(torch.clamp(t_xy, 0.0, 1.0), self.edge_drop_power)
        heat_z = 1.0 - torch.pow(tz, self.edge_drop_power)
        support_heat = heat_xy * heat_z * inside_outer.float()

        best_local = support_heat.argmax(dim=-1)
        best_heat = support_heat.gather(-1, best_local.unsqueeze(-1)).squeeze(-1)
        support_mask = inside_outer.any(dim=-1)
        core_mask = inside_core.any(dim=-1)

        best_piece = torch.full_like(best_local, -1)
        valid = support_mask
        best_piece[valid] = self.piece_indices[best_local[valid]]

        edge_penalty = support_mask.float() * (1.0 - best_heat)
        miss_penalty = (~support_mask).float()

        return {
            "support_heat": best_heat,
            "core_mask": core_mask.float(),
            "support_mask": support_mask.float(),
            "edge_penalty": edge_penalty,
            "miss_penalty": miss_penalty,
            "best_piece": best_piece,
        }

    def aggregate_contact_reward(
        self,
        points: Tensor,
        contact_mask: Tensor,
        core_bonus: float,
        edge_penalty_scale: float,
        miss_penalty_scale: float,
    ) -> Dict[str, Tensor]:
        if points.shape[:2] != contact_mask.shape:
            raise ValueError(
                f"Point/contact shape mismatch: points={tuple(points.shape)}, contact_mask={tuple(contact_mask.shape)}"
            )

        terms = self.evaluate_points(points)
        active = contact_mask.float()
        denom = active.sum(dim=-1).clamp(min=1.0)

        core_term = (terms["core_mask"] * active).sum(dim=-1) / denom
        edge_term = (terms["edge_penalty"] * active).sum(dim=-1) / denom
        miss_term = (terms["miss_penalty"] * active).sum(dim=-1) / denom
        contact_frac = active.sum(dim=-1) / max(active.shape[-1], 1)

        foothold_rew = core_bonus * core_term - edge_penalty_scale * edge_term - miss_penalty_scale * miss_term
        no_contact = active.sum(dim=-1) <= 0
        foothold_rew = torch.where(no_contact, torch.zeros_like(foothold_rew), foothold_rew)

        return {
            "foothold_rew": foothold_rew,
            "foothold_core_term": core_term,
            "foothold_edge_term": edge_term,
            "foothold_miss_term": miss_term,
            "foothold_contact_frac": contact_frac,
        }
