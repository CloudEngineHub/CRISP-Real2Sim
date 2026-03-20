from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class SkeletonStats:
    dof_body_ids: tuple[int, ...]
    dof_offsets: tuple[int, ...]
    num_bodies: int

    @property
    def num_joints(self) -> int:
        return len(self.dof_offsets) - 1

    @property
    def dof_obs_size(self) -> int:
        return self.num_joints * 6

    @property
    def num_act(self) -> int:
        return self.dof_offsets[-1]


_KNOWN_SKELETONS: dict[str, SkeletonStats] = {
    "amp_humanoid.xml": SkeletonStats(
        dof_body_ids=(1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14),
        dof_offsets=(0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28),
        num_bodies=15,
    ),
    "amp_humanoid_3d.xml": SkeletonStats(
        dof_body_ids=(1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14),
        dof_offsets=(0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28),
        num_bodies=15,
    ),
    "smpl_humanoid.xml": SkeletonStats(
        dof_body_ids=tuple(range(1, 24)),
        dof_offsets=tuple(range(0, 70, 3)),
        num_bodies=24,
    ),
    "smpl_humanoid_pulse.xml": SkeletonStats(
        dof_body_ids=tuple(range(1, 24)),
        dof_offsets=tuple(range(0, 70, 3)),
        num_bodies=24,
    ),
}


def _joint_size(joint: ET.Element) -> int:
    joint_type = joint.attrib.get("type", "hinge")
    if joint_type in {"hinge", "slide"}:
        return 1
    if joint_type == "ball":
        return 3
    if joint_type == "free":
        return 6
    raise ValueError(f"Unsupported joint type in MJCF: {joint_type}")


def _stats_from_mjcf(asset_path: Path) -> SkeletonStats:
    tree = ET.parse(asset_path)
    root = tree.getroot()

    bodies: list[ET.Element] = []
    for body in root.iter("body"):
        if "name" in body.attrib:
            bodies.append(body)

    dof_body_ids: list[int] = []
    dof_offsets: list[int] = [0]

    for body_id, body in enumerate(bodies):
        joint_size = sum(_joint_size(joint) for joint in body.findall("joint"))
        if joint_size <= 0:
            continue
        dof_body_ids.append(body_id)
        dof_offsets.append(dof_offsets[-1] + joint_size)

    if len(bodies) == 0 or len(dof_offsets) == 1:
        raise ValueError(f"Failed to extract skeleton stats from MJCF: {asset_path}")

    return SkeletonStats(
        dof_body_ids=tuple(dof_body_ids),
        dof_offsets=tuple(dof_offsets),
        num_bodies=len(bodies),
    )


def _resolve_asset_path(asset_file: str) -> Path | None:
    asset_path = Path(asset_file)
    if asset_path.is_absolute() and asset_path.exists():
        return asset_path

    assets_root = Path(__file__).resolve().parent
    candidates = [
        assets_root / asset_file,
        assets_root / os.path.basename(asset_file),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _get_stats(asset_file: str) -> SkeletonStats:
    asset_path = _resolve_asset_path(asset_file)
    if asset_path is not None:
        try:
            return _stats_from_mjcf(asset_path)
        except Exception:
            pass

    asset_name = os.path.basename(asset_file)
    if asset_name in _KNOWN_SKELETONS:
        return _KNOWN_SKELETONS[asset_name]

    raise KeyError(
        f"Unsupported asset file {asset_file!r}. "
        f"Add its MJCF under MotionTracking/motion_tracking/data/assets or extend _KNOWN_SKELETONS."
    )


def _num_obs_from_stats(
    stats: SkeletonStats,
    num_key_bodies: int,
    use_max_coords_obs: bool,
) -> int:
    if use_max_coords_obs:
        return (
            1
            + (stats.num_bodies - 1) * 3
            + stats.num_bodies * 6
            + stats.num_bodies * 3
            + stats.num_bodies * 3
        )

    return 1 + 6 + 3 + 3 + stats.dof_obs_size + stats.num_act + num_key_bodies * 3


def isaacgym_asset_file_to_stats(
    asset_file: str,
    num_key_bodies: int,
    use_max_coords_obs: bool,
):
    stats = _get_stats(asset_file)
    num_obs = _num_obs_from_stats(stats, num_key_bodies, use_max_coords_obs)
    return (
        list(stats.dof_body_ids),
        list(stats.dof_offsets),
        stats.dof_obs_size,
        num_obs,
        stats.num_act,
    )


def get_obs_and_act_sizes(config):
    num_key_bodies = len(getattr(config, "key_bodies", []))
    _, _, _, num_obs, num_act = isaacgym_asset_file_to_stats(
        config.asset.asset_file_name,
        num_key_bodies,
        bool(getattr(config, "use_max_coords_obs", False)),
    )
    return num_obs, num_act


def get_num_jd_obs(config):
    num_key_bodies = len(getattr(config, "key_bodies", []))
    _, _, _, num_disc_obs, _ = isaacgym_asset_file_to_stats(
        config.asset.asset_file_name,
        num_key_bodies,
        False,
    )
    return num_disc_obs
