"""
Single-frame planar primitive extraction.

This minimal script lifts the segmentation / primitive fitting logic out
of `visualizer_megasam.py` so you can experiment on a single RGB frame
without the rest of the pipeline.

Usage
-----
    python -m single_frame_demo.run_single_frame_planes \\
        --image /path/to/frame.jpg \\
        --output_dir ./single_frame_planes
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from toy_exp.simple_vis import SimpleVis
from utils import (
    build_global_segments_single_view,
    compute_segment_properties,
    process_global_segments,
    segment_single_frame_normals,
)


def depth_to_points(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Project a depth map back to 3D using intrinsics K."""
    H, W = depth.shape
    ys, xs = torch.meshgrid(
        torch.arange(H, device=depth.device),
        torch.arange(W, device=depth.device),
        indexing="ij",
    )
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = depth
    x = (xs - cx) / fx * z
    y = (ys - cy) / fy * z
    pts = torch.stack([x, y, z], dim=-1)
    return pts


def serialize_primitives(results: Dict[str, List[torch.Tensor]]) -> List[Dict]:
    entries = []
    for idx, (S, R, T) in enumerate(
        zip(results["S_items"], results["R_items"], results["T_items"])
    ):
        entries.append(
            {
                "id": idx,
                "half_sizes": torch.exp(S).cpu().tolist(),
                "rotation_world_to_body": R.cpu().tolist(),
                "center_world": T.cpu().tolist(),
            }
        )
    return entries


def main(image: Path, output_dir: Path, device: str = "cuda") -> None:
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    vis = SimpleVis(device=dev)

    _, _, _, _, (depth, normals, _), _, cam_intr = vis(image)

    depth_hw = depth[0, 0]
    normals_hw3 = normals[0].permute(1, 2, 0).contiguous()
    K = cam_intr[0]

    seg_map, _ = segment_single_frame_normals(
        normals_hw3,
        depth_hw,
        eps_spatial=2.0,
        n_normal_clusters=8,
        min_samples=10,
        min_points=22,
        device=dev,
    )

    pointclouds = depth_to_points(depth_hw, K)
    seg_props = compute_segment_properties(
        seg_map,
        normals_hw3,
        depth_hw,
        pointclouds,
        device=dev,
    )

    global_segments = build_global_segments_single_view([seg_props])
    results = process_global_segments(
        global_segments,
        [seg_props],
        min_frames=1,
        device=dev,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    entries = serialize_primitives(results)
    (output_dir / "planes.json").write_text(json.dumps(entries, indent=2))
    print(f"Wrote {len(entries)} planar primitives to {output_dir/'planes.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-frame planar fitting demo")
    parser.add_argument("--image", type=Path, required=True, help="Path to RGB frame")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./single_frame_planes"),
        help="Directory for serialized planar primitives",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device to use (falls back to CPU if unavailable)",
    )
    args = parser.parse_args()
    main(args.image, args.output_dir, device=args.device)
