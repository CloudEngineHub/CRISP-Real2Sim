#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a paired CRISP sequence into a holosoma-ready bundle with "
            "motion/, geometry/, terrain_meta/, and pairing/ directories."
        )
    )
    parser.add_argument("--sequence-root", type=Path, required=True, help="Path to a CRISP gv sequence directory.")
    parser.add_argument("--bundle-root", type=Path, required=True, help="Root output directory for the paired bundle.")
    parser.add_argument("--sequence-id", type=str, default=None, help="Optional explicit sequence id.")
    parser.add_argument("--motion-npz", type=Path, default=None, help="Optional explicit motion npz path.")
    parser.add_argument("--merged-obj", type=Path, default=None, help="Optional explicit merged scene obj path.")
    parser.add_argument("--pieces-dir", type=Path, default=None, help="Optional explicit pieces directory.")
    parser.add_argument("--scene-urdf", type=Path, default=None, help="Optional explicit scene urdf path.")
    parser.add_argument("--atlas-resolution", type=int, default=512, help="Atlas resolution passed to the exporter.")
    return parser.parse_args()


def infer_sequence_paths(args: argparse.Namespace) -> dict[str, Path | None]:
    seq_root = args.sequence_root.resolve()
    sequence_id = args.sequence_id or seq_root.parent.name

    motion_npz = args.motion_npz
    if motion_npz is None:
        candidate = seq_root / "hmr" / "human_motion.npz"
        motion_npz = candidate if candidate.exists() else None

    merged_obj = args.merged_obj
    if merged_obj is None:
        candidate = seq_root / "scene_mesh_sqs" / "scene_mesh_sqs.obj"
        if not candidate.exists():
            raise FileNotFoundError(f"Could not infer merged OBJ under {seq_root}")
        merged_obj = candidate

    pieces_dir = args.pieces_dir
    if pieces_dir is None:
        candidate = seq_root / "scene_mesh_sqs" / "pieces"
        if not candidate.exists():
            raise FileNotFoundError(f"Could not infer pieces directory under {seq_root}")
        pieces_dir = candidate

    scene_urdf = args.scene_urdf
    if scene_urdf is None:
        candidate = seq_root / "scene_mesh_sqs" / "scene_mesh_sqs.urdf"
        scene_urdf = candidate if candidate.exists() else None

    world_rotation = seq_root / "world_rotation.npy"
    world_rotation = world_rotation if world_rotation.exists() else None
    world_rotation_txt = seq_root / "world_rotation.txt"
    world_rotation_txt = world_rotation_txt if world_rotation_txt.exists() else None
    sqs_params = seq_root / "scene_mesh_sqs" / "sqs_params.npz"
    if not sqs_params.exists():
        candidate = seq_root / "scene_mesh_sqs" / "sqs_params.npy"
        sqs_params = candidate if candidate.exists() else None

    return {
        "sequence_id": sequence_id,
        "motion_npz": motion_npz.resolve() if motion_npz is not None else None,
        "merged_obj": merged_obj.resolve(),
        "pieces_dir": pieces_dir.resolve(),
        "scene_urdf": scene_urdf.resolve() if scene_urdf is not None else None,
        "world_rotation": world_rotation.resolve() if world_rotation is not None else None,
        "world_rotation_txt": world_rotation_txt.resolve() if world_rotation_txt is not None else None,
        "sqs_params": sqs_params.resolve() if sqs_params is not None else None,
        "sequence_root": seq_root,
    }


def summarize_motion(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    payload = np.load(path, allow_pickle=True)
    summary = {"source_motion_npz": str(path), "keys": {}}
    for key in sorted(payload.files):
        value = payload[key]
        summary["keys"][key] = {
            "shape": list(value.shape) if hasattr(value, "shape") else None,
            "dtype": str(value.dtype) if hasattr(value, "dtype") else type(value).__name__,
        }
    return summary


def write_pairing_csv(path: Path, sequence_id: str, has_motion: bool) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["seq_id", "motion_rel", "geometry_rel", "terrain_meta_rel", "has_motion"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "seq_id": sequence_id,
                "motion_rel": f"motion/{sequence_id}" if has_motion else "",
                "geometry_rel": f"geometry/{sequence_id}",
                "terrain_meta_rel": f"terrain_meta/{sequence_id}",
                "has_motion": int(has_motion),
            }
        )


def main() -> None:
    args = parse_args()
    paths = infer_sequence_paths(args)
    sequence_id = str(paths["sequence_id"])
    bundle_root = args.bundle_root.resolve()

    motion_dir = bundle_root / "motion" / sequence_id
    geometry_dir = bundle_root / "geometry" / sequence_id
    terrain_meta_dir = bundle_root / "terrain_meta" / sequence_id
    pairing_dir = bundle_root / "pairing"
    for directory in (motion_dir, geometry_dir, terrain_meta_dir, pairing_dir):
        directory.mkdir(parents=True, exist_ok=True)

    shutil.copy2(str(paths["merged_obj"]), str(geometry_dir / "scene.obj"))
    if paths["scene_urdf"] is not None:
        shutil.copy2(str(paths["scene_urdf"]), str(geometry_dir / "scene.urdf"))
    if paths["world_rotation"] is not None:
        shutil.copy2(str(paths["world_rotation"]), str(geometry_dir / "world_rotation.npy"))
    if paths["world_rotation_txt"] is not None:
        shutil.copy2(str(paths["world_rotation_txt"]), str(geometry_dir / "world_rotation.txt"))
    if paths["sqs_params"] is not None:
        shutil.copy2(str(paths["sqs_params"]), str(geometry_dir / paths["sqs_params"].name))

    motion_summary = summarize_motion(paths["motion_npz"])
    if paths["motion_npz"] is not None:
        shutil.copy2(str(paths["motion_npz"]), str(motion_dir / "motion.npz"))
        (motion_dir / "motion_manifest.json").write_text(json.dumps(motion_summary, indent=2), encoding="utf-8")

    exporter = Path(__file__).resolve().parent / "generate_piece_reward_heatmap.py"
    cmd = [
        sys.executable,
        str(exporter),
        "--pieces-dir",
        str(paths["pieces_dir"]),
        "--merged-obj",
        str(paths["merged_obj"]),
        "--output-dir",
        str(terrain_meta_dir),
        "--world-rotation-npy",
        str(paths["world_rotation"]) if paths["world_rotation"] is not None else "",
        "--sqs-params",
        str(paths["sqs_params"]) if paths["sqs_params"] is not None else "",
        "--piece-normal-thresh",
        "0.8",
        "--world-up-max-angle-deg",
        "45",
        "--heat-shape",
        "edge_aware",
        "--flat-ratio-xy",
        "0.92",
        "--flat-ratio-z",
        "0.75",
        "--superellipse-power",
        "24",
        "--edge-drop-power",
        "24",
        "--core-ratio-xy",
        "0.78",
        "--core-ratio-z",
        "0.60",
        "--solid-padding-xy",
        "0.01",
        "--solid-padding-z",
        "0.005",
        "--atlas-resolution",
        str(int(args.atlas_resolution)),
        "--cmap",
        "red_gray",
    ]
    if paths["world_rotation"] is None:
        del cmd[cmd.index("--world-rotation-npy"):cmd.index("--world-rotation-npy") + 2]
    if paths["sqs_params"] is None:
        del cmd[cmd.index("--sqs-params"):cmd.index("--sqs-params") + 2]
    subprocess.run(cmd, check=True)

    write_pairing_csv(pairing_dir / "all.csv", sequence_id, paths["motion_npz"] is not None)
    write_pairing_csv(pairing_dir / "train.csv", sequence_id, paths["motion_npz"] is not None)

    bundle_manifest = {
        "schema_version": 1,
        "sequence_id": sequence_id,
        "source_sequence_root": str(paths["sequence_root"]),
        "motion_dir": str(motion_dir) if paths["motion_npz"] is not None else None,
        "geometry_dir": str(geometry_dir),
        "terrain_meta_dir": str(terrain_meta_dir),
        "pairing_dir": str(pairing_dir),
        "world_rotation": str(paths["world_rotation"]) if paths["world_rotation"] is not None else None,
        "sqs_params": str(paths["sqs_params"]) if paths["sqs_params"] is not None else None,
        "motion_summary": motion_summary,
    }
    (bundle_root / "manifest.json").write_text(json.dumps(bundle_manifest, indent=2), encoding="utf-8")
    print(json.dumps(bundle_manifest, indent=2))


if __name__ == "__main__":
    main()
