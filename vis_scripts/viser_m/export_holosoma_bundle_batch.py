#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-export CRISP sequences into a holosoma-ready paired bundle."
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Root containing per-sequence CRISP output folders.")
    parser.add_argument("--bundle-root", type=Path, required=True, help="Destination bundle root.")
    parser.add_argument("--atlas-resolution", type=int, default=512, help="Atlas resolution for each exported sequence.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip sequence export if the expected terrain_meta files already exist.",
    )
    return parser.parse_args()


def write_pairing_csv(path: Path, sequence_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["seq_id", "motion_rel", "geometry_rel", "terrain_meta_rel", "has_motion"],
        )
        writer.writeheader()
        for sequence_id in sequence_ids:
            writer.writerow(
                {
                    "seq_id": sequence_id,
                    "motion_rel": f"motion/{sequence_id}",
                    "geometry_rel": f"geometry/{sequence_id}",
                    "terrain_meta_rel": f"terrain_meta/{sequence_id}",
                    "has_motion": 1,
                }
            )


def main() -> None:
    args = parse_args()
    exporter = Path(__file__).resolve().parent / "export_holosoma_sequence_bundle.py"
    input_root = args.input_root.resolve()

    sequence_dirs = []
    for candidate in sorted(input_root.iterdir()):
        gv_root = candidate / "gv"
        if (
            (gv_root / "hmr" / "human_motion.npz").exists()
            and (gv_root / "scene_mesh_sqs" / "scene_mesh_sqs.obj").exists()
            and (gv_root / "scene_mesh_sqs" / "pieces").exists()
        ):
            sequence_dirs.append(gv_root)

    if not sequence_dirs:
        raise FileNotFoundError(f"No exportable sequences found under {input_root}")

    print(f"Found {len(sequence_dirs)} exportable sequences.")
    sequence_ids = [seq_root.parent.name for seq_root in sequence_dirs]
    for idx, seq_root in enumerate(sequence_dirs, start=1):
        sequence_id = seq_root.parent.name
        terrain_meta_dir = args.bundle_root.resolve() / "terrain_meta" / sequence_id
        expected_files = [
            terrain_meta_dir / "obj_metadata.npz",
            terrain_meta_dir / "atlas.npz",
            terrain_meta_dir / "metadata.json",
        ]
        if args.skip_existing and all(path.exists() for path in expected_files):
            print(f"[{idx:02d}/{len(sequence_dirs):02d}] Skipping {sequence_id} (already exported)")
            continue

        print(f"[{idx:02d}/{len(sequence_dirs):02d}] Exporting {sequence_id}")
        cmd = [
            sys.executable,
            str(exporter),
            "--sequence-root",
            str(seq_root),
            "--bundle-root",
            str(args.bundle_root),
            "--atlas-resolution",
            str(int(args.atlas_resolution)),
        ]
        subprocess.run(cmd, check=True)

    pairing_dir = args.bundle_root.resolve() / "pairing"
    pairing_dir.mkdir(parents=True, exist_ok=True)
    write_pairing_csv(pairing_dir / "all.csv", sequence_ids)
    write_pairing_csv(pairing_dir / "train.csv", sequence_ids)

    manifest = {
        "schema_version": 1,
        "input_root": str(input_root),
        "bundle_root": str(args.bundle_root.resolve()),
        "num_sequences": len(sequence_ids),
        "sequence_ids": sequence_ids,
    }
    (args.bundle_root.resolve() / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
