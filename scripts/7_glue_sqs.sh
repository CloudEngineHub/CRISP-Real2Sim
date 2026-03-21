#!/usr/bin/env bash
set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
echo $REPO_ROOT
VISER_DIR="$REPO_ROOT/vis_scripts/viser_m"
VIS_SCRIPT="$VISER_DIR/vis.sh"
SCRIPT3="$VISER_DIR/run_nksr.sh"

DATA_ROOT="$REPO_ROOT/data"

usage() {
  cat <<'EOF'
Usage: sh 7_glue_sqs.sh <split_or_path> [hmr_type]

Examples:
  sh 7_glue_sqs.sh rebuttal gv
  sh 7_glue_sqs.sh ../data/rebuttal_img

The script will invoke vis_scripts/viser_m/vis.sh with save mode enabled for
each sequence directory inside the provided *_img split.
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

SPLIT_INPUT="${1%/}"
HMR_TYPE="${2:-gv}"
LOG_DIR="${LOG_DIR:-/tmp/vis_megasam_logs}"
mkdir -p "$LOG_DIR"

RUN_NKSR_RAW="${RUN_NKSR:-off}"
case "${RUN_NKSR_RAW,,}" in
  on|true|1|yes|y) RUN_NKSR=1 ;;
  off|false|0|no|n) RUN_NKSR=0 ;;
  *)
    echo "Invalid RUN_NKSR='$RUN_NKSR_RAW' (use on/off or true/false)" >&2
    exit 2
    ;;
esac

declare -a CANDIDATES=(
  "$SPLIT_INPUT"
  "${SPLIT_INPUT}_img"
  "$REPO_ROOT/$SPLIT_INPUT"
  "$REPO_ROOT/${SPLIT_INPUT}_img"
  "$DATA_ROOT/$SPLIT_INPUT"
  "$DATA_ROOT/${SPLIT_INPUT}_img"
)

DATA_PATH=""
for candidate in "${CANDIDATES[@]}"; do
  [[ -z "$candidate" ]] && continue
  if [[ -d "$candidate" ]]; then
    DATA_PATH="$(cd "$candidate" && pwd)"
    break
  fi
done

if [[ -z "$DATA_PATH" ]]; then
  echo "Could not locate data directory for '$SPLIT_INPUT'." >&2
  exit 1
fi

if [[ ! -x "$VIS_SCRIPT" ]]; then
  echo "vis.sh not found at $VIS_SCRIPT" >&2
  exit 1
fi

if (( RUN_NKSR == 1 )) && [[ ! -f "$SCRIPT3" ]]; then
  echo "run_nksr.sh not found at $SCRIPT3" >&2
  exit 1
fi

backfill_sqs_params_inline() {
  local scene_root="$1"
  local eps1="${BACKFILL_EPS1:-0.1}"
  local eps2="${BACKFILL_EPS2:-0.1}"

  python - "$scene_root" "$eps1" "$eps2" <<'PY'
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def load_obj_vertices(path: Path) -> np.ndarray:
    verts: list[list[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith("v "):
                _, x, y, z, *_ = line.split()
                verts.append([float(x), float(y), float(z)])
    if not verts:
        raise ValueError(f"No vertices found in {path}")
    return np.asarray(verts, dtype=np.float32)


def normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-8:
        return fallback.astype(np.float32)
    return (vec / norm).astype(np.float32)


def estimate_piece_axis(vertices: np.ndarray) -> np.ndarray:
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(centered.shape[0], 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmin(eigvals))].astype(np.float32)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if float(np.dot(axis, world_up)) < 0.0:
        axis = -axis
    return normalize(axis, world_up)


def build_tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(ref, normal))) > 0.95:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    tangent_u = normalize(np.cross(normal, ref), np.array([1.0, 0.0, 0.0], dtype=np.float32))
    tangent_v = normalize(np.cross(normal, tangent_u), np.array([0.0, 1.0, 0.0], dtype=np.float32))
    return tangent_u, tangent_v


def parse_piece_order_from_urdf(urdf_path: Path, pieces_dir: Path) -> list[Path]:
    root = ET.parse(urdf_path).getroot()
    refs: list[str] = []
    seen: set[str] = set()
    for mesh_elem in root.findall(".//mesh"):
        filename = mesh_elem.attrib.get("filename")
        if not filename or filename in seen:
            continue
        refs.append(filename)
        seen.add(filename)
    if refs:
        return [pieces_dir / ref for ref in refs]
    return sorted(pieces_dir.glob("part_*.obj"))


def clean_stale_pieces(pieces_dir: Path, keep_paths: list[Path]) -> list[str]:
    keep_names = {path.name for path in keep_paths}
    removed: list[str] = []
    for path in sorted(pieces_dir.glob("part_*.obj")):
        if path.name not in keep_names:
            path.unlink()
            removed.append(path.name)
    return removed


def main() -> None:
    scene_root = Path(sys.argv[1]).resolve()
    eps1 = float(sys.argv[2])
    eps2 = float(sys.argv[3])
    sqs_root = scene_root / "scene_mesh_sqs"
    pieces_dir = sqs_root / "pieces"
    urdf_path = sqs_root / "scene_mesh_sqs.urdf"
    if not pieces_dir.exists():
        raise FileNotFoundError(f"Missing pieces dir: {pieces_dir}")
    if not urdf_path.exists():
        raise FileNotFoundError(f"Missing scene URDF: {urdf_path}")

    piece_paths = parse_piece_order_from_urdf(urdf_path, pieces_dir)
    missing = [str(path) for path in piece_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"URDF references missing piece files: {missing}")

    removed = clean_stale_pieces(pieces_dir, piece_paths)

    params = []
    rot_mats = []
    piece_names = []
    for path in piece_paths:
        verts = load_obj_vertices(path)
        axis_z = estimate_piece_axis(verts)
        axis_x, axis_y = build_tangent_basis(axis_z)

        basis_rows = np.stack([axis_x, axis_y, axis_z], axis=0).astype(np.float32)
        local = verts @ basis_rows.T
        lo = local.min(axis=0)
        hi = local.max(axis=0)
        center_local = 0.5 * (lo + hi)
        half_axes = np.maximum(0.5 * (hi - lo), 1.0e-3).astype(np.float32)
        center_world = (center_local @ basis_rows).astype(np.float32)

        rot_world = np.stack([axis_x, axis_y, axis_z], axis=1).astype(np.float32)
        euler = Rotation.from_matrix(rot_world).as_euler("ZYX").astype(np.float32)
        params.append(
            [
                eps1,
                eps2,
                float(half_axes[0]),
                float(half_axes[1]),
                float(half_axes[2]),
                float(euler[0]),
                float(euler[1]),
                float(euler[2]),
                float(center_world[0]),
                float(center_world[1]),
                float(center_world[2]),
            ]
        )
        rot_mats.append(rot_world)
        piece_names.append(path.name)

    params_np = np.asarray(params, dtype=np.float32)
    rot_np = np.stack(rot_mats, axis=0).astype(np.float32) if rot_mats else np.zeros((0, 3, 3), dtype=np.float32)

    np.save(sqs_root / "sqs_params.npy", params_np)
    np.savez_compressed(
        sqs_root / "sqs_params.npz",
        params=params_np,
        piece_name_utf8=np.asarray(piece_names, dtype=f"<U{max((len(name) for name in piece_names), default=1)}"),
        piece_rot_p2w=rot_np,
    )

    summary = {
        "scene_root": str(scene_root),
        "num_pieces": int(len(piece_paths)),
        "removed_stale_pieces": removed,
        "sqs_params_npy": str((sqs_root / "sqs_params.npy").resolve()),
        "sqs_params_npz": str((sqs_root / "sqs_params.npz").resolve()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
PY
}

pushd "$VISER_DIR" >/dev/null

shopt -s nullglob
seq_dirs=("$DATA_PATH"/*/)
shopt -u nullglob

if (( ${#seq_dirs[@]} == 0 )); then
  echo "No sequence folders found under $DATA_PATH" >&2
  popd >/dev/null
  exit 1
fi

echo "Found ${#seq_dirs[@]} sequences in $DATA_PATH. Logs -> $LOG_DIR"

for seq_dir in "${seq_dirs[@]}"; do
  seq_name="$(basename "${seq_dir%/}")"
  results_file="$REPO_ROOT/results/output/scene/${seq_name}_${HMR_TYPE}_sgd_cvd_hr.npz"
  if [[ ! -f "$results_file" ]]; then
    echo "Skipping ${seq_name}: missing results file $results_file" >&2
    continue
  fi

  logfile="${LOG_DIR}/${seq_name}.log"
  echo "[$(date +'%F %T')] Running scripts for ${seq_name} (log: $logfile)"

  {
    scene_mesh_dir="$REPO_ROOT/results/output/scene/${seq_name}/${HMR_TYPE}/scene_mesh_sqs"
    if [[ -d "$scene_mesh_dir" && -f "$scene_mesh_dir/scene_mesh_sqs.obj" && -f "$scene_mesh_dir/scene_mesh_sqs.urdf" ]]; then
      echo "===== $(date +'%F %T') backfill_sqs_params_inline ====="
      backfill_sqs_params_inline "$REPO_ROOT/results/output/scene/${seq_name}/${HMR_TYPE}"
    else
      echo "===== $(date +'%F %T') vis.sh ====="
      HMR_TYPE="$HMR_TYPE" SAVE_MODE=on bash "$VIS_SCRIPT" "$seq_name"
    fi

    if (( RUN_NKSR == 1 )); then
      echo "===== $(date +'%F %T') nksr ====="
      HMR_TYPE="$HMR_TYPE" bash "$SCRIPT3" "$seq_name"
    fi

  } >"$logfile" 2>&1
done

popd >/dev/null

echo "All visualizations completed successfully."
