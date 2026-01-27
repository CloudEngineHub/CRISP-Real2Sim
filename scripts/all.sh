#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="/home/ubuntu/FAR/CRISP-Real2Sim/data"
SCRIPT_REL="all_gv.sh"   # assumes you run this from the folder that contains all_gv.sh
# If you want an absolute script path instead, set:
# SCRIPT="/home/ubuntu/FAR/CRISP-Real2Sim/scripts/all_gv.sh"

shopt -s nullglob

for videos_dir in "$DATA_ROOT"/_emdb*_videos; do
  [[ -d "$videos_dir" ]] || continue

  base="$(basename "$videos_dir")"  # e.g. vmm_a_videos
  if [[ "$base" != _emdb* ]]; then
    continue
  fi

  base="${videos_dir%_videos}"   # strip trailing "_videos" -> /.../vmm_a
  echo "==> Processing: $videos_dir  ->  bash $SCRIPT_REL $base"
  bash "$SCRIPT_REL" "$base"
done
