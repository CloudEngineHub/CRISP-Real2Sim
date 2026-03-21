#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  cat <<'EOF' >&2
Usage:
  bash scripts/all_gv_contact.sh <SEQ_ROOT> [OBJECT_NAME]

Example:
  bash scripts/all_gv_contact.sh /abs/path/to/data/demo stairs
EOF
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$1"
OBJECT_NAME="${2:-stairs}"
HMR_TYPE="gv"

bash "$SCRIPT_DIR/1_video2imgs.sh" "$ROOT_DIR"
bash "$SCRIPT_DIR/0_interactvlm.sh" "$ROOT_DIR" "$OBJECT_NAME"
bash "$SCRIPT_DIR/2_get_mask.sh" "$ROOT_DIR"
bash "$SCRIPT_DIR/3_megasam.sh" "$ROOT_DIR"
bash "$SCRIPT_DIR/4_post_camera.sh" "$ROOT_DIR"
bash "$SCRIPT_DIR/5_grav.sh" "$ROOT_DIR"
bash "$SCRIPT_DIR/0_ufm.sh" "$ROOT_DIR"
bash "$SCRIPT_DIR/6_align.sh" "$ROOT_DIR" "$HMR_TYPE"
bash "$SCRIPT_DIR/7_glue_sqs.sh" "$ROOT_DIR" "$HMR_TYPE"
