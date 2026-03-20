#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${1:-$SCRIPT_DIR/../prep/data/smpl}"
DST_DIR="${2:-$SCRIPT_DIR/data/smpl}"

if [[ ! -d "$SRC_DIR" ]]; then
    echo "Missing SMPL source directory: $SRC_DIR" >&2
    exit 1
fi

mkdir -p "$DST_DIR"

link_or_copy() {
    local src="$1"
    local dst="$2"
    if [[ ! -e "$src" ]]; then
        return 0
    fi
    if [[ -L "$dst" || -f "$dst" ]]; then
        rm -f "$dst"
    fi
    ln -s "$src" "$dst"
}

link_or_copy "$SRC_DIR/J_regressor_extra.npy" "$DST_DIR/J_regressor_extra.npy"
link_or_copy "$SRC_DIR/SMPL_MALE.pkl" "$DST_DIR/SMPL_MALE.pkl"
link_or_copy "$SRC_DIR/SMPL_NEUTRAL.pkl" "$DST_DIR/SMPL_NEUTRAL.pkl"

if [[ ! -e "$DST_DIR/SMPL_FEMALE.pkl" ]]; then
    if [[ -e "$DST_DIR/SMPL_NEUTRAL.pkl" ]]; then
        ln -s "$DST_DIR/SMPL_NEUTRAL.pkl" "$DST_DIR/SMPL_FEMALE.pkl"
    elif [[ -e "$DST_DIR/SMPL_MALE.pkl" ]]; then
        ln -s "$DST_DIR/SMPL_MALE.pkl" "$DST_DIR/SMPL_FEMALE.pkl"
    fi
fi

echo "$DST_DIR"
