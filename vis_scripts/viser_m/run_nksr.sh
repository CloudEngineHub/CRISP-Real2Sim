#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <sequence_name>" >&2
  exit 1
fi

SEQ="$1"
HMR_TYPE="${HMR_TYPE:-gv}"
NKSR_ENV="${NKSR_ENV:-crisp_nksr}"
NKSR_DETAIL_LEVEL="${NKSR_DETAIL_LEVEL:-0.1}"
NKSR_MISE_ITER="${NKSR_MISE_ITER:-1}"
NKSR_MAX_INPUT_POINTS="${NKSR_MAX_INPUT_POINTS:--1}"
NKSR_CHUNK_SIZE="${NKSR_CHUNK_SIZE:--1}"
NKSR_DEVICE="${NKSR_DEVICE:-cuda:0}"

exec conda run -n "$NKSR_ENV" python run_nksr.py \
  --sequence-name "$SEQ" \
  --hmr-type "$HMR_TYPE" \
  --detail-level "$NKSR_DETAIL_LEVEL" \
  --mise-iter "$NKSR_MISE_ITER" \
  --max-input-points "$NKSR_MAX_INPUT_POINTS" \
  --chunk-size "$NKSR_CHUNK_SIZE" \
  --device "$NKSR_DEVICE"
