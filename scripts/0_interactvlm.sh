#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    cat <<'EOF' >&2
Usage:
  bash scripts/0_interactvlm.sh <SEQ_ROOT> [OBJECT_NAME]

Example:
  bash scripts/0_interactvlm.sh /abs/path/to/data/demo/wall-kicking stairs

Notes:
  - Pass the sequence root without the `_img` suffix.
  - Uses the separate Contact-Predictor env `crisp_contact` by default.
  - Outputs are written under `results/init/contacts` and `results/init/contact_vis`.
EOF
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CP_DIR="$REPO_ROOT/prep/Contact-Predictor"
CONTACT_ENV="${CONTACT_ENV:-crisp_contact}"

ROOT_INPUT="$(realpath -m "$1")"
OBJ="${2:-stairs}"

if [[ "$ROOT_INPUT" == *_img ]]; then
    DATA_PATH="$ROOT_INPUT"
else
    DATA_PATH="${ROOT_INPUT%/}_img"
fi

[[ -d "$CP_DIR" ]] || { echo "Contact-Predictor not found: $CP_DIR" >&2; exit 1; }
[[ -d "$DATA_PATH" ]] || { echo "Image folder not found: $DATA_PATH" >&2; exit 1; }

command -v conda >/dev/null 2>&1 || { echo "conda not found on PATH." >&2; exit 1; }
conda run -n "$CONTACT_ENV" python -V >/dev/null 2>&1 || {
    echo "Conda env '$CONTACT_ENV' not found. Run: bash setup_crisp_contact.sh $CONTACT_ENV" >&2
    exit 1
}

GPU_COUNT=$(nvidia-smi -L | wc -l)
GPU_IDS=($(seq 0 $((GPU_COUNT - 1))))

echo "Found $GPU_COUNT GPUs -> ${GPU_IDS[*]}"
echo "Scanning $DATA_PATH"
echo "Using Contact-Predictor env: $CONTACT_ENV"

mapfile -d '' DIRS < <(find "$DATA_PATH" -mindepth 1 -maxdepth 1 -type d -print0)
NUM_DIRS=${#DIRS[@]}
echo "$NUM_DIRS folders to process"

worker() {
    local gpu_id="$1"
    shift
    local folders=("$@")

    eval "$(conda shell.bash hook)"
    conda activate "$CONTACT_ENV"
    cd "$CP_DIR"

    for cam_folder in "${folders[@]}"; do
        local seq
        seq="$(basename "$cam_folder")"
        local parent_dir
        parent_dir="$(dirname "$cam_folder")"

        echo "GPU $gpu_id | $seq"
        CUDA_VISIBLE_DEVICES="$gpu_id" \
            bash ./process.sh \
            "$parent_dir" \
            "$seq" \
            "$OBJ"
    done
}

for gpu_id in "${GPU_IDS[@]}"; do
    gpu_dirs=()
    for (( idx=gpu_id; idx<NUM_DIRS; idx+=GPU_COUNT )); do
        gpu_dirs+=("${DIRS[idx]}")
    done
    worker "$gpu_id" "${gpu_dirs[@]}" &
done

wait
echo "All jobs finished."
