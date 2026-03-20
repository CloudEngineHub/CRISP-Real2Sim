#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
PYTHON_BIN="${1:-python}"

if [[ "$PYTHON_BIN" != */* ]]; then
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        echo "Python executable not found: $PYTHON_BIN" >&2
        exit 1
    fi
    PYTHON_BIN="$(command -v "$PYTHON_BIN")"
fi

run_help_check() {
    local workdir="$1"
    shift
    local script_path="$1"
    shift

    echo "[env-check] $script_path --help"
    (
        cd "$workdir"
        "$PYTHON_BIN" "$script_path" --help >/dev/null
    )
}

echo "[env-check] Importing core CRISP modules"
"$PYTHON_BIN" - <<'PY'
import importlib

modules = [
    "torch",
    "torchvision",
    "torchaudio",
    "xformers",
    "nerfstudio",
    "pytorch3d",
    "viser",
    "sam2",
    "hmr4d",
    "uniflowmatch",
    "uniception",
    "pointops2",
    "droid_backends",
    "geocalib",
]

for module_name in modules:
    importlib.import_module(module_name)

print("core-imports-ok")
PY

run_help_check "$REPO_ROOT/prep/AutoMask" "custom_mask.py"
run_help_check "$REPO_ROOT/prep/MogeSAM" "inference.py"
run_help_check "$REPO_ROOT/prep/MogeSAM" "ufm.py"
run_help_check "$REPO_ROOT/prep/MogeSAM" "post_cam.py"
run_help_check "$REPO_ROOT/prep/HMR" "tools/demo/demo.py"
run_help_check "$REPO_ROOT/vis_scripts/viser_m" "visualizer_megasam.py"

echo "core-env-smoke-ok"
