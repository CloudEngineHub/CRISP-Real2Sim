#!/usr/bin/env bash
set -euo pipefail

################################################################################
# 1) Make sure conda’s shell functions are available in this script:
#    Option A: using the "conda shell.bash hook":
eval "$(conda shell.bash hook)"

#    OR Option B: source the conda profile script directly:
# source ~/miniconda3/etc/profile.d/conda.sh
################################################################################

DATA_PATH=$1
hmr_type="$2"
run_demo() {
  local script="$1"
  shift
  echo "Running: $script with arguments: $*"
  sh "$script" "$@" || echo "Error running $script with arguments: $*. Continuing to the next sequence."
}

conda activate som
for folder in "$DATA_PATH"/*/
do
    seq=$(basename "$folder")
    run_demo "./cvd_opt/postprocess.sh" "$DATA_PATH" "$seq" "$hmr_type"
done

conda deactivate

echo "All demos completed successfully."
