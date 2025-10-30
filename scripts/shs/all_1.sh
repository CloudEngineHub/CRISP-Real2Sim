set -euo pipefail

conda activate som
################################################################################
# 1) Make sure conda’s shell functions are available in this script:
#    Option A: using the "conda shell.bash hook":
eval "$(conda shell.bash hook)"

#    OR Option B: source the conda profile script directly:
# source ~/miniconda3/etc/profile.d/conda.sh
################################################################################
SKIP_EXISTING=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        -s|--skip-existing) SKIP_EXISTING=1; shift ;;
        --)                 shift; break          ;;
        -*) echo "Unknown option: $1" >&2; exit 1 ;;
        *)  break ;;
    esac
done

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 [-s|--skip-existing] <dataset_prefix>" >&2
    exit 1
fi

DATA_PATH=$1

run_demo() {
  local script="$1"
  shift
  echo "Running: $script with arguments: $*"
  sh "$script" "$@" || echo "Error running $script with arguments: $*. Continuing to the next sequence."
}


for folder in "$DATA_PATH"/*/
do
    seq=$(basename "$folder")
    run_demo "./cvd_opt/postcam.sh" "$DATA_PATH" "$seq"
done

conda deactivate

echo "All demos completed successfully."
