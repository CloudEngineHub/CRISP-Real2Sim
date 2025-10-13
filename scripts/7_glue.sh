#!/usr/bin/env bash
set -euo pipefail
eval "$(conda shell.bash hook)"

run_demo() {
  local script="$1"
  shift
  echo "Running: $script with arguments: $*"
  
  # If the python command fails, print an error but continue to next iteration
  if ! python "$script" "$@"; then
    echo "Error running $script with arguments: $*. Continuing to the next sequence."
  fi
}

# Usage: ./run.sh <config> <hmr_type>
CONFIG="$1"
HMR_TYPE="$2"
DATA_PATH="${1}_img"

conda activate mega_sam
cd /data3/zihanwa3/_Robotics/_vision/mega-sam

for folder in "$DATA_PATH"/*/
do
    [ -d "$folder" ] || continue  # skip non-directories

    seq="$(basename "$folder")"

    # Create symbolic link in datasets/megasam/$CONFIG/images/$seq → $DATA_PATH/$seq
    LINK_DIR="datasets/megasam/$CONFIG/images"
    SRC_DIR="$folder"
    DEST_LINK="$LINK_DIR/$seq"

    mkdir -p "$LINK_DIR"
    
    # Only create symlink if it doesn't already exist
    if [ ! -e "$DEST_LINK" ]; then
        ln -sfn "$SRC_DIR" "$DEST_LINK"
        echo "Created symlink: $DEST_LINK → $SRC_DIR"
    else
        echo "Symlink already exists: $DEST_LINK"
    fi

    # Run the demo (replace with actual demo script path and args)
    run_demo scripts/run_one_sequence.py --config "$CONFIG" --seq "$seq" --hmr_type "$HMR_TYPE"
done

conda deactivate
echo "All demos completed successfully."
