#!/usr/bin/env bash

################################################################################
# 1) Make sure conda’s shell functions are available in this script:
#    Option A: using the "conda shell.bash hook":
eval "$(conda shell.bash hook)"

conda activate crisp
cd /data3/zihanwa3/_Robotics/_vision/mega-sam 
  

#DATA_PATH='/data3/zihanwa3/_Robotics/_data/door_push'
DATA_PATH="${1}_img"

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