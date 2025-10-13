#!/usr/bin/env bash
set -euo pipefail

################################################################################
# 1) Make sure conda’s shell functions are available in this script:
#    Option A: using the "conda shell.bash hook":
eval "$(conda shell.bash hook)"

conda activate mega_sam
cd /data3/zihanwa3/_Robotics/_vision/mega-sam
#DATA_PATH='/data3/zihanwa3/_Robotics/_data/door_push'
DATA_PATH="${1}_img"

sh all_2.sh "$DATA_PATH"