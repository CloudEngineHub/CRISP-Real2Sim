
#!/usr/bin/env bash
set -eo pipefail

################################################################################
# 1) Make sure conda’s shell functions are available in this script:
#    Option A: using the "conda shell.bash hook":
eval "$(conda shell.bash hook)"


ROOT_DIR="$1"

sh 6_align.sh "$ROOT_DIR" gv
sh re_glue_sqs.sh "$ROOT_DIR" gv


bash 1_video2imgs.sh "$ROOT_DIR"
bash 2_get_mask.sh "$ROOT_DIR"
bash 3_megasam.sh "$ROOT_DIR" 

bash 4_post_camera.sh "$ROOT_DIR"
bash 5_grav.sh "$ROOT_DIR"
bash 0_ufm.sh "$ROOT_DIR"

bash 6_align.sh "$ROOT_DIR" "$HMR_TYPE" 
bash 7_glue_sqs.sh "$ROOT_DIR" "$HMR_TYPE" 
