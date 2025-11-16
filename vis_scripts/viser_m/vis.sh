
HMR_TYPE='gv'
ROOT_DIR="../../results/output/scene/${1}_${HMR_TYPE}_sgd_cvd_hr.npz"

python visualizer_megasam.py --data "$ROOT_DIR" --hmr_type "${HMR_TYPE}"

# video_777_777
# video_1533_1599
#
# 19_indoor_walk_off_mvs




