# MotionTracking Guide

This folder consumes aligned CRISP outputs and turns them into the directory
layout expected by MotionTracking training and evaluation.

## What MotionTracking Expects

MotionTracking does not train directly from `results/output/scene/...`.
It expects:

- `motion_data/<DATE>/<SEQ>_<METHOD>.npy`
- `motion_tracking/data/assets/urdf/<DATE>/<SEQ>/<METHOD>/<METHOD>.urdf`

That means the source must already be the aligned `post_scene` output:

```text
results/output/post_scene/<SEQ>/<HMR_TYPE>/
├── hmr/human_motion.npz
└── scene_mesh_sqs/
```

If `human_motion.npz` is missing, the CRISP post-scene step is not complete yet
and the sequence cannot be bridged into MotionTracking training.

## 1. Setup The Standalone Environment

From the repository root:

```bash
bash setup_motiontracking_viser_env.sh motiontracking_viser
source ~/miniconda3/etc/profile.d/conda.sh
conda activate motiontracking_viser
```

This environment is independent from the CRISP video environment. It supports:

- Isaac Gym based MotionTracking
- `robot-viser` visualization
- CRISP-to-MotionTracking motion conversion
- automatic runtime preparation of `MotionTracking/data/smpl`

## 2. Bridge A CRISP Sequence Into MotionTracking

Move into this folder:

```bash
cd MotionTracking
```

The fastest entrypoint is:

```bash
bash bridge_crisp_sequence.sh <SEQ_NAME> [DATE_TAG] [METHOD] [HMR_TYPE]
```

Example:

```bash
bash bridge_crisp_sequence.sh 40_indoor_walk_big_circle bridge0318 ours gv
```

This materializes:

```text
motion_data/<DATE>/<SEQ>_<METHOD>.npz
motion_data/<DATE>/<SEQ>_<METHOD>.npy
motion_tracking/data/assets/urdf/<DATE>/<SEQ>/<METHOD>/<METHOD>.urdf
motion_tracking/data/assets/urdf/<DATE>/<SEQ>/<METHOD>/<METHOD>.obj
motion_tracking/data/assets/urdf/<DATE>/<SEQ>/<METHOD>/part_*.obj
```

By default the bridge copies files into MotionTracking. If you want symlinks
instead:

```bash
bash bridge_crisp_sequence.sh <SEQ_NAME> <DATE_TAG> <METHOD> <HMR_TYPE> --materialize symlink
```

## 3. Train

Once the bridge step finishes:

```bash
bash run_bridged_train.sh <DATE_TAG> <SEQ_NAME> <METHOD>
```

Example:

```bash
bash run_bridged_train.sh bridge0318 40_indoor_walk_big_circle ours
```

To inspect the exact Hydra command without starting training:

```bash
PRINT_ONLY=1 bash run_bridged_train.sh bridge0318 40_indoor_walk_big_circle ours
```

## 4. Evaluate

Use the same bridged motion and scene layout with your checkpoint:

```bash
bash run_bridged_eval.sh <DATE_TAG> <SEQ_NAME> <METHOD> /abs/path/to/last.ckpt
```

Dry-run only:

```bash
PRINT_ONLY=1 bash run_bridged_eval.sh bridge0318 40_indoor_walk_big_circle ours /abs/path/to/last.ckpt
```

## 5. Robot Viser

For robot playback from exported rigid-body recordings:

```bash
bash run_motiontracking_robot_viser.sh /abs/path/to/record_dir --port 8080
```

## 6. Notes

- Do not run `pip install MotionTracking`. The current `setup.py` pins
  `torch==2.0.1`, which breaks the validated Isaac Gym + CUDA 12.4 stack.
- The bridge uses `hmr/human_motion.npz`, not the raw scene NPZ.
- During motion conversion, the bridge creates a temporary `data/smpl` view and
  falls back to `SMPL_NEUTRAL.pkl` when `SMPL_FEMALE.pkl` is absent. It does not
  modify the repository assets.
- The train/eval wrappers automatically prepare `MotionTracking/data/smpl` from
  `../prep/data/smpl` using symlinks, and they also create a fallback
  `SMPL_FEMALE.pkl` symlink when only neutral or male models are available.

## 7. Validated Example

The following sequence was successfully bridged into MotionTracking:

```text
motion_data/bridge0318/40_indoor_walk_big_circle_ours.npy
motion_tracking/data/assets/urdf/bridge0318/40_indoor_walk_big_circle/ours/ours.urdf
```
