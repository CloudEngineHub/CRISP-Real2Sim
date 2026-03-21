# MotionTracking

Use this after CRISP has produced aligned `post_scene` outputs:

```text
results/output/post_scene/<SEQ>/<HMR_TYPE>/
├── hmr/human_motion.npz
└── scene_mesh_sqs/
```

`results/output/scene/...` alone is not enough.

## 1. Install The Environment

From the repository root:

```bash
bash setups/setup_crisp_rl.sh
conda activate crisp_rl
```

## 2. Transfer CRISP Output Into The RL Layout

Full bridge into `MotionTracking/motion_data` and `MotionTracking/motion_tracking/data/assets/urdf`:

```bash
bash bridge_crisp_sequence.sh <SEQ_NAME> <DATE_TAG>
```

Optional motion-only helper from the repository root:

```bash
python vis_scripts/viser_m/process_to_rl.py --seq-names <SEQ_NAME> --date <DATE_TAG>
```

## 3. Train

Default training:

```bash
bash run_bridged_train.sh <DATE_TAG> <SEQ_NAME>
```

Live `viser` debug run instead of `headless=True`:

```bash
MT_VISER_PORT=8080 bash run_bridged_train.sh <DATE_TAG> <SEQ_NAME> headless=False num_envs=1 batch_size=8 visualize_markers=False
```

Use the second command for visualization/debug. Keep the first command for the normal large-batch recipe.

## 4. Evaluate

Headless evaluation:

```bash
bash run_bridged_eval.sh <DATE_TAG> <SEQ_NAME> /abs/path/to/last.ckpt
```

`viser` evaluation:

```bash
MT_VISER_PORT=8081 bash run_bridged_eval_viser.sh <DATE_TAG> <SEQ_NAME> /abs/path/to/last.ckpt
```

## 5. Export SMPL Parameters To File

```bash
bash run_bridged_export_motion.sh <DATE_TAG> <SEQ_NAME> /abs/path/to/last.ckpt
```

Default output:

```text
results/export_motion/<DATE_TAG>_<SEQ_NAME>_ours/000/trajectory_pose_aa_0.pkl
```

That file contains:

- `pose`
- `trans`
- `shape`
- `gender`

## 6. Replay Exported Robot Motion

```bash
bash run_motiontracking_robot_viser.sh /abs/path/to/record_dir --port 8080
```
