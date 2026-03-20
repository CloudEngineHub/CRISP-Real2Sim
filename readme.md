<div align="center">
	<h1>CRISP: Contact-guided Real2Sim from Monocular Video with Planar Scene Primitives</h1>
	<a href="https://arxiv.org/abs/2512.14696"><img src="https://img.shields.io/badge/arXiv-2512.14696-b31b1b" alt="arXiv"></a>
	<a href="https://crisp-real2sim.github.io/CRISP-Real2Sim/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
</div>
	
![teaser](https://raw.githubusercontent.com/Z1hanW/CRISP-Real2Sim/main/assets/crisp.png)

(Code is in beta test.)
---

## 1. Repository Setup

```bash
git clone --recursive https://github.com/Z1hanW/CRISP-Real2Sim.git
cd CRISP-Real2Sim
bash setup_crisp_video_env.sh crisp
conda activate crisp
bash validate_crisp_video_env.sh
```

Default environment names in this repo:
- `crisp`
- `crisp_contact`
- `crisp_rl`

Optional demo shortcut:

```bash
bash one_command_crisp_video_test.sh
```

### If You Need More Control

Use a custom environment name:

```bash
bash setup_crisp_video_env.sh my_crisp_env
conda activate my_crisp_env
bash validate_crisp_video_env.sh
```

Run the one-command demo wrapper with a custom environment or your own data root:

```bash
bash one_command_crisp_video_test.sh my_crisp_env /abs/path/to/data_split_root
```

---

## 2. Download Assets and Data

1. **SMPL / SMPL-X body models** (required for rendering and evaluation)
   - Register at [SMPL](https://smpl.is.tue.mpg.de/) and [SMPL-X](https://smpl-x.is.tue.mpg.de/).
   - For a clean clone, place the downloaded files using the structure below.

```text
prep/data/
└── body_models/
    ├── smpl/SMPL_{GENDER}.pkl
    └── smplx/SMPLX_{GENDER}.pkl or SMPLX_{GENDER}.npz
```

2. **Demo checkpoints and torch.hub caches** (optional, for the demo wrapper)

```bash
bash setup_crisp_video_env.sh crisp --with-assets
```

3. **Demo videos and metadata**

```bash
mkdir -p data
gdown --folder "https://drive.google.com/drive/folders/1k712Oj9StmWXRzSeSMiHZc3LtvsVk2Rw" -O data
```

> `gdown` is installed in the `crisp` environment. Use the `-O data` flag so Google Drive folders land under `CRISP-Real2Sim/data`.

---

## 3. Run the Full Pipeline

The wrapper and scripts expect your source sequences to live under either
`*_videos` or `*_img` folders. Remove that suffix when you feed paths to the
scripts.

```text
data/
├── demo_videos/
│   └── wall-kicking.mp4
└── YOUR_videos/
    ├── seq_a.mp4
    └── seq_b.mp4
```

```bash
bash run_crisp_video.sh --demo
```

For your own data:

```bash
bash run_crisp_video.sh /path/to/data/demo        # not /path/to/data/demo_videos
```

- The pipeline will iterate through every sequence under the root you supply.
- Intermediate outputs are written under `results/init/`.
- Final scene outputs are written under `results/output/scene/`.
- The main scene result is saved as:

```text
results/output/scene/<SEQ_NAME>_gv_sgd_cvd_hr.npz
```

- The SQS scene export is saved as:

```text
results/output/scene/<SEQ_NAME>/gv/scene_mesh_sqs/scene_mesh_sqs.urdf
```

### Validated Example

The current helper-based environment was validated by running one full video
through the new environment and producing:

```text
results/output/scene/wall-kicking-envtest-20260317_gv_sgd_cvd_hr.npz
results/output/scene/wall-kicking-envtest-20260317/gv/scene_mesh_sqs/scene_mesh_sqs.urdf
```

---

## 4. Contact Hallucination (Optional)

This step uses a separate environment because its dependency stack conflicts
with the main CRISP and MotionTracking environments.

```bash
bash setup_contact_predictor_env.sh crisp_contact
conda activate crisp_contact
cd prep/Contact-Predictor
bash fetch_data.sh hcontact-wScene
cd ../..
bash scripts/0_interactvlm.sh /abs/path/to/data/demo/wall-kicking stairs
```

Pass the sequence root without the `_img` suffix. The second argument is the
object name used in the contact prompt.

Outputs are written to:

```text
results/init/contacts/<camera>/*.npz
results/init/contact_vis/<camera>/*_vis.jpg
```

---

## 5. Visualize Human–Scene Reconstructions

Compile viser if needed:

```bash
cd vis_scripts/viser_m
pip install -e .
```

Visualize your sequences:

```bash
bash vis.sh ${SEQ_NAME}
```

Common flags (see script header for the full list):
- `--scene_name`: override the scene used for rendering.
- `--data_root`: custom data directory if not `./data`.
- `--out_dir`: write visualizations to a different folder.

---

## 6. Train Your Agent

```bash
cd MotionTracking
```

See [MotionTracking/README.md](MotionTracking/README.md).

That guide covers environment setup, CRISP-to-RL transfer, training, `viser`
debug runs, evaluation, and SMPL parameter export. The commands there assume
your working directory is already `MotionTracking`.

---

## 7. Visualize Your Agent

Agent visualization builds on the same `vis.sh` infrastructure:

```bash
python agents/vis_agent.py \
  --checkpoint path/to/checkpoint.pt \
  --seq ${SEQ_NAME} \
  --out_dir outputs/agent_viz/${SEQ_NAME}
```

Pass `--scene_name` or `--camera_pose_file` if your controller requires a custom scene or camera path.
