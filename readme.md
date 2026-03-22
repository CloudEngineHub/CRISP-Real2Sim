<div align="center">
	<h1>CRISP: Contact-guided Real2Sim from Monocular Video with Planar Scene Primitives</h1>
	<a href="https://arxiv.org/abs/2512.14696"><img src="https://img.shields.io/badge/arXiv-2512.14696-b31b1b" alt="arXiv"></a>
	<a href="https://openreview.net/pdf?id=xlr3NqxUqY"><img src="https://img.shields.io/badge/ICLR_Version-pdf-orange" alt="ICLR Version"></a>
	<a href="https://crisp-real2sim.github.io/CRISP-Real2Sim/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
	<a href="https://drive.google.com/drive/folders/1PX8Pqzqjlh5v0Z6xt-NjzTgpugk4igoN?usp=drive_link"><img src="https://img.shields.io/badge/Video_Dataset-blue" alt="Video Dataset"></a>
</div>
	
![teaser](https://raw.githubusercontent.com/Z1hanW/CRISP-Real2Sim/main/assets/crisp.png)

We open source the code and video we used. See [Video Dataset](#video-dataset).

Code pipeline, in one line: scripts `1-8` are `1)` video-to-images convention, `2)` human masks, `3)` improved scene reconstruction, `4)` camera postprocess, `5)` GVHMR, `6)` human-scene alignment and opitmization, `7)` planar fitting, `8)` post-scene alignment + bridge; `MotionTracking` then handles RL train/eval/viser.

---

### 1. Repository Setup

```bash
git clone --recursive https://github.com/Z1hanW/CRISP-Real2Sim.git
cd CRISP-Real2Sim
bash setups/setup_crisp.sh
conda activate crisp
```

Optional check:

```bash
bash setups/validate_crisp_video_env.sh
```

Optional demo shortcut: [`run_demo.sh`](setups/run_demo.sh)

---

### 2. Download Assets and Data

1. **SMPL / SMPL-X body models** (required for rendering and evaluation)
   - Register at [SMPL](https://smpl.is.tue.mpg.de/) and [SMPL-X](https://smpl-x.is.tue.mpg.de/).
   - Recommended:

```bash
cd prep
bash smplme.sh
cd ..
bash setups/fetch_crisp_assets.sh
```

   - `smplme.sh` fills the SMPL / SMPL-X body-model paths.
   - `setups/fetch_crisp_assets.sh` pulls the demo checkpoints and the extra neutral SMPL file used by HMR.
   - If the upstream download flow fails, place the files manually using the structure below.

```text
prep/data/
└── body_models/
    ├── smpl/SMPL_{GENDER}.pkl
    └── smplx/SMPLX_{GENDER}.pkl or SMPLX_{GENDER}.npz
```

2. **Demo videos and metadata**

```bash
mkdir -p data
gdown --folder "https://drive.google.com/drive/folders/1k712Oj9StmWXRzSeSMiHZc3LtvsVk2Rw" -O data
```

> `gdown` is installed in the `crisp` environment. Use the `-O data` flag so Google Drive folders land under `CRISP-Real2Sim/data`.

---

### 3. Run the Full Pipeline

Wrapper order:

```text
all_gv.sh:
  1_video2imgs -> 2_get_mask -> 3_megasam -> 4_post_camera -> 5_grav
  -> 0_ufm -> 6_align -> 7_glue_sqs -> 8_postprocessing

all_gv_contact.sh:
  1_video2imgs -> 0_interactvlm -> 2_get_mask -> 3_megasam -> 4_post_camera
  -> 5_grav -> 0_ufm -> 6_align -> 7_glue_sqs(use contact) -> 8_postprocessing
```

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
- By default this wrapper also runs `scripts/8_postprocessing.sh`, so it
  continues from `scene` into `post_scene` and the MotionTracking bridge.
- If you want that end-to-end path to finish in one shot, install `crisp_rl`
  first with `bash setups/setup_crisp_rl.sh`.
- The main scene result is saved as:

```text
results/output/scene/<SEQ_NAME>_gv_sgd_cvd_hr.npz
```

- The SQS scene export is saved as:

```text
results/output/scene/<SEQ_NAME>/gv/scene_mesh_sqs/scene_mesh_sqs.urdf
```

- The aligned post-processing output is saved as:

```text
results/output/post_scene/<SEQ_NAME>/gv/hmr/human_motion.npz
```

- The bridged MotionTracking motion is saved as:

```text
MotionTracking/motion_data/<DATE_TAG>/<SEQ_NAME>_ours.npy
```

Validated on the demo sequence:

```text
results/output/scene/wall-kicking-smoke_gv_sgd_cvd_hr.npz
results/output/scene/wall-kicking-smoke/gv/scene_mesh_sqs/scene_mesh_sqs.urdf
```
---

### 4. Contact Hallucination (Optional)

It is optional and is not part of `run_crisp_video.sh`. This step uses a separate environment because its dependency stack conflicts
with the main CRISP and MotionTracking environments.

```bash
bash setups/setup_crisp_contact.sh
cd prep/Contact-Predictor
bash fetch_data.sh hcontact-wScene
cd ../..
bash scripts/0_interactvlm.sh /abs/path/to/data/demo/pkr stairs # 'stairs' is the object name , replace it for other object
```

Outputs are written to:

```text
results/init/contacts/<camera>/*.npz
results/init/contact_vis/<camera>/*_vis.jpg
```

After this step, you can enable contact-aware visualization from the main
`crisp` environment.

If you want a single batch entry with contact hallucination included:

```bash
bash scripts/all_gv_contact.sh /abs/path/to/data/demo stairs
```

Like `run_crisp_video.sh`, this batch entry now also continues through
`post_scene` and the MotionTracking bridge by default.

<sub>Contact hallucination is currently not very stable, and it may not produce reasonable results for every video.</sub>

---

### 5. Visualize Human–Scene Reconstructions

Compile viser if needed:

```bash
cd vis_scripts/viser_m
pip install -e .
```

Visualize your sequences:

```bash
bash vis.sh ${SEQ_NAME}
```

If you also ran the optional Contact Hallucination step:

```bash
USE_CONTACT=on bash vis.sh ${SEQ_NAME}
```

Common flags (see script header for the full list):
- `--scene_name`: override the scene used for rendering.
- `--data_root`: custom data directory if not `./data`.
- `--out_dir`: write visualizations to a different folder.

---

### 6. Train Your Agent

If you already have an older run that stopped at `results/output/scene/...`,
you can rerun only the alignment + bridge stage from the repository root:

```bash
conda activate crisp
bash scripts/8_postprocessing.sh smoke gv
```

That step produces:

```text
results/output/post_scene/wall-kicking-smoke/gv/
```

and bridges the demo into MotionTracking under the default date tag `bridge`
(or a custom `RL_DATE` if you set one).

```bash
cd MotionTracking
```

See [MotionTracking/README.md](MotionTracking/README.md).

That guide covers environment setup, CRISP-to-RL transfer, training, `viser`
debug runs, evaluation, and SMPL parameter export. The commands there assume
your working directory is already `MotionTracking`.

---

### 7. Visualize Your Agent

Agent visualization builds on the same `vis.sh` infrastructure:

```bash
python agents/vis_agent.py \
  --checkpoint path/to/checkpoint.pt \
  --seq ${SEQ_NAME} \
  --out_dir outputs/agent_viz/${SEQ_NAME}
```

Pass `--scene_name` or `--camera_pose_file` if your controller requires a custom scene or camera path.

---

## Video Dataset

We release a curated and clipped video dataset here:
[Video Dataset](https://drive.google.com/drive/folders/1PX8Pqzqjlh5v0Z6xt-NjzTgpugk4igoN?usp=drive_link).

It includes both self-captured videos and internet videos we collect with
hours efforts. A substantial
portion of these videos currently fail in CRISP because HMR is still not
reliable under high-dynamics motion. We still decided to release them because
we know that finding and cleaning suitable videos is a real bottleneck for
such a real2sim pipeline.

If these video data are helpful for your work, please consider citing CRISP.

---

## Citation

If the idea, code, visualization, or video data are helpful for your research,
please consider citing CRISP.

```bibtex
@inproceedings{
wang2026contactguided,
title={Contact-guided Real2Sim from Monocular Video with Planar Scene Primitives},
author={Zihan Wang and Jiashun Wang and Jeff Tan and Yiwen Zhao and Jessica K. Hodgins and Shubham Tulsiani and Deva Ramanan},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=xlr3NqxUqY}
}
```

## Acknowledgment

We thank [viser](https://github.com/viser-project/viser) for supporting our visualization workflow.
