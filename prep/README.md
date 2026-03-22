# Prep

Use this page for asset, model, and optional contact-data preparation.

## 1. Core Assets and Data

### SMPL / SMPL-X body models

Register at [SMPL](https://smpl.is.tue.mpg.de/) and [SMPL-X](https://smpl-x.is.tue.mpg.de/), then run:

```bash
cd prep
bash smplme.sh
cd ..
bash setups/fetch_crisp_assets.sh
```

`smplme.sh` fills the SMPL / SMPL-X body-model paths.

`setups/fetch_crisp_assets.sh` pulls the demo checkpoints and the extra neutral
SMPL file used by HMR.

If the upstream download flow fails, place the files manually as:

```text
prep/data/
└── body_models/
    ├── smpl/SMPL_{GENDER}.pkl
    └── smplx/SMPLX_{GENDER}.pkl or SMPLX_{GENDER}.npz
```

### Demo videos and metadata

```bash
mkdir -p data
gdown --folder "https://drive.google.com/drive/folders/1k712Oj9StmWXRzSeSMiHZc3LtvsVk2Rw" -O data
```

`gdown` is installed in the `crisp` environment. Use `-O data` so Google Drive
folders land under `CRISP-Real2Sim/data`.

## 2. Optional Contact Hallucination

This step uses a separate environment because its dependency stack conflicts
with the main CRISP and MotionTracking environments.

```bash
bash setups/setup_crisp_contact.sh
cd prep/Contact-Predictor
bash fetch_data.sh hcontact-wScene
cd ../..
bash scripts/0_interactvlm.sh /abs/path/to/data/demo/pkr stairs
```

`stairs` is the default object name in the contact prompt. Replace it for other
objects if needed.

Outputs are written to:

```text
results/init/contacts/<camera>/*.npz
results/init/contact_vis/<camera>/*_vis.jpg
```

If you want a single batch entry with contact hallucination included:

```bash
bash scripts/all_gv_contact.sh /abs/path/to/data/demo stairs
```

That batch entry also continues through `post_scene` and the MotionTracking
bridge by default.

Contact hallucination is currently not very stable, and it may not produce
reasonable results for every video.
