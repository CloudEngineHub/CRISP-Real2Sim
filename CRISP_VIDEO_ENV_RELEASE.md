# CRISP Video Quickstart

## Main Entry

```bash
bash setups/setup_crisp.sh
conda activate crisp
```

Optional check:

```bash
bash setups/validate_crisp_video_env.sh
```

## Assets

```bash
cd prep
bash smplme.sh
cd ..
bash setups/fetch_crisp_assets.sh
```

## Run

Demo:

```bash
bash setups/run_demo.sh
```

Your own data:

```bash
bash run_crisp_video.sh /abs/path/to/data_split_root
```

Input layout:

```text
<root>_videos/*.mp4
```

or

```text
<root>_img/<sequence>/*
```

## Files To Share

- `setups/setup_crisp.sh`
- `setups/fetch_crisp_assets.sh`
- `setups/run_demo.sh`
- `run_crisp_video.sh`
- `setups/validate_crisp_video_env.sh`
- `requirements-crisp-video.txt`
- `runtime_shims/sitecustomize.py`
