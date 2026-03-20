# CRISP Video Quickstart

## Main Entry

```bash
bash setup_crisp.sh
conda activate crisp
```

Optional check:

```bash
bash validate_crisp_video_env.sh
```

## Assets

```bash
cd prep
bash smplme.sh
cd ..
bash fetch_crisp_assets.sh
```

## Run

Demo:

```bash
bash run_demo.sh
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

- `setup_crisp.sh`
- `fetch_crisp_assets.sh`
- `run_demo.sh`
- `run_crisp_video.sh`
- `validate_crisp_video_env.sh`
- `requirements-crisp-video.txt`
- `runtime_shims/sitecustomize.py`
