#!/usr/bin/env bash
set -eo pipefail

################################################################################
# 1) Make sure conda’s shell functions are available in this script:
eval "$(conda shell.bash hook)"

# --------------------- helpers ---------------------
# 打印并在指定 GPU 上运行 python 脚本（单进程独占该 GPU）
run_demo() {
  local gpu="$1"; shift
  local script="$1"; shift
  echo "[$(date +'%F %T')] [GPU ${gpu}] Run: $script $*"
  CUDA_VISIBLE_DEVICES="$gpu" python "$script" "$@" || \
    echo "[$(date +'%F %T')] [GPU ${gpu}] Error: $script $*"
}

# 发现可用 GPU：优先用 CUDA_VISIBLE_DEVICES，否则用 nvidia-smi
discover_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
  else
    mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || echo 0)
  fi
  NUM_GPUS="${#GPU_IDS[@]}"
  if (( NUM_GPUS == 0 )); then
    echo "No GPUs found. Set CUDA_VISIBLE_DEVICES or ensure nvidia-smi is available."
    exit 1
  fi
  echo "Using GPUs: ${GPU_IDS[*]}"
}

# 等待直到有一个 GPU 空闲，返回该 GPU 的索引（在 GPU_IDS 数组中的下标）
wait_for_free_gpu() {
  while true; do
    for i in "${!GPU_IDS[@]}"; do
      local pid="${GPU_PIDS[$i]:-}"
      if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        echo "$i"
        return
      fi
    done
    sleep 1
  done
}

# Ctrl-C 时干净地结束所有子进程
trap 'echo "Stopping..."; jobs -pr | xargs -r kill; wait' INT TERM

# --------------------- main ---------------------
conda activate sqs
cd /data3/zihanwa3/_Robotics/_geo/differentiable-blocksworld

# 参数
DATA_PATH="${1}_img"
HMR_TYPE="${2}"
SCRIPT="viser_m/visualizer_demo_ours.py"
LOG_DIR="${LOG_DIR:-/tmp/vis_megasam_logs}"
mkdir -p "$LOG_DIR"

discover_gpus
declare -a GPU_PIDS  # 按 GPU 索引记录当前在该 GPU 上跑的 PID

shopt -s nullglob
for folder in "$DATA_PATH"/*/; do
  seq="$(basename "$folder")"

  # 找到一个空闲 GPU（如果都忙就等待）
  idx="$(wait_for_free_gpu)"
  gpu="${GPU_IDS[$idx]}"
  logfile="${LOG_DIR}/${seq}.log"

  # 在该 GPU 上后台跑，并把 PID 记到对应槽位；输出单独写日志文件
  (
    run_demo "$gpu" "$SCRIPT" \
      --data "/data3/zihanwa3/_Robotics/_vision/mega-sam/postprocess/${seq}_${HMR_TYPE}_sgd_cvd_hr.npz"
  ) >"$logfile" 2>&1 &

  GPU_PIDS[$idx]=$!
  echo "Launched ${seq} on GPU ${gpu} (PID ${GPU_PIDS[$idx]}), log: $logfile"
done

# 等全部任务结束
wait
conda deactivate
echo "All demos completed successfully."
