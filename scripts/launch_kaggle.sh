#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/pretrain_7b_kaggle.yaml}"
MODE="${2:-train}"

# Ensure we're in the repo root
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="false"
# Memory optimization for 16GB T4
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32"
export CUDA_LAUNCH_BLOCKING=0

if [[ "$MODE" == "smoke" ]]; then
  python3 -m src.train --config "$CONFIG_PATH" --debug-smoke
  exit 0
fi

# Force single GPU on Kaggle (7B + optimizer states won't fit on 16GB x2 with DDP)
# Use gradient accumulation for effective batch size
echo "Using single GPU with memory optimizations (seq_len=256, grad_accum=256)"
export CUDA_VISIBLE_DEVICES=0
python3 -m src.train --config "$CONFIG_PATH"