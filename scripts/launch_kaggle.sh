#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/pretrain_7b_kaggle.yaml}"
MODE="${2:-train}"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export TOKENIZERS_PARALLELISM="false"

if [[ "$MODE" == "smoke" ]]; then
  python3 -m src.train --config "$CONFIG_PATH" --debug-smoke
  exit 0
fi

# Force single GPU on Kaggle (7B + optimizer states won't fit on 16GB x2 with DDP)
# Use gradient accumulation for effective batch size
echo "Using single GPU (7B model requires ~14GB weights + optimizer states)"
export CUDA_VISIBLE_DEVICES=0
python3 -m src.train --config "$CONFIG_PATH"