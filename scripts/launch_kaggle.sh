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

NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")
echo "Detected GPUs: $NUM_GPUS"

if [[ "$NUM_GPUS" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="$NUM_GPUS" -m src.train --config "$CONFIG_PATH"
else
  python3 -m src.train --config "$CONFIG_PATH"
fi