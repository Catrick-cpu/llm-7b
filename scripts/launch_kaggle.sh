#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/pretrain_7b_kaggle.yaml}"
MODE="${2:-train}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32"

# Explicitly unset any distributed env vars that Kaggle might set
unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT

if [[ "$MODE" == "smoke" ]]; then
  CUDA_VISIBLE_DEVICES=0 python3 -m src.train --config "$CONFIG_PATH" --debug-smoke
  exit 0
fi

# FORCE SINGLE GPU - no torchrun, no DDP
echo "=== FORCING SINGLE GPU (CUDA_VISIBLE_DEVICES=0) ==="
export CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=0 python3 -m src.train --config "$CONFIG_PATH"