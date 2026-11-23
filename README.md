# `llm-7b` — 7B Decoder-Only Transformer Pretraining for Kaggle

A clean, modular PyTorch pretraining pipeline for a ~7B Llama-style causal language model using **Grouped-Query Attention (GQA)**, **RoPE**, **RMSNorm**, **SwiGLU**, multi-corpus streaming/packing, and session-safe checkpointing for Kaggle.

## Quick Start

### 1. Fast Smoke Test (CPU or Single GPU)
```bash
pip install -r requirements.txt
bash scripts/launch_kaggle.sh configs/pretrain_7b_kaggle.yaml smoke
```

### 2. Multi-GPU / Kaggle Session Launch
```bash
bash scripts/launch_kaggle.sh configs/pretrain_7b_kaggle.yaml train
```

## Kaggle Operational Notes
* **Session Timeout Guard:** `max_runtime_seconds` saves a checkpoint and exits cleanly before Kaggle's ~12-hour runtime cap. On the next session, `resume_from_checkpoint: auto` automatically picks up from the latest `step_*` directory.
* **Memory Budget:** Gradient checkpointing + micro-batching + gradient accumulation keep peak activation memory bounded on 16 GB cards (`T4` / `P100`). For full multi-node/FSDP scale at 7B parameters, pair with `accelerate` or PyTorch FSDP.