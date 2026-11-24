import argparse
import glob
import json
import math
import os
import time
from typing import Optional

import torch
from torch.utils.data import DataLoader
import yaml

from src.data import PackedMixtureDataset
from src.eval import evaluate_model
from src.model import Llama7BForCausalLM, ModelConfig
from src.tokenizer import load_or_build_tokenizer


def setup_single_gpu() -> tuple[int, int, int, torch.device]:
    """Force single GPU training - no DDP, no distributed."""
    # Explicitly unset any distributed env vars that Kaggle might set
    for key in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]:
        os.environ.pop(key, None)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    return 0, 1, 0, device


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr_ratio: float) -> float:
    min_lr = max_lr * min_lr_ratio
    if step < warmup_steps:
        return max_lr * float(step + 1) / float(max(1, warmup_steps))
    if step >= max_steps:
        return min_lr
    progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


def get_amp_context(precision: str, device: torch.device):
    if device.type != "cuda" or precision == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    if precision == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        precision = "fp16"
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    ckpt_dirs = sorted(glob.glob(os.path.join(output_dir, "step_*")))
    if not ckpt_dirs:
        return None
    ckpt_dirs.sort(key=lambda p: int(os.path.basename(p).split("_")[-1]))
    return ckpt_dirs[-1]


def save_checkpoint(
    output_dir: str,
    step: int,
    model_raw: Llama7BForCausalLM,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    tokens_seen: int,
    max_to_keep: int,
) -> str:
    ckpt_dir = os.path.join(output_dir, f"step_{step:07d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model_raw.state_dict(), os.path.join(ckpt_dir, "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
    if scaler.is_enabled():
        torch.save(scaler.state_dict(), os.path.join(ckpt_dir, "scaler.pt"))
    with open(os.path.join(ckpt_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
        json.dump({"step": step, "tokens_seen": tokens_seen}, f, indent=2)

    all_ckpts = sorted(
        glob.glob(os.path.join(output_dir, "step_*")),
        key=lambda p: int(os.path.basename(p).split("_")[-1]),
    )
    if len(all_ckpts) > max_to_keep:
        for old_dir in all_ckpts[:-max_to_keep]:
            for fname in os.listdir(old_dir):
                os.remove(os.path.join(old_dir, fname))
            os.rmdir(old_dir)
    return ckpt_dir


def load_checkpoint(
    ckpt_dir: str,
    model_raw: Llama7BForCausalLM,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> tuple[int, int]:
    model_state = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location=device)
    model_raw.load_state_dict(model_state)

    opt_path = os.path.join(ckpt_dir, "optimizer.pt")
    if os.path.exists(opt_path):
        optimizer.load_state_dict(torch.load(opt_path, map_location=device))

    scaler_path = os.path.join(ckpt_dir, "scaler.pt")
    if scaler.is_enabled() and os.path.exists(scaler_path):
        scaler.load_state_dict(torch.load(scaler_path, map_location=device))

    state_path = os.path.join(ckpt_dir, "trainer_state.json")
    step, tokens_seen = 0, 0
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            st = json.load(f)
            step = int(st.get("step", 0))
            tokens_seen = int(st.get("tokens_seen", 0))
    return step, tokens_seen


def apply_debug_smoke_overrides(cfg: dict) -> dict:
    cfg["model"]["vocab_size"] = 1024
    cfg["model"]["hidden_size"] = 128
    cfg["model"]["intermediate_size"] = 256
    cfg["model"]["num_hidden_layers"] = 2
    cfg["model"]["num_attention_heads"] = 4
    cfg["model"]["num_key_value_heads"] = 2
    cfg["model"]["max_position_embeddings"] = 256
    cfg["model"]["gradient_checkpointing"] = False

    cfg["tokenizer"]["train_if_missing"] = False
    cfg["tokenizer"]["fallback_tokenizer"] = "gpt2"

    cfg["data"]["seq_len"] = 64
    cfg["data"]["num_workers"] = 0
    cfg["data"]["shuffle_buffer"] = 10
    # Use synthetic data for smoke test to avoid network
    cfg["data"]["sources"] = [
        {
            "name": "synthetic",
            "dataset_id": "synthetic",
            "config_name": None,
            "data_dir": None,
            "split": "train",
            "text_field": "text",
            "weight": 1.0,
            "streaming": False,
        }
    ]

    cfg["training"]["micro_batch_size"] = 2
    cfg["training"]["gradient_accumulation_steps"] = 1
    cfg["training"]["max_steps"] = 4
    cfg["training"]["warmup_steps"] = 1
    cfg["training"]["log_interval"] = 1
    cfg["training"]["save_interval"] = 2
    cfg["training"]["eval_interval"] = 2
    cfg["eval"]["eval_batches"] = 2
    cfg["eval"]["completion_prompts"] = []
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="7B LLM Pretraining Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--debug-smoke", action="store_true", help="Run tiny smoke test")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.debug_smoke:
        cfg = apply_debug_smoke_overrides(cfg)

    rank, world_size, _, device = setup_single_gpu()
    is_main = True  # Always main process in single GPU mode

    seed = int(cfg.get("seed", 42)) + rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if cfg.get("cache_dir"):
        os.environ.setdefault("HF_DATASETS_CACHE", cfg["cache_dir"])

    # Clear CUDA cache before model creation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    model_cfg = ModelConfig(**cfg["model"])
    tokenizer = load_or_build_tokenizer(cfg["tokenizer"], cfg["data"]["sources"], rank=rank)

    if not args.debug_smoke and hasattr(tokenizer, "vocab_size") and tokenizer.vocab_size:
        if int(cfg["model"].get("vocab_size", tokenizer.vocab_size)) != int(tokenizer.vocab_size):
            model_cfg.vocab_size = int(tokenizer.vocab_size)

    model = Llama7BForCausalLM(model_cfg).to(device)
    if is_main:
        counts = model.count_parameters()
        print(
            f"[model] Params -> total={counts['total']:,} ({counts['total']/1e9:.2f}B), "
            f"non_embed={counts['non_embedding']:,} ({counts['non_embedding']/1e9:.2f}B)"
        )

    model_raw = model  # No DDP in single GPU mode

    train_cfg = cfg["training"]
    # Use 8-bit AdamW from bitsandbytes to fit 7B on 16GB T4
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(
            model_raw.parameters(),
            lr=float(train_cfg["learning_rate"]),
            betas=(float(train_cfg["adam_beta1"]), float(train_cfg["adam_beta2"])),
            eps=float(train_cfg["adam_eps"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        print("[optimizer] Using 8-bit AdamW (bitsandbytes)")
    except ImportError:
        opt = torch.optim.AdamW(
            model_raw.parameters(),
            lr=float(train_cfg["learning_rate"]),
            betas=(float(train_cfg["adam_beta1"]), float(train_cfg["adam_beta2"])),
            eps=float(train_cfg["adam_eps"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        print("[optimizer] Using standard AdamW (bitsandbytes not available)")

    precision = train_cfg.get("precision", "bf16")
    use_scaler = (device.type == "cuda") and (precision == "fp16") and (not torch.cuda.is_bf16_supported())
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    step = 0
    tokens_seen = 0
    resume_mode = str(train_cfg.get("resume_from_checkpoint", "auto"))
    ckpt_to_load = None
    if resume_mode in ("auto", "latest"):
        ckpt_to_load = find_latest_checkpoint(output_dir)
    elif resume_mode != "none" and os.path.exists(resume_mode):
        ckpt_to_load = resume_mode

    if ckpt_to_load:
        step, tokens_seen = load_checkpoint(ckpt_to_load, model_raw, optimizer=opt, scaler=scaler, device=device)
        if is_main:
            print(f"[resume] Loaded checkpoint {ckpt_to_load} at step={step}, tokens_seen={tokens_seen:,}")

    dataset = PackedMixtureDataset(
        sources=cfg["data"]["sources"],
        tokenizer=tokenizer,
        seq_len=int(cfg["data"]["seq_len"]),
        seed=int(cfg.get("seed", 42)),
        shuffle_buffer=int(cfg["data"].get("shuffle_buffer", 10000)),
        rank=0,
        world_size=1,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg["micro_batch_size"]),
        num_workers=int(cfg["data"].get("num_workers", 2)),
        pin_memory=(device.type == "cuda"),
    )
    data_iter = iter(loader)

    grad_accum = int(train_cfg["gradient_accumulation_steps"])
    max_steps = int(train_cfg["max_steps"])
    warmup_steps = int(train_cfg["warmup_steps"])
    max_lr = float(train_cfg["learning_rate"])
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.1))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    log_interval = int(train_cfg.get("log_interval", 10))
    save_interval = int(train_cfg.get("save_interval", 1000))
    eval_interval = int(train_cfg.get("eval_interval", 1000))
    max_ckpts = int(train_cfg.get("max_checkpoints_to_keep", 3))
    max_runtime_sec = float(train_cfg.get("max_runtime_seconds", 39600))

    start_time = time.time()
    model.train()

    while step < max_steps:
        elapsed = time.time() - start_time
        if elapsed >= max_runtime_sec:
            if is_main:
                print(f"[time-guard] Reached {elapsed:.0f}s budget; saving checkpoint before exit.")
                save_checkpoint(output_dir, step, model_raw, opt, scaler, tokens_seen, max_ckpts)
            break

        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr_ratio)
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        step_tokens = 0

        for micro_idx in range(grad_accum):
            batch = next(data_iter)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            step_tokens += input_ids.numel()

            with get_amp_context(precision, device):
                out = model(input_ids=input_ids, labels=labels)
                loss = out["loss"] / grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss += float(loss.detach().item())
            # Periodic cache clear to prevent fragmentation
            if micro_idx % 32 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if scaler.is_enabled():
            scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model_raw.parameters(), max_grad_norm)

        if scaler.is_enabled():
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        step += 1
        tokens_seen += step_tokens

        if is_main and (step % log_interval == 0 or step == 1):
            dt = max(time.time() - start_time, 1e-6)
            tok_per_sec = tokens_seen / dt
            print(
                f"step={step}/{max_steps} | loss={accum_loss:.4f} | lr={lr:.2e} | "
                f"grad_norm={float(grad_norm):.3f} | tokens={tokens_seen:,} | avg_tok/s={tok_per_sec:,.0f}"
            )

        if eval_interval > 0 and (step % eval_interval == 0 or step == max_steps):
            eval_res = evaluate_model(
                model=model_raw,
                tokenizer=tokenizer,
                loader=loader,
                device=device,
                precision=precision,
                eval_batches=int(cfg["eval"].get("eval_batches", 10)),
                prompts=cfg["eval"].get("completion_prompts", []),
            )
            if is_main:
                print(
                    f"[eval @ step {step}] val_loss={eval_res['val_loss']:.4f} | "
                    f"ppl={eval_res['perplexity']:.2f}"
                )
                for sample in eval_res["generations"]:
                    print(f"  > Prompt: {sample['prompt']}\n    Out: {sample['completion']}")
            model.train()

        if is_main and save_interval > 0 and (step % save_interval == 0 or step == max_steps):
            ckpt_path = save_checkpoint(
                output_dir=output_dir,
                step=step,
                model_raw=model_raw,
                optimizer=opt,
                scaler=scaler,
                tokens_seen=tokens_seen,
                max_to_keep=max_ckpts,
            )
            print(f"[checkpoint] Saved {ckpt_path}")

    cleanup_single_gpu()


if __name__ == "__main__":
    main()