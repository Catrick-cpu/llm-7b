import math
from typing import Iterable

import torch
import torch.nn.functional as F

from src.model import Llama7BForCausalLM


@torch.no_grad()
def generate_greedy(
    model: Llama7BForCausalLM,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 48,
) -> str:
    was_training = model.training
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    max_len = model.config.max_position_embeddings

    for _ in range(max_new_tokens):
        ctx_ids = input_ids[:, -max_len:]
        out = model(input_ids=ctx_ids, labels=None)
        next_token = torch.argmax(out["logits"][:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    if was_training:
        model.train()
    return text


@torch.no_grad()
def evaluate_model(
    model: Llama7BForCausalLM,
    tokenizer,
    loader: Iterable,
    device: torch.device,
    precision: str = "bf16",
    eval_batches: int = 20,
    prompts: list[str] | None = None,
) -> dict:
    was_training = model.training
    model.eval()

    total_loss = 0.0
    count = 0
    loader_iter = iter(loader)

    for _ in range(eval_batches):
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        amp_enabled = device.type == "cuda" and precision in ("bf16", "fp16")
        amp_dtype = torch.bfloat16 if (precision == "bf16" and torch.cuda.is_bf16_supported()) else torch.float16
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            out = model(input_ids=input_ids, labels=labels)
            loss = out["loss"]

        if loss is not None and torch.isfinite(loss):
            total_loss += float(loss.item())
            count += 1

    mean_loss = total_loss / max(1, count)
    perplexity = math.exp(min(mean_loss, 20.0))

    generations = []
    for p in (prompts or []):
        full_out = generate_greedy(model, tokenizer, p, device=device, max_new_tokens=40)
        generations.append({"prompt": p, "completion": full_out})

    if was_training:
        model.train()

    return {
        "val_loss": mean_loss,
        "perplexity": perplexity,
        "generations": generations,
    }