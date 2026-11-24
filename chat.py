#!/usr/bin/env python3
"""
Simple chat script for trained models.
Usage: python chat.py [--config CONFIG] [--checkpoint CHECKPOINT_DIR]
"""
import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, '.')

from src.model import Llama7BForCausalLM, ModelConfig
from transformers import AutoTokenizer


def find_latest_checkpoint(output_dir: str) -> str:
    ckpt_dirs = sorted(
        glob.glob(os.path.join(output_dir, "step_*")),
        key=lambda p: int(os.path.basename(p).split("_")[-1]),
    )
    if not ckpt_dirs:
        raise FileNotFoundError(f"No checkpoints found in {output_dir}")
    return os.path.join(ckpt_dirs[-1], "model.pt")


def load_config_from_yaml(config_path: str) -> ModelConfig:
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    m = cfg["model"]
    return ModelConfig(
        vocab_size=m.get("vocab_size", 50257),
        hidden_size=m["hidden_size"],
        intermediate_size=m["intermediate_size"],
        num_hidden_layers=m["num_hidden_layers"],
        num_attention_heads=m["num_attention_heads"],
        num_key_value_heads=m.get("num_key_value_heads", m["num_attention_heads"]),
        max_position_embeddings=m.get("max_position_embeddings", 2048),
        rope_theta=m.get("rope_theta", 10000.0),
        rms_norm_eps=m.get("rms_norm_eps", 1e-5),
        tie_word_embeddings=m.get("tie_word_embeddings", False),
        use_flash_attn=m.get("use_flash_attn", False),
        gradient_checkpointing=False,  # Inference: no checkpointing
    )


def main():
    parser = argparse.ArgumentParser(description="Chat with trained model")
    parser.add_argument("--config", type=str, default="configs/train_elitebook.yaml",
                        help="Path to training config YAML")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Specific checkpoint dir (default: latest)")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    print(f"Loading config from {args.config}...")
    model_cfg = load_config_from_yaml(args.config)

    print("Initializing model...")
    model = Llama7BForCausalLM(model_cfg)

    # Determine checkpoint
    if args.checkpoint:
        ckpt_path = os.path.join(args.checkpoint, "model.pt")
    else:
        output_dir = "checkpoints/llm-50m-elitebook"
        if not os.path.exists(output_dir):
            # Try to infer from config
            import yaml
            with open(args.config) as f:
                cfg = yaml.safe_load(f)
            output_dir = cfg.get("output_dir", "./checkpoints/llm-50m-elitebook")
        ckpt_path = find_latest_checkpoint(output_dir)

    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    print(f"\n=== Chat ready (Ctrl+C to exit) ===")
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {ckpt_path}")
    print("Try: 'def fibonacci(n):'  |  'User: Hello\\nAssistant:'  |  'What is Python?'")

    while True:
        try:
            prompt = input("\nYou: ")
            if not prompt.strip():
                continue

            ids = tok.encode(prompt, return_tensors="pt")

            with torch.no_grad():
                out = model.generate(
                    ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    do_sample=True,
                    pad_token_id=tok.eos_token_id,
                    eos_token_id=tok.eos_token_id,
                )

            new_tokens = out[0][ids.shape[1]:]
            response = tok.decode(new_tokens, skip_special_tokens=True)
            print(f"Bot: {response}")

        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()