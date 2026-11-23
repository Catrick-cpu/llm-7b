from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 65536
    hidden_size: int = 4096
    intermediate_size: int = 13056
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 4096
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    use_flash_attn: bool = True
    gradient_checkpointing: bool = True

    @property
    def head_dim(self) -> int:
        assert self.hidden_size % self.num_attention_heads == 0
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        assert self.num_attention_heads % self.num_key_value_heads == 0
        return self.num_attention_heads // self.num_key_value_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        normed = x_fp32 * torch.rsqrt(variance + self.eps)
        return (normed * self.weight.float()).to(orig_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 500000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cos_sin(max_seq_len, device=inv_freq.device)

    def _build_cos_sin(self, seq_len: int, device: torch.device) -> None:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self._cached_len = seq_len

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > getattr(self, "_cached_len", 0) or self.cos_cached.device != device:
            self._build_cos_sin(max(seq_len, self.max_seq_len), device=device)
        cos = self.cos_cached[:seq_len].to(dtype=dtype)
        sin = self.sin_cached[:seq_len].to(dtype=dtype)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if position_ids is not None:
        cos_pos = cos[position_ids].unsqueeze(1)
        sin_pos = sin[position_ids].unsqueeze(1)
    else:
        seq_len = q.shape[2]
        cos_pos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
        sin_pos = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos_pos) + (_rotate_half(q) * sin_pos)
    k_embed = (k * cos_pos) + (_rotate_half(k) * sin_pos)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim
    )
    return expanded.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_key_value_groups
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=position_ids)

        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        is_causal = attention_mask is None and q_len > 1
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUFeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cos=cos,
            sin=sin,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class Llama7BForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(
            dim=config.head_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
        )
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        scale = 1.0 / math.sqrt(2.0 * config.num_hidden_layers)
        for layer in self.layers:
            nn.init.normal_(layer.self_attn.o_proj.weight, mean=0.0, std=0.02 * scale)
            nn.init.normal_(layer.mlp.down_proj.weight, mean=0.0, std=0.02 * scale)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        embed = self.embed_tokens.weight.numel()
        lm_head = 0 if self.lm_head is None else self.lm_head.weight.numel()
        non_embed = total - embed - lm_head
        return {
            "total": total,
            "trainable": trainable,
            "non_embedding": non_embed,
            "embeddings_and_head": embed + lm_head,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, Optional[torch.Tensor]]:
        bsz, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(seq_len=seq_len, device=hidden_states.device, dtype=hidden_states.dtype)

        use_ckpt = self.config.gradient_checkpointing and self.training
        for layer in self.layers:
            if use_ckpt:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    cos,
                    sin,
                    position_ids,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(
                    hidden_states=hidden_states,
                    cos=cos,
                    sin=sin,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                )

        hidden_states = self.norm(hidden_states)
        logits = (
            F.linear(hidden_states, self.embed_tokens.weight)
            if self.lm_head is None
            else self.lm_head(hidden_states)
        )

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits}