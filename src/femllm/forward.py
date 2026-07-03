import math
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class ModelConfig:
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    orig_dtype = x.dtype
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight * x).to(orig_dtype)


def _build_rope(seq_len: int, head_dim: int, theta: float, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).to(dtype)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).to(dtype)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = position_ids[-1].item() + 1
    head_dim = q.shape[-1]
    cos, sin = _build_rope(seq_len, head_dim, theta, q.dtype)
    cos = cos[position_ids]  # [seq, head_dim]
    sin = sin[position_ids]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


def forward_layer(
    weights: dict[str, torch.Tensor],
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]],
    layer_idx: int,
    config: ModelConfig,
) -> torch.Tensor:
    bsz, seq_len, _ = hidden_states.shape

    residual = hidden_states
    hidden_states = rms_norm(hidden_states, weights["input_layernorm.weight"], config.rms_norm_eps)

    q = hidden_states @ weights["self_attn.q_proj.weight"].T
    k = hidden_states @ weights["self_attn.k_proj.weight"].T
    v = hidden_states @ weights["self_attn.v_proj.weight"].T

    q = q.view(bsz, seq_len, config.num_heads, config.head_dim).transpose(1, 2)
    k = k.view(bsz, seq_len, config.num_kv_heads, config.head_dim).transpose(1, 2)
    v = v.view(bsz, seq_len, config.num_kv_heads, config.head_dim).transpose(1, 2)

    q, k = _apply_rope(q, k, position_ids, config.rope_theta)

    if layer_idx in kv_cache:
        k_past, v_past = kv_cache[layer_idx]
        k = torch.cat([k_past, k], dim=2)
        v = torch.cat([v_past, v], dim=2)
    kv_cache[layer_idx] = (k, v)

    groups = config.num_heads // config.num_kv_heads
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)

    attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=(seq_len > 1))
    attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, config.num_heads * config.head_dim)
    attn_out = attn_out @ weights["self_attn.o_proj.weight"].T
    hidden_states = residual + attn_out

    residual = hidden_states
    hidden_states = rms_norm(hidden_states, weights["post_attention_layernorm.weight"], config.rms_norm_eps)
    gate = F.silu(hidden_states @ weights["mlp.gate_proj.weight"].T)
    up = hidden_states @ weights["mlp.up_proj.weight"].T
    hidden_states = (gate * up) @ weights["mlp.down_proj.weight"].T

    return residual + hidden_states
