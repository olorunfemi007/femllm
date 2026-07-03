import torch
import pytest
from src.femllm.forward import forward_layer, rms_norm, ModelConfig
from src.femllm.layer_loader import load_layer_weights

TINYLLAMA_CONFIG = ModelConfig(
    hidden_size=2048,
    num_heads=32,
    num_kv_heads=4,
    head_dim=64,
    intermediate_size=5632,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
)

def test_rms_norm_output_shape():
    x = torch.randn(1, 10, 2048, dtype=torch.bfloat16)
    w = torch.ones(2048, dtype=torch.bfloat16)
    out = rms_norm(x, w)
    assert out.shape == (1, 10, 2048)

def test_rms_norm_unit_weight_near_unit_norm():
    x = torch.randn(1, 5, 2048, dtype=torch.bfloat16)
    w = torch.ones(2048, dtype=torch.bfloat16)
    out = rms_norm(x, w)
    norms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(norms, torch.ones_like(norms), atol=0.1)

def test_forward_layer_output_shape(shard_dir):
    weights = load_layer_weights(f"{shard_dir}/worker_0.safetensors", layer_idx=0)
    weights = {k: v.to(torch.bfloat16) for k, v in weights.items()}
    hidden_states = torch.randn(1, 5, 2048, dtype=torch.bfloat16)
    position_ids = torch.arange(5)
    kv_cache = {}
    out = forward_layer(weights, hidden_states, position_ids, kv_cache, layer_idx=0, config=TINYLLAMA_CONFIG)
    assert out.shape == (1, 5, 2048)

def test_forward_layer_populates_kv_cache(shard_dir):
    weights = load_layer_weights(f"{shard_dir}/worker_0.safetensors", layer_idx=0)
    weights = {k: v.to(torch.bfloat16) for k, v in weights.items()}
    hidden_states = torch.randn(1, 5, 2048, dtype=torch.bfloat16)
    position_ids = torch.arange(5)
    kv_cache = {}
    forward_layer(weights, hidden_states, position_ids, kv_cache, layer_idx=0, config=TINYLLAMA_CONFIG)
    assert 0 in kv_cache
    k, v = kv_cache[0]
    assert k.shape == (1, 4, 5, 64)  # [bsz, num_kv_heads, seq_len, head_dim]
    assert v.shape == (1, 4, 5, 64)

def test_forward_layer_decode_extends_kv_cache(shard_dir):
    weights = load_layer_weights(f"{shard_dir}/worker_0.safetensors", layer_idx=0)
    weights = {k: v.to(torch.bfloat16) for k, v in weights.items()}
    kv_cache = {}
    hidden = torch.randn(1, 5, 2048, dtype=torch.bfloat16)
    forward_layer(weights, hidden, torch.arange(5), kv_cache, layer_idx=0, config=TINYLLAMA_CONFIG)
    hidden = torch.randn(1, 1, 2048, dtype=torch.bfloat16)
    forward_layer(weights, hidden, torch.tensor([5]), kv_cache, layer_idx=0, config=TINYLLAMA_CONFIG)
    k, v = kv_cache[0]
    assert k.shape == (1, 4, 6, 64)  # 5 + 1

def test_forward_layer_dtype_preserved(shard_dir):
    weights = load_layer_weights(f"{shard_dir}/worker_0.safetensors", layer_idx=0)
    weights = {k: v.to(torch.bfloat16) for k, v in weights.items()}
    hidden_states = torch.randn(1, 3, 2048, dtype=torch.bfloat16)
    kv_cache = {}
    out = forward_layer(weights, hidden_states, torch.arange(3), kv_cache, layer_idx=0, config=TINYLLAMA_CONFIG)
    assert out.dtype == torch.bfloat16
