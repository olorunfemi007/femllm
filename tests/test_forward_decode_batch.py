import copy
import torch
import pytest
from src.femllm.forward import forward_layer, forward_layer_decode_batch, ModelConfig
from src.femllm.layer_loader import load_layer_weights

TINYLLAMA_CONFIG = ModelConfig(
    hidden_size=2048, num_heads=32, num_kv_heads=4, head_dim=64,
    intermediate_size=5632, rms_norm_eps=1e-5, rope_theta=10000.0,
)

def _weights(shard_dir):
    weights = load_layer_weights(f"{shard_dir}/worker_0.safetensors", layer_idx=0)
    return {k: v.to(torch.bfloat16) for k, v in weights.items()}

def test_decode_batch_output_shape(shard_dir):
    weights = _weights(shard_dir)
    hidden = torch.randn(3, 1, 2048, dtype=torch.bfloat16)
    position_ids = torch.tensor([2, 5, 9])
    kv_caches = [{}, {}, {}]
    out = forward_layer_decode_batch(weights, hidden, position_ids, kv_caches, layer_idx=0, config=TINYLLAMA_CONFIG)
    assert out.shape == (3, 1, 2048)

def test_decode_batch_updates_each_rows_own_kv_cache(shard_dir):
    weights = _weights(shard_dir)
    hidden = torch.randn(2, 1, 2048, dtype=torch.bfloat16)
    position_ids = torch.tensor([0, 0])
    kv_caches = [{}, {}]
    forward_layer_decode_batch(weights, hidden, position_ids, kv_caches, layer_idx=0, config=TINYLLAMA_CONFIG)
    assert 0 in kv_caches[0]
    assert 0 in kv_caches[1]
    k0, _ = kv_caches[0][0]
    k1, _ = kv_caches[1][0]
    assert k0.shape == (1, 4, 1, 64)
    assert k1.shape == (1, 4, 1, 64)

def test_decode_batch_matches_individual_calls_at_different_positions(shard_dir):
    """
    The core correctness claim: two requests with DIFFERENT accumulated KV-cache
    lengths (simulating different prompt lengths) must batch together and produce
    results identical to running each one individually through forward_layer.
    """
    weights = _weights(shard_dir)

    kv_cache_a = {}
    forward_layer(weights, torch.randn(1, 5, 2048, dtype=torch.bfloat16), torch.arange(5), kv_cache_a, layer_idx=0, config=TINYLLAMA_CONFIG)
    kv_cache_b = {}
    forward_layer(weights, torch.randn(1, 9, 2048, dtype=torch.bfloat16), torch.arange(9), kv_cache_b, layer_idx=0, config=TINYLLAMA_CONFIG)

    torch.manual_seed(0)
    new_token_a = torch.randn(1, 1, 2048, dtype=torch.bfloat16)
    new_token_b = torch.randn(1, 1, 2048, dtype=torch.bfloat16)

    ref_cache_a = copy.deepcopy(kv_cache_a)
    ref_out_a = forward_layer(weights, new_token_a.clone(), torch.tensor([5]), ref_cache_a, layer_idx=0, config=TINYLLAMA_CONFIG)
    ref_cache_b = copy.deepcopy(kv_cache_b)
    ref_out_b = forward_layer(weights, new_token_b.clone(), torch.tensor([9]), ref_cache_b, layer_idx=0, config=TINYLLAMA_CONFIG)

    batch_hidden = torch.cat([new_token_a.clone(), new_token_b.clone()], dim=0)
    batch_positions = torch.tensor([5, 9])
    kv_caches = [kv_cache_a, kv_cache_b]
    batch_out = forward_layer_decode_batch(weights, batch_hidden, batch_positions, kv_caches, layer_idx=0, config=TINYLLAMA_CONFIG)

    assert torch.allclose(batch_out[0:1], ref_out_a, atol=1e-2)
    assert torch.allclose(batch_out[1:2], ref_out_b, atol=1e-2)
