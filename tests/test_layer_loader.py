import torch
import pytest
from src.femllm.layer_loader import load_layer_weights

EXPECTED_KEYS = {
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
}

def test_load_layer_returns_all_weight_keys(shard_dir):
    weights = load_layer_weights(f"{shard_dir}/worker_0.safetensors", layer_idx=0)
    assert set(weights.keys()) == EXPECTED_KEYS

def test_load_layer_q_proj_shape(shard_dir):
    weights = load_layer_weights(f"{shard_dir}/worker_0.safetensors", layer_idx=0)
    assert weights["self_attn.q_proj.weight"].shape == (2048, 2048)

def test_load_layer_k_proj_shape(shard_dir):
    weights = load_layer_weights(f"{shard_dir}/worker_0.safetensors", layer_idx=0)
    assert weights["self_attn.k_proj.weight"].shape == (256, 2048)

def test_load_layer_v_proj_shape(shard_dir):
    weights = load_layer_weights(f"{shard_dir}/worker_0.safetensors", layer_idx=0)
    assert weights["self_attn.v_proj.weight"].shape == (256, 2048)

def test_load_layer_gate_proj_shape(shard_dir):
    weights = load_layer_weights(f"{shard_dir}/worker_0.safetensors", layer_idx=0)
    assert weights["mlp.gate_proj.weight"].shape == (5632, 2048)

def test_load_layer_strips_prefix(shard_dir):
    weights = load_layer_weights(f"{shard_dir}/worker_0.safetensors", layer_idx=0)
    assert not any(k.startswith("model.layers.") for k in weights)

def test_load_layer_raises_for_missing_layer(shard_dir):
    with pytest.raises(KeyError):
        load_layer_weights(f"{shard_dir}/worker_0.safetensors", layer_idx=99)

def test_load_layer_from_worker_1_returns_layer_1(shard_dir):
    weights = load_layer_weights(f"{shard_dir}/worker_1.safetensors", layer_idx=1)
    assert set(weights.keys()) == EXPECTED_KEYS
