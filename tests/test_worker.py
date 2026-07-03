import torch
import pytest
from src.femllm.worker import Worker
from src.femllm.forward import ModelConfig

TINYLLAMA_CONFIG = ModelConfig(
    hidden_size=2048, num_heads=32, num_kv_heads=4, head_dim=64,
    intermediate_size=5632, rms_norm_eps=1e-5, rope_theta=10000.0,
)

def test_worker_forward_output_shape(shard_dir):
    worker = Worker(f"{shard_dir}/worker_0.safetensors", [0, 4, 8, 12, 16, 20], TINYLLAMA_CONFIG, window_size=1)
    hidden = torch.randn(1, 5, 2048, dtype=torch.bfloat16)
    out = worker.forward(hidden, torch.arange(5), request_id="req1", start_layer_idx=0)
    assert out.shape == (1, 5, 2048)

def test_worker_forward_creates_kv_cache_entry(shard_dir):
    worker = Worker(f"{shard_dir}/worker_0.safetensors", [0, 4, 8, 12, 16, 20], TINYLLAMA_CONFIG, window_size=1)
    hidden = torch.randn(1, 5, 2048, dtype=torch.bfloat16)
    worker.forward(hidden, torch.arange(5), request_id="req1", start_layer_idx=0)
    assert "req1" in worker.kv_caches
    assert 0 in worker.kv_caches["req1"]

def test_worker_forward_processes_full_chunk_window_2(shard_dir_w2):
    worker = Worker(f"{shard_dir_w2}/worker_0.safetensors", [0, 1, 8, 9, 16, 17], TINYLLAMA_CONFIG, window_size=2)
    hidden = torch.randn(1, 5, 2048, dtype=torch.bfloat16)
    worker.forward(hidden, torch.arange(5), request_id="req1", start_layer_idx=0)
    # both layers of the chunk should have populated KV cache entries
    assert 0 in worker.kv_caches["req1"]
    assert 1 in worker.kv_caches["req1"]

def test_worker_decode_extends_kv_cache(shard_dir):
    worker = Worker(f"{shard_dir}/worker_0.safetensors", [0, 4, 8, 12, 16, 20], TINYLLAMA_CONFIG, window_size=1)
    hidden = torch.randn(1, 5, 2048, dtype=torch.bfloat16)
    worker.forward(hidden, torch.arange(5), request_id="req1", start_layer_idx=0)
    # cycle through all intermediate layers to return to layer 0 for decode pass
    for layer_idx in [4, 8, 12, 16, 20]:
        worker.forward(torch.randn(1, 5, 2048, dtype=torch.bfloat16), torch.arange(5), request_id="req1", start_layer_idx=layer_idx)
    hidden = torch.randn(1, 1, 2048, dtype=torch.bfloat16)
    worker.forward(hidden, torch.tensor([5]), request_id="req1", start_layer_idx=0)
    k, _ = worker.kv_caches["req1"][0]
    assert k.shape[2] == 6

def test_worker_reset_clears_kv_cache(shard_dir):
    worker = Worker(f"{shard_dir}/worker_0.safetensors", [0, 4, 8, 12, 16, 20], TINYLLAMA_CONFIG, window_size=1)
    hidden = torch.randn(1, 5, 2048, dtype=torch.bfloat16)
    worker.forward(hidden, torch.arange(5), request_id="req1", start_layer_idx=0)
    worker.reset("req1")
    assert "req1" not in worker.kv_caches

def test_worker_multiple_requests_isolated(shard_dir):
    worker = Worker(f"{shard_dir}/worker_0.safetensors", [0, 4, 8, 12, 16, 20], TINYLLAMA_CONFIG, window_size=1)
    h1 = torch.randn(1, 3, 2048, dtype=torch.bfloat16)
    worker.forward(h1, torch.arange(3), request_id="req1", start_layer_idx=0)
    # req1 must fully lap the cycle back to layer 0 before req2 can also start at layer 0,
    # since a worker only ever has one chunk resident (Task 4's invariant)
    for layer_idx in [4, 8, 12, 16, 20]:
        worker.forward(torch.randn(1, 3, 2048, dtype=torch.bfloat16), torch.arange(3), request_id="req1", start_layer_idx=layer_idx)
    h2 = torch.randn(1, 7, 2048, dtype=torch.bfloat16)
    worker.forward(h2, torch.arange(7), request_id="req2", start_layer_idx=0)
    k1, _ = worker.kv_caches["req1"][0]
    k2, _ = worker.kv_caches["req2"][0]
    assert k1.shape[2] == 3
    assert k2.shape[2] == 7
