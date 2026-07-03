# tests/test_coordinator_concurrent.py
"""
Requires 4 worker processes already running (see Task 9's start_workers.py).
Run: pytest tests/test_coordinator_concurrent.py -v -s
"""
import os
import pytest
from src.femllm.coordinator import Coordinator
from src.femllm.forward import ModelConfig

pytestmark = pytest.mark.skipif(os.environ.get("FEMLLM_E2E") != "1", reason="requires running workers (FEMLLM_E2E=1)")

TINYLLAMA_CONFIG = ModelConfig(
    hidden_size=2048, num_heads=32, num_kv_heads=4, head_dim=64,
    intermediate_size=5632, rms_norm_eps=1e-5, rope_theta=10000.0,
)
WORKER_PORTS = [50051, 50052, 50053, 50054]

def test_generate_concurrent_matches_sequential_generate():
    coord = Coordinator(
        model_dir="models/tinyllama",
        shard_dir="shards/tinyllama",
        worker_ports=WORKER_PORTS,
        config=TINYLLAMA_CONFIG,
    )
    prompts = ["The capital of France is", "The capital of Japan is", "Two plus two is"]

    sequential = [coord.generate(p, max_new_tokens=5) for p in prompts]
    concurrent = coord.generate_concurrent(prompts, max_new_tokens=5)

    assert concurrent == sequential

def test_generate_concurrent_returns_results_in_prompt_order():
    coord = Coordinator(
        model_dir="models/tinyllama",
        shard_dir="shards/tinyllama",
        worker_ports=WORKER_PORTS,
        config=TINYLLAMA_CONFIG,
    )
    prompts = ["Hello", "Goodbye", "Thank you"]
    results = coord.generate_concurrent(prompts, max_new_tokens=3)
    assert len(results) == 3
    assert all(isinstance(r, str) and len(r) > 0 for r in results)
