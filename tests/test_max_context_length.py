# tests/test_max_context_length.py
"""
Requires 4 worker processes already running (see Task 9's start_workers.py).
Run: pytest tests/test_max_context_length.py -v -s
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
WORKER_ADDRESSES = [f"localhost:{p}" for p in [50051, 50052, 50053, 50054]]

def test_oversized_prompt_rejected():
    coord = Coordinator(
        model_dir="models/tinyllama",
        shard_dir="shards/tinyllama",
        worker_addresses=WORKER_ADDRESSES,
        config=TINYLLAMA_CONFIG,
        max_context_length=5,
    )
    long_prompt = "This prompt has way more than five tokens in it for sure"
    with pytest.raises(ValueError, match="exceeds max_context_length"):
        coord.generate(long_prompt, max_new_tokens=10)

def test_generation_stops_at_max_context_length():
    coord = Coordinator(
        model_dir="models/tinyllama",
        shard_dir="shards/tinyllama",
        worker_addresses=WORKER_ADDRESSES,
        config=TINYLLAMA_CONFIG,
        max_context_length=8,
    )
    input_ids = coord.tokenizer("Hello there", return_tensors="pt")["input_ids"][0]
    prompt_len = len(input_ids)
    output = coord.generate("Hello there", max_new_tokens=50)
    total_len = prompt_len + len(coord.tokenizer(output)["input_ids"])
    assert total_len <= 8 + 2  # allow for tokenizer special-token slack
