# tests/test_concurrent_end_to_end.py
"""
Requires 4 worker processes already running (see Task 9's start_workers.py).
Run: FEMLLM_E2E=1 pytest tests/test_concurrent_end_to_end.py -v -s
"""
import os
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.femllm.coordinator import Coordinator
from src.femllm.forward import ModelConfig

pytestmark = pytest.mark.skipif(os.environ.get("FEMLLM_E2E") != "1", reason="requires running workers (FEMLLM_E2E=1)")

TINYLLAMA_CONFIG = ModelConfig(
    hidden_size=2048, num_heads=32, num_kv_heads=4, head_dim=64,
    intermediate_size=5632, rms_norm_eps=1e-5, rope_theta=10000.0,
)
WORKER_ADDRESSES = [f"localhost:{p}" for p in [50051, 50052, 50053, 50054]]

def test_many_concurrent_requests_match_individual_baselines():
    coord = Coordinator(
        model_dir="models/tinyllama",
        shard_dir="shards/tinyllama",
        worker_addresses=WORKER_ADDRESSES,
        config=TINYLLAMA_CONFIG,
        num_users=3,
    )
    prompts = [
        "The capital of France is",
        "The capital of Japan is",
        "The capital of Italy is",
        "The capital of Germany is",
        "The capital of Spain is",
    ]

    tokenizer = AutoTokenizer.from_pretrained("models/tinyllama")
    model = AutoModelForCausalLM.from_pretrained("models/tinyllama", torch_dtype=torch.bfloat16)
    model.eval()

    baselines = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        baselines.append(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())

    results = coord.generate_concurrent(prompts, max_new_tokens=5)
    results = [r.strip() for r in results]

    assert results == baselines, f"Distributed: {results} | Baseline: {baselines}"
