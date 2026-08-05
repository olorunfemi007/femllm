# tests/test_end_to_end.py
"""
Requires 4 worker processes already running. Start them with:
  python scripts/start_workers.py shards/tinyllama models/tinyllama
Then run:
  pytest tests/test_end_to_end.py -v -s
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

def test_coordinator_generates_nonempty_text():
    coord = Coordinator(
        model_dir="models/tinyllama",
        shard_dir="shards/tinyllama",
        worker_addresses=WORKER_ADDRESSES,
        config=TINYLLAMA_CONFIG,
    )
    output = coord.generate("What is 2 + 2?", max_new_tokens=20)
    assert isinstance(output, str)
    assert len(output) > 0

def test_coordinator_output_matches_baseline():
    """Output tokens must exactly match a single-process HuggingFace baseline (greedy)."""
    coord = Coordinator(
        model_dir="models/tinyllama",
        shard_dir="shards/tinyllama",
        worker_addresses=WORKER_ADDRESSES,
        config=TINYLLAMA_CONFIG,
    )
    prompt = "The capital of France is"
    distributed_output = coord.generate(prompt, max_new_tokens=5)

    tokenizer = AutoTokenizer.from_pretrained("models/tinyllama")
    model = AutoModelForCausalLM.from_pretrained("models/tinyllama", torch_dtype=torch.bfloat16)
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    baseline = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    assert distributed_output.strip() == baseline.strip(), \
        f"Distributed: '{distributed_output}' | Baseline: '{baseline}'"
