# tests/test_num_users.py
"""
Requires 4 worker processes already running (see Task 9's start_workers.py).
Run: pytest tests/test_num_users.py -v -s
"""
import os
import time
import threading
import pytest
from src.femllm.coordinator import Coordinator
from src.femllm.forward import ModelConfig

pytestmark = pytest.mark.skipif(os.environ.get("FEMLLM_E2E") != "1", reason="requires running workers (FEMLLM_E2E=1)")

TINYLLAMA_CONFIG = ModelConfig(
    hidden_size=2048, num_heads=32, num_kv_heads=4, head_dim=64,
    intermediate_size=5632, rms_norm_eps=1e-5, rope_theta=10000.0,
)
WORKER_ADDRESSES = [f"localhost:{p}" for p in [50051, 50052, 50053, 50054]]

def test_num_users_caps_concurrent_active_generations():
    coord = Coordinator(
        model_dir="models/tinyllama",
        shard_dir="shards/tinyllama",
        worker_addresses=WORKER_ADDRESSES,
        config=TINYLLAMA_CONFIG,
        num_users=2,
    )
    active_count = 0
    max_observed = 0
    lock = threading.Lock()

    # NOTE: patch _generate_admitted (not generate) so we track concurrency
    # of ADMITTED generations only. Patching coord.generate would increment
    # the counter before the semaphore is acquired (generate_concurrent submits
    # the patched coord.generate), so max_observed would reflect queued-but-
    # not-yet-admitted calls (up to len(prompts)) rather than the admission cap.
    original = coord._generate_admitted

    def tracked(prompt, max_new_tokens):
        nonlocal active_count, max_observed
        with lock:
            active_count += 1
            max_observed = max(max_observed, active_count)
        try:
            return original(prompt, max_new_tokens)
        finally:
            with lock:
                active_count -= 1

    coord._generate_admitted = tracked

    prompts = ["Hello", "Goodbye", "Thank you", "Good morning"]
    results = coord.generate_concurrent(prompts, max_new_tokens=5)

    assert len(results) == 4
    assert max_observed <= 2
