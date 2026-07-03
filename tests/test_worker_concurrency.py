import threading
import time
import torch
import pytest
from src.femllm.worker import Worker
from src.femllm.forward import ModelConfig

TINYLLAMA_CONFIG = ModelConfig(
    hidden_size=2048, num_heads=32, num_kv_heads=4, head_dim=64,
    intermediate_size=5632, rms_norm_eps=1e-5, rope_theta=10000.0,
)

@pytest.mark.timeout(10)
def test_single_request_still_works_through_the_belt(shard_dir):
    worker = Worker(f"{shard_dir}/worker_0.safetensors", [0, 4, 8, 12, 16, 20], TINYLLAMA_CONFIG, window_size=1)
    try:
        hidden = torch.randn(1, 5, 2048, dtype=torch.bfloat16)
        out = worker.forward(hidden, torch.arange(5), request_id="req1", start_layer_idx=0)
        assert out.shape == (1, 5, 2048)
    finally:
        worker.close()

@pytest.mark.timeout(10)
def test_late_arrival_for_already_passed_chunk_waits_instead_of_crashing(shard_dir):
    """
    This is the exact bug this task fixes: a request for a chunk that is no longer
    resident (the belt already moved past it) must wait for the belt to lap back
    around, not raise an exception.
    """
    worker = Worker(f"{shard_dir}/worker_0.safetensors", [0, 4, 8, 12, 16, 20], TINYLLAMA_CONFIG, window_size=1)
    try:
        hidden = torch.randn(1, 3, 2048, dtype=torch.bfloat16)
        # drive one request through chunk 0 and then chunk 4 -- this proves the belt
        # has moved past chunk 0 (it's now at chunk 8 or later) by the time this returns
        worker.forward(hidden, torch.arange(3), request_id="early-req", start_layer_idx=0)
        worker.forward(hidden, torch.arange(3), request_id="early-req", start_layer_idx=4)

        # a NEW request now asks for chunk 0 again -- NOT the currently-resident chunk.
        # under the Task 6 implementation this would raise immediately. it must instead
        # wait for the belt to complete its lap (chunks 8, 12, 16, 20, then back to 0).
        late_hidden = torch.randn(1, 2, 2048, dtype=torch.bfloat16)
        result = worker.forward(late_hidden, torch.arange(2), request_id="late-req", start_layer_idx=0)
        assert result.shape == (1, 2, 2048)
    finally:
        worker.close()

@pytest.mark.timeout(10)
def test_concurrent_requests_produce_correct_isolated_results(shard_dir):
    worker = Worker(f"{shard_dir}/worker_0.safetensors", [0, 4, 8, 12, 16, 20], TINYLLAMA_CONFIG, window_size=1)
    reference = Worker(f"{shard_dir}/worker_0.safetensors", [0, 4, 8, 12, 16, 20], TINYLLAMA_CONFIG, window_size=1)
    try:
        torch.manual_seed(0)
        hidden = torch.randn(1, 3, 2048, dtype=torch.bfloat16)
        position_ids = torch.arange(3)
        results = {}
        barrier = threading.Barrier(2)

        def call(request_id):
            barrier.wait()
            results[request_id] = worker.forward(hidden.clone(), position_ids, request_id=request_id, start_layer_idx=0)

        threads = [threading.Thread(target=call, args=(rid,)) for rid in ["req1", "req2"]]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = reference.forward(hidden.clone(), position_ids, request_id="reference", start_layer_idx=0)
        assert torch.allclose(results["req1"], expected)
        assert torch.allclose(results["req2"], expected)
    finally:
        worker.close()
        reference.close()

@pytest.mark.timeout(10)
def test_concurrent_requests_have_isolated_kv_caches(shard_dir):
    worker = Worker(f"{shard_dir}/worker_0.safetensors", [0, 4, 8, 12, 16, 20], TINYLLAMA_CONFIG, window_size=1)
    try:
        h1 = torch.randn(1, 3, 2048, dtype=torch.bfloat16)
        h2 = torch.randn(1, 7, 2048, dtype=torch.bfloat16)
        barrier = threading.Barrier(2)

        def call(request_id, hidden, seq_len):
            barrier.wait()
            worker.forward(hidden, torch.arange(seq_len), request_id=request_id, start_layer_idx=0)

        t1 = threading.Thread(target=call, args=("req1", h1, 3))
        t2 = threading.Thread(target=call, args=("req2", h2, 7))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        k1, _ = worker.kv_caches["req1"][0]
        k2, _ = worker.kv_caches["req2"][0]
        assert k1.shape[2] == 3
        assert k2.shape[2] == 7
    finally:
        worker.close()

@pytest.mark.timeout(10)
def test_close_unblocks_waiting_submit(shard_dir):
    worker = Worker(f"{shard_dir}/worker_0.safetensors", [0, 4, 8, 12, 16, 20], TINYLLAMA_CONFIG, window_size=1)
    hidden = torch.randn(1, 3, 2048, dtype=torch.bfloat16)
    # move the belt past chunk 0 so a new chunk-0 request must wait a full lap
    worker.forward(hidden, torch.arange(3), request_id="warm", start_layer_idx=0)

    errors = {}

    def blocked_call():
        try:
            worker.forward(hidden, torch.arange(3), request_id="stuck", start_layer_idx=0)
        except RuntimeError as e:
            errors["stuck"] = e

    t = threading.Thread(target=blocked_call)
    t.start()
    time.sleep(0.2)  # let it enqueue and block
    worker.close()
    t.join(timeout=5)
    assert not t.is_alive(), "submit() still blocked after close()"
    # either it got served just before close, or it raised — both acceptable; hanging is not
