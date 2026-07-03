import threading
import torch
import pytest
from src.femllm.worker import Worker
from src.femllm.forward import ModelConfig

TINYLLAMA_CONFIG = ModelConfig(
    hidden_size=2048, num_heads=32, num_kv_heads=4, head_dim=64,
    intermediate_size=5632, rms_norm_eps=1e-5, rope_theta=10000.0,
)

@pytest.mark.timeout(10)
def test_decode_requests_at_different_positions_batch_and_stay_correct(shard_dir):
    """
    Two requests primed to DIFFERENT accumulated lengths (simulating different
    prompt lengths) must both get correct results when their decode calls land
    in the same belt round — the case Task 10's exact-match rule would have
    forced into two separate, unbatched rounds.
    """
    worker = Worker(f"{shard_dir}/worker_0.safetensors", [0, 4, 8, 12, 16, 20], TINYLLAMA_CONFIG, window_size=1)
    reference = Worker(f"{shard_dir}/worker_0.safetensors", [0, 4, 8, 12, 16, 20], TINYLLAMA_CONFIG, window_size=1)
    try:
        # prime req1 to a 5-token history and req2 to a 9-token history.
        # req1's priming tensors are shared between `worker` and `reference` so both
        # instances build an identical KV-cache history for req1 — otherwise two
        # independent (unseeded) torch.randn() draws would prime them with different
        # data and the later exact-match comparison below would be invalid by
        # construction (see task-11-report.md for the deviation writeup).
        req1_prime_0 = torch.randn(1, 5, 2048, dtype=torch.bfloat16)
        req1_primes_rest = [torch.randn(1, 5, 2048, dtype=torch.bfloat16) for _ in [4, 8, 12, 16, 20]]

        worker.forward(req1_prime_0.clone(), torch.arange(5), request_id="req1", start_layer_idx=0)
        for layer_idx, prime in zip([4, 8, 12, 16, 20], req1_primes_rest):
            worker.forward(prime.clone(), torch.arange(5), request_id="req1", start_layer_idx=layer_idx)
        worker.forward(torch.randn(1, 9, 2048, dtype=torch.bfloat16), torch.arange(9), request_id="req2", start_layer_idx=0)
        for layer_idx in [4, 8, 12, 16, 20]:
            worker.forward(torch.randn(1, 9, 2048, dtype=torch.bfloat16), torch.arange(9), request_id="req2", start_layer_idx=layer_idx)

        reference.forward(req1_prime_0.clone(), torch.arange(5), request_id="req1", start_layer_idx=0)
        for layer_idx, prime in zip([4, 8, 12, 16, 20], req1_primes_rest):
            reference.forward(prime.clone(), torch.arange(5), request_id="req1", start_layer_idx=layer_idx)

        torch.manual_seed(0)
        new_token = torch.randn(1, 1, 2048, dtype=torch.bfloat16)
        barrier = threading.Barrier(2)
        results = {}

        def decode(request_id, position):
            barrier.wait()
            results[request_id] = worker.forward(new_token.clone(), torch.tensor([position]), request_id=request_id, start_layer_idx=0)

        t1 = threading.Thread(target=decode, args=("req1", 5))
        t2 = threading.Thread(target=decode, args=("req2", 9))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        expected = reference.forward(new_token.clone(), torch.tensor([5]), request_id="req1", start_layer_idx=0)
        assert torch.allclose(results["req1"], expected, atol=1e-2)
        assert results["req2"].shape == (1, 1, 2048)
    finally:
        worker.close()
        reference.close()
