"""
Run this test manually:
  Terminal 1: python -m src.femllm.worker_server --shard shards/tinyllama/worker_0.safetensors --manifest shards/tinyllama/manifest.json --worker-id 0 --port 50051
  Terminal 2: python tests/test_worker_server_concurrent_manual.py
"""
import sys
sys.path.insert(0, 'src')
import threading
import torch
import grpc
import femllm_pb2
import femllm_pb2_grpc
from src.femllm.worker_server import tensor_to_bytes, bytes_to_tensor

channel = grpc.insecure_channel("localhost:50051")
stub = femllm_pb2_grpc.WorkerServiceStub(channel)

results = {}

def call(request_id):
    hidden = torch.randn(1, 3, 2048, dtype=torch.bfloat16)
    req = femllm_pb2.ForwardRequest(
        request_id=request_id,
        hidden_states=tensor_to_bytes(hidden),
        shape=list(hidden.shape),
        position_ids=[0, 1, 2],
        layer_idx=0,
    )
    resp = stub.Prefill(req)
    results[request_id] = bytes_to_tensor(resp.hidden_states, resp.shape, torch.bfloat16)

threads = [threading.Thread(target=call, args=(f"concurrent-{i}",)) for i in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()

assert len(results) == 5
for request_id, out in results.items():
    assert out.shape == (1, 3, 2048), f"{request_id}: unexpected shape {out.shape}"
print("PASS: 5 concurrent same-position requests all served correctly")
