"""
Run this test manually:
  Terminal 1: python -m src.femllm.worker_server --shard shards/tinyllama/worker_0.safetensors --manifest shards/tinyllama/manifest.json --worker-id 0 --port 50051
  Terminal 2: python tests/test_worker_server_manual.py
"""
import sys
sys.path.insert(0, 'src')
import torch
import grpc
import femllm_pb2
import femllm_pb2_grpc
from src.femllm.worker_server import tensor_to_bytes, bytes_to_tensor

channel = grpc.insecure_channel("localhost:50051")
stub = femllm_pb2_grpc.WorkerServiceStub(channel)

hidden = torch.randn(1, 5, 2048, dtype=torch.bfloat16)
req = femllm_pb2.ForwardRequest(
    request_id="manual-test",
    hidden_states=tensor_to_bytes(hidden),
    shape=list(hidden.shape),
    position_ids=list(range(5)),
    layer_idx=0,
)
resp = stub.Prefill(req)
out = bytes_to_tensor(resp.hidden_states, resp.shape, torch.bfloat16)
print("Output shape:", out.shape)
assert out.shape == (1, 5, 2048), f"Expected (1, 5, 2048) got {out.shape}"
print("PASS")
