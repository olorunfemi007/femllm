# src/femllm/worker_server.py
import sys
import json
import argparse
from concurrent import futures

sys.path.insert(0, "src")

import torch
import grpc
import femllm_pb2
import femllm_pb2_grpc

from src.femllm.worker import Worker
from src.femllm.forward import ModelConfig

TINYLLAMA_CONFIG = ModelConfig(
    hidden_size=2048, num_heads=32, num_kv_heads=4, head_dim=64,
    intermediate_size=5632, rms_norm_eps=1e-5, rope_theta=10000.0,
)


def tensor_to_bytes(t: torch.Tensor) -> bytes:
    # torch.Tensor.numpy() has no bfloat16 support (numpy has no native bfloat16
    # dtype); view as int16 first, symmetric with bytes_to_tensor's reverse view.
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return t.contiguous().numpy().tobytes()


def bytes_to_tensor(data: bytes, shape: list[int], dtype: torch.dtype) -> torch.Tensor:
    import numpy as np
    np_dtype = {torch.bfloat16: "uint16", torch.float32: "float32"}[dtype]
    arr = np.frombuffer(data, dtype=np_dtype).reshape(shape)
    t = torch.from_numpy(arr.copy())
    if dtype == torch.bfloat16:
        t = t.view(torch.bfloat16)
    return t


class WorkerServicer(femllm_pb2_grpc.WorkerServiceServicer):
    def __init__(self, worker: Worker):
        self.worker = worker

    def _run_forward(self, request) -> femllm_pb2.ForwardResponse:
        hidden = bytes_to_tensor(request.hidden_states, list(request.shape), torch.bfloat16)
        position_ids = torch.tensor(list(request.position_ids), dtype=torch.long)
        out = self.worker.forward(hidden, position_ids, request.request_id, request.layer_idx)
        return femllm_pb2.ForwardResponse(
            hidden_states=tensor_to_bytes(out),
            shape=list(out.shape),
        )

    def Prefill(self, request, context):
        return self._run_forward(request)

    def Decode(self, request, context):
        return self._run_forward(request)

    def Reset(self, request, context):
        self.worker.reset(request.request_id)
        return femllm_pb2.ResetResponse()


def serve(shard_path: str, layer_indices: list[int], config: ModelConfig, port: int, window_size: int = 1) -> None:
    worker = Worker(shard_path, layer_indices, config, window_size)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=32))
    femllm_pb2_grpc.add_WorkerServiceServicer_to_server(WorkerServicer(worker), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"Worker serving layers {layer_indices} (window_size={window_size}) on port {port}")
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    worker_entry = next(w for w in manifest["workers"] if w["id"] == args.worker_id)

    serve(args.shard, worker_entry["layer_indices"], TINYLLAMA_CONFIG, args.port, manifest["window_size"])
