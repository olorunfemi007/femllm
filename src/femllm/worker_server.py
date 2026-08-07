# src/femllm/worker_server.py
import sys
import json
import argparse
import time
from concurrent import futures

sys.path.insert(0, "src")

import torch
import grpc
import femllm_pb2
import femllm_pb2_grpc

from src.femllm.worker import Worker
from src.femllm.forward import ModelConfig, load_model_config


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
    def __init__(self, worker: Worker, worker_id: int):
        self.worker = worker
        self.worker_id = worker_id

    def _run_forward(self, method: str, request) -> femllm_pb2.ForwardResponse:
        # Per-hop logging — without this there is no way to observe requests
        # actually reaching a worker or the round-robin routing in action;
        # kubectl logs on a worker showed nothing at all per-request before
        # this. Timed so a hang inside worker.forward() (e.g. stuck in the
        # belt's chunk queue) is visible as a start line with no matching
        # done line, not just silence.
        print(f"[worker {self.worker_id}] {method} request_id={request.request_id} layer_idx={request.layer_idx} shape={list(request.shape)}")
        start = time.monotonic()
        hidden = bytes_to_tensor(request.hidden_states, list(request.shape), torch.bfloat16)
        position_ids = torch.tensor(list(request.position_ids), dtype=torch.long)
        out = self.worker.forward(hidden, position_ids, request.request_id, request.layer_idx)
        elapsed = time.monotonic() - start
        print(f"[worker {self.worker_id}] {method} request_id={request.request_id} layer_idx={request.layer_idx} done in {elapsed:.3f}s")
        return femllm_pb2.ForwardResponse(
            hidden_states=tensor_to_bytes(out),
            shape=list(out.shape),
        )

    def Prefill(self, request, context):
        return self._run_forward("Prefill", request)

    def Decode(self, request, context):
        return self._run_forward("Decode", request)

    def Reset(self, request, context):
        print(f"[worker {self.worker_id}] Reset request_id={request.request_id}")
        self.worker.reset(request.request_id)
        return femllm_pb2.ResetResponse()


def serve(shard_path: str, layer_indices: list[int], config: ModelConfig, port: int, worker_id: int, window_size: int = 1) -> None:
    worker = Worker(shard_path, layer_indices, config, window_size)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=32))
    femllm_pb2_grpc.add_WorkerServiceServicer_to_server(WorkerServicer(worker, worker_id), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"Worker {worker_id} serving layers {layer_indices} (window_size={window_size}) on port {port}")
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    worker_entry = next(w for w in manifest["workers"] if w["id"] == args.worker_id)

    config = load_model_config(args.model_dir)
    serve(args.shard, worker_entry["layer_indices"], config, args.port, worker_id=args.worker_id, window_size=manifest["window_size"])
