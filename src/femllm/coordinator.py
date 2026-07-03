# src/femllm/coordinator.py
import sys
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "src")

import torch
import grpc
from safetensors import safe_open
from transformers import AutoTokenizer
import femllm_pb2
import femllm_pb2_grpc

from src.femllm.forward import rms_norm, ModelConfig
from src.femllm.worker_server import tensor_to_bytes, bytes_to_tensor


class Coordinator:
    def __init__(self, model_dir: str, shard_dir: str, worker_ports: list[int], config: ModelConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        with open(f"{shard_dir}/manifest.json") as f:
            self.manifest = json.load(f)
        self.window_size = self.manifest["window_size"]
        self.num_workers = self.manifest["num_workers"]
        self.num_layers = self.manifest["num_layers"]

        with safe_open(f"{shard_dir}/coordinator.safetensors", framework="pt", device="cpu") as f:
            self.embed_tokens = f.get_tensor("model.embed_tokens.weight").to(torch.bfloat16)
            self.norm_weight = f.get_tensor("model.norm.weight").to(torch.bfloat16)
            self.lm_head = f.get_tensor("lm_head.weight").to(torch.bfloat16)

        self.stubs = []
        for port in worker_ports:
            channel = grpc.insecure_channel(f"localhost:{port}")
            self.stubs.append(femllm_pb2_grpc.WorkerServiceStub(channel))

    def _make_request(self, request_id: str, hidden: torch.Tensor, position_ids: torch.Tensor, layer_idx: int) -> femllm_pb2.ForwardRequest:
        return femllm_pb2.ForwardRequest(
            request_id=request_id,
            hidden_states=tensor_to_bytes(hidden),
            shape=list(hidden.shape),
            position_ids=position_ids.tolist(),
            layer_idx=layer_idx,
        )

    def _pipeline(self, method: str, request_id: str, hidden: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        for start_layer_idx in range(0, self.num_layers, self.window_size):
            worker_id = (start_layer_idx // self.window_size) % self.num_workers
            stub = self.stubs[worker_id]
            req = self._make_request(request_id, hidden, position_ids, start_layer_idx)
            resp = getattr(stub, method)(req)
            hidden = bytes_to_tensor(resp.hidden_states, list(resp.shape), torch.bfloat16)
        return hidden

    def _logits(self, hidden: torch.Tensor) -> torch.Tensor:
        last = hidden[:, -1, :]
        last = rms_norm(last, self.norm_weight, self.config.rms_norm_eps)
        return last @ self.lm_head.T

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        request_id = str(uuid.uuid4())

        hidden = self.embed_tokens[input_ids].unsqueeze(0)
        position_ids = torch.arange(len(input_ids))
        hidden = self._pipeline("Prefill", request_id, hidden, position_ids)

        next_token = self._logits(hidden).argmax(dim=-1)
        generated = [next_token.item()]

        for step in range(max_new_tokens - 1):
            position = len(input_ids) + step
            hidden = self.embed_tokens[next_token].unsqueeze(0)
            position_ids = torch.tensor([position])
            hidden = self._pipeline("Decode", request_id, hidden, position_ids)
            next_token = self._logits(hidden).argmax(dim=-1)
            generated.append(next_token.item())
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        for stub in self.stubs:
            stub.Reset(femllm_pb2.ResetRequest(request_id=request_id))

        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def generate_concurrent(self, prompts: list[str], max_new_tokens: int = 50) -> list[str]:
        with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            futures = [pool.submit(self.generate, prompt, max_new_tokens) for prompt in prompts]
            return [f.result() for f in futures]
