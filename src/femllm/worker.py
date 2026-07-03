import torch
from src.femllm.layer_streamer import LayerStreamer
from src.femllm.forward import forward_layer, ModelConfig


class Worker:
    """
    This version assumes single-threaded, in-order calls (proven correct through Task 9's
    end-to-end test). Task 10 replaces forward()'s internals entirely to make concurrent,
    out-of-order calls safe — see that task for why this version isn't sufficient once
    multiple threads can call a worker at once.
    """

    def __init__(self, shard_path: str, layer_indices: list[int], config: ModelConfig, window_size: int = 1):
        self.config = config
        self.streamer = LayerStreamer(shard_path, layer_indices, window_size)
        self.kv_caches: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        request_id: str,
        start_layer_idx: int,
    ) -> torch.Tensor:
        assert start_layer_idx == self.streamer.current_start_layer(), (
            f"out-of-order call: requested chunk at layer {start_layer_idx}, "
            f"but the resident chunk starts at layer {self.streamer.current_start_layer()} "
            f"(expected for single-threaded, in-order use only — see Task 10)"
        )

        if request_id not in self.kv_caches:
            self.kv_caches[request_id] = {}
        kv_cache = self.kv_caches[request_id]

        chunk_weights = self.streamer.current_chunk()
        for layer_idx in sorted(chunk_weights.keys()):
            hidden_states = forward_layer(
                chunk_weights[layer_idx], hidden_states, position_ids, kv_cache, layer_idx, self.config
            )

        self.streamer.advance()
        return hidden_states

    def reset(self, request_id: str) -> None:
        self.kv_caches.pop(request_id, None)
