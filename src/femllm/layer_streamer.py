import threading
import torch
from src.femllm.layer_loader import load_layer_weights


def _load_chunk(shard_path: str, layer_indices: list[int]) -> dict[int, dict[str, torch.Tensor]]:
    return {layer_idx: load_layer_weights(shard_path, layer_idx) for layer_idx in layer_indices}


class LayerStreamer:
    def __init__(self, shard_path: str, layer_indices: list[int], window_size: int = 1):
        self.shard_path = shard_path
        self.window_size = window_size
        self.chunks: list[list[int]] = [
            layer_indices[i:i + window_size] for i in range(0, len(layer_indices), window_size)
        ]
        self._position = 0
        self._resident = _load_chunk(shard_path, self.chunks[0])
        self._prefetch_thread, self._prefetch_holder = self._start_prefetch((self._position + 1) % len(self.chunks))

    def _start_prefetch(self, position: int):
        holder: dict[str, dict[int, dict[str, torch.Tensor]]] = {}

        def _background_load():
            holder["weights"] = _load_chunk(self.shard_path, self.chunks[position])

        thread = threading.Thread(target=_background_load, daemon=True)
        thread.start()
        return thread, holder

    def current_start_layer(self) -> int:
        return self.chunks[self._position][0]

    def current_chunk(self) -> dict[int, dict[str, torch.Tensor]]:
        return self._resident

    def advance(self) -> None:
        self._prefetch_thread.join()
        self._position = (self._position + 1) % len(self.chunks)
        self._resident = self._prefetch_holder["weights"]
        self._prefetch_thread, self._prefetch_holder = self._start_prefetch((self._position + 1) % len(self.chunks))
