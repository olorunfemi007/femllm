import threading
import torch
from src.femllm.layer_streamer import LayerStreamer
from src.femllm.forward import forward_layer, forward_layer_decode_batch, ModelConfig

BATCH_COLLECTION_WINDOW_SECONDS = 0.01


class Worker:
    def __init__(self, shard_path: str, layer_indices: list[int], config: ModelConfig, window_size: int = 1):
        self.config = config
        self.streamer = LayerStreamer(shard_path, layer_indices, window_size)
        self.kv_caches: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}

        self._cv = threading.Condition()
        self._pending: list[tuple[str, torch.Tensor, torch.Tensor, int]] = []
        self._results: dict[str, torch.Tensor] = {}
        self._stopped = False
        self._belt_error = None  # type: Exception | None
        self._belt_thread = threading.Thread(target=self._run_belt, daemon=True)
        self._belt_thread.start()

    def _run_belt(self) -> None:
        try:
            while True:
                with self._cv:
                    if self._stopped:
                        return
                    current_start = self.streamer.current_start_layer()
                    has_pending = any(sl == current_start for (_, _, _, sl) in self._pending)
                    if has_pending:
                        # give near-simultaneous stragglers a short window to join this round
                        self._cv.wait(timeout=BATCH_COLLECTION_WINDOW_SECONDS)
                    batch = [(rid, h, p) for (rid, h, p, sl) in self._pending if sl == current_start]
                    self._pending = [item for item in self._pending if item[3] != current_start]

                if batch:
                    batch_results = self._compute_chunk(batch, current_start)
                    with self._cv:
                        self._results.update(batch_results)
                        self._cv.notify_all()

                self.streamer.advance()
        except Exception as e:
            with self._cv:
                self._belt_error = e
                self._cv.notify_all()
            return

    def _compute_chunk(
        self, batch: list[tuple[str, torch.Tensor, torch.Tensor]], start_layer_idx: int
    ) -> dict[str, torch.Tensor]:
        chunk_weights = self.streamer.current_chunk()
        layer_order = sorted(chunk_weights.keys())
        results: dict[str, torch.Tensor] = {}

        decode_items = [item for item in batch if item[1].shape[1] == 1]
        prefill_items = [item for item in batch if item[1].shape[1] > 1]

        if decode_items:
            request_ids = [r[0] for r in decode_items]
            hidden_states = torch.cat([r[1] for r in decode_items], dim=0)
            position_ids = torch.tensor([r[2].item() for r in decode_items], dtype=torch.long)
            kv_caches = [self.kv_caches.setdefault(rid, {}) for rid in request_ids]

            for layer_idx in layer_order:
                hidden_states = forward_layer_decode_batch(
                    chunk_weights[layer_idx], hidden_states, position_ids, kv_caches, layer_idx, self.config
                )

            for i, request_id in enumerate(request_ids):
                results[request_id] = hidden_states[i:i + 1]

        groups: dict[tuple, list[tuple[str, torch.Tensor, torch.Tensor]]] = {}
        for request_id, hidden_states, position_ids in prefill_items:
            key = tuple(position_ids.tolist())
            groups.setdefault(key, []).append((request_id, hidden_states, position_ids))

        for group in groups.values():
            request_ids = [r[0] for r in group]
            batched_hidden = torch.cat([r[1] for r in group], dim=0)
            position_ids = group[0][2]

            merged_kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            for layer_idx in layer_order:
                existing = [self.kv_caches.setdefault(rid, {}).get(layer_idx) for rid in request_ids]
                has_cache = [e is not None for e in existing]
                assert all(has_cache) or not any(has_cache), (
                    f"grouped requests {request_ids} disagree on KV-cache presence at layer {layer_idx}"
                )
                if all(has_cache):
                    merged_kv_cache[layer_idx] = (
                        torch.cat([e[0] for e in existing], dim=0),
                        torch.cat([e[1] for e in existing], dim=0),
                    )

            for layer_idx in layer_order:
                batched_hidden = forward_layer(
                    chunk_weights[layer_idx], batched_hidden, position_ids, merged_kv_cache, layer_idx, self.config
                )

            for i, request_id in enumerate(request_ids):
                results[request_id] = batched_hidden[i:i + 1]
                for layer_idx, (ks, vs) in merged_kv_cache.items():
                    self.kv_caches[request_id][layer_idx] = (ks[i:i + 1], vs[i:i + 1])

        return results

    # Concurrent calls must use unique request_ids (the coordinator uses uuid4 per request).
    def submit(
        self, request_id: str, hidden_states: torch.Tensor, position_ids: torch.Tensor, start_layer_idx: int
    ) -> torch.Tensor:
        with self._cv:
            self._pending.append((request_id, hidden_states, position_ids, start_layer_idx))
            self._cv.notify_all()
            while request_id not in self._results and not self._stopped and self._belt_error is None:
                self._cv.wait()
            if request_id in self._results:
                return self._results.pop(request_id)
            if self._belt_error is not None:
                raise RuntimeError(
                    f"worker belt thread died: {self._belt_error}"
                ) from self._belt_error
            raise RuntimeError(f"worker closed while request {request_id!r} was pending")

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor, request_id: str, start_layer_idx: int
    ) -> torch.Tensor:
        return self.submit(request_id, hidden_states, position_ids, start_layer_idx)

    def reset(self, request_id: str) -> None:
        self.kv_caches.pop(request_id, None)

    def close(self) -> None:
        with self._cv:
            self._stopped = True
            self._cv.notify_all()
        self._belt_thread.join(timeout=1.0)
