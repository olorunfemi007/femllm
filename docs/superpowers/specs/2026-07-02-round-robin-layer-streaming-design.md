# Round-Robin Layer Streaming Design

**Status:** Approved by user, pending translation into implementation plan.
**Supersedes:** The caching/sharding approach in `docs/superpowers/plans/2026-06-29-distributed-layer-streaming-inference.md` (Tasks 2, 4, 6, 7, 8, 9). Everything else in that plan (Tasks 1, 3, 5) is unaffected.

## Problem

The original plan assigns each worker a **contiguous** range of layers (worker 0: layers 0–5, worker 1: 6–11, etc.) and streams weights into a small LRU pool (`WeightCache`, `max_cached` parameter) with prefetch-during-compute for the *next layer in the same worker's range*.

This has a gap: a worker only gets one layer's worth of compute time as a prefetch window between its own layers, and gets essentially no advance warning before the *first* layer of its range is needed on each new request — there's no signal that tells it a request is coming until the request itself arrives. If disk I/O for a layer is slower than the compute time of the layer before it, the pipeline stalls waiting on disk, on every request, every token.

## Design

### Layer assignment: chunked round-robin, parameterized by `window_size`

Layers are grouped into consecutive chunks of `window_size` layers, and chunks (not individual layers) are assigned round-robin across workers:

```
chunk_idx(layer_idx) = layer_idx // window_size
worker_id(layer_idx) = chunk_idx(layer_idx) % num_workers
```

Example (TinyLlama, 22 layers, 4 workers, `window_size=1`):
- worker 0: `[0, 4, 8, 12, 16, 20]`
- worker 1: `[1, 5, 9, 13, 17, 21]`
- worker 2: `[2, 6, 10, 14, 18]`
- worker 3: `[3, 7, 11, 15, 19]`

Example (3 workers, `window_size=2`):
- worker 0: chunks `{0,1}, {6,7}, {12,13}, ...`
- worker 1: chunks `{2,3}, {8,9}, {14,15}, ...`
- worker 2: chunks `{4,5}, {10,11}, {16,17}, ...`

Each worker's chunk list is a **fixed cyclic sequence** — after finishing its last chunk, it wraps back to its first, forever, since layer weights are stateless across requests (only the KV cache is request-scoped).

`window_size` is a single dial spanning the whole design space. `window_size=1` gives the full single-layer-per-hop interleave — maximum hops, maximum prefetch slack, the primary "extreme memory constraint" story for this project. Larger values group more consecutive layers into each worker's turn, trading prefetch slack for fewer hops. At the extreme (`window_size = layers_per_worker`), this converges to the *original* contiguous-range design — the one with the stalling problem this design exists to fix. So bigger `window_size` is not strictly better; it's a real tradeoff, not a free upgrade.

Default remains `window_size=1`, a global setting applied uniformly to all workers (per earlier decision).

**Why round-robin at any window_size fixes the stall:** after worker 0 finishes a chunk, it doesn't need its next chunk again until every other worker has taken a full turn. That's a much larger and more reliable prefetch window than "the compute time of one prior layer" — and, at `window_size=1`, it applies uniformly to *every single layer* a worker owns, not just the first one in a range.

**Explicit cost:** hop count per token is `ceil(num_layers / window_size)` instead of a flat `num_layers`. For TinyLlama/4 workers: 22 hops at `window_size=1`, 11 at `window_size=2`, 6 at `window_size=4`. Each hop carries its own tensor serialize/deserialize overhead — this is a deliberate trade of network overhead for I/O-hiding reliability, tunable via `window_size` rather than fixed.

### Per-chunk mechanics

Replace `WeightCache` (LRU pool keyed by `max_cached`) with `LayerStreamer`, which operates on whole chunks as a unit:

- Each worker has its own fixed cyclic list of **chunks** (each `window_size` layers), not individual layers.
- **A chunk loads and evicts as a single batch.** All `window_size` layers of a chunk are loaded together *before* compute starts on any of them. Compute then proceeds straight through the chunk (layer *i*'s output feeds layer *i+1*, entirely in memory) with no further disk access needed until the chunk is done. Once the whole chunk's compute finishes, all `window_size` layers are evicted together. There's no intra-chunk prefetch step — once the batch is loaded, there's no more I/O left to hide until the next chunk.
- **Between chunks** (this worker's idle time while other workers take their turns): immediately after evicting the current chunk, the worker kicks off a background load for its *entire next chunk* (all `window_size` layers), using the round-robin idle window as slack. By the time this worker's turn comes around again, the next chunk should already be fully resident and ready — the same "keep the next-needed thing warm via idle time" principle as before, now applied to a whole batch instead of a single layer.
- **Resident memory at any instant is at most `window_size` layers** — this is the real memory/hop tradeoff `window_size` controls: `window_size=1` is the extreme-memory-constraint story (one layer resident, maximum hop count); larger values hold proportionally more memory resident per worker in exchange for fewer, larger hops.
- At startup, each worker synchronously loads its entire first chunk (all `window_size` layers) before serving, so the first request isn't a special case.

Because access order is fully deterministic (each worker's own fixed cyclic chunk list), there's no eviction *policy* to reason about the way an LRU cache needs one — the next eviction and the next prefetch target are always known in advance.

### Coordinator dispatch

`Coordinator._pipeline()` loops over chunk-starting layer indices (`0, window_size, 2*window_size, ...`), routing each to `stubs[(layer_idx // window_size) % num_workers]`. One RPC call per chunk — at `window_size=1` this is one call per layer, as originally designed.

### Proto change

`ForwardRequest` gains a `layer_idx` field (int32) carrying the *starting* layer of the chunk this call should process. The worker looks up its own chunk that starts there (from its manifest-assigned sequence) and processes all `window_size` layers in it sequentially — no need to enumerate the full layer list in the request.

### Manifest / shard format

`split_model.py`'s manifest changes from `start_layer`/`end_layer` ranges to a `layer_indices` list per worker (the flattened set of layers assigned via chunked round-robin), plus a top-level `window_size` field recording the chunk size the shards were built with — so the coordinator, every worker, and the splitter all derive identical chunk boundaries without hardcoding it in multiple places. Each `worker_{i}.safetensors` shard contains its assigned (possibly non-contiguous) layer set instead of a single contiguous block.

### What's unaffected

- `load_layer_weights` (Task 3): already loads one layer's weights by `layer_idx`; only *what* `split_model.py` puts in each shard file changes.
- `forward_layer` / `rms_norm` / KV cache (Task 5): already operate on one layer at a time, keyed by `layer_idx`. No change.
- Prefill vs. Decode as separate RPC methods: no behavioral difference between them today (both call the same internal forward logic); kept as-is for API clarity.
- Per-request KV cache isolation (`Worker.kv_caches[request_id]`): unaffected — still keyed by `layer_idx` within each worker's own owned set.

### Request batching (throughput): the conveyor-belt model

Naive concurrency conflicts with the memory story: if independent requests are free to sit at arbitrary pipeline positions, two requests could ask the same worker for two *different* chunks at once, forcing it to hold more than `window_size` layers resident. An admission cap (`max_in_flight ≤ num_workers`) avoids this but ties the system's whole concurrency ceiling to worker count — not the resource that should be setting it. The fix below decouples chunk residency from request demand, which removes *that specific, worker-count-tied ceiling*. It does not mean concurrency goes uncapped: `num_users`, introduced further down, is where the real, deliberate concurrency limit lives — sized to KV-cache memory budget rather than to `num_workers`.

**Each worker's chunk cycle runs on its own clock, never on request demand.** A worker moves through its fixed cyclic chunk sequence exactly as it always does — load a chunk, hold it resident, evict it, prefetch the next — regardless of how many requests exist or where they are. This is the same mechanism as the single-request design; nothing about *loading* changes.

**Requests queue for the chunk they need; the worker serves whoever's waiting when it gets there.** When a worker's current chunk is resident, it batches together *every* currently-queued request that needs that specific chunk — one, ten, a hundred, however many concurrent users happen to need it right now — into a single `forward_layer` call, returns results split back out by `request_id`, then evicts and moves on. A request needing a *different* chunk from the one currently up simply waits in that chunk's queue until the belt cycles back around to it.

This queue is worker-side application state, not a new wire protocol: the coordinator's RPC call for a given chunk is still a standard blocking unary gRPC call (no proto change, confirmed below) — it may just take longer to return if it has to wait for the worker's belt to reach that chunk. The coordinator issues these calls concurrently, one per active request-pipeline, so one request's wait never blocks another's progress.

This is the same structural pattern as elevator (SCAN) disk-arm scheduling: a resource sweeps a fixed path in a fixed order, servicing whatever's waiting at each stop, and anything that missed its stop waits for the next pass. It's a well-precedented way to get a batch size that scales with real demand out of a resource with a hard "only one thing resident at a time" constraint — batch size isn't tied to `num_workers` the way it would be under admission-capped lockstep.

**Why the memory invariant survives untouched, at any concurrency level up to `num_users`:** a worker never loads a chunk *because* a request asked for it — only because its own fixed cycle reached that position. No number of concurrent requests can force two chunks resident at once, because residency is entirely decoupled from demand. This is strictly better than the admission-cap approach for the actual goal: it removes the *worker-count-tied* ceiling while keeping exactly the same weight-memory guarantee — concurrency itself is still bounded, just by `num_users` instead.

**Latency tradeoff (accepted, not hidden):** a request's worst-case wait for a given hop is bounded by one full lap of that worker's own cycle — `num_chunks_per_worker − 1` other chunks ahead of it — not unbounded. In practice, with many concurrent users, demand naturally spreads across all chunk positions, so most chunks already have a waiting batch by the time the belt reaches them, keeping realized latency well under the worst case.

**Decode-step batching prioritized — correction: position alignment is not automatic.** Every decode call has `seq_len=1` regardless of prompt length, but that only means the *new* token is a plain batch-dimension concat — it does not mean the whole layer batches for free. Each request's *past* KV-cache length differs (different prompt lengths, different admission times), so requiring an exact position match before batching (as first drafted) means real staggered traffic rarely batches at all. The fix: split a layer into the parts that don't depend on per-request history — RMSNorm, Q/K/V/O projections, FFN, which dominate FLOPs and are what actually amortizes the chunk's disk-load cost — and batch those across *every* pending decode request regardless of position, while running attention as a per-row loop against each request's own untouched, natively-shaped KV cache (no padding, no masking, no change to how KV cache is stored). RoPE needs per-row position IDs instead of one shared 1-D tensor, since each row is genuinely at its own absolute position — a position-math detail, not a cache-management one. Prefill-step batching (variable prompt lengths) still needs padding plus an attention mask to batch at all, since even the projections require matching sequence length there — that stays out of scope, since decode steps dominate total hop count for any reasonably-long generation.

**`num_users`: bounding KV-cache memory, independently of the streaming cap.** `Worker.kv_caches` (Task 6) is keyed by `request_id` and grows with concurrent requests regardless of the belt policy above — that's a different resource than the layer weights this design constrains, and needs its own explicit cap. Add `num_users`, mirroring llama.cpp's `--parallel`/`n_parallel` (its number of concurrent sequence slots):

- The coordinator maintains `num_users` slots. A new request is admitted into the pipeline (submitted to chunk 0's queue) only if a slot is free; otherwise it waits in a FIFO admission queue.
- A slot frees when its request finishes (EOS or `max_new_tokens`) — the coordinator immediately calls `Reset(request_id)` on every worker (the RPC already defined in Task 7/8) to release that request's KV-cache entries, then admits the next queued request into the freed slot.
- This bounds total KV-cache memory to `num_users × (KV-cache size for one sequence at its current length)` — a predictable ceiling, independent of how many requests are waiting in the admission queue.

This is a genuinely separate axis from the belt/chunk-queue mechanism above: `window_size`/`num_workers` bound *weight* memory and are fully decoupled from request count (the belt absorbs concurrent demand at any level without needing its own count-based cap); `num_users` bounds *KV-cache* memory via a hard admission cap on concurrent sequences, and is the parameter that actually sets the system's overall concurrency ceiling. They're independently tunable — `num_users` can be set higher or lower than `num_workers` depending on which resource (weight streaming vs. KV-cache RAM) is the tighter budget. To be precise about the corrected claim: concurrency in this system is **bounded**, by `num_users` for how many sequences run at once and by `max_context_length` (below) for how large each one can grow — not unbounded anywhere. What the belt model removes is only the *narrower, incidental* ceiling that would otherwise come from tying concurrency to `num_workers`.

One difference from llama.cpp worth noting rather than silently matching: llama.cpp's `--parallel` pre-allocates a fixed-size KV-cache buffer per slot upfront (avoiding fragmentation, at the cost of reserving max-context memory even for short sequences). This design keeps Task 6's KV cache dynamically sized per request (`kv_caches[request_id]`, growing token-by-token) — `num_users` bounds slot *count* here, not a fixed memory reservation. Static pre-allocation is a reasonable future optimization, out of scope for this pass.

**Weights and KV cache have opposite eviction lifecycles — do not conflate them.** Weights are stateless and disposable: loaded fresh each chunk turn, evicted immediately after compute, identical for every request, so nothing is lost by discarding them. KV cache is stateful and cumulative: it's the model's memory of every prior token's attention keys/values for one specific sequence, and evicting it mid-generation is unrecoverable without recomputing the whole prefix. So `Worker.kv_caches[request_id][layer_idx]` persists and grows across a request's *entire* lifetime — every chunk turn, every decode step — and is only cleared, all at once, when `Reset(request_id)` fires on completion (the trigger already wired into `num_users` slot-freeing above). Weights evict every turn; KV cache evicts exactly once, in full, per request.

`num_users` bounds KV-cache memory across *concurrent* requests, but not how large a *single* request's KV cache grows within one long sequence — that's inherent to attention, not something this design constrains on its own. Closing that gap:

**`max_context_length`: capping a single sequence's KV-cache growth.** A global coordinator-level setting (analogous to llama.cpp's `--ctx-size`), independent of `num_users` and `window_size`:

- Enforced entirely on the coordinator side, since it's the only component tracking a request's accumulated sequence length (`prompt_tokens + tokens_generated_so_far`) — workers stay purely reactive and don't need to know the limit.
- If a submitted prompt's own token count already exceeds `max_context_length`, the request is rejected upfront with a clear error rather than starting prefill and failing later or silently truncating.
- During decoding, `max_context_length` becomes a third stopping condition alongside EOS and `max_new_tokens`: the coordinator checks accumulated length before each decode step and stops generation once the cap is hit, whichever of the three conditions comes first.
- Hitting this cap triggers the exact same completion path as any other stop: `Reset(request_id)` on every worker, freeing the `num_users` slot for the next queued request. No new mechanism needed there — just another way a sequence can finish.
- No proto change required — this is coordinator-side policy, not something workers enforce.

Combined with `num_users`, this gives a fully predictable worst-case KV-cache memory ceiling for the whole system: `num_users × max_context_length × (KV-cache bytes per token per layer) × (layers owned by that worker)`, per worker — the missing piece that made the earlier "weight memory bounded, KV memory not yet bounded" story incomplete.

**Coordinator becomes a scheduler, not a loop.** `Coordinator.generate()` (Task 9) changes from one blocking per-request loop to managing up to `num_users` concurrent request-pipelines: tracking each request's current chunk position, submitting it to that chunk's queue at the right worker, collecting results as batches return, and re-submitting each request to chunk 0's queue for its next decode step once it completes a full pass. `num_users` is the coordinator-side admission cap — the true concurrency ceiling; the queue-and-sweep behavior at each worker only guarantees that whatever population `num_users` admits doesn't also blow up weight memory. Summary correction: **concurrency here is bounded, controlled by `num_users` (KV-cache memory) and `max_context_length` (per-sequence growth) — not unbounded.**

### Why not just do what llama.cpp's RPC backend does

llama.cpp's RPC backend (`ggml-rpc`) looks superficially similar — worker processes each holding some layers, tensors flowing over the network — but solves a different problem. Its `rpc-server` receives tensor data **once** at load time and keeps it resident in RAM/VRAM for the whole session (it even offers a local cache specifically to *avoid* re-transferring tensors). It has no concept of a worker that can't hold its assigned weights simultaneously and must keep re-streaming them from disk — because it doesn't need one; it pools combined cluster memory rather than shrinking any single node's footprint. Its op-level graph scheduler is also more efficient in that regime (fewer, coarser network boundaries). None of that machinery helps with *this* project's actual goal — proving a node can run with far less memory than its assigned layers require — which is why round-robin assignment and window-based streaming are purpose-built here rather than borrowed.

## Plan tasks affected

| Task | Change |
|---|---|
| Task 2 (`split_model.py`) | Manifest: `layer_indices` list per worker instead of `start_layer`/`end_layer`, plus a top-level `window_size` field. Chunked round-robin assignment formula (`chunk_idx = layer_idx // window_size`, `worker_id = chunk_idx % num_workers`). Tests rewritten for new manifest shape. |
| Task 4 (`weight_cache.py`) | Renamed `layer_streamer.py`. `LayerStreamer` replaces `WeightCache`: loads/evicts one whole chunk (`window_size` layers) as a batch per turn, then background-loads the entire next chunk during idle time. No LRU, no generic `max_cached` pool. `window_size` param, default 1; at most `window_size` layers resident at any instant. |
| Task 6 (`worker.py`) | `Worker.forward()` runs one chunk (`window_size` consecutive layers, given the chunk's starting `layer_idx`) per call instead of looping its full assigned range. Maintains a per-chunk queue of pending requests; when the chunk is resident, batches every currently-queued request needing it into one `forward_layer` call, splits results back out by `request_id`. |
| Task 7 (`femllm.proto`) | `ForwardRequest` gains `layer_idx` field carrying the chunk's starting layer. |
| Task 8 (`worker_server.py`) | Passes `layer_idx` through to `Worker.forward()`; loads the worker's first chunk's first layer at startup; enqueues incoming calls against the worker's own chunk cycle rather than serving them immediately. |
| Task 9 (`coordinator.py`) | Rewritten from a single blocking per-request loop to a scheduler managing concurrent request-pipelines: submits each request to the appropriate chunk queue, collects batched results as they return, re-submits requests to chunk 0 for their next decode step. Weight-memory safety comes from the per-worker chunk cycle (no cap needed there). Adds a separate `num_users`-slot admission pool with a FIFO wait queue for requests beyond capacity, calling `Reset()` and admitting the next queued request whenever a slot's sequence finishes. Adds `max_context_length`: rejects prompts that already exceed it, and stops decoding (same `Reset()`-and-free-slot path) once accumulated length hits the cap, alongside the existing EOS/`max_new_tokens` stop conditions. |
| Tasks 1, 3, 5 | Unaffected. |
