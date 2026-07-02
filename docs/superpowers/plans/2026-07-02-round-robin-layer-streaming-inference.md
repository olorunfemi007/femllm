# Round-Robin Layer-Streaming LLM Inference Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a distributed LLM inference engine where each worker owns a chunked, round-robin-interleaved set of transformer layers, streams those weights from local storage into memory one whole chunk at a time (never more than `window_size` layers resident), uses round-robin idle time between turns to guarantee the next chunk is always ready before it's needed, and — once the core single-request pipeline is proven — serves concurrent requests via a conveyor-belt scheduling model whose weight-memory guarantee holds at any concurrency level, with overall concurrency itself deliberately bounded by an explicit `num_users` KV-cache cap and `max_context_length` per-sequence cap.

**Architecture:** The coordinator tokenizes input, embeds tokens, and drives each token's hidden states through a fixed sequence of chunk-hops (`0, window_size, 2*window_size, ...`), routing each hop to `stubs[(layer_idx // window_size) % num_workers]` via gRPC. Each worker's `LayerStreamer` cycles through its own fixed list of chunks forever — load whole chunk, serve it, evict it, prefetch the next chunk during idle time — completely decoupled from which or how many requests are asking. Later tasks add worker-side per-chunk batching and a coordinator-side concurrent scheduler so many requests can be in flight while every worker still only ever holds one chunk resident. Each worker independently maintains per-request KV cache, which (unlike weights) persists for a request's full lifetime and clears only on completion.

**Tech Stack:** Python 3.10+, PyTorch 2.x, safetensors, gRPC + protobuf, transformers (tokenizer only), pytest. Test model: TinyLlama-1.1B-Chat-v1.0 (22 layers, hidden_size=2048).

**Design reference:** `docs/superpowers/specs/2026-07-02-round-robin-layer-streaming-design.md` — read this first for the *why* behind every task below; this plan is the *how*.

## Global Constraints

- Python 3.10+ only — no walrus operators or newer syntax for broader compatibility
- PyTorch 2.x — use `F.scaled_dot_product_attention` (not manual attention)
- All tensors in bfloat16 during inference to match TinyLlama's native dtype
- TinyLlama config: num_layers=22, hidden_size=2048, num_heads=32, num_kv_heads=4, head_dim=64, intermediate_size=5632, vocab_size=32000, rms_norm_eps=1e-5, rope_theta=10000.0
- 4 workers for all tests, `window_size=1` unless a task explicitly tests otherwise
- Coordinator owns: `model.embed_tokens.weight`, `model.norm.weight`, `lm_head.weight`
- Greedy decoding only (argmax) — no sampling needed for this prototype
- Tests requiring model files use `models/tinyllama/` as the local path (downloaded in Task 1)
- Layer assignment formula throughout: `chunk_idx(layer_idx) = layer_idx // window_size`, `worker_id(layer_idx) = chunk_idx(layer_idx) % num_workers`

---

## File Structure

```
femllm/
├── proto/
│   └── femllm.proto                  # gRPC service + message definitions
├── src/
│   ├── femllm/
│   │   ├── __init__.py
│   │   ├── layer_loader.py           # load one layer's weights from a shard file
│   │   ├── layer_streamer.py         # chunk-based streaming loader with prefetch thread
│   │   ├── forward.py                # transformer layer forward pass (attention + FFN)
│   │   ├── kv_cache.py               # (concepts live in forward.py / worker.py; no separate file needed)
│   │   ├── worker.py                 # Worker class: owns a chunk cycle, runs forward pass, batches requests
│   │   ├── worker_server.py          # gRPC server wrapping Worker
│   │   └── coordinator.py            # Coordinator: embed → chunk pipeline → sample → concurrent scheduler
│   └── femllm_pb2.py                 # generated (do not edit)
│   └── femllm_pb2_grpc.py            # generated (do not edit)
├── tools/
│   └── split_model.py                # CLI: split HuggingFace model into per-worker shards, chunked round-robin
├── scripts/
│   └── start_workers.py              # launch N worker processes for local testing
├── tests/
│   ├── conftest.py                   # shared fixtures: tmp shards, model path
│   ├── test_split_model.py
│   ├── test_layer_loader.py
│   ├── test_layer_streamer.py
│   ├── test_forward.py
│   ├── test_worker.py
│   ├── test_worker_concurrency.py
│   ├── test_forward_decode_batch.py
│   ├── test_worker_decode_batching.py
│   ├── test_end_to_end.py
│   ├── test_coordinator_concurrent.py
│   ├── test_num_users.py
│   ├── test_max_context_length.py
│   └── test_concurrent_end_to_end.py
├── models/
│   └── tinyllama/                    # downloaded model (gitignored)
├── shards/                           # generated shards (gitignored)
├── requirements.txt
└── .gitignore
```

---

### Task 1: Project Setup and Model Download

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `src/femllm/__init__.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Produces: `models/tinyllama/` — HuggingFace model files on disk that all later tasks reference

- [ ] **Step 1: Create requirements.txt**

```
torch>=2.0.0
safetensors>=0.4.0
transformers>=4.35.0
grpcio>=1.60.0
grpcio-tools>=1.60.0
protobuf>=4.25.0
pytest>=7.4.0
pytest-timeout>=2.2.0
```

- [ ] **Step 2: Create .gitignore**

```
models/
shards/
src/femllm_pb2.py
src/femllm_pb2_grpc.py
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 3: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: all packages install without error

- [ ] **Step 4: Create src/femllm/__init__.py**

```python
```

(Empty file — marks the directory as a package.)

- [ ] **Step 5: Download TinyLlama**

Run:
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    local_dir='models/tinyllama',
    ignore_patterns=['*.bin', '*.msgpack', 'flax_model*', 'tf_model*']
)
"
```
Expected: `models/tinyllama/` contains `config.json`, `tokenizer.json`, and one or more `.safetensors` files.

- [ ] **Step 6: Verify model structure**

Run:
```bash
python -c "
from safetensors import safe_open
import os
files = [f for f in os.listdir('models/tinyllama') if f.endswith('.safetensors')]
print('Shard files:', files)
with safe_open(f'models/tinyllama/{files[0]}', framework='pt', device='cpu') as f:
    keys = list(f.keys())
print('Sample keys:', keys[:5])
print('Total tensors:', len(keys))
"
```
Expected output includes keys like `model.layers.0.self_attn.q_proj.weight` and `model.embed_tokens.weight`.

- [ ] **Step 7: Create tests/conftest.py**

```python
import pytest
import os

MODEL_DIR = "models/tinyllama"

@pytest.fixture(scope="session")
def model_dir():
    if not os.path.exists(MODEL_DIR):
        pytest.skip("TinyLlama not downloaded — run Task 1 setup first")
    return MODEL_DIR

@pytest.fixture(scope="session")
def shard_dir(tmp_path_factory, model_dir):
    from tools.split_model import split_model
    out = str(tmp_path_factory.mktemp("shards"))
    split_model(model_dir, out, num_workers=4, window_size=1)
    return out

@pytest.fixture(scope="session")
def shard_dir_w2(tmp_path_factory, model_dir):
    from tools.split_model import split_model
    out = str(tmp_path_factory.mktemp("shards_w2"))
    split_model(model_dir, out, num_workers=4, window_size=2)
    return out
```

- [ ] **Step 8: Commit**

```bash
git init
git add requirements.txt .gitignore src/ tests/conftest.py
git commit -m "feat: project setup and TinyLlama download"
```

---

### Task 2: Model Shard Splitter (chunked round-robin)

**Files:**
- Create: `tools/split_model.py`
- Create: `tests/test_split_model.py`

**Interfaces:**
- Consumes: `models/tinyllama/` — HuggingFace safetensors model files
- Produces: `split_model(model_dir: str, output_dir: str, num_workers: int, window_size: int = 1) -> None` — writes `coordinator.safetensors`, `worker_0.safetensors` … `worker_{n-1}.safetensors`, `manifest.json` to `output_dir`. Manifest shape:
  ```json
  {
    "num_workers": 4,
    "num_layers": 22,
    "window_size": 1,
    "workers": [
      {"id": 0, "layer_indices": [0, 4, 8, 12, 16, 20], "shard_file": "worker_0.safetensors"}
    ]
  }
  ```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_split_model.py
import json
import pytest
from safetensors import safe_open
from tools.split_model import split_model

def test_split_creates_coordinator_shard(tmp_path, model_dir):
    split_model(model_dir, str(tmp_path), num_workers=4, window_size=1)
    assert (tmp_path / "coordinator.safetensors").exists()

def test_split_creates_four_worker_shards(tmp_path, model_dir):
    split_model(model_dir, str(tmp_path), num_workers=4, window_size=1)
    for i in range(4):
        assert (tmp_path / f"worker_{i}.safetensors").exists()

def test_split_creates_manifest(tmp_path, model_dir):
    split_model(model_dir, str(tmp_path), num_workers=4, window_size=1)
    assert (tmp_path / "manifest.json").exists()

def test_manifest_records_window_size(tmp_path, model_dir):
    split_model(model_dir, str(tmp_path), num_workers=4, window_size=1)
    with open(tmp_path / "manifest.json") as f:
        manifest = json.load(f)
    assert manifest["window_size"] == 1
    assert manifest["num_workers"] == 4
    assert manifest["num_layers"] == 22

def test_manifest_layer_indices_round_robin_window_1(tmp_path, model_dir):
    split_model(model_dir, str(tmp_path), num_workers=4, window_size=1)
    with open(tmp_path / "manifest.json") as f:
        manifest = json.load(f)
    workers = {w["id"]: w["layer_indices"] for w in manifest["workers"]}
    assert workers[0] == [0, 4, 8, 12, 16, 20]
    assert workers[1] == [1, 5, 9, 13, 17, 21]
    assert workers[2] == [2, 6, 10, 14, 18]
    assert workers[3] == [3, 7, 11, 15, 19]

def test_manifest_layer_indices_round_robin_window_2(tmp_path, model_dir):
    split_model(model_dir, str(tmp_path), num_workers=4, window_size=2)
    with open(tmp_path / "manifest.json") as f:
        manifest = json.load(f)
    workers = {w["id"]: w["layer_indices"] for w in manifest["workers"]}
    # chunks of 2: [0,1] [2,3] [4,5] [6,7] ... assigned round-robin by chunk index
    assert workers[0] == [0, 1, 8, 9, 16, 17]
    assert workers[1] == [2, 3, 10, 11, 18, 19]
    assert workers[2] == [4, 5, 12, 13, 20, 21]
    assert workers[3] == [6, 7, 14, 15]

def test_every_layer_assigned_exactly_once(tmp_path, model_dir):
    split_model(model_dir, str(tmp_path), num_workers=4, window_size=1)
    with open(tmp_path / "manifest.json") as f:
        manifest = json.load(f)
    all_layers = []
    for w in manifest["workers"]:
        all_layers.extend(w["layer_indices"])
    assert sorted(all_layers) == list(range(22))

def test_coordinator_shard_has_embeddings(tmp_path, model_dir):
    split_model(model_dir, str(tmp_path), num_workers=4, window_size=1)
    with safe_open(str(tmp_path / "coordinator.safetensors"), framework="pt") as f:
        keys = list(f.keys())
    assert "model.embed_tokens.weight" in keys
    assert "model.norm.weight" in keys
    assert "lm_head.weight" in keys

def test_coordinator_shard_has_no_layers(tmp_path, model_dir):
    split_model(model_dir, str(tmp_path), num_workers=4, window_size=1)
    with safe_open(str(tmp_path / "coordinator.safetensors"), framework="pt") as f:
        keys = list(f.keys())
    assert not any("model.layers." in k for k in keys)

def test_worker_shard_contains_assigned_layers(tmp_path, model_dir):
    split_model(model_dir, str(tmp_path), num_workers=4, window_size=1)
    with safe_open(str(tmp_path / "worker_0.safetensors"), framework="pt") as f:
        keys = list(f.keys())
    layer_indices = {int(k.split(".")[2]) for k in keys if k.startswith("model.layers.")}
    assert layer_indices == {0, 4, 8, 12, 16, 20}

def test_worker_shard_contains_no_embeddings(tmp_path, model_dir):
    split_model(model_dir, str(tmp_path), num_workers=4, window_size=1)
    with safe_open(str(tmp_path / "worker_0.safetensors"), framework="pt") as f:
        keys = list(f.keys())
    assert "model.embed_tokens.weight" not in keys
    assert "lm_head.weight" not in keys
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_split_model.py -v`
Expected: `ModuleNotFoundError: No module named 'tools.split_model'`

- [ ] **Step 3: Implement tools/split_model.py**

```python
# tools/split_model.py
import json
import os
import sys
from safetensors import safe_open
from safetensors.torch import save_file


def split_model(model_dir: str, output_dir: str, num_workers: int, window_size: int = 1) -> None:
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    num_layers = config["num_hidden_layers"]

    tensors = {}
    for filename in sorted(os.listdir(model_dir)):
        if filename.endswith(".safetensors"):
            with safe_open(os.path.join(model_dir, filename), framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)

    os.makedirs(output_dir, exist_ok=True)

    coordinator_tensors = {k: v for k, v in tensors.items() if not k.startswith("model.layers.")}
    save_file(coordinator_tensors, os.path.join(output_dir, "coordinator.safetensors"))

    worker_layers: dict[int, list[int]] = {i: [] for i in range(num_workers)}
    for layer_idx in range(num_layers):
        chunk_idx = layer_idx // window_size
        worker_id = chunk_idx % num_workers
        worker_layers[worker_id].append(layer_idx)

    manifest = {
        "num_workers": num_workers,
        "num_layers": num_layers,
        "window_size": window_size,
        "workers": [],
    }

    for worker_idx in range(num_workers):
        layer_indices = worker_layers[worker_idx]
        worker_tensors = {
            k: v for k, v in tensors.items()
            if k.startswith("model.layers.") and int(k.split(".")[2]) in layer_indices
        }
        save_file(worker_tensors, os.path.join(output_dir, f"worker_{worker_idx}.safetensors"))

        manifest["workers"].append({
            "id": worker_idx,
            "layer_indices": layer_indices,
            "shard_file": f"worker_{worker_idx}.safetensors",
        })

    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    window_size = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    split_model(sys.argv[1], sys.argv[2], int(sys.argv[3]), window_size)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_split_model.py -v`
Expected: all 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tools/split_model.py tests/test_split_model.py
git commit -m "feat: chunked round-robin model shard splitter"
```

---

### Task 3: Layer Weight Loader

**Files:**
- Create: `src/femllm/layer_loader.py`
- Create: `tests/test_layer_loader.py`

**Interfaces:**
- Consumes: `shard_dir/worker_{i}.safetensors` produced by `split_model`
- Produces: `load_layer_weights(shard_path: str, layer_idx: int) -> dict[str, torch.Tensor]` — keys are local names like `self_attn.q_proj.weight` (prefix `model.layers.{N}.` stripped)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_layer_loader.py
import torch
import pytest
from src.femllm.layer_loader import load_layer_weights

EXPECTED_KEYS = {
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
}

def test_load_layer_returns_all_weight_keys(shard_dir):
    weights = load_layer_weights(f"{shard_dir}/worker_0.safetensors", layer_idx=0)
    assert set(weights.keys()) == EXPECTED_KEYS

def test_load_layer_q_proj_shape(shard_dir):
    weights = load_layer_weights(f"{shard_dir}/worker_0.safetensors", layer_idx=0)
    assert weights["self_attn.q_proj.weight"].shape == (2048, 2048)

def test_load_layer_k_proj_shape(shard_dir):
    weights = load_layer_weights(f"{shard_dir}/worker_0.safetensors", layer_idx=0)
    assert weights["self_attn.k_proj.weight"].shape == (256, 2048)

def test_load_layer_v_proj_shape(shard_dir):
    weights = load_layer_weights(f"{shard_dir}/worker_0.safetensors", layer_idx=0)
    assert weights["self_attn.v_proj.weight"].shape == (256, 2048)

def test_load_layer_gate_proj_shape(shard_dir):
    weights = load_layer_weights(f"{shard_dir}/worker_0.safetensors", layer_idx=0)
    assert weights["mlp.gate_proj.weight"].shape == (5632, 2048)

def test_load_layer_strips_prefix(shard_dir):
    weights = load_layer_weights(f"{shard_dir}/worker_0.safetensors", layer_idx=0)
    assert not any(k.startswith("model.layers.") for k in weights)

def test_load_layer_raises_for_missing_layer(shard_dir):
    with pytest.raises(KeyError):
        load_layer_weights(f"{shard_dir}/worker_0.safetensors", layer_idx=99)

def test_load_layer_from_worker_1_returns_layer_1(shard_dir):
    weights = load_layer_weights(f"{shard_dir}/worker_1.safetensors", layer_idx=1)
    assert set(weights.keys()) == EXPECTED_KEYS
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_layer_loader.py -v`
Expected: `ModuleNotFoundError: No module named 'src.femllm.layer_loader'`

- [ ] **Step 3: Implement src/femllm/layer_loader.py**

```python
# src/femllm/layer_loader.py
import torch
from safetensors import safe_open


def load_layer_weights(shard_path: str, layer_idx: int) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer_idx}."
    weights = {}
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(prefix):
                local_key = key[len(prefix):]
                weights[local_key] = f.get_tensor(key)
    if not weights:
        raise KeyError(f"No weights found for layer {layer_idx} in {shard_path}")
    return weights
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_layer_loader.py -v`
Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/femllm/layer_loader.py tests/test_layer_loader.py
git commit -m "feat: layer weight loader from safetensors shard"
```

---

### Task 4: Chunk-Based Layer Streamer

**Files:**
- Create: `src/femllm/layer_streamer.py`
- Create: `tests/test_layer_streamer.py`

**Interfaces:**
- Consumes: `load_layer_weights` from `src.femllm.layer_loader`
- Produces: `LayerStreamer(shard_path, layer_indices, window_size=1)` with methods:
  - `current_start_layer() -> int` — the starting layer index of the currently-resident chunk
  - `current_chunk() -> dict[int, dict[str, torch.Tensor]]` — `{layer_idx: weights}` for the currently-resident chunk; does not advance the cycle
  - `advance() -> None` — evicts the current chunk, promotes the already-prefetched next chunk to resident, kicks off prefetch for the chunk after that
  - `chunks: list[list[int]]` — precomputed chunk boundaries, e.g. `[[0], [4], [8], [12], [16], [20]]` for `window_size=1`

**Important:** this class assumes single-threaded access — it has no internal locking. It's only ever driven by one worker's dedicated "belt" thread (Task 10), never called directly by request-handling threads. This is a deliberate simplification: keeping "wait for your turn" logic out of `LayerStreamer` and entirely inside `Worker` (Task 10) avoids a race where a request for a chunk that isn't currently resident has nowhere correct to go.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_layer_streamer.py
import time
import torch
from src.femllm.layer_streamer import LayerStreamer

def test_chunks_computed_correctly_window_1(shard_dir):
    streamer = LayerStreamer(f"{shard_dir}/worker_0.safetensors", layer_indices=[0, 4, 8, 12, 16, 20], window_size=1)
    assert streamer.chunks == [[0], [4], [8], [12], [16], [20]]

def test_chunks_computed_correctly_window_2(shard_dir_w2):
    streamer = LayerStreamer(f"{shard_dir_w2}/worker_0.safetensors", layer_indices=[0, 1, 8, 9, 16, 17], window_size=2)
    assert streamer.chunks == [[0, 1], [8, 9], [16, 17]]

def test_current_start_layer_begins_at_first_chunk(shard_dir):
    streamer = LayerStreamer(f"{shard_dir}/worker_0.safetensors", layer_indices=[0, 4, 8, 12, 16, 20], window_size=1)
    assert streamer.current_start_layer() == 0

def test_current_chunk_returns_weights_for_every_layer_in_it(shard_dir_w2):
    streamer = LayerStreamer(f"{shard_dir_w2}/worker_0.safetensors", layer_indices=[0, 1, 8, 9, 16, 17], window_size=2)
    chunk = streamer.current_chunk()
    assert set(chunk.keys()) == {0, 1}
    assert "self_attn.q_proj.weight" in chunk[0]
    assert "self_attn.q_proj.weight" in chunk[1]

def test_advance_moves_to_next_chunk(shard_dir):
    streamer = LayerStreamer(f"{shard_dir}/worker_0.safetensors", layer_indices=[0, 4, 8, 12, 16, 20], window_size=1)
    streamer.advance()
    assert streamer.current_start_layer() == 4
    assert set(streamer.current_chunk().keys()) == {4}

def test_advance_wraps_around_cycle(shard_dir):
    streamer = LayerStreamer(f"{shard_dir}/worker_0.safetensors", layer_indices=[0, 4, 8, 12, 16, 20], window_size=1)
    for _ in range(6):
        streamer.advance()
    assert streamer.current_start_layer() == 0

def test_advance_reloads_consistently_on_next_lap(shard_dir):
    streamer = LayerStreamer(f"{shard_dir}/worker_0.safetensors", layer_indices=[0, 4, 8, 12, 16, 20], window_size=1)
    first_weight = streamer.current_chunk()[0]["self_attn.q_proj.weight"]
    for _ in range(6):
        streamer.advance()
    second_weight = streamer.current_chunk()[0]["self_attn.q_proj.weight"]
    assert torch.allclose(first_weight, second_weight)

def test_advance_next_chunk_ready_without_extra_wait(shard_dir):
    streamer = LayerStreamer(f"{shard_dir}/worker_0.safetensors", layer_indices=[0, 4, 8, 12, 16, 20], window_size=1)
    time.sleep(0.3)  # let background prefetch for the second chunk finish
    start = time.time()
    streamer.advance()
    elapsed = time.time() - start
    assert elapsed < 0.1  # should already be resident, not a fresh disk read
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_layer_streamer.py -v`
Expected: `ModuleNotFoundError: No module named 'src.femllm.layer_streamer'`

- [ ] **Step 3: Implement src/femllm/layer_streamer.py**

```python
# src/femllm/layer_streamer.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_layer_streamer.py -v`
Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/femllm/layer_streamer.py tests/test_layer_streamer.py
git commit -m "feat: chunk-based layer streamer with idle-time prefetch"
```

---

### Task 5: Transformer Layer Forward Pass

**Files:**
- Create: `src/femllm/forward.py`
- Create: `tests/test_forward.py`

**Interfaces:**
- Consumes: `dict[str, torch.Tensor]` from `LayerStreamer.current_chunk()[layer_idx]`
- Produces:
  - `forward_layer(weights, hidden_states, position_ids, kv_cache, layer_idx, config) -> torch.Tensor`
  - `rms_norm(x, weight, eps) -> torch.Tensor`
  - `ModelConfig` dataclass

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_forward.py
import torch
import pytest
from src.femllm.forward import forward_layer, rms_norm, ModelConfig
from src.femllm.layer_loader import load_layer_weights

TINYLLAMA_CONFIG = ModelConfig(
    hidden_size=2048,
    num_heads=32,
    num_kv_heads=4,
    head_dim=64,
    intermediate_size=5632,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
)

def test_rms_norm_output_shape():
    x = torch.randn(1, 10, 2048, dtype=torch.bfloat16)
    w = torch.ones(2048, dtype=torch.bfloat16)
    out = rms_norm(x, w)
    assert out.shape == (1, 10, 2048)

def test_rms_norm_unit_weight_near_unit_norm():
    x = torch.randn(1, 5, 2048, dtype=torch.bfloat16)
    w = torch.ones(2048, dtype=torch.bfloat16)
    out = rms_norm(x, w)
    norms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(norms, torch.ones_like(norms), atol=0.1)

def test_forward_layer_output_shape(shard_dir):
    weights = load_layer_weights(f"{shard_dir}/worker_0.safetensors", layer_idx=0)
    weights = {k: v.to(torch.bfloat16) for k, v in weights.items()}
    hidden_states = torch.randn(1, 5, 2048, dtype=torch.bfloat16)
    position_ids = torch.arange(5)
    kv_cache = {}
    out = forward_layer(weights, hidden_states, position_ids, kv_cache, layer_idx=0, config=TINYLLAMA_CONFIG)
    assert out.shape == (1, 5, 2048)

def test_forward_layer_populates_kv_cache(shard_dir):
    weights = load_layer_weights(f"{shard_dir}/worker_0.safetensors", layer_idx=0)
    weights = {k: v.to(torch.bfloat16) for k, v in weights.items()}
    hidden_states = torch.randn(1, 5, 2048, dtype=torch.bfloat16)
    position_ids = torch.arange(5)
    kv_cache = {}
    forward_layer(weights, hidden_states, position_ids, kv_cache, layer_idx=0, config=TINYLLAMA_CONFIG)
    assert 0 in kv_cache
    k, v = kv_cache[0]
    assert k.shape == (1, 4, 5, 64)  # [bsz, num_kv_heads, seq_len, head_dim]
    assert v.shape == (1, 4, 5, 64)

def test_forward_layer_decode_extends_kv_cache(shard_dir):
    weights = load_layer_weights(f"{shard_dir}/worker_0.safetensors", layer_idx=0)
    weights = {k: v.to(torch.bfloat16) for k, v in weights.items()}
    kv_cache = {}
    hidden = torch.randn(1, 5, 2048, dtype=torch.bfloat16)
    forward_layer(weights, hidden, torch.arange(5), kv_cache, layer_idx=0, config=TINYLLAMA_CONFIG)
    hidden = torch.randn(1, 1, 2048, dtype=torch.bfloat16)
    forward_layer(weights, hidden, torch.tensor([5]), kv_cache, layer_idx=0, config=TINYLLAMA_CONFIG)
    k, v = kv_cache[0]
    assert k.shape == (1, 4, 6, 64)  # 5 + 1

def test_forward_layer_dtype_preserved(shard_dir):
    weights = load_layer_weights(f"{shard_dir}/worker_0.safetensors", layer_idx=0)
    weights = {k: v.to(torch.bfloat16) for k, v in weights.items()}
    hidden_states = torch.randn(1, 3, 2048, dtype=torch.bfloat16)
    kv_cache = {}
    out = forward_layer(weights, hidden_states, torch.arange(3), kv_cache, layer_idx=0, config=TINYLLAMA_CONFIG)
    assert out.dtype == torch.bfloat16
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_forward.py -v`
Expected: `ModuleNotFoundError: No module named 'src.femllm.forward'`

- [ ] **Step 3: Implement src/femllm/forward.py**

```python
# src/femllm/forward.py
import math
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class ModelConfig:
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight * x).to(x.dtype)


def _build_rope(seq_len: int, head_dim: int, theta: float, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).to(dtype)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).to(dtype)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = position_ids[-1].item() + 1
    head_dim = q.shape[-1]
    cos, sin = _build_rope(seq_len, head_dim, theta, q.dtype)
    cos = cos[position_ids]  # [seq, head_dim]
    sin = sin[position_ids]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


def forward_layer(
    weights: dict[str, torch.Tensor],
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]],
    layer_idx: int,
    config: ModelConfig,
) -> torch.Tensor:
    bsz, seq_len, _ = hidden_states.shape

    residual = hidden_states
    hidden_states = rms_norm(hidden_states, weights["input_layernorm.weight"], config.rms_norm_eps)

    q = hidden_states @ weights["self_attn.q_proj.weight"].T
    k = hidden_states @ weights["self_attn.k_proj.weight"].T
    v = hidden_states @ weights["self_attn.v_proj.weight"].T

    q = q.view(bsz, seq_len, config.num_heads, config.head_dim).transpose(1, 2)
    k = k.view(bsz, seq_len, config.num_kv_heads, config.head_dim).transpose(1, 2)
    v = v.view(bsz, seq_len, config.num_kv_heads, config.head_dim).transpose(1, 2)

    q, k = _apply_rope(q, k, position_ids, config.rope_theta)

    if layer_idx in kv_cache:
        k_past, v_past = kv_cache[layer_idx]
        k = torch.cat([k_past, k], dim=2)
        v = torch.cat([v_past, v], dim=2)
    kv_cache[layer_idx] = (k, v)

    groups = config.num_heads // config.num_kv_heads
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)

    attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=(seq_len > 1))
    attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, config.num_heads * config.head_dim)
    attn_out = attn_out @ weights["self_attn.o_proj.weight"].T
    hidden_states = residual + attn_out

    residual = hidden_states
    hidden_states = rms_norm(hidden_states, weights["post_attention_layernorm.weight"], config.rms_norm_eps)
    gate = F.silu(hidden_states @ weights["mlp.gate_proj.weight"].T)
    up = hidden_states @ weights["mlp.up_proj.weight"].T
    hidden_states = (gate * up) @ weights["mlp.down_proj.weight"].T

    return residual + hidden_states
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_forward.py -v`
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/femllm/forward.py tests/test_forward.py
git commit -m "feat: transformer layer forward pass with GQA, RoPE, and KV cache"
```

---

### Task 6: Worker — Chunk-Scoped Forward Pass

**Files:**
- Create: `src/femllm/worker.py`
- Create: `tests/test_worker.py`

**Interfaces:**
- Consumes: `LayerStreamer`, `forward_layer`, `ModelConfig`
- Produces: `Worker(shard_path, layer_indices, config, window_size=1)` with methods:
  - `forward(hidden_states, position_ids, request_id, start_layer_idx) -> torch.Tensor` — runs every layer in the chunk starting at `start_layer_idx`, in order, for one request
  - `reset(request_id) -> None`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_worker.py
import torch
import pytest
from src.femllm.worker import Worker
from src.femllm.forward import ModelConfig

TINYLLAMA_CONFIG = ModelConfig(
    hidden_size=2048, num_heads=32, num_kv_heads=4, head_dim=64,
    intermediate_size=5632, rms_norm_eps=1e-5, rope_theta=10000.0,
)

def test_worker_forward_output_shape(shard_dir):
    worker = Worker(f"{shard_dir}/worker_0.safetensors", [0, 4, 8, 12, 16, 20], TINYLLAMA_CONFIG, window_size=1)
    hidden = torch.randn(1, 5, 2048, dtype=torch.bfloat16)
    out = worker.forward(hidden, torch.arange(5), request_id="req1", start_layer_idx=0)
    assert out.shape == (1, 5, 2048)

def test_worker_forward_creates_kv_cache_entry(shard_dir):
    worker = Worker(f"{shard_dir}/worker_0.safetensors", [0, 4, 8, 12, 16, 20], TINYLLAMA_CONFIG, window_size=1)
    hidden = torch.randn(1, 5, 2048, dtype=torch.bfloat16)
    worker.forward(hidden, torch.arange(5), request_id="req1", start_layer_idx=0)
    assert "req1" in worker.kv_caches
    assert 0 in worker.kv_caches["req1"]

def test_worker_forward_processes_full_chunk_window_2(shard_dir_w2):
    worker = Worker(f"{shard_dir_w2}/worker_0.safetensors", [0, 1, 8, 9, 16, 17], TINYLLAMA_CONFIG, window_size=2)
    hidden = torch.randn(1, 5, 2048, dtype=torch.bfloat16)
    worker.forward(hidden, torch.arange(5), request_id="req1", start_layer_idx=0)
    # both layers of the chunk should have populated KV cache entries
    assert 0 in worker.kv_caches["req1"]
    assert 1 in worker.kv_caches["req1"]

def test_worker_decode_extends_kv_cache(shard_dir):
    worker = Worker(f"{shard_dir}/worker_0.safetensors", [0, 4, 8, 12, 16, 20], TINYLLAMA_CONFIG, window_size=1)
    hidden = torch.randn(1, 5, 2048, dtype=torch.bfloat16)
    worker.forward(hidden, torch.arange(5), request_id="req1", start_layer_idx=0)
    hidden = torch.randn(1, 1, 2048, dtype=torch.bfloat16)
    worker.forward(hidden, torch.tensor([5]), request_id="req1", start_layer_idx=0)
    k, _ = worker.kv_caches["req1"][0]
    assert k.shape[2] == 6

def test_worker_reset_clears_kv_cache(shard_dir):
    worker = Worker(f"{shard_dir}/worker_0.safetensors", [0, 4, 8, 12, 16, 20], TINYLLAMA_CONFIG, window_size=1)
    hidden = torch.randn(1, 5, 2048, dtype=torch.bfloat16)
    worker.forward(hidden, torch.arange(5), request_id="req1", start_layer_idx=0)
    worker.reset("req1")
    assert "req1" not in worker.kv_caches

def test_worker_multiple_requests_isolated(shard_dir):
    worker = Worker(f"{shard_dir}/worker_0.safetensors", [0, 4, 8, 12, 16, 20], TINYLLAMA_CONFIG, window_size=1)
    h1 = torch.randn(1, 3, 2048, dtype=torch.bfloat16)
    worker.forward(h1, torch.arange(3), request_id="req1", start_layer_idx=0)
    # req1 must fully lap the cycle back to layer 0 before req2 can also start at layer 0,
    # since a worker only ever has one chunk resident (Task 4's invariant)
    for layer_idx in [4, 8, 12, 16, 20]:
        worker.forward(torch.randn(1, 3, 2048, dtype=torch.bfloat16), torch.arange(3), request_id="req1", start_layer_idx=layer_idx)
    h2 = torch.randn(1, 7, 2048, dtype=torch.bfloat16)
    worker.forward(h2, torch.arange(7), request_id="req2", start_layer_idx=0)
    k1, _ = worker.kv_caches["req1"][0]
    k2, _ = worker.kv_caches["req2"][0]
    assert k1.shape[2] == 3
    assert k2.shape[2] == 7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_worker.py -v`
Expected: `ModuleNotFoundError: No module named 'src.femllm.worker'`

- [ ] **Step 3: Implement src/femllm/worker.py**

```python
# src/femllm/worker.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_worker.py -v`
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/femllm/worker.py tests/test_worker.py
git commit -m "feat: worker runs one chunk per call via chunk-based layer streamer"
```

---

### Task 7: gRPC Protocol Definition

**Files:**
- Create: `proto/femllm.proto`
- Create: `src/femllm_pb2.py` (generated)
- Create: `src/femllm_pb2_grpc.py` (generated)

**Interfaces:**
- Produces: gRPC stubs importable as `import femllm_pb2` and `import femllm_pb2_grpc`
  - `WorkerServiceStub` with methods: `Prefill(ForwardRequest) -> ForwardResponse`, `Decode(ForwardRequest) -> ForwardResponse`, `Reset(ResetRequest) -> ResetResponse`
  - `ForwardRequest` fields: `request_id` (str), `hidden_states` (bytes), `shape` (repeated int32), `position_ids` (repeated int32), `layer_idx` (int32) — the starting layer of the chunk this call should process
  - `ForwardResponse` fields: `hidden_states` (bytes), `shape` (repeated int32)
  - `ResetRequest` fields: `request_id` (str)

- [ ] **Step 1: Write proto/femllm.proto**

```proto
syntax = "proto3";

package femllm;

service WorkerService {
  rpc Prefill(ForwardRequest) returns (ForwardResponse);
  rpc Decode(ForwardRequest) returns (ForwardResponse);
  rpc Reset(ResetRequest) returns (ResetResponse);
}

message ForwardRequest {
  string request_id = 1;
  bytes hidden_states = 2;
  repeated int32 shape = 3;
  repeated int32 position_ids = 4;
  int32 layer_idx = 5;
}

message ForwardResponse {
  bytes hidden_states = 1;
  repeated int32 shape = 2;
}

message ResetRequest {
  string request_id = 1;
}

message ResetResponse {}
```

- [ ] **Step 2: Generate gRPC stubs**

Run:
```bash
python -m grpc_tools.protoc \
  -I proto \
  --python_out=src \
  --grpc_python_out=src \
  proto/femllm.proto
```
Expected: `src/femllm_pb2.py` and `src/femllm_pb2_grpc.py` are created.

- [ ] **Step 3: Verify stubs import correctly**

Run:
```bash
python -c "
import sys; sys.path.insert(0, 'src')
import femllm_pb2, femllm_pb2_grpc
req = femllm_pb2.ForwardRequest(request_id='test', shape=[1,5,2048], layer_idx=0)
print('ForwardRequest ok:', req.request_id, req.layer_idx)
print('WorkerServiceStub ok:', femllm_pb2_grpc.WorkerServiceStub)
"
```
Expected: prints both lines without error.

- [ ] **Step 4: Commit**

```bash
git add proto/femllm.proto src/femllm_pb2.py src/femllm_pb2_grpc.py
git commit -m "feat: gRPC protocol with layer_idx for chunk-scoped calls"
```

---

### Task 8: Worker gRPC Server

**Files:**
- Create: `src/femllm/worker_server.py`

**Interfaces:**
- Consumes: `Worker`, `femllm_pb2`, `femllm_pb2_grpc`
- Produces: `serve(shard_path, layer_indices, config, port, window_size=1)` — starts blocking gRPC server on given port

Helper functions (also in `worker_server.py`, used by coordinator in Task 9):
- `tensor_to_bytes(t: torch.Tensor) -> bytes`
- `bytes_to_tensor(data: bytes, shape: list[int], dtype) -> torch.Tensor`

- [ ] **Step 1: Write a manual integration test (not pytest — requires running server)**

Create `tests/test_worker_server_manual.py`:

```python
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
```

- [ ] **Step 2: Implement src/femllm/worker_server.py**

```python
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
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
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
```

- [ ] **Step 3: Split shards and run manual test**

Run:
```bash
python tools/split_model.py models/tinyllama shards/tinyllama 4 1
```

Terminal 1:
```bash
python -m src.femllm.worker_server \
  --shard shards/tinyllama/worker_0.safetensors \
  --manifest shards/tinyllama/manifest.json \
  --worker-id 0 \
  --port 50051
```

Terminal 2:
```bash
python tests/test_worker_server_manual.py
```
Expected: `Output shape: torch.Size([1, 5, 2048])` and `PASS`

- [ ] **Step 4: Commit**

```bash
git add src/femllm/worker_server.py tests/test_worker_server_manual.py
git commit -m "feat: worker gRPC server reading layer assignment from manifest"
```

---

### Task 9: Coordinator — Single-Request End-to-End Token Generation

**Files:**
- Create: `src/femllm/coordinator.py`
- Create: `scripts/start_workers.py`
- Create: `tests/test_end_to_end.py`

**Interfaces:**
- Consumes: `WorkerServiceStub`, `tensor_to_bytes`, `bytes_to_tensor`, all workers running on known ports, `manifest.json`
- Produces: `Coordinator(model_dir, shard_dir, worker_ports, config)` with method:
  - `generate(prompt: str, max_new_tokens: int) -> str`

This task proves the redesigned chunk-based streaming pipeline produces identical output to a single-process HuggingFace baseline, for exactly one request at a time. Concurrency (Tasks 10-16) builds on top of this once it's proven correct.

- [ ] **Step 1: Write the end-to-end test**

```python
# tests/test_end_to_end.py
"""
Requires 4 worker processes already running. Start them with:
  python scripts/start_workers.py shards/tinyllama
Then run:
  pytest tests/test_end_to_end.py -v -s
"""
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.femllm.coordinator import Coordinator
from src.femllm.forward import ModelConfig

TINYLLAMA_CONFIG = ModelConfig(
    hidden_size=2048, num_heads=32, num_kv_heads=4, head_dim=64,
    intermediate_size=5632, rms_norm_eps=1e-5, rope_theta=10000.0,
)
WORKER_PORTS = [50051, 50052, 50053, 50054]

def test_coordinator_generates_nonempty_text():
    coord = Coordinator(
        model_dir="models/tinyllama",
        shard_dir="shards/tinyllama",
        worker_ports=WORKER_PORTS,
        config=TINYLLAMA_CONFIG,
    )
    output = coord.generate("What is 2 + 2?", max_new_tokens=20)
    assert isinstance(output, str)
    assert len(output) > 0

def test_coordinator_output_matches_baseline():
    """Output tokens must exactly match a single-process HuggingFace baseline (greedy)."""
    coord = Coordinator(
        model_dir="models/tinyllama",
        shard_dir="shards/tinyllama",
        worker_ports=WORKER_PORTS,
        config=TINYLLAMA_CONFIG,
    )
    prompt = "The capital of France is"
    distributed_output = coord.generate(prompt, max_new_tokens=5)

    tokenizer = AutoTokenizer.from_pretrained("models/tinyllama")
    model = AutoModelForCausalLM.from_pretrained("models/tinyllama", torch_dtype=torch.bfloat16)
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    baseline = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    assert distributed_output.strip() == baseline.strip(), \
        f"Distributed: '{distributed_output}' | Baseline: '{baseline}'"
```

- [ ] **Step 2: Run test to confirm it fails (no coordinator yet)**

Run: `pytest tests/test_end_to_end.py::test_coordinator_generates_nonempty_text -v`
Expected: `ModuleNotFoundError: No module named 'src.femllm.coordinator'`

- [ ] **Step 3: Implement src/femllm/coordinator.py**

```python
# src/femllm/coordinator.py
import sys
import json
import uuid
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
            hidden = self.embed_tokens[next_token].unsqueeze(0).unsqueeze(0)
            position_ids = torch.tensor([position])
            hidden = self._pipeline("Decode", request_id, hidden, position_ids)
            next_token = self._logits(hidden).argmax(dim=-1)
            generated.append(next_token.item())
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        for stub in self.stubs:
            stub.Reset(femllm_pb2.ResetRequest(request_id=request_id))

        return self.tokenizer.decode(generated, skip_special_tokens=True)
```

- [ ] **Step 4: Create scripts/start_workers.py**

```python
# scripts/start_workers.py
"""Launch N worker processes for local testing, reading layer assignment from manifest.json."""
import sys
import json
import subprocess


def main():
    shard_dir = sys.argv[1]
    base_port = 50051

    with open(f"{shard_dir}/manifest.json") as f:
        manifest = json.load(f)

    procs = []
    for worker in manifest["workers"]:
        port = base_port + worker["id"]
        cmd = [
            sys.executable, "-m", "src.femllm.worker_server",
            "--shard", f"{shard_dir}/{worker['shard_file']}",
            "--manifest", f"{shard_dir}/manifest.json",
            "--worker-id", str(worker["id"]),
            "--port", str(port),
        ]
        p = subprocess.Popen(cmd)
        procs.append(p)
        print(f"Started worker {worker['id']} (layers {worker['layer_indices']}) on port {port} PID={p.pid}")

    print(f"\n{len(procs)} workers running. Ctrl+C to stop all.")
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Split the model shards (if not done yet)**

Run:
```bash
python tools/split_model.py models/tinyllama shards/tinyllama 4 1
```
Expected: `shards/tinyllama/` contains `coordinator.safetensors`, `worker_0.safetensors` … `worker_3.safetensors`, `manifest.json`

- [ ] **Step 6: Start workers and run end-to-end tests**

Terminal 1:
```bash
python scripts/start_workers.py shards/tinyllama
```
Wait for all 4 "Worker serving" lines to appear.

Terminal 2:
```bash
pytest tests/test_end_to_end.py -v -s
```
Expected: `test_coordinator_generates_nonempty_text` PASS, `test_coordinator_output_matches_baseline` PASS.

If `test_coordinator_output_matches_baseline` fails, print both outputs side-by-side and trace the difference through single-chunk comparisons (compare hidden states after each `_pipeline` hop against the HuggingFace baseline's intermediate layer outputs) before assuming a bug elsewhere.

- [ ] **Step 7: Commit**

```bash
git add src/femllm/coordinator.py scripts/start_workers.py tests/test_end_to_end.py
git commit -m "feat: coordinator drives chunked round-robin pipeline, matches greedy baseline"
```

---

**Checkpoint: the core redesign is complete and proven correct for a single request.** Tasks 10-16 add concurrent request handling on top of this working foundation.

---

### Task 10: Thread-Safe Worker — Belt Thread and Request Queueing

**Files:**
- Modify: `src/femllm/worker.py`
- Create: `tests/test_worker_concurrency.py`

**Interfaces:**
- Consumes: `LayerStreamer.current_start_layer/current_chunk/advance` (Task 4), `forward_layer` (unchanged — `position_ids` stays 1-D)
- Produces: `Worker` gains a dedicated background "belt" thread and a thread-safe `submit()`/`forward()` entry point; `Worker.close()` stops the belt thread. `Worker.forward`'s external signature is unchanged from Task 6.

**Why this task exists, precisely:** Task 6's `Worker.forward()` only worked because every call was single-threaded and always arrived in the exact order the streamer's cycle expected. Under real concurrency, two requests calling the same worker can easily arrive with one "behind" where the streamer's cycle currently is — Task 6's implementation would hit its assert and crash instead of correctly waiting for the belt to lap back around to the chunk that request needs. The fix: only ONE thread — a dedicated belt thread owned by the `Worker`, started once at construction — is ever allowed to touch `LayerStreamer` or run `forward_layer`. It advances continuously on its own clock (never blocked on request demand, matching the design spec exactly), draining whatever's currently queued for the chunk it's at, computing that as one batch, and moving on. Every other thread (gRPC request handlers) only ever calls `submit()`, which enqueues its request and blocks on a condition variable until the belt serves it — regardless of how "out of sync" that request's timing was, it can only ever wait, never crash.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_worker_concurrency.py
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
def test_single_request_still_works_through_the_belt(shard_dir):
    worker = Worker(f"{shard_dir}/worker_0.safetensors", [0, 4, 8, 12, 16, 20], TINYLLAMA_CONFIG, window_size=1)
    try:
        hidden = torch.randn(1, 5, 2048, dtype=torch.bfloat16)
        out = worker.forward(hidden, torch.arange(5), request_id="req1", start_layer_idx=0)
        assert out.shape == (1, 5, 2048)
    finally:
        worker.close()

@pytest.mark.timeout(10)
def test_late_arrival_for_already_passed_chunk_waits_instead_of_crashing(shard_dir):
    """
    This is the exact bug this task fixes: a request for a chunk that is no longer
    resident (the belt already moved past it) must wait for the belt to lap back
    around, not raise an exception.
    """
    worker = Worker(f"{shard_dir}/worker_0.safetensors", [0, 4, 8, 12, 16, 20], TINYLLAMA_CONFIG, window_size=1)
    try:
        hidden = torch.randn(1, 3, 2048, dtype=torch.bfloat16)
        # drive one request through chunk 0 and then chunk 4 -- this proves the belt
        # has moved past chunk 0 (it's now at chunk 8 or later) by the time this returns
        worker.forward(hidden, torch.arange(3), request_id="early-req", start_layer_idx=0)
        worker.forward(hidden, torch.arange(3), request_id="early-req", start_layer_idx=4)

        # a NEW request now asks for chunk 0 again -- NOT the currently-resident chunk.
        # under the Task 6 implementation this would raise immediately. it must instead
        # wait for the belt to complete its lap (chunks 8, 12, 16, 20, then back to 0).
        late_hidden = torch.randn(1, 2, 2048, dtype=torch.bfloat16)
        result = worker.forward(late_hidden, torch.arange(2), request_id="late-req", start_layer_idx=0)
        assert result.shape == (1, 2, 2048)
    finally:
        worker.close()

@pytest.mark.timeout(10)
def test_concurrent_requests_produce_correct_isolated_results(shard_dir):
    worker = Worker(f"{shard_dir}/worker_0.safetensors", [0, 4, 8, 12, 16, 20], TINYLLAMA_CONFIG, window_size=1)
    reference = Worker(f"{shard_dir}/worker_0.safetensors", [0, 4, 8, 12, 16, 20], TINYLLAMA_CONFIG, window_size=1)
    try:
        torch.manual_seed(0)
        hidden = torch.randn(1, 3, 2048, dtype=torch.bfloat16)
        position_ids = torch.arange(3)
        results = {}
        barrier = threading.Barrier(2)

        def call(request_id):
            barrier.wait()
            results[request_id] = worker.forward(hidden.clone(), position_ids, request_id=request_id, start_layer_idx=0)

        threads = [threading.Thread(target=call, args=(rid,)) for rid in ["req1", "req2"]]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = reference.forward(hidden.clone(), position_ids, request_id="reference", start_layer_idx=0)
        assert torch.allclose(results["req1"], expected)
        assert torch.allclose(results["req2"], expected)
    finally:
        worker.close()
        reference.close()

@pytest.mark.timeout(10)
def test_concurrent_requests_have_isolated_kv_caches(shard_dir):
    worker = Worker(f"{shard_dir}/worker_0.safetensors", [0, 4, 8, 12, 16, 20], TINYLLAMA_CONFIG, window_size=1)
    try:
        h1 = torch.randn(1, 3, 2048, dtype=torch.bfloat16)
        h2 = torch.randn(1, 7, 2048, dtype=torch.bfloat16)
        barrier = threading.Barrier(2)

        def call(request_id, hidden, seq_len):
            barrier.wait()
            worker.forward(hidden, torch.arange(seq_len), request_id=request_id, start_layer_idx=0)

        t1 = threading.Thread(target=call, args=("req1", h1, 3))
        t2 = threading.Thread(target=call, args=("req2", h2, 7))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        k1, _ = worker.kv_caches["req1"][0]
        k2, _ = worker.kv_caches["req2"][0]
        assert k1.shape[2] == 3
        assert k2.shape[2] == 7
    finally:
        worker.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_worker_concurrency.py -v`
Expected: `AttributeError: 'Worker' object has no attribute 'close'` (or the late-arrival test hangs/crashes with the Task 6 assert — either way, failing)

- [ ] **Step 3: Replace Worker's implementation in src/femllm/worker.py**

```python
# src/femllm/worker.py
import threading
import torch
from src.femllm.layer_streamer import LayerStreamer
from src.femllm.forward import forward_layer, ModelConfig

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
        self._belt_thread = threading.Thread(target=self._run_belt, daemon=True)
        self._belt_thread.start()

    def _run_belt(self) -> None:
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

    def _compute_chunk(
        self, batch: list[tuple[str, torch.Tensor, torch.Tensor]], start_layer_idx: int
    ) -> dict[str, torch.Tensor]:
        chunk_weights = self.streamer.current_chunk()
        layer_order = sorted(chunk_weights.keys())
        results: dict[str, torch.Tensor] = {}

        groups: dict[tuple, list[tuple[str, torch.Tensor, torch.Tensor]]] = {}
        for request_id, hidden_states, position_ids in batch:
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

    def submit(
        self, request_id: str, hidden_states: torch.Tensor, position_ids: torch.Tensor, start_layer_idx: int
    ) -> torch.Tensor:
        with self._cv:
            self._pending.append((request_id, hidden_states, position_ids, start_layer_idx))
            self._cv.notify_all()
            while request_id not in self._results:
                self._cv.wait()
            return self._results.pop(request_id)

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_worker_concurrency.py -v`
Expected: all 4 tests PASS. Also re-run `pytest tests/test_worker.py -v` — Task 6's tests call the **old** `Worker` implementation's assert-based logic; since this step replaces that implementation, re-run them now to confirm they still pass against the new belt-based `Worker` (they should, since single-threaded in-order calls are exactly the case the belt handles trivially).

- [ ] **Step 5: Commit**

```bash
git add src/femllm/worker.py tests/test_worker_concurrency.py requirements.txt
git commit -m "fix: worker belt thread queues out-of-turn requests instead of crashing"
```

---

### Task 11: Flexible Decode Batching Across Positions

**Files:**
- Modify: `src/femllm/forward.py`
- Modify: `src/femllm/worker.py`
- Create: `tests/test_forward_decode_batch.py`
- Create: `tests/test_worker_decode_batching.py`

**Interfaces:**
- Consumes: `rms_norm`, `_rotate_half`, `_build_rope` from `src.femllm.forward` (Task 5, unmodified)
- Produces: `forward_layer_decode_batch(weights, hidden_states, position_ids, kv_caches, layer_idx, config) -> torch.Tensor` in `forward.py` — `hidden_states` is `[N, 1, hidden]`, `position_ids` is `[N]` (one absolute position per row), `kv_caches` is a `list[dict[int, tuple[Tensor, Tensor]]]`, one dict per row, mutated in place exactly like the existing per-request `kv_cache` argument to `forward_layer`. `Worker._compute_chunk` (Task 10) is rewritten to route decode calls (`hidden_states.shape[1] == 1`) through this regardless of position, and keep prefill calls (`shape[1] > 1`) on the existing exact-match path.

**Why this task exists:** Task 10's `_compute_chunk` only grouped requests that shared an identical `position_ids` tensor. For decode calls that means requiring the exact same accumulated sequence length across requests — something realistic staggered traffic (different prompt lengths, different admission times) essentially never produces, so the batching mechanism built in Task 10 rarely fires in practice. The fix: split a transformer layer into the parts that don't depend on per-request history (RMSNorm, Q/K/V/O projections, FFN — the parts that dominate FLOPs and are what amortizes the chunk's disk-load cost across requests) and batch those across *every* pending decode request regardless of position, while running attention as a per-row loop against each request's own untouched, natively-shaped KV cache. Nothing about how KV cache is stored or shaped changes — no padding, no masking, no reshaping stored state. The one real addition is RoPE needing a position *per row* instead of one shared value, since each request is genuinely at its own absolute position.

- [ ] **Step 1: Write the failing tests for forward_layer_decode_batch**

```python
# tests/test_forward_decode_batch.py
import copy
import torch
import pytest
from src.femllm.forward import forward_layer, forward_layer_decode_batch, ModelConfig
from src.femllm.layer_loader import load_layer_weights

TINYLLAMA_CONFIG = ModelConfig(
    hidden_size=2048, num_heads=32, num_kv_heads=4, head_dim=64,
    intermediate_size=5632, rms_norm_eps=1e-5, rope_theta=10000.0,
)

def _weights(shard_dir):
    weights = load_layer_weights(f"{shard_dir}/worker_0.safetensors", layer_idx=0)
    return {k: v.to(torch.bfloat16) for k, v in weights.items()}

def test_decode_batch_output_shape(shard_dir):
    weights = _weights(shard_dir)
    hidden = torch.randn(3, 1, 2048, dtype=torch.bfloat16)
    position_ids = torch.tensor([2, 5, 9])
    kv_caches = [{}, {}, {}]
    out = forward_layer_decode_batch(weights, hidden, position_ids, kv_caches, layer_idx=0, config=TINYLLAMA_CONFIG)
    assert out.shape == (3, 1, 2048)

def test_decode_batch_updates_each_rows_own_kv_cache(shard_dir):
    weights = _weights(shard_dir)
    hidden = torch.randn(2, 1, 2048, dtype=torch.bfloat16)
    position_ids = torch.tensor([0, 0])
    kv_caches = [{}, {}]
    forward_layer_decode_batch(weights, hidden, position_ids, kv_caches, layer_idx=0, config=TINYLLAMA_CONFIG)
    assert 0 in kv_caches[0]
    assert 0 in kv_caches[1]
    k0, _ = kv_caches[0][0]
    k1, _ = kv_caches[1][0]
    assert k0.shape == (1, 4, 1, 64)
    assert k1.shape == (1, 4, 1, 64)

def test_decode_batch_matches_individual_calls_at_different_positions(shard_dir):
    """
    The core correctness claim: two requests with DIFFERENT accumulated KV-cache
    lengths (simulating different prompt lengths) must batch together and produce
    results identical to running each one individually through forward_layer.
    """
    weights = _weights(shard_dir)

    kv_cache_a = {}
    forward_layer(weights, torch.randn(1, 5, 2048, dtype=torch.bfloat16), torch.arange(5), kv_cache_a, layer_idx=0, config=TINYLLAMA_CONFIG)
    kv_cache_b = {}
    forward_layer(weights, torch.randn(1, 9, 2048, dtype=torch.bfloat16), torch.arange(9), kv_cache_b, layer_idx=0, config=TINYLLAMA_CONFIG)

    torch.manual_seed(0)
    new_token_a = torch.randn(1, 1, 2048, dtype=torch.bfloat16)
    new_token_b = torch.randn(1, 1, 2048, dtype=torch.bfloat16)

    ref_cache_a = copy.deepcopy(kv_cache_a)
    ref_out_a = forward_layer(weights, new_token_a.clone(), torch.tensor([5]), ref_cache_a, layer_idx=0, config=TINYLLAMA_CONFIG)
    ref_cache_b = copy.deepcopy(kv_cache_b)
    ref_out_b = forward_layer(weights, new_token_b.clone(), torch.tensor([9]), ref_cache_b, layer_idx=0, config=TINYLLAMA_CONFIG)

    batch_hidden = torch.cat([new_token_a.clone(), new_token_b.clone()], dim=0)
    batch_positions = torch.tensor([5, 9])
    kv_caches = [kv_cache_a, kv_cache_b]
    batch_out = forward_layer_decode_batch(weights, batch_hidden, batch_positions, kv_caches, layer_idx=0, config=TINYLLAMA_CONFIG)

    assert torch.allclose(batch_out[0:1], ref_out_a, atol=1e-2)
    assert torch.allclose(batch_out[1:2], ref_out_b, atol=1e-2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_forward_decode_batch.py -v`
Expected: `ImportError: cannot import name 'forward_layer_decode_batch' from 'src.femllm.forward'`

- [ ] **Step 3: Add forward_layer_decode_batch to src/femllm/forward.py**

Add these two functions to `src/femllm/forward.py`, after `_apply_rope` and before `forward_layer`:

```python
def _apply_rope_per_row(
    q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    # q, k: [N, heads, 1, head_dim]; position_ids: [N], one absolute position per row
    max_pos = position_ids.max().item() + 1
    head_dim = q.shape[-1]
    cos, sin = _build_rope(max_pos, head_dim, theta, q.dtype)
    cos = cos[position_ids].unsqueeze(1).unsqueeze(1)  # [N, 1, 1, head_dim]
    sin = sin[position_ids].unsqueeze(1).unsqueeze(1)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


def forward_layer_decode_batch(
    weights: dict[str, torch.Tensor],
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    kv_caches: list[dict[int, tuple[torch.Tensor, torch.Tensor]]],
    layer_idx: int,
    config: ModelConfig,
) -> torch.Tensor:
    n = hidden_states.shape[0]

    residual = hidden_states
    hidden_states = rms_norm(hidden_states, weights["input_layernorm.weight"], config.rms_norm_eps)

    q = hidden_states @ weights["self_attn.q_proj.weight"].T
    k = hidden_states @ weights["self_attn.k_proj.weight"].T
    v = hidden_states @ weights["self_attn.v_proj.weight"].T

    q = q.view(n, 1, config.num_heads, config.head_dim).transpose(1, 2)
    k = k.view(n, 1, config.num_kv_heads, config.head_dim).transpose(1, 2)
    v = v.view(n, 1, config.num_kv_heads, config.head_dim).transpose(1, 2)

    q, k = _apply_rope_per_row(q, k, position_ids, config.rope_theta)

    groups = config.num_heads // config.num_kv_heads
    attn_outputs = []
    for i in range(n):
        k_i, v_i = k[i:i + 1], v[i:i + 1]
        if layer_idx in kv_caches[i]:
            k_past, v_past = kv_caches[i][layer_idx]
            k_i = torch.cat([k_past, k_i], dim=2)
            v_i = torch.cat([v_past, v_i], dim=2)
        kv_caches[i][layer_idx] = (k_i, v_i)

        k_rep = k_i.repeat_interleave(groups, dim=1)
        v_rep = v_i.repeat_interleave(groups, dim=1)
        attn_outputs.append(F.scaled_dot_product_attention(q[i:i + 1], k_rep, v_rep, is_causal=False))

    attn_out = torch.cat(attn_outputs, dim=0)
    attn_out = attn_out.transpose(1, 2).contiguous().view(n, 1, config.num_heads * config.head_dim)
    attn_out = attn_out @ weights["self_attn.o_proj.weight"].T
    hidden_states = residual + attn_out

    residual = hidden_states
    hidden_states = rms_norm(hidden_states, weights["post_attention_layernorm.weight"], config.rms_norm_eps)
    gate = F.silu(hidden_states @ weights["mlp.gate_proj.weight"].T)
    up = hidden_states @ weights["mlp.up_proj.weight"].T
    hidden_states = (gate * up) @ weights["mlp.down_proj.weight"].T

    return residual + hidden_states
```

`is_causal=False` here is correct, not a shortcut: each row has exactly one new token attending to its own full, unpadded history plus itself — there is no padding to mask and nothing later in the sequence to hide, so the built-in causal flag (which matters when there are multiple *new* query positions in a single call) doesn't apply.

- [ ] **Step 4: Run forward-level tests to verify they pass**

Run: `pytest tests/test_forward_decode_batch.py -v`
Expected: all 3 tests PASS

- [ ] **Step 5: Write the failing worker-level tests**

```python
# tests/test_worker_decode_batching.py
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
        # prime req1 to a 5-token history and req2 to a 9-token history
        worker.forward(torch.randn(1, 5, 2048, dtype=torch.bfloat16), torch.arange(5), request_id="req1", start_layer_idx=0)
        for layer_idx in [4, 8, 12, 16, 20]:
            worker.forward(torch.randn(1, 5, 2048, dtype=torch.bfloat16), torch.arange(5), request_id="req1", start_layer_idx=layer_idx)
        worker.forward(torch.randn(1, 9, 2048, dtype=torch.bfloat16), torch.arange(9), request_id="req2", start_layer_idx=0)
        for layer_idx in [4, 8, 12, 16, 20]:
            worker.forward(torch.randn(1, 9, 2048, dtype=torch.bfloat16), torch.arange(9), request_id="req2", start_layer_idx=layer_idx)

        reference.forward(torch.randn(1, 5, 2048, dtype=torch.bfloat16), torch.arange(5), request_id="req1", start_layer_idx=0)
        for layer_idx in [4, 8, 12, 16, 20]:
            reference.forward(torch.randn(1, 5, 2048, dtype=torch.bfloat16), torch.arange(5), request_id="req1", start_layer_idx=layer_idx)

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
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_worker_decode_batching.py -v`
Expected: FAIL — with Task 10's exact-match grouping, `req1` (position 5) and `req2` (position 9) never merge, so the test doesn't exercise the new code path yet and results reflect the old per-position grouping (the assertion on `results["req1"]` will fail once `_compute_chunk` is rewritten if the rewrite has a bug, which is exactly what this test is for).

- [ ] **Step 7: Rewrite _compute_chunk in src/femllm/worker.py**

Add the import and replace the `_compute_chunk` method:

```python
from src.femllm.forward import forward_layer, forward_layer_decode_batch, ModelConfig
```

```python
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
```

Prefill requests keep the exact-match grouping from Task 10 unchanged — batching them across different prompt lengths would need the same padding-and-masking machinery this task deliberately avoided, and that stays out of scope (per the design spec).

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_worker_decode_batching.py tests/test_worker_concurrency.py tests/test_worker.py -v`
Expected: all PASS — the new decode-batch path handles different-position decode requests correctly, and every earlier worker test (single-threaded, thread-safety) still passes unmodified since `forward_layer` itself was never touched.

- [ ] **Step 9: Commit**

```bash
git add src/femllm/forward.py src/femllm/worker.py tests/test_forward_decode_batch.py tests/test_worker_decode_batching.py
git commit -m "feat: batch decode requests regardless of position via split projection/attention path"
```

---

### Task 12: gRPC Server Uses the Thread-Safe Worker

**Files:**
- Modify: `src/femllm/worker_server.py`
- Create: `tests/test_worker_server_concurrent_manual.py`

**Interfaces:**
- Consumes: `Worker.forward` (Task 10 — already thread-safe, no batching logic needed at this layer)
- Produces: `WorkerServicer` unchanged in structure from Task 8, just backed by the new `Worker`

Since Task 10 moved all the queueing and batching logic inside `Worker` itself, the gRPC servicer needs **no changes to its control flow** — every concurrent RPC thread just calls `self.worker.forward(...)` exactly as in Task 8, and `Worker.submit()` correctly enqueues it and blocks until the belt thread serves it (possibly batched with other concurrently-arrived requests, possibly alone, possibly after waiting a full lap — the servicer doesn't need to know which). This is worth calling out explicitly: pushing the concurrency-safety into `Worker` made the transport layer *simpler*, not more complex.

- [ ] **Step 1: Write the manual concurrency test**

```python
# tests/test_worker_server_concurrent_manual.py
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
```

- [ ] **Step 2: Update src/femllm/worker_server.py**

Only two changes from Task 8's version: bump `ThreadPoolExecutor(max_workers=32)` (up from 4, since the server now needs enough threads to hold multiple concurrent RPC calls open at once while `Worker.submit()` blocks them), and confirm `_run_forward` calls `self.worker.forward(...)` unchanged. Full file:

```python
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
```

- [ ] **Step 3: Run the existing single-request manual test to confirm no regression**

Terminal 1:
```bash
python -m src.femllm.worker_server --shard shards/tinyllama/worker_0.safetensors --manifest shards/tinyllama/manifest.json --worker-id 0 --port 50051
```

Terminal 2:
```bash
python tests/test_worker_server_manual.py
```
Expected: `Output shape: torch.Size([1, 5, 2048])` and `PASS` (same as Task 8, now backed by the belt-based `Worker`).

- [ ] **Step 4: Run the new concurrent manual test**

Terminal 2 (same server still running):
```bash
python tests/test_worker_server_concurrent_manual.py
```
Expected: `PASS: 5 concurrent same-position requests all served correctly`

- [ ] **Step 5: Commit**

```bash
git add src/femllm/worker_server.py tests/test_worker_server_concurrent_manual.py
git commit -m "feat: gRPC server relies on thread-safe worker, no transport-layer batching needed"
```

---

### Task 13: Coordinator Concurrent Request Handling

**Files:**
- Modify: `src/femllm/coordinator.py`
- Create: `tests/test_coordinator_concurrent.py`

**Interfaces:**
- Consumes: `Coordinator.generate` (Task 9, unchanged)
- Produces: `Coordinator.generate_concurrent(prompts: list[str], max_new_tokens: int) -> list[str]`

No new scheduler is needed: each request's own pipeline is already just a sequence of blocking gRPC calls (Task 9), and Python releases the GIL during both network I/O and tensor compute, so running several `generate()` calls in their own threads naturally produces concurrent demand at every worker. Task 4's `LayerStreamer` guarantees each worker only ever has one chunk resident regardless of how many threads call into it, and Task 10's belt thread is what turns concurrent same-chunk calls into a batch — the coordinator doesn't need to know any of that is happening.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coordinator_concurrent.py
"""
Requires 4 worker processes already running (see Task 9's start_workers.py).
Run: pytest tests/test_coordinator_concurrent.py -v -s
"""
import pytest
from src.femllm.coordinator import Coordinator
from src.femllm.forward import ModelConfig

TINYLLAMA_CONFIG = ModelConfig(
    hidden_size=2048, num_heads=32, num_kv_heads=4, head_dim=64,
    intermediate_size=5632, rms_norm_eps=1e-5, rope_theta=10000.0,
)
WORKER_PORTS = [50051, 50052, 50053, 50054]

def test_generate_concurrent_matches_sequential_generate():
    coord = Coordinator(
        model_dir="models/tinyllama",
        shard_dir="shards/tinyllama",
        worker_ports=WORKER_PORTS,
        config=TINYLLAMA_CONFIG,
    )
    prompts = ["The capital of France is", "The capital of Japan is", "Two plus two is"]

    sequential = [coord.generate(p, max_new_tokens=5) for p in prompts]
    concurrent = coord.generate_concurrent(prompts, max_new_tokens=5)

    assert concurrent == sequential

def test_generate_concurrent_returns_results_in_prompt_order():
    coord = Coordinator(
        model_dir="models/tinyllama",
        shard_dir="shards/tinyllama",
        worker_ports=WORKER_PORTS,
        config=TINYLLAMA_CONFIG,
    )
    prompts = ["Hello", "Goodbye", "Thank you"]
    results = coord.generate_concurrent(prompts, max_new_tokens=3)
    assert len(results) == 3
    assert all(isinstance(r, str) and len(r) > 0 for r in results)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_coordinator_concurrent.py -v`
Expected: `AttributeError: 'Coordinator' object has no attribute 'generate_concurrent'`

- [ ] **Step 3: Add generate_concurrent to src/femllm/coordinator.py**

Add this method to the `Coordinator` class, and add `from concurrent.futures import ThreadPoolExecutor` to the imports at the top of the file:

```python
    def generate_concurrent(self, prompts: list[str], max_new_tokens: int = 50) -> list[str]:
        with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            futures = [pool.submit(self.generate, prompt, max_new_tokens) for prompt in prompts]
            return [f.result() for f in futures]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_coordinator_concurrent.py -v -s`
Expected: both tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/femllm/coordinator.py tests/test_coordinator_concurrent.py
git commit -m "feat: coordinator runs concurrent requests via thread pool"
```

---

### Task 14: `num_users` — Bounding KV-Cache Memory

**Files:**
- Modify: `src/femllm/coordinator.py`
- Create: `tests/test_num_users.py`

**Interfaces:**
- Modifies: `Coordinator.__init__(model_dir, shard_dir, worker_ports, config, num_users=4)` — new `num_users` parameter
- Produces: an admission gate around `generate()` so at most `num_users` requests are actively generating at once; excess requests block until a slot frees

- [ ] **Step 1: Write the failing test**

```python
# tests/test_num_users.py
"""
Requires 4 worker processes already running (see Task 9's start_workers.py).
Run: pytest tests/test_num_users.py -v -s
"""
import time
import threading
import pytest
from src.femllm.coordinator import Coordinator
from src.femllm.forward import ModelConfig

TINYLLAMA_CONFIG = ModelConfig(
    hidden_size=2048, num_heads=32, num_kv_heads=4, head_dim=64,
    intermediate_size=5632, rms_norm_eps=1e-5, rope_theta=10000.0,
)
WORKER_PORTS = [50051, 50052, 50053, 50054]

def test_num_users_caps_concurrent_active_generations():
    coord = Coordinator(
        model_dir="models/tinyllama",
        shard_dir="shards/tinyllama",
        worker_ports=WORKER_PORTS,
        config=TINYLLAMA_CONFIG,
        num_users=2,
    )
    active_count = 0
    max_observed = 0
    lock = threading.Lock()

    original_generate = coord.generate

    def tracked_generate(prompt, max_new_tokens):
        nonlocal active_count, max_observed
        with lock:
            active_count += 1
            max_observed = max(max_observed, active_count)
        try:
            return original_generate(prompt, max_new_tokens)
        finally:
            with lock:
                active_count -= 1

    coord.generate = tracked_generate

    prompts = ["Hello", "Goodbye", "Thank you", "Good morning"]
    results = coord.generate_concurrent(prompts, max_new_tokens=5)

    assert len(results) == 4
    assert max_observed <= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_num_users.py -v`
Expected: `TypeError: Coordinator.__init__() got an unexpected keyword argument 'num_users'`

- [ ] **Step 3: Add num_users admission gate to src/femllm/coordinator.py**

Modify `Coordinator.__init__` to accept and store `num_users`, and add a semaphore. Also add `import threading` at the top of the file:

```python
    def __init__(self, model_dir: str, shard_dir: str, worker_ports: list[int], config: ModelConfig, num_users: int = 4):
        self.config = config
        self.num_users = num_users
        self._admission = threading.Semaphore(num_users)
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
```

Wrap the body of `generate()` with the admission semaphore — replace the existing method with:

```python
    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        with self._admission:
            return self._generate_admitted(prompt, max_new_tokens)

    def _generate_admitted(self, prompt: str, max_new_tokens: int) -> str:
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        request_id = str(uuid.uuid4())

        hidden = self.embed_tokens[input_ids].unsqueeze(0)
        position_ids = torch.arange(len(input_ids))
        hidden = self._pipeline("Prefill", request_id, hidden, position_ids)

        next_token = self._logits(hidden).argmax(dim=-1)
        generated = [next_token.item()]

        for step in range(max_new_tokens - 1):
            position = len(input_ids) + step
            hidden = self.embed_tokens[next_token].unsqueeze(0).unsqueeze(0)
            position_ids = torch.tensor([position])
            hidden = self._pipeline("Decode", request_id, hidden, position_ids)
            next_token = self._logits(hidden).argmax(dim=-1)
            generated.append(next_token.item())
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        for stub in self.stubs:
            stub.Reset(femllm_pb2.ResetRequest(request_id=request_id))

        return self.tokenizer.decode(generated, skip_special_tokens=True)
```

This releases the semaphore (via the `with self._admission:` context manager in `generate()`) only after `Reset()` has already freed the request's KV-cache entries on every worker — the slot doesn't free until cleanup is fully done.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_num_users.py -v -s`
Expected: PASS. Also re-run `pytest tests/test_end_to_end.py tests/test_coordinator_concurrent.py -v` to confirm no regression (default `num_users=4` doesn't constrain the existing tests' concurrency).

- [ ] **Step 5: Commit**

```bash
git add src/femllm/coordinator.py tests/test_num_users.py
git commit -m "feat: num_users admission semaphore bounds concurrent KV-cache memory"
```

---

### Task 15: `max_context_length` — Bounding Single-Sequence KV-Cache Growth

**Files:**
- Modify: `src/femllm/coordinator.py`
- Create: `tests/test_max_context_length.py`

**Interfaces:**
- Modifies: `Coordinator.__init__(..., max_context_length=2048)` — new parameter
- Produces: `generate()` raises `ValueError` if the prompt alone exceeds `max_context_length`; decoding stops (same cleanup path as EOS) once accumulated length hits the cap

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_max_context_length.py
"""
Requires 4 worker processes already running (see Task 9's start_workers.py).
Run: pytest tests/test_max_context_length.py -v -s
"""
import pytest
from src.femllm.coordinator import Coordinator
from src.femllm.forward import ModelConfig

TINYLLAMA_CONFIG = ModelConfig(
    hidden_size=2048, num_heads=32, num_kv_heads=4, head_dim=64,
    intermediate_size=5632, rms_norm_eps=1e-5, rope_theta=10000.0,
)
WORKER_PORTS = [50051, 50052, 50053, 50054]

def test_oversized_prompt_rejected():
    coord = Coordinator(
        model_dir="models/tinyllama",
        shard_dir="shards/tinyllama",
        worker_ports=WORKER_PORTS,
        config=TINYLLAMA_CONFIG,
        max_context_length=5,
    )
    long_prompt = "This prompt has way more than five tokens in it for sure"
    with pytest.raises(ValueError, match="exceeds max_context_length"):
        coord.generate(long_prompt, max_new_tokens=10)

def test_generation_stops_at_max_context_length():
    coord = Coordinator(
        model_dir="models/tinyllama",
        shard_dir="shards/tinyllama",
        worker_ports=WORKER_PORTS,
        config=TINYLLAMA_CONFIG,
        max_context_length=8,
    )
    input_ids = coord.tokenizer("Hello there", return_tensors="pt")["input_ids"][0]
    prompt_len = len(input_ids)
    output = coord.generate("Hello there", max_new_tokens=50)
    total_len = prompt_len + len(coord.tokenizer(output)["input_ids"])
    assert total_len <= 8 + 2  # allow for tokenizer special-token slack
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_max_context_length.py -v`
Expected: `TypeError: Coordinator.__init__() got an unexpected keyword argument 'max_context_length'`

- [ ] **Step 3: Add max_context_length to src/femllm/coordinator.py**

Modify `Coordinator.__init__`'s signature to accept `max_context_length: int = 2048`, storing it as `self.max_context_length = max_context_length` alongside the existing `self.num_users` assignment.

Replace `_generate_admitted` with a version that checks the cap at both ends:

```python
    def _generate_admitted(self, prompt: str, max_new_tokens: int) -> str:
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        if len(input_ids) > self.max_context_length:
            raise ValueError(
                f"Prompt length {len(input_ids)} exceeds max_context_length={self.max_context_length}"
            )

        request_id = str(uuid.uuid4())

        hidden = self.embed_tokens[input_ids].unsqueeze(0)
        position_ids = torch.arange(len(input_ids))
        hidden = self._pipeline("Prefill", request_id, hidden, position_ids)

        next_token = self._logits(hidden).argmax(dim=-1)
        generated = [next_token.item()]

        total_len = len(input_ids) + 1
        for step in range(max_new_tokens - 1):
            if total_len >= self.max_context_length:
                break
            position = len(input_ids) + step
            hidden = self.embed_tokens[next_token].unsqueeze(0).unsqueeze(0)
            position_ids = torch.tensor([position])
            hidden = self._pipeline("Decode", request_id, hidden, position_ids)
            next_token = self._logits(hidden).argmax(dim=-1)
            generated.append(next_token.item())
            total_len += 1
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        for stub in self.stubs:
            stub.Reset(femllm_pb2.ResetRequest(request_id=request_id))

        return self.tokenizer.decode(generated, skip_special_tokens=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_max_context_length.py -v -s`
Expected: both tests PASS. Also re-run `pytest tests/test_end_to_end.py -v` to confirm the default `max_context_length=2048` doesn't affect existing short-prompt tests.

- [ ] **Step 5: Commit**

```bash
git add src/femllm/coordinator.py tests/test_max_context_length.py
git commit -m "feat: max_context_length caps single-sequence KV-cache growth"
```

---

### Task 16: Concurrent End-to-End Proof

**Files:**
- Create: `tests/test_concurrent_end_to_end.py`

**Interfaces:**
- Consumes: everything from Tasks 9-14

This is the final proof: many concurrent requests, correctness matching the single-request baseline, and both memory-bounding parameters exercised together.

- [ ] **Step 1: Write the test**

```python
# tests/test_concurrent_end_to_end.py
"""
Requires 4 worker processes already running (see Task 9's start_workers.py).
Run: pytest tests/test_concurrent_end_to_end.py -v -s
"""
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.femllm.coordinator import Coordinator
from src.femllm.forward import ModelConfig

TINYLLAMA_CONFIG = ModelConfig(
    hidden_size=2048, num_heads=32, num_kv_heads=4, head_dim=64,
    intermediate_size=5632, rms_norm_eps=1e-5, rope_theta=10000.0,
)
WORKER_PORTS = [50051, 50052, 50053, 50054]

def test_many_concurrent_requests_match_individual_baselines():
    coord = Coordinator(
        model_dir="models/tinyllama",
        shard_dir="shards/tinyllama",
        worker_ports=WORKER_PORTS,
        config=TINYLLAMA_CONFIG,
        num_users=3,
    )
    prompts = [
        "The capital of France is",
        "The capital of Japan is",
        "The capital of Italy is",
        "The capital of Germany is",
        "The capital of Spain is",
    ]

    tokenizer = AutoTokenizer.from_pretrained("models/tinyllama")
    model = AutoModelForCausalLM.from_pretrained("models/tinyllama", torch_dtype=torch.bfloat16)
    model.eval()

    baselines = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        baselines.append(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())

    results = coord.generate_concurrent(prompts, max_new_tokens=5)
    results = [r.strip() for r in results]

    assert results == baselines, f"Distributed: {results} | Baseline: {baselines}"
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_concurrent_end_to_end.py -v -s`
Expected: PASS — 5 concurrent requests (admitted 3 at a time via `num_users=3`, with worker-side batching coalescing whichever land on the same chunk together) produce byte-identical output to sequential single-process HuggingFace baselines.

- [ ] **Step 3: Commit**

```bash
git add tests/test_concurrent_end_to_end.py
git commit -m "test: concurrent generation matches individual greedy baselines"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| Chunked round-robin layer assignment, `window_size` parameter | Task 2 |
| Chunk batch-loads/evicts as a unit, idle-time prefetch of next chunk | Task 4 |
| Layer/RMSNorm/RoPE/GQA/KV-cache forward math (unaffected by redesign) | Task 5 (unchanged from original plan) |
| Worker runs one chunk per call, chunk-scoped KV cache | Task 6 |
| `layer_idx` in wire protocol | Task 7 |
| Worker server reads layer assignment from manifest | Task 8 |
| End-to-end correctness vs. HuggingFace baseline, single request | Task 9 |
| Weight vs. KV-cache eviction lifecycle distinction (weights evict every turn, KV cache persists until Reset) | Enforced by Task 4 (weights) vs. `Worker.kv_caches` (Task 6, Task 10) design and tests, not a separate task |
| Conveyor-belt batching: worker batches concurrent requests for the same chunk | Task 10 (belt thread + initial grouping logic) |
| Requests that don't align with the currently-resident chunk must wait, never error | Task 10 (`test_late_arrival_for_already_passed_chunk_waits_instead_of_crashing`) — see correction #1 below |
| Batching must not be so restrictive it rarely fires under realistic staggered traffic | Task 11 (decode batches regardless of position; KV-cache storage untouched) — see correction #2 below |
| gRPC layer stays correct under concurrency | Task 12 |
| Coordinator handles concurrent requests | Task 13 |
| `num_users` bounds concurrent KV-cache memory | Task 14 |
| `max_context_length` bounds single-sequence KV-cache growth | Task 15 |
| Full concurrent correctness proof | Task 16 |

No gaps found.

**Correction #1 found during review (fixed before this version):** the first draft of Task 4's `LayerStreamer` exposed a `get_chunk(start_layer_idx)` that raised `ValueError` if the requested chunk wasn't the currently-resident one. Under real concurrency, any request arriving even slightly out of sync with a worker's belt position — completely normal, not an edge case — would crash instead of correctly waiting for the belt to lap back around, contradicting the design spec's own stated behavior. Fixed by moving all streamer access behind a single dedicated belt thread per worker (Task 10); `LayerStreamer` itself (Task 4) is now explicitly single-threaded-access-only, and `Worker.submit()` is the only thread-safe entry point, with a regression test proving the exact failure scenario is now handled by waiting rather than raising.

**Correction #2 found during review (fixed before this version):** the first draft of Task 10's `_compute_chunk` only grouped requests sharing an *identical* `position_ids` tensor. For decode calls that meant requiring the exact same accumulated sequence length across requests — which realistic staggered traffic (different prompt lengths, different admission times) essentially never satisfies, so the batching mechanism was correctly built but would rarely actually trigger. Fixed in Task 11 by splitting a layer into its position-independent parts (RMSNorm, projections, FFN — batched across every pending decode request regardless of position) and its position-dependent part (attention — run per-row against each request's own untouched KV cache, no padding or reshaping of stored state). Prefill batching remains exact-match-only and out of scope, since batching different prompt lengths would need padding/masking machinery this fix deliberately avoided.

**Placeholder scan:** No "TBD", "TODO", or "add appropriate X" found — every step has complete, runnable code or an exact command with expected output.

**Type consistency check:** Traced every interface end to end:
- `split_model(...) → manifest.json {window_size, workers: [{id, layer_indices, shard_file}]}` (Task 2) is read identically by `worker_server.py`'s `serve()` (Task 8/12), `scripts/start_workers.py` (Task 9), and `Coordinator.__init__` (Task 9) — same keys, same shapes.
- `LayerStreamer.current_start_layer()/current_chunk()/advance()` (Task 4) are called only from `Worker`'s belt thread (Task 6's simple in-order version, then Task 10's thread-safe replacement) — never from request-handling threads directly, which is the invariant that makes single-threaded, lock-free access inside `LayerStreamer` safe.
- `forward_layer`'s signature (Task 5) is never modified — it's still used for Task 6's single-request path and Task 11's exact-match prefill path, both with a 1-D `position_ids`. The new `forward_layer_decode_batch` (Task 11) is a separate function with its own `[N]` per-row `position_ids` and `list[dict]` of per-row KV caches — it doesn't change `forward_layer`'s contract, it adds an alternative path for the case `forward_layer` can't handle (batched rows at different positions).
- `Worker._compute_chunk`'s signature (`batch`, `start_layer_idx` in, `dict[request_id -> Tensor]` out) is unchanged between Task 10 and Task 11 — only its internal routing changes (decode items go through the new function, prefill items keep the Task 10 path), so `_run_belt` (Task 10) never needs to change when Task 11 lands.
- `Coordinator.generate`'s signature is stable across Tasks 9, 14, 15 — each later task replaces its body (explicitly called out in each step) rather than layering on a second method with a different name, so `generate_concurrent` (Task 13) keeps working unmodified through Tasks 14-15.
- `Worker.forward`'s external signature (Task 6) is unchanged by Task 10's and Task 11's internal replacements, so Task 8/12's `WorkerServicer._run_forward` and Task 9's `Coordinator._pipeline` never need to change when either lands.

No mismatches found.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-02-round-robin-layer-streaming-inference.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
