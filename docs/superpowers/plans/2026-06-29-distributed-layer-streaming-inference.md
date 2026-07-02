# Distributed Layer-Streaming LLM Inference Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a distributed LLM inference engine where each worker owns a fixed range of transformer layers, streams those layer weights from local storage into a hot memory cache on demand, and passes activations to the next worker via gRPC — proving that multi-node inference can run under extreme per-node memory constraints.

**Architecture:** The coordinator tokenizes input, embeds tokens, and sends hidden states to Worker 0 via gRPC. Workers form a pipeline: each runs its assigned layer range by loading weights from a local safetensors shard into an LRU cache, then forwards activations to the next worker. The final worker returns hidden states to the coordinator, which applies the final norm, computes logits, samples, and repeats the decode loop. Each worker independently maintains its KV cache per request.

**Tech Stack:** Python 3.10+, PyTorch 2.x, safetensors, gRPC + protobuf, transformers (tokenizer only), pytest. Test model: TinyLlama-1.1B-Chat-v1.0 (22 layers, hidden_size=2048).

## Global Constraints

- Python 3.10+ only — no walrus operators or newer syntax for broader compatibility
- PyTorch 2.x — use `F.scaled_dot_product_attention` (not manual attention)
- All tensors in bfloat16 during inference to match TinyLlama's native dtype
- TinyLlama config: num_layers=22, hidden_size=2048, num_heads=32, num_kv_heads=4, head_dim=64, intermediate_size=5632, vocab_size=32000, rms_norm_eps=1e-5, rope_theta=10000.0
- 4 workers for all tests: Worker 0 → layers 0–5, Worker 1 → layers 6–11, Worker 2 → layers 12–17, Worker 3 → layers 18–21
- Coordinator owns: `model.embed_tokens.weight`, `model.norm.weight`, `lm_head.weight`
- Greedy decoding only (argmax) — no sampling needed for this prototype
- Tests requiring model files use `models/tinyllama/` as the local path (downloaded in Task 1)

---

## File Structure

```
air-llm/
├── proto/
│   └── airllm.proto                  # gRPC service + message definitions
├── src/
│   ├── airllm/
│   │   ├── __init__.py
│   │   ├── layer_loader.py           # load one layer's weights from a shard file
│   │   ├── weight_cache.py           # LRU weight cache with prefetch thread
│   │   ├── forward.py                # transformer layer forward pass (attention + FFN)
│   │   ├── kv_cache.py               # per-request, per-layer KV cache store
│   │   ├── worker.py                 # Worker class: owns layer range, runs forward pass
│   │   ├── worker_server.py          # gRPC server wrapping Worker
│   │   └── coordinator.py            # Coordinator: embed → pipeline → sample
│   └── airllm_pb2.py                 # generated (do not edit)
│   └── airllm_pb2_grpc.py            # generated (do not edit)
├── tools/
│   └── split_model.py                # CLI: split HuggingFace model into per-worker shards
├── tests/
│   ├── conftest.py                   # shared fixtures: tmp shards, model path
│   ├── test_split_model.py
│   ├── test_layer_loader.py
│   ├── test_weight_cache.py
│   ├── test_forward.py
│   ├── test_kv_cache.py
│   ├── test_worker.py
│   └── test_end_to_end.py
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
- Create: `src/airllm/__init__.py`
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
```

- [ ] **Step 2: Create .gitignore**

```
models/
shards/
src/airllm_pb2.py
src/airllm_pb2_grpc.py
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 3: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: all packages install without error

- [ ] **Step 4: Create src/airllm/__init__.py**

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
SHARD_DIR = "shards/tinyllama"

@pytest.fixture(scope="session")
def model_dir():
    if not os.path.exists(MODEL_DIR):
        pytest.skip("TinyLlama not downloaded — run Task 1 setup first")
    return MODEL_DIR

@pytest.fixture(scope="session")
def shard_dir(tmp_path_factory, model_dir):
    from tools.split_model import split_model
    out = str(tmp_path_factory.mktemp("shards"))
    split_model(model_dir, out, num_workers=4)
    return out
```

- [ ] **Step 8: Commit**

```bash
git init
git add requirements.txt .gitignore src/ tests/conftest.py
git commit -m "feat: project setup and TinyLlama download"
```

---

### Task 2: Model Shard Splitter

**Files:**
- Create: `tools/split_model.py`
- Create: `tests/test_split_model.py`

**Interfaces:**
- Consumes: `models/tinyllama/` — HuggingFace safetensors model files
- Produces: `split_model(model_dir, output_dir, num_workers)` — writes `coordinator.safetensors`, `worker_0.safetensors` … `worker_{n-1}.safetensors`, `manifest.json` to `output_dir`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_split_model.py
import json
import pytest
from safetensors import safe_open
from tools.split_model import split_model

def test_split_creates_coordinator_shard(tmp_path, model_dir):
    split_model(model_dir, str(tmp_path), num_workers=4)
    assert (tmp_path / "coordinator.safetensors").exists()

def test_split_creates_four_worker_shards(tmp_path, model_dir):
    split_model(model_dir, str(tmp_path), num_workers=4)
    for i in range(4):
        assert (tmp_path / f"worker_{i}.safetensors").exists()

def test_split_creates_manifest(tmp_path, model_dir):
    split_model(model_dir, str(tmp_path), num_workers=4)
    assert (tmp_path / "manifest.json").exists()

def test_manifest_layer_ranges(tmp_path, model_dir):
    split_model(model_dir, str(tmp_path), num_workers=4)
    with open(tmp_path / "manifest.json") as f:
        manifest = json.load(f)
    assert manifest["workers"][0]["start_layer"] == 0
    assert manifest["workers"][0]["end_layer"] == 5
    assert manifest["workers"][1]["start_layer"] == 6
    assert manifest["workers"][1]["end_layer"] == 11
    assert manifest["workers"][2]["start_layer"] == 12
    assert manifest["workers"][2]["end_layer"] == 17
    assert manifest["workers"][3]["start_layer"] == 18
    assert manifest["workers"][3]["end_layer"] == 21

def test_coordinator_shard_has_embeddings(tmp_path, model_dir):
    split_model(model_dir, str(tmp_path), num_workers=4)
    with safe_open(str(tmp_path / "coordinator.safetensors"), framework="pt") as f:
        keys = list(f.keys())
    assert "model.embed_tokens.weight" in keys
    assert "model.norm.weight" in keys
    assert "lm_head.weight" in keys

def test_coordinator_shard_has_no_layers(tmp_path, model_dir):
    split_model(model_dir, str(tmp_path), num_workers=4)
    with safe_open(str(tmp_path / "coordinator.safetensors"), framework="pt") as f:
        keys = list(f.keys())
    assert not any("model.layers." in k for k in keys)

def test_worker_shard_contains_correct_layers(tmp_path, model_dir):
    split_model(model_dir, str(tmp_path), num_workers=4)
    with safe_open(str(tmp_path / "worker_0.safetensors"), framework="pt") as f:
        keys = list(f.keys())
    layer_indices = {int(k.split(".")[2]) for k in keys if k.startswith("model.layers.")}
    assert layer_indices == {0, 1, 2, 3, 4, 5}

def test_worker_shard_contains_no_embeddings(tmp_path, model_dir):
    split_model(model_dir, str(tmp_path), num_workers=4)
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


def split_model(model_dir: str, output_dir: str, num_workers: int) -> None:
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

    layers_per_worker = num_layers // num_workers
    manifest = {"num_workers": num_workers, "num_layers": num_layers, "workers": []}

    for worker_idx in range(num_workers):
        start = worker_idx * layers_per_worker
        end = num_layers if worker_idx == num_workers - 1 else start + layers_per_worker

        worker_tensors = {
            k: v for k, v in tensors.items()
            if k.startswith("model.layers.") and start <= int(k.split(".")[2]) < end
        }
        save_file(worker_tensors, os.path.join(output_dir, f"worker_{worker_idx}.safetensors"))

        manifest["workers"].append({
            "id": worker_idx,
            "start_layer": start,
            "end_layer": end - 1,
            "shard_file": f"worker_{worker_idx}.safetensors",
        })

    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    split_model(sys.argv[1], sys.argv[2], int(sys.argv[3]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_split_model.py -v`
Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tools/split_model.py tests/test_split_model.py
git commit -m "feat: model shard splitter for per-worker safetensors files"
```

---

### Task 3: Layer Weight Loader

**Files:**
- Create: `src/airllm/layer_loader.py`
- Create: `tests/test_layer_loader.py`

**Interfaces:**
- Consumes: `shard_dir/worker_{i}.safetensors` produced by `split_model`
- Produces: `load_layer_weights(shard_path: str, layer_idx: int) -> dict[str, torch.Tensor]` — keys are local names like `self_attn.q_proj.weight` (prefix `model.layers.{N}.` stripped)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_layer_loader.py
import torch
import pytest
from src.airllm.layer_loader import load_layer_weights

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

def test_load_layer_from_worker_1_returns_layer_6(shard_dir):
    weights = load_layer_weights(f"{shard_dir}/worker_1.safetensors", layer_idx=6)
    assert set(weights.keys()) == EXPECTED_KEYS
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_layer_loader.py -v`
Expected: `ModuleNotFoundError: No module named 'src.airllm.layer_loader'`

- [ ] **Step 3: Implement src/airllm/layer_loader.py**

```python
# src/airllm/layer_loader.py
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
git add src/airllm/layer_loader.py tests/test_layer_loader.py
git commit -m "feat: layer weight loader from safetensors shard"
```

---

### Task 4: LRU Weight Cache

**Files:**
- Create: `src/airllm/weight_cache.py`
- Create: `tests/test_weight_cache.py`

**Interfaces:**
- Consumes: `load_layer_weights` from `src.airllm.layer_loader`
- Produces: `WeightCache(shard_path, layer_indices, max_cached)` with methods:
  - `get(layer_idx: int) -> dict[str, torch.Tensor]`
  - `prefetch(layer_idx: int) -> None` (non-blocking, background thread)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_weight_cache.py
import time
import pytest
from src.airllm.weight_cache import WeightCache

def test_get_returns_weights(shard_dir):
    cache = WeightCache(f"{shard_dir}/worker_0.safetensors", layer_indices=list(range(6)), max_cached=3)
    weights = cache.get(0)
    assert "self_attn.q_proj.weight" in weights

def test_get_same_layer_twice_uses_cache(shard_dir):
    cache = WeightCache(f"{shard_dir}/worker_0.safetensors", layer_indices=list(range(6)), max_cached=3)
    w1 = cache.get(0)
    w2 = cache.get(0)
    assert w1 is w2

def test_cache_evicts_lru_when_full(shard_dir):
    cache = WeightCache(f"{shard_dir}/worker_0.safetensors", layer_indices=list(range(6)), max_cached=2)
    cache.get(0)
    cache.get(1)
    cache.get(2)  # should evict layer 0
    assert len(cache.cache) == 2
    assert 0 not in cache.cache

def test_evicted_layer_reloaded_correctly(shard_dir):
    cache = WeightCache(f"{shard_dir}/worker_0.safetensors", layer_indices=list(range(6)), max_cached=2)
    w0_first = cache.get(0)
    cache.get(1)
    cache.get(2)  # evicts layer 0
    w0_second = cache.get(0)
    import torch
    assert torch.allclose(w0_first["self_attn.q_proj.weight"], w0_second["self_attn.q_proj.weight"])

def test_prefetch_loads_layer_in_background(shard_dir):
    cache = WeightCache(f"{shard_dir}/worker_0.safetensors", layer_indices=list(range(6)), max_cached=3)
    cache.prefetch(1)
    time.sleep(0.5)
    assert 1 in cache.cache

def test_prefetch_ignores_layer_not_in_range(shard_dir):
    cache = WeightCache(f"{shard_dir}/worker_0.safetensors", layer_indices=list(range(6)), max_cached=3)
    cache.prefetch(99)  # not in this worker's range, must not crash
    time.sleep(0.1)
    assert 99 not in cache.cache
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_weight_cache.py -v`
Expected: `ModuleNotFoundError: No module named 'src.airllm.weight_cache'`

- [ ] **Step 3: Implement src/airllm/weight_cache.py**

```python
# src/airllm/weight_cache.py
import threading
from collections import OrderedDict
import torch
from src.airllm.layer_loader import load_layer_weights


class WeightCache:
    def __init__(self, shard_path: str, layer_indices: list[int], max_cached: int = 3):
        self.shard_path = shard_path
        self.layer_indices = set(layer_indices)
        self.max_cached = max_cached
        self.cache: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, layer_idx: int) -> dict[str, torch.Tensor]:
        with self.lock:
            if layer_idx in self.cache:
                self.cache.move_to_end(layer_idx)
                return self.cache[layer_idx]
            weights = load_layer_weights(self.shard_path, layer_idx)
            self._insert(layer_idx, weights)
            return weights

    def prefetch(self, layer_idx: int) -> None:
        if layer_idx not in self.layer_indices:
            return
        threading.Thread(target=self._background_load, args=(layer_idx,), daemon=True).start()

    def _background_load(self, layer_idx: int) -> None:
        with self.lock:
            if layer_idx in self.cache:
                return
        weights = load_layer_weights(self.shard_path, layer_idx)
        with self.lock:
            if layer_idx not in self.cache:
                self._insert(layer_idx, weights)

    def _insert(self, layer_idx: int, weights: dict[str, torch.Tensor]) -> None:
        self.cache[layer_idx] = weights
        self.cache.move_to_end(layer_idx)
        if len(self.cache) > self.max_cached:
            self.cache.popitem(last=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_weight_cache.py -v`
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/airllm/weight_cache.py tests/test_weight_cache.py
git commit -m "feat: LRU weight cache with background prefetch thread"
```

---

### Task 5: Transformer Layer Forward Pass

**Files:**
- Create: `src/airllm/forward.py`
- Create: `tests/test_forward.py`

**Interfaces:**
- Consumes: `dict[str, torch.Tensor]` from `WeightCache.get`
- Produces:
  - `forward_layer(weights, hidden_states, position_ids, kv_cache, layer_idx, config) -> torch.Tensor`
  - `rms_norm(x, weight, eps) -> torch.Tensor`
  - `ModelConfig` dataclass

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_forward.py
import torch
import pytest
from src.airllm.forward import forward_layer, rms_norm, ModelConfig
from src.airllm.layer_loader import load_layer_weights

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
    # Prefill 5 tokens
    hidden = torch.randn(1, 5, 2048, dtype=torch.bfloat16)
    forward_layer(weights, hidden, torch.arange(5), kv_cache, layer_idx=0, config=TINYLLAMA_CONFIG)
    # Decode 1 token at position 5
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
Expected: `ModuleNotFoundError: No module named 'src.airllm.forward'`

- [ ] **Step 3: Implement src/airllm/forward.py**

```python
# src/airllm/forward.py
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

    # Pre-attention norm
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, weights["input_layernorm.weight"], config.rms_norm_eps)

    # QKV projections
    q = hidden_states @ weights["self_attn.q_proj.weight"].T
    k = hidden_states @ weights["self_attn.k_proj.weight"].T
    v = hidden_states @ weights["self_attn.v_proj.weight"].T

    # Reshape to [bsz, heads, seq, head_dim]
    q = q.view(bsz, seq_len, config.num_heads, config.head_dim).transpose(1, 2)
    k = k.view(bsz, seq_len, config.num_kv_heads, config.head_dim).transpose(1, 2)
    v = v.view(bsz, seq_len, config.num_kv_heads, config.head_dim).transpose(1, 2)

    # Rotary embeddings
    q, k = _apply_rope(q, k, position_ids, config.rope_theta)

    # Append to KV cache
    if layer_idx in kv_cache:
        k_past, v_past = kv_cache[layer_idx]
        k = torch.cat([k_past, k], dim=2)
        v = torch.cat([v_past, v], dim=2)
    kv_cache[layer_idx] = (k, v)

    # GQA: repeat KV heads to match Q heads
    groups = config.num_heads // config.num_kv_heads
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)

    # Attention — causal only during prefill (seq_len > 1)
    attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=(seq_len > 1))
    attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, config.num_heads * config.head_dim)
    attn_out = attn_out @ weights["self_attn.o_proj.weight"].T
    hidden_states = residual + attn_out

    # FFN
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
git add src/airllm/forward.py tests/test_forward.py
git commit -m "feat: transformer layer forward pass with GQA, RoPE, and KV cache"
```

---

### Task 6: Worker — Single-Node Layer-Range Inference

**Files:**
- Create: `src/airllm/worker.py`
- Create: `tests/test_worker.py`

**Interfaces:**
- Consumes: `WeightCache`, `forward_layer`, `ModelConfig`
- Produces: `Worker(shard_path, layer_indices, config, max_cached)` with methods:
  - `forward(hidden_states, position_ids, request_id) -> torch.Tensor`
  - `reset(request_id) -> None`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_worker.py
import torch
import pytest
from src.airllm.worker import Worker
from src.airllm.forward import ModelConfig

TINYLLAMA_CONFIG = ModelConfig(
    hidden_size=2048, num_heads=32, num_kv_heads=4, head_dim=64,
    intermediate_size=5632, rms_norm_eps=1e-5, rope_theta=10000.0,
)

def test_worker_forward_output_shape(shard_dir):
    worker = Worker(f"{shard_dir}/worker_0.safetensors", list(range(6)), TINYLLAMA_CONFIG, max_cached=3)
    hidden = torch.randn(1, 5, 2048, dtype=torch.bfloat16)
    out = worker.forward(hidden, torch.arange(5), request_id="req1")
    assert out.shape == (1, 5, 2048)

def test_worker_forward_creates_kv_cache_entry(shard_dir):
    worker = Worker(f"{shard_dir}/worker_0.safetensors", list(range(6)), TINYLLAMA_CONFIG, max_cached=3)
    hidden = torch.randn(1, 5, 2048, dtype=torch.bfloat16)
    worker.forward(hidden, torch.arange(5), request_id="req1")
    assert "req1" in worker.kv_caches
    assert 0 in worker.kv_caches["req1"]

def test_worker_decode_extends_kv_cache(shard_dir):
    worker = Worker(f"{shard_dir}/worker_0.safetensors", list(range(6)), TINYLLAMA_CONFIG, max_cached=3)
    hidden = torch.randn(1, 5, 2048, dtype=torch.bfloat16)
    worker.forward(hidden, torch.arange(5), request_id="req1")
    hidden = torch.randn(1, 1, 2048, dtype=torch.bfloat16)
    worker.forward(hidden, torch.tensor([5]), request_id="req1")
    k, _ = worker.kv_caches["req1"][0]
    assert k.shape[2] == 6

def test_worker_reset_clears_kv_cache(shard_dir):
    worker = Worker(f"{shard_dir}/worker_0.safetensors", list(range(6)), TINYLLAMA_CONFIG, max_cached=3)
    hidden = torch.randn(1, 5, 2048, dtype=torch.bfloat16)
    worker.forward(hidden, torch.arange(5), request_id="req1")
    worker.reset("req1")
    assert "req1" not in worker.kv_caches

def test_worker_multiple_requests_isolated(shard_dir):
    worker = Worker(f"{shard_dir}/worker_0.safetensors", list(range(6)), TINYLLAMA_CONFIG, max_cached=3)
    h1 = torch.randn(1, 3, 2048, dtype=torch.bfloat16)
    h2 = torch.randn(1, 7, 2048, dtype=torch.bfloat16)
    worker.forward(h1, torch.arange(3), request_id="req1")
    worker.forward(h2, torch.arange(7), request_id="req2")
    k1, _ = worker.kv_caches["req1"][0]
    k2, _ = worker.kv_caches["req2"][0]
    assert k1.shape[2] == 3
    assert k2.shape[2] == 7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_worker.py -v`
Expected: `ModuleNotFoundError: No module named 'src.airllm.worker'`

- [ ] **Step 3: Implement src/airllm/worker.py**

```python
# src/airllm/worker.py
import torch
from src.airllm.weight_cache import WeightCache
from src.airllm.forward import forward_layer, ModelConfig


class Worker:
    def __init__(self, shard_path: str, layer_indices: list[int], config: ModelConfig, max_cached: int = 3):
        self.layer_indices = layer_indices
        self.config = config
        self.cache = WeightCache(shard_path, layer_indices, max_cached)
        self.kv_caches: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor, request_id: str) -> torch.Tensor:
        if request_id not in self.kv_caches:
            self.kv_caches[request_id] = {}
        kv_cache = self.kv_caches[request_id]

        for i, layer_idx in enumerate(self.layer_indices):
            next_layer = self.layer_indices[i + 1] if i + 1 < len(self.layer_indices) else None
            if next_layer is not None:
                self.cache.prefetch(next_layer)
            weights = self.cache.get(layer_idx)
            hidden_states = forward_layer(weights, hidden_states, position_ids, kv_cache, layer_idx, self.config)

        return hidden_states

    def reset(self, request_id: str) -> None:
        self.kv_caches.pop(request_id, None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_worker.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/airllm/worker.py tests/test_worker.py
git commit -m "feat: worker runs assigned layer range with streaming weight cache"
```

---

### Task 7: gRPC Protocol Definition

**Files:**
- Create: `proto/airllm.proto`
- Create: `src/airllm_pb2.py` (generated)
- Create: `src/airllm_pb2_grpc.py` (generated)

**Interfaces:**
- Produces: gRPC stubs importable as `import airllm_pb2` and `import airllm_pb2_grpc`
  - `WorkerServiceStub` with methods: `Prefill(ForwardRequest) -> ForwardResponse`, `Decode(ForwardRequest) -> ForwardResponse`, `Reset(ResetRequest) -> ResetResponse`
  - `ForwardRequest` fields: `request_id` (str), `hidden_states` (bytes), `shape` (repeated int32), `position_ids` (repeated int32)
  - `ForwardResponse` fields: `hidden_states` (bytes), `shape` (repeated int32)
  - `ResetRequest` fields: `request_id` (str)

- [ ] **Step 1: Write proto/airllm.proto**

```proto
syntax = "proto3";

package airllm;

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
  proto/airllm.proto
```
Expected: `src/airllm_pb2.py` and `src/airllm_pb2_grpc.py` are created.

- [ ] **Step 3: Verify stubs import correctly**

Run:
```bash
python -c "
import sys; sys.path.insert(0, 'src')
import airllm_pb2, airllm_pb2_grpc
req = airllm_pb2.ForwardRequest(request_id='test', shape=[1,5,2048])
print('ForwardRequest ok:', req.request_id)
print('WorkerServiceStub ok:', airllm_pb2_grpc.WorkerServiceStub)
"
```
Expected: prints both lines without error.

- [ ] **Step 4: Commit**

```bash
git add proto/airllm.proto src/airllm_pb2.py src/airllm_pb2_grpc.py
git commit -m "feat: gRPC protocol definition and generated stubs"
```

---

### Task 8: Worker gRPC Server

**Files:**
- Create: `src/airllm/worker_server.py`

**Interfaces:**
- Consumes: `Worker`, `airllm_pb2`, `airllm_pb2_grpc`
- Produces: `serve(shard_path, layer_indices, config, port, max_cached)` — starts blocking gRPC server on given port

Helper functions (also in `worker_server.py`, used by coordinator in Task 9):
- `tensor_to_bytes(t: torch.Tensor) -> bytes`
- `bytes_to_tensor(data: bytes, shape: list[int], dtype) -> torch.Tensor`

- [ ] **Step 1: Write a manual integration test (not pytest — requires running server)**

Create `tests/test_worker_server_manual.py`:

```python
"""
Run this test manually:
  Terminal 1: python -m src.airllm.worker_server --shard shards/tinyllama/worker_0.safetensors --layers 0-5 --port 50051
  Terminal 2: python tests/test_worker_server_manual.py
"""
import sys
sys.path.insert(0, 'src')
import torch
import grpc
import airllm_pb2
import airllm_pb2_grpc
from src.airllm.worker_server import tensor_to_bytes, bytes_to_tensor

channel = grpc.insecure_channel("localhost:50051")
stub = airllm_pb2_grpc.WorkerServiceStub(channel)

hidden = torch.randn(1, 5, 2048, dtype=torch.bfloat16)
req = airllm_pb2.ForwardRequest(
    request_id="manual-test",
    hidden_states=tensor_to_bytes(hidden),
    shape=list(hidden.shape),
    position_ids=list(range(5)),
)
resp = stub.Prefill(req)
out = bytes_to_tensor(resp.hidden_states, resp.shape, torch.bfloat16)
print("Output shape:", out.shape)
assert out.shape == (1, 5, 2048), f"Expected (1, 5, 2048) got {out.shape}"
print("PASS")
```

- [ ] **Step 2: Implement src/airllm/worker_server.py**

```python
# src/airllm/worker_server.py
import sys
import argparse
from concurrent import futures

sys.path.insert(0, "src")

import torch
import grpc
import airllm_pb2
import airllm_pb2_grpc

from src.airllm.worker import Worker
from src.airllm.forward import ModelConfig

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


class WorkerServicer(airllm_pb2_grpc.WorkerServiceServicer):
    def __init__(self, worker: Worker):
        self.worker = worker

    def _run_forward(self, request) -> airllm_pb2.ForwardResponse:
        hidden = bytes_to_tensor(request.hidden_states, list(request.shape), torch.bfloat16)
        position_ids = torch.tensor(list(request.position_ids), dtype=torch.long)
        out = self.worker.forward(hidden, position_ids, request.request_id)
        return airllm_pb2.ForwardResponse(
            hidden_states=tensor_to_bytes(out),
            shape=list(out.shape),
        )

    def Prefill(self, request, context):
        return self._run_forward(request)

    def Decode(self, request, context):
        return self._run_forward(request)

    def Reset(self, request, context):
        self.worker.reset(request.request_id)
        return airllm_pb2.ResetResponse()


def serve(shard_path: str, layer_indices: list[int], config: ModelConfig, port: int, max_cached: int = 3) -> None:
    worker = Worker(shard_path, layer_indices, config, max_cached)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    airllm_pb2_grpc.add_WorkerServiceServicer_to_server(WorkerServicer(worker), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"Worker serving layers {layer_indices[0]}–{layer_indices[-1]} on port {port}")
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", required=True)
    parser.add_argument("--layers", required=True, help="e.g. 0-5")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-cached", type=int, default=3)
    args = parser.parse_args()

    start, end = map(int, args.layers.split("-"))
    layer_indices = list(range(start, end + 1))
    serve(args.shard, layer_indices, TINYLLAMA_CONFIG, args.port, args.max_cached)
```

- [ ] **Step 3: Start worker and run manual test**

Terminal 1:
```bash
python -m src.airllm.worker_server \
  --shard shards/tinyllama/worker_0.safetensors \
  --layers 0-5 \
  --port 50051
```

Terminal 2:
```bash
python tests/test_worker_server_manual.py
```
Expected: `Output shape: torch.Size([1, 5, 2048])` and `PASS`

- [ ] **Step 4: Commit**

```bash
git add src/airllm/worker_server.py tests/test_worker_server_manual.py
git commit -m "feat: worker gRPC server with Prefill, Decode, Reset endpoints"
```

---

### Task 9: Coordinator — End-to-End Token Generation

**Files:**
- Create: `src/airllm/coordinator.py`
- Create: `tests/test_end_to_end.py`

**Interfaces:**
- Consumes: `WorkerServiceStub`, `tensor_to_bytes`, `bytes_to_tensor`, all workers running on known ports
- Produces: `Coordinator(model_dir, shard_dir, worker_ports, config)` with method:
  - `generate(prompt: str, max_new_tokens: int) -> str`

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
import subprocess
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.airllm.coordinator import Coordinator
from src.airllm.forward import ModelConfig

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
Expected: `ModuleNotFoundError: No module named 'src.airllm.coordinator'`

- [ ] **Step 3: Implement src/airllm/coordinator.py**

```python
# src/airllm/coordinator.py
import sys
import uuid
sys.path.insert(0, "src")

import torch
import grpc
from safetensors import safe_open
from transformers import AutoTokenizer
import airllm_pb2
import airllm_pb2_grpc

from src.airllm.forward import rms_norm, ModelConfig
from src.airllm.worker_server import tensor_to_bytes, bytes_to_tensor


class Coordinator:
    def __init__(self, model_dir: str, shard_dir: str, worker_ports: list[int], config: ModelConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        with safe_open(f"{shard_dir}/coordinator.safetensors", framework="pt", device="cpu") as f:
            self.embed_tokens = f.get_tensor("model.embed_tokens.weight").to(torch.bfloat16)
            self.norm_weight = f.get_tensor("model.norm.weight").to(torch.bfloat16)
            self.lm_head = f.get_tensor("lm_head.weight").to(torch.bfloat16)

        self.stubs = []
        for port in worker_ports:
            channel = grpc.insecure_channel(f"localhost:{port}")
            self.stubs.append(airllm_pb2_grpc.WorkerServiceStub(channel))

    def _make_request(self, request_id: str, hidden: torch.Tensor, position_ids: torch.Tensor) -> airllm_pb2.ForwardRequest:
        return airllm_pb2.ForwardRequest(
            request_id=request_id,
            hidden_states=tensor_to_bytes(hidden),
            shape=list(hidden.shape),
            position_ids=position_ids.tolist(),
        )

    def _pipeline(self, method: str, request_id: str, hidden: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        for stub in self.stubs:
            req = self._make_request(request_id, hidden, position_ids)
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
            stub.Reset(airllm_pb2.ResetRequest(request_id=request_id))

        return self.tokenizer.decode(generated, skip_special_tokens=True)
```

- [ ] **Step 4: Create scripts/start_workers.py**

```python
# scripts/start_workers.py
"""Launch 4 worker processes for local testing."""
import sys
import os
import subprocess

def main():
    shard_dir = sys.argv[1]
    worker_assignments = [
        (0, "0-5",   50051),
        (1, "6-11",  50052),
        (2, "12-17", 50053),
        (3, "18-21", 50054),
    ]
    procs = []
    for worker_id, layers, port in worker_assignments:
        cmd = [
            sys.executable, "-m", "src.airllm.worker_server",
            "--shard", f"{shard_dir}/worker_{worker_id}.safetensors",
            "--layers", layers,
            "--port", str(port),
        ]
        p = subprocess.Popen(cmd)
        procs.append(p)
        print(f"Started worker {worker_id} (layers {layers}) on port {port} PID={p.pid}")

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
python tools/split_model.py models/tinyllama shards/tinyllama 4
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

If `test_coordinator_output_matches_baseline` fails, print both outputs side-by-side and trace the difference through single-layer comparisons before assuming a bug in coordinator logic.

- [ ] **Step 7: Commit**

```bash
git add src/airllm/coordinator.py scripts/start_workers.py tests/test_end_to_end.py
git commit -m "feat: coordinator orchestrates distributed pipeline and matches greedy baseline"
```

---

## Self-Review

**Spec coverage check:**

| Requirement | Task |
|---|---|
| Model split into per-worker layer shards | Task 2 |
| Workers stream layer weights from local storage on demand | Tasks 3, 4, 6 |
| Activations pass between workers via RPC | Tasks 7, 8 |
| Per-worker KV cache per request | Tasks 5, 6 |
| Correct end-to-end token generation | Task 9 |
| Lower per-node memory than full-shard residency | Demonstrated by `max_cached` < total layers per worker in Task 4 tests |
| Prefetch to hide weight loading latency | Task 6 (`cache.prefetch` in Worker.forward) |

**Placeholder scan:** None found — all steps contain real code.

**Type consistency check:** `load_layer_weights` returns `dict[str, torch.Tensor]` (Task 3), consumed by `WeightCache.get` (Task 4), consumed by `forward_layer` (Task 5), consumed through `Worker.forward` (Task 6), wrapped by `WorkerServicer` (Task 8). Names are consistent throughout.
