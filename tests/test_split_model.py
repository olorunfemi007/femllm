import json
import os
import pytest
from safetensors import safe_open

def test_split_creates_coordinator_shard(shard_dir):
    assert os.path.exists(os.path.join(shard_dir, "coordinator.safetensors"))

def test_split_creates_four_worker_shards(shard_dir):
    for i in range(4):
        assert os.path.exists(os.path.join(shard_dir, f"worker_{i}.safetensors"))

def test_split_creates_manifest(shard_dir):
    assert os.path.exists(os.path.join(shard_dir, "manifest.json"))

def test_manifest_records_window_size(shard_dir):
    with open(os.path.join(shard_dir, "manifest.json")) as f:
        manifest = json.load(f)
    assert manifest["window_size"] == 1
    assert manifest["num_workers"] == 4
    assert manifest["num_layers"] == 22

def test_manifest_layer_indices_round_robin_window_1(shard_dir):
    with open(os.path.join(shard_dir, "manifest.json")) as f:
        manifest = json.load(f)
    workers = {w["id"]: w["layer_indices"] for w in manifest["workers"]}
    assert workers[0] == [0, 4, 8, 12, 16, 20]
    assert workers[1] == [1, 5, 9, 13, 17, 21]
    assert workers[2] == [2, 6, 10, 14, 18]
    assert workers[3] == [3, 7, 11, 15, 19]

def test_manifest_layer_indices_round_robin_window_2(shard_dir_w2):
    with open(os.path.join(shard_dir_w2, "manifest.json")) as f:
        manifest = json.load(f)
    workers = {w["id"]: w["layer_indices"] for w in manifest["workers"]}
    # chunks of 2: [0,1] [2,3] [4,5] [6,7] ... assigned round-robin by chunk index
    assert workers[0] == [0, 1, 8, 9, 16, 17]
    assert workers[1] == [2, 3, 10, 11, 18, 19]
    assert workers[2] == [4, 5, 12, 13, 20, 21]
    assert workers[3] == [6, 7, 14, 15]

def test_every_layer_assigned_exactly_once(shard_dir):
    with open(os.path.join(shard_dir, "manifest.json")) as f:
        manifest = json.load(f)
    all_layers = []
    for w in manifest["workers"]:
        all_layers.extend(w["layer_indices"])
    assert sorted(all_layers) == list(range(22))

def test_coordinator_shard_has_embeddings(shard_dir):
    with safe_open(os.path.join(shard_dir, "coordinator.safetensors"), framework="pt") as f:
        keys = list(f.keys())
    assert "model.embed_tokens.weight" in keys
    assert "model.norm.weight" in keys
    assert "lm_head.weight" in keys

def test_coordinator_shard_has_no_layers(shard_dir):
    with safe_open(os.path.join(shard_dir, "coordinator.safetensors"), framework="pt") as f:
        keys = list(f.keys())
    assert not any("model.layers." in k for k in keys)

def test_worker_shard_contains_assigned_layers(shard_dir):
    with safe_open(os.path.join(shard_dir, "worker_0.safetensors"), framework="pt") as f:
        keys = list(f.keys())
    layer_indices = {int(k.split(".")[2]) for k in keys if k.startswith("model.layers.")}
    assert layer_indices == {0, 4, 8, 12, 16, 20}

def test_worker_shard_contains_no_embeddings(shard_dir):
    with safe_open(os.path.join(shard_dir, "worker_0.safetensors"), framework="pt") as f:
        keys = list(f.keys())
    assert "model.embed_tokens.weight" not in keys
    assert "lm_head.weight" not in keys
