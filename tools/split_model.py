import gc
import json
import os
import sys
from collections import defaultdict
from safetensors import safe_open
from safetensors.torch import save_file


def _index_source_files(model_dir: str) -> dict[str, str]:
    """Map each tensor key to the source .safetensors file that contains it,
    without loading any tensor data — safe_open only reads each file's header."""
    key_to_file: dict[str, str] = {}
    for filename in sorted(os.listdir(model_dir)):
        if filename.endswith(".safetensors"):
            path = os.path.join(model_dir, filename)
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    key_to_file[key] = path
    return key_to_file


def _save_shard(key_to_file: dict[str, str], keys: list[str], output_path: str) -> None:
    """Write one output shard, loading only the tensors it needs. Keys are
    grouped by source file so each source file is opened once per shard and
    only the needed tensors are pulled out of it — peak memory is bounded by
    this one shard's total tensor size, never the whole source model.

    Explicitly deletes the tensors dict and forces a GC pass before
    returning — safetensors' get_tensor() is backed by an mmap of the source
    file, and relying on refcounting alone to release that between shards
    left more than one shard's worth of memory resident at once on tightly
    constrained nodes (observed as an OOM-kill in production)."""
    keys_by_file: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        keys_by_file[key_to_file[key]].append(key)

    tensors = {}
    for path, file_keys in keys_by_file.items():
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in file_keys:
                tensors[key] = f.get_tensor(key)

    save_file(tensors, output_path)
    del tensors
    gc.collect()


def split_model(model_dir: str, output_dir: str, num_workers: int, window_size: int = 1) -> None:
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    num_layers = config["num_hidden_layers"]

    key_to_file = _index_source_files(model_dir)
    os.makedirs(output_dir, exist_ok=True)

    coordinator_keys = [k for k in key_to_file if not k.startswith("model.layers.")]
    _save_shard(key_to_file, coordinator_keys, os.path.join(output_dir, "coordinator.safetensors"))

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
        worker_keys = [
            k for k in key_to_file
            if k.startswith("model.layers.") and int(k.split(".")[2]) in layer_indices
        ]
        _save_shard(key_to_file, worker_keys, os.path.join(output_dir, f"worker_{worker_idx}.safetensors"))

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
