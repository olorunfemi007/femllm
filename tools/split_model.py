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
