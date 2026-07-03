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
