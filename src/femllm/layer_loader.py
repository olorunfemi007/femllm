import torch
from safetensors import safe_open


def load_layer_weights(shard_path: str, layer_idx: int) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer_idx}."
    weights = {}
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(prefix):
                local_key = key[len(prefix):]
                # Cast to bf16 regardless of the source checkpoint's native
                # dtype (fp16, fp32, ...) — activations flowing through the
                # pipeline are always bf16 (coordinator casts embed_tokens/
                # lm_head the same way), so uncast weights of another dtype
                # would fail at the first matmul.
                weights[local_key] = f.get_tensor(key).to(torch.bfloat16)
    if not weights:
        raise KeyError(f"No weights found for layer {layer_idx} in {shard_path}")
    return weights
