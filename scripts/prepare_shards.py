# scripts/prepare_shards.py
"""One-time model download + shard split, meant to run as a Kubernetes Job
against an NFS-backed PVC shared with the worker/coordinator pods.
Idempotent: skips all work if --ready-marker already exists.
"""
import argparse
import os
import sys
from huggingface_hub import snapshot_download

sys.path.insert(0, ".")
from tools.split_model import split_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="HuggingFace repo id, e.g. openlm-research/open_llama_3b_v2")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--ready-marker", required=True)
    args = parser.parse_args()

    if os.path.exists(args.ready_marker):
        print(f"{args.ready_marker} already exists, skipping.")
        return

    print(f"Downloading {args.repo_id} to {args.model_dir} ...")
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.model_dir,
        ignore_patterns=["*.bin", "*.msgpack", "flax_model*", "tf_model*"],
    )

    print(f"Splitting into {args.num_workers} shards (window_size={args.window_size}) ...")
    split_model(args.model_dir, args.shard_dir, args.num_workers, args.window_size)

    print("Removing full-weight safetensors from model dir (tokenizer/config kept for the coordinator) ...")
    for filename in os.listdir(args.model_dir):
        if filename.endswith(".safetensors"):
            os.remove(os.path.join(args.model_dir, filename))

    os.makedirs(os.path.dirname(args.ready_marker) or ".", exist_ok=True)
    with open(args.ready_marker, "w") as f:
        f.write("ready\n")
    print("Done.")


if __name__ == "__main__":
    main()
