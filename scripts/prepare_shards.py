# scripts/prepare_shards.py
"""One-time model download + shard split. Meant to run as an init container
on the coordinator Deployment, against an NFS-backed PVC shared with the
worker pods — the coordinator prepares shards, workers only ever wait for
them (see manifests/statefulset-worker.yaml's wait-for-shards init container
in femllm-deploy).

Idempotent: skips all work if --ready-marker already exists. Safe under
concurrent coordinator replicas: uses an mkdir-based lock (atomic on POSIX
and NFS) so only one replica does the download+split; the others wait for
the marker instead of racing to write the same shard files.
"""
import argparse
import os
import shutil
import sys
import time
from huggingface_hub import snapshot_download

sys.path.insert(0, ".")
from tools.split_model import split_model


def _wait_for_marker(ready_marker: str) -> None:
    print(f"Another replica is already preparing shards — waiting for {ready_marker} ...")
    while not os.path.exists(ready_marker):
        time.sleep(5)
    print("Shards ready (prepared by another replica).")


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

    lock_dir = args.ready_marker + ".lock"
    os.makedirs(os.path.dirname(args.ready_marker) or ".", exist_ok=True)
    try:
        os.mkdir(lock_dir)
    except FileExistsError:
        _wait_for_marker(args.ready_marker)
        return

    try:
        # Re-check: another replica may have finished between our first check and acquiring the lock.
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

        with open(args.ready_marker, "w") as f:
            f.write("ready\n")
        print("Done.")
    finally:
        shutil.rmtree(lock_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
