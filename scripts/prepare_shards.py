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

The lock is staleness-checked: if a replica dies mid-download (OOMKilled,
pod eviction, node preemption — anything that sends SIGKILL skips Python's
`finally` cleanup entirely), waiters would otherwise poll forever for a
marker nobody's still working toward. After LOCK_STALE_SECONDS with no sign
the lock holder is still around, a waiter assumes it's dead, clears it, and
retries acquiring it itself.
"""
import argparse
import os
import shutil
import sys
import time

# Must be set before huggingface_hub is imported — it checks this at import
# time to decide whether to use the Xet storage backend (hf_xet package,
# installed automatically as a transitive dependency). Xet does
# content-defined chunking/deduplication rather than a simple streaming
# write-to-disk, which needs meaningfully more memory for a single multi-GB
# file than plain HTTP download — confirmed as the actual cause of repeated
# OOM-kills here (always at ~89% through fetching, on the large safetensors
# file, regardless of tools/split_model.py's own memory-bounded design,
# which never even got a chance to run).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download

sys.path.insert(0, ".")
from tools.split_model import split_model

LOCK_STALE_SECONDS = 900  # generous vs. real split time (~seconds), bounds worst-case hang


def _acquire_lock_or_defer(ready_marker: str, lock_dir: str) -> bool:
    """True: caller acquired the lock and should do the work.
    False: the marker appeared (someone else finished) — caller should skip."""
    while True:
        if os.path.exists(ready_marker):
            return False
        try:
            os.mkdir(lock_dir)
            return True
        except FileExistsError:
            pass

        print(f"Another replica is already preparing shards — waiting for {ready_marker} ...")
        waited = 0
        while os.path.exists(lock_dir) and not os.path.exists(ready_marker):
            time.sleep(5)
            waited += 5
            if waited >= LOCK_STALE_SECONDS:
                print(f"{lock_dir} held for over {LOCK_STALE_SECONDS}s with no progress — assuming its owner died, clearing it.")
                shutil.rmtree(lock_dir, ignore_errors=True)
                break
        # Loop back around: marker may now exist, or the lock is free (cleared
        # or genuinely released) to try acquiring ourselves.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="HuggingFace repo id, must ship .safetensors (e.g. danielhanchen/open_llama_3b_600bt_preview, not openlm-research/open_llama_3b_v2 which only has pytorch_model.bin)")
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
    if not _acquire_lock_or_defer(args.ready_marker, lock_dir):
        print("Shards ready (prepared by another replica).")
        return

    try:
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
