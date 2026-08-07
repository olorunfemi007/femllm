# scripts/prepare_shards.py
"""One-time model download + shard split. Meant to run as an init container
on the coordinator Deployment, against an NFS-backed PVC shared with the
worker pods — the coordinator prepares shards, workers only ever wait for
them (see manifests/statefulset-worker.yaml's wait-for-shards init container
in femllm-deploy).

Idempotent: skips all work if --ready-marker already exists AND the existing
shard_dir/manifest.json was built with the same --num-workers/--window-size
as this invocation. SHARD_DIR is scoped to the model name only, not to those
two values — without this check, changing NUM_WORKERS in configmap.yaml and
redeploying would silently skip re-sharding and leave workers reading shard
files whose layer assignments were computed for a different worker count.
That's a correctness bug, not just wasted time: no error, just wrong output.
On a mismatch, this re-splits (fast — the downloaded model weights are kept
around specifically to make that cheap) rather than trusting a stale marker.

Safe under concurrent coordinator replicas: uses an mkdir-based lock (atomic
on POSIX and NFS) so only one replica does the download+split; the others
wait for the marker instead of racing to write the same shard files.

The lock is staleness-checked: if a replica dies mid-download (OOMKilled,
pod eviction, node preemption — anything that sends SIGKILL skips Python's
`finally` cleanup entirely), waiters would otherwise poll forever for a
marker nobody's still working toward. After LOCK_STALE_SECONDS with no sign
the lock holder is still around, a waiter assumes it's dead, clears it, and
retries acquiring it itself.
"""
import argparse
import json
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


def _shards_match_config(shard_dir: str, num_workers: int, window_size: int) -> bool:
    manifest_path = os.path.join(shard_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return False
    with open(manifest_path) as f:
        manifest = json.load(f)
    return manifest.get("num_workers") == num_workers and manifest.get("window_size") == window_size


def _marker_is_valid(ready_marker: str, shard_dir: str, num_workers: int, window_size: int) -> bool:
    return os.path.exists(ready_marker) and _shards_match_config(shard_dir, num_workers, window_size)


def _acquire_lock_or_defer(ready_marker: str, lock_dir: str, shard_dir: str, num_workers: int, window_size: int) -> bool:
    """True: caller acquired the lock and should do the work.
    False: a valid marker appeared (someone else finished with matching
    config) — caller should skip."""
    while True:
        if _marker_is_valid(ready_marker, shard_dir, num_workers, window_size):
            return False
        try:
            os.mkdir(lock_dir)
            return True
        except FileExistsError:
            pass

        print(f"Another replica is already preparing shards — waiting for {ready_marker} ...")
        waited = 0
        while os.path.exists(lock_dir) and not _marker_is_valid(ready_marker, shard_dir, num_workers, window_size):
            time.sleep(5)
            waited += 5
            if waited >= LOCK_STALE_SECONDS:
                print(f"{lock_dir} held for over {LOCK_STALE_SECONDS}s with no progress — assuming its owner died, clearing it.")
                shutil.rmtree(lock_dir, ignore_errors=True)
                break
        # Loop back around: a valid marker may now exist, or the lock is free
        # (cleared or genuinely released) to try acquiring ourselves.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="HuggingFace repo id, must ship .safetensors (e.g. danielhanchen/open_llama_3b_600bt_preview, not openlm-research/open_llama_3b_v2 which only has pytorch_model.bin)")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--ready-marker", required=True)
    args = parser.parse_args()

    if _marker_is_valid(args.ready_marker, args.shard_dir, args.num_workers, args.window_size):
        print(f"{args.ready_marker} already exists and matches num_workers={args.num_workers}/window_size={args.window_size} — skipping.")
        return

    lock_dir = args.ready_marker + ".lock"
    os.makedirs(os.path.dirname(args.ready_marker) or ".", exist_ok=True)
    if not _acquire_lock_or_defer(args.ready_marker, lock_dir, args.shard_dir, args.num_workers, args.window_size):
        print("Shards ready (prepared by another replica).")
        return

    try:
        # snapshot_download is itself resumable/idempotent — if the model was
        # already downloaded by an earlier run (kept around on purpose, see
        # module docstring), this verifies existing files and returns quickly
        # rather than re-fetching multiple GB.
        print(f"Downloading {args.repo_id} to {args.model_dir} ...")
        snapshot_download(
            repo_id=args.repo_id,
            local_dir=args.model_dir,
            ignore_patterns=["*.bin", "*.msgpack", "flax_model*", "tf_model*"],
        )

        if os.path.exists(args.shard_dir):
            print(f"Existing {args.shard_dir} doesn't match the requested config — clearing it before re-splitting.")
            shutil.rmtree(args.shard_dir)
        if os.path.exists(args.ready_marker):
            os.remove(args.ready_marker)

        print(f"Splitting into {args.num_workers} shards (window_size={args.window_size}) ...")
        split_model(args.model_dir, args.shard_dir, args.num_workers, args.window_size)

        with open(args.ready_marker, "w") as f:
            f.write("ready\n")
        print("Done.")
    finally:
        shutil.rmtree(lock_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
