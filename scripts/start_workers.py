# scripts/start_workers.py
"""Launch N worker processes for local testing, reading layer assignment from manifest.json."""
import sys
import json
import subprocess


def main():
    shard_dir = sys.argv[1]
    model_dir = sys.argv[2]
    base_port = 50051

    with open(f"{shard_dir}/manifest.json") as f:
        manifest = json.load(f)

    procs = []
    for worker in manifest["workers"]:
        port = base_port + worker["id"]
        cmd = [
            sys.executable, "-m", "src.femllm.worker_server",
            "--shard", f"{shard_dir}/{worker['shard_file']}",
            "--manifest", f"{shard_dir}/manifest.json",
            "--model-dir", model_dir,
            "--worker-id", str(worker["id"]),
            "--port", str(port),
        ]
        p = subprocess.Popen(cmd)
        procs.append(p)
        print(f"Started worker {worker['id']} (layers {worker['layer_indices']}) on port {port} PID={p.pid}")

    print(f"\n{len(procs)} workers running. Ctrl+C to stop all.")
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
