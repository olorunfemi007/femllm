# src/femllm/coordinator_server.py
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, "src")

from src.femllm.coordinator import Coordinator


class CoordinatorHandler(BaseHTTPRequestHandler):
    coordinator: Coordinator = None  # set by serve() before the server starts

    def _send_json(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            request = json.loads(self.rfile.read(length))
            prompt = request["prompt"]
            max_new_tokens = request.get("max_new_tokens", 50)
        except (json.JSONDecodeError, KeyError) as e:
            self._send_json(400, {"error": f"invalid request body: {e}"})
            return

        try:
            text = self.coordinator.generate(prompt, max_new_tokens=max_new_tokens)
        except ValueError as e:
            self._send_json(400, {"error": str(e)})
            return

        self._send_json(200, {"text": text})

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}")


def serve(
    model_dir: str,
    shard_dir: str,
    worker_addresses: list[str],
    port: int,
    num_users: int = 4,
    max_context_length: int = 2048,
) -> None:
    coordinator = Coordinator(
        model_dir=model_dir,
        shard_dir=shard_dir,
        worker_addresses=worker_addresses,
        num_users=num_users,
        max_context_length=max_context_length,
    )
    CoordinatorHandler.coordinator = coordinator

    server = ThreadingHTTPServer(("0.0.0.0", port), CoordinatorHandler)
    print(f"Coordinator serving on port {port} ({len(worker_addresses)} workers, num_users={num_users})")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--worker-addresses", required=True, help="comma-separated host:port list")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--num-users", type=int, default=4)
    parser.add_argument("--max-context-length", type=int, default=2048)
    args = parser.parse_args()

    serve(
        model_dir=args.model_dir,
        shard_dir=args.shard_dir,
        worker_addresses=args.worker_addresses.split(","),
        port=args.port,
        num_users=args.num_users,
        max_context_length=args.max_context_length,
    )
