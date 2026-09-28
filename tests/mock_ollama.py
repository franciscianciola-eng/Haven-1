"""A tiny HTTP server that speaks enough of Ollama's API to test the real client over a socket."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class MockOllama(ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.models: dict[str, dict] = {
            "haven": {"capabilities": ["completion"], "model_info": {"qwen3.context_length": 262144}},
        }
        self.replies: list[list[dict] | int] = []  # an int answers with that HTTP error status
        self.requests: list[tuple[str, dict | None]] = []
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def __enter__(self) -> MockOllama:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()
        self.server_close()


class _Handler(BaseHTTPRequestHandler):
    server: MockOllama

    def log_message(self, *args: object) -> None:
        pass

    def _json(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _stream(self, chunks: list[dict]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write((json.dumps(chunk) + "\n").encode())
            self.wfile.flush()

    def do_GET(self) -> None:
        self.server.requests.append((self.path, None))
        if self.path == "/api/version":
            self._json(200, {"version": "0.12.3"})
        elif self.path == "/api/tags":
            self._json(200, {"models": [{"name": name} for name in self.server.models]})
        else:
            self._json(404, {"error": "unknown endpoint"})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, body))
        model = body.get("model")
        if self.path == "/api/show":
            if model not in self.server.models:
                self._json(404, {"error": f"model '{model}' not found"})
            else:
                self._json(200, self.server.models[model])
        elif self.path == "/api/pull":
            self.server.models[model] = {"capabilities": ["completion", "tools"], "model_info": {}}
            self._stream(
                [
                    {"status": "pulling manifest"},
                    {"status": "downloading", "total": 10, "completed": 5},
                    {"status": "success"},
                ]
            )
        elif self.path == "/api/chat":
            if model not in self.server.models:
                self._json(404, {"error": f'model "{model}" not found, try pulling it first'})
                return
            script = self.server.replies.pop(0)
            if isinstance(script, int):
                self._json(script, {"error": "something broke"})
            else:
                self._stream(script)
        else:
            self._json(404, {"error": "unknown endpoint"})
