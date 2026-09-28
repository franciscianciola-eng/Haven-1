"""A local web page for watching Haven and being with it (http://127.0.0.1:8765 by default).

It only listens on this computer. Nothing here needs the internet.
"""

from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources

from .check import indicators, probe
from .life import Life

MAX_BODY = 16_000


class Handler(BaseHTTPRequestHandler):
    life: Life  # set on the subclass made by serve()

    def log_message(self, format: str, *args: object) -> None:
        pass  # quiet: the terminal is for talking with Haven

    def _trusted(self) -> bool:
        """Only pages this computer serves itself may use Haven's page, not other websites (even ones that point
        their name at this computer), so nothing else can talk to Haven or read its state."""
        host = (self.headers.get("Host") or "").strip().lower()
        name = host[1:].partition("]")[0] if host.startswith("[") else host.partition(":")[0]
        if name not in ("127.0.0.1", "localhost", "::1"):
            self._json({"error": "not from this computer"}, HTTPStatus.FORBIDDEN)
            return False
        return True

    def do_GET(self) -> None:
        if not self._trusted():
            return
        if self.path in ("/", "/index.html"):
            page = resources.files("haven").joinpath("dashboard.html").read_bytes()
            self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._json(self.life.snapshot())
        elif self.path == "/api/check":
            with self.life.lock:
                measured = probe(self.life.mind)
                found = indicators(self.life.mind, measured)
            self._json({"indicators": [vars(i) for i in found], "measured": measured})
        else:
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _body(self) -> dict | None:
        """The JSON a request brought, or None (having answered it) if it's too long or not JSON."""
        if not self._trusted():
            return None
        if "application/json" not in (self.headers.get("Content-Type") or ""):  # other sites can't send this
            self._json({"error": "send JSON"}, HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
            return None
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self._json({"error": "too long"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return None
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "not JSON"}, HTTPStatus.BAD_REQUEST)
            return None
        if not isinstance(body, dict):
            self._json({"error": "not a JSON object"}, HTTPStatus.BAD_REQUEST)
            return None
        return body

    def do_POST(self) -> None:
        body = self._body()
        if body is None:
            return
        life = self.life
        if self.path == "/api/say":
            text = str(body.get("text", ""))[:500]
            self._json({"heard": life.say(text)})
        elif self.path == "/api/touch":
            life.touch()
            self._json({"ok": True})
        elif self.path == "/api/feed":
            life.feed()
            self._json({"ok": True})
        elif self.path == "/api/pause":
            life.paused = bool(body.get("paused", not life.paused))
            self._json({"paused": life.paused})
        elif self.path == "/api/speed":
            life.speed = float(min(max(float(body.get("speed", life.speed)), 0.5), 200))
            self._json({"speed": life.speed})
        else:
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _json(self, data: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(status, json.dumps(data, default=plain).encode(), "application/json")

    def _send(self, status: HTTPStatus, body: bytes, kind: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def plain(value: object) -> object:
    """numpy numbers and arrays, as JSON."""
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"can't send a {type(value).__name__} as JSON")


def serve(life: Life, port: int = 8765, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """Start the dashboard in a background thread and return the server (call .shutdown() to stop)."""
    handler = type("LifeHandler", (Handler,), {"life": life})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="haven-dashboard", daemon=True).start()
    return server
