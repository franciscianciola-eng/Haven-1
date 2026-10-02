"""The page, served as GitHub Pages serves it (the repository's docs/ folder, at /docs/), and a stand-in for the Simple
English Wikipedia's API at /w/api.php (fake_wiki.py, with the articles of tests/fakes.py), for tests of the page in a
real browser (page.e2e.mjs). Prints the port it's on.

    python tests/web/serve.py [--port 8000]
"""

from __future__ import annotations

import argparse
import functools
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "tests")]

from fake_wiki import FakeWiki
from fakes import ARTICLES

WIKI = FakeWiki(ARTICLES, {"World War 2": "World War II"})


class Handler(SimpleHTTPRequestHandler):
    extensions_map: ClassVar[dict[str, str]] = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".mjs": "text/javascript",
        ".js": "text/javascript",
        ".wasm": "application/wasm",
        ".json": "application/json",
        ".onnx": "application/octet-stream",
    }

    def do_GET(self):
        if self.path.startswith("/w/api.php"):
            body = WIKI(self.path).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, *_args):
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), functools.partial(Handler, directory=str(ROOT)))
    print(server.server_address[1], flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
