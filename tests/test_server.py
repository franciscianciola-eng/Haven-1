import json
import urllib.request

from haven.life import Life
from haven.mind import Mind
from haven.server import serve
from haven.store import Store

NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def call(port, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    with NO_PROXY.open(request, timeout=30) as response:
        raw = response.read()
        return raw if path == "/" else json.loads(raw)


def test_dashboard_api(tmp_path):
    life = Life(Mind(seed=3), Store(tmp_path), speed=200)
    server = serve(life, port=0)
    port = server.server_port
    try:
        life.start()
        assert b"<title>Haven</title>" in call(port, "/")
        state = call(port, "/api/state")
        assert state["name"] == "Haven" and "readout" in state and state["welfare"]["ok"]
        assert call(port, "/api/say", {"text": "hello Haven"})["heard"] == ["hello", "haven"]
        assert call(port, "/api/touch", {})["ok"] and call(port, "/api/feed", {})["ok"]
        assert call(port, "/api/pause", {"paused": True})["paused"] is True
        assert call(port, "/api/speed", {"speed": 1000})["speed"] == 200
        assert len(call(port, "/api/check")["indicators"]) == 14
    finally:
        life.stop()
        server.shutdown()
    assert Store(tmp_path).exists()  # stopping saved the life
