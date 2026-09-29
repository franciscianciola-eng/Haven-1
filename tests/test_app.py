"""The app: Haven's window, where you talk with it and can watch it think."""

import argparse
import io
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

pytest.importorskip("torch")

from haven import __version__
from haven.app import Chat, let_rest, running, serve
from haven.cli import Terminal
from haven.cortex.think import Thinker
from haven.life import Life
from haven.mind import Mind
from haven.store import Store

NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def call(port, path, body=None, headers=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data, headers={"Content-Type": "application/json", **(headers or {})}
    )
    with NO_PROXY.open(request, timeout=30) as response:
        raw = response.read()
        return raw if not path.startswith("/api") else json.loads(raw)


def status(port, path, body=None, headers=None):
    try:
        call(port, path, body, headers)
    except urllib.error.HTTPError as error:
        return error.code
    return 200


class Stand(Thinker):
    """A stand-in cortex: it answers a piece at a time, thinking it over on the way, once it's allowed to."""

    name = "a stand-in"

    def __init__(self, root):
        super().__init__(root)
        self.go = threading.Event()

    def deliberate(self, life, text):
        self._tell("draft")
        for piece in ("I ", "feel ", "fine."):
            self._tell("words", piece)
        self._tell("pondering")
        self._tell("thought", "They asked how I feel.")
        self.go.wait(10)
        self._tell("draft")
        self._tell("words", "I feel fine, thank you.")
        return "I feel fine, thank you.", 0.8


@pytest.fixture
def app(tmp_path):
    life = Life(Mind(seed=4), Store(tmp_path), speed=50)
    chat = Chat(life, log=lambda s: None)
    chat.thinker = Stand(tmp_path)
    chat.status = {"stage": "ready", "text": "ready"}
    server = serve(life, chat, port=0)
    life.start()
    yield life, chat, server.server_port
    chat.thinker.go.set()
    life.stop()
    server.shutdown()


def wait_for(port, number):
    for _ in range(200):
        turn = call(port, f"/api/chat/{number}")
        if turn["done"]:
            return turn
        time.sleep(0.05)
    raise AssertionError("it never answered")


def test_talking_in_the_app(app, tmp_path):
    life, chat, port = app
    assert b"<title>Haven</title>" in call(port, "/") and b"Talk to" in call(port, "/")
    assert b"/static/world3d.js" in call(port, "/")  # its valley, in 3D
    assert b"Stream of consciousness" in call(port, "/dashboard")  # the full view of its mind is there too
    assert call(port, "/api/chat")["stage"] == "ready"
    turn = call(port, "/api/chat", {"text": "how do you feel?"})
    assert turn["done"] is False
    assert status(port, "/api/chat", {"text": "and now?"}) == 409  # one thing at a time
    time.sleep(0.3)
    following = call(port, f"/api/chat/{turn['id']}")  # its reply, as it takes shape
    assert following["phase"] == "thinking it over"
    assert [t["text"] for t in following["thoughts"]] == ["They asked how I feel."]
    assert following["draft"] == "I feel fine."
    chat.thinker.go.set()
    done = wait_for(port, turn["id"])
    assert done["reply"] == "I feel fine, thank you." and done["confidence"] == 0.8
    kinds = [t["kind"] for t in done["thoughts"]]
    assert kinds == ["thought", "draft"]  # the first words that came to it are kept, as a passing thought
    said = [(c["who"], c["text"]) for c in life.conversation]
    assert said[-2:] == [("you", "how do you feel?"), ("haven", "I feel fine, thank you.")]
    remembered = [json.loads(line) for line in (tmp_path / "cortex" / "conversations.jsonl").read_text().splitlines()]
    assert remembered[-1]["haven"] == "I feel fine, thank you."
    assert call(port, "/api/state")["conversation"][-1]["text"] == "I feel fine, thank you."
    assert status(port, "/api/chat/99") == 404


def test_only_this_computer_can_use_it(app):
    _, chat, port = app
    assert status(port, "/api/state", headers={"Host": "attacker.example"}) == 403  # a site pointed at this computer
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/chat", data=b'{"text": "hi"}', headers={"Content-Type": "text/plain"}
    )
    with pytest.raises(urllib.error.HTTPError) as refused:  # what another site's page could send without asking
        NO_PROXY.open(request, timeout=30)
    assert refused.value.code == 415
    assert not chat.turns


def test_resting_and_waking(app, tmp_path):
    _, chat, port = app
    assert running(port)
    out = io.StringIO()
    args = argparse.Namespace(port=port, no_browser=True)
    from haven.app import run

    assert run(args, Store(tmp_path), Terminal(out)) == 0  # a second start just goes to the one that's awake
    assert "already awake" in out.getvalue()
    assert call(port, "/api/rest", {})["ok"]
    assert chat.resting.wait(5)  # it answers first, then lets Haven rest


def test_an_older_haven_rests_for_a_newer_one(tmp_path):
    life = Life(Mind(seed=4), Store(tmp_path))
    chat = Chat(life, log=lambda s: None)
    server = serve(life, chat, port=0)
    port = server.server_port
    assert running(port)["version"] == __version__  # so a newer version can tell it's older

    def when_let_rest():
        if chat.resting.wait(10):  # as `haven app` does: it saves, rests and closes its window
            server.shutdown()
            server.server_close()

    threading.Thread(target=when_let_rest, daemon=True).start()
    let_rest(port, wait=10)
    assert chat.resting.is_set() and running(port) is None


def test_not_ready_yet(tmp_path):
    life = Life(Mind(seed=4), Store(tmp_path))
    chat = Chat(life, log=lambda s: None)
    server = serve(life, chat, port=0)
    try:
        assert status(server.server_port, "/api/chat", {"text": "hello"}) == 503
    finally:
        server.shutdown()


def test_the_app_gives_it_its_own_cortex(tmp_path):
    """The first time, it gets the language cortex it's born with (its own), and then it can talk."""
    from haven.cortex import starter
    from haven.cortex.think import OwnThinker

    if not starter.available() or not starter.fits(starter.FOLDER / "cortex.pt"):
        pytest.skip("this copy of Haven came without its starter cortex (or with an out-of-date one)")
    life = Life(Mind(seed=4), Store(tmp_path / "home"))
    chat = Chat(life, log=lambda s: None)
    chat.start()
    for _ in range(300):
        if chat.status["stage"] in ("ready", "error"):
            break
        time.sleep(0.1)
    assert chat.status["stage"] == "ready", chat.status
    assert isinstance(chat.thinker, OwnThinker) and life.thinker is chat.thinker
    assert "grown from scratch" in chat.thinker.describe()
    turn = chat.ask("What's your name?")
    for _ in range(300):
        if chat.turn(turn["id"])["done"]:
            break
        time.sleep(0.1)
    done = chat.turn(turn["id"])
    assert done["done"] and "Haven" in done["reply"]
    assert [t["kind"] for t in done["thoughts"]] == ["draft", "draft"]  # the other words that came to it
