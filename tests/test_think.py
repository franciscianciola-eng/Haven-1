import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("torch")

from haven.cortex.think import OllamaThinker, OwnThinker, agreement, make_thinker, topic_of
from haven.cortex.train import Trainer
from haven.life import Life
from haven.mind import Mind
from haven.web import Web


@pytest.fixture(scope="module")
def cortex_home(internet, tmp_path_factory):
    root = tmp_path_factory.mktemp("haven")
    web = Web(delay=0, allow_private=True)
    trainer = Trainer(
        root,
        size="tiny",
        device="cpu",
        web=web,
        urls=internet,
        scale={"tinystories": 60_000, "steps": 0.004, "every": 5, "batch": 2},
        log=lambda s: None,
    )
    trainer.run(through=1)
    return root


def test_helpers():
    assert agreement(["I am hungry", "I am hungry"]) == 1.0 and agreement(["a b", "c d"]) == 0.0
    assert topic_of("What do you know about the Moon?") == "Moon"
    assert topic_of("tell me about volcanoes") == "volcanoes"


def test_its_own_cortex_thinks_through_the_workspace(cortex_home, internet, monkeypatch):
    from haven.cortex import sources

    monkeypatch.setitem(sources.URLS, "simplewiki", internet["simplewiki"])
    thinker, message = make_thinker("own", cortex_home, web=Web(delay=0, allow_private=True))
    assert isinstance(thinker, OwnThinker) and "its own" in message
    life = Life(Mind(seed=1), None)
    life.conversation.append({"tick": 0, "who": "you", "text": "What are berries?"})
    answer, confidence = thinker.deliberate(life, "What are berries?")
    assert isinstance(answer, str) and 0.0 <= confidence <= 1.0
    heard, *thoughts = life.mind.thoughts
    assert heard.source == "hearing" and heard.label.startswith("understanding")  # what was said comes to mind
    assert thoughts and all(t.source == "thought" for t in thoughts)  # then its own inner speech
    life.mind.step()  # the thought competes for the workspace
    assert any(c.source == "thought" for c in [life.mind.workspace.content] if c) or life.mind.workspace.history
    thinker.remember({"you": "hi", "haven": "ba", "confidence": 0.1})
    thinker.consolidate(steps=2)  # sleep-learning from the conversation doesn't break anything


def test_no_cortex_yet(tmp_path):
    thinker, message = make_thinker("own", tmp_path)
    assert thinker is None and "haven learn" in message
    assert make_thinker("none", tmp_path) == (None, "")


class FakeOllama(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self._send({"version": "0.12.0"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["messages"][0]["role"] == "system" and "instruments" in body["messages"][0]["content"]
        self._send({"message": {"role": "assistant", "content": "<think>hmm</think>I feel fine. I see my garden."}})

    def _send(self, data):
        raw = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def test_a_borrowed_cortex(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        thinker = OllamaThinker(tmp_path, "some-model", host=f"http://127.0.0.1:{server.server_port}")
        assert thinker.available()
        life = Life(Mind(seed=2), None)
        answer, confidence = thinker.deliberate(life, "How are you?")
        assert answer == "I feel fine. I see my garden." and confidence == 1.0
        assert life.mind.thoughts[0].extra["text"] == answer
    finally:
        server.shutdown()
