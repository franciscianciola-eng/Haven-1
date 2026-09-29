import pytest

pytest.importorskip("torch")

from haven.cortex import starter
from haven.cortex.think import OwnThinker, agreement, make_thinker, topic_of
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


def test_no_cortex_yet(tmp_path, monkeypatch):
    monkeypatch.setattr(starter, "FOLDER", tmp_path / "no starter")  # a copy of Haven that came without one
    thinker, message = make_thinker("own", tmp_path)
    assert thinker is None and "haven learn" in message
    assert make_thinker("none", tmp_path) == (None, "")
    assert make_thinker("someone else's", tmp_path)[0] is None  # its language cortex is only ever its own
