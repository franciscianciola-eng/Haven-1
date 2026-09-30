import json

import pytest

pytest.importorskip("torch")

from haven.cortex import starter
from haven.cortex.think import OwnThinker, agreement, make_thinker, topic_of, unfounded
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


def test_no_cortex_yet(tmp_path, monkeypatch):
    monkeypatch.setattr(starter, "FOLDER", tmp_path / "no starter")  # a copy of Haven that came without one
    thinker, message = make_thinker("own", tmp_path)
    assert thinker is None and "haven learn" in message
    assert make_thinker("none", tmp_path) == (None, "")
    assert make_thinker("someone else's", tmp_path)[0] is None  # its language cortex is only ever its own


def test_it_notices_when_its_words_go_round_in_circles():
    from haven.cortex.think import repeats

    assert repeats("I can climb the hill and climb the hill and swim", "I can climb the hill and swim")
    story = "I noticed a new kind of thing: a stone. I noticed a new kind of thing: a flower."
    assert not repeats(story, f"I remember: {story}")  # saying again what came to mind isn't going in circles


def test_it_learns_from_its_day_in_its_sleep_and_keeps_it_only_if_it_helps(cortex_home, monkeypatch):
    import torch

    from haven.cortex import sleep

    monkeypatch.setattr(sleep, "STEPS", 3)
    thinker, _ = make_thinker("own", cortex_home, web=Web(delay=0, allow_private=True))
    life = Life(Mind(seed=2), None)
    events = []
    life.listeners.append(lambda kind, text: events.append(text))
    assert thinker.sleep_on_it(life) is None  # nothing to go over yet
    with (cortex_home / "cortex" / "readings.jsonl").open("a") as f:  # and something it read, to go over too
        f.write(json.dumps({"title": "Frog", "text": "A frog is a small animal. It can jump far.", "why": "curious"}))
        f.write("\n")
    assert [e.title for e in thinker.own_readings()] == ["Frog"]
    for _ in range(45):
        life.mind.live(10)
        thinker.notice(life.mind)
    before = {k: v.clone() for k, v in thinker.model.state_dict().items()}
    report = thinker.sleep_on_it(life)
    assert report["moments"] == 45 and report["steps"] == 3 and set(report["after"]) == {"own day", "other lives"}
    changed = any(not torch.equal(before[k], v) for k, v in thinker.model.state_dict().items())
    assert changed == report["kept"]
    assert (cortex_home / "cortex" / "nights.jsonl").exists() and "in its sleep" in events[-1]
    assert not thinker.day  # a new day starts
    first = json.loads((cortex_home / "cortex" / "sleep.json").read_text())["other lives"]
    assert first == report["before"]["other lives"] and thinker._first_night() == first  # the floor from now on


def test_it_prefers_drafts_it_can_back_up():
    mind = "I worked it out: 12 times 7 is 84. What's 12 times 7? You can call me mehmet."
    assert not unfounded("12 times 7 is 84.", mind) and unfounded("12 times 7 is 844.", mind)
    assert not unfounded("Nice to meet you, Mehmet!", mind) and unfounded("Nice to meet you, Mehmetmet!", mind)
    assert not unfounded("I feel fine. It's a nice day.", mind)
