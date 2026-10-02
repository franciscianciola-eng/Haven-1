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


def test_it_goes_over_the_story_it_heard_in_its_sleep_and_tells_it(cortex_home, monkeypatch):
    from haven.cortex import hearing, sleep, talk

    passage = " ".join(f"A little hen sat on her nest by the barn, day {i}. She sang to her eggs." for i in range(40))
    book = f"*** START OF THE PROJECT GUTENBERG EBOOK HEN ***\n\n{passage}\n\n{passage}\n\n{passage}\n\n*** END"
    monkeypatch.setattr(sleep, "STEPS", 3)
    monkeypatch.setattr(hearing, "book", lambda web, repo, number: book)
    monkeypatch.setattr(hearing, "SHELF", (7,))
    monkeypatch.setattr(hearing, "repos", lambda: {7: "The-Tale-of-a-Hen_7"})
    thinker, _ = make_thinker("own", cortex_home, web=Web(delay=0, allow_private=True))
    life = Life(Mind(seed=3), None)
    events = []
    life.listeners.append(lambda kind, text: events.append(text))
    story = thinker.bedtime_story(life)
    assert story.title == "The Tale of a Hen" and events == ["heard a bedtime story: The Tale of a Hen"]
    thinker.hearing.heard_said("Hello, little one")
    for _ in range(45):
        life.mind.live(10)
        thinker.notice(life.mind)
    report = thinker.sleep_on_it(life)
    assert report["heard"] >= 1 and report["said to it"] == 1 and "following" in report["after"]
    assert "the story it heard" in events[-1] and thinker.hearing.tonight == []
    assert thinker.recollect(life.mind, "Tell me a story.") == [talk.tale_note(story)]  # the one it heard
    heard = thinker.recollect(life.mind, "What did you hear last night?")
    assert heard[0].startswith("The last story I heard was The Tale of a Hen")
    assert "words here" in thinker.describe()


def test_a_long_telling_may_say_a_phrase_twice_but_not_three_times():
    from haven.cortex.think import repeats

    story = "The fox ran to the wood. The fox ran home. Then it slept."
    assert repeats(story) and not repeats(story, times=2)
    assert repeats(story + " The fox ran away.", times=2)


def test_it_knows_when_it_says_again_what_it_said_just_now(cortex_home, monkeypatch):
    thinker, _ = make_thinker("own", cortex_home, web=Web(delay=0, allow_private=True))
    life = Life(Mind(seed=1), None)
    life.conversation += [
        {"tick": 0, "who": "you", "text": "What's your name?"},
        {"tick": 0, "who": "haven", "text": "My name is Haven."},
        {"tick": 0, "who": "you", "text": "What's your name?"},
    ]
    minded = []  # (what came to mind, each time it put a reply into words)

    def say(prompt, state, drafts, most=100, heard=None):
        minded.append(thinker.tok.decode(prompt))
        return ("My name is Haven." if len(minded) == 1 else "Like I said, my name is Haven."), 0.9

    monkeypatch.setattr(thinker, "_say", say)
    answer, _ = thinker.deliberate(life, "What's your name?")
    assert answer == "Like I said, my name is Haven."
    assert "I said that just now: My name is Haven." in minded[1] and "I said that just now" not in minded[0]


def test_it_notices_when_it_tells_as_read_what_it_didnt_read():
    from haven.cortex.think import misread

    came = "I read about World War II: Japan formally surrendered on September 2, 1945."
    assert not misread("I read about World War II. It says: Japan formally surrendered on September 2, 1945.", came)
    assert misread("I read about World War II. It says: Japan formally surreamed on September 2, 1945.", came)
    assert not misread("I feel cuddly, because you stroked me.", came)  # (it isn't telling what it read)
    more = came + " More that I read about World War II: The war ended with an Allied victory."
    assert not misread("It also says: The war ended with an Allied victory.", more)


def test_what_it_read_about_something_new_it_tells_as_if_they_had_only_just_asked(cortex_home, monkeypatch):
    from dataclasses import replace

    thinker, _ = make_thinker("own", cortex_home, web=Web(delay=0, allow_private=True))
    life = Life(Mind(seed=1), None)
    minded = []  # (what came to mind, and what was said in view, each time it put a reply into words)

    def say(prompt, state, drafts, most=100, heard=None):
        minded.append(thinker.tok.decode(prompt))
        return "Okay.", 0.9

    monkeypatch.setattr(thinker, "_say", say)
    monkeypatch.setattr(thinker.model, "cfg", replace(thinker.model.cfg, context=4096))  # (room for all of it)
    answers = ("Answer number one.", "Answer number two.", "Answer number three.", "Answer number four.")
    for text, answer in zip(("What's your name?", "What is the moon?", "Tell me more.", "How are you?"), answers):
        life.conversation.append({"tick": 0, "who": "you", "text": text})
        thinker.deliberate(life, text)
        life.conversation.append({"tick": 0, "who": "haven", "text": answer})
    _, moon, more, feeling = minded
    assert "I read about Moon" in moon and answers[0] not in moon  # (something new it read: as if they just asked)
    assert answers[1] in more and "I read about Moon" in more  # (asked for more: what it said, and what it read)
    assert answers[2] in feeling  # (anything else: with what was said lately in view)
