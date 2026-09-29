"""The language cortex Haven is born with: its own, and it answers from its own state and what comes to mind."""

import random

import pytest

pytest.importorskip("torch")

from haven.cortex import starter
from haven.cortex.curriculum import ScratchReader, _plain
from haven.cortex.grounding import gather
from haven.cortex.talk import DONT_KNOW, recall, thing_answer
from haven.cortex.think import OwnThinker, make_thinker

pytestmark = pytest.mark.skipif(
    not starter.available() or not starter.fits(starter.FOLDER / "cortex.pt"),
    reason="this copy of Haven came without its starter cortex (or with an out-of-date one)",
)


@pytest.fixture(scope="module")
def reader(tmp_path_factory):
    root = tmp_path_factory.mktemp("home")
    thinker, message = make_thinker("own", root)
    assert isinstance(thinker, OwnThinker) and (root / "cortex" / "cortex.pt").exists()
    assert "grown from scratch" in message and "talking about itself" in message
    return ScratchReader(thinker.model, thinker.tok)


@pytest.fixture(scope="module")
def moments():
    return gather(seed=4242, days=1.5)  # a life it never lived while it learned


def ask(reader, m, question, **context):
    return _plain(reader.reply(m["state"], recall(m["memo"], question, **context), question))


def test_it_says_how_it_feels_from_its_state(reader, moments):
    rng = random.Random(0)
    asked = rng.sample(moments, 60)
    right = sum(ask(reader, m, "How are you feeling?") == _plain(m["answers"]["feel"]) for m in asked)
    assert right / len(asked) >= 0.6


def test_it_answers_people(reader, moments):
    m = moments[-1]
    assert ask(reader, m, "What's your name?") == "my name is haven"
    assert ask(reader, m, "Where are you?") == _plain(m["answers"]["place"])
    assert ask(reader, m, "What is the capital of France?") == _plain(DONT_KNOW)
    assert ask(reader, m, "What did you do today?").startswith(("today i", "i haven't done much"))


def test_it_talks_about_the_things_in_its_valley(reader, moments):
    m = moments[-1]
    for name, question in (("bush", "Tell me about berries."), ("thorns", "Do you like the thorns?")):
        thing = m["memo"]["things"][name]
        kind = "about" if question.startswith("Tell") else "like"
        assert ask(reader, m, question) == _plain(thing_answer(kind, name, thing["stats"], thing["where"]))


def test_it_remembers_who_you_are_and_what_you_told_it(reader, moments):
    m = moments[-1]
    assert ask(reader, m, "Hi, I'm Marisol.", person="Marisol") == "nice to meet you marisol"
    assert ask(reader, m, "What's my name?", person="Marisol") == "your name is marisol"
    assert ask(reader, m, "What's my name?") == "you haven't told me your name yet"
    told = ["you have a dog named Waffle", "your favorite color is green"]
    assert ask(reader, m, "What's my dog called?", told=told) == "you told me that you have a dog named waffle"
    said = ask(reader, m, "I live in Lisbon.", just="you live in Lisbon")
    assert said == "okay i'll remember that you live in lisbon"


def test_it_remembers_what_it_read(reader, moments):
    m = moments[-1]
    read = [("Moon", "The Moon is the Earth's only natural satellite.")]
    said = ask(reader, m, "What is the moon?", read=read)
    assert said == "i read about moon it says the moon is the earth's only natural satellite"
    assert ask(reader, m, "What is the moon?") == _plain(DONT_KNOW)
