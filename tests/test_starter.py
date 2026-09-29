"""The language cortex Haven is born with: its own, and it answers from its own state."""

import random

import pytest

pytest.importorskip("torch")

from haven.cortex import starter
from haven.cortex.curriculum import ScratchReader, _plain
from haven.cortex.grounding import gather
from haven.cortex.think import OwnThinker, make_thinker

pytestmark = pytest.mark.skipif(not starter.available(), reason="this copy of Haven came without its starter cortex")


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


def test_it_says_how_it_feels_from_its_state(reader, moments):
    rng = random.Random(0)
    asked = rng.sample(moments, 60)
    right = sum(
        _plain(reader.reply(m["state"], m["notes"], "How are you feeling?")) == _plain(m["answers"]["feel"])
        for m in asked
    )
    assert right / len(asked) >= 0.6


def test_it_answers_people(reader, moments):
    m = moments[-1]
    assert _plain(reader.reply(m["state"], m["notes"], "What's your name?")) == "my name is haven"
    assert _plain(reader.reply(m["state"], m["notes"], "Where are you?")) == _plain(m["answers"]["place"])
    assert _plain(reader.reply(m["state"], m["notes"], "What is the capital of France?")).startswith("i don't know")
