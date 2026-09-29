"""What comes to Haven's mind when someone talks with it, and the answers it learns to give."""

import random

import pytest

from haven.cortex.talk import (
    DONT_KNOW,
    THINGS,
    answers,
    best_fact,
    best_reading,
    conversation,
    first_sentence,
    introduced,
    memo,
    mentioned,
    notes,
    recall,
    second_person,
    statement,
    thing_answer,
)
from haven.mind import Mind


@pytest.fixture(scope="module")
def lived():
    mind = Mind(seed=3)
    mind.live(2400)
    return mind


def test_it_hears_people_say_their_name():
    for said in ("My name is Sam", "my name is sam", "Hi, I'm Sam.", "I'm Sam", "call me Sam", "Hey Haven, I'm Sam"):
        assert introduced(said) == "Sam", said
    for said in ("I'm tired", "I'm fine thanks", "i'm back", "I'm a teacher", "It's cold", "I'm from Oslo"):
        assert introduced(said) is None, said


def test_it_keeps_what_people_tell_it_about_themselves():
    assert statement("I have a dog named Rex.") == "you have a dog named Rex"
    assert statement("my favorite color is blue") == "your favorite color is blue"
    assert statement("I am 30 years old") == "you are 30 years old"
    assert statement("Hey Haven, I live in Boston") == "you live in Boston"
    for said in ("I love you", "I'm tired", "What do I like?", "Hi", "I'm sorry", "I feel sad"):
        assert statement(said) is None, said
    assert second_person("I was born in May") == "you were born in May"


def test_it_recalls_the_right_thing_they_told_it():
    told = ["you have a dog named Rex", "your favorite color is blue", "you're a nurse", "you live in Lima"]
    assert best_fact("What's my dog called?", told) == "you have a dog named Rex"
    assert best_fact("what's my favorite color", told) == "your favorite color is blue"
    assert best_fact("What's my job?", told) == "you're a nurse"
    assert best_fact("Where do I live?", told) == "you live in Lima"
    assert best_fact("Do I have a pet?", told) == "you have a dog named Rex"
    assert best_fact("What's my favorite food?", told) is None  # never said
    assert best_fact("Do you remember what I told you?", told) == "you live in Lima"
    assert best_fact("Tell me about the moon", told) is None  # not about them


def test_it_remembers_what_it_read():
    read = [("Moon", "The Moon is the Earth's only natural satellite."), ("Albert Einstein", "Albert Einstein was…")]
    assert best_reading("What is the moon?", read)[0] == "Moon"
    assert best_reading("Tell me about Einstein", read)[0] == "Albert Einstein"
    assert best_reading("What is the sun?", read) is None
    assert first_sentence("Paris (French: Paris) is the capital of France. It has 2 million people.") == (
        "Paris is the capital of France."
    )


def test_what_comes_to_mind_depends_on_what_is_said(lived):
    known = memo(lived)
    plain = recall(known, "How are you?")
    assert plain.startswith("I'm Haven,") and "The bell:" not in plain
    about = recall(known, "Where is the bell?")
    assert "The bell:" in about and "mound" in about
    assert mentioned("Do you like berries or apples?") == ["bush", "apple"]
    told = recall(known, "What's my name?", person="Ada", told=["you like tea"])
    assert "I'm talking with Ada." in told and "You told me" not in told
    assert "You told me that you like tea." in recall(known, "What do I like?", told=["you like tea"])
    assert "I read about Moon:" in recall(known, "what's the moon", read=[("Moon", "The Moon is a moon.")])


def test_it_knows_the_things_it_has_met(lived):
    known = memo(lived)
    for name, t in THINGS.items():
        stats = known["things"][name]["stats"]
        said = thing_answer("about", name, stats)
        assert said and (("haven't seen" in said) != bool(stats)), (name, said)
    bush = known["things"]["bush"]["stats"]
    if bush.get("ate"):
        assert thing_answer("eat", "bush", bush) == "Yes, they are good to eat."


def test_conversations_to_learn_from_are_answerable_from_what_comes_to_mind(lived):
    moment = {"memo": memo(lived), "answers": answers(lived)}
    rng = random.Random(0)
    for _ in range(300):
        thought, turns = conversation(moment, rng)
        for turn in turns:
            if turn.kind == "their name" and turn.answer.startswith(("Your name is", "Nice to meet you")):
                assert turn.answer.split()[-1].strip(".!") in thought
            if turn.kind == "what they told it" and turn.answer.startswith("You told me that"):
                assert turn.answer in thought
            if turn.kind == "being told":
                assert turn.answer.removeprefix("Okay, I'll remember that ").rstrip(".") in thought
            if turn.kind == "what it read":
                title = turn.answer.removeprefix("I read about ").split(". It says: ")[0]
                assert f"I read about {title}:" in thought
            if turn.kind == "what it doesn't know":
                assert turn.answer == DONT_KNOW


def test_the_mind_keeps_names_and_facts(lived):
    mind = Mind(seed=1)
    mind.person, mind.told = "Ada", [(5, "you like tea")]
    restored = Mind(seed=2)
    restored.load_state(mind.to_state())
    assert (restored.person, restored.told) == ("Ada", [(5, "you like tea")])
    assert "I'm talking with Ada." in notes(restored, "hello")
