"""Following a conversation, and deciding for itself: what comes to mind for it, and what it learns to say."""

import random

from haven.cortex import engage
from haven.cortex.talk import request
from haven.mind import Mind
from haven.will import consider


def test_being_told_something_makes_it_wonder():
    asks = engage.wonders("your favorite animal is an octopus")
    assert "Where do octopuses live?" in asks and "Have you ever seen an octopus?" in asks
    assert engage.wonders("you have a dog named Max")[0] == "What does Max like to do?"
    assert "What's it like in Paris?" in engage.wonders("you live in Paris")
    assert "Is the piano hard?" in engage.wonders("you play the piano")
    assert engage.wonders("the sky is blue") == []  # (not something about them that it wonders about)


def test_the_answer_to_what_it_wondered_is_something_learned():
    assert engage.learned_from("Where do octopuses live?", "in the sea") == "octopuses live in the sea"
    assert engage.learned_from("Where do octopuses live?", "They live in the ocean.") == "octopuses live in the ocean"
    assert engage.learned_from("What does pizza taste like?", "Yummy!") == "pizza tastes yummy"
    assert engage.learned_from("Where do octopuses live?", "why do you ask?") is None
    assert engage.learned_from("What does a plumber do?", "They fix pipes.") == "a plumber fixes pipes"
    assert engage.learned_from("What does Max like to do?", "Run around.") == "Max likes to run around"


def test_it_reacts_says_its_own_and_asks():
    rng = random.Random(0)
    said = engage.told_reply(
        "your favorite animal is an octopus",
        "Ooh!",
        "My favorite animal is the butterfly. They fly around my valley.",
        "Where do octopuses live?",
        rng,
    )
    assert (
        said
        == "Ooh, an octopus! My favorite animal is the butterfly. They fly around my valley. Where do octopuses live?"
    )
    assert engage.told_reply("you like cats", "Oh!", None, None, rng).endswith("I'll remember that you like cats.")


def test_its_mood_comes_from_its_brains_chemistry():
    usual = dict.fromkeys(("dopamine", "noradrenaline", "serotonin", "acetylcholine", "oxytocin"), 1.0)
    assert engage.mood_words(usual) is None
    assert engage.mood_words({**usual, "dopamine": 2.4}) == "full of beans"
    assert engage.mood_words({**usual, "serotonin": 0.3}) == "a bit grumpy"
    assert engage.mood_note({**usual, "oxytocin": 3.0}) == "I feel cuddly."


def test_it_decides_for_itself():
    rng = random.Random(0)
    base = {
        "fear": 0.0, "fear_reason": "The fire burned me", "need": 0.1, "need_words": "hungry", "meets_need": False,
        "tired": 0.1, "sleep_asked": False, "bond": 0.7, "appeal": 0.7, "bored": 0.0, "done_lately": "rung the bell",
        "busy": 0.0, "busy_words": "busy", "rather": "dance", "rather_urge": 0.3, "dopamine": 1.0, "friendly": 0.6,
        "insisted": False, "rng": rng,
    }  # fmt: skip
    assert engage.decide(**base).answer == "glad"
    assert engage.decide(**{**base, "fear": 0.9}).answer == "scared"
    assert engage.decide(**{**base, "need": 0.8}).answer == "later"
    bored = engage.decide(**{**base, "bored": 1.0, "appeal": 0.1, "bond": 0.2})
    assert bored.answer == "bored" and bored.instead == "dance"
    assert engage.decide(**{**base, "bored": 0.5, "appeal": 0.1, "bond": 0.6, "insisted": True}).answer == "persuaded"
    note = engage.decision_note("go to the fire", engage.decide(**{**base, "fear": 0.9}))
    assert note == "You asked me to go to the fire. I don't want to: The fire burned me."
    assert engage.why_reply("The fire burned me") == "Because the fire burned me."


def test_a_haven_that_got_burned_wont_go_to_the_fire():
    mind = Mind(seed=4)
    mind.things["fire"] = {"hurt": 4.0, "x": 3.0, "y": 3.0}
    decision = consider(mind, request("Go to the fire."), rng=random.Random(0))
    assert decision.answer == "scared" and "burned" in decision.reason


def test_it_follows_what_was_said_with_what_comes_to_mind():
    rng = random.Random(3)
    practice = engage.told_practice(
        "I have a dog named Max.", "you have a dog named Max", "full of beans", 1.0, {}, rng
    )
    first = practice.turns[0][1]
    assert first.startswith("Ooh")
    for asked in [q for q in engage.wonders("you have a dog named Max") if q in first]:
        assert engage.wonder_note(asked) in practice.notes
