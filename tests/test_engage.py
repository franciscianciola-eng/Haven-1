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
    assert engage.learned_from("What does Rex look like?", "Chase balls.") is None  # (an answer to another question)


def test_babble_and_a_favorites_reason():
    from haven.cortex.think import Thread, garbled, misnamed, misworded

    assert garbled("Tababababababababa") and not garbled("Octopuses live in the sea. Thank you for telling me!")
    assert misworded("A octopuses eat little crabs.") and misworded("I rang an bell.")
    assert not misworded("An octopus eats crabs. I saw a unicorn, an hour ago, and a bell.")
    told = "My favorite animal is an octopus"
    assert misnamed("Oh, actopus! Where do octopuses live?", told) and misnamed("Oh, a lion!", told)
    assert not misnamed("Ooh, an octopus! Where do octopuses live?", told) and not misnamed("Ooh, octopuses!", told)
    talk = Thread()
    talk.mine("My favorite animal is the butterfly. They fly around my valley.")
    assert engage.why_reply(talk.reason) == "Because butterflies fly around my valley."


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
    assert engage.mood_note({**usual, "oxytocin": 3.0}, "you stroked me") == "I feel cuddly, because you stroked me."


def test_it_knows_why_it_feels_as_it_does_or_that_it_doesnt():
    stirred = {"oxytocin+": (100, "you stroked me"), "dopamine+": (10, "I ate an apple")}
    assert engage.mood_cause("cuddly", stirred, 150) == "you stroked me"
    assert engage.mood_cause("cuddly", stirred, 100 + engage.LATELY + 1) is None  # (too long ago to be why)
    assert engage.mood_cause("on edge", stirred, 150) is None and engage.mood_cause(None, stirred, 150) is None
    assert engage.feel_reply("cuddly", "cuddly", "you stroked me") == "Because you stroked me."
    assert engage.feel_reply("that", "cuddly", "you stroked me") == "Because you stroked me."
    assert engage.feel_reply("cuddly", "cuddly", None) == "I don't know. I just feel that way."
    assert engage.feel_reply("a bit grumpy", "cuddly", None, "grumpy") == "I'm not grumpy. I feel cuddly."
    assert engage.feel_reply("a bit grumpy", None, None, "cross") == "I'm not cross."
    rng = random.Random(0)
    assert all(engage.synthetic_cause("dreamy", rng) in (None, "I just woke up") for _ in range(20))


def test_it_says_again_what_it_said_just_now_as_such():
    mine = "My favorite animal is the butterfly. They fly around my valley."
    assert engage.said_again(mine) == "Like I said, my favorite animal is the butterfly. They fly around my valley."
    assert engage.said_again("Yes, I'm very hungry.") == "Like I said, I'm very hungry."
    assert engage.said_again("I'm 2 days old.", "a bit grumpy") == "I told you already. I'm 2 days old."
    earlier = [f"Ooh, an octopus! {mine} Where do octopuses live?", "Your name is Sam."]
    assert engage.said_before(mine, earlier) and engage.said_before("your name is sam", earlier)
    assert not engage.said_before("My favorite food is apples.", earlier)
    assert not engage.said_before("Okay.", ["Okay."])  # (short, everyday replies don't count)
    assert not engage.said_before(
        "I don't know. I haven't learned about that.", ["I don't know. I haven't learned about that."]
    )


def test_what_it_learns_brings_its_own_valley_to_mind():
    rng = random.Random(0)
    sea = "I've never seen the sea. I only have my pond."
    assert engage.relate("octopuses live in the sea") == sea and engage.relate("Rex looks small and brown") is None
    assert (
        engage.learned_reply("octopuses live in the sea", None, rng, None, sea) == f"Octopuses live in the sea! {sea}"
    )
    flat = engage.learned_reply("octopuses live in the sea", None, rng, "a bit grumpy", sea)
    assert flat == "Octopuses live in the sea. Hm. Okay."  # (not in the mood to chat)
    assert engage.learned_reply("Rex looks small", None, rng, "cuddly").endswith("Aww, thank you for telling me!")


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
