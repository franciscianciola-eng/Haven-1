"""What comes to Haven's mind when someone talks with it, and the answers it learns to give."""

import random
import re

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
    for said in (
        "My name is Sam",
        "my name is sam",
        "Hi, I'm Sam.",
        "I'm Sam",
        "call me Sam",
        "Hey Haven, I'm Sam",
        "Hi, I'm Sam, nice to meet you!",
    ):
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


def test_every_wording_it_practises_brings_the_memory_it_asks_for_to_mind():
    from haven.cortex.talk import ASKED_FOR, QUESTIONS

    asks = {"favorite place": "favorites", "favorite season": "favorites", "afraid": "fears"}
    asks |= {"best day": "best day", "worst day": "worst day"}  # (memories that only come to mind when asked for)
    for intent, piece in asks.items():
        for question in QUESTIONS[intent]:
            assert re.search(ASKED_FOR[piece], question.lower()), (intent, question)


def test_conversations_to_learn_from_are_answerable_from_what_comes_to_mind(lived):
    moment = {"memo": memo(lived), "answers": answers(lived)}
    rng = random.Random(0)
    kinds = set()
    for _ in range(600):
        thought, turns = conversation(moment, rng)
        kinds |= {turn.kind for turn in turns}
        for turn in turns:
            if turn.kind == "their name" and turn.answer.startswith(("Your name is", "Nice to meet you")):
                assert turn.answer.split()[-1].strip(".!") in thought
            if turn.kind == "what they told it" and "You told me that" in turn.answer:
                told = turn.answer.split("You told me that ", 1)[1].removesuffix(".")
                for fact in re.split(r",(?: and)? that ", told):  # everything it knows about them, or one thing
                    assert f"You told me that {fact}." in thought
            if turn.kind in ("what it was taught", "alive"):  # (what it found out about itself comes to mind)
                assert turn.answer in thought
            if turn.kind == "request":
                assert "You asked me to " in thought
            if turn.kind == "more" and turn.answer.startswith("It also says: "):
                assert turn.answer.removeprefix("It also says: ") in thought
            if turn.kind == "being told":  # what it'll remember, or what it wonders and has of its own, is in mind
                if turn.answer.startswith("Okay, I'll remember that "):
                    assert turn.answer.removeprefix("Okay, I'll remember that ").rstrip(".") in thought
                else:
                    asked = re.findall(r"[A-Z][^.!?]*\?", turn.answer)
                    assert all(f"I wonder: {q}" in thought for q in asked), (turn.answer, thought)
                    assert "You just told me that " in thought
            if turn.kind == "learning":  # what it learned, and what that brings to mind from its own valley
                learned = re.split(r"[.!]", turn.answer)[0]
                assert f"you just told me that {learned.lower()}." in thought.lower()
                related = re.match(r"[^.!]*! (.*?)(?: [A-Z][^.!?]*\?)?$", turn.answer)
                if related:
                    assert f"From my own life: {related.group(1)}" in thought, (turn.answer, thought)
            if turn.kind == "said again":  # it knows it's saying it again
                again = turn.answer.removeprefix("Like I said, ").removeprefix("I told you already. ")
                assert again.lower() in thought.lower() and "I said that just now: " in thought
                first = [t for t in turns if t is not turn and t.answer.lower().endswith(again.lower())]
                assert any(not t.learn for t in first)  # (the time before is there to read, not to learn)
            if turn.kind == "why it feels":  # how it feels, and why, is in mind (or it isn't: then it says so)
                felt = re.search(r"I feel ([a-z ]+?)(?:, because ([^.]+))?\.", thought)
                if turn.answer.startswith("Because "):
                    assert felt and felt.group(2) and turn.answer == f"Because {felt.group(2)}."
                elif turn.answer.startswith("I don't know"):
                    assert felt and not felt.group(2)
                else:
                    assert turn.answer.startswith(("I'm not ", "I feel okay"))
            if turn.kind in ("why", "about you", "its brain", "pastime"):
                assert any(
                    n in thought for n in ("Why I said that:", "They want to know", "My brain:", "Right now I'm")
                )
            if turn.kind == "about you" and turn.answer.startswith(("Like I said", "I told you already")):
                assert "I said that just now: " in thought  # (and what it said first is there to read)
                assert any(not t.learn for t in turns[: turns.index(turn)])
            if turn.kind == "feel" and ", because " in turn.answer:  # why it feels as it does is in mind
                assert turn.answer.split("I feel ", 1)[1] in thought
            if turn.kind in ("what it read", "a fact"):
                title = turn.answer.removeprefix("I read about ").split(". It says: ")[0]
                assert f"I read about {title}:" in thought
            if turn.kind == "sums":
                worked, result = turn.answer.removesuffix(".").rsplit(" is ", 1)
                assert f"I worked it out: {worked[0].lower()}{worked[1:]} is {result}." in thought
            if turn.kind == "lately":
                assert turn.answer.replace("I read about", "Lately I read about") in thought
            if turn.kind == "what it doesn't know":
                assert turn.answer == DONT_KNOW
    assert {"request", "more", "a word", "sums", "lately", "a fact", "what they told it", "what it was taught"} <= kinds
    assert {"said again", "why it feels", "learning"} <= kinds
    for _ in range(500):  # a conversation always has the turns asked for, even when a try comes to nothing
        assert conversation(moment, rng, turns=1)[1]


def test_the_mind_keeps_names_and_facts(lived):
    mind = Mind(seed=1)
    mind.person, mind.told = "Ada", [(5, "you like tea")]
    restored = Mind(seed=2)
    restored.load_state(mind.to_state())
    assert (restored.person, restored.told) == ("Ada", [(5, "you like tea")])
    assert "I'm talking with Ada." in notes(restored, "hello")


def test_what_they_just_said_comes_to_mind_as_just_said():
    mind = Mind(seed=1)
    mind.told = [(1, "you like tea"), (2, "you live in Lisbon")]
    thought = notes(mind, "I live in Lisbon.", just="you live in Lisbon")
    assert "You just told me that you live in Lisbon." in thought and "You told me that you live" not in thought


def test_it_is_born_having_read_a_little_book(tmp_path):
    from haven.cortex.think import Thinker

    read = Thinker(tmp_path).readings()
    assert ("Moon", "The Moon is the Earth's only natural satellite.") in read
    assert best_reading("tell me about dinosaurs", read)[0] == "Dinosaur"


def test_its_story_tells_its_firsts_that_matter(lived):
    from haven.cortex.talk import story

    told = story(lived)
    assert told.startswith("I came into the world") and told.count(".") <= 4
    notable = [t for _, t in lived.me.milestones[1:] if not t.startswith("noticed a new kind")]
    if len(notable) >= 3:
        assert "noticed a new kind" not in told


def test_it_reads_about_what_it_is_curious_about(tmp_path, internet, monkeypatch):
    from haven.cortex import sources
    from haven.cortex.library import topics_in
    from haven.cortex.think import Thinker
    from haven.life import Life
    from haven.web import Web

    assert topics_in("I went to Paris with Maria last summer") == ["Paris"]  # not people's names
    assert topics_in("My favorite food is sushi and I have a dog named Rex.") == ["sushi", "food", "dog"]
    monkeypatch.setitem(sources.URLS, "simplewiki", internet["simplewiki"])
    mind = Mind(seed=1)
    mind.person = "Sam"
    mind.things = {"butterfly": {"seen": 3}, "bell": {"seen": 9, "rang": 2}}
    life = Life(mind, None)
    life.conversation = [{"tick": 0, "who": "you", "text": "Hi, I'm Sam. I live in Boston and I love chess."}]
    thinker = Thinker(tmp_path, web=Web(delay=0, allow_private=True))
    events = []
    life.listeners.append(lambda kind, text: events.append(text))
    assert thinker.wonder(life) == "Boston"  # chess it has read about already, in its little book
    assert [thinker.read_for_fun(life) for _ in range(3)] == ["Boston", "Bell", "Butterfly"]
    assert "read about Boston, out of curiosity" in events
    assert [text for _, text in life.news][:1] == ["read about Boston, out of curiosity"]  # the app shows it
    assert life.snapshot()["news"][0]["text"] == "read about Boston, out of curiosity"
    assert thinker.library.find("Tell me about Boston") == ("Boston", 0)
    assert Thinker(tmp_path).library.sources["Bell"] == "curious"  # it keeps what it read

    life.thinker, before = thinker, life._last_wonder
    assert life.snapshot()["read"] == ["Butterfly", "Bell", "Boston"]  # the latest first, for the app
    life.say("hello")
    life._wonder()  # someone's talking with it: not now
    assert life._last_wonder == before


def test_it_works_out_sums():
    from haven.cortex.talk import sum_answer, sum_of

    assert sum_of("What's 12 times 7?") == ("12 times 7", "84")
    assert sum_of("hey haven, what is two plus two") == ("2 plus 2", "4")
    assert sum_of("How much is 100 divided by 8?") == ("100 divided by 8", "12.5")
    assert sum_answer(sum_of("9 - 12")) == "9 minus 12 is -3."
    for said in ("What is the capital of France?", "two and two", "What's 5 divided by 0?", "I'm 7"):
        assert sum_of(said) is None, said


def test_it_can_be_taught_about_the_world():
    from haven.cortex.talk import best_lesson, lesson

    assert lesson("The capital of Peru is Lima.") == "the capital of Peru is Lima"
    assert lesson("hey haven, frogs can jump very far") == "frogs can jump very far"
    assert lesson("No, Mark Twain wrote Tom Sawyer.") == "Mark Twain wrote Tom Sawyer"
    for said in ("I have a dog named Rex", "That's a frog.", "You are cute", "Ring the bell.", "What is 2 plus 2?"):
        assert lesson(said) is None, said
    taught = ["the capital of Peru is Lima", "frogs can jump very far", "Mark Twain wrote Tom Sawyer"]
    assert best_lesson("What's the capital city of Peru?", taught) == "the capital of Peru is Lima"
    assert best_lesson("What did Mark Twain write?", taught) == "Mark Twain wrote Tom Sawyer"
    assert best_lesson("What is the capital of France?", taught) is None


def test_questions_about_it_are_not_about_what_it_read():
    from haven.cortex.talk import about_haven

    for said in ("Who made you?", "Is it day or night?", "Do you speak English?", "What should you stay away from?"):
        assert about_haven(said), said
    for said in (
        "What do you know about volcanoes?",
        "Do you know who wrote Hamlet?",
        "What is the capital of France?",
    ):
        assert not about_haven(said), said


def test_asked_about_itself_it_says_what_it_has_found_out(lived):
    from haven.cortex.talk import found_out

    said = found_out(lived)
    assert said.endswith(lived.me.conclusions()[-1])  # what it makes of it
    assert said in recall(memo(lived), "Are you alive?") and said not in recall(memo(lived), "How are you?")
    assert answers(lived)["alive"] == said
    assert answers(lived)["what"].endswith(answers(lived)["personality"])  # what it is: a creature, and what it's like
