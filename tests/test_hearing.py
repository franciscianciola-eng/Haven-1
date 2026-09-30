import json
import random
import re
import time

import pytest

from haven.cortex import hearing, talk
from haven.cortex.stories import STORIES, Story, from_passage, sentences


def fake_book(number: int) -> str:
    rng = random.Random(number)
    names = ["Peter Rabbit", "Reddy Fox", "Jimmy Skunk", "Mrs. Duck"]
    paragraphs = [
        " ".join(
            f"{rng.choice(names)} went down to the {rng.choice(['pond', 'meadow', 'wood'])} one fine morning. "
            f"It was a very happy day for him, and he sang a little song."
            for _ in range(3)
        )
        for _ in range(12)
    ]
    return (
        "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n\n[Illustration: A rabbit.]\n\nCHAPTER I\n\n"
        + "\n\n".join(paragraphs)
        + "\n\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***"
    )


@pytest.fixture
def shelf(monkeypatch):
    got = []

    def book(web, repo, number):
        got.append(number)
        return fake_book(number)

    monkeypatch.setattr(hearing, "book", book)
    monkeypatch.setattr(hearing, "SHELF", (1, 2))
    monkeypatch.setattr(hearing, "repos", lambda: {1: "The-Adventures-of-Peter-Rabbit_1", 2: "Mother-Goose_2"})
    return got


def test_what_was_said_to_children_reads_as_it_was_said():
    assert hearing.utterance("aha i think i see child .") == "Aha I think I see child."
    assert hearing.utterance("who's that ?") == "Who's that?"
    assert hearing.utterance("xxx .") == "" and hearing.utterance("  ") == ""
    stretches = hearing.speech_passages("\n".join(["look at the doggy ."] * 200), size=300)
    assert len(stretches) > 5 and all(s.count("\n") >= 5 for s in stretches)


def test_books_lose_their_notices_pictures_and_word_lists():
    text = (
        "Please take a look at the important information in this header.\n\n*END*THE SMALL PRINT! FOR PUBLIC DOMAIN "
        "ETEXTS*Ver.04.29.93*END*\n\nLESSON I.\n\nNed    eggs    black    left\n\n[Illustration: Boy feeding\na hen.]"
        "\n\nNed has fed the _black_ hen. She has left the nest, and Ned is glad.\n\nHey diddle diddle,\n"
        "The cat and the fiddle.\n\nEnd of the Project Gutenberg Etext of Readers"
    )
    cleaned = hearing.clean_book(text)
    assert (
        "Ned has fed the black hen" in cleaned and "Hey diddle diddle,\nThe cat" in cleaned
    )  # (verse keeps its lines)
    for gone in ("header", "SMALL PRINT", "LESSON", "eggs    black", "Illustration", "Etext"):
        assert gone not in cleaned


def test_stories_it_remembers_and_stories_from_what_it_hears():
    assert len(STORIES) >= 40 and all(s.opening and s.then and s.title for s in STORIES)
    assert sentences("Mr. Fox ran. He was quick!") == ["Mr. Fox ran.", "He was quick!"]
    story = from_passage(
        "A Tale", "Mr. Fox ran to the wood. He was very quick. The hens all ran away. It rained all day."
    )
    assert story.opening == ("Mr. Fox ran to the wood.", "He was very quick.") and len(story.then) == 2


def test_asking_for_a_story_or_what_it_heard():
    for text in (
        "Tell me a story.",
        "can you tell me a story",
        "Do you know any stories?",
        "What stories do you know?",
    ):
        assert talk.story_asked(text) == "tale", text
    for text in ("Tell me a fairy tale!", "Story time! Tell me one.", "I'd like to hear a story"):
        assert talk.story_asked(text) == "tale", text
    for text in ("What did you hear last night?", "Did anyone read you a story?", "What was your bedtime story?"):
        assert talk.story_asked(text) == "heard", text
    for text in ("Tell me your story.", "Tell me the story of your life.", "How are you?"):
        assert talk.story_asked(text) is None, text
    fox = talk.story_for("Tell me a story about a fox", STORIES)
    assert fox is not None and "Fox" in fox.title
    assert talk.story_for("Tell me a story", STORIES) is None
    story = Story("The Tale of a Hen", ("A hen sat.",), ("She laid an egg.",))
    assert talk.tale_note(story) == "A story I know: The Tale of a Hen. It begins: A hen sat."
    assert talk.tale_answer(story) == "Here's a story I heard, The Tale of a Hen. A hen sat. She laid an egg."
    assert talk.heard_note(story, True).startswith("The last story I heard was The Tale of a Hen.")
    assert "Nobody has read me a story here yet" in talk.heard_note(story, False)
    life_story = re.compile(talk.TOPICS["story"])
    assert life_story.search("tell me your story.") and life_story.search("tell me the story of your life")
    assert not life_story.search("tell me a story.")  # a story is something else than its life story


def test_it_learns_to_tell_stories_and_what_it_heard(monkeypatch):
    from haven.cortex.grounding import gather

    monkeypatch.setattr(talk, "STORY_TURNS", 1.0)
    monkeypatch.setattr(talk, "SPEAKING_UP", 0.0)
    moment = gather(seed=5, days=0.3)[-1]
    heard = Story("The Tale of a Hen", ("A hen sat on her nest.",), ("She laid a big egg.",))
    kinds = set()
    for seed in range(40):
        thought, turns = talk.conversation({**moment, "heard": (heard,)}, random.Random(seed), turns=1)
        for turn in turns:
            kinds.add(turn.kind)
            if turn.kind == "a story":
                assert turn.answer.startswith("Here's a story I heard, ") and "It begins:" in thought
            if turn.kind == "what it heard":
                assert turn.answer in thought
    assert {"a story", "what it heard"} <= kinds


def test_it_hears_a_book_at_bedtime_one_passage_at_a_time(tmp_path, shelf):
    ears = hearing.Listening(tmp_path, web=object())
    assert ears.latest() is None and ears.words() == 0
    story, passage = ears.bedtime()
    assert story.title == "The Adventures of Peter Rabbit" and passage.startswith(story.opening[0])
    assert ears.tonight == [passage] and ears.words() == len(passage.split())
    assert (tmp_path / "cortex" / "hearing" / "1.txt").exists() and shelf == [1]  # downloaded once, then kept
    upcoming = ears.upcoming()
    _, again = ears.bedtime()
    assert again == upcoming and ears.before() == [] and len(ears.tonight) == 2
    ears.heard_said("I like your valley very much")
    assert ears.people == ["I like your valley very much"] and ears.words() > 6
    ears.slept()
    assert ears.tonight == [] and ears.people == []
    for _ in range(len(hearing._passages(hearing.clean_book(fake_book(1)))) - 1):  # to the end of the book, and on
        story, _ = ears.bedtime()
    assert story.title == "Mother Goose" and ears.state["book"] == 1
    kept = json.loads((tmp_path / "cortex" / "hearing.json").read_text())
    again = hearing.Listening(tmp_path)
    assert again.state == kept and again.latest().title == "Mother Goose"
    assert hearing.Listening(tmp_path / "offline").bedtime() is None  # without the internet, no new book


def test_it_hears_a_story_as_it_falls_asleep_and_tells_of_it_when_it_wakes(monkeypatch):
    from haven import life as life_module
    from haven.life import Life
    from haven.mind import Mind

    class Stand:
        busy = __import__("threading").Lock()

        def __init__(self):
            self.stories = []

        def bedtime_story(self, life):
            story = Story("The Tale of a Hen", ("A hen sat on her nest.",), ())
            self.stories.append(story)
            return story

    life = Life(Mind(seed=4), None)
    life.thinker = Stand()
    life._last_story = time.monotonic() - life_module.STORY_EVERY - 1
    life._bedtime()
    for _ in range(100):
        if life.initiative.pending:
            break
        time.sleep(0.02)
    assert life.thinker.stories and life.initiative.pending[-1][1] == "heard"
    assert "The Tale of a Hen" in life.initiative.pending[-1][2]
    life._bedtime()  # not again so soon
    time.sleep(0.1)
    assert len(life.thinker.stories) == 1
