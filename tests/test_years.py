"""Years in the valley: the seasons, the character Haven grows into, speaking up of its own accord, and letting time
go by."""

import copy
import random
import re
import time

import numpy as np

from haven.cortex import talk
from haven.life import Life
from haven.mind import Mind
from haven.personality import TRAITS, Character
from haven.speaking import Initiative
from haven.store import Store
from haven.world import BUTTERFLY, COLORS, DAY, FLOOR, FLOWER, SEASON_DAYS, SEASONS, TREE, YEAR, World


def first(world: World, kind: int) -> tuple[int, int]:
    return next((int(x), int(y)) for y, x in zip(*np.nonzero(world.grid == kind)))


def test_the_year_turns():
    w = World(0)
    seen = []
    for day in range(YEAR):
        w.tick = day * DAY + DAY // 2
        seen.append(w.season)
    assert seen == [s for s in SEASONS for _ in range(SEASON_DAYS)]
    warmth = {}
    for season, day in (("spring", 1), ("summer", 8), ("autumn", 13), ("winter", 20)):
        w.tick = day * DAY + DAY // 2
        warmth[season] = w.ambient(5, 12)
    assert warmth["summer"] > warmth["spring"] > warmth["autumn"] > warmth["winter"]
    flower, tree = first(w, FLOWER), first(w, TREE)
    assert w.thing(*flower) == FLOOR and not any(w.thing(*b) == BUTTERFLY for b in w.butterflies)  # winter
    assert w._color(*tree, TREE) != COLORS[TREE]  # bare branches
    w.tick = 2 * DAY  # spring again
    assert w.thing(*flower) == FLOWER and w._color(*tree, TREE) == COLORS[TREE]


def test_little_grows_back_in_winter():
    def regrown(day: int) -> int:
        w = World(0)
        w.tick = day * DAY
        spot = next(iter(w.berries))
        w.berries[spot], w.growth[spot] = 0, 0
        for _ in range(500):
            w._grow()
            w.tick += 1
        return w.berries[spot]

    assert regrown(7) > regrown(19)  # summer, winter


def test_a_new_season_is_noted_and_a_year_is_a_milestone():
    mind = Mind(seed=3)
    mind.world.tick = YEAR * DAY - 1  # the last moment of its first winter
    mind.step()
    assert mind.world.season == "spring" and mind.log[-1][1] == "saw spring come to the valley"
    assert "year 1" in mind.me.firsts and mind.me.milestones[-1][1].endswith("a whole year had gone by")


def live_days(mind: Mind, days: int, **day) -> None:
    """Days of a kind (what the character's tallies would be), lived through to midnight."""
    c = mind.character
    for _ in range(days):
        c.day.update(ticks=1200, awake=1000, **day)
        mind.world.tick += DAY
        c.end_day(mind)


def test_a_character_grows_out_of_how_it_lives():
    mind = Mind(seed=2)
    c = mind.character
    assert c.traits == c.temperament  # a newborn is all temperament
    lively = {"explore": 600, "plays": 15, "far": 900, "feel": 6.0, "arousal": 350, "social": 10, "social_feel": 0.4}
    live_days(mind, 2 * YEAR, **lively)  # (it takes growing up for how it lives to count for most of who it is)
    assert all(c.traits[t] > 0.6 for t in ("curious", "playful", "brave", "cheerful", "friendly")), c.traits
    assert len(c.years) == 2 and c.best and c.worst
    said = talk.character_words(c.traits, c.days, c.changes())
    assert said.startswith("I'm ") and any(word in said for word in ("curious", "playful", "brave", "cheerful"))
    quiet = {"explore": 20, "plays": 0, "far": 500, "feel": 0.5, "arousal": 500, "social": 10, "social_feel": 0.0}
    live_days(mind, 3 * YEAR, **quiet)  # its life changed: it changes too
    assert c.traits["curious"] < 0.5 and c.traits["playful"] < 0.5
    assert c.changes() and all(d < 0 for _, d in c.changes())  # it has become less so, since it was little
    assert "since I was little" in talk.character_words(c.traits, c.days, c.changes())
    restored = Character(seed=9)
    restored.load_state(copy.deepcopy(c.to_state()))
    assert restored.traits == c.traits and restored.habits == c.habits and restored.years == c.years


def test_its_character_leans_what_it_wants_to_do():
    c = Character(seed=0)
    wants = {"food": 0.1, "warmth": 0.1, "healing": 0.0, "sleep": 0.1, "explore": 0.3, "play": 0.3}
    c.traits.update(curious=0.9, playful=0.1)
    leaned = c.lean(wants)
    assert leaned["explore"] > wants["explore"] and leaned["play"] < wants["play"]


def test_it_can_say_what_it_is_like():
    traits = dict.fromkeys(TRAITS, 0.5) | {"brave": 0.85, "friendly": 0.35, "curious": 0.3}
    assert talk.character_words(traits, 30) == "I'm very brave, a homebody and shy."
    assert talk.character_words(traits, 1) == "I'm still finding out what I'm like."
    assert talk.trait_asked("Are you shy?") == ("friendly", False) and talk.trait_asked("are you brave") == (
        "brave",
        True,
    )
    assert talk.trait_reply(traits, "friendly", False) == "Yes, I'm shy."
    assert talk.trait_reply(traits, "brave", False) == "No, I'm very brave."
    assert talk.trait_reply(traits, "calm", True) == "Sometimes. I'm about as calm as most."
    known = {**talk.memo(Mind(seed=1)), "traits": traits}
    assert "I'm shy." in talk.recall(known, "Are you shy?")  # asked about a trait, it comes to mind


def test_it_understands_short_answers_to_what_it_asked():
    assert talk.answer_to("your favorite food", "pizza") == "My favorite food is pizza."
    assert talk.answer_to("your favorite food", "Probably sushi!") == "My favorite food is sushi."
    assert talk.answer_to("how old you are", "34") == "I'm 34 years old."
    assert talk.answer_to("what you do", "teacher") == "I'm a teacher."
    assert talk.answer_to("your name", "it's Priya") == "My name is Priya."
    assert talk.answer_to("where you live", "What?") is None
    assert talk.answer_to("your favorite food", "I love pizza") is None  # a sentence of their own: taken as it is
    assert talk.statement(talk.answer_to("where you live", "Boston")) == "you live in Boston"


def test_what_it_says_of_its_own_accord_comes_from_what_comes_to_mind():
    for text, said in (
        ("got burned", "Ouch! I got burned."),
        ("did what it was asked: ring the bell", "I did it! I rang the bell."),
        ("stopped trying to push the ball: it was very tired", "I stopped trying to push the ball. I'm very tired."),
        ("fainted, and woke up later in its nest", "I fainted. I woke up in my nest."),
    ):
        assert talk.event_answer(talk.event_note(text)) == said
    assert talk.event_note('said "ba"') is None
    assert talk.need_answer(talk.need_note(0, 0.8, False)) == "I'm very hungry! I'm looking for food."
    assert talk.question_answer(talk.question_note("your favorite food")) == "What's your favorite food?"
    note = talk.memory_note("My best day was in my second summer. I rang the bell.")
    assert (
        talk.memory_answer(note)
        == "I was just thinking about my best day. It was in my second summer. I rang the bell."
    )
    note = talk.passed_note(YEAR, ["I saw summer, autumn and winter."])
    assert talk.passed_answer(note) == "A whole year went by! I saw summer, autumn and winter."
    assert talk.back_answer({"character": "I'm very friendly.", "today": "Today I rang the bell."}, "Sam") == (
        "You're back, Sam! I missed you. Today I rang the bell."
    )


def test_conversations_where_it_speaks_up_first():
    mind = Mind(seed=3)
    mind.live(1500)
    from haven.cortex.grounding import moment_of

    moment = moment_of(mind)
    rng = random.Random(1)
    spoke = 0
    for _ in range(400):
        thought, turns = talk.conversation(moment, rng)
        if turns[0].said is not None:
            continue
        spoke += 1
        said = turns[0].answer
        derived = {  # (friendly or shy: as its character, in what comes to mind, says)
            talk.back_answer({"character": thought, "today": moment["memo"]["today"]}, person)
            for person in (None, *re.findall(r"I'm talking with (\w+)\.", thought))
        }
        notes = re.findall(r"(?:Just now, I|I did what you asked|I stopped trying|I gave up trying)[^.]*\.", thought)
        derived |= {talk.event_answer(n) for n in notes}
        derived |= {talk.need_answer(n) for n in re.findall(r"I'm [a-z ]+, so [^.]*\.", thought)}
        derived |= {talk.question_answer(n) for n in re.findall(r"I'd like to know [^.]*\.", thought)}
        derived |= {a for n, a in talk.SEASON_NEWS.values() if n in thought}
        derived |= {talk.SLEEP_ANSWER} if talk.SLEEP_NOTE in thought else set()
        derived |= {talk.WAKE_ANSWER} if talk.WAKE_NOTE in thought else set()
        for title, begins in re.findall(
            r"I heard a story last night: (.+?)\. It begins: (.+?) I want to tell", thought
        ):
            derived.add(f"Last night I heard a story, {title}. It begins: {begins}")  # (a bedtime story)
        if " went by!" in said:
            derived.add(said if said.replace(" went by!", " went by.", 1) in thought else "")
        if said.startswith("I read about"):
            title, sentence = said.removeprefix("I read about ").split(". It says: ", 1)
            derived.add(said if f"I read about {title}: {sentence}" in thought else "")
        remembered = thought.split("I'm remembering: ", 1)[-1].split(". ")
        pieces = (". ".join(remembered[:n]) for n in range(1, 5))
        derived |= {talk.memory_answer("I'm remembering: " + p + ("" if p.endswith(".") else ".")) for p in pieces}
        assert said in derived, (said, thought[-400:])
    assert spoke > 50


def test_it_speaks_up_when_someone_is_there_and_it_is_quiet():
    mind = Mind(seed=5)
    i = Initiative(random.Random(0))
    t = time.monotonic()
    i.noticed("got burned", mind.tick, False, "morning")
    assert i.occasion(mind, mind.tick, 0.5, list, now=t) is None  # nobody is there
    i.looked(t)
    assert i.occasion(mind, mind.tick, 0.5, list, now=t) == ("event", "Just now, I got burned.", None)
    i.spoke(t)
    i.noticed("rang the bell", mind.tick, False, "morning")
    i.looked(t + 5)
    assert i.occasion(mind, mind.tick, 0.5, list, now=t + 5) is None  # not straight after it spoke
    i.talked(t + 100)
    i.looked(t + 110)
    assert i.occasion(mind, mind.tick, 0.5, list, now=t + 110) is None  # nor right after someone said something
    i.looked(t + 200)
    assert i.occasion(mind, mind.tick, 0.5, list, now=t + 200)[1] == "Just now, I rang the bell."
    i.spoke(t + 200)
    mind.body.energy = 0.1  # very hungry
    i.looked(t + 400)
    kind, note, _ = i.occasion(mind, mind.tick, 0.5, list, now=t + 400)
    assert kind == "need" and note.startswith("I'm very hungry")


def test_when_it_is_quiet_for_a_while_it_asks_about_them():
    mind = Mind(seed=5)
    i = Initiative(random.Random(1))
    t = time.monotonic()
    found = None
    for k in range(60):  # it brings something up now and then
        i.looked(t + k)
        found = i.occasion(mind, mind.tick, 0.5, list, now=t + k)
        if found:
            break
    assert found == ("question", "I'd like to know your name.", "your name")  # the first thing it wants to know


def test_it_greets_them_when_they_come_back():
    mind = Mind(seed=5)
    i = Initiative(random.Random(0))
    t = time.monotonic()
    i.looked(t)
    assert i.back is None  # a first meeting isn't a coming back
    i = Initiative(random.Random(0))
    i.away_for(8 * 3600)  # (when it wakes up: they last talked eight hours ago)
    i.looked(t + 1)
    assert i.occasion(mind, mind.tick, 0.5, list, now=t + 1) == ("back", "You came back after a long time.", None)
    i.looked(t + 300)  # a glance five minutes later: they never went away
    assert i.back is None
    i.looked(t + 1500)  # twenty minutes later: they're back
    assert i.occasion(mind, mind.tick, 0.5, list, now=t + 1500) == ("back", "You came back.", None)
    i = Initiative(random.Random(0))
    i.away_for(120)  # (it woke up two minutes after they last talked: they haven't been away)
    i.looked(t + 1)
    assert i.back is None


def test_letting_time_pass(tmp_path):
    life = Life(Mind(seed=6), Store(tmp_path))
    start = life.mind.world.day
    assert life.pass_time(1) and not life.pass_time(1)  # (it's already going by)
    for _ in range(1200):
        if life.passing is None:
            break
        time.sleep(0.05)
    assert life.passing is None and life.mind.world.day == start + 1
    assert life.initiative.passed.startswith("A whole day went by.")
    assert Store(tmp_path).load()["world"]["tick"] == life.mind.tick  # saved
