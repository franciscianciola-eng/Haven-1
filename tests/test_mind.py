import copy

import numpy as np

from haven.check import indicators, probe
from haven.mind import Mind
from haven.report import readout, snapshot
from haven.world import BUSH, THORN


def test_it_learns_to_live(grown):
    m = grown
    assert len(m.vision.kinds.alive()) >= 4  # kinds of things, found for itself
    assert m.counts["ate"] >= 1 and m.counts["fainted"] == 0
    assert m.counts["dreams"] >= 1  # it slept, and dreamed
    assert m.workspace.ignitions > 100
    edible = [k for k in m.vision.kinds.alive() if m.knowledge.eaten[k] >= 1 and m.knowledge.edible(k) > 0.5]
    assert edible, "it should have found out what is good to eat"
    assert m.me.alive > 0.3


def test_it_remembers_what_it_has_seen_and_done_with_things(grown):
    things = grown.things
    assert things["bush"]["ate"] >= 1 and sum(bool(t.get("seen")) for t in things.values()) >= 6
    assert grown.today or grown.counts["ate"]  # what it did lately, to tell about
    assert any(grown.kind_name(k) for k in grown.vision.kinds.alive())  # it knows what some of its kinds are called


def test_its_metacognition_and_attention_schema_work(grown):
    insight = grown.meta.insight()
    assert insight is not None and insight > 0.6  # confidence tells right percepts from wrong ones
    _, on_switch = grown.schema.accuracy()
    assert on_switch > 0.3  # well above chance (1 in 8) at foreseeing where attention goes
    assert grown.model.agency.mean > 0.0  # its actions explain what it sees change


def test_the_probe_measures_every_indicator(grown):
    measured = probe(grown, ticks=400)
    assert measured["belief_accuracy"] > 0.7
    found = indicators(grown, measured)
    assert [i.code for i in found] == [
        "RPT-1",
        "RPT-2",
        "GWT-1",
        "GWT-2",
        "GWT-3",
        "GWT-4",
        "HOT-1",
        "HOT-2",
        "HOT-3",
        "HOT-4",
        "AST-1",
        "PP-1",
        "AE-1",
        "AE-2",
    ]
    assert grown.world.tick == 3360  # the probe used a copy


def test_readouts_and_snapshot(grown):
    lines = readout(grown)
    assert lines and all(isinstance(line, str) for line in lines)
    snap = snapshot(grown)
    assert snap["name"] == "Haven" and len(snap["beliefs"]["kind"]) == 24
    assert snap["world"]["heights"][4][18] == "3" and snap["world"]["ball"]  # the valley, with its hill
    assert snap["kinds"] and snap["self"]["conclusions"]


def test_saving_and_loading_continue_the_same_life(grown):
    original = copy.deepcopy(grown)
    restored = Mind(seed=99)
    restored.load_state(original.to_state())
    original.live(200)
    restored.live(200)
    assert (original.world.x, original.world.y, original.world.tick) == (
        restored.world.x,
        restored.world.y,
        restored.world.tick,
    )
    assert np.isclose(original.body.energy, restored.body.energy)
    assert original.log == restored.log


def test_it_does_what_it_is_asked(grown):
    for do, thing, action, did in (
        ("ring the bell", "bell", "use", "rang"),
        ("push the ball", "ball", "use", "pushed"),
        ("eat some berries", "bush", "eat", "ate"),
        ("go to the pond", "pond", "go", None),
    ):
        mind = copy.deepcopy(grown)
        before = mind.counts.get(did, 0)
        mind.take_errand(do, thing, action)
        for _ in range(300):
            mind.step()
            if mind.errand is None:
                break
        assert mind.log[-1][1] == f"did what it was asked: {do}", (do, mind.log[-4:])
        assert did is None or mind.counts[did] > before
        assert mind.counts["errands"] == 1 and "errand" in mind.me.firsts


def test_it_goes_to_sleep_when_asked_and_keeps_errands_when_saved(grown):
    mind = copy.deepcopy(grown)
    mind.take_errand("go to sleep", "nest", "sleep", 3)
    restored = Mind(seed=4)
    restored.load_state(mind.to_state())
    assert restored.errand["do"] == "go to sleep"
    for _ in range(300):
        restored.step()
        if restored.errand is None:
            break
    assert restored.body.asleep and restored.log[-1][1] == "did what it was asked: go to sleep"


def test_it_gives_up_on_an_errand_when_something_else_matters_more(grown):
    mind = copy.deepcopy(grown)
    mind.take_errand("ring the bell", "bell", "use")
    mind.body.energy = 0.05  # suddenly starving
    mind.step()
    assert mind.errand is None and mind.log[-1][1].startswith("stopped trying to ring the bell: it was")


def test_it_learns_words_from_a_person():
    mind = Mind(seed=0)
    mind.live(2400)
    last = -100
    for _ in range(5000):
        mind.step()
        c, t = mind.workspace.content, mind.world.tick
        if c is None or c.source != "vision" or not c.extra.get("cell") or t - last < 30:
            continue
        x, y = c.extra["cell"]
        cell = mind.world._cell(x, y)
        if cell == BUSH and mind.world.berries[(x, y)]:
            mind.hear("berry")
            last = t
        elif cell == THORN:
            mind.hear("ouch")
            last = t
    assert "berry" in mind.lexicon.vocabulary() or "ouch" in mind.lexicon.vocabulary()
    for word in mind.lexicon.vocabulary():
        kind = mind.lexicon.kind_of(word)
        color = mind.kind_color(kind)
        assert color in {"berry": ("red", "orange-brown", "pink"), "ouch": ("purple", "pink", "blue")}[word]


def test_welfare_it_cannot_die():
    mind = Mind(seed=5)
    mind.body.energy = 0.02
    mind.live(5)
    assert mind.counts["fainted"] == 1 and mind.body.energy >= 0.3
    assert (mind.world.x, mind.world.y) == mind.world.nest
