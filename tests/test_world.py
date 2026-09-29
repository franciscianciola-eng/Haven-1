import numpy as np

from haven.body import Body
from haven.world import (
    ACTIONS,
    APPLE,
    BELL,
    BUSH,
    DAY,
    FIRE,
    FLOWER,
    MAX_BERRIES,
    NEST,
    REGROW,
    THORN,
    TOADSTOOL,
    TREE,
    WATER,
    World,
)


def at(w: World, kind: int) -> tuple[int, int]:
    return tuple(int(v) for v in np.argwhere(w.grid == kind)[0][::-1])


def facing(w: World, spot: tuple[int, int]) -> None:
    """Stand just west of a spot, facing it (east)."""
    w.x, w.y, w.heading = spot[0] - 1, spot[1], 2


def test_senses_have_the_right_shape():
    s = World(0).sense()
    assert s.colors.shape == (5, 3) and s.heights.shape == (5,) and s.nearness.shape == (5,)
    assert 0 <= s.light <= 1 and 0 <= s.scent <= 1


def test_days_and_nights():
    w = World(0)
    lights = []
    for _ in range(DAY):
        w.act("rest")
        lights.append(w.light)
    assert min(lights) < 0.1 and max(lights) > 0.9


def test_walls_block_and_bumping_is_felt():
    w = World(0)
    w.x, w.y, w.heading = 1, 1, 0  # north, into the wall around the valley
    outcome = w.act("forward")
    assert outcome.bumped and not outcome.moved and w.sense().bump == 1


def test_the_hill_can_be_climbed_but_not_the_cliff():
    w = World(0)
    w.x, w.y, w.heading = 14, 5, 2  # east, at the foot of the hill's gentle side
    assert w.act("forward").climbed and w.level(w.x, w.y) == 1
    w.x, w.y, w.heading = 18, 8, 0  # north, at the foot of the cliff
    assert w.act("forward").bumped
    w.x, w.y, w.heading = 18, 7, 4  # south, off the top of the cliff: jumping down is fine
    assert w.act("forward").moved and w.level(w.x, w.y) == 0


def test_it_sees_how_tall_things_are_and_not_past_rising_ground():
    w = World(0)
    tree = at(w, TREE)
    w.x, w.y, w.heading = tree[0], tree[1] + 1, 0  # right under a tree, looking up at it
    s = w.sense()
    assert s.nearness[2] == 1 and s.heights[2] > 0.8  # trees are tall
    w.x, w.y, w.heading = 13, 4, 2  # east, toward the hill: the slope blocks the view beyond it
    s = w.sense()
    assert s.nearness[2] > 0.5 and s.heights[2] < 0.1


def test_eating_berries_and_regrowth():
    w = World(0)
    bush = next(iter(w.berries))
    facing(w, bush)
    assert w.grid[bush[1], bush[0]] == BUSH
    outcome = w.act("eat")
    assert outcome.ate and outcome.food > 0 and w.berries[bush] == MAX_BERRIES - 1
    for _ in range(REGROW + 5):
        w.act("rest")
    assert w.berries[bush] == MAX_BERRIES


def test_shaking_a_tree_drops_an_apple_to_eat():
    w = World(0)
    tree = at(w, TREE)
    facing(w, tree)
    assert w.act("use").shook and len(w.apples) == 1 and w.fruit[tree] == 2
    apple = w.apples[0]
    w.x, w.y = apple[0] - 1, apple[1]
    w.heading = 2
    if w.thing(*w.ahead()) == APPLE:
        outcome = w.act("eat")
        assert outcome.ate and outcome.food > 0.3 and apple not in w.apples  # (others may fall meanwhile)


def test_toadstools_make_it_sick():
    w = World(0)
    facing(w, at(w, TOADSTOOL))
    outcome = w.act("eat")
    assert outcome.ate and outcome.sick > 0
    assert w.thing(*w.ahead()) != TOADSTOOL  # eaten; it grows back later


def test_things_to_use():
    w = World(0)
    facing(w, at(w, WATER))
    assert w.act("use").drank
    bell = at(w, BELL)
    w.x, w.y, w.heading = bell[0], bell[1] + 1, 0
    assert w.act("use").rang and w.sense().sound == 1.0 and w.sense().sound == 0.0
    facing(w, at(w, FLOWER))
    assert w.act("use").smelled


def test_pushing_the_ball():
    w = World(0)
    bx, by = w.ball
    w.x, w.y, w.heading = bx - 1, by, 2
    outcome = w.act("forward")
    assert outcome.pushed and outcome.moved and w.ball == (bx + 1, by) and (w.x, w.y) == (bx, by)
    assert w.act("use").pushed and w.ball[0] > bx + 1  # a kick sends it further


def test_thorns_hurt_and_fire_burns_but_warms():
    w = World(0)
    facing(w, at(w, THORN))
    assert w.act("forward").pain > 0.5
    fire = at(w, FIRE)
    w.x, w.y = fire[0] - 1, fire[1]
    beside = w.ambient()
    assert beside > w.ambient(fire[0] - 4, fire[1])
    w.heading = 2
    assert w.act("forward").pain > 0.5


def test_the_nest_is_warm_and_butterflies_flit():
    w = World(0)
    assert w.grid[w.y, w.x] == NEST
    assert w.ambient() > w.ambient(5, 5)
    start = list(w.butterflies)
    for _ in range(40):
        w.act("rest")
    assert w.butterflies != start


def test_state_round_trip():
    w = World(0)
    for action in ACTIONS * 10:
        w.act(action)
    copy = World(1)
    copy.load(w.state())
    assert (copy.x, copy.y, copy.heading, copy.tick, copy.berries) == (w.x, w.y, w.heading, w.tick, w.berries)
    assert (copy.ball, copy.apples, copy.butterflies, copy.fruit) == (w.ball, w.apples, w.butterflies, w.fruit)


def test_body_needs_and_fainting():
    b = Body()
    assert b.drives().shape == (4,)
    hungry = Body(energy=0.2)
    assert hungry.drives()[0] > b.drives()[0] and hungry.discomfort() > b.discomfort()
    b.energy = 0.01
    assert b.collapsed()
    b.faint()
    assert b.asleep and b.energy >= 0.35 and not b.collapsed()
