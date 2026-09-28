import numpy as np

from haven.body import Body
from haven.world import ACTIONS, BUSH, DAY, MAX_BERRIES, NEST, REGROW, THORN, World


def test_senses_have_the_right_shape():
    s = World(0).sense()
    assert s.colors.shape == (5, 3) and s.nearness.shape == (5,)
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
    w.heading = 0  # north, into the wall above the nest
    outcome = w.act("forward")
    assert outcome.bumped and not outcome.moved and w.sense().bump == 1


def test_eating_berries_and_regrowth():
    w = World(0)
    bush = next(iter(w.berries))
    w.x, w.y, w.heading = bush[0] - 1, bush[1], 2  # just west of it, facing east
    assert w.grid[bush[1], bush[0]] == BUSH
    assert w.act("eat").ate and w.berries[bush] == MAX_BERRIES - 1
    for _ in range(REGROW + 5):
        w.act("rest")
    assert w.berries[bush] == MAX_BERRIES


def test_thorns_hurt():
    w = World(0)
    thorn = tuple(int(v) for v in np.argwhere(w.grid == THORN)[0][::-1])
    w.x, w.y, w.heading = thorn[0] - 1, thorn[1], 2
    assert w.act("forward").pain > 0.5


def test_the_nest_is_warm():
    w = World(0)
    assert w.grid[w.y, w.x] == NEST
    assert w.ambient() > w.ambient(5, 5)


def test_state_round_trip():
    w = World(0)
    for action in ACTIONS * 10:
        w.act(action)
    copy = World(1)
    copy.load(w.state())
    assert (copy.x, copy.y, copy.heading, copy.tick, copy.berries) == (w.x, w.y, w.heading, w.tick, w.berries)


def test_body_needs_and_fainting():
    b = Body()
    assert b.drives().shape == (4,)
    hungry = Body(energy=0.2)
    assert hungry.drives()[0] > b.drives()[0] and hungry.discomfort() > b.discomfort()
    b.energy = 0.01
    assert b.collapsed()
    b.faint()
    assert b.asleep and b.energy >= 0.35 and not b.collapsed()
