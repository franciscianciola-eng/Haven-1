"""Its pastimes: what it does when its needs leave it free, and how doing the same thing wears thin."""

from haven import activities
from haven.mind import Mind
from haven.world import BUTTERFLY, DIRECTIONS, World


def test_a_haven_with_time_to_itself_finds_pastimes():
    mind = Mind(seed=3)
    for tick in range(3000):
        mind.company = (tick // 300) % 3 == 0  # someone there a third of the time
        mind.step()
    pastimes = [
        text for _, text in mind.log if text.startswith(("watched", "sang", "danced", "chased", "sat with you"))
    ]
    assert pastimes, "it should have found something to do besides its needs"
    assert set(mind.visited) >= {"my nest"}  # it keeps track of where it has been, to go back


def test_doing_the_same_thing_wears_thin_and_comes_back():
    mind = Mind(seed=1)
    for _ in range(30):
        activities.wear(mind, "sing")
    worn = activities.freshness(mind, "sing")
    assert worn < 0.3 and activities.freshness(mind, "dance") == 1.0
    for _ in range(400):
        activities.wear(mind, None)
    assert activities.freshness(mind, "sing") > worn + 0.5


def test_it_gets_bored_of_a_plaything_for_a_while():
    mind = Mind(seed=2)
    mind.played = {5: 9.0, 6: 1.0}
    assert mind.bored() == {5}


def test_butterflies_can_be_chased():
    world = World(0)
    x, y = world.butterflies[0]
    dx, dy = DIRECTIONS[world.heading]
    world.x, world.y = x - dx, y - dy
    assert world.thing(*world.ahead()) == BUTTERFLY
    outcome = world.act("use")
    assert outcome.chased and tuple(world.butterflies[0]) != (x, y)


def test_its_pastimes_feel_good():
    mind = Mind(seed=5)
    assert activities.felt(mind, "dance", "left", None) > 0.2
    assert activities.felt(mind, "watch", "forward", None) == 0.0  # (watching is sitting still and looking)
