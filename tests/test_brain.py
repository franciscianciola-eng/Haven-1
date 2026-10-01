"""Haven's spiking brain: its regions do what their real counterparts do."""

import pytest

torch = pytest.importorskip("torch")

from haven.brain.brain import CHEMICALS, OPTIONS, Brain
from haven.brain.neurons import CELLS, STRENGTH, Network


@pytest.fixture(scope="module")
def brain():
    b = Brain(seed=1, size="tiny")
    for _ in range(10):
        b.moment({}, {}, {})
    return b


def test_cell_types_fire_like_their_kinds():
    """Regular-spiking pyramidal cells start firing at about 60 pA; spiny cells need far more (they sit in a
    down-state); fast-spiking interneurons jump straight to a high rate."""
    currents = torch.arange(0, 801, 20.0)
    rates = {}
    for kind in ("RS", "MSN", "FS"):
        net = Network(0)
        net.population("x", [(kind, len(currents))])
        net.build()
        net.bias[: len(currents)] = currents
        net.prepare()
        net.begin()
        for _ in range(1000):
            net.step()
        rates[kind] = net.spikes[: len(currents)].float()
    first = {k: float(currents[(r > 0).nonzero()[0]]) for k, r in rates.items()}
    assert 40 <= first["RS"] <= 80
    assert first["MSN"] >= 250
    assert rates["FS"][(rates["FS"] > 0).nonzero()[0]] >= 15  # (class 2: no slow start)
    assert set(CELLS) >= {"RS", "IB", "CH", "FS", "LTS", "TC", "MSN"}


def test_synapses_come_in_26_sizes_over_a_sixty_fold_range():
    assert len(STRENGTH) == 26 and STRENGTH[0] == 0  # (size 0: a silent synapse)
    assert abs(float(STRENGTH[-1] / STRENGTH[1]) - 60.0) < 1e-3


def test_resting_brain_is_quiet_and_balanced(brain):
    r = brain.moment({}, {}, {})
    assert r.rates["cortex"] < 3.0  # sparse, like a real cortex at rest
    assert r.rates["pallidum"] > 20.0  # the pallidum fires on its own, holding the motor thalamus back
    assert r.choice is None
    assert set(r.chemistry) == set(CHEMICALS)


def test_basal_ganglia_let_through_what_is_urged(brain):
    for urges, wanted in (({"play": 0.8, "food": 0.2}, "play"), ({"food": 0.9, "play": 0.2}, "food")):
        for _ in range(14):  # (the old choice takes a few moments to let go)
            r = brain.moment({}, {}, urges)
        assert r.choice == wanted
    assert set(r.channels) == set(OPTIONS)


def test_chemistry_answers_what_happens():
    b = Brain(seed=2, size="tiny")
    for signals, chemical, up in (
        ({"reward": 1.0}, "dopamine", True),
        ({"reward": -1.0}, "dopamine", False),
        ({"pain": 1.0}, "serotonin", False),
        ({"touch": 1.0, "social": 1.0}, "oxytocin", True),
    ):
        for _ in range(20):
            b.moment({}, {}, {})
        before = b.levels[chemical]
        for _ in range(12):
            r = b.moment({}, signals, {})
        assert (r.chemistry[chemical] > 1.5 * before) if up else (r.chemistry[chemical] < 0.7 * before), chemical


def test_what_hurt_it_becomes_frightening():
    b = Brain(seed=3, size="tiny")
    for _ in range(10):
        b.moment({}, {}, {})
    assert b.fear_of({"see:thorns": 1.0}) < 0.2
    for _ in range(3):
        b.moment({"see:thorns": 1.0}, {"pain": 0.9}, {})
    for _ in range(30):
        b.moment({}, {}, {})
    assert b.fear_of({"see:thorns": 1.0}) > 0.5
    assert b.fear_of({"see:apple": 1.0}) < 0.3


def test_what_it_meets_over_and_over_feels_less_new(brain):
    novelties = [brain.moment({"see:bell": 1.0}, {}, {}).novelty for _ in range(60)]
    assert sum(novelties[-10:]) / 10 < sum(novelties[1:11]) / 10
    assert brain.freshness("see:bell") < 0.75
    assert brain.freshness("see:something never met") == 1.0


def test_it_keeps_its_brain(tmp_path):
    b = Brain(seed=4, size="tiny")
    for _ in range(3):
        b.moment({"see:thorns": 1.0}, {"pain": 0.9}, {})
    for _ in range(5):
        b.moment({"see:bell": 1.0, "word:bell": 1.0}, {}, {"play": 0.7})
    b.save(tmp_path)
    again = Brain.load(tmp_path)
    assert torch.equal(again.recurrent.sizes, b.recurrent.sizes)
    assert torch.equal(again.net.syn_size, b.net.syn_size)
    assert set(again.senses) == set(b.senses)
    assert again.fear_of({"see:thorns": 1.0}) > 0.5
    assert again.neurons() == b.neurons() and again.synapses() == b.synapses()
