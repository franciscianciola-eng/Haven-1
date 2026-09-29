import numpy as np

from haven.store import Store, join, split


def test_split_and_join():
    tree = {"a": np.arange(3), "b": [np.ones((2, 2)), {"c": 1.5}], "d": "text", "e": np.float64(2.0)}
    arrays = {}
    flat = split(tree, arrays)
    back = join(flat, arrays)
    assert np.array_equal(back["a"], tree["a"]) and back["b"][1]["c"] == 1.5 and back["e"] == 2.0


def test_save_load_and_archive(tmp_path):
    store = Store(tmp_path)
    assert not store.exists()
    store.save({"x": np.array([1.0, 2.0]), "name": "Haven"})
    assert store.exists()
    loaded = store.load()
    assert loaded["name"] == "Haven" and list(loaded["x"]) == [1.0, 2.0]
    where = store.archive()
    assert where is not None and (where / "mind.json").exists() and not store.exists()


def test_a_haven_from_the_old_garden_moves_to_the_valley(tmp_path):
    from haven.life import open_mind
    from haven.mind import MOVED, VERSION, Mind

    old = Mind(seed=3, name="Pip").to_state()  # stands in for a life saved before the valley
    del old["version"]
    old["world"]["tick"] = 5000
    old["self"]["firsts"] += ["first meal", "kind:2"]
    old["self"]["milestones"] += [[4000, "found food and ate for the first time"]]
    old["counts"]["ate"] = 12
    store = Store(tmp_path)
    store.save(old)
    mind = open_mind(store)
    assert (mind.me.name, mind.world.tick, mind.counts["ate"]) == ("Pip", 5000, 12)  # same name, age, history
    assert mind.me.milestones[-1] == (5000, MOVED) and "kind:2" not in mind.me.firsts
    assert not mind.me.milestone("first meal", 5001, "")  # its firsts stay firsts
    assert (mind.world.x, mind.world.y) == mind.world.nest
    assert store.load()["version"] == VERSION and list((tmp_path / "archive").iterdir())  # the old life is kept
    assert open_mind(store).world.tick == 5000  # from now on it simply wakes up
