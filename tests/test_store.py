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
