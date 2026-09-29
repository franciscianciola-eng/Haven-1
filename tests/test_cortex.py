import pytest

torch = pytest.importorskip("torch")

from haven.cortex.model import SLOTS, Cortex, CortexConfig
from haven.workspace import D


def small():
    torch.manual_seed(0)
    return Cortex(CortexConfig(vocab=300, d=32, layers=2, heads=2, context=64))


def test_reads_its_workspace():
    model = small()
    tokens = torch.randint(0, 300, (2, 20))
    logits, meaning = model(tokens)
    assert logits.shape == (2, 20, 300) and meaning.shape == (2, 20, D)
    other, _ = model(tokens, torch.randn(2, SLOTS, D))
    assert not torch.allclose(logits, other)  # what it experiences changes what it expects to say


def test_scoring_generation_and_growth():
    model = small()
    scores = model.score([1, 2, 3], [[4, 5], [6]])
    assert len(scores) == 2 and all(s < 0 for s in scores)
    tokens, logprobs = model.generate([1, 2], max_new=5, temperature=0.0)
    assert len(tokens) == 5 and len(logprobs) == 5
    before = model.embed.weight[:300].clone()
    model.grow_vocabulary(305, [(1, 2)] * 5)
    assert model.embed.weight.shape[0] == 305 and torch.equal(model.embed.weight[:300], before)
    assert torch.allclose(model.embed.weight[300], (before[1] + before[2]) / 2)
    assert model.head.weight is model.embed.weight


def test_a_cortex_grown_for_another_workspace_is_archived(tmp_path):
    from haven.cortex import starter

    folder = tmp_path / "cortex"
    folder.mkdir()
    old = Cortex(CortexConfig(vocab=300, d=32, layers=2, heads=2, context=64, core=36))  # the old garden's size
    torch.save({"config": dict(vars(old.cfg)), "model": old.state_dict(), "progress": {}}, folder / "cortex.pt")
    (folder / "conversations.jsonl").write_text("{}\n")
    assert not starter.fits(folder / "cortex.pt")
    usable = starter.available() and starter.fits(starter.FOLDER / "cortex.pt")
    assert starter.install(tmp_path) == ("replaced" if usable else "retired")
    assert list((tmp_path / "archive").glob("cortex-*/cortex.pt"))  # kept, not deleted
    assert (folder / "conversations.jsonl").exists()  # what it heard stays
    if usable:
        assert starter.fits(folder / "cortex.pt") and starter.install(tmp_path) == ""
