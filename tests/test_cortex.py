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


def test_an_older_starter_is_replaced_but_a_cortex_that_has_read_is_kept(tmp_path):
    import json
    import shutil

    from haven.cortex import starter

    if not (starter.available() and starter.stamp(starter.FOLDER)):
        return  # (a copy of Haven without a stamped starter)
    for name, levels, expected in (
        ("older", {"2": {"status": "passed", "steps": 12925}}, "updated"),
        ("read", {"1": {"status": "studying", "steps": 300}, "2": {"status": "passed", "steps": 12925}}, ""),
    ):
        root = tmp_path / name
        (root / "cortex").mkdir(parents=True)
        for f in starter.FILES:
            shutil.copyfile(starter.FOLDER / f, root / "cortex" / f)
        (root / "cortex" / "progress.json").write_text(json.dumps({"level": 1, "levels": levels}))  # (no stamp)
        assert starter.install(root) == expected, name
        assert bool(list((root / "archive").glob("cortex-*/cortex.pt"))) == (expected == "updated")
        assert (starter.stamp(root / "cortex") == starter.stamp(starter.FOLDER)) == (expected == "updated")
        assert starter.install(root) == ""  # and it stays as it is from then on


def test_growing_wider_and_deeper_keeps_what_it_computes():
    from haven.cortex.model import grow

    model = small()
    with torch.no_grad():  # (so every part of it matters)
        for name, p in model.named_parameters():
            p.add_(torch.randn_like(p) * (0.3 if "norm" in name else 0.05))
    model.eval()
    tokens, state = torch.randint(0, 300, (2, 30)), torch.randn(2, SLOTS, D)
    logits, meaning = model(tokens, state)
    for width, layers in ((2, None), (1, 5), (2, 5), (3, 4)):
        grown = grow(model, width, layers).eval()
        assert grown.cfg.d == 32 * width and grown.cfg.layers == (layers or 2)
        assert grown.parameters_count() > model.parameters_count() or width == 1
        new_logits, new_meaning = grown(tokens, state)
        assert torch.allclose(new_logits, logits, atol=1e-4) and torch.allclose(new_meaning, meaning, atol=1e-4)
        assert grown.head.weight is grown.embed.weight
    grown = grow(model, 2, 4)
    assert (
        grown.generate([1, 2, 3], max_new=10, temperature=0.0)[0]
        == model.generate([1, 2, 3], max_new=10, temperature=0.0)[0]
    )
    loss = torch.nn.functional.cross_entropy(grown(tokens)[0].reshape(-1, 300), tokens.reshape(-1))
    loss.backward()
    torch.optim.SGD(grown.parameters(), lr=0.1).step()
    w = grown.blocks[0].mlp[0].weight
    assert (w[0::2] - w[1::2]).abs().mean() > 0  # the copies of each unit learn apart
    assert grown.blocks[1].proj.weight.abs().sum() > 0  # and a new layer starts to add something
    with pytest.raises(ValueError):
        grow(model, 1, 1)  # a cortex only grows


def test_packed_weights_come_back_nearly_as_they_were(tmp_path):
    from haven.cortex.starter import packed, unpacked

    model = small()
    weights = packed(model.state_dict())
    assert "head.weight" not in weights and weights["embed.weight"]["int8"].dtype == torch.int8
    torch.save(weights, tmp_path / "cortex.pt")
    again = Cortex(model.cfg)
    again.load_state_dict(unpacked(torch.load(tmp_path / "cortex.pt", weights_only=False)))
    assert again.head.weight is again.embed.weight
    tokens = torch.randint(0, 300, (1, 20))
    assert torch.allclose(again(tokens)[0], model(tokens)[0], atol=0.05)


def test_a_long_telling_does_not_go_round_in_circles():
    model = small().eval()

    def grams(tokens):
        return [tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)]

    looping, _ = model.generate([1, 2, 3], max_new=60, temperature=0.0)
    assert len(grams(looping)) > len(set(grams(looping)))  # (an untrained cortex goes round in circles)
    told, _ = model.generate([1, 2, 3], max_new=60, temperature=0.0, no_repeat=3)
    assert len(grams(told)) == len(set(grams(told)))  # but not when it mustn't

    def cycling(tokens, state=None, past=None):  # a cortex that always wants to say 0, 1, 2, 3, 4, 0, 1, ...
        logits = torch.full((1, tokens.shape[1], 300), -10.0)
        logits[0, -1, (int(tokens[0, -1]) + 1) % 5] = 10.0
        logits[0, -1, 9] = 5.0  # (and next best, 9)
        return logits, None, [None] * len(model.blocks)

    model.run = cycling
    told, _ = model.generate([4], max_new=8, temperature=0.0, no_repeat=3)
    assert told == [0, 1, 2, 3, 4, 0, 1, 9]  # once round, then not again
    told, _ = model.generate([0, 1, 2, 3, 4], max_new=11, temperature=0.0, no_repeat=3)
    assert told == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 9]  # but what came to mind, it can say again


def test_a_haven_that_came_without_its_cortex_gets_it_the_first_time(tmp_path, monkeypatch):
    import hashlib
    import json

    from haven.cortex import starter

    folder = tmp_path / "starter"
    folder.mkdir()
    model = small()
    torch.save({"config": vars(model.cfg), "model": starter.packed(model.state_dict())}, tmp_path / "made.pt")
    body = (tmp_path / "made.pt").read_bytes()
    (folder / "tokenizer.json").write_text(json.dumps({"merges": []}))
    (folder / "progress.json").write_text(json.dumps({"starter": "1"}))
    url = "https://raw.githubusercontent.com/someone/Haven/abc/haven/cortex/starter/cortex.pt"
    (folder / "cortex.json").write_text(json.dumps({"url": url, "sha256": "0" * 64, "bytes": len(body)}))
    monkeypatch.setattr(starter, "FOLDER", folder)
    monkeypatch.setattr(starter, "fits", lambda path: True)
    asked = []

    class Web:
        def get(self, url, most):
            asked.append(url)
            return body

    assert starter.install(tmp_path / "home", Web()) == "" and not (folder / "cortex.pt").exists()  # (not the one)
    (folder / "cortex.json").write_text(
        json.dumps({"url": url, "sha256": hashlib.sha256(body).hexdigest(), "bytes": 9})
    )
    assert starter.install(tmp_path / "home", Web()) == "installed" and asked == [url, url]
    assert (tmp_path / "home" / "cortex" / "cortex.pt").read_bytes() == body
    assert starter.install(tmp_path / "home", Web()) == "" and len(asked) == 2  # (got once, then kept)
