"""The grafted cortex: an open model wired into Haven's workspace.

These tests use a tiny stand-in with Qwen3's architecture and chat format, taught here to
talk the way Haven talks (and to reason before answering), since real models can't be
downloaded while testing.
"""

import numpy as np
import pytest

pytest.importorskip("transformers")

from fakes import teach_base, tiny_base

from haven.cortex.curriculum import balanced, self_report
from haven.cortex.graft import DESCRIBE, STATE, SYSTEM, Graft, GraftTrainer, exists, pick_base
from haven.cortex.grounding import mind_state
from haven.cortex.think import GraftThinker, make_thinker
from haven.cortex.train import simulated_moments
from haven.life import Life
from haven.mind import Mind
from haven.web import Web

SCALE = {"every": 100, "min_steps": 100, "max_steps": 300, "tests": 10, "reports": 16, "reads": 60, "batch": 8}


@pytest.fixture(scope="module")
def moments(tmp_path_factory):
    cache = tmp_path_factory.mktemp("reading")
    return cache, simulated_moments(cache, lambda s: None, lives=((1, 1.5), (2, 1.5), (3, 0.8)))


@pytest.fixture(scope="module")
def stand_in(tmp_path_factory, moments):
    """A tiny model that talks like Haven about whatever state is written into its system message, the way a real
    model reads context, and that can reason before answering."""
    from transformers import AutoTokenizer

    folder = tmp_path_factory.mktemp("stand-in")
    train = moments[1]["train"]
    tiny_base(folder, " ".join(m["text"] + " " + m["answer"] for m in train[:2000]))
    tok = AutoTokenizer.from_pretrained(folder)

    def prompt(question, m, think=False):
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=think)
        return text.replace(STATE, f"{m['need']}, seeing {m['color'] or 'nothing'}")

    texts = []
    for i, m in enumerate(train):
        texts.append(prompt(DESCRIBE, m) + m["text"] if i % 2 else prompt(m["question"], m) + m["answer"])
        if i % 5 == 0:
            texts.append(
                prompt(m["question"], m, think=True)
                + f"<think>\nThey asked me something. {m['text']}\n</think>\n\n{m['answer']}"
            )
    teach_base(folder, texts, steps=500)
    return folder


def trainer(root, base, cache, internet, **kwargs):
    root.joinpath("cortex").mkdir(parents=True, exist_ok=True)
    reading = root / "cortex" / "reading"
    if not reading.exists():
        reading.symlink_to(cache)
    web = Web(delay=0, allow_private=True)
    return GraftTrainer(
        root, base=str(base), device="cpu", web=web, urls=internet, log=lambda s: None, scale=SCALE, **kwargs
    )


@pytest.fixture(scope="module")
def grafted(tmp_path_factory, stand_in, moments, internet):
    """The stand-in, wired into Haven through the whole curriculum."""
    cache, data = moments
    root = tmp_path_factory.mktemp("haven")
    t = trainer(root, stand_in, cache, internet)
    probe = balanced(data["held"], 24)
    before = self_report(t.graft, probe)  # unwired, it says what it usually says, whatever its state
    assert t.run() == "done"
    return root, t, probe, before


def test_the_graft_learns_to_say_its_state(grafted, stand_in, moments, internet):
    root, t, probe, before = grafted
    first, second = t.progress["levels"]["1"], t.progress["levels"]["2"]
    assert first["status"] in ("passed", "not yet") and "cloze" in first["tests"]
    assert second["status"] in ("passed", "plateaued", "moved on") and second["steps"] >= 100
    after = self_report(t.graft, probe)
    assert after >= before + 0.1, (before, after)  # its state tokens alone now tell it what to say
    assert 0.0 <= second["tests"]["wits"] <= 1.0
    assert all(str(n) in t.progress["levels"] for n in range(1, 9)) and "grafted onto" in t.report()[0]
    assert exists(root)
    again = trainer(root, stand_in, moments[0], internet)  # the wiring comes back from disk
    assert again.progress["level"] == 9
    assert np.allclose(again.graft.meaning("I'm hungry."), t.graft.meaning("I'm hungry."), atol=1e-5)
    assert self_report(again.graft, probe) == after


def test_it_thinks_through_the_workspace(grafted):
    root = grafted[0]
    thinker, message = make_thinker("own", root, web=Web(delay=0, allow_private=True))
    assert isinstance(thinker, GraftThinker) and "grafted onto" in message
    life = Life(Mind(seed=4), None)
    life.conversation.append({"tick": 0, "who": "you", "text": "How do you feel?"})
    answer, confidence = thinker.deliberate(life, "How do you feel?")
    assert isinstance(answer, str) and 0.0 <= confidence <= 1.0
    heard = life.mind.thoughts[0]
    assert heard.source == "hearing" and heard.label.startswith("understanding")  # what was said came to mind
    assert all(t.source == "thought" for t in life.mind.thoughts[1:])

    # When it isn't sure, it reasons step by step, and each step comes to mind as inner speech as it's produced.
    graft = thinker.graft
    said = "<think>\nThey asked me how I feel. My body says I'm warm enough.\n</think>\n\nI feel fine."

    def reasoning(ids, state, max_new=0, temperature=0.0, on_text=None):
        for i in range(0, len(said), 7):
            on_text(said[i : i + 7])
        graft.last_tokens = graft.encode(said)
        return said, [-0.1] * len(graft.last_tokens)

    graft.generate = reasoning
    life.mind.thoughts.clear()
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "How do you feel?"}]
    reasoned, confidence = thinker._reason(life, messages, mind_state(life.mind))
    steps = [t.extra["text"] for t in life.mind.thoughts]
    assert reasoned == "I feel fine." and confidence > 0.8
    assert steps == ["They asked me how I feel.", "My body says I'm warm enough.", "I feel fine."]

    thinker.remember({"you": "The blue flowers by the pond are called irises.", "haven": "Irises.", "confidence": 0.5})
    assert any("irises" in note.lower() for note in thinker.recall("What are those blue flowers called?"))


def test_it_picks_a_base_by_hardware():
    import torch

    assert pick_base(torch.device("cpu")) in ("qwen3-0.6b", "qwen3-1.7b")
    assert pick_base(torch.device("mps")) == "qwen3-1.7b"


def test_graft_forward_shapes(stand_in):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    graft = Graft(AutoModelForCausalLM.from_pretrained(stand_in), AutoTokenizer.from_pretrained(stand_in), "stand-in")
    import torch

    ids, mask = graft.batch([graft.encode("I feel fine."), graft.encode("Hi")])
    h = graft.hidden(ids, torch.zeros(2, 4, 36), mask)
    assert h.shape[:2] == ids.shape and graft.meanings(["a", "b c"]).shape == (2, 36)
    assert graft.pick("I feel", [" fine.", " zebra quantum."]) in (0, 1)


def test_chat(tmp_path, monkeypatch, capsys):
    """`haven chat`: it gets a cortex the first time, and then you can just talk."""
    from fakes import tiny_base as make

    from haven.cli import main

    base = make(tmp_path / "base")
    monkeypatch.setenv("HAVEN_HOME", str(tmp_path / "home"))
    lines = iter(["hello, how are you?", "/status", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    assert main(["chat", "--base", str(base), "--no-web"]) == 0
    out = capsys.readouterr().out
    assert "is born" in out and "no language cortex yet" in out and "You're talking with Haven" in out
    assert "Haven: " in out and "In its mind" in out and "resting until you come back" in out
    assert exists(tmp_path / "home")
