import random

import pytest

pytest.importorskip("torch")

from haven.cortex import sources
from haven.cortex.curriculum import LEVELS, cloze_items, next_sentence_items, passed
from haven.cortex.train import Trainer
from haven.web import Web

TEST_SCALE = {"tinystories": 60_000, "articles": 40, "steps": 0.004, "every": 5, "batch": 2}


def test_sources_read_their_formats(internet, tmp_path):
    web = Web(delay=0, allow_private=True)
    stories = sources.tinystories(web, tmp_path, 20_000, internet)
    assert stories.train and stories.held_out and all("Once upon a time" in s for s in stories.train)
    books = sources.gutenberg(web, tmp_path, (11, 16), "books", internet)
    assert books.train and "START OF THE PROJECT" not in " ".join(books.train)
    wiki = sources.wiki(web, tmp_path, internet["simplewiki"], 30, "wiki")
    assert len(wiki.train) + len(wiki.held_out) >= 30
    squad = sources.squad(web, tmp_path, internet)
    item = squad.items["questions"][0]
    assert item["prefix"].endswith("Answer:") and len(item["choices"]) == 4
    math = sources.gsm8k(web, tmp_path, internet)
    assert "<<" not in math.train[0] and "The answer is" in math.train[0]
    assert len(math.items["math"][0]["choices"]) == 4
    assert sources.article(web, internet["simplewiki"], "berries")[0] == "Berries"


def test_tests_are_well_formed():
    docs = ["Lily saw a big red ball in the garden today. She wanted to play with it all morning long. " * 3] * 20
    rng = random.Random(0)
    for item in cloze_items(docs, 10, rng) + next_sentence_items(docs, 10, rng):
        assert len(item["choices"]) == 4 and len(set(item["choices"])) >= 2
    assert passed(LEVELS[0], {"fluency": 1.0, "cloze": 0.9})
    assert not passed(LEVELS[0], {"fluency": 3.0, "cloze": 0.9})


def test_the_whole_curriculum_runs_and_resumes(internet, tmp_path):
    web = Web(delay=0, allow_private=True)
    trainer = Trainer(tmp_path, size="tiny", device="cpu", web=web, urls=internet, scale=TEST_SCALE, log=lambda s: None)
    assert trainer.run(through=1) == "done"
    first = trainer.progress["levels"]["1"]
    assert first["status"] in ("passed", "moved on", "plateaued") and "fluency" in first["tests"]
    resumed = Trainer(tmp_path, web=web, urls=internet, scale=TEST_SCALE, log=lambda s: None)
    assert resumed.progress["level"] == 2 and resumed.model is not None
    assert resumed.run(steps=3) == "steps"
    assert resumed.run() == "done"
    card = "\n".join(resumed.report())
    assert all(level.name in card for level in LEVELS)
    assert len(resumed.tok) > 260 and resumed.model.cfg.vocab == len(resumed.tok)
