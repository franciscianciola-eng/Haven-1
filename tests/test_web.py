"""Haven in the browser (docs/): its own conversation code, run by Pyodide in the page, has the same things in mind
as the desktop Haven before each reply, turn by turn, reading the same articles; and the page's cortex says what the
desktop's does (tests/web/*.test.mjs). These need Node, and `npm install` in tests/web."""

from __future__ import annotations

import io
import json
import random
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from fakes import ARTICLES, tfrecord

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "tests" / "web"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(
    NODE is None or not (WEB / "node_modules" / "pyodide").exists(), reason="needs Node, and npm install in tests/web"
)
sys.path.insert(0, str(ROOT / "packaging" / "web"))

CONVERSATION = [
    ("Hi, I'm Sam", "Nice to meet you, Sam!"),
    ("What is the capital of France?", "I read about France. It says: Its capital city is Paris."),
    (
        "Who was Einstein?",
        "I read about Albert Einstein. It says: Albert Einstein was a scientist who came up with the theory of relativity.",
    ),
    ("When was he born?", "I read about Albert Einstein. It says: Einstein was born in Ulm in 1879."),
    ("Tell me more", "It also says: He worked on theoretical physics."),
    ("What do koalas eat?", "I read about Koala. It says: They eat leaves of eucalyptus trees."),
    ("When did World War 2 end?", "I read about World War II. It says: The war ended with an Allied victory in 1945."),
    ("How are you?", "I feel fine."),
    ("My favorite animal is the octopus", "Oh, an octopus! What do octopuses eat?"),
    ("What's my name?", "Your name is Sam."),
    ("What is 12 times 7?", "12 times 7 is 84."),
    (
        "Tell me about volcanoes",
        "I read about Volcano. It says: A volcano is a mountain that has lava coming out of it.",
    ),
]


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    import build

    path = tmp_path_factory.mktemp("page") / "haven.zip"
    build.bundle(path)
    return path


@pytest.fixture(scope="module")
def shelf(tmp_path_factory):
    from haven.cortex import encyclopedia
    from haven.cortex.shelf import Shelf

    shelf = Shelf(tmp_path_factory.mktemp("shelf"))
    parts = [tfrecord(ARTICLES[i::4]) for i in range(4)]
    assert shelf.read_encyclopedia(encyclopedia.SIMPLE, lambda part: io.BytesIO(parts[part]))
    yield shelf
    shelf.close()


def weights(shelf) -> dict:
    """How many sentences on the shelf say each word (by its stem): what the page's shelf weighs words by."""
    db = sqlite3.connect(shelf.folder / "simple.sqlite")
    db.execute("CREATE VIRTUAL TABLE temp.vocab USING fts5vocab(main, sentences, 'row')")
    counts = dict(db.execute("SELECT term, doc FROM temp.vocab"))
    db.close()
    return {"total": sum(v.size() for v in shelf.volumes.values()) + 1, "counts": counts}


def desktop(shelf, tmp_path) -> tuple[list[list[str]], dict]:
    """What came to mind before each reply, for the desktop Haven (its replies are the conversation's); and its
    character's traits (which the page's Haven is given, to have the same)."""
    from haven.cortex import starter
    from haven.cortex.think import OwnThinker
    from haven.life import Life
    from haven.mind import Mind

    (tmp_path / "cortex").mkdir()
    for name in ("cortex.pt", "tokenizer.json", "progress.json"):
        shutil.copyfile(starter.FOLDER / name, tmp_path / "cortex" / name)
    thinker = OwnThinker(tmp_path)
    thinker.shelf = shelf
    life = Life(Mind(seed=1), None)
    turns, prompts, reply = [], [], [""]

    def say(prompt, state, drafts, most=100, heard=None):
        prompts.append(thinker.tok.decode(prompt))
        return reply[0], 0.9

    thinker._say = say
    for text, answer in CONVERSATION:
        reply[0] = answer
        prompts.clear()
        random.seed(0)
        life.conversation.append({"tick": life.mind.tick, "who": "you", "text": text})
        words, _ = thinker.deliberate(life, text)
        life.mind.hear(text)
        life.reply(words, "")
        turns.append(list(prompts))
    return turns, dict(life.mind.character.traits)


def page(spec: dict) -> dict:
    found = subprocess.run(
        [NODE, str(WEB / "page_haven.mjs")],
        input=json.dumps(spec),
        capture_output=True,
        text=True,
        timeout=900,
        cwd=WEB,
        check=False,
    )
    assert found.returncode == 0, found.stderr[-3000:]
    return json.loads(found.stdout)


def test_the_page_has_in_mind_what_the_desktop_haven_does(bundle, shelf, tmp_path):
    expected, traits = desktop(shelf, tmp_path)
    found = page(
        {
            "zip": str(bundle),
            "conversation": [text for text, _ in CONVERSATION],
            "answers": [answer for _, answer in CONVERSATION],
            "articles": ARTICLES,
            "redirects": {"World War 2": "World War II"},
            "weights": weights(shelf),
            "traits": traits,
            "calm": True,  # (the desktop's mind here has no brain: its chemistry stays as usual)
        }
    )
    assert found["problem"] is None
    for (text, _), want, got in zip(CONVERSATION, expected, found["turns"], strict=True):
        assert got["prompts"] == want, text
        assert got["answer"]


def test_the_pages_python_is_haven_s_code_as_it_is_now(bundle):
    """docs/py/haven.zip is what packaging/web/build.py makes of the code as it is (run it after changing Haven)."""
    assert (ROOT / "docs" / "py" / "haven.zip").read_bytes() == bundle.read_bytes()


def _chromium() -> bool:
    """Whether Playwright and its Chromium are here (for the page's tests in a browser)."""
    where = subprocess.run(
        [NODE, "-e", "import('playwright').then((p) => console.log(p.chromium.executablePath()))"],
        cwd=WEB,
        capture_output=True,
        text=True,
        check=False,
    )
    return where.returncode == 0 and Path(where.stdout.strip()).exists()


def test_the_page_in_a_browser():
    """The page itself, in headless Chromium (page.e2e.mjs): it wakes up, answers from what it reads online, and
    keeps it all; opened in another tab meanwhile, it waits there, and comes there with everything it had."""
    if not _chromium():
        pytest.skip(
            "needs Playwright and its Chromium (npm install in tests/web, then npx playwright install chromium)"
        )
    found = subprocess.run([NODE, "page.e2e.mjs"], cwd=WEB, capture_output=True, text=True, timeout=1500, check=False)
    assert found.returncode == 0, found.stderr[-3000:]
    out = json.loads(found.stdout)
    first, second, newborn = out["visits"]
    replies = [t["haven"] for t in first["turns"] + second["turns"]]
    assert "Sam" in replies[0]
    assert replies[1].startswith("I read about Volcano. It says: A volcano is a mountain")
    assert "eucalyptus" in replies[2]
    assert out["waited"] == "Haven is open in another tab"
    assert second["earlier"] == 1  # (what was said before)
    assert replies[3].startswith("I read about Volcano.")
    assert "Sam" in replies[4]
    read = [q["titles"] for q in out["wiki"] if q.get("prop") == "extracts"]
    assert read == ["Volcano", "Koala"]  # (read once: kept on its shelf, in the browser's storage)
    assert "cuddly" in first["state"]
    assert "53.0M connections" in out["about"]
    offline = second["turns"][-1]
    assert any("Couldn't reach the Simple English Wikipedia" in t for t in offline["thoughts"])
    assert re.search(r"% sure|couldn't find", offline["note"])  # (it answers from what it has)
    assert newborn["hello"] == 1 and newborn["earlier"] == 0  # (started over: nothing said before)
    cortex = [code for path, code in out["asked"] if path == "/docs/cortex/cortex.onnx"]
    assert cortex[0] == 200 and set(cortex[1:]) == {304}  # (downloaded once, then kept in the browser's storage)
    assert not [line for line in out["console"] if line.startswith(("error", "pageerror"))]


def test_the_cortex_on_a_graphics_card():
    """Its cortex on a graphics card (WebGPU: the browser's software one, gpu.e2e.mjs) says and means what the desktop's
    does."""
    if not _chromium():
        pytest.skip(
            "needs Playwright and its Chromium (npm install in tests/web, then npx playwright install chromium)"
        )
    found = subprocess.run([NODE, "gpu.e2e.mjs"], cwd=WEB, capture_output=True, text=True, timeout=1500, check=False)
    assert found.returncode == 0, found.stderr[-3000:]
    out = json.loads(found.stdout)
    want = out["want"]
    assert out["device"] == "webgpu"
    assert out["greedy"] == want["greedy"][: len(out["greedy"])] and len(out["greedy"]) == min(6, len(want["greedy"]))
    assert all(abs(a - b) < 1e-3 for a, b in zip(out["logprobs"], want["logprobs"], strict=False))
    assert out["drafts"][0] == out["greedy"]  # (the careful draft, beside two freer ones)
    assert all(abs(a - b) < 1e-3 for a, b in zip(out["meaning"], want["meaning"], strict=True))
