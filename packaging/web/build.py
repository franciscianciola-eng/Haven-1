"""Haven in the browser: builds what docs/ (the GitHub Pages site) needs besides its JavaScript.

    python packaging/web/build.py [--library ~/.haven/library]

- docs/py/haven.zip: Haven's own conversation code (haven/cortex: what comes to mind, its shelf, its little book, its
  thinking), for Pyodide, with browser.py (the page's mind, life, shelf and cortex: see that file), and in place of
  the parts of Haven a Haven in its nest in the page doesn't run (its valley, its brain, PyTorch, numpy), small
  modules with just what the conversation code takes from them, made from the real ones here. think.py's three ways
  of using the cortex (deliberate, _say, speak_up) become coroutines, since the page's cortex answers in its own time.
- docs/data/weights.json: how many sentences of the Simple English Wikipedia and the dictionary say each word (by its
  stem), from a shelf the desktop Haven has read (so a word weighs as much as there).
- docs/dictionary/*.json: its dictionary (WordNet), by each word's first two letters, as on that shelf.

The cortex itself is exported by export_cortex.py; ONNX Runtime Web and Pyodide are copied by vendor.py.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import io
import json
import os
import sqlite3
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HAVEN_BRAIN", "off")

DOCS = ROOT / "docs"
COPIED = [
    "haven/__init__.py",
    "haven/web.py",
    "haven/cortex/__init__.py",
    "haven/cortex/talk.py",
    "haven/cortex/engage.py",
    "haven/cortex/library.py",
    "haven/cortex/book.py",
    "haven/cortex/stories.py",
    "haven/cortex/encyclopedia.py",
    "haven/cortex/shelf.py",
    "haven/cortex/tokenizer.py",
    "haven/cortex/sources.py",
]
AWAITED = {"deliberate", "_say", "speak_up"}  # think.py's ways of using the cortex, which the page's cortex awaits


class _Await(ast.NodeTransformer):
    """In the awaited methods: await what they ask of the cortex, and each other."""

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        f = node.func
        if (
            isinstance(f, ast.Attribute)
            and isinstance(f.value, ast.Name)
            and f.value.id == "self"
            and f.attr in AWAITED
        ):
            return ast.Await(node)
        if (
            isinstance(f, ast.Attribute)
            and f.attr in ("generate", "meaning")
            and isinstance(f.value, ast.Attribute)
            and f.value.attr == "model"
        ):
            return ast.Await(node)
        return node


def awaited_think() -> str:
    """think.py with its three ways of using the cortex made coroutines."""
    tree = ast.parse((ROOT / "haven/cortex/think.py").read_text())
    made = set()
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        for i, node in enumerate(cls.body):
            if isinstance(node, ast.FunctionDef) and node.name in AWAITED and cls.name == "OwnThinker":
                fn = ast.AsyncFunctionDef(**{f: getattr(node, f) for f in node._fields if hasattr(node, f)})
                fn = _Await().visit(fn)
                cls.body[i] = ast.copy_location(fn, node)
                made.add(node.name)
    if made != AWAITED:
        raise SystemExit(f"think.py has changed: couldn't find {sorted(AWAITED - made)} in OwnThinker")
    return (
        "# Made from haven/cortex/think.py by packaging/web/build.py (deliberate, _say, speak_up awaited).\n"
        + ast.unparse(ast.fix_missing_locations(tree))
    )


def stubs() -> dict[str, str]:
    """Modules with just what the conversation code takes from the parts of Haven the page doesn't run."""
    from haven import activities, attention, mind, personality, selfmodel, will, workspace, world
    from haven.cortex import engage

    def values(module, names):
        return "".join(f"{name} = {getattr(module, name)!r}\n" for name in names)

    head = '"""Made by packaging/web/build.py: what the conversation code takes from {}, for the page."""\n\n'
    return {
        "haven/mind.py": head.format("haven/mind.py")
        + values(mind, ["POWERS", "CHEMICALS"])
        + "\n\n"
        + inspect.getsource(mind.need_words)
        + "\n\nclass Mind:  # (the page's mind is browser.Mind)\n    pass\n",
        "haven/personality.py": head.format("haven/personality.py") + values(personality, ["TRAITS"]),
        "haven/selfmodel.py": head.format("haven/selfmodel.py") + values(selfmodel, ["VERDICTS"]),
        "haven/world.py": head.format("haven/world.py")
        + values(
            world,
            [
                "BELL",
                "DAY",
                "FIRE",
                "NEST",
                "SAND",
                "SEASON_DAYS",
                "SEASONS",
                "THORN",
                "TREE",
                "TURNING",
                "WATER",
                "YEAR",
                "CLIMATE",
            ],
        ),
        "haven/attention.py": head.format("haven/attention.py") + values(attention, ["GOALS"]),
        "haven/workspace.py": head.format("haven/workspace.py")
        + values(workspace, ["D", "SOURCES", "Q", "QUALITY", "DRIVES", "AFFECT", "CONFIDENCE", "IGNITION", "DECAY"]),
        "haven/activities.py": head.format("haven/activities.py") + values(activities, ["SIGHTS"]),
        "haven/will.py": head.format("haven/will.py")
        + values(will, ["DONE"])
        + "\n\ndef consider(mind, req, insisted=False, rng=None):  # (in its nest it's asked nothing to do: see browser.py)\n"
        + "    raise NotImplementedError\n",
        "haven/cortex/grounding.py": head.format("haven/cortex/grounding.py")
        + "def mind_state(mind):\n    return [mind.state()]  # (browser.Mind.state: the workspace tokens, flattened)\n\n\n"
        + "def moment_of(mind):\n    return {}\n",
        "numpy.py": '"""Made by packaging/web/build.py: the little of numpy the conversation code uses."""\n\n'
        "def argmax(values):\n    values = list(values)\n    return max(range(len(values)), key=values.__getitem__)\n\n\n"
        "def mean(values):\n    values = list(values)\n    return sum(values) / len(values) if values else float('nan')\n\n\n"
        "def clip(value, low, high):\n    return min(max(value, low), high)\n",
        "haven/newborn.py": newborn(engage),
    }


def newborn(engage) -> str:
    """What a newborn Haven knows of itself and its valley (talk.memo), its readouts the page doesn't simulate (its
    attention schema and self-model, as its cortex reads them: grounding.mind_state), captured from one."""
    from haven.attention import GOALS
    from haven.cortex import talk
    from haven.cortex.grounding import mind_state
    from haven.life import open_mind
    from haven.mind import CHEMICALS
    from haven.store import Store
    from haven.workspace import D
    from haven.world import CLIMATE

    with tempfile.TemporaryDirectory() as home:
        m = open_mind(Store(Path(home)), seed=1)  # (one newborn: the same page for the same code)
        memo = talk.memo(m)
        state = mind_state(m)
        evidence = dict(m.me.evidence)
        alive = float(m.me.alive)
        traits = dict(m.character.traits)
    return (
        '"""Made by packaging/web/build.py: a newborn Haven, as it knows itself (talk.memo), and its attention schema and\n'
        'self-model as its cortex reads them (grounding.mind_state), captured from one."""\n\n'
        f"MEMO = {memo!r}\n"
        f"TRAITS = {traits!r}\n"
        f"ATTENTION = {[round(float(v), 6) for v in state[2]]!r}\n"
        f"SELF = {{'evidence': {evidence!r}, 'alive': {alive!r}}}\n"
        f"GOALS = {tuple(GOALS)!r}\n"
        f"CHEMICALS = {tuple(CHEMICALS)!r}\n"
        f"CLIMATE = {dict(CLIMATE)!r}\n"
        f"D = {int(D)!r}\n"
    )


def bundle(out: Path) -> int:
    """docs/py/haven.zip: the page's Python."""
    files = {path: (ROOT / path).read_text() for path in COPIED}
    files["haven/cortex/think.py"] = awaited_think()
    files["haven/browser.py"] = (ROOT / "packaging/web/browser.py").read_text()
    files.update(stubs())
    for path, text in files.items():
        compile(text, path, "exec")  # (it must at least read as Python)
    out.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(files):
            info = zipfile.ZipInfo(path, (2026, 1, 1, 0, 0, 0))  # (the same bytes for the same code)
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, files[path])
    out.write_bytes(buffer.getvalue())
    return len(files)


def weights(library: Path, out: Path) -> int:
    """How many sentences say each word (its stem), on a desktop shelf: the encyclopedia and the dictionary."""
    counts: dict[str, int] = {}
    total = 1
    for name in ("simple", "dictionary"):
        db = sqlite3.connect(library / f"{name}.sqlite")
        db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp.vocab USING fts5vocab(main, sentences, 'row')")
        for term, doc in db.execute("SELECT term, doc FROM temp.vocab"):
            counts[term] = counts.get(term, 0) + doc
        total += db.execute("SELECT max(last) FROM articles").fetchone()[0] or 0
        db.close()
    kept = {t: n for t, n in counts.items() if n >= 2 and len(t) <= 30}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"total": total, "counts": kept}, separators=(",", ":")))
    return len(kept)


def dictionary(library: Path, folder: Path) -> int:
    """The dictionary from a desktop shelf, by each word's first two letters: {word: [(title, sentences, names)]}."""
    db = sqlite3.connect(library / "dictionary.sqlite")
    files: dict[str, dict] = {}
    rows = db.execute(
        "SELECT a.title, a.first, a.last, group_concat(n.name, '\x1f') FROM articles a JOIN names n ON n.article = a.id "
        "GROUP BY a.id"
    )
    count = 0
    for title, first, last, names in rows:
        sentences = [
            t
            for (t,) in db.execute(
                "SELECT text FROM sentences WHERE rowid BETWEEN ? AND ? ORDER BY rowid", (first, last)
            )
        ]
        names = names.split("\x1f")
        for name in names:
            key = "".join(c if "a" <= c <= "z" else "_" for c in name.lower()[:2].ljust(2, "_"))
            files.setdefault(key, {}).setdefault(name.lower(), []).append([title, sentences, names])
        count += 1
    db.close()
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.json"):
        old.unlink()
    for key, words in files.items():
        (folder / f"{key}.json").write_text(json.dumps(words, separators=(",", ":"), ensure_ascii=False))
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", help="a shelf the desktop Haven has read (for its weights and its dictionary)")
    args = parser.parse_args()
    n = bundle(DOCS / "py" / "haven.zip")
    print(f"docs/py/haven.zip: {n} files, {(DOCS / 'py' / 'haven.zip').stat().st_size / 1e3:.0f} kB")
    if args.library:
        library = Path(args.library).expanduser()
        print(f"docs/data/weights.json: {weights(library, DOCS / 'data' / 'weights.json'):,} words")
        print(f"docs/dictionary: {dictionary(library, DOCS / 'dictionary'):,} words")


if __name__ == "__main__":
    main()
