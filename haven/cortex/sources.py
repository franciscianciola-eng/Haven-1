"""What Haven reads, and where it comes from: public texts and datasets on the internet.

Everything is downloaded once, carefully (see haven/web.py), and kept in a local cache so
later runs don't need the internet. Each source is split into text to learn from and
held-out text that is only used to test how well it understands.
"""

from __future__ import annotations

import io
import json
import random
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from ..web import Web, WebError

URLS = {
    "tinystories": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
    "tinystories-valid": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
    # Project Gutenberg asks programs to download from its mirrors, never from www.gutenberg.org.
    "gutenberg": "https://aleph.pglaf.org",
    "gutenberg-mirror": "https://mirrors.pglaf.org/gutenberg",
    "simplewiki": "https://simple.wikipedia.org/w/api.php",
    "wikipedia": "https://en.wikipedia.org/w/api.php",
    "squad-train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
    "squad-dev": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
    "gsm8k-train": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
    "gsm8k-test": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
}

# Project Gutenberg books: children's classics, then books written for adults.
CHILDREN = (11, 16, 55, 113, 120, 236, 2591, 21, 271, 500, 74, 2781)
CLASSICS = (1342, 84, 98, 1661, 76, 2701, 345, 158, 1260, 174)


@dataclass
class Reading:
    train: list[str]
    held_out: list[str]
    items: dict[str, list] = field(default_factory=dict)  # prepared test items, by test

    def to_json(self) -> dict:
        return {"train": self.train, "held_out": self.held_out, "items": self.items}


def cached(cache: Path, name: str, make) -> Reading:
    path = cache / f"{name}.json"
    if path.exists():
        data = json.loads(path.read_text())
        return Reading(data["train"], data["held_out"], data.get("items", {}))
    reading = make()
    cache.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(reading.to_json()))
    tmp.replace(path)
    return reading


def split(documents: list[str], held: float = 0.05, seed: int = 0) -> tuple[list[str], list[str]]:
    documents = [d for d in documents if d.strip()]
    rng = random.Random(seed)
    order = list(range(len(documents)))
    rng.shuffle(order)
    n = max(1, int(len(documents) * held)) if len(documents) > 1 else 0
    held_out = {order[i] for i in range(n)}
    return (
        [d for i, d in enumerate(documents) if i not in held_out],
        [d for i, d in enumerate(documents) if i in held_out],
    )


# --- the sources -------------------------------------------------------------------------


def tinystories(web: Web, cache: Path, budget: int, urls: dict) -> Reading:
    def make() -> Reading:
        train = _stories(web.get(urls["tinystories"], budget).decode("utf-8", "replace"))
        held = _stories(web.get(urls["tinystories-valid"], max(budget // 20, 200_000)).decode("utf-8", "replace"))
        return Reading(train, held[:2000])

    return cached(cache, f"tinystories-{budget}", make)


def _stories(text: str) -> list[str]:
    parts = text.split("<|endoftext|>")
    if len(parts) > 1:
        parts = parts[:-1]  # the last one was cut off by the download limit
    return [" ".join(p.split()) for p in parts if len(p.split()) > 5]


def gutenberg(web: Web, cache: Path, books: tuple[int, ...], name: str, urls: dict) -> Reading:
    def make() -> Reading:
        train, held = [], []
        failures = []
        for book in books:
            try:
                text = _book(web, book, urls)
            except WebError as error:
                failures.append(str(error))
                continue
            passages = _passages(_strip_gutenberg(text))
            cut = max(1, len(passages) // 20)
            train += passages[:-cut]
            held += passages[-cut:]
        if not train:
            raise WebError("couldn't download any books (" + "; ".join(failures[:3]) + ")")
        return Reading(train, held)

    return cached(cache, name, make)


def gutenberg_path(book: int) -> str:
    """Where a mirror keeps a book: #2591 is under 2/5/9/2591/, and #7 under 0/7/."""
    digits = str(book)
    folders = "0" if len(digits) == 1 else "/".join(digits[:-1])
    return f"{folders}/{digits}/{digits}"


def _book(web: Web, book: int, urls: dict) -> str:
    """A book's plain text from a mirror, whichever of the usual files it has."""
    mirrors = [urls["gutenberg"], *([urls["gutenberg-mirror"]] if urls.get("gutenberg-mirror") else [])]
    for mirror in mirrors:
        stem = f"{mirror.rstrip('/')}/{gutenberg_path(book)}"
        for suffix, encoding in (("-0", "utf-8"), ("", "utf-8"), ("-8", "latin-1")):
            for ext in (".txt", ".zip"):
                try:
                    body = web.get(stem + suffix + ext, 12_000_000)
                except WebError:
                    continue
                if ext == ".zip":
                    body = _unzip_text(body)
                    if body is None:
                        continue
                return body.decode(encoding, "replace")
    raise WebError(f"no mirror had book #{book}")


def _unzip_text(body: bytes) -> bytes | None:
    try:
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            name = next((n for n in archive.namelist() if n.lower().endswith(".txt")), None)
            return archive.read(name) if name else None
    except (zipfile.BadZipFile, OSError):
        return None


def _strip_gutenberg(text: str) -> str:
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.IGNORECASE)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", text, re.IGNORECASE)
    return text[start.end() if start else 0 : end.start() if end else len(text)]


def _passages(text: str, size: int = 2000) -> list[str]:
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text)]
    passages, current = [], ""
    for paragraph in paragraphs:
        if not paragraph:
            continue
        if current and len(current) + len(paragraph) > size:
            passages.append(current)
            current = ""
        current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        passages.append(current)
    return [p for p in passages if len(p) > 200]


def wiki(web: Web, cache: Path, api: str, articles: int, name: str) -> Reading:
    """Random encyclopedia articles (their opening sections), twenty at a time."""

    def make() -> Reading:
        seen, documents = set(), []
        attempts = 0
        while len(documents) < articles and attempts < articles // 10 + 20:
            attempts += 1
            data = web.json(
                api,
                {
                    "action": "query",
                    "format": "json",
                    "generator": "random",
                    "grnnamespace": 0,
                    "grnlimit": 20,
                    "prop": "extracts",
                    "exintro": 1,
                    "explaintext": 1,
                    "exlimit": 20,
                },
            )
            for page in (data.get("query", {}).get("pages", {}) or {}).values():
                text = (page.get("extract") or "").strip()
                if page.get("pageid") in seen or len(text) < 200:
                    continue
                seen.add(page.get("pageid"))
                documents.append(f"{page.get('title', '')}\n\n{text}")
        train, held = split(documents)
        return Reading(train, held)

    return cached(cache, f"{name}-{articles}", make)


def article(web: Web, api: str, topic: str) -> tuple[str, str] | None:
    """The encyclopedia article that best matches a topic: (title, text)."""
    found = web.json(api, {"action": "query", "format": "json", "list": "search", "srsearch": topic, "srlimit": 1})
    hits = found.get("query", {}).get("search", [])
    if not hits:
        return None
    title = hits[0]["title"]
    data = web.json(
        api,
        {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": 1,
            "titles": title,
            "redirects": 1,
        },
    )
    pages = list((data.get("query", {}).get("pages", {}) or {}).values())
    if not pages or not pages[0].get("extract"):
        return None
    return pages[0].get("title", title), pages[0]["extract"]


def squad(web: Web, cache: Path, urls: dict, seed: int = 0) -> Reading:
    """Passages with questions and answers (SQuAD 1.1, Rajpurkar et al. 2016)."""

    def make() -> Reading:
        train = []
        for topic in web.json(urls["squad-train"])["data"]:
            for paragraph in topic["paragraphs"]:
                qa = "\n\n".join(
                    f"Question: {q['question']}\nAnswer: {q['answers'][0]['text']}"
                    for q in paragraph["qas"]
                    if q["answers"]
                )
                train.append(f"{paragraph['context']}\n\n{qa}")
        rng = random.Random(seed)
        items = []
        for topic in web.json(urls["squad-dev"])["data"]:
            answers = sorted({a["text"] for p in topic["paragraphs"] for q in p["qas"] for a in q["answers"][:1]})
            for paragraph in topic["paragraphs"]:
                for q in paragraph["qas"]:
                    right = q["answers"][0]["text"]
                    wrong = [a for a in answers if a.lower() != right.lower()]
                    if len(wrong) < 3:
                        continue
                    items.append(
                        {
                            "prefix": f"{paragraph['context']}\n\nQuestion: {q['question']}\nAnswer:",
                            "choices": [right, *rng.sample(wrong, 3)],
                        }
                    )
        rng.shuffle(items)
        return Reading(train, [], {"questions": items[:400]})

    return cached(cache, "squad", make)


def gsm8k(web: Web, cache: Path, urls: dict, seed: int = 0) -> Reading:
    """Grade-school math word problems with worked solutions (GSM8K, Cobbe et al. 2021)."""

    def make() -> Reading:
        train = [_problem(p) for p in _jsonl(web.get(urls["gsm8k-train"], 20_000_000))]
        rng = random.Random(seed)
        items = []
        for p in _jsonl(web.get(urls["gsm8k-test"], 20_000_000)):
            answer = p["answer"].split("####")[-1].strip().replace(",", "")
            try:
                value = float(answer)
            except ValueError:
                continue
            wrong = {_number(value + d) for d in (-1, 1, 2, -2, 10)} | {_number(value * 2), _number(value / 2)}
            wrong.discard(answer)
            items.append(
                {
                    "prefix": f"Question: {p['question']}\nAnswer: The answer is",
                    "choices": [f" {answer}.", *(f" {w}." for w in rng.sample(sorted(wrong), 3))],
                    "question": p["question"],
                    "answer": answer,
                }
            )
        rng.shuffle(items)
        return Reading(train, [], {"math": items[:300]})

    return cached(cache, "gsm8k", make)


def _jsonl(body: bytes) -> list[dict]:
    lines = body.decode("utf-8", "replace").splitlines()
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # the last line may have been cut off
    return rows


def _problem(p: dict) -> str:
    steps, _, answer = p["answer"].partition("####")
    steps = re.sub(r"<<[^>]*>>", "", steps).strip()
    return f"Question: {p['question']}\nAnswer: {steps}\nThe answer is {answer.strip().replace(',', '')}."


def _number(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:.2f}".rstrip("0").rstrip(".")


def local(folder: Path) -> Reading:
    """Text files of the person's own choosing."""
    documents = []
    for path in sorted(folder.rglob("*.txt")):
        documents += _passages(path.read_text(errors="replace"))
    train, held = split(documents)
    return Reading(train, held)
