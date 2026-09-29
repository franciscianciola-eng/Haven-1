"""The curriculum: levels of reading, from stories for small children to encyclopedias and math.

Each level has its own reading and its own tests of understanding, all on text held out
from what it learns from:

  fluency        bits per byte on unseen text of that level (lower is better)
  cloze          pick the missing word in a sentence, from four
  next sentence  pick how a passage goes on, from four
  questions      answer a question about a passage, from four candidate answers
  math           pick the answer to a word problem, from four
  self-report    say what state it's in (checked against its actual state)
  understanding  reading a description of a state brings that state to mind
  conversation   answer what people ask it, as its state and what it knows say it should

Chance on the four-way tests is 25%. A level is passed when every test reaches its mark
after a minimum amount of study. If progress stalls it moves on and says so, since a
small model can't master everything; the report card keeps the honest record.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F

from ..workspace import DRIVES
from .grounding import need_index
from .model import Cortex
from .tokenizer import END, HAVEN, THINK, YOU, Tokenizer


@dataclass(frozen=True)
class Level:
    number: int
    name: str
    about: str
    source: str
    tests: dict  # test name → (">=" or "<=", mark)
    new_pieces: int  # how many new tokenizer pieces it learns from this level's reading
    min_steps: int
    max_steps: int


LEVELS = (
    Level(
        1,
        "First stories",
        "very simple stories for small children (TinyStories)",
        "tinystories",
        {"fluency": ("<=", 1.4), "cloze": (">=", 0.6)},
        3000,
        1500,
        12000,
    ),
    Level(
        2,
        "Talking about itself",
        "putting its own states into words, understanding words about states, and answering people",
        "grounded",
        {"self-report": (">=", 0.7), "understanding": (">=", 0.6), "conversation": (">=", 0.8)},
        200,
        500,
        4000,
    ),
    Level(
        3,
        "Children's books",
        "fairy tales, fables and children's classics (Project Gutenberg)",
        "children",
        {"fluency": ("<=", 2.0), "next sentence": (">=", 0.45)},
        1000,
        1500,
        12000,
    ),
    Level(
        4,
        "Simple facts",
        "short articles about the world in plain words (Simple English Wikipedia)",
        "simplewiki",
        {"fluency": ("<=", 2.0), "cloze": (">=", 0.5)},
        1000,
        1500,
        12000,
    ),
    Level(
        5,
        "Reading comprehension",
        "answering questions about passages (SQuAD)",
        "squad",
        {"questions": (">=", 0.45)},
        500,
        1500,
        12000,
    ),
    Level(
        6,
        "Literature",
        "novels written for adults (Project Gutenberg)",
        "classics",
        {"fluency": ("<=", 2.1), "next sentence": (">=", 0.45)},
        1000,
        1500,
        15000,
    ),
    Level(
        7,
        "Encyclopedia",
        "articles about everything (English Wikipedia)",
        "wikipedia",
        {"fluency": ("<=", 2.1), "cloze": (">=", 0.5)},
        1500,
        1500,
        15000,
    ),
    Level(
        8,
        "Reasoning with numbers",
        "grade-school math word problems, worked step by step (GSM8K)",
        "gsm8k",
        {"math": (">=", 0.4)},
        300,
        1500,
        12000,
    ),
)

STOP = {
    "the",
    "and",
    "that",
    "with",
    "they",
    "this",
    "from",
    "have",
    "were",
    "there",
    "their",
    "what",
    "when",
    "which",
    "would",
    "could",
    "about",
    "into",
    "then",
    "than",
    "them",
    "been",
    "will",
    "your",
    "said",
    "very",
    "just",
}


def balanced(moments: list[dict], n: int) -> list[dict]:
    """Up to n moments with every need equally represented, so saying the same thing every time can't pass."""
    groups: dict[str, list[dict]] = {}
    for m in moments:
        groups.setdefault(m["need"], []).append(m)
    per = max(n // max(len(groups), 1), 1)
    chosen = []
    for group in groups.values():
        step = max(len(group) // per, 1)
        chosen += group[::step][:per]
    return chosen[:n]


def passed(level: Level, results: dict) -> bool:
    for test, (direction, mark) in level.tests.items():
        value = results.get(test)
        if value is None or (value > mark if direction == "<=" else value < mark):
            return False
    return True


def progress_score(level: Level, results: dict) -> float:
    """One number for "is it still getting better?": each test's value relative to its mark."""
    total = 0.0
    for test, (direction, mark) in level.tests.items():
        value = results.get(test)
        if value is not None:
            total += mark / max(value, 1e-6) if direction == "<=" else value / mark
    return total


# --- preparing tests -------------------------------------------------------------------------

SENTENCE = re.compile(r"(?<=[.!?])\s+")


def cloze_items(documents: list[str], n: int, rng: random.Random) -> list[dict]:
    """Sentences with one word to fill in, and three wrong words to choose from."""
    sentences, vocabulary = [], set()
    for doc in documents:
        parts = SENTENCE.split(doc)
        for i, s in enumerate(parts):
            words = s.split()
            vocabulary.update(w for w in words if w.isalpha() and w.islower() and len(w) >= 4 and w not in STOP)
            if 6 <= len(words) <= 30:
                sentences.append((" ".join(parts[max(0, i - 3) : i]), s))
    vocabulary = sorted(vocabulary)
    rng.shuffle(sentences)
    items = []
    for context, sentence in sentences:
        words = sentence.split()
        slots = [i for i, w in enumerate(words) if w.isalpha() and w.islower() and len(w) >= 4 and w not in STOP]
        if not slots or len(vocabulary) < 10:
            continue
        i = rng.choice(slots)
        right = words[i]
        similar = [w for w in vocabulary if w != right and abs(len(w) - len(right)) <= 2]
        if len(similar) < 3:
            continue
        choices = [right, *rng.sample(similar, 3)]
        before, after = " ".join(words[:i]), " ".join(words[i + 1 :])
        items.append({"prefix": context, "choices": [f" {before} {c} {after}".rstrip() for c in choices]})
        if len(items) >= n:
            break
    return items


def next_sentence_items(documents: list[str], n: int, rng: random.Random) -> list[dict]:
    """A passage, how it really goes on, and three sentences from elsewhere."""
    pool = [s for doc in documents for s in SENTENCE.split(doc) if 5 <= len(s.split()) <= 40]
    items = []
    docs = list(documents)
    rng.shuffle(docs)
    for doc in docs:
        parts = SENTENCE.split(doc)
        if len(parts) < 4 or len(pool) < 10:
            continue
        i = rng.randrange(2, len(parts))
        right = parts[i]
        if not 5 <= len(right.split()) <= 40:
            continue
        wrong = [s for s in rng.sample(pool, 6) if s != right][:3]
        if len(wrong) < 3:
            continue
        items.append(
            {"prefix": " ".join(parts[max(0, i - 6) : i]), "choices": [f" {right}", *(f" {w}" for w in wrong)]}
        )
        if len(items) >= n:
            break
    return items


# --- running tests ------------------------------------------------------------------------------


class Reader(Protocol):
    """What the tests need from a language cortex, whichever kind it is."""

    def bits(self, text: str, max_tokens: int) -> tuple[float, int, int]:
        """(negative log-likelihood in nats, bytes predicted, tokens predicted) for a text."""

    def pick(self, prefix: str, choices: list[str]) -> int:
        """Which choice it finds most likely to follow the prefix (judged per byte)."""

    def describe(self, state: np.ndarray) -> str:
        """What it says about itself, given only its state tokens."""

    def meaning(self, text: str) -> np.ndarray:
        """What reading a text brings to mind, in the workspace's layout."""


class ScratchReader:
    """The reader interface for the cortex Haven grows from scratch."""

    def __init__(self, model: Cortex, tok: Tokenizer):
        self.model, self.tok = model, tok

    @property
    def device(self) -> torch.device:
        return self.model.embed.weight.device

    @torch.no_grad()
    def bits(self, text: str, max_tokens: int) -> tuple[float, int, int]:
        self.model.eval()
        ids = self.tok.encode(text)[: min(self.model.cfg.context, max_tokens) + 1]
        if len(ids) < 2:
            return 0.0, 0, 0
        logits, _ = self.model(torch.tensor([ids[:-1]], device=self.device))
        target = torch.tensor(ids[1:], device=self.device)
        nll = float(F.cross_entropy(logits[0].float(), target, reduction="sum"))
        return nll, len(self.tok.decode(ids[1:]).encode("utf-8")), len(ids) - 1

    @torch.no_grad()
    def pick(self, prefix: str, choices: list[str]) -> int:
        self.model.eval()
        ids = self.tok.encode(prefix)[-(self.model.cfg.context // 2) :]
        scores = self.model.score(ids, [self.tok.encode(c) for c in choices])
        return int(np.argmax([s / max(len(c.encode("utf-8")), 1) for s, c in zip(scores, choices, strict=True)]))

    @torch.no_grad()
    def describe(self, state: np.ndarray) -> str:
        self.model.eval()
        tensor = torch.tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)
        tokens, _ = self.model.generate([HAVEN], tensor, max_new=40, temperature=0.0, stop=(END,))
        return self.tok.decode(tokens)

    @torch.no_grad()
    def meaning(self, text: str) -> np.ndarray:
        self.model.eval()
        return self.model.meaning([HAVEN, *self.tok.encode(text)]).float().cpu().numpy()

    @torch.no_grad()
    def reply(self, state: np.ndarray, notes: str, question: str) -> str:
        """Its answer to a question, with its state in mind and what it knows recalled first."""
        self.model.eval()
        tensor = torch.tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)
        prompt = [THINK, *self.tok.encode(notes), YOU, *self.tok.encode(question), HAVEN]
        tokens, _ = self.model.generate(prompt, tensor, max_new=100, temperature=0.0, stop=(END, YOU))
        return self.tok.decode(tokens).strip()


def fluency(reader: Reader, documents: list[str], max_tokens: int = 12000) -> float | None:
    """Bits per byte on text it hasn't learned from: how well it predicts the next piece of text."""
    nll, count_bytes, seen = 0.0, 0, 0
    for doc in documents:
        n, b, t = reader.bits(doc, max_tokens - seen)
        nll, count_bytes, seen = nll + n, count_bytes + b, seen + t
        if seen >= max_tokens:
            break
    return nll / count_bytes / math.log(2) if count_bytes else None


def choose(reader: Reader, items: list[dict]) -> float | None:
    """Accuracy on four-way choices, judged by the likelihood of each choice (per byte)."""
    if not items:
        return None
    return sum(reader.pick(item["prefix"], item["choices"]) == 0 for item in items) / len(items)


def self_report(reader: Reader, items: list[dict]) -> float | None:
    """Does what it says about itself match its actual state? (its need, and the color it's looking at)"""
    if not items:
        return None
    checked, right = 0, 0
    for item in items:
        said = reader.describe(item["state"]).lower()
        need = item["need"]
        checked += 1
        right += (need in said) if need != "fine" else ("fine" in said or "comfortable" in said)
        if item.get("color"):
            checked += 1
            right += item["color"] in said
    return right / checked


def conversation(reader: Reader, items: list[dict]) -> float | None:
    """Does it answer people as its state and what it knows say it should? (the answer's words, exactly)"""
    if not items or not hasattr(reader, "reply"):
        return None
    right = sum(_plain(reader.reply(i["state"], i["notes"], i["question"])) == _plain(i["answer"]) for i in items)
    return right / len(items)


def _plain(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def understanding(reader: Reader, items: list[dict]) -> float | None:
    """Reading a description (with no state given) should bring the described need to mind."""
    scored = [i for i in items if need_index(i["need"]) is not None]
    if not scored:
        return None
    if hasattr(reader, "meanings"):  # read many at once when the cortex can
        vectors = np.concatenate(
            [reader.meanings([i["text"] for i in scored[j : j + 16]]) for j in range(0, len(scored), 16)]
        )
    else:
        vectors = np.stack([reader.meaning(i["text"]) for i in scored])
    right = sum(int(np.argmax(v[DRIVES])) == need_index(i["need"]) for v, i in zip(vectors, scored, strict=True))
    return right / len(scored)
