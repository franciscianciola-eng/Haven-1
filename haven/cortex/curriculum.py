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

Chance on the four-way tests is 25%. A level is passed when every test reaches its mark
after a minimum amount of study. If progress stalls it moves on and says so, since a
small model can't master everything; the report card keeps the honest record.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..workspace import DRIVES
from .grounding import need_index
from .model import Cortex
from .tokenizer import END, HAVEN, Tokenizer


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
        "putting its own states into words, and understanding words about states",
        "grounded",
        {"self-report": (">=", 0.7), "understanding": (">=", 0.6)},
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


def device_of(model: Cortex) -> torch.device:
    return model.embed.weight.device


@torch.no_grad()
def fluency(model: Cortex, tok: Tokenizer, documents: list[str], max_tokens: int = 12000) -> float | None:
    """Bits per byte on text it hasn't learned from: how well it predicts the next piece of text."""
    model.eval()
    device = device_of(model)
    nll, count_bytes, seen = 0.0, 0, 0
    window = model.cfg.context
    for doc in documents:
        ids = tok.encode(doc)[: window + 1]
        if len(ids) < 2:
            continue
        x = torch.tensor([ids[:-1]], device=device)
        y = torch.tensor([ids[1:]], device=device)
        logits, _ = model(x)
        nll += float(F.cross_entropy(logits[0].float(), y[0], reduction="sum"))
        count_bytes += len(tok.decode(ids[1:]).encode("utf-8"))
        seen += len(ids) - 1
        if seen >= max_tokens:
            break
    if not count_bytes:
        return None
    return nll / count_bytes / math.log(2)


@torch.no_grad()
def choose(model: Cortex, tok: Tokenizer, items: list[dict]) -> float | None:
    """Accuracy on four-way choices, judged by the likelihood of each choice (per byte)."""
    if not items:
        return None
    model.eval()
    right = 0
    for item in items:
        prefix = tok.encode(item["prefix"])[-(model.cfg.context // 2) :]
        conts = [tok.encode(c) for c in item["choices"]]
        scores = model.score(prefix, conts)
        per_byte = [s / max(len(c.encode("utf-8")), 1) for s, c in zip(scores, item["choices"], strict=True)]
        right += int(np.argmax(per_byte)) == 0
    return right / len(items)


def _state(item: dict, device: torch.device) -> torch.Tensor:
    return torch.tensor(np.asarray(item["state"], dtype=np.float32), device=device).unsqueeze(0)


@torch.no_grad()
def self_report(model: Cortex, tok: Tokenizer, items: list[dict]) -> float | None:
    """Does what it says about itself match its actual state? (its need, and the color it's looking at)"""
    if not items:
        return None
    model.eval()
    device = device_of(model)
    checked, right = 0, 0
    for item in items:
        tokens, _ = model.generate([HAVEN], _state(item, device), max_new=40, temperature=0.0, stop=(END,))
        said = tok.decode(tokens).lower()
        need = item["need"]
        checked += 1
        right += (need in said) if need != "fine" else ("fine" in said or "comfortable" in said)
        if item.get("color"):
            checked += 1
            right += item["color"] in said
    return right / checked


@torch.no_grad()
def understanding(model: Cortex, tok: Tokenizer, items: list[dict]) -> float | None:
    """Reading a description (with no state given) should bring the described need to mind."""
    model.eval()
    scored = [i for i in items if need_index(i["need"]) is not None]
    if not scored:
        return None
    right = 0
    for item in scored:
        meaning = model.meaning([HAVEN, *tok.encode(item["text"])]).float().cpu().numpy()
        right += int(np.argmax(meaning[DRIVES])) == need_index(item["need"])
    return right / len(scored)
