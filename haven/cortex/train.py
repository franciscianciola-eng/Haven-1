"""Studying: the language cortex works through the curriculum, level by level.

Progress is saved as it goes (the model, its tokenizer, and a report card), so study can
stop at any time and pick up where it left off.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..web import Web
from . import sources
from .curriculum import (
    LEVELS,
    Level,
    ScratchReader,
    balanced,
    choose,
    cloze_items,
    conversation,
    fluency,
    next_sentence_items,
    passed,
    progress_score,
    self_report,
    understanding,
)
from .grounding import gather
from .model import SIZES, Cortex, CortexConfig
from .talk import OTHER_QUESTIONS, QUESTIONS, UNKNOWN_FORMS, UNKNOWN_TOPICS
from .talk import conversation as converse
from .tokenizer import END, HAVEN, THINK, YOU, Tokenizer

SCALE = {  # how much it reads and studies, by the size of its cortex
    "tiny": {
        "tinystories": 8_000_000,
        "articles": 600,
        "pieces": 0.5,
        "steps": 1.0,
        "batch": 16,
        "lr": 1e-3,
        "every": 250,
    },
    "small": {
        "tinystories": 40_000_000,
        "articles": 2000,
        "pieces": 1.0,
        "steps": 1.5,
        "batch": 24,
        "lr": 6e-4,
        "every": 500,
    },
    "medium": {
        "tinystories": 150_000_000,
        "articles": 6000,
        "pieces": 1.5,
        "steps": 3.0,
        "batch": 32,
        "lr": 4e-4,
        "every": 500,
    },
    "large": {
        "tinystories": 400_000_000,
        "articles": 15000,
        "pieces": 2.0,
        "steps": 5.0,
        "batch": 32,
        "lr": 3e-4,
        "every": 1000,
    },
}
REVIEW = 0.15  # share of each batch spent re-reading earlier levels, so it doesn't forget them


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Stop(Exception):
    pass


class Trainer:
    def __init__(
        self,
        root: Path,
        size: str = "auto",
        device: str = "auto",
        web: Web | None = None,
        urls: dict | None = None,
        log: Callable[[str], None] = print,
        scale: dict | None = None,
    ):
        self.dir = Path(root) / "cortex"
        self.cache = self.dir / "reading"
        self.log = log
        self.web = web or Web()
        self.urls = urls or sources.URLS
        self.device = pick_device(device)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.model: Cortex | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        if (self.dir / "cortex.pt").exists():
            self._load()
        else:
            if size == "auto":
                size = {"cuda": "medium", "mps": "small"}.get(self.device.type, "tiny")
            if size not in SIZES:
                raise ValueError(f"size must be one of {', '.join(SIZES)}")
            self.tok = Tokenizer()
            self.progress = {"size": size, "level": 1, "levels": {}, "steps": 0, "tokens": 0, "started": time.time()}
        self.scale = {**SCALE[self.progress["size"]], **(scale or {})}
        self.rng = random.Random(self.progress["steps"])
        self._streams: dict[int, np.ndarray] = {}
        self._grounded: list[dict] | None = None

    # --- saving -----------------------------------------------------------------------------------

    def _load(self) -> None:
        checkpoint = torch.load(self.dir / "cortex.pt", map_location="cpu", weights_only=False)
        self.tok = Tokenizer.load(self.dir / "tokenizer.json")
        self.progress = checkpoint["progress"]
        self.model = Cortex(CortexConfig(**checkpoint["config"]))
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device)
        self._make_optimizer()
        if checkpoint.get("optimizer"):
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            except ValueError:
                pass  # the vocabulary grew since; start the optimizer's moments afresh

    def save(self) -> None:
        if self.model is None:
            return
        self.tok.save(self.dir / "tokenizer.json")
        tmp = self.dir / "cortex.pt.tmp"
        torch.save(
            {
                "config": asdict(self.model.cfg),
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict() if self.optimizer else None,
                "progress": self.progress,
            },
            tmp,
        )
        os.replace(tmp, self.dir / "cortex.pt")
        (self.dir / "progress.json").write_text(json.dumps(self.progress, indent=1))

    # --- the plan -----------------------------------------------------------------------------------

    def run(self, through: int | None = None, minutes: float | None = None, steps: int | None = None) -> str:
        """Study from the current level on. Returns why it stopped: done, time, steps or interrupted."""
        deadline = None if minutes is None else time.monotonic() + 60 * minutes
        budget = [steps]
        try:
            while self.progress["level"] <= len(LEVELS):
                level = LEVELS[self.progress["level"] - 1]
                if through is not None and level.number > through:
                    return "done"
                data = self.prepare(level)
                self.study(level, data, deadline, budget)
                self.progress["level"] += 1
                self.save()
            return "done"
        except Stop as reason:
            self.save()
            return str(reason)
        except KeyboardInterrupt:
            self.save()
            return "interrupted"

    def prepare(self, level: Level) -> dict:
        """Get the level's reading, grow the vocabulary with it, and set up its tests."""
        record = self.progress["levels"].setdefault(str(level.number), {"status": "starting", "steps": 0, "tests": {}})
        self.log(f"Level {level.number} · {level.name}: {level.about}.")
        if level.source == "grounded":
            moments = self.grounded()
            train_texts = [m["text"] for m in moments["train"]] + [m["notes"] for m in moments["train"][::5]]
            train_texts += [a for m in moments["train"][::5] for a in m["answers"].values()]
            train_texts += [q for qs in QUESTIONS.values() for q in qs] * 20 + list(OTHER_QUESTIONS) * 5
            train_texts += [form.format(topic) for form in UNKNOWN_FORMS for topic in UNKNOWN_TOPICS]
            rng = random.Random(0)
            for m in moments["train"][::2]:  # conversations: what comes to mind, what's asked, and what it says
                thought, turns = converse(m, rng)
                train_texts += [thought, *(t.said for t in turns), *(t.answer for t in turns)]
            data = {
                "grounded": moments["train"],
                "items": {
                    "self-report": balanced(moments["held"], 60),
                    "understanding": moments["held"][:200],
                    "conversation": conversation_items(moments["held"], 120),
                },
            }
        else:
            reading = self.read(level)
            self.log(
                f"  reading: {len(reading.train):,} texts to learn from, {len(reading.held_out):,} held out for tests"
            )
            train_texts = reading.train
            rng = random.Random(level.number)
            items = dict(reading.items)
            if "cloze" in level.tests:
                items["cloze"] = cloze_items(reading.held_out, 120, rng)
            if "next sentence" in level.tests:
                items["next sentence"] = next_sentence_items(reading.held_out, 80, rng)
            data = {"held_out": reading.held_out, "items": items}
        if "pieces" not in record:
            sample = _sample(train_texts, 6_000_000, random.Random(0))
            added = self.tok.learn(sample, int(level.new_pieces * self.scale["pieces"]))
            record["pieces"] = added
            self.log(f"  vocabulary: +{added:,} pieces learned from this reading ({len(self.tok):,} in all)")
            self._grow_model()
            self._streams.pop(level.number, None)
            self.save()
        if level.source != "grounded":
            data["stream"] = self.stream(level.number, train_texts)
        return data

    def read(self, level: Level) -> sources.Reading:
        return read_level(level, self.web, self.cache, self.urls, self.scale)

    def grounded(self) -> dict:
        """Moments from simulated lives, each with its state tokens and words for it."""
        if self._grounded is None:
            self._grounded = simulated_moments(self.cache, self.log, lives=self.scale.get("lives", LIVES))
        return self._grounded

    def stream(self, number: int, texts: list[str]) -> np.ndarray:
        if number in self._streams:
            return self._streams[number]
        path = self.dir / "tokens" / f"level-{number}.npy"
        if path.exists():
            ids = np.load(path)
        else:
            self.log("  turning the reading into tokens…")
            pieces = []
            for text in texts:
                pieces.extend(self.tok.encode(text))
                pieces.append(END)
            ids = np.array(pieces, dtype=np.int32)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, ids)
        self._streams[number] = ids
        return ids

    def _grow_model(self) -> None:
        if self.model is None:
            cfg = CortexConfig.sized(self.progress["size"], len(self.tok))
            self.model = Cortex(cfg).to(self.device)
            self.log(f"  a new language cortex: {self.model.parameters_count() / 1e6:.1f}M connections")
        else:
            old = self.model.cfg.vocab
            parts = self.tok.merges[old - (len(self.tok) - len(self.tok.merges)) :]
            self.model.grow_vocabulary(len(self.tok), parts)
        self._make_optimizer()

    def _make_optimizer(self) -> None:
        decay = [p for n, p in self.model.named_parameters() if p.dim() >= 2]
        other = [p for n, p in self.model.named_parameters() if p.dim() < 2]
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
            lr=self.scale["lr"] if hasattr(self, "scale") else 6e-4,
            betas=(0.9, 0.95),
        )

    # --- studying -------------------------------------------------------------------------------------

    def study(self, level: Level, data: dict, deadline: float | None, budget: list) -> None:
        record = self.progress["levels"][str(level.number)]
        record["status"] = "studying"
        min_steps = int(level.min_steps * self.scale["steps"])
        max_steps = int(level.max_steps * self.scale["steps"])
        every = self.scale["every"]
        while True:
            if record["steps"] >= max_steps:
                record["status"] = "moved on"
                self.log(
                    f"  Moved on after {record['steps']:,} steps without reaching every mark (see the report card)."
                )
                return
            loss = self.step(level, data, record["steps"])
            record["steps"] += 1
            self.progress["steps"] += 1
            if budget[0] is not None:
                budget[0] -= 1
            if record["steps"] % every == 0:
                results = self.evaluate(level, data)
                record["tests"] = results
                self.log(f"  step {record['steps']:>6,}  loss {loss:.3f}  " + describe(level, results))
                if passed(level, results) and record["steps"] >= min_steps:
                    record["status"] = "passed"
                    self.log(f"  Passed level {level.number} after {record['steps']:,} steps.")
                    self.save()
                    return
                score = progress_score(level, results)
                if record.get("best") is None or score > record["best"] * 1.01:
                    record["best"], record["stale"] = score, 0
                else:
                    record["stale"] = record.get("stale", 0) + 1
                if record["stale"] >= 4 and record["steps"] >= min_steps:
                    record["status"] = "plateaued"
                    self.log("  Its scores have stopped improving at this size; moving on (see the report card).")
                    self.save()
                    return
                self.save()
            if budget[0] is not None and budget[0] <= 0:
                raise Stop("steps")
            if deadline is not None and time.monotonic() > deadline:
                raise Stop("time")

    def step(self, level: Level, data: dict, n: int) -> float:
        model, opt = self.model, self.optimizer
        model.train()
        warmup = 200
        lr = self.scale["lr"] * min(1.0, (n + 1) / warmup)
        for group in opt.param_groups:
            group["lr"] = lr
        # Once it can talk about itself, it keeps talking while it reads, so it doesn't forget how.
        talks = self.progress["levels"].get("2", {}).get("status") in ("passed", "moved on", "plateaued", "studied")
        grounded_turn = level.source == "grounded" or (talks and n % 5 == 0)
        if grounded_turn:
            loss = self._grounded_loss(self.grounded()["train"] if level.source != "grounded" else data["grounded"])
        else:
            loss = self._text_loss(level, data)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return float(loss.detach())

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return torch.autocast("cpu", enabled=False)

    def _text_loss(self, level: Level, data: dict) -> torch.Tensor:
        t = self.model.cfg.context
        b = self.scale["batch"]
        rows = []
        earlier = [n for n in range(1, level.number) if n != 2 and (self.dir / "tokens" / f"level-{n}.npy").exists()]
        for _ in range(b):
            stream = data["stream"]
            if earlier and self.rng.random() < REVIEW:
                stream = self.stream(self.rng.choice(earlier), [])
            if len(stream) <= t + 1:
                row = np.full(t + 1, END, dtype=np.int64)
                row[: len(stream)] = stream
            else:
                start = self.rng.randrange(0, len(stream) - t - 1)
                row = stream[start : start + t + 1].astype(np.int64)
            rows.append(row)
        batch = torch.tensor(np.stack(rows), device=self.device)
        self.progress["tokens"] += b * t
        with self._autocast():
            logits, _ = self.model(batch[:, :-1])
            return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))

    def _grounded_loss(self, moments: list[dict]) -> torch.Tensor:
        """Talking about itself: saying what it's experiencing, understanding words about states, and answering people.

        Only what Haven says is marked: the notes it recalls and what people say are there to be read.
        """
        b = self.scale["batch"]
        chosen = [moments[self.rng.randrange(len(moments))] for _ in range(b)]
        seqs, marks, states, understand = [], [], [], []
        for i, m in enumerate(chosen):
            if i % 6 in (0, 1):  # saying what it's experiencing; or (0) working out a state from the words alone
                seq = [HAVEN, *self.tok.encode(m["text"]), END]
                mark = [False] + [True] * (len(seq) - 1)
            else:
                seq, mark = conversation_ids(self.tok, m, self.rng)
            limit = self.model.cfg.context + 1
            seqs.append(seq[-limit:])
            marks.append(mark[-limit:])
            states.append(np.asarray(m["state"], dtype=np.float32))
            understand.append(i % 6 == 0)
        length = max(len(s) for s in seqs)
        ids = torch.full((b, length), END, dtype=torch.long)
        targets = torch.full((b, length - 1), -100, dtype=torch.long)
        for i, (seq, mark) in enumerate(zip(seqs, marks, strict=True)):
            ids[i, : len(seq)] = torch.tensor(seq)
            keep = torch.tensor(mark[1:])
            targets[i, : len(seq) - 1] = torch.where(keep, torch.tensor(seq[1:]), torch.tensor(-100))
        state = torch.tensor(np.stack(states), device=self.device)
        given = state.clone()
        mask = torch.tensor(understand, device=self.device)
        given[mask] = 0.0
        ids, targets = ids.to(self.device), targets.to(self.device)
        with self._autocast():
            logits, meanings = self.model(ids[:, :-1], given)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
            last = torch.tensor([len(s) - 2 for s in seqs], device=self.device)
            meaning = meanings[torch.arange(b, device=self.device), last].float()
            if mask.any():
                loss = loss + 0.5 * F.mse_loss(meaning[mask], state[mask, 0])
        return loss

    # --- testing ---------------------------------------------------------------------------------------

    def evaluate(self, level: Level, data: dict) -> dict:
        return evaluate(ScratchReader(self.model, self.tok), level, data)

    def report(self) -> list[str]:
        """The report card."""
        lines = [
            f"Language cortex: {self.progress['size']}, "
            + (f"{self.model.parameters_count() / 1e6:.1f}M connections, " if self.model else "")
            + f"{len(self.tok):,} vocabulary pieces, {self.progress['steps']:,} study steps, "
            f"{self.progress['tokens'] / 1e6:.1f}M tokens read."
        ]
        for level in LEVELS:
            record = self.progress["levels"].get(str(level.number))
            if record is None:
                lines.append(f"  {level.number}. {level.name}: not started")
                continue
            lines.append(
                f"  {level.number}. {level.name}: {record['status']} after {record['steps']:,} steps"
                + (f" — {describe(level, record['tests'])}" if record.get("tests") else "")
            )
        return lines


def read_level(level: Level, web: Web, cache: Path, urls: dict, scale: dict) -> sources.Reading:
    """A level's reading, from the internet the first time and from the cache after that."""
    s = level.source
    if s == "tinystories":
        return sources.tinystories(web, cache, scale["tinystories"], urls)
    if s == "children":
        return sources.gutenberg(web, cache, sources.CHILDREN, "children", urls)
    if s == "classics":
        return sources.gutenberg(web, cache, sources.CLASSICS, "classics", urls)
    if s in ("simplewiki", "wikipedia"):
        return sources.wiki(web, cache, urls[s], scale["articles"], s)
    if s == "squad":
        return sources.squad(web, cache, urls)
    if s == "gsm8k":
        return sources.gsm8k(web, cache, urls)
    raise ValueError(f"unknown source {s}")


LIVES = ((101, 4.0), (202, 4.0), (404, 4.0), (505, 3.0), (606, 3.0), (303, 3.0))  # (seed, days); the last is held out


def simulated_moments(cache: Path, log: Callable[[str], None], lives: tuple = LIVES) -> dict:
    """Moments from simulated lives of Haven, each with its state tokens and words for it (cached)."""
    path = cache / "grounded.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as npz:
            meta = json.loads(str(npz["meta"]))
            states = npz["states"]
    else:
        log("  living a few simulated lives to learn words for its own states…")
        moments = [m for seed, days in lives for m in gather(seed=seed, days=days)]
        states = np.stack([m.pop("state") for m in moments]).astype(np.float32)
        meta = moments
        cache.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            np.savez_compressed(f, states=states, meta=np.array(json.dumps(meta)))
    for m, state in zip(meta, states, strict=True):
        m["state"] = state
    cut = int(len(meta) * 0.86)  # about the last simulated life is held out for testing
    return {"train": meta[:cut], "held": meta[cut:]}


def evaluate(reader, level: Level, data: dict, limit: int = 120) -> dict:
    """Run a level's tests, on held-out material, with any kind of cortex."""
    items = data["items"]
    results = {}
    for test in level.tests:
        if test == "fluency":
            results[test] = fluency(reader, data["held_out"], max_tokens=100 * limit)
        elif test in ("cloze", "next sentence", "questions", "math"):
            results[test] = choose(reader, items.get(test, [])[:limit])
        elif test == "self-report":
            results[test] = self_report(reader, items["self-report"][: max(limit // 2, 10)])
        elif test == "understanding":
            results[test] = understanding(reader, items["understanding"])
        elif test == "conversation":
            results[test] = conversation(reader, items.get("conversation", [])[:limit])
    return {k: v for k, v in results.items() if v is not None}


def describe(level: Level, results: dict) -> str:
    parts = []
    for test, (direction, mark) in level.tests.items():
        value = results.get(test)
        if value is None:
            continue
        if test == "fluency":
            parts.append(f"fluency {value:.2f} bits/byte (mark ≤ {mark})")
        else:
            parts.append(f"{test} {value:.0%} (mark {mark:.0%})")
    return " · ".join(parts)


def _sample(texts: list[str], limit: int, rng: random.Random) -> list[str]:
    total = sum(len(t) for t in texts)
    if total <= limit:
        return texts
    keep = limit / total
    return [t for t in texts if rng.random() < keep]


__all__ = ["LEVELS", "SCALE", "Trainer", "math", "pick_device"]


def conversation_ids(tok: Tokenizer, moment: dict, rng: random.Random) -> tuple[list[int], list[bool]]:
    """A conversation at one moment, as tokens: what it recalls, then what's said, turn by turn.

    Also returns which tokens are Haven's to say (its answers, and knowing when to stop).
    """
    thought, turns = converse(moment, rng)
    ids = [THINK, *tok.encode(thought)]
    mark = [False] * len(ids)
    for question, answer in ((t.said, t.answer) for t in turns):
        asked = [YOU, *tok.encode(question), HAVEN]
        ids += asked
        mark += [bool(mark) and mark[-1]] + [False] * (len(asked) - 1)  # the turn after its answer ends it
        said = tok.encode(answer)
        ids += said
        mark += [True] * len(said)
    ids.append(END)
    mark.append(True)
    return ids, mark


def conversation_items(moments: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Held-out questions, at held-out moments, with the answers its state and what comes to mind give."""
    rng = random.Random(seed)
    items = []
    for m in rng.sample(moments, min(n, len(moments))):
        thought, turns = converse(m, rng, turns=1)
        turn = rng.choice(turns)
        items.append(
            {"state": m["state"], "notes": thought, "question": turn.said, "answer": turn.answer, "kind": turn.kind}
        )
    return items
