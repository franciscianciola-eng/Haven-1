"""Thinking in words: inner speech that passes through Haven's global workspace.

When someone talks to Haven, its language cortex (its own, grown from scratch) reads
Haven's state straight from its workspace, recalls what it knows in a line of inner
speech, and drafts a few replies. How sure it is comes from its own signals: how likely
it found its own words, and whether its drafts agree. What it says enters the workspace
as a thought, competing with everything else Haven is aware of, and is remembered like
any other experience. When someone asks about something it has never learned about, it
says so, and if it's allowed to, it reads about it in an encyclopedia and tells them
what it read. While it sleeps, it goes over what it read and what people said to it.
"""

from __future__ import annotations

import contextlib
import json
import math
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..web import Web, WebError
from . import sources
from .grounding import mind_state

_COMMON = (
    "what when where which who whom whose why how does did do is are was were be been being have has had "
    "the a an and or but if then than that this these those there their they them you your yours about "
    "with from into onto for of on in at by to it its it's can could would should will shall may might "
    "must tell know think please"
)
STOPWORDS = frozenset(_COMMON.split())


def agreement(texts: list[str]) -> float:
    """How much independent drafts say the same thing (word overlap, averaged over pairs)."""
    sets = [set(re.findall(r"[a-z']+", t.lower())) for t in texts]
    pairs = [(a, b) for i, a in enumerate(sets) for b in sets[i + 1 :]]
    if not pairs:
        return 0.0
    return float(np.mean([len(a & b) / max(len(a | b), 1) for a, b in pairs]))


def topic_of(text: str) -> str | None:
    """What a question is about: a name if there is one, else its most specific word."""
    names = re.findall(r"(?<!^)(?<![.?!] )\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)", text)
    if names:
        return names[0]
    words = [w for w in re.findall(r"[A-Za-z]+", text) if w.lower() not in STOPWORDS and len(w) >= 4]
    return max(words, key=len) if words else None


class Thinker:
    """What Life needs from a language cortex."""

    name = "none"

    def __init__(self, root: Path, web: Web | None = None):
        self.root = Path(root)
        self.web = web
        self.busy = threading.Lock()
        # Someone following along as it answers (the app): hears ("draft" | "words" | "pondering" | "thought" |
        # "reading" | "read", text) as a reply takes shape.
        self.listener: Callable[[str, str], None] | None = None

    def describe(self) -> str:
        return self.name

    def _tell(self, kind: str, text: str = "") -> None:
        if self.listener is not None:
            with contextlib.suppress(Exception):  # a broken listener mustn't stop a thought
                self.listener(kind, text)

    def respond_later(self, life, text: str) -> None:
        if self.busy.locked():
            return  # it's still thinking about the last thing
        threading.Thread(target=self._respond, args=(life, text), daemon=True, name="haven-thinking").start()

    def _respond(self, life, text: str) -> None:
        with self.busy:
            try:
                answer, confidence = self.deliberate(life, text)
            except Exception as error:  # noqa: BLE001  thinking going wrong mustn't end a life
                life._emit("event", f"couldn't put a thought into words ({error})")
                return
            if answer:
                life.reply(answer, f"{self.name}, confidence {confidence:.0%}")
                self.remember({"you": text, "haven": answer, "confidence": confidence, "time": time.time()})

    def deliberate(self, life, text: str) -> tuple[str, float]:
        raise NotImplementedError

    def remember(self, exchange: dict) -> None:
        path = self.root / "cortex" / "conversations.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(exchange) + "\n")

    def look_up(self, topic: str, life=None) -> str | None:
        """Read about something in the Simple English Wikipedia. Returns a short excerpt."""
        if self.web is None:
            return None
        self._tell("reading", topic)
        try:
            found = sources.article(self.web, sources.URLS["simplewiki"], topic)
        except WebError:
            return None
        if found is None:
            return None
        title, text = found
        path = self.root / "cortex" / "readings.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps({"title": title, "text": text[:20000], "time": time.time()}) + "\n")
        if life is not None:
            life._emit("event", f"read about {title} to answer that")
        self._tell("read", title)
        first = " ".join(text.split())[:700]
        return f"{title}: {first.rsplit('. ', 1)[0]}."

    def consolidate(self) -> None:
        """Learn from recent conversations and reading (called while Haven sleeps)."""


class OwnThinker(Thinker):
    name = "its own cortex"

    def __init__(self, root: Path, device: str = "cpu", web: Web | None = None):  # it's small: the processor is plenty
        super().__init__(root, web)
        import torch

        from .model import Cortex, CortexConfig
        from .tokenizer import Tokenizer
        from .train import pick_device

        self.torch = torch
        self.device = pick_device(device)
        checkpoint = torch.load(self.root / "cortex" / "cortex.pt", map_location="cpu", weights_only=False)
        self.progress = checkpoint["progress"]
        self.tok = Tokenizer.load(self.root / "cortex" / "tokenizer.json")
        self.model = Cortex(CortexConfig(**checkpoint["config"]))
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device).eval()
        self.model_lock = threading.Lock()
        self._optimizer = None
        self._since_save = 0

    def describe(self) -> str:
        from .curriculum import LEVELS

        done = [
            LEVELS[int(n) - 1].name.lower()
            for n, r in sorted(self.progress["levels"].items(), key=lambda item: int(item[0]))
            if r.get("status") in ("passed", "moved on", "plateaued", "studied")
        ]
        return f"its own, grown from scratch ({self.model.parameters_count() / 1e6:.1f}M connections); " + (
            f"it has learned: {', '.join(done)}" if done else "it hasn't studied yet"
        )

    def deliberate(self, life, text: str, drafts: int = 3) -> tuple[str, float]:
        from .talk import DONT_KNOW, notes
        from .tokenizer import HAVEN, THINK, YOU

        torch = self.torch
        with life.lock:  # what it's experiencing as it's asked, and what it knows
            mind = life.mind
            state = torch.tensor(mind_state(mind), device=self.device).unsqueeze(0)
            known = notes(mind)
            history = [t for t in life.conversation[-5:-1] if not t.get("earlier")]
        with self.model_lock:
            heard = self.model.meaning([YOU, *self.tok.encode(text)], state).float().cpu().numpy()
        with life.lock:
            mind.understand(text, heard)  # what it made of what was said comes to mind
        prompt = [THINK, *self.tok.encode(known)]
        for turn in history:
            prompt += [YOU if turn["who"] == "you" else HAVEN, *self.tok.encode(turn["text"])]
        prompt += [YOU, *self.tok.encode(text), HAVEN]
        words, confidence = self._say(prompt, state, drafts)
        if words.startswith(DONT_KNOW[:24]) and self.web is not None:  # "I don't know. I haven't…" (not "…yet")
            topic = topic_of(text)
            excerpt = self.look_up(topic, life) if topic else None
            if excerpt:  # it didn't know, so it read about it: it tells them what it read
                title, _, read = excerpt.partition(": ")
                first = read.split(". ")[0].rstrip(".") + "."
                words, confidence = f"I didn't know, so I read about {title}. It says: {first}", 0.5
                self._tell("draft")
                self._tell("words", words)
        with self.model_lock:
            meaning = self.model.meaning([HAVEN, *self.tok.encode(words)], state).float().cpu().numpy()
        with life.lock:
            mind.think(words, meaning, confidence)  # what it says enters its workspace
        return words, confidence

    def _say(self, prompt: list[int], state, drafts: int) -> tuple[str, float]:
        """A few drafts of a reply; the likeliest is what it says, and how much they agree is part of how sure it is."""
        from .tokenizer import END, YOU

        found = []
        with self.model_lock:
            for _ in range(drafts):
                tokens, logprobs = self.model.generate(
                    prompt[-(self.model.cfg.context - 60) :],
                    state,
                    max_new=60,
                    temperature=0.6,
                    stop=(END, YOU),
                )
                words = self.tok.decode(tokens).strip()
                found.append((words, math.exp(float(np.mean(logprobs))) if logprobs else 0.0))
        found.sort(key=lambda d: d[1])
        for words, _ in found:  # the ones it didn't pick pass by as thoughts; the likeliest comes last
            self._tell("draft")
            self._tell("words", words)
        words, likely = found[-1]
        return words, 0.5 * likely + 0.5 * agreement([d[0] for d in found])

    def consolidate(self, steps: int = 2) -> None:
        """A little learning in its sleep, from what it read and what people said to it."""
        from .tokenizer import END

        torch = self.torch
        texts = []
        for name in ("conversations.jsonl", "readings.jsonl"):
            path = self.root / "cortex" / name
            if path.exists():
                for line in path.read_text().splitlines()[-200:]:
                    item = json.loads(line)
                    text = item.get("you") if "you" in item else item.get("text", "")[:4000]
                    if text:
                        texts.append(self.tok.encode(text) + [END])
        if not texts:
            return
        rng = np.random.default_rng()
        with self.model_lock:
            model = self.model
            if self._optimizer is None:
                self._optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.0)
            model.train()
            for _ in range(steps):
                ids = texts[int(rng.integers(len(texts)))][: model.cfg.context + 1]
                if len(ids) < 3:
                    continue
                batch = torch.tensor([ids], device=self.device)
                logits, _ = model(batch[:, :-1])
                loss = torch.nn.functional.cross_entropy(logits[0].float(), batch[0, 1:])
                self._optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self._optimizer.step()
            model.eval()
            self._since_save += steps
            if self._since_save >= 50:
                self.save()

    def save(self) -> None:
        torch = self.torch
        path = self.root / "cortex" / "cortex.pt"
        with self.model_lock:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            checkpoint["model"] = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
            tmp = path.with_suffix(".pt.tmp")
            torch.save(checkpoint, tmp)
            tmp.replace(path)
            self._since_save = 0


def make_thinker(spec: str, root: Path, web: Web | None = None) -> tuple[Thinker | None, str]:
    """Its language cortex ("own"), or none. The first time, it gets the one it starts life with."""
    from . import starter

    web = web if web is not None else Web()
    if spec == "none":
        return None, ""
    if spec != "own":
        return None, f'Its language cortex is its own: "{spec}" isn\'t one it can use.'
    starter.install(root)
    if not (Path(root) / "cortex" / "cortex.pt").exists():
        return None, "It has no language cortex yet (train one with: haven learn). It can still learn words from you."
    thinker = OwnThinker(root, web=web)
    return thinker, f"Language cortex: {thinker.describe()}."
