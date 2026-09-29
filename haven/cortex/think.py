"""Thinking in words: inner speech that passes through Haven's global workspace.

When someone talks to Haven, its language cortex (its own, grown from scratch) reads
Haven's state straight from its workspace, recalls what it knows in a line of inner
speech, and drafts a few replies. How sure it is comes from its own signals: how likely
it found its own words, and whether its drafts agree. What it says enters the workspace
as a thought, competing with everything else Haven is aware of, and is remembered like
any other experience. When someone asks about something it has never learned about, it
says so, and if it's allowed to, it reads about it in an encyclopedia and tells them
what it read (and keeps it: see library.py). When nobody is talking with it, it now and
then reads about something it's curious about. While it sleeps, it goes over moments of
its day, and keeps what it learned only if it still talks as well (see sleep.py).
"""

from __future__ import annotations

import collections
import contextlib
import copy
import json
import math
import random
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
MOMENTS = 40  # moments of its day it needs, at least, to learn from them in its sleep


def agreement(texts: list[str]) -> float:
    """How much independent drafts say the same thing (word overlap, averaged over pairs)."""
    sets = [set(re.findall(r"[a-z']+", t.lower())) for t in texts]
    pairs = [(a, b) for i, a in enumerate(sets) for b in sets[i + 1 :]]
    if not pairs:
        return 0.0
    return float(np.mean([len(a & b) / max(len(a | b), 1) for a, b in pairs]))


def repeats(text: str, context: str = "") -> bool:
    """Whether words go round in circles ("I can climb the hill and climb the hill"): say something more often
    than it came to mind."""

    def grams(words: str) -> dict[tuple, int]:
        found = re.findall(r"[a-z']+", words.lower())
        counted: dict[tuple, int] = {}
        for i in range(len(found) - 2):
            counted[tuple(found[i : i + 3])] = counted.get(tuple(found[i : i + 3]), 0) + 1
        return counted

    known = grams(context)
    return any(n > 1 and n > known.get(gram, 0) for gram, n in grams(text).items())


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
        self._library = None
        self.shown: dict[str, set[int]] = {}  # what it has told of each thing it read, in this conversation
        self.last_read: str | None = None  # what it last told them it read
        self.wondered: set[str] = set()  # what it has tried reading about out of curiosity
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

    def readings(self) -> list[tuple[str, str]]:
        """What it has read, (title, first sentence), oldest first: the little book it was born having read, then
        whatever it has looked up since."""
        from .talk import READINGS, first_sentence

        book = [(title, sentence) for _, title, sentence in READINGS]

        path = self.root / "cortex" / "readings.jsonl"
        try:
            stamp = path.stat().st_mtime
        except FileNotFoundError:
            return book
        if getattr(self, "_read_stamp", None) != stamp:
            found = []
            for line in path.read_text().splitlines()[-500:]:
                with contextlib.suppress(ValueError, KeyError, TypeError):
                    item = json.loads(line)
                    found.append((str(item["title"]), first_sentence(str(item["text"]))))
            self._read, self._read_stamp = found, stamp
        return [*book, *self._read]

    def listen(self, mind, text: str) -> str | None:
        """What it takes from what someone said: their name, or something about themselves or the world to remember."""
        from .talk import introduced, lesson, statement

        name = introduced(text)
        if name:
            if name != mind.person:
                mind._note(mind.tick, f"learned that the person talking with it is called {name}")
            mind.person = name
            return None
        fact = statement(text)
        if fact:
            mind.told = [*(item for item in mind.told if item[1] != fact), (mind.tick, fact)][-50:]
            mind._note(mind.tick, f"was told that {fact}")
            return fact
        taught = lesson(text)
        if taught:
            mind.lessons = [*(item for item in mind.lessons if item[1] != taught), (mind.tick, taught)][-200:]
            mind._note(mind.tick, f"was taught that {taught}")
        return taught

    @property
    def library(self):
        """Everything it has read (made the first time it's needed)."""
        if self._library is None:
            from .library import Library

            self._library = Library(self.root)
        self._library.refresh()
        return self._library

    def recollect(self, mind, text: str, req=None) -> list[str]:
        """What else comes to mind at someone's words: being asked to do something, a sum worked out, or what it read
        (that answers them, that it read lately, or that it could tell them)."""
        from .talk import (
            FACT_ASK,
            LATELY,
            MORE,
            about_haven,
            about_them,
            bare,
            best_lesson,
            lately_note,
            memo,
            mentioned,
            more_note,
            request_note,
            sum_note,
            sum_of,
            thing_note,
        )

        if req is not None:
            thing = memo(mind)["things"][req.thing]
            return [thing_note(req.thing, thing["stats"], thing["where"]), request_note(req)]
        worked = sum_of(text)
        if worked is not None:
            return [sum_note(worked)]
        library = self.library
        if LATELY.search(bare(text)):  # what it has read lately (not its little book)
            return [lately_note([t for t in library.titles() if library.sources[t] != "book"][-3:])]
        if FACT_ASK.search(bare(text)):  # something it read that it hasn't told them yet, the latest first
            fresh = [t for t in library.titles() if not self.shown.get(t)]
            if not fresh:
                return []
            title = fresh[-1] if library.sources[fresh[-1]] != "book" else random.choice(fresh)
            self.shown.setdefault(title, set()).add(0)
            self.last_read = title
            return [f"I read about {title}: {library.sentence(title, 0)}"]
        if MORE.search(text):  # more of what it read
            title = library.title_for(text) or self.last_read
            if title is None or title not in library:
                return []
            told = self.shown.setdefault(title, set())
            self.last_read = title
            if not told:
                told.add(0)
                return [f"I read about {title}: {library.sentence(title, 0)}"]
            first = min(told)  # what it told them, and what it read next (as it practised it)
            index = next((i for i in range(len(library.docs[title])) if i not in told), None)
            if index is not None:
                told.add(index)
            more = more_note(title, None if index is None else library.sentence(title, index))
            return [f"I read about {title}: {library.sentence(title, first)}", more]
        if about_them(text) or mentioned(text):  # about them, or about its valley: not what it read
            return []
        if about_haven(text) and not library.title_for(text):  # about it ("who made you?"), not something it read
            return []
        taught = best_lesson(text, [item for _, item in mind.lessons])
        if taught:  # something someone taught it
            return [f"You told me that {taught}."]
        found = library.find(text)
        if found is None:
            return []
        title, index = found
        self.shown.setdefault(title, set()).add(index)
        self.last_read = title
        return [f"I read about {title}: {library.sentence(title, index)}"]

    def look_up(self, topic: str, life=None, why: str = "asked") -> str | None:
        """Read about something in the Simple English Wikipedia (`why`: "asked", or "curious"). Returns the title of
        what it read."""
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
            f.write(json.dumps({"title": title, "text": text[:20000], "time": time.time(), "why": why}) + "\n")
        if life is not None:
            life._emit(
                "event",
                f"read about {title}" + {"asked": " to answer that", "curious": ", out of curiosity"}.get(why, ""),
            )
        self._tell("read", title)
        return title

    # --- reading out of curiosity ------------------------------------------------------------------

    def wonder(self, life) -> str | None:
        """Something it would like to read about: what people talked with it about lately, the things it has met in
        its valley, or what something it read says a thing is. Not what it has read, or tried to, already."""
        from .library import VALLEY, topics_in, what_it_is

        with life.lock:
            said = [t["text"] for t in life.conversation[-20:] if t["who"] == "you"]
            mind = life.mind
            skip = {mind.person.lower()} if mind.person else set()
            met = sorted(
                (n for n, r in mind.things.items() if n in VALLEY and r.get("seen")),
                key=lambda n: -mind.things[n]["seen"],
            )
        library = self.library
        wanted = [topic for text in reversed(said) for topic in topics_in(text, skip)]
        wanted += [VALLEY[name] for name in met]
        for title in reversed(library.titles("curious") + library.titles("asked")):
            kind = what_it_is(library.sentence(title, 0) or "")
            if kind:
                wanted.append(kind)
        return next((t for t in wanted if t.lower() not in self.wondered and not library.title_for(t)), None)

    def read_for_fun(self, life) -> str | None:
        """Read about something it's curious about. Returns what it read about, if it read anything."""
        topic = None if self.web is None else self.wonder(life)
        if topic is None:
            return None
        self.wondered.add(topic.lower())
        return self.look_up(topic, life, why="curious")

    def notice(self, mind) -> None:
        """A moment of its day, noted to go over while it sleeps (called now and then while it's awake)."""

    def sleep_on_it(self, life=None) -> dict | None:
        """Learning in its sleep, from the moments of its day (see sleep.py). Returns how it went, if it did."""
        return None


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
        self.day: collections.deque = collections.deque(maxlen=240)  # moments of its day, to go over in its sleep
        self._others = self._rehearse = None  # the test about other lives, and another life to go over (made once)

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
        from .library import asked_to_read
        from .talk import DONT_KNOW, MORE, notes, request
        from .tokenizer import HAVEN, THINK, YOU

        torch = self.torch
        wanted = asked_to_read(text)
        if wanted and self.web is not None and not self.library.title_for(wanted):
            self.look_up(wanted, life)  # asked to read about something: it reads it first
        with life.lock:  # what it's experiencing as it's asked, and what comes to mind
            mind = life.mind
            just = self.listen(mind, text)
            req = None if just else request(text)
            extra = [] if just else self.recollect(mind, text, req)
            state = torch.tensor(mind_state(mind), device=self.device).unsqueeze(0)
            known = notes(mind, text, (), just, extra)
            # It answers from what comes to mind, as it learned to; only "tell me more" needs what was just said.
            history = [t for t in life.conversation[-3:-1] if not t.get("earlier")] if MORE.search(text) else []
        with self.model_lock:
            heard = self.model.meaning([YOU, *self.tok.encode(text)], state).float().cpu().numpy()
        with life.lock:
            mind.understand(text, heard)  # what it made of what was said comes to mind

        def prompt_for(known: str) -> list[int]:
            prompt = [THINK, *self.tok.encode(known)]
            for turn in history:
                prompt += [YOU if turn["who"] == "you" else HAVEN, *self.tok.encode(turn["text"])]
            return prompt + [YOU, *self.tok.encode(text), HAVEN]

        words, confidence = self._say(prompt_for(known), state, drafts)
        topic = topic_of(text)
        if words.startswith(DONT_KNOW[:24]) and self.web is not None and topic and not self.library.title_for(topic):
            if self.look_up(topic, life):  # it didn't know, so it reads about it, and says what it read
                with life.lock:
                    extra = self.recollect(mind, text)
                    known = notes(mind, text, (), just, extra)
                if extra:
                    words, confidence = self._say(prompt_for(known), state, drafts)
        if req is not None and words.startswith("Okay, I'll"):
            with life.lock:
                mind.take_errand(req.do, req.thing, req.action, req.need)  # it said it would, so it sets off
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
            for i in range(drafts):  # its most careful draft first, then freer ones
                tokens, logprobs = self.model.generate(
                    prompt[-(self.model.cfg.context - 100) :],
                    state,
                    max_new=100,
                    temperature=0.0 if i == 0 else 0.6,
                    stop=(END, YOU),
                )
                words = self.tok.decode(tokens).strip()
                found.append((words, math.exp(float(np.mean(logprobs))) if logprobs else 0.0))
        mind = self.tok.decode(prompt)  # what came to mind, and what was said
        found.sort(key=lambda d: (not repeats(d[0], mind), d[1]))  # a draft that goes round in circles loses
        for words, _ in found:  # the ones it didn't pick pass by as thoughts; the likeliest comes last
            self._tell("draft")
            self._tell("words", words)
        words, likely = found[-1]
        return words, 0.5 * likely + 0.5 * agreement([d[0] for d in found])

    def notice(self, mind) -> None:
        from .grounding import moment_of

        self.day.append(moment_of(mind))

    def own_readings(self, most: int = 20) -> tuple:
        """What it has read besides its little book (the latest), to practise telling about in its sleep."""
        from .book import Entry

        library = self.library
        titles = [t for t in library.titles() if library.sources[t] != "book"][-most:]
        return tuple(Entry(re.sub(r"\s*\(.*?\)", "", t), t, tuple(library.docs[t][:3])) for t in titles)

    def sleep_on_it(self, life=None) -> dict | None:
        """Practise talking about moments of its day (and what it read) on a copy of its cortex, and keep the copy
        only if it talks at least as well (see sleep.py)."""
        from . import sleep

        day = list(self.day)
        if len(day) < MOMENTS:
            return None
        readings = self.own_readings()
        day = [{**m, "readings": readings} for m in day]  # it goes over what it read, too
        if self._others is None:
            self._others, self._rehearse = sleep.other_lives(), sleep.rehearsal()
        with self.model_lock:
            current = copy.deepcopy(self.model)
        learned, report = sleep.night(current, self.tok, day, self._others, random.Random(), rehearse=self._rehearse)
        if learned is not None:
            with self.model_lock:
                self.model.load_state_dict(learned.state_dict())
            self.save()
        self.day.clear()
        report["time"] = time.time()
        with (self.root / "cortex" / "nights.jsonl").open("a") as f:
            f.write(json.dumps(report) + "\n")
        if life is not None:
            b, a = report["before"]["own day"], report["after"]["own day"]
            life._emit(
                "event",
                f"went over its day in its sleep, and learned from it (answers about its day: {b:.0%} → {a:.0%})"
                if learned is not None
                else "went over its day in its sleep, but kept what it knew: practising didn't help this time",
            )
        return report

    def save(self) -> None:
        torch = self.torch
        path = self.root / "cortex" / "cortex.pt"
        with self.model_lock:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            checkpoint["model"] = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
            tmp = path.with_suffix(".pt.tmp")
            torch.save(checkpoint, tmp)
            tmp.replace(path)


def make_thinker(spec: str, root: Path, web: Web | None = None) -> tuple[Thinker | None, str]:
    """Its language cortex ("own"), or none. The first time, it gets the one it starts life with."""
    from . import starter

    web = web if web is not None else Web()
    if spec == "none":
        return None, ""
    if spec != "own":
        return None, f'Its language cortex is its own: "{spec}" isn\'t one it can use.'
    installed = starter.install(root)
    moved = (
        " (its old one was grown for its old world: it's kept in archive/)"
        if installed in ("replaced", "retired")
        else ""
    )
    if not (Path(root) / "cortex" / "cortex.pt").exists():
        return (
            None,
            f"It has no language cortex yet{moved}. Train one with: haven learn. It can still learn words from you.",
        )
    thinker = OwnThinker(root, web=web)
    return thinker, f"Language cortex: {thinker.describe()}.{moved}"
