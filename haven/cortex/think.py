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


def repeats(text: str, context: str = "", times: int = 1) -> bool:
    """Whether words go round in circles ("I can climb the hill and climb the hill"): say something more often
    than it came to mind (by more than `times`: a long telling can say "said the fox" twice)."""

    def grams(words: str) -> dict[tuple, int]:
        found = re.findall(r"[a-z']+", words.lower())
        counted: dict[tuple, int] = {}
        for i in range(len(found) - 2):
            counted[tuple(found[i : i + 3])] = counted.get(tuple(found[i : i + 3]), 0) + 1
        return counted

    known = grams(context)
    return any(n > times and n > known.get(gram, 0) for gram, n in grams(text).items())


def garbled(text: str) -> bool:
    """Whether a draft has come out as babble ("Tababababa", a "word" far longer than any it knows)."""
    return bool(re.search(r"[A-Za-z]{19,}|(\w{1,3})\1{5,}", text))


def unfounded(text: str, context: str) -> bool:
    """Whether a draft says a number or a name that nothing in what came to mind, or was said, says (a slip in copying
    "84" or "Mehmet"; the answers it learned never do)."""
    numbers = set(re.findall(r"\d+(?:\.\d+)?", context))
    if any(n not in numbers for n in re.findall(r"\d+(?:\.\d+)?", text)):
        return True
    low = context.lower()
    return any(name.lower() not in low for name in re.findall(r"(?<=[a-z,;:] )[A-Z][a-z]+", text))


def topic_of(text: str) -> str | None:
    """What a question is about: a name if there is one, else its most specific word."""
    names = re.findall(r"(?<!^)(?<![.?!] )\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)", text)
    if names:
        return names[0]
    words = [w for w in re.findall(r"[A-Za-z]+", text) if w.lower() not in STOPWORDS and len(w) >= 4]
    return max(words, key=len) if words else None


class Thread:
    """What a conversation is about lately: what it wondered and asked, what it said of its own, what it decided."""

    LASTS = 600.0  # seconds what it wondered stays open for an answer

    def __init__(self):
        self.wondered: tuple[str, float] | None = None
        self.own: str | None = None  # what it said of its own, for "what about you?"
        self.reason: str | None = None  # why it said what it said, for "why?"
        self.request = None  # (what it was asked, what it decided), for being asked again

    def wonder(self, question: str) -> None:
        self.wondered = (question, time.monotonic())

    def wondering(self) -> str | None:
        found, self.wondered = self.wondered, None
        return found[0] if found and time.monotonic() - found[1] < self.LASTS else None

    def mine(self, own: str) -> None:
        from .engage import own_reason

        self.own = own
        self.reason = own_reason(own) or self.reason  # ("...the butterfly. They fly around my valley.": the reason)

    def asked_to(self, req, decision) -> None:
        self.request = (req, decision)
        if decision.reason:
            self.reason = decision.reason

    def declined(self) -> bool:
        from .engage import WILLING

        return self.request is not None and self.request[1].answer not in WILLING


def own_favorites(mind) -> dict[str, str]:
    """Its own favorites, as it says them (only what it has found out for itself)."""
    found = {}
    season = mind.character.favorite_season()
    if season:
        found["season"] = f"My favorite season is {season}."
    days = mind.age / 1200
    found["age"] = f"I'm {int(days)} days old." if days < 24 else f"I'm {int(days // 24)} years old."
    return found


own_favorites_of = own_favorites


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
        self._hearing = None
        self.told_stories: set[str] = set()  # the stories it has told in this conversation

    def describe(self) -> str:
        return self.name

    @property
    def hearing(self):
        """What it hears at home: bedtime stories, and what people say to it (see hearing.py)."""
        if self._hearing is None:
            from .hearing import Listening

            self._hearing = Listening(self.root, self.web)
        return self._hearing

    def bedtime_story(self, life=None):
        """Hear the next passage of the book it's hearing, as it falls asleep. Returns the story, if it heard one."""
        found = self.hearing.bedtime()
        if found is None:
            return None
        story, _ = found
        if life is not None:
            life._emit("event", f"heard a bedtime story: {story.title}")
        return story

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
        from .stories import STORIES
        from .talk import (
            FACT_ASK,
            LATELY,
            MORE,
            about_haven,
            about_them,
            bare,
            best_lesson,
            heard_note,
            lately_note,
            memo,
            mentioned,
            more_note,
            request_note,
            story_asked,
            story_for,
            sum_note,
            sum_of,
            tale_note,
            thing_note,
        )

        if req is not None:
            thing = memo(mind)["things"][req.thing]
            return [thing_note(req.thing, thing["stats"], thing["where"]), request_note(req)]
        worked = sum_of(text)
        if worked is not None:
            return [sum_note(worked)]
        asked = story_asked(text)
        if asked is not None:  # a story, or what it heard
            heard = self.hearing.stories()
            if asked == "heard":
                return [heard_note(heard[-1], True) if heard else heard_note(random.choice(STORIES), False)]
            story = story_for(text, [*STORIES, *heard])
            if story is None:  # the latest it heard, unless it has told that one already: then another
                fresh = [s for s in heard[-3:] if s.title not in self.told_stories]
                fresh = fresh or [s for s in (*heard, *STORIES) if s.title not in self.told_stories] or list(STORIES)
                story = fresh[-1] if fresh[-1] in heard else random.choice(fresh)
            self.told_stories.add(story.title)
            return [tale_note(story)]
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
        from .starter import unpacked
        from .tokenizer import Tokenizer
        from .train import pick_device

        self.torch = torch
        self.device = pick_device(device)
        checkpoint = torch.load(self.root / "cortex" / "cortex.pt", map_location="cpu", weights_only=False)
        self.progress = checkpoint["progress"]
        self.tok = Tokenizer.load(self.root / "cortex" / "tokenizer.json")
        self.model = Cortex(CortexConfig(**checkpoint["config"]))
        self.model.load_state_dict(unpacked(checkpoint["model"]))
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
        grew, here = int(self.progress.get("heard", {}).get("words", 0)), self.hearing.words()
        heard = f"; it grew up hearing {grew / 1e6:.1f} million words" if grew else ""
        if here:
            heard += f"{',' if grew else '; it has heard'} {here:,} {'more ' if grew else ''}words here"
        return (
            f"its own, grown from scratch ({self.model.parameters_count() / 1e6:.1f}M connections); "
            + (f"it has learned: {', '.join(done)}" if done else "it hasn't studied yet")
            + heard
        )

    def deliberate(self, life, text: str, drafts: int = 3) -> tuple[str, float]:
        from ..will import consider
        from . import engage
        from .library import asked_to_read
        from .talk import DONT_KNOW, addressed, answer_to, notes, request, request_note, story_asked, sum_of
        from .tokenizer import HAVEN, THINK, YOU

        torch = self.torch
        self.hearing.heard_said(text)  # (what people say to it, it goes over in its sleep, as it does what it hears)
        text = addressed(text, life.mind.me.name)  # ("Hi Pip!" is being greeted, as "Hi Haven!" is)
        if getattr(life, "talk", None) is None:
            life.talk = Thread()
        talk = life.talk  # (what this conversation is about, lately)
        asked = (
            life.question() if hasattr(life, "question") else None
        )  # it asked them something, and this is the answer
        meant = answer_to(asked, text) if asked else None  # ("pizza": "My favorite food is pizza.")
        wondered = talk.wondering()  # it wondered something about what they told it, and this may be the answer
        learned = engage.learned_from(wondered, text) if wondered and not meant else None
        wanted = None if meant or learned else asked_to_read(text)
        if wanted and self.web is not None and not self.library.title_for(wanted):
            self.look_up(wanted, life)  # asked to read about something: it reads it first
        decision = None
        with life.lock:  # what it's experiencing as it's asked, and what comes to mind
            mind = life.mind
            if learned:  # the answer to what it wondered: something learned
                mind.lessons = [*(item for item in mind.lessons if item[1] != learned), (mind.tick, learned)][-200:]
                mind._note(mind.tick, f"was taught that {learned}")
                just = learned
            else:
                just = self.listen(mind, meant or text)
            req = None if just else request(text)
            extra = [] if just else self.recollect(mind, text, req)
            follow, question = self._follow(mind, talk, text, just, meant or text, learned)
            if req is not None:  # asked to do something: it makes up its own mind
                decision = consider(mind, req)
                extra = [engage.decision_note(req.do, decision) if e == request_note(req) else e for e in extra]
                talk.asked_to(req, decision)
            elif engage.INSIST.match(text) and talk.declined():  # asked again: it may give in
                req, _ = talk.request
                decision = consider(mind, req, insisted=True)
                follow.append(engage.decision_note(req.do, decision, again=True))
                talk.asked_to(req, decision)
            state = torch.tensor(mind_state(mind), device=self.device).unsqueeze(0)
            known = notes(mind, text, (), just, extra + follow)
            # It answers from what comes to mind, with what was said lately in view.
            history = [
                t for t in life.conversation[-5:-1] if not t.get("earlier") and mind.tick - t.get("tick", 0) < 2400
            ]
        with self.model_lock:
            heard = self.model.meaning([YOU, *self.tok.encode(text)], state).float().cpu().numpy()
        with life.lock:
            mind.understand(text, heard)  # what it made of what was said comes to mind

        def prompt_for(known: str) -> list[int]:
            prompt = [THINK, *self.tok.encode(known)]
            for turn in history:
                prompt += [YOU if turn["who"] == "you" else HAVEN, *self.tok.encode(turn["text"])]
            return prompt + [YOU, *self.tok.encode(text), HAVEN]

        long = 180 if story_asked(text) == "tale" else 100  # (a story takes longer to tell)
        words, confidence = self._say(prompt_for(known), state, drafts, long)
        topic = None if sum_of(text) else topic_of(text)  # (a sum isn't something to look up)
        if words.startswith(DONT_KNOW[:24]) and self.web is not None and topic and not self.library.title_for(topic):
            if self.look_up(topic, life):  # it didn't know, so it reads about it, and says what it read
                with life.lock:
                    extra = self.recollect(mind, text)
                    known = notes(mind, text, (), just, extra + follow)
                if extra:
                    words, confidence = self._say(prompt_for(known), state, drafts)
        if decision is not None and decision.answer in engage.WILLING:
            with life.lock:
                mind.take_errand(req.do, req.thing, req.action, req.need)  # it chose to, so it sets off
        if question and question.rstrip("?").lower() in words.lower():
            talk.wonder(question)  # it asked what it wondered: the answer may teach it something
        with self.model_lock:
            meaning = self.model.meaning([HAVEN, *self.tok.encode(words)], state).float().cpu().numpy()
        with life.lock:
            mind.think(words, meaning, confidence)  # what it says enters its workspace
        return words, confidence

    def _follow(self, mind, talk, text: str, just: str | None, said: str, learned: str | None):
        """What else comes to mind to follow the conversation: what it wonders about what it was told, what it has of
        its own that goes with it, what it would say about itself if asked the same, why it said what it said, its
        mood, its brain, what it's doing. Returns (notes, the question it wonders, if any)."""
        from ..activities import SIGHTS
        from . import engage
        from .talk import statement

        follow, question = [], None
        mood = engage.mood_note(mind.chemistry)
        if mood:
            follow.append(mood)
        fact = None if learned else (statement(said) if just else None)
        favorites = own_favorites_of(mind)
        if fact:  # something about themselves: it wonders, and has something of its own to say
            curious = mind.character.traits["curious"]
            choices = engage.wonders(fact)
            can_learn = [q for q in choices if engage.learnable(q)]
            if can_learn and random.random() < 0.7:  # (most often what it can learn from the answer to)
                choices = can_learn
            if choices and random.random() < 0.6 + 0.4 * curious:  # (a curious Haven nearly always asks)
                question = random.choice(choices)
                follow.append(engage.wonder_note(question))
            own = engage.own_for(fact, favorites)
            if own:
                follow.append(engage.own_note(own))
                talk.mine(own)
        elif learned and random.random() < 0.3 * mind.character.traits["curious"]:
            choices = [q for q in engage.wonders(learned) if q]
            if choices:
                question = random.choice(choices)
                follow.append(engage.wonder_note(question))
        if engage.ABOUT_YOU.match(text) and talk.own:
            follow.append(engage.about_you_note(talk.own))
        if engage.WHY.match(text) and talk.reason:
            follow.append(engage.why_note(talk.reason))
        if engage.BRAIN_ASKED.search(text) and mind.brain is not None:
            follow.append(engage.brain_note(mind.brain.neurons(), mind.brain.synapses(), self.connections()))
        busy = engage.pastime_words(
            engage.BUSY, mind.activity, SIGHTS.get(mind.watching or "", mind.watching), mind.visiting
        )
        if busy and re.search(r"\bwhat (?:are|r) (?:you|u) (?:doing|up to)\b|\bwhatcha\b", text, re.IGNORECASE):
            follow.append(f"Right now I'm {busy}.")
        return follow, question

    def connections(self) -> int:
        """How many connections its language cortex has."""
        model = getattr(self, "model", None)
        return int(model.parameters_count()) if model is not None else 0

    def speak_up(self, life, note: str, drafts: int = 3) -> tuple[str, float]:
        """Say something of its own accord: what it has to say comes to mind last (see speaking.py), and it puts it
        into words from there, as it answers anything."""
        from .talk import notes
        from .tokenizer import HAVEN, THINK

        torch = self.torch
        with life.lock:
            mind = life.mind
            state = torch.tensor(mind_state(mind), device=self.device).unsqueeze(0)
            known = notes(mind, "", (), None, [note])
        words, confidence = self._say([THINK, *self.tok.encode(known), HAVEN], state, drafts)
        with self.model_lock:
            meaning = self.model.meaning([HAVEN, *self.tok.encode(words)], state).float().cpu().numpy()
        with life.lock:
            mind.think(words, meaning, confidence)  # what it says enters its workspace, like anything it says
        return words, confidence

    def _say(self, prompt: list[int], state, drafts: int, most: int = 100) -> tuple[str, float]:
        """A few drafts of a reply; the likeliest is what it says, and how much they agree is part of how sure it is."""
        from .tokenizer import END, YOU

        found = []
        with self.model_lock:
            for i in range(drafts):  # its most careful draft first, then freer ones
                tokens, logprobs = self.model.generate(
                    prompt[-(self.model.cfg.context - most) :],
                    state,
                    max_new=most,
                    temperature=0.0 if i == 0 else 0.4 if most > 100 else 0.6,
                    stop=(END, YOU),
                    no_repeat=4 if most > 100 else 0,  # (a story mustn't go round in circles)
                )
                words = self.tok.decode(tokens).strip()
                found.append((words, math.exp(float(np.mean(logprobs))) if logprobs else 0.0))
        mind = self.tok.decode(prompt)  # what came to mind, and what was said
        # A draft that's babble loses, then one that goes round in circles, then one that says a number or name it
        # can't back up.
        times = 2 if most > 100 else 1  # (a story can say "said the fox" twice)
        found.sort(key=lambda d: (not garbled(d[0]), not repeats(d[0], mind, times), not unfounded(d[0], mind), d[1]))
        for words, _ in found:  # the ones it didn't pick pass by as thoughts; the likeliest comes last
            self._tell("draft")
            self._tell("words", words)
        words, likely = found[-1]
        return words, 0.5 * likely + 0.5 * agreement([d[0] for d in found])

    def notice(self, mind) -> None:
        from .grounding import moment_of

        self.day.append(moment_of(mind))

    def _first_night(self, score: float | None = None) -> float | None:
        """How the cortex it has now did on other lives the first night it learned in its sleep (or record that)."""
        from .starter import stamp

        path = self.root / "cortex" / "sleep.json"
        began = stamp(self.root / "cortex")
        if score is not None:
            path.write_text(json.dumps({"starter": began, "other lives": score}))
            return score
        with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
            found = json.loads(path.read_text())
            if found["starter"] == began:
                return float(found["other lives"])
        return None

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
        readings, hearing = self.own_readings(), self.hearing
        stories = tuple(hearing.stories()[-5:])
        day = [{**m, "readings": readings, "heard": stories} for m in day]  # it goes over what it read and heard, too
        if self._others is None:
            self._others, self._rehearse = sleep.other_lives(), sleep.rehearsal()
        with self.model_lock:
            current = copy.deepcopy(self.model)
        first = self._first_night()
        heard = [*hearing.tonight, *hearing.before()] if hearing.tonight else []  # (and a little from before)
        learned, report = sleep.night(
            current,
            self.tok,
            day,
            self._others,
            random.Random(),
            rehearse=self._rehearse,
            floor=first,
            heard=heard,
            people=list(hearing.people),
            upcoming=hearing.upcoming() if heard else None,
        )
        hearing.slept()
        if first is None:  # how it did the first night, the floor from now on (until it gets a new cortex)
            self._first_night(report["before"]["other lives"])
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
            what = "its day and the story it heard" if heard else "its day"
            follows = report["after"].get("following"), report["before"].get("following")
            better = f"; how the story goes on: {follows[1]:.2f} → {follows[0]:.2f} bits a letter" if follows[0] else ""
            life._emit(
                "event",
                f"went over {what} in its sleep, and learned from it (answers about its day: {b:.0%} → {a:.0%}{better})"
                if learned is not None
                else f"went over {what} in its sleep, but kept what it knew: practising didn't help this time",
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
    installed = starter.install(root, web)
    moved = {
        "replaced": " (its old one was grown for its old world: it's kept in archive/)",
        "retired": " (its old one was grown for its old world: it's kept in archive/)",
        "updated": " (the newer one this Haven came with; its old one is kept in archive/)",
    }.get(installed, "")
    if not (Path(root) / "cortex" / "cortex.pt").exists():
        return (
            None,
            f"It has no language cortex yet{moved}. Train one with: haven learn. It can still learn words from you.",
        )
    thinker = OwnThinker(root, web=web)
    return thinker, f"Language cortex: {thinker.describe()}.{moved}"
