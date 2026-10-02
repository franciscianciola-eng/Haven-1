"""Haven in the browser (docs/): its own conversation code (haven/cortex), run by Pyodide in the page, with

- a mind for Haven in its nest: a body whose needs are kept near set points as in body.py (in its nest, it finds
  food and sleeps by itself), its brain's chemistry at rest unless something stirs it (being stroked, fed, talked
  with), time and seasons as in world.py, and what it knows of itself and its valley as a newborn Haven does (captured
  from one by packaging/web/build.py, in newborn.py);
- a shelf (shelf.py) of what it reads, filled as it's asked: articles of the Simple English Wikipedia, read online,
  and words from its dictionary, kept in the page's storage, so what it read once it has;
- its own cortex, the same network as the desktop's, run by the page (docs/haven/cortex.js): see `Cortex`.

This file is packed into the page's Python by packaging/web/build.py, as haven/browser.py.
"""

from __future__ import annotations

import collections
import contextlib
import json
import math
import random
import re
import threading
import time
from pathlib import Path

from .cortex import encyclopedia, shelf, talk, think
from .cortex.shelf import FILLER, MEANING, NOT_ASKED, QUESTION, ROMAN, Shelf, Volume, names_of, singulars
from .cortex.tokenizer import Tokenizer
from .newborn import ATTENTION, CHEMICALS, CLIMATE, GOALS, MEMO, SELF, TRAITS, D
from .workspace import AFFECT, CONFIDENCE, DECAY, DRIVES, IGNITION, QUALITY, SOURCES, Q
from .world import DAY, SEASON_DAYS, SEASONS, TURNING, YEAR

SET_ENERGY, SET_TEMPERATURE, SET_FATIGUE = 0.85, 0.5, 0.15  # (as body.py)
STIRS_FOR = 960  # moments a stir of its chemistry is why it feels as it does (engage.LATELY)
FADES = 0.996  # how much of a stir of its chemistry is left a moment later
ASKED = 600.0  # seconds a question it asked stays open for an answer (as life.py)


# --- its body ---------------------------------------------------------------------------------------------------


class Body:
    """Its needs, as body.py keeps them: energy, temperature, integrity (health) and fatigue, near set points."""

    def __init__(self):
        self.energy, self.temperature, self.integrity, self.fatigue = 0.8, 0.5, 1.0, 0.2
        self.asleep = False

    def drives(self) -> list[float]:
        clip = lambda v: min(max(v, 0.0), 1.0)
        return [
            clip((SET_ENERGY - self.energy) / SET_ENERGY),
            clip(abs(self.temperature - SET_TEMPERATURE) / 0.35),
            clip(1 - self.integrity),
            clip((self.fatigue - SET_FATIGUE) / (1 - SET_FATIGUE)),
        ]

    def cold(self) -> bool:
        return self.temperature < SET_TEMPERATURE

    def live(self, warmth: float, fed: float = 0.0) -> None:
        """One moment, in its nest (body.py's metabolism, for a Haven that potters about its nest by day)."""
        burn = 0.0002 if self.asleep else 0.0004
        self.energy += fed - burn
        self.temperature += 0.03 * (warmth - self.temperature)
        if self.energy > 0.25:
            self.integrity += 0.004
        self.fatigue += -0.005 if self.asleep else 0.0006
        self.energy, self.temperature, self.integrity, self.fatigue = (
            min(max(v, 0.0), 1.0) for v in (self.energy, self.temperature, self.integrity, self.fatigue)
        )


# --- its mind ---------------------------------------------------------------------------------------------------


class Content:
    """What comes to mind (workspace.py's Candidate): what it heard, or its own thought."""

    def __init__(self, source: str, label: str, meaning, salience: float, confidence: float = 1.0):
        self.source, self.label, self.salience, self.confidence = source, label, salience, confidence
        meaning = _floats(meaning)
        self.quality = meaning[QUALITY] if len(meaning) == D else meaning[:Q]  # (as mind.py's _quality)
        self.kind, self.extra = -1, {}


class Workspace:
    """Its global workspace (workspace.py): what it hears, and its own thoughts, come to the fore one at a time, and
    fade; what it holds is broadcast laid out as there, with its body's needs and its feelings always in it."""

    def __init__(self):
        self.content: Content | None = None
        self.offered: Content | None = None  # (what came to mind, to ignite the next moment)
        self.strength = 0.0
        self.vector = [0.0] * D
        self.dwell = 0
        self.history: list[Content] = []

    def offer(self, content: Content) -> None:
        self.offered = content

    def compete(self) -> None:
        """A moment of it (as workspace.py's compete, with one thing at a time coming to mind, scored by salience)."""
        self.strength *= DECAY
        best, self.offered = self.offered, None
        if best is not None and best.salience > IGNITION and best.salience > self.strength:
            self.content, self.strength, self.dwell = best, best.salience, 0
            self.history = [*self.history[-19:], best]
        else:
            self.dwell += 1
            if self.strength < 0.1:
                self.content = None  # (nothing is in mind)

    def broadcast(self, drives: list[float], affect: tuple[float, float]) -> None:
        v = [0.0] * D
        if self.content is not None:  # (as workspace.py's Candidate.vector, and broadcast)
            v[SOURCES.index(self.content.source)] = 1.0
            v[QUALITY.start : QUALITY.start + len(self.content.quality)] = self.content.quality
            v[CONFIDENCE] = self.content.confidence
            v = [x * min(1.0, 0.5 + self.strength) for x in v]
        v[DRIVES], v[AFFECT] = list(drives), list(affect)
        self.vector = v


class Me:
    def __init__(self, name: str, born: int):
        self.name, self.born = name, born
        self.milestones = [(born, "came into the world, in its nest")]
        self.evidence = dict(SELF["evidence"])
        self.conclusions: list = []
        self.alive = SELF["alive"]


class Character:
    def __init__(self, traits: dict[str, float]):
        self.traits = traits
        self.best = self.worst = None

    def favorite_season(self) -> None:
        return None  # (it hasn't lived a whole year yet, in its nest)

    def favorite_area(self) -> None:
        return None


class World:
    """Its valley's clock, as world.py keeps it: days of 1,200 moments, seasons of six days, years of four seasons."""

    def __init__(self, tick: int = 360):
        self.tick = tick  # (a Haven is born in the morning)
        self.voice = None

    @property
    def day(self) -> int:
        return self.tick // DAY

    @property
    def year(self) -> int:
        return self.day // YEAR

    @property
    def season(self) -> str:
        return SEASONS[(self.day % YEAR) // SEASON_DAYS]

    @property
    def season_day(self) -> int:
        return self.day % SEASON_DAYS

    @property
    def summer(self) -> float:
        days = (self.tick % (DAY * YEAR)) / DAY
        i = int(days // SEASON_DAYS)
        now, after = CLIMATE[SEASONS[i]], CLIMATE[SEASONS[(i + 1) % len(SEASONS)]]
        turned = max(0.0, (days - i * SEASON_DAYS - (SEASON_DAYS - TURNING)) / TURNING)
        return now + (after - now) * turned

    @property
    def light(self) -> float:
        phase = 2 * math.pi * (self.tick % DAY) / DAY
        return min(max(0.5 + 0.08 * self.summer - 0.55 * math.cos(phase), 0.06), 1.0)


class _Nothing:
    """A part of the desktop's mind a Haven in its nest doesn't have yet (no words learned, no kinds of things met)."""

    def vocabulary(self) -> list:
        return []

    def name_for(self, *_args) -> None:
        return None

    def alive(self) -> list:
        return []

    def describe(self, *_args, **_kwargs) -> str:
        return ""


class Goals:
    def __init__(self):
        self.current = GOALS[0]

    def index(self) -> int:
        return GOALS.index(self.current)


class Mind:
    """What the conversation code reads from a mind (talk.py, think.py, engage.py), for Haven in its nest."""

    def __init__(self, name: str = "Haven", seed: int | None = None):
        self.rng = random.Random(seed)
        self.world = World()
        self.body = Body()
        self.me = Me(name, self.world.tick)
        # (a temperament of its own, as personality.py's Character draws one)
        self.character = Character({t: min(max(0.5 + self.rng.gauss(0, 0.1), 0.25), 0.75) for t in TRAITS})
        self.workspace = Workspace()
        self.lexicon = self.knowledge = _Nothing()
        self.vision = _Nothing()
        self.vision.kinds = _Nothing()
        self.goals = Goals()
        self.chemistry = dict.fromkeys(CHEMICALS, 1.0)
        self.stirred: dict[str, tuple[int, str]] = {}
        self.person: str | None = None
        self.told: list[tuple[int, str]] = []
        self.lessons: list[tuple[int, str]] = []
        self.said: list[tuple[int, str]] = []
        self.today: dict[str, int] = {}
        self.counts: dict[str, int] = {}
        self.things: dict[str, dict] = {}
        self.notes: list[tuple[int, str]] = []
        self.valence = self.arousal = self.mood = 0.0
        self.activity = self.watching = self.visiting = self.brain = None
        self._drives = self.body.drives()

    # (what the conversation code asks of a mind)

    @property
    def tick(self) -> int:
        return self.world.tick

    @property
    def age(self) -> int:
        return self.world.tick - self.me.born

    @property
    def years(self) -> int:
        return self.age // (DAY * YEAR)

    def time_of_day(self) -> str:
        phase = (self.world.tick % DAY) / DAY
        for limit, name in (
            (0.2, "night"),
            (0.3, "dawn"),
            (0.45, "morning"),
            (0.55, "midday"),
            (0.7, "afternoon"),
            (0.8, "dusk"),
        ):
            if phase < limit:
                return name
        return "night"

    def kind_name(self, *_args) -> None:
        return None

    def kind_color(self, *_args) -> None:
        return None

    def _note(self, tick: int, text: str) -> None:
        self.notes = [*self.notes[-199:], (tick, text)]

    def understand(self, text: str, heard) -> None:
        """What it made of what was said comes to mind (as Mind.understand)."""
        self.workspace.offer(Content("hearing", f'understanding "{text[:80]}"', heard, 0.7))
        self.wake()
        self._stir("oxytocin", 0.25, "you're talking with me")

    def think(self, words: str, meaning, confidence: float) -> None:
        """Its own words come to mind (as Mind.think)."""
        self.workspace.offer(
            Content("thought", f'thinking "{words[:80]}"', meaning, 0.55 + 0.3 * confidence, confidence)
        )

    def hear(self, text: str) -> list[str]:
        return []

    def take_errand(self, *_args) -> None:
        pass  # (in its nest, there's nowhere to go)

    # (what happens to it)

    def wake(self) -> None:
        if self.body.asleep:
            self.body.asleep = False
            self._note(self.tick, "woke up")

    def stroke(self) -> None:
        self._stir("oxytocin", 1.6, "you stroked me")
        self._did("was stroked")

    def feed(self) -> None:
        self.body.energy = min(1.0, self.body.energy + 0.2)
        self._stir("dopamine", 1.3, "you gave me food")
        self._did("ate a berry")

    def _did(self, what: str) -> None:
        self.today[what] = self.today.get(what, 0) + 1
        self.counts[what] = self.counts.get(what, 0) + 1
        self._note(self.tick, what)

    def _stir(self, chemical: str, amount: float, why: str) -> None:
        self.chemistry[chemical] = min(self.chemistry[chemical] + amount, 3.0)
        self.stirred[chemical + "+"] = (self.tick, why)

    def step(self) -> list[str]:
        """One moment of its life in its nest. Returns what happened, as events."""
        w, b = self.world, self.body
        events = []
        day = w.day
        w.tick += 1
        if w.day != day:
            self.today = {}
        night = w.light < 0.2
        warmth = SET_TEMPERATURE + 0.04 * w.summer - (0.04 if night else 0.0)
        b.live(warmth)
        if not b.asleep and b.energy < 0.6:  # (it finds berries by its nest)
            b.energy = min(1.0, b.energy + 0.3)
            self._stir("dopamine", 0.6, "I ate a berry")
            self._did("ate a berry")
            events.append("ate a berry")
        if not b.asleep and night and b.fatigue > 0.35:
            b.asleep = True
            events.append("fell asleep in its nest")
        elif b.asleep and (b.fatigue <= 0.05 or (not night and b.fatigue < 0.2)):
            b.asleep = False
            events.append("woke up")
        for chemical in self.chemistry:  # (its chemistry settles back, slowly)
            self.chemistry[chemical] = 1.0 + (self.chemistry[chemical] - 1.0) * FADES
        drives = b.drives()
        felt = sum(a - c for a, c in zip(self._drives, drives, strict=True))  # (needs met feel good)
        self._drives = drives
        self.valence = 0.95 * self.valence + 0.05 * (20 * felt + 0.5 * (self.chemistry["oxytocin"] - 1) - max(drives))
        self.arousal = 0.95 * self.arousal + 0.05 * (0.2 if b.asleep else 0.5 + 0.3 * (self.chemistry["dopamine"] - 1))
        self.mood = 0.999 * self.mood + 0.001 * self.valence
        self.workspace.compete()
        self.workspace.broadcast(drives, (self.valence, self.arousal))
        return events

    # (what its cortex reads of it: grounding.py's mind_state, for this mind)

    def state(self) -> list[float]:
        """What Haven is experiencing, as its cortex's workspace tokens (4 slots of D), flattened."""
        w, b = self.world, self.body
        phase = 2 * math.pi * (w.tick % 1200) / 1200
        goal = [0.0] * len(GOALS)
        goal[self.goals.index()] = 1.0
        body = [
            *b.drives(),
            self.valence,
            self.arousal,
            self.mood * 5,
            b.energy,
            b.temperature,
            b.integrity,
            b.fatigue,
            float(b.asleep),
            float(b.cold()),
            w.light,
            math.sin(phase),
            math.cos(phase),
            *goal,
        ]
        attention = list(ATTENTION)
        self_part = [*self.me.evidence.values(), self.me.alive, min(self.age / 12000, 1.0), 0.0, 0.0, 1 / 30]
        slots = []
        for part in (self.workspace.vector, body, attention, self_part):
            slots += [float(v) for v in part][:D] + [0.0] * (D - min(len(part), D))
        return slots

    # (keeping it)

    def save(self) -> dict:
        return {
            "name": self.me.name,
            "born": self.me.born,
            "tick": self.world.tick,
            "body": vars(self.body),
            "chemistry": self.chemistry,
            "stirred": self.stirred,
            "person": self.person,
            "told": self.told,
            "lessons": self.lessons,
            "said": self.said[-50:],
            "today": self.today,
            "counts": self.counts,
            "valence": self.valence,
            "arousal": self.arousal,
            "mood": self.mood,
            "traits": self.character.traits,
            "notes": self.notes[-200:],
        }

    @classmethod
    def load(cls, saved: dict) -> Mind:
        mind = cls(saved.get("name", "Haven"))
        mind.me.born = int(saved.get("born", mind.me.born))
        mind.me.milestones = [(mind.me.born, "came into the world, in its nest")]
        mind.world.tick = int(saved.get("tick", mind.world.tick))
        for key, value in saved.get("body", {}).items():
            setattr(mind.body, key, value)
        mind.chemistry.update(saved.get("chemistry", {}))
        mind.stirred = {k: tuple(v) for k, v in saved.get("stirred", {}).items()}
        mind.person = saved.get("person")
        mind.told = [tuple(t) for t in saved.get("told", [])]
        mind.lessons = [tuple(t) for t in saved.get("lessons", [])]
        mind.said = [tuple(t) for t in saved.get("said", [])]
        mind.today = dict(saved.get("today", {}))
        mind.counts = dict(saved.get("counts", {}))
        mind.valence, mind.arousal, mind.mood = (float(saved.get(k, 0.0)) for k in ("valence", "arousal", "mood"))
        mind.character.traits = dict(saved.get("traits", mind.character.traits))
        mind.notes = [tuple(t) for t in saved.get("notes", [])]
        mind._drives = mind.body.drives()
        return mind


def _floats(vector) -> list[float]:
    values = vector.tolist() if hasattr(vector, "tolist") else vector
    return [float(v) for v in values]


def memo(mind: Mind) -> dict:
    """What it knows that could come to mind (talk.memo): a newborn Haven's, captured from one, but its own name, age,
    season and day (it lives in its nest: what it knows of its valley stays as a newborn's)."""
    found = {key: (dict(value) if isinstance(value, dict) else value) for key, value in MEMO.items()}
    found["me"] = f"I'm {mind.me.name}, {talk.age_words(mind)}."
    found["season"] = talk.season_words(mind)
    found["today"] = talk.today(mind)
    found["traits"] = {t: round(v, 3) for t, v in mind.character.traits.items()}
    return found


talk.memo = memo  # (think.py and talk.py ask talk.memo, at the time)
talk.request = lambda text: None  # (asked to do something in its valley: in its nest, it hears that as anything else)


# --- its life ---------------------------------------------------------------------------------------------------


class Life:
    """Haven living in the page: its mind, what's been said, and what it said (as life.py keeps them)."""

    def __init__(self, mind: Mind):
        self.mind = mind
        self.lock = threading.RLock()
        self.conversation: list[dict] = []
        self.asked: tuple[str, float] | None = None
        self.listeners: list = []
        self.thinker = None

    def question(self) -> str | None:
        asked, self.asked = self.asked, None
        return asked[0] if asked and time.monotonic() - asked[1] < ASKED else None

    def heard(self, text: str) -> None:
        with self.lock:
            self.conversation = [*self.conversation[-99:], {"tick": self.mind.tick, "who": "you", "text": text}]

    def reply(self, text: str, source: str = "", spoken: bool = False) -> None:
        with self.lock:
            self.mind.world.voice = (self.mind.tick, text)
            self.mind.said = [*self.mind.said[-49:], (self.mind.tick, text)]
            self.conversation = [*self.conversation[-99:], {"tick": self.mind.tick, "who": "haven", "text": text}]
        self._emit("said", text)

    def step(self, moments: int = 1) -> list[str]:
        events = []
        with self.lock:
            for _ in range(moments):
                events += self.mind.step()
        for event in events:
            self._emit("event", event)
        return events

    def _emit(self, kind: str, text: str) -> None:
        for listener in self.listeners:
            with contextlib.suppress(Exception):
                listener(kind, text)

    def status(self) -> dict:
        """How it is, for the page's line at the top."""
        mind, b = self.mind, self.mind.body
        from .cortex import engage

        drives = b.drives()
        need = max(range(4), key=lambda i: drives[i])
        return {
            "name": mind.me.name,
            "age": talk.age_words(mind),
            "season": mind.world.season,
            "time": mind.time_of_day(),
            "asleep": b.asleep,
            "need": None if drives[need] < 0.3 else ("hungry", "cold" if b.cold() else "hot", "hurt", "tired")[need],
            "cold": b.cold(),
            "needs": dict(zip(("hunger", "warmth", "hurt", "tiredness"), (round(d, 3) for d in drives), strict=True)),
            "mood": engage.mood_words(mind.chemistry),
            "feeling": round(mind.valence, 3),
            "person": mind.person,
            "light": round(mind.world.light, 3),
            "cortex": None if self.thinker is None else self.thinker.describe(),
        }

    def save(self) -> dict:
        return {"mind": self.mind.save(), "conversation": self.conversation[-40:]}

    @classmethod
    def load(cls, saved: dict | None, name: str = "Haven") -> Life:
        if not saved:
            return cls(Mind(name))
        life = cls(Mind.load(saved.get("mind", {})))
        life.conversation = [dict(t, earlier=True) for t in saved.get("conversation", [])]  # (said in an earlier visit)
        return life


# --- its cortex, run by the page ----------------------------------------------------------------------------------


class Config:
    def __init__(self, config: dict):
        for key, value in config.items():
            setattr(self, key, value)


class Vector(list):
    """A meaning, as the desktop's cortex gives it (a tensor: .float().cpu().numpy())."""

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self

    def tolist(self):
        return list(self)


class State:
    def __init__(self, values):
        self.values = [float(v) for v in values]

    def unsqueeze(self, _dim):
        return self


class _Torch:
    """What think.py asks of torch: turning its mind's state into what the cortex reads."""

    @staticmethod
    def tensor(values, device=None):
        flat = []
        for row in values:
            flat += list(row) if isinstance(row, list | tuple) else [row]
        return State(flat)


class Cortex:
    """Its language cortex, run by the page (docs/haven/cortex.js): generate() and meaning() as model.py's, awaited.
    `page` is the page's cortex (or, in tests, anything with the same two async methods and a config)."""

    def __init__(self, page, config: dict):
        self.page = page
        self.cfg = Config(config)
        self._count = None

    def parameters_count(self) -> int:
        if self._count is None:
            self._count = int(self.page.parametersCount())
        return self._count

    async def generate(
        self, prompt, state=None, max_new=60, temperature=0.8, top_k=40, stop=(), generator=None, no_repeat=0
    ):
        options = {
            "maxNew": int(max_new),
            "temperature": float(temperature),
            "topK": int(top_k),
            "stop": list(stop),
            "noRepeat": int(no_repeat),
        }
        tokens, logprobs = await self.page.generate(list(prompt), None if state is None else state.values, options)
        return [int(t) for t in tokens], [float(p) for p in logprobs]

    async def meaning(self, ids, state=None):
        return Vector(float(v) for v in await self.page.meaning(list(ids), None if state is None else state.values))


class Quiet:
    """What it hears at home (hearing.py): in the page, no bedtime stories yet."""

    tonight = ()

    def heard_said(self, text: str) -> None:
        pass

    def words(self) -> int:
        return 0

    def stories(self) -> list:
        return []

    def bedtime(self) -> None:
        return None


class PageThinker(think.OwnThinker):
    """Its own cortex thinking, as the desktop's (think.py), run by the page."""

    def __init__(self, root: Path, cortex: Cortex, merges: list, progress: dict, shelf=None):
        think.Thinker.__init__(self, root, None)  # (not OwnThinker's: that one loads PyTorch)
        self.torch = _Torch()
        self.device = "page"
        self.progress = progress
        self.tok = Tokenizer([tuple(m) for m in merges])
        self.model = cortex
        # (the page's cortex does one thing at a time itself; a lock held over an awaited reply would stop everything)
        self.model_lock = contextlib.nullcontext()
        self.day = collections.deque(maxlen=240)
        self._others = self._rehearse = None
        self._shelf = shelf
        self._hearing = Quiet()

    def describe(self) -> str:
        return self.progress.get("described") or self.name  # (as the desktop's says it: docs/cortex/cortex.json)

    def notice(self, mind) -> None:
        pass  # (it doesn't learn in its sleep in the page)

    def sleep_on_it(self, life) -> None:
        return None


# --- its shelf, filled as it's asked ------------------------------------------------------------------------------

API = "https://simple.wikipedia.org/w/api.php"


class WikiShelf(Shelf):
    """Its shelf (shelf.py), filled as it's asked: before it looks for what answers a question, it reads, online, the
    articles of the Simple English Wikipedia the question names (or, naming none, the ones it finds that say most of
    it), and looks up its words in its dictionary. What it read stays on its shelf.

    `fetch(url)` gets a page's text (in the page, synchronously: Pyodide runs in a worker); `dictionary(word)` gives
    the dictionary's entries for a word, as (title, sentences, names); `weights`: how many sentences of the whole Simple
    English Wikipedia and its dictionary say each word (by its stem), so a word weighs as much as on the desktop's
    shelf."""

    def __init__(self, folder: Path, fetch, dictionary=None, weights: dict | None = None, api: str = API):
        super().__init__(folder)
        self.fetch, self.dictionary, self.api = fetch, dictionary, api
        self.counted = dict((weights or {}).get("counts", {}))
        self.total = int((weights or {}).get("total", 0))
        self.looked: dict[str, str | None] = {}  # names it has looked up online: the article, or None
        self.searched: set[str] = set()
        self.problem: str | None = None
        self.tell = lambda kind, text: None  # (someone following along: "reading" an article online, then "read")
        self._stems = None

    def find(self, question: str):
        try:
            self.gather(question)
        except Exception as error:  # noqa: BLE001  (offline, or Wikipedia didn't answer: it answers from what it has)
            self.problem = f"{type(error).__name__}: {error}"
            self.tell("unread", self.problem)
        return super().find(question)

    # (reading what it's asked about)

    def gather(self, question: str) -> None:
        text = " ".join(question.strip().split())
        if not QUESTION.search(text) or NOT_ASKED.search(text):
            return
        said = [t.split("'")[0].split("’")[0] for t in re.findall(r"[^\W_]+(?:['’][^\W_]+)?", text)]
        tokens = [t.lower() for t in said]
        content = [t for t in tokens if t not in FILLER]
        if not content:
            return
        meaning = bool(MEANING.search(text))
        if meaning:
            content = [t for t in content if t not in ("mean", "means", "meaning", "define", "definition", "word")]
        if self.dictionary is not None and (meaning or len(content) == 1):
            for word in content:
                self._words(word)
        if meaning:
            return
        named = self._names(tokens)
        found = self._titles([n for n in named if n not in self.looked])
        for name, title in found.items():
            self.looked[name] = title
        titles = list(dict.fromkeys(self.looked[n] for n in named if self.looked.get(n)))
        for title in titles[:4]:
            self._article(title, [n for n in named if self.looked.get(n) == title], named)
        if not titles:  # (it names nothing: what says the most of it, anywhere)
            key = " ".join(content)
            if key not in self.searched:
                self.searched.add(key)
                for title in self._search(content)[:2]:
                    self._article(title, [], named)

    def _names(self, tokens: list[str]) -> list[str]:
        """What a question might be naming, the longest first (as shelf.py's _named looks them up)."""
        found = []
        for n in range(min(6, len(tokens)), 0, -1):
            for i in range(len(tokens) - n + 1):
                gram = tokens[i : i + n]
                if gram[0] in FILLER or gram[-1] in FILLER:
                    continue
                names = [" ".join(gram), *(" ".join([*gram[:-1], one]) for one in sorted(singulars(gram[-1])))]
                if gram[-1] in ROMAN and n > 1:
                    names.append(" ".join([*gram[:-1], ROMAN[gram[-1]]]))
                found += [name for name in names if name not in found]
        return found

    def _api(self, **params) -> dict:
        params = {"action": "query", "format": "json", "formatversion": "2", "origin": "*", **params}
        url = self.api + "?" + "&".join(f"{k}={_quote(str(v))}" for k, v in params.items())
        return json.loads(self.fetch(url)).get("query", {})

    def _titles(self, names: list[str]) -> dict[str, str | None]:
        """The articles names are of, if there are any (as Wikipedia has them: titles, and redirects to them)."""
        found: dict[str, str | None] = {}
        asked: dict[str, list[str]] = {}  # each title tried, and the names it's for
        for name in names[:24]:
            words = [w.upper() if w in ROMAN.values() else w[:1].upper() + w[1:] for w in name.split()]
            for title in dict.fromkeys((name[:1].upper() + name[1:], " ".join(words))):
                asked.setdefault(title, []).append(name)
            found[name] = None
        tried = list(asked)
        for start in range(0, len(tried), 50):
            query = self._api(
                titles="|".join(tried[start : start + 50]), redirects="1", prop="pageprops", ppprop="disambiguation"
            )
            moved = {n["from"]: n["to"] for n in query.get("normalized", [])}
            moved.update({r["from"]: r["to"] for r in query.get("redirects", [])})
            pages = {p["title"]: p for p in query.get("pages", [])}
            for title in tried[start : start + 50]:
                final = title
                for _ in range(3):
                    final = moved.get(final, final)
                page = pages.get(final)
                if (
                    page is None
                    or page.get("missing")
                    or page.get("invalid")
                    or "disambiguation" in page.get("pageprops", {})
                ):
                    continue
                for name in asked[title]:
                    found[name] = found[name] or page["title"]
        return found

    def _search(self, words: list[str]) -> list[str]:
        query = self._api(list="search", srsearch=" ".join(words), srlimit="3", srnamespace="0", srprop="")
        return [hit["title"] for hit in query.get("search", [])]

    def _article(self, title: str, asked: list[str], named: list[str]) -> None:
        """An article onto its shelf (as the desktop reads an encyclopedia: shelf.py's _articles), and, if none of the
        names it would have there (names_of) is among what the question might be naming, the names it was asked for
        by (Wikipedia's redirects: "WW2" for World War II), so its shelf finds it as the desktop's would."""
        volume = self.volume("simple", make=True)
        with self.lock:
            row = volume.db.execute(
                "SELECT id FROM articles WHERE title = ? AND source = 'simple'", (title,)
            ).fetchone()
            if row is not None:
                have = {n for (n,) in volume.db.execute("SELECT name FROM names WHERE article = ?", row)}
                with volume.db:
                    volume.db.executemany(
                        "INSERT INTO names (name, article) VALUES (?, ?)", [(n, row[0]) for n in asked if n not in have]
                    )
                return
        self.tell("reading", title)
        pages = self._api(prop="extracts", explaintext="1", exsectionformat="plain", redirects="1", titles=title).get(
            "pages", []
        )
        text = pages[0].get("extract", "") if pages else ""
        text = re.sub(r"\n+", "\n\n", text)  # (each line of what Wikipedia gives is a paragraph or a heading)
        paragraphs = encyclopedia.prose(text, encyclopedia.SIMPLE.keep)
        found = [said for p in paragraphs for s in encyclopedia.sentences(p) if (said := encyclopedia.speakable(s))]
        if not found:
            return
        own = names_of(title, found[0])
        names = sorted(own if own & set(named) else own | set(asked))
        with self.lock, volume.db:
            Volume.put(volume.db, title, found, "simple", 0, len(text), Volume.next_row(volume.db), names)
        self._changed()
        self.tell("read", title)

    def _words(self, word: str) -> None:
        """A word's entries from its dictionary onto its shelf (the word, and what it may be the plural of)."""
        volume = self.volume("dictionary", make=True)
        for lemma in dict.fromkeys((word, *sorted(singulars(word)))):
            with self.lock:
                if volume.db.execute("SELECT 1 FROM names WHERE name = ? LIMIT 1", (lemma,)).fetchone():
                    continue
            entries = self.dictionary(lemma) or []
            with self.lock, volume.db:
                for title, sentences, names in entries:
                    Volume.put(volume.db, title, sentences, "dictionary", 0, 0, Volume.next_row(volume.db), names)
            if entries:
                self._changed()

    # (how telling a word is: as on the desktop's shelf, of the whole encyclopedia and dictionary)

    def _weight(self, word: str) -> float:
        if not self.total:
            return super()._weight(word)
        return math.log(1 + self.total / (1 + self.counted.get(self._stem(word), 0)))

    def _stem(self, word: str) -> str:
        """A word as its shelf's full-text search knows it (Porter's stem, as SQLite's FTS5 makes it)."""
        if self._stems is None:
            self._stems = shelf.sqlite3.connect(":memory:")
            self._stems.execute(
                "CREATE VIRTUAL TABLE t USING fts5 (x, tokenize = 'porter unicode61 remove_diacritics 2')"
            )
            self._stems.execute("CREATE VIRTUAL TABLE v USING fts5vocab (t, 'row')")
        db = self._stems
        db.execute("DELETE FROM t")
        db.execute("INSERT INTO t (x) VALUES (?)", (word,))
        found = db.execute("SELECT term FROM v").fetchone()
        return found[0] if found else word


class Dictionary:
    """Its dictionary (WordNet), as the page keeps it: a file of words for each first two letters (docs/dictionary),
    each word's entries as the desktop's dictionary has them on its shelf: (title, sentences, names)."""

    def __init__(self, fetch, base: str):
        self.fetch, self.base = fetch, base.rstrip("/")
        self.files: dict[str, dict] = {}

    def __call__(self, word: str) -> list:
        key = re.sub(r"[^a-z]", "_", word.lower()[:2].ljust(2, "_"))
        if key not in self.files:
            try:
                self.files[key] = json.loads(self.fetch(f"{self.base}/{key}.json"))
            except Exception:  # noqa: BLE001  (no such file: no words starting so)
                self.files[key] = {}
        return [tuple(entry) for entry in self.files[key].get(word.lower(), [])]


def _quote(text: str) -> str:
    from urllib.parse import quote

    return quote(text, safe="")


# --- the page's way in ------------------------------------------------------------------------------------------


class Haven:
    """What the page talks to: Haven's life, its thinking, and its shelf (see docs/haven/worker.js)."""

    def __init__(self, root: str, cortex: Cortex, merges: list, progress: dict, fetch, dictionary=None, weights=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        saved = None
        with contextlib.suppress(OSError, ValueError):
            saved = json.loads((self.root / "life.json").read_text())
        self.life = Life.load(saved)
        self.shelf = WikiShelf(self.root / "library", fetch, dictionary, weights)
        self.thinker = PageThinker(self.root, cortex, merges, progress, self.shelf)
        self.life.thinker = self.thinker
        self.heard: list[tuple[str, str]] = []  # (what passed through its mind as it answered, for the page)
        self.on_thought = None  # (the page, following along as it answers)
        self.thinker.listener = self._thought
        self.shelf.tell = self._thought

    def _thought(self, kind: str, text: str) -> None:
        self.heard.append((kind, text))
        if self.on_thought is not None:
            self.on_thought(kind, text)

    async def say(self, text: str) -> dict:
        """Someone says something to it: its answer, how sure it was, and what passed through its mind."""
        self.heard = []
        self.life.heard(text)
        answer, confidence = await self.thinker.deliberate(self.life, text)
        self.life.mind.hear(text)
        if answer:
            self.life.reply(answer)
        self.save()
        return {
            "answer": answer,
            "confidence": confidence,
            "thoughts": [list(t) for t in self.heard],
            "status": self.life.status(),
        }

    def touch(self, what: str) -> dict:
        mind = self.life.mind
        {"stroke": mind.stroke, "feed": mind.feed}.get(what, lambda: None)()
        self.save()
        return self.life.status()

    def step(self, moments: int = 1) -> list[str]:
        return self.life.step(moments)

    def status(self) -> dict:
        return self.life.status()

    def conversation(self) -> list[dict]:
        return [t for t in self.life.conversation if t.get("who") in ("you", "haven")]

    def save(self) -> None:
        tmp = self.root / "life.json.tmp"
        tmp.write_text(json.dumps(self.life.save()))
        tmp.replace(self.root / "life.json")
