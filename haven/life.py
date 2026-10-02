"""Running Haven's life in real time, in the background, saving as it goes."""

from __future__ import annotations

import contextlib
import pickle
import re
import threading
import time
from collections.abc import Callable

from .mind import VERSION, Mind, moved
from .report import snapshot
from .speaking import Initiative
from .store import Store
from .world import DAY

STORY_EVERY = 1200.0  # seconds, at least, between bedtime stories (see cortex/hearing.py)

AUTOSAVE = 600  # ticks between saves
SYNAPSES_EVERY = 900.0  # seconds, at least, between saves of every synapse of its brain (hundreds of MB)
QUIET = 180.0  # seconds nobody has said anything before Haven reads about something out of curiosity
CURIOUS_EVERY = 900.0  # seconds, at least, between the things it reads out of curiosity
SLEEP_EVERY = 1800.0  # seconds, at least, between nights it learns from its day (its days are short)
FIRST_NIGHT = 600.0  # seconds after it wakes up in the app before the first time it can
NOTICE = 25  # ticks between the moments of its day it notes down, to learn from
SPEAK = 10  # ticks between looks at whether there's something to say of its own accord
PASSING_SAVE = 6000  # ticks between saves while time goes by quickly
ASKED = 600.0  # seconds a question it asked stays open for an answer
NEWS = re.compile(  # what it does that's worth telling the person talking with it
    r"^(?:set off to|did what it was asked|stopped trying to|gave up trying to|couldn't .*: it didn't know where|"
    r"read about|went over its day|learned the word|was taught that|heard a bedtime story|"
    r"found food everywhere|felt the sun come out|saw butterflies come|was healed|was caught in|felt a heat wave|"
    r"saw a fire spread|felt the ground shake|saw the food wither|saw thorns grow|was hurt by you)"
)
FIRST_STORY = 120.0  # seconds after it wakes up in the app before it can hear its first bedtime story


def open_mind(store: Store, seed: int | None = None, name: str = "Haven") -> Mind:
    """Wake the Haven that lives in this store, or bring a new one into the world."""
    if store.exists():
        state = store.load()
        if int(state.get("version", 1)) < VERSION:  # it lived in an older world: it moves to this one
            mind = moved(state)
            store.archive()  # its old life stays on disk, in archive/
            store.save(mind.to_state())
            return mind
        mind = Mind(int(state["seed"]), state["self"]["name"])
        mind.load_state(state)
        return mind
    return Mind(int(time.time()) % 100000 if seed is None else seed, name)


class Life:
    def __init__(self, mind: Mind, store: Store | None, speed: float = 8.0):
        self.mind = mind
        self.store = store
        self.speed = speed  # moments per second
        self.lock = threading.RLock()
        self.paused = False
        self.listeners: list[Callable[[str, str], None]] = []  # (kind, text) for things worth showing
        self.conversation: list[dict] = []  # what the person said, and anything Haven said back
        self.thinker = None  # a language cortex, once it has one
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._since_save = 0
        self._seen_log = mind.log[-1] if mind.log else None
        self._seen_said = mind.said[-1] if mind.said else None
        self._last_words = self._last_wonder = time.monotonic() - CURIOUS_EVERY  # when it last talked, and read
        self._last_night = time.monotonic() - SLEEP_EVERY + FIRST_NIGHT  # when it last learned in its sleep
        self._learning = threading.Lock()
        self._last_story = time.monotonic() - STORY_EVERY + FIRST_STORY  # when it last heard a bedtime story
        self._was_asleep = mind.body.asleep
        self.news: list[tuple[int, str]] = []  # (number, what): what it did lately that's worth telling
        self._news = 0
        self.initiative = Initiative()  # when it speaks up of its own accord, and what about
        self.spoken: list[tuple[int, str]] = []  # (number, what): what it said of its own accord lately
        self._spoken = 0
        self.asked: tuple[str, float] | None = None  # what it asked the person, and when, until they answer
        self.passing: dict | None = None  # while time goes by quickly: how far along it is
        self.talk = None  # what the conversation is about lately (cortex/think.Thread), once there is one
        self.brain_state = "none"  # its brain of neurons: "waking" (growing or loading), "awake", or "none"
        self._synapses_saved = time.monotonic()  # when every synapse of its brain was last saved

    # --- running --------------------------------------------------------------------

    def wake_brain(self) -> None:
        """Load its brain (or grow a newborn one), in the background: it lives on meanwhile, and the brain joins in
        when it's ready."""
        if self.store is None or self.mind.brain is not None:
            return
        self.brain_state = "waking"

        def waking() -> None:
            try:
                from .brain import wake

                brain = wake(self.store.root, self.mind.seed)
            except (OSError, RuntimeError, ValueError, KeyError, EOFError, pickle.UnpicklingError) as error:
                # (a brain that won't load mustn't stop its life)
                self.brain_state = "none"
                self._emit("event", f"its brain couldn't wake up ({type(error).__name__}); it lives on without it")
                return
            with self.lock:
                self.mind.brain = brain
            self.brain_state = "awake" if brain is not None else "none"

        threading.Thread(target=waking, name="haven-brain-waking", daemon=True).start()

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="haven-life", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.save(synapses=True)

    def _run(self) -> None:
        next_tick = time.monotonic()
        while not self._stop.is_set():
            if self.paused or self.passing is not None:
                time.sleep(0.05)
                next_tick = time.monotonic()
                continue
            self.advance(1)
            next_tick += 1.0 / max(self.speed, 0.1)
            delay = next_tick - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_tick = time.monotonic()  # running behind: don't try to catch up in a burst

    def advance(self, ticks: int) -> None:
        for _ in range(ticks):
            with self.lock:
                self.mind.brain_budget = 550.0 / max(self.speed, 0.1)  # (its brain may take about half of a moment)
                self.mind.company = self.initiative.present()
                self.mind.step()
                self._since_save += 1
                self._announce()
                asleep = self.mind.body.asleep
                tick = self.mind.tick
                if self.thinker is not None and not asleep and tick % NOTICE == 0:
                    self.thinker.notice(self.mind)
            if self._since_save >= AUTOSAVE:
                self.save()
            if asleep and not self._was_asleep:
                self._bedtime()
            self._was_asleep = asleep
            if asleep and tick % 200 == 0:
                self._sleep_on_it()
            elif not asleep and tick % 50 == 0:
                self._wonder()
            if tick % SPEAK == 0:
                self._speak_up()

    # --- time going by quickly ---------------------------------------------------------------

    def pass_time(self, days: int) -> bool:
        """Let `days` of its life go by as fast as it can live them (in the background). Its language cortex rests
        meanwhile: no reading, no learning in its sleep. Returns False if time is already going by."""
        with self.lock:
            if self.passing is not None:
                return False
            mind = self.mind
            self.passing = {
                "days": int(days),
                "done": 0,
                "total": int(days) * DAY,
                "from": mind.tick,
                "before": dict(mind.character.traits),
            }
        threading.Thread(target=self._pass, name="haven-time-passing", daemon=True).start()
        return True

    def _pass(self) -> None:
        passing = self.passing
        self.mind.brain_resting = True  # (time goes by too fast for its brain of neurons to live through it)
        self.mind.chemistry = {k: 1.0 for k in self.mind.chemistry}  # (so its mood is its usual one meanwhile)
        hurt_before = self.mind.hurts()
        try:
            while passing["done"] < passing["total"] and not self._stop.is_set():
                with self.lock:
                    for _ in range(min(200, passing["total"] - passing["done"])):
                        self.mind.step()
                        passing["done"] += 1
                    self._announce(quietly=True)
                if passing["done"] % PASSING_SAVE < 200:
                    self.save()
        finally:
            from .cortex.talk import passed_highlights, passed_note

            with self.lock:
                mind = self.mind
                days = passing["done"] // DAY
                if days:
                    self.initiative.passed = passed_note(
                        days, passed_highlights(mind, passing["from"], passing["before"])
                    )
                    self.initiative.pending.clear()
                if self.thinker is not None and hasattr(self.thinker, "day"):
                    self.thinker.day.clear()  # what it noted of its day before is long ago now
                self._last_night = time.monotonic()
                mind.remember_hurts(hurt_before)  # (what hurt it meanwhile, its brain learns to fear now)
                mind.brain_resting = False
            self.save()
            self.passing = None
            self._emit("event", f"lived {days} days" + ("" if days == passing["days"] else " (it was stopped)"))

    # --- speaking up ----------------------------------------------------------------------------

    def present(self) -> None:
        """Someone is looking at its window."""
        self.initiative.looked()

    def _speak_up(self) -> None:
        thinker = self.thinker
        if thinker is None or not hasattr(thinker, "speak_up") or thinker.busy.locked() or self.passing:
            return
        with self.lock:
            mind = self.mind
            found = self.initiative.occasion(mind, mind.tick, mind.character.traits["friendly"], self._lately)
        if found is None:
            return
        self.initiative.spoke()
        threading.Thread(target=self._say_up, args=found, name="haven-speaking-up", daemon=True).start()

    def _lately(self) -> list[tuple[str, str]]:
        """What it read out of curiosity lately: (title, first sentence)."""
        library = getattr(self.thinker, "_library", None)
        if library is None:
            return []
        return [(t, library.sentence(t, 0)) for t in library.titles("curious")[-3:]]

    def _say_up(self, kind: str, note: str, asked: str | None) -> None:
        thinker = self.thinker
        if not thinker.busy.acquire(blocking=False):
            return  # it's answering someone
        try:
            words, confidence = thinker.speak_up(self, note)
        except Exception as error:  # noqa: BLE001  thinking going wrong mustn't end a life
            self._emit("event", f"couldn't put a thought into words ({error})")
            return
        finally:
            thinker.busy.release()
        if words:
            self.reply(words, "", spoken=True)
            thinker.remember(
                {"you": None, "haven": words, "confidence": confidence, "time": time.time(), "about": kind}
            )
            if asked:
                self.asked = (asked, time.monotonic())
            if kind == "wonder" and "I wonder: " in note:  # it asked what it wondered: the answer may teach it
                from .cortex.think import Thread

                if getattr(self, "talk", None) is None:
                    self.talk = Thread()
                self.talk.wonder(note.split("I wonder: ", 1)[1])

    def question(self) -> str | None:
        """What it asked the person, if it's still waiting for an answer (and it's no longer waiting after this)."""
        asked, self.asked = self.asked, None
        return asked[0] if asked and time.monotonic() - asked[1] < ASKED else None

    def _sleep_on_it(self) -> None:
        """While Haven sleeps (at most every so often), its language cortex goes over moments of its day."""
        thinker, now = self.thinker, time.monotonic()
        if thinker is None or now - self._last_night < SLEEP_EVERY or self._learning.locked():
            return
        self._last_night = now
        threading.Thread(target=self._learn_in_sleep, name="haven-sleep-learning", daemon=True).start()

    def _bedtime(self) -> None:
        """As it falls asleep (at most every so often, in real time), it hears the next part of a story."""
        thinker, now = self.thinker, time.monotonic()
        if thinker is None or not hasattr(thinker, "bedtime_story") or now - self._last_story < STORY_EVERY:
            return
        self._last_story = now
        threading.Thread(target=self._hear_story, name="haven-bedtime-story", daemon=True).start()

    def _hear_story(self) -> None:
        from .cortex.talk import bedtime_note

        try:
            story = self.thinker.bedtime_story(self)
        except Exception as error:  # noqa: BLE001  a story going wrong mustn't end a life
            self._emit("event", f"couldn't hear a story ({error})")
            return
        if story is not None:
            with self.lock:
                self.initiative.pending.append((self.mind.tick, "heard", bedtime_note(story)))

    def _learn_in_sleep(self) -> None:
        with self._learning:
            try:
                self.thinker.sleep_on_it(self)
            except Exception as error:  # noqa: BLE001  learning going wrong mustn't end a life
                self._emit("event", f"couldn't learn in its sleep ({error})")

    def _wonder(self) -> None:
        """When nobody has said anything for a while, Haven now and then reads about something it's curious about
        (if it's allowed to use the internet)."""
        thinker, now = self.thinker, time.monotonic()
        if thinker is None or thinker.web is None or thinker.busy.locked():
            return
        if now - self._last_words < QUIET or now - self._last_wonder < CURIOUS_EVERY:
            return
        self._last_wonder = now
        threading.Thread(target=self._read_for_fun, name="haven-curious", daemon=True).start()

    def _read_for_fun(self) -> None:
        thinker = self.thinker
        if not thinker.busy.acquire(blocking=False):
            return  # it's answering someone
        try:
            thinker.read_for_fun(self)
        except Exception as error:  # noqa: BLE001  a failed reading mustn't end a life
            self._emit("event", f"couldn't read ({error})")
        finally:
            thinker.busy.release()

    def save(self, synapses: bool = False) -> None:
        """Save its life; its brain's state too, and every synapse of it when `synapses` (or every so often)."""
        if self.store is None:
            return
        with self.lock:
            state = self.mind.to_state()
            self._since_save = 0
            brain = self.mind.brain
            if brain is not None:
                synapses = synapses or time.monotonic() - self._synapses_saved > SYNAPSES_EVERY
                brain.save(self.store.root / "brain", synapses=synapses)
                if synapses:
                    self._synapses_saved = time.monotonic()
        self.store.save(state)

    # --- the person ------------------------------------------------------------------

    def say(self, text: str) -> list[str]:
        self._last_words = time.monotonic()
        self.initiative.talked()
        with self.lock:
            words = self.mind.hear(text)
            self.conversation = [*self.conversation[-99:], {"tick": self.mind.tick, "who": "you", "text": text}]
        if self.thinker is not None and text.strip():
            self.thinker.respond_later(self, text)
        return words

    def touch(self) -> None:
        with self.lock:
            self.mind.touch()

    def feed(self) -> None:
        with self.lock:
            self.mind.feed()

    def bring(self, what: str) -> bool:
        """Make something happen in its valley, good or bad, or heal or hurt it (see Mind.bring)."""
        with self.lock:
            return self.mind.bring(what)

    def reply(self, text: str, source: str, spoken: bool = False) -> None:
        """Something Haven says in words, from its language cortex (`spoken`: of its own accord)."""
        self._last_words = time.monotonic()
        self.initiative.talked()
        with self.lock:
            self.mind.world.voice = (self.mind.tick, text)
            self.mind.said = [*self.mind.said[-49:], (self.mind.tick, text)]
            self.conversation = [*self.conversation[-99:], {"tick": self.mind.tick, "who": "haven", "text": text}]
            if spoken:
                self._spoken += 1
                self.spoken = [*self.spoken[-19:], (self._spoken, text)]
        self._emit("said", text + (f"   [{source}]" if source else ""))

    def snapshot(self) -> dict:
        with self.lock:
            state = snapshot(self.mind)
        state["paused"] = self.paused
        state["speed"] = self.speed
        state["conversation"] = self.conversation[-20:]
        state["news"] = [{"id": n, "text": text} for n, text in self.news]
        state["spoken"] = [{"id": n, "text": text} for n, text in self.spoken]
        passing = self.passing
        state["passing"] = None if passing is None else {k: passing[k] for k in ("days", "done", "total")}
        library = getattr(self.thinker, "_library", None)  # what it has read, besides the book it was born with
        sources = dict(library.sources) if library else {}  # (a copy: it may be reading something right now)
        state["read"] = [title for title, source in sources.items() if source != "book"][-8:][::-1]
        state["cortex"] = None if self.thinker is None else self.thinker.describe()
        hearing = getattr(self.thinker, "_hearing", None)  # (the last bedtime story it heard, once it has heard one)
        latest = hearing.latest() if hearing is not None else None
        state["story"] = latest.title if latest else None
        if state.get("brain") is None and self.brain_state == "waking":
            state["brain"] = {"waking": True}
        return state

    # --- telling the people watching -------------------------------------------------

    def _announce(self, quietly: bool = False) -> None:
        """Tell the people watching what it said and did (`quietly`: while time goes by, only keep up with it)."""
        mind = self.mind
        for item in after(mind.said, self._seen_said):
            self._seen_said = item
            if not quietly:
                self._emit("said", item[1])
        for item in after(mind.log, self._seen_log):
            self._seen_log = item
            if not item[1].startswith("said ") and not quietly:
                self._emit("event", item[1])
                self.initiative.noticed(item[1], item[0], mind.body.asleep, mind.time_of_day)

    def _emit(self, kind: str, text: str) -> None:
        if kind == "event" and NEWS.match(text):
            self._news += 1
            self.news = [*self.news[-19:], (self._news, text)]
        for listener in self.listeners:
            with contextlib.suppress(Exception):  # a broken listener mustn't stop a life
                listener(kind, text)


def after(items: list, last: object) -> list:
    """The items that came after `last` in an append-only list that forgets its oldest items."""
    if last is None:
        return items
    for i in range(len(items) - 1, -1, -1):
        if items[i] == last:
            return items[i + 1 :]
    return items
