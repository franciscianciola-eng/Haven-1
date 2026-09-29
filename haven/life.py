"""Running Haven's life in real time, in the background, saving as it goes."""

from __future__ import annotations

import contextlib
import re
import threading
import time
from collections.abc import Callable

from .mind import VERSION, Mind, moved
from .report import snapshot
from .store import Store

AUTOSAVE = 600  # ticks between saves
QUIET = 180.0  # seconds nobody has said anything before Haven reads about something out of curiosity
CURIOUS_EVERY = 900.0  # seconds, at least, between the things it reads out of curiosity
SLEEP_EVERY = 1800.0  # seconds, at least, between nights it learns from its day (its days are short)
FIRST_NIGHT = 600.0  # seconds after it wakes up in the app before the first time it can
NOTICE = 25  # ticks between the moments of its day it notes down, to learn from
NEWS = re.compile(  # what it does that's worth telling the person talking with it
    r"^(?:set off to|did what it was asked|stopped trying to|gave up trying to|couldn't .*: it didn't know where|"
    r"read about|went over its day|learned the word|was taught that)"
)


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
        self.news: list[tuple[int, str]] = []  # (number, what): what it did lately that's worth telling
        self._news = 0

    # --- running --------------------------------------------------------------------

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="haven-life", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.save()

    def _run(self) -> None:
        next_tick = time.monotonic()
        while not self._stop.is_set():
            if self.paused:
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
                self.mind.step()
                self._since_save += 1
                self._announce()
                asleep = self.mind.body.asleep
                tick = self.mind.tick
                if self.thinker is not None and not asleep and tick % NOTICE == 0:
                    self.thinker.notice(self.mind)
            if self._since_save >= AUTOSAVE:
                self.save()
            if asleep and tick % 200 == 0:
                self._sleep_on_it()
            elif not asleep and tick % 50 == 0:
                self._wonder()

    def _sleep_on_it(self) -> None:
        """While Haven sleeps (at most every so often), its language cortex goes over moments of its day."""
        thinker, now = self.thinker, time.monotonic()
        if thinker is None or now - self._last_night < SLEEP_EVERY or self._learning.locked():
            return
        self._last_night = now
        threading.Thread(target=self._learn_in_sleep, name="haven-sleep-learning", daemon=True).start()

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

    def save(self) -> None:
        if self.store is None:
            return
        with self.lock:
            state = self.mind.to_state()
            self._since_save = 0
        self.store.save(state)

    # --- the person ------------------------------------------------------------------

    def say(self, text: str) -> list[str]:
        self._last_words = time.monotonic()
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

    def reply(self, text: str, source: str) -> None:
        """Something Haven says in words, from its language cortex."""
        self._last_words = time.monotonic()
        with self.lock:
            self.mind.world.voice = (self.mind.tick, text)
            self.mind.said = [*self.mind.said[-49:], (self.mind.tick, text)]
            self.conversation = [*self.conversation[-99:], {"tick": self.mind.tick, "who": "haven", "text": text}]
        self._emit("said", text + (f"   [{source}]" if source else ""))

    def snapshot(self) -> dict:
        with self.lock:
            state = snapshot(self.mind)
        state["paused"] = self.paused
        state["speed"] = self.speed
        state["conversation"] = self.conversation[-20:]
        state["news"] = [{"id": n, "text": text} for n, text in self.news]
        library = getattr(self.thinker, "_library", None)  # what it has read, besides the book it was born with
        sources = dict(library.sources) if library else {}  # (a copy: it may be reading something right now)
        state["read"] = [title for title, source in sources.items() if source != "book"][-8:][::-1]
        state["cortex"] = None if self.thinker is None else self.thinker.describe()
        return state

    # --- telling the people watching -------------------------------------------------

    def _announce(self) -> None:
        mind = self.mind
        for item in after(mind.said, self._seen_said):
            self._seen_said = item
            self._emit("said", item[1])
        for item in after(mind.log, self._seen_log):
            self._seen_log = item
            if not item[1].startswith("said "):
                self._emit("event", item[1])

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
