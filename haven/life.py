"""Running Haven's life in real time, in the background, saving as it goes."""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable

from .mind import VERSION, Mind, moved
from .report import snapshot
from .store import Store

AUTOSAVE = 600  # ticks between saves


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
            if self._since_save >= AUTOSAVE:
                self.save()
            if asleep and tick % 200 == 0:
                self._consolidate()

    def _consolidate(self) -> None:
        """While Haven sleeps, its language cortex goes over what it heard and read."""
        thinker = self.thinker
        if thinker is None or thinker.busy.locked():
            return
        threading.Thread(target=thinker.consolidate, name="haven-sleep-learning", daemon=True).start()

    def save(self) -> None:
        if self.store is None:
            return
        with self.lock:
            state = self.mind.to_state()
            self._since_save = 0
        self.store.save(state)

    # --- the person ------------------------------------------------------------------

    def say(self, text: str) -> list[str]:
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
