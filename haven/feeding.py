"""Feeding Haven what to read: whole encyclopedias, texts, files and web pages, onto its shelf (see cortex/shelf.py).

The first time Haven wakes up in the app, it starts reading the Simple English Wikipedia by
itself, in the background, while it lives (unless it isn't allowed on the internet). People
can give it more: the whole English Wikipedia, something pasted in, a file, a web page.
Whatever it has read, it can answer from at once, and it says where it read it.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path

from .cortex.encyclopedia import ENCYCLOPEDIAS, Encyclopedia
from .cortex.shelf import Shelf, page_text, shelf_path, title_for
from .web import Web, WebError

MOST_PAGE = 5_000_000  # bytes of a web page it reads
MOST_TEXT = 20_000_000  # characters of a text it's given at once


class FeedError(Exception):
    pass


class Feeding:
    def __init__(
        self,
        root: Path,
        web: Web | None,
        log: Callable[[str], None] = print,
        emit: Callable[[str], None] | None = None,
    ):
        self.root, self.web, self.log = Path(root), web, log
        self.emit = emit or (lambda text: None)  # (what it does, as news in the conversation)
        self.problem: str | None = None
        try:
            self.shelf: Shelf | None = Shelf(shelf_path(self.root))
        except (sqlite3.Error, OSError) as error:  # (a Python without full-text search, or no room on the disk)
            self.shelf, self.problem = None, f"it can't keep a shelf of what it reads here ({error})"
            self.log(f"Haven can't keep a shelf of what it reads on this computer ({error}).")
        self.reading: dict | None = None  # the encyclopedia it's reading now, and how far it's got
        self.last: dict | None = None  # how its last reading went
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def paused_path(self) -> Path:
        return shelf_path(self.root) / "paused"

    # --- what it has read ------------------------------------------------------------------------------------------

    def _shelf(self) -> Shelf:
        if self.shelf is None:
            raise FeedError(self.problem or "it has no shelf to read onto")
        return self.shelf

    def status(self) -> dict:
        if self.shelf is None:
            return {
                "articles": 0,
                "sentences": 0,
                "by source": {},
                "reading": None,
                "last": None,
                "online": False,
                "encyclopedias": [],
                "given": [],
                "problem": self.problem,
            }
        counts = self.shelf.counts()
        shelves = {s["name"]: s for s in counts["shelves"]}
        reading = dict(self.reading) if self.reading else None
        if reading:
            reading["articles"] = counts["by source"].get(reading["name"], 0)
        return {
            "articles": counts["articles"],
            "sentences": counts["sentences"],
            "by source": counts["by source"],
            "reading": reading,
            "last": self.last,
            "online": self.web is not None,
            "encyclopedias": [
                {
                    "name": e.name,
                    "title": e.title,
                    "articles": e.articles,
                    "size": e.size,
                    "about": e.about,
                    "read": counts["by source"].get(e.name, 0),
                    "finished": bool(shelves.get(e.name, {}).get("finished")),
                }
                for e in ENCYCLOPEDIAS.values()
            ],
            "given": self.shelf.titles("given", 12) + self.shelf.titles("web", 12),
        }

    # --- reading a whole encyclopedia, in the background -------------------------------------------------------------

    FIRST = ("dictionary", "simple")  # what it reads by itself, the first time: a dictionary, and an encyclopedia

    def start(self) -> bool:
        """The first time (and every time after, until it has finished, unless someone paused it): read a dictionary
        and the Simple English Wikipedia. Returns whether it started reading."""
        if self.web is None or self.shelf is None or self.paused_path.exists():
            return False
        done = {e["name"] for e in self.status()["encyclopedias"] if e["finished"]}
        left = [name for name in self.FIRST if name not in done]
        return bool(left) and self.read(*left)

    def read(self, *names: str) -> bool:
        """Start reading encyclopedias, one after another, in the background. Returns False if it's reading already."""
        found = [ENCYCLOPEDIAS.get(name) for name in names]
        if not names or None in found:
            wrong = next((n for n, w in zip(names, found, strict=True) if w is None), "")
            raise FeedError(f"there's no encyclopedia called {wrong!r} (there's {', '.join(ENCYCLOPEDIAS)})")
        if self.web is None:
            raise FeedError("Haven isn't allowed on the internet here, so it can't get an encyclopedia")
        self._shelf()
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop.clear()
            self.paused_path.unlink(missing_ok=True)
            self.reading = {"name": found[0].name, "title": found[0].title, "part": 0, "parts": found[0].files}
            self._thread = threading.Thread(target=self._read_all, args=(found,), name="haven-reading", daemon=True)
            self._thread.start()
        return True

    def _read_all(self, encyclopedias: list[Encyclopedia]) -> None:
        for which in encyclopedias:
            if self._stop.is_set():
                break
            self.reading = {"name": which.name, "title": which.title, "part": 0, "parts": which.files}
            self._read(which)

    def stop(self, pause: bool = True) -> None:
        """Stop reading for now (what it finished stays read). `pause`: and don't start again by itself."""
        self._stop.set()
        if pause:
            self.paused_path.parent.mkdir(parents=True, exist_ok=True)
            self.paused_path.write_text("stopped by the person\n")

    def wait(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _read(self, which: Encyclopedia) -> None:
        try:
            self._reading(which)
        except sqlite3.Error as error:  # (no room left on the disk, say)
            self.last = {"name": which.name, "title": which.title, "finished": False, "error": str(error)}
            self.emit(f"couldn't go on reading {which.title} ({error})")
            self.log(f"Haven couldn't go on reading {which.title}: {error}")
        finally:
            self.reading = None

    def _reading(self, which: Encyclopedia) -> None:
        urls = which.urls()
        began = self.shelf.counts()["by source"].get(which.name, 0)

        def progress(found: dict) -> None:
            if self.reading is not None:
                self.reading.update(part=found["part"], parts=found["parts"])

        self.emit(f"started reading {which.title}")
        self.log(f"Haven is reading {which.title} ({which.size / 1e6:,.0f} MB to download), in the background.")
        tries, finished, error = 0, False, None
        while True:
            try:
                finished = self.shelf.read_encyclopedia(
                    which, lambda part: self.web.open(urls[part]), progress, self._stop
                )
                break
            except (WebError, OSError, ValueError) as problem:  # (it carries on where it was: what it read stays read)
                error = str(problem)
                tries += 1
                if tries > 5 or self._stop.wait(min(60, 2**tries)):
                    break
        read = self.shelf.counts()["by source"].get(which.name, 0)
        self.last = {"name": which.name, "title": which.title, "finished": finished, "articles": read, "error": error}
        if finished:
            self.emit(f"finished reading {which.title}: {read:,} {which.unit}")
            self.log(f"Haven has read {which.title}: {read:,} {which.unit}.")
        elif self._stop.is_set():
            self.emit(f"stopped reading {which.title} for now, {read:,} {which.unit} in")
        else:
            self.emit(f"couldn't go on reading {which.title} ({error}); it read {read - began:,} more {which.unit}")
            self.log(f"Haven couldn't go on reading {which.title}: {error}")

    # --- texts, files and web pages ---------------------------------------------------------------------------------

    def give(self, text: str, title: str | None = None, source: str = "given") -> tuple[str, int]:
        """Something to read: a text pasted in, or a file. Returns (its title, how many sentences it kept)."""
        text = text[:MOST_TEXT]
        if not text.strip():
            raise FeedError("there's nothing in it to read")
        title = " ".join((title or title_for(text)).split())[:120]
        if re.search(r"<(?:html|body|p|div)\b", text[:20000], re.IGNORECASE):  # (a web page saved as a file)
            found, text = page_text(text, title)
            title = title or found
        kept = self._shelf().add(title, text, source)
        if not kept:
            raise FeedError("it couldn't find any sentences in it to read")
        self.emit(f"read what you gave it: {title} ({kept:,} sentences)")
        return title, kept

    def page(self, url: str) -> tuple[str, int]:
        """Read a web page. Returns (its title, how many sentences it kept)."""
        if self.web is None:
            raise FeedError("Haven isn't allowed on the internet here")
        url = url.strip()
        if "://" not in url:
            url = "https://" + url
        try:
            body = self.web.get(url, MOST_PAGE, "text/html, text/plain;q=0.9, */*;q=0.5")
        except WebError as error:
            raise FeedError(str(error)) from None
        title, text = page_text(body.decode("utf-8", "replace"), url)
        kept = self._shelf().add(title, text, "web")
        if not kept:
            raise FeedError("it couldn't find any sentences on that page to read")
        self.emit(f"read the page {title} ({kept:,} sentences)")
        return title, kept

    def forget(self, title: str) -> int:
        return self._shelf().forget(title)
