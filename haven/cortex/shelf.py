"""Haven's shelf: whole encyclopedias, and anything else it's given to read, kept sentence by sentence in a database
on this computer, and searched for the sentence that answers a question.

Its cortex is small: it can't hold an encyclopedia in its connections. So it keeps what it
reads on a shelf (a SQLite database with full-text search, in its home folder), and when
someone asks it something, what it read that answers them comes to mind, as a note it
reads before it speaks ("I read about France: The capital of France is Paris."). It says
it in its own words, from its own cortex, and says where it read it.

What's on the shelf: an encyclopedia it reads in full (see encyclopedia.py: the Simple
English Wikipedia, or the whole English Wikipedia), texts people give it, and web pages
they ask it to read. Finding the answer: a question names what it's about (an article's
title is in it: "the capital of France"), and the rest of the question says what about it
("capital"); the sentence of that article that says the most of the rest is the answer.
If no title is named, the sentences anywhere that say the most of what was asked are.
"""

from __future__ import annotations

import contextlib
import html
import math
import re
import sqlite3
import threading
import time
import unicodedata
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

from . import encyclopedia

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    part INTEGER NOT NULL DEFAULT 0,
    size INTEGER NOT NULL,
    first INTEGER NOT NULL,
    last INTEGER NOT NULL,
    added REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS articles_by_source ON articles (source, part);
CREATE INDEX IF NOT EXISTS articles_by_title ON articles (title);
CREATE INDEX IF NOT EXISTS articles_by_last ON articles (last);
CREATE TABLE IF NOT EXISTS names (name TEXT NOT NULL, article INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS names_by_name ON names (name);
CREATE TABLE IF NOT EXISTS shelves (
    name TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    parts INTEGER NOT NULL,
    done TEXT NOT NULL DEFAULT '',
    articles INTEGER NOT NULL DEFAULT 0,
    started REAL,
    finished REAL
);
"""
SENTENCES = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS sentences USING fts5 "
    "(text, article UNINDEXED, tokenize = 'porter unicode61 remove_diacritics 2')"
)
FILLER = frozenset(
    """a an the of to in on at for with by from and or but is are was were be been being am do does did done have has
    had having can could would should will shall may might must please tell me us about know knows knew you your yours
    i my we our it its this that these those there here what whats who whos whom whose where wheres when whens why how
    hows which explain describe define definition meaning mean means look up find out read reading anything something
    everything heard hear any some more much lot lots exactly really actually ever thing things kind sort type hey hi
    hello haven ok okay so well just also too very like let lets see show give teach learn learned want wanna wondering
    wonder idea ideas info information fact facts called named name many s t he she they him her them his hers their
    theirs""".split()
)
QUESTION = re.compile(
    r"\?\s*$|^(?:(?:hey|hi|ok|okay|so|and|but|well|haven)[, ]+)*(?:what|who|whom|whose|where|when|why|how|which|"
    r"is|are|was|were|do|does|did|can|could|tell|explain|describe|define|teach|look up|read about|find out|"
    r"have you heard|i want to know|i wonder)\b",
    re.I,
)
MEANING = re.compile(  # asked what a word means: its dictionary first
    r"\bmean(?:s|ing)?\b|\bdefin(?:e|ition)\b|\bsynonyms?\b|\bwhat(?:'s| is) (?:a|an|the) word\b", re.IGNORECASE
)
TELLING = re.compile(  # asked to tell about something (two sentences), not what it is (one)
    r"\b(?:tell (?:me|us) (?:something |anything |all |more )?about|what (?:do|did) you know about|"
    r"(?:know|read) anything about|explain|describe|teach (?:me|us) about|i want to know about|"
    r"what can you tell (?:me|us) about|have you (?:ever )?heard (?:of|about))\b",
    re.I,
)
MOST_GIVEN = 20_000  # sentences it keeps of something it's given to read


def shelf_path(root: Path) -> Path:
    """Where Haven's shelf is kept, in its home folder: a folder, with a file for each encyclopedia it has read, and
    one for what people gave it."""
    return Path(root) / "library"


def plain_name(text: str) -> str:
    """A name as it's looked up: lower case, no accents, no brackets or punctuation ("Évian-les-Bains": "evian les
    bains", "Rock (geology)": "rock")."""
    text = re.sub(r"\s*\([^()]*\)", "", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return text[4:] if text.startswith("the ") else text


def names_of(title: str, first: str = "") -> set[str]:
    """What people might call what an article is about: its title, the part before a comma ("Hearst, Ontario":
    "hearst"), and for a person, their surname ("Einstein"), or a ruler's name without the number ("Cleopatra")."""
    found = {plain_name(title)}
    if ", " in title:
        found.add(plain_name(title.split(", ")[0]))
    found.update(plain_name(n) for n in encyclopedia.other_names(title, first))
    return {n for n in found if n}


ROMAN = {"1": "i", "2": "ii", "3": "iii", "4": "iv", "5": "v", "6": "vi", "7": "vii", "8": "viii"}
SAME = {  # words that mean the same, for finding what answers a question (each with the others it might be said as)
    "biggest": ("largest",),
    "largest": ("biggest",),
    "big": ("large",),
    "large": ("big",),
    "tallest": ("highest",),
    "highest": ("tallest",),
    "tall": ("high", "height"),
    "high": ("tall", "height"),
    "far": ("distance", "away"),
    "hot": ("temperature", "heat"),
    "cold": ("temperature",),
    "old": ("age", "years"),
    "small": ("little",),
    "smallest": ("tiniest",),
    "fast": ("speed",),
    "long": ("length",),
    "heavy": ("weight", "weighs"),
    "money": ("currency",),
    "currency": ("money",),
    "invented": ("invention", "inventor", "patent", "patented"),
    "invent": ("invention", "inventor", "patent"),
    "wrote": ("written", "writer", "author"),
    "painted": ("painting", "painter", "painted"),
    "discovered": ("discovery", "discoverer", "found"),
    "made": ("created", "invented", "built"),
    "live": ("lives", "found", "habitat"),
    "born": ("birth",),
    "died": ("death", "killed", "assassinated"),
    "die": ("died", "death", "killed", "assassinated"),
    "sink": ("sank", "sunk"),
    "fall": ("fell", "fallen"),
    "begin": ("began", "begun", "started"),
    "start": ("started", "began", "begun"),
    "build": ("built",),
    "win": ("won",),
    "fight": ("fought",),
    "lose": ("lost",),
    "find": ("found",),
    "speak": ("language", "languages", "spoken"),
    "people": ("population", "inhabitants"),
    "eat": ("food", "diet", "prey", "feed"),
    "synonym": ("same",),
    "synonyms": ("same",),
}
NOT_ASKED = re.compile(  # questions about here and now, or about it: not for an encyclopedia
    r"\b(?:what time is it|what(?:'s| is) the time|what(?:'s| is) the weather|how(?:'s| is) the weather|"
    r"what day is (?:it|today)|what(?:'s| is) today|what(?:'s| is) up|what are you|who are you|how are you|"
    r"how old are you|where are you|what do you (?:like|want|need|think|feel)|do you like|are you)\b",
    re.I,
)


def singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ches", "shes", "xes", "sses")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


IRREGULAR = {"mice": "mouse", "geese": "goose", "children": "child", "women": "woman", "men": "man", "feet": "foot",
             "teeth": "tooth"}


def singulars(word: str) -> set[str]:
    """What a word may be the plural of ("volcanoes": a volcano, "wolves": a wolf, "mice": a mouse)."""
    found = {singular(word)}
    if word in IRREGULAR:
        found.add(IRREGULAR[word])
    if len(word) > 4 and word.endswith("oes"):
        found.add(word[:-2])
    if len(word) > 4 and word.endswith("ves"):
        found |= {word[:-3] + "f", word[:-3] + "fe"}
    return found


PRONOUNS = ("it", "they", "he", "she")  # (not "his" or "its": "His brother died" is about his brother)
DATE = re.compile(
    r"\b\d{3,4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b"
)


def about(sentence: str, named: list[str]) -> bool:
    """Whether a sentence is about what was asked: it starts by naming it ("Koalas do not drink often", every word of
    the name, not just "war" for World War II), or with "it", "they", "he" or "she" ("They eat leaves of eucalyptus
    trees.")."""
    said = re.findall(r"[^\W_]+", sentence)
    words = [word.lower() for word in said]
    if words and words[0] in PRONOUNS:
        return True
    name = [word for word in named if not word.isdigit() and word not in ROMAN.values()]
    start = {form for word in words[: len(name) + 2] for form in (word, singular(word))}
    if len(name) > 1 and name[-1] in words[:3] and said[words.index(name[-1])][0].isupper():
        return True  # (someone's name, the second time: "Einstein was born in Ulm")
    return bool(name) and all(word in start or singulars(word) & start for word in name)


def says(sentence: str, words: tuple[str, ...]) -> bool:
    """Whether a sentence says one of these words as a word of its own, not as part of another ("born", not
    "German-born")."""
    return any(re.search(rf"(?<![-\w]){re.escape(word)}", sentence, re.IGNORECASE) for word in words)


@dataclass
class Found:
    title: str
    sentences: list[str]  # the article's first sentences, and the one that answers (if it isn't one of them)
    index: int  # which of them answers
    count: int  # how many to tell, from there (two when asked to tell about something)
    score: float
    source: str

    @property
    def said(self) -> str:
        return " ".join(self.sentences[self.index : self.index + self.count])


GIVEN = "given"  # the volume of what people give it to read (texts, files, web pages)
ORDER = (GIVEN, "dictionary", "simple", "english")  # where it looks first for an article by its title


class Volume:
    """One part of the shelf, in a file of its own: an encyclopedia, a dictionary, or what people gave it. (A whole
    encyclopedia can be taken off the shelf at once, and what it reads never waits on what it's given.)"""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.db = self.connect()

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, check_same_thread=False, timeout=60)
        db.execute("PRAGMA journal_mode = WAL")
        db.execute("PRAGMA synchronous = NORMAL")
        db.executescript(SCHEMA)
        db.execute(SENTENCES)
        db.commit()
        return db

    def close(self) -> None:
        self.db.close()

    def sentences(self, first: int, last: int, most: int = 12) -> list[str]:
        rows = self.db.execute(
            "SELECT text FROM sentences WHERE rowid BETWEEN ? AND ? ORDER BY rowid LIMIT ?", (first, last, most)
        ).fetchall()
        return [t for (t,) in rows]

    def size(self) -> int:
        """How many sentences are in it (near enough: what's been taken off leaves gaps)."""
        return self.db.execute("SELECT max(last) FROM articles").fetchone()[0] or 0

    @staticmethod
    def next_row(db: sqlite3.Connection) -> int:
        return (db.execute("SELECT max(last) FROM articles").fetchone()[0] or 0) + 1

    @staticmethod
    def put(
        db: sqlite3.Connection,
        title: str,
        sentences: list[str],
        source: str,
        part: int,
        size: int,
        row: int,
        names: list[str] | None = None,
    ) -> int:
        """One article into it (in an open transaction), its sentences from row `row` on, and the names it's looked up
        by (what its title says, unless they're given). Returns the next row."""
        cursor = db.execute(
            "INSERT INTO articles (title, source, part, size, first, last, added) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (title, source, part, size, row, row + len(sentences) - 1, time.time()),
        )
        article = cursor.lastrowid
        db.executemany(
            "INSERT INTO sentences (rowid, text, article) VALUES (?, ?, ?)",
            [(row + i, s, article) for i, s in enumerate(sentences)],
        )
        called = names if names is not None else names_of(title, sentences[0])
        db.executemany("INSERT INTO names (name, article) VALUES (?, ?)", [(n, article) for n in called])
        return row + len(sentences)

    @staticmethod
    def remove(db: sqlite3.Connection, article: int, first: int, last: int) -> None:
        db.execute("DELETE FROM sentences WHERE rowid BETWEEN ? AND ?", (first, last))
        db.execute("DELETE FROM names WHERE article = ?", (article,))
        db.execute("DELETE FROM articles WHERE id = ?", (article,))


def fts5() -> None:
    """Raise sqlite3.OperationalError if this Python's SQLite has no full-text search (FTS5)."""
    with contextlib.closing(sqlite3.connect(":memory:")) as db:
        db.execute("CREATE VIRTUAL TABLE t USING fts5 (x)")


class Shelf:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        fts5()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.volumes: dict[str, Volume] = {path.stem: Volume(path) for path in sorted(self.folder.glob("*.sqlite"))}
        self._counts: dict[str, int] = {}  # how many sentences say a word (how telling it is)
        self._total: int | None = None
        self._counted: tuple[float, dict] | None = None  # (when, what was on it then)

    @property
    def path(self) -> Path:
        return self.folder

    def volume(self, name: str, make: bool = False) -> Volume | None:
        """The volume a source is kept in ("given" and "web" share one), made if `make` and it isn't there yet."""
        name = GIVEN if name in (GIVEN, "web") else name
        with self.lock:
            if name not in self.volumes and make:
                self.volumes[name] = Volume(self.folder / f"{name}.sqlite")
            return self.volumes.get(name)

    def _changed(self) -> None:
        self._total = self._counted = None
        self._counts.clear()

    def close(self) -> None:
        with self.lock:
            for volume in self.volumes.values():
                volume.close()
            self.volumes = {}

    # --- what's on it ------------------------------------------------------------------------------------------

    def counts(self) -> dict:
        """How much is on the shelf (counted again at most every few seconds: a whole Wikipedia takes counting)."""
        counted = self._counted
        if counted is not None and time.monotonic() - counted[0] < 5:
            return counted[1]
        by_source: dict[str, int] = {}
        sentences, shelves = 0, []
        with self.lock:
            for volume in self.volumes.values():
                for source, n in volume.db.execute("SELECT source, count(*) FROM articles GROUP BY source"):
                    by_source[source] = by_source.get(source, 0) + n
                sentences += volume.size()
                shelves += volume.db.execute(
                    "SELECT name, title, parts, done, articles, started, finished FROM shelves"
                ).fetchall()
        found = {
            "articles": sum(by_source.values()),
            "by source": by_source,
            "sentences": sentences,
            "shelves": [
                {
                    "name": name,
                    "title": title,
                    "parts": parts,
                    "done": len([p for p in done.split(",") if p]),
                    "articles": n,
                    "started": started,
                    "finished": finished,
                }
                for name, title, parts, done, n, started, finished in shelves
            ],
        }
        self._counted = (time.monotonic(), found)
        return found

    def titles(self, source: str | None = None, most: int = 50) -> list[str]:
        """The latest titles it was given or read (of a source, if one is named)."""
        with self.lock:
            volumes = [self.volume(source)] if source else list(self.volumes.values())
            found = []
            for volume in volumes:
                if volume is None:
                    continue
                found += volume.db.execute(
                    "SELECT added, title FROM articles"
                    + (" WHERE source = ?" if source else "")
                    + " ORDER BY id DESC LIMIT ?",
                    (source, most) if source else (most,),
                ).fetchall()
        return [t for _, t in sorted(found, reverse=True)[:most]]

    def article(self, title: str, source: str | None = None) -> tuple[str, list[str], str] | None:
        """(title, its sentences, where it came from), of the article with this title, if it's on the shelf."""
        with self.lock:
            names = [source] if source else [n for n in (*ORDER, *self.volumes) if n in self.volumes]
            for name in dict.fromkeys(names):
                volume = self.volume(name)
                if volume is None:
                    continue
                row = volume.db.execute(
                    "SELECT title, first, last, source FROM articles WHERE title = ? ORDER BY id DESC LIMIT 1",
                    (title,),
                ).fetchone()
                if row is not None:
                    return row[0], volume.sentences(row[1], row[2]), row[3]
        return None

    # --- putting things on it ----------------------------------------------------------------------------------

    def add(self, title: str, text: str, source: str = GIVEN, keep: int | None = None) -> int:
        """Something to read: its sentences go on the shelf (replacing anything of the same title it was given before).
        Returns how many sentences it kept."""
        found = []
        paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n|\r\n\s*\r\n", text)]
        if len(paragraphs) == 1:  # (lines, not paragraphs)
            paragraphs = [" ".join(p.split()) for p in text.splitlines()]
        for paragraph in paragraphs[:keep]:
            for sentence in encyclopedia.sentences(paragraph):
                said = encyclopedia.speakable(sentence)
                if said:
                    found.append(said)
            if len(found) >= MOST_GIVEN:
                break
        if not found:
            return 0
        with self.lock:
            db = self.volume(source, make=True).db
            with db:
                for article, first, last in db.execute(
                    "SELECT id, first, last FROM articles WHERE title = ? AND source = ?", (title, source)
                ).fetchall():
                    Volume.remove(db, article, first, last)
                Volume.put(db, title, found[:MOST_GIVEN], source, 0, len(text), Volume.next_row(db))
        self._changed()
        return len(found[:MOST_GIVEN])

    def forget(self, title: str, source: str | None = None) -> int:
        """Take an article off the shelf. Returns how many were taken off."""
        n = 0
        with self.lock:
            for volume in [self.volume(source)] if source else list(self.volumes.values()):
                if volume is None:
                    continue
                with volume.db:
                    rows = volume.db.execute(
                        "SELECT id, first, last FROM articles WHERE title = ?" + (" AND source = ?" if source else ""),
                        (title, source) if source else (title,),
                    ).fetchall()
                    for article, first, last in rows:
                        Volume.remove(volume.db, article, first, last)
                n += len(rows)
        self._changed()
        return n

    def forget_all(self, source: str) -> int:
        """Take a whole encyclopedia off the shelf (or all it was given): its file goes, and the room it took with it.
        Returns how many articles that was."""
        with self.lock:
            volume = self.volume(source)
            if volume is None:
                return 0
            n = volume.db.execute("SELECT count(*) FROM articles").fetchone()[0]
            volume.close()
            del self.volumes[volume.path.stem]
            for path in (volume.path, Path(f"{volume.path}-wal"), Path(f"{volume.path}-shm")):
                path.unlink(missing_ok=True)
        self._changed()
        return n

    # --- reading a whole encyclopedia ---------------------------------------------------------------------------

    def read_encyclopedia(
        self,
        which: encyclopedia.Encyclopedia,
        open_part: Callable[[int], contextlib.AbstractContextManager],
        progress: Callable[[dict], None] = lambda _: None,
        stop: threading.Event | None = None,
    ) -> bool:
        """Read every article of an encyclopedia onto the shelf, part by part (each a file of records, opened by
        `open_part`). It can stop and carry on later: what it finished stays read. Returns whether it read it all."""
        writer = self.volume(which.name, make=True).connect()  # (its own connection: it can answer while it reads)
        try:
            with writer:
                writer.execute(
                    "INSERT OR IGNORE INTO shelves (name, title, parts, started) VALUES (?, ?, ?, ?)",
                    (which.name, which.title, which.files, time.time()),
                )
            for part in range(which.files):
                done = writer.execute("SELECT done FROM shelves WHERE name = ?", (which.name,)).fetchone()[0]
                if str(part) in done.split(","):
                    continue
                if stop is not None and stop.is_set():
                    return False
                with writer:  # (anything left from a part it didn't finish)
                    for article, first, last in writer.execute(
                        "SELECT id, first, last FROM articles WHERE part = ?", (part,)
                    ).fetchall():
                        Volume.remove(writer, article, first, last)
                n = self._read_part(writer, which, part, open_part, progress, stop)
                if n is None:
                    return False
                with writer:
                    writer.execute(
                        "UPDATE shelves SET done = done || ? || ',', articles = articles + ? WHERE name = ?",
                        (str(part), n, which.name),
                    )
                self._changed()
            with writer:
                writer.execute("UPDATE shelves SET finished = ? WHERE name = ?", (time.time(), which.name))
            return True
        finally:
            writer.close()

    def _read_part(self, writer, which, part: int, open_part, progress, stop) -> int | None:
        """Read one part of an encyclopedia onto the shelf, a batch of articles at a time (so finding answers is never
        held up for long). Returns how many articles it read, or None if it was stopped (what it read of this part is
        taken off again, next time)."""
        n, batch = 0, []
        with open_part(part) as stream:
            for article in self._articles(which, stream):
                if stop is not None and stop.is_set():
                    return None
                batch.append(article)
                if len(batch) >= 2000:
                    n += self._write(writer, which, part, batch)
                    batch = []
                    progress({"part": part, "parts": which.files, "articles": n})
        n += self._write(writer, which, part, batch)
        progress({"part": part + 1, "parts": which.files, "articles": n})
        return n

    def _write(self, db: sqlite3.Connection, which, part: int, batch: list) -> int:
        if not batch:
            return 0
        with db:
            row = Volume.next_row(db)
            for title, found, size, names in batch:
                row = Volume.put(db, title, found, which.name, part, size, row, names)
        self._counted = None
        return len(batch)

    def _articles(self, which, stream):
        """(title, the sentences it keeps, how big the article is, its names or None) of each article in a part."""
        if which.url.endswith(".zip"):  # a dictionary: WordNet's files, zipped
            import io
            import zipfile

            for title, sentences, names in encyclopedia.dictionary(zipfile.ZipFile(io.BytesIO(stream.read()))):
                yield title, sentences, 0, names
            return
        for title, text in encyclopedia.articles(stream):
            paragraphs = encyclopedia.prose(text, which.keep)
            found = [said for p in paragraphs for s in encyclopedia.sentences(p) if (said := encyclopedia.speakable(s))]
            if found:
                yield title, found, len(text), None

    # --- finding what answers a question ------------------------------------------------------------------------

    def find(self, question: str) -> Found | None:
        """What it read that answers a question, if anything does."""
        text = " ".join(question.strip().split())
        if not QUESTION.search(text) or NOT_ASKED.search(text):
            return None  # (only questions about the world, and being asked to tell about something)
        count = 2 if TELLING.search(text) else 1
        said = [t.split("'")[0].split("’")[0] for t in re.findall(r"[^\W_]+(?:['’][^\W_]+)?", text)]
        tokens = [t.lower() for t in said]
        content = [t for t in tokens if t not in FILLER]
        if not content:
            return None
        who = tokens[0] == "who" or tokens[:2] in (["who", "was"], ["who", "is"])
        when = tokens[0] == "when" or tokens[:2] == ["what", "year"]
        meaning = bool(MEANING.search(text))
        if meaning:
            content = [t for t in content if t not in ("mean", "means", "meaning", "define", "definition", "word")]
            if not content:
                return None
        with self.lock:
            named = self._named(tokens, content, count, who, when, said)
            found = [*named, *self._anywhere(content, count, bool(named))]
        for f in found:  # (a dictionary says what words mean; an encyclopedia, what things are)
            if f.source == "dictionary":
                f.score += 4.0 if meaning else -1.5
        if not found:
            return None
        return max(found, key=lambda f: f.score)

    def _named(
        self,
        tokens: list[str],
        content: list[str],
        count: int,
        who: bool = False,
        when: bool = False,
        said: list[str] | None = None,
    ) -> list[Found]:
        """Answers from articles a question names, by their titles (or what else they're called). (`said`: the words
        as they were said, in capitals or not.)"""
        said = said or tokens
        why = tokens[0] == "why"
        found = []
        seen: set[tuple[str, int]] = set()
        for n in range(min(6, len(tokens)), 0, -1):
            for i in range(len(tokens) - n + 1):
                gram = tokens[i : i + n]
                if gram[0] in FILLER or gram[-1] in FILLER:
                    continue
                if why and n > 1 and i + n == len(tokens):
                    continue  # ("why is the sky blue": about the sky, not the colour sky blue)
                capital = all(word[:1].isupper() for word in said[i : i + n])  # (said as a name)
                names = {" ".join(gram), *(" ".join([*gram[:-1], one]) for one in singulars(gram[-1]))}
                if gram[-1] in ROMAN and n > 1:  # ("World War 2": "World War II")
                    names.add(" ".join([*gram[:-1], ROMAN[gram[-1]]]))
                for name in names:
                    for volume in self.volumes.values():
                        rows = volume.db.execute(  # (the likeliest few of its namesakes)
                            "SELECT a.id, a.title, a.first, a.last, a.size, a.source FROM names n "
                            "JOIN articles a ON a.id = n.article WHERE n.name = ? ORDER BY a.size DESC LIMIT 4",
                            (name,),
                        ).fetchall()
                        for article, title, first, last, size, source in rows:
                            if (volume.path.stem, article) in seen:
                                continue
                            seen.add((volume.path.stem, article))
                            named = [t for t in gram if t not in FILLER]
                            rest = [t for t in content if t not in gram and t not in ROMAN]
                            exact = plain_name(title) == name and "(" not in title  # (its own article, not a namesake)
                            answer = self._in_article(
                                volume, title, first, last, size, source, named, rest, count, exact, who, when, capital
                            )
                            if answer is not None:
                                found.append(answer)
            if found and n > 1:
                break  # (the longest names first: "New York City", not "York")
        return found

    def _in_article(
        self, volume, title, first, last, size, source, named, rest, count, exact, who, when, capital=True
    ) -> Found | None:
        lead = volume.sentences(first, last)
        if not (capital or exact) and len(named) == 1 and lead and encyclopedia.PERSON.search(lead[0][:200]):
            return None  # (someone of that name, but "the moon" isn't Mr Moon)
        base = 3.0 * len(named) + 0.3 * math.log10(1 + size) + (1.0 if exact else 0.0)
        base += 0.5 if source == "simple" else 0.0  # (in simple words: easier to tell)
        if who and lead and encyclopedia.PERSON.search(lead[0][:200]):
            base += 2.0  # (asked who: a person)
        if not rest:  # just what it is: the start of the article
            return Found(title, lead, 0, min(count, len(lead)), 10 + base, source)
        weights = {t: self._weight(t) for t in set(rest)}
        has: dict[int, float] = {}
        said: dict[int, str] = {}
        literal: set[int] = set()  # (sentences that say a word as it was asked, not another word for it)
        for term, weight in weights.items():
            for row, sentence in volume.db.execute(
                "SELECT rowid, text FROM sentences WHERE sentences MATCH ? AND rowid BETWEEN ? AND ? LIMIT 400",
                (self._query(term), first, last),
            ):
                if says(sentence, (term,)):  # (the word it was asked, as a word of its own)
                    literal.add(row)
                joined = not says(sentence, (term, *SAME.get(term, ()))) and re.search(
                    rf"-(?:{'|'.join(map(re.escape, (term, *SAME.get(term, ()))))})", sentence, re.IGNORECASE
                )  # (only as part of another word: "German-born", asked when someone was born)
                has[row] = has.get(row, 0.0) + (weight / 2 if joined else weight)
                said[row] = sentence
        if not has:  # nothing it read about it says that: what it is, at least (less likely the answer; and a namesake,
            return Found(title, lead, 0, 1, (3.0 if exact else 1.0) + base, source)  # like a Mr Panda, less still)
        lean = 0.3 * max(weights.values())  # (of sentences that say as much, the first one about what was asked,
        rank = {  # and asked when, one that says when; and one that says what was asked as it was asked: "eat", not "feeds")
            r: has[r]
            + (lean if about(said[r], named) else 0.0)
            + (lean if when and DATE.search(said[r]) else 0.0)
            + (0.8 * lean if r in literal else 0.0)
            for r in has
        }
        row = min(rank, key=lambda r: (-rank[r], r))
        covered = has[row] / sum(weights.values())
        sentence = said[row]
        index = row - first
        sentences = lead
        if index >= len(lead):
            sentences, index = [*lead, sentence], len(lead)
        return Found(title, sentences, index, 1, 4 + 6 * covered + base, source)

    def _anywhere(self, content: list[str], count: int, named: bool = False) -> list[Found]:
        """Answers from sentences anywhere that say the most of what was asked (when no title is named, or as well)."""
        terms = sorted(set(content), key=self._weight, reverse=True)  # the most telling words first
        total = sum(self._weight(t) for t in terms)
        if not terms or total <= 0:
            return []
        for size in range(len(terms), 0, -1):
            if size < min(2, len(terms)):
                break
            used = terms[:size]
            covered = sum(self._weight(t) for t in used) / total
            if covered < 0.6:
                break
            found = []
            for volume in self.volumes.values():
                rows = volume.db.execute(
                    "SELECT rowid, text, article, bm25(sentences) FROM sentences WHERE sentences MATCH ? "
                    "ORDER BY rank LIMIT 8",
                    (" AND ".join(self._query(t) for t in used),),
                ).fetchall()
                for row, sentence, article, rank in rows:
                    title, first, last, size_, source = volume.db.execute(
                        "SELECT title, first, last, size, source FROM articles WHERE id = ?", (article,)
                    ).fetchone()
                    lead = volume.sentences(first, last)
                    index = row - first
                    sentences = lead if index < len(lead) else [*lead, sentence]
                    index = min(index, len(sentences) - 1)
                    words = set(plain_name(title).split())
                    titled = sum(1 for t in used if t in words or singulars(t) & words)
                    score = 6 * covered + 2 * titled - 0.05 * index + 0.3 * math.log10(1 + size_) - 0.01 * rank
                    score -= 2.0 if named else 0.0  # (an article the question names is likelier to answer it)
                    found.append(Found(title, sentences, index, 1, score, source))
            if found:
                return found
        return []

    def _term(self, word: str) -> str:
        return '"' + word.replace('"', "") + '"'

    def _query(self, word: str) -> str:
        """A word to search for, and the words that mean the same ("biggest": biggest or largest)."""
        said = [word, *SAME.get(word, ())]
        return self._term(word) if len(said) == 1 else "(" + " OR ".join(self._term(w) for w in said) + ")"

    def _weight(self, word: str) -> float:
        """How telling a word is: rarer words say more about what was asked."""
        if word not in self._counts:
            self._counts[word] = sum(
                v.db.execute("SELECT count(*) FROM sentences WHERE sentences MATCH ?", (self._term(word),)).fetchone()[
                    0
                ]
                for v in self.volumes.values()
            )
        if self._total is None:
            self._total = sum(v.size() for v in self.volumes.values()) + 1
        return math.log(1 + self._total / (1 + self._counts[word]))


# --- web pages, as text ---------------------------------------------------------------------------------------------


class _Text(HTMLParser):
    SKIP = frozenset({"script", "style", "nav", "header", "footer", "aside", "form", "noscript", "svg", "button"})
    BLOCK = frozenset({"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article", "td"})

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def page_text(page: str, url: str = "") -> tuple[str, str]:
    """(title, text) of a web page: its words, without menus, scripts or styles."""
    if not re.search(r"<(?:html|body|p|div)\b", page[:20000], re.I):
        name = urllib.parse.unquote(url.rstrip("/").rsplit("/", 1)[-1]) or url
        return name, page  # plain text
    reader = _Text()
    reader.feed(page)
    reader.close()
    text = "".join(reader.parts)
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text)]
    title = html.unescape(" ".join(reader.title.split())) or urllib.parse.urlsplit(url).netloc or url
    return title, "\n\n".join(p for p in paragraphs if len(p.split()) >= 6)


def title_for(text: str) -> str:
    """A title for something pasted in: its first line, if that's short and not a sentence, or its first words."""
    first = text.strip().splitlines()[0].strip() if text.strip() else ""
    if 0 < len(first) <= 80 and not re.search(r"[.!?]$", first):
        return first
    words = text.split()[:6]
    return " ".join(words).rstrip(".,;:") + "…" if words else "Something you gave me"
