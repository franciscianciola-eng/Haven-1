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
    wonder idea ideas info information fact facts called named name many s t""".split()
)
QUESTION = re.compile(
    r"\?\s*$|^(?:(?:hey|hi|ok|okay|so|and|but|well|haven)[, ]+)*(?:what|who|whom|whose|where|when|why|how|which|"
    r"is|are|was|were|do|does|did|can|could|tell|explain|describe|define|teach|look up|read about|find out|"
    r"have you heard|i want to know|i wonder)\b",
    re.I,
)
TELLING = re.compile(  # asked to tell about something (two sentences), not what it is (one)
    r"\b(?:tell (?:me|us) (?:something |anything |all |more )?about|what (?:do|did) you know about|"
    r"(?:know|read) anything about|explain|describe|teach (?:me|us) about|i want to know about|"
    r"what can you tell (?:me|us) about|have you (?:ever )?heard (?:of|about))\b",
    re.I,
)
MOST_GIVEN = 20_000  # sentences it keeps of something it's given to read


def shelf_path(root: Path) -> Path:
    """Where Haven's shelf is kept, in its home folder."""
    return Path(root) / "library" / "shelf.sqlite"


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
    "died": ("death",),
    "speak": ("language", "languages", "spoken"),
    "people": ("population", "inhabitants"),
    "eat": ("food", "diet", "prey", "feed"),
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


class Shelf:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = self._connect()
        self._counts: dict[str, int] = {}  # how many sentences say a word (how telling it is)
        self._total: int | None = None
        self._counted: tuple[float, dict] | None = None  # (when, what was on it then)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, check_same_thread=False, timeout=60)
        db.execute("PRAGMA journal_mode = WAL")
        db.execute("PRAGMA synchronous = NORMAL")
        db.executescript(SCHEMA)
        db.execute(SENTENCES)
        db.commit()
        return db

    def close(self) -> None:
        with self.lock:
            self.db.close()

    # --- what's on it ------------------------------------------------------------------------------------------

    def counts(self) -> dict:
        """How much is on the shelf (counted again at most every few seconds: a whole Wikipedia takes counting)."""
        counted = self._counted
        if counted is not None and time.monotonic() - counted[0] < 5:
            return counted[1]
        with self.lock:
            articles = self.db.execute("SELECT source, count(*) FROM articles GROUP BY source").fetchall()
            sentences = self.db.execute("SELECT max(last) FROM articles").fetchone()[0] or 0
            shelves = self.db.execute(
                "SELECT name, title, parts, done, articles, started, finished FROM shelves"
            ).fetchall()
        found = {
            "articles": sum(n for _, n in articles),
            "by source": dict(articles),
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
            rows = self.db.execute(
                "SELECT title FROM articles" + (" WHERE source = ?" if source else "") + " ORDER BY id DESC LIMIT ?",
                (source, most) if source else (most,),
            ).fetchall()
        return [t for (t,) in rows]

    def article(self, title: str) -> tuple[str, list[str], str] | None:
        """(title, its sentences, where it came from), of the article with this title, if it's on the shelf."""
        with self.lock:
            row = self.db.execute(
                "SELECT id, title, first, last, source FROM articles WHERE title = ? ORDER BY id DESC LIMIT 1",
                (title,),
            ).fetchone()
            if row is None:
                return None
            return row[1], self._sentences(row[2], row[3]), row[4]

    def _sentences(self, first: int, last: int, most: int = 12) -> list[str]:
        rows = self.db.execute(
            "SELECT text FROM sentences WHERE rowid BETWEEN ? AND ? ORDER BY rowid LIMIT ?", (first, last, most)
        ).fetchall()
        return [t for (t,) in rows]

    # --- putting things on it ----------------------------------------------------------------------------------

    def _next_row(self, db: sqlite3.Connection) -> int:
        return (db.execute("SELECT max(last) FROM articles").fetchone()[0] or 0) + 1

    def _put(
        self, db: sqlite3.Connection, title: str, sentences: list[str], source: str, part: int, size: int, row: int
    ) -> int:
        """One article onto the shelf (in an open transaction), its sentences from row `row` on. Returns the next row."""
        cursor = db.execute(
            "INSERT INTO articles (title, source, part, size, first, last, added) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (title, source, part, size, row, row + len(sentences) - 1, time.time()),
        )
        article = cursor.lastrowid
        db.executemany(
            "INSERT INTO sentences (rowid, text, article) VALUES (?, ?, ?)",
            [(row + i, s, article) for i, s in enumerate(sentences)],
        )
        db.executemany(
            "INSERT INTO names (name, article) VALUES (?, ?)", [(n, article) for n in names_of(title, sentences[0])]
        )
        return row + len(sentences)

    def add(self, title: str, text: str, source: str = "given", keep: int | None = None) -> int:
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
            self.forget(title, source)
            with self.db:
                self._put(self.db, title, found[:MOST_GIVEN], source, 0, len(text), self._next_row(self.db))
        self._total = self._counted = None
        return len(found[:MOST_GIVEN])

    def forget(self, title: str, source: str | None = None) -> int:
        """Take an article off the shelf. Returns how many were taken off."""
        with self.lock, self.db:
            rows = self.db.execute(
                "SELECT id, first, last FROM articles WHERE title = ?" + (" AND source = ?" if source else ""),
                (title, source) if source else (title,),
            ).fetchall()
            for article, first, last in rows:
                self._remove(self.db, article, first, last)
        self._total = self._counted = None
        return len(rows)

    def _remove(self, db: sqlite3.Connection, article: int, first: int, last: int) -> None:
        db.execute("DELETE FROM sentences WHERE rowid BETWEEN ? AND ?", (first, last))
        db.execute("DELETE FROM names WHERE article = ?", (article,))
        db.execute("DELETE FROM articles WHERE id = ?", (article,))

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
        writer = self._connect()  # (its own connection: it can find answers while it reads)
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
                        "SELECT id, first, last FROM articles WHERE source = ? AND part = ?", (which.name, part)
                    ).fetchall():
                        self._remove(writer, article, first, last)
                n = self._read_part(writer, which, part, open_part, progress, stop)
                if n is None:
                    return False
                with writer:
                    writer.execute(
                        "UPDATE shelves SET done = done || ? || ',', articles = articles + ? WHERE name = ?",
                        (str(part), n, which.name),
                    )
                self._total = self._counted = None
                self._counts.clear()
            with writer:
                writer.execute("UPDATE shelves SET finished = ? WHERE name = ?", (time.time(), which.name))
            return True
        finally:
            writer.close()

    def _read_part(self, writer, which, part: int, open_part, progress, stop) -> int | None:
        n, batch = 0, 0
        row = self._next_row(writer)
        writer.execute("BEGIN")
        try:
            with open_part(part) as stream:
                for title, text in encyclopedia.articles(stream):
                    if stop is not None and stop.is_set():
                        writer.rollback()
                        return None
                    paragraphs = encyclopedia.prose(text, which.keep)
                    found = [
                        said
                        for p in paragraphs
                        for s in encyclopedia.sentences(p)
                        if (said := encyclopedia.speakable(s))
                    ]
                    if not found:
                        continue
                    row = self._put(writer, title, found, which.name, part, len(text), row)
                    n += 1
                    batch += 1
                    if batch >= 2000:  # (in steps, so finding answers isn't held up for long)
                        writer.commit()
                        writer.execute("BEGIN")
                        batch = 0
                        self._counted = None
                        progress({"part": part, "parts": which.files, "articles": n})
            writer.commit()
        except BaseException:
            writer.rollback()
            raise
        progress({"part": part + 1, "parts": which.files, "articles": n})
        return n

    # --- finding what answers a question ------------------------------------------------------------------------

    def find(self, question: str) -> Found | None:
        """What it read that answers a question, if anything does."""
        text = " ".join(question.strip().split())
        if not QUESTION.search(text) or NOT_ASKED.search(text):
            return None  # (only questions about the world, and being asked to tell about something)
        count = 2 if TELLING.search(text) else 1
        tokens = re.findall(r"[^\W_]+(?:['’][^\W_]+)?", text.lower())
        tokens = [t.split("'")[0].split("’")[0] for t in tokens]
        content = [t for t in tokens if t not in FILLER]
        if not content:
            return None
        who = tokens[0] == "who" or tokens[:2] in (["who", "was"], ["who", "is"])
        with self.lock:
            named = self._named(tokens, content, count, who)
            found = [*named, *self._anywhere(content, count, bool(named))]
        if not found:
            return None
        return max(found, key=lambda f: f.score)

    def _named(self, tokens: list[str], content: list[str], count: int, who: bool = False) -> list[Found]:
        """Answers from articles a question names, by their titles (or what else they're called)."""
        why = tokens[0] in ("why", "how")
        found = []
        seen: set[int] = set()
        for n in range(min(6, len(tokens)), 0, -1):
            for i in range(len(tokens) - n + 1):
                gram = tokens[i : i + n]
                if gram[0] in FILLER or gram[-1] in FILLER:
                    continue
                if why and n > 1 and i + n == len(tokens):
                    continue  # ("why is the sky blue": about the sky, not the colour sky blue)
                said = {" ".join(gram), " ".join([*gram[:-1], singular(gram[-1])])}
                if gram[-1] in ROMAN and n > 1:  # ("World War 2": "World War II")
                    said.add(" ".join([*gram[:-1], ROMAN[gram[-1]]]))
                for name in said:
                    for article, title, first, last, size, source in self.db.execute(
                        "SELECT a.id, a.title, a.first, a.last, a.size, a.source FROM names n "
                        "JOIN articles a ON a.id = n.article WHERE n.name = ? ORDER BY a.size DESC LIMIT 4",
                        (name,),
                    ):
                        if article in seen:
                            continue
                        seen.add(article)
                        named = [t for t in gram if t not in FILLER]
                        rest = [t for t in content if t not in gram and t not in ROMAN]
                        exact = plain_name(title) == name and "(" not in title  # (its own article, not a namesake's)
                        found.append(self._in_article(title, first, last, size, source, named, rest, count, exact, who))
            if found and n > 1:
                break  # (the longest names first: "New York City", not "York")
        return found

    def _in_article(self, title, first, last, size, source, named, rest, count, exact, who) -> Found:
        lead = self._sentences(first, last)
        base = 3.0 * len(named) + 0.3 * math.log10(1 + size) + (1.0 if exact else 0.0)
        if who and lead and encyclopedia.PERSON.search(lead[0][:200]):
            base += 2.0  # (asked who: a person)
        if not rest:  # just what it is: the start of the article
            return Found(title, lead, 0, min(count, len(lead)), 10 + base, source)
        weights = {t: self._weight(t) for t in set(rest)}
        has: dict[int, float] = {}
        for term, weight in weights.items():
            for (row,) in self.db.execute(
                "SELECT rowid FROM sentences WHERE sentences MATCH ? AND rowid BETWEEN ? AND ? LIMIT 400",
                (self._query(term), first, last),
            ):
                has[row] = has.get(row, 0.0) + weight
        if not has:  # nothing it read about it says that: what it is, at least (less likely the answer)
            return Found(title, lead, 0, 1, 3 + base, source)
        row = min(has, key=lambda r: (-has[r], r))
        covered = has[row] / sum(weights.values())
        (sentence,) = self.db.execute("SELECT text FROM sentences WHERE rowid = ?", (row,)).fetchone()
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
            rows = self.db.execute(
                "SELECT rowid, text, article, bm25(sentences) FROM sentences WHERE sentences MATCH ? "
                "ORDER BY rank LIMIT 8",
                (" AND ".join(self._query(t) for t in used),),
            ).fetchall()
            if rows:
                found = []
                for row, sentence, article, rank in rows:
                    title, first, last, size_, source = self.db.execute(
                        "SELECT title, first, last, size, source FROM articles WHERE id = ?", (article,)
                    ).fetchone()
                    lead = self._sentences(first, last)
                    index = row - first
                    sentences = lead if index < len(lead) else [*lead, sentence]
                    index = min(index, len(sentences) - 1)
                    words = set(plain_name(title).split())
                    titled = sum(1 for t in used if t in words or singular(t) in words)
                    score = 6 * covered + 2 * titled - 0.05 * index + 0.3 * math.log10(1 + size_) - 0.01 * rank
                    score -= 2.0 if named else 0.0  # (an article the question names is likelier to answer it)
                    found.append(Found(title, sentences, index, 1, score, source))
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
            self._counts[word] = self.db.execute(
                "SELECT count(*) FROM sentences WHERE sentences MATCH ?", (self._term(word),)
            ).fetchone()[0]
        if self._total is None:
            self._total = (self.db.execute("SELECT max(last) FROM articles").fetchone()[0] or 0) + 1
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
