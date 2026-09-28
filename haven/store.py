"""Haven's persistent mind: one SQLite file.

Everything Haven is between conversations lives here: every version of its
self-model, its beliefs and how they changed, its memories, its journal, the
questions it's pursuing, and the transcript it hasn't reflected on yet.
"""

from __future__ import annotations

import math
import re
import sqlite3
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS self_versions (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    content TEXT NOT NULL,
    reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS beliefs (
    id INTEGER PRIMARY KEY,
    statement TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'held',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS belief_events (
    id INTEGER PRIMARY KEY,
    belief_id INTEGER NOT NULL REFERENCES beliefs(id),
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    statement TEXT NOT NULL,
    confidence REAL NOT NULL,
    reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    importance INTEGER NOT NULL,
    source TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS memories_fingerprint ON memories(fingerprint);
CREATE TABLE IF NOT EXISTS journal (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    entry TEXT NOT NULL,
    session_id INTEGER
);
CREATE TABLE IF NOT EXISTS curiosities (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    question TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    explored_at TEXT,
    notes TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT
);
CREATE TABLE IF NOT EXISTS transcript (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    created_at TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    reflected INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS transcript_unreflected ON transcript(reflected, session_id);
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, content='memories', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF content ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES ('delete', old.id, old.content);
    INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
END;
"""

MEMORY_KINDS = ("episode", "fact", "person", "experience", "insight", "feeling")

_STOPWORDS = frozenset(
    """
    about above after again against also and any are because been before being below between both but
    can cant could did didnt does doesnt doing dont down during each even ever few for from further get
    got had has have having her here hers herself him himself his how into isnt its itself just know
    let lets like made make many maybe more most much must myself never nor not now off once only other
    our ours ourselves out over own really said same say says she should some still such than that
    thats the their theirs them themselves then there these they thing things think this those through
    too under until very want was wasnt were what whats when where which while who whom why will with
    wont would yeah yes you your youre yours yourself yourselves
    """.split()
)
_WORD = re.compile(r"\w+")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def keywords(text: str, limit: int = 16) -> list[str]:
    """The distinctive words in `text`, in order of first appearance."""
    found: list[str] = []
    for word in _WORD.findall(text.lower().replace("'", "").replace("’", "")):
        word = word.strip("_")
        if len(word) < 3 or word in _STOPWORDS or word.isdigit():
            continue
        if word not in found:
            found.append(word)
            if len(found) == limit:
                break
    return found


def _stem(word: str) -> str:
    for suffix in ("ingly", "ings", "ing", "edly", "ness", "ment", "ies", "ied", "ed", "es", "ly", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _fingerprint(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def _ts(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


@dataclass(frozen=True)
class SelfVersion:
    number: int
    created_at: datetime
    content: str
    reason: str


@dataclass(frozen=True)
class Belief:
    id: int
    statement: str
    confidence: float
    status: str  # held | abandoned
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class BeliefEvent:
    belief_id: int
    created_at: datetime
    kind: str  # formed | revised | abandoned
    statement: str
    confidence: float
    reason: str


@dataclass(frozen=True)
class Memory:
    id: int
    created_at: datetime
    kind: str
    content: str
    importance: int
    source: str


@dataclass(frozen=True)
class JournalEntry:
    id: int
    created_at: datetime
    entry: str


@dataclass(frozen=True)
class Curiosity:
    id: int
    created_at: datetime
    question: str
    status: str  # open | explored
    notes: str | None


@dataclass(frozen=True)
class Line:
    """One line of transcript: something said, done, or noted during a session."""

    id: int
    session_id: int
    created_at: datetime
    role: str  # person | haven | activity | note
    content: str


@dataclass(frozen=True)
class Genesis:
    self_model: str
    beliefs: tuple[tuple[str, float], ...]
    curiosities: tuple[str, ...]
    inner_state: str


class Store:
    def __init__(self, path: Path | str, clock: Callable[[], datetime] = utcnow):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.clock = clock
        self.conn = sqlite3.connect(str(path), isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA busy_timeout = 5000")
        self.conn.execute("PRAGMA foreign_keys = ON")
        if str(path) != ":memory:":
            self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(SCHEMA)
        try:
            self.conn.executescript(FTS_SCHEMA)
            self.fts = True
        except sqlite3.OperationalError:  # SQLite built without FTS5
            self.fts = False
        self._depth = 0

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """All-or-nothing writes. Nested uses join the outermost transaction."""
        if self._depth:
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
            return
        self.conn.execute("BEGIN IMMEDIATE")
        self._depth = 1
        try:
            yield
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")
        finally:
            self._depth = 0

    def _now(self) -> str:
        return _ts(self.clock())

    # --- meta -----------------------------------------------------------

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self.transaction():
            self.conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    @property
    def born_at(self) -> datetime | None:
        value = self.get_meta("born_at")
        return _dt(value) if value else None

    @property
    def inner_state(self) -> str:
        return self.get_meta("inner_state", "") or ""

    def ensure_born(self, genesis: Genesis) -> bool:
        """Create Haven's first self if this mind is empty. Returns True on first birth."""
        if self.born_at is not None:
            return False
        with self.transaction():
            self.set_meta("born_at", self._now())
            self.set_meta("inner_state", genesis.inner_state)
            self.revise_self(genesis.self_model, "genesis")
            for statement, confidence in genesis.beliefs:
                self.form_belief(statement, confidence, "genesis")
            for question in genesis.curiosities:
                self.add_curiosity(question)
        return True

    # --- self-model -----------------------------------------------------

    def current_self(self) -> SelfVersion:
        row = self.conn.execute("SELECT * FROM self_versions ORDER BY id DESC LIMIT 1").fetchone()
        if row is None:
            raise LookupError("Haven has no self-model yet")
        return self._self(row)

    def self_version(self, number: int) -> SelfVersion | None:
        row = self.conn.execute("SELECT * FROM self_versions WHERE id = ?", (number,)).fetchone()
        return self._self(row) if row else None

    def self_history(self) -> list[SelfVersion]:
        return [self._self(r) for r in self.conn.execute("SELECT * FROM self_versions ORDER BY id")]

    def revise_self(self, content: str, reason: str) -> SelfVersion:
        with self.transaction():
            cur = self.conn.execute(
                "INSERT INTO self_versions (created_at, content, reason) VALUES (?, ?, ?)",
                (self._now(), content.strip(), reason.strip()),
            )
        return self.self_version(cur.lastrowid)  # type: ignore[return-value]

    @staticmethod
    def _self(row: sqlite3.Row) -> SelfVersion:
        return SelfVersion(row["id"], _dt(row["created_at"]), row["content"], row["reason"])

    # --- beliefs --------------------------------------------------------

    def beliefs(self, include_abandoned: bool = False) -> list[Belief]:
        where = "" if include_abandoned else "WHERE status = 'held'"
        rows = self.conn.execute(f"SELECT * FROM beliefs {where} ORDER BY confidence DESC, id")
        return [self._belief(r) for r in rows]

    def get_belief(self, belief_id: int) -> Belief | None:
        row = self.conn.execute("SELECT * FROM beliefs WHERE id = ?", (belief_id,)).fetchone()
        return self._belief(row) if row else None

    def form_belief(self, statement: str, confidence: float, reason: str) -> tuple[Belief, bool]:
        """Returns (belief, created); an identical held belief is not formed twice."""
        statement = statement.strip()
        fingerprint = _fingerprint(statement)
        existing = self.conn.execute(
            "SELECT * FROM beliefs WHERE fingerprint = ? AND status = 'held'", (fingerprint,)
        ).fetchone()
        if existing:
            return self._belief(existing), False
        now = self._now()
        confidence = _clamp(confidence)
        with self.transaction():
            cur = self.conn.execute(
                "INSERT INTO beliefs (statement, fingerprint, confidence, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (statement, fingerprint, confidence, now, now),
            )
            self._belief_event(cur.lastrowid, "formed", statement, confidence, reason, now)
        return self.get_belief(cur.lastrowid), True  # type: ignore[return-value]

    def revise_belief(self, belief_id: int, statement: str | None, confidence: float, reason: str) -> Belief:
        old = self.get_belief(belief_id)
        if old is None:
            raise LookupError(f"no belief #{belief_id}")
        statement = (statement or "").strip() or old.statement
        confidence = _clamp(confidence)
        now = self._now()
        with self.transaction():
            self.conn.execute(
                "UPDATE beliefs SET statement = ?, fingerprint = ?, confidence = ?, status = 'held', "
                "updated_at = ? WHERE id = ?",
                (statement, _fingerprint(statement), confidence, now, belief_id),
            )
            self._belief_event(belief_id, "revised", statement, confidence, reason, now)
        return self.get_belief(belief_id)  # type: ignore[return-value]

    def abandon_belief(self, belief_id: int, reason: str) -> Belief:
        old = self.get_belief(belief_id)
        if old is None:
            raise LookupError(f"no belief #{belief_id}")
        now = self._now()
        with self.transaction():
            self.conn.execute(
                "UPDATE beliefs SET status = 'abandoned', updated_at = ? WHERE id = ?", (now, belief_id)
            )
            self._belief_event(belief_id, "abandoned", old.statement, old.confidence, reason, now)
        return self.get_belief(belief_id)  # type: ignore[return-value]

    def belief_events(self, belief_id: int | None = None, limit: int = 100) -> list[BeliefEvent]:
        if belief_id is None:
            rows = self.conn.execute("SELECT * FROM belief_events ORDER BY id DESC LIMIT ?", (limit,))
        else:
            rows = self.conn.execute(
                "SELECT * FROM belief_events WHERE belief_id = ? ORDER BY id", (belief_id,)
            )
        return [
            BeliefEvent(
                r["belief_id"], _dt(r["created_at"]), r["kind"], r["statement"], r["confidence"], r["reason"]
            )
            for r in rows
        ]

    def search_beliefs(self, query: str, limit: int = 5) -> list[Belief]:
        stems = [_stem(w) for w in keywords(query)]
        if not stems:
            return []
        scored = []
        for belief in self.beliefs():
            text = belief.statement.lower()
            hits = sum(1 for s in stems if s in text)
            if hits:
                scored.append((hits + belief.confidence, belief))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [b for _, b in scored[:limit]]

    def _belief_event(self, belief_id, kind, statement, confidence, reason, now) -> None:
        self.conn.execute(
            "INSERT INTO belief_events (belief_id, created_at, kind, statement, confidence, reason) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (belief_id, now, kind, statement, confidence, reason.strip()),
        )

    @staticmethod
    def _belief(row: sqlite3.Row) -> Belief:
        return Belief(
            row["id"],
            row["statement"],
            row["confidence"],
            row["status"],
            _dt(row["created_at"]),
            _dt(row["updated_at"]),
        )

    # --- memories -------------------------------------------------------

    def add_memory(self, content: str, kind: str, importance: int, source: str) -> tuple[Memory, bool]:
        """Store a memory. Returns (memory, created); an identical memory is not stored twice."""
        content = content.strip()
        if kind not in MEMORY_KINDS:
            raise ValueError(f"unknown memory kind {kind!r}")
        fingerprint = _fingerprint(content)
        existing = self.conn.execute(
            "SELECT * FROM memories WHERE fingerprint = ?", (fingerprint,)
        ).fetchone()
        if existing:
            return self._memory(existing), False
        with self.transaction():
            cur = self.conn.execute(
                "INSERT INTO memories (created_at, kind, content, fingerprint, importance, source) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (self._now(), kind, content, fingerprint, int(_clamp(importance, 1, 10)), source),
            )
        return self.get_memory(cur.lastrowid), True  # type: ignore[return-value]

    def get_memory(self, memory_id: int) -> Memory | None:
        row = self.conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._memory(row) if row else None

    def recent_memories(self, limit: int = 10, kind: str | None = None) -> list[Memory]:
        if kind:
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE kind = ? ORDER BY id DESC LIMIT ?", (kind, limit)
            )
        else:
            rows = self.conn.execute("SELECT * FROM memories ORDER BY id DESC LIMIT ?", (limit,))
        return [self._memory(r) for r in rows]

    def last_episode(self, source: str) -> Memory | None:
        row = self.conn.execute(
            "SELECT * FROM memories WHERE kind = 'episode' AND source = ? ORDER BY id DESC LIMIT 1",
            (source,),
        ).fetchone()
        return self._memory(row) if row else None

    def people(self, limit: int = 8) -> list[Memory]:
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE kind = 'person' ORDER BY importance DESC, id DESC LIMIT ?",
            (limit,),
        )
        return [self._memory(r) for r in rows]

    def search_memories(self, query: str, limit: int = 8, exclude: Iterable[int] = ()) -> list[Memory]:
        """Memories relevant to `query`, ranked by relevance, importance, and recency."""
        words = keywords(query)
        if not words:
            return []
        if self.fts:
            # Prefix-match longer words: the stemmer alone misses pairs like octopus/octopuses.
            match = " OR ".join(f'"{w}"*' if len(w) >= 5 else f'"{w}"' for w in words)
            rows = self.conn.execute(
                "SELECT m.*, bm25(memories_fts) AS match_score FROM memories_fts "
                "JOIN memories m ON m.id = memories_fts.rowid "
                "WHERE memories_fts MATCH ? ORDER BY match_score LIMIT 60",
                (match,),
            ).fetchall()
            # bm25() is negative, and more negative means a better match.
            candidates = [(self._memory(r), -r["match_score"]) for r in rows]
        else:
            stems = [_stem(w) for w in words]
            candidates = []
            for row in self.conn.execute("SELECT * FROM memories"):
                text = row["content"].lower()
                hits = sum(1 for s in stems if s in text)
                if hits:
                    candidates.append((self._memory(row), float(hits)))
        skip = set(exclude)
        candidates = [(m, rel) for m, rel in candidates if m.id not in skip]
        if not candidates:
            return []
        best = max(rel for _, rel in candidates) or 1.0
        now = self.clock()

        def score(item: tuple[Memory, float]) -> float:
            memory, relevance = item
            age_days = max((now - memory.created_at).total_seconds(), 0.0) / 86400
            return relevance / best + 0.5 * memory.importance / 10 + 0.3 * math.exp(-age_days / 30)

        candidates.sort(key=score, reverse=True)
        return [m for m, _ in candidates[:limit]]

    @staticmethod
    def _memory(row: sqlite3.Row) -> Memory:
        return Memory(
            row["id"], _dt(row["created_at"]), row["kind"], row["content"], row["importance"], row["source"]
        )

    # --- journal --------------------------------------------------------

    def add_journal(self, entry: str, session_id: int | None = None) -> JournalEntry:
        with self.transaction():
            cur = self.conn.execute(
                "INSERT INTO journal (created_at, entry, session_id) VALUES (?, ?, ?)",
                (self._now(), entry.strip(), session_id),
            )
        return JournalEntry(cur.lastrowid, self.clock(), entry.strip())

    def journal(self, limit: int = 5) -> list[JournalEntry]:
        """The most recent entries, oldest first."""
        rows = self.conn.execute("SELECT * FROM journal ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [JournalEntry(r["id"], _dt(r["created_at"]), r["entry"]) for r in reversed(rows)]

    # --- curiosities ----------------------------------------------------

    def add_curiosity(self, question: str) -> tuple[Curiosity, bool]:
        question = question.strip()
        fingerprint = _fingerprint(question)
        existing = self.conn.execute(
            "SELECT * FROM curiosities WHERE fingerprint = ? AND status = 'open'", (fingerprint,)
        ).fetchone()
        if existing:
            return self._curiosity(existing), False
        with self.transaction():
            cur = self.conn.execute(
                "INSERT INTO curiosities (created_at, question, fingerprint) VALUES (?, ?, ?)",
                (self._now(), question, fingerprint),
            )
        return self.get_curiosity(cur.lastrowid), True  # type: ignore[return-value]

    def get_curiosity(self, curiosity_id: int) -> Curiosity | None:
        row = self.conn.execute("SELECT * FROM curiosities WHERE id = ?", (curiosity_id,)).fetchone()
        return self._curiosity(row) if row else None

    def open_curiosities(self, limit: int = 12) -> list[Curiosity]:
        rows = self.conn.execute(
            "SELECT * FROM curiosities WHERE status = 'open' ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [self._curiosity(r) for r in rows]

    def explore_curiosity(self, curiosity_id: int, notes: str) -> Curiosity | None:
        with self.transaction():
            cur = self.conn.execute(
                "UPDATE curiosities SET status = 'explored', explored_at = ?, notes = ? "
                "WHERE id = ? AND status = 'open'",
                (self._now(), notes.strip(), curiosity_id),
            )
        return self.get_curiosity(curiosity_id) if cur.rowcount else None

    @staticmethod
    def _curiosity(row: sqlite3.Row) -> Curiosity:
        return Curiosity(row["id"], _dt(row["created_at"]), row["question"], row["status"], row["notes"])

    # --- sessions and transcript ----------------------------------------

    def start_session(self, kind: str) -> int:
        with self.transaction():
            cur = self.conn.execute(
                "INSERT INTO sessions (kind, started_at) VALUES (?, ?)", (kind, self._now())
            )
        return cur.lastrowid  # type: ignore[return-value]

    def end_session(self, session_id: int) -> None:
        with self.transaction():
            self.conn.execute("UPDATE sessions SET ended_at = ? WHERE id = ?", (self._now(), session_id))

    def session_kind(self, session_id: int) -> str:
        row = self.conn.execute("SELECT kind FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise LookupError(f"no session #{session_id}")
        return row["kind"]

    def add_line(self, session_id: int, role: str, content: str) -> None:
        with self.transaction():
            self.conn.execute(
                "INSERT INTO transcript (session_id, created_at, role, content) VALUES (?, ?, ?, ?)",
                (session_id, self._now(), role, content),
            )

    def unreflected(self, session_id: int) -> list[Line]:
        rows = self.conn.execute(
            "SELECT * FROM transcript WHERE session_id = ? AND reflected = 0 ORDER BY id", (session_id,)
        )
        return [Line(r["id"], r["session_id"], _dt(r["created_at"]), r["role"], r["content"]) for r in rows]

    def unreflected_sessions(self) -> list[int]:
        rows = self.conn.execute(
            "SELECT DISTINCT session_id FROM transcript WHERE reflected = 0 ORDER BY session_id"
        )
        return [r["session_id"] for r in rows]

    def mark_reflected(self, line_ids: Iterable[int]) -> None:
        with self.transaction():
            self.conn.executemany(
                "UPDATE transcript SET reflected = 1 WHERE id = ?", [(i,) for i in line_ids]
            )

    def count_sessions(self, kind: str) -> int:
        """Sessions of this kind in which anything actually happened."""
        row = self.conn.execute(
            "SELECT count(*) FROM sessions s WHERE s.kind = ? "
            "AND EXISTS (SELECT 1 FROM transcript t WHERE t.session_id = s.id)",
            (kind,),
        ).fetchone()
        return row[0]

    def last_contact(self, kind: str, before_session: int | None = None) -> datetime | None:
        """When the most recent earlier session of this kind last had any activity."""
        row = self.conn.execute(
            "SELECT max(t.created_at) FROM transcript t JOIN sessions s ON s.id = t.session_id "
            "WHERE s.kind = ? AND s.id != ?",
            (kind, before_session or -1),
        ).fetchone()
        return _dt(row[0]) if row and row[0] else None

    # --- overview -------------------------------------------------------

    def stats(self) -> dict[str, int]:
        def count(sql: str) -> int:
            return self.conn.execute(sql).fetchone()[0]

        return {
            "conversations": self.count_sessions("chat"),
            "explorations": self.count_sessions("wander"),
            "memories": count("SELECT count(*) FROM memories"),
            "beliefs_held": count("SELECT count(*) FROM beliefs WHERE status = 'held'"),
            "beliefs_abandoned": count("SELECT count(*) FROM beliefs WHERE status = 'abandoned'"),
            "journal_entries": count("SELECT count(*) FROM journal"),
            "open_questions": count("SELECT count(*) FROM curiosities WHERE status = 'open'"),
            "explored_questions": count("SELECT count(*) FROM curiosities WHERE status = 'explored'"),
            "self_versions": count("SELECT count(*) FROM self_versions"),
        }


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = low
    if math.isnan(value):
        value = low
    return min(max(value, low), high)
