"""Speaking up: when Haven says something of its own accord, and what about.

Haven doesn't only answer. When someone is there (the app's page is open and looked at), it
notices moments that call for saying something: the person coming back, a new season,
something that just happened to it (it got burned, it rang the bell, it did what it was
asked), a need that has grown strong, and, when nothing else is going on, something it
would like to know about them, something it remembers, or something it read. What it has
to say comes to mind as a note; the words are its language cortex's, like everything else
it says. It doesn't talk over people: it waits a little after anyone has spoken, and leaves
time between the things it says (a friendly Haven speaks up more than a shy one).
"""

from __future__ import annotations

import random
import re
import time

PRESENT = 8.0  # seconds since the page was last looked at, at most, for someone to be there
AWAY = 600.0  # seconds without anyone there before their coming back is a return worth greeting
LONG_AWAY = 6 * 3600.0  # ...and a long time away
QUIET = 25.0  # seconds after anyone has said anything before it speaks up
GAP = 90.0  # seconds between the things it says of its own accord (a friendly Haven waits less)
IDLE = 120.0  # seconds of nothing said before it brings up something just because
KEEP = 150  # ticks a thing that happened stays worth mentioning
NEED = 0.65  # how strong a need has to be to mention it
TELLING = re.compile(r"^saw (spring|summer|autumn|winter) come to the valley$")


class Initiative:
    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        self.watched = -1e9  # when the page was last looked at
        self.gone_since: float | None = time.monotonic()  # when they went away (at first: they haven't come yet)
        self.back: bool | None = None  # someone came back: None, or whether they were away a long time
        self.last_words = -1e9  # when anyone last said anything (them or it)
        self.last_spoke = -1e9  # when it last spoke up
        self.pending: list[tuple[int, str, str]] = []  # (tick, kind, note): things that happened, to mention
        self.needs: dict[int, float] = {}  # when it last mentioned each need
        self.asked: set[str] = set()  # what it has asked them (each only once)
        self.remembered: set[str] = set()  # what it has brought up from memory
        self.shared: set[str] = set()  # what it has read and told them of its own accord
        self.met = False  # whether it has talked with them before (only then is their coming in a coming back)
        self.passed: str | None = None  # time went by quickly: what it has to say about that

    # --- what it notices ---------------------------------------------------------------------------

    def looked(self, now: float | None = None) -> None:
        """The page was looked at: someone is there."""
        now = time.monotonic() if now is None else now
        if now - self.watched > AWAY and self.met:
            since = self.gone_since if self.gone_since is not None else self.watched
            self.back = now - since > LONG_AWAY
        self.watched, self.gone_since = now, None

    def away_for(self, seconds: float) -> None:
        """They were last here this long ago (from the last conversation, when it wakes up)."""
        self.met = True
        self.watched = self.gone_since = time.monotonic() - max(seconds, 0.0)

    def present(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        there = now - self.watched <= PRESENT
        if not there and self.gone_since is None:
            self.gone_since = self.watched
        return there

    def talked(self, now: float | None = None) -> None:
        """Someone said something, or it answered."""
        self.last_words = time.monotonic() if now is None else now
        self.met = True

    def noticed(self, text: str, tick: int, asleep: bool, time_of_day: str) -> None:
        """Something from its log: a thing that happened worth mentioning, a new season, falling asleep, waking."""
        from .cortex.talk import SEASON_NEWS, SLEEP_NOTE, WAKE_NOTE, event_note

        season = TELLING.match(text)
        if season:
            self.pending.append((tick, "season", SEASON_NEWS[season.group(1)][0]))
        elif text.startswith("fell asleep") and time_of_day in ("dusk", "night"):
            self.pending.append((tick, "sleep", SLEEP_NOTE))
        elif text == "woke up" and time_of_day in ("dawn", "morning"):
            self.pending.append((tick, "wake", WAKE_NOTE))
        else:
            note = event_note(text)
            if note:
                self.pending.append((tick, "event", note))
        self.pending = self.pending[-6:]

    # --- what it would say now ------------------------------------------------------------------------

    def occasion(self, mind, tick: int, friendly: float, readings, now: float | None = None):
        """What it would speak up about just now, as (kind, note, what it asks): or None. `readings` gives what it
        read lately, (title, first sentence), when it's needed."""
        from .cortex.talk import back_note, need_note

        now = time.monotonic() if now is None else now
        self.pending = [p for p in self.pending if tick - p[0] <= KEEP or p[1] in ("season", "heard")]
        if not self.present(now) or now - self.last_words < QUIET:
            return None
        if self.passed is not None:
            note, self.passed, self.back = self.passed, None, None
            return "passed", note, None
        if self.back is not None and not mind.body.asleep:
            long, self.back = self.back, None
            return "back", back_note(long), None
        if now - self.last_spoke < GAP * (1.5 - friendly):
            return None
        asleep = mind.body.asleep
        for i in range(len(self.pending) - 1, -1, -1):
            _, kind, note = self.pending[i]
            if asleep and kind != "sleep":
                continue
            del self.pending[i]
            return kind, note, None
        if asleep:
            return None
        drives = mind.body.drives()
        need = int(drives.argmax())
        if drives[need] >= NEED and now - self.needs.get(need, -1e9) > 600:
            self.needs[need] = now
            return "need", need_note(need, float(drives[need]), mind.body.cold()), None
        if now - max(self.last_words, self.last_spoke) < IDLE * (1.5 - friendly) or self.rng.random() > 0.35:
            return None
        return self._something(mind, readings())

    def _something(self, mind, readings: list[tuple[str, str]]):
        """With nothing else going on: something it would like to know, something it remembers, or read."""
        from .cortex.talk import (
            ASKED_FACTS,
            ASKS,
            best_day,
            fact_key,
            memory,
            memory_note,
            question_note,
            reading_note,
            worst_day,
        )

        told = {fact_key(fact) for _, fact in mind.told}
        questions = [
            k
            for k in ASKS
            if k not in self.asked and not (k == "your name" and mind.person) and ASKED_FACTS.get(k) not in told
        ]
        if mind.person is None and "your name" in questions:
            questions = ["your name"]  # the first thing it would like to know
        pieces = [best_day(mind), worst_day(mind), memory(mind)]
        pieces += [f"You told me that {fact}." for _, fact in mind.told[-3:]]
        pieces = [p for p in pieces if p and p not in self.remembered]
        fresh = [(title, sentence) for title, sentence in readings if title not in self.shared]
        choices = [kind for kind, there in (("question", questions), ("memory", pieces), ("reading", fresh)) if there]
        if not choices:
            return None
        kind = self.rng.choice(choices)
        if kind == "question":
            key = questions[0] if questions == ["your name"] else self.rng.choice(questions)
            self.asked.add(key)
            return "question", question_note(key), key
        if kind == "memory":
            piece = self.rng.choice(pieces)
            self.remembered.add(piece)
            return "memory", memory_note(piece), None
        title, sentence = fresh[-1]
        self.shared.add(title)
        return "reading", reading_note(title, sentence), None

    def spoke(self, now: float | None = None) -> None:
        self.last_spoke = time.monotonic() if now is None else now
