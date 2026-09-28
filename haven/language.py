"""Words, learned the way a small child learns them: by hearing them while something is in mind.

A word it attends to opens a short window. Whatever passes through its workspace in that
window is linked to the word, so over many hearings a word comes to mean what reliably
goes with it: a kind of thing, a feeling, a need. Later, hearing the word brings its
meaning to mind, and when something is in mind that it has a word for, it can say it.
It only learns words it actually attends to.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np

from .workspace import DRIVES, QUALITY, D, Q

WINDOW = 12  # ticks after hearing a word during which what's in mind is linked to it
SYLLABLES = ("ba", "ma", "da", "na", "pa", "mi", "bi", "nu", "la", "wa", "ga", "yi")


def sound(word: str) -> np.ndarray:
    """What a word sounds like, as a code: words that share letter groups sound alike."""
    code = np.zeros(Q)
    padded = f"#{word}#"
    for i in range(len(padded) - 2):
        h = zlib.crc32(padded[i : i + 3].encode())
        code[h % Q] += 1.0 if (h >> 8) & 1 else -1.0
    norm = np.linalg.norm(code)
    return code / norm if norm else code


@dataclass
class Word:
    form: str
    meaning: np.ndarray = field(default_factory=lambda: np.zeros(D))
    heard: int = 0
    said: int = 0
    first: int = 0
    kinds: dict[int, float] = field(default_factory=dict)
    feeling: np.ndarray = field(default_factory=lambda: np.zeros(4))  # the needs it goes with
    learned: int = 0  # when it first understood it


class Lexicon:
    def __init__(self):
        self.words: dict[str, Word] = {}
        self.pending: list[dict] = []
        self.background = np.zeros(D)  # the average content, so meanings stand out against it
        self.kind_freq: dict[int, float] = {}
        self.drive_mean = np.zeros(4)

    def hear(self, form: str, tick: int, referent: tuple[np.ndarray, str, int] | None = None) -> Word:
        """It attended to a word: open a window to link it with what's in mind.

        The referent is what it was attending to when the word came: people mostly name what
        the listener is already looking at (joint attention), so that counts the most.
        """
        word = self.words.get(form)
        if word is None:
            word = self.words[form] = Word(form, first=tick)
        word.heard += 1
        window = {"form": form, "until": tick + WINDOW, "sum": np.zeros(D), "n": 0, "kinds": {}, "referent": None}
        if referent is not None:
            vector, source, kind = referent
            if source != "hearing":
                window["sum"] += 3 * vector
                window["n"] += 3
                if source == "vision" and kind >= 0:
                    window["referent"] = kind
        self.pending.append(window)
        return word

    def attend(self, broadcast: np.ndarray, source: str | None, kind: int, drives: np.ndarray, tick: int) -> list[str]:
        """Take in this moment's content. Returns words whose meaning just got clear enough to count as known."""
        self.background = 0.995 * self.background + 0.005 * broadcast
        self.drive_mean = 0.995 * self.drive_mean + 0.005 * drives
        if kind >= 0 and source == "vision":
            self.kind_freq[kind] = self.kind_freq.get(kind, 0.0) + 1
        learned = []
        still = []
        for window in self.pending:
            if source not in (None, "hearing"):
                window["sum"] += broadcast
                window["n"] += 1
                if kind >= 0 and source == "vision":
                    window["kinds"][kind] = window["kinds"].get(kind, 0.0) + 1
            if tick < window["until"]:
                still.append(window)
                continue
            word = self.words[window["form"]]
            if window["n"]:
                seen = window["sum"] / window["n"]
                rate = 1 / min(word.heard, 8)
                word.meaning = (1 - rate) * word.meaning + rate * seen
                word.feeling = (1 - rate) * word.feeling + rate * seen[DRIVES]
                # What it was attending to when the word came gets most of the credit.
                referent = window["referent"]
                rest = 1.0 if referent is None else 0.3
                if referent is not None:
                    word.kinds[referent] = word.kinds.get(referent, 0.0) + 0.7
                total = sum(window["kinds"].values())
                for k, count in window["kinds"].items():
                    word.kinds[k] = word.kinds.get(k, 0.0) + rest * count / total
                if not total and referent is not None:
                    word.kinds[referent] += 0.3
            if not word.learned and self.known(word.form):
                word.learned = tick
                learned.append(word.form)
        self.pending = still
        return learned

    def known(self, form: str) -> bool:
        word = self.words.get(form)
        return bool(word and word.heard >= 3 and (self.kind_of(form) is not None or self.need_of(form) is not None))

    def kind_of(self, form: str) -> int | None:
        """The kind of thing a word names, if its hearings consistently went with one."""
        word = self.words.get(form)
        if not word or not word.kinds or word.heard < 3:
            return None
        k, count = max(word.kinds.items(), key=lambda item: item[1])
        share = count / max(word.heard, 1)
        total = sum(self.kind_freq.values()) or 1.0
        lift = share / (self.kind_freq.get(k, 0.0) / total + 1e-3)
        # Mostly heard with that kind, or heard with it much more than that kind is usually around.
        return k if share >= 0.5 or (share >= 0.35 and lift > 1.5) else None

    def need_of(self, form: str) -> int | None:
        """The need a word goes with (like "hungry"), if it's heard when that need is strong."""
        word = self.words.get(form)
        if not word or word.heard < 3:
            return None
        excess = word.feeling - self.drive_mean
        i = int(np.argmax(excess))
        return i if excess[i] > 0.2 and word.feeling[i] > 0.35 else None

    def name_for(self, kind: int) -> str | None:
        names = [form for form in self.words if self.kind_of(form) == kind]
        return max(names, key=lambda f: self.words[f].heard) if names else None

    def word_for_need(self, need: int) -> str | None:
        words = [form for form in self.words if self.need_of(form) == need]
        return max(words, key=lambda f: self.words[f].heard) if words else None

    def evoke(self, form: str) -> np.ndarray | None:
        """What a known word brings to mind: the quality of what it means."""
        if not self.known(form):
            return None
        meaning = self.words[form].meaning
        return meaning[QUALITY] - self.background[QUALITY]

    def express(self, source: str | None, kind: int, drives: np.ndarray, rng: np.random.Generator) -> str:
        """Something to say about what's in mind: its words if it has them, babble if not."""
        words = []
        if source == "vision" and kind >= 0:
            name = self.name_for(kind)
            if name:
                words.append(name)
        need = int(np.argmax(drives))
        if drives[need] > 0.4:
            word = self.word_for_need(need)
            if word:
                words.append(word)
        for form in words:
            self.words[form].said += 1
        if words:
            return " ".join(dict.fromkeys(words))
        heard = [w.form for w in self.words.values() if w.heard >= 2]
        if heard and rng.random() < 0.3:
            return str(rng.choice(heard))  # trying out a word it has heard
        return "".join(rng.choice(SYLLABLES, size=int(rng.integers(1, 3))))

    def vocabulary(self) -> list[str]:
        return sorted(form for form in self.words if self.known(form))

    def to_state(self) -> dict:
        return {
            "words": {
                form: {
                    "meaning": w.meaning,
                    "heard": w.heard,
                    "said": w.said,
                    "first": w.first,
                    "kinds": [[k, v] for k, v in w.kinds.items()],
                    "feeling": w.feeling,
                    "learned": w.learned,
                }
                for form, w in self.words.items()
            },
            "background": self.background,
            "kind_freq": [[k, v] for k, v in self.kind_freq.items()],
            "drive_mean": self.drive_mean,
        }

    def load_state(self, state: dict) -> None:
        self.words = {}
        for form, w in state["words"].items():
            self.words[form] = Word(
                form,
                meaning=np.array(w["meaning"], dtype=float),
                heard=int(w["heard"]),
                said=int(w["said"]),
                first=int(w["first"]),
                kinds={int(k): float(v) for k, v in w["kinds"]},
                feeling=np.array(w["feeling"], dtype=float),
                learned=int(w["learned"]),
            )
        self.background = np.array(state["background"], dtype=float)
        self.kind_freq = {int(k): float(v) for k, v in state["kind_freq"]}
        self.drive_mean = np.array(state["drive_mean"], dtype=float)
