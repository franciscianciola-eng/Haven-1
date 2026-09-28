"""The global workspace: where one thing at a time comes to the fore and is shared with the whole mind.

Specialist modules work in parallel, each on its own kind of information (GWT-1). They
compete to put their content into a workspace that holds one item at a time (GWT-2).
The winner "ignites": it is sustained, and broadcast to every module at once (GWT-3).
What is in the workspace, and what Haven is trying to do, biases what gets in next, so
the workspace can query modules one after another (GWT-4). This is the architecture of
Baars's and Dehaene's global workspace theories.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SOURCES = ("vision", "smell", "touch", "body", "hearing", "memory", "imagination", "thought")
Q = 16  # size of the quality part of a content

# Layout of the broadcast vector.
SOURCE = slice(0, len(SOURCES))
QUALITY = slice(SOURCE.stop, SOURCE.stop + Q)
WHERE = slice(QUALITY.stop, QUALITY.stop + 2)  # (bearing from -1 left to 1 right, nearness)
DRIVES = slice(WHERE.stop, WHERE.stop + 4)  # the body's needs, always in the background
AFFECT = slice(DRIVES.stop, DRIVES.stop + 2)  # (valence, arousal)
CONFIDENCE, VALUE, NOVELTY, AGENCY = range(AFFECT.stop, AFFECT.stop + 4)
D = AGENCY + 1

IGNITION = 0.35  # how strong a content must be to enter the workspace
DECAY = 0.9  # how fast a content fades unless something keeps it going
HABITUATION = 15  # ticks over which an unchanging content loses its hold


@dataclass
class Candidate:
    """A content that a module offers to the workspace."""

    source: str
    quality: np.ndarray
    salience: float
    where: tuple[float, float] = (0.0, 0.0)
    confidence: float = 1.0
    value: float = 0.0
    novelty: float = 0.0
    agency: float = 0.0
    kind: int = -1
    label: str = ""  # a few words about it, for readouts
    key: tuple = ()  # identifies "the same thing" from one moment to the next
    extra: dict = field(default_factory=dict)

    def vector(self, drives: np.ndarray, affect: np.ndarray) -> np.ndarray:
        v = np.zeros(D)
        v[SOURCES.index(self.source)] = 1.0
        quality = np.asarray(self.quality, dtype=float)[:Q]
        v[QUALITY.start : QUALITY.start + len(quality)] = quality
        v[WHERE] = self.where
        v[DRIVES] = drives
        v[AFFECT] = affect
        v[CONFIDENCE], v[VALUE], v[NOVELTY], v[AGENCY] = self.confidence, self.value, self.novelty, self.agency
        return v


class Workspace:
    def __init__(self):
        self.content: Candidate | None = None
        self.previous: tuple[Candidate, np.ndarray] | None = None  # what the last ignition displaced
        self.vector = np.zeros(D)
        self.strength = 0.0
        self.dwell = 0  # ticks the current content has held the workspace
        self.trace = np.zeros(D)  # a fading echo of recent contents: what "just happened"
        self.ignitions = 0
        self.competitions = 0
        self.entrants = 0  # candidates that competed
        self.history: list[tuple[int, str, str]] = []  # (tick, source, label) of recent ignitions

    def compete(
        self, candidates: list[Candidate], scores: list[float], inertia: float, tick: int
    ) -> tuple[Candidate | None, bool]:
        """Let the candidates compete. Returns the workspace content and whether a new one ignited."""
        self.competitions += 1
        self.entrants += len(candidates)
        self.strength *= DECAY
        current = self.content
        best, best_score = None, -np.inf
        for candidate, score in zip(candidates, scores, strict=True):
            if current is not None and candidate.key == current.key:
                # The same thing is still there: it stays vivid while it's perceived, but habituates.
                self.content = current = candidate
                self.strength = max(self.strength, (score + inertia) * np.exp(-self.dwell / HABITUATION))
            elif score > best_score:
                best, best_score = candidate, score
        ignited = best is not None and best_score > IGNITION and best_score > self.strength
        if ignited:
            if self.content is not None:
                self.previous = (self.content, self.vector.copy())
            self.content, self.strength, self.dwell = best, best_score, 0
            self.ignitions += 1
            self.history = [*self.history[-39:], (tick, best.source, best.label)]
        else:
            self.dwell += 1
            if self.strength < 0.1:
                self.content = None  # nothing is in mind: the mind can wander
        return self.content, ignited

    def broadcast(self, drives: np.ndarray, affect: np.ndarray) -> np.ndarray:
        if self.content is None:
            self.vector = np.zeros(D)
            self.vector[DRIVES] = drives
            self.vector[AFFECT] = affect
        else:
            self.vector = self.content.vector(drives, affect) * min(1.0, 0.5 + self.strength)
            self.vector[DRIVES], self.vector[AFFECT] = drives, affect
        self.trace = 0.9 * self.trace + 0.1 * self.vector
        return self.vector

    def to_state(self) -> dict:
        return {
            "strength": self.strength,
            "dwell": self.dwell,
            "trace": self.trace,
            "ignitions": self.ignitions,
            "competitions": self.competitions,
            "entrants": self.entrants,
            "history": [list(item) for item in self.history],
        }

    def load_state(self, state: dict) -> None:
        self.strength, self.dwell = float(state["strength"]), int(state["dwell"])
        self.trace = np.array(state["trace"], dtype=float)
        self.ignitions, self.competitions = int(state["ignitions"]), int(state["competitions"])
        self.entrants = int(state["entrants"])
        self.history = [tuple(item) for item in state["history"]]
