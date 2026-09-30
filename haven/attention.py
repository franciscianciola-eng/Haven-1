"""The attention schema: Haven's simplified model of its own attention (AST-1).

Just as a body schema models the body so it can be controlled, the attention schema
models attention: what Haven is attending to, how long it will stay there, and where
it will go next (Graziano's attention schema theory). Haven uses it to steady its
attention when it's pursuing something, and it notices when its attention is grabbed
by something it didn't expect. When it reports what it's aware of, the report comes
from this model, not from the machinery itself.
"""

from __future__ import annotations

import numpy as np

from .nets import softmax
from .workspace import SOURCES

NOTHING = len(SOURCES)  # the "nothing in mind" state
N_STATES = len(SOURCES) + 1
GOALS = ("food", "warmth", "healing", "sleep", "explore", "play")


def features(focus: int, dwell: int, arousal: float, drives: np.ndarray, light: float, goal: int, heard: bool):
    return np.concatenate(
        [
            np.eye(N_STATES)[focus],
            [np.log1p(dwell) / 3, arousal, light, float(heard), 1.0],
            drives,
            np.eye(len(GOALS))[goal],
        ]
    )


N_FEATURES = N_STATES + 5 + 4 + len(GOALS)


class AttentionSchema:
    def __init__(self):
        self.W = np.zeros((N_STATES, N_FEATURES))
        self.W[:, :N_STATES] = 2.0 * np.eye(N_STATES)  # at first it expects attention to stay put
        self.lr = 0.05
        self.focus = NOTHING
        self.expected = np.full(N_STATES, 1 / N_STATES)
        self.captured = False  # attention just went somewhere it didn't expect
        self.predictions = 0
        self.hits = 0
        self.switches = 0
        self.switch_hits = 0
        self._x: np.ndarray | None = None

    def anticipate(self, x: np.ndarray) -> np.ndarray:
        """Where will attention be next moment?"""
        self._x = x
        self.expected = softmax(self.W @ x)
        return self.expected

    def observe(self, focus: int) -> None:
        """Where attention actually went: learn, and notice surprises."""
        if self._x is not None:
            self.W += self.lr * np.outer(np.eye(N_STATES)[focus] - self.expected, self._x)
            guess = int(np.argmax(self.expected))
            self.predictions += 1
            self.hits += guess == focus
            if focus != self.focus:
                # When attention moves, did it foresee where to? (Its best guess other than staying.)
                elsewhere = self.expected.copy()
                elsewhere[self.focus] = -1.0
                self.switches += 1
                self.switch_hits += int(np.argmax(elsewhere)) == focus
            self.captured = bool(focus != self.focus and self.expected[focus] < 0.2)
        self.focus = focus

    def steadiness(self, relevance: float) -> float:
        """How much to hold attention where it is: more when it matters and is likely to slip."""
        slipping = 1 - float(self.expected[self.focus])
        return 0.1 + 0.4 * relevance * slipping

    def accuracy(self) -> tuple[float, float]:
        """How often it foresaw its next focus, overall and when attention moved."""
        overall = self.hits / self.predictions if self.predictions else 0.0
        on_switch = self.switch_hits / self.switches if self.switches else 0.0
        return overall, on_switch

    def to_state(self) -> dict:
        return {
            "W": self.W,
            "focus": self.focus,
            "counts": [self.predictions, self.hits, self.switches, self.switch_hits],
            "expected": self.expected,
            "captured": self.captured,
            "features": getattr(self, "_x", None),
        }

    def load_state(self, state: dict) -> None:
        self.W = np.array(state["W"], dtype=float)
        self.focus = int(state["focus"])
        self.predictions, self.hits, self.switches, self.switch_hits = (int(c) for c in state["counts"])
        if "expected" in state:
            self.expected = np.array(state["expected"], dtype=float)
            self.captured = bool(state["captured"])
        if state.get("features") is not None:
            self._x = np.array(state["features"], dtype=float)
