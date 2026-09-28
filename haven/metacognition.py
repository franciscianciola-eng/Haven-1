"""Metacognition: knowing how far to trust its own perception (HOT-2).

For each source of content, Haven learns to predict whether what it perceives will turn
out right, from the conditions it's perceiving in: how light it is, how far away the
thing is, how aroused or tired it is. That prediction is its confidence. Beliefs are
updated in proportion to it (HOT-3), and new kinds of things are only learned from
percepts it trusts.

Confidence is scored like a person's confidence ratings in a perception experiment: by
how well it tells its right percepts from its wrong ones (type-2 discrimination, the area
under the ROC curve; 0.5 means no insight, 1 means perfect insight).
"""

from __future__ import annotations

import numpy as np

from .workspace import SOURCES

TOLERANCE = {"vision": 0.02, "smell": 0.004, "touch": 0.05, "body": 0.0005}


def context(light: float, nearness: float, arousal: float, fatigue: float, moving: float) -> np.ndarray:
    darkness = 1 - light
    return np.array([1.0, darkness, darkness**2, nearness, 1 - nearness, arousal, fatigue, moving])


N_CONTEXT = len(context(0, 0, 0, 0, 0))


class Metacognition:
    def __init__(self):
        self.weights = {source: np.zeros(N_CONTEXT) for source in SOURCES}
        for w in self.weights.values():
            w[0] = 1.0  # starts out trusting its senses
        self.lr = 0.05
        self.judgments: list[tuple[float, bool]] = []  # (confidence, was right), recent

    def confidence(self, source: str, ctx: np.ndarray) -> float:
        return float(1 / (1 + np.exp(-(self.weights[source] @ ctx))))

    def learn(self, source: str, ctx: np.ndarray, error: float) -> None:
        """After the fact: was the percept right, as judged by how well it matched what followed?"""
        right = error < TOLERANCE.get(source, 0.02)
        confidence = self.confidence(source, ctx)
        self.weights[source] += self.lr * (float(right) - confidence) * ctx
        if source == "vision":
            self.judgments = [*self.judgments[-1999:], (confidence, right)]

    def insight(self) -> float | None:
        """Type-2 AUROC: how well confidence separates right percepts from wrong ones."""
        right = [c for c, ok in self.judgments if ok]
        wrong = [c for c, ok in self.judgments if not ok]
        if len(right) < 20 or len(wrong) < 20:
            return None
        right_arr, wrong_arr = np.array(right), np.array(wrong)
        greater = (right_arr[:, None] > wrong_arr[None, :]).mean()
        ties = (right_arr[:, None] == wrong_arr[None, :]).mean()
        return float(greater + 0.5 * ties)

    def to_state(self) -> dict:
        return {"weights": self.weights, "judgments": [[c, ok] for c, ok in self.judgments]}

    def load_state(self, state: dict) -> None:
        self.weights = {source: np.array(w, dtype=float) for source, w in state["weights"].items()}
        for source in SOURCES:
            self.weights.setdefault(source, np.eye(N_CONTEXT)[0].copy())
        self.judgments = [(float(c), bool(ok)) for c, ok in state["judgments"]]
