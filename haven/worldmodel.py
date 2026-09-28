"""The world model: what Haven expects to sense next, given what it's about to do.

Before each action a copy of the motor command goes to the world model, which predicts
the consequences (an efference copy). The prediction is the top-down expectation that
perception is compared against (PP-1, HOT-1). Comparing what happened with what would
have been expected had it done nothing tells it which changes it caused itself: a sense
of agency, built from modelling how its outputs change its inputs (AE-2). And running
the model without acting is imagination: trying things out in its head first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nets import MLP, RunningStat
from .perception import BODY_FEATURES, VISION_FEATURES
from .workspace import D
from .world import ACTIONS

N_IN = D + VISION_FEATURES + len(BODY_FEATURES) + 4 + len(ACTIONS)
N_OUT = VISION_FEATURES + len(BODY_FEATURES) + 4
DRIVE_SCALE = 50.0  # drive changes per tick are tiny; scale them up (and squash big ones) to predict them
PAIN = VISION_FEATURES + BODY_FEATURES.index("pain")


@dataclass
class Prediction:
    vision: np.ndarray
    body: np.ndarray
    drive_change: np.ndarray

    @property
    def pain(self) -> float:
        return float(np.clip(self.body[BODY_FEATURES.index("pain")], 0, 1))


class WorldModel:
    """Predicts the next moment's sensations and the change in its needs.

    It predicts how the view and smells will change rather than what they will be, so
    "nothing changes" is the easy default and learning goes into what actions do.
    """

    def __init__(self, rng: np.random.Generator):
        self.net = MLP([N_IN, 64, N_OUT], rng, lr=0.02, clip=5.0)
        self.error = RunningStat(0.01, 0.05, 0.001)
        self.agency = RunningStat(0.02, 0.0, 0.1)

    @staticmethod
    def inputs(broadcast: np.ndarray, vision: np.ndarray, body: np.ndarray, drives: np.ndarray, action: int):
        return np.concatenate([broadcast, vision, body, drives, np.eye(len(ACTIONS))[action]])

    @staticmethod
    def target(vision: np.ndarray, body: np.ndarray, drive_change: np.ndarray) -> np.ndarray:
        return np.concatenate([vision, body, np.tanh(drive_change * DRIVE_SCALE)])

    @staticmethod
    def _now(x: np.ndarray) -> np.ndarray:
        """The part of the input that the prediction is a change from: current view, scent, warmth."""
        base = np.zeros(N_OUT)
        base[:VISION_FEATURES] = x[D : D + VISION_FEATURES]
        base[VISION_FEATURES : VISION_FEATURES + 2] = x[D + VISION_FEATURES : D + VISION_FEATURES + 2]
        return base

    def _predict_vector(self, x: np.ndarray) -> np.ndarray:
        return self._now(x) + self.net(x)

    def predict(self, x: np.ndarray) -> Prediction:
        y = self._predict_vector(x)
        return Prediction(
            vision=y[:VISION_FEATURES],
            body=y[VISION_FEATURES : VISION_FEATURES + len(BODY_FEATURES)],
            drive_change=np.arctanh(np.clip(y[VISION_FEATURES + len(BODY_FEATURES) :], -0.99, 0.99)) / DRIVE_SCALE,
        )

    def imagine(self, x: np.ndarray, action: int) -> Prediction:
        """What would happen if it did this? The same model, run offline."""
        x = x.copy()
        x[-len(ACTIONS) :] = np.eye(len(ACTIONS))[action]
        return self.predict(x)

    def learn(self, x: np.ndarray, target: np.ndarray, weight: float = 1.0) -> float:
        # Rare, important things (pain) get more weight, the way they grab more learning in animals.
        weights = np.ones(N_OUT)
        weights[PAIN] = 4.0
        error = self.net.learn(x, target - self._now(x), weights * weight)
        self.error.update(error)
        return error

    def sense_of_agency(self, x: np.ndarray, action: int, actual: np.ndarray) -> float:
        """How much better its own action explains what happened than doing nothing would.

        Positive when it caused the change itself; about zero when the change came from
        outside (a touch, a word) or nothing changed.
        """
        if ACTIONS[action] == "rest":
            return 0.0
        with_action = self._predict_vector(x)
        rest = x.copy()
        rest[-len(ACTIONS) :] = np.eye(len(ACTIONS))[ACTIONS.index("rest")]
        without_vec = self._predict_vector(rest)
        e_action = float(np.sum((actual - with_action) ** 2))
        e_rest = float(np.sum((actual - without_vec) ** 2))
        score = (e_rest - e_action) / (e_rest + e_action + 1e-6)
        self.agency.update(score)
        return score

    def to_state(self) -> dict:
        return {"net": self.net.to_state(), "error": self.error.to_state(), "agency": self.agency.to_state()}

    def load_state(self, state: dict) -> None:
        self.net.load_state(state["net"])
        self.error.load_state(state["error"])
        self.agency.load_state(state["agency"])
