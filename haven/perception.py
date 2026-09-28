"""Perception: specialist modules that each predict their own input and report what's new.

Each sense compares what arrives with what the world model predicted a moment ago
(predictive coding, PP-1). What it perceives is built by combining the two, trusting the
senses in proportion to how reliable they are just now (HOT-1, HOT-2). Vision settles
over a few recurrent steps between its quality code and the kinds of things Haven has
learned, the way recognition feeds back into seeing (RPT-1), and it describes the scene
as things of learned kinds at places (RPT-2). The codes are sparse and smooth (HOT-4).
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field

import numpy as np

from .nets import Prototypes, QualityCoder
from .workspace import Candidate, Q
from .world import DIRECTIONS, RAY_RANGE, RAYS, Senses

N_RAYS = len(RAYS)
VISION_FEATURES = N_RAYS * 4  # color and nearness along each ray
BODY_FEATURES = ("scent", "warmth", "pain", "bump", "touch", "fed")


@dataclass
class Observation:
    """Sensations after the first, innate stage of processing."""

    light: float
    colors: np.ndarray  # (5, 3), corrected for the light (color constancy), so noisier in the dark
    nearness: np.ndarray
    scent: float
    warmth: float
    bump: float
    pain: float
    touch: float
    fed: float
    words: list[str] = field(default_factory=list)

    @classmethod
    def of(cls, senses: Senses) -> Observation:
        brightness = 0.2 + 0.8 * senses.light
        return cls(
            light=senses.light,
            colors=np.clip(senses.colors / brightness, 0.0, 1.5),
            nearness=senses.nearness.copy(),
            scent=senses.scent,
            warmth=senses.warmth,
            bump=senses.bump,
            pain=senses.pain,
            touch=senses.touch,
            fed=senses.fed,
            words=list(senses.words),
        )

    def vision(self) -> np.ndarray:
        return np.concatenate([self.colors, self.nearness[:, None]], axis=1).ravel()

    def body(self) -> np.ndarray:
        return np.array([getattr(self, name) for name in BODY_FEATURES])


@dataclass
class Percept:
    ray: int
    bearing: float  # -1 (90° left) to 1 (90° right)
    nearness: float
    distance: int  # cells away, RAY_RANGE + 1 if nothing is in range
    cell: tuple[int, int] | None  # where it is, if something is in range
    color: np.ndarray
    code: np.ndarray
    kind: int
    typicality: float  # 1 if it looks just like its kind, lower if it's hard to make out
    error: float  # how far it differed from what was predicted
    reliability: float
    settled: float = 0.0  # how much recognition reshaped the code as it settled


class Vision:
    """Sees along five rays and learns, by itself, what kinds of things there are."""

    def __init__(self, rng: np.random.Generator):
        self.coder = QualityCoder(3, Q, rng, width=0.16, active=3)
        self.kinds = Prototypes(Q, capacity=16, radius=0.55, lr=0.04)
        self.seen = 0  # ticks of looking

    def perceive(
        self,
        obs: Observation,
        predicted: np.ndarray | None,
        reliability: np.ndarray,
        pose: tuple[int, int, int],
        learn: bool,
    ) -> list[Percept]:
        self.seen += 1
        x, y, heading = pose
        features = obs.vision().reshape(N_RAYS, 4)
        expected = None if predicted is None else predicted.reshape(N_RAYS, 4)
        percepts = []
        for i, turn in enumerate(RAYS):
            color, nearness, r = obs.colors[i], float(obs.nearness[i]), float(reliability[i])
            prior = None if expected is None else self.coder.tuning(np.clip(expected[i, :3], 0, 1.5))
            code, _ = self.coder.infer(color, prior, r)
            first = code.copy()
            for _ in range(2):  # recognition feeds back into seeing: codes are drawn toward their kind
                kind, distance = self.kinds.nearest(code)
                if kind < 0 or distance > self.kinds.radius:
                    break
                code = self.coder.sparsen(0.75 * code + 0.25 * self.kinds.centers[kind])
            if learn and r > 0.3:
                self.coder.learn(color, 0.03 * r)
            may_create = learn and r > 0.7 and self.seen > 100
            kind = self.kinds.assign(code, weight=r if learn else 0.0, may_create=may_create)
            _, distance = self.kinds.nearest(code)
            error = 0.0 if expected is None else float(np.mean((features[i] - expected[i]) ** 2))
            steps = round(RAY_RANGE + 1 - RAY_RANGE * nearness)
            cell = None
            if steps <= RAY_RANGE:
                dx, dy = DIRECTIONS[(heading + turn) % 8]
                cell = (x + dx * steps, y + dy * steps)
            percepts.append(
                Percept(
                    ray=i,
                    bearing=turn / 2,
                    nearness=nearness,
                    distance=steps,
                    cell=cell,
                    color=color,
                    code=code,
                    kind=kind,
                    typicality=float(np.exp(-distance / self.kinds.radius)) if kind >= 0 else 0.0,
                    error=error,
                    reliability=r,
                    settled=float(np.linalg.norm(code - first)),
                )
            )
        return percepts

    def novelty(self, kind: int) -> float:
        if kind < 0:
            return 1.0
        return float(1 / np.sqrt(1 + self.kinds.counts[kind] / 20))

    def to_state(self) -> dict:
        return {"coder": self.coder.to_state(), "kinds": self.kinds.to_state(), "seen": self.seen}

    def load_state(self, state: dict) -> None:
        self.coder.load_state(state["coder"])
        self.kinds.load_state(state["kinds"])
        self.seen = int(state["seen"])


def vision_candidates(percepts: list[Percept], values: np.ndarray, novelty: callable) -> list[Candidate]:
    candidates = []
    for p in percepts:
        surprise = p.reliability * float(np.tanh(p.error / 0.05))
        value = float(values[p.kind]) if p.kind >= 0 else 0.0
        fresh = novelty(p.kind)
        salience = 0.5 * surprise + 0.15 * p.nearness + 0.25 * fresh * p.reliability
        where = "ahead" if p.bearing == 0 else f"{'left' if p.bearing < 0 else 'right'}"
        distance = (
            "far off"
            if p.cell is None
            else "right there"
            if p.distance == 1
            else "close"
            if p.distance <= 2
            else "a way off"
        )
        candidates.append(
            Candidate(
                source="vision",
                quality=p.code,
                salience=salience,
                where=(p.bearing, p.nearness),
                confidence=p.reliability * (0.5 + 0.5 * p.typicality),
                value=value,
                novelty=fresh,
                kind=p.kind,
                label=f"something {color_name(p.color)} {where}, {distance}",
                key=("vision", p.cell if p.cell is not None else ("far", p.ray)),
                extra={"ray": p.ray, "cell": p.cell, "distance": p.distance, "color": color_name(p.color)},
            )
        )
    return candidates


def color_name(rgb: np.ndarray) -> str:
    """A plain-English name for a color, for people reading about what Haven sees."""
    r, g, b = (float(c) for c in np.clip(rgb, 0, 1))
    hue, lightness, saturation = colorsys.rgb_to_hls(r, g, b)
    if saturation < 0.15 or max(r, g, b) - min(r, g, b) < 0.12:
        return "pale" if lightness > 0.75 else "gray" if lightness > 0.4 else "dark gray"
    hue *= 360
    for limit, name in (
        (15, "red"),
        (50, "orange-brown"),
        (70, "yellow"),
        (165, "green"),
        (200, "teal"),
        (255, "blue"),
        (320, "purple"),
        (345, "pink"),
        (360, "red"),
    ):
        if hue < limit:
            return f"dark {name}" if name == "green" and lightness < 0.3 else name
    return "red"
