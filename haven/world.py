"""The garden Haven lives in: day and night, warmth and cold, berries, thorns, and a nest.

The world only produces raw, noisy sensations. Everything Haven knows about it, including
what a berry is, it has to learn.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

LAYOUT = (
    "##############",
    "#N.....#.....#",
    "#......#..B..#",
    "#..T...#.....#",
    "#......S..T..#",
    "#.B..........#",
    "#.....T......#",
    "#.........B..#",
    "#..S.........#",
    "#.......##...#",
    "#.B.....#..T.#",
    "#....S..#..B.#",
    "#............#",
    "##############",
)
FLOOR, WALL, BUSH, THORN, STONE, NEST = range(6)
_SYMBOLS = {".": FLOOR, "#": WALL, "B": BUSH, "T": THORN, "S": STONE, "N": NEST}
BLOCKING = (WALL, BUSH, STONE)

COLORS = {
    FLOOR: (0.35, 0.62, 0.33),
    WALL: (0.28, 0.27, 0.33),
    THORN: (0.62, 0.22, 0.78),
    STONE: (0.62, 0.62, 0.6),
    NEST: (0.86, 0.74, 0.3),  # straw
}
BERRIES = (0.92, 0.16, 0.2)  # a bush with berries on it
BARE_BUSH = (0.12, 0.4, 0.18)
# Eight compass directions; Haven faces one of the four main ones and sees along five of them.
DIRECTIONS = ((0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))
RAYS = (-2, -1, 0, 1, 2)  # 90° left, 45° left, ahead, 45° right, 90° right
RAY_RANGE = 5
DAY = 1200  # ticks in a day
MAX_BERRIES = 3
REGROW = 160  # ticks for a bush to grow back one berry
SUNNY = (range(8, 13), range(1, 5))  # a warm corner in the day
ACTIONS = ("forward", "left", "right", "eat", "rest", "speak")


@dataclass
class Senses:
    """One moment of raw sensation, before any interpretation."""

    light: float
    colors: np.ndarray  # (5, 3): the color seen along each ray
    nearness: np.ndarray  # (5,): 1 for something right there, 0 for nothing within range
    scent: float
    warmth: float
    bump: float
    pain: float
    touch: float  # a gentle touch from the person
    fed: float  # the person gave it food
    words: list[str] = field(default_factory=list)


@dataclass
class Outcome:
    moved: bool = False
    turned: bool = False
    bumped: bool = False
    ate: bool = False
    pain: float = 0.0


class World:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.grid = np.array([[_SYMBOLS[c] for c in row] for row in LAYOUT])
        self.height, self.width = self.grid.shape
        self.berries = {(int(x), int(y)): MAX_BERRIES for y, x in zip(*np.nonzero(self.grid == BUSH))}
        self.growth = dict.fromkeys(self.berries, 0)
        self.nest = next((int(x), int(y)) for y, x in zip(*np.nonzero(self.grid == NEST)))
        self.x, self.y = self.nest
        self.heading = 2  # east
        self.tick = int(DAY * 0.3)  # a morning
        self.pain = 0.0
        self.bumped = False
        self._heard: list[str] = []
        self._touched = False
        self._fed = False
        self.voice: tuple[int, str] | None = None  # the last word Haven said, and when

    # --- time and weather -------------------------------------------------

    @property
    def light(self) -> float:
        phase = 2 * math.pi * (self.tick % DAY) / DAY
        return float(np.clip(0.5 - 0.55 * math.cos(phase), 0.06, 1.0))

    @property
    def day(self) -> int:
        return self.tick // DAY

    def ambient(self, x: int | None = None, y: int | None = None) -> float:
        x, y = (self.x, self.y) if x is None else (x, y)
        light = self.light
        warmth = 0.26 + 0.34 * light  # cold nights, warm days
        if x in SUNNY[0] and y in SUNNY[1] and light > 0.5:
            warmth += 0.15 * light
        if self.grid[y, x] == NEST:
            warmth += 0.25
        return float(np.clip(warmth, 0.0, 1.0))

    # --- what the person does -----------------------------------------------

    def say(self, text: str) -> list[str]:
        words = re.findall(r"[a-z']+", text.lower())
        self._heard += words
        return words

    def touch(self) -> None:
        self._touched = True

    def feed(self) -> None:
        self._fed = True

    # --- sensing --------------------------------------------------------------

    def sense(self) -> Senses:
        light = self.light
        colors = np.zeros((len(RAYS), 3))
        nearness = np.zeros(len(RAYS))
        for i, turn in enumerate(RAYS):
            dx, dy = DIRECTIONS[(self.heading + turn) % 8]
            color, distance = COLORS[FLOOR], RAY_RANGE + 1
            for step in range(1, RAY_RANGE + 1):
                cell = self._cell(self.x + dx * step, self.y + dy * step)
                if cell != FLOOR:
                    color, distance = self._color(self.x + dx * step, self.y + dy * step, cell), step
                    break
            noise = 0.01 + 0.3 * (1 - light) ** 2 + 0.03 * (min(distance, RAY_RANGE) - 1) / RAY_RANGE
            seen = np.array(color) * (0.2 + 0.8 * light) + self.rng.normal(0, noise, 3)
            colors[i] = np.clip(seen, 0.0, 1.0)
            nearness[i] = max(0.0, 1 - (distance - 1) / RAY_RANGE)
        scent = sum(count * math.exp(-math.dist((self.x, self.y), spot) / 2.5) for spot, count in self.berries.items())
        heard, self._heard = self._heard, []
        senses = Senses(
            light=float(np.clip(light + self.rng.normal(0, 0.03), 0, 1)),
            colors=colors,
            nearness=nearness,
            scent=float(np.clip(scent / 3 + self.rng.normal(0, 0.01), 0, 1)),
            warmth=float(np.clip(self.ambient() + self.rng.normal(0, 0.01), 0, 1)),
            bump=float(self.bumped),
            pain=self.pain,
            touch=float(self._touched),
            fed=float(self._fed),
            words=heard,
        )
        self._touched = self._fed = False
        return senses

    def _cell(self, x: int, y: int) -> int:
        if 0 <= x < self.width and 0 <= y < self.height:
            return int(self.grid[y, x])
        return WALL

    def _color(self, x: int, y: int, cell: int) -> tuple[float, float, float]:
        if cell == BUSH:
            return BERRIES if self.berries[(x, y)] else BARE_BUSH
        return COLORS[cell]

    def ahead(self) -> tuple[int, int]:
        dx, dy = DIRECTIONS[self.heading]
        return self.x + dx, self.y + dy

    # --- acting -----------------------------------------------------------------

    def act(self, action: str) -> Outcome:
        outcome = Outcome()
        if action == "forward":
            nx, ny = self.ahead()
            if self._cell(nx, ny) in BLOCKING:
                outcome.bumped = True
            else:
                self.x, self.y = nx, ny
                outcome.moved = True
        elif action in ("left", "right"):
            self.heading = (self.heading + (-2 if action == "left" else 2)) % 8
            outcome.turned = True
        elif action == "eat":
            spot = self.ahead()
            if self.berries.get(spot, 0) > 0:
                self.berries[spot] -= 1
                outcome.ate = True
        if self.grid[self.y, self.x] == THORN:
            outcome.pain = 0.8 if outcome.moved else 0.3  # stepping on thorns hurts; staying keeps hurting
        self.bumped, self.pain = outcome.bumped, outcome.pain
        self._grow()
        self.tick += 1
        return outcome

    def carry_home(self) -> None:
        """After Haven faints, it wakes up in its nest."""
        self.x, self.y = self.nest
        self.pain = 0.0

    def _grow(self) -> None:
        for spot, count in self.berries.items():
            if count < MAX_BERRIES:
                self.growth[spot] += 1
                if self.growth[spot] >= REGROW:
                    self.berries[spot] += 1
                    self.growth[spot] = int(self.rng.integers(0, REGROW // 4))

    # --- saving -------------------------------------------------------------------

    def state(self) -> dict:
        return {
            "tick": self.tick,
            "x": self.x,
            "y": self.y,
            "heading": self.heading,
            "berries": [[x, y, n] for (x, y), n in self.berries.items()],
            "growth": [[x, y, n] for (x, y), n in self.growth.items()],
        }

    def load(self, state: dict) -> None:
        self.tick, self.x, self.y, self.heading = state["tick"], state["x"], state["y"], state["heading"]
        self.berries = {(x, y): n for x, y, n in state["berries"]}
        self.growth = {(x, y): n for x, y, n in state["growth"]}
