"""The valley Haven lives in: a walled world with a hill, an orchard, a pond and a meadow.

It is a 3D world on a grid. Every place has a height: the ground rises into a hill in one
corner, with a cliff on its south face, and there's a mound in the meadow. Things have
heights too: trees tower, bushes come up to its eyes, flowers and mushrooms are low. So
what Haven can see depends on where it stands: the ground rising in front of it blocks
its view, and from the hilltop it sees over everything lower. Climbing costs effort, and
cliffs can only be jumped down.

There is a lot to deal with. Berries on bushes, apples that fall from the trees (or that
it can shake down), and mushrooms, some of which make it sick. A pond to drink from,
which cools it down. Flowers to smell, a ball to push (it rolls downhill), a bell to
ring, a campfire that's lovely to sit by at night but burns if it steps in, thorns,
stones, and butterflies that won't keep still. Days are warm and nights cold; the shade
under the trees is cool, and the hilltop is windy.

The world only produces raw, noisy sensations. Everything Haven knows about it, including
what an apple is, it has to learn.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

MAP = (
    "########################",
    "#....f.................#",
    "#.N.....A..A..A..S.....#",
    "#....................f.#",
    "#...f..................#",
    "#........A..A..........#",
    "#f....B................#",
    "#......................#",
    "#........b....m........#",
    "#...B......B...........#",
    "#.....f......t.........#",
    "#...............m......#",
    "#.......B...f.....t..m.#",
    "#.................B....#",
    "#..B.............::....#",
    "#.....T...##...::WW::..#",
    "#.............:WWWWWW:f#",
    "#...T..S......:WWWWWW:.#",
    "#.S.......F...WWWWWWWW.#",
    "#............B:WWWWWW:.#",
    "#....S..T.....:WWWWWW:.#",
    "#.T............::WW::f.#",
    "#......B.......f.......#",
    "########################",
)
HEIGHTS = (  # how high the ground is, in steps: a hill in the north-east, a mound where the bell is
    "000000000000000000000000",
    "000000000000000000000000",
    "000000000000000011111100",
    "000000000000000122222100",
    "000000000000000123332100",
    "000000000000000123332100",
    "000000000000000123332100",
    "000000001110000122222100",
    "000000001110000100000100",
    "000000001110000000000000",
    *("000000000000000000000000",) * 14,
)
LAYOUT = MAP  # the name the rest of Haven (and its dashboard) knows the map by

(FLOOR, WALL, BUSH, THORN, STONE, NEST, TREE, WATER, SAND, FLOWER, MUSHROOM, TOADSTOOL, BELL, FIRE) = range(14)
BALL, APPLE, BUTTERFLY = 14, 15, 16  # things that move, or come and go
_SYMBOLS = {
    ".": FLOOR,
    "#": WALL,
    "B": BUSH,
    "T": THORN,
    "S": STONE,
    "N": NEST,
    "A": TREE,
    "W": WATER,
    ":": SAND,
    "f": FLOWER,
    "m": MUSHROOM,
    "t": TOADSTOOL,
    "b": BELL,
    "F": FIRE,
}
NAMES = {  # what people call them (Haven has to learn its own words for them, if it ever does)
    FLOOR: "grass",
    WALL: "wall",
    BUSH: "bush",
    THORN: "thorns",
    STONE: "stone",
    NEST: "nest",
    TREE: "tree",
    WATER: "water",
    SAND: "sand",
    FLOWER: "flower",
    MUSHROOM: "mushroom",
    TOADSTOOL: "toadstool",
    BELL: "bell",
    FIRE: "fire",
    BALL: "ball",
    APPLE: "apple",
    BUTTERFLY: "butterfly",
}
BLOCKING = (WALL, BUSH, STONE, TREE, WATER, BELL)
GROUND = (FLOOR, SAND)  # what its eyes pass over
LEVEL = 0.6  # the height of one step of the ground
EYE = 0.5  # how high its eyes are above the ground it's on
TALL = {
    WALL: 2.4,
    BUSH: 0.9,
    THORN: 0.5,
    STONE: 0.6,
    NEST: 0.35,
    TREE: 3.2,
    WATER: 0.05,
    FLOWER: 0.35,
    MUSHROOM: 0.25,
    TOADSTOOL: 0.3,
    BELL: 1.6,
    FIRE: 0.7,
    BALL: 0.45,
    APPLE: 0.15,
    BUTTERFLY: 0.7,
}
TALLEST = 3.2

COLORS = {
    FLOOR: (0.35, 0.62, 0.33),
    WALL: (0.28, 0.27, 0.33),
    THORN: (0.62, 0.22, 0.78),
    STONE: (0.62, 0.62, 0.6),
    NEST: (0.86, 0.74, 0.3),  # straw
    TREE: (0.18, 0.46, 0.2),
    WATER: (0.2, 0.45, 0.85),
    SAND: (0.86, 0.79, 0.56),
    MUSHROOM: (0.6, 0.42, 0.25),
    TOADSTOOL: (0.9, 0.3, 0.45),  # red with white spots
    BELL: (0.85, 0.66, 0.16),  # brass
    FIRE: (1.0, 0.55, 0.12),
    BALL: (0.1, 0.8, 0.8),
    APPLE: (0.88, 0.14, 0.12),
    BUTTERFLY: (0.98, 0.9, 0.25),
}
FLOWER_COLORS = ((0.95, 0.5, 0.75), (0.98, 0.85, 0.2), (0.55, 0.45, 0.95))
BERRIES = (0.92, 0.16, 0.2)  # a bush with berries on it
BARE_BUSH = (0.12, 0.4, 0.18)
SLOPE = (0.28, 0.52, 0.26)  # grass rising up in front of it
# Eight compass directions; Haven faces one of the four main ones and sees along five of them.
DIRECTIONS = ((0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))
RAYS = (-2, -1, 0, 1, 2)  # 90° left, 45° left, ahead, 45° right, 90° right
RAY_RANGE = 6
DAY = 1200  # ticks in a day
MAX_BERRIES = 3
REGROW = 160  # ticks for a bush to grow back one berry
MAX_FRUIT = 3  # apples on a tree
RIPEN = 260  # ticks for a tree to grow another apple
FALLEN = 6  # at most this many apples lying on the ground
MUSHROOMS_REGROW = 500
FOOD = {BUSH: 0.3, APPLE: 0.4, MUSHROOM: 0.15, TOADSTOOL: 0.05}  # energy from eating each
SICKNESS = 0.18  # what a toadstool does to its health
SUNNY = (range(3, 13), range(7, 14))  # the meadow: warm in the sun
BALL_START = (13, 13)
BUTTERFLIES_START = ((5, 2), (12, 11), (21, 15))
ACTIONS = ("forward", "left", "right", "eat", "use", "rest", "speak")


@dataclass
class Senses:
    """One moment of raw sensation, before any interpretation."""

    light: float
    colors: np.ndarray  # (5, 3): the color seen along each ray
    heights: np.ndarray  # (5,): how tall each thing it sees is, 0 (flat) to 1 (as tall as a tree)
    nearness: np.ndarray  # (5,): 1 for something right there, 0 for nothing within range
    scent: float
    warmth: float
    bump: float
    pain: float
    touch: float  # a gentle touch from the person
    fed: float  # the person gave it food
    sound: float = 0.0  # a ringing sound
    words: list[str] = field(default_factory=list)


@dataclass
class Outcome:
    moved: bool = False
    turned: bool = False
    bumped: bool = False
    ate: bool = False
    food: float = 0.0  # energy in what it ate
    pain: float = 0.0
    sick: float = 0.0  # what something it ate did to its health
    drank: bool = False
    rang: bool = False
    pushed: bool = False
    shook: bool = False
    smelled: bool = False
    warmed: bool = False
    climbed: bool = False


class World:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.grid = np.array([[_SYMBOLS[c] for c in row] for row in MAP])
        self.ground = np.array([[int(c) for c in row] for row in HEIGHTS])
        self.height, self.width = self.grid.shape
        self.berries = {(int(x), int(y)): MAX_BERRIES for y, x in zip(*np.nonzero(self.grid == BUSH))}
        self.growth = dict.fromkeys(self.berries, 0)
        self.fruit = {(int(x), int(y)): MAX_FRUIT for y, x in zip(*np.nonzero(self.grid == TREE))}
        self.ripening = dict.fromkeys(self.fruit, 0)
        self.apples: list[tuple[int, int]] = []
        caps = np.nonzero(np.isin(self.grid, (MUSHROOM, TOADSTOOL)))
        self.mushrooms = {(int(x), int(y)): 0 for y, x in zip(*caps)}  # ticks until it has grown back (0: it's there)
        self.ball = BALL_START
        self.butterflies = list(BUTTERFLIES_START)
        self.homes = list(BUTTERFLIES_START)  # the flowers they keep near
        self.rang = -(10**9)  # when the bell last rang
        self.nest = next((int(x), int(y)) for y, x in zip(*np.nonzero(self.grid == NEST)))
        self.x, self.y = self.nest
        self.heading = 2  # east
        self.tick = int(DAY * 0.3)  # a morning
        self.pain = 0.0
        self.bumped = False
        self._sound = 0.0
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

    def level(self, x: int, y: int) -> int:
        return int(self.ground[y, x]) if self.inside(x, y) else 0

    def inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def ambient(self, x: int | None = None, y: int | None = None) -> float:
        """How warm it is at a place: sun and night, shade, the fire, the pond, the wind up high, the nest."""
        x, y = (self.x, self.y) if x is None else (x, y)
        light = self.light
        warmth = 0.26 + 0.34 * light  # cold nights, warm days
        if x in SUNNY[0] and y in SUNNY[1] and light > 0.5:
            warmth += 0.12 * light
        near = self.grid[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]
        if light > 0.5 and (near == TREE).any():
            warmth -= 0.12 * light  # shade
        if (near == WATER).any():
            warmth -= 0.04  # a cool breeze off the pond
        warmth -= 0.05 * self.level(x, y)  # windy up high
        fire = self._nearest(x, y, FIRE)
        warmth += {0: 0.3, 1: 0.24, 2: 0.1}.get(fire, 0.0)
        if self.grid[y, x] == NEST:
            warmth += 0.25
        return float(np.clip(warmth, 0.0, 1.0))

    def _nearest(self, x: int, y: int, kind: int, reach: int = 2) -> int | None:
        """How many steps away the nearest thing of a kind is (None if it's further than `reach`)."""
        for r in range(reach + 1):
            box = self.grid[max(0, y - r) : y + r + 1, max(0, x - r) : x + r + 1]
            if (box == kind).any():
                return r
        return None

    # --- what the person does -----------------------------------------------

    def say(self, text: str) -> list[str]:
        words = re.findall(r"[a-z']+", text.lower())
        self._heard += words
        return words

    def touch(self) -> None:
        self._touched = True

    def feed(self) -> None:
        self._fed = True

    # --- what's where ---------------------------------------------------------

    def thing(self, x: int, y: int) -> int:
        """What's at a place, as seen from outside it: a butterfly, the ball, an apple, or what's on the ground."""
        if not self.inside(x, y):
            return WALL
        if (x, y) in self.butterflies:
            return BUTTERFLY
        if (x, y) == self.ball:
            return BALL
        if (x, y) in self.apples:
            return APPLE
        cell = int(self.grid[y, x])
        if cell in (MUSHROOM, TOADSTOOL) and self.mushrooms[(x, y)]:
            return FLOOR  # eaten, and not grown back yet
        return cell

    def _cell(self, x: int, y: int) -> int:
        """What's on the ground at a place (not counting things that move)."""
        return int(self.grid[y, x]) if self.inside(x, y) else WALL

    def free(self, x: int, y: int) -> bool:
        """Whether something could be put down there: open ground, with nothing else on it."""
        return (
            self.inside(x, y)
            and self.grid[y, x] in (FLOOR, SAND)
            and (x, y) != self.ball
            and (x, y) not in self.apples
            and (x, y) != (self.x, self.y)
        )

    # --- sensing --------------------------------------------------------------

    def sense(self) -> Senses:
        light = self.light
        colors = np.zeros((len(RAYS), 3))
        heights = np.zeros(len(RAYS))
        nearness = np.zeros(len(RAYS))
        eye = self.level(self.x, self.y) * LEVEL + EYE
        for i, turn in enumerate(RAYS):
            dx, dy = DIRECTIONS[(self.heading + turn) % 8]
            color, distance, tall, glowing = COLORS[FLOOR], RAY_RANGE + 1, 0.0, False
            for step in range(1, RAY_RANGE + 1):
                cx, cy = self.x + dx * step, self.y + dy * step
                thing = self.thing(cx, cy)
                if thing not in GROUND:
                    color, distance, tall, glowing = self._color(cx, cy, thing), step, TALL[thing], thing == FIRE
                    break
                rise = self.level(cx, cy) * LEVEL - eye
                if rise > 0:  # the ground rises above its eyes: a slope it can't see past
                    color, distance, tall = SLOPE, step, rise
                    break
            brightness = 1.0 if glowing else 0.2 + 0.8 * light  # the fire gives its own light
            dark = 0.0 if glowing else (1 - light) ** 2
            noise = 0.01 + 0.3 * dark + 0.03 * (min(distance, RAY_RANGE) - 1) / RAY_RANGE
            seen = np.array(color) * brightness + self.rng.normal(0, noise, 3)
            colors[i] = np.clip(seen, 0.0, 1.0)
            heights[i] = np.clip(tall / TALLEST + self.rng.normal(0, 0.02 + 0.1 * dark), 0.0, 1.0)
            nearness[i] = max(0.0, 1 - (distance - 1) / RAY_RANGE)
        heard, self._heard = self._heard, []
        senses = Senses(
            light=float(np.clip(light + self.rng.normal(0, 0.03), 0, 1)),
            colors=colors,
            heights=heights,
            nearness=nearness,
            scent=float(np.clip(self._scent() / 3 + self.rng.normal(0, 0.01), 0, 1)),
            warmth=float(np.clip(self.ambient() + self.rng.normal(0, 0.01), 0, 1)),
            bump=float(self.bumped),
            pain=self.pain,
            touch=float(self._touched),
            fed=float(self._fed),
            sound=self._sound,
            words=heard,
        )
        self._touched = self._fed = False
        self._sound = 0.0
        return senses

    def _scent(self) -> float:
        """Sweet smells: ripe berries and fallen apples most, flowers a little."""
        here = (self.x, self.y)
        total = sum(n * math.exp(-math.dist(here, spot) / 2.5) for spot, n in self.berries.items())
        total += sum(1.5 * math.exp(-math.dist(here, spot) / 2.5) for spot in self.apples)
        flowers = np.nonzero(self.grid == FLOWER)
        total += sum(0.15 * math.exp(-math.dist(here, (int(fx), int(fy))) / 2.0) for fy, fx in zip(*flowers))
        return total

    def _color(self, x: int, y: int, thing: int) -> tuple[float, float, float]:
        if thing == BUSH:
            return BERRIES if self.berries[(x, y)] else BARE_BUSH
        if thing == FLOWER:
            return FLOWER_COLORS[(x + y) % len(FLOWER_COLORS)]
        return COLORS[thing]

    def ahead(self) -> tuple[int, int]:
        dx, dy = DIRECTIONS[self.heading]
        return self.x + dx, self.y + dy

    # --- acting -----------------------------------------------------------------

    def act(self, action: str) -> Outcome:
        outcome = Outcome()
        if action == "forward":
            self._step(outcome)
        elif action in ("left", "right"):
            self.heading = (self.heading + (-2 if action == "left" else 2)) % 8
            outcome.turned = True
        elif action == "eat":
            self._eat(outcome)
        elif action == "use":
            self._use(outcome)
        here = int(self.grid[self.y, self.x])
        if here == THORN:
            outcome.pain = max(outcome.pain, 0.8 if outcome.moved else 0.3)  # stepping on thorns hurts
        elif here == FIRE:
            outcome.pain = max(outcome.pain, 0.9 if outcome.moved else 0.6)  # and fire burns
        self.bumped, self.pain = outcome.bumped, outcome.pain
        self._grow()
        self.tick += 1
        return outcome

    def _step(self, outcome: Outcome) -> None:
        nx, ny = self.ahead()
        rise = self.level(nx, ny) - self.level(self.x, self.y)
        if (nx, ny) == self.ball:  # it pushes the ball along, if there's room
            dx, dy = DIRECTIONS[self.heading]
            if not self._roll(dx, dy, 1):
                outcome.bumped = True
                return
            outcome.pushed = True
        if self._cell(nx, ny) in BLOCKING or rise > 1:  # a cliff can only be jumped down
            outcome.bumped = True
            return
        self.x, self.y = nx, ny
        outcome.moved = True
        outcome.climbed = rise > 0

    def _eat(self, outcome: Outcome) -> None:
        spot = self.ahead()
        thing = self.thing(*spot)
        if thing == BUSH and self.berries[spot] > 0:
            self.berries[spot] -= 1
        elif thing == APPLE:
            self.apples.remove(spot)
        elif thing in (MUSHROOM, TOADSTOOL):
            self.mushrooms[spot] = MUSHROOMS_REGROW
            outcome.sick = SICKNESS if thing == TOADSTOOL else 0.0
        else:
            if thing == BUTTERFLY:
                self._flutter(self.butterflies.index(spot), away=True)
            return
        outcome.ate = True
        outcome.food = FOOD[thing]

    def _use(self, outcome: Outcome) -> None:
        """Doing something with what's in front of it: drinking, shaking, ringing, sniffing, kicking."""
        spot = self.ahead()
        thing = self.thing(*spot)
        if thing == WATER:
            outcome.drank = True
        elif thing == TREE:
            if self.fruit[spot] > 0 and self._drop_apple(spot):
                self.fruit[spot] -= 1
                outcome.shook = True  # and an apple fell
        elif thing == BELL:
            outcome.rang = True
            self.rang = self.tick
            self._sound = 1.0
        elif thing == FLOWER:
            outcome.smelled = True
        elif thing == BALL:
            dx, dy = DIRECTIONS[self.heading]
            outcome.pushed = self._roll(dx, dy, 3)
        elif thing == FIRE:
            outcome.warmed = True
        elif thing == THORN:
            outcome.pain = 0.3  # pricked

    def _roll(self, dx: int, dy: int, cells: int) -> bool:
        """Moves the ball up to `cells` along a direction. It goes up a step if pushed, not up a cliff."""
        moved = False
        for _ in range(cells):
            bx, by = self.ball
            nx, ny = bx + dx, by + dy
            if not self.free(nx, ny) or self.level(nx, ny) > self.level(bx, by) + 1:
                break
            self.ball, moved = (nx, ny), True
        return moved

    def _drop_apple(self, tree: tuple[int, int]) -> bool:
        spots = [(tree[0] + dx, tree[1] + dy) for dx, dy in DIRECTIONS]
        spots = [s for s in spots if self.free(*s)]
        if not spots or len(self.apples) >= FALLEN:
            return False
        self.apples.append(spots[int(self.rng.integers(len(spots)))])
        return True

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
        for spot, count in self.fruit.items():
            if count < MAX_FRUIT:
                self.ripening[spot] += 1
                if self.ripening[spot] >= RIPEN:
                    self.fruit[spot] += 1
                    self.ripening[spot] = int(self.rng.integers(0, RIPEN // 4))
            elif self.rng.random() < 1 / 350 and self._drop_apple(spot):  # ripe apples fall by themselves
                self.fruit[spot] -= 1
        for spot, wait in self.mushrooms.items():
            if wait:
                self.mushrooms[spot] = wait - 1
        if self.tick % 4 == 0:  # the ball rolls down slopes
            bx, by = self.ball
            lower = [(dx, dy) for dx, dy in DIRECTIONS[::2] if self.level(bx + dx, by + dy) < self.level(bx, by)]
            if lower:
                self._roll(*lower[0], 1)
        for i in range(len(self.butterflies)):
            if self.rng.random() < 0.25:
                self._flutter(i)

    def _flutter(self, i: int, away: bool = False) -> None:
        """A butterfly flits to a nearby spot, staying near its flowers."""
        x, y = self.butterflies[i]
        hx, hy = self.homes[i]
        for _ in range(4):
            dx, dy = DIRECTIONS[int(self.rng.integers(8))]
            nx, ny = x + dx * (2 if away else 1), y + dy * (2 if away else 1)
            if (
                self.inside(nx, ny)
                and self.grid[ny, nx] not in (WALL, TREE, BELL)
                and max(abs(nx - hx), abs(ny - hy)) <= 4
                and (nx, ny) != (self.x, self.y)
            ):
                self.butterflies[i] = (nx, ny)
                return

    # --- saving -------------------------------------------------------------------

    def state(self) -> dict:
        return {
            "tick": self.tick,
            "x": self.x,
            "y": self.y,
            "heading": self.heading,
            "berries": [[x, y, n] for (x, y), n in self.berries.items()],
            "growth": [[x, y, n] for (x, y), n in self.growth.items()],
            "fruit": [[x, y, n] for (x, y), n in self.fruit.items()],
            "ripening": [[x, y, n] for (x, y), n in self.ripening.items()],
            "apples": [list(a) for a in self.apples],
            "mushrooms": [[x, y, n] for (x, y), n in self.mushrooms.items()],
            "ball": list(self.ball),
            "butterflies": [list(b) for b in self.butterflies],
            "rang": self.rang,
        }

    def load(self, state: dict) -> None:
        self.tick, self.x, self.y, self.heading = state["tick"], state["x"], state["y"], state["heading"]
        self.berries = {(x, y): n for x, y, n in state["berries"]}
        self.growth = {(x, y): n for x, y, n in state["growth"]}
        self.fruit = {(x, y): n for x, y, n in state["fruit"]}
        self.ripening = {(x, y): n for x, y, n in state["ripening"]}
        self.apples = [tuple(a) for a in state["apples"]]
        self.mushrooms = {(x, y): n for x, y, n in state["mushrooms"]}
        self.ball = tuple(state["ball"])
        self.butterflies = [tuple(b) for b in state["butterflies"]]
        self.rang = int(state["rang"])
