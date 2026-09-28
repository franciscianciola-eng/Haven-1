"""Agency: beliefs about the world, goals that compete, plans, and learning from what happens.

Haven keeps a map of what it believes is where, built from what it has seen and weighted
by how much it trusted each look (HOT-3). It learns what each kind of thing is like by
dealing with it: which can be eaten, which block the way, which hurt, which are warm.
Its needs compete to set its goal (AE-1), a planner searches its beliefs for a way to
reach it, and an actor-critic learns from how good or bad each moment feels.
Innate reflexes get it through its first hours, the way a newborn's do.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable

import numpy as np

from .attention import GOALS
from .nets import MLP
from .world import ACTIONS, DIRECTIONS, RAY_RANGE, RAYS

UNKNOWN, OPEN = -1, -2
FACING = (0, 2, 4, 6)  # it faces north, east, south or west


class BeliefMap:
    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.kind = np.full((height, width), UNKNOWN, dtype=int)
        self.confidence = np.zeros((height, width))
        self.seen = np.full((height, width), -1, dtype=int)
        self.visits = np.zeros((height, width), dtype=int)
        self.warmth = np.full((height, width), np.nan)
        self.hurt = np.zeros((height, width))  # places where it got hurt, whatever was there
        self.updates = 0
        self.weighted = 0.0  # total confidence of the evidence it has taken in

    def inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def update(self, percepts: list, pose: tuple[int, int, int], tick: int) -> None:
        x, y, heading = pose
        for p in percepts:
            trust = p.reliability
            if trust < 0.3:
                continue  # too unsure to change its mind about anything
            dx, dy = DIRECTIONS[(heading + RAYS[p.ray]) % 8]
            for step in range(1, min(p.distance, RAY_RANGE + 1)):
                self.believe(x + dx * step, y + dy * step, OPEN, trust, tick)
            if p.cell is not None and p.kind >= 0:
                self.believe(*p.cell, p.kind, trust * (0.5 + 0.5 * p.typicality), tick)

    def believe(self, x: int, y: int, kind: int, trust: float, tick: int) -> None:
        if not self.inside(x, y):
            return
        self.updates += 1
        self.weighted += trust
        if self.kind[y, x] == kind:
            self.confidence[y, x] += trust * (1 - self.confidence[y, x])
        elif trust >= self.confidence[y, x]:
            self.kind[y, x], self.confidence[y, x] = kind, trust
        else:
            self.confidence[y, x] *= 1 - 0.5 * trust  # contrary evidence it doesn't fully trust
        self.seen[y, x] = tick

    def stand(self, x: int, y: int, warmth: float, pain: float = 0.0) -> None:
        self.visits[y, x] += 1
        self.hurt[y, x] = 0.7 * self.hurt[y, x] + 0.3 * pain if pain else 0.995 * self.hurt[y, x]
        old = self.warmth[y, x]
        self.warmth[y, x] = warmth if np.isnan(old) else 0.8 * old + 0.2 * warmth
        if self.kind[y, x] == UNKNOWN:
            self.kind[y, x], self.confidence[y, x] = OPEN, 0.5

    def known(self) -> float:
        return float(np.mean(self.kind != UNKNOWN))

    def to_state(self) -> dict:
        return {
            "kind": self.kind,
            "confidence": self.confidence,
            "seen": self.seen,
            "visits": self.visits,
            "warmth": self.warmth,
            "hurt": self.hurt,
            "counts": [self.updates, self.weighted],
        }

    def load_state(self, state: dict) -> None:
        self.kind = np.array(state["kind"], dtype=int)
        self.confidence = np.array(state["confidence"], dtype=float)
        self.seen = np.array(state["seen"], dtype=int)
        self.visits = np.array(state["visits"], dtype=int)
        self.warmth = np.array(state["warmth"], dtype=float)
        self.hurt = np.array(state["hurt"], dtype=float)
        self.updates, self.weighted = int(state["counts"][0]), float(state["counts"][1])


class KindKnowledge:
    """What Haven has found out about each kind of thing, by dealing with it."""

    FIELDS = ("eat_tries", "eaten", "walk_tries", "blocked", "stepped", "pain", "warm_sum", "valence_sum", "met")

    def __init__(self, capacity: int = 16):
        for name in self.FIELDS:
            setattr(self, name, np.zeros(capacity))

    def edible(self, k: int) -> float:
        return float((self.eaten[k] + 0.3) / (self.eat_tries[k] + 2))

    def solid(self, k: int) -> float:
        return float((self.blocked[k] + 0.2) / (self.walk_tries[k] + 1))

    def painful(self, k: int) -> float:
        return float(self.pain[k] / (self.stepped[k] + 1))

    def warm(self, k: int) -> float:
        return float(self.warm_sum[k] / self.stepped[k]) if self.stepped[k] else 0.0

    def value(self, k: int) -> float:
        return float(self.valence_sum[k] / (self.met[k] + 2))

    def values(self) -> np.ndarray:
        return self.valence_sum / (self.met + 2)

    def tried_eating(self, k: int, ate: bool) -> None:
        self.eat_tries[k] += 1
        self.eaten[k] += ate

    def tried_walking(self, k: int, blocked: bool) -> None:
        self.walk_tries[k] += 1
        self.blocked[k] += blocked

    def stood_on(self, k: int, pain: float, warmth: float) -> None:
        self.stepped[k] += 1
        self.pain[k] += pain
        self.warm_sum[k] += warmth

    def felt(self, k: int, valence: float) -> None:
        self.met[k] += 1
        self.valence_sum[k] += valence

    def verdicts(self, k: int) -> tuple:
        """What it has concluded about a kind (None where it hasn't found out yet)."""
        return (
            None if self.walk_tries[k] < 2 else self.solid(k) > 0.6,
            None if self.eat_tries[k] < 2 else self.edible(k) > 0.5,
            None if self.stepped[k] < 1 else self.painful(k) > 0.2,
        )

    def merge(self, keep: int, gone: int) -> None:
        for name in self.FIELDS:
            values = getattr(self, name)
            values[keep] += values[gone]
            values[gone] = 0.0

    def describe(self, k: int) -> list[str]:
        """What it has learned about a kind, in words, for readouts."""
        facts = []
        if self.eat_tries[k] >= 2:
            facts.append("good to eat" if self.edible(k) > 0.5 else "not something to eat")
        if self.walk_tries[k] >= 2 and self.solid(k) > 0.6:
            facts.append("in the way")
        if self.stepped[k] >= 1 and self.painful(k) > 0.2:
            facts.append("it hurts")
        if self.stepped[k] >= 3 and self.warm(k) > 0.62:
            facts.append("warm")
        return facts

    def to_state(self) -> dict:
        return {name: getattr(self, name) for name in self.FIELDS}

    def load_state(self, state: dict) -> None:
        for name in self.FIELDS:
            setattr(self, name, np.array(state[name], dtype=float))


class Planner:
    """Searches its beliefs for the cheapest way to reach a goal (turns and steps)."""

    def __init__(self):
        self.plan: list[str] = []
        self.poses: list[tuple[int, int, int]] = []
        self.goal: tuple = ()
        self.made = 0
        self.searches = 0

    def next_action(
        self,
        pose: tuple[int, int, int],
        goal: tuple,
        beliefs: BeliefMap,
        knowledge: KindKnowledge,
        tick: int,
    ) -> str | None:
        """goal is (mode, cells): mode "face" to end facing one of the cells, "enter" to stand on one."""
        mode, cells = goal
        if not cells:
            self.plan = []
            return None
        fresh = self.plan and self.goal == goal and self.poses and self.poses[0] == pose and tick - self.made < 25
        if not fresh:
            self.searches += 1
            self.goal, self.made = goal, tick
            # Which way to turn when both are as good: no built-in handedness.
            order = ("forward", "left", "right") if self.searches % 2 else ("forward", "right", "left")
            self.plan, self.poses = search(pose, mode, set(cells), beliefs, knowledge, order)
        if not self.plan:
            return None
        self.poses.pop(0)
        action = self.plan.pop(0)
        return action

    def to_state(self) -> dict:
        return {"searches": self.searches}

    def load_state(self, state: dict) -> None:
        self.searches = int(state["searches"])


def search(
    start: tuple[int, int, int],
    mode: str,
    cells: set,
    beliefs: BeliefMap,
    knowledge: KindKnowledge,
    order: tuple[str, ...] = ("forward", "left", "right"),
) -> tuple[list[str], list[tuple[int, int, int]]]:
    """Dijkstra over (x, y, heading). Returns the actions and the pose before each of them."""

    def done(state: tuple[int, int, int]) -> bool:
        x, y, h = state
        if mode == "enter":
            return (x, y) in cells
        dx, dy = DIRECTIONS[h]
        return (x + dx, y + dy) in cells

    x0, y0, h0 = start
    if h0 not in FACING:
        h0 = FACING[0]
    start = (x0, y0, h0)
    if done(start):
        return [], []
    cost = {start: 0.0}
    came: dict[tuple, tuple] = {}
    queue = [(0.0, 0, start)]
    counter = 0
    while queue:
        c, _, state = heapq.heappop(queue)
        if c > cost[state]:
            continue
        if done(state):
            actions, poses = [], []
            while state in came:
                state, action = came[state]
                actions.append(action)
                poses.append(state)
            return actions[::-1], poses[::-1]
        x, y, h = state
        for action in order:
            if action == "forward":
                dx, dy = DIRECTIONS[h]
                step = step_cost(x + dx, y + dy, beliefs, knowledge)
                if step is None:
                    continue
                nxt = (x + dx, y + dy, h)
            else:
                step, nxt = 1.0, (x, y, (h + (-2 if action == "left" else 2)) % 8)
            new = c + step
            if new < cost.get(nxt, np.inf):
                cost[nxt], came[nxt] = new, (state, action)
                counter += 1
                heapq.heappush(queue, (new, counter, nxt))
    return [], []


def step_cost(x: int, y: int, beliefs: BeliefMap, knowledge: KindKnowledge) -> float | None:
    if not beliefs.inside(x, y):
        return None
    kind = beliefs.kind[y, x]
    hurt = 25 * beliefs.hurt[y, x]
    if beliefs.visits[y, x]:
        return 1.0 + hurt  # it has stood there, so whatever it thinks is there, it can go there
    if kind == UNKNOWN:
        return 1.5 + hurt
    if kind == OPEN:
        return 1.0 + hurt
    if knowledge.solid(kind) > 0.6:
        return None
    return 1.0 + hurt + 25 * knowledge.painful(kind)


class Goals:
    """Its needs compete to decide what it's trying to do (flexible pursuit of competing goals)."""

    def __init__(self):
        self.current = "explore"
        self.since = 0
        self.switches = 0

    def urgencies(self, drives: np.ndarray, cold: bool, light: float, curiosity: float) -> dict[str, float]:
        hunger, temperature, damage, tiredness = drives
        night = light < 0.3
        return {
            "food": 1.2 * hunger,
            "warmth": temperature * (1.0 if cold else 0.6),
            "healing": 0.9 * damage,
            "sleep": tiredness + (0.45 if night else 0.0),
            "explore": 0.2 + 0.2 * curiosity,
        }

    def choose(self, urgencies: dict[str, float], tick: int) -> str:
        best = max(urgencies, key=urgencies.get)
        settled = tick - self.since > 30 or urgencies[best] > 0.6  # don't flit between goals
        if best != self.current and settled and urgencies[best] > urgencies[self.current] + 0.12:
            self.current, self.since = best, tick
            self.switches += 1
        return self.current

    def index(self) -> int:
        return GOALS.index(self.current)

    def to_state(self) -> dict:
        return {"current": self.current, "since": self.since, "switches": self.switches}

    def load_state(self, state: dict) -> None:
        self.current, self.since, self.switches = state["current"], int(state["since"]), int(state["switches"])


class ActorCritic:
    """Learns which situations are good (critic) and what to do in them (actor) from how things feel."""

    def __init__(self, n_state: int, rng: np.random.Generator):
        self.critic = MLP([n_state, 48, 1], rng, lr=0.004)
        self.actor = MLP([n_state, 48, len(ACTIONS)], rng, lr=0.003)
        self.actor.W[-1] *= 0.1  # start without strong habits
        self.td = 0.0

    def value(self, state: np.ndarray) -> float:
        return float(self.critic(state)[0])

    def preferences(self, state: np.ndarray) -> np.ndarray:
        return self.actor(state)

    def learn(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, gamma: float, policy: np.ndarray
    ) -> float:
        target = reward + gamma * self.value(next_state)
        delta = target - self.value(state)
        self.critic.learn(state, np.array([target]))
        logits = self.actor(state)
        push = np.clip(delta, -1, 1) * (np.eye(len(ACTIONS))[action] - policy)
        self.actor.learn(state, logits + push)
        self.td = 0.95 * self.td + 0.05 * abs(delta)
        return delta

    def to_state(self) -> dict:
        return {"critic": self.critic.to_state(), "actor": self.actor.to_state(), "td": self.td}

    def load_state(self, state: dict) -> None:
        self.critic.load_state(state["critic"])
        self.actor.load_state(state["actor"])
        self.td = float(state["td"])


def reflexes(
    bump: float,
    pain: float,
    ahead_near: bool,
    ahead_kind: int,
    knowledge: KindKnowledge,
    hunger: float,
    social: bool,
    tired: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Innate tendencies, like a newborn's: they get it started until it learns better."""
    bias = np.zeros(len(ACTIONS))
    a = {name: i for i, name in enumerate(ACTIONS)}
    bias[a["speak"]] -= 2.0  # it vocalizes now and then
    bias[a["rest"]] -= 0.5
    if bump:
        bias[a["left" if rng.random() < 0.5 else "right"]] += 2.0
        bias[a["forward"]] -= 2.0
    if pain > 0:
        bias[a["forward"]] += 2.5  # get off whatever hurts
    if ahead_near:
        untried = ahead_kind < 0 or knowledge.eat_tries[ahead_kind] < 2
        if hunger > 0.15 and (untried or knowledge.edible(ahead_kind) > 0.5):
            bias[a["eat"]] += 1.0 + 2.0 * hunger  # put it in its mouth
    else:
        bias[a["eat"]] -= 1.5
    if social:
        bias[a["speak"]] += 2.5  # answer when someone's there
    if tired > 0.7:
        bias[a["rest"]] += 2.0
    return bias


def goal_cells(
    goal: str,
    pose: tuple[int, int, int],
    beliefs: BeliefMap,
    knowledge: KindKnowledge,
    cold: bool,
    temperature: float,
    remembered: Callable[[str], tuple[int, int] | None],
    tick: int,
) -> tuple[str, list[tuple[int, int]]]:
    """Where it could go to pursue a goal, according to what it believes (never onto what hurts)."""
    mode, cells = _goal_cells(goal, pose, beliefs, knowledge, cold, temperature, remembered, tick)
    if mode == "enter":
        x, y, _ = pose
        cells = [c for c in cells if not hurts(c, beliefs, knowledge)]
        cells = sorted(cells, key=lambda c: abs(c[0] - x) + abs(c[1] - y))[:12]
    return mode, cells


def hurts(cell: tuple[int, int], beliefs: BeliefMap, knowledge: KindKnowledge) -> bool:
    x, y = cell
    kind = beliefs.kind[y, x]
    return bool(beliefs.hurt[y, x] > 0.1 or (kind >= 0 and knowledge.painful(kind) > 0.3))


def _goal_cells(
    goal: str,
    pose: tuple[int, int, int],
    beliefs: BeliefMap,
    knowledge: KindKnowledge,
    cold: bool,
    temperature: float,
    remembered: Callable[[str], tuple[int, int] | None],
    tick: int,
) -> tuple[str, list[tuple[int, int]]]:
    x, y, _ = pose
    kinds = beliefs.kind
    if goal == "food":
        edible = [k for k in range(len(knowledge.eaten)) if knowledge.edible(k) > 0.5]
        cells = [(int(cx), int(cy)) for cy, cx in zip(*np.nonzero(np.isin(kinds, edible)))] if edible else []
        if cells:
            return "face", cells
        place = remembered("ate")
        if place is not None and place != (x, y):
            return "enter", [place]
        return "enter", frontier(beliefs, pose, tick)
    if goal in ("warmth", "sleep", "healing"):
        warmth = np.nan_to_num(beliefs.warmth, nan=-1.0)
        if goal == "warmth" and not cold:
            visited = np.nonzero((warmth >= 0) & (warmth < temperature - 0.05))
            return "enter", [(int(cx), int(cy)) for cy, cx in zip(*visited)]
        warmth[beliefs.hurt > 0.1] = -1.0  # not somewhere it got hurt
        if warmth.max() > 0.62:
            best = np.nonzero(warmth >= warmth.max() - 0.03)
            return "enter", [(int(cx), int(cy)) for cy, cx in zip(*best)]
        return "enter", [] if goal != "warmth" else frontier(beliefs, pose, tick)
    return "enter", frontier(beliefs, pose, tick)


def frontier(beliefs: BeliefMap, pose: tuple[int, int, int], tick: int) -> list[tuple[int, int]]:
    """Places at the edge of what it knows, or that it hasn't seen in a long while."""
    x, y, _ = pose
    passable = (beliefs.kind == OPEN) | (beliefs.visits > 0)
    unknown = beliefs.kind == UNKNOWN
    edge = np.zeros_like(passable)
    edge[1:, :] |= unknown[:-1, :]
    edge[:-1, :] |= unknown[1:, :]
    edge[:, 1:] |= unknown[:, :-1]
    edge[:, :-1] |= unknown[:, 1:]
    candidates = passable & edge
    if not candidates.any():
        stale = passable & (beliefs.seen < tick - 1500)
        candidates = stale if stale.any() else passable & (beliefs.visits == beliefs.visits[passable].min())
    return [(int(cx), int(cy)) for cy, cx in zip(*np.nonzero(candidates)) if (cx, cy) != (x, y)]
