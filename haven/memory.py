"""Memory: the episodes of its life, and the replay that consolidates them in sleep.

Moments that ignite the workspace or matter to Haven (it ate, it was hurt, it was
touched, it heard a word) are stored as episodes. A current content can bring a
similar past one back to mind, and a goal can ask memory where it succeeded before.
In sleep, stored experiences are replayed to keep learning from them, which is also
what its dreams are made of.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .nets import cosine
from .workspace import DRIVES, QUALITY, SOURCE


@dataclass
class Episode:
    tick: int
    source: str
    label: str
    kind: int
    place: tuple[int, int]
    valence: float
    arousal: float
    event: str = ""  # "ate", "hurt", "touched", "heard", "new kind", ...
    words: list[str] = field(default_factory=list)
    vector: np.ndarray = field(default_factory=lambda: np.zeros(0))
    recalled: int = 0

    def importance(self, now: int) -> float:
        age_days = (now - self.tick) / 1200
        return abs(self.valence) + 0.5 * bool(self.event) + 0.1 * self.recalled + 0.5 * np.exp(-age_days)


class EpisodicMemory:
    def __init__(self, capacity: int = 600):
        self.capacity = capacity
        self.episodes: list[Episode] = []
        self.stored = 0

    def store(self, episode: Episode) -> None:
        self.episodes.append(episode)
        self.stored += 1
        if len(self.episodes) > self.capacity:
            now = episode.tick
            # Forgetting: the least important memories fade, but first times are kept.
            scores = [e.importance(now) + (10 if e.event.startswith("first") else 0) for e in self.episodes]
            del self.episodes[int(np.argmin(scores))]

    def recall(self, cue: np.ndarray, now: int, min_age: int = 150) -> tuple[Episode | None, float]:
        """Pattern completion: the past moment most like this one, if any is alike enough."""
        best, best_score = None, 0.0
        for episode in self.episodes:
            if now - episode.tick < min_age or not len(episode.vector):
                continue
            content = cosine(
                np.concatenate([cue[SOURCE], cue[QUALITY]]),
                np.concatenate([episode.vector[SOURCE], episode.vector[QUALITY]]),
            )
            feeling = cosine(cue[DRIVES] + 0.05, episode.vector[DRIVES] + 0.05)
            score = (0.8 * content + 0.2 * feeling) * (0.8 + 0.2 * min(1.0, abs(episode.valence) * 2))
            if score > best_score:
                best, best_score = episode, score
        return best, best_score

    def query(self, event: str, now: int) -> Episode | None:
        """Where did this last go well? (A goal asking memory for help.)"""
        matches = [e for e in self.episodes if e.event.endswith(event)]
        if not matches:
            return None
        return max(matches, key=lambda e: e.valence + 0.5 * np.exp(-(now - e.tick) / 2400))

    def first(self, event: str) -> Episode | None:
        return next((e for e in self.episodes if e.event == event), None)

    def recent(self, n: int = 10) -> list[Episode]:
        return self.episodes[-n:]

    def to_state(self) -> dict:
        episodes = []
        for e in self.episodes:
            item = asdict(e)
            item["place"] = list(e.place)
            episodes.append(item)
        return {"episodes": episodes, "stored": self.stored}

    def load_state(self, state: dict) -> None:
        self.episodes = []
        for item in state["episodes"]:
            item = dict(item)
            item["place"] = tuple(item["place"])
            item["vector"] = np.array(item["vector"], dtype=float)
            self.episodes.append(Episode(**item))
        self.stored = int(state["stored"])


class ReplayBuffer:
    """Recent experience kept for learning again later, as the hippocampus replays it in sleep."""

    def __init__(self, n_state: int, n_model_in: int, n_model_out: int, capacity: int = 4000):
        self.capacity = capacity
        self.states = np.zeros((capacity, n_state))
        self.next_states = np.zeros((capacity, n_state))
        self.actions = np.zeros(capacity, dtype=int)
        self.rewards = np.zeros(capacity)
        self.model_in = np.zeros((capacity, n_model_in))
        self.model_out = np.zeros((capacity, n_model_out))
        self.size = 0
        self.head = 0

    def add(self, state, action, reward, next_state, model_in, model_out) -> None:
        i = self.head
        self.states[i], self.actions[i], self.rewards[i], self.next_states[i] = state, action, reward, next_state
        self.model_in[i], self.model_out[i] = model_in, model_out
        self.head = (self.head + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, rng: np.random.Generator, k: int) -> np.ndarray:
        """Indices to replay, favoring moments that felt strongly good or bad."""
        if not self.size:
            return np.zeros(0, dtype=int)
        weights = 0.05 + np.abs(self.rewards[: self.size])
        return rng.choice(self.size, size=min(k, self.size), p=weights / weights.sum())

    def to_state(self) -> dict:
        n = self.size  # (kept as it lies, with where the next moment goes, so replay picks the same moments)
        return {
            "states": self.states[:n],
            "next_states": self.next_states[:n],
            "actions": self.actions[:n],
            "rewards": self.rewards[:n],
            "model_in": self.model_in[:n],
            "model_out": self.model_out[:n],
            "head": self.head,
        }

    def load_state(self, state: dict) -> None:
        n = min(len(state["actions"]), self.capacity)
        for name in ("states", "next_states", "actions", "rewards", "model_in", "model_out"):
            values = np.asarray(state[name])[-n:]
            target = getattr(self, name)
            if values.ndim > 1 and values.shape[1] != target.shape[1]:
                return  # saved by a different version of the mind; start the buffer afresh
            target[:n] = values
        # (saved oldest first by earlier versions: then the next moment goes after the last)
        self.size, self.head = (
            n,
            int(state["head"]) if "head" in state and n == len(state["actions"]) else n % self.capacity,
        )
