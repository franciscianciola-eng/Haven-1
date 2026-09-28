"""Small neural networks that learn online, one experience at a time."""

from __future__ import annotations

from itertools import pairwise

import numpy as np


class MLP:
    """A small tanh network trained by gradient descent on each new experience."""

    def __init__(self, sizes: list[int], rng: np.random.Generator, lr: float = 0.01, clip: float = 1.0):
        self.W = [rng.normal(0, 1 / np.sqrt(a), (b, a)) for a, b in pairwise(sizes)]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.lr = lr
        self.clip = clip

    def _forward(self, x: np.ndarray) -> list[np.ndarray]:
        activations = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b, strict=True)):
            z = W @ activations[-1] + b
            activations.append(np.tanh(z) if i < len(self.W) - 1 else z)
        return activations

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x)[-1]

    def learn(self, x: np.ndarray, target: np.ndarray, weight: float | np.ndarray = 1.0) -> float:
        """One step of gradient descent on squared error. Returns the error before the step."""
        activations = self._forward(x)
        error = activations[-1] - target
        grad = error * weight
        for i in reversed(range(len(self.W))):
            gW, gb = np.outer(grad, activations[i]), grad
            if i > 0:
                grad = (self.W[i].T @ grad) * (1 - activations[i] ** 2)
            scale = self.lr * min(1.0, self.clip / (np.linalg.norm(gW) + 1e-12))
            self.W[i] -= scale * gW
            self.b[i] -= scale * gb
        return float(np.mean(error**2))

    def to_state(self) -> dict:
        return {"W": self.W, "b": self.b}

    def load_state(self, state: dict) -> None:
        self.W = [np.array(W, dtype=float) for W in state["W"]]
        self.b = [np.array(b, dtype=float) for b in state["b"]]


class QualityCoder:
    """A population code: units tuned to regions of an input space, like color-tuned neurons.

    Only a few units respond to any input (the code is sparse), and similar inputs excite
    similar units (it is smooth), so the codes form a space of qualities in which a red and a
    slightly different red are close and a red and a green are far apart.

    Inference combines the bottom-up evidence with a top-down prediction, trusting each in
    proportion to how reliable the evidence is; the part of the input the code fails to
    explain is its prediction error. Learning moves the best-tuned units toward what is seen.
    """

    def __init__(self, n_in: int, n_units: int, rng: np.random.Generator, width: float = 0.12, active: int = 3):
        self.centers = rng.uniform(0.1, 0.9, (n_units, n_in))
        self.width = width
        self.active = active

    def tuning(self, x: np.ndarray, reliability: float = 1.0) -> np.ndarray:
        width = self.width / np.sqrt(max(reliability, 0.05))  # less reliable input, less precise tuning
        return self.sparsen(np.exp(-np.sum((self.centers - x) ** 2, axis=1) / (2 * width * width)))

    def sparsen(self, z: np.ndarray) -> np.ndarray:
        z = z.copy()
        if self.active < len(z):
            z[np.argsort(z)[: -self.active]] = 0.0
        return z

    def infer(
        self, x: np.ndarray, prior: np.ndarray | None = None, reliability: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        z = self.tuning(x, reliability)
        if prior is not None:
            z = self.sparsen(reliability * z + (1 - reliability) * prior)
        return z, x - self.reconstruct(z)

    def reconstruct(self, z: np.ndarray) -> np.ndarray:
        total = z.sum()
        return z @ self.centers / total if total > 1e-9 else self.centers.mean(axis=0)

    def learn(self, x: np.ndarray, lr: float) -> None:
        order = np.argsort(np.sum((self.centers - x) ** 2, axis=1))
        for rank, unit in enumerate(order[:4]):  # the best-tuned unit most, its runners-up a little
            self.centers[unit] += lr * (0.3 * np.exp(-rank) if rank else 1.0) * (x - self.centers[unit])

    def to_state(self) -> dict:
        return {"centers": self.centers}

    def load_state(self, state: dict) -> None:
        self.centers = np.array(state["centers"], dtype=float)


class Prototypes:
    """Learns kinds of things by clustering codes, one sighting at a time.

    Something unlike every known kind is only a candidate at first; it becomes a kind of its
    own once it has been seen several times, so a single odd glimpse doesn't create one.
    """

    def __init__(self, dim: int, capacity: int = 12, radius: float = 0.35, lr: float = 0.05, confirm: int = 5):
        self.centers = np.zeros((0, dim))
        self.counts = np.zeros(0)
        self.capacity = capacity
        self.radius = radius
        self.lr = lr
        self.confirm = confirm
        self.candidates: list[list] = []  # [center, sightings]

    def nearest(self, z: np.ndarray) -> tuple[int, float]:
        if not len(self.centers):
            return -1, np.inf
        distances = np.linalg.norm(self.centers - z, axis=1)
        index = int(np.argmin(distances))
        return index, float(distances[index])

    def assign(self, z: np.ndarray, weight: float = 1.0, may_create: bool = True) -> int:
        """The kind this code belongs to, learning from it. -1 if it's unlike anything known."""
        index, distance = self.nearest(z)
        if distance <= self.radius:
            self.centers[index] += self.lr * weight * (z - self.centers[index])
            self.counts[index] += 1
            return index
        if may_create and (len(self.centers) < self.capacity or (self.counts == 0).any()):
            return self._consider(z)
        return -1

    def _consider(self, z: np.ndarray) -> int:
        for i, candidate in enumerate(self.candidates):
            if np.linalg.norm(candidate[0] - z) <= self.radius:
                candidate[0] += (z - candidate[0]) / (candidate[1] + 1)
                candidate[1] += 1
                if candidate[1] >= self.confirm:
                    del self.candidates[i]
                    free = np.flatnonzero(self.counts == 0)
                    if len(free):  # a place left by two kinds that were merged
                        index = int(free[0])
                        self.centers[index], self.counts[index] = candidate[0], float(candidate[1])
                        return index
                    self.centers = np.vstack([self.centers, candidate[0]])
                    self.counts = np.append(self.counts, float(candidate[1]))
                    return len(self.centers) - 1
                return -1
        self.candidates = [*self.candidates[-7:], [z.copy(), 1]]
        return -1

    def merge(self, keep: int, gone: int) -> None:
        """Two kinds turned out to be one: pool them, and free the other's place."""
        total = self.counts[keep] + self.counts[gone]
        if total:
            self.centers[keep] = (
                self.counts[keep] * self.centers[keep] + self.counts[gone] * self.centers[gone]
            ) / total
        self.counts[keep] = total
        self.centers[gone] = 1e3  # nothing is near it any more
        self.counts[gone] = 0.0

    def alive(self) -> list[int]:
        return [int(k) for k in np.flatnonzero(self.counts > 0)]

    def to_state(self) -> dict:
        return {
            "centers": self.centers,
            "counts": self.counts,
            "candidates": [[c, n] for c, n in self.candidates],
        }

    def load_state(self, state: dict) -> None:
        self.centers = np.array(state["centers"], dtype=float).reshape(-1, self.centers.shape[1])
        self.counts = np.array(state["counts"], dtype=float)
        self.candidates = [[np.array(c, dtype=float), int(n)] for c, n in state.get("candidates", [])]


class RunningStat:
    """An exponentially weighted mean and variance."""

    def __init__(self, rate: float = 0.01, mean: float = 0.0, var: float = 1.0):
        self.rate = rate
        self.mean = mean
        self.var = var

    def update(self, value: float) -> None:
        delta = value - self.mean
        self.mean += self.rate * delta
        self.var = (1 - self.rate) * (self.var + self.rate * delta * delta)

    def to_state(self) -> dict:
        return {"mean": self.mean, "var": self.var}

    def load_state(self, state: dict) -> None:
        self.mean, self.var = float(state["mean"]), float(state["var"])


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = (x - np.max(x)) / max(temperature, 1e-6)
    e = np.exp(z)
    return e / e.sum()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 1e-9 else 0.0
