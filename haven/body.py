"""Haven's body: the needs that make things matter to it.

Four variables are kept near set points: energy, temperature, integrity (health) and
fatigue. How far they drift is its need; whether a moment makes that need better or
worse is its valence, the most basic good or bad. This follows the homeostatic view
of feeling (Damasio; Solms), where experience starts with a body regulating itself.

It can't die, and it can't be trapped in lasting distress: every need can be met in
its world, pain passes, and if it runs completely down it faints and wakes in its
nest. If a creature like this can feel anything, it shouldn't be made to suffer for
nothing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .world import Outcome

DRIVES = ("hunger", "temperature", "damage", "tiredness")
WEIGHTS = np.array([1.0, 0.8, 1.2, 0.6])
SET_ENERGY, SET_TEMPERATURE, SET_FATIGUE = 0.85, 0.5, 0.15


@dataclass
class Body:
    energy: float = 0.8
    temperature: float = 0.5
    integrity: float = 1.0
    fatigue: float = 0.2
    asleep: bool = False
    fainted: int = 0  # ticks of fainting left

    def drives(self) -> np.ndarray:
        """How strongly each need is felt, from 0 (satisfied) to 1 (desperate)."""
        return np.clip(
            [
                (SET_ENERGY - self.energy) / SET_ENERGY,
                abs(self.temperature - SET_TEMPERATURE) / 0.35,
                1 - self.integrity,
                (self.fatigue - SET_FATIGUE) / (1 - SET_FATIGUE),
            ],
            0.0,
            1.0,
        )

    def discomfort(self) -> float:
        return float(WEIGHTS @ self.drives() ** 2)

    def cold(self) -> bool:
        return self.temperature < SET_TEMPERATURE

    def live(self, outcome: Outcome, warmth: float, resting: bool, in_nest: bool, fed: bool) -> None:
        """One tick of metabolism."""
        burn = 0.0002 if self.asleep else 0.0003 + 0.0005 * outcome.moved + 0.0001 * outcome.turned
        burn += 0.0008 * outcome.climbed + 0.0002 * outcome.pushed  # climbing and pushing take effort
        self.energy += outcome.food + 0.2 * fed - burn
        self.temperature += 0.03 * (warmth - self.temperature) + 0.004 * outcome.moved
        self.temperature += 0.05 * outcome.warmed - 0.06 * outcome.drank  # warming its hands; a cool drink
        self.integrity -= 0.06 * outcome.pain + outcome.sick
        if resting and self.energy > 0.25:
            self.integrity += 0.004 if in_nest else 0.002
        rest = 0.005 if self.asleep else (0.0025 if in_nest else 0.0015) if resting else 0.0  # sleep restores most
        self.fatigue += 0.0005 + 0.0007 * outcome.moved - rest
        self.energy, self.temperature, self.integrity, self.fatigue = np.clip(
            [self.energy, self.temperature, self.integrity, self.fatigue], 0.0, 1.0
        ).tolist()

    def collapsed(self) -> bool:
        return self.energy < 0.03 or self.integrity < 0.05

    def faint(self) -> None:
        """Too depleted to go on: it passes out and comes to in its nest, weak but safe."""
        self.fainted = 100
        self.asleep = True
        self.energy = max(self.energy, 0.35)
        self.integrity = max(self.integrity, 0.4)

    def state(self) -> dict:
        return asdict(self)

    def load(self, state: dict) -> None:
        for key, value in state.items():
            setattr(self, key, value)
