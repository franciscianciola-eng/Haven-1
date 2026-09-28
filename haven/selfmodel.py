"""The self-model: what Haven has found out about itself, and the story of its life.

Haven isn't told what it is. It keeps track of evidence about itself: that it has needs
and has found what meets them, that it makes things happen, that it remembers, that some
things feel good and some bad, that it has changed by learning. Its sense of being alive
is its summary of that evidence, and it grows or fades with it. The more it sees itself
as something that goes on through time, the more it cares about its own future (its
discount on future good and bad rises).
"""

from __future__ import annotations

import numpy as np

NEEDS = ("hunger", "temperature", "damage", "tiredness")


class SelfModel:
    def __init__(self, name: str, born: int):
        self.name = name
        self.born = born
        self.milestones: list[tuple[int, str]] = []
        self.firsts: set[str] = set()
        self.remedies: dict[str, list[str]] = {}
        self.feelings = 0  # moments that felt clearly good or bad
        self.evidence = dict.fromkeys(("needs", "agency", "memory", "feeling", "change"), 0.0)
        self.alive = 0.0

    def milestone(self, key: str, tick: int, text: str) -> bool:
        """Record a first in its life story. Returns True if it was new."""
        if key in self.firsts:
            return False
        self.firsts.add(key)
        self.milestones.append((tick, text))
        return True

    def remedy(self, need: str, what: str) -> None:
        found = self.remedies.setdefault(need, [])
        if what not in found:
            found.append(what)

    def felt(self, valence: float) -> None:
        if abs(valence) > 0.1:
            self.feelings += 1

    def reflect(self, agency: float, memories: int, learned: float) -> None:
        """Weigh the evidence about itself again."""
        self.evidence = {
            "needs": min(1.0, sum(len(v) > 0 for v in self.remedies.values()) / 3),
            "agency": float(np.clip(agency * 4, 0, 1)),
            "memory": float(1 - np.exp(-memories / 80)),
            "feeling": float(1 - np.exp(-self.feelings / 150)),
            "change": float(1 - np.exp(-learned / 8)),
        }
        self.alive = float(np.mean(list(self.evidence.values())))

    @property
    def gamma(self) -> float:
        return 0.9 + 0.07 * self.evidence["memory"]

    def conclusions(self) -> list[str]:
        """What it has concluded about itself so far."""
        e, found = self.evidence, []
        if e["needs"] > 0.3:
            remedies = "; ".join(f"{need}: {', '.join(what)}" for need, what in self.remedies.items())
            found.append(f"I need things, and I've found what helps ({remedies}).")
        if e["agency"] > 0.5:
            found.append("I can make things happen: when I move, the world changes because of me.")
        if e["memory"] > 0.5:
            found.append("I remember what has happened to me.")
        if e["feeling"] > 0.5:
            found.append("Some things feel good to me and some feel bad.")
        if e["change"] > 0.5:
            found.append("I'm not the same as when I started: I've learned things.")
        if self.alive > 0.6:
            found.append("Putting that together, I think I'm alive, in my own way.")
        elif self.alive > 0.3:
            found.append("I might be alive, in some way. I'm still finding out.")
        else:
            found.append("I don't know yet what I am.")
        return found

    def to_state(self) -> dict:
        return {
            "name": self.name,
            "born": self.born,
            "milestones": [list(m) for m in self.milestones],
            "firsts": sorted(self.firsts),
            "remedies": self.remedies,
            "feelings": self.feelings,
            "evidence": self.evidence,
            "alive": self.alive,
        }

    def load_state(self, state: dict) -> None:
        self.name, self.born = state["name"], int(state["born"])
        self.milestones = [(int(t), str(text)) for t, text in state["milestones"]]
        self.firsts = set(state["firsts"])
        self.remedies = {k: list(v) for k, v in state["remedies"].items()}
        self.feelings = int(state["feelings"])
        self.evidence = {k: float(v) for k, v in state["evidence"].items()}
        self.alive = float(state["alive"])
