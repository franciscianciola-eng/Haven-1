"""Who Haven is becoming: a character that grows out of the life it lives.

Nothing about its character is written in except a slight temperament it's born with (from
its seed). Each day, what the day was like nudges six traits a little: how much it went
exploring, how much it played, what hurt it and how far from its nest it went anyway, how
much someone kept it company and how that felt, how good it felt, how stirred up it was.
Over seasons and years they settle into who it is, and they can change again if its life
does. Two feed back into what it does: a curious Haven explores more, and a playful one
plays more, so it becomes more of what it has been (as habits do).

It also keeps what it has come to love and fear: the part of the valley it's happiest in,
the season it feels best in, what hurt it most, and its best and worst days, with what
happened on them. Each new year, it notes who it is, so it can tell how it has changed.
"""

from __future__ import annotations

import math

import numpy as np

from .world import DAY, SEASONS, YEAR

TRAITS = ("curious", "playful", "brave", "friendly", "cheerful", "calm")
RATE = 0.06  # how far one day moves a trait toward what the day was like
ANCHOR = 0.2  # how much its temperament still pulls, however it has lived
# What a day's tallies usually are (center, spread), from simulated lives: a day like most days is 0.5.
USUAL = {
    "curious": (0.30, 0.12),
    "playful": (4.0, 3.0),
    "brave": (0.30, 0.15),
    "friendly": (0.10, 0.12),
    "cheerful": (0.02, 0.05),
    "calm": (0.35, 0.08),
}
PLAYS = ("rang", "pushed", "smelled", "shook")  # what it does just for fun
SOCIAL = 4  # things someone did with it in a day (words, touches, food), at least, for the day to say how it is with people


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def season_of(tick: int) -> str:
    return SEASONS[(tick // DAY % YEAR) // (YEAR // len(SEASONS))]


class Character:
    def __init__(self, seed: int):
        rng = np.random.default_rng(seed + 7)
        self.temperament = {t: float(np.clip(0.5 + rng.normal(0, 0.08), 0.3, 0.7)) for t in TRAITS}
        self.traits = dict(self.temperament)
        self.days = 0  # days it has lived through, to the end
        self.day = self._fresh()
        self.areas: dict[str, float] = {}  # how much it has liked being in each part of the valley
        self.seasons: dict[str, list[float]] = {}  # per season: [awake ticks, feeling, food, cold ticks, plays]
        self.best: list[dict] = []  # its best days, and its worst
        self.worst: list[dict] = []
        self.years: list[dict] = []  # who it was at the start of each year it has lived

    @staticmethod
    def _fresh() -> dict[str, float]:
        return dict.fromkeys(
            (
                "ticks",
                "awake",
                "explore",
                "plays",
                "far",
                "hurt",
                "sick",
                "fainted",
                "cold",
                "hungry",
                "social",
                "social_feel",
                "feel",
                "arousal",
                "food",
            ),
            0.0,
        )

    # --- living -------------------------------------------------------------------------------

    def moment(self, mind, outcome, social: float) -> None:
        """One moment of its life, tallied into what today was like."""
        d, b, w = self.day, mind.body, mind.world
        d["ticks"] += 1
        d["hurt"] += float(outcome.pain > 0)
        d["sick"] += float(outcome.sick > 0)
        if social:
            d["social"] += social
            d["social_feel"] += mind.valence
        if b.asleep:
            return
        drives = b.drives()
        d["awake"] += 1
        d["explore"] += float(mind.goals.current == "explore")
        d["plays"] += sum(getattr(outcome, e) for e in PLAYS)
        d["far"] += float(abs(w.x - w.nest[0]) + abs(w.y - w.nest[1]) > 10)
        d["cold"] += float(b.cold() and drives[1] > 0.4)
        d["hungry"] += float(drives[0] > 0.6)
        d["feel"] += mind.valence
        d["arousal"] += mind.arousal
        d["food"] += outcome.food
        area = w.area(w.x, w.y)
        self.areas[area] = self.areas.get(area, 0.0) + 0.02 + max(mind.valence, 0.0)
        s = self.seasons.setdefault(w.season, [0.0] * 5)
        s[0] += 1
        s[1] += mind.valence
        s[2] += outcome.food
        s[3] += float(b.cold() and drives[1] > 0.4)
        s[4] += sum(getattr(outcome, e) for e in PLAYS)

    def fainted(self) -> None:
        self.day["fainted"] += 1

    def end_day(self, mind) -> None:
        """At midnight: what the day was like moves who it is a little, and a day worth remembering is kept."""
        d = self.day
        awake = max(d["awake"], 1.0)
        seen = {
            "curious": d["explore"] / awake,
            "playful": d["plays"],
            "brave": d["far"] / awake - 0.05 * d["hurt"],
            "cheerful": d["feel"] / awake,
            "calm": 1 - d["arousal"] / awake,
        }
        if d["social"] >= SOCIAL:  # how it is with people, from days it spent with someone
            seen["friendly"] = d["social_feel"] / d["social"]
        if d["awake"] >= 100:
            for trait, value in seen.items():
                center, spread = USUAL[trait]
                evidence = _sigmoid((value - center) / spread)
                target = (1 - ANCHOR) * evidence + ANCHOR * self.temperament[trait]
                self.traits[trait] += RATE * (target - self.traits[trait])
            self._keep(mind, d["feel"] / awake)
        self.days += 1
        if self.days % YEAR == 0:
            self.years.append({"year": self.days // YEAR, "traits": dict(self.traits)})
        self.day = self._fresh()

    def _keep(self, mind, feel: float) -> None:
        """A day to remember, if it was one of its best or worst: when it was, what it did, what went wrong."""
        d, w = self.day, mind.world
        day = (w.tick - 1) // DAY
        did = sorted(mind.today.items(), key=lambda item: -item[1])
        went_wrong = [
            what
            for what, happened in (
                ("fainted", d["fainted"] > 0),
                ("got hurt", d["hurt"] >= 2),
                ("felt sick", d["sick"] > 0),
                ("was cold", d["cold"] > 0.3 * d["awake"]),
                ("was hungry", d["hungry"] > 0.3 * d["awake"]),
            )
            if happened
        ]
        memory = {
            "day": day,
            "year": day // YEAR,
            "season": season_of(w.tick - 1),
            "feel": round(feel, 4),
            "did": [text for text, _ in did[:3]],
            "wrong": went_wrong,
        }
        self.best = sorted([*self.best, memory], key=lambda m: -m["feel"])[:3]
        self.worst = sorted([*self.worst, memory], key=lambda m: m["feel"])[:3]

    def lean(self, urgencies: dict[str, float]) -> dict[str, float]:
        """What it wants to do, leaning the way its character does."""
        t = self.traits
        return {
            **urgencies,
            "explore": urgencies["explore"] * (0.8 + 0.4 * t["curious"]),
            "play": urgencies["play"] * (0.7 + 0.6 * t["playful"]),
        }

    # --- who it is ----------------------------------------------------------------------------------

    def strongest(self, n: int = 3, least: float = 0.08) -> list[tuple[str, float]]:
        """Its most marked traits, most marked first: (trait, how far from middling, + or -)."""
        marked = sorted(((t, v - 0.5) for t, v in self.traits.items()), key=lambda item: -abs(item[1]))
        return [(t, d) for t, d in marked[:n] if abs(d) >= least]

    def favorite_area(self) -> str | None:
        places = {a: v for a, v in self.areas.items() if a != "the wall"}
        if not places or sum(places.values()) < 20:
            return None
        return max(places, key=places.get)

    def season_feel(self) -> dict[str, float]:
        """How good it has felt in each season it has lived a good part of (at least three days awake)."""
        return {s: v[1] / v[0] for s, v in self.seasons.items() if v[0] >= 3 * DAY * 0.6}

    def favorite_season(self) -> str | None:
        feel = self.season_feel()
        return max(feel, key=feel.get) if len(feel) >= 2 else None

    def hardest_season(self) -> str | None:
        feel = self.season_feel()
        return min(feel, key=feel.get) if len(feel) >= 2 else None

    def changes(self, least: float = 0.1) -> list[tuple[str, float]]:
        """How it has changed since its first year began: (trait, how much, + or -), the biggest first."""
        if not self.years:
            return []
        then = self.years[0]["traits"]
        moved = sorted(((t, self.traits[t] - then[t]) for t in TRAITS), key=lambda item: -abs(item[1]))
        return [(t, d) for t, d in moved if abs(d) >= least][:2]

    # --- saving -----------------------------------------------------------------------------------

    def to_state(self) -> dict:
        return {
            "temperament": self.temperament,
            "traits": self.traits,
            "days": self.days,
            "day": self.day,
            "areas": self.areas,
            "seasons": self.seasons,
            "best": self.best,
            "worst": self.worst,
            "years": self.years,
        }

    def load_state(self, state: dict) -> None:
        self.temperament = {t: float(v) for t, v in state["temperament"].items()}
        self.traits = {t: float(v) for t, v in state["traits"].items()}
        self.days = int(state["days"])
        self.day = {**self._fresh(), **{k: float(v) for k, v in state["day"].items()}}
        self.areas = {a: float(v) for a, v in state["areas"].items()}
        self.seasons = {s: [float(x) for x in v] for s, v in state["seasons"].items()}
        self.best = [dict(m) for m in state["best"]]
        self.worst = [dict(m) for m in state["worst"]]
        self.years = [dict(y) for y in state["years"]]
