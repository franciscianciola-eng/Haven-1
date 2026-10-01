"""What Haven does when its needs leave it free: pastimes of its own.

Besides meeting its needs, exploring and playing with what it has found fun, it can sit and
watch something lovely or new (the pond, the fire, butterflies, the flowers, the sunset or the
stars from the top of the hill), sing, dance, chase butterflies, go and visit a part of the
valley it likes or hasn't been to for a while, and keep someone company.

How much it wants each comes from how it feels (its needs have to leave it free), from who it
is (a cheerful Haven sings more, a playful one dances more, a calm one watches more, a friendly
one keeps you company more), from its brain's chemistry when it has a brain (dopamine for
dancing and playing, serotonin for singing and watching, oxytocin for company), and from what's
around it. Doing the same thing over and over wears the urge down: the brain's relay cells for
what it's doing tire (habituation, see brain/brain.py), so the urge to keep at it fades and
something else wins. Without a brain the same wearing down is kept as a simple running tally.
"""

from __future__ import annotations

import numpy as np

from .world import BUTTERFLY, DAY

ACTIVITIES = ("watch", "sing", "dance", "chase", "visit", "company")
BASE = {"watch": "explore", "visit": "explore", "sing": "play", "dance": "play", "chase": "play", "company": "play"}
WATCHABLE = {  # what's lovely to watch, by what people call it: how lovely
    "water": 0.7,
    "fire": 0.9,
    "butterfly": 1.0,
    "flower": 0.6,
    "ball": 0.3,
    "bell": 0.2,
}
SIGHTS = {"water": "the pond", "fire": "the fire", "butterfly": "a butterfly", "flower": "the flowers"}
SKY = {"dusk": ("sunset", "the sunset"), "night": ("stars", "the stars")}  # from the top of the hill
TIRING, RESTING = 0.06, 0.004  # without a brain: how fast an activity wears thin, and comes back


def free(drives: np.ndarray) -> float:
    """How free its needs leave it for pastimes, 0 (a need is pressing) to 1 (all met)."""
    return float(max(0.0, 1.0 - 1.6 * float(np.max(drives))))


def freshness(mind, activity: str) -> float:
    """How much it still feels like doing something it has been doing (1 = not done lately)."""
    brain = mind.brain
    if brain is not None:
        return brain.freshness(f"doing {activity}")
    return float(1.0 - mind.worn.get(activity, 0.0))


def wear(mind, doing: str | None) -> None:
    """Without a brain: what it's doing wears thin a little each moment, and everything else comes back."""
    for activity in list(mind.worn):
        mind.worn[activity] *= 1.0 - RESTING
        if mind.worn[activity] < 0.01:
            del mind.worn[activity]
    if doing is not None:
        mind.worn[doing] = mind.worn.get(doing, 0.0) + TIRING * (1.0 - mind.worn.get(doing, 0.0))


def sights(mind, percepts: list) -> list[tuple[float, str, tuple[int, int]]]:
    """Lovely things in view: (how much it would like to watch it, what, where), the best first."""
    from .mind import name_of

    found = []
    for p in percepts:
        if p.cell is None:
            continue
        name = name_of(mind.world, *p.cell)
        appeal = WATCHABLE.get(name or "", 0.0)
        if name == "fire" and mind.world.light > 0.5:
            appeal *= 0.4  # (a fire is best in the dark)
        if appeal > 0:
            found.append((appeal * (0.4 + 0.6 * freshness(mind, f"watch {name}")), name, p.cell))
    w = mind.world
    sky = SKY.get(mind.time_of_day)
    if sky and w.level(w.x, w.y) >= 3:
        found.append((0.9 * (0.4 + 0.6 * freshness(mind, f"watch {sky[0]}")), sky[0], (w.x, w.y)))
    return sorted(found, reverse=True)


def urges(mind, percepts: list, drives: np.ndarray) -> dict[str, float]:
    """How much it wants each pastime right now."""
    t = mind.character.traits
    chem = mind.chemistry
    room = free(drives)
    night = mind.world.light < 0.3
    seen = sights(mind, percepts)
    butterfly = any(name == "butterfly" for _, name, _ in seen)
    happy = max(0.0, min(1.0, 0.4 + 2.0 * mind.mood + 0.5 * mind.valence))
    w = mind.world
    here = w.area(w.x, w.y)
    pull = 0.0
    for area, last in mind.visited.items():
        if area != here:
            pull = max(pull, min(1.0, (mind.world.tick - last) / (2 * DAY)))
    favorite = mind.character.favorite_area()
    if favorite and favorite != here:
        pull = max(pull, 0.7)
    wants = {
        "watch": room * (0.25 + 0.35 * t["calm"]) * (seen[0][0] if seen else 0.0) * (0.6 + 0.4 * chem["serotonin"]),
        "sing": room * 0.3 * (0.4 + t["cheerful"]) * happy * (0.4 if night else 1.0) * min(chem["serotonin"], 2.0),
        "dance": room * 0.3 * (0.4 + t["playful"]) * happy * (0.3 if night else 1.0) * min(chem["dopamine"], 2.0),
        "chase": room * 0.45 * (0.4 + t["playful"]) * float(butterfly),
        "visit": room * 0.3 * (0.4 + t["curious"]) * pull * (0.3 if night else 1.0),
        "company": 0.55 * (0.3 + t["friendly"]) * float(mind.company) * min(chem["oxytocin"], 2.0),
    }
    return {a: v * (0.25 + 0.75 * freshness(mind, a)) for a, v in wants.items()}


def plan(mind, activity: str, pose: tuple[int, int, int], percepts: list, tick: int):
    """Where it goes for a pastime, and what to do once it's there: (target, action or None)."""
    from .agency import frontier
    from .mind import name_of

    w = mind.world
    x, y, _ = pose
    if activity == "watch":
        seen = sights(mind, percepts)
        if not seen:
            return ("enter", frontier(mind.beliefs, pose, tick)), None
        _, name, cell = seen[0]
        mind.watching = name
        ahead = percepts[2] if percepts else None
        close = name in ("sunset", "stars") or (
            ahead is not None and ahead.cell == cell and ahead.distance <= 4 and name_of(w, *cell) == name
        )
        if close:
            return ("enter", [(x, y)]), "rest"  # it sits down and watches
        return ("face", [cell]), None
    if activity == "sing":
        return ("enter", [(x, y)]), "speak"
    if activity == "dance":
        return ("enter", [(x, y)]), ("left" if tick % 2 else "right")  # round and round
    if activity == "chase":
        flutter = [p.cell for p in percepts if p.cell is not None and w.thing(*p.cell) == BUTTERFLY]
        if not flutter:
            return ("enter", frontier(mind.beliefs, pose, tick)), None
        cell = min(flutter, key=lambda c: abs(c[0] - x) + abs(c[1] - y))
        ahead = percepts[2] if percepts else None
        if ahead is not None and ahead.cell == cell and ahead.distance == 1:
            return ("face", [cell]), "use"  # a pounce
        return ("face", [cell]), None
    if activity == "visit":
        here = w.area(x, y)
        if mind.visiting is None or mind.visiting == here:
            mind.visiting = _where_to_visit(mind, here)
        cells = [
            (cx, cy)
            for cy in range(w.height)
            for cx in range(w.width)
            if w.areas[cy][cx] == mind.visiting and w.free(cx, cy)
        ]
        return ("enter", sorted(cells, key=lambda c: abs(c[0] - x) + abs(c[1] - y))[:12]), None
    if activity == "company":
        return ("enter", [(x, y)]), "rest"  # it settles down by you
    return ("enter", []), None


def _where_to_visit(mind, here: str) -> str:
    favorite = mind.character.favorite_area()
    if favorite and favorite != here and mind.world.tick - mind.visited.get(favorite, -(10**9)) > DAY // 2:
        return favorite
    w = mind.world
    hurt = {w.areas[y][x] for y, x in zip(*np.nonzero(mind.beliefs.hurt > 0.1), strict=True)}  # (not where it got hurt)
    places = [a for a in mind.visited if a != here and not a.endswith("wall") and a not in hurt] or ["the meadow"]
    return min(places, key=lambda a: mind.visited.get(a, -(10**9)))


def felt(mind, activity: str | None, action: str, outcome) -> float:
    """How pleasant this moment of a pastime is (added to what it feels)."""
    if activity is None:
        return 0.0
    t = mind.character.traits
    if activity == "watch" and action == "rest":
        return 0.25 + 0.2 * t["calm"]
    if activity == "sing" and action == "speak":
        return 0.25 + 0.25 * t["cheerful"]
    if activity == "dance" and action in ("left", "right"):
        return 0.25 + 0.25 * t["playful"]
    if activity == "chase" and getattr(outcome, "chased", False):
        return 0.5 + 0.3 * t["playful"]
    if activity == "company" and action == "rest" and mind.company:
        return 0.2 + 0.3 * t["friendly"]
    return 0.0


def bout(mind, activity: str | None, action: str, outcome, tick: int) -> None:
    """Note a pastime the first time in a while that it's really doing it (for telling about its day)."""
    if activity is None:
        return
    done = None
    if activity == "watch" and action == "rest" and mind.watching:
        done = ("watched", f"watched {SIGHTS.get(mind.watching, 'the ' + mind.watching)}")
    elif activity == "sing" and action == "speak":
        done = ("sang", "sang a little song")
    elif activity == "dance" and action in ("left", "right"):
        done = ("danced", "danced")
    elif activity == "chase" and getattr(outcome, "chased", False):
        done = ("chased", "chased a butterfly")
    elif activity == "company" and action == "rest" and mind.company:
        done = ("kept company", "sat with you")
    elif activity == "visit" and mind.visiting and mind.world.area(mind.world.x, mind.world.y) == mind.visiting:
        done = ("visited", f"went to {mind.visiting}")
        mind.visiting = None
    if done is None or mind.bouts.get(done[1], -(10**9)) > tick - 120:
        return
    mind.bouts[done[1]] = tick
    mind._did(*done)


def voice(text: str) -> str:
    """What singing sounds like: its own sounds, to a tune."""
    return f"♪ {text} ♪"
