"""Whether Haven does what it's asked: a choice of its own.

It isn't a servant. Asked to do something, it weighs it up as a creature would: is it afraid
of that (its amygdala's answer, at the thought of it), does a need come first, is it too tired,
does it like the one asking (its bond: oxytocin, and how they've been with it), does it like
doing that or has it done it so much it's bored of it, is it in the middle of something, is
there something it would rather do. Then it's glad to, or will, or not now, or would rather
not, and it says why (see cortex/engage.py, which makes the choice from these). Asked again
("please!"), it may give in.
"""

from __future__ import annotations

import random

import numpy as np

from .activities import ACTIVITIES, SIGHTS, freshness, urges
from .cortex.engage import BUSY, RATHER, Decision, decide, pastime_words
from .mind import Mind, need_words

DONE = {  # what doing it is, as it says it has done it lately
    "use": {
        "bell": "rung the bell",
        "ball": "pushed the ball",
        "flower": "smelled the flowers",
        "tree": "shaken the trees",
        "pond": "drunk from the pond",
        "fire": "sat by the fire",
    },
    "eat": {"bush": "eaten berries", "apple": "eaten apples"},
}
EFFECT = {"bell": "rang", "ball": "pushed", "flower": "smelled", "tree": "shook", "pond": "drank", "fire": "warmed"}


def fear_of(mind: Mind, thing: str) -> tuple[float, str]:
    """How afraid it is of a thing, and why: its amygdala's answer at the thought of it (with a brain), or what it
    remembers of being hurt by it."""
    stats = mind.things.get(thing, {})
    hurt = float(stats.get("hurt", 0.0)) + 2.0 * float(stats.get("sick", 0.0))
    why = "the fire burned me" if thing == "fire" else f"the {thing} hurt me" if hurt else f"I'm scared of the {thing}"
    if mind.brain is not None:
        afraid = mind.brain.fear_of({f"see:{thing}": 1.0})
    else:
        afraid = min(1.0, hurt / 4.0)
    return afraid, why


def bond(mind: Mind) -> float:
    """How it feels about the one asking, 0 to 1: oxytocin, its friendliness, knowing who they are, their kindness."""
    oxytocin = mind.chemistry.get("oxytocin", 1.0)
    kind = min(1.0, (mind.counts.get("touched", 0) + mind.counts.get("fed", 0)) / 20.0)
    known = 0.15 if mind.person else 0.0
    return float(
        np.clip(0.3 + 0.25 * (oxytocin - 1.0) + 0.3 * mind.character.traits["friendly"] + known + 0.2 * kind, 0, 1)
    )


def consider(mind: Mind, req, insisted: bool = False, rng: random.Random | None = None) -> Decision:
    """Whether it'll do what it's asked (`req`: cortex/talk.Request), and why."""
    rng = rng or random.Random()
    drives = mind.body.drives()
    need = int(np.argmax(drives))
    afraid, why = fear_of(mind, req.thing)
    stats = mind.things.get(req.thing, {})
    joy = float(stats.get("joy", 0.0))
    tried = sum(float(v) for k, v in stats.items() if k in ("ate", "drank", "rang", "pushed", "smelled", "shook"))
    appeal = min(1.0, joy / 3.0) if tried else 0.3 + 0.4 * mind.character.traits["curious"]  # (new: curious about it)
    effect = EFFECT.get(req.thing)
    bored = 1.0 - freshness(mind, f"event:{effect}") if effect and mind.brain is not None else 0.0
    if effect and mind.brain is None:
        kinds = [k for k, names in mind.kind_names.items() if names.get(req.thing)]
        bored = max([min(1.0, mind.played.get(k, 0.0) / 5.0) for k in kinds] + [0.0])
    activity = mind.activity
    busy = freshness(mind, activity) if activity in ACTIVITIES and activity != "company" else 0.0
    sight = SIGHTS.get(mind.watching or "", mind.watching)
    wants = urges(mind, [], drives) if mind.vision.seen else {}
    best = max(wants, key=wants.get) if wants else None
    rather = pastime_words(RATHER, best, sight, mind.visiting) if best and best != activity else None
    return decide(
        fear=afraid,
        fear_reason=why[0].upper() + why[1:],
        need=float(drives[need]),
        need_words=need_words(need, float(drives[need]), mind.body.cold()),
        meets_need=req.need == need,
        tired=float(drives[3]),
        sleep_asked=req.action == "sleep",
        bond=bond(mind),
        appeal=appeal,
        bored=bored,
        done_lately=DONE.get(req.action, {}).get(req.thing, "done that"),
        busy=busy,
        busy_words=pastime_words(BUSY, activity, sight, mind.visiting) or "busy",
        rather=rather,
        rather_urge=float(wants.get(best, 0.0)) if best else 0.0,
        dopamine=float(mind.chemistry.get("dopamine", 1.0)),
        friendly=float(mind.character.traits["friendly"]),
        insisted=insisted,
        rng=rng,
    )
