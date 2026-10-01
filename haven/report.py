"""Readouts: plain-English descriptions of what is going on inside Haven, for the people watching.

These are instrument readings, not Haven's own words. Every line is computed from its
current internal state: what's in its workspace, what its attention schema says, what it
feels, what it's trying to do, and what it has concluded about itself. Haven's own words
are only the ones it learned, or what its language cortex says once it has one.
"""

from __future__ import annotations

import numpy as np

from .attention import NOTHING
from .body import DRIVES
from .mind import Mind, need_words
from .workspace import SOURCES
from .world import ACTIONS, BUSH, DAY, FLOWER, HEIGHTS, LAYOUT, YEAR

GOALS = {
    "food": "find food",
    "warmth": "get to a comfortable temperature",
    "healing": "rest and heal",
    "sleep": "get to its nest and sleep",
    "explore": "explore",
    "play": "play with something it enjoys",
}


def feeling_words(valence: float, arousal: float) -> str:
    if valence > 0.5:
        tone = "very good"
    elif valence > 0.1:
        tone = "good"
    elif valence < -0.5:
        tone = "very bad"
    elif valence < -0.1:
        tone = "bad"
    else:
        tone = "neutral"
    energy = "excited" if arousal > 0.6 else "alert" if arousal > 0.3 else "calm"
    return f"{tone}, {energy}"


def readout(mind: Mind) -> list[str]:
    b, ws, schema = mind.body, mind.workspace, mind.schema
    lines = []
    content = ws.content
    if b.asleep:
        if content is not None and content.source == "memory" and content.label.startswith("dreaming"):
            lines.append(f"Asleep, {content.label}.")
        else:
            lines.append("Asleep" + (" (fainted)" if b.fainted else "") + ".")
    elif content is None:
        lines.append("Nothing in particular is in its mind right now.")
    else:
        lines.append(f"In its mind: {content.label} ({content.source}).")
        if schema.captured:
            lines.append("Its attention was just grabbed by that; its attention schema didn't see it coming.")
        elif schema.expected[schema.focus] > 0.8:
            lines.append("Its attention schema expects its attention to stay there for now.")
        if content.source == "vision" and content.kind >= 0:
            facts = mind.knowledge.describe(content.kind)
            if facts:
                lines.append("What it has learned about that kind of thing: " + ", ".join(facts) + ".")
            lines.append(f"How sure it is of what it sees: {content.confidence:.0%}.")
    drives = b.drives()
    needs = [need_words(i, level, b.cold()) for i, level in enumerate(drives) if level >= 0.15]
    lines.append(
        f"Feeling {feeling_words(mind.valence, mind.arousal)}"
        + (f"; {', '.join(needs)}." if needs else "; its needs are met.")
    )
    if not b.asleep:
        goal = GOALS[mind.goals.current]
        step = f" Next: {mind.suggestion}." if mind.suggestion else ""
        lines.append(f"Trying to {goal}.{step}")
        if (
            mind.agency_now > 0.3
            and mind.last is not None
            and ACTIONS[mind.last.action] in ("forward", "left", "right")
        ):
            lines.append("It sensed that it caused the last change it saw.")
    return lines


def kinds(mind: Mind) -> list[dict]:
    result = []
    for k in mind.vision.kinds.alive():
        rgb = np.clip(mind.vision.coder.reconstruct(mind.vision.kinds.centers[k])[:3], 0, 1)
        result.append(
            {
                "id": k,
                "color": "#" + "".join(f"{int(c * 255):02x}" for c in rgb),
                "looks": mind.kind_look(k),
                "name": mind.lexicon.name_for(k),
                "facts": mind.knowledge.describe(k),
                "seen": int(mind.vision.kinds.counts[k]),
            }
        )
    return result


def snapshot(mind: Mind) -> dict:
    """Everything the dashboard shows, in one JSON-able dict."""
    w, b = mind.world, mind.body
    content = mind.workspace.content
    schema = mind.schema
    focus = SOURCES[schema.focus] if schema.focus != NOTHING else None
    words = []
    for form in mind.lexicon.vocabulary():
        kind, need = mind.lexicon.kind_of(form), mind.lexicon.need_of(form)
        words.append(
            {
                "word": form,
                "means": mind.kind_color(kind) + " things"
                if kind is not None
                else DRIVES[need]
                if need is not None
                else "?",
                "heard": mind.lexicon.words[form].heard,
                "said": mind.lexicon.words[form].said,
            }
        )
    return {
        "name": mind.me.name,
        "tick": w.tick,
        "age_days": round(mind.age / 1200, 2),
        "day": w.day,
        "time_of_day": mind.time_of_day,
        "season": w.season,
        "season_day": w.season_day,
        "year": w.year,
        "light": round(w.light, 2),
        "world": world(mind),
        "beliefs": {
            "kind": mind.beliefs.kind.tolist(),
            "confidence": np.round(mind.beliefs.confidence, 2).tolist(),
            "hurt": np.round(mind.beliefs.hurt, 2).tolist(),
        },
        "kinds": kinds(mind),
        "body": {
            "energy": round(b.energy, 3),
            "temperature": round(b.temperature, 3),
            "integrity": round(b.integrity, 3),
            "fatigue": round(b.fatigue, 3),
            "asleep": b.asleep,
            "fainted": b.fainted > 0,
        },
        "drives": {name: round(float(v), 3) for name, v in zip(DRIVES, b.drives(), strict=True)},
        "feeling": {"valence": round(mind.valence, 3), "arousal": round(mind.arousal, 3), "mood": round(mind.mood, 3)},
        "workspace": None
        if content is None
        else {
            "source": content.source,
            "label": content.label,
            "strength": round(mind.workspace.strength, 2),
            "dwell": mind.workspace.dwell,
            "confidence": round(content.confidence, 2),
        },
        "stream": [{"tick": t, "source": s, "label": label} for t, s, label in mind.workspace.history[-14:]][::-1],
        "attention": {
            "focus": focus,
            "expected": {
                (SOURCES[i] if i != NOTHING else "nothing"): round(float(p), 2)
                for i, p in enumerate(schema.expected)
                if p > 0.05
            },
            "captured": schema.captured,
        },
        "goal": mind.goals.current,
        "goal_text": f"{mind.errand['do']}, as it was asked" if mind.errand else doing(mind),
        "brain": brain(mind),
        "next": mind.suggestion,
        "said": [{"tick": t, "text": text} for t, text in mind.said[-6:]][::-1],
        "log": [{"tick": t, "text": text} for t, text in mind.log[-14:]][::-1],
        "words": words,
        "self": {
            "conclusions": mind.me.conclusions(),
            "alive": round(mind.me.alive, 2),
            "evidence": {k: round(v, 2) for k, v in mind.me.evidence.items()},
            "milestones": [{"tick": t, "text": text} for t, text in mind.me.milestones[-12:]][::-1],
        },
        "readout": readout(mind),
        "character": character(mind),
        "welfare": welfare(mind),
        "today": mind.today,
        "things": things(mind),
        "you": {
            "name": mind.person,
            "told": [fact for _, fact in mind.told[-12:]][::-1],
            "taught": [lesson for _, lesson in mind.lessons[-12:]][::-1],
        },
    }


def character(mind: Mind) -> dict:
    """Who it's becoming: its traits (and the temperament it was born with), and what it says about itself."""
    from .cortex.talk import TRAIT_WORDS, best_day, favorites, fears, worst_day
    from .cortex.talk import character as in_words

    c = mind.character
    return {
        "traits": [
            {"trait": t, "less": TRAIT_WORDS[t][1], "value": round(v, 3), "born": round(c.temperament[t], 3)}
            for t, v in c.traits.items()
        ],
        "says": [p for p in (in_words(mind), favorites(mind), fears(mind), best_day(mind), worst_day(mind)) if p],
        "days": c.days,
    }


def things(mind: Mind) -> list[str]:
    """What it has found out about each thing in its valley that it has come across."""
    from .cortex.talk import THINGS, knows_of, thing_note

    notes = []
    for name in THINGS:
        stats = {k: v for k, v in mind.things.get(name, {}).items() if k not in ("x", "y", "last")}
        if knows_of(stats):
            notes.append(thing_note(name, stats))
    return notes


def world(mind: Mind) -> dict:
    """The world as it really is (for the people watching; Haven only knows what it has seen of it)."""
    w = mind.world
    return {
        "layout": list(LAYOUT),
        "heights": list(HEIGHTS),
        "tick": w.tick,
        "light": round(w.light, 3),
        "berries": [[x, y, n] for (x, y), n in w.berries.items()],
        "bushes": [[x, y] for (x, y) in w.berries if w.grid[y, x] == BUSH],
        "fruit": [[x, y, n] for (x, y), n in w.fruit.items()],
        "apples": [list(a) for a in w.apples],
        "mushrooms": [[x, y, int(wait == 0)] for (x, y), wait in w.mushrooms.items()],
        "ball": list(w.ball),
        "butterflies": [] if w.season == "winter" else [list(b) for b in w.butterflies],
        "rang": w.tick - w.rang,
        "flowers": [
            [int(x), int(y), _hex(w._color(int(x), int(y), FLOWER))] for y, x in zip(*np.nonzero(w.grid == FLOWER))
        ],
        "phase": round((w.tick % DAY) / DAY, 4),
        "season": w.season,
        "year_phase": round((w.tick % (DAY * YEAR)) / (DAY * YEAR), 4),
        "x": w.x,
        "y": w.y,
        "heading": w.heading,
        "asleep": mind.body.asleep,
        "did": did(mind),
        "voice": None if w.voice is None else {"ago": w.tick - w.voice[0], "text": w.voice[1]},
        "errand": errand(mind),
        "doing": mind.activity,  # a pastime (activities.py): watch, sing, dance, chase, visit, company
    }


def errand(mind: Mind) -> dict | None:
    """What it's doing because someone asked, and the place it believes it's going (nearest first), if it knows."""
    if mind.errand is None:
        return None
    thing = "nest" if mind.errand["action"] == "sleep" else mind.errand["thing"]
    places = sorted(mind.places_of(thing), key=lambda c: abs(c[0] - mind.world.x) + abs(c[1] - mind.world.y))
    return {"do": mind.errand["do"], "to": list(places[0]) if places else None}


def did(mind: Mind) -> list[str]:
    """What just happened to it, for showing: what it did, and what it felt."""
    last = mind.last
    if last is None:
        return []
    o = last.outcome
    happened = [
        e
        for e in ("ate", "drank", "rang", "pushed", "smelled", "shook", "warmed", "sick", "bumped", "chased")
        if getattr(o, e)
    ]
    return happened + ["hurt"] * bool(o.pain)


def _hex(rgb) -> str:
    return "#" + "".join(f"{int(round(float(c) * 255)):02x}" for c in rgb[:3])


def welfare(mind: Mind) -> dict:
    """Is it doing all right? Sustained bad feeling is flagged so a person can help."""
    if mind.distress > 600:
        return {
            "ok": False,
            "message": "Haven has felt bad for a long while. Consider feeding it, touching it, or pausing it.",
        }
    if mind.body.fainted:
        return {"ok": False, "message": "Haven fainted and is recovering in its nest."}
    return {"ok": True, "message": ""}


PASTIMES = {  # what it's doing, as the window says it
    "sing": "sing a little song",
    "dance": "dance",
    "chase": "chase a butterfly",
    "company": "keep you company",
}


def doing(mind: Mind) -> str:
    """What it's trying to do, in a few words: a need, or one of its pastimes."""
    activity = mind.activity
    if activity == "watch":
        from .activities import SIGHTS

        return (
            f"watch {SIGHTS.get(mind.watching or '', 'something lovely')}"
            if mind.watching
            else "find something to watch"
        )
    if activity == "visit":
        return f"go to {mind.visiting}" if mind.visiting else "go somewhere it likes"
    return PASTIMES.get(activity or "", GOALS[mind.goals.current])


def brain(mind: Mind) -> dict | None:
    """Its brain of spiking neurons, for the window: how big, how it's firing, its chemistry."""
    found = mind.brain
    if found is None:
        return None
    reading = mind.reading
    return {
        "size": found.size,
        "neurons": found.neurons(),
        "synapses": found.synapses(),
        "chemistry": {k: round(v, 2) for k, v in mind.chemistry.items()},
        "rates": {k: round(v, 2) for k, v in reading.rates.items()} if reading else {},
        "novelty": round(mind.novelty, 2),
        "fear": round(mind.fear, 2),
        "senses": len(found.senses),
        "ms": round(found.timing, 1),
    }
