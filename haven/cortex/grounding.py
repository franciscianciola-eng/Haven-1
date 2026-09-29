"""The bridge between Haven's experience and its language.

`mind_state` turns what Haven is experiencing into the workspace tokens the language
cortex reads. `describe` puts the same moment into plain first-person words. Pairs of the
two, gathered from simulated lives, are the "talking about itself" level of the
curriculum: they teach the cortex to say what state it's in (reading its state tokens)
and to understand words about states (reading "hungry" brings hunger to mind).

What the cortex learns to say is tied to Haven's actual state, not to what a person
would say: if its self-model hasn't found evidence that it's alive, the words it learns
for that moment are "I don't know yet what I am."
"""

from __future__ import annotations

import numpy as np

from ..attention import GOALS
from ..mind import Mind, need_words
from ..workspace import SOURCES, D
from .model import SLOTS

NEED_WORDS = ("hungry", "cold", "hot", "hurt", "tired", "fine")


def mind_state(mind: Mind) -> np.ndarray:
    """What Haven is experiencing, as the language cortex's workspace tokens (SLOTS × D)."""
    b, schema, me = mind.body, mind.schema, mind.me
    state = np.zeros((SLOTS, D), dtype=np.float32)
    state[0] = mind.workspace.vector
    phase = 2 * np.pi * (mind.world.tick % 1200) / 1200
    body = [
        *b.drives(),
        mind.valence,
        mind.arousal,
        mind.mood * 5,
        b.energy,
        b.temperature,
        b.integrity,
        b.fatigue,
        float(b.asleep),
        float(b.cold()),
        mind.world.light,
        np.sin(phase),
        np.cos(phase),
        *np.eye(len(GOALS))[mind.goals.index()],
    ]
    state[1, : len(body)] = body
    focus = np.eye(len(SOURCES) + 1)[schema.focus]
    attention = [*schema.expected, *focus, float(schema.captured), min(mind.workspace.dwell / 50, 1.0)]
    state[2, : len(attention)] = attention
    evidence = list(me.evidence.values())
    self_part = [
        *evidence,
        me.alive,
        min(mind.age / 12000, 1.0),
        min(len(mind.lexicon.vocabulary()) / 20, 1.0),
        min(len(mind.vision.kinds.alive()) / 16, 1.0),
        min(len(me.milestones) / 30, 1.0),
    ]
    state[3, : len(self_part)] = self_part
    return state


def dominant_need(mind: Mind) -> str:
    drives = mind.body.drives()
    i = int(np.argmax(drives))
    if drives[i] < 0.3:
        return "fine"
    return ("hungry", "cold" if mind.body.cold() else "hot", "hurt", "tired")[i]


def describe(mind: Mind) -> tuple[str, dict]:
    """This moment in plain first-person words, with the facts a listener could check."""
    b = mind.body
    content = mind.workspace.content
    facts: dict = {"need": dominant_need(mind), "color": None}
    parts = []
    if b.asleep:
        if content is not None and content.source == "memory" and content.label.startswith("dreaming"):
            parts.append(f"I'm asleep, {content.label}.")
        else:
            parts.append("I'm asleep.")
        return " ".join(parts), facts
    drives = b.drives()
    need = int(np.argmax(drives))
    if facts["need"] == "fine":
        parts.append("I feel fine.")
    else:
        parts.append(f"I'm {need_words(need, float(drives[need]), b.cold())}.")
    if content is not None:
        if content.source == "vision":
            parts.append(f"I see {content.label.split(' (')[0]}.")
            facts["color"] = content.extra.get("color")
            if content.kind >= 0:
                knowledge = mind.knowledge.describe(content.kind)
                if "good to eat" in knowledge:
                    parts.append("I think it's good to eat.")
                elif "it hurts" in knowledge:
                    parts.append("I think it hurts.")
        elif content.source == "smell":
            parts.append("I smell something sweet.")
        elif content.source == "touch":
            parts.append(
                {
                    "pain, where it's standing": "Ouch, it hurts.",
                    "bumped into something": "I bumped into something.",
                    "a gentle touch": "Someone touched me.",
                    "someone gave it food": "Someone gave me food.",
                }.get(content.label, "I felt something.")
            )
        elif content.source == "hearing":
            if content.label.startswith("the word "):
                parts.append(f"I heard {content.label.removeprefix('the word ')}.")
            else:
                parts.append("Someone is talking to me.")
        elif content.source in ("memory", "imagination"):
            parts.append(f"I'm {content.label}.")
    goal = mind.goals.current
    parts.append(
        {
            "food": "I want to find food.",
            "warmth": "I want to get warm." if b.cold() else "I want to cool down.",
            "healing": "I need to rest.",
            "sleep": "I want to go to my nest and sleep.",
            "explore": "I want to look around.",
        }[goal]
    )
    if mind.valence > 0.2:
        parts.append("That feels good.")
    elif mind.valence < -0.2:
        parts.append("That feels bad.")
    return " ".join(parts), facts


# What someone keeping it company calls the things it looks at, by their color.
NAMES = {"red": "berry", "purple": "thorn", "gray": "stone", "yellow": "nest", "dark gray": "wall", "pale": "stone"}


def visit(mind: Mind, rng: np.random.Generator) -> None:
    """Someone keeping it company now and then: naming what it's looking at, saying hello, touching and feeding it."""
    content = mind.workspace.content
    if content is not None and content.source == "vision" and rng.random() < 0.05:
        word = NAMES.get(content.extra.get("color"))
        if word:
            mind.hear(word)
    drives = mind.body.drives()
    need = int(np.argmax(drives))
    if drives[need] > 0.5 and rng.random() < 0.01:
        mind.hear(("hungry", "cold" if mind.body.cold() else "hot", "hurt", "tired")[need])
    roll = rng.random()
    if roll < 0.001:
        mind.touch()
    elif roll < 0.0015:
        mind.feed()
    elif roll < 0.003:
        mind.hear(str(rng.choice(["hello haven", "hi", "good morning haven", "hello"])))


def gather(seed: int, days: float, every: int = 7, company: bool = True) -> list[dict]:
    """Live a simulated life and note down the moments: state tokens, words for them, and what it would answer."""
    from .talk import answers, notes

    mind = Mind(seed)
    rng = np.random.default_rng(seed)
    moments = []
    for tick in range(int(days * 1200)):
        if company:
            visit(mind, rng)
        mind.step()
        # Moments of need are rarer than moments of feeling fine, so they're noted more often.
        needy = dominant_need(mind) != "fine" or (mind.workspace.content and mind.workspace.content.source != "vision")
        if tick % (2 if needy else every):
            continue
        text, facts = describe(mind)
        moments.append(
            {"state": mind_state(mind), "text": text, "notes": notes(mind), "answers": answers(mind), **facts}
        )
    return moments


def need_index(word: str) -> int | None:
    return {"hungry": 0, "cold": 1, "hot": 1, "hurt": 2, "tired": 3}.get(word)
