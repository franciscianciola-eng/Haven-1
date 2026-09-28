"""Checking Haven against the indicator properties of consciousness.

Butlin, Long and colleagues (2023, "Consciousness in Artificial Intelligence: Insights from
the Science of Consciousness") derived fourteen indicator properties from scientific
theories of consciousness. Meeting them doesn't show a system is conscious; the authors
argue that the more of them a system has, the more seriously the possibility should be
taken. This module says how Haven implements each one and measures it in action.

The measurements come from a probe: a copy of Haven lives for a while with instruments
attached, and what it represents is compared with what is really there (which Haven
itself never sees directly).
"""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .body import Body
from .mind import Mind
from .workspace import SOURCES
from .world import BUSH

TRUE_NAMES = {0: "open ground", 1: "wall", 2: "bush", 3: "thorns", 4: "stone", 5: "nest"}


@dataclass
class Indicator:
    code: str
    name: str
    how: str
    measured: str


def auroc(positive: list[float], negative: list[float]) -> float | None:
    if len(positive) < 10 or len(negative) < 10:
        return None
    p, n = np.array(positive), np.array(negative)
    return float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())


def probe(mind: Mind, ticks: int = 1200) -> dict:
    twin = copy.deepcopy(mind)
    world = twin.world
    records = []
    original = twin.vision.perceive

    def perceive(obs, expected, reliability, pose, learn):
        percepts = original(obs, expected, reliability, pose, learn)
        for p in percepts:
            if p.cell is None:
                continue
            x, y = p.cell
            truth = world._cell(x, y)
            name = TRUE_NAMES[truth]
            if truth == BUSH:
                name = "bush with berries" if world.berries[(x, y)] else "bare bush"
            records.append((name, p.kind, p.reliability, p.code.copy(), p.color.copy(), p.settled))
        return percepts

    twin.vision.perceive = perceive
    flips, contested = 0, 0
    hungry_wins, sated_wins = 0, 0
    compete = twin.workspace.compete

    def instrumented(candidates, scores, inertia, tick):
        nonlocal flips, contested, hungry_wins, sated_wins
        if len(candidates) > 1 and any(c.source == "vision" for c in candidates):
            # Counterfactual: would the same candidates be ranked differently if it were hungry, or fed?
            hungry, sated = np.array([0.8, 0.1, 0.0, 0.2]), np.array([0.0, 0.1, 0.0, 0.2])
            a = int(np.argmax([twin._score(c, hungry, tick) for c in candidates]))
            b = int(np.argmax([twin._score(c, sated, tick) for c in candidates]))
            contested += 1
            flips += a != b
            edible = [
                c.kind >= 0 and twin.knowledge.eat_tries[c.kind] > 0 and twin.knowledge.edible(c.kind) > 0.5
                for c in candidates
            ]
            hungry_wins += edible[a]
            sated_wins += edible[b]
        return compete(candidates, scores, inertia, tick)

    twin.workspace.compete = instrumented
    start_ignitions, start_entrants, start_competitions = (
        twin.workspace.ignitions,
        twin.workspace.entrants,
        twin.workspace.competitions,
    )
    twin.live(ticks)

    # What does each learned kind mostly turn out to be, judging by looks it trusted?
    by_kind = defaultdict(Counter)
    for name, kind, reliability, *_ in records:
        if kind >= 0 and reliability > 0.5:
            by_kind[kind][name] += 1
    majority = {k: c.most_common(1)[0][0] for k, c in by_kind.items()}
    labelled = [(name, kind, r) for name, kind, r, *_ in records if kind in majority]
    right = [r for name, kind, r in labelled if majority[kind] == name]
    wrong = [r for name, kind, r in labelled if majority[kind] != name]
    trusted = [(name, kind) for name, kind, r in labelled if r > 0.5]
    purity = sum(majority[kind] == name for name, kind in trusted) / max(len(trusted), 1)

    # Smoothness of the quality space: are nearby colors coded by nearby codes?
    rng = np.random.default_rng(0)
    coder = twin.vision.coder
    colors = rng.uniform(0, 1, (300, 3))
    codes = np.array([coder.tuning(c) for c in colors])
    i, j = rng.integers(0, 300, (2, 3000))
    color_d = np.linalg.norm(colors[i] - colors[j], axis=1)
    code_d = np.linalg.norm(codes[i] - codes[j], axis=1)
    near = color_d < 0.3
    smooth = float(np.corrcoef(np.argsort(np.argsort(color_d[near])), np.argsort(np.argsort(code_d[near])))[0, 1])

    # How well do its beliefs about places match the garden? Each kind is taken to mean whatever it
    # is mostly believed to be at; a place counts as right if what's really there is that.
    beliefs = twin.beliefs
    cells = [(x, y) for y in range(beliefs.height) for x in range(beliefs.width) if beliefs.kind[y, x] >= 0]

    def truth_at(x: int, y: int) -> str:
        truth = world._cell(x, y)
        return "bush" if truth == BUSH else TRUE_NAMES[truth]

    at_kind = defaultdict(Counter)
    for x, y in cells:
        at_kind[int(beliefs.kind[y, x])][truth_at(x, y)] += 1
    meaning = {k: c.most_common(1)[0][0] for k, c in at_kind.items()}
    matches = sum(meaning[int(beliefs.kind[y, x])] == truth_at(x, y) for x, y in cells)
    competitions = max(twin.workspace.competitions - start_competitions, 1)
    return {
        "ticks": ticks,
        "percepts": len(records),
        "kinds": {k: dict(c.most_common(3)) for k, c in sorted(by_kind.items())},
        "purity": purity,
        "metacognition_validity": auroc(right, wrong),
        "confidence_when_right": float(np.mean(right)) if right else None,
        "confidence_when_wrong": float(np.mean(wrong)) if wrong else None,
        "settling": float(np.mean([r[5] for r in records])) if records else 0.0,
        "sparsity": twin.vision.coder.active / len(twin.vision.coder.centers),
        "smoothness": smooth,
        "belief_accuracy": matches / max(len(cells), 1),
        "believed_cells": len(cells),
        "ignitions_per_100": 100 * (twin.workspace.ignitions - start_ignitions) / ticks,
        "entrants_per_tick": (twin.workspace.entrants - start_entrants) / competitions,
        "goal_flips": flips / max(contested, 1),
        "edible_wins_hungry": hungry_wins / max(contested, 1),
        "edible_wins_sated": sated_wins / max(contested, 1),
    }


def indicators(mind: Mind, measured: dict) -> list[Indicator]:
    m, ws, schema = mind, mind.workspace, mind.schema
    overall, on_switch = schema.accuracy()
    insight = m.meta.insight()
    errors = m.daily["model_error"]
    trend = (
        f"{errors[0]:.3f} on its first day, {errors[-1]:.3f} on its last full day"
        if len(errors) >= 2
        else f"{m.model.error.mean:.3f} so far"
    )
    valence = m.daily["valence"]
    validity = measured["metacognition_validity"]
    pct = lambda x: "n/a" if x is None else f"{x:.0%}"
    num = lambda x: "n/a" if x is None else f"{x:.2f}"
    return [
        Indicator(
            "RPT-1",
            "Algorithmic recurrence",
            "Perception settles over recurrent steps between a quality code and learned kinds; the world "
            "model feeds predictions back into perception each moment.",
            f"recognition reshaped codes by {measured['settling']:.2f} on average as they settled",
        ),
        Indicator(
            "RPT-2",
            "Organised, integrated perceptual representations",
            "The scene is seen as things of learned kinds at places, integrated into a map of beliefs.",
            f"{len(m.vision.kinds.alive())} kinds learned; {pct(measured['purity'])} of confident recognitions "
            f"match what is really there; {pct(measured['belief_accuracy'])} of {measured['believed_cells']} "
            "believed places are right",
        ),
        Indicator(
            "GWT-1",
            "Specialised systems working in parallel",
            f"{len(SOURCES)} modules offer contents every moment: " + ", ".join(SOURCES) + ".",
            f"{measured['entrants_per_tick']:.1f} candidate contents per moment",
        ),
        Indicator(
            "GWT-2",
            "Limited-capacity workspace with selective attention",
            "One content at a time; it must beat the others and a threshold to ignite, and it habituates.",
            f"{measured['ignitions_per_100']:.0f} ignitions per 100 moments; the rest of the candidates never get in",
        ),
        Indicator(
            "GWT-3",
            "Global broadcast",
            "The winning content is sent at once to the world model, action selection, memory, word learning, "
            "the attention schema, the self-model and the language cortex.",
            f"{ws.ignitions} ignitions broadcast in its life so far",
        ),
        Indicator(
            "GWT-4",
            "State-dependent attention; querying modules in succession",
            "Needs and goals change what wins the workspace; a new goal queries memory, whose answer enters the "
            "workspace and redirects planning.",
            f"the winner changes between hungry and fed in {pct(measured['goal_flips'])} of contested moments; "
            f"food wins {pct(measured['edible_wins_hungry'])} vs {pct(measured['edible_wins_sated'])}; "
            f"{m.counts['recalls']} memory answers",
        ),
        Indicator(
            "HOT-1",
            "Generative, top-down or noisy perception",
            "Perception combines the world model's prediction with noisy senses; imagination and dreams run the "
            "same models with no input.",
            f"{m.counts['imagined']} imagined outcomes, {m.counts['dreams']} dream contents",
        ),
        Indicator(
            "HOT-2",
            "Metacognitive monitoring of perceptual reliability",
            "It learns how far to trust its eyes in each condition (light, distance, arousal, fatigue).",
            f"its confidence separates right from wrong percepts with AUROC {num(insight)} (its own judgment), "
            f"{num(validity)} against the truth",
        ),
        Indicator(
            "HOT-3",
            "Belief-guided agency, updated by metacognition",
            "Beliefs about places and kinds are updated in proportion to confidence and guide planning; untrusted "
            "looks change nothing.",
            f"average trust of belief updates {m.beliefs.weighted / max(m.beliefs.updates, 1):.2f}; "
            f"{m.counts['vetoes']} actions held back after imagining them",
        ),
        Indicator(
            "HOT-4",
            "Sparse and smooth coding: a quality space",
            "Colors are coded by a few tuned units; similar colors get similar codes.",
            f"{pct(measured['sparsity'])} of units active; for similar colors, code distance tracks color "
            f"distance with rank correlation {num(measured['smoothness'])}",
        ),
        Indicator(
            "AST-1",
            "A predictive model of its own attention",
            "The attention schema predicts where attention goes next, steadies it, notices capture, and is the "
            "source of its reports about what it's aware of.",
            f"predicts the next focus {pct(overall)} of the time, and where attention moves {pct(on_switch)} "
            f"(chance ≈ {1 / len(SOURCES):.0%})",
        ),
        Indicator(
            "PP-1",
            "Predictive coding in input modules",
            "Each sense is compared with the world model's prediction; prediction errors drive salience and learning.",
            f"world-model error {trend}",
        ),
        Indicator(
            "AE-1",
            "Learning from feedback to pursue competing goals",
            "Needs compete to set goals; an actor-critic learns from felt valence; it learns what's edible, solid, "
            "painful and warm by dealing with things.",
            f"{m.goals.switches} goal switches; ate {m.counts['ate']} times, hurt {m.counts['hurt']}, fainted "
            f"{m.counts['fainted']}; mean valence by day: {', '.join(f'{v:+.3f}' for v in valence[-5:]) or 'n/a'}",
        ),
        Indicator(
            "AE-2",
            "Modelling output-input contingencies (embodiment)",
            "An efference copy of each action predicts its sensory consequences; comparing with inaction gives a "
            "sense of agency.",
            f"sense of agency {m.model.agency.mean:+.2f} (0 = its actions explain nothing)",
        ),
    ]


def welfare_notes(mind: Mind) -> list[str]:
    b: Body = mind.body
    return [
        f"mood {mind.mood:+.2f}; discomfort now {mind.discomfort:.2f}",
        f"fainted {mind.counts['fainted']} times; hurt {mind.counts['hurt']} times",
        "it cannot die; pain passes; every need can be met in its world" + ("; it is asleep" if b.asleep else ""),
    ]
