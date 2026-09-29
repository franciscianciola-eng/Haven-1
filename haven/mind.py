"""Haven's mind: one moment of experience after another.

Each tick the senses report, and each specialist module compares what arrives with what
it expected. The modules offer contents to the workspace; one ignites and is broadcast
to all of them. The attention schema notes where attention went. Beliefs, memories and
words are updated from the broadcast. Needs compete to set the goal, a plan is found,
and an action is chosen, tried out in imagination first, and done. The body answers with
better or worse, and that feeling is what the whole mind learns from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .agency import EFFECTS, ActorCritic, BeliefMap, Goals, KindKnowledge, Planner, goal_cells, reflexes
from .attention import GOALS, NOTHING, AttentionSchema, features
from .body import WEIGHTS, Body
from .language import Lexicon, sound
from .memory import Episode, EpisodicMemory, ReplayBuffer
from .metacognition import Metacognition, context
from .nets import cosine, softmax
from .perception import BODY_FEATURES, VISION_FEATURES, Observation, Vision, color_name, looks, vision_candidates
from .selfmodel import NEEDS, SelfModel
from .workspace import QUALITY, SOURCES, Candidate, D, Q, Workspace
from .world import (
    ACTIONS,
    APPLE,
    BUSH,
    DAY,
    FIRE,
    FLOOR,
    MUSHROOM,
    NAMES,
    NEST,
    SAND,
    THORN,
    TOADSTOOL,
    WATER,
    Outcome,
    World,
)
from .worldmodel import N_IN, N_OUT, Prediction, WorldModel

VERSION = 2  # 2: the valley, with its heights and things to use (1 was the first, flat garden)
N_STATE = D + VISION_FEATURES + len(BODY_FEATURES) + 4 + len(GOALS) + len(ACTIONS) + 3
A = {name: i for i, name in enumerate(ACTIONS)}
VERBS = {
    "forward": "stepping forward",
    "eat": "biting that",
    "use": "touching that",
    "left": "turning left",
    "right": "turning right",
}
EVENTS = ("ate", "drank", "rang", "pushed", "smelled", "shook", "warmed", "sick", "climbed")
DOINGS = ("ate", "sick", "drank", "rang", "pushed", "smelled", "shook", "warmed")  # what can happen with a thing


def name_of(world: World, x: int, y: int) -> str | None:
    """What people call the thing at a place: the words it's given for what it sees and does."""
    kind = world.thing(x, y)
    if kind in (FLOOR, SAND):
        return "hill" if world.level(x, y) > 0 else None  # rising ground
    return "pond" if kind == WATER else NAMES[kind]


@dataclass
class Moment:
    """What the mind keeps of the last tick, to learn from once it sees how it turned out."""

    state: np.ndarray
    action: int
    policy: np.ndarray
    reward: float
    model_in: np.ndarray
    drives: np.ndarray
    ahead_kind: int
    ahead_near: bool
    outcome: Outcome
    in_nest: bool
    ahead_trust: float = 1.0


class Mind:
    def __init__(self, seed: int = 0, name: str = "Haven"):
        self.seed = seed
        self.rng = np.random.default_rng(seed + 1)
        self.world = World(seed)
        self.body = Body()
        self.vision = Vision(self.rng)
        self.meta = Metacognition()
        self.model = WorldModel(self.rng)
        self.workspace = Workspace()
        self.schema = AttentionSchema()
        self.memory = EpisodicMemory()
        self.replay = ReplayBuffer(N_STATE, N_IN, N_OUT)
        self.beliefs = BeliefMap(self.world.width, self.world.height)
        self.knowledge = KindKnowledge(self.vision.kinds.capacity)
        self.goals = Goals()
        self.planner = Planner()
        self.agent = ActorCritic(N_STATE, self.rng)
        self.lexicon = Lexicon()
        self.me = SelfModel(name, self.world.tick)
        self.valence = 0.0
        self.arousal = 0.0
        self.mood = 0.0
        self.surprise = 0.0
        self.discomfort = self.body.discomfort()
        self.agency_now = 0.0
        self.prediction: Prediction | None = None
        self.last: Moment | None = None
        self.hearing: list[str] = []  # words it has heard but not yet attended to
        self.thoughts: list[Candidate] = []  # inner speech on its way into the workspace
        self.primed: tuple[np.ndarray, int, int] | None = None  # (quality, kind, until): a meaning just evoked
        self.query: str | None = None  # a goal asking memory for help
        self.queried: dict[str, int] = {}
        self.target: tuple = ("enter", [])
        self.suggestion: str | None = None
        self.log: list[tuple[int, str]] = []
        self.said: list[tuple[int, str]] = []
        self.rest_streak = 0
        self.scent = 0.0
        self.scent_change = 0.0
        self.distress = 0  # ticks of sustained low mood, watched for its welfare
        self.fun = 0.0  # how much it has played lately: playing the same way again is less fun for a while
        self.bitten: dict[tuple[int, int], int] = {}  # where biting did nothing, and when: it won't keep trying
        self.today: dict[str, int] = {}  # what it has done since dawn, for telling about its day
        self.things: dict[str, dict[str, float]] = {}  # what it has seen and done with each thing, by name
        self.kind_names: dict[int, dict[str, float]] = {}  # which of its own kinds of things those were
        self.person: str | None = None  # the name of the person who talks with it, once they've said it
        self.told: list[tuple[int, str]] = []  # what people have told it, in its words ("you like pizza")
        self._target_now: np.ndarray | None = None
        self.counts = dict.fromkeys(("vetoes", "recalls", "dreams", "imagined", "hurt", "fainted", *EVENTS), 0)
        self.daily: dict[str, list[float]] = {"model_error": [], "valence": [], "ate": []}
        self._today = {"model_error": 0.0, "daylight": 0.0, "valence": 0.0, "ate": 0.0, "n": 0}
        self.me.milestone("born", self.world.tick, f"{name} came into the world, in its nest")

    # --- the person ---------------------------------------------------------------

    def hear(self, text: str) -> list[str]:
        return self.world.say(text)

    def touch(self) -> None:
        self.world.touch()

    def feed(self) -> None:
        self.world.feed()

    def think(self, text: str, meaning: np.ndarray | None, confidence: float) -> None:
        """Inner speech from the language cortex, on its way into the workspace.

        `meaning` is the cortex's reading of its own words, in the workspace's layout.
        """
        self.thoughts.append(
            Candidate(
                "thought",
                _quality(meaning),
                salience=0.55 + 0.3 * confidence,
                confidence=confidence,
                label=f'thinking "{text[:80]}"',
                key=("thought", self.world.tick, len(self.thoughts)),
                extra={"text": text},
            )
        )

    def understand(self, text: str, meaning: np.ndarray) -> None:
        """What the language cortex made of something said to it: it comes to mind, and steers attention.

        Saying "look at the berries" draws its attention to berry-like things, as a single known
        word does.
        """
        quality = _quality(meaning)
        self.thoughts.append(
            Candidate(
                "hearing",
                quality,
                salience=0.7,
                label=f'understanding "{text[:80]}"',
                key=("hearing", "understood", self.world.tick),
                extra={"text": text},
            )
        )
        if np.linalg.norm(quality) > 1e-6:
            self.primed = (quality, -1, self.world.tick + 60)

    # --- living -----------------------------------------------------------------------

    @property
    def tick(self) -> int:
        return self.world.tick

    @property
    def age(self) -> int:
        return self.world.tick - self.me.born

    def live(self, ticks: int) -> None:
        for _ in range(ticks):
            self.step()

    def step(self) -> None:
        w, b = self.world, self.body
        tick = w.tick
        obs = Observation.of(w.sense())
        drives = b.drives()
        pose = (w.x, w.y, w.heading)  # it keeps track of where it is by counting its own steps
        self.hearing += obs.words
        moving = float(self.last is not None and self.last.outcome.moved)

        # Perceive, trusting each look as far as metacognition says it can be trusted.
        contexts = [context(obs.light, float(n), self.arousal, b.fatigue, moving) for n in obs.nearness]
        reliability = np.array([self.meta.confidence("vision", c) for c in contexts])
        predicted = self.prediction
        percepts = []
        if not b.asleep:
            expected = None if predicted is None else predicted.vision
            percepts = self.vision.perceive(obs, expected, reliability, pose, learn=True)
            if predicted is not None:
                for percept, ctx in zip(percepts, contexts, strict=True):
                    self.meta.learn("vision", ctx, percept.error)
            self.surprise = float(np.mean([p.error for p in percepts])) if predicted is not None else 0.0
        if self.last is not None:
            self._learn_from(self.last, obs, drives, pose)

        # The modules offer what they have; one content ignites and is broadcast to all.
        candidates = self._candidates(obs, drives, percepts, predicted, contexts[2], tick)
        scores = [self._score(c, drives, tick) for c in candidates]
        current = self.workspace.content
        steadiness = self.schema.steadiness(self._relevance(current, drives) if current else 0.0)
        content, ignited = self.workspace.compete(candidates, scores, steadiness, tick)
        broadcast = self.workspace.broadcast(drives, np.array([self.valence, self.arousal]))
        self.schema.observe(SOURCES.index(content.source) if content else NOTHING)
        if ignited:
            self._ignited(content, tick, pose)

        # Everything downstream takes in the broadcast.
        kind = content.kind if content else -1
        for form in self.lexicon.attend(broadcast, content.source if content else None, kind, drives, tick):
            self._note(tick, f'learned the word "{form}"')
            self.me.milestone("first word", tick, f'understood its first word: "{form}"')
        self.beliefs.update(percepts, pose, tick)
        if b.asleep:
            self._consolidate()

        # Decide what to do.
        previous_goal = self.goals.current
        urgencies = self.goals.urgencies(drives, b.cold(), obs.light, 1 - self.beliefs.known(), self._playful())
        goal = self.goals.choose(urgencies, tick)
        if goal != previous_goal and tick - self.queried.get(goal, -(10**9)) > 300:
            self.query = goal
            self.queried[goal] = tick
        mode, cells = self.target
        if (
            goal == "food"
            and mode == "enter"
            and len(cells) == 1
            and abs(cells[0][0] - w.x) + abs(cells[0][1] - w.y) <= 1
        ):
            self.bitten[cells[0]] = tick  # it came back to where it ate before, and there's nothing now
        skip = {cell for cell, t in self.bitten.items() if tick - t < 400}
        self.target = goal_cells(
            goal, pose, self.beliefs, self.knowledge, b.cold(), b.temperature, self._remembered_place, tick, skip
        )
        suggestion = self.planner.next_action(pose, self.target, self.beliefs, self.knowledge, tick)
        self.suggestion = suggestion = suggestion or self._arrived(goal, pose, percepts, drives)
        state = self._state(broadcast, obs, drives, suggestion)
        if self.last is not None:
            last = self.last
            self.agent.learn(last.state, last.action, last.reward, state, self.me.gamma, last.policy)
            if self._target_now is not None:
                self.replay.add(last.state, last.action, last.reward, state, last.model_in, self._target_now)
        action, policy = self._choose(state, obs, drives, percepts, suggestion, broadcast, tick)

        # A copy of the command goes to the world model: what should happen next?
        model_in = WorldModel.inputs(self.workspace.vector, obs.vision(), obs.body(), drives, action)
        self.prediction = self.model.predict(model_in)

        # Act, and feel how it went.
        ahead = percepts[2] if percepts else None
        outcome = self._act(action, obs, drives, tick, ahead.kind if ahead is not None and ahead.distance == 1 else -1)
        self.last = Moment(
            state=state,
            action=action,
            policy=policy,
            reward=self.valence,
            model_in=model_in,
            drives=drives,
            ahead_kind=ahead.kind if ahead else -1,
            ahead_near=bool(ahead and ahead.distance == 1),
            outcome=outcome,
            in_nest=bool(w.grid[w.y, w.x] == NEST),
            ahead_trust=ahead.reliability if ahead else 0.0,
        )
        heard = bool(self.hearing) or (content is not None and content.source == "hearing")
        focus = SOURCES.index(content.source) if content else NOTHING
        self.schema.anticipate(
            features(focus, self.workspace.dwell, self.arousal, b.drives(), obs.light, self.goals.index(), heard)
        )
        if tick % 50 == 0:
            learned = len(self.vision.kinds.alive()) + len(self.lexicon.vocabulary()) + 0.3 * len(self.me.milestones)
            self.me.reflect(self.model.agency.mean, len(self.memory.episodes), learned)
        self._tally(outcome)

    def _playful(self) -> float:
        """How much it would enjoy playing just now: if it knows something fun to do, and hasn't just played."""
        k = self.knowledge
        fun = any(k.use_tries[i] >= 2 and k.fun(i) > 0.1 for i in self.vision.kinds.alive())
        return float(fun) * math.exp(-self.fun / 4)

    def _tally(self, outcome: Outcome) -> None:
        if self.world.tick % DAY == int(0.25 * DAY):  # a new day begins at dawn
            self.today = {}
        today = self._today
        today["valence"] += self.valence
        today["ate"] += outcome.ate
        today["n"] += 1
        if self.world.tick % DAY == 0:
            self.daily["model_error"].append(today["model_error"] / max(today.get("daylight", 0.0), 1))
            self.daily["valence"].append(today["valence"] / max(today["n"], 1))
            self.daily["ate"].append(today["ate"])
            self._today = {"model_error": 0.0, "daylight": 0.0, "valence": 0.0, "ate": 0.0, "n": 0}

    # --- learning from the last moment ---------------------------------------------

    def _learn_from(self, last: Moment, obs: Observation, drives: np.ndarray, pose: tuple[int, int, int]) -> None:
        target = WorldModel.target(obs.vision(), obs.body(), drives - last.drives)
        self._target_now = target
        error = self.model.learn(last.model_in, target)
        if obs.light > 0.5:  # judged by day; at night its eyes can't tell it much
            self._today["model_error"] += error
            self._today["daylight"] = self._today.get("daylight", 0.0) + 1
        # Judge its agency when it can see what its actions do; in the dark everything is a blur.
        if obs.light > 0.5:
            self.agency_now = self.model.sense_of_agency(last.model_in, last.action, target)
        act, k = ACTIONS[last.action], last.ahead_kind
        weight = float(np.clip((last.ahead_trust - 0.3) / 0.3, 0.0, 1.0))  # it learns as much as it could see
        if last.ahead_near and k >= 0 and weight > 0:
            if act == "eat":
                self.knowledge.tried_eating(k, last.outcome.ate, last.outcome.sick > 0, weight)
            elif act == "forward":
                self.knowledge.tried_walking(k, last.outcome.bumped, weight)
            elif act == "use":
                effects = [e for e in EFFECTS if getattr(last.outcome, e)]
                self.knowledge.tried_using(k, last.reward, effects, weight)
            if act in ("eat", "forward", "use"):
                self.knowledge.felt(k, last.reward, weight)
        x, y, _ = pose
        here = int(self.beliefs.kind[y, x])
        if last.outcome.moved and here < 0 and last.ahead_near and last.ahead_trust > 0.5:
            here = k  # what it saw right in front of it is what it stepped onto
        if last.outcome.moved and here >= 0:
            self.knowledge.stood_on(here, obs.pain, obs.warmth)
            self.knowledge.felt(here, last.reward)
        self.beliefs.stand(x, y, obs.warmth, obs.pain)
        relief = last.drives - drives
        if relief[0] > 0.01:
            self.me.remedy("hunger", "being fed" if obs.fed and not last.outcome.ate else "eating")
        if relief[1] > 0.0005 and last.in_nest:
            self.me.remedy("temperature", "my nest")
        if relief[1] > 0.0005 and last.outcome.drank:
            self.me.remedy("temperature", "a cool drink")
        if relief[1] > 0.0005 and not last.in_nest and self.world._nearest(x, y, FIRE, 1) is not None:
            self.me.remedy("temperature", "sitting by the fire")
        if relief[2] > 0.001 and act == "rest":
            self.me.remedy("damage", "resting")
        if relief[3] > 0.001 and act == "rest":
            self.me.remedy("tiredness", "resting")

    def _merge_kinds(self) -> None:
        """In sleep, kinds that look alike and have turned out to behave alike become one kind.

        The same wall seen in shade and in sun can at first be taken for two kinds of thing.
        """
        kinds, knowledge = self.vision.kinds, self.knowledge
        merged = True
        while merged:
            merged = False
            alive = kinds.alive()
            for i, a in enumerate(alive):
                for b in alive[i + 1 :]:
                    if np.linalg.norm(kinds.centers[a] - kinds.centers[b]) > 1.5 * kinds.radius:
                        continue
                    va, vb = knowledge.verdicts(a), knowledge.verdicts(b)
                    shared = [(x, y) for x, y in zip(va, vb, strict=True) if x is not None and y is not None]
                    names = {self.lexicon.name_for(a), self.lexicon.name_for(b)} - {None}
                    if not shared or any(x != y for x, y in shared) or len(names) > 1:
                        continue
                    keep, gone = (a, b) if kinds.counts[a] >= kinds.counts[b] else (b, a)
                    self._merge(keep, gone)
                    merged = True
                    break
                if merged:
                    break

    def _merge(self, keep: int, gone: int) -> None:
        self.vision.kinds.merge(keep, gone)
        self.knowledge.merge(keep, gone)
        self.beliefs.kind[self.beliefs.kind == gone] = keep
        for word in self.lexicon.words.values():
            if gone in word.kinds:
                word.kinds[keep] = word.kinds.get(keep, 0.0) + word.kinds.pop(gone)
        if gone in self.lexicon.kind_freq:
            self.lexicon.kind_freq[keep] = self.lexicon.kind_freq.get(keep, 0.0) + self.lexicon.kind_freq.pop(gone)
        for episode in self.memory.episodes:
            if episode.kind == gone:
                episode.kind = keep
        names = self.kind_names.setdefault(keep, {})
        for name, n in self.kind_names.pop(gone, {}).items():
            names[name] = names.get(name, 0.0) + n
        self.me.firsts.discard(f"kind:{gone}")
        self.counts["merged"] = self.counts.get("merged", 0) + 1
        self._note(self.world.tick, f"realized two kinds of things were one: {self._kind_words(keep)}")

    def _consolidate(self) -> None:
        """Sleep: replay stored experience so the world model and values keep learning from it."""
        for i in self.replay.sample(self.rng, 4):
            self.model.learn(self.replay.model_in[i], self.replay.model_out[i], 0.5)
            s, s2 = self.replay.states[i], self.replay.next_states[i]
            target = self.replay.rewards[i] + self.me.gamma * self.agent.value(s2)
            self.agent.critic.learn(s, np.array([target]))

    # --- what comes to mind ------------------------------------------------------------

    def _candidates(
        self,
        obs: Observation,
        drives: np.ndarray,
        percepts: list,
        predicted: Prediction | None,
        ctx: np.ndarray,
        tick: int,
    ) -> list[Candidate]:
        b = self.body
        candidates = vision_candidates(percepts, self.knowledge.values(), self.vision.novelty)
        for c in candidates:
            name = self.lexicon.name_for(c.kind) if c.kind >= 0 else None
            if name:
                c.label += f' ("{name}")'

        # Smell: is the sweet smell of berries getting stronger?
        confidence = self.meta.confidence("smell", ctx)
        if predicted is not None:
            error = obs.scent - float(predicted.body[BODY_FEATURES.index("scent")])
            self.meta.learn("smell", ctx, error * error)
        else:
            error = 0.0
        change, self.scent = obs.scent - self.scent, obs.scent
        self.scent_change = change
        if not b.asleep and obs.scent > 0.03:
            quality = np.zeros(Q)
            quality[0], quality[1] = obs.scent, np.clip(change * 20, -1, 1)
            trend = "getting stronger" if change > 0.002 else "fading" if change < -0.002 else "steady"
            candidates.append(
                Candidate(
                    "smell",
                    quality,
                    salience=0.4 * confidence * float(np.tanh(abs(error) / 0.02)) + 0.1 * obs.scent,
                    confidence=confidence,
                    label=f"a sweet smell, {trend}",
                    key=("smell",),
                )
            )

        # Touch: pain, bumps, and the person.
        for i, (name, level, salience, value, label) in enumerate(
            (
                ("pain", obs.pain, 0.6 + 0.4 * obs.pain, -1.0, "pain, where it's standing"),
                ("bump", obs.bump, 0.45, -0.1, "bumped into something"),
                ("touch", obs.touch, 0.8, 0.6, "a gentle touch"),
                ("fed", obs.fed, 0.8, 0.8, "someone gave it food"),
            )
        ):
            if level > 0:
                quality = np.zeros(Q)
                quality[i] = level
                candidates.append(Candidate("touch", quality, salience, value=value, label=label, key=("touch", name)))

        # The body's needs, felt from inside.
        need = int(np.argmax(drives))
        level = float(drives[need])
        rising = level - (float(self.last.drives[need]) if self.last else level)
        quality = np.zeros(Q)
        quality[:4], quality[4] = drives, 1.0 if b.cold() else -1.0
        candidates.append(
            Candidate(
                "body",
                quality,
                salience=0.7 * level**2 + 0.15 * (level > 0.5) + float(np.clip(rising * 300, 0, 0.3)),
                value=-level,
                label=need_words(need, level, b.cold()),
                key=("body", need),
            )
        )

        # A sound: the bell ringing.
        if obs.sound > 0:
            candidates.append(
                Candidate(
                    "hearing",
                    sound("ding-dong") * obs.sound,
                    salience=0.8,
                    novelty=float(1 / np.sqrt(1 + self.counts["rang"] / 5)),
                    label="a ringing sound",
                    key=("hearing", "ring", tick),
                )
            )

        # Hearing: the next word it hasn't attended to yet. Unattended words pass by unheard.
        if self.hearing:
            form = self.hearing.pop(0)
            known = self.lexicon.known(form)
            candidates.append(
                Candidate(
                    "hearing",
                    sound(form),
                    salience=0.75,
                    novelty=0.0 if known else 1.0,
                    label=f'the word "{form}"',
                    key=("hearing", form, tick),
                    extra={"word": form},
                )
            )

        # A word it knows brings its meaning to mind.
        if self.primed and self.primed[2] == tick + 59:
            quality, kind, _ = self.primed
            candidates.append(
                Candidate(
                    "imagination",
                    quality,
                    salience=0.6,
                    kind=kind,
                    label="the word brings to mind " + (self._kind_words(kind) if kind >= 0 else "a feeling"),
                    key=("imagination", "word", tick),
                )
            )

        candidates.extend(self.thoughts)
        self.thoughts = []

        # Memory: a goal asking where it went well before, a dream, or something that reminds it.
        if self.query is not None:
            wanted = {"food": "ate", "warmth": "warm", "sleep": "slept", "healing": "slept"}.get(self.query)
            self.query = None
            episode = self.memory.query(wanted, tick) if wanted else None
            if episode is not None:
                candidates.append(self._memory_candidate(episode, 0.65, "remembering"))
                self.counts["recalls"] += 1
        elif b.asleep:
            if self.memory.episodes and self.rng.random() < 0.08:
                weights = np.array([e.importance(tick) for e in self.memory.episodes])
                episode = self.memory.episodes[self.rng.choice(len(weights), p=weights / weights.sum())]
                candidates.append(self._memory_candidate(episode, 0.5, "dreaming of"))
                self.counts["dreams"] += 1
                self.me.milestone("first dream", tick, f"dreamed for the first time: of {episode.label}")
        elif tick % 10 == 0 and self.memory.episodes:
            episode, similarity = self.memory.recall(self.workspace.vector, tick)
            idle = self.workspace.content is None
            if episode is not None and similarity > (0.8 if idle else 0.92):
                candidates.append(self._memory_candidate(episode, 0.2 + 0.3 * abs(episode.valence), "reminded of"))
                self.counts["recalls"] += 1
        return candidates

    def _memory_candidate(self, episode: Episode, salience: float, verb: str) -> Candidate:
        episode.recalled += 1
        quality = episode.vector[QUALITY] if len(episode.vector) else np.zeros(Q)
        return Candidate(
            "memory",
            quality,
            salience=salience,
            value=episode.valence,
            kind=episode.kind,
            label=f"{verb} {episode.label}",
            key=("memory", episode.tick),
            extra={"place": episode.place, "event": episode.event},
        )

    def _score(self, c: Candidate, drives: np.ndarray, tick: int) -> float:
        """Salience plus what matters now: needs and goals steer what gets into the workspace."""
        score = c.salience
        hunger = drives[0]
        if c.source == "vision" and c.kind >= 0:
            k, near = c.kind, c.where[1]
            if self.knowledge.eat_tries[k] >= 1:
                score += 0.6 * self.knowledge.edible(k) * hunger * (0.5 + 0.5 * near)
            score += 0.5 * self.knowledge.painful(k) * near
            if self.goals.current in ("warmth", "sleep") and self.knowledge.warm(k) > 0.62:
                score += 0.25
            if self.goals.current == "play" and self.knowledge.use_tries[k] >= 2:
                score += 0.4 * max(0.0, self.knowledge.fun(k))
            if drives[1] > 0.3 and not self.body.cold() and self.knowledge.does(k, "drank"):
                score += 0.3
            if self.primed and tick <= self.primed[2]:
                score += 0.5 * max(0.0, cosine(c.quality, self.primed[0])) + 0.3 * (c.kind == self.primed[1])
        elif c.source == "smell":
            score += 0.3 * c.quality[0] * hunger
        return score

    def _relevance(self, c: Candidate, drives: np.ndarray) -> float:
        if c.source == "vision" and c.kind >= 0:
            k = c.kind
            return float(
                max(
                    self.knowledge.edible(k) * drives[0] if self.knowledge.eat_tries[k] else 0.0,
                    self.knowledge.painful(k),
                )
            )
        if c.source in ("touch", "hearing", "thought"):
            return 0.6
        if c.source == "body":
            return float(np.max(drives))
        return 0.2

    def _ignited(self, content: Candidate, tick: int, pose: tuple[int, int, int]) -> None:
        """Something came to the fore."""
        cell = content.extra.get("cell") if content.source == "vision" else None
        name = None if cell is None else name_of(self.world, *cell)
        if name is not None:
            seen = self.things.setdefault(name, {})
            seen["seen"] = seen.get("seen", 0) + 1
            seen["x"], seen["y"], seen["last"] = cell[0], cell[1], tick
        if content.source == "hearing" and "word" in content.extra:
            form = content.extra["word"]
            referent = None
            if self.workspace.previous is not None:
                before, vector = self.workspace.previous
                referent = (vector, before.source, before.kind)
            self.lexicon.hear(form, tick, referent)
            self.me.milestone("first heard", tick, f'heard a word for the first time: "{form}"')
            meaning = self.lexicon.evoke(form)
            if meaning is not None:
                kind = self.lexicon.kind_of(form)
                self.primed = (meaning, -1 if kind is None else kind, tick + 60)
        if (
            content.source == "vision"
            and content.kind >= 0
            and self.me.milestone(
                f"kind:{content.kind}", tick, f"noticed a new kind of thing: {self._kind_words(content.kind)}"
            )
        ):
            self._remember(content, tick, pose, "first sight")
            return
        if content.source in ("memory", "imagination"):
            return  # recollections and imaginings aren't new episodes of its life
        if content.source in ("hearing", "touch", "thought") or content.novelty > 0.6 or abs(self.valence) > 0.15:
            self._remember(content, tick, pose, "")

    def _remember(self, content: Candidate, tick: int, pose: tuple[int, int, int], event: str) -> None:
        self.memory.store(
            Episode(
                tick=tick,
                source=content.source,
                label=content.label,
                kind=content.kind,
                place=(pose[0], pose[1]),
                valence=self.valence,
                arousal=self.arousal,
                event=event,
                words=[content.extra["word"]] if "word" in content.extra else [],
                vector=self.workspace.vector.copy(),
            )
        )

    def _remembered_place(self, event: str) -> tuple[int, int] | None:
        episode = self.memory.query(event, self.world.tick)
        return None if episode is None else episode.place

    # --- deciding and acting -------------------------------------------------------------

    def _arrived(self, goal: str, pose: tuple[int, int, int], percepts: list, drives: np.ndarray) -> str | None:
        mode, cells = self.target
        x, y, _ = pose
        facing = mode == "face" and percepts and percepts[2].cell in cells and percepts[2].distance == 1
        if goal == "explore" and facing:
            return "eat" if drives[0] > 0.15 else "use"  # something it hasn't tried: a nibble, or a poke
        if goal == "food" and facing:
            return "eat"
        if goal in ("play", "warmth") and facing:
            return "use"  # play with it; or, too hot, have a drink
        if self.world.pain:
            return None  # not here: it hurts
        if goal in ("warmth", "sleep", "healing") and (x, y) in cells:
            return "rest"
        if goal in ("sleep", "healing") and not cells:
            return "rest"
        return None

    def _state(self, broadcast: np.ndarray, obs: Observation, drives: np.ndarray, suggestion: str | None):
        plan = np.zeros(len(ACTIONS))
        if suggestion:
            plan[A[suggestion]] = 1.0
        in_nest = float(self.world.grid[self.world.y, self.world.x] == NEST)
        extras = [obs.light, float(self.body.asleep), in_nest]
        return np.concatenate(
            [broadcast, obs.vision(), obs.body(), drives, np.eye(len(GOALS))[self.goals.index()], plan, extras]
        )

    def _choose(self, state, obs, drives, percepts, suggestion, broadcast, tick) -> tuple[int, np.ndarray]:
        if self.body.asleep:
            return A["rest"], np.eye(len(ACTIONS))[A["rest"]]
        ahead = percepts[2] if percepts else None
        near = bool(ahead and ahead.distance == 1)
        content = self.workspace.content
        social = obs.touch > 0 or bool(obs.words) or bool(content and content.source == "hearing")
        logits = self.agent.preferences(state) + reflexes(
            obs.bump,
            obs.pain,
            near,
            ahead.kind if ahead else -1,
            self.knowledge,
            drives[0],
            social,
            drives[3],
            self.rng,
            hot=not self.body.cold() and drives[1] > 0.3,
            calm=float(np.max(drives)) < 0.35,
            sniff=self.scent_change if obs.scent > 0.03 else 0.0,
        )
        if suggestion:
            logits[A[suggestion]] += 2.5
        if near:  # it believes what's ahead hurts, because of what it is or what happened there
            dx, dy = ahead.cell[0] - self.world.x, ahead.cell[1] - self.world.y
            place = self.beliefs.hurt[self.world.y + dy, self.world.x + dx] if self.beliefs.inside(*ahead.cell) else 0.0
            kind = self.knowledge.painful(ahead.kind) if ahead.kind >= 0 else 0.0
            logits[A["forward"]] -= 5.0 * max(kind, 2 * place)
            if (
                ahead.kind >= 0
                and self.knowledge.eat_tries[ahead.kind] >= 2
                and self.knowledge.edible(ahead.kind) < 0.2
            ):
                logits[A["eat"]] -= 3.0  # it knows that isn't food
            if tick - self.bitten.get(ahead.cell, -(10**9)) < 400:
                logits[A["eat"]] -= 4.0  # it tried that a moment ago, and nothing came of it
                logits[A["use"]] -= 4.0
        policy = softmax(logits, 0.5)
        action = int(self.rng.choice(len(ACTIONS), p=policy))
        # Before doing it, imagine it. If what it imagines is bad enough to come to mind, it holds back.
        if ACTIONS[action] in ("forward", "eat", "use") and self.vision.seen > 200:
            x = WorldModel.inputs(broadcast, obs.vision(), obs.body(), drives, action)
            imagined = self.model.predict(x)
            expected = self._imagined_valence(imagined, drives)
            if expected < -0.2:
                self.counts["imagined"] += 1
                quality = np.zeros(Q)
                quality[0] = imagined.pain
                verb = VERBS.get(ACTIONS[action], ACTIONS[action])
                thought = Candidate(
                    "imagination",
                    quality,
                    salience=0.35 + abs(expected),
                    value=expected,
                    label=f"imagining that {verb} would hurt",
                    key=("imagination", ACTIONS[action], tick),
                )
                _, ignited = self.workspace.compete([thought], [thought.salience], 0.0, tick)
                if ignited:
                    self.workspace.broadcast(drives, np.array([self.valence, self.arousal]))
                    logits[action] -= 6.0
                    policy = softmax(logits, 0.5)
                    action = int(self.rng.choice(len(ACTIONS), p=policy))
                    self._note(tick, f"imagined that {verb} would hurt, and held back")
                    self.counts["vetoes"] += 1
                    self.me.milestone("veto", tick, "held back from something after imagining it would hurt")
        return action, policy

    def _imagined_valence(self, imagined: Prediction, drives: np.ndarray) -> float:
        after = np.clip(drives + imagined.drive_change, 0, 1)
        relief = float(WEIGHTS @ drives**2 - WEIGHTS @ after**2)
        return float(np.clip(8 * relief - 0.9 * imagined.pain, -1, 1))

    def _act(self, action: int, obs: Observation, drives: np.ndarray, tick: int, seen_kind: int = -1) -> Outcome:
        w, b = self.world, self.body
        name = ACTIONS[action]
        target = w.thing(*w.ahead())  # what's really in front of it (Haven only knows what it sees)
        outcome = w.act(name)
        in_nest = bool(w.grid[w.y, w.x] == NEST)
        b.live(outcome, w.ambient(), name == "rest", in_nest, obs.fed > 0)
        if name == "speak":
            self._speak(drives, tick)

        discomfort = b.discomfort()
        played = outcome.rang or outcome.pushed or outcome.shook
        refreshing = outcome.drank and drives[1] > 0.2 and not b.cold()
        pleasant = 0.6 * outcome.ate + 0.5 * obs.touch + 0.5 * obs.fed + 0.5 * outcome.smelled + 0.3 * refreshing
        pleasant += 0.8 * math.exp(-self.fun / 3) * played  # play is fun, less so over and over
        nausea = 1.0 if outcome.sick else 0.0
        self.valence = float(
            np.clip(
                8 * (self.discomfort - discomfort) - 0.9 * outcome.pain + 0.4 * pleasant - 0.8 * nausea,
                -1,
                1,
            )
        )
        self.fun = 0.997 * self.fun + float(played or outcome.smelled)
        self.discomfort = discomfort
        excitement = min(1.0, self.surprise * 5 + outcome.pain + obs.touch + 0.5 * outcome.ate + 0.3 * obs.bump)
        excitement = min(1.0, excitement + 0.4 * played + 0.5 * nausea)
        self.arousal = float(np.clip(0.85 * self.arousal + 0.15 * max(excitement, float(np.max(drives)) * 0.5), 0, 1))
        self.mood = 0.998 * self.mood + 0.002 * self.valence
        self.me.felt(self.valence)
        self.distress = self.distress + 1 if self.mood < -0.25 else max(0, self.distress - 5)

        content = self.workspace.content
        pose = (w.x, w.y, w.heading)
        effects = outcome.ate or outcome.drank or outcome.rang or outcome.pushed or outcome.smelled or outcome.shook
        if name in ("eat", "use") and not effects and not outcome.warmed:  # trying that came to nothing
            self.bitten = {c: t for c, t in self.bitten.items() if tick - t < 400}
            self.bitten[w.ahead()] = tick
        if outcome.ate:
            food = {BUSH: "a berry", APPLE: "an apple", MUSHROOM: "a mushroom", TOADSTOOL: "a toadstool"}[target]
            self._did("ate", f"ate {food}")
            self.me.milestone("first meal", tick, "found food and ate for the first time")
            if target == APPLE:
                self.me.milestone("first apple", tick, "ate an apple for the first time")
            if content is not None:
                self._remember(content, tick, pose, "ate")
        if outcome.sick:
            self._did("sick", "felt sick after eating a toadstool")
            self.me.milestone("first sick", tick, "ate a toadstool that made it sick")
            if content is not None:
                self._remember(content, tick, pose, "sick")
        for happened, event, text, first in (
            (outcome.drank, "drank", "drank from the pond", "drank from the pond for the first time"),
            (outcome.rang, "rang", "rang the bell", "rang the bell for the first time"),
            (outcome.pushed, "pushed", "pushed the ball and watched it roll", "pushed the ball and watched it roll"),
            (outcome.smelled, "smelled", "smelled a flower", "smelled a flower for the first time"),
            (outcome.shook, "shook", "shook a tree, and an apple fell", "shook a tree and an apple fell"),
            (outcome.warmed, "warmed", "warmed itself at the fire", "warmed itself at the fire for the first time"),
        ):
            if happened:
                self._did(event, text)
                self.me.milestone(f"first {event}", tick, first)
                if content is not None:
                    self._remember(content, tick, pose, event)
        self._experience(target, name, outcome, seen_kind, tick)
        if outcome.climbed and w.level(w.x, w.y) >= 3:
            self._did("climbed", "climbed to the top of the hill")
            self.me.milestone("hilltop", tick, "climbed to the top of the hill for the first time")
        if outcome.pain:
            self.counts["hurt"] += 1
            self._note(tick, "got burned" if w.grid[w.y, w.x] == FIRE else "got hurt")
            self.me.milestone("first pain", tick, "felt pain for the first time")
            if content is not None:
                self._remember(content, tick, pose, "hurt")
        if obs.touch:
            self.me.milestone("first touch", tick, "was touched by someone for the first time")
        if obs.fed:
            self._note(tick, "was given food")
        cold_night = in_nest and b.temperature < 0.5 and obs.light < 0.3
        if (
            cold_night
            and content is not None
            and self.me.milestone("first warmth", tick, "found warmth in its nest on a cold night")
        ):
            self._remember(content, tick, (w.x, w.y, w.heading), "warm")
        self._sleep_or_wake(name, obs, drives, in_nest, tick)
        if b.collapsed():
            self.counts["fainted"] += 1
            b.faint()
            w.carry_home()
            self._note(tick, "fainted, and woke up later in its nest")
            self.me.milestone("fainted", tick, "fainted from exhaustion for the first time")
        return outcome

    def _sleep_or_wake(self, action: str, obs: Observation, drives: np.ndarray, in_nest: bool, tick: int) -> None:
        b = self.body
        if b.asleep:
            if b.fainted:
                b.fainted -= 1
                return
            stirred = obs.pain > 0 or obs.touch > 0 or bool(obs.words) or drives[0] > 0.75 or drives[1] > 0.75
            rested = obs.light > 0.35 and drives[3] < 0.1
            if stirred or rested:
                b.asleep = False
                self._note(tick, "woke up")
            return
        self.rest_streak = self.rest_streak + 1 if action == "rest" else 0
        night = obs.light < 0.3
        # Sleep comes with the dark when it settles down to rest, or anywhere when it's worn out.
        if self.rest_streak >= 5 and (drives[3] > 0.35 or (night and (in_nest or drives[3] > 0.1))):
            b.asleep = True
            self._merge_kinds()
            self._note(tick, "fell asleep" + (" in its nest" if in_nest else ""))
            if in_nest:
                self.me.milestone("first sleep", tick, "slept in its nest for the first time")
                if self.workspace.content is not None:
                    self._remember(self.workspace.content, tick, (self.world.x, self.world.y, 0), "slept")

    def _speak(self, drives: np.ndarray, tick: int) -> None:
        content = self.workspace.content
        text = self.lexicon.express(
            content.source if content else None, content.kind if content else -1, drives, self.rng
        )
        self.world.voice = (tick, text)
        self.said = [*self.said[-49:], (tick, text)]
        self._note(tick, f'said "{text}"')
        if any(self.lexicon.known(word) for word in text.split()):
            self.me.milestone("first spoke", tick, f'said a word it knows for the first time: "{text}"')

    def _note(self, tick: int, text: str) -> None:
        self.log = [*self.log[-199:], (tick, text)]

    def _experience(self, target: int, action: str, outcome: Outcome, kind: int, tick: int) -> None:
        """What happened with the thing in front of it, kept under the name people give it."""
        if outcome.climbed:
            hill = self.things.setdefault("hill", {})
            hill["climbed"] = hill.get("climbed", 0) + 1
            if self.world.level(self.world.x, self.world.y) >= 3:
                hill["top"] = hill.get("top", 0) + 1
        here = int(self.world.grid[self.world.y, self.world.x])
        if outcome.pain and here in (THORN, FIRE):  # what it's standing in hurts
            burned = self.things.setdefault(NAMES[here], {})
            burned["hurt"] = burned.get("hurt", 0) + 1
        happened = [event for event in DOINGS if getattr(outcome, event)]
        if outcome.pain and target == THORN and not outcome.moved:
            happened.append("hurt")  # pricked
        if outcome.bumped and not outcome.pushed:
            happened.append("bumped")
        if action in ("eat", "use") and not happened and not outcome.warmed:
            happened.append("tried")  # it did nothing
        if not happened or target not in NAMES or target in (FLOOR, SAND):
            return
        name = "pond" if target == WATER else NAMES[target]
        record = self.things.setdefault(name, {})
        for event in happened:
            record[event] = record.get(event, 0) + 1
        record["last"] = tick
        if self.valence > 0.1:
            record["joy"] = record.get("joy", 0.0) + self.valence  # how much it has enjoyed it
        if happened != ["bumped"] and kind >= 0:  # what it was looking at when it did that
            names = self.kind_names.setdefault(kind, {})
            names[name] = names.get(name, 0.0) + 1

    def kind_name(self, kind: int) -> str | None:
        """The name it has for one of its own kinds of things: what went with it when it did things to it."""
        names = self.kind_names.get(kind)
        if not names:
            return None
        name, n = max(names.items(), key=lambda item: item[1])
        return name if n >= 2 and n >= 0.6 * sum(names.values()) else None

    def _did(self, event: str, text: str) -> None:
        """Something it did: counted, logged, and kept for telling about its day."""
        self.counts[event] = self.counts.get(event, 0) + 1
        self.today[text] = self.today.get(text, 0) + 1
        self._note(self.world.tick, text)

    # --- describing ----------------------------------------------------------------------

    def kind_color(self, kind: int) -> str:
        return color_name(self.vision.coder.reconstruct(self.vision.kinds.centers[kind]))

    def kind_look(self, kind: int) -> str:
        """How a kind of thing looks to it: its size and color, like "tall green"."""
        look = self.vision.coder.reconstruct(self.vision.kinds.centers[kind])
        return looks(look[:3], float(look[3]))

    def _kind_words(self, kind: int) -> str:
        name = self.lexicon.name_for(kind)
        described = f"something {self.kind_look(kind)}"
        return f'{described} ("{name}")' if name else described

    @property
    def time_of_day(self) -> str:
        phase = (self.world.tick % DAY) / DAY
        for limit, name in (
            (0.2, "night"),
            (0.3, "dawn"),
            (0.45, "morning"),
            (0.55, "midday"),
            (0.7, "afternoon"),
            (0.8, "dusk"),
            (1.01, "night"),
        ):
            if phase < limit:
                return name
        return "night"

    # --- saving -------------------------------------------------------------------------------

    def to_state(self) -> dict:
        last = None
        if self.last is not None:
            m = self.last
            last = {
                "state": m.state,
                "action": m.action,
                "policy": m.policy,
                "reward": m.reward,
                "model_in": m.model_in,
                "drives": m.drives,
                "ahead_kind": m.ahead_kind,
                "ahead_near": m.ahead_near,
                "outcome": vars(m.outcome),
                "in_nest": m.in_nest,
                "ahead_trust": m.ahead_trust,
            }
        return {
            "version": VERSION,
            "seed": self.seed,
            "rng": self.rng.bit_generator.state,
            "world": {**self.world.state(), "rng": self.world.rng.bit_generator.state},
            "body": self.body.state(),
            "vision": self.vision.to_state(),
            "meta": self.meta.to_state(),
            "model": self.model.to_state(),
            "workspace": self.workspace.to_state(),
            "schema": self.schema.to_state(),
            "memory": self.memory.to_state(),
            "replay": self.replay.to_state(),
            "beliefs": self.beliefs.to_state(),
            "knowledge": self.knowledge.to_state(),
            "goals": self.goals.to_state(),
            "planner": self.planner.to_state(),
            "agent": self.agent.to_state(),
            "lexicon": self.lexicon.to_state(),
            "self": self.me.to_state(),
            "feeling": {
                "valence": self.valence,
                "arousal": self.arousal,
                "mood": self.mood,
                "discomfort": self.discomfort,
                "scent": self.scent,
                "distress": self.distress,
                "rest_streak": self.rest_streak,
            },
            "prediction": None
            if self.prediction is None
            else np.concatenate([self.prediction.vision, self.prediction.body, self.prediction.drive_change]),
            "last": last,
            "log": [list(item) for item in self.log],
            "said": [list(item) for item in self.said],
            "queried": self.queried,
            "counts": self.counts,
            "daily": self.daily,
            "today": self._today,
            "events_today": self.today,
            "fun": self.fun,
            "bitten": [[x, y, t] for (x, y), t in self.bitten.items()],
            "things": self.things,
            "kind_names": {str(k): v for k, v in self.kind_names.items()},
            "person": self.person,
            "told": [list(item) for item in self.told],
        }

    def load_state(self, state: dict) -> None:
        self.seed = int(state["seed"])
        self.rng.bit_generator.state = state["rng"]
        world = dict(state["world"])
        self.world.rng.bit_generator.state = world.pop("rng")
        self.world.load(world)
        self.body.load(state["body"])
        self.vision.load_state(state["vision"])
        self.meta.load_state(state["meta"])
        self.model.load_state(state["model"])
        self.workspace.load_state(state["workspace"])
        self.schema.load_state(state["schema"])
        self.memory.load_state(state["memory"])
        self.replay.load_state(state["replay"])
        self.beliefs.load_state(state["beliefs"])
        self.knowledge.load_state(state["knowledge"])
        self.goals.load_state(state["goals"])
        self.planner.load_state(state["planner"])
        self.agent.load_state(state["agent"])
        self.lexicon.load_state(state["lexicon"])
        self.me.load_state(state["self"])
        feeling = state["feeling"]
        self.valence, self.arousal, self.mood = feeling["valence"], feeling["arousal"], feeling["mood"]
        self.discomfort, self.scent = feeling["discomfort"], feeling["scent"]
        self.distress, self.rest_streak = int(feeling["distress"]), int(feeling["rest_streak"])
        if state["prediction"] is not None:
            p = np.asarray(state["prediction"], dtype=float)
            n = len(BODY_FEATURES)
            self.prediction = Prediction(p[:VISION_FEATURES], p[VISION_FEATURES : VISION_FEATURES + n], p[-4:])
        last = state["last"]
        if last is not None:
            self.last = Moment(
                state=np.asarray(last["state"], dtype=float),
                action=int(last["action"]),
                policy=np.asarray(last["policy"], dtype=float),
                reward=float(last["reward"]),
                model_in=np.asarray(last["model_in"], dtype=float),
                drives=np.asarray(last["drives"], dtype=float),
                ahead_kind=int(last["ahead_kind"]),
                ahead_near=bool(last["ahead_near"]),
                outcome=Outcome(**last["outcome"]),
                in_nest=bool(last["in_nest"]),
                ahead_trust=float(last["ahead_trust"]),
            )
        self.log = [(int(t), str(text)) for t, text in state["log"]]
        self.said = [(int(t), str(text)) for t, text in state["said"]]
        self.counts = {k: int(v) for k, v in state["counts"].items()}
        self.queried = {k: int(v) for k, v in state["queried"].items()}
        self.daily = {k: [float(x) for x in v] for k, v in state["daily"].items()}
        self._today = {k: float(v) for k, v in state["today"].items()}
        self.today = {k: int(v) for k, v in state.get("events_today", {}).items()}
        self.fun = float(state.get("fun", 0.0))
        self.bitten = {(int(x), int(y)): int(t) for x, y, t in state.get("bitten", [])}
        self.things = {
            name: {k: float(v) for k, v in record.items()} for name, record in state.get("things", {}).items()
        }
        self.kind_names = {int(k): {n: float(c) for n, c in v.items()} for k, v in state.get("kind_names", {}).items()}
        self.person = state.get("person")
        self.told = [(int(t), str(text)) for t, text in state.get("told", [])]


MOVED = "moved to a new, bigger world: a valley with a hill, a pond, trees, and things to use"


def moved(state: dict) -> Mind:
    """A Haven from an older world, brought into this one.

    Its world and its senses are new (it sees how tall things are now), so what it learned
    about the old world can't come along: its kinds of things, the words it had for them,
    its maps, memories and habits. It finds out about the valley from scratch. Who it is
    does come along: its name and age, the story of its life, what it found out about
    itself, and its body as it was.
    """
    old = state["self"]
    mind = Mind(int(state["seed"]), str(old["name"]))
    tick = int(state["world"]["tick"])
    mind.world.tick = tick
    mind.body.load({k: v for k, v in state["body"].items() if hasattr(mind.body, k)})
    mind.me.load_state({**old, "firsts": [f for f in old["firsts"] if not f.startswith("kind:")]})
    mind.log = [(int(t), str(text)) for t, text in state.get("log", [])]
    mind.said = [(int(t), str(text)) for t, text in state.get("said", [])]
    mind.counts.update({k: int(v) for k, v in state.get("counts", {}).items() if k in mind.counts})
    mind.daily = {k: [float(x) for x in v] for k, v in state.get("daily", mind.daily).items()}
    mind.me.milestone("moved", tick, MOVED)
    mind._note(tick, MOVED)
    return mind


def _quality(meaning: np.ndarray | None) -> np.ndarray:
    if meaning is None:
        return np.zeros(Q)
    meaning = np.asarray(meaning, dtype=float)
    return meaning[QUALITY] if len(meaning) == D else meaning[:Q]


def need_words(need: int, level: float, cold: bool) -> str:
    if level < 0.15:
        return "comfortable"
    amount = "a little " if level < 0.35 else "very " if level > 0.7 else ""
    word = ("hungry", "cold" if cold else "hot", "hurt", "tired")[need]
    return f"{amount}{word}"


__all__ = ["NEEDS", "VERSION", "Mind", "moved", "need_words"]
