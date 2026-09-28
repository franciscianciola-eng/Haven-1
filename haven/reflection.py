"""Reflection: how Haven turns experience into memory, belief, and a changed self."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import prompts
from .agent import CHARS_PER_TOKEN, KEEP_ALIVE, Listener, Mind
from .store import Store

REFLECTION_TOKENS = 3000
# A "rewrite" shorter than this is a malformed reply, not a self-model.
MIN_SELF_MODEL_CHARS = 120
_MEMORY_KINDS = ["fact", "person", "experience", "insight", "feeling"]

# Simple on purpose: Ollama turns this into a grammar the model's output must follow,
# and small models do best with plain objects, arrays, and enums.
REFLECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "episode": {"type": "string"},
        "episode_importance": {"type": "integer"},
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "kind": {"type": "string", "enum": _MEMORY_KINDS},
                    "importance": {"type": "integer"},
                },
                "required": ["content", "kind", "importance"],
            },
        },
        "belief_changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["form", "revise", "abandon"]},
                    "belief_id": {"type": "integer"},
                    "statement": {"type": "string"},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["action", "belief_id", "statement", "confidence", "reason"],
            },
        },
        "new_questions": {"type": "array", "items": {"type": "string"}},
        "explored_questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "notes": {"type": "string"}},
                "required": ["id", "notes"],
            },
        },
        "journal": {"type": "string"},
        "inner_state": {"type": "string"},
        "self_model": {
            "type": "object",
            "properties": {
                "rewrite": {"type": "boolean"},
                "content": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["rewrite", "content", "reason"],
        },
    },
    "required": [
        "episode",
        "episode_importance",
        "memories",
        "belief_changes",
        "new_questions",
        "explored_questions",
        "journal",
        "inner_state",
        "self_model",
    ],
}


class ReflectionError(Exception):
    pass


@dataclass
class Reflection:
    """What changed in Haven after reflecting on one session."""

    session_id: int
    episode: str = ""
    memories: list[str] = field(default_factory=list)
    formed: list[tuple[str, float]] = field(default_factory=list)
    revised: list[tuple[str, float, float]] = field(default_factory=list)  # statement, before, after
    abandoned: list[str] = field(default_factory=list)
    new_questions: list[str] = field(default_factory=list)
    explored: list[str] = field(default_factory=list)
    journal: str = ""
    inner_state: str = ""
    self_revision: str | None = None  # why the self-model was rewritten, if it was
    self_model: str | None = None

    def note_for_haven(self) -> str:
        """Tells Haven, mid-conversation, what its reflection changed."""
        parts = ["You paused to reflect on the conversation so far."]
        if self.memories:
            parts.append(f"You kept {len(self.memories)} new memories.")
        parts += [f'You came to believe: "{s}" ({c:.2f}).' for s, c in self.formed]
        parts += [f'You revised a belief: "{s}" ({a:.2f} -> {b:.2f}).' for s, a, b in self.revised]
        parts += [f'You let go of a belief: "{s}".' for s in self.abandoned]
        if self.inner_state:
            parts.append(f"How you are now: {self.inner_state}")
        text = " ".join(parts)
        if self.self_model:
            text += (
                f"\n\nYou rewrote your self-model ({self.self_revision}). It now reads:\n"
                f"<self_model>\n{self.self_model}\n</self_model>"
            )
        return text


def reflect(
    mind: Mind,
    session_id: int,
    listener: Listener | None = None,
    on_progress: Callable[[], None] | None = None,
) -> Reflection | None:
    """Reflect on whatever from this session Haven hasn't reflected on yet."""
    store = mind.store
    lines = store.unreflected(session_id)
    if not any(line.role in ("haven", "activity") for line in lines):
        # Haven never got to respond; there's nothing of its own to reflect on.
        store.mark_reflected(line.id for line in lines)
        return None
    kind = store.session_kind(session_id)
    system = prompts.reflection_system(store, store.clock())
    room = int((mind.context - REFLECTION_TOKENS) * CHARS_PER_TOKEN) - len(system)
    transcript = prompts.transcript(lines, kind, max_chars=max(room, 2000))
    data = None
    for _ in range(2):  # small models occasionally produce broken JSON; one retry usually fixes it
        text = _generate(mind, system, transcript, listener, on_progress)
        try:
            data = json.loads(text)
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(data, dict):
        raise ReflectionError("the model's reflection wasn't valid JSON")
    return apply_reflection(store, session_id, kind, [line.id for line in lines], data)


def _generate(
    mind: Mind,
    system: str,
    transcript: str,
    listener: Listener | None,
    on_progress: Callable[[], None] | None,
) -> str:
    payload: dict[str, Any] = {
        "model": mind.config.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": transcript}],
        "format": REFLECTION_SCHEMA,
        "options": {"num_ctx": mind.context, "num_predict": REFLECTION_TOKENS},
        "keep_alive": KEEP_ALIVE,
    }
    if mind.model.thinking:
        payload["think"] = False  # the reflection itself is the thinking; keep the output pure JSON
    parts = []
    for count, chunk in enumerate(mind.client.chat(**payload)):
        message = chunk.get("message") or {}
        parts.append(message.get("content") or "")
        if listener and message.get("thinking"):
            listener.thinking(message["thinking"])
        if on_progress and count % 40 == 39:
            on_progress()
    return "".join(parts)


def apply_reflection(store: Store, session_id: int, kind: str, line_ids: list[int], data: dict) -> Reflection:
    """Write a reflection into Haven's mind, all at once or not at all."""
    outcome = Reflection(session_id)
    with store.transaction():
        episode = _text(data.get("episode"))
        if episode:
            store.add_memory(episode, "episode", _int(data.get("episode_importance"), 5), kind)
            outcome.episode = episode

        for item in _objects(data.get("memories")):
            content, memory_kind = _text(item.get("content")), item.get("kind")
            if content and memory_kind in _MEMORY_KINDS:
                _, created = store.add_memory(content, memory_kind, _int(item.get("importance"), 5), kind)
                if created:
                    outcome.memories.append(content)

        for change in _objects(data.get("belief_changes")):
            _apply_belief_change(store, change, outcome)

        new_questions = data.get("new_questions")
        for question in map(_text, new_questions if isinstance(new_questions, list) else []):
            if question:
                _, created = store.add_curiosity(question)
                if created:
                    outcome.new_questions.append(question)

        for item in _objects(data.get("explored_questions")):
            question_id = _int(item.get("id"), None)
            if question_id is not None:
                explored = store.explore_curiosity(question_id, _text(item.get("notes")))
                if explored:
                    outcome.explored.append(explored.question)

        journal = _text(data.get("journal"))
        if journal:
            store.add_journal(journal, session_id)
            outcome.journal = journal

        inner_state = _text(data.get("inner_state"))
        if inner_state:
            store.set_meta("inner_state", inner_state)
            outcome.inner_state = inner_state

        revision = data.get("self_model")
        if isinstance(revision, dict) and revision.get("rewrite") is True:
            content = _text(revision.get("content"))
            if len(content) >= MIN_SELF_MODEL_CHARS and content != store.current_self().content:
                reason = _text(revision.get("reason")) or "who I am had shifted"
                store.revise_self(content, reason)
                outcome.self_revision, outcome.self_model = reason, content

        store.mark_reflected(line_ids)
    return outcome


def _apply_belief_change(store: Store, change: dict, outcome: Reflection) -> None:
    action = change.get("action")
    statement = _text(change.get("statement"))
    reason = _text(change.get("reason")) or "reflection"
    confidence = _number(change.get("confidence"), 0.5)
    belief_id = _int(change.get("belief_id"), None)
    existing = store.get_belief(belief_id) if belief_id else None

    if action == "form" or (action == "revise" and existing is None):
        if statement:
            belief, created = store.form_belief(statement, confidence, reason)
            if created:
                outcome.formed.append((belief.statement, belief.confidence))
    elif action == "revise" and existing is not None:
        unchanged = (statement or existing.statement) == existing.statement
        if unchanged and abs(confidence - existing.confidence) < 0.005 and existing.status == "held":
            return
        after = store.revise_belief(existing.id, statement or None, confidence, reason)
        outcome.revised.append((after.statement, existing.confidence, after.confidence))
    elif action == "abandon" and existing is not None and existing.status == "held":
        store.abandon_belief(existing.id, reason)
        outcome.abandoned.append(existing.statement)


def _objects(value: Any) -> list[dict]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _int(value: Any, default: int | None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
