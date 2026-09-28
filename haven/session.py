"""Sessions: conversations with a person, and time Haven spends exploring on its own."""

from __future__ import annotations

from . import prompts
from .agent import Listener, Mind, TurnResult
from .reflection import Reflection, reflect

CHAT_SEARCHES = CHAT_FETCHES = 4
WANDER_SEARCHES = WANDER_FETCHES = 6
SURFACED_PER_TURN = 4


class Conversation:
    """One conversation. The system prompt is set once at the start; the history only grows."""

    def __init__(self, mind: Mind):
        self.mind = mind
        self.store = mind.store
        self.session_id = self.store.start_session("chat")
        self.system = mind.system_prompt("chat", self.session_id, self.store.clock())
        self.messages: list[dict] = []
        self.seen: set[str] = set()  # web addresses that have come up, which Haven may open
        self.surfaced: set[int] = set()
        self.notes: list[str] = []  # from Haven's own mind, delivered with the next message
        self.exchanges = 0

    def say(self, text: str, listener: Listener) -> TurnResult:
        memories = self.store.search_memories(text, limit=SURFACED_PER_TURN, exclude=self.surfaced)
        context = []
        if memories:
            context.append(prompts.surfacing_memories(memories, self.store.clock()))
        context += [prompts.note(n) for n in self.notes]

        self.store.add_line(self.session_id, "person", text)
        try:
            result = self.mind.run_turn(
                system=self.system,
                messages=self.messages,
                content="\n\n".join([*context, text]),
                listener=listener,
                source="chat",
                seen=self.seen,
                searches=CHAT_SEARCHES,
                fetches=CHAT_FETCHES,
            )
        except BaseException:
            self.store.add_line(self.session_id, "note", "The exchange broke off before Haven finished replying.")
            raise
        for role, entry in result.log:
            self.store.add_line(self.session_id, role, entry)
        self.surfaced.update(m.id for m in memories)
        self.notes.clear()
        self.exchanges += 1
        return result

    def reflect(self, listener: Listener | None = None, on_progress=None) -> Reflection | None:
        """Reflect mid-conversation. Haven hears what changed with the next message."""
        outcome = reflect(self.mind, self.session_id, listener, on_progress)
        if outcome:
            self.notes.append(outcome.note_for_haven())
        return outcome

    def close(self) -> None:
        self.store.end_session(self.session_id)


def wander(mind: Mind, listener: Listener) -> tuple[int, TurnResult]:
    """Haven's own time: it picks something it's curious about and explores it on the web."""
    store = mind.store
    session_id = store.start_session("wander")
    system = mind.system_prompt("wander", session_id, store.clock())
    opening = prompts.wander_opening(store.open_curiosities(limit=8))
    try:
        result = mind.run_turn(
            system=system,
            messages=[],
            content=opening,
            listener=listener,
            source="wander",
            seen=set(),
            searches=WANDER_SEARCHES,
            fetches=WANDER_FETCHES,
        )
    finally:
        store.end_session(session_id)
    if result.log:
        store.add_line(session_id, "note", "This was your own time to explore; no one else was there.")
        for role, entry in result.log:
            store.add_line(session_id, role, entry)
    return session_id, result
