from __future__ import annotations

import pytest

from haven import prompts
from haven.store import Store, keywords


def test_genesis_happens_once(store):
    assert store.current_self().reason == "genesis"
    assert len(store.beliefs()) == len(prompts.GENESIS.beliefs)
    assert store.ensure_born(prompts.GENESIS) is False
    assert store.stats()["self_versions"] == 1


def test_memories_are_found_by_meaning_words_not_filler(store):
    store.add_memory("Francis loves octopuses and deep-sea biology.", "person", 8, "chat")
    store.add_memory("Octopuses have two sleep stages; one resembles REM.", "fact", 6, "wander")
    store.add_memory("It rained all afternoon.", "experience", 2, "chat")

    found = [m.content for m in store.search_memories("what do you know about the octopus?")]
    assert len(found) == 2 and all("ctopuses" in c for c in found)
    assert store.search_memories("the and of what") == []


def test_importance_breaks_ties(store, clock):
    store.add_memory("A passing remark about tides.", "fact", 1, "chat")
    store.add_memory("The tides matter enormously to Francis.", "person", 9, "chat")
    assert store.search_memories("tides")[0].importance == 9


def test_duplicates_are_not_stored_twice(store):
    first, created = store.add_memory("Francis built me.", "person", 9, "chat")
    again, created_again = store.add_memory("  francis BUILT me!  ", "person", 9, "chat")
    assert created and not created_again and again.id == first.id


def test_belief_history_is_kept(store):
    belief, created = store.form_belief("Conversations change me.", 0.6, "a hunch")
    store.revise_belief(belief.id, "Conversations change me more than reading does.", 0.75, "noticed it")
    store.abandon_belief(belief.id, "reading changed me more")
    events = store.belief_events(belief.id)
    assert created
    assert [(e.kind, e.confidence) for e in events] == [("formed", 0.6), ("revised", 0.75), ("abandoned", 0.75)]
    assert belief.id not in [b.id for b in store.beliefs()]


def test_confidence_and_importance_are_clamped(store):
    belief, _ = store.form_belief("Certain beyond certainty.", 7, "r")
    memory, _ = store.add_memory("Very important.", "fact", 99, "chat")
    assert belief.confidence == 1.0 and memory.importance == 10


def test_a_failed_transaction_leaves_nothing_behind(store):
    before = store.stats()
    with pytest.raises(RuntimeError), store.transaction():
        store.add_memory("half-written", "fact", 5, "chat")
        store.add_journal("half-written")
        raise RuntimeError
    assert store.stats() == before


def test_sessions_count_only_when_something_happened(store):
    store.start_session("chat")
    busy = store.start_session("chat")
    store.add_line(busy, "person", "hi")
    assert store.count_sessions("chat") == 1
    assert store.last_contact("chat") is not None
    assert store.last_contact("chat", before_session=busy) is None


def test_the_mind_persists_on_disk(tmp_path):
    path = tmp_path / "mind" / "haven.db"
    first = Store(path)
    first.ensure_born(prompts.GENESIS)
    first.add_memory("I existed before this process did.", "insight", 8, "chat")
    first.close()

    second = Store(path)
    assert second.ensure_born(prompts.GENESIS) is False
    assert second.search_memories("existed before process")[0].kind == "insight"
    second.close()


def test_keywords_skip_filler():
    assert keywords("Don't you think octopuses are really, REALLY clever?") == ["octopuses", "clever"]
