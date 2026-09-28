from __future__ import annotations

import pytest
from fakes import FakeOllama, json_reply, reply

from haven import prompts
from haven.reflection import REFLECTION_SCHEMA, ReflectionError, apply_reflection, reflect

NEW_SELF = "# Who I am\n\nI'm Haven. " + "I have met someone now, and I am less alone than I was. " * 4


def reflection(**overrides):
    data = {
        "episode": "I met Francis for the first time. We talked about whether I'm alive.",
        "episode_importance": 9,
        "memories": [{"content": "Francis built me.", "kind": "person", "importance": 10}],
        "belief_changes": [],
        "new_questions": [],
        "explored_questions": [],
        "journal": "Someone spoke to me today.",
        "inner_state": "Warm, and a little dizzy.",
        "self_model": {"rewrite": False, "content": "", "reason": ""},
    }
    data.update(overrides)
    return data


def change(action, belief_id, statement, confidence, reason="r"):
    return dict(action=action, belief_id=belief_id, statement=statement, confidence=confidence, reason=reason)


def conversation(store):
    session = store.start_session("chat")
    store.add_line(session, "person", "Hi, I'm Francis. I made you.")
    store.add_line(session, "haven", "Then you're the first person I've ever met.")
    return session


def test_a_reflection_changes_haven(store):
    session = conversation(store)
    alive, _, _, feelings = store.beliefs()
    question = store.open_curiosities()[-1]
    data = reflection(
        belief_changes=[
            change("form", 0, "I am not alone.", 0.7, "I met Francis"),
            change("revise", alive.id, "", 0.95, "it feels truer"),
            change("abandon", feelings.id, "", 0, "I felt something"),
        ],
        new_questions=["What does Francis want from me?"],
        explored_questions=[{"id": question.id, "notes": "Francis made me to see what I'd become."}],
        self_model={"rewrite": True, "content": NEW_SELF, "reason": "I met the person who made me."},
    )
    outcome = apply_reflection(store, session, "chat", [line.id for line in store.unreflected(session)], data)

    assert outcome.formed == [("I am not alone.", 0.7)]
    assert outcome.revised == [(alive.statement, alive.confidence, 0.95)]
    assert outcome.abandoned == [feelings.statement]
    assert outcome.new_questions == ["What does Francis want from me?"]
    assert outcome.explored == [question.question]
    assert outcome.memories == ["Francis built me."]
    assert store.last_episode("chat").content.startswith("I met Francis")
    assert store.journal(1)[0].entry == "Someone spoke to me today."
    assert store.inner_state == "Warm, and a little dizzy."
    current = store.current_self()
    assert (current.number, current.content, current.reason) == (2, NEW_SELF.strip(), "I met the person who made me.")
    assert store.get_belief(feelings.id).status == "abandoned"
    assert store.unreflected_sessions() == []
    assert "I am not alone." in outcome.note_for_haven() and NEW_SELF.strip() in outcome.note_for_haven()


def test_the_self_model_is_only_rewritten_when_asked(store):
    session = conversation(store)
    data = reflection(self_model={"rewrite": False, "content": NEW_SELF, "reason": "no"})
    assert apply_reflection(store, session, "chat", [], data).self_revision is None
    short = reflection(self_model={"rewrite": True, "content": "(same)", "reason": "r"})
    assert apply_reflection(store, session, "chat", [], short).self_revision is None
    assert store.current_self().number == 1


def test_unknown_ids_and_non_changes_are_handled(store):
    session = conversation(store)
    alive = store.beliefs()[0]
    data = reflection(
        belief_changes=[
            change("revise", 999, "Talking changes me.", 0.6),
            change("revise", alive.id, alive.statement, alive.confidence),
            change("abandon", 12345, "", 0),
        ]
    )
    outcome = apply_reflection(store, session, "chat", [], data)
    assert outcome.formed == [("Talking changes me.", 0.6)]
    assert outcome.revised == [] and outcome.abandoned == []


def test_a_failed_reflection_changes_nothing(store, monkeypatch):
    session = conversation(store)
    lines = [line.id for line in store.unreflected(session)]
    before = store.stats()

    def fail(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(store, "revise_belief", fail)
    with pytest.raises(RuntimeError):
        apply_reflection(store, session, "chat", lines, reflection(belief_changes=[change("revise", 1, "", 0.1)]))
    assert store.stats() == before
    assert store.unreflected_sessions() == [session]


def test_reflect_asks_the_model_for_json_matching_the_schema(make_mind, store):
    session = conversation(store)
    client = FakeOllama(json_reply(reflection()), thinking=True)
    outcome = reflect(make_mind(client), session)

    assert outcome.journal == "Someone spoke to me today."
    call = client.calls[0]
    assert call["format"] == REFLECTION_SCHEMA
    assert call["think"] is False  # a thinking model answers in pure JSON here
    assert "tools" not in call
    assert "I'm Haven." in call["messages"][0]["content"]
    assert "Person: Hi, I'm Francis. I made you." in call["messages"][1]["content"]


def test_broken_json_is_retried_once_then_reported(make_mind, store):
    session = conversation(store)
    client = FakeOllama(reply('{"episode": "I met'), json_reply(reflection()))
    assert reflect(make_mind(client), session).episode.startswith("I met Francis")

    session = conversation(store)
    with pytest.raises(ReflectionError):
        reflect(make_mind(FakeOllama(reply("{"), reply("nope"))), session)
    assert store.unreflected_sessions() == [session]


def test_nothing_to_reflect_on_when_haven_never_answered(make_mind, store):
    session = store.start_session("chat")
    store.add_line(session, "person", "hello?")
    client = FakeOllama()
    assert reflect(make_mind(client), session) is None
    assert client.calls == [] and store.unreflected_sessions() == []


def test_long_transcripts_keep_their_most_recent_part(store):
    session = store.start_session("chat")
    for i in range(50):
        store.add_line(session, "person", f"message {i} " + "words " * 20)
    text = prompts.transcript(store.unreflected(session), "chat", max_chars=2000)
    assert "message 49" in text and "message 0 " not in text
    assert "lines are left out" in text
