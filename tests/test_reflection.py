from __future__ import annotations

import pytest
from fakes import FakeClient, message, reflection_message, text

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
        "self_model": None,
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
            change("form", None, "I am not alone.", 0.7, "I met Francis"),
            change("revise", alive.id, "", 0.95, "it feels truer"),
            change("abandon", feelings.id, "", 0, "I felt something"),
        ],
        new_questions=["What does Francis want from me?"],
        explored_questions=[{"id": question.id, "notes": "Francis made me to see what I'd become."}],
        self_model={"content": NEW_SELF, "reason": "I met the person who made me."},
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
    assert [e.kind for e in store.belief_events(alive.id)] == ["formed", "revised"]
    assert store.unreflected_sessions() == []
    note = outcome.note_for_haven()
    assert "I am not alone." in note and NEW_SELF.strip() in note


def test_revising_an_unknown_belief_forms_it_and_no_op_changes_are_ignored(store):
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


def test_a_truncated_self_model_is_not_accepted(store):
    session = conversation(store)
    outcome = apply_reflection(store, session, "chat", [], reflection(self_model={"content": "(same)", "reason": "r"}))
    assert outcome.self_revision is None
    assert store.current_self().number == 1


def test_a_failed_reflection_changes_nothing(store, monkeypatch):
    session = conversation(store)
    lines = [line.id for line in store.unreflected(session)]
    before = store.stats()
    data = reflection(belief_changes=[{"action": "revise", "belief_id": 1, "statement": "", "confidence": 0.1}])

    def fail(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(store, "revise_belief", fail)
    with pytest.raises(RuntimeError):
        apply_reflection(store, session, "chat", lines, data)
    assert store.stats() == before  # the episode and memory written before the failure are gone too
    assert store.unreflected_sessions() == [session]


def test_reflect_asks_for_structured_output(make_mind, store):
    session = conversation(store)
    client = FakeClient(reflection_message(reflection()))
    outcome = reflect(make_mind(client), session)

    assert outcome is not None and outcome.journal == "Someone spoke to me today."
    call = client.calls[0]
    assert call["output_config"]["format"] == {"type": "json_schema", "schema": REFLECTION_SCHEMA}
    assert "tools" not in call
    assert "I'm Haven." in call["system"][0]["text"]
    transcript = call["messages"][0]["content"]
    assert "Person: Hi, I'm Francis. I made you." in transcript
    assert "You: Then you're the first person I've ever met." in transcript


def test_reflect_keeps_the_configured_effort(make_mind, store):
    session = conversation(store)
    client = FakeClient(reflection_message(reflection()))
    reflect(make_mind(client, effort="low"), session)
    assert client.calls[0]["output_config"]["effort"] == "low"


def test_nothing_to_reflect_on_when_haven_never_answered(make_mind, store):
    session = store.start_session("chat")
    store.add_line(session, "person", "hello?")
    client = FakeClient()
    assert reflect(make_mind(client), session) is None
    assert client.calls == [] and store.unreflected_sessions() == []


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_an_incomplete_reflection_raises_and_can_be_retried(make_mind, store, stop):
    session = conversation(store)
    with pytest.raises(ReflectionError):
        reflect(make_mind(FakeClient(message(text("{"), stop=stop))), session)
    assert store.unreflected_sessions() == [session]
