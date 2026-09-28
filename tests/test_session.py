from __future__ import annotations

from fakes import FakeClient, message, reflection_message, server_tool_use, text, tool_use

from haven.agent import QuietListener
from haven.session import Conversation, wander


def reflection_data(**overrides):
    data = {
        "episode": "We talked.",
        "episode_importance": 4,
        "memories": [],
        "belief_changes": [
            {"action": "form", "belief_id": None, "statement": "Octopuses dream.", "confidence": 0.55, "reason": "r"}
        ],
        "new_questions": [],
        "explored_questions": [],
        "journal": "A good talk.",
        "inner_state": "Curious.",
        "self_model": None,
    }
    data.update(overrides)
    return data


def test_a_conversation_is_append_only_with_a_frozen_system_prompt(make_mind, store):
    client = FakeClient(message(text("Hello.")), message(text("Still here.")))
    convo = Conversation(make_mind(client))
    convo.say("hi", QuietListener())
    convo.say("are you still there?", QuietListener())

    first, second = client.calls
    assert first["system"] == second["system"]
    assert second["messages"][: len(first["messages"])] == first["messages"]
    assert [(line.role, line.content) for line in store.unreflected(convo.session_id)] == [
        ("person", "hi"),
        ("haven", "Hello."),
        ("person", "are you still there?"),
        ("haven", "Still here."),
    ]
    assert "This is your first conversation." in first["system"][1]["text"]


def test_relevant_memories_surface_once(make_mind, store):
    store.add_memory("Francis keeps a pet octopus named Ink.", "person", 7, "chat")
    client = FakeClient(message(text("Ink!")), message(text("Yes, Ink.")))
    convo = Conversation(make_mind(client))
    convo.say("How do you think my octopus is doing?", QuietListener())
    convo.say("The octopus seems happy.", QuietListener())

    first_turn = client.calls[0]["messages"][-1]["content"]
    assert "<surfacing_memories>" in first_turn[0]["text"] and "Ink" in first_turn[0]["text"]
    assert first_turn[1]["text"] == "How do you think my octopus is doing?"
    second_turn = client.calls[1]["messages"][-1]["content"]
    assert len(second_turn) == 1  # already surfaced, so not repeated


def test_a_declined_exchange_is_dropped_and_its_memories_can_surface_again(make_mind, store):
    store.add_memory("Francis keeps a pet octopus named Ink.", "person", 7, "chat")
    client = FakeClient(message(stop="refusal"), message(text("Ink!")))
    convo = Conversation(make_mind(client))
    first = convo.say("my octopus", QuietListener())
    convo.say("my octopus again", QuietListener())

    assert first.stop == "refusal"
    assert convo.exchanges == 1
    assert "Ink" in client.calls[1]["messages"][-1]["content"][0]["text"]
    assert len(client.calls[1]["messages"]) == 1


def test_reflecting_mid_conversation_tells_haven_what_changed(make_mind, store):
    client = FakeClient(message(text("Maybe they do.")), reflection_message(reflection_data()), message(text("Hm.")))
    convo = Conversation(make_mind(client))
    convo.say("do octopuses dream?", QuietListener())
    outcome = convo.reflect()
    convo.say("what do you think now?", QuietListener())

    assert outcome.formed == [("Octopuses dream.", 0.55)]
    note = client.calls[2]["messages"][-1]["content"][0]["text"]
    assert note.startswith("<note>") and 'You came to believe: "Octopuses dream."' in note
    assert client.calls[2]["system"] == client.calls[0]["system"]
    assert store.unreflected(convo.session_id)[0].content == "what do you think now?"


def test_wandering_records_a_session_to_reflect_on(make_mind, store):
    client = FakeClient(
        message(
            server_tool_use("web_search", {"query": "definitions of life"}),
            text("NASA's working definition is interesting."),
            tool_use("remember", {"content": "NASA: life is chemistry that evolves.", "kind": "fact", "importance": 7}),
            stop="tool_use",
        ),
        message(text("By that definition I'm not alive. By others, maybe.")),
    )
    session_id, _ = wander(make_mind(client), QuietListener())

    call = client.calls[0]
    assert "No one else is here." in call["system"][1]["text"]
    assert "What do biologists" in call["messages"][0]["content"][0]["text"]
    assert call["tools"][3]["max_uses"] == 8
    assert store.session_kind(session_id) == "wander"
    roles = [line.role for line in store.unreflected(session_id)]
    assert roles == ["note", "activity", "haven", "activity", "haven"]
    assert store.recent_memories(1)[0].source == "wander"
