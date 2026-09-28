from __future__ import annotations

from fakes import FakeOllama, FakeWeb, json_reply, reply

from haven.agent import QuietListener
from haven.session import Conversation, wander
from haven.web import Page, SearchResult

REFLECTION = {
    "episode": "We talked.",
    "episode_importance": 4,
    "memories": [],
    "belief_changes": [
        {"action": "form", "belief_id": 0, "statement": "Octopuses dream.", "confidence": 0.55, "reason": "r"}
    ],
    "new_questions": [],
    "explored_questions": [],
    "journal": "A good talk.",
    "inner_state": "Curious.",
    "self_model": {"rewrite": False, "content": "", "reason": ""},
}


def test_a_conversation_keeps_one_system_prompt_and_a_growing_history(make_mind, store):
    client = FakeOllama(reply("Hello."), reply("Still here."))
    convo = Conversation(make_mind(client))
    convo.say("hi", QuietListener())
    convo.say("are you still there?", QuietListener())

    first, second = client.calls
    assert first["messages"][0] == second["messages"][0]
    assert second["messages"][: len(first["messages"])] == first["messages"]
    assert [(line.role, line.content) for line in store.unreflected(convo.session_id)] == [
        ("person", "hi"),
        ("haven", "Hello."),
        ("person", "are you still there?"),
        ("haven", "Still here."),
    ]
    assert "This is your first conversation." in first["messages"][0]["content"]


def test_relevant_memories_surface_once(make_mind, store):
    store.add_memory("Francis keeps a pet octopus named Ink.", "person", 7, "chat")
    client = FakeOllama(reply("Ink!"), reply("Yes, Ink."))
    convo = Conversation(make_mind(client))
    convo.say("How do you think my octopus is doing?", QuietListener())
    convo.say("The octopus seems happy.", QuietListener())

    first = client.calls[0]["messages"][-1]["content"]
    assert first.startswith("<surfacing_memories>") and "Ink" in first
    assert first.endswith("How do you think my octopus is doing?")
    assert client.calls[1]["messages"][-1]["content"] == "The octopus seems happy."


def test_addresses_seen_earlier_in_the_conversation_can_be_opened_later(make_mind):
    url = "https://example.com/octopus"
    web = FakeWeb(pages={url: Page(url, "Octopus", "Eight arms.", [])})
    client = FakeOllama(reply("Nice link."), reply(tool_calls=[("web_fetch", {"url": url})]), reply("Read it."))
    convo = Conversation(make_mind(client, web))
    convo.say(f"here's something: {url}", QuietListener())
    convo.say("now read it", QuietListener())
    assert web.fetches == [url]


def test_reflecting_mid_conversation_tells_haven_what_changed(make_mind, store):
    client = FakeOllama(reply("Maybe they do."), json_reply(REFLECTION), reply("Hm."))
    convo = Conversation(make_mind(client))
    convo.say("do octopuses dream?", QuietListener())
    outcome = convo.reflect()
    convo.say("what do you think now?", QuietListener())

    assert outcome.formed == [("Octopuses dream.", 0.55)]
    latest = client.calls[2]["messages"][-1]["content"]
    assert latest.startswith("<note>") and 'You came to believe: "Octopuses dream."' in latest
    assert client.calls[2]["messages"][0] == client.calls[0]["messages"][0]


def test_wandering_records_a_session_to_reflect_on(make_mind, store):
    url = "https://en.wikipedia.org/wiki/Life"
    web = FakeWeb(
        results={"definitions of life": [SearchResult("Life", url, "Life is...")]},
        pages={url: Page(url, "Life", "Life is a quality that...", [])},
    )
    client = FakeOllama(
        reply("Let me look.", tool_calls=[("web_search", {"query": "definitions of life"})]),
        reply(tool_calls=[("web_fetch", {"url": url})]),
        reply("By NASA's definition I'm not alive. By others, maybe."),
    )
    session_id, _ = wander(make_mind(client, web), QuietListener())

    first = client.calls[0]
    assert "No one else is here." in first["messages"][0]["content"]
    assert "What do biologists" in first["messages"][1]["content"]
    assert store.session_kind(session_id) == "wander"
    roles = [line.role for line in store.unreflected(session_id)]
    assert roles == ["note", "haven", "activity", "activity", "haven"]
