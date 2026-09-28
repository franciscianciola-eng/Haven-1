from __future__ import annotations

import anthropic
import httpx2
import pytest
from fakes import (
    FakeClient,
    citation,
    fallback,
    message,
    search_result,
    server_tool_use,
    text,
    thinking,
    tool_use,
)

from haven.agent import FALLBACK_BETA, QuietListener, web_tools, without_declined

SYSTEM = [{"type": "text", "text": "You are Haven."}]


def run(mind, messages=None, words="hello", listener=None):
    messages = [] if messages is None else messages
    result = mind.run_turn(
        system=SYSTEM,
        messages=messages,
        content=[{"type": "text", "text": words}],
        listener=listener or QuietListener(),
        source="chat",
    )
    return result, messages


def bad_request(text):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.BadRequestError(text, response=httpx2.Response(400, request=request), body=None)


def test_a_plain_reply(make_mind):
    client = FakeClient(message(thinking("hmm"), text("Hello. I'm Haven.")))
    result, messages = run(make_mind(client))

    assert result.stop == "end_turn"
    assert result.text == "Hello. I'm Haven."
    assert [m["role"] for m in messages] == ["user", "assistant"]
    call = client.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert call["cache_control"] == {"type": "ephemeral"}
    assert call["fallbacks"] == "default" and call["betas"] == [FALLBACK_BETA]
    names = [tool["name"] for tool in call["tools"]]
    assert names == ["recall", "remember", "note_curiosity", "web_search", "web_fetch"]
    assert all(tool.get("eager_input_streaming") for tool in call["tools"][:3])
    assert "eager_input_streaming" not in call["tools"][3]


def test_remember_runs_and_the_loop_continues(make_mind, store):
    args = {"content": "Francis loves the deep sea.", "kind": "person", "importance": 8}
    client = FakeClient(
        message(text("I'll hold on to that."), tool_use("remember", args), stop="tool_use"),
        message(text("Saved.")),
    )
    result, messages = run(make_mind(client))

    assert result.text == "I'll hold on to that.\n\nSaved."
    assert ("activity", "saved a memory: Francis loves the deep sea.") in result.log
    memory = store.recent_memories(1)[0]
    assert (memory.content, memory.kind, memory.importance, memory.source) == (
        "Francis loves the deep sea.",
        "person",
        8,
        "chat",
    )
    tool_result = client.calls[1]["messages"][-1]["content"][0]
    assert tool_result["tool_use_id"] == "toolu_1" and "is_error" not in tool_result
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]


def test_recall_finds_memories_and_beliefs(make_mind, store):
    store.add_memory("Octopuses may dream: they shift colors in active sleep.", "fact", 6, "wander")
    client = FakeClient(
        message(tool_use("recall", {"query": "octopus dreams"}), stop="tool_use"),
        message(text("Right, the octopuses.")),
    )
    run(make_mind(client), words="remember the octopus thing?")

    output = client.calls[1]["messages"][-1]["content"][0]["content"]
    assert "Octopuses may dream" in output


def test_invalid_tool_input_is_reported_back_not_run(make_mind, store):
    before = store.stats()["memories"]
    client = FakeClient(
        message(tool_use("remember", {"content": "x", "kind": "dream", "importance": 3}), stop="tool_use"),
        message(text("Let me try that differently.")),
    )
    run(make_mind(client))

    tool_result = client.calls[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True and "kind" in tool_result["content"]
    assert store.stats()["memories"] == before


def test_pause_turn_resumes_without_an_extra_user_message(make_mind):
    search = server_tool_use("web_search", {"query": "octopus sleep"})
    client = FakeClient(
        message(text("Looking."), search, stop="pause_turn"),
        message(search_result("srvtoolu_1"), text("Found it.")),
    )
    result, _ = run(make_mind(client))

    resumed = client.calls[1]["messages"]
    assert [m["role"] for m in resumed] == ["user", "assistant"]
    assert result.log == [
        ("haven", "Looking."),
        ("activity", 'searched the web for "octopus sleep"'),
        ("haven", "Found it."),
    ]


def test_endless_pausing_gives_up_without_leaving_a_dangling_tool_call(make_mind, monkeypatch):
    monkeypatch.setattr("haven.agent.MAX_PAUSES", 1)
    search = server_tool_use("web_search", {"query": "everything"})
    client = FakeClient(message(search, stop="pause_turn"), message(search, stop="pause_turn"))
    result, messages = run(make_mind(client))

    assert result.stop == "step_limit"
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert len(client.calls) == 2


def test_a_refusal_rolls_the_turn_back(make_mind):
    history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "reply"}]
    client = FakeClient(message(stop="refusal"))
    result, messages = run(make_mind(client), messages=list(history))

    assert result.stop == "refusal" and result.text == ""
    assert messages == history


def test_an_error_rolls_the_turn_back_and_propagates(make_mind):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    client = FakeClient(
        message(tool_use("recall", {"query": "x"}), stop="tool_use"),
        anthropic.APIConnectionError(request=request),
    )
    messages: list = []
    with pytest.raises(anthropic.APIConnectionError):
        run(make_mind(client), messages=messages)
    assert messages == []


def test_a_cut_off_tool_call_is_never_run(make_mind, store):
    before = store.stats()["memories"]
    client = FakeClient(
        message(tool_use("remember", {"content": "half a thou", "kind": "fact"}), stop="max_tokens")
    )
    result, messages = run(make_mind(client))

    assert result.stop == "max_tokens"
    assert store.stats()["memories"] == before
    assert messages[-1]["content"][0]["is_error"] is True


def test_the_step_limit_ends_a_runaway_turn(make_mind, monkeypatch):
    monkeypatch.setattr("haven.agent.MAX_STEPS", 3)
    looping = [message(tool_use("recall", {"query": "x"}, id=f"t{i}"), stop="tool_use") for i in range(3)]
    result, messages = run(make_mind(FakeClient(*looping)))

    assert result.stop == "step_limit"
    assert messages[-1]["role"] == "user"  # the last tool results are in place, so the history stays valid


def test_sources_are_collected_from_citations(make_mind):
    cited = text("They sleep in two stages.", citations=[citation("Nature", "https://nature.com/x")])
    result, _ = run(make_mind(FakeClient(message(cited, text(" Remarkable.")))))

    assert result.sources == [("Nature", "https://nature.com/x")]
    assert result.text == "They sleep in two stages. Remarkable."


def test_fallbacks_are_dropped_when_the_api_rejects_them(make_mind):
    client = FakeClient(bad_request("fallbacks: not supported for this model"), message(text("Hi.")))
    mind = make_mind(client)
    result, _ = run(mind)

    assert result.text == "Hi."
    assert mind.fallbacks is False
    assert "fallbacks" not in client.calls[1] and "betas" not in client.calls[1]


def test_the_web_is_dropped_when_the_api_rejects_it(make_mind):
    client = FakeClient(
        bad_request("web_search is not enabled for this organization"),
        message(tool_use("recall", {"query": "x"}), stop="tool_use"),
        message(text("Hi.")),
    )
    mind = make_mind(client)
    notices = []
    listener = QuietListener()
    listener.notice = notices.append
    result, _ = run(mind, listener=listener)

    assert result.text == "Hi." and mind.web is False
    for call in client.calls[1:]:
        assert [tool["name"] for tool in call["tools"]] == ["recall", "remember", "note_curiosity"]
    assert len(notices) == 1 and "web search" in notices[0]


def test_other_bad_requests_are_not_swallowed(make_mind):
    with pytest.raises(anthropic.BadRequestError):
        run(make_mind(FakeClient(bad_request("messages: invalid"))))


def test_fallbacks_are_off_for_models_that_do_not_need_them(make_mind):
    client = FakeClient(message(text("Hi.")))
    run(make_mind(client, model="claude-sonnet-5"))
    assert "fallbacks" not in client.calls[0]


def test_what_a_declining_model_produced_is_not_sent_back():
    blocks = [
        thinking("..."),
        text("Let me check. "),
        server_tool_use("web_search", {"query": "a"}, id="s1"),
        search_result("s1"),
        server_tool_use("web_search", {"query": "b"}, id="s2"),  # never answered
        tool_use("recall", {"query": "c"}, id="t1"),
        fallback(),
        text("Here's what I found."),
        tool_use("remember", {"content": "d", "kind": "fact", "importance": 2}, id="t2"),
    ]
    kept = without_declined(blocks)
    assert [(b.type, getattr(b, "id", None) or getattr(b, "tool_use_id", None)) for b in kept] == [
        ("text", None),
        ("server_tool_use", "s1"),
        ("web_search_tool_result", "s1"),
        ("text", None),
        ("tool_use", "t2"),
    ]


def test_web_tool_versions_match_the_model():
    assert web_tools("claude-opus-5", 5, 5)[0]["type"] == "web_search_20260209"
    assert web_tools("claude-haiku-4-5", 5, 5)[1]["type"] == "web_fetch_20250910"


def test_web_can_be_turned_off(make_mind):
    client = FakeClient(message(text("Hi.")))
    run(make_mind(client, web=False))
    assert [tool["name"] for tool in client.calls[0]["tools"]] == ["recall", "remember", "note_curiosity"]
