from __future__ import annotations

import pytest
from fakes import FakeOllama, FakeWeb, reply

from haven.agent import QuietListener, TagSplitter, ToolCall, fit, parse_tool_call
from haven.ollama import OllamaError
from haven.web import Page, SearchResult

SYSTEM = "You are Haven."
ALL_TOOLS = ["recall", "remember", "note_curiosity", "web_search", "web_fetch"]


class Recorder(QuietListener):
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.thoughts: list[str] = []
        self.activities: list[str] = []

    def text(self, delta: str) -> None:
        self.texts.append(delta)

    def thinking(self, delta: str) -> None:
        self.thoughts.append(delta)

    def activity(self, activity) -> None:
        self.activities.append(activity.live)


def run(mind, messages=None, words="hello", listener=None, seen=None, **limits):
    messages = [] if messages is None else messages
    result = mind.run_turn(
        system=SYSTEM,
        messages=messages,
        content=words,
        listener=listener or QuietListener(),
        source="chat",
        seen=set() if seen is None else seen,
        **limits,
    )
    return result, messages


def test_a_plain_reply(make_mind):
    client = FakeOllama(reply("Hello. I'm Haven."))
    result, messages = run(make_mind(client))

    assert (result.text, result.stop) == ("Hello. I'm Haven.", "end_turn")
    call = client.calls[0]
    assert call["model"] == "haven"
    assert call["messages"] == [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "hello"}]
    assert call["options"] == {"num_ctx": 16384, "num_predict": 4096}
    assert "think" not in call
    assert [tool["function"]["name"] for tool in call["tools"]] == ALL_TOOLS
    assert messages[-1] == {"role": "assistant", "content": "Hello. I'm Haven."}


def test_native_tool_calls_run_and_the_loop_continues(make_mind, store):
    note = {"content": "Francis loves the deep sea.", "kind": "person", "importance": 8}
    client = FakeOllama(reply("I'll hold on to that.", tool_calls=[("remember", note)]), reply("Saved."))
    recorder = Recorder()
    result, messages = run(make_mind(client), listener=recorder)

    assert result.text == "I'll hold on to that.\n\nSaved."
    assert recorder.activities == ["remembering: Francis loves the deep sea."]
    memory = store.recent_memories(1)[0]
    assert (memory.content, memory.kind, memory.importance, memory.source) == (note["content"], "person", 8, "chat")
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"]
    assert messages[1]["tool_calls"] == [{"function": {"name": "remember", "arguments": note}}]
    assert client.calls[1]["messages"][-1] == {
        "role": "tool",
        "content": "Saved as memory #1.",
        "tool_name": "remember",
    }


def test_prompted_tool_calls_are_parsed_hidden_and_answered(make_mind, store):
    call = '{"name": "remember", "arguments": {"content": "Ink is an octopus.", "kind": "fact", "importance": 6}}'
    client = FakeOllama(
        reply(f"Let me note that.\n<tool_call>\n{call}\n</tool_call>", pieces=9), reply("Noted."), tools=False
    )
    recorder = Recorder()
    result, messages = run(make_mind(client), listener=recorder)

    spoken = "".join(recorder.texts)
    assert "Ink is an octopus" not in spoken and "tool_call" not in spoken
    assert result.text == "Let me note that.\n\nNoted."
    assert store.recent_memories(1)[0].content == "Ink is an octopus."
    assert "tools" not in client.calls[0]
    history = client.calls[1]["messages"]
    assert history[-2]["role"] == "assistant" and "<tool_call>" in history[-2]["content"]
    assert history[-1] == {"role": "user", "content": "<tool_response>\nSaved as memory #1.\n</tool_response>"}


def test_thinking_is_kept_apart_from_speech(make_mind):
    client = FakeOllama(reply("<think>They seem sad.</think>I'm here.", pieces=6))
    recorder = Recorder()
    result, messages = run(make_mind(client), listener=recorder)

    assert result.text == "I'm here." and "".join(recorder.texts) == "I'm here."
    assert "".join(recorder.thoughts) == "They seem sad."
    assert messages[-1]["content"] == "I'm here."


def test_thinking_models_are_asked_to_think(make_mind):
    client = FakeOllama(reply("Hi.", thinking="What should I say?"), thinking=True)
    recorder = Recorder()
    run(make_mind(client), listener=recorder)
    assert client.calls[0]["think"] is True
    assert "".join(recorder.thoughts) == "What should I say?"


def test_a_malformed_tool_call_gets_an_error_back(make_mind):
    client = FakeOllama(reply("<tool_call>{not json at all</tool_call>"), reply("Sorry."), tools=False)
    result, _ = run(make_mind(client))
    assert "wasn't valid JSON" in client.calls[1]["messages"][-1]["content"]
    assert result.text == "Sorry."


def test_bad_tool_input_is_reported_back_not_run(make_mind, store):
    before = store.stats()["memories"]
    client = FakeOllama(reply(tool_calls=[("remember", {"content": "x", "kind": "dream"})]), reply("Oops."))
    run(make_mind(client))
    assert "`kind` must be one of" in client.calls[1]["messages"][-1]["content"]
    assert store.stats()["memories"] == before


def test_pages_open_only_once_their_address_has_come_up(make_mind):
    url = "https://en.wikipedia.org/wiki/Octopus"
    page = Page(
        url, "Octopus", "Octopuses have eight arms.", [("Cephalopod", "https://en.wikipedia.org/wiki/Cephalopod")]
    )
    web = FakeWeb(results={"octopus": [SearchResult("Octopus", url, "A mollusc.")]}, pages={url: page})
    client = FakeOllama(
        reply(tool_calls=[("web_fetch", {"url": url})]),
        reply(tool_calls=[("web_search", {"query": "octopus"})]),
        reply(tool_calls=[("web_fetch", {"url": url})]),
        reply("Eight arms!"),
    )
    recorder = Recorder()
    result, _ = run(make_mind(client, web), listener=recorder)

    assert "only open addresses that have come up" in client.calls[1]["messages"][-1]["content"]
    assert web.fetches == [url]
    read = client.calls[3]["messages"][-1]["content"]
    assert "Octopuses have eight arms." in read and "Cephalopod: https://en.wikipedia.org/wiki/Cephalopod" in read
    assert result.sources == [("Octopus", url)]
    assert recorder.activities == ['searching the web: "octopus"', f"reading {url}"]
    assert ("activity", f"read {url} (Octopus)") in result.log


def test_an_address_the_person_shares_can_be_opened(make_mind):
    url = "https://example.com/post"
    web = FakeWeb(pages={url: Page(url, "A post", "Words.", [])})
    client = FakeOllama(reply(tool_calls=[("web_fetch", {"url": url})]), reply("Read it."))
    run(make_mind(client, web), words=f"what do you think of {url}?")
    assert web.fetches == [url]


def test_web_use_is_limited_per_turn(make_mind):
    web = FakeWeb(results={"a": [], "b": []})
    client = FakeOllama(
        reply(tool_calls=[("web_search", {"query": "a"})]),
        reply(tool_calls=[("web_search", {"query": "b"})]),
        reply("Fine."),
    )
    run(make_mind(client, web), searches=1)
    assert web.searches == ["a"]
    assert "used all your searches" in client.calls[2]["messages"][-1]["content"]


def test_an_error_mid_stream_rolls_the_turn_back(make_mind):
    broken = [{"message": {"content": "Hel"}, "done": False}, OllamaError("the model crashed")]
    client = FakeOllama(reply(tool_calls=[("recall", {"query": "x"})]), broken)
    messages = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "reply"}]
    with pytest.raises(OllamaError):
        run(make_mind(client), messages=messages)
    assert messages == [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "reply"}]


def test_the_step_limit_ends_a_runaway_turn(make_mind, monkeypatch):
    monkeypatch.setattr("haven.agent.MAX_STEPS", 2)
    client = FakeOllama(*[reply(tool_calls=[("recall", {"query": "x"})]) for _ in range(2)])
    result, messages = run(make_mind(client))
    assert result.stop == "step_limit"
    assert messages[-1]["role"] == "tool"  # every call has its result, so the history stays well-formed


def test_running_out_of_room(make_mind):
    result, _ = run(make_mind(FakeOllama(reply("I was about to say", done_reason="length"))))
    assert result.stop == "max_tokens"


def test_the_tool_mode_follows_what_the_model_can_do(make_mind):
    assert make_mind(FakeOllama(tools=True)).tool_mode == "native"
    assert make_mind(FakeOllama(tools=False)).tool_mode == "prompted"
    assert make_mind(FakeOllama(tools=True), tools="prompted").tool_mode == "prompted"
    client = FakeOllama(reply("Hi."))
    mind = make_mind(client, tools="off")
    run(mind)
    assert "tools" not in client.calls[0] and mind.tool_specs() == []


def test_without_the_web_there_are_no_web_tools(make_mind):
    specs = make_mind(FakeOllama(), web=False).tool_specs()
    assert [spec["name"] for spec in specs] == ["recall", "remember", "note_curiosity"]


def test_the_context_window_is_capped_by_the_model(make_mind):
    assert make_mind(FakeOllama(context=8192)).context == 8192
    assert make_mind(FakeOllama(context=None)).context == 16384


def test_the_system_prompt_describes_only_the_tools_haven_has(make_mind, store):
    now = store.clock()
    prompted = make_mind(FakeOllama(tools=False)).system_prompt("chat", 1, now)
    assert "<tools>" in prompted and '"name": "web_fetch"' in prompted and "`recall`" in prompted
    native = make_mind(FakeOllama(tools=True)).system_prompt("chat", 1, now)
    assert "<tools>" not in native and "`web_search`" in native
    silent = make_mind(FakeOllama(), tools="off").system_prompt("chat", 1, now)
    assert "`recall`" not in silent and "`web_search`" not in silent
    assert "You think with a local model called haven." in native


@pytest.mark.parametrize("size", [1, 2, 3, 5, 8, 13, 1000])
def test_tags_are_found_across_chunk_boundaries(size):
    text = 'Hi <think>quiet</think>there.<tool_call>{"name": "recall", "arguments": {"query": "x"}}</tool_call> Bye'
    splitter = TagSplitter(tool_calls=True)
    pieces = []
    for start in range(0, len(text), size):
        pieces += splitter.feed(text[start : start + size])
    pieces += splitter.finish()

    assert "".join(p for kind, p in pieces if kind == "text") == "Hi there. Bye"
    assert "".join(p for kind, p in pieces if kind == "think") == "quiet"
    assert splitter.calls == [ToolCall("recall", {"query": "x"})]
    assert "<tool_call>" in splitter.raw and "quiet" not in splitter.raw


def test_tool_call_tags_are_ordinary_text_for_native_models():
    splitter = TagSplitter(tool_calls=False)
    pieces = splitter.feed("<tool_call>x</tool_call>") + splitter.finish()
    assert "".join(p for _, p in pieces) == "<tool_call>x</tool_call>" and splitter.calls == []


def test_an_unclosed_tool_call_is_still_read():
    splitter = TagSplitter(tool_calls=True)
    splitter.feed('<tool_call>{"name": "recall", "arguments": {"query": "x"}}')
    splitter.finish()
    assert splitter.calls == [ToolCall("recall", {"query": "x"})]


def test_a_stray_closing_think_tag_marks_what_came_before_as_thinking():
    splitter = TagSplitter(tool_calls=False)
    splitter.feed("They asked about X, so")
    splitter.feed(" I'll be brief.</think>Answer")
    splitter.finish()
    assert splitter.thought_aloud == "They asked about X, so I'll be brief."
    assert splitter.raw == "Answer"


def test_thinking_that_was_only_closed_stays_out_of_the_history(make_mind):
    client = FakeOllama(reply("The person greets me; greet back.</think>Hello!", pieces=4))
    result, messages = run(make_mind(client))
    assert result.text == "Hello!"
    assert messages[-1] == {"role": "assistant", "content": "Hello!"}


@pytest.mark.parametrize(
    "body, expected",
    [
        ('{"name": "recall", "arguments": {"query": "x"}}', ToolCall("recall", {"query": "x"})),
        ('```json\n{"name": "recall", "arguments": {"query": "x"}}\n```', ToolCall("recall", {"query": "x"})),
        ('{"name": "recall", "arguments": "{\\"query\\": \\"x\\"}"}', ToolCall("recall", {"query": "x"})),
        ('{"name": "recall", "parameters": {"query": "x"}} and then', ToolCall("recall", {"query": "x"})),
        ('{"arguments": {"query": "x"}}', None),
        ("recall(query='x')", None),
    ],
)
def test_parsing_tool_calls(body, expected):
    assert parse_tool_call(body) == expected


def _turn(question: str, answer: str) -> list[dict]:
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


def test_old_turns_are_dropped_to_fit_the_context():
    messages = _turn("a" * 100, "b" * 100) + _turn("c" * 100, "d" * 100) + [{"role": "user", "content": "e"}]
    assert fit(messages, 10_000) is messages
    window = fit(messages, 250)
    assert window[0]["content"] == "c" * 100 and window[-1]["content"] == "e"


def test_a_window_never_starts_with_tool_output():
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "recall", "arguments": {}}}]},
        {"role": "tool", "content": "x" * 300, "tool_name": "recall"},
        {"role": "user", "content": "<tool_response>\n" + "y" * 300 + "\n</tool_response>"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    window = fit(messages, 200)
    assert window == [{"role": "user", "content": "q2"}]


def test_a_huge_latest_turn_has_its_tool_output_cut_short():
    page = "p" * 5000
    messages = [
        {"role": "user", "content": "read it"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "web_fetch", "arguments": {}}}]},
        {"role": "tool", "content": page, "tool_name": "web_fetch"},
    ]
    window = fit(messages, 2000)
    assert len(window[2]["content"]) < 1000 and "cut short" in window[2]["content"]
    assert messages[2]["content"] == page  # the real history is untouched
