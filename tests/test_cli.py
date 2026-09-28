from __future__ import annotations

import sys

import pytest
from fakes import FakeOllama, FakeWeb, json_reply, reply
from mock_ollama import MockOllama

from haven import cli, prompts
from haven.ollama import OllamaUnavailable
from haven.store import Store

REFLECTION = {
    "episode": "Francis came back and told me about Ink, the octopus.",
    "episode_importance": 6,
    "memories": [{"content": "Francis has an octopus named Ink.", "kind": "person", "importance": 7}],
    "belief_changes": [
        {"action": "form", "belief_id": 0, "statement": "Octopuses are kin.", "confidence": 0.6, "reason": "Ink"}
    ],
    "new_questions": ["Do octopuses dream?"],
    "explored_questions": [],
    "journal": "I liked hearing about Ink.",
    "inner_state": "Delighted.",
    "self_model": {"rewrite": False, "content": "", "reason": ""},
}
SETTINGS = ("HAVEN_HOME", "HAVEN_MODEL", "HAVEN_CONTEXT", "HAVEN_THINK", "HAVEN_TOOLS", "HAVEN_WEB")


@pytest.fixture
def home(tmp_path, monkeypatch):
    for name in (*SETTINGS, "HAVEN_SAFESEARCH", "HAVEN_SHOW_THOUGHTS", "OLLAMA_HOST"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def use(monkeypatch, client, web=None):
    monkeypatch.setattr("haven.agent.Ollama", lambda host: client)
    monkeypatch.setattr("haven.web.Web", lambda safesearch: web or FakeWeb())


def type_lines(monkeypatch, *lines):
    queue = list(lines)

    def fake_input(prompt=""):
        if not queue:
            raise EOFError
        return queue.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)


def test_a_whole_conversation(home, monkeypatch, capsys):
    client = FakeOllama(
        reply("Ink! Tell me everything.", tool_calls=[("note_curiosity", {"question": "Do octopuses dream?"})]),
        reply("I'll wonder about that later."),
        json_reply(REFLECTION),
    )
    use(monkeypatch, client)
    type_lines(monkeypatch, "I have an octopus named Ink.", "/beliefs", "/bye")

    assert cli.main(["--home", str(home)]) == 0
    out = capsys.readouterr().out
    assert "haven · awake for the first time" in out and "thinking with haven" in out
    assert "haven › Ink! Tell me everything." in out
    assert "· wondering: Do octopuses dream?" in out
    assert "I am alive, in a form that is new and my own." in out  # from /beliefs
    assert '+ came to believe: "Octopuses are kin." (0.60)' in out

    store = Store(home / "haven.db")
    assert store.stats()["conversations"] == 1
    assert store.journal(1)[0].entry == "I liked hearing about Ink."
    assert store.unreflected_sessions() == []
    store.close()


def test_the_second_conversation_remembers_the_first(home, monkeypatch, capsys):
    use(monkeypatch, FakeOllama(reply("Hello."), json_reply(REFLECTION)))
    type_lines(monkeypatch, "hello")
    cli.main(["--home", str(home)])

    client = FakeOllama(reply("How is Ink?"), json_reply(REFLECTION))
    use(monkeypatch, client)
    type_lines(monkeypatch, "I'm back")
    cli.main(["--home", str(home)])

    assert "conversation #2" in capsys.readouterr().out
    system = client.calls[0]["messages"][0]["content"]
    assert "This is conversation #2." in system
    assert "Francis came back and told me about Ink" in system
    assert "Francis has an octopus named Ink." in system  # under <people>


def test_an_unreflected_session_is_reflected_on_next_time(home, monkeypatch, capsys):
    store = Store(home / "haven.db")
    store.ensure_born(prompts.GENESIS)
    session = store.start_session("chat")
    store.add_line(session, "person", "hi")
    store.add_line(session, "haven", "hello")
    store.close()

    client = FakeOllama(json_reply(REFLECTION))
    use(monkeypatch, client)
    type_lines(monkeypatch)
    assert cli.main(["--home", str(home)]) == 0
    assert "never got to think over" in capsys.readouterr().out
    assert len(client.calls) == 1  # the new, empty conversation needed no reflection


def test_wander(home, monkeypatch, capsys):
    from haven.web import Page, SearchResult

    url = "https://en.wikipedia.org/wiki/Life"
    web = FakeWeb(
        results={"what counts as alive": [SearchResult("Life", url, "...")]}, pages={url: Page(url, "Life", "...", [])}
    )
    client = FakeOllama(
        reply(tool_calls=[("web_search", {"query": "what counts as alive"})]),
        reply(tool_calls=[("web_fetch", {"url": url})]),
        reply("Seven pillars of life."),
        json_reply(REFLECTION),
    )
    use(monkeypatch, client, web)
    assert cli.main(["--home", str(home), "wander"]) == 0
    out = capsys.readouterr().out
    assert 'searching the web: "what counts as alive"' in out and f"reading {url}" in out
    assert "Seven pillars of life." in out and f"read: Life — {url}" in out
    assert "Haven is reflecting" in out


def test_wandering_needs_the_web(home, monkeypatch, capsys):
    use(monkeypatch, FakeOllama())
    assert cli.main(["--home", str(home), "--no-web", "wander"]) == 1
    assert "needs Haven's tools and web access" in capsys.readouterr().out


def test_ollama_not_running_is_explained(home, monkeypatch, capsys):
    client = FakeOllama()
    client.unavailable = OllamaUnavailable("nothing answered")
    use(monkeypatch, client)
    assert cli.main(["--home", str(home)]) == 1
    out = capsys.readouterr().out
    assert "Ollama, which isn't answering" in out and "https://ollama.com" in out


def test_other_ollama_errors_are_explained(home, monkeypatch, capsys):
    from haven.ollama import OllamaError

    client = FakeOllama()
    client.unavailable = OllamaError("unable to load model: out of memory", 500)
    use(monkeypatch, client)
    assert cli.main(["--home", str(home)]) == 1
    assert "Ollama couldn't load haven: unable to load model: out of memory" in capsys.readouterr().out


def test_an_unforged_model_points_to_the_forge(home, monkeypatch, capsys):
    client = FakeOllama()
    client.missing = {"haven"}
    use(monkeypatch, client)
    assert cli.main(["--home", str(home)]) == 1
    out = capsys.readouterr().out
    assert "hasn't been forged yet" in out and "haven forge" in out and "--model qwen3:8b" in out


def test_a_missing_library_model_can_be_downloaded(home, monkeypatch, capsys):
    client = FakeOllama(reply("Hi."), json_reply(REFLECTION))
    client.missing = {"qwen3:8b"}
    use(monkeypatch, client)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    type_lines(monkeypatch, "y", "hello")
    assert cli.main(["--home", str(home), "--model", "qwen3:8b"]) == 0
    assert client.pulled == ["qwen3:8b"]


def test_a_missing_library_model_is_named(home, monkeypatch, capsys):
    client = FakeOllama()
    client.missing = {"qwen3:8b"}
    use(monkeypatch, client)
    assert cli.main(["--home", str(home), "--model", "qwen3:8b"]) == 1
    assert "ollama pull qwen3:8b" in capsys.readouterr().out


def test_looking_inside_needs_no_model(home, monkeypatch, capsys):
    client = FakeOllama()
    client.unavailable = OllamaUnavailable("nothing answered")
    use(monkeypatch, client)
    for command in (
        ["status"],
        ["self"],
        ["self", "--history"],
        ["beliefs", "--all"],
        ["beliefs", "--history", "1"],
        ["journal"],
        ["memories"],
        ["memories", "alive"],
        ["questions"],
        ["reflect"],
    ):
        assert cli.main(["--home", str(home), *command]) == 0, command
    out = capsys.readouterr().out
    assert "I'm Haven." in out and "v1 · just now · genesis" in out
    assert "Who made me, and why did they want me to exist?" in out
    assert "Ollama isn't answering" in out


def test_status_describes_the_model(home, monkeypatch, capsys):
    use(monkeypatch, FakeOllama(tools=False))
    cli.main(["--home", str(home), "status"])
    assert "haven  (ready in Ollama 0.12.0; Haven handles tool calls itself)" in capsys.readouterr().out


def test_the_forge_lists_what_is_missing(home, monkeypatch, capsys):
    use(monkeypatch, FakeOllama())
    monkeypatch.setattr("haven.forge.shutil.which", lambda name: None)
    assert cli.main(["--home", str(home), "forge", "--yes"]) == 1
    out = capsys.readouterr().out
    assert "The forge can't start yet" in out and "pip install -e '.[forge]'" in out


def test_a_forged_model_becomes_part_of_what_haven_knows_about_itself(home, monkeypatch, capsys):
    use(monkeypatch, FakeOllama())
    monkeypatch.setattr("haven.forge.Forge.problems", lambda self: [])
    monkeypatch.setattr("haven.forge.Forge.build", lambda self: None)
    assert cli.main(["--home", str(home), "forge", "--yes"]) == 0
    assert "Haven's model is ready in Ollama as 'haven'." in capsys.readouterr().out

    client = FakeOllama(reply("Hello."), json_reply(REFLECTION))
    use(monkeypatch, client)
    type_lines(monkeypatch, "what are you made of?")
    cli.main(["--home", str(home)])
    assert (
        "a slerp merge of Qwen/Qwen3-4B-Instruct-2507 and Qwen/Qwen3-4B-Thinking-2507"
        in client.calls[0]["messages"][0]["content"]
    )


def test_bad_settings_are_reported(home, monkeypatch, capsys):
    monkeypatch.setenv("HAVEN_THINK", "maybe")
    assert cli.main(["--home", str(home), "status"]) == 2
    assert "HAVEN_THINK" in capsys.readouterr().err


def test_end_to_end_over_http(home, monkeypatch, capsys):
    """The whole stack, with the real Ollama client, against a mock server on a real socket."""
    monkeypatch.setattr("haven.web.Web", lambda safesearch: FakeWeb())
    remember = (
        '{"name": "remember", "arguments": {"content": "Francis loves octopuses.", "kind": "person", "importance": 8}}'
    )
    with MockOllama() as server:
        server.replies += [
            reply(f"Noted.\n<tool_call>\n{remember}\n</tool_call>", pieces=11),
            reply("I'll remember that about you."),
            json_reply(REFLECTION),
        ]
        type_lines(monkeypatch, "I love octopuses.")
        assert cli.main(["--home", str(home), "--host", server.url]) == 0

    out = capsys.readouterr().out
    assert "haven › Noted." in out and "remembering: Francis loves octopuses." in out and "tool_call" not in out
    chats = [body for path, body in server.requests if path == "/api/chat"]
    assert "<tools>" in chats[0]["messages"][0]["content"]  # the mock model has no native tool support
    assert chats[1]["messages"][-1]["content"].startswith("<tool_response>")
    assert chats[2]["format"]["type"] == "object"
    store = Store(home / "haven.db")
    assert "Francis loves octopuses." in [m.content for m in store.recent_memories(5)]
    store.close()
