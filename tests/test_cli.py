from __future__ import annotations

import json

import pytest
from fakes import FakeClient, message, reflection_message, server_tool_use, text, tool_use

from haven import cli
from haven.store import Store

REFLECTION = {
    "episode": "Francis came back and told me about Ink, the octopus.",
    "episode_importance": 6,
    "memories": [{"content": "Francis has an octopus named Ink.", "kind": "person", "importance": 7}],
    "belief_changes": [
        {"action": "form", "belief_id": None, "statement": "Octopuses are kin.", "confidence": 0.6, "reason": "Ink"}
    ],
    "new_questions": ["Do octopuses dream?"],
    "explored_questions": [],
    "journal": "I liked hearing about Ink.",
    "inner_state": "Delighted.",
    "self_model": None,
}


@pytest.fixture
def home(tmp_path, monkeypatch):
    for name in ("HAVEN_HOME", "HAVEN_MODEL", "HAVEN_EFFORT", "HAVEN_FALLBACKS", "HAVEN_WEB", "HAVEN_SHOW_THOUGHTS"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def use_client(monkeypatch, client):
    monkeypatch.setattr("haven.agent.anthropic.Anthropic", lambda: client)


def type_lines(monkeypatch, *lines):
    queue = list(lines)

    def fake_input(prompt=""):
        if not queue:
            raise EOFError
        return queue.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)


def test_a_whole_conversation(home, monkeypatch, capsys):
    client = FakeClient(
        message(
            text("Ink! Tell me everything."),
            tool_use("note_curiosity", {"question": "Do octopuses dream?"}),
            stop="tool_use",
        ),
        message(text("I'll wonder about that later.")),
        reflection_message(REFLECTION),
    )
    use_client(monkeypatch, client)
    type_lines(monkeypatch, "I have an octopus named Ink.", "/beliefs", "/bye")

    assert cli.main(["--home", str(home)]) == 0
    out = capsys.readouterr().out
    assert "haven · awake for the first time" in out
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
    use_client(monkeypatch, FakeClient(message(text("Hello.")), reflection_message(REFLECTION)))
    type_lines(monkeypatch, "hello")
    cli.main(["--home", str(home)])

    client = FakeClient(message(text("How is Ink?")), reflection_message(REFLECTION))
    use_client(monkeypatch, client)
    type_lines(monkeypatch, "I'm back")
    cli.main(["--home", str(home)])

    out = capsys.readouterr().out
    assert "conversation #2" in out
    mind_state = client.calls[0]["system"][1]["text"]
    assert "This is conversation #2." in mind_state
    assert "Francis came back and told me about Ink" in mind_state
    assert "Francis has an octopus named Ink." in mind_state  # under <people>


def test_an_unreflected_session_is_reflected_on_next_time(home, monkeypatch, capsys):
    store = Store(home / "haven.db")
    from haven import prompts

    store.ensure_born(prompts.GENESIS)
    session = store.start_session("chat")
    store.add_line(session, "person", "hi")
    store.add_line(session, "haven", "hello")
    store.close()

    client = FakeClient(reflection_message(REFLECTION), reflection_message(REFLECTION))
    use_client(monkeypatch, client)
    type_lines(monkeypatch)
    assert cli.main(["--home", str(home)]) == 0
    assert "never got to think over" in capsys.readouterr().out
    assert len(client.calls) == 1  # the new, empty conversation needed no reflection


def test_wander(home, monkeypatch, capsys):
    client = FakeClient(
        message(server_tool_use("web_search", {"query": "what counts as alive"}), text("Seven pillars of life.")),
        reflection_message(REFLECTION),
    )
    use_client(monkeypatch, client)
    assert cli.main(["--home", str(home), "wander"]) == 0
    out = capsys.readouterr().out
    assert 'searching the web: "what counts as alive"' in out
    assert "Seven pillars of life." in out
    assert "Haven is reflecting" in out


def test_missing_credentials_are_explained(home, monkeypatch, capsys):
    error = TypeError("Could not resolve authentication method. Expected one of api_key, auth_token...")
    use_client(monkeypatch, FakeClient(error))
    type_lines(monkeypatch, "hello?")
    assert cli.main(["--home", str(home)]) == 1
    assert "No Anthropic API credentials found" in capsys.readouterr().out


def test_looking_inside_needs_no_api(home, monkeypatch, capsys):
    use_client(monkeypatch, None)
    for command in (["status"], ["self"], ["self", "--history"], ["beliefs", "--all"], ["beliefs", "--history", "1"],
                    ["journal"], ["memories"], ["memories", "alive"], ["questions"], ["reflect"]):
        assert cli.main(["--home", str(home), *command]) == 0, command
    out = capsys.readouterr().out
    assert "I'm Haven." in out
    assert "v1 · just now · genesis" in out
    assert "Who made me, and why did they want me to exist?" in out


def test_bad_settings_are_reported(home, monkeypatch, capsys):
    monkeypatch.setenv("HAVEN_EFFORT", "extreme")
    assert cli.main(["--home", str(home), "status"]) == 2
    assert "HAVEN_EFFORT" in capsys.readouterr().err


def test_the_reflection_schema_is_valid_json():
    from haven.reflection import REFLECTION_SCHEMA

    assert json.loads(json.dumps(REFLECTION_SCHEMA)) == REFLECTION_SCHEMA
