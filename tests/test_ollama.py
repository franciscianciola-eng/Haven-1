"""The real Ollama client, over a real socket, against a mock server."""

from __future__ import annotations

import socket

import pytest
from mock_ollama import MockOllama

from haven.ollama import ModelInfo, ModelNotFound, Ollama, OllamaError, OllamaUnavailable


@pytest.fixture
def server():
    with MockOllama() as mock:
        yield mock


def test_chat_streams_chunks(server):
    server.replies.append(
        [
            {"message": {"content": "Hel"}, "done": False},
            {"message": {"content": "lo"}, "done": True, "done_reason": "stop"},
        ]
    )
    chunks = list(Ollama(server.url).chat(model="haven", messages=[{"role": "user", "content": "hi"}]))

    assert "".join(chunk["message"]["content"] for chunk in chunks) == "Hello"
    assert chunks[-1]["done_reason"] == "stop"
    path, body = server.requests[-1]
    assert path == "/api/chat" and body["stream"] is True and body["messages"][0]["content"] == "hi"


def test_an_error_in_the_middle_of_a_stream_is_raised(server):
    server.replies.append([{"message": {"content": "I"}, "done": False}, {"error": "out of memory"}])
    with pytest.raises(OllamaError, match="out of memory"):
        list(Ollama(server.url).chat(model="haven", messages=[]))


def test_http_errors_carry_ollamas_explanation(server):
    server.replies.append(500)
    with pytest.raises(OllamaError) as caught:
        list(Ollama(server.url).chat(model="haven", messages=[]))
    assert (caught.value.message, caught.value.status) == ("something broke", 500)


def test_inspect_reads_what_a_model_can_do(server):
    server.models["qwen3:8b"] = {
        "capabilities": ["completion", "tools", "thinking"],
        "model_info": {"qwen3.context_length": 40960},
    }
    client = Ollama(server.url)
    assert client.inspect("haven") == ModelInfo("haven", False, False, 262144)
    assert client.inspect("qwen3:8b") == ModelInfo("qwen3:8b", True, True, 40960)


def test_missing_models_are_named_as_such(server):
    client = Ollama(server.url)
    with pytest.raises(ModelNotFound):
        client.inspect("nope")
    with pytest.raises(ModelNotFound):
        list(client.chat(model="nope", messages=[]))


def test_pull_streams_progress(server):
    statuses = [chunk["status"] for chunk in Ollama(server.url).pull("qwen3:8b")]
    assert statuses == ["pulling manifest", "downloading", "success"]


def test_version_and_models(server):
    client = Ollama(server.url)
    assert client.version() == "0.12.3"
    assert client.models() == ["haven"]


def test_nothing_listening_is_reported_as_unavailable():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with pytest.raises(OllamaUnavailable):
        Ollama(f"http://127.0.0.1:{port}").version()
