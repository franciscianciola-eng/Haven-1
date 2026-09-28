"""Stand-ins for Ollama and the web that replay scripted responses."""

from __future__ import annotations

import copy
import dataclasses
import json

from haven.ollama import ModelInfo, ModelNotFound
from haven.web import Page, SearchResult, WebError


def _pieces(text: str, count: int) -> list[str]:
    if not text:
        return []
    size = max(1, -(-len(text) // count))
    return [text[i : i + size] for i in range(0, len(text), size)]


def reply(
    text: str = "",
    *,
    thinking: str = "",
    tool_calls: list[tuple[str, dict]] = (),
    done_reason: str = "stop",
    pieces: int = 3,
) -> list[dict]:
    """The chunks Ollama streams for one reply, with text split into `pieces` chunks."""
    chunks = [
        {"message": {"role": "assistant", "content": "", "thinking": t}, "done": False}
        for t in _pieces(thinking, pieces)
    ]
    chunks += [{"message": {"role": "assistant", "content": t}, "done": False} for t in _pieces(text, pieces)]
    if tool_calls:
        calls = [{"function": {"name": name, "arguments": args}} for name, args in tool_calls]
        chunks.append({"message": {"role": "assistant", "content": "", "tool_calls": calls}, "done": False})
    chunks.append({"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": done_reason})
    return chunks


def json_reply(data: dict) -> list[dict]:
    return reply(json.dumps(data), pieces=5)


class FakeOllama:
    """Each chat() consumes the next scripted reply. An exception in the script is raised when reached."""

    def __init__(self, *replies, tools: bool = True, thinking: bool = False, context: int | None = 32768):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.info = ModelInfo("haven", tools, thinking, context)
        self.missing: set[str] = set()
        self.unavailable: Exception | None = None
        self.pulled: list[str] = []
        self.shown: dict = {}

    def _check(self, model: str) -> None:
        if self.unavailable:
            raise self.unavailable
        if model in self.missing:
            raise ModelNotFound(f"model '{model}' not found", 404)

    def inspect(self, model: str) -> ModelInfo:
        self._check(model)
        return dataclasses.replace(self.info, name=model)

    def show(self, model: str) -> dict:
        self._check(model)
        return self.shown

    def version(self, timeout: float = 3) -> str:
        if self.unavailable:
            raise self.unavailable
        return "0.12.0"

    def pull(self, model: str):
        self.pulled.append(model)
        self.missing.discard(model)
        yield {"status": "pulling manifest"}
        yield {"status": "downloading", "total": 100, "completed": 50}
        yield {"status": "success"}

    def chat(self, **payload):
        self._check(payload["model"])
        self.calls.append(copy.deepcopy(payload))
        if not self.replies:
            raise AssertionError("the fake model ran out of scripted replies")
        script = self.replies.pop(0)
        if isinstance(script, BaseException):
            raise script
        return self._stream(script)

    @staticmethod
    def _stream(script: list):
        for chunk in script:
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk


class FakeWeb:
    def __init__(self, results: dict | None = None, pages: dict | None = None):
        self.results = results or {}
        self.pages = pages or {}
        self.searches: list[str] = []
        self.fetches: list[str] = []

    def search(self, query: str, limit: int = 6) -> list[SearchResult]:
        self.searches.append(query)
        found = self.results.get(query, [])
        if isinstance(found, Exception):
            raise found
        return found

    def fetch(self, url: str, max_chars: int = 10000) -> Page:
        self.fetches.append(url)
        page = self.pages.get(url, WebError("the site answered with an error (404 Not Found)"))
        if isinstance(page, Exception):
            raise page
        return page
