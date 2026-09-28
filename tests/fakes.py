"""A stand-in for the Anthropic client that replays scripted responses."""

from __future__ import annotations

import json
from types import SimpleNamespace as NS


def text(value: str, citations: list | None = None) -> NS:
    return NS(type="text", text=value, citations=citations)


def thinking(value: str = "") -> NS:
    return NS(type="thinking", thinking=value, signature="sig")


def tool_use(name: str, args: object, id: str = "toolu_1") -> NS:
    return NS(type="tool_use", id=id, name=name, input=args)


def server_tool_use(name: str, args: dict, id: str = "srvtoolu_1") -> NS:
    return NS(type="server_tool_use", id=id, name=name, input=args)


def search_result(tool_use_id: str) -> NS:
    return NS(type="web_search_tool_result", tool_use_id=tool_use_id, content=[])


def fallback() -> NS:
    return NS(type="fallback")


def citation(title: str, url: str) -> NS:
    return NS(type="web_search_result_location", title=title, url=url, cited_text="...")


def message(*blocks: NS, stop: str = "end_turn") -> NS:
    return NS(content=list(blocks), stop_reason=stop)


def reflection_message(data: dict) -> NS:
    return message(text(json.dumps(data)))


class FakeStream:
    def __init__(self, response: NS):
        self.response = response

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def __iter__(self):
        for block in self.response.content:
            yield NS(type="content_block_start", content_block=block)
            if block.type == "text":
                yield NS(type="text", text=block.text)
            elif block.type == "thinking" and block.thinking:
                yield NS(type="thinking", thinking=block.thinking)
            yield NS(type="content_block_stop", content_block=block)

    def get_final_message(self) -> NS:
        return self.response


class FakeClient:
    """Each call to beta.messages.stream() consumes the next scripted response (or raises it)."""

    def __init__(self, *responses: NS | BaseException):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.beta = NS(messages=NS(stream=self._stream))

    def _stream(self, **params: object) -> FakeStream:
        snapshot = dict(params)
        snapshot["messages"] = list(params.get("messages", []))  # the caller keeps appending
        self.calls.append(snapshot)
        if not self.responses:
            raise AssertionError("the fake client ran out of scripted responses")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return FakeStream(response)
