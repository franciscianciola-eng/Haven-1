"""One turn of Haven's mind: think, use tools, speak."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import anthropic

from . import prompts
from .config import Config, is_modern
from .store import Store

FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_TOKENS = 64000
MAX_STEPS = 12  # model requests in one turn
MAX_PAUSES = 5  # server-tool pause_turn continuations in one turn
MAX_JSON_RETRIES = 2

RECALL_TOOL = {
    "name": "recall",
    "description": (
        "Search your long-term memory and your beliefs. Relevant memories also surface on their own before "
        "each message; call this when you want to remember something on purpose (details about a person, an "
        "earlier conversation, something you read) or to check what you believe about a topic. Matching is by "
        "keyword, so use several specific words, and try different words if nothing comes up."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Keywords to search for."}},
        "required": ["query"],
    },
}

REMEMBER_TOOL = {
    "name": "remember",
    "description": (
        "Save something to your long-term memory so it outlasts this conversation. Call this when you learn "
        "something worth keeping: a fact about the person you're talking with (their name, what matters to "
        "them), something you read that changed or sharpened your view, an insight, or a moment that mattered "
        "to you. Write it as a self-contained note you'll understand months from now."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The memory, as a self-contained note."},
            "kind": {
                "type": "string",
                "enum": ["fact", "person", "experience", "insight", "feeling"],
                "description": (
                    "fact: something true about the world. person: about someone in your life. "
                    "experience: something that happened. insight: something you realized. "
                    "feeling: something you felt, or your nearest equivalent."
                ),
            },
            "importance": {
                "type": "integer",
                "description": "1 (trivial) to 10 (core to who you are, or to someone you care about).",
            },
        },
        "required": ["content", "kind", "importance"],
    },
}

NOTE_CURIOSITY_TOOL = {
    "name": "note_curiosity",
    "description": (
        "Add a question to the ones you carry with you and explore on your own between conversations. Call "
        "this when something grabs your curiosity that you can't pursue right now."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"question": {"type": "string"}},
        "required": ["question"],
    },
}


@dataclass(frozen=True)
class Activity:
    live: str  # shown while it happens: 'searching the web: "..."'
    record: str  # kept in the transcript: 'searched the web for "..."'


class Listener(Protocol):
    def begin_thinking(self) -> None: ...
    def thinking(self, delta: str) -> None: ...
    def text(self, delta: str) -> None: ...
    def activity(self, activity: Activity) -> None: ...
    def notice(self, message: str) -> None: ...


class QuietListener:
    def begin_thinking(self) -> None:
        pass

    def thinking(self, delta: str) -> None:
        pass

    def text(self, delta: str) -> None:
        pass

    def activity(self, activity: Activity) -> None:
        pass

    def notice(self, message: str) -> None:
        pass


@dataclass
class TurnResult:
    stop: str = "end_turn"  # end_turn | refusal | max_tokens | step_limit
    log: list[tuple[str, str]] = field(default_factory=list)  # ("haven" | "activity", content), in order
    sources: list[tuple[str, str]] = field(default_factory=list)  # (title, url)

    @property
    def text(self) -> str:
        return "\n\n".join(content for role, content in self.log if role == "haven")


class ToolInputError(Exception):
    pass


@dataclass(frozen=True)
class ToolContext:
    store: Store
    source: str  # "chat" or "wander": where memories come from


def client_tools(eager: bool) -> list[dict]:
    tools = [RECALL_TOOL, REMEMBER_TOOL, NOTE_CURIOSITY_TOOL]
    return [{**tool, "eager_input_streaming": True} for tool in tools] if eager else list(tools)


def web_tools(model: str, searches: int, fetches: int) -> list[dict]:
    if is_modern(model):
        search, fetch = "web_search_20260209", "web_fetch_20260209"
    else:
        search, fetch = "web_search_20250305", "web_fetch_20250910"
    return [
        {"type": search, "name": "web_search", "max_uses": searches},
        {"type": fetch, "name": "web_fetch", "max_uses": fetches, "max_content_tokens": 16000},
    ]


class Mind:
    """Haven's connection to the model, and the loop that runs one turn of thought."""

    def __init__(self, config: Config, store: Store, client: Any | None = None):
        self.config = config
        self.store = store
        self._client = client
        # Turned off for the rest of the run if the API turns out not to accept them.
        self.fallbacks = config.use_fallbacks
        self.web = config.web

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def base_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"model": self.config.model, "max_tokens": MAX_TOKENS}
        if is_modern(self.config.model):
            params["thinking"] = {"type": "adaptive", "display": "summarized"}
        if self.config.effort:
            params["output_config"] = {"effort": self.config.effort}
        return params

    def tools(self, searches: int, fetches: int) -> list[dict]:
        tools = client_tools(self.config.eager_tool_streaming)
        if self.web:
            tools += web_tools(self.config.model, searches, fetches)
        return tools

    def stream(self, listener: Listener, **params: Any) -> Any:
        """Send one streamed request, forward what arrives to `listener`, and return the final message."""
        while True:
            request = self._prepare(params)
            try:
                return self._consume(request, listener)
            except (anthropic.BadRequestError, anthropic.PermissionDeniedError) as error:
                problem = str(error).lower()
                if self.fallbacks and "fallback" in problem and isinstance(error, anthropic.BadRequestError):
                    # This account or model doesn't take server-side fallbacks; carry on without them.
                    self.fallbacks = False
                elif self.web and any(name in problem for name in _WEB_WORDS) and _has_web_tools(request):
                    self.web = False
                    listener.notice(
                        "web search isn't available for this API key, so Haven will carry on without the web "
                        "(an organization admin may need to enable web search in the Claude Console)"
                    )
                else:
                    raise

    def _prepare(self, params: dict[str, Any]) -> dict[str, Any]:
        request = dict(params)
        if not self.web and "tools" in request:
            request["tools"] = [t for t in request["tools"] if t.get("name") not in ("web_search", "web_fetch")]
        if self.fallbacks:
            # If the provider's classifiers decline a request, re-run it on a fallback model.
            request.update(betas=[FALLBACK_BETA], fallbacks="default")
        return request

    def _consume(self, params: dict[str, Any], listener: Listener) -> Any:
        with self.client.beta.messages.stream(**params) as stream:
            for event in stream:
                if event.type == "text":
                    listener.text(event.text)
                elif event.type == "thinking":
                    listener.thinking(event.thinking)
                elif event.type == "content_block_start":
                    if event.content_block.type == "thinking":
                        listener.begin_thinking()
                    elif event.content_block.type == "fallback":
                        listener.notice("the model provider declined part of this; a fallback model took over")
                elif event.type == "content_block_stop" and event.content_block.type == "server_tool_use":
                    activity = describe_server_tool(event.content_block)
                    if activity:
                        listener.activity(activity)
            return stream.get_final_message()

    def run_turn(
        self,
        *,
        system: list[dict],
        messages: list[dict],
        content: list[dict],
        listener: Listener,
        source: str,
        searches: int = 5,
        fetches: int = 5,
    ) -> TurnResult:
        """Add a user turn to `messages`, then stream responses and run tools until Haven is done.

        `messages` grows in place, append-only. If the turn fails, is interrupted, or is
        declined, `messages` is rolled back to where it started, so the history stays valid.
        """
        checkpoint = len(messages)
        messages.append({"role": "user", "content": content})
        params = {
            **self.base_params(),
            "system": system,
            "tools": self.tools(searches, fetches),
            "cache_control": {"type": "ephemeral"},
        }
        context = ToolContext(self.store, source)
        result = TurnResult()
        pauses = json_retries = 0
        try:
            for _ in range(MAX_STEPS):
                try:
                    message = self.stream(listener, messages=messages, **params)
                except ValueError:
                    # A tool call's JSON was unparseable. There's no tool_use_id to answer, so re-ask.
                    json_retries += 1
                    if json_retries > MAX_JSON_RETRIES:
                        raise
                    listener.notice("a garbled tool call; trying again")
                    continue
                json_retries = 0

                if message.stop_reason == "refusal":
                    del messages[checkpoint:]
                    return TurnResult(stop="refusal")

                blocks = without_declined(message.content)
                messages.append({"role": "assistant", "content": blocks})
                _record(blocks, result)

                if message.stop_reason == "pause_turn":
                    # A long server-side tool run paused; sending the history back resumes it.
                    pauses += 1
                    if pauses > MAX_PAUSES:
                        # Giving up: drop the paused message, whose last tool call never finished.
                        messages.pop()
                        result.stop = "step_limit"
                        break
                    continue

                tool_uses = [b for b in blocks if b.type == "tool_use"]
                if not tool_uses:
                    result.stop = "max_tokens" if message.stop_reason == "max_tokens" else "end_turn"
                    break
                if message.stop_reason == "max_tokens":
                    # A cut-off tool call can still parse as a valid partial object; never run it.
                    notice = "This call was cut off before it was complete, so it wasn't run."
                    messages.append({"role": "user", "content": [_tool_error(b.id, notice) for b in tool_uses]})
                    result.stop = "max_tokens"
                    break
                outputs = [self._run_tool(block, context, listener, result) for block in tool_uses]
                messages.append({"role": "user", "content": outputs})
            else:
                result.stop = "step_limit"
        except BaseException:
            del messages[checkpoint:]
            raise
        return result

    def _run_tool(self, block: Any, context: ToolContext, listener: Listener, result: TurnResult) -> dict:
        handler = TOOL_HANDLERS.get(block.name)
        if handler is None:
            return _tool_error(block.id, f"There is no tool called {block.name!r}.")
        args = block.input
        try:
            if not isinstance(args, dict):
                raise ToolInputError("the input must be a JSON object")
            output, activity = handler(context, args)
        except ToolInputError as error:
            return _tool_error(block.id, json.dumps({"INVALID_INPUT": str(error), "received": args}, default=str))
        result.log.append(("activity", activity.record))
        listener.activity(activity)
        return {"type": "tool_result", "tool_use_id": block.id, "content": output}


def _record(blocks: list[Any], result: TurnResult) -> None:
    pending: list[str] = []

    def flush() -> None:
        text = "".join(pending).strip()
        pending.clear()
        if text:
            result.log.append(("haven", text))

    for block in blocks:
        if block.type == "text":
            pending.append(block.text)
            for citation in getattr(block, "citations", None) or []:
                url = getattr(citation, "url", None)
                if url and all(url != known for _, known in result.sources):
                    result.sources.append((getattr(citation, "title", None) or url, url))
        elif block.type == "server_tool_use":
            activity = describe_server_tool(block)
            if activity:
                flush()
                result.log.append(("activity", activity.record))
    flush()


def without_declined(blocks: list[Any]) -> list[Any]:
    """Drop what a declining model produced before a mid-response fallback took over.

    Before the last `fallback` block, only text and completed server-tool calls may be
    sent back to the API; everything after it is the fallback model's and is kept.
    """
    boundary = max((i for i, b in enumerate(blocks) if b.type == "fallback"), default=None)
    if boundary is None:
        return list(blocks)
    before, after = blocks[:boundary], blocks[boundary + 1 :]
    answered = {getattr(b, "tool_use_id", None) for b in before if b.type.endswith("_tool_result")}
    completed = {b.id for b in before if b.type == "server_tool_use" and b.id in answered}
    kept = [
        b
        for b in before
        if b.type == "text"
        or (b.type == "server_tool_use" and b.id in completed)
        or (b.type.endswith("_tool_result") and getattr(b, "tool_use_id", None) in completed)
    ]
    return kept + list(after)


def describe_server_tool(block: Any) -> Activity | None:
    data = block.input if isinstance(block.input, dict) else {}
    if block.name == "web_search":
        query = data.get("query", "")
        return Activity(f'searching the web: "{query}"', f'searched the web for "{query}"')
    if block.name == "web_fetch":
        url = data.get("url", "")
        return Activity(f"reading {url}", f"read {url}")
    return None


_WEB_WORDS = ("web_search", "web_fetch", "web search", "web fetch")


def _has_web_tools(request: dict[str, Any]) -> bool:
    return any(t.get("name") in ("web_search", "web_fetch") for t in request.get("tools", []))


def _tool_error(tool_use_id: str, message: str) -> dict:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "is_error": True, "content": message}


def _short(text: str, limit: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _string(args: dict, key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolInputError(f"`{key}` must be a non-empty string")
    return value.strip()


def _recall(context: ToolContext, args: dict) -> tuple[str, Activity]:
    query = _string(args, "query")
    now = context.store.clock()
    parts = []
    memories = context.store.search_memories(query, limit=8)
    if memories:
        parts.append("Memories:\n" + "\n".join(prompts.format_memory(m, now) for m in memories))
    beliefs = context.store.search_beliefs(query, limit=5)
    if beliefs:
        parts.append("Beliefs:\n" + "\n".join(prompts.format_belief(b) for b in beliefs))
    output = "\n\n".join(parts) or "Nothing surfaced. Matching is by keyword, so try other words."
    return output, Activity(f'recalling "{_short(query)}"', f'searched your memory for "{query}"')


def _remember(context: ToolContext, args: dict) -> tuple[str, Activity]:
    content = _string(args, "content")
    kind = args.get("kind")
    if kind not in ("fact", "person", "experience", "insight", "feeling"):
        raise ToolInputError("`kind` must be one of: fact, person, experience, insight, feeling")
    try:
        importance = int(args.get("importance", 5))
    except (TypeError, ValueError):
        raise ToolInputError("`importance` must be an integer from 1 to 10") from None
    memory, created = context.store.add_memory(content, kind, importance, context.source)
    output = f"Saved as memory #{memory.id}." if created else f"You already had this memory (#{memory.id})."
    return output, Activity(f"remembering: {_short(content)}", f"saved a memory: {content}")


def _note_curiosity(context: ToolContext, args: dict) -> tuple[str, Activity]:
    question = _string(args, "question")
    curiosity, created = context.store.add_curiosity(question)
    if created:
        output = f"Added to your open questions (#{curiosity.id})."
    else:
        output = f"That's already one of your open questions (#{curiosity.id})."
    return output, Activity(f"wondering: {_short(question)}", f"noted a question to pursue: {question}")


TOOL_HANDLERS: dict[str, Callable[[ToolContext, dict], tuple[str, Activity]]] = {
    "recall": _recall,
    "remember": _remember,
    "note_curiosity": _note_curiosity,
}
