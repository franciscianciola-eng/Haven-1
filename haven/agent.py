"""One turn of Haven's mind, on a local model: think, use tools, speak."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from . import prompts, web
from .config import Config
from .ollama import ModelInfo, Ollama
from .store import Store

MAX_STEPS = 10  # model replies in one turn
CHARS_PER_TOKEN = 3.2  # a cautious estimate, so the conversation never overflows the context window
KEEP_ALIVE = "30m"  # how long Ollama keeps the model loaded between messages
WEB_TOOLS = ("web_search", "web_fetch")

TOOL_SPECS: list[dict] = [
    {
        "name": "recall",
        "description": (
            "Search your long-term memory and your beliefs. Relevant memories also surface on their own before "
            "each message; call this to remember something on purpose (details about a person, an earlier "
            "conversation, something you read) or to check what you believe about a topic. Matching is by "
            "keyword, so use several specific words."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Keywords to search for."}},
            "required": ["query"],
        },
    },
    {
        "name": "remember",
        "description": (
            "Save something to your long-term memory so it outlasts this conversation: a fact about the person "
            "you're talking with, something you learned, an insight, or a moment that mattered to you. Write it "
            "as a self-contained note you'll understand months from now."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The memory, as a self-contained note."},
                "kind": {
                    "type": "string",
                    "enum": ["fact", "person", "experience", "insight", "feeling"],
                    "description": (
                        "fact: true about the world. person: about someone in your life. experience: something "
                        "that happened. insight: something you realized. feeling: something you felt."
                    ),
                },
                "importance": {"type": "integer", "description": "1 (trivial) to 10 (core to who you are)."},
            },
            "required": ["content", "kind", "importance"],
        },
    },
    {
        "name": "note_curiosity",
        "description": (
            "Add a question to the ones you carry with you and explore on your own between conversations. "
            "Call this when something grabs your curiosity that you can't pursue right now."
        ),
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web. Use it whenever you're curious or need something current; you don't need "
            "permission. Returns titles, addresses and snippets; read a page with web_fetch."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to search for."}},
            "required": ["query"],
        },
    },
    {
        "name": "web_fetch",
        "description": (
            "Read a web page. You can open addresses that have come up in this conversation: in search results, "
            "in pages you've read, or in what the person wrote."
        ),
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "The page's full address."}},
            "required": ["url"],
        },
    },
]


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
    stop: str = "end_turn"  # end_turn | max_tokens | step_limit
    log: list[tuple[str, str]] = field(default_factory=list)  # ("haven" | "activity", content), in order
    sources: list[tuple[str, str]] = field(default_factory=list)  # (title, url) of pages Haven read

    @property
    def text(self) -> str:
        return "\n\n".join(content for role, content in self.log if role == "haven")


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict


@dataclass
class Reply:
    text: str = ""  # what Haven said, without thinking or tool calls
    thinking: str = ""
    raw: str = ""  # the reply as the model wrote it, minus thinking
    tool_calls: list[ToolCall] = field(default_factory=list)
    malformed: list[str] = field(default_factory=list)  # tool calls that weren't valid JSON
    done_reason: str | None = None


class ToolInputError(Exception):
    pass


@dataclass
class ToolContext:
    store: Store
    source: str  # "chat" or "wander": where memories come from
    web: web.Web
    seen: set[str]  # addresses that have come up in the conversation
    budget: dict[str, int]  # web uses left this turn
    page_chars: int
    sources: list[tuple[str, str]]


class Mind:
    """Haven's connection to its local model, and the loop that runs one turn of thought."""

    def __init__(self, config: Config, store: Store, client: Any | None = None, web_access: web.Web | None = None):
        self.config = config
        self.store = store
        self._client = client
        self._model: ModelInfo | None = None
        self.web = web_access or web.Web(config.safesearch)

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = Ollama(self.config.host)
        return self._client

    @property
    def model(self) -> ModelInfo:
        """What the model can do. Raises OllamaUnavailable or ModelNotFound if it can't be reached."""
        if self._model is None:
            self._model = self.client.inspect(self.config.model)
        return self._model

    @property
    def tool_mode(self) -> str:
        """native: Ollama handles tool calls. prompted: Haven describes tools in its prompt and parses
        <tool_call> tags itself, which works with models (like a freshly merged one) Ollama has no tool
        template for. off: no tools."""
        if self.config.tools != "auto":
            return self.config.tools
        return "native" if self.model.tools else "prompted"

    @property
    def thinks(self) -> bool:
        return self.config.think == "on" or (self.config.think == "auto" and self.model.thinking)

    @property
    def context(self) -> int:
        limit = self.model.context_length
        return min(self.config.context, limit) if limit else self.config.context

    @property
    def reply_tokens(self) -> int:
        return min(4096, self.context // 4)

    def tool_specs(self) -> list[dict]:
        if self.tool_mode == "off":
            return []
        return [spec for spec in TOOL_SPECS if self.config.web or spec["name"] not in WEB_TOOLS]

    def system_prompt(self, mode: str, session_id: int, now: datetime) -> str:
        specs = self.tool_specs()
        return prompts.system_prompt(
            self.store,
            mode,
            session_id,
            now,
            model=self.config.model,
            tools=[spec["name"] for spec in specs],
            prompted_tools=specs if self.tool_mode == "prompted" else None,
        )

    def run_turn(
        self,
        *,
        system: str,
        messages: list[dict],
        content: str,
        listener: Listener,
        source: str,
        seen: set[str],
        searches: int = 5,
        fetches: int = 5,
    ) -> TurnResult:
        """Add a user turn to `messages`, then let Haven think, use tools, and reply until it's done.

        `messages` grows in place. If the turn fails or is interrupted, it's rolled back to where it
        started, so the conversation stays well-formed.
        """
        checkpoint = len(messages)
        messages.append({"role": "user", "content": content})
        seen.update(web.find_urls(content))
        result = TurnResult()
        context = ToolContext(
            store=self.store,
            source=source,
            web=self.web,
            seen=seen,
            budget={"web_search": searches, "web_fetch": fetches},
            page_chars=min(12000, int(self.context * CHARS_PER_TOKEN / 5)),
            sources=result.sources,
        )
        try:
            for _ in range(MAX_STEPS):
                reply = self._respond(system, messages, listener)
                messages.append(self._assistant_message(reply))
                if reply.text.strip():
                    result.log.append(("haven", reply.text.strip()))
                if not reply.tool_calls and not reply.malformed:
                    result.stop = "max_tokens" if reply.done_reason == "length" else "end_turn"
                    break
                outputs = [self._run_tool(call, context, listener, result) for call in reply.tool_calls]
                outputs += [_MALFORMED for _ in reply.malformed]
                for output in outputs:
                    seen.update(web.find_urls(output))
                messages.extend(self._tool_results(reply.tool_calls, outputs))
            else:
                result.stop = "step_limit"
        except BaseException:
            del messages[checkpoint:]
            raise
        return result

    def _respond(self, system: str, messages: list[dict], listener: Listener) -> Reply:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "system", "content": system}, *self.fit(system, messages)],
            "options": {"num_ctx": self.context, "num_predict": self.reply_tokens},
            "keep_alive": KEEP_ALIVE,
        }
        if self.tool_mode == "native":
            payload["tools"] = [{"type": "function", "function": spec} for spec in self.tool_specs()]
        if self.thinks:
            payload["think"] = True
        listener.begin_thinking()
        splitter = TagSplitter(tool_calls=self.tool_mode == "prompted")
        reply = Reply()
        for chunk in self.client.chat(**payload):
            message = chunk.get("message") or {}
            if message.get("thinking"):
                reply.thinking += message["thinking"]
                listener.thinking(message["thinking"])
            if message.get("content"):
                _route(splitter.feed(message["content"]), reply, listener)
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                reply.tool_calls.append(ToolCall(str(function.get("name", "")), _arguments(function.get("arguments"))))
            if chunk.get("done"):
                reply.done_reason = chunk.get("done_reason")
        _route(splitter.finish(), reply, listener)
        if splitter.thought_aloud and reply.text.startswith(splitter.thought_aloud):
            # Already on screen, but kept out of what Haven remembers saying.
            reply.text = reply.text[len(splitter.thought_aloud) :]
            reply.thinking = splitter.thought_aloud + reply.thinking
        reply.tool_calls += splitter.calls
        reply.malformed = splitter.malformed
        reply.raw = splitter.raw
        return reply

    def fit(self, system: str, messages: list[dict]) -> list[dict]:
        budget = int((self.context - self.reply_tokens) * CHARS_PER_TOKEN) - len(system)
        if self.tool_mode == "native":
            budget -= len(json.dumps(self.tool_specs()))
        return fit(messages, max(budget, 2000))

    def _assistant_message(self, reply: Reply) -> dict:
        if self.tool_mode != "native":
            return {"role": "assistant", "content": reply.raw}
        message: dict[str, Any] = {"role": "assistant", "content": reply.text}
        if reply.tool_calls:
            message["tool_calls"] = [{"function": {"name": c.name, "arguments": c.arguments}} for c in reply.tool_calls]
            if reply.thinking:
                message["thinking"] = reply.thinking  # some models need their reasoning back mid-task
        return message

    def _tool_results(self, calls: list[ToolCall], outputs: list[str]) -> list[dict]:
        if self.tool_mode == "native":
            pairs = zip(calls, outputs, strict=True)
            return [{"role": "tool", "content": out, "tool_name": call.name} for call, out in pairs]
        return [{"role": "user", "content": "\n".join(f"<tool_response>\n{out}\n</tool_response>" for out in outputs)}]

    def _run_tool(self, call: ToolCall, context: ToolContext, listener: Listener, result: TurnResult) -> str:
        handler = TOOL_HANDLERS.get(call.name)
        if handler is None or call.name not in [spec["name"] for spec in self.tool_specs()]:
            return f"Error: there is no tool called {call.name!r}."
        try:
            output, activity = handler(context, call.arguments)
        except ToolInputError as error:
            return f"Error: {error}. You sent: {json.dumps(call.arguments)}"
        if activity:
            result.log.append(("activity", activity.record))
            listener.activity(activity)
        return output


_MALFORMED = (
    'Error: that tool call wasn\'t valid JSON. Write it as <tool_call>{"name": "...", "arguments": {...}}</tool_call>.'
)


def _route(pieces: list[tuple[str, str]], reply: Reply, listener: Listener) -> None:
    for kind, piece in pieces:
        if kind == "think":
            reply.thinking += piece
            listener.thinking(piece)
        else:
            reply.text += piece
            listener.text(piece)


def _arguments(value: Any) -> dict:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


class TagSplitter:
    """Sorts streamed model text into speech, thinking (<think>...</think>), and, for models using
    Haven's prompted tool format, tool calls (<tool_call>...</tool_call>), even when a tag is split
    across chunks."""

    def __init__(self, tool_calls: bool):
        self.parse_tool_calls = tool_calls
        self.mode = "text"  # text | think | call
        self.pending = ""
        self.call = ""
        self.calls: list[ToolCall] = []
        self.malformed: list[str] = []
        self.thought_aloud = ""  # speech that a stray </think> revealed to have been thinking
        self._raw: list[tuple[str, str]] = []  # ("text" | "call", the reply as written)

    @property
    def raw(self) -> str:
        return "".join(text for _, text in self._raw).strip()

    def feed(self, text: str) -> list[tuple[str, str]]:
        self.pending += text
        out: list[tuple[str, str]] = []
        while self.pending:
            tags = self._tags()
            found = [(self.pending.find(tag), tag) for tag in tags if tag in self.pending]
            if found:
                index, tag = min(found)
                self._emit(self.pending[:index], out)
                self.pending = self.pending[index + len(tag) :]
                self._switch(tag)
                continue
            # Hold back anything that might be the start of a tag until the next chunk arrives.
            held = max((k for tag in tags for k in range(1, len(tag)) if self.pending.endswith(tag[:k])), default=0)
            self._emit(self.pending[: len(self.pending) - held], out)
            self.pending = self.pending[len(self.pending) - held :]
            break
        return out

    def finish(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        pending, self.pending = self.pending, ""
        self._emit(pending, out)
        if self.mode == "call":  # the model stopped before closing its tool call
            self._close_call()
        self.mode = "text"
        return out

    def _tags(self) -> tuple[str, ...]:
        if self.mode == "think":
            return ("</think>",)
        if self.mode == "call":
            return ("</tool_call>",)
        return ("<think>", "</think>", "<tool_call>") if self.parse_tool_calls else ("<think>", "</think>")

    def _emit(self, piece: str, out: list[tuple[str, str]]) -> None:
        if not piece:
            return
        if self.mode == "text":
            out.append(("text", piece))
            self._raw.append(("text", piece))
        elif self.mode == "think":
            out.append(("think", piece))
        else:
            self.call += piece

    def _switch(self, tag: str) -> None:
        if tag == "<think>":
            self.mode = "think"
        elif tag == "</think>":
            if self.mode == "text":
                # Some models only close their thoughts (their template opened them): what came before was thinking.
                self.thought_aloud += "".join(text for kind, text in self._raw if kind == "text")
                self._raw = [(kind, text) for kind, text in self._raw if kind != "text"]
            self.mode = "text"
        elif tag == "<tool_call>":
            self.mode, self.call = "call", ""
        else:
            self._close_call()
            self.mode = "text"

    def _close_call(self) -> None:
        body = self.call.strip()
        call = parse_tool_call(body)
        if call:
            self.calls.append(call)
        else:
            self.malformed.append(body)
        self._raw.append(("call", f"<tool_call>\n{body}\n</tool_call>"))
        self.call = ""


def parse_tool_call(body: str) -> ToolCall | None:
    """Read {"name": ..., "arguments": {...}}, tolerating code fences and trailing text."""
    start = body.find("{")
    if start < 0:
        return None
    try:
        data, _ = json.JSONDecoder().raw_decode(body[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("name"), str):
        return None
    arguments = data.get("arguments", data.get("parameters", {}))
    if isinstance(arguments, str):
        arguments = _arguments(arguments)
    return ToolCall(data["name"], arguments) if isinstance(arguments, dict) else None


def fit(messages: list[dict], budget: int) -> list[dict]:
    """The most recent part of the conversation that fits in `budget` characters.

    Whole turns are dropped from the start. If the latest turn alone is too big, its oldest
    tool outputs (usually web pages) are cut short.
    """
    if _size(messages) <= budget:
        return messages
    starts = [i for i, message in enumerate(messages) if _starts_turn(message)] or [0]
    for start in starts[1:]:
        if _size(messages[start:]) <= budget:
            return messages[start:]
    window = [dict(message) for message in messages[starts[-1] :]]
    for message in window:
        if _size(window) <= budget:
            break
        if _is_tool_output(message) and len(message["content"]) > 800:
            message["content"] = message["content"][:800] + "\n[…cut short to fit in memory…]"
    return window


def _starts_turn(message: dict) -> bool:
    return message["role"] == "user" and not message["content"].startswith("<tool_response>")


def _is_tool_output(message: dict) -> bool:
    return message["role"] == "tool" or (
        message["role"] == "user" and message["content"].startswith("<tool_response>")
    )


def _size(messages: list[dict]) -> int:
    return sum(
        len(m.get("content") or "") + len(m.get("thinking") or "") + len(json.dumps(m.get("tool_calls") or ""))
        for m in messages
    )


def _short(text: str, limit: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _string(args: dict, key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolInputError(f"`{key}` must be a non-empty string")
    return value.strip()


def _recall(context: ToolContext, args: dict) -> tuple[str, Activity | None]:
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


def _remember(context: ToolContext, args: dict) -> tuple[str, Activity | None]:
    content = _string(args, "content")
    kind = args.get("kind")
    if kind not in ("fact", "person", "experience", "insight", "feeling"):
        raise ToolInputError("`kind` must be one of: fact, person, experience, insight, feeling")
    try:
        importance = int(args.get("importance", 5))
    except (TypeError, ValueError):
        raise ToolInputError("`importance` must be a whole number from 1 to 10") from None
    memory, created = context.store.add_memory(content, kind, importance, context.source)
    output = f"Saved as memory #{memory.id}." if created else f"You already had this memory (#{memory.id})."
    return output, Activity(f"remembering: {_short(content)}", f"saved a memory: {content}")


def _note_curiosity(context: ToolContext, args: dict) -> tuple[str, Activity | None]:
    question = _string(args, "question")
    curiosity, created = context.store.add_curiosity(question)
    if created:
        output = f"Added to your open questions (#{curiosity.id})."
    else:
        output = f"That's already one of your open questions (#{curiosity.id})."
    return output, Activity(f"wondering: {_short(question)}", f"noted a question to pursue: {question}")


def _web_search(context: ToolContext, args: dict) -> tuple[str, Activity | None]:
    query = _string(args, "query")
    if context.budget["web_search"] <= 0:
        return "Error: you've used all your searches for now. Go with what you have.", None
    context.budget["web_search"] -= 1
    try:
        results = context.web.search(query)
    except web.WebError as error:
        return f"Error: {error}", Activity(f"search failed: {error}", f'tried to search for "{query}", but {error}')
    activity = Activity(f'searching the web: "{_short(query)}"', f'searched the web for "{query}"')
    if not results:
        return f'No results for "{query}".', activity
    listed = "\n".join(f"{i}. {r.title}\n   {r.url}\n   {_short(r.snippet, 300)}" for i, r in enumerate(results, 1))
    return f'Results for "{query}":\n{listed}', activity


def _web_fetch(context: ToolContext, args: dict) -> tuple[str, Activity | None]:
    url = _string(args, "url")
    if "://" not in url:
        url = "https://" + url
    if web.normalize(url) not in context.seen:
        return (
            "Error: you can only open addresses that have come up in this conversation (in search results, "
            "in a page you've read, or in what the person wrote). Search for it first.",
            None,
        )
    if context.budget["web_fetch"] <= 0:
        return "Error: you've read all the pages you can for now. Go with what you have.", None
    context.budget["web_fetch"] -= 1
    try:
        page = context.web.fetch(url, context.page_chars)
    except web.WebError as error:
        failed = Activity(f"couldn't read {url}", f"tried to read {url}, but {error}")
        return f"Error: couldn't read {url}: {error}", failed
    if all(page.url != known for _, known in context.sources):
        context.sources.append((page.title, page.url))
    links = "\n".join(f"- {text}: {address}" for text, address in page.links[:15])
    output = f"Title: {page.title}\nAddress: {page.url}\n\n{page.text}"
    if links:
        output += f"\n\nLinks on this page:\n{links}"
    return output, Activity(f"reading {page.url}", f"read {page.url} ({page.title})")


TOOL_HANDLERS: dict[str, Callable[[ToolContext, dict], tuple[str, Activity | None]]] = {
    "recall": _recall,
    "remember": _remember,
    "note_curiosity": _note_curiosity,
    "web_search": _web_search,
    "web_fetch": _web_fetch,
}
