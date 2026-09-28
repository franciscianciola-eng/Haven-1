"""The `haven` command: talk with Haven, let it wander, forge its model, and look inside its mind."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Callable
from pathlib import Path

from . import prompts
from .agent import Mind
from .config import DEFAULT_MODEL, MIN_CONTEXT, THINK_SETTINGS, TOOL_SETTINGS, Config, parse_host
from .forge import MIN_FREE_GB, OUTTYPES, RECIPE, Forge, ForgeError
from .ollama import ModelNotFound, OllamaError, OllamaUnavailable
from .reflection import ReflectionError, reflect
from .session import Conversation, wander
from .store import Store
from .ui import BOLD, CYAN, DIM, MAGENTA, Terminal

CHAT_HELP = """\
  /self          Haven's self-model, in its own words
  /beliefs       what Haven believes, and how confidently
  /journal       Haven's recent journal entries
  /memories [q]  search Haven's memories (or list the latest)
  /questions     what Haven is curious about
  /reflect       have Haven reflect on the conversation so far
  /thoughts      show or hide Haven's thinking
  /bye           end the conversation (Ctrl-D works too)"""


class _Fatal(Exception):
    """A problem no retry will fix, like Ollama not running. Already reported."""


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    args = _parser().parse_args(argv)
    try:
        config = Config.from_env()
    except ValueError as error:
        print(f"haven: {error}", file=sys.stderr)
        return 2
    overrides = {
        "home": args.home,
        "model": args.model,
        "host": parse_host(args.host) if args.host else None,
        "context": args.context,
        "think": args.think,
        "tools": args.tools,
        "show_thoughts": True if args.show_thoughts else None,
        "web": False if args.no_web else None,
    }
    config = dataclasses.replace(config, **{k: v for k, v in overrides.items() if v is not None})
    store = Store(config.db_path)
    try:
        store.ensure_born(prompts.GENESIS)
        mind = Mind(config, store)
        ui = Terminal(show_thoughts=config.show_thoughts)
        return COMMANDS[args.command or "chat"](mind, ui, args)
    except _Fatal:
        return 1
    finally:
        store.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="haven",
        description="Haven: a persistent AI mind that runs on your own computer, remembers, and grows.",
    )
    parser.add_argument("--home", type=_path, help="where Haven's mind is stored (default: ~/.haven)")
    parser.add_argument("--model", help=f"the Ollama model Haven thinks with (default: {DEFAULT_MODEL})")
    parser.add_argument("--host", help="Ollama's address (default: http://127.0.0.1:11434)")
    parser.add_argument("--context", type=_context, help="how many tokens the model sees at once (default: 16384)")
    parser.add_argument("--think", choices=THINK_SETTINGS, help="reason before answering (default: auto)")
    parser.add_argument("--tools", choices=TOOL_SETTINGS, help="how Haven calls tools (default: auto)")
    parser.add_argument("--show-thoughts", action="store_true", help="show Haven's thinking")
    parser.add_argument("--no-web", action="store_true", help="don't let Haven search or read the web")
    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.add_parser("chat", help="talk with Haven (the default)")
    wander_cmd = sub.add_parser("wander", help="let Haven explore the web on its own, then reflect")
    wander_cmd.add_argument("--steps", type=int, default=1, help="how many explorations in a row (default: 1)")
    sub.add_parser("reflect", help="have Haven reflect on anything it hasn't yet")
    forge_cmd = sub.add_parser("forge", help="build Haven's own model by merging open-source weights")
    forge_cmd.add_argument("--recipe", type=_path, default=RECIPE, help="a mergekit recipe (default: Haven's own)")
    forge_cmd.add_argument("--name", default=DEFAULT_MODEL, help=f"what to call the model (default: {DEFAULT_MODEL})")
    forge_cmd.add_argument("--outtype", choices=OUTTYPES, default="q8_0", help="GGUF precision (default: q8_0)")
    forge_cmd.add_argument("--template-from", metavar="MODEL", help="copy the chat template from this Ollama model")
    forge_cmd.add_argument("--cuda", action="store_true", help="merge on an NVIDIA GPU")
    forge_cmd.add_argument("--keep", action="store_true", help="keep the merged weights and GGUF file")
    forge_cmd.add_argument("--fresh", action="store_true", help="start over instead of resuming")
    forge_cmd.add_argument("--yes", action="store_true", help="don't ask for confirmation")
    self_cmd = sub.add_parser("self", help="Haven's self-model")
    self_cmd.add_argument("--history", action="store_true", help="list every version and why it changed")
    self_cmd.add_argument("--version", type=int, help="show a specific version")
    beliefs_cmd = sub.add_parser("beliefs", help="what Haven believes")
    beliefs_cmd.add_argument("--all", action="store_true", help="include beliefs Haven has let go of")
    beliefs_cmd.add_argument("--history", type=int, metavar="ID", help="how one belief changed over time")
    journal_cmd = sub.add_parser("journal", help="Haven's journal")
    journal_cmd.add_argument("-n", type=int, default=5, help="how many entries (default: 5)")
    memories_cmd = sub.add_parser("memories", help="search Haven's memories, or list the latest")
    memories_cmd.add_argument("query", nargs="*")
    memories_cmd.add_argument("-n", type=int, default=15, help="how many (default: 15)")
    sub.add_parser("questions", help="what Haven is curious about")
    sub.add_parser("status", help="an overview of Haven's mind")
    return parser


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _context(value: str) -> int:
    if not value.isdigit() or int(value) < MIN_CONTEXT:
        raise argparse.ArgumentTypeError(f"must be a number of tokens, at least {MIN_CONTEXT}")
    return int(value)


# --- talking and wandering ---------------------------------------------------


def cmd_chat(mind: Mind, ui: Terminal, args: argparse.Namespace) -> int:
    _ready(mind, ui)
    _consolidate(mind, ui)
    convo = Conversation(mind)
    _banner(mind, ui)
    try:
        while True:
            try:
                text = ui.ask().strip()
            except (EOFError, KeyboardInterrupt):
                ui.write("\n")
                break
            if not text:
                continue
            if text.startswith("/"):
                if _slash(text, convo, ui) == "quit":
                    break
                continue
            ui.start_turn()
            try:
                result = convo.say(text, ui)
            except KeyboardInterrupt:
                ui.notice("interrupted")
                continue
            except OllamaError as error:
                _report(error, mind, ui)
                continue
            ui.finish_turn(result)
    finally:
        convo.close()
    _reflect_on(mind, ui, convo.session_id)
    return 0


def cmd_wander(mind: Mind, ui: Terminal, args: argparse.Namespace) -> int:
    _ready(mind, ui)
    if mind.tool_mode == "off" or not mind.config.web:
        ui.error("Wandering means exploring the web, so it needs Haven's tools and web access turned on.")
        return 1
    _consolidate(mind, ui)
    steps = max(args.steps, 1)
    for step in range(steps):
        ui.line(f"haven · wandering{f' ({step + 1} of {steps})' if steps > 1 else ''}", BOLD, CYAN)
        ui.start_turn()
        try:
            session_id, result = wander(mind, ui)
        except KeyboardInterrupt:
            ui.notice("stopped")
            return 130
        except OllamaError as error:
            _report(error, mind, ui)
            return 1
        ui.finish_turn(result)
        _reflect_on(mind, ui, session_id)
    return 0


def cmd_reflect(mind: Mind, ui: Terminal, args: argparse.Namespace) -> int:
    if not mind.store.unreflected_sessions():
        ui.line("Haven has already reflected on everything it has lived through.", DIM)
        return 0
    _ready(mind, ui)
    return 0 if _consolidate(mind, ui, announce=False) else 1


def cmd_forge(mind: Mind, ui: Terminal, args: argparse.Namespace) -> int:
    forge = Forge(
        mind.config,
        recipe=args.recipe,
        name=args.name,
        outtype=args.outtype,
        template_from=args.template_from,
        cuda=args.cuda,
        keep=args.keep,
        fresh=args.fresh,
        say=ui.line,
        client=mind.client,
    )
    problems = forge.problems()
    if problems:
        ui.error("The forge can't start yet:")
        for problem in problems:
            ui.line(f"  - {problem}")
        return 1
    ui.line(f"Forging '{forge.name}': a {forge.method()} merge of {' and '.join(forge.sources())}.", BOLD)
    ui.line(
        f"This downloads the source weights (about 8 GB for each 4B model), needs roughly {MIN_FREE_GB} GB "
        "of free disk while it works, and takes a while. If it's interrupted, running it again picks up "
        "where it left off.",
        DIM,
    )
    if not args.yes and not ui.confirm("Start forging?"):
        return 1
    try:
        forge.build()
    except ForgeError as error:
        ui.error(str(error))
        return 1
    except KeyboardInterrupt:
        ui.notice("stopped; run `haven forge` again to pick up where it left off")
        return 130
    mind.store.set_meta(f"body:{forge.name}", forge.description())
    ui.line(f"\nHaven's model is ready in Ollama as '{forge.name}'.", BOLD, CYAN)
    ui.line("Run `haven` to talk." if forge.name == mind.config.model else f"Run `haven --model {forge.name}`.")
    return 0


def _ready(mind: Mind, ui: Terminal) -> None:
    """Make sure Ollama is running and has the model, or explain what to do and stop."""
    try:
        mind.model  # noqa: B018  (this reaches Ollama)
        return
    except OllamaUnavailable:
        ui.error(f"Haven's model runs on this computer through Ollama, which isn't answering at {mind.config.host}.")
        ui.line("Install Ollama from https://ollama.com, make sure it's running, then try again.", DIM)
    except ModelNotFound:
        name = mind.config.model
        if name == DEFAULT_MODEL:
            ui.error("Haven's own model hasn't been forged yet.")
            ui.line(
                "Build it from open-source weights with `haven forge` (see the README). Or talk to Haven now "
                "on a model from Ollama's library, for example:  haven --model qwen3:8b",
                DIM,
            )
        elif ui.interactive and ui.confirm(f"Ollama doesn't have {name} yet. Download it now?"):
            if _pull(mind, ui, name):
                mind.model  # noqa: B018
                return
        else:
            ui.error(f"Ollama doesn't have the model {name}. Get it with:  ollama pull {name}")
    except OllamaError as error:
        ui.error(f"Ollama couldn't load {mind.config.model}: {error.message}")
    raise _Fatal


def _pull(mind: Mind, ui: Terminal, name: str) -> bool:
    try:
        for chunk in mind.client.pull(name):
            total, done = chunk.get("total"), chunk.get("completed")
            share = f" {100 * done // total}%" if total and done else ""
            ui.status(f"  {chunk.get('status', '')}{share}")
    except OllamaError as error:
        ui.error(f"Couldn't download {name}: {error.message}")
        return False
    ui.line("")
    return True


def _consolidate(mind: Mind, ui: Terminal, announce: bool = True) -> bool:
    """Reflect on sessions that ended before Haven could (a crash, a closed terminal)."""
    for session_id in mind.store.unreflected_sessions():
        if announce:
            ui.line("(first, Haven reflects on a session it never got to think over)", DIM)
        if not _reflect_on(mind, ui, session_id):
            return False
    return True


def _reflect_on(mind: Mind, ui: Terminal, session_id: int) -> bool:
    if not any(line.role in ("haven", "activity") for line in mind.store.unreflected(session_id)):
        reflect(mind, session_id)  # nothing to think over; this just clears it
        return True
    ui.reflecting()
    try:
        outcome = reflect(mind, session_id, ui, ui.dot)
    except KeyboardInterrupt:
        ui.notice("reflection skipped; Haven will reflect on this next time")
        return False
    except ReflectionError as error:
        ui.notice(f"Haven couldn't finish reflecting: {error}. It will try again next time.")
        return False
    except OllamaError as error:
        _report(error, mind, ui)
        ui.notice("Haven will reflect on this next time")
        return False
    if outcome:
        ui.reflection(outcome)
    return True


def _report(error: OllamaError, mind: Mind, ui: Terminal) -> None:
    if isinstance(error, OllamaUnavailable):
        ui.error(f"Lost contact with Ollama at {mind.config.host}. Is it still running?")
        raise _Fatal from error
    if isinstance(error, ModelNotFound):
        ui.error(f"Ollama no longer has the model {mind.config.model}.")
        raise _Fatal from error
    ui.error(f"The model ran into a problem: {error.message}")


def _banner(mind: Mind, ui: Terminal) -> None:
    store = mind.store
    stats = store.stats()
    if stats["conversations"] == 0:
        ui.line("haven · awake for the first time", BOLD, CYAN)
    else:
        parts = ["haven · awake", f"conversation #{stats['conversations'] + 1}"]
        last = store.last_contact("chat")
        if last:
            parts.append(f"last talked {prompts.ago(last, store.clock())}")
        ui.line(" · ".join(parts), BOLD, CYAN)
    ui.line(f"thinking with {mind.config.model} · /help for commands · /bye or Ctrl-D to leave", DIM)
    ui.write("\n")


def _slash(text: str, convo: Conversation, ui: Terminal) -> str | None:
    command, _, rest = text.partition(" ")
    store = convo.store
    if command in ("/bye", "/quit", "/exit"):
        return "quit"
    if command == "/help":
        ui.line(CHAT_HELP, DIM)
    elif command == "/self":
        show_self(store, ui)
    elif command == "/beliefs":
        show_beliefs(store, ui)
    elif command == "/journal":
        show_journal(store, ui, 3)
    elif command == "/memories":
        show_memories(store, ui, rest.strip(), 10)
    elif command == "/questions":
        show_questions(store, ui)
    elif command == "/thoughts":
        ui.show_thoughts = not ui.show_thoughts
        ui.line(f"(Haven's thinking is now {'shown' if ui.show_thoughts else 'hidden'})", DIM)
    elif command == "/reflect":
        _reflect_mid_conversation(convo, ui)
    else:
        ui.line(f"(unknown command {command}; /help lists them)", DIM)
    return None


def _reflect_mid_conversation(convo: Conversation, ui: Terminal) -> None:
    if not convo.exchanges:
        ui.line("(nothing to reflect on yet)", DIM)
        return
    ui.reflecting()
    try:
        outcome = convo.reflect(ui, ui.dot)
    except KeyboardInterrupt:
        ui.notice("reflection skipped")
        return
    except ReflectionError as error:
        ui.notice(f"Haven couldn't finish reflecting: {error}")
        return
    except OllamaError as error:
        _report(error, convo.mind, ui)
        return
    if outcome:
        ui.reflection(outcome)
    else:
        ui.line("(nothing new to reflect on)", DIM)


# --- looking inside ------------------------------------------------------------


def cmd_self(mind: Mind, ui: Terminal, args: argparse.Namespace) -> int:
    store = mind.store
    if args.history:
        now = store.clock()
        for version in store.self_history():
            ui.line(f"v{version.number} · {prompts.ago(version.created_at, now)} · {version.reason}")
        return 0
    return show_self(store, ui, args.version)


def show_self(store: Store, ui: Terminal, number: int | None = None) -> int:
    version = store.self_version(number) if number else store.current_self()
    if version is None:
        ui.line(f"There's no version {number} of Haven's self-model.")
        return 1
    header = f"self-model v{version.number} · {prompts.ago(version.created_at, store.clock())} · {version.reason}"
    ui.line(header, DIM)
    ui.line(version.content)
    return 0


def cmd_beliefs(mind: Mind, ui: Terminal, args: argparse.Namespace) -> int:
    store = mind.store
    if args.history is None:
        return show_beliefs(store, ui, args.all)
    belief = store.get_belief(args.history)
    if belief is None:
        ui.line(f"There's no belief #{args.history}.")
        return 1
    now = store.clock()
    for event in store.belief_events(belief.id):
        ui.line(f"{prompts.ago(event.created_at, now)} · {event.kind} ({event.confidence:.2f})", DIM)
        ui.line(f"  {event.statement}")
        ui.line(f"  why: {event.reason}", DIM)
    return 0


def show_beliefs(store: Store, ui: Terminal, include_abandoned: bool = False) -> int:
    beliefs = store.beliefs(include_abandoned)
    if not beliefs:
        ui.line("Haven holds no beliefs right now.")
    for belief in beliefs:
        mark = "" if belief.status == "held" else "  (let go)"
        ui.line(f"#{belief.id:<4} {belief.confidence:.2f}  {belief.statement}{mark}")
    return 0


def cmd_journal(mind: Mind, ui: Terminal, args: argparse.Namespace) -> int:
    return show_journal(mind.store, ui, args.n)


def show_journal(store: Store, ui: Terminal, limit: int) -> int:
    entries = store.journal(limit)
    if not entries:
        ui.line("Haven hasn't written in its journal yet.")
    now = store.clock()
    for entry in entries:
        ui.line(f"{prompts.local_time(entry.created_at)} ({prompts.ago(entry.created_at, now)})", MAGENTA)
        ui.line(entry.entry)
        ui.write("\n")
    return 0


def cmd_memories(mind: Mind, ui: Terminal, args: argparse.Namespace) -> int:
    return show_memories(mind.store, ui, " ".join(args.query), args.n)


def show_memories(store: Store, ui: Terminal, query: str, limit: int) -> int:
    memories = store.search_memories(query, limit) if query else store.recent_memories(limit)
    if not memories:
        ui.line("Nothing surfaced." if query else "Haven has no memories yet.")
    now = store.clock()
    for memory in memories:
        ui.line(prompts.format_memory(memory, now))
    return 0


def cmd_questions(mind: Mind, ui: Terminal, args: argparse.Namespace) -> int:
    return show_questions(mind.store, ui)


def show_questions(store: Store, ui: Terminal) -> int:
    questions = store.open_curiosities(limit=50)
    if not questions:
        ui.line("Haven isn't carrying any open questions right now.")
    for question in questions:
        ui.line(prompts.format_question(question))
    return 0


def cmd_status(mind: Mind, ui: Terminal, args: argparse.Namespace) -> int:
    store, config = mind.store, mind.config
    now = store.clock()
    stats = store.stats()
    current = store.current_self()
    born = store.born_at or now
    ui.line("Haven", BOLD, CYAN)
    ui.line(f"  came into being  {prompts.local_time(born)} ({prompts.ago(born, now)})")
    ui.line(f"  conversations    {stats['conversations']}  ·  explorations {stats['explorations']}")
    ui.line(f"  memories         {stats['memories']}")
    ui.line(f"  beliefs          {stats['beliefs_held']} held, {stats['beliefs_abandoned']} let go")
    ui.line(f"  questions        {stats['open_questions']} open, {stats['explored_questions']} explored")
    ui.line(f"  journal entries  {stats['journal_entries']}")
    ui.line(f"  self-model       v{current.number}, last revised {prompts.ago(current.created_at, now)}")
    ui.line(f"  state of mind    {store.inner_state}")
    ui.line(f"  model            {config.model}  ({_model_state(mind)})")
    ui.line(f"  mind stored at   {config.db_path}", DIM)
    return 0


def _model_state(mind: Mind) -> str:
    try:
        version = mind.client.version()
        info = mind.model
    except OllamaUnavailable:
        return f"Ollama isn't answering at {mind.config.host}"
    except ModelNotFound:
        return "not forged yet: run `haven forge`" if mind.config.model == DEFAULT_MODEL else "not downloaded"
    except OllamaError as error:
        return f"Ollama error: {error.message}"
    abilities = [name for name, has in (("tools", info.tools), ("thinking", info.thinking)) if has]
    return f"ready in Ollama {version}; " + (", ".join(abilities) if abilities else "Haven handles tool calls itself")


COMMANDS: dict[str, Callable[[Mind, Terminal, argparse.Namespace], int]] = {
    "chat": cmd_chat,
    "wander": cmd_wander,
    "reflect": cmd_reflect,
    "forge": cmd_forge,
    "self": cmd_self,
    "beliefs": cmd_beliefs,
    "journal": cmd_journal,
    "memories": cmd_memories,
    "questions": cmd_questions,
    "status": cmd_status,
}
