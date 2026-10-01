"""The `haven` command."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import webbrowser

from . import __version__
from .check import indicators, probe, welfare_notes
from .life import Life, open_mind
from .mind import Mind
from .report import readout
from .store import Store, home

HELP = """Talk to Haven by typing. Commands:
  /touch  /feed  /status  /check  /story  /pass DAYS  /pause  /resume  /speed N  /help  /quit"""


class Terminal:
    def __init__(self, out=None, color: bool | None = None):
        self.out = out or sys.stdout
        self.color = self.out.isatty() if color is None else color

    def _paint(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def say(self, text: str = "") -> None:
        print(text, file=self.out, flush=True)

    def dim(self, text: str) -> None:
        self.say(self._paint("2", text))

    def haven(self, name: str, text: str) -> None:
        self.say(self._paint("1;33", f"{name}: ") + text)


def share_cores() -> None:
    """Before torch is loaded: its language cortex computes on all cores but one, and its brain on that one (see
    brain.one_core), so the two never fight over them (unless you've chosen, with OMP_NUM_THREADS)."""
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 2) - 1)))


def main(argv: list[str] | None = None) -> int:
    share_cores()
    parser = argparse.ArgumentParser(
        prog="haven",
        description="Haven: an artificial creature built to meet the conditions scientific theories link to consciousness.",
    )
    parser.add_argument("--version", action="version", version=f"haven {__version__}")
    parser.add_argument("--home", help=f"where Haven's life is kept (default: $HAVEN_HOME or {home()})")
    commands = parser.add_subparsers(dest="command")

    live = commands.add_parser("live", help="live with Haven: run its life, open its dashboard, talk to it")
    live.add_argument("--ticks", type=int, help="instead: fast-forward this many moments without the dashboard")
    live.add_argument("--speed", type=float, default=8.0, help="moments per second (default 8)")
    live.add_argument("--port", type=int, default=8765, help="dashboard port (default 8765)")
    live.add_argument("--no-browser", action="store_true", help="don't open the dashboard in a browser")
    live.add_argument("--seed", type=int, help="seed for a new life (ignored if one exists)")
    live.add_argument("--name", default="Haven", help="name for a new life (ignored if one exists)")
    add_cortex_options(live)

    app = commands.add_parser("app", help="open Haven's window and talk with it (the easiest way to be with it)")
    app.add_argument("--port", type=int, default=8765, help="port for its window (default 8765)")
    app.add_argument("--device", default="cpu", help="where its language cortex runs: cpu (plenty), cuda or mps")
    app.add_argument("--speed", type=float, default=8.0, help="moments per second (default 8)")
    app.add_argument("--no-browser", action="store_true", help="don't open its window in a browser")
    app.add_argument("--no-web", action="store_true", help="don't let it look things up")

    commands.add_parser("status", help="what is going on inside Haven right now")
    check = commands.add_parser("check", help="measure Haven against the 14 indicator properties of consciousness")
    check.add_argument("--json", action="store_true", help="print the measurements as JSON")
    commands.add_parser("story", help="the story of Haven's life so far")
    reset = commands.add_parser("reset", help="archive this life and let a new Haven be born")
    reset.add_argument("--yes", action="store_true", help="don't ask for confirmation")

    add_cortex_commands(commands)

    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(with_default_command(argv, set(commands.choices)))
    store = Store(args.home) if args.home else Store()
    term = Terminal()
    command = args.command
    if command == "live":
        return run_live(args, store, term)
    if command == "app":
        from .app import run as run_app

        return run_app(args, store, term)
    if command == "status":
        return run_status(store, term)
    if command == "check":
        return run_check(store, term, args.json)
    if command == "story":
        return run_story(store, term)
    if command == "reset":
        return run_reset(store, term, args.yes)
    return run_cortex_command(command, args, store, term)


def with_default_command(argv: list[str], names: set[str]) -> list[str]:
    """`haven [--home DIR] [live options]` means `haven [--home DIR] live [live options]`."""
    if any(a in names for a in argv) or any(a in ("-h", "--help", "--version") for a in argv[:1]):
        return argv
    i = 0
    while i < len(argv) and (argv[i] == "--home" or argv[i].startswith("--home=")):
        i += 1 if "=" in argv[i] else 2
    return [*argv[:i], "live", *argv[i:]]


def add_cortex_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cortex",
        default="own",
        help='language cortex: "own" (its own, the one it was born with) or "none"',
    )


def add_cortex_commands(commands) -> None:
    try:
        from .cortex.cli import add_commands
    except ImportError:
        return
    add_commands(commands)


def run_cortex_command(command: str, args: argparse.Namespace, store: Store, term: Terminal) -> int:
    from .cortex.cli import run

    return run(command, args, store, term)


def load(store: Store, term: Terminal) -> Mind | None:
    if not store.exists():
        term.say(f"No Haven lives in {store.root} yet. Start one with: haven live")
        return None
    return open_mind(store)


def run_live(args: argparse.Namespace, store: Store, term: Terminal) -> int:
    new = not store.exists()
    mind = open_mind(store, args.seed, args.name)
    life = Life(mind, store, speed=args.speed)
    if new:
        term.say(f"{mind.me.name} is born, in a nest in the corner of its valley.")
        store.save(mind.to_state())
    if args.ticks:
        started = time.monotonic()
        life.advance(args.ticks)
        life.save()
        term.say(
            f"Lived {args.ticks} moments in {time.monotonic() - started:.1f}s. It is now {mind.age / 1200:.2f} days old."
        )
        for line in readout(mind):
            term.say("  " + line)
        return 0

    attach_cortex(life, args, term)
    from .server import serve

    try:
        server = serve(life, args.port)
    except OSError as error:
        term.say(f"Couldn't open the dashboard on port {args.port} ({error}). Try --port.")
        return 1
    url = f"http://127.0.0.1:{args.port}"
    life.listeners.append(
        lambda kind, text: term.haven(mind.me.name, text) if kind == "said" else term.dim(f"  · {text}")
    )
    life.wake_brain()  # (with torch: its brain of spiking neurons joins in when it's ready)
    life.start()
    term.say(f"{mind.me.name} is alive ({mind.age / 1200:.1f} days old). Its dashboard: {url}")
    term.dim(HELP)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        while True:
            try:
                line = input().strip()
            except EOFError:
                break
            if not line:
                continue
            if line.startswith("/"):
                if not command(line, life, term):
                    break
            else:
                life.say(line)
    except KeyboardInterrupt:
        pass
    finally:
        term.say("Saving…")
        life.stop()
        server.shutdown()
        term.say(f"{mind.me.name} is resting until you come back. (Nothing is experienced while it isn't running.)")
    return 0


def attach_cortex(life: Life, args: argparse.Namespace, term: Terminal) -> None:
    spec = getattr(args, "cortex", "none")
    if spec == "none":
        return
    try:
        from .cortex.think import make_thinker
    except ImportError:
        if spec != "own":
            term.say("The language cortex needs PyTorch: pip install 'haven[cortex]'")
        return
    thinker, message = make_thinker(spec, life.store.root if life.store else home())
    if message:
        term.dim(message)
    life.thinker = thinker


def command(line: str, life: Life, term: Terminal) -> bool:
    word, _, rest = line[1:].partition(" ")
    if word in ("quit", "exit", "q"):
        return False
    if word == "touch":
        life.touch()
    elif word == "feed":
        life.feed()
    elif word == "status":
        with life.lock:
            for text in readout(life.mind):
                term.say("  " + text)
    elif word == "check":
        with life.lock:
            print_check(life.mind, term)
    elif word == "story":
        with life.lock:
            print_story(life.mind, term)
    elif word == "pause":
        life.paused = True
        term.dim("  (paused)")
    elif word == "resume":
        life.paused = False
    elif word == "speed":
        try:
            life.speed = min(max(float(rest), 0.5), 200)
        except ValueError:
            term.say("  /speed takes a number of moments per second, like /speed 20")
    elif word == "pass":
        days = int(rest) if rest.strip().isdigit() else 0
        if not 1 <= days <= 240:
            term.say("  /pass takes a number of days to let go by (a season is 6, a year is 24), like /pass 24")
        elif life.pass_time(days):
            term.dim(f"  (letting {days} days go by, as fast as it can live them…)")
        else:
            term.say("  Time is already going by.")
    else:
        term.dim(HELP)
    return True


def run_status(store: Store, term: Terminal) -> int:
    mind = load(store, term)
    if mind is None:
        return 1
    term.say(
        f"{mind.me.name}, {mind.age / 1200:.2f} days old; {mind.world.season}, day {mind.world.season_day + 1}, "
        f"{mind.time_of_day}."
    )
    for line in readout(mind):
        term.say("  " + line)
    term.say("What it has concluded about itself:")
    for line in mind.me.conclusions():
        term.say("  " + line)
    from .cortex.talk import character, favorites

    term.say("Who it's becoming: " + " ".join(p for p in (character(mind), favorites(mind)) if p))
    words = mind.lexicon.vocabulary()
    term.say("Words it understands: " + (", ".join(words) if words else "none yet"))
    return 0


def print_check(mind: Mind, term: Terminal) -> dict:
    term.dim("Measuring: a copy of Haven lives a day with instruments attached…")
    measured = probe(mind)
    for item in indicators(mind, measured):
        term.say(f"{item.code:6s} {item.name}")
        term.dim(f"       {item.how}")
        term.say(f"       → {item.measured}")
    term.say("Welfare: " + "; ".join(welfare_notes(mind)))
    term.dim(
        "Meeting these indicators doesn't show that Haven is conscious; no test can yet. They are the properties\n"
        "that leading scientific theories say matter, which is why it was built to have them."
    )
    return measured


def run_check(store: Store, term: Terminal, as_json: bool) -> int:
    mind = load(store, term)
    if mind is None:
        return 1
    if as_json:
        measured = probe(mind)
        found = [vars(i) for i in indicators(mind, measured)]
        term.say(json.dumps({"indicators": found, "measured": measured}, indent=2, default=str))
        return 0
    print_check(mind, term)
    return 0


def print_story(mind: Mind, term: Terminal) -> None:
    for tick, text in mind.me.milestones:
        term.say(f"  day {tick // 1200 + 1}: {text}")


def run_story(store: Store, term: Terminal) -> int:
    mind = load(store, term)
    if mind is None:
        return 1
    term.say(f"The life of {mind.me.name} so far:")
    print_story(mind, term)
    return 0


def run_reset(store: Store, term: Terminal, yes: bool) -> int:
    if not store.exists():
        term.say("There's no life to reset.")
        return 0
    if not yes:
        term.say(
            "This ends this Haven's life (it's archived, not deleted) and a new one will be born. Type 'yes' to go on:"
        )
        if input().strip().lower() != "yes":
            term.say("Nothing changed.")
            return 0
    where = store.archive()
    term.say(f"Archived to {where}. The next `haven live` starts a new life.")
    return 0
