"""Commands for Haven's language cortex: learn, read, ask."""

from __future__ import annotations

import argparse
import json
import time

from ..web import Web, WebError


def add_commands(commands) -> None:
    learn = commands.add_parser("learn", help="train Haven's language cortex through the reading curriculum")
    learn.add_argument(
        "--size",
        default="auto",
        choices=("auto", "tiny", "small", "medium", "large"),
        help="size of a new cortex: tiny suits a CPU, small an Apple GPU, medium or large an NVIDIA GPU "
        "(default: auto, by the hardware; ignored once one exists)",
    )
    learn.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    learn.add_argument("--through", type=int, help="stop after this level")
    learn.add_argument("--minutes", type=float, help="study for this long, then stop (it picks up where it left off)")
    learn.add_argument("--steps", type=int, help="study for this many steps, then stop")
    learn.add_argument("--report", action="store_true", help="just show the report card")

    read = commands.add_parser("read", help="have Haven read an encyclopedia article about something")
    read.add_argument("topic", nargs="+")
    read.add_argument("--full", action="store_true", help="read English Wikipedia instead of Simple English")

    ask = commands.add_parser("ask", help="ask Haven something, and see its thoughts")
    ask.add_argument("question", nargs="+")
    ask.add_argument("--cortex", default="own", help='"own" or "ollama:MODEL"')
    ask.add_argument("--no-web", action="store_true", help="don't let it look things up")


def run(command: str, args: argparse.Namespace, store, term) -> int:
    if command == "learn":
        return learn(args, store, term)
    if command == "read":
        return read(args, store, term)
    if command == "ask":
        return ask(args, store, term)
    return 2


def _torch_or_explain(term) -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        term.say(
            "The language cortex needs PyTorch. Install it with: pip install 'haven[cortex]'  (or pip install torch)"
        )
        return False
    return True


def learn(args: argparse.Namespace, store, term) -> int:
    if not _torch_or_explain(term):
        return 1
    from .train import Trainer

    trainer = Trainer(store.root, size=args.size, device=args.device, log=term.say)
    if args.report:
        for line in trainer.report():
            term.say(line)
        return 0
    term.say(f"Studying on {trainer.device} (Ctrl+C stops safely; run it again to carry on).")
    started = time.monotonic()
    why = trainer.run(through=args.through, minutes=args.minutes, steps=args.steps)
    term.say(
        {
            "done": "Finished the curriculum" if trainer.progress["level"] > 8 else "Reached the level you asked for",
            "time": "Time's up for now",
            "steps": "Did the steps you asked for",
            "interrupted": "Stopped",
        }.get(why, why)
        + f" after {(time.monotonic() - started) / 60:.1f} minutes. Progress is saved."
    )
    for line in trainer.report():
        term.say(line)
    return 0


def read(args: argparse.Namespace, store, term) -> int:
    from . import sources

    topic = " ".join(args.topic)
    api = sources.URLS["wikipedia" if args.full else "simplewiki"]
    try:
        found = sources.article(Web(), api, topic)
    except WebError as error:
        term.say(f"Couldn't read about that: {error}")
        return 1
    if found is None:
        term.say(f"The encyclopedia has nothing on {topic!r}.")
        return 1
    title, text = found
    path = store.root / "cortex" / "readings.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"title": title, "text": text[:20000], "time": time.time()}) + "\n")
    term.say(f"Haven read “{title}” ({len(text.split()):,} words). It will learn from it the next times it sleeps.")
    term.dim(" ".join(text.split())[:500] + "…")
    if store.exists():
        from ..life import open_mind
        from ..memory import Episode

        mind = open_mind(store)
        mind.memory.store(
            Episode(
                tick=mind.tick,
                source="thought",
                label=f"reading about {title}",
                kind=-1,
                place=(mind.world.x, mind.world.y),
                valence=0.0,
                arousal=0.3,
                event="read",
            )
        )
        mind.me.milestone("first reading", mind.tick, f"read its first encyclopedia article: {title}")
        store.save(mind.to_state())
    return 0


def ask(args: argparse.Namespace, store, term) -> int:
    from ..life import Life, open_mind

    if not store.exists():
        term.say("No Haven lives here yet. Start one with: haven live")
        return 1
    if args.cortex == "own" and not _torch_or_explain(term):
        return 1
    from .think import make_thinker

    mind = open_mind(store)
    life = Life(mind, store)
    thinker, message = make_thinker(args.cortex, store.root, web=None if args.no_web else Web())
    if thinker is None:
        term.say(message)
        return 1
    life.listeners.append(lambda kind, text: term.dim(f"  · {text}"))
    question = " ".join(args.question)
    life.conversation.append({"tick": mind.tick, "who": "you", "text": question})
    thoughts = []
    original = mind.think

    def noting(text, meaning, confidence):
        thoughts.append((text, confidence))
        term.dim(f"  thinking: {text!r} (confidence {confidence:.0%})")
        original(text, meaning, confidence)

    mind.think = noting
    answer, confidence = thinker.deliberate(life, question)
    term.haven(mind.me.name, f"{answer}   [{thinker.name}, confidence {confidence:.0%}]")
    mind.think = original
    mind.step()  # let the thought come to mind, and be remembered
    store.save(mind.to_state())
    return 0
