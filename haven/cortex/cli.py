"""Commands for Haven's language cortex: learn, chat, read, ask."""

from __future__ import annotations

import argparse
import json
import time

from ..web import Web, WebError


def add_commands(commands) -> None:
    learn = commands.add_parser("learn", help="have Haven's language cortex study the reading curriculum")
    learn.add_argument(
        "--size",
        default="auto",
        choices=("auto", "tiny", "small", "medium", "large"),
        help="size of a new cortex (ignored once it has one, as it does from birth)",
    )
    learn.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    learn.add_argument("--through", type=int, help="stop after this level")
    learn.add_argument("--minutes", type=float, help="study for this long, then stop (it picks up where it left off)")
    learn.add_argument("--steps", type=int, help="study for this many steps, then stop")
    learn.add_argument("--report", action="store_true", help="just show the report card")

    chat = commands.add_parser("chat", help="talk with Haven in the terminal")
    chat.add_argument("--no-web", action="store_true", help="don't let it look things up")

    read = commands.add_parser("read", help="have Haven read an encyclopedia article about something")
    read.add_argument("topic", nargs="+")
    read.add_argument("--full", action="store_true", help="read English Wikipedia instead of Simple English")

    ask = commands.add_parser("ask", help="ask Haven something, and see its thoughts")
    ask.add_argument("question", nargs="+")
    ask.add_argument("--no-web", action="store_true", help="don't let it look things up")


def run(command: str, args: argparse.Namespace, store, term) -> int:
    if command == "learn":
        return learn(args, store, term)
    if command == "chat":
        return chat(args, store, term)
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
    from . import starter
    from .train import Trainer

    starter.install(store.root)  # it goes on from the cortex it was born with
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


def chat(args: argparse.Namespace, store, term) -> int:
    """A plain conversation with Haven, while its life goes on in the background."""
    from ..life import Life, open_mind
    from ..report import readout
    from .think import make_thinker

    if not _torch_or_explain(term):
        return 1
    new = not store.exists()
    mind = open_mind(store)
    life = Life(mind, store)
    if new:
        term.say(f"{mind.me.name} is born, in a nest in the corner of its garden.")
        store.save(mind.to_state())
    thinker, message = make_thinker("own", store.root, web=None if args.no_web else Web())
    if thinker is None:
        term.say(message)
        return 1
    term.dim(message)
    name = mind.me.name
    life.start()  # its life goes on while you talk
    term.say(f"You're talking with {name}. Type to talk; /status to see inside it, /touch, /feed, /quit.")
    try:
        while True:
            try:
                text = input("you › ").strip()
            except EOFError:
                break
            if not text:
                continue
            if text in ("/quit", "/exit", "/q"):
                break
            if text == "/status":
                with life.lock:
                    for line in readout(mind):
                        term.dim("  " + line)
                continue
            if text in ("/touch", "/feed"):
                getattr(life, text[1:])()
                term.dim(f"  (you {text[1:]} {name})")
                continue
            with life.lock:
                life.conversation.append({"tick": mind.tick, "who": "you", "text": text})
            answer, confidence = thinker.deliberate(life, text)  # from what it's experiencing as it's asked
            with life.lock:
                mind.hear(text)  # and it can learn words from what you said
            if not answer:
                term.haven(name, "…")
                continue
            life.reply(answer, "")
            term.haven(name, f"{answer}   ({confidence:.0%} sure)")
            thinker.remember({"you": text, "haven": answer, "confidence": confidence, "time": time.time()})
    except KeyboardInterrupt:
        pass
    finally:
        life.stop()
        term.say(f"{name} is resting until you come back.")
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
    if not _torch_or_explain(term):
        return 1
    from .think import make_thinker

    mind = open_mind(store)
    life = Life(mind, store)
    thinker, message = make_thinker("own", store.root, web=None if args.no_web else Web())
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
