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
    chat.add_argument("--voice", action="store_true", help="it says its replies aloud, in the computer's voice")
    chat.add_argument("--no-reading", action="store_true", help="don't let it read the encyclopedia by itself")

    feed = commands.add_parser(
        "feed",
        help="give Haven something to read: an encyclopedia (simple, english), files, or web pages",
        description="Haven keeps what it reads on its shelf, on this computer, and answers from it. "
        "`haven feed simple` reads the Simple English Wikipedia (231,282 articles, 284 MB to download); "
        "`haven feed english` reads the whole English Wikipedia (6.7 million articles, 21 GB to download, "
        "about 10 GB on disk, hours). Files and web pages are read as they are.",
    )
    feed.add_argument("what", nargs="*", help="simple, english, files to read, or web addresses")
    feed.add_argument("--text", help="something to read, given here")
    feed.add_argument("--title", help="what to call it")
    feed.add_argument("--forget", metavar="TITLE", help="take something off its shelf (or a whole encyclopedia)")

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
    if command == "feed":
        return feed(args, store, term)
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
        term.say(f"{mind.me.name} is born, in a nest in the corner of its valley.")
        store.save(mind.to_state())
    web = None if args.no_web else Web()
    thinker, message = make_thinker("own", store.root, web=web)
    if thinker is None:
        term.say(message)
        return 1
    term.dim(message)
    from ..feeding import FeedError, Feeding
    from ..voice import Voice

    feeding = Feeding(store.root, web, log=term.dim)
    if feeding.shelf is not None:
        thinker.shelf = feeding.shelf  # (what it has read, to answer from)
    voice = Voice() if args.voice else None
    if voice is not None and not voice.available:
        term.dim("This computer has no speech voice Haven can use (on Linux: install espeak-ng), so it types.")
        voice = None
    name = mind.me.name
    life.wake_brain()  # (its brain of spiking neurons joins in when it's ready)
    life.start()  # its life goes on while you talk
    if not args.no_reading and feeding.start():
        term.dim("(It's reading the Simple English Wikipedia in the background, the first time. Ask it anything.)")
    term.say(
        f"You're talking with {name}. Type to talk, or ask it anything. "
        "/read FILE or ADDRESS to give it something to read, /library, /voice, /status, /touch, /feed, /quit."
    )
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
            if text == "/library":
                shelf_status(feeding, term)
                continue
            if text == "/voice":
                voice = None if voice else Voice()
                term.dim(
                    "  (it speaks its replies aloud now)" if voice and voice.available else "  (it only types now)"
                )
                voice = voice if voice and voice.available else None
                continue
            if text.startswith("/read "):
                try:
                    title, kept = give(feeding, text[6:].strip())
                    term.dim(f"  ({name} read “{title}”: {kept:,} sentences. Ask it about it.)")
                except (FeedError, OSError) as error:
                    term.dim(f"  ({name} couldn't read that: {error})")
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
            if voice is not None:
                voice.say(answer)
            thinker.remember({"you": text, "haven": answer, "confidence": confidence, "time": time.time()})
    except KeyboardInterrupt:
        pass
    finally:
        feeding.stop(pause=False)
        if voice is not None:
            voice.close()
        life.stop()
        feeding.wait(10)
        term.say(f"{name} is resting until you come back.")
    return 0


def give(feeding, what: str, title: str | None = None) -> tuple[str, int]:
    """Give Haven a file or a web page to read. Returns (its title, how many sentences it kept)."""
    from pathlib import Path

    path = Path(what).expanduser()
    if path.is_file():
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix.lower() in (".html", ".htm"):
            from .shelf import page_text

            found, text = page_text(text, path.name)
            title = title or found
        return feeding.give(text, title or path.stem)
    return feeding.page(what)


def shelf_status(feeding, term) -> None:
    status = feeding.status()
    term.say(f"On its shelf: {status['articles']:,} articles, {status['sentences']:,} sentences.")
    for e in status["encyclopedias"]:
        state = (
            "read" if e["finished"] else f"{e['read']:,} of about {e['articles']:,} articles read" if e["read"] else ""
        )
        term.say(f"  {e['title'][0].upper() + e['title'][1:]}: {state or 'not read yet'} (haven feed {e['name']})")
    if status["reading"]:
        term.say(f"  It's reading {status['reading']['title']} now.")
    if status["given"]:
        term.say("  What you gave it: " + "; ".join(status["given"]))


def feed(args: argparse.Namespace, store, term) -> int:
    """Give Haven something to read, onto its shelf: an encyclopedia, files, web pages."""
    from ..feeding import ENCYCLOPEDIAS, FeedError, Feeding

    web = Web()
    feeding = Feeding(store.root, web, log=term.dim)
    if args.forget:
        if args.forget in ENCYCLOPEDIAS:  # a whole encyclopedia
            term.say(f"Taking {ENCYCLOPEDIAS[args.forget].title} off its shelf (a big one takes a while)…")
            n = feeding.shelf.forget_all(args.forget) if feeding.shelf is not None else 0
            term.say(f"Took {n:,} articles off its shelf. (`haven feed {args.forget}` reads it again.)")
            return 0
        n = feeding.forget(args.forget)
        term.say(f"Took {n} thing{'s' if n != 1 else ''} called “{args.forget}” off its shelf.")
        return 0
    if args.text:
        title, kept = feeding.give(args.text, args.title)
        term.say(f"Haven read “{title}” ({kept:,} sentences).")
    for what in args.what:
        if what in ENCYCLOPEDIAS:
            which = ENCYCLOPEDIAS[what]
            term.say(
                f"Haven is reading {which.title}: {which.articles:,} {which.unit}, {which.size / 1e6:,.0f} MB to download."
            )
            term.dim("(Ctrl+C stops it; run this again to carry on where it was.)")
            feeding.read(what)
            try:
                shown = -1
                while feeding.reading is not None:
                    read = feeding.status()["by source"].get(what, 0)
                    if read != shown:
                        term.dim(f"  {read:,} {which.unit} read…")
                        shown = read
                    time.sleep(10)
            except KeyboardInterrupt:
                feeding.stop(pause=False)
                feeding.wait(30)
                term.say("Stopped for now; what it read stays read.")
                return 0
            feeding.wait()
            last = feeding.last or {}
            if not last.get("finished"):
                term.say(f"It couldn't finish: {last.get('error')}. Run this again to carry on where it was.")
            continue
        try:
            title, kept = give(feeding, what, args.title)
            term.say(f"Haven read “{title}” ({kept:,} sentences).")
        except (FeedError, OSError) as error:
            term.say(f"Haven couldn't read {what}: {error}")
    if not args.what and not args.text:
        shelf_status(feeding, term)
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
        f.write(json.dumps({"title": title, "text": text[:20000], "time": time.time(), "why": "asked"}) + "\n")
    term.say(f"Haven read “{title}” ({len(text.split()):,} words). It keeps what it read: ask it about {title}.")
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
