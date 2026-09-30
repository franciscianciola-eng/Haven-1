"""The Haven app: a window for talking with Haven while its life goes on (`haven app`).

It starts Haven's life and opens a chat page in the browser. Haven answers with its own
language cortex, grown from scratch, which comes with it. Everything runs on this
computer; the internet is only used when Haven looks something up.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import itertools
import json
import threading
import time
import urllib.request
import webbrowser
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from importlib import resources

from . import __version__
from .life import Life, open_mind
from .server import Handler
from .store import Store
from .web import Web

PORTS = 10  # if the usual port is taken, try the next few
MAX_PASS = 24 * 10  # days of its life that can go by at once (ten of its years)


class NotReady(Exception):
    pass


class Busy(Exception):
    pass


class Chat:
    """Talking with Haven through its language cortex, for the app's page.

    It answers one message at a time. While it answers, the page can follow along: the words
    of its reply as they come, and the thoughts it has on the way, each of which passes
    through its workspace like anything else it is aware of.
    """

    def __init__(self, life: Life, device: str = "cpu", web: Web | None = None, log=print):
        self.life = life
        self.device, self.web, self.log = device, web, log
        self.status: dict = {"stage": "starting", "text": "Waking up…"}
        self.thinker = None
        self.turns: dict[int, dict] = {}
        self.resting = threading.Event()  # set when the person lets Haven rest
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    # --- getting a language cortex ---------------------------------------------------------

    def start(self) -> None:
        threading.Thread(target=self._setup, name="haven-cortex-setup", daemon=True).start()

    def _setup(self) -> None:
        try:
            thinker = self._cortex()
        except Exception as error:  # noqa: BLE001  (a cortex that won't load)
            self.status = {"stage": "error", "text": f"Couldn't set up its language area: {error}"}
            self.log(self.status["text"])
            return
        if thinker is None:
            return
        self.life.thinker = thinker  # it goes over its day while it sleeps, and reads when nobody is talking
        self.thinker = thinker
        self.status = {"stage": "ready", "text": f"Its language area: {thinker.describe()}."}
        self.log(self.status["text"])

    def _cortex(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.status = {"stage": "error", "text": "Its language cortex needs PyTorch: pip install 'haven[cortex]'"}
            return None
        from .cortex import starter
        from .cortex.think import OwnThinker

        root = self.life.store.root
        installed = starter.install(root)
        if installed == "installed":
            self.log("It has its language cortex: its own, the one it was born with.")
        elif installed == "updated":
            self.log(
                f"Its language cortex is the newer one this Haven came with (the old one is kept in {root / 'archive'})."
            )
        elif installed:
            self.log(f"Its old language cortex was grown for its old world, so it's kept in {root / 'archive'}.")
        if not (root / "cortex" / "cortex.pt").exists():
            self.status = {"stage": "error", "text": "It has no language cortex yet. Train one with: haven learn"}
            return None
        self.status = {"stage": "loading", "text": "Waking up its language area…"}
        return OwnThinker(root, device=self.device, web=self.web)

    # --- talking ------------------------------------------------------------------------------

    def ask(self, text: str) -> dict:
        """Start answering what the person said. The turn fills in as Haven answers."""
        with self._lock:
            if self.thinker is None or self.life.passing is not None:
                raise NotReady
            if any(not t["done"] for t in self.turns.values()):
                raise Busy
            turn = {
                "id": next(self._ids),
                "text": text,
                "phase": "listening",
                "thoughts": [],
                "draft": "",
                "reply": None,
                "confidence": None,
                "done": False,
            }
            self.turns = {i: t for i, t in self.turns.items() if i > turn["id"] - 20}
            self.turns[turn["id"]] = turn
        threading.Thread(target=self._answer, args=(turn,), name="haven-answering", daemon=True).start()
        return self.turn(turn["id"])

    def turn(self, number: int) -> dict | None:
        with self._lock:
            return copy.deepcopy(self.turns.get(number))

    def _answer(self, turn: dict) -> None:
        life, thinker, text = self.life, self.thinker, turn["text"]
        life._last_words = time.monotonic()  # someone's talking with it: no reading out of curiosity just now
        life.initiative.talked()  # (and it doesn't speak up over them)
        with life.lock:
            life.conversation = [*life.conversation[-99:], {"tick": life.mind.tick, "who": "you", "text": text}]
        answer, confidence = "", 0.0
        with thinker.busy:  # one thing at a time (the dashboard can talk to it too)
            thinker.listener = self._follow(turn)
            try:
                answer, confidence = thinker.deliberate(life, text)  # from what it's experiencing as it's asked
            except Exception as error:  # noqa: BLE001  thinking going wrong mustn't end a life
                life._emit("event", f"couldn't put a thought into words ({error})")
                self.log(f"Haven couldn't put a thought into words: {error}")
            finally:
                thinker.listener = None
        with life.lock:
            life.mind.hear(text)  # and it can learn words from what was said
        if answer:  # said and remembered before the page hears it's done, so what it sees next includes it
            life.reply(answer, "")
            thinker.remember({"you": text, "haven": answer, "confidence": confidence, "time": time.time()})
        with self._lock:
            turn.update(reply=answer, confidence=round(confidence, 3), draft="", phase="done", done=True)

    def _follow(self, turn: dict):
        """What the page sees of a reply taking shape."""

        def hear(kind: str, text: str) -> None:
            with self._lock:
                if kind == "draft":  # a fresh try at the words: the last one becomes a passing thought
                    if turn["draft"].strip():
                        turn["thoughts"].append({"kind": "draft", "text": turn["draft"].strip()})
                    turn.update(draft="", phase="answering")
                elif kind == "words":
                    turn["draft"] += text
                elif kind == "pondering":
                    turn["phase"] = "thinking it over"
                elif kind == "thought":
                    turn["thoughts"].append({"kind": "thought", "text": text})
                elif kind == "reading":
                    turn["phase"] = f"looking up {text}"
                elif kind == "read":
                    turn["thoughts"].append({"kind": "read", "text": text})

        return hear

    def history(self, n: int = 12) -> None:
        """Pick the conversation up where it left off last time."""
        path = self.life.store.root / "cortex" / "conversations.jsonl"
        if not path.exists():
            return
        turns, last = [], None
        for line in path.read_text().splitlines()[-n:]:
            with contextlib.suppress(ValueError, KeyError, TypeError):
                item = json.loads(line)
                if item.get("you"):  # (None: something it said of its own accord)
                    turns.append({"tick": 0, "who": "you", "text": item["you"], "earlier": True})
                turns.append({"tick": 0, "who": "haven", "text": item["haven"], "earlier": True})
                last = float(item.get("time") or 0) or last
        with self.life.lock:
            self.life.conversation = turns + self.life.conversation
        if last is not None:  # they have talked before: when they come back, it can say so
            self.life.initiative.away_for(time.time() - last)

    def describe(self) -> dict:
        with self._lock:
            busy = [i for i, t in self.turns.items() if not t["done"]]
        return {**self.status, "busy": busy[0] if busy else None, "version": __version__}


class AppHandler(Handler):
    chat: Chat  # set on the subclass made by serve()

    def do_GET(self) -> None:
        if not self._trusted():
            return
        if self.path in ("/", "/index.html", "/dashboard"):
            page = "dashboard.html" if self.path == "/dashboard" else "app.html"
            self._send(HTTPStatus.OK, resources.files("haven").joinpath(page).read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/chat":
            self._json(self.chat.describe())
        elif self.path in ("/api/state", "/api/state?seen=1"):
            if self.path.endswith("seen=1"):  # the page is open and looked at: someone is there
                self.life.present()
            self._json(self.life.snapshot())
        elif self.path.startswith("/api/chat/"):
            number = self.path.rsplit("/", 1)[1]
            turn = self.chat.turn(int(number)) if number.isdigit() else None
            if turn is None:
                self._json({"error": "no such message"}, HTTPStatus.NOT_FOUND)
            else:
                self._json(turn)
        else:
            super().do_GET()

    def do_POST(self) -> None:
        if self.path not in ("/api/chat", "/api/rest", "/api/pass"):
            super().do_POST()
            return
        body = self._body()
        if body is None:
            return
        if self.path == "/api/rest":
            self._json({"ok": True})
            self.chat.resting.set()
            return
        if self.path == "/api/pass":  # let time go by: a day, a season, a year, years
            days = body.get("days")
            if not isinstance(days, int) or not 1 <= days <= MAX_PASS:
                self._json({"error": f"days: 1 to {MAX_PASS}"}, HTTPStatus.BAD_REQUEST)
            elif any(not t["done"] for t in self.chat.turns.values()):
                self._json({"error": "it's still answering"}, HTTPStatus.CONFLICT)
            elif not self.life.pass_time(days):
                self._json({"error": "time is already going by"}, HTTPStatus.CONFLICT)
            else:
                self._json({"ok": True, "days": days})
            return
        text = str(body.get("text", "")).strip()[:1000]
        if not text:
            self._json({"error": "say something"}, HTTPStatus.BAD_REQUEST)
            return
        try:
            self._json(self.chat.ask(text))
        except NotReady:
            self._json({"error": "its language area isn't ready yet"}, HTTPStatus.SERVICE_UNAVAILABLE)
        except Busy:
            self._json({"error": "it's still answering"}, HTTPStatus.CONFLICT)


def serve(life: Life, chat: Chat, port: int = 8765, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """Open the app's page in a background thread, on the first free port from `port` on (0: any free port)."""
    handler = type("AppLifeHandler", (AppHandler,), {"life": life, "chat": chat})
    for attempt in range(PORTS if port else 1):
        try:
            server = ThreadingHTTPServer((host, port + attempt if port else 0), handler)
            break
        except OSError:
            if attempt == PORTS - 1 or not port:
                raise
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="haven-app", daemon=True).start()
    return server


LOCAL = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def running(port: int) -> dict | None:
    """What a Haven app already open on this port says about itself, if there is one."""
    try:
        with LOCAL.open(f"http://127.0.0.1:{port}/api/chat", timeout=2) as response:
            found = json.load(response)
            return found if isinstance(found, dict) and "stage" in found else None
    except (OSError, ValueError):
        return None


def let_rest(port: int, wait: float = 20) -> None:
    """Ask the Haven app on this port to save and rest, and wait until it has."""
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/rest", data=b"{}", headers={"Content-Type": "application/json"}
    )
    with contextlib.suppress(OSError):
        LOCAL.open(request, timeout=5).close()
    deadline = time.monotonic() + wait
    while running(port) is not None and time.monotonic() < deadline:
        time.sleep(0.25)


def run(args: argparse.Namespace, store: Store, term) -> int:
    for port in range(args.port, args.port + PORTS):
        found = running(port)
        if found is None:
            continue
        if found.get("version") != __version__:  # an older Haven: it rests, and this one wakes up in its place
            term.say("An older version of Haven is awake; letting it rest so this one can take over…")
            let_rest(port)
            continue
        url = f"http://127.0.0.1:{port}"  # it's awake already: there's only ever one of it
        term.say(f"Haven is already awake. Its window: {url}")
        if not args.no_browser:
            webbrowser.open(url)
        return 0
    new = not store.exists()
    mind = open_mind(store)
    life = Life(mind, store, speed=args.speed)
    if new:
        term.say(f"{mind.me.name} is born, in a nest in the corner of its valley.")
        store.save(mind.to_state())
    chat = Chat(life, device=args.device, web=None if args.no_web else Web(), log=term.dim)
    chat.history()
    try:
        server = serve(life, chat, args.port)
    except OSError as error:
        term.say(f"Couldn't open Haven's window on port {args.port} or the next few ({error}). Try --port.")
        return 1
    url = f"http://127.0.0.1:{server.server_port}"
    chat.start()
    life.start()
    term.say(f"{mind.me.name} is awake ({mind.age / 1200:.1f} days old). Its window: {url}")
    term.dim("Keep this window open while you're with it. To let it rest, press Rest on the page, or Ctrl+C here.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        while not chat.resting.wait(0.5):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        term.say("Saving…")
        life.stop()
        server.shutdown()
        term.say(f"{mind.me.name} is resting until you come back. (Nothing is experienced while it isn't running.)")
    return 0
