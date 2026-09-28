"""The Haven app: a window for talking with Haven while its life goes on (`haven app`).

It starts Haven's life, opens a chat page in the browser, and the first time, gets Haven a
language cortex: an open model, downloaded once and wired into its mind. Everything runs
on this computer. The internet is only used to download that model, and for Haven to look
things up when it isn't sure.
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
from pathlib import Path

from .life import Life, open_mind
from .server import Handler
from .store import Store
from .web import Web

PORTS = 10  # if the usual port is taken, try the next few


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

    def __init__(self, life: Life, base: str | None = None, device: str = "auto", web: Web | None = None, log=print):
        self.life = life
        self.base, self.device, self.web, self.log = base, device, web, log
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
        except Exception as error:  # noqa: BLE001  (a download or a model that won't load)
            self.status = {"stage": "error", "text": f"Couldn't set up its language area: {error}"}
            self.log(self.status["text"])
            return
        if thinker is None:
            return
        self.life.thinker = thinker  # it goes over what it heard and read while it sleeps
        self.thinker = thinker
        self.status = {"stage": "ready", "text": f"Its language area: {thinker.describe()}."}
        self.log(self.status["text"])

    def _cortex(self):
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            self.status = {
                "stage": "error",
                "text": "Its language area needs PyTorch and transformers: pip install 'haven[cortex]'",
            }
            return None
        from .cortex import graft as grafting
        from .cortex.think import GraftThinker, OwnThinker
        from .cortex.train import pick_device

        root = self.life.store.root
        if grafting.exists(root):
            info = json.loads((root / "cortex" / "graft.json").read_text())
            self._download(info["base"])  # in case its files aren't on this computer any more
            self.status = {"stage": "loading", "text": "Waking up its language area…"}
            return GraftThinker(root, device=self.device, web=self.web)
        if (root / "cortex" / "cortex.pt").exists():  # a cortex it grew from scratch
            self.status = {"stage": "loading", "text": "Waking up its language area…"}
            return OwnThinker(root, device=self.device, web=self.web)
        name = self.base or grafting.pick_base(pick_device(self.device))
        self._download(name)
        self.status = {"stage": "loading", "text": "Wiring the model into Haven's mind…"}
        trainer = grafting.GraftTrainer(root, base=name, device=self.device, web=self.web, log=self.log)
        return GraftThinker(root, device=self.device, web=self.web, graft=trainer.graft, progress=trainer.progress)

    def _download(self, name: str) -> None:
        from .cortex.graft import BASES, download

        repo = BASES.get(name, name)
        if Path(repo).is_dir():  # a model already on this computer
            return
        text = f"Getting its language area: {repo}, an open model, downloaded once."

        def progress(done: int, total: int | None) -> None:
            self.status = {"stage": "download", "text": text, "done": done, "total": total}

        self.status = {"stage": "download", "text": text, "done": 0, "total": None}
        self.log(text)
        download(repo, progress)

    # --- talking ------------------------------------------------------------------------------

    def ask(self, text: str) -> dict:
        """Start answering what the person said. The turn fills in as Haven answers."""
        with self._lock:
            if self.thinker is None:
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
            turn = copy.deepcopy(self.turns.get(number))
        if turn and turn["draft"]:
            from .cortex.think import spoken

            turn["draft"] = spoken(turn["draft"])
        return turn

    def _answer(self, turn: dict) -> None:
        life, thinker, text = self.life, self.thinker, turn["text"]
        with life.lock:
            life.mind.hear(text)
            life.conversation = [*life.conversation[-99:], {"tick": life.mind.tick, "who": "you", "text": text}]
        answer, confidence = "", 0.0
        with thinker.busy:  # one thing at a time (the dashboard can talk to it too)
            thinker.listener = self._follow(turn)
            try:
                answer, confidence = thinker.deliberate(life, text)
            except Exception as error:  # noqa: BLE001  thinking going wrong mustn't end a life
                life._emit("event", f"couldn't put a thought into words ({error})")
                self.log(f"Haven couldn't put a thought into words: {error}")
            finally:
                thinker.listener = None
        with self._lock:
            turn.update(reply=answer, confidence=round(confidence, 3), draft="", phase="done", done=True)
        if answer:
            life.reply(answer, "")
            thinker.remember({"you": text, "haven": answer, "confidence": confidence, "time": time.time()})

    def _follow(self, turn: dict):
        """What the page sees of a reply taking shape."""
        from .cortex.think import spoken

        def hear(kind: str, text: str) -> None:
            with self._lock:
                if kind == "draft":  # a fresh try at the words: the last one becomes a passing thought
                    if spoken(turn["draft"]):
                        turn["thoughts"].append({"kind": "draft", "text": spoken(turn["draft"])})
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
        turns = []
        for line in path.read_text().splitlines()[-n:]:
            with contextlib.suppress(ValueError, KeyError):
                item = json.loads(line)
                turns += [
                    {"tick": 0, "who": "you", "text": item["you"]},
                    {"tick": 0, "who": "haven", "text": item["haven"]},
                ]
        with self.life.lock:
            self.life.conversation = turns + self.life.conversation

    def describe(self) -> dict:
        with self._lock:
            busy = [i for i, t in self.turns.items() if not t["done"]]
        return {**self.status, "busy": busy[0] if busy else None}


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
        if self.path not in ("/api/chat", "/api/rest"):
            super().do_POST()
            return
        body = self._body()
        if body is None:
            return
        if self.path == "/api/rest":
            self._json({"ok": True})
            self.chat.resting.set()
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


def running(port: int) -> bool:
    """Whether a Haven app is already open on this port (then that's the one to go to)."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{port}/api/chat", timeout=2) as response:
            return "stage" in json.load(response)
    except (OSError, ValueError):
        return False


def run(args: argparse.Namespace, store: Store, term) -> int:
    url = f"http://127.0.0.1:{args.port}"
    if running(args.port):
        term.say(f"Haven is already awake. Its window: {url}")
        if not args.no_browser:
            webbrowser.open(url)
        return 0
    new = not store.exists()
    mind = open_mind(store)
    life = Life(mind, store, speed=args.speed)
    if new:
        term.say(f"{mind.me.name} is born, in a nest in the corner of its garden.")
        store.save(mind.to_state())
    chat = Chat(life, base=args.base, device=args.device, web=None if args.no_web else Web(), log=term.dim)
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
