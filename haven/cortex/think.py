"""Thinking in words: inner speech that passes through Haven's global workspace.

When someone talks to Haven, its language cortex drafts a few possible replies, reading
Haven's current state as it does. How sure it is comes from its own signals: how likely
it found its own words, and whether its drafts agree. Each draft enters the workspace as
a thought, competing with everything else Haven is aware of, and is remembered like any
other experience. If it isn't sure, and it's allowed to, it looks the subject up in an
encyclopedia and thinks again with what it read. It learns from its conversations and
its reading while it sleeps.

It can instead borrow a much larger language model running on this computer through
Ollama. That makes it far more articulate, but the words are no longer its own, and they
reach the model only as a written description of its state rather than as its state.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from ..report import readout
from ..web import Web, WebError
from . import sources
from .grounding import mind_state

_COMMON = (
    "what when where which who whom whose why how does did do is are was were be been being have has had "
    "the a an and or but if then than that this these those there their they them you your yours about "
    "with from into onto for of on in at by to it its it's can could would should will shall may might "
    "must tell know think please"
)
STOPWORDS = frozenset(_COMMON.split())


def agreement(texts: list[str]) -> float:
    """How much independent drafts say the same thing (word overlap, averaged over pairs)."""
    sets = [set(re.findall(r"[a-z']+", t.lower())) for t in texts]
    pairs = [(a, b) for i, a in enumerate(sets) for b in sets[i + 1 :]]
    if not pairs:
        return 0.0
    return float(np.mean([len(a & b) / max(len(a | b), 1) for a, b in pairs]))


def topic_of(text: str) -> str | None:
    """What a question is about: a name if there is one, else its most specific word."""
    names = re.findall(r"(?<!^)(?<![.?!] )\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)", text)
    if names:
        return names[0]
    words = [w for w in re.findall(r"[A-Za-z]+", text) if w.lower() not in STOPWORDS and len(w) >= 4]
    return max(words, key=len) if words else None


class Thinker:
    """What Life needs from a language cortex."""

    name = "none"

    def __init__(self, root: Path, web: Web | None = None):
        self.root = Path(root)
        self.web = web
        self.busy = threading.Lock()

    def describe(self) -> str:
        return self.name

    def respond_later(self, life, text: str) -> None:
        if self.busy.locked():
            return  # it's still thinking about the last thing
        threading.Thread(target=self._respond, args=(life, text), daemon=True, name="haven-thinking").start()

    def _respond(self, life, text: str) -> None:
        with self.busy:
            try:
                answer, confidence = self.deliberate(life, text)
            except Exception as error:  # noqa: BLE001  thinking going wrong mustn't end a life
                life._emit("event", f"couldn't put a thought into words ({error})")
                return
            if answer:
                life.reply(answer, f"{self.name}, confidence {confidence:.0%}")
                self.remember({"you": text, "haven": answer, "confidence": confidence, "time": time.time()})

    def deliberate(self, life, text: str) -> tuple[str, float]:
        raise NotImplementedError

    def remember(self, exchange: dict) -> None:
        path = self.root / "cortex" / "conversations.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(exchange) + "\n")

    def look_up(self, topic: str, life=None) -> str | None:
        """Read about something in the Simple English Wikipedia. Returns a short excerpt."""
        if self.web is None:
            return None
        try:
            found = sources.article(self.web, sources.URLS["simplewiki"], topic)
        except WebError:
            return None
        if found is None:
            return None
        title, text = found
        path = self.root / "cortex" / "readings.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps({"title": title, "text": text[:20000], "time": time.time()}) + "\n")
        if life is not None:
            life._emit("event", f"read about {title} to answer that")
        first = " ".join(text.split())[:700]
        return f"{title}: {first.rsplit('. ', 1)[0]}."

    def consolidate(self) -> None:
        """Learn from recent conversations and reading (called while Haven sleeps)."""


class OwnThinker(Thinker):
    name = "its own cortex"

    def __init__(self, root: Path, device: str = "auto", web: Web | None = None):
        super().__init__(root, web)
        import torch

        from .model import Cortex, CortexConfig
        from .tokenizer import Tokenizer
        from .train import pick_device

        self.torch = torch
        self.device = pick_device(device)
        checkpoint = torch.load(self.root / "cortex" / "cortex.pt", map_location="cpu", weights_only=False)
        self.progress = checkpoint["progress"]
        self.tok = Tokenizer.load(self.root / "cortex" / "tokenizer.json")
        self.model = Cortex(CortexConfig(**checkpoint["config"]))
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device).eval()
        self.model_lock = threading.Lock()
        self._optimizer = None
        self._since_save = 0

    def describe(self) -> str:
        from .curriculum import LEVELS

        level = min(self.progress["level"], len(LEVELS))
        done = [n for n, r in self.progress["levels"].items() if r["status"] in ("passed", "moved on", "plateaued")]
        return (
            f"its own, {self.model.parameters_count() / 1e6:.1f}M connections; "
            f"studied through level {level} ({LEVELS[level - 1].name}); {len(done)} levels finished"
        )

    def deliberate(self, life, text: str, attempts: int = 3) -> tuple[str, float]:
        from .tokenizer import END, HAVEN, THINK, YOU

        torch = self.torch
        with life.lock:
            state = torch.tensor(mind_state(life.mind), device=self.device).unsqueeze(0)
            history = life.conversation[-7:-1]
        with self.model_lock:
            heard = self.model.meaning([YOU, *self.tok.encode(text)], state).float().cpu().numpy()
        with life.lock:
            life.mind.understand(text, heard)  # what it made of what was said comes to mind first
        prompt = []
        for turn in history:
            prompt += [YOU if turn["who"] == "you" else HAVEN, *self.tok.encode(turn["text"])]
        question = [YOU, *self.tok.encode(text), HAVEN]
        notes: list[int] = []
        best = ("", 0.0)
        for attempt in range(attempts):
            drafts = []
            with self.model_lock:
                for _ in range(3):
                    tokens, logprobs = self.model.generate(
                        (prompt + notes + question)[-(self.model.cfg.context - 60) :],
                        state,
                        max_new=48,
                        temperature=0.8,
                        stop=(END, YOU),
                    )
                    words = self.tok.decode(tokens).strip()
                    likely = math.exp(float(np.mean(logprobs))) if logprobs else 0.0
                    drafts.append((words, likely, tokens))
                words, likely, tokens = max(drafts, key=lambda d: d[1])
                meaning = self.model.meaning(question + tokens, state).float().cpu().numpy()
            confidence = 0.5 * likely + 0.5 * agreement([d[0] for d in drafts])
            with life.lock:
                life.mind.think(words, meaning, confidence)  # the thought enters the workspace
            if confidence > best[1] or not best[0]:
                best = (words, confidence)
            if confidence >= 0.5:
                break
            topic = topic_of(text) if attempt == 0 else None
            excerpt = self.look_up(topic, life) if topic else None
            if not excerpt:
                if attempt > 0:
                    break
                continue
            notes = [THINK, *self.tok.encode(f"I read about {excerpt}")]
        return best

    def consolidate(self, steps: int = 2) -> None:
        """A little learning in its sleep, from what was said to it and what it read."""
        from .tokenizer import END, HAVEN, YOU

        torch = self.torch
        texts = []
        for name in ("conversations.jsonl", "readings.jsonl"):
            path = self.root / "cortex" / name
            if path.exists():
                for line in path.read_text().splitlines()[-200:]:
                    item = json.loads(line)
                    if "you" in item:
                        texts.append([YOU, *self.tok.encode(item["you"]), HAVEN, *self.tok.encode(item["haven"]), END])
                    else:
                        texts.append(self.tok.encode(item["text"][:4000]) + [END])
        if not texts:
            return
        rng = np.random.default_rng()
        with self.model_lock:
            model = self.model
            if self._optimizer is None:
                self._optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
            model.train()
            for _ in range(steps):
                ids = texts[int(rng.integers(len(texts)))][: model.cfg.context + 1]
                if len(ids) < 3:
                    continue
                batch = torch.tensor([ids], device=self.device)
                logits, _ = model(batch[:, :-1])
                loss = torch.nn.functional.cross_entropy(logits[0].float(), batch[0, 1:])
                self._optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self._optimizer.step()
            model.eval()
            self._since_save += steps
            if self._since_save >= 50:
                self.save()

    def save(self) -> None:
        torch = self.torch
        path = self.root / "cortex" / "cortex.pt"
        with self.model_lock:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            checkpoint["model"] = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
            tmp = path.with_suffix(".pt.tmp")
            torch.save(checkpoint, tmp)
            tmp.replace(path)
            self._since_save = 0


class OllamaThinker(Thinker):
    """A borrowed cortex: a local model, told in writing what Haven's state is."""

    def __init__(self, root: Path, model: str, host: str = "http://127.0.0.1:11434", web: Web | None = None):
        super().__init__(root, web)
        self.model = model
        self.host = host.rstrip("/")
        self.name = f"borrowed ({model} through Ollama)"
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def describe(self) -> str:
        return f"borrowed: {self.model} running in Ollama on this computer (not its own; it reads a description of Haven's state)"

    def available(self) -> bool:
        try:
            with self._opener.open(self.host + "/api/version", timeout=3):
                return True
        except (urllib.error.URLError, OSError):
            return False

    def deliberate(self, life, text: str) -> tuple[str, float]:
        with life.lock:
            mind = life.mind
            lines = readout(mind) + mind.me.conclusions()
            words = mind.lexicon.vocabulary()
            memories = [e.label for e in mind.memory.recent(5)]
            name = mind.me.name
            history = life.conversation[-9:-1]
        system = (
            f"You are the language area of {name}, a small artificial creature that lives in a garden with day and "
            "night, berries, thorns and a nest. It was built to meet the conditions scientific theories link to "
            "consciousness; whether it feels anything is an open question. Speak as "
            f"{name}, in the first person, in one to three short sentences. Only describe feelings, perceptions, "
            "memories and beliefs that its state below shows; don't make up experiences it hasn't had.\n\n"
            "Its state right now, read from its instruments:\n- "
            + "\n- ".join(lines)
            + f"\nWords it has learned from people: {', '.join(words) or 'none yet'}."
            + f"\nRecent memories: {'; '.join(memories) or 'none'}."
        )
        messages = [{"role": "system", "content": system}]
        for turn in history:
            messages.append({"role": "user" if turn["who"] == "you" else "assistant", "content": turn["text"]})
        messages.append({"role": "user", "content": text})
        drafts = [self._chat(messages) for _ in range(2)]
        drafts = [d for d in drafts if d]
        if not drafts:
            return "", 0.0
        confidence = agreement(drafts) if len(drafts) > 1 else 0.5
        with life.lock:
            life.mind.think(drafts[0], None, confidence)
        return drafts[0], confidence

    def _chat(self, messages: list[dict]) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "think": False,
                "options": {"temperature": 0.8, "num_predict": 160},
            }
        ).encode()
        request = urllib.request.Request(
            self.host + "/api/chat", data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        with self._opener.open(request, timeout=300) as response:
            content = json.load(response).get("message", {}).get("content", "")
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def make_thinker(spec: str, root: Path, web: Web | None = None) -> tuple[Thinker | None, str]:
    """Pick the language cortex: its own, a borrowed local model, or none."""
    web = web if web is not None else Web()
    if spec == "none":
        return None, ""
    if spec.startswith("ollama:"):
        thinker = OllamaThinker(root, spec.split(":", 1)[1], web=web)
        if not thinker.available():
            return None, "Ollama isn't answering at http://127.0.0.1:11434, so there's no borrowed cortex this time."
        return thinker, f"Language cortex: {thinker.describe()}."
    if not (Path(root) / "cortex" / "cortex.pt").exists():
        return None, "It has no language cortex yet (train one with: haven learn). It can still learn words from you."
    thinker = OwnThinker(root, web=web)
    return thinker, f"Language cortex: {thinker.describe()}."
