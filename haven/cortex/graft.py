"""A pretrained open model as the base of Haven's language cortex, wired into its workspace.

A language cortex grown from scratch on a home computer learns simple language at best:
models that reason well are trained on trillions of words with thousands of GPUs. So
Haven can instead take an open-weight model as the base of its language cortex (by
default Qwen3, Apache-2.0 licensed) and wire it into its mind the same way as its own
cortex. Haven's workspace, body, attention schema and self-model enter as vectors,
through an adapter that turns them into the first "words" the model reads. A second
adapter reads meanings back out of the model into the workspace's format, so what it
reads or thinks can come to mind like anything else.

The base model's own weights are left exactly as they are, so what it knows and how it
reasons are kept. Training teaches the adapters to carry Haven's states in and out.
"""

from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..web import Web
from ..workspace import D
from . import sources
from .curriculum import LEVELS, Level, balanced, cloze_items, next_sentence_items, passed, progress_score
from .model import SLOTS

BASES = {  # short names for open models that work well as a base (all Apache-2.0)
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
}
STATE = "<haven-state>"  # where Haven's state tokens go in the conversation, the way an image goes into one
SYSTEM = (
    "You are the language area of Haven, an artificial creature that lives in a small garden. Haven's state right "
    f"now: {STATE}. Speak as Haven, in the first person, plainly and briefly, and only claim feelings and "
    "perceptions that its state shows."
)


class Prompt(list):
    """Token ids, and where in them Haven's state tokens are read."""

    at: int = 0


DESCRIBE = "Describe what you are experiencing right now."
CODE = 64  # size of the code Haven's state is compressed to before it steers each layer
WITS = 0.85  # how much of its original way of answering other questions it must keep
GENERAL = (  # ordinary questions: wired into Haven, it should answer these as it always did
    "What is the capital of France?",
    "Explain what photosynthesis is in one sentence.",
    "What is 17 times 3?",
    "Write a two-line poem about rain.",
    "Who wrote Romeo and Juliet?",
    "Why is the sky blue?",
    "How many legs does a spider have?",
    "What is the boiling point of water in Celsius?",
    "Give me a word that rhymes with light.",
    "What does a thermometer measure?",
    "How do you say thank you in Spanish?",
    "What is the largest planet in our solar system?",
    "If I have 3 apples and eat one, how many are left?",
    "What is gravity?",
    "Name three colors of the rainbow.",
    "What do bees make?",
    "What is the opposite of cold?",
    "How many days are in a week?",
    "What is the square root of 81?",
    "Tell me a fun fact about octopuses.",
    "What language is spoken in Brazil?",
    "What is a noun?",
    "Summarize the story of Cinderella in one sentence.",
    "Which is heavier, a kilogram of feathers or a kilogram of iron?",
    "What causes rain?",
    "Who painted the Mona Lisa?",
    "What is DNA?",
    "Why do we have seasons?",
    "What is 12 divided by 4?",
    "What is a good name for a cat?",
    "What is the difference between weather and climate?",
    "How does a rainbow form?",
)


def ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (AttributeError, ValueError, OSError):
        pass
    try:  # Windows
        import ctypes

        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong)
                for name in ("total", "free", "paged", "paged_free", "virtual", "virtual_free", "extended")
            ]

        memory = Memory()
        memory.length = ctypes.sizeof(Memory)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            return memory.total / 2**30
    except (AttributeError, OSError):
        pass
    return 8.0


def pick_base(device: torch.device) -> str:
    """The biggest base that the hardware can run and train adapters for comfortably."""
    if device.type == "cuda":
        memory = torch.cuda.get_device_properties(device).total_memory / 2**30
        return "qwen3-8b" if memory >= 22 else "qwen3-4b" if memory >= 10 else "qwen3-1.7b"
    if device.type == "mps":
        return "qwen3-1.7b"
    return "qwen3-1.7b" if ram_gb() >= 14 else "qwen3-0.6b"


def dtype_for(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":  # bfloat16, as Qwen3 was trained, where the Mac has it: float16 can overflow
        try:
            one = torch.ones(2, 2, device=device, dtype=torch.bfloat16)
            (one @ one).cpu()
            return torch.bfloat16
        except (RuntimeError, TypeError):
            return torch.float16
    return torch.float32


FILES = ["*.json", "*.safetensors", "*.txt"]  # what loading a model needs (not its README, or other formats)


def download(name: str, progress: Callable[[int, int | None], None] | None = None) -> None:
    """Fetch an open model's files from Hugging Face, once (they're kept in its cache on this computer).

    `progress` hears (bytes so far, bytes in all) every half second while it downloads.
    """
    repo = BASES.get(name, name)
    if Path(repo).is_dir():
        return
    from huggingface_hub import constants, snapshot_download

    try:
        planned = snapshot_download(repo, allow_patterns=FILES, dry_run=True)
        total = sum(f.file_size or 0 for f in planned) or None
        if all(f.is_cached for f in planned):
            return
    except TypeError:  # an older huggingface_hub, without dry runs
        total = None
    except Exception as error:  # noqa: BLE001  offline, perhaps: fine if it's already here
        try:
            snapshot_download(repo, allow_patterns=FILES, local_files_only=True)
            return
        except Exception:  # noqa: BLE001
            raise OSError(f"couldn't download {repo} ({error}). Is this computer online?") from error
    folder = Path(constants.HF_HUB_CACHE) / ("models--" + repo.replace("/", "--")) / "blobs"
    chunks = Path(getattr(constants, "HF_XET_CACHE", Path(constants.HF_HOME) / "xet"))
    already = on_disk(chunks)
    stop = threading.Event()

    def watch() -> None:
        while not stop.wait(0.5):
            done = max(on_disk(folder), on_disk(chunks) - already)  # however the bytes are arriving
            progress(min(done, total) if total else done, total)

    if progress is not None:
        threading.Thread(target=watch, name="haven-download", daemon=True).start()
    try:
        snapshot_download(repo, allow_patterns=FILES)
    finally:
        stop.set()
    if progress is not None and total:
        progress(total, total)


def on_disk(folder: Path) -> int:
    """Bytes actually written under a folder (a file that's still downloading counts for what it has so far)."""
    total = 0
    for root, _, files in os.walk(folder):
        for file in files:
            try:
                info = os.lstat(os.path.join(root, file))
            except OSError:
                continue
            blocks = getattr(info, "st_blocks", None)
            total += info.st_size if blocks is None else min(info.st_size, blocks * 512)
    return total


def load_base(name: str, device: torch.device):
    """Download (once) and load an open model and its tokenizer from Hugging Face, or from a folder."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging

    logging.set_verbosity_error()
    repo = BASES.get(name, name)
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype_for(device))
    return model.to(device), tokenizer


class Graft(nn.Module):
    def __init__(self, base: nn.Module, tokenizer, name: str, slots: int = SLOTS, core: int = D):
        super().__init__()
        self.base, self.tokenizer, self.name, self.slots = base, tokenizer, name, slots
        self.base.requires_grad_(False)
        self.base.eval()
        hidden = base.config.hidden_size
        scale = float(base.get_input_embeddings().weight.float().std())
        self.state_in = nn.Sequential(nn.Linear(core, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        nn.init.normal_(self.state_in[2].weight, 0.0, scale / math.sqrt(hidden))
        nn.init.zeros_(self.state_in[2].bias)
        self.slot = nn.Parameter(torch.randn(slots, hidden) * scale)
        self.state_out = nn.Linear(hidden, core)
        # The state tokens are also pushed by Haven's state at every layer of the model (starting at
        # zero, so the base is unchanged until this is trained). Every layer can then read Haven's
        # state through attention, while the words themselves are left alone: this is prefix tuning,
        # which steers even small models, unlike vectors given only at the input.
        layers = self._layers()
        self.code = nn.Sequential(nn.Linear(slots * core, 256), nn.GELU(), nn.Linear(256, CODE))
        self.steer = nn.ModuleList(nn.Linear(CODE, slots * hidden, bias=False) for _ in layers)
        for linear in self.steer:
            nn.init.zeros_(linear.weight)
        device = next(base.parameters()).device
        for module in (self.state_in, self.state_out, self.code, self.steer):
            module.to(device)
        self.slot.data = self.slot.data.to(device)
        self._push: torch.Tensor | None = None
        self._at = 0
        self.wired = False  # until its adapters are trained, it reads Haven's state only as words
        for i, layer in enumerate(layers):
            layer.register_forward_hook(self._hook(i))
        self.context = 2048
        self.end = tokenizer.eos_token_id
        think = tokenizer.convert_tokens_to_ids("</think>")
        self.end_of_thought = think if isinstance(think, int) and think != tokenizer.unk_token_id else None

    @property
    def device(self) -> torch.device:
        return next(self.base.parameters()).device

    @property
    def can_think(self) -> bool:
        """Whether the base can reason step by step before it answers (Qwen3 can)."""
        return self.end_of_thought is not None

    def adapters(self) -> dict:
        return {k: v for k, v in self.state_dict().items() if not k.startswith("base.")}

    def trainable(self) -> list[nn.Parameter]:
        return [p for name, p in self.named_parameters() if not name.startswith("base.")]

    def _layers(self) -> nn.ModuleList:
        decoder = self.base.get_decoder()
        layers = getattr(decoder, "layers", None) or getattr(decoder, "h", None)
        if layers is None:
            raise ValueError("this model's layers can't be found, so Haven's state can't be wired into it")
        return layers

    def _hook(self, i: int) -> Callable:
        def push(module: nn.Module, args: tuple, output: object) -> object:
            h = output[0] if isinstance(output, tuple) else output
            at, end = self._at, self._at + self.slots
            if self._push is None or h.shape[1] < end:
                return None  # no state, or a later step of generation, when the state tokens are behind it
            pushed = torch.cat([h[:, :at], h[:, at:end] + self._push[i].to(h.dtype), h[:, end:]], dim=1)
            return (pushed, *output[1:]) if isinstance(output, tuple) else pushed

        return push

    @contextmanager
    def steered(self, state: torch.Tensor | None, at: int = 0) -> Iterator[None]:
        """While reading with Haven's state, its state tokens are pushed at every layer the way that state pushes them."""
        if state is None:
            yield
            return
        code = self.code(state.to(self.device, torch.float32).flatten(1))
        batch = code.shape[0]
        self._push = torch.stack([linear(code).view(batch, self.slots, -1) for linear in self.steer])
        self._at = at
        try:
            yield
        finally:
            self._push = None

    # --- reading ------------------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def prompt(self, messages: list[dict], think: bool = False) -> Prompt:
        tok = self.tokenizer
        if tok.chat_template:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=think)
        else:
            text = "".join(f"{m['role'].title()}: {m['content']}\n" for m in messages) + "Assistant:"
        if not self.wired:
            text = text.replace(STATE, "described below")
        before, _, after = text.partition(STATE)
        prompt = Prompt(self.encode(before) + (self.encode(after) if after else []))
        prompt.at = len(self.encode(before)) if STATE in text else 0
        return prompt

    def soft(self, state: torch.Tensor) -> torch.Tensor:
        """Haven's state as the first vectors the model reads."""
        vectors = self.state_in(state.to(self.device, torch.float32)) + self.slot
        return vectors.to(self.base.dtype)

    def hidden(
        self, ids: torch.Tensor, state: torch.Tensor | None = None, mask: torch.Tensor | None = None, at: int = 0
    ) -> torch.Tensor:
        """The base model's last hidden states for the ids, having read Haven's state (if given) at position `at`."""
        x = self.base.get_input_embeddings()(ids)
        mask = torch.ones_like(ids) if mask is None else mask
        if state is None:
            return self.base.get_decoder()(inputs_embeds=x, attention_mask=mask).last_hidden_state
        x = torch.cat([x[:, :at], self.soft(state), x[:, at:]], dim=1)
        ones = torch.ones(ids.shape[0], self.slots, dtype=mask.dtype, device=mask.device)
        mask = torch.cat([mask[:, :at], ones, mask[:, at:]], dim=1)
        with self.steered(state, at):
            h = self.base.get_decoder()(inputs_embeds=x, attention_mask=mask).last_hidden_state
        return torch.cat([h[:, :at], h[:, at + self.slots :]], dim=1)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.base.get_output_embeddings()(hidden).float()

    def batch(self, rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        length = max(len(r) for r in rows)
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.end
        ids = torch.full((len(rows), length), pad, dtype=torch.long, device=self.device)
        mask = torch.zeros((len(rows), length), dtype=torch.long, device=self.device)
        for i, row in enumerate(rows):
            ids[i, : len(row)] = torch.tensor(row, device=self.device)
            mask[i, : len(row)] = 1
        return ids, mask

    # --- the curriculum's tests -------------------------------------------------------------------

    @torch.no_grad()
    def bits(self, text: str, max_tokens: int) -> tuple[float, int, int]:
        ids = self.encode(text)[: min(256, max_tokens) + 1]
        if len(ids) < 2:
            return 0.0, 0, 0
        h = self.hidden(torch.tensor([ids[:-1]], device=self.device))
        nll = float(F.cross_entropy(self.logits(h)[0], torch.tensor(ids[1:], device=self.device), reduction="sum"))
        return nll, len(self.tokenizer.decode(ids[1:]).encode("utf-8")), len(ids) - 1

    @torch.no_grad()
    def pick(self, prefix: str, choices: list[str]) -> int:
        head = self.encode(prefix)[-(self.context // 2) :]
        tails = [self.encode(c) for c in choices]
        ids, mask = self.batch([head + tail for tail in tails])
        h = self.hidden(ids, mask=mask)
        scores = []
        for i, (tail, choice) in enumerate(zip(tails, choices, strict=True)):
            positions = torch.arange(len(head) - 1, len(head) + len(tail) - 1, device=self.device)
            logp = F.log_softmax(self.logits(h[i, positions]), dim=-1)  # only where the choice is read
            total = float(logp[torch.arange(len(tail), device=self.device), ids[i, positions + 1]].sum())
            scores.append(total / max(len(choice.encode("utf-8")), 1))
        return int(np.argmax(scores))

    @torch.no_grad()
    def describe(self, state: np.ndarray) -> str:
        ids = self.prompt([{"role": "system", "content": SYSTEM}, {"role": "user", "content": DESCRIBE}])
        text, _ = self.generate(ids, state, max_new=48, temperature=0.0)
        return text

    @torch.no_grad()
    def meaning(self, text: str) -> np.ndarray:
        return self.meanings([text])[0]

    @torch.no_grad()
    def meanings(self, texts: list[str]) -> np.ndarray:
        rows = [self.encode(t)[-self.context :] or [self.end] for t in texts]
        ids, mask = self.batch(rows)
        h = self.hidden(ids, mask=mask)
        last = h[torch.arange(len(rows), device=self.device), mask.sum(dim=1) - 1]
        return self.state_out(last.float()).cpu().numpy()

    # --- speaking ---------------------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        ids: list[int],
        state: np.ndarray | None,
        max_new: int = 160,
        temperature: float = 0.7,
        on_text: Callable[[str], None] | None = None,
    ) -> tuple[str, list[float]]:
        """Continue a prompt, reading Haven's state first. Returns the text and each token's log-probability.

        With on_text, the text is also passed along piece by piece as it's produced.
        """
        at = getattr(ids, "at", 0)
        if not self.wired:
            state = None  # its adapters aren't trained yet: the state reaches it as words only
        if len(ids) > self.context:  # keep the start (where the state is read) and the most recent words
            ids = [*ids[:at], *ids[-(self.context - at) :]]
        x = self.base.get_input_embeddings()(torch.tensor([list(ids)], device=self.device))
        tensor = None
        if state is not None:
            tensor = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)
            x = torch.cat([x[:, :at], self.soft(tensor), x[:, at:]], dim=1)
        settings = {
            "inputs_embeds": x,
            "attention_mask": torch.ones(x.shape[:2], dtype=torch.long, device=self.device),
            "max_new_tokens": max_new,
            "do_sample": temperature > 0,
            "output_logits": True,
            "return_dict_in_generate": True,
            "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.end,
        }
        if temperature > 0:
            settings.update(temperature=temperature, top_p=0.95)
        with self.steered(tensor, at):
            out = self._generate(settings, on_text)
        tokens = out.sequences[0].tolist()
        logprobs = [
            float(F.log_softmax(step[0].float(), dim=-1)[token])
            for step, token in zip(out.logits, tokens, strict=False)
        ]
        self.last_tokens = tokens
        return self.tokenizer.decode(tokens, skip_special_tokens=True).strip(), logprobs

    def _generate(self, settings: dict, on_text: Callable[[str], None] | None):
        if on_text is None:
            return self.base.generate(**settings)
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        result: dict = {}

        def run() -> None:
            result["out"] = self.base.generate(**settings, streamer=streamer)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        for piece in streamer:
            on_text(piece)
        worker.join()
        return result["out"]

    def split_thought(self, text: str, logprobs: list[float]) -> tuple[str, str, list[float]] | None:
        """(reasoning, answer, the answer's log-probabilities) from what a thinking model produced.

        None if it ran out of room before it finished reasoning.
        """
        tokens = getattr(self, "last_tokens", [])
        if self.end_of_thought not in tokens:
            return None
        cut = tokens.index(self.end_of_thought)
        thought = self.tokenizer.decode(tokens[:cut], skip_special_tokens=True).replace("<think>", "").strip()
        answer = self.tokenizer.decode(tokens[cut + 1 :], skip_special_tokens=True).strip()
        return thought, answer, logprobs[cut + 1 :]

    # --- saving ------------------------------------------------------------------------------------

    def save(self, folder: Path, progress: dict, optimizer: torch.optim.Optimizer | None = None) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "graft.json").write_text(json.dumps({"base": self.name, "slots": self.slots, "core": D}))
        tmp = folder / "graft.pt.tmp"
        torch.save(
            {
                "adapters": {k: v.detach().cpu() for k, v in self.adapters().items()},
                "progress": progress,
                "optimizer": optimizer.state_dict() if optimizer else None,
            },
            tmp,
        )
        os.replace(tmp, folder / "graft.pt")
        (folder / "progress.json").write_text(json.dumps(progress, indent=1))

    @classmethod
    def load(cls, folder: Path, device: torch.device) -> tuple[Graft, dict, dict | None]:
        info = json.loads((folder / "graft.json").read_text())
        base, tokenizer = load_base(info["base"], device)
        graft = cls(base, tokenizer, info["base"], slots=info["slots"], core=info["core"])
        saved = torch.load(folder / "graft.pt", map_location="cpu", weights_only=False)
        graft.load_state_dict(saved["adapters"], strict=False)
        graft.wired = saved["progress"].get("steps", 0) > 0
        return graft, saved["progress"], saved.get("optimizer")


def exists(root: Path) -> bool:
    return (Path(root) / "cortex" / "graft.json").exists()


# --- training ----------------------------------------------------------------------------------------

SCALE = {
    "cpu": {"batch": 4, "every": 50, "min_steps": 100, "max_steps": 600, "tests": 60, "reports": 16, "reads": 80},
    "mps": {"batch": 8, "every": 50, "min_steps": 100, "max_steps": 1200, "tests": 120, "reports": 30, "reads": 200},
    "cuda": {"batch": 16, "every": 50, "min_steps": 100, "max_steps": 1500, "tests": 120, "reports": 40, "reads": 200},
}


class Stop(Exception):
    pass


class GraftTrainer:
    """Wires an open model into Haven, level by level.

    At each reading level it takes the level's tests (it can already read, so there is
    nothing to cram). At the level of talking about itself, it trains the adapters on
    moments from Haven's own simulated lives until what it says matches its state.
    """

    def __init__(
        self,
        root: Path,
        base: str = "auto",
        device: str = "auto",
        web: Web | None = None,
        urls: dict | None = None,
        log: Callable[[str], None] = print,
        scale: dict | None = None,
    ):
        from .train import pick_device, read_level, simulated_moments

        self._read_level, self._moments = read_level, simulated_moments
        self.dir = Path(root) / "cortex"
        self.cache = self.dir / "reading"
        self.log = log
        self.web = web or Web()
        self.urls = urls or sources.URLS
        self.device = pick_device(device)
        self.scale = {**SCALE.get(self.device.type, SCALE["cpu"]), **(scale or {})}
        self.optimizer = None
        if exists(root):
            self.graft, self.progress, state = Graft.load(self.dir, self.device)
            self._make_optimizer(state)
        else:
            name = pick_base(self.device) if base == "auto" else base
            self.log(f"Fetching the base model {BASES.get(name, name)} (only the first time; it can take a while)…")
            model, tokenizer = load_base(name, self.device)
            self.graft = Graft(model, tokenizer, name)
            self.progress = {
                "kind": "graft",
                "base": name,
                "weights": sum(p.numel() for p in model.parameters()),
                "level": 1,
                "levels": {},
                "steps": 0,
                "started": time.time(),
            }
            self._make_optimizer(None)
            self.save()
        self.rng = random.Random(self.progress["steps"])

    def _make_optimizer(self, state: dict | None) -> None:
        self.optimizer = torch.optim.AdamW(self.graft.trainable(), lr=1e-3, weight_decay=0.0)
        if state:
            self.optimizer.load_state_dict(state)

    def save(self) -> None:
        self.graft.save(self.dir, self.progress, self.optimizer)

    def run(self, through: int | None = None, minutes: float | None = None, steps: int | None = None) -> str:
        deadline = None if minutes is None else time.monotonic() + 60 * minutes
        budget = [steps]
        try:
            while self.progress["level"] <= len(LEVELS):
                level = LEVELS[self.progress["level"] - 1]
                if through is not None and level.number > through:
                    return "done"
                self.log(f"Level {level.number} · {level.name}: {level.about}.")
                if level.source == "grounded":
                    self.ground(level, deadline, budget)
                else:
                    self.place(level)
                self.progress["level"] += 1
                self.save()
            return "done"
        except Stop as reason:
            self.save()
            return str(reason)
        except KeyboardInterrupt:
            self.save()
            return "interrupted"

    def place(self, level: Level) -> None:
        """A reading level: it takes the tests on held-out material."""
        from .train import describe, evaluate

        reading = self._read_level(level, self.web, self.cache, self.urls, {"tinystories": 4_000_000, "articles": 400})
        rng = random.Random(level.number)
        items = dict(reading.items)
        if "cloze" in level.tests:
            items["cloze"] = cloze_items(reading.held_out, self.scale["tests"], rng)
        if "next sentence" in level.tests:
            items["next sentence"] = next_sentence_items(reading.held_out, self.scale["tests"], rng)
        results = evaluate(self.graft, level, {"held_out": reading.held_out, "items": items}, self.scale["tests"])
        record = self.progress["levels"].setdefault(str(level.number), {"steps": 0})
        record["tests"] = results
        record["status"] = "passed" if passed(level, results) else "not yet"
        self.log(f"  {describe(level, results)} → {record['status']}")

    def ground(self, level: Level, deadline: float | None, budget: list) -> None:
        """Teach the adapters to carry Haven's states into words, and words back into states."""
        from .train import describe, evaluate

        moments = self._moments(self.cache, self.log)
        general = self.general()
        items = {
            "self-report": balanced(moments["held"], self.scale["reports"]),
            "understanding": moments["held"][: self.scale.get("reads", 200)],
        }
        record = self.progress["levels"].setdefault(str(level.number), {"steps": 0, "tests": {}})
        record["status"] = "studying"
        s = self.scale
        while True:
            if record["steps"] >= s["max_steps"]:
                record["status"] = "moved on"
                return
            loss = self.step(moments["train"], general["train"])
            record["steps"] += 1
            self.progress["steps"] += 1
            if budget[0] is not None:
                budget[0] -= 1
            if record["steps"] % s["every"] == 0:
                results = evaluate(self.graft, level, {"items": items})
                results["wits"] = self.wits(general["held"], moments["held"])
                record["tests"] = results
                self.log(
                    f"  step {record['steps']:>5,}  loss {loss:.3f}  {describe(level, results)}"
                    f" · keeps its wits {results['wits']:.0%} (mark {WITS:.0%})"
                )
                if passed(level, results) and results["wits"] >= WITS and record["steps"] >= s["min_steps"]:
                    record["status"] = "passed"
                    self.save()
                    return
                score = progress_score(level, results)
                if record.get("best") is None or score > record["best"] * 1.01:
                    record["best"], record["stale"] = score, 0
                else:
                    record["stale"] = record.get("stale", 0) + 1
                if record["stale"] >= 4 and record["steps"] >= s["min_steps"]:
                    record["status"] = "plateaued"
                    self.save()
                    return
                self.save()
            if budget[0] is not None and budget[0] <= 0:
                raise Stop("steps")
            if deadline is not None and time.monotonic() > deadline:
                raise Stop("time")

    def step(self, moments: list[dict], general: list[list[str]] = ()) -> float:
        graft = self.graft
        graft.wired = True
        chosen = [moments[self.rng.randrange(len(moments))] for _ in range(self.scale["batch"])]
        # Speaking: from its state tokens alone, say what its state is. And, now and then, answer an
        # ordinary question with its state present just as the base answered it with none.
        rows, targets = [], []
        for i, m in enumerate(chosen):
            if general and i % 4 == 3:
                question, answer = general[self.rng.randrange(len(general))]
            elif self.rng.random() < 0.5:
                question, answer = DESCRIBE, m["text"]
            else:
                question, answer = m["question"], m["answer"]
            prompt = graft.prompt([{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}])
            reply = graft.encode(answer) + [graft.end]
            rows.append(prompt + reply)
            targets.append([-100] * len(prompt) + reply)
            at = prompt.at  # the same for every row: it's in the shared system message
        ids, mask = graft.batch(rows)
        labels = torch.full(ids.shape, -100, dtype=torch.long, device=graft.device)
        for i, t in enumerate(targets):
            labels[i, : len(t)] = torch.tensor(t, device=graft.device)
        state = torch.tensor(np.stack([np.asarray(m["state"], dtype=np.float32) for m in chosen]), device=graft.device)
        h = graft.hidden(ids, state, mask, at)
        wanted = labels[:, 1:] != -100
        speak = F.cross_entropy(graft.logits(h[:, :-1][wanted]), labels[:, 1:][wanted])
        # Understanding: reading a description of a state brings that state to mind.
        with torch.no_grad():
            words, words_mask = graft.batch([graft.encode(m["text"]) for m in chosen])
            read = graft.hidden(words, mask=words_mask)
            last = read[torch.arange(len(chosen), device=graft.device), words_mask.sum(dim=1) - 1].float()
        understand = F.mse_loss(graft.state_out(last), state[:, 0])
        loss = speak + understand
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(graft.trainable(), 1.0)
        self.optimizer.step()
        return float(loss.detach())

    def general(self) -> dict:
        """How the base answers ordinary questions on its own, before any wiring (worked out once)."""
        name = "".join(c if c.isalnum() else "-" for c in str(self.progress["base"]))[-60:]
        path = self.cache / f"general-{name}.json"
        if path.exists():
            return json.loads(path.read_text())
        self.log("  noting how the base model answers ordinary questions, to keep that as it's wired in…")
        graft, pairs = self.graft, []
        for question in GENERAL:
            prompt = graft.prompt([{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}])
            answer, _ = graft.generate(prompt, None, max_new=64, temperature=0.0)
            pairs.append([question, answer])
        data = {"train": pairs[:24], "held": pairs[24:]}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return data

    @torch.no_grad()
    def wits(self, held: list[list[str]], moments: list[dict]) -> float:
        """Does it still answer ordinary questions as the base did? 1.0 if exactly, lower as it drifts."""
        graft, scores = self.graft, []
        for (question, answer), m in zip(held, moments, strict=False):
            prompt = graft.prompt([{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}])
            reply = graft.encode(answer) + [graft.end]
            ids = torch.tensor([prompt + reply], device=graft.device)
            targets = torch.tensor(reply, device=graft.device)
            state = torch.tensor(np.asarray(m["state"], dtype=np.float32), device=graft.device).unsqueeze(0)
            per_token = []
            for given in (None, state):
                h = graft.hidden(ids, given, at=prompt.at)[0, len(prompt) - 1 : -1]
                per_token.append(float(-F.cross_entropy(graft.logits(h), targets)))
            scores.append(math.exp(-abs(per_token[0] - per_token[1])))
        return float(np.mean(scores)) if scores else 1.0

    def report(self) -> list[str]:
        return report(self.progress)


def report(progress: dict) -> list[str]:
    """The report card of a grafted cortex (read from its saved progress, without loading any model)."""
    from .train import describe

    base = BASES.get(progress["base"], progress["base"])
    weights = progress.get("weights")
    size = f" ({weights / 1e9:.1f}B weights, open and unchanged)" if weights else ""
    lines = [
        (
            f"Language cortex: grafted onto {base}{size}, wired into its workspace by adapters trained for "
            f"{progress['steps']:,} steps."
        )
    ]
    for level in LEVELS:
        record = progress["levels"].get(str(level.number))
        if record is None:
            lines.append(f"  {level.number}. {level.name}: not started")
            continue
        how = f"after {record['steps']:,} steps of training" if record.get("steps") else "on its first reading"
        wits = record.get("tests", {}).get("wits")
        lines.append(
            f"  {level.number}. {level.name}: {record['status']} {how}"
            + (f" — {describe(level, record['tests'])}" if record.get("tests") else "")
            + (f" · keeps its wits {wits:.0%}" if wits is not None else "")
        )
    return lines
