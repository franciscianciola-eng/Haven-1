"""Grows the language cortex Haven starts with, and lets it hear what a small child hears.

    python packaging/build_corpus.py               # first, what it hears (into dist/corpus/)
    python packaging/grow_cortex.py --hours 14     # then this (it can be stopped, and run again to carry on)

It starts from the starter cortex (haven/cortex/starter/). Its vocabulary grows with words
from what it hears, without changing how it reads anything it already knew, and the
cortex itself grows (twice as wide, and deeper) so that it starts out saying exactly what
it said before (see model.grow). Then it listens and reads: two steps in three are
what it hears (stretches of speech to children, and passages of children's books), and
every third step it practises talking about itself as before, so it keeps that. It's
tested as it goes, on held-out speech and books and held-out conversations, and a
snapshot is kept at each test, to choose from at the end.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from haven.cortex import hearing, starter, talk
from haven.cortex.curriculum import ScratchReader, balanced, conversation, self_report, understanding
from haven.cortex.model import Cortex, CortexConfig, grow
from haven.cortex.sources import _passages
from haven.cortex.stories import from_passage
from haven.cortex.tokenizer import END, Tokenizer
from haven.cortex.train import Trainer, conversation_items, known_texts, simulated_moments

MOMENTS = ROOT / "dist" / "starter-valley" / "cortex" / "reading" / "grounded.npz"  # moments of simulated lives


def load(folder: Path) -> tuple[Cortex, Tokenizer, dict]:
    checkpoint = torch.load(folder / "cortex.pt", map_location="cpu", weights_only=False)
    model = Cortex(CortexConfig(**checkpoint["config"]))
    model.load_state_dict({k: v.float() for k, v in checkpoint["model"].items()})
    return model.eval(), Tokenizer.load(folder / "tokenizer.json"), checkpoint["progress"]


def create(work: Path, heard: hearing.Heard, pieces: int, width: int, layers: int) -> None:
    """The starter cortex, grown: more pieces of words, more units, more layers, the same answers."""
    folder = work / "cortex"
    (folder / "reading").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(MOMENTS, folder / "reading" / "grounded.npz")
    model, tok, progress = load(starter.FOLDER)
    moments = simulated_moments(folder / "reading", print)
    old = len(tok)
    print(f"starter: {model.parameters_count() / 1e6:.1f}M connections, {old:,} pieces")
    added = tok.learn(hearing.mixed(heard), pieces, keep=known_texts(moments["train"] + moments["held"]))
    model.grow_vocabulary(len(tok), tok.merges[-added:] if added else [])
    grown = grow(model, width, layers, noise=0.1, seed=0).eval()
    print(f"grown: {grown.parameters_count() / 1e6:.1f}M connections, {len(tok):,} pieces (+{added:,})")
    old_reader = ScratchReader(load(starter.FOLDER)[0], Tokenizer.load(starter.FOLDER / "tokenizer.json"))
    new_reader = ScratchReader(grown, tok)
    items = conversation_items(moments["held"], 60, seed=3)
    same = sum(
        old_reader.reply(i["state"], i["notes"], i["question"])
        == new_reader.reply(i["state"], i["notes"], i["question"])
        for i in items
    )
    print(f"the grown cortex says the same as before to {same} of {len(items)} held-out questions")
    progress = {
        **progress,
        "size": "grown",
        "grown": {"from": progress.get("starter"), "width": width, "layers": layers, "pieces": added},
        "hearing": {"steps": 0, "seconds": 0.0, "tokens": 0, "tests": []},
    }
    tok.save(folder / "tokenizer.json")
    torch.save(
        {"config": asdict(grown.cfg), "model": grown.state_dict(), "optimizer": None, "progress": progress},
        folder / "cortex.pt",
    )
    (folder / "progress.json").write_text(json.dumps(progress, indent=1))


def story_pool(folder: Path, heard: hearing.Heard, seed: int = 0) -> list:
    """Stories to be asked for as it practises talking: bits of the books it learns from (not the held-out ones)."""
    rng = random.Random(seed)
    pool = []
    for kind in ("bedtime", "tales", "rhymes", "series", "classics"):
        for n, number in enumerate(hearing.KINDS[kind]):
            path = folder / "books" / f"{number}.txt"
            if n % hearing.HELD == hearing.HELD // 2 or str(number) not in heard.books or not path.exists():
                continue
            title = hearing.title_of(hearing.repos().get(number, heard.books[str(number)]))
            for passage in _passages(path.read_text()):
                story = from_passage(title, passage, rng)
                if story is not None:
                    pool.append(story)
    return pool


def stream(trainer: Trainer, heard: hearing.Heard) -> np.ndarray:
    """Everything it hears, shuffled together, as one long stream of tokens (made once)."""
    path = trainer.dir / "tokens" / "heard.npy"
    if path.exists():
        return np.load(path)
    ids: list[int] = []
    for passage in hearing.mixed(heard):
        ids += trainer.tok.encode(passage)
        ids.append(END)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.array(ids, dtype=np.int32))
    return np.load(path)


def text_loss(trainer: Trainer, ids: np.ndarray, batch: int) -> tuple[torch.Tensor, int]:
    t = trainer.model.cfg.context
    starts = [trainer.rng.randrange(0, len(ids) - t - 1) for _ in range(batch)]
    rows = torch.tensor(np.stack([ids[s : s + t + 1] for s in starts]).astype(np.int64))
    with trainer._autocast():
        logits, _ = trainer.model(rows[:, :-1])
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), rows[:, 1:].reshape(-1))
    return loss, batch * t


@torch.no_grad()
def bits_per_byte(reader: ScratchReader, passages: list[str]) -> float:
    nll, count = 0.0, 0
    for passage in passages:
        n, b, _ = reader.bits(passage, reader.model.cfg.context)
        nll, count = nll + n, count + b
    return nll / max(count, 1) / math.log(2)


def test(trainer: Trainer, items: dict, held: dict[str, list[str]]) -> dict:
    reader = ScratchReader(trainer.model, trainer.tok)
    results = {
        "conversation": conversation(reader, items["conversation"]),
        "self-report": self_report(reader, items["self-report"]),
        "understanding": understanding(reader, items["understanding"]),
    }
    results.update({f"bpb {kind}": bits_per_byte(reader, passages) for kind, passages in held.items()})
    trainer.model.train()
    return {k: round(v, 4) for k, v in results.items() if v is not None}


def pack(work: Path, heard: hearing.Heard, step: str, out: Path) -> None:
    """The snapshot at a step, saved as the cortex Haven starts with (see starter.pack)."""
    trainer = Trainer(work, device="cpu")
    snapshot = torch.load(work / "snapshots" / f"step-{step}.pt", map_location="cpu", weights_only=False)
    trainer.model.load_state_dict({k: v.float() for k, v in snapshot["model"].items()})
    trainer.progress["heard"] = {
        "words": sum(heard.words().values()),
        "speech": heard.words()["speech"],
        "books": len(heard.books),
    }
    trainer.progress["hearing"]["kept"] = int(step)
    trainer.progress["level"] = 1  # at home, it can go on to read the curriculum (haven learn)
    starter.pack(trainer, out)
    print(f"saved the cortex at step {step} to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default=str(ROOT / "dist" / "grown"))
    parser.add_argument("--corpus", default=str(ROOT / "dist" / "corpus" / "heard.json"))
    parser.add_argument("--hours", type=float, default=14.0, help="how long it listens and reads, in all")
    parser.add_argument("--pieces", type=int, default=3000, help="new pieces of words it learns from what it hears")
    parser.add_argument("--width", type=int, default=2)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch", type=int, default=16, help="passages of what it hears, a step")
    parser.add_argument("--talk", type=int, default=3, help="one step in this many is practice talking about itself")
    parser.add_argument("--every", type=int, default=400)
    parser.add_argument("--settle", type=float, default=0.0, help="hours, after that, of mostly practising talking")
    parser.add_argument("--settle-lr", type=float, default=5e-5, help="learning rate while it settles")
    parser.add_argument(
        "--saying",
        type=int,
        default=1,
        help="while it settles, of six moments practised, how many are saying its state",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--pack", metavar="STEP", help="save the snapshot at this step as the cortex Haven starts with")
    parser.add_argument("--out", default=str(starter.FOLDER), help="where --pack saves it")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    work = Path(args.work)
    heard = hearing.load(Path(args.corpus))
    if args.pack:
        pack(work, heard, args.pack, Path(args.out))
        return
    if not (work / "cortex" / "cortex.pt").exists():
        create(work, heard, args.pieces, args.width, args.layers)
    torch.manual_seed(0)
    talk.TALES[:] = story_pool(Path(args.corpus).parent, heard)  # (stories to be asked for, as it practises)
    trainer = Trainer(work, device="cpu", scale={"batch": 32, "lr": args.lr, "bf16": True})
    trainer.progress["heard"] = {
        "words": sum(heard.words().values()),
        "speech": heard.words()["speech"],
        "books": len(heard.books),
    }
    record = trainer.progress["hearing"]
    trainer.rng = random.Random(record["steps"])
    ids = stream(trainer, heard)
    moments = trainer.grounded()
    items = {
        "conversation": conversation_items(moments["held"], 300, seed=1),
        "self-report": balanced(moments["held"], 60),
        "understanding": moments["held"][:200],
    }
    rng = random.Random(5)
    held = {kind: rng.sample(passages, min(12, len(passages))) for kind, passages in heard.held.items() if passages}
    print(
        f"{len(talk.TALES):,} bits of stories to be asked for; "
        f"{trainer.model.parameters_count() / 1e6:.1f}M connections; {len(ids) / 1e6:.1f}M tokens to hear "
        f"({sum(heard.words().values()) / 1e6:.1f}M words); step {record['steps']:,}",
        flush=True,
    )
    if not record["tests"]:
        record["tests"].append({"step": 0, **test(trainer, items, held)})
        print("before:", record["tests"][-1], flush=True)
    budget, settle = 3600 * args.hours, 3600 * args.settle
    snapshots = work / "snapshots"
    snapshots.mkdir(exist_ok=True)
    while record["seconds"] < budget or record.get("settled", 0.0) < settle:
        began = time.monotonic()
        n = record["steps"]
        settling = record["seconds"] >= budget  # (then three steps in four are practice talking, at a low rate)
        if settling:
            lr, talking = args.settle_lr, n % 4 != 3
        else:
            done = record["seconds"] / budget
            lr = args.lr * min(1.0, (n + 1) / 300) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * done)))
            talking = n % args.talk == args.talk - 1
        for group in trainer.optimizer.param_groups:
            group["lr"] = lr
        trainer.model.train()
        if talking:
            loss, tokens = trainer._grounded_loss(moments["train"], args.saying if settling else 1), 0
        else:
            loss, tokens = text_loss(trainer, ids, args.batch)
        trainer.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
        trainer.optimizer.step()
        record["steps"] += 1
        record["tokens"] += tokens
        trainer.progress["steps"] += 1
        spent = time.monotonic() - began
        if settling:
            record["settled"] = record.get("settled", 0.0) + spent
        else:
            record["seconds"] += spent
        if record["steps"] % 50 == 0:
            print(
                f"step {record['steps']:>6,}  loss {float(loss.detach()):.3f}  lr {lr:.2e}  "
                f"heard {record['tokens'] / 1e6:.1f}M tokens  ({record['seconds'] / 3600:.2f} h"
                + (f", settling {record['settled'] / 3600:.2f} h)" if settling else ")"),
                flush=True,
            )
        ended = record["seconds"] >= budget if not settling else record["settled"] >= settle
        if record["steps"] % args.every == 0 or ended:
            results = test(trainer, items, held)
            record["tests"].append({"step": record["steps"], **results})
            print(f"step {record['steps']:,}:", results, flush=True)
            trainer.save()
            half = {k: (v.half() if v.is_floating_point() else v) for k, v in trainer.model.state_dict().items()}
            torch.save({"config": asdict(trainer.model.cfg), "model": half}, snapshots / f"step-{record['steps']}.pt")
    print("done:", record["tests"][-1], flush=True)


if __name__ == "__main__":
    main()
