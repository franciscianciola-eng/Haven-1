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
import zlib
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from haven.cortex import encyclopedia, hearing, starter, talk
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


HELD_WIKI = 50  # one encyclopedia article in this many is held out, to test on


def held_article(title: str) -> bool:
    return zlib.crc32(title.encode()) % HELD_WIKI == 0


def wiki_articles(folder: Path):
    """(title, text) of every article of the encyclopedia downloaded to this folder (see encyclopedia.py)."""
    for path in sorted(folder.glob("*tfrecord*")):
        with path.open("rb") as f:
            yield from encyclopedia.articles(f)


def wiki_entries(folder: Path) -> tuple[list, list]:
    """What it reads in the encyclopedia, to practise telling: entries of the articles (to learn from, held out)."""
    learn, held = [], []
    for title, text in wiki_articles(folder):
        entry = talk.entry_of(title, text)
        if entry is not None:
            (held if held_article(title) else learn).append(entry)
    return learn, held


def wiki_stream(trainer: Trainer, folder: Path, most: int) -> np.ndarray:
    """The start of every article it doesn't hold out, as one long stream of tokens (made once)."""
    path = trainer.dir / "tokens" / "wiki.npy"
    if path.exists():
        return np.load(path)
    ids: list[int] = []
    for title, text in wiki_articles(folder):
        paragraphs = encyclopedia.prose(text, 3)
        if held_article(title) or not paragraphs:
            continue
        ids += trainer.tok.encode(title + "\n\n" + "\n\n".join(paragraphs))
        ids.append(END)
        if len(ids) >= most:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.array(ids, dtype=np.int32))
    return np.load(path)


def wiki_held(folder: Path, n: int = 12, seed: int = 5) -> list[str]:
    """Held-out articles' first paragraphs, to see how well it reads an encyclopedia it hasn't learned from."""
    found = [
        title + "\n\n" + "\n\n".join(encyclopedia.prose(text, 2))
        for title, text in wiki_articles(folder)
        if held_article(title) and len(encyclopedia.prose(text, 2)) == 2
    ]
    return random.Random(seed).sample(found, min(n, len(found)))


def reading_items(moments: list[dict], entries: list, n: int, seed: int = 0) -> list[dict]:
    """Held-out questions about held-out articles, with what it read in mind, and what it should say."""
    rng = random.Random(seed)
    kept, share = list(talk.ENCYCLOPEDIA), talk.KNOWLEDGE_TURNS
    talk.ENCYCLOPEDIA[:], talk.KNOWLEDGE_TURNS = entries, 1.0
    items = []
    try:
        while len(items) < n:
            m = rng.choice(moments)
            thought, turns = talk.conversation(m, rng, turns=1)
            turn = next((t for t in turns if t.kind.startswith("encyclopedia") and t.said), None)
            if turn is not None:
                items.append(
                    {
                        "state": m["state"],
                        "notes": thought,
                        "question": turn.said,
                        "answer": turn.answer,
                        "kind": "read",
                    }
                )
    finally:
        talk.ENCYCLOPEDIA[:], talk.KNOWLEDGE_TURNS = kept, share
    return items


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
    if items.get("reading"):
        results["reading"] = conversation(reader, items["reading"])
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
    parser.add_argument("--wiki", help="a folder with an encyclopedia's files of records, to read and practise telling")
    parser.add_argument("--knowledge", type=float, default=0.3, help="with --wiki, how often a turn asks about it")
    parser.add_argument("--wiki-share", type=float, default=0.7, help="with --wiki, how much of its reading is it")
    parser.add_argument("--wiki-tokens", type=int, default=45_000_000)
    parser.add_argument("--text-every", type=int, default=4, help="while it settles, one step in this many is reading")
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
    wiki = None
    if args.wiki:  # an encyclopedia to read, and to practise telling what it says
        folder = Path(args.wiki)
        learn, held_entries = wiki_entries(folder)
        talk.ENCYCLOPEDIA[:], talk.KNOWLEDGE_TURNS = learn, args.knowledge
        items["reading"] = reading_items(moments["held"], held_entries, 150, seed=2)
        wiki = wiki_stream(trainer, folder, args.wiki_tokens)
        held["wiki"] = wiki_held(folder)
        print(
            f"the encyclopedia: {len(learn):,} articles to practise telling ({len(held_entries):,} held out), "
            f"{len(wiki) / 1e6:.1f}M tokens to read",
            flush=True,
        )
    print(
        f"{len(talk.TALES):,} bits of stories to be asked for; "
        f"{trainer.model.parameters_count() / 1e6:.1f}M connections; {len(ids) / 1e6:.1f}M tokens to hear "
        f"({sum(heard.words().values()) / 1e6:.1f}M words); step {record['steps']:,}",
        flush=True,
    )
    if not record["tests"] or (wiki is not None and "reading" not in record["tests"][-1]):
        record["tests"].append({"step": record["steps"], **test(trainer, items, held)})
        print("before:", record["tests"][-1], flush=True)
    budget, settle = 3600 * args.hours, 3600 * args.settle
    snapshots = work / "snapshots"
    snapshots.mkdir(exist_ok=True)
    while record["seconds"] < budget or record.get("settled", 0.0) < settle:
        began = time.monotonic()
        n = record["steps"]
        settling = record["seconds"] >= budget  # (then three steps in four are practice talking, at a low rate)
        if settling:
            lr, talking = args.settle_lr, n % args.text_every != args.text_every - 1
        else:
            done = record["seconds"] / budget
            lr = args.lr * min(1.0, (n + 1) / 300) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * done)))
            talking = n % args.talk == args.talk - 1
        for group in trainer.optimizer.param_groups:
            group["lr"] = lr
        trainer.model.train()
        if talking:
            loss, tokens = trainer._grounded_loss(moments["train"], args.saying if settling else 1), 0
        elif wiki is not None and trainer.rng.random() < args.wiki_share:  # reading the encyclopedia
            loss, tokens = text_loss(trainer, wiki, args.batch)
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
