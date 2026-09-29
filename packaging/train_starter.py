"""Trains the language cortex Haven starts with (haven/cortex/starter/), from scratch.

    python packaging/train_starter.py --minutes 90

It works through the curriculum's "talking about itself" level on moments from Haven's
own simulated lives (no reading from the internet is needed), keeps the version that
answers held-out questions best, and saves it into the package.
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from haven.cortex import starter
from haven.cortex.curriculum import LEVELS, ScratchReader
from haven.cortex.train import Trainer, describe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=90)
    parser.add_argument("--size", default="small")
    parser.add_argument("--work", default=str(ROOT / "dist" / "starter-work"))
    parser.add_argument("--every", type=int, default=500)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument(
        "--lr", type=float, help="learning rate (default: the size's own; lower it to carry on training)"
    )
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    scale = {"pieces": 4.0, "batch": 32, **({"lr": args.lr} if args.lr else {})}
    trainer = Trainer(Path(args.work), size=args.size, device="cpu", scale=scale)  # carries on from the work folder
    trainer.progress["level"] = 2
    level = LEVELS[1]
    data = trainer.prepare(level)
    record = trainer.progress["levels"]["2"]
    record["status"] = "studying"
    best, best_state = -1.0, None
    deadline = time.monotonic() + 60 * args.minutes
    started = time.monotonic()
    lr = trainer.scale["lr"]
    while time.monotonic() < deadline:
        left = (deadline - time.monotonic()) / (60 * args.minutes)
        trainer.scale["lr"] = lr * min(1.0, 0.1 + left / 0.4)  # settle down over the last part
        loss = trainer.step(level, data, record["steps"])
        record["steps"] += 1
        trainer.progress["steps"] += 1
        if record["steps"] % args.every == 0:
            results = trainer.evaluate(level, data)
            record["tests"] = results
            score = results.get("conversation", 0) + 0.5 * results.get("self-report", 0)
            print(
                f"step {record['steps']:>6,}  loss {loss:.3f}  {describe(level, results)}  "
                f"({(time.monotonic() - started) / 60:.0f} min)",
                flush=True,
            )
            if score > best:
                best, best_state = score, copy.deepcopy(trainer.model.state_dict())
                trainer.save()
    if best_state is not None:
        trainer.model.load_state_dict(best_state)
    results = trainer.evaluate(level, data)
    record["tests"], record["status"] = (
        results,
        "passed"
        if all(
            results.get(t, 0) >= mark if d == ">=" else results.get(t, 9) <= mark
            for t, (d, mark) in level.tests.items()
        )
        else "studied",
    )
    print("kept:", describe(level, results))
    reader = ScratchReader(trainer.model, trainer.tok)
    rng = random.Random(1)
    for m in rng.sample(data["items"]["conversation"], 12):
        print(
            f"  you: {m['question']}\n  haven: {reader.reply(m['state'], m['notes'], m['question'])}   (its state says: {m['answer']})"
        )
    m = data["grounded"][-1]
    for question in ("hi", "how are you", "where are you?", "what's your name", "hey haven, are you hungry?", "bye"):
        print(f"  you: {question}\n  haven: {reader.reply(m['state'], m['notes'], question)}")
    trainer.progress["level"] = 1  # at home, it goes on to read its first stories, then the rest
    starter.pack(trainer)
    print("saved to", starter.FOLDER)


if __name__ == "__main__":
    main()
