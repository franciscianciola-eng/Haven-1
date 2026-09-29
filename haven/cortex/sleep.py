"""Learning in its sleep: going over moments of its own day, and keeping what it learned only if it still talks as
well as it did.

While it's awake, Haven notes down moments of its life now and then: its state, the words for it, what it knows, and
what it would truthfully answer (worked out from its own state, as when it first learned to talk). While it sleeps
(at most every so often, in real time), a copy of its language cortex practises conversations about those moments:
about its own things, its own places, its own firsts and the people it has met. Then the copy and the cortex it has
take the same two tests: questions about moments of its day it didn't practise, and a fixed set about other lives.
It keeps the copy only if it does at least as well on its own day, and no worse on other lives (so learning about
its own life can't cost it what it knew). Either way, how it went is written down in cortex/nights.jsonl.
"""

from __future__ import annotations

import copy
import random

import numpy as np
import torch
import torch.nn.functional as F

from .tokenizer import END, HAVEN, THINK, YOU

STEPS = 60  # practice steps a night
BATCH = 8
LR = 1e-4  # (a sweep: 5e-5 helped less; 2e-4 helped its day more but cost it on other lives)
HELD = 4  # one moment in this many is kept back, to test on
TOLERANCE = 0.02  # how much worse on other lives still counts as no worse (the tests are small)


def practice_loss(model, tok, moments: list[dict], rng: random.Random) -> torch.Tensor:
    """Saying what it's experiencing, and conversations about moments of its day (only its own words are learned)."""
    from .train import conversation_ids

    seqs, marks, states = [], [], []
    limit = model.cfg.context + 1
    for i in range(BATCH):
        m = moments[rng.randrange(len(moments))]
        if i % 4 == 0:
            seq = [HAVEN, *tok.encode(m["text"]), END]
            mark = [False] + [True] * (len(seq) - 1)
        else:
            seq, mark = conversation_ids(tok, m, rng)
        seqs.append(seq[-limit:])
        marks.append(mark[-limit:])
        states.append(np.asarray(m["state"], dtype=np.float32))
    device = model.embed.weight.device
    length = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), length), END, dtype=torch.long)
    targets = torch.full((len(seqs), length - 1), -100, dtype=torch.long)
    for i, (seq, mark) in enumerate(zip(seqs, marks, strict=True)):
        ids[i, : len(seq)] = torch.tensor(seq)
        keep = torch.tensor(mark[1:])
        targets[i, : len(seq) - 1] = torch.where(keep, torch.tensor(seq[1:]), torch.tensor(-100))
    ids, targets = ids.to(device), targets.to(device)
    logits, _ = model(ids[:, :-1], torch.tensor(np.stack(states), device=device))
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)


@torch.no_grad()
def exam(model, tok, items: list[dict], batch: int = 16) -> float:
    """The share of questions it answers exactly as it should: every word of the answer, and then stopping, its most
    likely next word (which is what it says when it answers without hesitating)."""
    if not items:
        return 0.0
    model.eval()
    device = model.embed.weight.device
    right = 0
    for start in range(0, len(items), batch):
        chunk = items[start : start + batch]
        seqs, answers = [], []
        for item in chunk:
            prompt = [THINK, *tok.encode(item["notes"]), YOU, *tok.encode(item["question"]), HAVEN]
            answer = tok.encode(item["answer"])
            seq = (prompt + answer + [END])[-model.cfg.context :]
            seqs.append(seq)
            answers.append(answer)
        length = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), length), END, dtype=torch.long)
        for i, seq in enumerate(seqs):
            ids[i, : len(seq)] = torch.tensor(seq)
        state = torch.tensor(np.stack([np.asarray(i["state"], dtype=np.float32) for i in chunk]), device=device)
        guess = model(ids.to(device), state)[0].argmax(-1).cpu()
        for i, (seq, answer) in enumerate(zip(seqs, answers, strict=True)):
            first = len(seq) - len(answer) - 1  # where its answer starts
            said = guess[i, first - 1 : len(seq) - 2].tolist()
            right += said == answer and int(guess[i, len(seq) - 2]) in (END, YOU)
    return right / len(items)


def night(
    model,
    tok,
    day: list[dict],
    others: list[dict],
    rng: random.Random,
    steps: int | None = None,
    lr: float | None = None,
):
    """A night's practice on a copy of the cortex. Returns the copy if it's worth keeping (else None), and a report."""
    from .train import conversation_items

    steps = STEPS if steps is None else steps
    held = day[::HELD]
    practice = [m for i, m in enumerate(day) if i % HELD] or day
    own = conversation_items(held * 2, 2 * len(held), seed=rng.randrange(10**6))  # two questions at each
    before = {"own day": exam(model, tok, own), "other lives": exam(model, tok, others)}
    student = copy.deepcopy(model)
    student.train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR if lr is None else lr, weight_decay=0.0)
    losses = []
    for _ in range(steps):
        loss = practice_loss(student, tok, practice, rng)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    student.eval()
    after = {"own day": exam(student, tok, own), "other lives": exam(student, tok, others)}
    kept = after["own day"] >= before["own day"] and after["other lives"] >= before["other lives"] - TOLERANCE
    report = {
        "moments": len(day),
        "steps": steps,
        "loss": [round(float(np.mean(losses[:10])), 3), round(float(np.mean(losses[-10:])), 3)] if losses else None,
        "before": {k: round(v, 3) for k, v in before.items()},
        "after": {k: round(v, 3) for k, v in after.items()},
        "kept": kept,
    }
    return (student if kept else None), report


def other_lives(n: int = 80, seed: int = 11) -> list[dict]:
    """The fixed test about other lives: the same questions every night, at moments of a simulated life."""
    from .grounding import gather
    from .train import conversation_items

    return conversation_items(gather(seed, days=0.6), n, seed=seed)
