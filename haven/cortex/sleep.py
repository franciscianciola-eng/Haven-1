"""Learning in its sleep: going over moments of its own day, and keeping what it learned only if it still talks as
well as it did.

While it's awake, Haven notes down moments of its life now and then: its state, the words for it, what it knows, and
what it would truthfully answer (worked out from its own state, as when it first learned to talk). While it sleeps
(at most every so often, in real time), a copy of its language cortex practises conversations about those moments:
about its own things, its own places, its own firsts, the people it has met and what it has read. Every other one is
about a moment of another life instead, so that it doesn't forget how to talk about others. Then the copy and the
cortex it has take the same two tests: questions about moments of its day it didn't practise, and a fixed set about
other lives. It keeps the copy only if it does at least as well on its own day, and no worse on other lives (so
learning about its own life can't cost it what it knew). Either way, how it went is written down in
cortex/nights.jsonl.

It also goes over what it heard since it last slept, as a small child's brain does: the bedtime story it was read (and
a little of the book from before, so it doesn't forget), and what people said to it. Then it's tested on how the story
goes on, the part it hasn't heard yet, and it keeps what it learned only if it follows that at least as well as before.
"""

from __future__ import annotations

import copy
import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from .tokenizer import END, HAVEN, THINK, YOU

STEPS = 120  # practice steps a night, half on its own day and half going over another life
BATCH = 8
LR = 1e-4  # (sweeps: without going over another life too, practice cost it on other lives, and was thrown away)
HELD = 4  # one moment in this many is kept back, to test on
TOLERANCE = 0.02  # how much worse on other lives still counts as no worse (the tests are small)
LISTENING = 3  # with things it heard to go over, at most one practice step in this many is hearing them again...
PASSES = 2  # ...and it hears each of them about twice (more, and a cortex this size learns them by heart)
SIZED = 5.3e6  # connections of the cortex the learning rate was found for (a bigger one learns more gently)
FOLLOWING = 0.01  # how much worse at following a story (bits per byte) still counts as no worse


def practice_loss(model, tok, moments: list[dict], rng: random.Random, rehearse: list[dict] = ()) -> torch.Tensor:
    """Saying what it's experiencing, and conversations about moments of its day (only its own words are learned).

    With `rehearse`, every other example is a moment of another life, so it doesn't forget how to talk about others.
    """
    from .train import conversation_ids

    seqs, marks, states = [], [], []
    limit = model.cfg.context + 1
    for i in range(BATCH):
        pool = rehearse if rehearse and i % 2 else moments
        m = pool[rng.randrange(len(pool))]
        if i % 4 == 0:
            seq = [HAVEN, *tok.encode(m["text"]), END]
            mark = [False] + [True] * (len(seq) - 1)
        else:
            seq, mark = conversation_ids(tok, m, rng)
        seqs.append(seq[:limit])  # (a long conversation loses its end, not what came to mind first)
        marks.append(mark[:limit])
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


def listening_loss(model, tok, passages: list[str], people: list[str], rng: random.Random) -> torch.Tensor:
    """Hearing again what it heard: stories, each piece of them expected from what came before, and what people said
    to it (as they say it to it)."""
    limit = model.cfg.context + 1
    seqs = []
    for i in range(BATCH // 2):
        if people and (i % 2 or not passages):
            seq = [YOU, *tok.encode(rng.choice(people)), END]
        else:
            ids = [END, *tok.encode(rng.choice(passages)), END]
            start = rng.randrange(0, max(1, len(ids) - limit + 1))
            seq = ids[start : start + limit]
        seqs.append(seq[-limit:])
    device = model.embed.weight.device
    length = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), length), END, dtype=torch.long)
    targets = torch.full((len(seqs), length - 1), -100, dtype=torch.long)
    for i, seq in enumerate(seqs):
        ids[i, : len(seq)] = torch.tensor(seq)
        targets[i, : len(seq) - 1] = torch.tensor(seq[1:])
    logits, _ = model(ids[:, :-1].to(device))
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), targets.to(device).reshape(-1), ignore_index=-100
    )


@torch.no_grad()
def following(model, tok, passage: str) -> float:
    """How well it follows a passage it hasn't heard: bits per byte (lower is better)."""
    model.eval()
    ids = [END, *tok.encode(passage)][: model.cfg.context + 1]
    logits, _ = model(torch.tensor([ids[:-1]], device=model.embed.weight.device))
    nll = float(F.cross_entropy(logits[0].float(), torch.tensor(ids[1:], device=logits.device), reduction="sum"))
    return nll / max(len(tok.decode(ids[1:]).encode("utf-8")), 1) / math.log(2)


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
            asked = [] if item["question"] is None else [YOU, *tok.encode(item["question"])]  # (None: it spoke up)
            prompt = [THINK, *tok.encode(item["notes"]), *asked, HAVEN]
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
    rehearse: list[dict] = (),
    floor: float | None = None,
    heard: list[str] = (),
    people: list[str] = (),
    upcoming: str | None = None,
):
    """A night's practice on a copy of the cortex. Returns the copy if it's worth keeping (else None), and a report.

    `floor`: how it did on other lives the first night (so that small losses can't add up over many nights). `heard`:
    passages it heard (and a little from before), to go over; `people`: what people said to it; `upcoming`: how the
    story it's hearing goes on, to test how well it follows it.
    """
    from .train import conversation_items

    steps = STEPS if steps is None else steps
    held = day[::HELD]
    practice = [m for i, m in enumerate(day) if i % HELD] or day
    own = conversation_items(held * 2, 2 * len(held), seed=rng.randrange(10**6))  # two questions at each
    before = {"own day": exam(model, tok, own), "other lives": exam(model, tok, others)}
    if upcoming:
        before["following"] = following(model, tok, upcoming)
    student = copy.deepcopy(model)
    student.train()
    rate = LR * min(1.0, SIZED / model.parameters_count()) if lr is None else lr  # (ten times bigger: a tenth)
    optimizer = torch.optim.AdamW(student.parameters(), lr=rate, weight_decay=0.0)
    losses = []
    pieces = sum(len(tok.encode(text)) + 2 for text in (*heard, *people))
    listens = min(steps // LISTENING, math.ceil(PASSES * pieces / ((BATCH // 2) * (model.cfg.context + 1))))
    when = {int((i + 0.5) * steps / listens) for i in range(listens)} if listens else set()  # (spread out)
    for step in range(steps):
        if step in when:
            loss = listening_loss(student, tok, list(heard), list(people), rng)
        else:
            loss = practice_loss(student, tok, practice, rng, rehearse)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    student.eval()
    after = {"own day": exam(student, tok, own), "other lives": exam(student, tok, others)}
    if upcoming:
        after["following"] = following(student, tok, upcoming)
    worse = min(before["other lives"], floor if floor is not None else 1.0) - TOLERANCE  # not worse, night after night
    kept = after["own day"] >= before["own day"] and after["other lives"] >= worse
    kept = kept and after.get("following", 0.0) <= before.get("following", 0.0) + FOLLOWING
    report = {
        "moments": len(day),
        "heard": len(heard),
        "said to it": len(people),
        "steps": steps,
        "listening": len(when),
        "rate": rate,
        "loss": [round(float(np.mean(losses[:10])), 3), round(float(np.mean(losses[-10:])), 3)] if losses else None,
        "before": {k: round(v, 3) for k, v in before.items()},
        "after": {k: round(v, 3) for k, v in after.items()},
        "kept": kept,
    }
    return (student if kept else None), report


def rehearsal(seed: int = 12) -> list[dict]:
    """Moments of another life, to go over along with its own (not the life it's tested on)."""
    from .grounding import gather

    return gather(seed, days=0.6)


def other_lives(n: int = 80, seed: int = 11) -> list[dict]:
    """The fixed test about other lives: the same questions every night, at moments of a simulated life."""
    from .grounding import gather
    from .train import conversation_items

    return conversation_items(gather(seed, days=0.6), n, seed=seed)
