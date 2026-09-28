"""The language cortex: a transformer Haven trains from scratch, wired into its global workspace.

Before the words, it reads a few "workspace tokens": projections of what is in Haven's
global workspace, its body and feelings, its attention schema and its self-model. So
whatever it says is conditioned on what Haven is experiencing. After reading, its final
state is projected back into the workspace's format: the meaning of what it just read or
thought, which can then compete for the workspace like any other content. Words it reads
can in this way bring states to mind ("hungry" evokes hunger), and its own inner speech
enters the same bottleneck as everything else Haven is aware of.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..workspace import D

SLOTS = 4  # workspace tokens: focal content, body and feeling, attention schema, self-model

SIZES = {
    "tiny": {"d": 128, "layers": 4, "heads": 4, "context": 256},
    "small": {"d": 256, "layers": 6, "heads": 8, "context": 512},
    "medium": {"d": 512, "layers": 8, "heads": 8, "context": 768},
    "large": {"d": 768, "layers": 12, "heads": 12, "context": 1024},
}


@dataclass
class CortexConfig:
    vocab: int
    d: int = 256
    layers: int = 6
    heads: int = 8
    context: int = 512
    slots: int = SLOTS
    core: int = D
    dropout: float = 0.0

    @classmethod
    def sized(cls, size: str, vocab: int) -> CortexConfig:
        return cls(vocab=vocab, **SIZES[size])


class Block(nn.Module):
    def __init__(self, cfg: CortexConfig):
        super().__init__()
        self.heads = cfg.heads
        self.norm1 = nn.LayerNorm(cfg.d)
        self.qkv = nn.Linear(cfg.d, 3 * cfg.d, bias=False)
        self.proj = nn.Linear(cfg.d, cfg.d, bias=False)
        self.norm2 = nn.LayerNorm(cfg.d)
        self.mlp = nn.Sequential(nn.Linear(cfg.d, 4 * cfg.d), nn.GELU(), nn.Linear(4 * cfg.d, cfg.d))
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, past: tuple | None = None) -> tuple[torch.Tensor, tuple]:
        """With `past` (the keys and values of earlier positions), x is the next single position."""
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).split(d, dim=2)
        shape = (b, t, self.heads, d // self.heads)
        q, k, v = (z.view(shape).transpose(1, 2) for z in (q, k, v))
        if past is not None:
            k, v = torch.cat([past[0], k], dim=2), torch.cat([past[1], v], dim=2)
        dropout = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=past is None, dropout_p=dropout)
        x = x + self.proj(y.transpose(1, 2).reshape(b, t, d))
        return x + self.mlp(self.norm2(x)), (k, v)


class Cortex(nn.Module):
    def __init__(self, cfg: CortexConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d)
        self.position = nn.Embedding(cfg.slots + cfg.context, cfg.d)
        self.state_in = nn.Linear(cfg.core, cfg.d)  # workspace vectors become tokens it reads
        self.slot = nn.Parameter(torch.zeros(cfg.slots, cfg.d))
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.norm = nn.LayerNorm(cfg.d)
        self.head = nn.Linear(cfg.d, cfg.vocab, bias=False)
        self.head.weight = self.embed.weight  # reading and writing share one vocabulary of meanings
        self.state_out = nn.Linear(cfg.d, cfg.core)  # what it read or thought, as workspace content
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith(("proj.weight", "mlp.2.weight")):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.layers))

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, 0.0, 0.02)

    def parameters_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, tokens: torch.Tensor, state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """tokens (B, T); state (B, slots, core) or None for "just reading".

        Returns next-token logits (B, T, vocab) and the meaning after each token (B, T, core).
        """
        logits, meanings, _ = self.run(tokens, state)
        return logits, meanings

    def run(
        self, tokens: torch.Tensor, state: torch.Tensor | None = None, past: list | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, list]:
        """The forward pass, also returning the keys and values so generation can carry on from them."""
        b = tokens.shape[0]
        cfg = self.cfg
        if past is None:
            if state is None:
                state = torch.zeros(b, cfg.slots, cfg.core, device=tokens.device)
            x = torch.cat([self.state_in(state) + self.slot, self.embed(tokens)], dim=1)
            start, skip = 0, cfg.slots
        else:
            x = self.embed(tokens)
            start, skip = past[0][0].shape[2], 0
        x = x + self.position(torch.arange(start, start + x.shape[1], device=tokens.device))
        cache = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, None if past is None else past[i])
            cache.append(kv)
        x = self.norm(x)[:, skip:]
        return self.head(x), self.state_out(x), cache

    # --- growing ------------------------------------------------------------------------

    def grow_vocabulary(self, size: int, parts: list[tuple[int, int]]) -> None:
        """Add rows for new tokens, each starting as the average of the two pieces it's made of."""
        old = self.embed.weight.data
        if size <= old.shape[0]:
            return
        new = torch.empty(size, old.shape[1], device=old.device, dtype=old.dtype)
        new[: old.shape[0]] = old
        for i in range(old.shape[0], size):
            a, b = parts[i - old.shape[0]]
            new[i] = 0.5 * (new[a] + new[b])
        self.embed = nn.Embedding(size, old.shape[1]).to(old.device)
        self.embed.weight.data = new
        self.head.weight = self.embed.weight
        self.cfg.vocab = size

    # --- using --------------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: list[int],
        state: torch.Tensor | None = None,
        max_new: int = 60,
        temperature: float = 0.8,
        top_k: int = 40,
        stop: tuple[int, ...] = (),
        generator: torch.Generator | None = None,
    ) -> tuple[list[int], list[float]]:
        """Continue the prompt. Returns the new tokens and the log-probability of each."""
        device = self.embed.weight.device
        room = self.cfg.context - 1
        ids = list(prompt)[-max(room - max_new, 1) :]
        out, logprobs = [], []
        logits_all, _, past = self.run(torch.tensor([ids], device=device), state)
        logits = logits_all[0, -1].float()
        for step in range(max_new):
            if step:
                if len(ids) >= room:
                    break  # no room left to think in
                logits_all, _, past = self.run(torch.tensor([[ids[-1]]], device=device), past=past)
                logits = logits_all[0, -1].float()
            probs = F.softmax(logits / max(temperature, 1e-4), dim=-1)
            if top_k and top_k < probs.numel():
                cutoff = torch.topk(probs, top_k).values[-1]
                probs = torch.where(probs >= cutoff, probs, torch.zeros_like(probs))
                probs = probs / probs.sum()
            if temperature <= 1e-4:
                token = int(torch.argmax(probs))
            else:
                token = int(torch.multinomial(probs.cpu(), 1, generator=generator))
            logprobs.append(float(F.log_softmax(logits, dim=-1)[token]))
            if token in stop:
                break
            out.append(token)
            ids.append(token)
        return out, logprobs

    @torch.no_grad()
    def score(
        self, prefix: list[int], continuations: list[list[int]], state: torch.Tensor | None = None, batch: int = 16
    ) -> list[float]:
        """The log-probability of each continuation after the prefix (for multiple-choice tests)."""
        device = self.embed.weight.device
        scores: list[float] = []
        for start in range(0, len(continuations), batch):
            group = continuations[start : start + batch]
            seqs = [(prefix + cont)[-(self.cfg.context + 1) :] for cont in group]
            length = max(len(s) for s in seqs)
            ids = torch.zeros(len(seqs), length, dtype=torch.long, device=device)
            for i, seq in enumerate(seqs):
                ids[i, : len(seq)] = torch.tensor(seq, device=device)
            expanded = None if state is None else state.expand(len(seqs), -1, -1)
            logits, _ = self(ids[:, :-1], expanded)
            logp = F.log_softmax(logits.float(), dim=-1)
            for i, (seq, cont) in enumerate(zip(seqs, group, strict=True)):
                n = min(len(cont), len(seq) - 1)
                positions = torch.arange(len(seq) - 1 - n, len(seq) - 1, device=device)
                targets = ids[i, positions + 1]
                scores.append(float(logp[i, positions, targets].sum()))
        return scores

    @torch.no_grad()
    def meaning(self, ids: list[int], state: torch.Tensor | None = None) -> torch.Tensor:
        """What a piece of text comes to, in the workspace's format."""
        device = self.embed.weight.device
        _, meanings = self(torch.tensor([ids[-self.cfg.context :]], device=device), state)
        return meanings[0, -1]


def config_dict(cfg: CortexConfig) -> dict:
    return asdict(cfg)
