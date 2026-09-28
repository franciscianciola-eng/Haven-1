"""Haven's own tokenizer: byte-level byte-pair encoding, learned from what it reads.

It starts from the 256 possible bytes, so it can read any text, and learns merges of
frequent pairs into larger pieces. It can keep growing: when Haven starts a new level of
reading, new merges are learned from the new material and appended, so earlier pieces keep
their numbers and the language cortex only has to learn the new ones.
"""

from __future__ import annotations

import heapq
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path

SPECIALS = ("<|end|>", "<|you|>", "<|haven|>", "<|think|>")
END, YOU, HAVEN, THINK = range(256, 256 + len(SPECIALS))
BASE = 256 + len(SPECIALS)
PIECES = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]{1,3}| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")
_SPECIAL = re.compile("(" + "|".join(re.escape(s) for s in SPECIALS) + ")")


class Tokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges: list[tuple[int, int]] = []
        self.ranks: dict[tuple[int, int], int] = {}
        self.pieces: list[bytes] = [bytes([i]) for i in range(256)] + [s.encode() for s in SPECIALS]
        self._cache: dict[str, list[int]] = {}
        for pair in merges or []:
            self._add(tuple(pair))

    def __len__(self) -> int:
        return len(self.pieces)

    def _add(self, pair: tuple[int, int]) -> int:
        new = len(self.pieces)
        self.merges.append(pair)
        self.ranks[pair] = len(self.merges) - 1
        self.pieces.append(self.pieces[pair[0]] + self.pieces[pair[1]])
        return new

    # --- learning -------------------------------------------------------------------------

    def learn(self, texts: Iterable[str], new_pieces: int, min_count: int = 3) -> int:
        """Learn up to `new_pieces` more merges from these texts. Returns how many were added."""
        counts = Counter()
        for text in texts:
            for part in _SPECIAL.split(text):
                if part and part not in SPECIALS:
                    counts.update(PIECES.findall(part))
        words = [self._encode_piece(w) for w in counts]
        freqs = list(counts.values())
        pair_counts: Counter = Counter()
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for i, ids in enumerate(words):
            for pair in pairwise(ids):
                pair_counts[pair] += freqs[i]
                where[pair].add(i)
        heap = [(-c, pair) for pair, c in pair_counts.items()]
        heapq.heapify(heap)
        added = 0
        while added < new_pieces and heap:
            negative, pair = heapq.heappop(heap)
            if -negative != pair_counts.get(pair, 0):
                continue  # stale entry
            if -negative < min_count:
                break
            new = self._add(pair)
            added += 1
            changed: Counter = Counter()
            for i in list(where.pop(pair, ())):
                ids, f = words[i], freqs[i]
                merged, j = [], 0
                while j < len(ids):
                    if j + 1 < len(ids) and (ids[j], ids[j + 1]) == pair:
                        merged.append(new)
                        j += 2
                    else:
                        merged.append(ids[j])
                        j += 1
                for old in pairwise(ids):
                    changed[old] -= f
                for fresh in pairwise(merged):
                    changed[fresh] += f
                    where[fresh].add(i)
                words[i] = merged
            for p, delta in changed.items():
                if delta:
                    pair_counts[p] += delta
                    if pair_counts[p] <= 0:
                        pair_counts.pop(p, None)
                    else:
                        heapq.heappush(heap, (-pair_counts[p], p))
            pair_counts.pop(pair, None)
        self._cache.clear()
        return added

    # --- using --------------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for part in _SPECIAL.split(text):
            if not part:
                continue
            if part in SPECIALS:
                ids.append(256 + SPECIALS.index(part))
                continue
            for word in PIECES.findall(part):
                cached = self._cache.get(word)
                if cached is None:
                    cached = self._encode_piece(word)
                    if len(self._cache) > 200_000:
                        self._cache.clear()
                    self._cache[word] = cached
                ids.extend(cached)
        return ids

    def _encode_piece(self, word: str) -> list[int]:
        ids = list(word.encode("utf-8"))
        while len(ids) > 1:
            best, rank = None, None
            for pair in pairwise(ids):
                r = self.ranks.get(pair)
                if r is not None and (rank is None or r < rank):
                    best, rank = pair, r
            if best is None:
                break
            new = BASE + rank
            merged, j = [], 0
            while j < len(ids):
                if j + 1 < len(ids) and (ids[j], ids[j + 1]) == best:
                    merged.append(new)
                    j += 2
                else:
                    merged.append(ids[j])
                    j += 1
            ids = merged
        return ids

    def decode(self, ids: Iterable[int], specials: bool = False) -> str:
        out = bytearray()
        for i in ids:
            if 256 <= i < BASE and not specials:
                continue
            if 0 <= i < len(self.pieces):
                out += self.pieces[i]
        return out.decode("utf-8", errors="replace")

    # --- saving -------------------------------------------------------------------------

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"merges": self.merges}))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> Tokenizer:
        return cls([tuple(pair) for pair in json.loads(path.read_text())["merges"]])
