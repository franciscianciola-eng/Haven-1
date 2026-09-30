"""Downloads and prepares what Haven hears as it grows up (see haven/cortex/hearing.py), for training its cortex.

    python packaging/build_corpus.py

Keeps the downloads in dist/corpus/ (so it can be run again, and resumed), and writes
dist/corpus/heard.json: passages to learn from, and held-out passages for tests.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from haven.cortex import hearing
from haven.web import Web


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "dist" / "corpus"))
    args = parser.parse_args()
    out = Path(args.out)
    heard = hearing.gather(Web(delay=0.3), out)
    hearing.save(heard, out / "heard.json")
    train, held = heard.words("train"), heard.words("held")
    for kind in train:
        print(f"{kind:>9}: {train[kind]:>10,} words to learn from, {held.get(kind, 0):>8,} held out")
    print(f"{'in all':>9}: {sum(train.values()):>10,} words to learn from, {sum(held.values()):>8,} held out")
    print(f"{len(heard.books)} books")


if __name__ == "__main__":
    main()
