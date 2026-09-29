"""The language cortex Haven starts life with: its own, trained from scratch on moments of its simulated lives.

It comes with Haven (haven/cortex/starter/), so Haven can talk about itself from the
start, in words it learned for its own states. It keeps learning from there: `haven
learn` has it read stories, books and encyclopedias, level by level.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

FOLDER = Path(__file__).parent / "starter"
FILES = ("cortex.pt", "tokenizer.json", "progress.json")


def available() -> bool:
    return all((FOLDER / name).exists() for name in FILES)


def install(root: Path) -> bool:
    """Give a Haven that has no language cortex yet the one it starts with. Returns whether it did."""
    target = Path(root) / "cortex"
    if (target / "cortex.pt").exists() or not available():
        return False
    target.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copyfile(FOLDER / name, target / name)
    return True


def pack(trainer, folder: Path = FOLDER) -> None:
    """Save a trainer's cortex as the one Haven starts with: its weights at half size, without the optimizer."""
    import torch

    folder.mkdir(parents=True, exist_ok=True)
    model = trainer.model
    trainer.tok.save(folder / "tokenizer.json")
    weights = {k: (v.half() if v.is_floating_point() else v).cpu() for k, v in model.state_dict().items()}
    progress = dict(trainer.progress)
    torch.save(
        {"config": dict(vars(model.cfg)), "model": weights, "optimizer": None, "progress": progress},
        folder / "cortex.pt",
    )
    (folder / "progress.json").write_text(json.dumps(progress, indent=1))
