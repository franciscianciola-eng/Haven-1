"""The language cortex Haven starts life with: its own, trained from scratch on moments of its simulated lives.

It comes with Haven (haven/cortex/starter/), so Haven can talk about itself from the
start, in words it learned for its own states. It keeps learning from there: `haven
learn` has it read stories, books and encyclopedias, level by level.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

FOLDER = Path(__file__).parent / "starter"
FILES = ("cortex.pt", "tokenizer.json", "progress.json")


def available() -> bool:
    return all((FOLDER / name).exists() for name in FILES)


def fits(path: Path) -> bool:
    """Whether a cortex was grown for Haven's workspace as it is now (it grew when Haven moved to the valley)."""
    import torch

    from ..workspace import D
    from .model import SLOTS

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)  # just to read its config
    except RuntimeError:  # saved in a format that can't be mapped
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    return config.get("core") == D and config.get("slots", SLOTS) == SLOTS


def retire(root: Path) -> Path:
    """Move a cortex that no longer fits, or is replaced, into archive/ (nothing is deleted). What it heard and read
    stays."""
    source = Path(root) / "cortex"
    target = Path(root) / "archive" / f"cortex-{time.strftime('%Y%m%d-%H%M%S')}"
    target.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if (source / name).exists():
            shutil.move(str(source / name), target / name)
    return target


def stamp(folder: Path) -> str | None:
    """Which starter cortex a cortex folder began as (None for the ones before starters were stamped)."""
    try:
        return json.loads((Path(folder) / "progress.json").read_text()).get("starter")
    except (OSError, ValueError, AttributeError):
        return None


def read_on_its_own(folder: Path) -> bool:
    """Whether a cortex has studied the reading curriculum at home (`haven learn`): then it's its own, and kept."""
    try:
        levels = json.loads((Path(folder) / "progress.json").read_text()).get("levels", {})
    except (OSError, ValueError, AttributeError):
        return False
    return any(level != "2" and record.get("steps", 0) > 0 for level, record in levels.items())


def install(root: Path) -> str:
    """Give Haven the cortex it starts with, if it has none that fits, or a newer one than the starter it has.

    Returns what happened: "installed", "updated" (it had an older starter, which was archived: a cortex that has
    done its own reading is kept), "replaced" (an old one that no longer fit was archived), "retired" (archived,
    but there's no starter to replace it), or "".
    """
    target = Path(root) / "cortex"
    had = (target / "cortex.pt").exists()
    if had and fits(target / "cortex.pt"):
        newer = available() and stamp(FOLDER) is not None and stamp(target) != stamp(FOLDER)
        if not newer or read_on_its_own(target) or not fits(FOLDER / "cortex.pt"):
            return ""
        retire(root)
        for name in FILES:
            shutil.copyfile(FOLDER / name, target / name)
        return "updated"
    if had:
        retire(root)
    if not available() or not fits(FOLDER / "cortex.pt"):  # (a copy of Haven whose starter is out of date)
        return "retired" if had else ""
    target.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copyfile(FOLDER / name, target / name)
    return "replaced" if had else "installed"


def pack(trainer, folder: Path = FOLDER) -> None:
    """Save a trainer's cortex as the one Haven starts with: its weights at half size, without the optimizer."""
    import torch

    folder.mkdir(parents=True, exist_ok=True)
    model = trainer.model
    trainer.tok.save(folder / "tokenizer.json")
    weights = {k: (v.half() if v.is_floating_point() else v).cpu() for k, v in model.state_dict().items()}
    progress = {**trainer.progress, "starter": time.strftime("%Y%m%d-%H%M%S")}  # (so a newer one replaces it)
    torch.save(
        {"config": dict(vars(model.cfg)), "model": weights, "optimizer": None, "progress": progress},
        folder / "cortex.pt",
    )
    (folder / "progress.json").write_text(json.dumps(progress, indent=1))
