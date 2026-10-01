"""Haven's brain: spiking neurons modelled on real ones (neurons.py), wired into the regions a mammal's brain has
(brain.py).

wake() brings a Haven's brain back (or, the first time, grows one sized for the computer), from the folder it's kept
in; it needs torch, and returns None without it (Haven lives on without a brain of neurons then, as it used to).
"""

from __future__ import annotations

import os
from pathlib import Path

FOLDER = "brain"
SETTING = "HAVEN_BRAIN"  # "off", or a size ("small", "standard", "large", "huge") to grow one of that size


def wake(root: Path, seed: int):
    """Its brain, from root/brain, or a newborn one. None if it can't have one (no torch, or switched off)."""
    setting = os.environ.get(SETTING, "").strip().lower()
    if setting in ("off", "no", "none", "0"):
        return None
    try:
        from .brain import SIZES, Brain, fit
    except ImportError:
        return None
    folder = Path(root) / FOLDER
    if (folder / "state.pt").exists() and (folder / "cortex.u8").exists():
        return Brain.load(folder)
    return Brain(seed, setting if setting in SIZES else fit())
