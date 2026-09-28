"""Keeping Haven's life on disk between runs: arrays in an .npz file, everything else in JSON.

Saves are atomic (written beside the old file, then swapped in), so a crash mid-save
can't corrupt a life. Resetting archives a life instead of deleting it.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

HOME_ENV = "HAVEN_HOME"


def home() -> Path:
    return Path(os.environ.get(HOME_ENV) or Path.home() / ".haven").expanduser()


def split(tree, arrays: dict[str, np.ndarray], path: str = "") -> object:
    """Replace the numpy arrays in a nested structure with references to entries in `arrays`."""
    if isinstance(tree, np.ndarray):
        key = path.strip(".") or "root"
        arrays[key] = tree
        return {"__array__": key}
    if isinstance(tree, dict):
        return {str(k): split(v, arrays, f"{path}.{k}") for k, v in tree.items()}
    if isinstance(tree, list | tuple):
        return [split(v, arrays, f"{path}.{i}") for i, v in enumerate(tree)]
    if isinstance(tree, np.generic):
        return tree.item()
    return tree


def join(tree, arrays) -> object:
    if isinstance(tree, dict):
        if set(tree) == {"__array__"}:
            return arrays[tree["__array__"]]
        return {k: join(v, arrays) for k, v in tree.items()}
    if isinstance(tree, list):
        return [join(v, arrays) for v in tree]
    return tree


class Store:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else home()
        self.json_path = self.root / "mind.json"
        self.npz_path = self.root / "mind.npz"

    def exists(self) -> bool:
        return self.json_path.exists() and self.npz_path.exists()

    def save(self, state: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        tree = split(state, arrays)
        tmp_npz = self.npz_path.with_suffix(".tmp.npz")
        with open(tmp_npz, "wb") as f:
            np.savez_compressed(f, **arrays)
        tmp_json = self.json_path.with_suffix(".json.tmp")
        tmp_json.write_text(json.dumps({"saved": time.time(), "state": tree}))
        # The arrays go in first: a JSON file is only ever next to the arrays it refers to.
        os.replace(tmp_npz, self.npz_path)
        os.replace(tmp_json, self.json_path)

    def load(self) -> dict:
        data = json.loads(self.json_path.read_text())
        with np.load(self.npz_path, allow_pickle=False) as npz:
            arrays = {key: npz[key] for key in npz.files}
        return join(data["state"], arrays)

    def archive(self) -> Path | None:
        """Move the current life into archive/ (nothing is deleted)."""
        if not self.exists():
            return None
        stamp = time.strftime("%Y%m%d-%H%M%S")
        target = self.root / "archive" / f"life-{stamp}"
        target.mkdir(parents=True, exist_ok=True)
        for path in (self.json_path, self.npz_path):
            shutil.move(str(path), target / path.name)
        return target
