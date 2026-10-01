"""Builds dist/Haven-for-Windows.zip: Haven's folder, to unzip and start with a double-click.

    python packaging/make_zip.py

It holds what's in the repository (not the tests or the packaging, nor its language
cortex, which Haven downloads from GitHub the first time it runs), in a folder called
Haven, with `Start Haven.bat` (Windows), `Start Haven.command` (Mac) and `start-haven.sh`
(Linux) at the top, and `READ ME FIRST.txt` beside them. Line endings are as the
repository keeps them for each system (see .gitattributes).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from haven.cortex.starter import source

LEAVE_OUT = ("tests/", "packaging/", ".git")
CORTEX = "haven/cortex/starter/cortex.pt"  # (left out, to keep the download small: Haven gets it from GitHub)


def build(root: Path, out: Path) -> Path:
    listed = subprocess.run(["git", "ls-files"], cwd=root, check=True, capture_output=True, text=True).stdout
    files = [f for f in listed.splitlines() if not f.startswith(LEAVE_OUT) and (root / f).is_file() and f != CORTEX]
    out.parent.mkdir(parents=True, exist_ok=True)
    moment = time.localtime()[:6]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            info = zipfile.ZipInfo(f"Haven/{f}", moment)
            info.create_system = 3  # (made on Unix: so the launchers stay runnable on a Mac or Linux)
            runnable = f.endswith((".command", ".sh"))
            info.external_attr = (0o100755 if runnable else 0o100644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, (root / f).read_bytes())
        info = zipfile.ZipInfo("Haven/" + CORTEX.replace("cortex.pt", "cortex.json"), moment)
        info.create_system, info.external_attr, info.compress_type = 3, 0o100644 << 16, zipfile.ZIP_DEFLATED
        z.writestr(info, json.dumps(source(root), indent=1))  # (where Haven gets its cortex, the first time)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT), help="the checkout to pack")
    parser.add_argument("--out", help="where to write the zip (default: dist/Haven-for-Windows.zip there)")
    args = parser.parse_args()
    root = Path(args.root)
    print("built", build(root, Path(args.out) if args.out else root / "dist" / "Haven-for-Windows.zip"))
