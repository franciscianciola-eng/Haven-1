"""Builds haven/static/world3d.js: the 3D view of Haven's valley, bundled with three.js.

    python packaging/world3d/build.py

Needs Node.js. It installs three.js and esbuild into packaging/world3d/node_modules
(from the npm registry) the first time. The bundle is kept in the repository, so
people running Haven don't need any of this.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "haven" / "static"
THREE = "0.186.1"


def main() -> int:
    if not (HERE / "node_modules" / "three").exists() or not (HERE / "node_modules" / "esbuild").exists():
        subprocess.run(
            ["npm", "install", "--no-audit", "--no-fund", "--no-save", f"three@{THREE}", "esbuild@0.25"],
            cwd=HERE,
            check=True,
        )
    OUT.mkdir(exist_ok=True)
    subprocess.run(
        [
            str(HERE / "node_modules" / ".bin" / "esbuild"),
            str(HERE / "world3d.js"),
            "--bundle",
            "--minify",
            "--format=iife",
            "--global-name=HavenWorld",
            "--target=es2020",
            "--legal-comments=inline",
            f"--outfile={OUT / 'world3d.js'}",
        ],
        cwd=HERE,
        check=True,
    )
    shutil.copyfile(HERE / "node_modules" / "three" / "LICENSE", OUT / "three-LICENSE.txt")
    print(f"built {OUT / 'world3d.js'} ({(OUT / 'world3d.js').stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
