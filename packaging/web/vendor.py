"""Haven in the browser: copies what the page runs on into docs/vendor (from tests/web/node_modules: cd tests/web,
npm install first). Kept in the repository, so the page needs nothing from anywhere else to run.

    python packaging/web/vendor.py

- ONNX Runtime Web (MIT license, Microsoft), which runs Haven's cortex: on the graphics card (ort.webgpu), or the
  processor (ort.wasm).
- Pyodide (Mozilla Public License 2.0), Python for the browser, which runs Haven's conversation code; with it, Python's
  standard library (Python Software Foundation License).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULES = ROOT / "tests" / "web" / "node_modules"
OUT = ROOT / "docs" / "vendor"
FILES = {
    "ort": (
        "onnxruntime-web/dist",
        [
            "ort.webgpu.min.mjs",
            "ort-wasm-simd-threaded.asyncify.mjs",
            "ort-wasm-simd-threaded.asyncify.wasm",
            "ort.wasm.min.mjs",
            "ort-wasm-simd-threaded.mjs",
            "ort-wasm-simd-threaded.wasm",
        ],
    ),
    "pyodide": (
        "pyodide",
        ["pyodide.mjs", "pyodide.asm.mjs", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json"],
    ),
}
NOTICE = """What Haven's page runs on, as published on npm:

ort/      ONNX Runtime Web {ort}, (c) Microsoft Corporation, MIT License:
          https://github.com/microsoft/onnxruntime/blob/main/LICENSE
pyodide/  Pyodide {pyodide}, Mozilla Public License 2.0: https://github.com/pyodide/pyodide/blob/main/LICENSE
          (with Python's standard library, Python Software Foundation License: https://docs.python.org/3/license.html)
"""


def main() -> None:
    versions = {}
    for name, (folder, files) in FILES.items():
        source = MODULES / folder
        package = json.loads(
            (source.parent / "package.json" if folder.endswith("dist") else source / "package.json").read_text()
        )
        versions[name] = package["version"]
        target = OUT / name
        target.mkdir(parents=True, exist_ok=True)
        for old in target.iterdir():
            if old.name not in files:
                old.unlink()
        for file in files:
            shutil.copyfile(source / file, target / file)
    (OUT / "NOTICE.txt").write_text(NOTICE.format(**versions))
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"docs/vendor: ONNX Runtime Web {versions['ort']}, Pyodide {versions['pyodide']} ({size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
