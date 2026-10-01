"""Builds Haven for the Mac: dist/Haven-for-Mac.zip, holding Haven.app.

    python packaging/mac/make_app.py

The app carries Haven's code. The first time it's opened it sets itself up in ~/.haven-app
(see the `Haven` script next to this file), and after that it wakes Haven up and opens its
window in the browser. It isn't signed by an Apple developer account, so the first time,
macOS asks the person to allow it in System Settings, Privacy & Security.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORTEX = "haven/cortex/starter/cortex.pt"  # (left out, to keep the download small: Haven gets it from GitHub)
ROOT = HERE.parents[1]
PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Haven</string>
  <key>CFBundleDisplayName</key><string>Haven</string>
  <key>CFBundleIdentifier</key><string>io.github.franciscianciola-eng.haven</string>
  <key>CFBundleVersion</key><string>{version}</string>
  <key>CFBundleShortVersionString</key><string>{version}</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>Haven</string>
  <key>CFBundleIconFile</key><string>Haven</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>LSUIElement</key><true/>
  <key>NSHumanReadableCopyright</key><string>Haven: an artificial creature, and its language cortex.</string>
</dict>
</plist>
"""


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def build(out: Path) -> Path:
    from haven import __version__
    from haven.cortex.starter import source

    listed = git("ls-files", "haven", "pyproject.toml", "README.md").splitlines()
    files = [f for f in listed if (ROOT / f).is_file() and f != CORTEX]  # (its cortex is got the first time it runs)
    stamp = git("rev-parse", "--short", "HEAD") + (
        "-changed" if git("status", "--porcelain", "haven", "pyproject.toml") else ""
    )
    stamp += f"-{int(time.time())}" if stamp.endswith("-changed") else ""
    out.parent.mkdir(parents=True, exist_ok=True)
    contents = "Haven.app/Contents/"
    moment = time.localtime()[:6]

    def entry(name: str, mode: int) -> zipfile.ZipInfo:
        info = zipfile.ZipInfo(name, moment)
        info.create_system = 3  # made on Unix: so the permissions below are honored
        info.external_attr = mode << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        return info

    with zipfile.ZipFile(out, "w") as z:
        folders = {"Haven.app/", contents, contents + "MacOS/", contents + "Resources/"}
        for f in files:
            parts = f.split("/")[:-1]
            for i in range(len(parts)):
                folders.add(contents + "Resources/haven/" + "/".join(parts[: i + 1]) + "/")
        folders.add(contents + "Resources/haven/")
        for folder in sorted(folders):
            z.writestr(entry(folder, 0o40755), b"")
        z.writestr(entry(contents + "Info.plist", 0o100644), PLIST.format(version=__version__))
        z.writestr(entry(contents + "PkgInfo", 0o100644), "APPL????")
        z.writestr(entry(contents + "MacOS/Haven", 0o100755), (HERE / "Haven").read_bytes())
        z.writestr(entry(contents + "Resources/Haven.icns", 0o100644), (HERE / "Haven.icns").read_bytes())
        for f in files:
            z.writestr(entry(contents + "Resources/haven/" + f, 0o100644), (ROOT / f).read_bytes())
        where = json.dumps(source(ROOT), indent=1).encode()
        z.writestr(entry(contents + "Resources/haven/" + CORTEX.replace("cortex.pt", "cortex.json"), 0o100644), where)
        # Beside Haven's code, not in it: a Mac's disk doesn't tell BUILD from the build/ folder installing makes.
        z.writestr(entry(contents + "Resources/build-id", 0o100644), stamp + "\n")
    return out


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    print("built", build(ROOT / "dist" / "Haven-for-Mac.zip"))
