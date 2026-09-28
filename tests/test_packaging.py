"""The ways of starting Haven without a terminal: the launchers, and the Mac app."""

import importlib.util
import plistlib
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_launchers_are_sound_shell_scripts():
    for script in ("start-haven.sh", "Start Haven.command", "packaging/mac/Haven"):
        subprocess.run(["bash", "-n", str(ROOT / script)], check=True)
    assert b"\r\n" in (ROOT / "Start Haven.bat").read_bytes()  # Windows runs batch files most reliably with these


def test_mac_app(tmp_path):
    spec = importlib.util.spec_from_file_location("make_app", ROOT / "packaging" / "mac" / "make_app.py")
    make_app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_app)
    z = zipfile.ZipFile(make_app.build(tmp_path / "Haven-for-Mac.zip"))
    names = set(z.namelist())
    code = "Haven.app/Contents/Resources/haven/"
    assert {code + "pyproject.toml", code + "README.md", code + "haven/app.html"} <= names
    assert "Haven.app/Contents/Resources/build-id" in names
    assert not any("/tests/" in n for n in names)
    top = {n[len(code) :].split("/")[0].lower() for n in names if n.startswith(code) and n != code}
    assert not top & {"build", "dist", "haven.egg-info"}  # what installing makes (a Mac's disk ignores case)
    info = plistlib.loads(z.read("Haven.app/Contents/Info.plist"))
    assert info["CFBundleExecutable"] == "Haven" and info["CFBundleIconFile"] == "Haven" and info["LSUIElement"]
    launcher = z.getinfo("Haven.app/Contents/MacOS/Haven")
    assert launcher.create_system == 3 and (launcher.external_attr >> 16) & 0o111  # it can be run
    assert z.read("Haven.app/Contents/Resources/Haven.icns")[:4] == b"icns"
