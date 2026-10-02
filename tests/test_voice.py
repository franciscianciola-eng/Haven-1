"""Haven's voice in the terminal: its words, spoken by the computer's own speech voice."""

import sys
import time

from haven import voice
from haven.voice import Voice, speaker


def test_it_finds_the_computers_voice(monkeypatch):
    monkeypatch.setattr(voice.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(voice.shutil, "which", lambda name: f"/usr/bin/{name}" if name == "say" else None)
    assert speaker() == ["say", "-r", "185"]
    monkeypatch.setattr(voice.platform, "system", lambda: "Windows")
    monkeypatch.setattr(voice.shutil, "which", lambda name: "C:/powershell.exe" if name == "powershell" else None)
    command = speaker()
    assert command[0] == "C:/powershell.exe" and "SpeechSynthesizer" in command[-1]
    monkeypatch.setattr(voice.platform, "system", lambda: "Linux")
    monkeypatch.setattr(voice.shutil, "which", lambda name: "/usr/bin/espeak-ng" if name == "espeak-ng" else None)
    assert speaker(1.2)[:3] == ["espeak-ng", "-s", "198"]
    monkeypatch.setattr(voice.shutil, "which", lambda name: None)
    assert speaker() is None
    assert not Voice().available  # (no voice: it just types)


def test_it_says_one_thing_after_another(monkeypatch, tmp_path):
    heard = tmp_path / "heard.txt"
    fake = [sys.executable, "-c", f"import sys; open({str(heard)!r}, 'a').write(sys.stdin.read() + '|')"]
    monkeypatch.setattr(voice, "speaker", lambda rate=1.0: fake)
    v = Voice()
    v.say("Hi! I'm Haven.")
    v.say("I read about volcanoes.")
    v.say("   ")  # (nothing to say)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and (not heard.exists() or heard.read_text().count("|") < 2):
        time.sleep(0.05)
    v.close()
    assert heard.read_text() == "Hi! I'm Haven.|I read about volcanoes.|"
