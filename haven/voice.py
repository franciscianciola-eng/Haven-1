"""Haven's voice in the terminal: what it says, spoken aloud by the computer's own speech voice.

The words are Haven's own, from its cortex; only the sound is the computer's: `say` on a
Mac, the speech voice built into Windows, and espeak (or speech-dispatcher) on Linux.
Nothing is sent anywhere. In the app, the browser speaks instead, and can listen too.
"""

from __future__ import annotations

import contextlib
import platform
import queue
import shutil
import subprocess
import threading

WINDOWS = (  # the speech voice that comes with Windows, reading what to say from its input
    "[Console]::InputEncoding = [System.Text.Encoding]::UTF8; Add-Type -AssemblyName System.Speech; "
    "$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "$voice.Rate = {rate}; $voice.Speak([Console]::In.ReadToEnd())"
)


def speaker(rate: float = 1.0) -> list[str] | None:
    """The command that speaks text given on its input, on this computer, if it has one."""
    system = platform.system()
    if system == "Darwin" and shutil.which("say"):
        return ["say", "-r", str(round(185 * rate))]  # (it reads from its input when given no words)
    if system == "Windows":
        shell = shutil.which("powershell") or shutil.which("pwsh")
        if shell:
            return [shell, "-NoProfile", "-NonInteractive", "-Command", WINDOWS.format(rate=round(4 * (rate - 1)))]
    for name in ("espeak-ng", "espeak"):
        if shutil.which(name):
            return [name, "-s", str(round(165 * rate)), "--stdin"]
    if shutil.which("spd-say"):
        return ["spd-say", "-w", "-e"]  # (-e: what to say comes on its input)
    return None


class Voice:
    """Speaks one thing after another, in the background, so the conversation doesn't wait for it."""

    def __init__(self, rate: float = 1.0):
        self.command = speaker(rate)
        self._said: queue.Queue[str | None] = queue.Queue()
        self._now: subprocess.Popen | None = None
        self._thread = None
        if self.command:
            self._thread = threading.Thread(target=self._speak, name="haven-voice", daemon=True)
            self._thread.start()

    @property
    def available(self) -> bool:
        return self.command is not None

    def say(self, text: str) -> None:
        if self.command and text.strip():
            self._said.put(text)

    def hush(self) -> None:
        """Stop what it's saying now, and what it was going to say."""
        with contextlib.suppress(queue.Empty):
            while True:
                self._said.get_nowait()
        if self._now is not None and self._now.poll() is None:
            self._now.terminate()

    def close(self) -> None:
        self.hush()
        self._said.put(None)

    def _speak(self) -> None:
        while (text := self._said.get()) is not None:
            try:
                self._now = subprocess.Popen(
                    self.command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                self._now.communicate(text.encode("utf-8", "replace"), timeout=120)
            except (OSError, subprocess.SubprocessError):
                if self._now is not None:
                    self._now.kill()
            finally:
                self._now = None
