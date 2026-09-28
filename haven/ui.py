"""The terminal: how Haven's words, thoughts, and growth are shown."""

from __future__ import annotations

import os
import sys
from typing import TextIO

from .agent import Activity, TurnResult
from .reflection import Reflection

try:
    import readline  # gives input() line editing and history
except ImportError:  # Windows
    readline = None

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RED = "\033[31m"


class Terminal:
    """Renders a streaming turn. Also serves as the agent's Listener."""

    def __init__(self, out: TextIO | None = None, show_thoughts: bool = False):
        self.out = out = out or sys.stdout
        self.interactive = out.isatty()
        self.color = self.interactive and "NO_COLOR" not in os.environ
        self.show_thoughts = show_thoughts
        self._at_line_start = True
        self._prefixed = False  # "haven ›" printed this turn
        self._speaking = False  # in the middle of Haven's text
        self._indicator = False  # a transient "thinking…" is on screen
        self._thought_open = False

    def paint(self, text: str, *styles: str) -> str:
        return "".join(styles) + text + RESET if self.color and styles else text

    def write(self, text: str) -> None:
        if not text:
            return
        self.out.write(text)
        self.out.flush()
        self._at_line_start = text.endswith("\n")

    def line(self, text: str = "", *styles: str) -> None:
        self._settle()
        self.write(self.paint(text, *styles) + "\n")

    def ask(self) -> str:
        prompt = "you › "
        if self.color and readline is not None:
            # \001 and \002 tell readline the escape codes take no space on screen.
            prompt = f"\001{BOLD}{GREEN}\002{prompt}\001{RESET}\002"
        elif self.color:
            prompt = self.paint(prompt, BOLD, GREEN)
        return input(prompt)

    # --- the agent's Listener interface --------------------------------

    def begin_thinking(self) -> None:
        if self.show_thoughts or not self.interactive or self._speaking or self._indicator:
            return
        self._newline()
        self.write(self.paint("  thinking…", DIM))
        self._indicator = True

    def thinking(self, delta: str) -> None:
        if not self.show_thoughts or not delta:
            return
        self._clear_indicator()
        if not self._thought_open:
            self._newline()
            self.write(self.paint("  (thinking) ", DIM, ITALIC))
            self._thought_open = True
            self._speaking = False
        self.write(self.paint(delta.replace("\n", "\n  "), DIM, ITALIC))

    def text(self, delta: str) -> None:
        self._clear_indicator()
        self._close_thought()
        if not self._prefixed:
            self._newline()
            self.write(self.paint("haven › ", BOLD, CYAN))
            self._prefixed = True
            delta = delta.lstrip("\n")
        elif not self._speaking:
            self._newline()
            delta = delta.lstrip("\n")
        self.write(delta)
        self._speaking = True

    def activity(self, activity: Activity) -> None:
        self.line(f"  · {activity.live}", DIM)

    def notice(self, message: str) -> None:
        self.line(f"  ({message})", YELLOW)

    def error(self, message: str) -> None:
        self.line(f"  {message}", RED)

    def confirm(self, question: str) -> bool:
        self._settle()
        try:
            answer = input(f"{question} [y/N] ")
        except (EOFError, KeyboardInterrupt):
            self.write("\n")
            return False
        return answer.strip().lower() in ("y", "yes")

    def status(self, text: str) -> None:
        """A line that rewrites itself in place, for progress."""
        if self.interactive:
            self.write("\r\033[K" + self.paint(text, DIM))

    def dot(self) -> None:
        if self.interactive:
            self.write(self.paint(".", MAGENTA, DIM))

    # --- turns -----------------------------------------------------------

    def start_turn(self) -> None:
        self._prefixed = self._speaking = False

    def finish_turn(self, result: TurnResult) -> None:
        self._settle()
        for title, url in result.sources[:6]:
            self.line(f"  read: {title} — {url}", DIM)
        if result.stop == "end_turn" and not result.log:
            self.notice("Haven didn't say anything")
        elif result.stop == "max_tokens":
            self.notice("Haven ran out of room mid-thought")
        elif result.stop == "step_limit":
            self.notice("Haven reached its limit of steps for one turn")
        self.write("\n")

    def _settle(self) -> None:
        self._clear_indicator()
        self._close_thought()
        self._newline()
        self._speaking = False

    def _newline(self) -> None:
        if not self._at_line_start:
            self.write("\n")

    def _clear_indicator(self) -> None:
        if self._indicator:
            self.write("\r\033[K")
            self._at_line_start = True
            self._indicator = False

    def _close_thought(self) -> None:
        if self._thought_open:
            self.write("\n")
            self._thought_open = False

    # --- reflection --------------------------------------------------------

    def reflecting(self, label: str = "Haven is reflecting") -> None:
        self._settle()
        self.write(self.paint(f"~ {label} ~ ", MAGENTA))  # progress dots follow on this line

    def reflection(self, r: Reflection) -> None:
        details = []
        if r.memories:
            details.append(f"kept {len(r.memories)} new {'memory' if len(r.memories) == 1 else 'memories'}")
        details += [f'+ came to believe: "{s}" ({c:.2f})' for s, c in r.formed]
        details += [f'~ revised: "{s}" ({a:.2f} -> {b:.2f})' for s, a, b in r.revised]
        details += [f'- let go of: "{s}"' for s in r.abandoned]
        details += [f'? new question: "{q}"' for q in r.new_questions]
        details += [f'* explored: "{q}"' for q in r.explored]
        if r.self_revision:
            details.append(f"# rewrote its self-model: {r.self_revision}")
        if r.journal:
            details.append(f'journal: "{_short(r.journal, 200)}"')
        if r.inner_state:
            details.append(f"state of mind: {r.inner_state}")
        for detail in details or ["nothing changed"]:
            self.line(f"  {detail}", MAGENTA, DIM)
        self.write("\n")


def _short(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
