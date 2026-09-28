"""Runtime settings, read from environment variables (CLI flags can override them)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "claude-opus-5"
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

# Models new enough for adaptive thinking and the dynamic-filtering web tools.
_MODERN_PREFIXES = (
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
    "claude-fable-5",
    "claude-mythos-5",
)
# Models whose requests the API can re-run on a fallback model when the
# provider's classifiers decline them (the `fallbacks` request parameter).
_FALLBACK_PREFIXES = ("claude-opus-5", "claude-fable-5")


def is_modern(model: str) -> bool:
    return model.startswith(_MODERN_PREFIXES)


@dataclass(frozen=True)
class Config:
    home: Path
    model: str = DEFAULT_MODEL
    effort: str | None = None
    fallbacks: str = "auto"  # auto | on | off
    web: bool = True
    show_thoughts: bool = False
    # Stream tool inputs as they're generated. Proxies in front of the API may reject
    # the field, so it's only sent when talking to the API directly.
    eager_tool_streaming: bool = True

    @property
    def db_path(self) -> Path:
        return self.home / "haven.db"

    @property
    def use_fallbacks(self) -> bool:
        if self.fallbacks == "on":
            return True
        if self.fallbacks == "off":
            return False
        return self.model.startswith(_FALLBACK_PREFIXES)

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Config:
        effort = env.get("HAVEN_EFFORT") or None
        if effort is not None and effort not in EFFORT_LEVELS:
            raise ValueError(f"HAVEN_EFFORT must be one of {', '.join(EFFORT_LEVELS)} (got {effort!r})")
        fallbacks = env.get("HAVEN_FALLBACKS", "auto").lower()
        if fallbacks not in ("auto", "on", "off"):
            raise ValueError(f"HAVEN_FALLBACKS must be auto, on, or off (got {fallbacks!r})")
        return cls(
            home=Path(env.get("HAVEN_HOME") or Path.home() / ".haven").expanduser(),
            model=env.get("HAVEN_MODEL") or DEFAULT_MODEL,
            effort=effort,
            fallbacks=fallbacks,
            web=_flag(env.get("HAVEN_WEB"), default=True),
            show_thoughts=_flag(env.get("HAVEN_SHOW_THOUGHTS"), default=False),
            eager_tool_streaming=not env.get("ANTHROPIC_BASE_URL"),
        )


def _flag(value: str | None, *, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")
