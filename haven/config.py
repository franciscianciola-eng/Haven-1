"""Runtime settings, read from environment variables (CLI flags can override them)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "haven"  # Haven's own model, built by `haven forge`
DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_CONTEXT = 16384
MIN_CONTEXT = 4096
THINK_SETTINGS = ("auto", "on", "off")
TOOL_SETTINGS = ("auto", "native", "prompted", "off")
SAFESEARCH_SETTINGS = ("off", "moderate", "on")


@dataclass(frozen=True)
class Config:
    home: Path
    model: str = DEFAULT_MODEL
    host: str = DEFAULT_HOST
    context: int = DEFAULT_CONTEXT  # tokens the model sees at once (Ollama's num_ctx)
    think: str = "auto"  # auto: think if the model can
    tools: str = "auto"  # auto: the model's own tool calling if it has it, else Haven's prompted format
    web: bool = True
    safesearch: str = "off"
    show_thoughts: bool = False

    @property
    def db_path(self) -> Path:
        return self.home / "haven.db"

    @property
    def forge_dir(self) -> Path:
        return self.home / "forge"

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Config:
        context = env.get("HAVEN_CONTEXT") or str(DEFAULT_CONTEXT)
        if not context.isdigit() or int(context) < MIN_CONTEXT:
            raise ValueError(f"HAVEN_CONTEXT must be a number of tokens, at least {MIN_CONTEXT} (got {context!r})")
        return cls(
            home=Path(env.get("HAVEN_HOME") or Path.home() / ".haven").expanduser(),
            model=env.get("HAVEN_MODEL") or DEFAULT_MODEL,
            host=parse_host(env.get("OLLAMA_HOST")),
            context=int(context),
            think=_choice(env, "HAVEN_THINK", THINK_SETTINGS, "auto"),
            tools=_choice(env, "HAVEN_TOOLS", TOOL_SETTINGS, "auto"),
            web=_flag(env.get("HAVEN_WEB"), default=True),
            safesearch=_choice(env, "HAVEN_SAFESEARCH", SAFESEARCH_SETTINGS, "off"),
            show_thoughts=_flag(env.get("HAVEN_SHOW_THOUGHTS"), default=False),
        )


def parse_host(value: str | None) -> str:
    """Read OLLAMA_HOST the way Ollama does: '0.0.0.0', ':11435', 'http://box:80/ollama', ..."""
    value = (value or "").strip()
    if not value:
        return DEFAULT_HOST
    scheme, separator, rest = value.partition("://")
    if not separator:
        scheme, rest, default_port = "http", value, "11434"
    else:
        default_port = "443" if scheme == "https" else "80"
    address, _, path = rest.partition("/")
    if address.startswith("["):  # IPv6
        end = address.find("]") + 1
        host, port = address[:end], address[end + 1 :] if address[end : end + 1] == ":" else ""
    else:
        host, _, port = address.partition(":")
    path = path.strip("/")
    return f"{scheme}://{host or '127.0.0.1'}:{port or default_port}" + (f"/{path}" if path else "")


def _choice(env: Mapping[str, str], name: str, allowed: tuple[str, ...], default: str) -> str:
    value = (env.get(name) or default).strip().lower()
    if value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(allowed)} (got {value!r})")
    return value


def _flag(value: str | None, *, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")
