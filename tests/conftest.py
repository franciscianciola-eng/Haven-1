from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from haven import prompts  # noqa: E402
from haven.agent import Mind  # noqa: E402
from haven.config import Config  # noqa: E402
from haven.store import Store  # noqa: E402


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now += timedelta(**delta)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(clock: Clock) -> Store:
    store = Store(":memory:", clock=clock)
    store.ensure_born(prompts.GENESIS)
    yield store
    store.close()


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(home=tmp_path, eager_tool_streaming=True)


@pytest.fixture
def make_mind(config: Config, store: Store):
    def make(client, **overrides) -> Mind:
        cfg = Config(**{**config.__dict__, **overrides})
        return Mind(cfg, store, client)

    return make
