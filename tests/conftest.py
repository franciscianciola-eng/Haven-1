import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fakes import serve

from haven.mind import Mind


@pytest.fixture(scope="session")
def grown() -> Mind:
    """A Haven that has lived two and a half days (shared by tests that only look)."""
    mind = Mind(seed=0)
    mind.live(3000)
    return mind


@pytest.fixture(scope="session")
def internet():
    server, urls = serve()
    yield urls
    server.shutdown()


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("HAVEN_HOME", str(tmp_path / "haven"))
    return tmp_path / "haven"
