from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fakes import FakeOllama, reply

from haven.config import Config
from haven.forge import CHATML, RECIPE, Forge, ForgeError
from haven.ollama import OllamaUnavailable

TOOL_CALL = '<tool_call>{"name": "recall", "arguments": {"query": "octopuses"}}</tool_call>'


class Runner:
    """Stands in for subprocess.run: records each command and creates the files that tool would."""

    def __init__(self, fail: str | None = None):
        self.commands: list[list[str]] = []
        self.fail = fail

    def __call__(self, command: list[str], cwd: Path | None = None) -> SimpleNamespace:
        self.commands.append(command)
        if self.fail and self.fail in " ".join(command):
            return SimpleNamespace(returncode=1)
        if Path(command[0]).name == "mergekit-yaml":
            merged = Path(command[2])
            merged.mkdir(parents=True)
            (merged / "config.json").write_text("{}")
            (merged / "model.safetensors").write_bytes(b"weights")
        elif command[0] == "git":
            converter_dir = Path(command[-1])
            converter_dir.mkdir(parents=True)
            (converter_dir / "convert_hf_to_gguf.py").write_text("")
        elif command[1].endswith("convert_hf_to_gguf.py"):
            Path(command[command.index("--outfile") + 1]).write_bytes(b"GGUF")
        return SimpleNamespace(returncode=0)


@pytest.fixture
def tools(monkeypatch):
    monkeypatch.setattr("haven.forge.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("haven.forge.shutil.disk_usage", lambda path: SimpleNamespace(free=100e9))


def make_forge(tmp_path, runner, client=None, **options):
    said: list[str] = []
    client = client or FakeOllama(reply("Hello there."), reply(TOOL_CALL))
    forge = Forge(Config(home=tmp_path), run=runner, client=client, say=said.append, **options)
    return forge, said


def test_the_recipe_says_what_the_model_is_made_of(tmp_path):
    forge, _ = make_forge(tmp_path, Runner())
    assert forge.sources() == ["Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-4B-Thinking-2507"]
    assert forge.method() == "slerp"
    assert forge.description() == (
        "a model called 'haven', forged for you from open-source weights: "
        "a slerp merge of Qwen/Qwen3-4B-Instruct-2507 and Qwen/Qwen3-4B-Thinking-2507"
    )


def test_a_full_forge(tmp_path, tools):
    runner = Runner()
    forge, said = make_forge(tmp_path, runner)
    forge.build()

    merge, clone, convert, create = runner.commands
    assert Path(merge[0]).name == "mergekit-yaml" and merge[1] == str(RECIPE.resolve())
    assert merge[2] == str(tmp_path / "forge" / "haven" / "merged") and "--lazy-unpickle" in merge
    assert clone[:4] == ["git", "clone", "--depth", "1"] and clone[-1].endswith("forge/llama.cpp")
    assert convert[0] == sys.executable and convert[1].endswith("llama.cpp/convert_hf_to_gguf.py")
    assert convert[-2:] == ["--outtype", "q8_0"]
    assert create == ["ollama", "create", "haven", "-f", "Modelfile"]

    modelfile = (tmp_path / "forge" / "haven" / "Modelfile").read_text()
    assert "FROM ./haven-q8_0.gguf" in modelfile and f'TEMPLATE """{CHATML}"""' in modelfile
    assert 'PARAMETER stop "<|im_end|>"' in modelfile and "PARAMETER num_ctx 16384" in modelfile
    assert "  it says: Hello there." in said and "  tool calls: working" in said
    assert not (tmp_path / "forge" / "haven" / "merged").exists()
    assert not (tmp_path / "forge" / "haven" / "haven-q8_0.gguf").exists()


def test_a_failed_forge_picks_up_where_it_stopped(tmp_path, tools):
    forge, _ = make_forge(tmp_path, Runner(fail="convert_hf_to_gguf.py"))
    with pytest.raises(ForgeError, match="Converting to a single q8_0 GGUF file failed") as caught:
        forge.build()
    assert "requirements-convert_hf_to_gguf.txt" in str(caught.value)

    runner = Runner()
    forge, said = make_forge(tmp_path, runner, keep=True)
    forge.build()
    assert [command[0] for command in runner.commands] == [sys.executable, "ollama"]
    assert any("skipping the merge" in line for line in said)
    assert (tmp_path / "forge" / "haven" / "merged" / "config.json").exists()  # --keep


def test_half_finished_work_is_never_reused(tmp_path, tools):
    workdir = tmp_path / "forge" / "haven"
    (workdir / "merged").mkdir(parents=True)
    (workdir / "merged" / "config.json").write_text("{}")  # an interrupted merge
    (workdir / "haven-q8_0.gguf.part").write_bytes(b"GG")  # an interrupted conversion
    runner = Runner()
    forge, _ = make_forge(tmp_path, runner)
    forge.build()
    assert [Path(command[0]).name for command in runner.commands][:1] == ["mergekit-yaml"]
    assert any(command[0] == sys.executable for command in runner.commands)


def test_problems_are_found_before_anything_is_downloaded(tmp_path, monkeypatch):
    monkeypatch.setattr("haven.forge.shutil.which", lambda name: None)
    monkeypatch.setattr("haven.forge.shutil.disk_usage", lambda path: SimpleNamespace(free=5e9))
    runner = Runner()
    forge, _ = make_forge(tmp_path, runner)
    problems = "\n".join(forge.problems())
    assert "mergekit isn't installed" in problems
    assert "git is needed" in problems and "ollama command isn't installed" in problems
    assert "only 5 GB of disk is free" in problems
    with pytest.raises(ForgeError):
        forge.build()
    assert runner.commands == []


def test_ollama_must_be_running(tmp_path, tools):
    client = FakeOllama()
    client.unavailable = OllamaUnavailable("nothing answered")
    forge, _ = make_forge(tmp_path, Runner(), client=client)
    assert any("isn't answering" in problem for problem in forge.problems())


def test_the_chat_template_can_come_from_another_model(tmp_path):
    client = FakeOllama()
    client.shown = {
        "template": "{{ .Prompt }}",
        "parameters": 'stop                           "<|eot_id|>"\ntemperature 0.6',
    }
    forge, _ = make_forge(tmp_path, Runner(), client=client, template_from="llama3.1:8b")
    text = forge.modelfile_text()
    assert 'TEMPLATE """{{ .Prompt }}"""' in text and 'PARAMETER stop "<|eot_id|>"' in text
    assert "<|im_end|>" not in text


def test_a_model_that_cant_call_tools_is_reported(tmp_path, tools):
    forge, said = make_forge(tmp_path, Runner(), client=FakeOllama(reply("Hello."), reply("I'd rather not.")))
    forge.build()
    assert any("didn't make a clean tool call" in line for line in said)
