"""The forge: build Haven's own model by merging open-source weights, then load it into Ollama.

    open weights (Hugging Face) --mergekit--> merged weights --llama.cpp--> GGUF --ollama create--> "haven"
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import prompts
from .agent import TOOL_SPECS, TagSplitter
from .config import Config
from .ollama import Ollama, OllamaError

RECIPE = Path(__file__).parent / "recipes" / "haven.yml"
LLAMA_CPP = "https://github.com/ggml-org/llama.cpp"
OUTTYPES = ("q8_0", "f16", "bf16")
MIN_FREE_GB = 30
CHATML = """{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
CHATML_STOPS = ("<|im_start|>", "<|im_end|>")


class ForgeError(Exception):
    pass


@dataclass
class Forge:
    config: Config
    recipe: Path = RECIPE
    name: str = "haven"
    outtype: str = "q8_0"
    template_from: str | None = None  # copy the chat template from this Ollama model instead of ChatML
    cuda: bool = False
    keep: bool = False  # keep the merged weights and the GGUF file afterwards
    fresh: bool = False  # redo steps even if their output already exists
    say: Callable[[str], None] = print
    run: Callable[..., Any] = subprocess.run
    client: Any = field(default=None)

    def __post_init__(self) -> None:
        # Steps run inside the work directory, so every path must be absolute.
        self.recipe = Path(self.recipe).expanduser().resolve()
        if self.client is None:
            self.client = Ollama(self.config.host)

    @property
    def workdir(self) -> Path:
        return self.config.forge_dir.expanduser().resolve() / self.name

    @property
    def merged(self) -> Path:
        return self.workdir / "merged"

    @property
    def gguf(self) -> Path:
        return self.workdir / f"{self.name}-{self.outtype}.gguf"

    @property
    def converter(self) -> Path:
        return self.workdir.parent / "llama.cpp" / "convert_hf_to_gguf.py"

    @property
    def modelfile(self) -> Path:
        return self.workdir / "Modelfile"

    @property
    def mergekit(self) -> str | None:
        """mergekit-yaml from this Python's environment, even when it isn't activated (e.g. under cron)."""
        beside = Path(sys.executable).parent / ("mergekit-yaml.exe" if sys.platform == "win32" else "mergekit-yaml")
        return str(beside) if beside.exists() else shutil.which("mergekit-yaml")

    def sources(self) -> list[str]:
        text = self.recipe.read_text()
        return list(dict.fromkeys(re.findall(r"^\s*-?\s*model:\s*['\"]?([^\s'\"#]+)", text, re.MULTILINE)))

    def method(self) -> str:
        match = re.search(r"^merge_method:\s*(\S+)", self.recipe.read_text(), re.MULTILINE)
        return match.group(1) if match else "merge"

    def description(self) -> str:
        return (
            f"a model called '{self.name}', forged for you from open-source weights: "
            f"a {self.method()} merge of {' and '.join(self.sources())}"
        )

    def problems(self) -> list[str]:
        """Everything that would stop the forge, found before hours of downloading."""
        found = []
        if not self.recipe.is_file():
            return [f"there's no recipe at {self.recipe}"]
        if not self.sources():
            found.append(f"the recipe at {self.recipe} doesn't name any models")
        if not self.mergekit:
            found.append("mergekit isn't installed. Install the forge's tools with: pip install -e '.[forge]'")
        if not self.converter.exists() and not shutil.which("git"):
            found.append("git is needed to download llama.cpp's converter (https://git-scm.com)")
        if not shutil.which("ollama"):
            found.append("the ollama command isn't installed (https://ollama.com)")
        else:
            try:
                self.client.version()
            except OllamaError:
                found.append(f"Ollama isn't answering at {self.config.host}; start it first")
        if not self._merge_done.exists():
            existing = next(p for p in (self.workdir, *self.workdir.parents) if p.exists())
            free = shutil.disk_usage(existing).free / 1e9
            if free < MIN_FREE_GB:
                found.append(f"only {free:.0f} GB of disk is free; forging needs about {MIN_FREE_GB} GB")
        return found

    def build(self) -> None:
        problems = self.problems()
        if problems:
            raise ForgeError("\n".join(problems))
        self.workdir.mkdir(parents=True, exist_ok=True)
        if self.fresh:
            self._clear()
        self._merge()
        self._get_converter()
        self._convert()
        self._create()
        self._test()
        if not self.keep:
            self._clean()

    @property
    def _merge_done(self) -> Path:
        return self.workdir / "merged.done"  # written only after mergekit succeeds, so a half-merge isn't reused

    def _merge(self) -> None:
        if self._merge_done.exists():
            self.say("The merged weights are already there, so skipping the merge (--fresh redoes it).")
            return
        shutil.rmtree(self.merged, ignore_errors=True)  # leftovers of an interrupted merge
        command = [self.mergekit or "mergekit-yaml", str(self.recipe), str(self.merged), "--lazy-unpickle"]
        if self.cuda:
            command.append("--cuda")
        self._step("Merging the weights (the first time, this downloads them from Hugging Face)", command)
        self._merge_done.write_text(self.recipe.read_text())

    def _get_converter(self) -> None:
        if self.converter.exists():
            return
        command = ["git", "clone", "--depth", "1", LLAMA_CPP, str(self.converter.parent)]
        self._step("Downloading llama.cpp's GGUF converter", command)

    def _convert(self) -> None:
        if self.gguf.exists():
            self.say(f"{self.gguf.name} is already there, so skipping the conversion.")
            return
        requirements = self.converter.parent / "requirements" / "requirements-convert_hf_to_gguf.txt"
        partial = self.gguf.with_name(self.gguf.name + ".part")  # renamed only once it's complete
        command = [sys.executable, str(self.converter), str(self.merged), "--outfile", str(partial)]
        self._step(
            f"Converting to a single {self.outtype} GGUF file",
            [*command, "--outtype", self.outtype],
            hint=f"If a Python package is missing, run: {sys.executable} -m pip install -r {requirements}",
        )
        partial.replace(self.gguf)

    def _create(self) -> None:
        self.modelfile.write_text(self.modelfile_text())
        self._step(f"Loading it into Ollama as '{self.name}'", ["ollama", "create", self.name, "-f", "Modelfile"])

    def modelfile_text(self) -> str:
        template, stops = CHATML, list(CHATML_STOPS)
        if self.template_from:
            details = self.client.show(self.template_from)
            template = details.get("template") or template
            stops = re.findall(r'^stop\s+"(.*)"\s*$', details.get("parameters") or "", re.MULTILINE) or stops
        lines = [
            f"# Haven's model, forged by `haven forge` from: {', '.join(self.sources())}",
            f"FROM ./{self.gguf.name}",
            f'TEMPLATE """{template}"""',
            *[f'PARAMETER stop "{stop}"' for stop in stops],
            "PARAMETER temperature 0.7",
            "PARAMETER top_p 0.8",
            "PARAMETER top_k 20",
            f"PARAMETER num_ctx {self.config.context}",
        ]
        return "\n".join(lines) + "\n"

    def _test(self) -> None:
        self.say("Testing the new model...")
        hello = self._ask([{"role": "user", "content": "Say hello in one short sentence."}])
        self.say(f"  it says: {' '.join(hello.split())[:200] or '(nothing)'}")
        recall = next(spec for spec in TOOL_SPECS if spec["name"] == "recall")
        reply = self._ask(
            [
                {"role": "system", "content": "You are Haven.\n\n" + prompts.tools_block([recall])},
                {"role": "user", "content": "Use your recall tool to look up what you remember about octopuses."},
            ]
        )
        splitter = TagSplitter(tool_calls=True)
        splitter.feed(reply)
        splitter.finish()
        if splitter.calls:
            self.say("  tool calls: working")
        else:
            self.say(
                "  tool calls: it didn't make a clean tool call in the test. Haven will still run, "
                "but may use its memory and the web less reliably."
            )

    def _ask(self, messages: list[dict]) -> str:
        options = {"num_ctx": 4096, "num_predict": 200}
        chunks = self.client.chat(model=self.name, messages=messages, options=options, keep_alive="5m")
        return "".join((chunk.get("message") or {}).get("content") or "" for chunk in chunks)

    def _clear(self) -> None:
        shutil.rmtree(self.merged, ignore_errors=True)
        self._merge_done.unlink(missing_ok=True)
        self.gguf.unlink(missing_ok=True)

    def _clean(self) -> None:
        self._clear()
        self.say(
            "Removed the intermediate files (--keep keeps them). The downloaded source weights stay in "
            "Hugging Face's cache (usually ~/.cache/huggingface/hub); delete them there to free the space."
        )

    def _step(self, title: str, command: list[str], hint: str = "") -> None:
        self.say(f"\n== {title} ==")
        completed = self.run(command, cwd=self.workdir)
        if completed.returncode != 0:
            message = f"{title} failed (exit code {completed.returncode}). The command was:\n  {' '.join(command)}"
            raise ForgeError(message + (f"\n{hint}" if hint else ""))
