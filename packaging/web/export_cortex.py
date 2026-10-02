"""Haven's language cortex for the browser: docs/cortex/cortex.onnx (and its tokenizer beside it).

    python packaging/web/export_cortex.py

The same network as the cortex Haven starts with, step by step, for ONNX Runtime Web: it
reads input pieces (and, the first time, the workspace tokens: Haven's state), carrying
on from the keys and values of what it read before, and gives the next piece's scores
and what it all means (in the workspace's format). Its big matrices stay as they're
packed (each row as whole numbers from -127 to 127, with one scale for the row), so the
browser's copy is the desktop's to the last digit, and about as small (55 MB).
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from haven.cortex import starter
from haven.cortex.model import Cortex, CortexConfig

OUT = ROOT / "docs" / "cortex"


class Step(nn.Module):
    """One step of the cortex with a cache, for B drafts side by side. tokens (B, T); state (B, S, core), S = slots
    the first time, else 0; then each layer's past keys and values (B, heads, P, head size). Returns the next piece's
    scores (B, vocab) and the meaning (B, core) after the last position, and each layer's keys and values of
    everything read so far."""

    def __init__(self, cortex: Cortex):
        super().__init__()
        self.c = cortex

    def forward(self, tokens, state, *past):
        c, cfg = self.c, self.c.cfg
        d, heads = cfg.d, cfg.heads
        size = d // heads
        prefix = c.state_in(state) + c.slot[: state.shape[1]]
        x = torch.cat([prefix, c.embed(tokens)], dim=1)
        b, n, before = x.shape[0], x.shape[1], past[0].shape[2]
        x = x + c.position(torch.arange(n) + before)
        rows = torch.arange(n).unsqueeze(1) + before
        cols = torch.arange(n + before).unsqueeze(0)
        mask = torch.where(cols > rows, torch.tensor(float("-inf")), torch.tensor(0.0))
        cache = []
        for i, block in enumerate(c.blocks):
            q, k, v = block.qkv(block.norm1(x)).split(d, dim=2)
            q, k, v = (z.reshape(b, n, heads, size).transpose(1, 2) for z in (q, k, v))
            k = torch.cat([past[2 * i], k], dim=2)
            v = torch.cat([past[2 * i + 1], v], dim=2)
            att = torch.softmax(q @ k.transpose(2, 3) / math.sqrt(size) + mask, dim=-1)
            x = x + block.proj((att @ v).transpose(1, 2).reshape(b, n, d))
            x = x + block.mlp(block.norm2(x))
            cache += [k, v]
        last = c.norm(x[:, -1])
        return (c.head(last), c.state_out(last), *cache)


def load() -> tuple[Cortex, dict, dict]:
    checkpoint = torch.load(starter.FOLDER / "cortex.pt", map_location="cpu", weights_only=False)
    model = Cortex(CortexConfig(**checkpoint["config"]))
    model.load_state_dict(starter.unpacked(checkpoint["model"]))
    return model.eval(), checkpoint["model"], checkpoint["progress"]


def described(model: Cortex, progress: dict) -> str:
    """What its cortex is, as the desktop's says it (think.py's OwnThinker.describe), for the page to show."""
    from types import SimpleNamespace

    from haven.cortex.think import OwnThinker

    return OwnThinker.describe(
        SimpleNamespace(progress=progress, hearing=SimpleNamespace(words=lambda: 0), model=model)
    )


def export(model: Cortex, path: Path) -> None:
    cfg = model.cfg
    step = Step(model).eval()
    size = cfg.d // cfg.heads
    tokens = torch.tensor([[5, 6, 7]])
    state = torch.zeros(1, cfg.slots, cfg.core)
    past = [torch.zeros(1, cfg.heads, 2, size) for _ in range(2 * cfg.layers)]
    cache = [f"{i}.{kind}" for i in range(cfg.layers) for kind in ("keys", "values")]
    torch.onnx.export(
        step,
        (tokens, state, *past),
        str(path),
        input_names=["tokens", "state", *(f"past.{name}" for name in cache)],
        output_names=["logits", "meaning", *(f"now.{name}" for name in cache)],
        dynamic_axes={
            "tokens": {0: "drafts", 1: "new"},
            "state": {0: "drafts", 1: "slots"},
            "logits": {0: "drafts"},
            "meaning": {0: "drafts"},
            **{f"past.{name}": {0: "drafts", 2: "before"} for name in cache},
            **{f"now.{name}": {0: "drafts", 2: "all"} for name in cache},
        },
        opset_version=17,
        do_constant_folding=False,
        dynamo=False,
    )


def packed_weights(graph, packed: dict, model: Cortex) -> int:
    """Swap each big matrix for its packed rows, unpacked as the desktop unpacks them: the whole numbers as decimals,
    times each row's scale. (Plain arithmetic, so ONNX Runtime works the matrix out once, as it loads, rather than
    treat it as quantized: its quantized arithmetic is quicker, but not the same.)"""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    params = dict(model.named_parameters())
    swapped = 0
    keep = []
    for init in graph.initializer:
        name = init.name.removeprefix("c.")
        if isinstance(packed.get(name), dict) and name in params:
            q = packed[name]["int8"].numpy()
            scale = packed[name]["scale"].float().numpy().reshape(-1, 1)
            graph.node.insert(0, helper.make_node("Mul", [init.name + ".rows", init.name + ".scale"], [init.name]))
            graph.node.insert(
                0, helper.make_node("Cast", [init.name + ".int8"], [init.name + ".rows"], to=TensorProto.FLOAT)
            )
            keep += [
                numpy_helper.from_array(q, init.name + ".int8"),
                numpy_helper.from_array(scale.astype(np.float32), init.name + ".scale"),
            ]
            swapped += 1
        else:
            keep.append(init)
    del graph.initializer[:]
    graph.initializer.extend(keep)
    onnx.checker.check_model(onnx.helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]))
    return swapped


def main() -> None:
    import onnx

    OUT.mkdir(parents=True, exist_ok=True)
    model, packed, learned = load()
    raw = OUT / "cortex-float.onnx"
    export(model, raw)
    proto = onnx.load(str(raw))
    swapped = packed_weights(proto.graph, packed, model)
    path = OUT / "cortex.onnx"
    onnx.save(proto, str(path))
    raw.unlink()
    shutil.copyfile(starter.FOLDER / "tokenizer.json", OUT / "tokenizer.json")
    progress = json.loads((starter.FOLDER / "progress.json").read_text())
    (OUT / "cortex.json").write_text(
        json.dumps(
            {
                "config": vars(model.cfg),
                "starter": progress.get("starter"),
                "progress": {"levels": learned["levels"], "heard": learned.get("heard", {})},
                "described": described(model, learned),
            },
            indent=1,
        )
        + "\n"
    )
    print(f"{path}: {path.stat().st_size / 1e6:.1f} MB, {swapped} packed matrices")


if __name__ == "__main__":
    main()
