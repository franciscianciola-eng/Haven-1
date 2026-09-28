"""A fake internet for tests: small stand-ins for every source Haven reads, served on this machine."""

from __future__ import annotations

import json
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

NAMES = ["Lily", "Tom", "Sam", "Mia", "Ben", "Anna"]
THINGS = ["ball", "cat", "dog", "tree", "cake", "boat", "kite", "bird"]
COLORS = ["red", "blue", "green", "yellow", "big", "little"]
PLACES = ["park", "garden", "house", "river", "forest", "beach"]


def story(rng: random.Random) -> str:
    name, thing, color, place = rng.choice(NAMES), rng.choice(THINGS), rng.choice(COLORS), rng.choice(PLACES)
    friend = rng.choice([n for n in NAMES if n != name])
    return (
        f"Once upon a time, there was a girl named {name}. She had a {color} {thing}. "
        f"One day, {name} went to the {place} with her {thing}. She met her friend {friend}. "
        f'{friend} said, "I like your {color} {thing}!" {name} was happy. They played in the {place} all day. '
        f"At night, {name} went home and hugged her {thing}. The end."
    )


def stories(n: int, seed: int) -> str:
    rng = random.Random(seed)
    return "\n<|endoftext|>\n".join(story(rng) for _ in range(n)) + "\n<|endoftext|>\n"


def book(book_id: int) -> str:
    rng = random.Random(book_id)
    paragraphs = []
    for _ in range(60):
        paragraphs.append(" ".join(story(rng).split(". ")[:4]) + ".")
    return (
        f"The Project Gutenberg eBook of Book {book_id}\n\n*** START OF THE PROJECT GUTENBERG EBOOK BOOK {book_id} ***\n\n"
        + "\n\n".join(paragraphs)
        + f"\n\n*** END OF THE PROJECT GUTENBERG EBOOK BOOK {book_id} ***\nLicense text."
    )


def article(i: int) -> dict:
    rng = random.Random(i)
    thing, place = rng.choice(THINGS), rng.choice(PLACES)
    text = (
        f"A {thing} is a thing that people often see in a {place}. Many {thing}s are {rng.choice(COLORS)}. "
        f"People in many countries like the {thing}. The {thing} was first written about long ago. "
        f"Today there are many kinds of {thing}. Some people keep a {thing} at home. " * 2
    )
    return {"pageid": i, "title": f"{thing.title()} {i}", "extract": text}


def squad(n: int, seed: int) -> dict:
    rng = random.Random(seed)
    paragraphs = []
    for i in range(n):
        name, place, thing = rng.choice(NAMES), rng.choice(PLACES), rng.choice(THINGS)
        context = (
            f"{name} lived near the {place}. Every morning {name} took a {thing} to the {place} and sat by the water."
        )
        paragraphs.append(
            {
                "context": context,
                "qas": [
                    {
                        "id": f"{i}a",
                        "question": f"Where did {name} live?",
                        "answers": [{"text": f"near the {place}", "answer_start": 0}],
                    },
                    {
                        "id": f"{i}b",
                        "question": f"What did {name} take?",
                        "answers": [{"text": f"a {thing}", "answer_start": 0}],
                    },
                ],
            }
        )
    return {"data": [{"title": "Stories", "paragraphs": paragraphs}]}


def gsm8k(n: int, seed: int) -> str:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a, b = rng.randint(2, 20), rng.randint(2, 20)
        name, thing = rng.choice(NAMES), rng.choice(THINGS)
        rows.append(
            json.dumps(
                {
                    "question": f"{name} has {a} {thing}s and gets {b} more. How many {thing}s does {name} have now?",
                    "answer": f"{name} has {a} + {b} = <<{a}+{b}={a + b}>>{a + b} {thing}s.\n#### {a + b}",
                }
            )
        )
    return "\n".join(rows) + "\n"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def do_GET(self) -> None:
        url = urlsplit(self.path)
        path, query = url.path, parse_qs(url.query)
        if path == "/tinystories.txt":
            body, kind = stories(300, 1), "text/plain"
        elif path == "/tinystories-valid.txt":
            body, kind = stories(40, 2), "text/plain"
        elif path.startswith("/mirror/") and path.endswith("-0.txt"):
            number = int(path.split("/")[-2])
            if number == 16:  # this one only has an old-style file, like some real books
                self.send_response(404)
                self.end_headers()
                return
            body, kind = book(number), "text/plain"
        elif path.startswith("/mirror/") and path.endswith("/16.txt"):
            body, kind = book(16), "text/plain"
        elif path == "/w/api.php":
            if query.get("generator") == ["random"]:
                start = random.randint(0, 10_000)
                pages = {str(i): article(i) for i in range(start, start + 20)}
                body = json.dumps({"query": {"pages": pages}})
            elif query.get("list") == ["search"]:
                term = query["srsearch"][0]
                body = json.dumps({"query": {"search": [{"title": f"{term.title()}"}]}})
            else:
                title = query["titles"][0]
                body = json.dumps(
                    {
                        "query": {
                            "pages": {
                                "1": {"pageid": 1, "title": title, "extract": f"{title} is a thing in a garden. " * 20}
                            }
                        }
                    }
                )
            kind = "application/json"
        elif path == "/squad-train.json":
            body, kind = json.dumps(squad(80, 3)), "application/json"
        elif path == "/squad-dev.json":
            body, kind = json.dumps(squad(20, 4)), "application/json"
        elif path == "/gsm8k-train.jsonl":
            body, kind = gsm8k(120, 5), "text/plain"
        elif path == "/gsm8k-test.jsonl":
            body, kind = gsm8k(30, 6), "text/plain"
        else:
            self.send_response(404)
            self.end_headers()
            return
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def serve() -> tuple[ThreadingHTTPServer, dict]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    urls = {
        "tinystories": f"{base}/tinystories.txt",
        "tinystories-valid": f"{base}/tinystories-valid.txt",
        "gutenberg": f"{base}/mirror",
        "simplewiki": f"{base}/w/api.php",
        "wikipedia": f"{base}/w/api.php",
        "squad-train": f"{base}/squad-train.json",
        "squad-dev": f"{base}/squad-dev.json",
        "gsm8k-train": f"{base}/gsm8k-train.jsonl",
        "gsm8k-test": f"{base}/gsm8k-test.jsonl",
    }
    return server, urls


CHAT = """{%- for message in messages -%}
{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- '<|im_start|>assistant\\n' -}}
{%- if enable_thinking is defined and enable_thinking is false -%}
{{- '<think>\\n\\n</think>\\n\\n' -}}
{%- endif -%}
{%- endif -%}"""


def tiny_base(folder, text: str = ""):
    """A tiny, untrained model with the same architecture and chat format as Qwen3, saved like a download."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    rng = random.Random(0)
    text = text or " ".join(story(rng) for _ in range(200)) + " I feel fine. I'm hungry. I see something red ahead."
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=700,
        special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator([text], trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>", chat_template=CHAT
    )
    fast.add_tokens(["<think>", "</think>"])  # ordinary tokens, as in Qwen3
    fast.save_pretrained(folder)
    config = Qwen3Config(
        vocab_size=len(fast),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
        eos_token_id=fast.eos_token_id,
        pad_token_id=fast.pad_token_id,
    )
    torch.manual_seed(0)
    Qwen3ForCausalLM(config).save_pretrained(folder)
    return folder


def teach_base(folder, texts: list[str], steps: int = 300, seed: int = 0):
    """Pretrain the tiny stand-in on some text, so that it is a (very small) language model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(folder)
    model = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32)
    model.train()
    rows = [tok(t + tok.eos_token, add_special_tokens=False).input_ids[:512] for t in texts]
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rng = random.Random(seed)
    for _ in range(steps):
        batch = [rng.choice(rows) for _ in range(16)]
        length = max(len(r) for r in batch)
        ids = torch.full((len(batch), length), tok.pad_token_id)
        labels = torch.full((len(batch), length), -100)
        for i, r in enumerate(batch):
            ids[i, : len(r)] = torch.tensor(r)
            labels[i, : len(r)] = torch.tensor(r)
        loss = model(input_ids=ids, labels=labels).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.save_pretrained(folder)
    return float(loss)
