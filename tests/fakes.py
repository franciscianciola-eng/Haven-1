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


# A little encyclopedia, as TensorFlow Datasets keeps Wikipedia: the plain text of each article, headings on lines of
# their own, categories at the end.
ARTICLES = [
    (
        "France",
        "France (officially the French Republic) is a country in Western Europe. It has a long history.\n\n"
        "The capital of France is Paris. About 68 million people live in France.\n\nHistory\n\n"
        "France was a kingdom for a long time. The French Revolution began in 1789.\n\n"
        "References\n\nCategory:Countries in Europe",
    ),
    (
        "Paris",
        "Paris is the capital and largest city of France. The river Seine flows through it.\n\n"
        "The Eiffel Tower was built in 1889.\n\nCategory:Capitals in Europe",
    ),
    (
        "Albert Einstein",
        "Albert Einstein (14 March 1879 – 18 April 1955) was a German-born American scientist. He worked on "
        "theoretical physics. He developed the theory of relativity.\n\nEarly life\nEinstein was born in Ulm in "
        "1879.\n\nCategory:Physicists",
    ),
    (
        "Einstein field equations",
        "The Einstein field equations are equations that describe gravity in the classical sense.",
    ),
    (
        "Spider",
        "Spiders (order Araneae) are air-breathing arthropods. They have eight legs, and fangs that inject venom. "
        "Most make silk. Over twenty classifications have been proposed since 1900.p3 \n\n"
        "Spiders live on every continent except for Antarctica. Almost all spiders are predators, and most eat insects.",
    ),
    ("Cat", "Cats are small, carnivorous mammals. They have been kept as pets for 10,000 years.\n\nCategory:Cats"),
    ("Cat (zodiac)", "The Cat is the fourth animal symbol in the Vietnamese zodiac."),
    (
        "World War II",
        "World War II was a global war that lasted from 1939 to 1945. Most of the world's countries fought in it. "
        "The war ended with an Allied victory in 1945.",
    ),
    ("Romeo and Juliet", "Romeo and Juliet is a play written by William Shakespeare. It is a tragedy."),
    (
        "Mount Everest",
        "Mount Everest is the highest mountain on Earth. It is in the Himalayas, on the border of Nepal and China.",
    ),
    ("Capital of France", "This article is about the French national capital in general. The capital of France is Paris."),
    ("Rome", "Rome is the capital city of Italy. About 2.8 million people live in Rome."),
]


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte, n = n & 0x7F, n >> 7
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _field(number: int, payload: bytes) -> bytes:
    return _varint(number << 3 | 2) + _varint(len(payload)) + payload


def tfrecord(articles: list[tuple[str, str]]) -> bytes:
    """Articles as a TFRecord file of tf.train.Examples, the way TensorFlow Datasets keeps Wikipedia."""
    out = bytearray()
    for title, text in articles:
        entries = b"".join(
            _field(1, _field(1, key.encode()) + _field(2, _field(1, _field(1, value.encode()))))
            for key, value in (("text", text), ("title", title))
        )
        example = _field(1, entries)
        out += len(example).to_bytes(8, "little") + b"\0\0\0\0" + example + b"\0\0\0\0"
    return bytes(out)


PAGE = """<html><head><title>Bread - a page</title><script>var x = 1;</script></head><body>
<nav>Home | About | Contact us today</nav>
<h1>Bread</h1><p>Bread is a food made from flour, water and yeast. People have baked bread for thousands of years.</p>
<p>Sourdough bread is made with wild yeast. It tastes a little sour.</p>
<footer>Copyright 2024, all rights kept by the bakers</footer></body></html>"""


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
        elif path.startswith("/tfds/") and "tfrecord-" in path:  # an encyclopedia in four parts
            part = int(path.rsplit("tfrecord-", 1)[1][:5])
            data = tfrecord(ARTICLES[part::4])
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        elif path == "/bread.html":
            body, kind = PAGE, "text/html"
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
        "tfds": f"{base}/tfds",
        "page": f"{base}/bread.html",
    }
    return server, urls
