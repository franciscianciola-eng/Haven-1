"""What Haven hears: real language, the kind a small child hears.

A child hears millions of words in its first years: people talking to it, and stories read
to it. Haven hears the same kinds of things:

  speech   what parents and others said to children under six, from the CHILDES
           transcripts (the AO-CHILDES collection: about five million words, youngest
           children first; MacWhinney 2000, Huebner & Willits 2021)
  books    children's books from Project Gutenberg (through GITenberg, its copy on
           GitHub): bedtime stories, fairy tales and fables, nursery rhymes, first
           readers and a children's magazine, simple chapter books, and the classics
           read aloud to children

Everything is downloaded once and kept. Some of each is held out (whole books, and whole
stretches of speech), and is only used to test how well it understands what it hasn't
heard before.
"""

from __future__ import annotations

import contextlib
import json
import random
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

from ..web import Web, WebError
from .sources import _passages, _strip_gutenberg
from .stories import Story, from_passage

SPEECH = "https://raw.githubusercontent.com/phueb/BabyBERTa/master/data/corpora/aochildes.txt"
CATALOG = "https://raw.githubusercontent.com/gitenberg-dev/gitberg/master/gitenberg/data/GITenberg_repo_list.tsv"
BOOK = "https://raw.githubusercontent.com/GITenberg/{repo}/master/{file}"

# Project Gutenberg's numbers for the books it hears, by kind.
BEDTIME = (  # bedtime stories: Burgess's animals, Uncle Wiggily, the Tuck-me-in and Sleepy-time tales, Beatrix Potter
    1825, 2441, 2493, 2557, 3074, 4670, 4698, 4979, 5110, 5262, 5701, 5727, 5791, 5844, 5955, 6754, 9462, 11156,
    13087, 13355, 14375, 14407, 14732, 14797, 14814, 14837, 14838, 14848, 14868, 14872, 14877, 14958, 15168, 15280,
    15281, 15282, 15284, 15521, 15528, 15575, 16663, 17250, 17807, 18599, 18626, 18630, 18652, 18656, 18662, 20716,
    20877, 21015, 21078, 21203, 21286, 21322, 21412, 21426, 21619, 21836, 21844, 21845, 22816, 23213, 24589, 24590,
    24592, 24608, 24628, 24731, 24872, 24881, 25090, 25301, 25824, 30667, 37952, 39706, 42574, 43596, 44914, 45264,
    45265, 46866, 46950, 46951, 46952, 46988, 50405, 54995, 56950, 60017, 60625, 61671, 61695, 61735, 67990, 69458,
    70017, 70295, 70627, 70783, 71185, 71515, 71594, 72607, 72612, 72746, 19409, 11757, 18190, 17371,
)  # fmt: skip
TALES = (  # fairy tales, fables and folk tales, and stories for telling to children
    21, 19994, 2591, 11027, 1597, 27200, 503, 540, 640, 641, 2435, 3027, 3282, 3454, 5615, 6746, 7277, 27826, 7439,
    14241, 7885, 7128, 4018, 14916, 17208, 23661, 22248, 11167, 18155, 18735, 23322, 25877, 1988, 473, 5835, 16693,
    2781, 236, 1937,
)  # fmt: skip
RHYMES = (  # nursery rhymes and verses for children
    10607, 4901, 4921, 5312, 18546, 20511, 23794, 24623, 26197, 22014, 23350, 24117, 136, 23433, 13646, 13647, 13648,
)  # fmt: skip
READERS = (  # first readers, and The Nursery ("a monthly magazine for youngest readers")
    14642, 14640, 14668, 14766, 14880, 1489, 13853, 15170, 15659, 16936, 6685, 14170, 14335, 14493, 15928, 16522,
    16524, 17536, 19821, 21047, 24474, 24475, 24476, 24477, 24478, 24479, 24938, 24939, 24940, 24941, 24942, 24943,
    28129, 28130, 28131, 28132, 28133,
)  # fmt: skip
SERIES = (  # simple chapter books: the Bobbsey Twins, Bunny Brown and his sister Sue, the Twins of other lands
    714, 737, 5617, 5948, 5952, 6055, 6576, 6950, 15169, 16756, 17412, 18420, 20311, 5732, 16956, 17095, 17096, 17097,
    17878, 18421, 19555, 19565, 20133, 20134, 20309, 3496, 3497, 3642, 3774, 4012, 4086, 4091, 9966, 16644, 6692,
)  # fmt: skip
CLASSICS = (  # the classics read aloud to children: Alice, Peter Pan, Oz, the Jungle Books, Heidi, Pinocchio...
    11, 12, 16, 1332, 55, 54, 485, 486, 517, 419, 420, 955, 956, 957, 958, 959, 960, 961, 45, 74, 113, 146, 157, 225,
    271, 289, 479, 498, 500, 501, 1154, 514, 2788, 708, 709, 764, 770, 794, 778, 836, 837, 1448, 1450, 1874, 2770,
    3536, 5347, 25564, 9407,
)  # fmt: skip
KINDS = {
    "bedtime": BEDTIME,
    "tales": TALES,
    "rhymes": RHYMES,
    "readers": READERS,
    "series": SERIES,
    "classics": CLASSICS,
}
HELD = 20  # one book in this many (of each kind), and one stretch of speech in this many, is held out for tests


@dataclass
class Heard:
    """Passages to learn from and held-out passages, by source ("speech", or a kind of book)."""

    train: dict[str, list[str]] = field(default_factory=dict)
    held: dict[str, list[str]] = field(default_factory=dict)
    books: dict[str, str] = field(default_factory=dict)  # which books it heard: number → title

    def words(self, part: str = "train") -> dict[str, int]:
        return {k: sum(len(p.split()) for p in v) for k, v in getattr(self, part).items()}

    def to_json(self) -> dict:
        return {"train": self.train, "held": self.held, "books": self.books}

    @classmethod
    def from_json(cls, data: dict) -> Heard:
        return cls(data["train"], data["held"], data.get("books", {}))


# --- speech -------------------------------------------------------------------------------------

_I = re.compile(r"\bi(?=\b|')")
_SPACED = re.compile(r" +([.?!,;:])")


def utterance(line: str) -> str:
    """One thing said to a child, as it's written in the transcripts ("i think i see child ." → "I think I see child.")."""
    text = " ".join(line.split())
    if not text or not re.search(r"[a-z]", text):
        return ""
    text = _SPACED.sub(r"\1", text)
    text = _I.sub("I", text)
    text = re.sub(r"\b(xxx|yyy|www)\b", "", text).strip()
    text = re.sub(r"\s{2,}", " ", text)
    if not re.search(r"[a-zA-Z]{2,}|\b[aI]\b", text):
        return ""
    if text[-1] not in ".?!":
        text += "."
    return text[0].upper() + text[1:]


def speech_passages(text: str, size: int = 1500) -> list[str]:
    """Stretches of what was said, a line for each thing said."""
    passages, current, length = [], [], 0
    for line in text.splitlines():
        said = utterance(line)
        if not said:
            continue
        current.append(said)
        length += len(said) + 1
        if length >= size:
            passages.append("\n".join(current))
            current, length = [], 0
    if len(current) > 5:
        passages.append("\n".join(current))
    return passages


# --- books ---------------------------------------------------------------------------------------

_ILLUSTRATION = re.compile(
    r"\[(?:Illustration|Footnote|Transcriber|Sidenote|Note|Decoration)[^\]]*\]", re.IGNORECASE | re.DOTALL
)
_PRODUCED = re.compile(
    r"^(?:Produced by|E-?text prepared by|This e-?book was produced|Transcribed by|Transcriber)", re.IGNORECASE
)
_NOTICE = re.compile(
    r"gutenberg|copyright|act of congress|librarian of congress|table of contents|all rights reserved", re.IGNORECASE
)
_TABLE = re.compile(r"\S {3,}\S")  # a word list, or a table of contents
_PAGE = re.compile(r"(?:\s|\.)\d+\s*$")
_HEADING = re.compile(r"^(?:CHAPTER|LESSON|PART|BOOK|STORY|CONTENTS|ILLUSTRATIONS|LIST OF)\b", re.IGNORECASE)
QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "—": "--", "–": "-", "…": "...", " ": " "})


_SMALL_PRINT = re.compile(r"\*END\*?\s*THE SMALL PRINT!.*?\*END\*", re.IGNORECASE | re.DOTALL)  # an old notice's end
_OLD_END = re.compile(r"\n[ *]*End of (?:the )?Project Gutenberg", re.IGNORECASE)


def clean_book(text: str) -> str:
    """A book's text, without Project Gutenberg's notices, illustrations, tables of contents, word lists or headings."""
    text = text.replace("\r\n", "\n")
    notice = _SMALL_PRINT.search(text)  # (books put online long ago have their notice first, and it ends like this)
    if notice:
        text = text[notice.end() :]
    end = _OLD_END.search(text)
    if end:
        text = text[: end.start()]
    text = _strip_gutenberg(text).translate(QUOTES)
    text = _ILLUSTRATION.sub("", text)
    text = re.sub(r"(?<![\w_])_([^_\n]+(?:\n[^_\n]+)*)_(?![\w_])", r"\1", text)  # _italics_
    text = re.sub(r"(?<![\w=])=([^=\n]+)=(?![\w=])", r"\1", text)  # =bold=
    kept = []
    for paragraph in re.split(r"\n\s*\n", text):
        lines = [line for line in paragraph.splitlines() if line.strip()]
        if not lines:
            continue
        joined = " ".join(" ".join(lines).split())
        letters = [c for c in joined if c.isalpha()]
        if (
            not letters
            or _PRODUCED.match(joined)
            or sum(bool(_TABLE.search(line.strip())) for line in lines) > len(lines) / 2
            or sum(bool(_PAGE.search(line)) for line in lines) > len(lines) / 2
            or (sum(c.isupper() for c in letters) > 0.6 * len(letters) and len(letters) > 3)
            or (_HEADING.match(joined) and len(joined.split()) <= 8)
            or len(joined.split()) < 2
            or _NOTICE.search(joined)
        ):
            continue
        kept.append("\n".join(" ".join(line.split()) for line in lines) if _verse(lines) else joined)
    return "\n\n".join(kept)


def _verse(lines: list[str]) -> bool:
    """Whether a paragraph is verse (its line breaks are part of it): several short lines."""
    return len(lines) >= 2 and max(len(line.strip()) for line in lines) < 60


def catalog(web: Web, cache: Path) -> dict[int, str]:
    """GITenberg's list of books: Project Gutenberg's number → the repository it's kept in."""
    path = cache / "gitenberg.tsv"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        path.write_bytes(web.get(CATALOG, 20_000_000))
    found = {}
    for line in path.read_text(errors="replace").splitlines():
        number, _, repo = line.strip().partition("\t")
        if number.isdigit() and repo:
            found.setdefault(int(number), repo.strip())
    return found


def book(web: Web, repo: str, number: int) -> str:
    """A book's plain text, whichever of the usual files it has."""
    for suffix, encoding in (("", "utf-8"), ("-0", "utf-8"), ("-8", "latin-1")):
        try:
            body = web.get(BOOK.format(repo=repo, file=f"{number}{suffix}.txt"), 12_000_000)
        except WebError:
            continue
        return body.decode(encoding, "replace")
    raise WebError(f"no text for book #{number}")


def title_of(repo: str) -> str:
    """A book's title, from the name of the repository it's kept in ("The-Tale-of-Jimmy-RabbitSleepy-TimeTales_24628"
    is The Tale of Jimmy Rabbit): without its subtitle, or what's cut off."""
    title = re.sub(r"_+\d+$", "", repo).replace("-s-", "'s ").replace("--13-", ": ").replace("-", " ")
    title = re.sub(r"(?<!\bMc)(?<=[a-z])[A-Z][a-z].*$", "", title)  # (a subtitle, run on)
    title = re.split(r": | [Oo]r (?=[A-Z])| b (?=[A-Z])| Complete\b| Being the\b", title)[0]
    return " ".join(title.split())


# --- all of it -----------------------------------------------------------------------------------


def gather(web: Web, cache: Path, kinds: dict[str, tuple[int, ...]] = KINDS, log=print) -> Heard:
    """Download (once) and prepare everything it hears. Books are kept one file each, so it can be resumed."""
    heard = Heard()
    speech_path = cache / "speech.txt"
    if not speech_path.exists():
        log("  hearing people talk to children (CHILDES)…")
        cache.mkdir(parents=True, exist_ok=True)
        speech_path.write_bytes(web.get(SPEECH, 60_000_000))
    stretches = speech_passages(speech_path.read_text(errors="replace"))
    blocks = max(1, len(stretches) // 200)  # held out in whole stretches (things said one after another go together)
    heard.train["speech"] = [p for i, p in enumerate(stretches) if (i // blocks) % HELD != HELD // 2]
    heard.held["speech"] = [p for i, p in enumerate(stretches) if (i // blocks) % HELD == HELD // 2]
    books = catalog(web, cache)
    shelf = cache / "books"
    shelf.mkdir(parents=True, exist_ok=True)
    for kind, numbers in kinds.items():
        train, held = [], []
        for n, number in enumerate(numbers):
            repo = books.get(number)
            if repo is None:
                log(f"  (book #{number} isn't in GITenberg's list)")
                continue
            path = shelf / f"{number}.txt"
            if not path.exists():
                try:
                    path.write_text(clean_book(book(web, repo, number)))
                except WebError as error:
                    log(f"  (couldn't get book #{number}: {error})")
                    continue
            passages = _passages(path.read_text())
            if not passages:
                continue
            heard.books[str(number)] = title_of(repo)
            (held if n % HELD == HELD // 2 else train).extend(passages)
        heard.train[kind], heard.held[kind] = train, held
    return heard


def load(path: Path) -> Heard:
    return Heard.from_json(json.loads(Path(path).read_text()))


def save(heard: Heard, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(heard.to_json()))
    tmp.replace(path)


def mixed(heard: Heard, part: str = "train", seed: int = 0) -> list[str]:
    """Everything it heard, shuffled together (for learning from)."""
    passages = [p for ps in getattr(heard, part).values() for p in ps]
    random.Random(seed).shuffle(passages)
    return passages


# --- hearing at home -----------------------------------------------------------------------------

SHELF = BEDTIME + TALES + RHYMES + SERIES + CLASSICS  # what it hears at bedtime at home: one book after another
STORY_EVERY = 1200.0  # seconds, at least, between bedtime stories (about as many words an hour as a small child hears)
KEEP = 20  # stories it keeps in mind


def repos() -> dict[int, str]:
    """Where GITenberg keeps each book it may hear (shelf.json)."""
    return {int(k): v for k, v in json.loads((Path(__file__).parent / "shelf.json").read_text()).items()}


class Listening:
    """What it hears at home: now and then at bedtime, a passage of a book (one book after another, as a child is read
    to), and what people say to it. How far it has got, and the stories it has in mind, are kept in
    cortex/hearing.json; the books, in cortex/hearing/."""

    def __init__(self, root: Path, web: Web | None = None):
        self.dir = Path(root) / "cortex"
        self.web = web
        self.path = self.dir / "hearing.json"
        self.state: dict = {"book": 0, "passage": 0, "words": 0, "said": 0, "stories": []}
        with contextlib.suppress(OSError, ValueError, TypeError):
            self.state.update(json.loads(self.path.read_text()))
        self.tonight: list[str] = []  # passages it heard since it last slept on them
        self.people: list[str] = []  # what people said to it since then
        self.lock = threading.Lock()

    # --- what it heard ---------------------------------------------------------------------------

    def stories(self) -> list[Story]:
        """The stories it heard at home, the latest last."""
        return [Story(s["title"], tuple(s["opening"]), tuple(s["then"])) for s in self.state["stories"]]

    def latest(self) -> Story | None:
        found = self.stories()
        return found[-1] if found else None

    def words(self) -> int:
        """How many words it has heard at home: stories, and people."""
        return int(self.state["words"]) + int(self.state["said"])

    def heard_said(self, text: str) -> None:
        """Someone said something to it."""
        with self.lock:
            self.people = [*self.people, text][-200:]
            self.state["said"] += len(text.split())

    # --- a bedtime story -----------------------------------------------------------------------

    def bedtime(self) -> tuple[Story, str] | None:
        """The next passage of the book it's hearing (downloading the book first, if it must): the story as it has it
        in mind, and the passage. None if there's nothing to hear (no internet to get the next book)."""
        for _ in range(len(SHELF)):
            with self.lock:
                number = SHELF[self.state["book"] % len(SHELF)]
            passages = self._book(number)  # (it may have to be downloaded)
            if passages is None:
                return None
            with self.lock:
                if self.state["passage"] >= len(passages):
                    self.state["book"], self.state["passage"] = self.state["book"] + 1, 0
                    continue
                passage = passages[self.state["passage"]]
                self.state["passage"] += 1
                story = from_passage(title_of(repos().get(number, str(number))), passage)
                if story is None:
                    continue
                self.state["words"] += len(passage.split())
                entry = {"title": story.title, "opening": list(story.opening), "then": list(story.then)}
                self.state["stories"] = [*self.state["stories"], entry][-KEEP:]
                self.tonight = [*self.tonight, passage][-12:]
                self._save()
                return story, passage
        return None

    def before(self, n: int = 4) -> list[str]:
        """Passages it heard a while ago in the book it's hearing (to go over again, so it doesn't forget them)."""
        passages = self._book(SHELF[self.state["book"] % len(SHELF)]) or []
        done = self.state["passage"]
        return passages[max(0, done - len(self.tonight) - n) : max(0, done - len(self.tonight))]

    def upcoming(self) -> str | None:
        """The next passage of the book it's hearing, which it hasn't heard yet (to test how well it follows)."""
        passages = self._book(SHELF[self.state["book"] % len(SHELF)]) or []
        return passages[self.state["passage"]] if self.state["passage"] < len(passages) else None

    def slept(self) -> None:
        """It went over what it heard, in its sleep."""
        with self.lock:
            self.tonight, self.people = [], []

    def _book(self, number: int) -> list[str] | None:
        path = self.dir / "hearing" / f"{number}.txt"
        if not path.exists():
            if self.web is None or number not in repos():
                return None
            try:
                text = clean_book(book(self.web, repos()[number], number))
            except WebError:
                return []  # (it can't be had: the next one)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return _passages(path.read_text())

    def _save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state))
        tmp.replace(self.path)
