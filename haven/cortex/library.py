"""Everything Haven has read, sentence by sentence, and finding the sentence that answers a question.

Its cortex is small, so it doesn't soak up what it reads the way it learns words for its
own states; it keeps what it read, like a notebook, and when someone asks about something
the sentence that answers them comes to mind (it says where it read it). It keeps the
start of each article: the little book it was born having read, the articles it looked up
because it didn't know something, and the ones it read out of curiosity.
"""

from __future__ import annotations

import contextlib
import json
import math
import re
from pathlib import Path

from .book import BOOK
from .talk import _ADDRESS, PEOPLE, PET_NAMES, _tokens
from .talk import STOP as STOPWORDS

NAMES = frozenset(PEOPLE + PET_NAMES)

KEEP = 12  # sentences it keeps from the start of each article
ASKS_WHAT = re.compile(  # questions about what something is, answered by the first thing it read about it
    r"^(?:what|who)(?:'s| is| are| was| were)\b|^(?:can you )?tell me about\b|^(?:do you know|what do you know) about\b|"
    r"^have you (?:heard|read) (?:of|about)\b|^explain\b|^(?:please )?(?:read|look up|find out) about\b|^look up\b",
    re.I,
)
READ_ASK = re.compile(r"\b(?:read about|look up|find out about)\s+(.+?)[?.!]*$", re.I)


PLAIN = frozenset("have has had get got let like go goes went come came say said thing things".split())
SAME = {  # words that mean the same, for finding what answers a question
    "biggest": "largest",
    "big": "large",
    "huge": "large",
    "tallest": "highest",
    "tall": "high",
    "little": "small",
    "tiny": "small",
    "plane": "airplane",
    "planes": "airplane",
    "kids": "children",
    "kid": "children",
}


def words(text: str) -> set[str]:
    """The words of a sentence that say what it's about, without their endings ("walked" and "walks" are "walk")."""
    found = set()
    text = re.sub(r"(\w{2,})ies\b", r"\1y", text.lower())  # "flies", "berries"
    for word in _tokens(" ".join(SAME.get(w, w) for w in re.findall(r"[\w']+", text))):
        if word in PLAIN:
            continue
        for ending in ("ing", "ed", "es"):
            if len(word) > len(ending) + 3 and word.endswith(ending):
                word = word[: -len(ending)]
                break
        found.add(word)
    return found


def sentences_of(text: str, keep: int = KEEP) -> list[str]:
    """The first sentences of an article, without headings, lists or asides in brackets."""
    body = text.split("\n\n==", 1)[0]  # the lead, before the first section heading
    body = re.sub(r"\s*\([^()]*\)", "", " ".join(body.split()))
    found = []
    for sentence in re.split(r"(?<=[a-z0-9)\"'])[.!?] +(?=[A-Z\"'])", body):
        sentence = sentence.strip().rstrip(".") + "."
        if 12 <= len(sentence) <= 240 and not sentence.startswith(("=", "*", "|")):
            found.append(sentence)
        if len(found) >= keep:
            break
    return found


class Library:
    def __init__(self, root: Path | None = None):
        self.path = None if root is None else Path(root) / "cortex" / "readings.jsonl"
        self.docs: dict[str, list[str]] = {}
        self.sources: dict[str, str] = {}
        self.postings: dict[str, set[tuple[str, int]]] = {}
        self.count = 0  # sentences, for weighing how telling a word is
        self.questions: list[tuple[set[str], str, int]] = []  # questions it knows which sentence answers
        for entry in BOOK:
            self.add(entry.title, list(entry.sentences), "book")
            self.questions += [(words(q), entry.title, i) for q, i in entry.questions]
        self._stamp = None
        self.refresh()

    # --- what it has read ------------------------------------------------------------------------

    def refresh(self) -> None:
        """Take in whatever it has read since (what it looked up is saved to readings.jsonl as it goes)."""
        if self.path is None or not self.path.exists():
            return
        stamp = self.path.stat().st_mtime
        if stamp == self._stamp:
            return
        self._stamp = stamp
        for line in self.path.read_text().splitlines():
            with contextlib.suppress(ValueError, KeyError, TypeError):
                item = json.loads(line)
                self.add(str(item["title"]), str(item["text"]), item.get("why", "read"))

    def add(self, title: str, text: str | list[str], source: str = "read") -> bool:
        """Keep an article (the latest reading of a title replaces an older one). Returns whether it was new."""
        sentences = text if isinstance(text, list) else sentences_of(text)
        if not sentences:
            return False
        new = title not in self.docs
        if not new:
            for i, _ in enumerate(self.docs[title]):
                for word in self._words(title, i):
                    self.postings.get(word, set()).discard((title, i))
            self.count -= len(self.docs[title])
        self.docs[title], self.sources[title] = sentences, source
        self.count += len(sentences)
        for i in range(len(sentences)):
            for word in self._words(title, i):
                self.postings.setdefault(word, set()).add((title, i))
        return new

    def _words(self, title: str, i: int) -> set[str]:
        return words(self.docs[title][i])

    def titles(self, source: str | None = None) -> list[str]:
        return [t for t in self.docs if source is None or self.sources[t] == source]

    def __contains__(self, title: str) -> bool:
        return title in self.docs

    # --- finding what answers a question ------------------------------------------------------------

    def title_for(self, text: str) -> str | None:
        """The article a question is about, by its title: most of the title's words are in the question."""
        asked = words(text)
        best, score = None, 0.0
        for title in self.docs:
            named = words(title)
            if not named:
                continue
            s = len(named & asked) / len(named) + 0.01 * (len(named & asked) - len(named))
            if s > score:
                best, score = title, s
        return best if score >= 0.5 else None

    def find(self, text: str) -> tuple[str, int] | None:
        """The sentence it read that answers a question: (title, which sentence), or None if it read nothing that does."""
        question = _ADDRESS.sub("", " ".join(text.strip().split()), count=1)
        asked = words(question)
        if not asked:
            return None
        for known, doc, i in self.questions:  # a question it has practised: the sentence that answers it
            if doc in self.docs and len(known & asked) >= 0.75 * len(known | asked):
                return doc, i
        title = self.title_for(question)
        topic = words(title) if title else set()
        if title and ASKS_WHAT.search(question) and not asked - topic:  # just what it is
            return title, 0
        best, score = None, 0.0
        for word in asked:
            for doc, i in self.postings.get(word, ()):
                if doc == title:  # in the article it's about: anything more than the name will do
                    overlap = (asked - topic) & self._words(doc, i)
                    enough, bonus = 1, 2.0
                else:  # anywhere else: at least two telling words
                    overlap = asked & self._words(doc, i)
                    enough, bonus = 2, 0.0
                if len(overlap) < enough:
                    continue
                s = sum(self._weight(w) for w in overlap) + bonus - 0.05 * i
                if s > score:
                    best, score = (doc, i), s
        if best is None and title and ASKS_WHAT.search(question):
            return title, 0  # about it, but nothing it read says more exactly
        return best

    def _weight(self, word: str) -> float:
        return math.log(1 + self.count / max(1, len(self.postings.get(word, ()))))

    def sentence(self, title: str, i: int) -> str | None:
        found = self.docs.get(title, [])
        return found[i] if 0 <= i < len(found) else None


# --- what it would like to read about ----------------------------------------------------------------

VALLEY = {  # things in its valley, and what the articles about them are called
    "butterfly": "Butterfly",
    "flower": "Flower",
    "apple": "Apple",
    "tree": "Apple tree",
    "bush": "Berry",
    "mushroom": "Mushroom",
    "toadstool": "Toadstool",
    "pond": "Pond",
    "fire": "Fire",
    "bell": "Bell",
    "ball": "Ball",
    "hill": "Hill",
    "stone": "Rock (geology)",
    "thorns": "Thorn",
    "nest": "Nest",
}
NOT_TOPICS = frozenset(
    "thing things stuff lot lots bit way time times day days today tomorrow yesterday week year years morning night "
    "evening favorite favourite good bad new old big little small nice great best other same few many much more most "
    "very really name job work home place people person question answer something anything everything nothing "
    "someone anyone everyone moment while kind sort type part end start rest idea fun one two three minute hour "
    "haven valley world life trip visit body group member form piece number collection set series lot couple "
    "mom mum dad mother father sister brother wife husband son daughter kids children baby friend friends boyfriend "
    "girlfriend partner grandma grandpa aunt uncle cousin boss neighbor neighbour".split()
)
_AFTER = re.compile(
    r"\b(?:a|an|the|my|your|some|about|like|love|likes|loves)\s+(?:(?:big|little|small|new|old|favou?rite|pet|"
    r"good|best|lovely)\s+)?([a-z]{3,})\b"
)
_FAVORITE = re.compile(r"\bfavou?rite \w+ (?:is|are) (?:a |an |the )?([a-z]{3,}(?: [a-z]{3,})?)\b")
_PROPER = re.compile(r"(?<=[a-z,;:] )([A-Z][a-z]+(?: [A-Z][a-z]+)*)")
_NAMING = re.compile(r"\b(?:named|called|name is|name's|i'm|i am|call me|this is|with|met)\s*$", re.I)
_WHAT_IT_IS = re.compile(
    r"\b(?:is|are|was|were) (?:a |an |the )?(?:kind of |type of |sort of )?((?:[A-Za-z-]+ ){0,5}?[a-z-]+)"
    r"(?= in\b| of\b| that\b| which\b| with\b| from\b| and\b| by\b| to\b| for\b|[,.;])"
)


def topics_in(text: str, skip: set[str] = frozenset()) -> list[str]:
    """What someone mentioned that it could read about: names of places and things in capitals, and things after
    "a", "the", "my"… (not people's names, and not `skip`)."""
    found = [
        m.group(1)
        for m in _PROPER.finditer(text)
        if not _NAMING.search(text[: m.start()]) and m.group(1) not in NAMES  # not someone's name
    ]
    found += [" ".join(w for w in f.split() if w not in STOPWORDS) for f in _FAVORITE.findall(text.lower())]
    found += [w for w in _AFTER.findall(text.lower()) if w not in NOT_TOPICS and w not in STOPWORDS]
    topics: dict[str, str] = {}
    for topic in found:
        within = any(topic.lower() in other.split() for other in topics)  # "rolling", of "Rolling Stones"
        if topic.lower() not in skip and topic.lower() not in ("haven", "i") and not within:
            topics.setdefault(topic.lower(), topic)
    return list(topics.values())


def what_it_is(sentence: str) -> str | None:
    """What a first sentence says something is, in a word ("Butterflies are insects in the order…": "insects")."""
    found = _WHAT_IT_IS.search(sentence)
    if not found:
        return None
    word = found.group(1).split()[-1]
    return None if word in NOT_TOPICS or word in STOPWORDS or len(word) < 3 else word


def asked_to_read(text: str) -> str | None:
    """What someone is asking it to read about ("read about volcanoes"), if they are."""
    found = READ_ASK.search(" ".join(text.strip().split()))
    return found.group(1).strip() if found else None
