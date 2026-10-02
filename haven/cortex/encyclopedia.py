"""Whole encyclopedias for Haven to read: getting them, and turning articles into sentences it can say.

The Wikipedia it reads is the plain text of every article, as prepared by TensorFlow
Datasets from Wikipedia's own dumps and kept in a public bucket on Google Cloud Storage:
the Simple English Wikipedia (231,282 articles, 284 MB) and, for the patient, the whole
English Wikipedia (6.7 million articles, 21 GB). Each comes in files of records, each
record one article: its title and its text (see `articles`). Nothing here needs anything
beyond Python itself.

An article's text is paragraphs, with headings on lines of their own; `prose` keeps the
paragraphs and drops the headings, lists, tables, categories and the sections at the end
(references, other websites). `sentences` splits a paragraph into sentences without
splitting at "Dr." or "J. R. R.", and `speakable` makes a sentence plain enough to say:
no asides in brackets, no stray spaces.
"""

from __future__ import annotations

import re
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from typing import BinaryIO

# --- the encyclopedias -----------------------------------------------------------------------------------------

BUCKET = "https://storage.googleapis.com/tfds-data/datasets/wikipedia"


@dataclass(frozen=True)
class Encyclopedia:
    name: str  # what Haven's library calls it
    title: str  # what people call it
    folder: str  # where its files are, in the bucket
    files: int  # how many files of records it comes in
    articles: int
    size: int  # bytes to download
    keep: int  # paragraphs it keeps of each article (the start of each article holds most of what it's about)
    about: str = ""  # how long it takes, and how much room it needs
    url: str = ""  # (a dictionary: one file, WordNet's)

    def urls(self) -> list[str]:
        if self.url:
            return [self.url]
        return [
            f"{BUCKET}/{self.folder}/wikipedia-train.tfrecord-{i:05d}-of-{self.files:05d}" for i in range(self.files)
        ]


SIMPLE = Encyclopedia(
    "simple", "the Simple English Wikipedia", "20230601.simple/1.0.0", 4, 231_282, 283_905_360, 12,
    "a few minutes; about 300 MB on disk",
)  # fmt: skip
ENGLISH = Encyclopedia(
    "english", "the English Wikipedia", "20230601.en/1.0.0", 256, 6_200_000, 21_454_958_141, 3,
    "hours; about 10 GB on disk",
)  # fmt: skip
DICTIONARY = Encyclopedia(
    "dictionary", "WordNet, a dictionary of English", "", 1, 147_000, 11_058_667, 0, "under a minute; about 70 MB on disk",
    "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet31.zip",
)  # fmt: skip
ENCYCLOPEDIAS = {e.name: e for e in (DICTIONARY, SIMPLE, ENGLISH)}


# --- reading the records ---------------------------------------------------------------------------------------


def _varint(data: bytes, i: int) -> tuple[int, int]:
    shift = value = 0
    while True:
        byte = data[i]
        i += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, i
        shift += 7


def _fields(data: bytes) -> Iterator[tuple[int, bytes | int]]:
    """The fields of a protocol buffer message: (field number, value)."""
    i, n = 0, len(data)
    while i < n:
        key, i = _varint(data, i)
        number, kind = key >> 3, key & 7
        if kind == 2:  # bytes, strings and messages
            length, i = _varint(data, i)
            yield number, data[i : i + length]
            i += length
        elif kind == 0:
            value, i = _varint(data, i)
            yield number, value
        elif kind == 5:
            yield number, data[i : i + 4]
            i += 4
        elif kind == 1:
            yield number, data[i : i + 8]
            i += 8
        else:
            raise ValueError(f"not a record Haven can read (wire type {kind})")


def example(data: bytes) -> dict[str, list[bytes]]:
    """A tf.train.Example: each feature's list of byte strings (titles and texts are byte strings)."""
    found: dict[str, list[bytes]] = {}
    for number, features in _fields(data):
        if number != 1 or not isinstance(features, bytes):
            continue
        for number, entry in _fields(features):  # the map of features: (name, feature) entries
            if number != 1 or not isinstance(entry, bytes):
                continue
            name, feature = "", b""
            for number, value in _fields(entry):
                if number == 1 and isinstance(value, bytes):
                    name = value.decode("utf-8", "replace")
                elif number == 2 and isinstance(value, bytes):
                    feature = value
            values = []
            for number, listed in _fields(feature):
                if number == 1 and isinstance(listed, bytes):  # a list of byte strings
                    values += [v for n, v in _fields(listed) if n == 1 and isinstance(v, bytes)]
            found[name] = values
    return found


def records(stream: BinaryIO) -> Iterator[bytes]:
    """The records in a TFRecord file: a length, a checksum of it, the record, a checksum of that."""
    while True:
        head = stream.read(12)
        if len(head) < 12:
            return
        (length,) = struct.unpack("<Q", head[:8])
        data = stream.read(length)
        if len(data) < length:
            return  # (cut short)
        stream.read(4)
        yield data


def articles(stream: BinaryIO) -> Iterator[tuple[str, str]]:
    """(title, text) of each article in a file of records."""
    for record in records(stream):
        found = example(record)
        if found.get("title") and found.get("text"):
            yield found["title"][0].decode("utf-8", "replace"), found["text"][0].decode("utf-8", "replace")


# --- turning an article into sentences -------------------------------------------------------------------------

LAST_SECTIONS = re.compile(
    r"^(?:references|notes|sources|other websites|external links|related pages|see also|further reading|"
    r"bibliography|footnotes|citations|gallery|discography|filmography|works cited)\b",
    re.I,
)
_ENDS = re.compile(r"[.!?][\"'”’)\]]*$")
_SENTENCE_END = re.compile(r"[a-z)][.!?] +[A-Z]")  # (a paragraph whose last sentence ends in a stray reference mark)


def prose(text: str, keep: int | None = None) -> list[str]:
    """The paragraphs of an article (the first `keep`), each one line, without headings, lists or what comes at the end."""
    found: list[str] = []
    lines: list[str] = []

    def close() -> bool:  # (whether that's all it keeps)
        if lines:
            found.append(" ".join(lines))
            lines.clear()
        return keep is not None and len(found) >= keep

    for line in text.splitlines():
        line = " ".join(line.split())
        if line.startswith("Category:"):
            continue
        if not line:
            if close():
                break
            continue
        words = len(line.split())
        if (not _ENDS.search(line) or words < 3) and (words < 12 or not _SENTENCE_END.search(line)):  # a heading…
            if close() or LAST_SECTIONS.match(line):
                break
            continue
        lines.append(line)
    else:
        close()
    return found[:keep] if keep is not None else found


ABBREVIATIONS = frozenset(
    "mr mrs ms dr st jr sr mt ft vs etc no nos inc ltd co corp approx est gen gov lt col sgt capt prof rev hon pres sen "
    "rep jan feb mar apr jun jul aug sep sept oct nov dec fig vol op ca cf al bros dept univ assn ave blvd".split()
)
_SPLIT = re.compile(r"[.!?][\"'”’)\]]*(?=\s+[\"'“‘(\[]?[A-Z0-9])")


def sentences(paragraph: str) -> list[str]:
    """A paragraph's sentences (not split at "Dr. Smith", "J. R. R. Tolkien", "U.S. Navy" or "St. Louis")."""
    text, parts, start = paragraph.strip(), [], 0
    for end in _SPLIT.finditer(text):
        parts.append(text[start : end.end()])
        start = end.end()
    parts = [p.strip() for p in [*parts, text[start:]] if p.strip()]
    found: list[str] = []
    for part in parts:
        if found:
            last = found[-1].rstrip("\"'”’)]").rstrip(".")
            word = re.split(r"[\s(]", last)[-1] if last else ""
            if word.lower() in ABBREVIATIONS or re.fullmatch(r"(?:[A-Z]\.)*[A-Z]", word):  # "Dr.", "J.", "U.S."
                found[-1] += " " + part
                continue
        found.append(part)
    return [s.strip() for s in found]


_ASIDE = re.compile(r"\s*\([^()]*\)")
_ABOUT_ITSELF = re.compile(  # what an encyclopedia says about its own pages, not about the world
    r"^(?:This (?:article|page|list|is a list)\b|For other uses\b|.{0,80}\b(?:may|can) (?:also )?refer to\b|"
    r".{0,60}\bis a disambiguation\b|For (?:the |other |more |a )?.{0,80}\bsee\b|See also\b|"
    r"(?:The|A|An) (?:is|was|are|were|has|had)\b)",  # (and "The is a museum…": a name was in a script it couldn't keep)
    re.I,
)
_SPEAKABLE_START = re.compile(r"^[\"“'‘]?[A-Z0-9]")


def speakable(sentence: str) -> str | None:
    """A sentence as Haven would say it: no asides in brackets (dates of birth, other names, how to say it), no
    stray spaces; or None if what's left isn't a sentence to say."""
    text = sentence
    for _ in range(3):  # (brackets in brackets)
        text = _ASIDE.sub("", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", " ".join(text.split()))
    text = re.sub(r",(?=[,.;:])", "", text)  # ("Saturn, , is" when what was in between had gone)
    text = re.sub(r"[\[\]{}]|\s'\s", " ", text).strip()
    text = " ".join(text.split())
    if not _SPEAKABLE_START.match(text) or not 12 <= len(text) <= 320 or not _ENDS.search(text):
        return None
    if _ABOUT_ITSELF.match(text):
        return None
    if text.count('"') % 2 or text.count("(") != text.count(")"):
        return None
    return text


def lead(text: str, keep: int = 12) -> list[str]:
    """The first sentences of an article that can be said, from its first paragraphs."""
    found: list[str] = []
    for paragraph in prose(text, keep=4):
        for sentence in sentences(paragraph):
            said = speakable(sentence)
            if said:
                found.append(said)
            if len(found) >= keep:
                return found
    return found


# --- what an article is about ----------------------------------------------------------------------------------

JOBS = (
    "actor actress singer songwriter rapper musician composer pianist guitarist drummer politician president leader "
    "king queen emperor empress prince princess pharaoh ruler monarch writer author poet novelist playwright "
    "journalist painter artist sculptor architect photographer scientist physicist chemist biologist mathematician "
    "astronomer philosopher economist historian inventor engineer explorer businessman businesswoman entrepreneur "
    "lawyer judge jurist soldier general admiral director producer comedian model dancer footballer player cricketer "
    "athlete boxer wrestler swimmer cyclist racer driver skater skier coach manager priest bishop pope saint monk nun "
    "teacher professor doctor nurse surgeon designer chef broadcaster presenter host activist minister governor mayor "
    "senator diplomat theologian prophet conqueror commander"
).split()
PERSON = re.compile(r"\b(?:is|was) (?:(?:an?|the) )?(?:[\w'-]+ ){0,5}?(" + "|".join(JOBS) + r")s?\b", re.I)
PLACE = re.compile(
    r"\bis (?:an?|the) (?:[\w'-]+ ){0,3}?(?:city|town|village|commune|municipality|country|state|province|region|"
    r"island|river|lake|mountain|county|district|capital|suburb|borough|parish|prefecture|department|canton|"
    r"archipelago|sea|desert|volcano|peninsula|bay|neighbourhood|neighborhood|settlement|hamlet)\b"
)
_REGNAL = re.compile(r"^(.+?) (?:[IVX]+|the Great|the Elder|the Younger)$")


def other_names(title: str, first: str) -> list[str]:
    """Shorter names people use for what an article is about, from its first sentence: a person's surname
    ("Einstein"), a ruler's name without the number ("Cleopatra")."""
    base = re.sub(r"\s*\([^()]*\)", "", title).strip()
    found = []
    if PERSON.search(first[:200]):
        regnal = _REGNAL.match(base)
        if regnal:
            found.append(regnal.group(1))
        elif len(base.split()) >= 2 and base.split()[-1][:1].isupper():
            found.append(base.split()[-1])
    return found


# --- a dictionary: what words mean -------------------------------------------------------------------------------

_PARTS = {"noun": "n", "verb": "v", "adj": "a", "adv": "r"}
_SENSE_PART = {"1": "n", "2": "v", "3": "a", "4": "r", "5": "a"}  # (in WordNet's sense keys; 5: adjectives too)
_COUNTED = re.compile(r"^(?:a|an|the|any|one|some|someone|something|somebody)\b", re.I)


def _lines(archive, name: str):
    import io

    with archive.open(name) as f:
        yield from io.TextIOWrapper(f, encoding="utf-8", errors="replace")


def dictionary(archive) -> Iterator[tuple[str, list[str], list[str]]]:
    """What each word in WordNet means (Princeton University's dictionary of English, WordNet 3.1): (a title, sentences
    saying what it means, its commonest meanings first, the names it's looked up by), from WordNet's zip file."""
    files = {name.rsplit("/", 1)[-1]: name for name in archive.namelist()}
    meant: dict[tuple[str, str], tuple[str, str, list[str]]] = {}  # (part of speech, synset): (meaning, example, words)
    shown: dict[str, str] = {}  # how each word is written ("Einstein", not "einstein")
    for part, letter in _PARTS.items():
        for line in _lines(archive, files[f"data.{part}"]):
            if line.startswith("  "):  # (the licence, at the top)
                continue
            head, _, gloss = line.partition(" | ")
            fields = head.split()
            words = [re.sub(r"\(.*\)$", "", fields[4 + 2 * i]) for i in range(int(fields[3], 16))]
            for word in words:
                shown.setdefault(word.lower(), word)
            pieces = [p.strip() for p in gloss.strip().split("; ")]
            definition = next((p for p in pieces if not p.startswith('"')), "")
            example = next((p.strip('"') for p in pieces if p.startswith('"')), "")
            meant[(letter, fields[0])] = (definition, example, [w.replace("_", " ") for w in words])
    tagged: dict[tuple[str, str, str], int] = {}  # how often each meaning of each word was found in real text
    for line in _lines(archive, files["index.sense"]):
        key, offset, _, count = line.split()
        lemma, _, rest = key.partition("%")
        tagged[(lemma, _SENSE_PART.get(rest[:1], "n"), offset)] = int(count)
    senses: dict[str, list[tuple[int, int, str, str]]] = {}
    for part, letter in _PARTS.items():
        for line in _lines(archive, files[f"index.{part}"]):
            if line.startswith("  "):
                continue
            fields = line.split()
            lemma, pointers = fields[0], int(fields[3])
            for order, offset in enumerate(fields[6 + pointers :]):
                count = tagged.get((lemma, letter, offset), 0)
                senses.setdefault(lemma, []).append((-count, order, letter, offset))
    for lemma, found in senses.items():
        word = shown.get(lemma, lemma).replace("_", " ")
        said = []
        same: list[str] = []
        for _, _, letter, offset in sorted(found)[:3]:
            definition, example, words = meant.get((letter, offset), ("", "", []))
            if not definition:
                continue
            same += [w for w in words if w.lower() != word.lower() and w not in same]
            if said:
                sentence = f"{word[0].upper() + word[1:]} can also mean {definition}."
            elif letter == "v":
                sentence = f"To {word} means to {definition}."
            elif letter == "n" and _COUNTED.match(definition) and word[:1].islower():
                sentence = f"{'An' if word[0] in 'aeiou' else 'A'} {word} is {definition}."
            elif letter == "n":
                sentence = f"{word[0].upper() + word[1:]} is {definition}."
            else:
                sentence = f"{word[0].upper() + word[1:]} means {definition}."
            said.append(sentence)
            if len(said) == 1 and example:
                said.append(f'For example: "{example[0].upper() + example[1:]}."')
        if same:  # (other words that mean the same)
            said.append(f"{word[0].upper() + word[1:]} means the same as {' or '.join(same[:3])}.")
        sentences = [s for s in (speakable(s) for s in said) if s]
        if sentences:
            yield f"the word {word}", sentences, [lemma.replace("_", " ").lower()]
