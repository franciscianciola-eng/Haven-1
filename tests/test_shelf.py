"""Haven's shelf: whole encyclopedias and what it's given to read, kept on this computer and searched for the answer."""

import io
import json
import threading
import time
import urllib.request

import pytest
from fakes import ARTICLES, tfrecord

from haven.cortex import encyclopedia
from haven.cortex.encyclopedia import articles, lead, other_names, prose, sentences, speakable
from haven.cortex.shelf import Shelf, page_text, plain_name, shelf_path, title_for
from haven.feeding import FeedError, Feeding
from haven.web import Web

NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# --- turning articles into sentences ---------------------------------------------------------------------------


def test_it_reads_the_records_wikipedia_comes_in():
    found = list(articles(io.BytesIO(tfrecord(ARTICLES[:3]))))
    assert found == ARTICLES[:3]
    assert list(articles(io.BytesIO(tfrecord(ARTICLES[:2])[:-30]))) == ARTICLES[:1]  # (a file cut short: what's whole)


def test_it_keeps_the_paragraphs_not_the_headings_or_what_comes_at_the_end():
    text = dict(ARTICLES)["France"]
    kept = prose(text)
    assert kept[0].startswith("France (officially") and len(kept) == 3
    assert not any("Category:" in p or p == "History" or p == "References" for p in kept)
    spider = prose(dict(ARTICLES)["Spider"])  # (a paragraph that ends with a stray reference mark is still one)
    assert spider[0].startswith("Spiders (order Araneae)")


def test_it_splits_sentences_where_they_end():
    said = sentences(
        'Dr. Smith went to St. Louis. He met J. R. R. Tolkien there. It was 3.5 km away! "Wow." Then he left.'
    )
    assert said == [
        "Dr. Smith went to St. Louis.",
        "He met J. R. R. Tolkien there.",
        "It was 3.5 km away!",
        '"Wow."',
        "Then he left.",
    ]


def test_it_says_sentences_plainly_and_leaves_out_what_cant_be_said():
    assert speakable("Albert Einstein (14 March 1879 – 18 April 1955) was a German-born scientist.") == (
        "Albert Einstein was a German-born scientist."
    )
    assert speakable("This article is about the French capital.") is None  # (about the encyclopedia itself)
    assert speakable("For other uses, see Paris (disambiguation).") is None
    assert speakable("The is a museum about money in Japan.") is None  # (a name it couldn't keep was there)
    assert speakable("was a town in Japan.") is None
    assert lead(dict(ARTICLES)["Albert Einstein"], 2) == [
        "Albert Einstein was a German-born American scientist.",
        "He worked on theoretical physics.",
    ]


def test_people_have_shorter_names():
    assert other_names("Albert Einstein", "Albert Einstein was a German-born American scientist.") == ["Einstein"]
    assert other_names("Cleopatra VII", "Cleopatra was Queen of the Ptolemaic Kingdom of Egypt.") == ["Cleopatra"]
    assert other_names("Mount Everest", "Mount Everest is the highest mountain on Earth.") == []
    assert plain_name("Rock (geology)") == "rock" and plain_name("Évian-les-Bains") == "evian les bains"


# --- the shelf -------------------------------------------------------------------------------------------------


@pytest.fixture
def shelf(tmp_path):
    shelf = Shelf(shelf_path(tmp_path))
    parts = [tfrecord(ARTICLES[i::4]) for i in range(4)]
    which = encyclopedia.SIMPLE
    assert shelf.read_encyclopedia(which, lambda part: io.BytesIO(parts[part]))
    yield shelf
    shelf.close()


@pytest.mark.parametrize(
    ("question", "title", "says"),
    [
        ("What is the capital of France?", None, "The capital of France is Paris."),  # (France, or Capital of France)
        ("who was albert einstein", "Albert Einstein", "Albert Einstein was a German-born American scientist."),
        ("Who was Einstein?", "Albert Einstein", "Albert Einstein was a German-born American scientist."),
        ("how many legs does a spider have", "Spider", "They have eight legs, and fangs that inject venom."),
        ("when did world war 2 end", "World War II", "The war ended with an Allied victory in 1945."),
        (
            "who wrote romeo and juliet",
            "Romeo and Juliet",
            "Romeo and Juliet is a play written by William Shakespeare.",
        ),
        ("What's the tallest mountain?", "Mount Everest", "Mount Everest is the highest mountain on Earth."),
        ("what is a cat", "Cat", "Cats are small, carnivorous mammals."),
        ("How many people live in Rome?", "Rome", "About 2.8 million people live in Rome."),
        ("What do koalas eat?", "Koala", "They eat leaves of eucalyptus trees."),  # (about them, not just "eat")
        ("When was Albert Einstein born?", "Albert Einstein", "Einstein was born in Ulm in 1879."),  # (not German-born)
        ("When did Elvis Presley die?", "Elvis Presley", "He died on August 16, 1977."),  # (asked when: a date)
        ("What do pandas eat?", "Giant panda", "The panda's diet is mostly bamboo."),  # (not a Mr Panda)
        ("What is the moon made of?", "Moon", "It is made of rock and dust."),  # (not Mr Moon: "moon" isn't a name)
        ("What do tigers eat?", "Tiger", "Tigers eat many types of prey, mostly large mammals."),  # (not "feeds")
        ("When did the Titanic sink?", "Titanic", "It sank on 15 April 1912 after it hit an iceberg."),  # (sank)
        (
            "Who was Martin Luther King?",
            "Martin Luther King Jr.",
            "Martin Luther King Jr. was an American minister and civil rights activist.",
        ),
    ],
)
def test_it_finds_the_sentence_that_answers(shelf, question, title, says):
    found = shelf.find(question)
    assert found is not None and found.said == says and title in (None, found.title)


def test_asked_to_tell_about_something_it_tells_two_sentences(shelf):
    found = shelf.find("Tell me about Paris")
    assert found.title == "Paris" and found.count == 2
    assert found.said == "Paris is the capital and largest city of France. The river Seine flows through it."


def test_it_knows_what_a_plural_is_of(shelf):
    from haven.cortex.shelf import singulars

    assert {"volcano", "potato", "wolf", "mouse", "child", "leaf"} <= set().union(
        *map(singulars, ("volcanoes", "potatoes", "wolves", "mice", "children", "leaves"))
    )
    found = shelf.find("Tell me about volcanoes")  # (Volcano, not Volcanoes in Iceland)
    assert found.title == "Volcano" and found.said.startswith("A volcano is a mountain")


def test_only_questions_about_the_world(shelf):
    for said in ("I like pizza", "how are you?", "what time is it", "what are you doing?", "Paris"):
        assert shelf.find(said) is None, said


def test_what_it_was_given_it_can_answer_from_and_forget(shelf):
    shelf.add("My dog", "My dog is called Rex. Rex is a brown dog who loves to swim in the lake.", "given")
    assert shelf.find("What does Rex love?").title == "My dog"
    assert shelf.titles("given") == ["My dog"]
    assert shelf.forget("My dog") == 1
    assert shelf.titles("given") == [] and shelf.article("My dog") is None
    assert shelf.find("What does Rex love?") is None or shelf.find("What does Rex love?").title != "My dog"


def test_reading_stops_and_carries_on_where_it_was(tmp_path):
    shelf = Shelf(shelf_path(tmp_path))
    parts = [tfrecord(ARTICLES[i::4]) for i in range(4)]
    stop = threading.Event()

    def opener(part):
        if part == 2:
            stop.set()  # (they stop it while it reads the third part)
        return io.BytesIO(parts[part])

    assert not shelf.read_encyclopedia(encyclopedia.SIMPLE, opener, stop=stop)
    first = shelf.counts()
    assert first["shelves"][0]["done"] == 2 and 0 < first["articles"] < len(ARTICLES)
    assert shelf.read_encyclopedia(encyclopedia.SIMPLE, lambda part: io.BytesIO(parts[part]))
    counts = shelf.counts()
    assert counts["shelves"][0]["done"] == 4 and counts["articles"] == len(ARTICLES)
    assert counts["shelves"][0]["finished"]


def test_web_pages_are_read_without_menus_or_scripts():
    from fakes import PAGE

    title, text = page_text(PAGE, "https://example.org/bread")
    assert title == "Bread - a page"
    assert text.startswith("Bread is a food made from flour") and "var x" not in text and "Contact" not in text
    assert title_for("Bread\n\nBread is a food.") == "Bread"
    assert title_for("Bread is a food made from flour, water and yeast.").startswith("Bread is a food made")


# --- feeding it: in the background, from the internet ---------------------------------------------------------


@pytest.fixture
def feeding(tmp_path, internet, monkeypatch):
    import dataclasses

    monkeypatch.setattr(encyclopedia, "BUCKET", internet["tfds"])
    fake = dataclasses.replace(encyclopedia.DICTIONARY, url=internet["wordnet"])
    monkeypatch.setitem(encyclopedia.ENCYCLOPEDIAS, "dictionary", fake)
    events = []
    feeding = Feeding(tmp_path, Web(delay=0, allow_private=True), log=lambda s: None, emit=events.append)
    feeding.events = events
    yield feeding
    feeding.stop(pause=False)
    feeding.wait(10)


def test_the_first_time_it_reads_a_dictionary_and_the_simple_english_wikipedia_by_itself(feeding):
    assert feeding.start()
    feeding.wait(30)
    status = feeding.status()
    assert status["by source"] == {"dictionary": 5, "simple": len(ARTICLES)} and status["reading"] is None
    simple = next(e for e in status["encyclopedias"] if e["name"] == "simple")
    assert simple["finished"] and simple["read"] == len(ARTICLES)
    assert feeding.events[0] == "started reading WordNet, a dictionary of English"
    assert feeding.events[-1].startswith("finished reading the Simple English Wikipedia")
    assert not feeding.start()  # (it has read them: it doesn't read them again)
    shelf = feeding.shelf
    assert shelf.find("What does ubiquitous mean?").said == "Ubiquitous means being present everywhere at once."
    assert shelf.find("what does dog mean").said == "A dog is a member of the genus Canis."  # (asked what it means)
    assert shelf.find("What is a dog?").title == "the word dog"  # (only its dictionary has dogs, here)
    assert shelf.article("the word dog")[1][-1] == "Dog means the same as domestic dog."
    assert shelf.find("Who was Einstein?").title == "Albert Einstein"  # (the encyclopedia's, not the dictionary's)


def test_a_whole_encyclopedia_can_be_taken_off_the_shelf(feeding):
    assert feeding.read("dictionary")
    feeding.wait(30)
    feeding.give("Owls can turn their heads a long way round.", "Owls")
    assert feeding.shelf.forget_all("dictionary") == 5
    assert feeding.status()["by source"] == {"given": 1}
    assert not next(e for e in feeding.status()["encyclopedias"] if e["name"] == "dictionary")["finished"]


def test_paused_it_doesnt_start_again_by_itself(feeding):
    feeding.stop()
    assert not feeding.start()
    assert feeding.read("simple")  # (unless they ask it to)
    feeding.wait(30)
    assert feeding.status()["articles"] == len(ARTICLES)


def test_what_its_given_while_it_reads_an_encyclopedia_takes_rows_of_its_own(tmp_path):
    shelf = Shelf(shelf_path(tmp_path))
    parts = [tfrecord(ARTICLES[i::4]) for i in range(4)]
    given = []

    def opener(part):  # (someone gives it something to read while it reads each part)
        given.append(shelf.add(f"Notes {part}", f"Note number {part} is about owls and their big eyes."))
        return io.BytesIO(parts[part])

    assert shelf.read_encyclopedia(encyclopedia.SIMPLE, opener)
    assert given == [1, 1, 1, 1] and shelf.counts()["articles"] == len(ARTICLES) + 4
    assert sorted(p.name for p in shelf.folder.glob("*.sqlite")) == ["given.sqlite", "simple.sqlite"]  # (each its own)
    for volume in shelf.volumes.values():
        rows = volume.db.execute("SELECT first, last FROM articles ORDER BY first").fetchall()
        assert all(a[1] < b[0] for a, b in zip(rows, rows[1:], strict=False))  # (no two share a row)
    assert shelf.find("What is note number 2 about?").title == "Notes 2"
    assert shelf.forget_all("simple") == len(ARTICLES)  # (a whole encyclopedia goes at once, and its file with it)
    assert not (shelf.folder / "simple.sqlite").exists() and shelf.counts()["articles"] == 4


def test_it_reads_what_its_given_and_web_pages(feeding, internet):
    title, kept = feeding.give("Grandma's soup\n\nGrandma's soup has carrots, leeks and barley. It cooks for hours.")
    assert (title, kept) == ("Grandma's soup", 2)
    title, kept = feeding.page(internet["page"])
    assert title == "Bread - a page" and kept == 4
    assert feeding.shelf.find("What is sourdough bread made with?").title == "Bread - a page"
    assert feeding.status()["given"] == ["Grandma's soup", "Bread - a page"]
    with pytest.raises(FeedError):
        feeding.give("   ")
    assert any(e.startswith("read the page Bread") for e in feeding.events)


def test_without_the_internet_it_reads_only_what_its_given(tmp_path):
    feeding = Feeding(tmp_path, None, log=lambda s: None)
    assert not feeding.start()
    with pytest.raises(FeedError):
        feeding.read("simple")
    with pytest.raises(FeedError):
        feeding.page("https://example.org")
    assert feeding.give("Owls can turn their heads a long way round.")[1] == 1


# --- what comes to mind when it's asked ---------------------------------------------------------------------------


def test_what_it_read_on_its_shelf_comes_to_mind(shelf, tmp_path):
    from haven.cortex.think import Thinker
    from haven.mind import Mind

    thinker = Thinker(tmp_path)
    thinker.shelf = shelf
    mind = Mind(seed=3)
    heard = []
    thinker.listener = lambda kind, text: heard.append((kind, text))
    assert thinker.recollect(mind, "Where is Mount Everest?") == [
        "I read about Mount Everest: Mount Everest is the highest mountain on Earth."
    ]
    assert ("found", "Mount Everest") in heard
    assert thinker.recollect(mind, "Tell me more.") == [  # (and it has it in mind now, to tell more of)
        "I read about Mount Everest: Mount Everest is the highest mountain on Earth.",
        "More that I read about Mount Everest: It is in the Himalayas, on the border of Nepal and China.",
    ]
    assert thinker.recollect(mind, "When did World War 2 end?") == [
        "I read about World War II: The war ended with an Allied victory in 1945."
    ]
    assert thinker.recollect(mind, "What is the capital of France?") == [  # (its little book first: what it practised)
        "I read about France: Its capital city is Paris."
    ]
    assert thinker.recollect(mind, "Tell me about Romeo and Juliet") == [
        "I read about Romeo and Juliet: Romeo and Juliet is a play written by William Shakespeare. It is a tragedy."
    ]
    assert thinker.recollect(mind, "What is the moon?") == [  # (its little book first: what it practised)
        "I read about Moon: The Moon is the Earth's only natural satellite."
    ]


def test_he_or_they_just_after_it_told_them_something_is_what_it_told(shelf, tmp_path):
    from haven.cortex.think import Thinker, resolved
    from haven.mind import Mind

    assert resolved("When was he born?", "Albert Einstein") == "When was Albert Einstein born?"
    assert resolved("What is its capital?", "France") == "What is France's capital?"
    assert resolved("Where do they live?", "Cat (zodiac)") == "Where do Cat live?"
    thinker = Thinker(tmp_path)
    thinker.shelf = shelf
    mind = Mind(seed=3)
    assert thinker.recollect(mind, "Who was Elvis Presley?") == [
        "I read about Elvis Presley: Elvis Presley was an American singer and actor."
    ]
    assert thinker.recollect(mind, "When did he die?") == ["I read about Elvis Presley: He died on August 16, 1977."]


def test_what_it_read_comes_to_mind_in_plain_letters_and_valley_words_can_name_other_things(shelf, tmp_path):
    from haven.cortex.think import Thinker
    from haven.mind import Mind

    thinker = Thinker(tmp_path)
    thinker.shelf = shelf
    mind = Mind(seed=3)
    assert thinker.recollect(mind, "Where is Wieluń?") == [  # (an accent is hard for it to say back)
        "I read about Wielun: Wielun is a town in Lodz Voivodeship, in the middle of Poland."
    ]
    assert thinker.recollect(mind, "What is Apple Macintosh?") == [  # (not the apples in its valley)
        "I read about Apple Macintosh: The Apple Macintosh is a line of personal computers made by Apple."
    ]
    assert thinker.recollect(mind, "What do apples taste like?") == []  # (that it knows from its valley)


def test_what_it_read_on_its_shelf_takes_the_place_of_its_little_books_telling(shelf, tmp_path):
    from haven.cortex.think import Thinker
    from haven.mind import Mind

    thinker = Thinker(tmp_path)
    thinker.shelf = shelf
    mind = Mind(seed=3)
    assert thinker.library.find("What is lava?") == ("Volcano", 1)  # (practised, from its little book)
    assert thinker.recollect(mind, "Tell me about volcanoes") == [
        "I read about Volcano: A volcano is a mountain that has lava coming out of it. Volcanoes are formed by the "
        "movement of tectonic plates."
    ]
    assert thinker.library.sources["Volcano"] == "shelf"  # (what it told is what it tells more of)
    assert all(doc != "Volcano" for _, doc, _ in thinker.library.questions)  # (its practised questions were about the book's)


# --- the app's library -----------------------------------------------------------------------------------------------


def test_the_app_lets_people_feed_it(tmp_path, internet, monkeypatch):
    pytest.importorskip("torch")
    from haven.app import Chat, serve
    from haven.life import Life
    from haven.mind import Mind
    from haven.store import Store

    monkeypatch.setattr(encyclopedia, "BUCKET", internet["tfds"])
    life = Life(Mind(seed=4), Store(tmp_path), speed=50)
    feeding = Feeding(tmp_path, Web(delay=0, allow_private=True), log=lambda s: None)
    chat = Chat(life, log=lambda s: None, feeding=feeding)
    server = serve(life, chat, port=0)
    port = server.server_port

    def call(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}", data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with NO_PROXY.open(request, timeout=30) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    try:
        assert call("/api/library")[1]["articles"] == 0
        assert call("/api/library/read", {"which": "simple"})[1]["ok"]  # (the dictionary isn't served here)
        deadline = time.monotonic() + 30
        while call("/api/library")[1]["reading"] and time.monotonic() < deadline:
            time.sleep(0.2)
        assert call("/api/library")[1]["articles"] == len(ARTICLES)
        big = "Some notes about whales. " + "Whales are big animals that live in the sea. " * 1000  # (more than chat)
        code, given = call("/api/library/give", {"text": big, "title": "Whales"})
        assert code == 200 and given["title"] == "Whales"
        assert call("/api/library/page", {"url": internet["page"]})[1]["title"] == "Bread - a page"
        assert call("/api/library/read", {"which": "everything"})[0] == 400
        assert call("/api/library/forget", {"title": "Whales"})[1]["forgot"] == 1
        assert call("/api/library/give", {"text": " "})[0] == 400
    finally:
        feeding.stop(pause=False)
        server.shutdown()


def test_out_of_curiosity_it_reads_from_its_shelf_even_without_the_internet(shelf, tmp_path):
    from haven.cortex.think import Thinker

    thinker = Thinker(tmp_path)  # (no internet)
    thinker.shelf = shelf
    assert thinker.look_up("spiders", why="curious") == "Spider"
    assert "Spider" in thinker.library.titles("curious")  # (it's among what it has read lately)
    assert thinker.look_up("quasars", why="curious") is None  # (nothing on its shelf about that, and no internet)


def test_without_a_shelf_it_lives_on_and_says_why(tmp_path, monkeypatch):
    import sqlite3

    from haven import feeding as feeding_module

    def no_fts(path):
        raise sqlite3.OperationalError("no such module: fts5")

    monkeypatch.setattr(feeding_module, "Shelf", no_fts)
    logged = []
    feeding = Feeding(tmp_path, Web(delay=0, allow_private=True), log=logged.append)
    assert feeding.shelf is None and "fts5" in feeding.problem and logged
    assert not feeding.start()
    assert feeding.status()["problem"] == feeding.problem
    with pytest.raises(FeedError):
        feeding.give("Owls can turn their heads a long way round.")


def test_forgetting_what_it_was_given_leaves_its_encyclopedia_alone(feeding):
    assert feeding.read("simple")
    feeding.wait(30)
    feeding.give("Paris is where my aunt lives. She has a cat.", "Paris")
    assert feeding.forget("Paris") == 1
    assert feeding.shelf.article("Paris", "simple") is not None  # (the encyclopedia's Paris is still there)
