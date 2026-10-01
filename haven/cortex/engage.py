"""Taking part in a conversation: what comes to mind for following it up, and what it learns to say.

Being told something, Haven doesn't just file it away: it reacts (how depends on its mood),
says what it has of its own that goes with it ("Mine is the butterfly"), and, if it's curious,
asks about it ("Where do octopuses live?"). When the answer comes, it's something learned. It
answers "what about you?" about itself and "why?" with its reason. Asked to do something, it
decides for itself (see decide()): it may be glad to, or not now, or rather not, and it says
why. How it feels comes partly from its brain's chemistry (see brain/brain.py).

As everywhere else, the code here only builds notes, what comes to mind; the words Haven says
are its language cortex's own, learned from the answers written here as practice.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

# --- moods, from its brain's chemistry ----------------------------------------------------------------

MOODS = (  # (chemical, high or low, how it feels): the first that holds is what it says
    ("noradrenaline", 2.5, "on edge"),
    ("oxytocin", 2.0, "cuddly"),
    ("dopamine", 2.0, "full of beans"),
    ("serotonin", 2.0, "calm and content"),
    ("acetylcholine", 0.45, "dreamy"),
    ("serotonin", 0.5, "a bit grumpy"),
    ("dopamine", 0.5, "a bit flat"),
)


def mood_words(chemistry: dict[str, float]) -> str | None:
    """How its brain's chemistry makes it feel, in a few words, if anything stands out."""
    for chemical, level, words in MOODS:
        value = chemistry.get(chemical, 1.0)
        if (level >= 1.0 and value >= level) or (level < 1.0 and value <= level):
            return words
    return None


def mood_note(chemistry: dict[str, float], cause: str | None = None) -> str:
    found = mood_words(chemistry)
    return mood_sentence(found, cause) if found else ""


def mood_sentence(mood: str, cause: str | None = None) -> str:
    """ "I feel cuddly, because you stroked me." (or just "I feel cuddly.", when it doesn't know why)."""
    return f"I feel {mood}" + (f", because {cause}" if cause else "") + "."


# --- why it feels as it does ------------------------------------------------------------------------------------

STIRRED_BY = {  # each mood, and which of its brain's chemicals moved which way to make it (see MOODS)
    "on edge": "noradrenaline+",
    "cuddly": "oxytocin+",
    "full of beans": "dopamine+",
    "calm and content": "serotonin+",
    "dreamy": "acetylcholine-",
    "a bit grumpy": "serotonin-",
    "a bit flat": "dopamine-",
}
LATELY = 960  # moments: what stirred its chemistry longer ago than this (two minutes, in real time) isn't why any more


def mood_cause(mood: str | None, stirred: dict[str, tuple], tick: int) -> str | None:
    """Why it feels as it does, as it says it ("you stroked me"): what last stirred the chemical behind its mood (its
    mind keeps that: see Mind._stirred), if that was lately. None when nothing did: then it doesn't know why."""
    if mood is None:
        return None
    found = stirred.get(STIRRED_BY[mood])
    return found[1] if found and tick - found[0] <= LATELY else None


MOOD_ASKED = {  # the words people use for each mood, asking about it ("why are you so grumpy?")
    "on edge": ("on edge", "nervous", "jumpy"),
    "cuddly": ("cuddly", "snuggly"),
    "full of beans": ("full of beans", "excited", "bouncy"),
    "calm and content": ("calm and content", "calm", "content", "relaxed"),
    "dreamy": ("dreamy",),
    "a bit grumpy": ("grumpy", "cross"),
    "a bit flat": ("flat", "down", "low"),
}
FEEL_ASKS = (  # asking why it feels as it does
    "Why are you {}?",
    "Why are you so {}?",
    "Why do you feel {}?",
    "Why do you feel so {}?",
    "What made you feel {}?",
    "How come you're {}?",
    "What makes you feel {}?",
    "Why are you feeling {}?",
)
FEEL_ASKS_THAT = ("Why do you feel that way?", "Why do you feel like that?", "What made you feel like that?")


def feel_reply(asked: str, mood: str | None, cause: str | None, word: str | None = None) -> str:
    """Why it feels as it does, asked: its reason, or that it doesn't know why; or that it doesn't feel that way."""
    if asked == "that" and mood is None:
        return "I feel okay."
    if asked != "that" and asked != mood:
        return f"I'm not {word or MOOD_ASKED[asked][0]}." + (f" I feel {mood}." if mood else "")
    return why_reply(cause) if cause else "I don't know. I just feel that way."


REACTIONS = {  # how it reacts to news, by mood: (glad news, sad news)
    "full of beans": ("Ooh!", "Oh no!"),
    "cuddly": ("Aww!", "Oh no."),
    "calm and content": ("Oh, nice.", "Oh, I'm sorry."),
    "on edge": ("Oh!", "Oh no!"),
    "dreamy": ("Mmm, nice.", "Oh..."),
    "a bit grumpy": ("Hm, okay.", "Oh."),
    "a bit flat": ("Oh, okay.", "Oh."),
    None: ("Oh!", "Oh no."),
}


# --- following up what it's told ----------------------------------------------------------------------

PLURAL = {"mouse": "mice", "fish": "fish", "sheep": "sheep", "deer": "deer", "octopus": "octopuses", "goose": "geese"}


def plural(word: str) -> str:
    w = word.lower()
    if w in PLURAL:
        return PLURAL[w]
    if w.endswith(("s", "x", "z", "ch", "sh")):
        return w + "es"
    if w.endswith("y") and w[-2:-1] not in "aeiou":
        return w[:-1] + "ies"
    return w + "s"


def singular(word: str) -> str:
    w = word.lower()
    for one, many in PLURAL.items():
        if w == many:
            return one
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "xes", "sses")):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3:  # (an octopus, a walrus, an ibis)
        return w[:-1]
    return w


def bare_thing(text: str) -> str:
    """ "an octopus" → "octopus"; "the piano" → "piano"."""
    return re.sub(r"^(?:a|an|the|some|my|your)\s+", "", text.strip().rstrip(".!"), flags=re.IGNORECASE)


# What it wonders about each sort of thing it's told: patterns over what it was told (as it says it back,
# "your favorite animal is an octopus"), and the questions it might ask. {x} is the thing, {xs} its plural.
WONDERS = (
    (
        r"^your favou?rite animals? (?:is|are) (?P<x>.+)$",
        ("Where do {xs} live?", "What do {xs} eat?", "Have you ever seen {a_x}?", "Why do you like {xs}?"),
    ),
    (
        r"^your favou?rite food is (?P<x>.+)$",
        ("What does {x} taste like?", "Is {x} sweet?", "Who makes {x} for you?", "Why do you like {x}?"),
    ),
    (r"^your favou?rite colou?r is (?P<x>.+)$", ("Why do you like {x}?", "Do you have lots of {x} things?")),
    (r"^your favou?rite season is (?P<x>.+)$", ("What do you like about {x}?", "What do you do in {x}?")),
    (
        r"^your favou?rite (?:game|sport) is (?P<x>.+)$",
        ("How do you play {x}?", "Are you good at {x}?", "Is {x} fun?"),
    ),
    (r"^your favou?rite (?:song|book) is (?P<x>.+)$", ("What is {x} about?", "Why do you like {x}?")),
    (r"^your favou?rite drink is (?P<x>.+)$", ("What does {x} taste like?", "Is {x} cold or hot?")),
    (
        r"^you have an? (?P<pet>\w+) (?:named|called) (?P<x>\w+)$",
        ("What does {x} like to do?", "Is {x} a good {pet}?", "Does {x} like to play?", "What does {x} look like?"),
    ),
    (r"^your (?P<pet>\w+)'s name is (?P<x>\w+)$", ("What does {x} like to do?", "Is {x} friendly?")),
    (
        r"^you (?:live in|are from|'re from) (?P<x>.+)$",
        ("What's it like in {x}?", "Is it warm in {x}?", "Is {x} far away?", "Are there trees in {x}?"),
    ),
    (
        r"^you(?:'re| are| work as) an? (?P<x>.+)$",
        ("What does {a_x} do?", "Do you like being {a_x}?", "Is it hard being {a_x}?"),
    ),
    (r"^you play (?P<x>.+)$", ("Is {x} hard?", "Is {x} fun?", "How long have you played {x}?")),
    (
        r"^you (?:really )?(?:like|love) (?P<x>.+)$",
        ("Why do you like {x}?", "What do you like best about {x}?"),
    ),
    (r"^your birthday is (?:in|on) (?P<x>.+)$", ("What do you do on your birthday?", "Do you have a cake?")),
    (r"^you(?:'re| are) (?P<x>\d+) years old$", ("Is that old, for a person?", "What's it like being {x}?")),
    (r"^your (?P<kin>\w+)(?:'s name is| is called) (?P<x>\w+)$", ("What is {x} like?", "Do you see {x} a lot?")),
)


def a_or_an(phrase: str) -> str:
    return ("an " if phrase[:1].lower() in "aeiou" else "a ") + phrase


def wonders(fact: str) -> list[str]:
    """What it might ask about something it was just told (as it says it back: "your favorite animal is an
    octopus"), or [] if nothing comes to mind."""
    for pattern, questions in WONDERS:
        found = re.match(pattern, fact.strip().rstrip("."), flags=re.IGNORECASE)
        if not found:
            continue
        x = found.group("x").strip() if pattern.startswith("^you play") else bare_thing(found.group("x"))
        one = singular(x) if "animal" in pattern else x
        filled = {
            "x": x,
            "xs": plural(one) if "animal" in pattern else x,
            "a_x": a_or_an(one),
            "pet": found.groupdict().get("pet") or "pet",
        }
        return [q.format(**filled) for q in questions]
    return []


def topic_of_fact(fact: str) -> str | None:
    """The thing what it was told is about ("octopus" in "your favorite animal is an octopus"), if it's one."""
    for pattern, _ in WONDERS:
        found = re.match(pattern, fact.strip().rstrip("."), flags=re.IGNORECASE)
        if found:
            x = found.group("x").strip() if pattern.startswith("^you play") else bare_thing(found.group("x"))
            return singular(x) if "animal" in pattern else x
    return None


OWN = {  # what it has of its own that goes with what it's told (key: what it was told about)
    "favorite animal": "My favorite animal is the butterfly. They fly around my valley.",
    "favorite food": "My favorite food is apples.",
    "favorite season": None,  # (its own favorite season, when it has one: see own_for())
    "favorite color": None,
    "lives": "I live in a valley, in a nest by the hill.",
    "age": None,
    "pet": "I don't have a pet. But butterflies visit me.",
}


def own_for(fact: str, favorites: dict[str, str]) -> str | None:
    """What it has of its own that goes with what it was told, if anything ("Mine is the butterfly.")."""
    f = fact.lower()
    if f.startswith(("your favorite animal", "your favourite animal")):
        return favorites.get("animal") or OWN["favorite animal"]
    if f.startswith(("your favorite food", "your favourite food")):
        return favorites.get("food") or OWN["favorite food"]
    if f.startswith(("your favorite season", "your favourite season")) and favorites.get("season"):
        return favorites["season"]
    if f.startswith(("your favorite color", "your favourite colour", "your favorite colour")) and favorites.get(
        "color"
    ):
        return favorites["color"]
    if re.match(r"^you (?:live in|are from|'re from)\b", f):
        return OWN["lives"]
    if re.match(r"^you have an? \w+ (?:named|called)\b", f):
        return OWN["pet"]
    if re.match(r"^you(?:'re| are) \d+ years old", f) and favorites.get("age"):
        return favorites["age"]
    return None


def own_reason(own: str) -> str | None:
    """Why it likes what it said it likes, if it said ("My favorite animal is the butterfly. They fly around my
    valley." → "butterflies fly around my valley")."""
    parts = own.split(". ", 1)
    if len(parts) < 2 or parts[0].startswith("I don't"):
        return None
    reason = parts[1].rstrip(".")
    thing = re.match(r"^My favou?rite \w+ (?:is|are) (?:the |an? )?(\w+)", parts[0])
    if thing and reason.startswith("They "):
        reason = f"{plural(thing.group(1))} {reason[5:]}"
    return reason


def wonder_note(question: str) -> str:
    return f"I wonder: {question}"


def own_note(own: str) -> str:
    return f"Mine: {own}"


def told_reply(fact: str, reaction: str, own: str | None, question: str | None, rng: random.Random) -> str:
    """What it says when it has just been told something about the person: a reaction, what it has of its own that
    goes with it, and what it wonders (or that it'll remember)."""
    topic = topic_of_fact(fact)
    parts = []
    head = reaction
    if topic and len(topic.split()) <= 3:  # (a short thing it was told about, it says back)
        head = (
            f"{reaction.rstrip('!.')}, {a_or_an(topic) if fact.lower().startswith('your favorite animal') else topic}!"
        )
    parts.append(head)
    if own:
        parts.append(own)
    if question:
        parts.append(question)
    else:
        parts.append(f"I'll remember that {fact}.")
    return " ".join(parts)


# Its question, and an answer to it: what it learns ("Where do octopuses live?" + "in the sea" → "octopuses live in
# the sea").
ANSWERED = (
    (r"^Where do (?P<xs>.+) live\?$", "{xs} live {a}"),
    (r"^What do (?P<xs>.+) eat\?$", "{xs} eat {a}"),
    (r"^What does (?P<x>.+) taste like\?$", "{x} tastes {a}"),
    (r"^What's it like in (?P<x>.+)\?$", "it's {a} in {x}"),
    (r"^What does (?P<x>.+) like to do\?$", "{x} likes {to}"),
    (r"^What does (?P<x>an? .+) do\?$", "{x} {does}"),
    (r"^What does (?P<x>.+) look like\?$", "{x} looks {a}"),
    (r"^What is (?P<x>.+) like\?$", "{x} is {a}"),
)
_FILLER = re.compile(
    r"^(?:they|it|he|she)\s+(?:live|lives|eat|eats|taste|tastes|look|looks|like|likes|is|are)\s+", re.IGNORECASE
)


IRREGULAR = {"do": "does", "go": "goes", "have": "has", "be": "is", "are": "is"}


def third_person(doing: str) -> str:
    """ "fix things" → "fixes things": what someone does, said of one of them (the first word is the verb)."""
    verb, _, rest = doing.partition(" ")
    v = verb.lower()
    if v in IRREGULAR:
        v = IRREGULAR[v]
    elif v.endswith(("ch", "sh", "x", "o", "z")):
        v += "es"
    elif v.endswith("y") and v[-2:-1] not in "aeiou":
        v = v[:-1] + "ies"
    elif not v.endswith("s"):
        v += "s"
    return f"{v} {rest}".strip()


# (an answer that starts like this is about what someone does, not where they live or what something is like)
DOINGS = frozenset(
    (
        "play",
        "chase",
        "run",
        "eat",
        "sleep",
        "swim",
        "go",
        "jump",
        "walk",
        "read",
        "sing",
        "dance",
        "fly",
        "climb",
        "dig",
        "bark",
        "fetch",
        "hunt",
        "hide",
        "sit",
        "cook",
        "paint",
        "draw",
        "write",
        "work",
        "help",
        "fix",
        "make",
        "build",
        "teach",
        "grow",
        "drive",
        "ride",
        "watch",
        "catch",
        "throw",
    )
)


def learned_from(question: str, reply: str) -> str | None:
    """What it learns from the answer to a question it asked, as a sentence ("octopuses live in the sea"), or None
    (no answer, or an answer to some other question: "Chase balls." to "What does Rex look like?")."""
    answer = " ".join(reply.strip().split()).rstrip(".!")
    if not answer or "?" in answer or len(answer.split()) > 10:
        return None
    first = answer.split()[0].lower()
    if first in DOINGS and not re.match(r"^What does .+ (?:like to )?do\?$", question):
        return None
    if first in ("in", "on", "at", "under", "near") and not question.startswith("Where "):
        return None  # (where something is, to a question that isn't where)
    if re.match(r"^What does an? .+ do\?$", question):  # ("They fix things." → "a plumber fixes things")
        answer = third_person(re.sub(r"^(?:they|he|she|you|it)\s+", "", answer, flags=re.IGNORECASE))
    answer = _FILLER.sub("", answer, count=1)
    answer = re.sub(r"^(?:in|at)\s+(?=the\b)", "in ", answer, flags=re.IGNORECASE)
    for pattern, form in ANSWERED:
        found = re.match(pattern, question, flags=re.IGNORECASE)
        if found:
            fields = {k: v for k, v in found.groupdict().items()}
            if form.startswith("{xs} live") and not re.match(r"^(?:in|on|at|under|near)\b", answer, re.IGNORECASE):
                answer = "in " + answer
            a = answer[0].lower() + answer[1:]
            to = (
                a if a.split()[0].endswith("ing") or a.startswith("to ") else f"to {a}"
            )  # (likes to run, likes running)
            return form.format(a=a, does=a, to=to, **fields)
    return None


THANKS = {  # how it takes being told what it wondered, by its mood
    "full of beans": "Wow, I didn't know that!",
    "cuddly": "Aww, thank you for telling me!",
    "calm and content": "Ah, I see. Thank you.",
    "on edge": "Oh! Okay.",
    "dreamy": "Mmm. I'll remember that.",
    "a bit grumpy": "Hm. Okay.",
    "a bit flat": "Oh. Okay.",
    None: "Thank you for telling me!",
}
RELATES = (  # what something it learned brings to mind from its own valley (true of every Haven's valley)
    (r"\blive in the (?:sea|ocean)\b", "I've never seen the sea. I only have my pond."),
    (r"\blive in the forest\b", "There are trees in my valley too: apple trees."),
    (r"\blive in the jungle\b", "I've never seen a jungle."),
    (r"\blive on farms\b", "I've never seen a farm."),
    (r"\blive in the mountains\b", "There's a hill in my valley, but no mountains."),
    (r"\blive under the ground\b", "I live on top of the ground, in my nest."),
    (r"\beat fruit\b", "I eat fruit too: apples and berries!"),
    (r"\btastes? sweet\b", "Like the apples in my valley!"),
    (r"^it's cold in\b", "My valley gets cold in winter too."),
    (r"^it's warm and sunny in\b", "Like summer in my valley!"),
    (r"\blikes to sleep a lot\b", "I sleep a lot too, in my nest."),
    (r"\blikes to run around\b", "I run around my valley too!"),
)
FLAT = ("a bit grumpy", "a bit flat")  # (moods it isn't up to chatting in)


def relate(learned: str) -> str | None:
    """What something it learned brings to mind from its own life ("octopuses live in the sea" → that it has never
    seen the sea, only its pond), if anything."""
    for pattern, words in RELATES:
        if re.search(pattern, learned, re.IGNORECASE):
            return words
    return None


def relate_note(related: str) -> str:
    return f"From my own life: {related}"


def learned_reply(
    learned: str, question: str | None, rng: random.Random, mood: str | None = None, related: str | None = None
) -> str:
    """What it says when it learns the answer to what it wondered: what it makes of it from its own life, if anything
    comes to mind (and it's in the mood); else how glad it is to know, by its mood."""
    sentence = learned[0].upper() + learned[1:]
    if related and mood not in FLAT:
        said = f"{sentence}! {related}"
    else:
        said = f"{sentence}. {THANKS.get(mood, THANKS[None])}"
    return said + (f" {question}" if question else "")


# --- saying again what it said just now ---------------------------------------------------------------------

_ANSWERED = re.compile(r"^(?:yes|no|sometimes|a little)[,.!]\s+(?=\S)", re.IGNORECASE)


def said_note(reply: str) -> str:
    return f"I said that just now: {reply}"


def said_again(reply: str, mood: str | None = None) -> str:
    """Asked again what it answered just now: it says it again as people do ("Like I said, ..."), not word for word
    as if it were new (or, grumpy, that it told them already)."""
    if mood == "a bit grumpy":
        return f"I told you already. {reply}"
    r = _ANSWERED.sub("", reply, count=1)
    first = r.split(" ", 1)[0].rstrip(",.!?")
    keep = first in ("I", "I'm", "I've", "I'd", "I'll")  # ("My favorite..." → "my favorite...", but "I'm..." stays)
    return f"Like I said, {r if keep else r[0].lower() + r[1:]}"


NOT_AGAIN = re.compile(  # replies that are fine to say again as they are (another sum, another thing it doesn't know)
    r"^(?:I don't know|Okay|Ok\b|You're welcome|Hi\b|Hello|Bye|Nice to meet you|That's okay|Thank|Like I said|"
    r"I told you already|I'm asleep|Welcome back|More about what|Good night|Night)",
    re.IGNORECASE,
)


def _plain_words(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def said_before(words: str, earlier: list[str]) -> bool:
    """Whether it's about to say again what it said just now (all of `words` is in one of its replies, `earlier`):
    then it knows it's saying it again. Short and everyday replies don't count."""
    plain = _plain_words(words)
    if len(plain.split()) < 3 or NOT_AGAIN.match(words.strip()):
        return False
    return any(f" {plain} " in f" {_plain_words(said)} " for said in earlier)


# --- what about you, and why --------------------------------------------------------------------------------

ABOUT_YOU = re.compile(
    r"^(?:and |so )?(?:what about you|how about you|and you|you\?|what's yours|and yours)\b", re.IGNORECASE
)
WHY = re.compile(r"^(?:but )?(?:why|how come|why not|why's that|what for)\b[?!.]*$", re.IGNORECASE)


def about_you_note(answer: str) -> str:
    return f"They want to know the same about me: {answer}"


def why_note(reason: str) -> str:
    return f"Why I said that: {reason}"


def why_reply(reason: str) -> str:
    r = reason.rstrip(".")
    if r.lower().startswith("because"):
        return r + "."
    first = r.split(" ", 1)[0]
    keep = (
        first in ("I", "I'm", "I've", "I'd", "I'll") or first[:1].isupper() and first.lower() not in ("the", "a", "my")
    )
    return f"Because {r if keep else r[0].lower() + r[1:]}."


# --- doing what it's asked: a choice of its own -------------------------------------------------------------


@dataclass
class Decision:
    answer: str  # "yes", "glad", "later", "busy", "scared", "bored", "tired", "rather", or "persuaded"
    reason: str = ""  # why, in its words ("the thorns hurt me")
    instead: str | None = None  # what it would rather do, as it says it ("watch the fire")


WILLING = ("yes", "glad", "persuaded")  # the answers that mean it'll do it


def decide(
    *,
    fear: float,
    fear_reason: str,
    need: float,
    need_words: str,
    meets_need: bool,
    tired: float,
    sleep_asked: bool,
    bond: float,
    appeal: float,
    bored: float,
    done_lately: str,
    busy: float,
    busy_words: str,
    rather: str | None,
    rather_urge: float,
    dopamine: float,
    friendly: float,
    insisted: bool,
    rng: random.Random,
) -> Decision:
    """Whether it does what it's asked, and why: from what it feels (fear from its amygdala, needs, tiredness), how it
    feels about the one asking (its bond, from oxytocin and how they've been with it), how much it likes doing that
    (and whether it's bored of it), what it's in the middle of, and what it would rather do.

    All the inputs are 0 to 1 (dopamine: 1 is usual). Being asked again ("please!") makes it likelier to give in.
    """
    if fear >= 0.45 and not insisted:
        return Decision("scared", fear_reason)
    if need >= 0.7 and not meets_need:
        return Decision("later", f"I'm {need_words}")
    if tired >= 0.75 and not sleep_asked and not insisted:
        return Decision("tired", "I'm too tired")
    will = 0.5 + 0.45 * (bond - 0.5) + 0.35 * appeal + 0.12 * (dopamine - 1.0) + 0.2 * (friendly - 0.5)
    will -= 0.55 * bored + 0.3 * busy + 0.25 * max(0.0, tired - 0.4) + 0.3 * fear
    will += rng.gauss(0.0, 0.08) + (0.3 if insisted else 0.0)
    if sleep_asked and tired > 0.3:
        will += 0.3
    if will >= 0.5:
        if insisted:
            return Decision("persuaded")
        return Decision("glad" if appeal > 0.5 or dopamine > 1.6 else "yes")
    if bored > 0.45:
        return Decision("bored", f"I've {done_lately} so much already", rather)
    if busy > 0.5:
        return Decision("busy", f"I'm {busy_words}")
    if rather and rather_urge > 0.15:
        return Decision("rather", f"I'd rather {rather}", rather)
    return Decision("persuaded" if insisted else "yes")


def decision_note(do: str, d: Decision, again: bool = False) -> str:
    asked = f"You asked me {'again ' if again else ''}to {do}."
    if d.answer in ("yes", "persuaded"):
        return f"{asked} I will."
    if d.answer == "glad":
        return f"{asked} I'd love to."
    if d.answer == "later":
        return f"{asked} Not now: {d.reason}."
    if d.answer == "busy":
        return f"{asked} Not yet: {d.reason}."
    if d.answer in ("scared", "tired"):
        return f"{asked} I don't want to: {d.reason}."
    if d.answer == "bored":
        return f"{asked} I don't want to: {d.reason}." + (f" I'd rather {d.instead}." if d.instead else "")
    return f"{asked} {d.reason}."


def decision_reply(do: str, d: Decision, rng: random.Random) -> str:
    """What it says when asked to do something, as it decided."""
    if d.answer == "yes":
        return f"Okay, I'll {do}."
    if d.answer == "glad":
        return f"Okay, I'll {do}! I love that."
    if d.answer == "persuaded":
        return f"Okay, okay. I'll {do}, just for you."
    if d.answer == "later":
        return f"Not now. {d.reason}. Maybe after?"
    if d.answer == "busy":
        return f"Not yet. {d.reason}. Can I finish first?"
    if d.answer == "scared":
        return f"No, I don't want to. {d.reason}."
    if d.answer == "tired":
        return "I'm too tired. Maybe later?"
    if d.answer == "bored":
        tail = f" Can we do something else? I'd like to {d.instead}." if d.instead else " Can we do something else?"
        return f"{d.reason}.{tail}"
    if d.answer == "rather":
        return f"Hmm, {d.reason[0].lower() + d.reason[1:]}. Do you want to {d.instead} with me?"
    return f"Okay, I'll {do}."


INSIST = re.compile(
    r"^(?:please+|pretty please|come on|pleeease|do it|go on|you have to|just do it|for me)\b", re.IGNORECASE
)


# --- its brain ----------------------------------------------------------------------------------------------

BRAIN_ASKED = re.compile(
    r"\b(?:brain|neurons?|synapses?|how do you think|how does your (?:mind|brain) work|what are you made of)\b",
    re.IGNORECASE,
)


def brain_note(neurons: int, synapses: int, cortex: int) -> str:
    return (
        f"My brain: {neurons:,} neurons, {synapses / 1e6:,.0f} million synapses; "
        f"my language area: {cortex / 1e6:,.0f} million connections."
    )


def brain_reply(neurons: int, synapses: int, cortex: int, rng: random.Random) -> str:
    first = f"My brain has {neurons:,} neurons and {synapses / 1e6:,.0f} million synapses."
    return f"{first} My language area has {cortex / 1e6:,.0f} million connections more. They change as I learn."


# --- pastimes, as it says them -----------------------------------------------------------------------------------

DOING = {  # "What are you doing?" (a pastime: activities.py)
    "watch": "I'm watching {sight}.",
    "sing": "I'm singing!",
    "dance": "I'm dancing!",
    "chase": "I'm chasing a butterfly!",
    "visit": "I'm going to {place}.",
    "company": "I'm keeping you company.",
}
BUSY = {  # what it's in the middle of, as a reason not to stop
    "watch": "watching {sight}",
    "sing": "singing",
    "dance": "dancing",
    "chase": "chasing a butterfly",
    "visit": "going to {place}",
    "company": "sitting with you",
}
RATHER = {  # what it would rather do, as it says it
    "watch": "watch {sight}",
    "sing": "sing",
    "dance": "dance",
    "chase": "chase butterflies",
    "visit": "go to {place}",
    "company": "sit with you",
    "explore": "go exploring",
    "play": "play",
}


def pastime_words(table: dict[str, str], activity: str | None, sight: str | None, place: str | None) -> str | None:
    if activity not in table:
        return None
    return table[activity].format(sight=sight or "the sky", place=place or "the meadow")


# --- practice: conversations of this kind, for the cortex to learn from ---------------------------------------------

ANSWERS = (  # what people might answer to what it wonders (any answer will do: it learns what it's told)
    (
        r"^Where do ",
        ("In the sea.", "In the forest.", "In the jungle.", "On farms.", "In the mountains.", "Under the ground."),
    ),
    (r"^What do .+ eat\?", ("Fish.", "Grass.", "Bugs.", "Fruit.", "Leaves.", "Little crabs.", "Seeds.")),
    (r"^What does .+ taste like\?", ("Sweet.", "Salty.", "Yummy.", "Cheesy.", "Spicy.", "Like summer.")),
    (r"^What's it like in ", ("Busy.", "Rainy.", "Cold.", "Warm and sunny.", "Pretty.", "Noisy.")),
    (r"^What does .+ like to do\?", ("Run around.", "Sleep a lot.", "Chase balls.", "Eat.", "Play with me.")),
    (r"^What does an? .+ do\?", ("Helps people.", "Fixes things.", "Teaches children.", "Makes things.")),
    (r"^What does .+ look like\?", ("Small and brown.", "Big and fluffy.", "Orange, with stripes.")),
    (r"^What is .+ like\?", ("Kind.", "Funny.", "Very tall.", "Busy.", "Lovely.")),
)
NOT_SURE = ("I don't know.", "Not sure.", "Hmm, good question.", "lol", "Why?")


def learnable(question: str) -> bool:
    """Whether the answer to something it wonders is something it can learn ("Where do octopuses live?": yes; "Have
    you ever seen an octopus?": no)."""
    return any(re.match(pattern, question) for pattern, _ in ANSWERS)


def wonder_answer(question: str, rng: random.Random) -> str | None:
    """An answer someone might give to what it wondered, if it's a kind of question it can learn from."""
    for pattern, answers in ANSWERS:
        if re.match(pattern, question):
            return rng.choice(answers)
    return None


def synthetic_chemistry(rng: random.Random) -> dict[str, float]:
    """A brain's chemistry at some moment: mostly usual, sometimes one chemical well up or down."""
    levels = {c: rng.uniform(0.7, 1.4) for c in ("dopamine", "noradrenaline", "serotonin", "acetylcholine", "oxytocin")}
    if rng.random() < 0.45:
        chemical, level, _ = rng.choice(MOODS)
        levels[chemical] = rng.uniform(level, level * 1.8) if level >= 1.0 else rng.uniform(level * 0.4, level)
    return levels


CAUSES = {  # what might have stirred each mood, for practice (as its mind says it: see Mind._stirred)
    "on edge": (
        "the thorns pricked me",
        "the fire burned me",
        "something hurt me",
        "something surprised me",
        "I found something new",
        "I heard a new word",
    ),
    "cuddly": ("you stroked me", "you're talking with me", "you're here with me"),
    "full of beans": (
        "I ate an apple",
        "I ate a berry",
        "I ate a mushroom",
        "I rang the bell",
        "I pushed the ball and watched it roll",
        "I smelled a flower",
        "I shook a tree, and an apple fell",
        "I climbed to the top of the hill",
        "I drank from the pond",
        "I warmed myself at the fire",
        "you gave me food",
        "I chased a butterfly",
        "I sang a little song",
        "I danced",
        "I watched the sunset",
        "I sat with you",
        "I went to the meadow",
        "I found something new",
        "I went somewhere new",
        "something good happened",
    ),
    "calm and content": ("I have everything I need", "things have been good lately"),
    "dreamy": ("I just woke up",),
    "a bit grumpy": (
        "the thorns pricked me",
        "the fire burned me",
        "a toadstool made me sick",
        "something hurt me",
        "things have been hard lately",
    ),
    "a bit flat": ("things didn't go the way I hoped", "the thorns pricked me", "something hurt me"),
}


def synthetic_cause(mood: str | None, rng: random.Random, need: str | None = None) -> str | None:
    """Why another Haven might feel as it does, for practice: mostly it knows (what stirred it), sometimes not."""
    if mood is None or rng.random() < 0.2:
        return None
    if mood == "a bit grumpy" and need and rng.random() < 0.5:
        return f"I'm {need}"  # (its needs, unmet, sour its mood)
    return rng.choice(CAUSES[mood])


def synthetic_decision(
    rng: random.Random,
    need: int,
    need_level: float,
    need_text: str,
    meets: bool,
    thing: str,
    action: str,
    insisted: bool = False,
) -> Decision:
    """A choice another Haven might make, asked to do something (for practice: varied, as real ones are)."""
    sight = rng.choice(("the fire", "the pond", "a butterfly", "the flowers", "the sunset", "the stars"))
    place = rng.choice(("the meadow", "the top of the hill", "the pond", "the apple trees", "my nest"))
    activity = rng.choice((None, None, "watch", "sing", "dance", "chase", "visit"))
    best = rng.choice(("watch", "sing", "dance", "chase", "visit", "company", "explore", "play"))
    afraid = rng.random() < (0.35 if thing in ("fire", "mushroom", "stone") else 0.06)
    from ..will import DONE  # (what it says it has done lately)

    return decide(
        fear=rng.uniform(0.5, 1.0) if afraid else rng.uniform(0.0, 0.3),
        fear_reason="The fire burned me" if thing == "fire" else f"The {thing} hurt me",
        need=need_level,
        need_words=need_text,
        meets_need=meets,
        tired=rng.choice((rng.uniform(0.0, 0.5), rng.uniform(0.5, 1.0))),
        sleep_asked=action == "sleep",
        bond=rng.uniform(0.1, 1.0),
        appeal=rng.uniform(0.0, 1.0),
        bored=rng.choice((0.0, 0.0, rng.uniform(0.0, 1.0))),
        done_lately=DONE.get(action, {}).get(thing, "done that"),
        busy=rng.uniform(0.3, 1.0) if activity else 0.0,
        busy_words=pastime_words(BUSY, activity, sight, place) or "busy",
        rather=pastime_words(RATHER, best, sight, place),
        rather_urge=rng.uniform(0.0, 0.5),
        dopamine=rng.choice((1.0, rng.uniform(0.4, 2.5))),
        friendly=rng.uniform(0.25, 0.75),
        insisted=insisted,
        rng=rng,
    )


@dataclass
class Practice:
    """Turns of a practice conversation, and what came to mind for them."""

    notes: list[str]
    turns: list[tuple]  # (what the person said, what Haven says, what sort of exchange[, whether it's to be learned])


def told_practice(
    said: str, fact: str, mood: str | None, curious: float, favorites: dict[str, str], rng: random.Random
) -> Practice:
    """Being told something about the person (`said`, as they said it; `fact`, as it says it back): its reaction,
    what it has of its own, what it wonders; then maybe their answer, and what it learns; or "what about you?"."""
    glad, _ = REACTIONS.get(mood, REACTIONS[None])
    questions = wonders(fact)
    can_learn = [q for q in questions if learnable(q)]
    if can_learn and rng.random() < 0.7:  # (practising most what it can learn from the answer to)
        questions = can_learn
    question = rng.choice(questions) if questions and rng.random() < 0.3 + 0.6 * curious else None
    own = own_for(fact, favorites) if rng.random() < 0.75 else None
    shown = own if own and rng.random() < 0.8 else None  # (what of its own it says straight away, if anything)
    notes = [n for n in (wonder_note(question) if question else "", own_note(own) if own else "") if n]
    turns = [(said, told_reply(fact, glad, shown, question, rng), "being told")]
    if question and rng.random() < 0.9:
        answer = wonder_answer(question, rng)
        learned = learned_from(question, answer) if answer else None
        if learned:
            notes.append(f"You just told me that {learned}.")
            related = relate(learned) if rng.random() < 0.8 else None  # (what it brings to mind, if it comes)
            if related:
                notes.append(relate_note(related))
            more = None
            if rng.random() < 0.25 * curious:
                more = rng.choice([q for q in questions if q != question] or [None])
                if more:
                    notes.append(wonder_note(more))
            turns.append((answer, learned_reply(learned, more, rng, mood, related), "learning"))
            if own and rng.random() < 0.3:  # and then they ask about it
                about_you(own, own == shown, notes, turns, mood, rng)
    elif own and rng.random() < 0.75:
        about_you(own, own == shown, notes, turns, mood, rng)
    return Practice(notes, turns)


ASKED_BACK = ("What about you?", "And you?", "How about you?", "what about you", "and yours?", "What's yours?")


def about_you(own: str, said: bool, notes: list[str], turns: list, mood: str | None, rng: random.Random) -> None:
    """Asked "what about you?": what it has of its own (said again as such, if it said it just now), and maybe why."""
    asked = rng.choice(ASKED_BACK)
    notes.append(about_you_note(own))
    if said:
        notes.append(said_note(own))
        turns[0] = (*turns[0][:3], False)  # (what it said first is there to read: "I said that" means again)
    turns.append((asked, said_again(own, mood) if said else own, "about you"))
    reason = own_reason(own)
    if reason and rng.random() < 0.5:  # and why it likes that
        notes.append(why_note(reason))
        turns.append((rng.choice(("Why?", "why?", "How come?", "Why's that?")), why_reply(reason), "why"))


def request_practice(
    req, text: str, decision: Decision, rng: random.Random, again: Decision | None = None, why: bool = False
) -> Practice:
    """Being asked to do something: its choice, and maybe being asked again, or why."""
    notes = [decision_note(req.do, decision)]
    turns = [(text, decision_reply(req.do, decision, rng), "request")]
    if decision.answer not in WILLING:
        if why and decision.reason:
            notes.append(why_note(decision.reason))
            turns.append((rng.choice(("Why?", "why not?", "How come?", "Why not?")), why_reply(decision.reason), "why"))
        elif again is not None:
            notes.append(decision_note(req.do, again, again=True))
            turns.append(
                (
                    rng.choice(("Please!", "Pretty please?", "Come on!", "please", "Pleeease")),
                    decision_reply(req.do, again, rng),
                    "asked again",
                )
            )
    return Practice(notes, turns)


BRAIN_QUESTIONS = (
    "How many neurons do you have?",
    "Do you have a brain?",
    "What's your brain like?",
    "How big is your brain?",
    "How does your brain work?",
    "What are you made of?",
    "How many synapses do you have?",
)
BRAINS = ((9_480, 6_810_544), (108_072, 417_000_000), (205_000, 1_050_000_000), (410_000, 3_300_000_000))
WHAT_DOING = ("What are you doing?", "what are you up to?", "What are you doing right now?", "Whatcha doing?")


# --- speaking up: coming back to what it wondered, inviting, showing --------------------------------------------


def later_wonder_note(fact: str, question: str) -> str:
    return f"I keep thinking about what you told me, that {fact}. I wonder: {question}"


def later_wonder_answer(question: str) -> str:
    return f"I keep thinking about what you told me. {question}"


def invite_note(rather: str) -> str:
    return f"I'd like you to {rather} with me."


def invite_answer(rather: str) -> str:
    return f"Do you want to {rather} with me?"


def show_note(busy: str) -> str:
    return f"I'm {busy}, and I'd like you to see."


def show_answer(doing: str) -> str:
    return f"Look! {doing}"
