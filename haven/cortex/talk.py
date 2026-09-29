"""Talking with people: how Haven learns to answer in its own words, from its own state.

Every answer it learns is worked out from a moment of one of its simulated lives: how it
feels, what it sees, what it wants and the time of day, which it reads straight from its
workspace tokens; and what it knows (its name and age, where it is, the words and facts
it has learned, what it remembers), which comes to mind first as a line of inner speech
before it answers. When a question isn't about anything it has experienced or learned,
the answer it learns is that it doesn't know. It can then go and read about it.

The sentences are templates written by people, like the words a parent gives a child for
what the child is feeling. What Haven learns is to say the right one at the right time,
because its state is what it is.
"""

from __future__ import annotations

import random

import numpy as np

from ..mind import Mind, need_words
from ..world import BUSH, NEST, STONE, THORN, WALL

QUESTIONS: dict[str, tuple[str, ...]] = {
    "feel": (
        "How are you?",
        "How are you feeling?",
        "How do you feel?",
        "How's it going?",
        "Are you okay?",
        "How are you doing?",
        "How do you feel right now?",
        "How are you today?",
        "how are you",
        "you ok?",
        "What's up?",
        "Hi, how are you?",
    ),
    "hungry": ("Are you hungry?", "Do you want something to eat?", "Are you starving?", "are you hungry"),
    "cold": ("Are you cold?", "Do you feel cold?", "Are you freezing?", "are you cold"),
    "hot": ("Are you hot?", "Do you feel hot?", "Are you too warm?"),
    "tired": ("Are you tired?", "Are you sleepy?", "Do you need to sleep?", "are you tired"),
    "hurt": ("Are you hurt?", "Does anything hurt?", "Are you in pain?", "Does it hurt?"),
    "good": ("Are you happy?", "Do you feel good?", "Are you in a good mood?", "Is everything good?"),
    "bad": ("Are you sad?", "Do you feel bad?", "Are you upset?", "Is something wrong?"),
    "see": ("What do you see?", "What are you looking at?", "What's in front of you?", "What can you see?"),
    "smell": ("What do you smell?", "Can you smell anything?", "Do you smell something?"),
    "doing": (
        "What are you doing?",
        "What are you up to?",
        "What do you want?",
        "What do you want to do?",
        "Where are you going?",
        "what are you doing",
    ),
    "thinking": ("What are you thinking about?", "What's on your mind?", "What are you thinking?"),
    "time": ("What time is it?", "Is it day or night?", "What time of day is it?", "Is it dark?"),
    "asleep": ("Are you asleep?", "Are you sleeping?"),
    "awake": ("Are you awake?", "Are you up?"),
    "name": ("What's your name?", "What is your name?", "Who are you?", "what's your name", "Do you have a name?"),
    "age": ("How old are you?", "What's your age?", "How long have you been alive?"),
    "place": ("Where are you?", "Where are you right now?", "Where are you now?", "What's around you?"),
    "home": ("Where do you live?", "Where is your home?", "Where do you sleep?"),
    "words": ("What words do you know?", "Do you know any words?", "What words have you learned?"),
    "talk": ("Can you talk?", "Can you speak?", "Do you understand me?", "Can you hear me?"),
    "learned": ("What have you learned?", "What do you know?", "What do you know about your garden?"),
    "eat": ("What is good to eat?", "What do you eat?", "What food do you like?"),
    "danger": ("What hurts you?", "What's dangerous?", "What should you stay away from?"),
    "remember": ("What do you remember?", "What happened to you?", "Tell me something you remember."),
    "alive": ("Are you alive?", "Do you think you're alive?", "Are you real?", "Are you conscious?"),
    "feelings": ("Do you have feelings?", "Can you feel things?", "Do you feel anything?"),
    "what": ("What are you?", "What kind of thing are you?", "Tell me about yourself.", "Describe yourself."),
    "hello": (
        "Hi",
        "Hello",
        "Hi Haven",
        "Hello Haven",
        "Hey",
        "Good morning",
        "Hi there",
        "Hello there",
        "Hey there",
        "Good evening",
        "Howdy",
    ),
    "thanks": ("Thank you", "Thanks", "Thanks Haven", "Thank you so much.", "Thanks a lot"),
    "bye": ("Bye", "Goodbye", "See you later", "Bye Haven", "Good night", "I have to go now.", "See you", "Bye bye"),
}

# Questions about the world beyond its garden: it hasn't experienced or read about these.
UNKNOWN_FORMS = (
    "What is {}?",
    "Tell me about {}.",
    "Do you know about {}?",
    "What do you know about {}?",
    "Can you tell me about {}?",
    "Have you heard of {}?",
    "Explain {}.",
)
UNKNOWN_TOPICS = (
    "the moon",
    "the sun",
    "the ocean",
    "Paris",
    "France",
    "Japan",
    "dinosaurs",
    "elephants",
    "whales",
    "music",
    "the internet",
    "computers",
    "cars",
    "trains",
    "airplanes",
    "the weather",
    "snow",
    "rain",
    "volcanoes",
    "mountains",
    "the stars",
    "Mars",
    "Jupiter",
    "gravity",
    "electricity",
    "money",
    "school",
    "football",
    "chess",
    "the piano",
    "pizza",
    "chocolate",
    "coffee",
    "the president",
    "history",
    "science",
    "math",
    "photosynthesis",
    "atoms",
    "the brain",
    "love",
    "friendship",
    "the city",
    "London",
    "New York",
    "China",
    "Egypt",
    "the pyramids",
    "robots",
    "phones",
    "books",
    "painting",
    "Shakespeare",
    "Einstein",
    "cats",
    "dogs",
    "horses",
    "birds",
    "fish",
    "the sea",
    "rivers",
    "forests",
    "deserts",
    "winter",
    "summer",
)
OTHER_QUESTIONS = (
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What is two plus two?",
    "Why is the sky blue?",
    "How far away is the moon?",
    "Who is the president?",
    "What's the weather like today?",
    "How do airplanes fly?",
    "What is the biggest animal?",
    "How many days are in a year?",
    "What language do people speak in Brazil?",
    "Can you help me with my homework?",
    "What should I cook for dinner?",
    "What's your favorite movie?",
    "Do you like music?",
    "What is the meaning of life?",
)
DONT_KNOW = "I don't know. I haven't learned about that."


def age_words(mind: Mind) -> str:
    days = int(mind.age // 1200)
    return "less than a day old" if days < 1 else "one day old" if days == 1 else f"{days} days old"


def place(mind: Mind) -> str:
    """Where it is in its garden, from what's around it."""
    w = mind.world
    x, y = w.x, w.y
    grid = w.grid
    if grid[y, x] == NEST:
        return "I'm in my nest."

    def near(kind: int, reach: int) -> bool:
        box = grid[max(0, y - reach) : y + reach + 1, max(0, x - reach) : x + reach + 1]
        return bool((box == kind).any())

    if near(NEST, 2):
        return "I'm near my nest."
    if near(BUSH, 1):
        return "I'm next to a berry bush."
    if near(THORN, 1):
        return "I'm near some thorns."
    if near(BUSH, 2):
        return "I'm near a berry bush."
    if near(STONE, 1):
        return "I'm next to a stone."
    if near(WALL, 1):
        return "I'm by the wall of my garden."
    return "I'm out in the middle of my garden."


def learned_facts(mind: Mind) -> dict[str, list[str]]:
    """What it has found out about the kinds of things in its world, by what they're like."""
    facts: dict[str, list[str]] = {"eat": [], "hurt": [], "way": [], "warm": []}
    for k in mind.vision.kinds.alive():
        known = mind.knowledge.describe(k)
        color = mind.kind_color(k)
        name = mind.lexicon.name_for(k)
        thing = f"{color} things" + (f' ("{name}")' if name else "")
        if "good to eat" in known:
            facts["eat"].append(thing)
        if "it hurts" in known:
            facts["hurt"].append(thing)
        if "in the way" in known:
            facts["way"].append(thing)
        if "warm" in known:
            facts["warm"].append(f"{color} places")
    return {k: list(dict.fromkeys(v)) for k, v in facts.items()}


def listing(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def knowledge(mind: Mind) -> str:
    facts = learned_facts(mind)
    said = []
    if facts["eat"]:
        said.append(f"{listing(facts['eat'][:2]).capitalize()} are good to eat.")
    if facts["hurt"]:
        said.append(f"{listing(facts['hurt'][:2]).capitalize()} hurt.")
    if facts["warm"]:
        said.append(f"{listing(facts['warm'][:1]).capitalize()} are warm.")
    if facts["way"] and len(said) < 2:
        said.append(f"{listing(facts['way'][:2]).capitalize()} are in the way.")
    return " ".join(said)


def first_person(milestone: str, name: str) -> str:
    text = milestone.replace(f"{name} came into the world", "came into the world").replace("it knows", "I know")
    text = text.replace(" its ", " my ").replace(": of ", ": ")
    return f"I {text}."


def memory(mind: Mind) -> str:
    stones = [text for _, text in mind.me.milestones[-3:]]
    if not stones:
        return ""
    return first_person(stones[-1], mind.me.name)


def known_words(words: list[str]) -> str:
    if not words:
        return "I don't know any words yet."
    return f"I know the word {words[0]}." if len(words) == 1 else f"I know the words {listing(words[:6])}."


def notes(mind: Mind) -> str:
    """What it knows, brought to mind as inner speech before it answers someone."""
    words = mind.lexicon.vocabulary()
    parts = [
        f"I'm {mind.me.name}, {age_words(mind)}.",
        place(mind),
        known_words(words),
        knowledge(mind),
        memory(mind),
    ]
    return " ".join(p for p in parts if p)


def _drive(mind: Mind, i: int) -> float:
    return float(mind.body.drives()[i])


def _yes_no(level: float, word: str) -> str:
    if level >= 0.7:
        return f"Yes, I'm very {word}."
    if level >= 0.35:
        return f"Yes, I'm {word}."
    if level >= 0.15:
        return f"A little. I'm a little {word}."
    return f"No, I'm not {word}."


def feeling(mind: Mind) -> str:
    """How it is, in a sentence or two: its strongest need, or that it's fine, and how that feels."""
    b = mind.body
    if b.asleep:
        return "I'm asleep."
    drives = b.drives()
    need = int(np.argmax(drives))
    if drives[need] < 0.3:
        first = "I feel good." if mind.valence > 0.2 else "I feel fine."
    else:
        first = f"I'm {need_words(need, float(drives[need]), b.cold())}."
        if mind.valence < -0.2:
            first += " It feels bad."
    return first


def seeing(mind: Mind) -> str:
    content = mind.workspace.content
    if mind.body.asleep:
        return "Nothing. I'm asleep."
    if content is not None and content.source == "vision":
        seen = content.label.split(" (")[0]
        if content.kind >= 0:
            known = mind.knowledge.describe(content.kind)
            if "good to eat" in known:
                return f"I see {seen}. I think it's good to eat."
            if "it hurts" in known:
                return f"I see {seen}. I think it hurts."
        return f"I see {seen}."
    return "Nothing special right now."


def doing(mind: Mind) -> str:
    if mind.body.asleep:
        return "I'm sleeping."
    return {
        "food": "I'm looking for food.",
        "warmth": "I'm trying to get warm." if mind.body.cold() else "I'm trying to cool down.",
        "healing": "I'm resting, so I can heal.",
        "sleep": "I'm going to my nest to sleep.",
        "explore": "I'm looking around.",
    }[mind.goals.current]


def thinking(mind: Mind) -> str:
    content = mind.workspace.content
    if content is None:
        return "Nothing much."
    if content.source == "vision":
        return f"I'm looking at {content.label.split(' (')[0]}."
    if content.source in ("memory", "imagination"):
        return f"I'm {content.label}."
    if content.source == "touch":
        return "About how that felt."
    if content.source == "hearing":
        return "About what you're saying."
    if content.source == "body":
        return f"About being {content.label}."
    if content.source == "smell":
        return "About something sweet I can smell."
    return "About what you said."


def answers(mind: Mind) -> dict[str, str]:
    """What its state and what it knows say the answer to each kind of question is, right now."""
    b = mind.body
    conclusions = mind.me.conclusions()
    facts = learned_facts(mind)
    words = mind.lexicon.vocabulary()
    hurt = _drive(mind, 2)
    temperature = _drive(mind, 1)
    content = mind.workspace.content
    greeting = "Hi! I'm asleep." if b.asleep else f"Hi! {feeling(mind)}"
    return {
        "feel": feeling(mind),
        "hungry": _yes_no(_drive(mind, 0), "hungry"),
        "cold": _yes_no(temperature, "cold") if b.cold() or temperature < 0.15 else "No, I'm hot.",
        "hot": _yes_no(temperature, "hot") if not b.cold() or temperature < 0.15 else "No, I'm cold.",
        "tired": _yes_no(_drive(mind, 3), "tired"),
        "hurt": "Yes, it hurts." if hurt >= 0.35 else "A little." if hurt >= 0.15 else "No, nothing hurts.",
        "good": "Yes, I feel good."
        if mind.valence > 0.2
        else "No, I feel bad."
        if mind.valence < -0.2
        else "I feel okay.",
        "bad": "Yes, I feel bad."
        if mind.valence < -0.2
        else "No, I feel good."
        if mind.valence > 0.2
        else "No, I feel okay.",
        "see": seeing(mind),
        "smell": "I smell something sweet."
        if content is not None and content.source == "smell"
        else "Nothing right now.",
        "doing": doing(mind),
        "thinking": thinking(mind),
        "time": f"It's {mind.time_of_day}.",
        "asleep": "Yes, I'm asleep." if b.asleep else "No, I'm awake.",
        "awake": "No, I'm asleep." if b.asleep else "Yes, I'm awake.",
        "name": f"My name is {mind.me.name}.",
        "age": f"I'm {age_words(mind)}.",
        "place": place(mind),
        "home": "I live in a small garden. My nest is in the corner.",
        "words": known_words(words),
        "talk": "A little. I'm still learning words.",
        "learned": knowledge(mind) or "Not much yet. I'm still finding out.",
        "eat": f"{listing(facts['eat'][:2]).capitalize()} are good to eat." if facts["eat"] else "I don't know yet.",
        "danger": f"{listing(facts['hurt'][:2]).capitalize()} hurt." if facts["hurt"] else "Nothing has hurt me yet.",
        "remember": memory(mind) or "Not much yet.",
        "alive": conclusions[-1],
        "feelings": "Yes. Some things feel good to me and some feel bad."
        if mind.me.evidence["feeling"] > 0.5
        else "I don't know yet.",
        "what": f"I'm {mind.me.name}. I live in a garden. {conclusions[-1]}",
        "hello": greeting,
        "thanks": "You're welcome.",
        "bye": "Bye! Come back soon.",
    }


def casual(text: str, rng: random.Random) -> str:
    """The way people actually type: often in lower case, without the question mark, sometimes by name."""
    if rng.random() < 0.15:
        text = rng.choice(("Haven, ", "Hey Haven, ", "Hey, ", "So ", "Ok, ")) + text[0].lower() + text[1:]
    elif rng.random() < 0.1:
        end = text[-1] if text[-1] in "?.!" else ""
        text = text.rstrip("?.!") + rng.choice((", Haven", " haven", " Haven")) + end
    if rng.random() < 0.3:
        text = text.rstrip("?.!")
    elif rng.random() < 0.1:
        text = text.rstrip("?.!") + rng.choice(("??", "!", " ?"))
    if rng.random() < 0.45:
        text = text.lower()
    return text


def dialog(moment: dict, rng: random.Random, turns: int | None = None) -> list[tuple[str, str]]:
    """A short conversation at one moment: (what the person said, what Haven answers), a few times over."""
    turns = turns or rng.choice((1, 1, 2, 3))
    said = []
    for _ in range(turns):
        roll = rng.random()
        if roll < 0.1:
            said.append((rng.choice(UNKNOWN_FORMS).format(rng.choice(UNKNOWN_TOPICS)), DONT_KNOW))
        elif roll < 0.14:
            said.append((rng.choice(OTHER_QUESTIONS), DONT_KNOW))
        else:
            intent = rng.choice(list(QUESTIONS))
            said.append((rng.choice(QUESTIONS[intent]), moment["answers"][intent]))
    return [(casual(question, rng), answer) for question, answer in said]
