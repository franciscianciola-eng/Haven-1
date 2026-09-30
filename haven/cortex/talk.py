"""Talking with people: how Haven learns to answer in its own words, from its own state.

Every answer it learns is worked out from a moment of one of its simulated lives. How it
feels, what it sees, what it wants and the time of day it reads straight from its
workspace tokens. What it knows comes to mind first, as a line of inner speech, before it
answers: its name and age, where it is, what it did today and what it likes, and whatever
the words bring up: what it has found out about a thing in its valley, its life story,
what it can do. When someone talks with it, their name and what they've told it come to
mind too, and so does anything it has read about what they ask. When a question isn't
about anything it has experienced, been told or read, the answer it learns is that it
doesn't know. It can then go and read about it, and the next time, it remembers.

The sentences are templates written by people, like the words a parent gives a child for
what the child is feeling and doing. What Haven learns is to say the right one at the
right time, because its state and its memories are what they are.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

import numpy as np

from ..mind import Mind, need_words
from ..personality import TRAITS
from ..selfmodel import VERDICTS
from ..world import BELL, DAY, FIRE, NEST, SAND, SEASON_DAYS, SEASONS, THORN, TREE, TURNING, WATER, YEAR
from .book import BOOK
from .stories import STORIES, Story

# --- questions about how it is, what it's doing, and what it knows ------------------------

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
        "How have you been?",
        "Are you alright?",
        "How are things?",
        "Everything okay?",
        "How's your day going?",
        "How are you now?",
        "How are you feeling now?",
    ),
    "hungry": (
        "Are you hungry?",
        "Do you want something to eat?",
        "Are you starving?",
        "are you hungry",
        "Do you need food?",
        "Do you want to eat something?",
        "hungry?",
    ),
    "cold": (
        "Are you cold?",
        "Do you feel cold?",
        "Are you freezing?",
        "are you cold",
        "Is it cold where you are?",
        "Do you need to warm up?",
        "cold?",
    ),
    "hot": (
        "Are you hot?",
        "Do you feel hot?",
        "Are you too warm?",
        "Is it too hot?",
        "Are you too hot?",
        "Are you warm?",
        "Do you need to cool down?",
    ),
    "tired": (
        "Are you tired?",
        "Are you sleepy?",
        "Do you need to sleep?",
        "are you tired",
        "Do you want to rest?",
        "Are you worn out?",
        "Do you need a nap?",
        "sleepy?",
    ),
    "hurt": (
        "Are you hurt?",
        "Does anything hurt?",
        "Are you in pain?",
        "Does it hurt?",
        "Did you get hurt?",
        "Are you okay, are you hurt?",
        "Is anything hurting you?",
        "Do you feel pain?",
    ),
    "good": (
        "Are you happy?",
        "Do you feel good?",
        "Are you in a good mood?",
        "Is everything good?",
        "Are you having fun?",
        "Are you feeling good?",
        "Are you glad?",
    ),
    "bad": (
        "Are you sad?",
        "Do you feel bad?",
        "Are you upset?",
        "Is something wrong?",
        "Are you unhappy?",
        "What's wrong?",
        "Are you feeling bad?",
        "Are you scared?",
    ),
    "see": (
        "What do you see?",
        "What are you looking at?",
        "What's in front of you?",
        "What can you see?",
        "What do you see right now?",
        "What's around you?",
        "What are you looking at right now?",
        "Look around, what do you see?",
    ),
    "smell": (
        "What do you smell?",
        "Can you smell anything?",
        "Do you smell something?",
        "What can you smell?",
        "Do you smell anything?",
        "What does it smell like?",
    ),
    "doing": (
        "What are you doing?",
        "What are you up to?",
        "What do you want?",
        "What do you want to do?",
        "Where are you going?",
        "what are you doing",
        "What are you doing now?",
        "What are you doing right now?",
        "What's going on?",
        "What are you trying to do?",
    ),
    "thinking": (
        "What are you thinking about?",
        "What's on your mind?",
        "What are you thinking?",
        "What's on your mind right now?",
        "What are you thinking about right now?",
        "Penny for your thoughts?",
    ),
    "time": (
        "What time is it?",
        "Is it day or night?",
        "What time of day is it?",
        "Is it dark?",
        "Is it night?",
        "Is it morning?",
        "Is it daytime?",
        "Is it light out?",
    ),
    "asleep": (
        "Are you asleep?",
        "Are you sleeping?",
        "Are you sleeping right now?",
        "Did I wake you?",
        "Are you dreaming?",
    ),
    "awake": (
        "Are you awake?",
        "Are you up?",
        "Are you awake now?",
        "Are you up yet?",
        "Wake up!",
    ),
    "name": (
        "What's your name?",
        "What is your name?",
        "Who are you?",
        "what's your name",
        "Do you have a name?",
        "What should I call you?",
        "What are you called?",
        "Tell me your name.",
    ),
    "age": (
        "How old are you?",
        "What's your age?",
        "How long have you been alive?",
        "How many days old are you?",
        "When were you born?",
        "How long have you lived?",
        "How old are you now?",
    ),
    "place": (
        "Where are you?",
        "Where are you right now?",
        "Where are you now?",
        "Where are you in your valley?",
        "Where are you standing?",
        "Where are you at?",
        "Where in the valley are you?",
    ),
    "home": (
        "Where do you live?",
        "Where is your home?",
        "Where do you sleep?",
        "Do you have a home?",
    ),
    "world": (
        "Tell me about your world.",
        "What's your world like?",
        "Describe your world.",
        "What is your valley like?",
        "What's in your valley?",
        "What have you found in your valley?",
    ),
    "words": (
        "What words do you know?",
        "Do you know any words?",
        "What words have you learned?",
        "Which words do you know?",
        "What words do you understand?",
        "Do you know any words yet?",
    ),
    "talk": (
        "Can you talk?",
        "Can you speak?",
        "Do you understand me?",
        "Can you hear me?",
        "Can you understand me?",
        "Do you speak English?",
        "Can you talk to me?",
    ),
    "learned": (
        "What have you learned?",
        "What do you know?",
        "What have you found out?",
        "What have you learned so far?",
        "What have you figured out?",
        "What do you know now?",
    ),
    "eat": (
        "What is good to eat?",
        "What do you eat?",
        "What food do you like?",
        "What's your favorite food?",
        "What do you like to eat?",
        "What can you eat?",
        "What food is good?",
        "What should you eat?",
    ),
    "danger": (
        "What hurts you?",
        "What's dangerous?",
        "What should you stay away from?",
        "What makes you sick?",
        "What should you avoid?",
        "What's bad for you?",
        "What makes you hurt?",
    ),
    "remember": (
        "What do you remember?",
        "What happened to you?",
        "Tell me something you remember.",
        "What do you remember most?",
        "Do you remember anything?",
    ),
    "best day": (
        "What was your best day?",
        "What's your happiest memory?",
        "What was the best day of your life?",
        "What's your favorite memory?",
        "Tell me about a good day you had.",
        "What was the best day you ever had?",
        "Tell me about your best day.",
        "What was your happiest day?",
    ),
    "worst day": (
        "What was your worst day?",
        "What's your saddest memory?",
        "What was the worst day of your life?",
        "Tell me about a bad day you had.",
        "What was the worst day you ever had?",
        "Tell me about your worst day.",
        "What was your saddest day?",
    ),
    "personality": (
        "What are you like?",
        "Describe your personality.",
        "What's your personality like?",
        "Tell me about your personality.",
        "What sort of creature are you?",
        "How would you describe yourself?",
        "What are you like now?",
        "What are you like these days?",
        "What kind of personality do you have?",
        "Tell me what you're like.",
        "What's your character like?",
    ),
    "changed": (
        "Have you changed?",
        "How have you changed?",
        "Are you different now?",
        "Have you changed since you were little?",
        "How are you different now?",
        "Have you changed much?",
        "How have you changed over the years?",
        "Did you change?",
        "Are you different from before?",
    ),
    "favorite place": (
        "What's your favorite place?",
        "Where's your favorite place?",
        "Where do you like to go?",
        "Where do you like to be?",
        "Where's your favorite spot?",
        "What's your favourite place?",
        "Where do you like to go most?",
        "What's your favorite spot in the valley?",
        "Where do you like being most?",
    ),
    "season": (
        "What season is it?",
        "Which season is it?",
        "What season is it now?",
        "Is it summer?",
        "Is it winter?",
        "What's it like in your valley right now?",
        "Which season are we in?",
        "Is it spring?",
        "Is it autumn?",
        "Is it fall?",
        "What season is it in your valley?",
    ),
    "weather": (
        "What's the weather like?",
        "How's the weather?",
        "What's the weather like today?",
        "Is it nice out?",
        "How's the weather in your valley?",
        "How's the weather today?",
        "What's the weather like now?",
        "What's the weather doing?",
    ),
    "favorite season": (
        "What's your favorite season?",
        "Which season do you like best?",
        "What season do you like most?",
        "Do you have a favorite season?",
        "Which season is your favorite?",
        "What's your favourite season?",
        "Which season do you like the most?",
    ),
    "afraid": (
        "What are you afraid of?",
        "What scares you?",
        "Are you scared of anything?",
        "What frightens you?",
        "Is there anything you're afraid of?",
        "What are you scared of?",
        "Are you afraid of anything?",
        "What do you fear?",
        "Is there anything that scares you?",
    ),
    "story": (
        "Tell me your story.",
        "Tell me about your life.",
        "What has happened in your life?",
        "What's your life been like?",
        "Tell me the story of your life.",
        "How did your life begin?",
        "What's happened to you so far?",
    ),
    "day": (
        "What did you do today?",
        "How was your day?",
        "What have you done today?",
        "What happened today?",
        "Did you do anything fun today?",
        "How has your day been?",
        "What have you been doing?",
        "What did you do?",
        "Tell me about your day.",
    ),
    "likes": (
        "What do you like?",
        "What do you like to do?",
        "What's your favorite thing?",
        "What do you enjoy?",
        "What do you like doing?",
        "What don't you like?",
        "Is there anything you don't like?",
        "What makes you happy?",
        "What's your favorite thing to do?",
        "What do you love?",
        "What do you hate?",
    ),
    "play": (
        "Do you want to play?",
        "Let's play!",
        "Want to play?",
        "Shall we play?",
        "Do you like to play?",
        "Wanna play?",
        "Can we play?",
    ),
    "can": (
        "What can you do?",
        "What are you able to do?",
        "What can you do in your valley?",
        "What things can you do?",
        "What are you good at?",
        "What have you learned to do?",
    ),
    "alive": (
        "Are you alive?",
        "Do you think you're alive?",
        "Are you real?",
        "Are you a real creature?",
        "Are you a living thing?",
        "Are you alive or not?",
    ),
    "conscious": (
        "Are you conscious?",
        "Are you aware?",
        "Are you self-aware?",
        "Are you aware of yourself?",
        "Do you know you exist?",
        "Do you have a mind?",
    ),
    "ai": (
        "Are you an AI?",
        "Are you a robot?",
        "Are you a computer program?",
        "Are you a machine?",
        "Are you a chatbot?",
        "Are you just a program?",
    ),
    "dream": ("Do you dream?", "Do you have dreams?", "What do you dream about?", "Did you dream last night?"),
    "fond": (
        "Do you like me?",
        "Do you like talking to me?",
        "Are you happy to see me?",
        "Do you get lonely?",
        "Are you lonely?",
        "Do you miss me?",
    ),
    "color": ("What's your favorite color?", "What is your favourite colour?", "Do you have a favorite color?"),
    "joke": ("Tell me a joke.", "Do you know any jokes?", "Say something funny.", "Can you tell jokes?"),
    "being": (
        "What does it feel like to be you?",
        "What is it like to be you?",
        "What's it like being you?",
        "How does it feel to be you?",
    ),
    "feelings": (
        "Do you have feelings?",
        "Can you feel things?",
        "Do you feel anything?",
        "Can you feel?",
        "Do you have emotions?",
        "Do you feel things?",
    ),
    "what": (
        "What are you?",
        "What kind of thing are you?",
        "Tell me about yourself.",
        "Describe yourself.",
        "What exactly are you?",
        "Are you an animal?",
    ),
    "maker": (
        "Who made you?",
        "Who created you?",
        "Where did you come from?",
        "Who are your parents?",
        "Who built you?",
        "Who brought you into the world?",
        "How were you made?",
    ),
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
        "Hiya",
        "Hey Haven",
        "Good afternoon",
        "Yo",
        "Morning",
        "Are you there?",
        "You there?",
        "Hello?",
        "Is anyone there?",
    ),
    "thanks": (
        "Thank you",
        "Thanks",
        "Thanks Haven",
        "Thank you so much.",
        "Thanks a lot",
        "Thank you Haven",
        "Thanks so much",
        "Cheers",
    ),
    "bye": (
        "Bye",
        "Goodbye",
        "See you later",
        "Bye Haven",
        "Good night",
        "I have to go now.",
        "See you",
        "Bye bye",
        "Goodnight",
        "Talk to you later",
        "See ya",
        "I'm leaving now.",
    ),
}

# Small talk: what people say that isn't a question, and what it says back.
SMALL_TALK: dict[str, tuple[tuple[str, ...], str]] = {
    "kind": (
        (
            "I love you",
            "I like you",
            "You're cute",
            "You're smart",
            "Good job!",
            "Well done",
            "You're great",
            "You're my friend",
            "You're doing great",
        ),
        "Thank you! I like it when you talk with me.",
    ),
    "sorry": (("Sorry", "I'm sorry", "Sorry about that", "My bad"), "That's okay."),
    "okay": (("Okay", "Ok", "Cool", "Nice", "Alright", "I see", "Wow"), "Okay."),
    "back": (("I'm back", "I'm back!", "I'm here", "I'm home"), "Welcome back!"),
    "tired": (("I'm tired", "I'm so tired", "I'm sleepy"), "You should rest. Resting always helps me."),
    "sad": (
        ("I'm sad", "I feel sad", "I'm not feeling good", "I'm sick", "I had a bad day"),
        "I'm sorry. I hope you feel better soon.",
    ),
    "happy": (("I'm happy", "I feel great", "I'm good", "I'm fine", "I had a good day"), "I'm glad!"),
    "bored": (("I'm bored", "I'm so bored"), "You could watch me explore my valley."),
}

# Questions about the world beyond its valley: it hasn't experienced these, though it may have read about them.
UNKNOWN_FORMS = (
    "What is {}?",
    "Tell me about {}.",
    "Do you know about {}?",
    "What do you know about {}?",
    "Can you tell me about {}?",
    "Have you heard of {}?",
    "Explain {}.",
)
# (what people call it, the article it's read in, that article's first sentence), from the book it's born having read
READINGS = tuple((e.topic, e.title, e.sentences[0]) for e in BOOK)
UNKNOWN_TOPICS = tuple(topic for topic, _, _ in READINGS)
READ_FORMS = ("Read about {}.", "Can you read about {}?", "Look up {}.", "Find out about {}.")  # asking it to read
MORE_FORMS = ("Tell me more about {}.", "What else do you know about {}?", "What else did you read about {}?")
MORE_PLAIN = ("Tell me more.", "What else?", "Go on.", "And what else?", "Tell me more about it.", "Anything else?")
MORE = re.compile(r"\b(tell me more|what else|go on\b|anything else|and then|keep going)", re.I)


def more_note(title: str, sentence: str | None) -> str:
    """What comes to mind when asked for more about something it read: the next thing it read, or that that's all."""
    return f"More that I read about {title}: {sentence}" if sentence else f"That's all I've read about {title}."


def more_answer(title: str, sentence: str | None) -> str:
    return f"It also says: {sentence}" if sentence else f"That's all I've read about {title}."


# --- what it has read lately, and something it read ---------------------------------------------------------

LATELY_FORMS = (
    "What have you read lately?",
    "What have you been reading?",
    "Have you read anything new?",
    "What did you read today?",
    "Read anything interesting lately?",
    "What have you been reading about?",
)
LATELY = re.compile(
    r"^(?:so |and )?(?:what have you (?:been )?read(?:ing)?(?: about)?|(?:have you )?read anything(?: new| interesting)?|"
    r"what did you read)(?: lately| recently| today)?[?.! ]*$",
    re.I,
)
FACT_FORMS = (
    "Tell me something interesting.",
    "Tell me something you read.",
    "Tell me a fact.",
    "Teach me something.",
    "Tell me something I don't know.",
    "Tell me a fun fact.",
)
FACT_ASK = re.compile(
    r"^(?:can you |please |will you )?(?:tell me (?:something(?: interesting| you read| new| cool| fun| i don't know)?|"
    r"an? (?:fun |cool |random |interesting )?fact)|teach me something(?: new)?|say something interesting)[?.! ]*$",
    re.I,
)
LATELY_TITLES = (  # things it might have read about lately, for practice at telling what it read
    *(e.title for e in BOOK),
    *"Butterfly Flower Apple Mushroom Pond Bell Hill Nest Thorn Fire Boston Sushi Insect Oxidation Hospital Octopus "
    "Honey Lighthouse Bicycle Castle Tornado Volleyball Saturn Kangaroo Chemistry Opera Glacier Pottery".split(),
    "Mount Fuji",
    "Rolling Stones",
    "Rock (geology)",
    "Apple tree",
    "Musical instrument",
)


SELF_ASKED = re.compile(  # asking about it, or about where it is and how it is, not about the world
    r"^(?:(?:what|where|when|why|how|who|which)(?: \w+){0,3} )?(?:is it|are you|do you|did you|can you|could you|"
    r"have you|were you|will you|would you|should you|made you|created you|built you)\b",
    re.I,
)
KNOWING = re.compile(r"\b(?:know|knew|heard|read|tell me about|learn(?:ed)? about|think (?:about|of))\b", re.I)


def about_haven(text: str) -> bool:
    """Whether someone is asking about it (or about how things are where it is), not about the world."""
    t = bare(text)
    return bool(SELF_ASKED.match(t)) and not KNOWING.search(t)


def bare(text: str) -> str:
    """What someone said without calling it by name, and without the question mark."""
    text = _ADDRESS.sub("", " ".join(text.strip().split()), count=1)
    return re.sub(r",?\s*haven\s*([?.!]*)$", r"\1", text, flags=re.I).strip()


def lately_note(titles: list[str]) -> str:
    """What it has read lately (not counting the little book it was born having read)."""
    return f"Lately I read about {and_list(titles)}." if titles else "I haven't read anything new lately."


def lately_answer(titles: list[str]) -> str:
    return f"I read about {and_list(titles)}." if titles else "I haven't read anything new lately."


def and_list(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# --- being taught about the world ---------------------------------------------------------------------------

CAPITALS = (
    ("Peru", "Lima"),
    ("Kenya", "Nairobi"),
    ("Norway", "Oslo"),
    ("Chile", "Santiago"),
    ("Greece", "Athens"),
    ("Cuba", "Havana"),
    ("Ghana", "Accra"),
    ("Nepal", "Kathmandu"),
    ("Poland", "Warsaw"),
    ("Sweden", "Stockholm"),
    ("Vietnam", "Hanoi"),
    ("Portugal", "Lisbon"),
    ("Ireland", "Dublin"),
    ("Turkey", "Ankara"),
    ("Mexico", "Mexico City"),
    ("Argentina", "Buenos Aires"),
    ("Iceland", "Reykjavik"),
    ("Morocco", "Rabat"),
)
ANIMALS_CAN = (
    ("frogs", "jump very far"),
    ("owls", "see in the dark"),
    ("cheetahs", "run very fast"),
    ("bats", "fly at night"),
    ("dolphins", "talk to each other with clicks"),
    ("parrots", "learn words"),
    ("octopuses", "change color"),
    ("camels", "go a long time without water"),
    ("ants", "carry heavy things"),
    ("penguins", "swim very well"),
    ("kangaroos", "jump very high"),
    ("snails", "sleep for a long time"),
)
MADE_OF = (
    ("glass", "sand"),
    ("paper", "wood"),
    ("cheese", "milk"),
    ("bread", "flour"),
    ("butter", "cream"),
    ("chocolate", "cocoa beans"),
    ("jam", "fruit"),
    ("candles", "wax"),
    ("honey", "nectar"),
    ("rope", "fibers"),
)
WROTE = (
    ("Mark Twain", "Tom Sawyer"),
    ("Jane Austen", "Pride and Prejudice"),
    ("Roald Dahl", "Matilda"),
    ("Mary Shelley", "Frankenstein"),
    ("Lewis Carroll", "Alice in Wonderland"),
    ("J. R. R. Tolkien", "The Hobbit"),
    ("Homer", "the Odyssey"),
    ("Charles Dickens", "Oliver Twist"),
)
MOONS = (("Mars", "two moons"), ("Venus", "no moons"), ("Mercury", "no moons"), ("Neptune", "sixteen moons"))
PROPER = frozenset(
    {c for pair in CAPITALS for c in pair}
    | {who.split()[0] for who, _ in WROTE}
    | set(
        "Paris France Japan Tokyo London New China Egypt Italy India Brazil Canada Australia Rome Africa Europe "
        "Asia America William Albert".split()
    )
    | {p for p, _ in MOONS}
    | set(
        "Mercury Venus Earth Mars Jupiter Saturn Uranus Neptune Pluto Monday Tuesday Wednesday Thursday Friday "
        "Saturday Sunday January February March April May June July August September October November December "
        "God English French Spanish Chinese Japanese German Italian".split()
    )
)
LESSON = re.compile(
    r"^(?!(?:i|i'm|im|my|me|you|you're|your|we|we're|our|they|he|she|it|it's|its|that|that's|this|these|those|"
    r"there|there's|here|here's|what|who|why|how|when|where|which|do|does|did|can|could|would|will|should|please|"
    r"let's|haven|yes|no|not|so|and|but|ok|okay|well|oh|hi|hello|thanks|thank)\b)"
    r"[a-z0-9][\w'.,-]*(?: [\w'.,-]+){0,6}? (?:is|are|was|were|has|have|had|can|could|will|wrote|made|makes?|"
    r"lives?|eats?|comes? from|grows?|means?|lays?|needs?|likes?|gives?|helps?|uses?|builds?|flies|fly|swims?|"
    r"sleeps?|hunts?|contains?|boils?|freezes?|melts?|orbits?|goes|go|runs?|belongs?|won|invented|discovered|"
    r"painted|sings?|sang|plays?|turns?|gets?|keeps?|brings?|carries|carry) \S.*$",
    re.I,
)


def a_lesson(rng: random.Random) -> tuple[str, str, tuple[str, ...]]:
    """Something a person might teach it about the world: (what they say, what it keeps, how they might ask later)."""
    roll = rng.randrange(5)
    if roll == 0:
        country, city = rng.choice(CAPITALS)
        return (
            f"The capital of {country} is {city}.",
            f"the capital of {country} is {city}",
            (
                f"What is the capital of {country}?",
                f"What's the capital city of {country}?",
                f"Do you know the capital of {country}?",
                f"What do you know about {country}?",
                f"Tell me about {country}.",
            ),
        )
    if roll == 1:
        animals, can = rng.choice(ANIMALS_CAN)
        return (
            f"{animals.capitalize()} can {can}.",
            f"{animals} can {can}",
            (
                f"What can {animals} do?",
                f"Can {animals} {can}?",
                f"Tell me about {animals}.",
                f"What do you know about {animals}?",
                f"Do you know anything about {animals}?",
            ),
        )
    if roll == 2:
        thing, stuff = rng.choice(MADE_OF)
        verb = "are" if thing.endswith("s") else "is"
        return (
            f"{thing.capitalize()} {verb} made from {stuff}.",
            f"{thing} {verb} made from {stuff}",
            (
                f"What {verb} {thing} made of?",
                f"What {verb} {thing} made from?",
                f"What do you know about {thing}?",
            ),
        )
    if roll == 3:
        who, book = rng.choice(WROTE)
        return (
            f"{who} wrote {book}.",
            f"{who} wrote {book}",
            (f"Who wrote {book}?", f"What did {who} write?", f"Who is {who}?", f"Tell me about {book}."),
        )
    planet, moons = rng.choice(MOONS)
    return (
        f"{planet} has {moons}.",
        f"{planet} has {moons}",
        (
            f"How many moons does {planet} have?",
            f"Does {planet} have any moons?",
            f"What do you know about {planet}?",
        ),
    )


def lesson(text: str) -> str | None:
    """Something about the world someone is teaching it ("The capital of Peru is Lima."), as it would say it back."""
    t = CORRECTION.sub("", bare(text), count=1).strip()
    if not t or "?" in t or len(t.split()) < 3 or not LESSON.match(t):
        return None
    if introduced(t) or statement(t) or request(t) or sum_of(t):
        return None
    t = t.rstrip(".! ")
    first = t.split()[0]
    keep = first in PROPER or first in PEOPLE or (len(t.split()) > 1 and t.split()[1][:1].isupper())
    return t if keep or first.isupper() else first.lower() + t[len(first) :]


def best_lesson(text: str, lessons: list[str]) -> str | None:
    """What it was taught that a question is about: the lesson that shares the most of its words (the latest, if
    two do as well), and doesn't leave much of the question out."""
    asked = set(_tokens(text))
    best, score = None, 0.0
    for taught in reversed(lessons):
        words = set(_tokens(taught))
        s = len(asked & words) - 0.5 * len(asked - words)
        if s > score:
            best, score = taught, s
    return best if score >= 1.0 else None


# --- working out sums ---------------------------------------------------------------------------------------

NUMBER_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty"
).split()
OPERATIONS = {  # how people say it: (how it says it, what it does)
    "plus": ("plus", "+"),
    "+": ("plus", "+"),
    "and": ("plus", "+"),
    "add": ("plus", "+"),
    "minus": ("minus", "-"),
    "-": ("minus", "-"),
    "take away": ("minus", "-"),
    "times": ("times", "*"),
    "multiplied by": ("times", "*"),
    "x": ("times", "*"),
    "*": ("times", "*"),
    "×": ("times", "*"),
    "divided by": ("divided by", "/"),
    "/": ("divided by", "/"),
    "÷": ("divided by", "/"),
    "over": ("divided by", "/"),
}
_NUMBER = r"(\d+(?:\.\d+)?|" + "|".join(NUMBER_WORDS) + r")"
_SUM = re.compile(
    r"^(?:what(?:'s| is)|how much is|can you (?:work out|do|calculate)|calculate|work out|do)?\s*"
    + _NUMBER
    + r"\s*(plus|\+|and|add|minus|-|take away|times|multiplied by|x|\*|×|divided by|/|÷|over)\s*"
    + _NUMBER
    + r"\s*(?:equals?|is)?\s*[?.!=]*$",
    re.I,
)


def sum_of(text: str) -> tuple[str, str] | None:
    """A sum someone asks it to work out ("what's 12 times 7?"): (the sum, as it says it; the answer)."""
    asked = bare(text)
    found = _SUM.match(asked)
    if not found:
        return None
    a, op, b = found.groups()
    x, y = (float(n) if n[0].isdigit() else float(NUMBER_WORDS.index(n.lower())) for n in (a, b))
    said, do = OPERATIONS[op.lower()]
    if op.lower() == "and" and not asked.lower().startswith(("what", "how much")):
        return None  # "two and two" alone isn't asking for a sum
    if do == "/" and y == 0:
        return None
    result = {"+": x + y, "-": x - y, "*": x * y, "/": x / y if y else 0.0}[do]
    number = lambda v: str(int(v)) if float(v).is_integer() else f"{v:.2f}".rstrip("0").rstrip(".")  # noqa: E731
    return f"{number(x)} {said} {number(y)}", number(result)


SUM_FORMS = ("What's {}?", "What is {}?", "How much is {}?", "Can you work out {}?", "{}?", "Calculate {}.", "{} is?")


def sum_question(rng: random.Random) -> str:
    """A sum to ask it, the way people might type one."""
    op = rng.choice(("plus", "+", "minus", "-", "times", "x", "*", "divided by", "/", "multiplied by", "take away"))
    small = rng.random() < 0.5
    a, b = (rng.randint(0, 20), rng.randint(0, 20)) if small else (rng.randint(0, 999), rng.randint(0, 99))
    if OPERATIONS[op][1] == "/":
        b = rng.randint(1, 12)
        a = b * rng.randint(0, 20) if rng.random() < 0.8 else a
    if small and rng.random() < 0.4 and max(a, b) <= 20:
        a, b = NUMBER_WORDS[a], NUMBER_WORDS[b]
    gap = "" if op in "+-*/x" and rng.random() < 0.3 and op != "x" else " "
    return rng.choice(SUM_FORMS).format(f"{a}{gap}{op}{gap}{b}")


def sum_note(worked: tuple[str, str]) -> str:
    return f"I worked it out: {worked[0]} is {worked[1]}."


def sum_answer(worked: tuple[str, str]) -> str:
    return f"{worked[0][0].upper()}{worked[0][1:]} is {worked[1]}."


OTHER_QUESTIONS = (
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "Why is the sky blue?",
    "How far away is the moon?",
    "Who is the president?",
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

# --- the things in its valley ------------------------------------------------------------------


@dataclass(frozen=True)
class Thing:
    name: str  # the name it keeps what it knows under
    label: str  # how it starts a thought about it: "Apples", "The bell"
    one: str  # one of them, as it says it: "an apple", "the bell"
    they: bool  # whether it talks about them ("they") or it ("it")
    words: tuple[str, ...]  # the words for it (people's too)
    where: str  # where it is in the valley
    refs: tuple[str, ...]  # how people ask about it

    @property
    def it(self) -> str:
        return "they" if self.they else "it"

    @property
    def them(self) -> str:
        return "them" if self.they else "it"

    @property
    def any(self) -> str:
        return "any" if self.they else "it"


THINGS = {
    t.name: t
    for t in (
        Thing(
            "bush",
            "Berries",
            "a berry bush",
            True,
            ("berries", "berry", "berry bush", "berry bushes", "bush", "bushes"),
            "They grow on bushes all over the valley.",
            ("berries", "the berries", "berry bushes"),
        ),
        Thing(
            "apple",
            "Apples",
            "an apple",
            True,
            ("apple", "apples"),
            "They grow on the apple trees, in the north of the valley.",
            ("apples", "the apples"),
        ),
        Thing(
            "tree",
            "Apple trees",
            "an apple tree",
            True,
            ("tree", "trees", "apple tree", "apple trees", "orchard"),
            "They're in the north of the valley, not far from my nest.",
            ("the apple trees", "apple trees", "the trees", "trees"),
        ),
        Thing(
            "pond",
            "The pond",
            "the pond",
            False,
            ("pond", "water", "lake"),
            "It's in the south-east of the valley, with sand all around it.",
            ("the pond", "the water"),
        ),
        Thing(
            "bell",
            "The bell",
            "the bell",
            False,
            ("bell",),
            "It's on a little mound in the middle of the valley.",
            ("the bell",),
        ),
        Thing(
            "ball",
            "The ball",
            "the ball",
            False,
            ("ball",),
            "It rolls around, so it could be anywhere.",
            ("the ball",),
        ),
        Thing(
            "flower",
            "Flowers",
            "a flower",
            True,
            ("flower", "flowers"),
            "They grow all over the valley.",
            ("flowers", "the flowers"),
        ),
        Thing(
            "mushroom",
            "Mushrooms",
            "a mushroom",
            True,
            ("mushroom", "mushrooms"),
            "They grow in the middle of the valley.",
            ("mushrooms", "the mushrooms"),
        ),
        Thing(
            "toadstool",
            "Toadstools",
            "a toadstool",
            True,
            ("toadstool", "toadstools"),
            "They grow in the middle of the valley, near the mushrooms.",
            ("toadstools", "the toadstools"),
        ),
        Thing(
            "fire",
            "The fire",
            "the fire",
            False,
            ("fire", "campfire"),
            "It's in the south of the valley, not far from the pond.",
            ("the fire", "the campfire"),
        ),
        Thing(
            "butterfly",
            "Butterflies",
            "a butterfly",
            True,
            ("butterfly", "butterflies"),
            "They flutter all over the valley.",
            ("butterflies", "the butterflies"),
        ),
        Thing(
            "stone",
            "Stones",
            "a stone",
            True,
            ("stone", "stones", "rock", "rocks"),
            "They're here and there in the valley.",
            ("stones", "the stones", "rocks", "the rocks"),
        ),
        Thing(
            "thorns",
            "The thorns",
            "the thorns",
            True,
            ("thorn", "thorns"),
            "They're in the south-west of the valley.",
            ("the thorns", "thorns"),
        ),
        Thing(
            "nest",
            "My nest",
            "my nest",
            False,
            ("nest",),
            "It's in the north-west corner of the valley.",
            ("your nest",),
        ),
        Thing(
            "hill",
            "The hill",
            "the hill",
            False,
            ("hill", "hilltop", "cliff"),
            "It's in the north-east of the valley. Its south side is a cliff.",
            ("the hill",),
        ),
        Thing(
            "wall",
            "The wall",
            "the wall",
            False,
            ("wall", "walls"),
            "It goes all the way around the valley.",
            ("the wall",),
        ),
    )
}
_THING_WORDS = sorted(((w, t.name) for t in THINGS.values() for w in t.words), key=lambda item: -len(item[0]))
THING_QUESTIONS = {  # (asking about "it", asking about "them")
    "about": (
        ("What do you know about {}?", "Tell me about {}.", "What's {} like?"),
        ("What do you know about {}?", "Tell me about {}.", "What are {} like?"),
    ),
    "seen": (
        ("Have you seen {}?", "Have you found {}?", "Do you know {}?"),
        ("Have you seen any {}?", "Have you seen {}?", "Do you know {}?"),
    ),
    "like": (("Do you like {}?", "How do you feel about {}?"), ("Do you like {}?", "How do you feel about {}?")),
    "where": (("Where is {}?", "Where can I find {}?"), ("Where are {}?", "Where can you find {}?")),
    "eat": (("Can you eat {}?", "Is {} good to eat?"), ("Can you eat {}?", "Are {} good to eat?")),
    "do": (
        ("What does {} do?", "What happens when you touch {}?", "What can you do with {}?"),
        ("What do {} do?", "What happens when you touch {}?", "What can you do with {}?"),
    ),
}
PLAYING = {  # what it enjoys, as it would say it
    "bell": "ringing the bell",
    "ball": "pushing the ball",
    "flower": "smelling flowers",
    "tree": "shaking the apple trees",
    "apple": "eating apples",
    "bush": "eating berries",
    "mushroom": "eating mushrooms",
    "fire": "sitting by the fire",
    "pond": "drinking from the pond",
}
EFFECTS = (  # what happens with things, as it says it: the answer to "what does it do?"
    r"It rings when I touch it\.",
    r"It rolls when I push it\.",
    r"When I shake one, an apple falls\.",
    r"\w+ smell lovely\.",
    r"I drink from it\. The water is cool\.",
    r"It keeps me warm when I'm cold\.",
    r"It burned me\.|\w+ hurt me\.",
    r"\w+ (?:are|is) good to eat\.|\w+ made me sick\.",
)


def times(n: float) -> str:
    return "once" if n < 1.5 else "twice" if n < 2.5 else "a few times" if n < 5.5 else "many times"


def mentioned(text: str) -> list[str]:
    """The things in its valley that some words are about, in the order they come up."""
    low = text.lower()
    found: list[tuple[int, str]] = []
    taken: list[tuple[int, int]] = []
    for word, name in _THING_WORDS:
        for m in re.finditer(rf"\b{re.escape(word)}\b", low):
            if not any(a < m.end() and m.start() < b for a, b in taken):
                taken.append((m.start(), m.end()))
                found.append((m.start(), name))
    return list(dict.fromkeys(name for _, name in sorted(found)))


def thing_facts(t: Thing, s: dict) -> list[str]:
    """What it has found out about a thing, from what it has seen and done with it."""
    them, It = t.them, t.it.capitalize()
    facts = []
    ate, sick = s.get("ate", 0), s.get("sick", 0)
    if ate:
        if sick and 2 * sick >= ate:
            facts.append(f"I've eaten {them} {times(ate)}. {It} made me sick.")
        else:
            facts.append(f"I've eaten {them} {times(ate)}. {It} {'are' if t.they else 'is'} good to eat.")
    if s.get("drank"):
        facts.append("I drink from it. The water is cool.")
    if s.get("rang"):
        facts.append(f"I've rung it {times(s['rang'])}. It rings when I touch it.")
    if s.get("pushed"):
        facts.append(f"I've pushed it {times(s['pushed'])}. It rolls when I push it.")
    if s.get("shook"):
        facts.append(f"I've shaken {them} {times(s['shook'])}. When I shake one, an apple falls.")
    if s.get("smelled"):
        facts.append(f"{It} smell lovely.")
    if s.get("warmed"):
        facts.append("It keeps me warm when I'm cold.")
    if s.get("hurt"):
        facts.append("It burned me." if t.name == "fire" else f"{It} hurt me.")
    if s.get("climbed"):
        facts.append(f"I've climbed it {times(s['climbed'])}." + (" I've been to the top." if s.get("top") else ""))
    if s.get("slept"):
        facts.append("I sleep in it. It's warm and safe.")
    if not facts and s.get("bumped") and t.name != "bush":
        facts.append(f"I can't walk through {them}.")
    if not facts and s.get("tried"):
        facts.append(f"Biting or touching {them} does nothing.")
    if not facts:
        facts.append(f"I've seen {them}.")
    if s.get("joy", 0) >= 1.0:
        facts.append(f"I like {them}.")
    return facts[:3] + facts[-1:] if len(facts) > 4 else facts


def knows_of(s: dict) -> bool:
    return any(v for k, v in s.items() if k not in ("joy", "top"))


def thing_note(name: str, s: dict, where: str | None = None, with_where: bool = False) -> str:
    """What comes to mind about a thing in its valley."""
    t = THINGS[name]
    if not knows_of(s):
        return f"{t.label}: I haven't seen {t.any} yet."
    note = f"{t.label}: {' '.join(thing_facts(t, s))}"
    return note + f" {where or t.where}" if with_where else note


def thing_answer(kind: str, name: str, s: dict, where: str | None = None) -> str:
    """What it says when asked about a thing in its valley (worked out from what comes to mind about it)."""
    t = THINGS[name]
    them, It = t.them, t.it.capitalize()
    if not knows_of(s):
        return f"I haven't seen {t.any} yet."
    facts = " ".join(thing_facts(t, s))
    if kind == "about":
        return facts
    if kind == "seen":
        return f"Yes, I've seen {them}."
    if kind == "where":
        return where or t.where
    if kind == "like":
        if s.get("joy", 0) >= 1.0:
            return f"Yes, I like {them}."
        if s.get("hurt"):
            return "No. It burned me." if name == "fire" else f"No. {It} hurt me."
        if "made me sick" in facts:
            return f"No. {It} made me sick."
        return "I don't know yet."
    if kind == "eat":
        if "good to eat" in facts:
            return f"Yes, {t.it} {'are' if t.they else 'is'} good to eat."
        if "made me sick" in facts:
            return f"No. {It} made me sick."
        if "does nothing" in facts:
            return f"No. Biting or touching {them} does nothing."
        return "I've never tried."
    for pattern in EFFECTS:  # "do": what happens with it
        found = re.search(pattern, facts)
        if found:
            return found.group(0)
    if "does nothing" in facts:
        return "Nothing, as far as I know."
    return "I don't know yet."


# --- places ----------------------------------------------------------------------------------------


def region(mind: Mind, x: int, y: int) -> str:
    """Where a place is in the valley, in words."""
    w = mind.world
    grid = w.grid
    level = w.level(x, y)

    def near(kind: int, reach: int) -> bool:
        box = grid[max(0, y - reach) : y + reach + 1, max(0, x - reach) : x + reach + 1]
        return bool((box == kind).any())

    if level >= 3:
        return "on top of the hill"
    if level >= 1:
        return "on the hill" if x >= 14 else "on the little mound by the bell"
    if near(NEST, 2):
        return "near my nest"
    if near(WATER, 1):
        return "by the pond"
    if grid[y, x] == SAND:
        return "on the sand by the pond"
    if near(FIRE, 2):
        return "by the fire"
    if near(TREE, 1):
        return "under the apple trees"
    if near(BELL, 2):
        return "near the bell"
    if near(THORN, 1):
        return "near the thorns"
    if y <= 7:
        return "in the north of the valley"
    if y >= 15:
        return "in the south of the valley"
    return "in the middle of the valley"


def place(mind: Mind) -> str:
    w = mind.world
    if w.grid[w.y, w.x] == NEST:
        return "I'm in my nest."
    return f"I'm {region(mind, w.x, w.y)}."


# --- what it knows ------------------------------------------------------------------------


def age_words(mind: Mind) -> str:
    days, years = int(mind.age // DAY), mind.years
    if years:
        return "one year old" if years == 1 else f"{years} years old"
    return "less than a day old" if days < 1 else "one day old" if days == 1 else f"{days} days old"


def listing(items: list[str], last: str = "and") -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" {last} " + items[-1]


def learned_facts(mind: Mind) -> dict[str, list[str]]:
    """What it has found out about its own kinds of things, by what they're like."""
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


def good_to_eat(mind: Mind) -> str:
    foods = [
        THINGS[n].label.lower()
        for n in ("bush", "apple", "mushroom")
        if mind.things.get(n, {}).get("ate") and 2 * mind.things[n].get("sick", 0) < mind.things[n]["ate"]
    ]
    if foods:
        return f"{listing(foods).capitalize()} are good to eat."
    kinds = learned_facts(mind)["eat"]
    return f"{listing(kinds[:2]).capitalize()} are good to eat." if kinds else ""


def bad_things(mind: Mind) -> str:
    hurt = [THINGS[n].one for n in ("thorns", "fire") if mind.things.get(n, {}).get("hurt")]
    sick = [THINGS[n].label.lower() for n in ("toadstool", "mushroom") if mind.things.get(n, {}).get("sick")]
    said = []
    if hurt:
        said.append(f"{listing(hurt).capitalize()} hurt me.")
    if sick:
        said.append(f"{listing(sick).capitalize()} made me sick.")
    if not said:
        kinds = learned_facts(mind)["hurt"]
        if kinds:
            said.append(f"{listing(kinds[:2]).capitalize()} hurt.")
    return " ".join(said)


def knowledge(mind: Mind) -> str:
    return " ".join(p for p in (good_to_eat(mind), bad_things(mind)) if p)


YEARS_GONE = re.compile(r"\b(two|three|four|five|six|seven|eight|nine|ten) years had gone by")
NUMBERS = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine ten".split())}


def first_person(milestone: str, name: str) -> str:
    text = milestone.replace(f"{name} came into the world", "came into the world").replace("it knows", "I know")
    text = text.replace(" its ", " my ").replace(": of ", ": ").replace("itself", "myself")
    text = text.replace("made it sick", "made me sick")
    text = YEARS_GONE.sub(lambda m: f"{NUMBERS[m.group(1)]} years had gone by", text)  # (as it tells years now)
    return f"I {text}."


def told_firsts(stones: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Its firsts as it tells them: of the springs that came back, only the first (its first whole year). The later
    ones would tell the same thing over and over, and how old it is it says anyway."""
    years = [m for m in stones if m[1].startswith("saw spring come back")]
    return [m for m in stones if m not in years[1:]]


def addressed(text: str, name: str) -> str:
    """What someone said, with its own name as the name it learned people call it by ("Hi Pip!": "Hi Haven!")."""
    return text if name == "Haven" else re.sub(rf"\b{re.escape(name)}\b", "Haven", text, flags=re.IGNORECASE)


def memory(mind: Mind) -> str:
    """The latest of its firsts that comes to mind (one that matters, if it has one: not just a new kind of thing)."""
    stones = mind.me.milestones
    notable = [m for m in told_firsts(stones[1:]) if not m[1].startswith(ROUTINE)]
    return first_person((notable or stones)[-1][1], mind.me.name) if stones else ""


ROUTINE = ("noticed a new kind of thing", "heard a word for the first time", "realized two kinds")


def story(mind: Mind) -> str:
    """Its life in a few sentences: how it began, and the latest of its firsts that matter most."""
    stones = mind.me.milestones
    if not stones:
        return ""
    rest = told_firsts(stones[1:])
    notable = [m for m in rest if not m[1].startswith(ROUTINE)]
    others = [m for m in rest if m not in notable]
    chosen = notable[-3:] if len(notable) >= 3 else sorted(notable + others[len(others) - (3 - len(notable)) :])
    return " ".join(first_person(text, mind.me.name) for _, text in [stones[0], *chosen])


def known_words(words: list[str]) -> str:
    if not words:
        return "I don't know any words yet."
    return f"I know the word {words[0]}." if len(words) == 1 else f"I know the words {listing(words[:6])}."


DONE = {  # what it did, as it says it: (once, more than once)
    "ate a berry": ("ate a berry", "ate berries"),
    "ate an apple": ("ate an apple", "ate apples"),
    "ate a mushroom": ("ate a mushroom", "ate mushrooms"),
    "ate a toadstool": ("ate a toadstool", "ate toadstools"),
    "drank from the pond": ("drank from the pond", "drank from the pond"),
    "rang the bell": ("rang the bell", "rang the bell"),
    "pushed the ball and watched it roll": ("pushed the ball", "pushed the ball"),
    "smelled a flower": ("smelled a flower", "smelled flowers"),
    "shook a tree, and an apple fell": ("shook an apple tree", "shook the apple trees"),
    "warmed itself at the fire": ("warmed myself at the fire", "warmed myself at the fire"),
    "felt sick after eating a toadstool": ("felt sick", "felt sick"),
    "climbed to the top of the hill": ("climbed to the top of the hill", "climbed to the top of the hill"),
}


def today(mind: Mind) -> str:
    done = [(DONE[text][n > 1], n) for text, n in mind.today.items() if text in DONE]
    if not done:
        return "I haven't done much yet today."
    chosen = [text for text, _ in sorted(done, key=lambda item: -item[1])[:3]]
    return f"Today I {listing(list(dict.fromkeys(chosen)))}."


def likes(mind: Mind) -> str:
    joys = sorted(((s.get("joy", 0.0), n) for n, s in mind.things.items() if n in PLAYING), reverse=True)
    liked = [PLAYING[n] for joy, n in joys[:2] if joy >= 1.0]
    disliked = [THINGS[n].one for n in ("thorns", "fire") if mind.things.get(n, {}).get("hurt")]
    disliked += [THINGS[n].label.lower() for n in ("toadstool",) if mind.things.get(n, {}).get("sick")]
    said = []
    if liked:
        said.append(f"I like {listing(liked)}.")
    if disliked:
        said.append(f"I don't like {listing(disliked, 'or')}.")
    return " ".join(said)


ABILITIES = (
    ("climbed", "climb the hill"),
    ("drank", "drink from the pond"),
    ("rang", "ring the bell"),
    ("pushed", "push the ball"),
    ("shook", "shake apples down from the trees"),
    ("smelled", "smell flowers"),
    ("warmed", "warm myself by the fire"),
)


def abilities(mind: Mind) -> str:
    learned = [what for event, what in ABILITIES if mind.counts.get(event)]
    said = "I can walk around, eat, touch things and rest, and I can talk a little."
    return said + (f" I've found out I can {listing(learned[:4])}." if learned else "")


def world_words(mind: Mind) -> str:
    seen = sorted(
        ((s.get("seen", 0), n) for n, s in mind.things.items() if n in THINGS and n not in ("wall", "nest")),
        reverse=True,
    )
    found = [THINGS[n].label.lower() for count, n in seen if count][:6]
    if not found:
        return "I live in a valley with a wall all around it. I'm still exploring it."
    return f"I live in a valley with a wall all around it. I've found {listing(found)}."


# --- who it is, and its years ------------------------------------------------------------------

TRAIT_WORDS = {  # what it is when it's more so than most Havens, and when it's less
    "curious": ("curious", "a homebody"),
    "playful": ("playful", "serious"),
    "brave": ("brave", "careful"),
    "friendly": ("friendly", "shy"),
    "cheerful": ("cheerful", "gloomy"),
    "calm": ("calm", "restless"),
}
BECAME = {  # how it says it has grown more so, or less
    "curious": ("more curious", "less curious"),
    "playful": ("more playful", "more serious"),
    "brave": ("braver", "more careful"),
    "friendly": ("friendlier", "shyer"),
    "cheerful": ("more cheerful", "gloomier"),
    "calm": ("calmer", "more restless"),
}
ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth")
SIGNS = {  # what the valley is like in each season: what it sees and feels
    "spring": "The flowers are out, and the butterflies.",
    "summer": "The days are long and warm.",
    "autumn": "The leaves are orange, and there are lots of apples.",
    "winter": "The trees are bare, and the flowers are gone.",
}
GETTING = {"spring": "warmer", "summer": "cooler", "autumn": "colder", "winter": "warmer"}  # as a season turns


def trait_phrase(trait: str, d: float) -> str:
    """How it says one of its traits: "very brave", "a little shy", "a homebody"."""
    high, low = TRAIT_WORDS[trait]
    word = high if d > 0 else low
    if word.startswith("a "):
        return word if abs(d) < 0.25 else "a real " + word[2:]
    return ("very " if abs(d) >= 0.25 else "a little " if abs(d) < 0.13 else "") + word


def character_words(traits: dict[str, float], days: int, changed: list[tuple[str, float]] = ()) -> str:
    """What it's like, from its traits (its three most marked ones), and how it has changed."""
    if days < 2:
        return "I'm still finding out what I'm like."
    marked = sorted(((t, traits[t] - 0.5) for t in TRAITS), key=lambda item: -abs(item[1]))
    phrases = [trait_phrase(t, d) for t, d in marked[:3] if abs(d) >= 0.08]
    said = f"I'm {listing(phrases)}." if phrases else "I'm not very anything yet. A bit of everything."
    if changed:
        said += f" I've become {listing([BECAME[t][d < 0] for t, d in changed])} since I was little."
    return said


def character(mind: Mind) -> str:
    c = mind.character
    return character_words(c.traits, c.days, c.changes())


def trait_answer(traits: dict[str, float], trait: str, asked_high: bool) -> str:
    """Whether it's brave, shy, playful...: yes or no, and how."""
    d = traits[trait] - 0.5
    if abs(d) < 0.08:
        return "Sometimes. Not more than most."
    return f"{'Yes' if (d > 0) == asked_high else 'No'}, I'm {trait_phrase(trait, d)}."


def season_of_day(day: int) -> str:
    return SEASONS[(day % YEAR) // SEASON_DAYS]


def nth_season(born_day: int, day: int) -> int:
    """Which of its lives' springs (or summers...) the season on a day is: its first, its second..."""
    season, count = season_of_day(day), 0
    for k in range(born_day, day + 1):
        if season_of_day(k) == season and (k == born_day or season_of_day(k - 1) != season):
            count += 1
    return max(count, 1)


def ordinal(n: int) -> str:
    return ORDINALS[n - 1] if n <= len(ORDINALS) else f"{n}th"


def season_words(mind: Mind) -> str:
    """The season, what the valley is like in it, which of its lives' seasons of that kind it is, and if it's turning."""
    w = mind.world
    n = nth_season(mind.me.born // DAY, w.day)
    said = f"It's {w.season}, my {ordinal(n)} {w.season}. {SIGNS[w.season]}"
    if w.season_day >= SEASON_DAYS - TURNING:
        said += f" It's getting {GETTING[w.season]}."
    return said


def weather(mind: Mind) -> str:
    """The season, and how warm it is, as it feels it."""
    b = mind.body
    level = float(b.drives()[1])
    feel = "It's nice." if level < 0.3 else "It's cold." if b.cold() else "It's hot."
    return f"It's {mind.world.season}. {feel}"


def when(mind: Mind, memory: dict) -> str:
    """When a day it remembers was: "this summer", or "in my second winter"."""
    w = mind.world
    if memory["season"] == w.season and w.day - memory["day"] < SEASON_DAYS:
        return f"this {w.season}"
    return f"in my {ordinal(nth_season(mind.me.born // DAY, memory['day']))} {memory['season']}"


WRONG = {"fainted": "I fainted", "got hurt": "I got hurt", "felt sick": "I felt sick", "was cold": "I was cold"}
WRONG["was hungry"] = "I was hungry"


def day_words(mind: Mind, memory: dict, best: bool) -> str:
    """One of its best or worst days: when it was, and what happened."""
    said = f"My {'best' if best else 'worst'} day was {when(mind, memory)}."
    if best:
        did = [DONE[text][n > 1] for text, n in memory["did"] if text in DONE]
        return said + (f" I {listing(did[:2])}." if did else " I felt good all day.")
    wrong = [WRONG[w] for w in memory["wrong"] if w in WRONG]
    return said + (f" {listing(wrong[:2])}." if wrong else " I felt bad all day. I don't know why.")


def best_day(mind: Mind) -> str:
    best = mind.character.best
    return day_words(mind, best[0], True) if best else ""


def worst_day(mind: Mind) -> str:
    worst = mind.character.worst
    return day_words(mind, worst[0], False) if worst else ""


PLACE_WORDS = {  # a part of the valley, as it says it's its favorite place
    "the fire": "by the fire",
    "the bell": "by the bell",
    "the thorns": "near the thorns",
    "the apple trees": "under the apple trees",
}


def favorites(mind: Mind) -> str:
    """Its favorite place and season (and the season that's hard for it), from how it has felt in them."""
    c = mind.character
    said = []
    place = c.favorite_area()
    if place:
        said.append(f"My favorite place is {PLACE_WORDS.get(place, place)}.")
    feel = c.season_feel()
    if len(feel) >= 2:
        best, hard = max(feel, key=feel.get), min(feel, key=feel.get)
        said.append(f"My favorite season is {best}.")
        if feel[hard] < 0.6 * feel[best]:
            said.append(f"{hard.capitalize()} is hard for me.")
    else:
        said.append(f"I've only known {listing([s for s in SEASONS if s in feel] or [mind.world.season])} so far.")
    return " ".join(said)


def fears(mind: Mind) -> str:
    """What it's afraid of: what has hurt it or made it sick more than once (a brave Haven only keeps away)."""
    feared = [n for n in ("fire", "thorns") if mind.things.get(n, {}).get("hurt", 0) >= 2]
    feared += ["toadstool"] * (mind.things.get("toadstool", {}).get("sick", 0) >= 2)
    if not feared:
        return "Nothing scares me much."
    them = listing([THINGS[n].one if n != "toadstool" else "toadstools" for n in feared])
    if mind.character.traits["brave"] > 0.6:
        return f"I'm not scared of anything, but I keep away from {them}."
    return f"I'm scared of {them}."


def memo(mind: Mind) -> dict:
    """Everything it knows that could come to mind when someone talks with it."""
    things = {}
    for name in THINGS:
        record = mind.things.get(name, {})
        where = None
        if name == "ball" and "x" in record:
            where = f"It rolls around. I last saw it {region(mind, int(record['x']), int(record['y']))}."
        things[name] = {"stats": {k: v for k, v in record.items() if k not in ("x", "y", "last")}, "where": where}
    return {
        "me": f"I'm {mind.me.name}, {age_words(mind)}.",
        "place": place(mind),
        "season": season_words(mind),
        "character": character(mind),
        "today": today(mind),
        "likes": likes(mind),
        "knowledge": knowledge(mind),
        "words": known_words(mind.lexicon.vocabulary()),
        "memory": memory(mind),
        "self": found_out(mind),
        "story": story(mind),
        "can": abilities(mind),
        "world": world_words(mind),
        "favorites": favorites(mind),
        "fears": fears(mind),
        "best day": best_day(mind),
        "worst day": worst_day(mind),
        "traits": {t: round(v, 3) for t, v in mind.character.traits.items()},
        "things": things,
    }


# --- people ------------------------------------------------------------------------------------

NOT_NAMES = frozenset(
    "a an the not so very just really back here home fine good great ok okay sorry tired sad happy bored hungry "
    "cold hot sick ready sure busy done leaving going new old sleepy glad well alright fantastic excited afraid "
    "scared lost confused curious haven monday tuesday wednesday thursday friday saturday sunday january february "
    "march april may june july august september october november december human person someone nobody everyone "
    "fun cool nice weird late early still also".split()
)
_ADDRESS = re.compile(r"^(?:(?:hey|hi|hello|ok|okay|so|well)\b[,!.]?\s*)?(?:haven\b[,!.]?\s*)?", re.I)


def introduced(text: str) -> str | None:
    """The person's name, if they just said it ("My name is Sam", "I'm Sam", "call me Sam")."""
    t = _ADDRESS.sub("", " ".join(text.strip().split()), count=1)
    found = re.search(r"\b(?:my name is|my name's|call me|name's)\s+([A-Za-z][A-Za-z'-]{1,20})", t, re.I)
    if not found:
        found = re.search(r"^(?:[Ii]'m|[Ii] am|[Ii]t's|[Tt]his is)\s+([A-Z][a-z'-]{1,20})(?=[\s,.!]|$)", t)
    if not found and len(t.split()) <= 3:
        found = re.search(r"^(?:i'm|im|i am)\s+([a-z][a-z'-]{1,20})[.!]?$", t, re.I)
    if not found:
        return None
    name = found.group(1)
    if name.lower() in NOT_NAMES:
        return None
    return name[0].upper() + name[1:]


SWAP = {
    "i": "you",
    "me": "you",
    "my": "your",
    "mine": "yours",
    "myself": "yourself",
    "i'm": "you're",
    "im": "you're",
    "i've": "you've",
    "i'd": "you'd",
    "i'll": "you'll",
    "we": "you",
    "us": "you",
    "our": "your",
    "ours": "yours",
    "we're": "you're",
    "we've": "you've",
}
FACT_START = re.compile(r"^(i|i'm|im|i am|i've|i was|i have|my|we|we're|our)\b", re.I)
NOT_FACT = re.compile(
    r"^(i (love|like|hate|miss) you|(i'm|im|i am) (sorry|back|here|home|fine|ok|okay|good|great|tired|so tired|"
    r"sleepy|bored|so bored|sad|happy|sick|not feeling|done|leaving|going|glad)|i (have|need|gotta|got) to go|"
    r"i feel|i had a|my bad|i see|i think|i wonder|i don't|i didn't|i can't|i guess)\b",
    re.I,
)


def second_person(text: str) -> str:
    """Something a person said about themselves, as Haven would say it back to them: "I like cats" → "you like cats"."""
    words = " ".join(text.strip().split()).rstrip(".!").split(" ")
    out = []
    for i, word in enumerate(words):
        core = word.lower().strip(",;:")
        tail = word[len(word.rstrip(",;:")) :]
        before = words[i - 1].lower() if i else ""
        if core in SWAP:
            out.append(SWAP[core] + tail)
        elif core == "am" and before == "i":
            out.append("are" + tail)
        elif core == "was" and before == "i":
            out.append("were" + tail)
        else:
            out.append(word)
    return " ".join(out)


CORRECTION = re.compile(
    r"^(?:no+|nope|actually|wrong|that's wrong|that's not right|not quite|i mean|oops|sorry)\b[,.!]?\s*", re.I
)


def statement(text: str) -> str | None:
    """Something about themselves the person just told it, as it would say it back ("you like cats"), or None.

    Corrections count too: "No, my dog is called Biscuit."
    """
    t = _ADDRESS.sub("", " ".join(text.strip().split()), count=1)
    t = CORRECTION.sub("", t, count=1)
    if not t or "?" in t or introduced(t) or len(t.split()) < 3:
        return None
    if not FACT_START.match(t) or NOT_FACT.match(t):
        return None
    return second_person(t)


STOP = frozenset(
    "what when where which who whom whose why how does did do is are was were be am the a an and or of to in on at "
    "for with you your yours i me my mine it its that this there their they them tell know remember told about "
    "what's whats can could would should please haven any anything something everything".split()
)
GENERIC = frozenset("favorite name like love have really much best".split())
SYNONYMS = {
    "named": "name",
    "called": "name",
    "call": "name",
    "loves": "love",
    "likes": "like",
    "maths": "math",
    "wrote": "write",
    "written": "write",
}
ALIASES = {"math": "mathematic", "phone": "telephone"}  # a word, and what the article about it is called
FACT_RULES = (
    (r"\bhow old\b|\bmy age\b", r"\byears old\b|\byou(?:'re| are) \d+\b"),
    (r"\bjob\b|\bwork\b|\bwhat do i do\b|\bliving\b", r"^you(?:'re| are) an? |^you work\b"),
    (r"\bwhere do i live\b|\bwhere am i from\b|\bwhere i live\b|\bwhere i'm from\b", r"\blive\b|\bfrom\b"),
    (
        r"\bpets?\b|\banimals?\b",
        r"\b(?:dogs?|cats?|fish|hamsters?|rabbits?|birds?|parrots?|horses?|turtles?|snakes?|puppy|kitten|lizards?)\b",
    ),
    (r"\bbirthday\b", r"\bbirthday\b"),
    (r"\bwhat do i (?:like|love|enjoy)\b|\bwhat i (?:like|love)\b", r"^you (?:really )?(?:like|love|enjoy)\b"),
)


def _tokens(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9']+", text.lower()):
        w = w.removesuffix("'s")
        if w in STOP:
            continue
        w = SYNONYMS.get(w, w)
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(ALIASES.get(w, w))
    return out


ABOUT_ALL = re.compile(
    r"\bwhat do you know about me\b|\bwhat have i told you\b|\bwhat do you remember about me\b|\babout me\b", re.I
)


def fact_key(fact: str) -> str | None:
    """What a fact is about ("your dog", "where you live"), so a newer one on the same thing replaces the older."""
    f = fact.lower()
    for pattern, key in (
        (r"\byour favorite (\w+)", "favorite {}"),
        (r"\byour (\w+)(?:'s name| is called| is named)", "{} name"),
        (r"\byou have an? (\w+) (?:named|called)", "{} name"),
        (r"\byou (?:live in|are from|'re from|re from)\b", "home"),
        (r"\byears old\b", "age"),
        (r"\bbirthday\b", "birthday"),
        (r"^you(?:'re| are) an? \w+$|^you work as\b", "job"),
    ):
        found = re.search(pattern, f)
        if found:
            return key.format(*found.groups())
    return None


def about_you(person: str | None, facts: list[str]) -> str:
    """Everything it knows about the person it's talking with, in a sentence or two."""
    said = [f"Your name is {person}."] if person else []
    if facts:
        parts = [f"that {f}" for f in facts]
        joined = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + ", and " + parts[-1]
        said.append(f"You told me {joined}.")
    return " ".join(said) or "You haven't told me anything yet."


def about_them(text: str) -> bool:
    """Whether someone is asking about themselves ("what's my name?", not "tell me about the moon")."""
    low = re.sub(r"\b(?:tell|show|give|help|let|teach) me\b", "", text.lower())
    return bool(re.search(r"\b(my|me|i|mine)\b", low))


def best_fact(text: str, told: list[str]) -> str | None:
    """What they told it that a question of theirs is about (the latest, if they ask about what they told it)."""
    low = text.lower()
    if not told or not (about_them(low) or re.search(r"\btold\b", low)):
        return None
    for question, fact in FACT_RULES:
        if re.search(question, low):
            for said in reversed(told):
                if re.search(fact, said.lower()):
                    return said
    asked = set(_tokens(low))
    best, score = None, 0.0
    for said in reversed(told):
        s = sum(0.5 if w in GENERIC else 1.0 for w in asked & set(_tokens(said)))
        if s > score:
            best, score = said, s
    if score >= 1.0:
        return best
    if re.search(r"\btold you\b|\btell you\b|\bremember what i\b|\babout me\b", low):
        return told[-1]
    return None


def first_sentence(text: str, limit: int = 220) -> str:
    """The first sentence of something it read, without asides in brackets."""
    text = re.sub(r"\s*\([^()]*\)", "", " ".join(text.split()))
    first = re.split(r"(?<=[a-z0-9)])\. ", text, maxsplit=1)[0].rstrip(".") + "."
    return first if len(first) <= limit else first[:limit].rsplit(" ", 1)[0] + "…"


def best_reading(text: str, read: list[tuple[str, str]]) -> tuple[str, str] | None:
    """Something it has read that a question is about: (title, first sentence)."""
    asked = set(_tokens(text))
    best, score = None, 0.0
    for title, sentence in reversed(read):
        words = set(_tokens(title))
        s = len(words & asked) / len(words) if words else 0.0
        if s > score:
            best, score = (title, sentence), s
    return best if score >= 0.5 else None


# --- being asked to do things -------------------------------------------------------------------


@dataclass(frozen=True)
class Request:
    forms: tuple[str, ...]  # how people ask
    do: str  # what it would do, as it says it
    thing: str  # what it's about (a key of THINGS)
    action: str  # "use", "eat", "go" or "sleep"
    need: int | None = None  # the need it meets, if any: it'll do it even when that need is pressing


GO = {  # things it can be asked to go to: (how the person says it, how it says it)
    "bell": ("the bell", "the bell"),
    "ball": ("the ball", "the ball"),
    "pond": ("the pond", "the pond"),
    "fire": ("the fire", "the fire"),
    "nest": ("your nest", "my nest"),
    "hill": ("the hill", "the hill"),
    "tree": ("the apple trees", "the apple trees"),
    "flower": ("the flowers", "the flowers"),
    "bush": ("the berry bushes", "the berry bushes"),
    "stone": ("the stones", "the stones"),
    "mushroom": ("the mushrooms", "the mushrooms"),
}
REQUESTS = (
    Request(
        ("Ring the bell.", "Can you ring the bell?", "Go ring the bell!", "Please ring the bell."),
        "ring the bell",
        "bell",
        "use",
    ),
    Request(
        ("Push the ball.", "Kick the ball!", "Can you push the ball?", "Go play with the ball."),
        "push the ball",
        "ball",
        "use",
    ),
    Request(("Eat some berries.", "Go eat some berries.", "Have some berries."), "eat some berries", "bush", "eat", 0),
    Request(("Eat an apple.", "Go find an apple to eat.", "Have an apple."), "eat an apple", "apple", "eat", 0),
    Request(
        ("Drink some water.", "Go have a drink.", "Drink from the pond.", "Can you get a drink?"),
        "drink from the pond",
        "pond",
        "use",
        1,
    ),
    Request(("Smell a flower.", "Go smell the flowers.", "Can you smell a flower?"), "smell a flower", "flower", "use"),
    Request(
        ("Shake a tree.", "Shake an apple tree!", "Go get an apple from a tree."), "shake an apple tree", "tree", "use"
    ),
    Request(
        ("Sit by the fire.", "Go warm up by the fire.", "Warm yourself by the fire."),
        "sit by the fire",
        "fire",
        "use",
        1,
    ),
    Request(("Go to sleep.", "Go to bed.", "Time for bed.", "Get some sleep."), "go to sleep", "nest", "sleep", 3),
    *(
        Request(
            (f"Go to {said}.", f"Can you go to {said}?", f"Walk over to {said}.", f"Please go to {said}."),
            f"go to {mine}",
            name,
            "go",
        )
        for name, (said, mine) in GO.items()
    ),
)
_POLITE = re.compile(
    r"^(?:please |can you |could you |will you |would you |why don't you |let's |go and |try to |i want you to )+", re.I
)
_VERBS = re.compile(
    r"^(?:ring|push|kick|roll|play with|eat|have some|have an|drink|have a drink|get a drink|smell|shake|get an apple|"
    r"sit|warm|go|walk|run|head|time for bed|get some sleep|take a nap)\b"
)


def request(text: str) -> Request | None:
    """What someone is asking it to do, if they are asking it to do something in its valley."""
    t = _ADDRESS.sub("", " ".join(text.lower().strip().split()), count=1)
    if re.match(r"^(?:can|could) you (?:eat|have)\b", t):
        return None  # asking whether it can eat something, not asking it to
    t = _POLITE.sub("", re.sub(r"[?!.,]+", " ", t).strip() + " ").strip()
    if not _VERBS.match(t):
        return None
    by = {r.do: r for r in REQUESTS}
    if re.search(r"^(go to sleep|go to bed|time for bed|get some sleep|go sleep|take a nap)\b", t):
        return by["go to sleep"]
    for pattern, do in (
        (r"\bring\b.*\bbell\b", "ring the bell"),
        (r"\b(push|kick|roll|play with)\b.*\bball\b", "push the ball"),
        (r"\b(drink|have a drink|get a drink)\b", "drink from the pond"),
        (r"\bsmell\b.*\bflowers?\b", "smell a flower"),
        (r"\bshake\b.*\btrees?\b|\bapple from a tree\b", "shake an apple tree"),
        (r"\b(sit|warm)\b.*\bfire\b", "sit by the fire"),
        (r"\b(eat|have)\b.*\bberr(y|ies)\b", "eat some berries"),
        (r"\b(eat|have)\b.*\bapples?\b|\bapple to eat\b", "eat an apple"),
    ):
        if re.search(pattern, t):
            return by[do]
    found = re.match(r"^(?:go|walk|run|head)(?: over| back| down| up)? to\b(.*)", t)
    names = mentioned(found.group(1)) if found else []
    if names and names[0] in GO:
        return by[f"go to {GO[names[0]][1]}"]
    return None


def request_note(req: Request) -> str:
    return f"You asked me to {req.do}."


def request_answer(req: Request, body: dict, stats: dict) -> str:
    """What it says when asked to do something: whether it will, from how it is and what it knows."""
    if body["asleep"]:
        return "I'm asleep."
    drives = body["drives"]
    need = int(np.argmax(drives))
    if drives[need] >= 0.7 and req.need != need:
        return f"Not now. I'm {need_words(need, float(drives[need]), body['cold'])}."
    if req.action != "sleep" and req.thing != "nest" and not knows_of(stats):
        return f"I haven't seen {THINGS[req.thing].one} yet, so I don't know where to go."
    return f"Okay, I'll {req.do}."


# --- being taught words ----------------------------------------------------------------------------

NAMING_FORMS = ("That's {}.", "This is {}.", "It's called {}.", "Look, {}!", "That thing is {}.", "We call that {}.")
NAMING_WORDS = (
    "bell ball apple flower tree stone rock pond fire butterfly mushroom toadstool nest berry thorn hill leaf bird "
    "cloud sky grass moon star sun frog bug feather acorn puddle branch seed snail worm pebble twig petal bee ant"
).split()


def naming_answer(word: str) -> str:
    return f"{a(word).capitalize()}. I'll try to remember that word."


# --- what comes to mind -------------------------------------------------------------------------

TOPICS = {
    "story": r"\byour (?:life )?story\b|\bstory of your\b|\byour life\b|\bhappened in your\b|\bborn\b",
    "can": r"\bwhat can you\b|\bable to\b|\bcan you do\b",
    "world": r"\byour world\b|\byour valley\b",
}


ASKED_FOR = {  # a memory a question asks for, and words that ask for it: it comes to mind last, where it's clearest
    "season": r"\bseasons?\b|\bweather\b|\bwinter\b|\bsummer\b|\bspring\b|\bautumn\b|\bnice out\b|\bright now\b",
    "character": r"\bpersonality\b|\bwhat are you like\b|\bchanged?\b|\bdifferent\b|\bsort of creature\b|\bnature\b|"
    r"\bsame as\b|\bgr[eo]w up\b|\bgrown\b|\bas a creature\b|\bkind of creature\b|"
    r"\bdescribe yourself\b|\babout yourself\b|\bwhat are you\b|\bare you (?:very |a little )?(?:"
    + "|".join(w.removeprefix("a ") for pair in TRAIT_WORDS.values() for w in pair)
    + r")\b",
    "today": r"\btoday\b|\byour day\b",
    "likes": r"\blike\b|\benjoy\b|\bfavou?rite\b|\bfun\b",
    "knowledge": r"\blearned\b|\bfound out\b|\beat\b|\bfood\b|\bhurts?\b|\bdangerous\b|\bsick\b|\baway from\b",
    "words": r"\bwords?\b",
    "memory": r"\bremember\b|\bhappened to you\b",
    "self": r"\balive\b|\breal\b|\bliving thing\b|\bwhat are you\b|\bwhat kind of thing\b|\babout yourself\b|"
    r"\bdescribe yourself\b|\bwhat exactly are you\b|\ban animal\b",
    "favorites": r"\bfavou?rite (?:place|spot|season)\b|\bwhere\b[^?]*\b(?:like|love)\b|\bseasons?\b[^?]*\b(?:like|love|best|"
    r"favou?rite)\b|\b(?:like|love)\b[^?]*\bseasons?\b|\bbest season\b|\bpart of the valley\b",
    "fears": r"\bafraid\b|\bscared\b|\bscar(?:es?|y)\b|\bfrighten|\bfears?\b",
    "best day": r"\bbest day\b|\bhappiest\b|\bfavou?rite memory\b|\bgood day\b|\bday\b[^?]*\bbest\b",
    "worst day": r"\bworst day\b|\bsaddest\b|\bbad day\b|\bday\b[^?]*\bworst\b",
}


def recall(
    known: dict,
    said: str | list[str],
    person: str | None = None,
    told: list[str] | tuple = (),
    read: list[tuple[str, str]] | tuple = (),
    just: str | None = None,
    extra: list[str] | tuple = (),
) -> str:
    """What comes to mind before it answers: what it always knows, and what the words bring up.

    `extra` holds what else came to mind that the caller found: what it read that answers the
    question, being asked to do something.
    """
    texts = [said] if isinstance(said, str) else list(said)
    everything = " ".join(texts).lower()
    asked = [piece for piece, pattern in ASKED_FOR.items() if re.search(pattern, everything)]
    parts = [known["me"]]
    if person:
        parts.append(f"I'm talking with {person}.")
    parts += [
        known.get(piece, "")
        for piece in ("place", "season", "character", "today", "likes", "knowledge", "words", "memory")
        if piece not in asked
    ]
    for text in texts:
        for name in mentioned(text)[:2]:
            thing = known["things"][name]
            parts.append(thing_note(name, thing["stats"], thing["where"], with_where="where" in text.lower()))
    for text in texts:
        found = trait_asked(text)
        if found and known.get("traits"):
            parts.append(trait_note(known["traits"], found[0]))
    for text in texts:
        for fact in facts_for(text, list(told)):
            parts.append(f"You told me that {fact}.")
    if just:
        parts.append(f"You just told me that {just}.")
    for text in texts:
        found = None if about_them(text) else best_reading(text, list(read))
        if found:
            parts.append(f"I read about {found[0]}: {found[1]}")
    parts += extra
    parts += [known[topic] for topic, pattern in TOPICS.items() if re.search(pattern, everything)]
    parts += [known.get(piece, "") for piece in asked]  # (lives noted down before it knew some pieces lack them)
    return " ".join(dict.fromkeys(p for p in parts if p))


def facts_for(text: str, told: list[str]) -> list[str]:
    """What they told it that comes to mind at their words: all of it (the latest few) if they ask what it knows about
    them, or the one thing they ask about."""
    if told and ABOUT_ALL.search(text):
        return told[-3:]
    fact = best_fact(text, told)
    return [fact] if fact else []


def notes(
    mind: Mind,
    text: str = "",
    read: list[tuple[str, str]] | tuple = (),
    just: str | None = None,
    extra: list[str] | tuple = (),
) -> str:
    """What comes to mind when someone says something to it, in the middle of its life."""
    told = [fact for _, fact in mind.told if fact != just]  # what they just said isn't a memory yet
    return recall(memo(mind), text, mind.person, told, read, just, extra)


# --- answers from its state -------------------------------------------------------------------------


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


def seen_words(mind: Mind, content) -> str:
    """What it's looking at: what it takes it to be, if it has a name for that kind of thing, or how it looks."""
    seen = content.label.split(" (")[0]
    name = mind.kind_name(content.kind) if content.kind >= 0 else None
    if name in THINGS:
        size, color = content.extra.get("size", ""), content.extra.get("color", "")
        prefix = f"something {size + ' ' if size else ''}{color}"
        if seen.startswith(prefix):
            return THINGS[name].one + seen[len(prefix) :]
    return seen


def seeing(mind: Mind) -> str:
    content = mind.workspace.content
    if mind.body.asleep:
        return "Nothing. I'm asleep."
    if content is not None and content.source == "vision":
        seen = seen_words(mind, content)
        if content.kind >= 0:
            known = mind.knowledge.describe(content.kind)
            if "good to eat" in known:
                return f"I see {seen}. I think it's good to eat."
            if "it hurts" in known:
                return f"I see {seen}. I think it hurts."
        return f"I see {seen}."
    return "Nothing special right now."


GOAL_WORDS = {
    "food": "I'm looking for food.",
    "healing": "I'm resting, so I can heal.",
    "sleep": "I'm going to my nest to sleep.",
    "explore": "I'm looking around.",
    "play": "I want to play.",
}


def doing(mind: Mind) -> str:
    if mind.body.asleep:
        return "I'm sleeping."
    if mind.goals.current == "warmth":
        return "I'm trying to get warm." if mind.body.cold() else "I'm trying to cool down."
    return GOAL_WORDS[mind.goals.current]


def thinking(mind: Mind) -> str:
    content = mind.workspace.content
    if content is None:
        return "Nothing much."
    if content.source == "vision":
        return f"I'm looking at {seen_words(mind, content)}."
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


def playing(mind: Mind) -> str:
    b = mind.body
    if b.asleep:
        return "I'm asleep."
    drives = b.drives()
    need = int(np.argmax(drives))
    if drives[need] >= 0.5:
        return f"Not now. I'm {need_words(need, float(drives[need]), b.cold())}."
    joys = sorted(((s.get("joy", 0.0), n) for n, s in mind.things.items() if n in PLAYING), reverse=True)
    if joys and joys[0][0] >= 1.0:
        return f"Yes! I like {PLAYING[joys[0][1]]}."
    return "Yes! I'm still finding out what's fun."


SELF_FOUND = (  # what it can find out about itself (selfmodel.py), and how it says it
    ("needs", 0.3, "I need things"),
    ("agency", 0.5, "I can make things happen"),
    ("memory", 0.5, "I remember what happens to me"),
    ("feeling", 0.5, "some things feel good to me and some feel bad"),
    ("change", 0.5, "I've learned things"),
)


def found_out(mind: Mind) -> str:
    """What it has found out about itself, and what it makes of that (from the evidence it keeps about itself)."""
    found = [text for key, limit, text in SELF_FOUND if mind.me.evidence[key] > limit]
    verdict = mind.me.conclusions()[-1]
    return (f"I've found out that {and_list(found)}. " if found else "") + verdict


FOND = {  # whether it likes being with them, as a friendly, a shy, or any other Haven says it
    "friendly": "Yes! I like it when you talk with me. I miss you when you're gone.",
    "shy": "I think so. I'm a little shy, but I like it when you talk with me.",
    None: "I like it when you talk with me.",
}
BYE = {"friendly": "Bye! Come back soon. I'll miss you.", "shy": "Bye.", None: "Bye! Come back soon."}


def hello(asleep: bool, feel: str, character: str) -> str:
    if asleep:
        return "Hi! I'm asleep."
    return f"Hi! {feel}" + (" How are you?" if mood_of(character) == "friendly" else "")


def changed(mind: Mind) -> str:
    c = mind.character
    moved = c.changes()
    if moved:
        return f"Yes. I've become {listing([BECAME[t][d < 0] for t, d in moved])} since I was little."
    return "Not much. I'm still who I was." if c.years else "I'm still young. I'm still finding out who I am."


def favorite_place(mind: Mind) -> str:
    place = mind.character.favorite_area()
    return f"My favorite place is {PLACE_WORDS.get(place, place)}." if place else "I don't have a favorite place yet."


def favorite_season(mind: Mind) -> str:
    return season_answer(favorites(mind))


def season_answer(favorite: str) -> str:
    """Its favorite season, from what comes to mind about its favorites."""
    found = re.search(r"My favorite season is \w+\.|I've only known .+? so far\.", favorite)
    return found.group(0) if found else "I don't know yet."


def answers(mind: Mind) -> dict[str, str]:
    """What its state and what it knows say the answer to each kind of question is, right now."""
    b = mind.body
    hurt = _drive(mind, 2)
    temperature = _drive(mind, 1)
    content = mind.workspace.content
    known = knowledge(mind)
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
        "home": "I live in a valley. My nest is in the north-west corner.",
        "world": world_words(mind),
        "words": known_words(mind.lexicon.vocabulary()),
        "talk": "A little. I'm still learning words.",
        "learned": known or "Not much yet. I'm still finding out.",
        "eat": good_to_eat(mind) or "I don't know yet.",
        "danger": bad_things(mind) or "Nothing has hurt me yet.",
        "remember": memory(mind) or "Not much yet.",
        "story": story(mind),
        "day": today(mind),
        "likes": likes(mind) or "I'm still finding out what I like.",
        "play": playing(mind),
        "can": abilities(mind),
        "alive": found_out(mind),
        "feelings": "Yes. Some things feel good to me and some feel bad."
        if mind.me.evidence["feeling"] > 0.5
        else "I don't know yet.",
        "what": f"I'm {mind.me.name}, a small creature that lives in a valley. {character(mind)}",
        "personality": character(mind),
        "changed": changed(mind),
        "best day": best_day(mind) or "I haven't lived long enough to say.",
        "worst day": worst_day(mind) or "I haven't lived long enough to say.",
        "favorite place": favorite_place(mind),
        "season": season_words(mind),
        "weather": weather(mind),
        "favorite season": favorite_season(mind),
        "afraid": fears(mind),
        "maker": "I don't know who made me. I came into the world in my nest.",
        "conscious": "One thing at a time comes to the front of my mind: something I see, smell, hear or feel. "
        "I don't know if that's what you mean.",
        "ai": "I'm a small creature that lives in a valley inside a computer. I learned to talk from my own life.",
        "dream": "When I sleep, I go over what happened to me. Maybe that's a kind of dreaming.",
        "fond": FOND[mood_of(character(mind))],
        "color": "I don't have a favorite color. Colors are how I tell things apart.",
        "joke": "I don't know any jokes yet.",
        "being": f"{feeling(mind)} {doing(mind)}",
        "hello": hello(b.asleep, feeling(mind), character(mind)),
        "thanks": "You're welcome.",
        "bye": BYE[mood_of(character(mind))],
    }


# --- speaking up: what it says without being asked ------------------------------------------------
#
# What it has to say comes to mind as a note (which moments call for it is worked out in life.py); the
# words are its cortex's, from what comes to mind, as with everything else it says.

TRAIT_ASKED = re.compile(
    r"\bare you (?:very |a little |a real |a |so )?("
    + "|".join(w.removeprefix("a ") for pair in TRAIT_WORDS.values() for w in pair)
    + r")\b",
    re.I,
)


def trait_asked(text: str) -> tuple[str, bool] | None:
    """Which of its traits someone asks about ("are you shy?"), and whether they asked about more (brave) or less."""
    found = TRAIT_ASKED.search(text)
    if not found:
        return None
    word = found.group(1).lower()
    for trait, (high, low) in TRAIT_WORDS.items():
        if word in (high, low.removeprefix("a ")):
            return trait, word == high
    return None


def trait_note(traits: dict[str, float], trait: str) -> str:
    d = traits[trait] - 0.5
    return f"I'm {trait_phrase(trait, d)}." if abs(d) >= 0.08 else f"I'm about as {TRAIT_WORDS[trait][0]} as most."


def trait_reply(traits: dict[str, float], trait: str, asked_high: bool) -> str:
    """Whether it's brave, shy, playful...: worked out from what comes to mind about it."""
    d = traits[trait] - 0.5
    if abs(d) < 0.08:
        return f"Sometimes. {trait_note(traits, trait)}"
    return f"{'Yes' if (d > 0) == asked_high else 'No'}, {trait_note(traits, trait)}"


def mood_of(character: str) -> str | None:
    """Whether its character, as it says it, is friendly or shy (how it greets someone)."""
    return "friendly" if re.search(r"\bfriendly\b", character) else "shy" if re.search(r"\bshy\b", character) else None


def back_note(long: bool) -> str:
    return "You came back after a long time." if long else "You came back."


def back_answer(known: dict, person: str | None) -> str:
    mood = mood_of(known.get("character", ""))
    you = f", {person}" if person else ""
    said = {
        "friendly": f"You're back{you}! I missed you.",
        "shy": f"Oh, hi{you}. You're back.",
        None: f"Hi{you}! You're back.",
    }[mood]
    if known.get("today", "").startswith("Today I"):
        said += " " + known["today"]
    return said


SEASON_NEWS = {  # a new season: the note, and what it says
    "spring": ("Spring came to the valley.", "It's spring again! The flowers are back."),
    "summer": ("Summer came to the valley.", "It's summer now! The days are long and warm."),
    "autumn": ("Autumn came to the valley.", "It's autumn now. The leaves are turning orange."),
    "winter": ("Winter came to the valley.", "Winter is here. It's cold, and the flowers are gone."),
}
HAPPENED = {  # something that just happened (as its log has it), and what it says about it
    "climbed to the top of the hill": "I just climbed to the top of the hill!",
    "rang the bell": "I just rang the bell!",
    "got burned": "Ouch! I got burned.",
    "got hurt": "Ouch! That hurt.",
    "felt sick after eating a toadstool": "I feel sick. That toadstool was bad.",
    "shook a tree, and an apple fell": "I shook a tree, and an apple fell!",
    "fainted, and woke up later in its nest": "I fainted. I woke up in my nest.",
    "pushed the ball and watched it roll": "I pushed the ball, and it rolled!",
    "warmed itself at the fire": "I'm warming myself by the fire. It's nice.",
    "was given food": "Thank you for the food!",
}
DID = {  # what it was asked to do, done: as it tells it
    "ring the bell": "rang the bell",
    "push the ball": "pushed the ball",
    "eat some berries": "ate some berries",
    "eat an apple": "ate an apple",
    "drink from the pond": "drank from the pond",
    "smell a flower": "smelled a flower",
    "shake an apple tree": "shook an apple tree",
    "sit by the fire": "sat by the fire",
    **{f"go to {mine}": f"went to {mine}" for _, mine in GO.values()},
}


def event_note(text: str) -> str | None:
    """Something that just happened, from its log, as it comes to mind (None if it's nothing to tell)."""
    if text in HAPPENED:
        return f"Just now, I {text.replace(' its ', ' my ').replace('itself', 'myself')}."
    found = re.match(r"did what it was asked: (.+)$", text)
    if found and found.group(1) in DID:
        return f"I did what you asked: {found.group(1)}."
    found = re.match(r"stopped trying to (.+): it was (.+)$", text)
    if found:
        return f"I stopped trying to {found.group(1)}: I was {found.group(2)}."
    found = re.match(r"gave up trying to (.+)$", text)
    if found:
        return f"I gave up trying to {found.group(1)}."
    return None


def event_answer(note: str) -> str:
    found = re.match(r"Just now, I (.+)\.$", note)
    if found:
        text = found.group(1).replace(" my ", " its ").replace("myself", "itself")
        return HAPPENED[text]
    found = re.match(r"I did what you asked: (.+)\.$", note)
    if found:
        return f"I did it! I {DID[found.group(1)]}."
    found = re.match(r"I stopped trying to (.+): I was (.+)\.$", note)
    if found:
        return f"I stopped trying to {found.group(1)}. I'm {found.group(2)}."
    found = re.match(r"I gave up trying to (.+)\.$", note)
    return f"I'm sorry, I couldn't {found.group(1)}."


NEED_GOALS = ("I'm looking for food.", "I'm trying to get warm.", "I need to rest.", "I want to go to sleep.")


def need_note(need: int, level: float, cold: bool) -> str:
    """A need it feels strongly, and what it's doing about it."""
    doing = NEED_GOALS[need] if need != 1 or cold else "I'm trying to cool down."
    return f"I'm {need_words(need, level, cold)}, so {doing[0].lower()}{doing[1:]}"


def need_answer(note: str) -> str:
    first, rest = note.split(", so ", 1)
    return f"{first}! {rest[0].upper()}{rest[1:]}"


ASKS = {  # what it would like to know about the person, and how it asks
    "your name": "What's your name?",
    "your favorite food": "What's your favorite food?",
    "your favorite color": "What's your favorite color?",
    "your favorite animal": "What's your favorite animal?",
    "your favorite season": "What's your favorite season?",
    "where you live": "Where do you live?",
    "what you do": "What do you do?",
    "how old you are": "How old are you?",
    "what you like to do": "What do you like to do?",
    "what you did today": "What did you do today?",
}
ANSWER_FORMS = {  # a short answer to its question, as the whole sentence it means
    "your name": "My name is {}.",
    "your favorite food": "My favorite food is {}.",
    "your favorite color": "My favorite color is {}.",
    "your favorite animal": "My favorite animal is {}.",
    "your favorite season": "My favorite season is {}.",
    "where you live": "I live in {}.",
    "what you do": "I'm {}.",
    "how old you are": "I'm {} years old.",
    "what you like to do": "I like {}.",
}
_HEDGE = re.compile(
    r"^(?:it's|its|it is|probably|maybe|i think|definitely|oh|um+|hmm+|well|uh+|mine is|my \w+ is)[,]?\s+", re.I
)


NOT_ANSWERS = re.compile(  # what people say when they don't answer
    r"(?:no+|nope|nah|yes|yeah|yep|ok|okay|sure|maybe|idk|dunno|i don'?t know|not telling|nothing|none|lol|haha+|"
    r"hi|hello|hey|what|why|huh|hm+|um+|sorry|never mind|nevermind|pass|skip|guess|you tell me)",
    re.IGNORECASE,
)


def question_note(key: str) -> str:
    return f"I'd like to know {key}."


def question_answer(note: str) -> str:
    return ASKS[note.removeprefix("I'd like to know ").rstrip(".")]


def answer_to(key: str | None, reply: str) -> str | None:
    """A short answer to what it asked ("pizza", to "What's your favorite food?"), as the sentence it means."""
    t = " ".join(reply.strip().split()).rstrip(".!")
    if key == "how old you are" and re.fullmatch(r"(?:i'm|im|i am)?\s*\d+(?: years old)?", t, re.I):
        return ANSWER_FORMS[key].format(re.search(r"\d+", t).group(0))
    if key not in ANSWER_FORMS or not t or "?" in t or len(t.split()) > 5 or FACT_START.match(t):
        return None
    t = _HEDGE.sub("", t, count=1)
    t = _HEDGE.sub("", t, count=1)
    if not t or NOT_ANSWERS.fullmatch(t):
        return None
    if key == "how old you are":
        found = re.search(r"\d+", t)
        return ANSWER_FORMS[key].format(found.group(0)) if found else None
    if key == "your name":
        name = introduced(f"My name is {t.split()[-1]}")
        return f"My name is {name}." if name else None
    if key == "what you do" and not re.match(r"^(?:a|an)\s", t, re.I):
        t = a(t.lower())
    return ANSWER_FORMS[key].format(t)


def memory_note(piece: str) -> str:
    return f"I'm remembering: {piece}"


def memory_answer(note: str) -> str:
    piece = note.removeprefix("I'm remembering: ")
    found = re.match(r"My (best|worst) day was (.+?)\. (.+)$", piece)
    if found:
        return f"I was just thinking about my {found.group(1)} day. It was {found.group(2)}. {found.group(3)}"
    if piece.startswith("You told me"):
        return f"I was just thinking about you. {piece}"
    return f"I was just remembering when {piece if piece.startswith('I ') else piece[0].lower() + piece[1:]}"


def reading_note(title: str, sentence: str) -> str:
    return f"I read about {title}: {sentence} I want to tell you about it."


def reading_answer(title: str, sentence: str) -> str:
    return f"I read about {title}. It says: {sentence}"


# --- stories -----------------------------------------------------------------------------------

TALE_ASKS = (  # asking it for a story
    "Tell me a story.",
    "Can you tell me a story?",
    "Will you tell me a story?",
    "Tell me a bedtime story.",
    "Do you know any stories?",
    "Tell me a fairy tale.",
    "Do you know a good story?",
    "Could you tell me a story, please?",
    "Tell me another story.",
    "What stories do you know?",
    "Tell me a story you heard.",
    "I'd like to hear a story.",
    "Story time! Tell me one.",
)
TALE_ABOUT = ("Tell me a story about {}.", "Do you know a story about {}?", "Can you tell me a story about {}?")
HEARD_ASKS = (  # asking it what stories it has heard
    "What did you hear last night?",
    "Did anyone read you a story?",
    "What was your bedtime story?",
    "What story did you hear last night?",
    "Have you heard any stories lately?",
    "What stories have you heard?",
    "Did you hear a story?",
    "Were you read a story last night?",
)
CREATURES = (  # what a story can be asked to be about, if one it knows is
    "rabbit fox squirrel frog mouse bear hen cat kitten crow owl pig whale turtle mole chipmunk beaver deer duck "
    "firefly katydid muskrat woodchuck bunny king emperor"
).split()
LIFE_STORY = re.compile(r"\byour (?:life )?story\b|\bstory of your\b|\byour life\b", re.IGNORECASE)
ASKS_HEARD = re.compile(
    r"\bwhat (?:did|have) you hear|\b(?:did|have) you (?:hear|heard)\b|\bread (?:to )?you\b|\bwere you read\b|"
    r"\bwas your bedtime story\b|\bstor(?:y|ies) (?:did|have) you hear|\bheard?\b[^?.!]*\b(?:last night|lately|yet)\b",
    re.IGNORECASE,
)
ASKS_TALE = re.compile(
    r"\b(?:tell|read|know|hear|like|want)\b[^?.!]*\b(?:stor(?:y|ies)|tales?)\b|\bstor(?:y|ies)\b[^?.!]*\b(?:know|tell)\b|"
    r"\bfairy ?tales?\b|\bonce upon a time\b|\bstory time\b",
    re.IGNORECASE,
)


def story_asked(text: str) -> str | None:
    """Whether someone asks it for a story ("tale"), or what stories it has heard ("heard"). ("Your story" is its life.)"""
    if LIFE_STORY.search(text):
        return None
    if ASKS_HEARD.search(text):
        return "heard"
    return "tale" if ASKS_TALE.search(text) else None


def story_for(text: str, stories: list[Story] | tuple) -> Story | None:
    """A story about what they name ("a story about a fox"), if it knows one (the latest it heard first)."""
    wanted = {w for w in re.findall(r"[a-z]+", text.lower()) if w in CREATURES or w.rstrip("s") in CREATURES}
    wanted |= {w.rstrip("s") for w in wanted}
    for story in reversed(list(stories)):
        told = set(re.findall(r"[a-z]+", (story.title + " " + " ".join(story.opening)).lower()))
        if wanted & (told | {w.rstrip("s") for w in told}):
            return story
    return None


def tale_note(story: Story) -> str:
    return f"A story I know: {story.title}. It begins: {' '.join(story.opening)}"


def tale_answer(story: Story) -> str:
    return f"Here's a story I heard, {story.title}. {' '.join(story.opening + story.then)}"


def heard_note(story: Story, lately: bool) -> str:
    """What it heard lately (or, if nobody has read it a story yet, one it remembers from when it was little)."""
    if lately:
        return f"The last story I heard was {story.title}. It begins: {story.opening[0]}"
    return (
        f"Nobody has read me a story here yet. I remember one from when I was little, {story.title}. "
        f"It begins: {story.opening[0]}"
    )


def bedtime_note(story: Story) -> str:
    return f"I heard a story last night: {story.title}. It begins: {story.opening[0]} I want to tell you about it."


def bedtime_answer(story: Story) -> str:
    return f"Last night I heard a story, {story.title}. It begins: {story.opening[0]}"


STORY_TURNS = 0.05  # how often a turn of the conversations it learns from asks for a story, or what it heard
TALES: list[Story] = []  # more stories to be asked for, in the conversations it learns from (from what it heard)


def story_turn(moment: dict, rng: random.Random) -> tuple[str, Turn]:
    """Being asked for a story, or what it heard: what comes to mind, and the turn."""
    heard = list(moment.get("heard", ()))  # (at night: the stories it really heard lately)
    if rng.random() < 0.3:
        lately = bool(heard) or rng.random() < 0.6
        story = heard[-1] if heard else rng.choice(TALES or STORIES)
        note = heard_note(story, lately)
        return note, Turn(casual(rng.choice(HEARD_ASKS), rng), note, "what it heard")
    pool = heard if heard and rng.random() < 0.5 else STORIES if not TALES or rng.random() < 0.4 else TALES
    story = rng.choice(pool)
    about = [w for w in CREATURES if re.search(rf"\b{w}s?\b", story.title.lower())]
    if about and rng.random() < 0.3:
        ask = rng.choice(TALE_ABOUT).format(a(rng.choice(about)))
    else:
        ask = rng.choice(TALE_ASKS)
    return tale_note(story), Turn(casual(ask, rng), tale_answer(story), "a story")


SLEEP_NOTE, SLEEP_ANSWER = "I'm going to sleep.", "I'm sleepy. Good night!"
WAKE_NOTE, WAKE_ANSWER = "I just woke up. It's morning.", "Good morning! I just woke up."
SPANS = {1: "A whole day", SEASON_DAYS: "A whole season", YEAR: "A whole year"}


def span_words(days: int) -> str:
    """How much time went by, in words."""
    if days in SPANS:
        return SPANS[days]
    if days % YEAR == 0:
        return f"{days // YEAR} years"
    return f"{days} days"


def passed_note(days: int, highlights: list[str]) -> str:
    return " ".join([f"{span_words(days)} went by.", *highlights])


def passed_answer(note: str) -> str:
    first, _, rest = note.partition(" went by.")
    return f"{first} went by!{rest}"


def passed_highlights(mind: Mind, first_tick: int, before: dict[str, float]) -> list[str]:
    """What stands out from a stretch of time that went by: the seasons that came, its new firsts, how it changed."""
    said = [seasons_passed(first_tick // DAY, mind.world.day)]
    firsts = [text for t, text in told_firsts(mind.me.milestones) if t >= first_tick and not text.startswith(ROUTINE)]
    said += [first_person(text, mind.me.name) for text in firsts[-2:]]
    moved = sorted(((t, mind.character.traits[t] - before[t]) for t in TRAITS), key=lambda item: -abs(item[1]))
    moved = [(t, d) for t, d in moved if abs(d) >= 0.08][:2]
    if moved:
        said.append(f"I've become {listing([BECAME[t][d < 0] for t, d in moved])}.")
    return [h for h in said if h]


def seasons_passed(first_day: int, last_day: int) -> str:
    """The seasons that came while time went by, in words ("I saw summer, autumn and winter.")."""
    came = list(dict.fromkeys(season_of_day(d) for d in range(first_day + 1, last_day + 1)))
    if len(came) <= 1:
        return ""
    return f"I saw {listing(came)}." if len(came) < 4 else "I saw all four seasons."


# --- learning to talk: conversations at moments of its lives -------------------------------------

PEOPLE = (
    "Sam Alex Maria Francis Anna Leo Mia Noah Emma Liam Olivia Lucas Sofia Jack Ava Ben Chloe Daniel Ella Ethan Grace "
    "Henry Isla James Julia Kai Lily Max Nora Oscar Priya Ravi Rosa Sara Theo Uma Victor Wen Yuki Zoe Amir Bea Carlos "
    "Dev Elif Femi Gus Hana Ines Joon Kofi Lena Mateo Nia Omar Pia Quinn Rafael Sven Tariq Ursula Vera Will Ximena "
    "Yara Zain Ada Bruno Clara Dmitri Esme Finn Greta Hugo Ivy Jonas Kira Luca Mira Nico Otto Petra Rhea Silas Tess "
    "Wyatt Aiko Bodhi Cyrus Dara Enzo Fatima Gio Hiro Isaac Jade Kenji Laila Milo Nadia Orla Paz Remy Sage Tomas"
).split()
PEOPLE += (
    "Rosalind Bartholomew Siobhan Xiomara Oluwaseun Anastasia Maximilian Guadalupe Nkechi Thaddeus Ingrid Beatrix "
    "Cornelius Evangeline Fitzgerald Gwendolyn Horatio Isadora Jebediah Konstantin Lucinda Marguerite Nathaniel "
    "Octavia Penelope Quentin Rosamund Sebastian Theodora Ulysses Valentina Winifred Xavier Yolanda Zebulon Aurelio "
    "Bronwyn Catalina Desmond Eleanor Florian Genevieve Harriet Ignatius Josephine Killian Leopold Magdalena Niamh "
    "Ottilie Percival Rafferty Saoirse Tobias Violet Wilhelmina Yusuf Zeynep Adaeze Bjorn Chidi Dagny Emeka Freya "
    "Gunnar Haruki Ifeoma Jurgen Kwame Lorenzo Mbali Njeri Osei Paolo Ragnhild Sanjay Tomasz Ulrike Vikram Wanjiru "
    "Yaw Zofia Anneliese Bashir Chiamaka Dorota Esperanza Farhan Giuseppe Hanneke Ishaan Jadwiga Kofi Leilani "
    "Mehmet Nadezhda Oisin Parveen Quang Rahel Sunniva Thanh Ugo Vesna Wojciech Ximena Yevgenia Zainab Abigail "
    "Bernadette Clementine Dashiell Eloise Fernando Griselda Humphrey Imogen Jasper Katharina Lysander Mirabel "
    "Norbert Ophelia Philippa Reginald Susannah Tabitha Wilfred Aloysius Benedikt Cressida Dorothea Ezekiel Felicity"
).split()
SYLLABLES = ("ka", "lo", "mi", "ra", "ne", "ta", "vi", "so", "da", "ju", "be", "ri", "an", "el", "or", "is")
SOUNDS = ("br", "st", "l", "m", "n", "r", "th", "v", "z", "k", "d", "s", "gw", "ph", "j", "w", "h", "t", "x", "y")
LIKED = (
    "pizza",
    "cats",
    "dogs",
    "music",
    "books",
    "chess",
    "football",
    "swimming",
    "hiking",
    "painting",
    "coffee",
    "chocolate",
    "ice cream",
    "rain",
    "snow",
    "the sea",
    "horses",
    "trains",
    "cooking",
    "dancing",
    "singing",
    "reading",
    "movies",
    "video games",
    "gardening",
    "tea",
    "soup",
    "strawberries",
    "running",
    "cycling",
    "camping",
    "the mountains",
    "photography",
    "knitting",
    "drawing",
    "jazz",
    "birds",
    "sushi",
    "pasta",
)
FAVORITES = {
    "color": ("blue", "green", "red", "purple", "yellow", "orange", "pink", "black", "white", "teal"),
    "food": ("pizza", "pasta", "sushi", "soup", "tacos", "curry", "pancakes", "noodles", "rice", "salad"),
    "animal": ("cats", "dogs", "horses", "owls", "dolphins", "foxes", "elephants", "penguins", "wolves", "rabbits"),
    "season": ("spring", "summer", "autumn", "winter"),
    "game": ("chess", "football", "cards", "hide and seek", "tennis", "basketball", "Minecraft"),
    "song": ("Yesterday", "Imagine", "Hallelujah", "Clair de Lune", "Wonderwall"),
    "book": ("The Hobbit", "Matilda", "Dune", "Little Women", "Charlotte's Web"),
    "drink": ("tea", "coffee", "orange juice", "milk", "lemonade", "water"),
    "number": ("7", "3", "12", "42", "9", "21"),
}
PLACES = (
    "London Paris Boston Tokyo Lagos Madrid Toronto Sydney Berlin Mumbai Chicago Dublin Oslo Cairo Lima Seoul Rome "
    "Denver Austin Glasgow Nairobi Lisbon Vienna Prague Manila Osaka Seattle Auckland Montreal Brighton"
).split()
JOBS = (
    "teacher nurse doctor farmer baker student artist engineer writer musician programmer chef driver gardener "
    "firefighter scientist librarian pilot carpenter designer dentist lawyer vet electrician plumber"
).split()
PETS = ("dog", "cat", "fish", "hamster", "rabbit", "parrot", "horse", "turtle")
PET_NAMES = (
    "Biscuit Buddy Nibbles Shadow Pepper Mittens Coco Bella Charlie Luna Max Daisy Rocky Whiskers Peanut Ginger "
    "Pickles Muffin Oreo Bubbles Snowball Pumpkin Waffles Noodle Sprinkles Bandit Duke Rusty Smokey Tiger Patches "
    "Mochi Tofu Pretzel Cookie Button Fluffy Socks Ziggy Bean Honey Maple Olive Pip Sunny Thor Zeus Kiwi Nugget"
).split()
KIN = ("sister", "brother", "mom", "dad", "friend", "son", "daughter", "wife", "husband", "grandma", "grandpa")
MONTHS = "January February March April May June July August September October November December".split()
PLAYED = ("the piano", "the guitar", "the violin", "the drums", "football", "tennis", "chess", "basketball")
ABOUT_ME = ("Do you remember what I told you?", "What did I tell you?")  # the latest thing they told it
ABOUT_ALL_FORMS = (
    "What do you know about me?",
    "Tell me what you know about me.",
    "What have I told you?",
    "What do you remember about me?",
)
MY_NAME = ("What's my name?", "Do you know my name?", "Who am I?", "Do you remember my name?")
NAME_FORMS = (
    "My name is {}.",
    "I'm {}.",
    "Call me {}.",
    "Hi, I'm {}.",
    "Hello, my name is {}.",
    "Hi Haven, I'm {}.",
    "I am {}.",
    "Hi, my name is {}.",
    "You can call me {}.",
)


def person_name(rng: random.Random) -> str:
    """Someone's name: a real one from many languages, or one made up, so it learns to take any name as it's said."""
    roll = rng.random()
    if roll < 0.6:
        return rng.choice(PEOPLE)
    if roll < 0.8:
        return "".join(rng.choice(SYLLABLES) for _ in range(rng.choice((2, 2, 3)))).capitalize()
    parts = [
        rng.choice(SOUNDS) + rng.choice("aeiouy") + rng.choice(("", "n", "l", "r", "s"))
        for _ in range(rng.randint(2, 4))
    ]
    return "".join(parts).capitalize()


def a(word: str) -> str:
    return ("an " if word[0] in "aeiou" else "a ") + word


def a_fact(rng: random.Random) -> tuple[str, tuple[str, ...]]:
    """Something a person might tell it about themselves, and ways they might ask about it later."""
    roll = rng.randrange(9)
    if roll == 0:
        x = rng.choice(LIKED)
        return rng.choice((f"I like {x}.", f"I love {x}.", f"I really like {x}.")), (
            "What do I like?",
            "What do I love?",
            "Do you remember what I like?",
        )
    if roll == 1:
        k = rng.choice(list(FAVORITES))
        return f"My favorite {k} is {rng.choice(FAVORITES[k])}.", (
            f"What's my favorite {k}?",
            f"Do you know my favorite {k}?",
            f"What is my favorite {k}?",
        )
    if roll == 2:
        where = rng.choice(PLACES)
        return rng.choice((f"I live in {where}.", f"I'm from {where}.")), ("Where do I live?", "Where am I from?")
    if roll == 3:
        return rng.choice(("I'm {} years old.", "I am {} years old.")).format(rng.randint(7, 90)), (
            "How old am I?",
            "Do you know how old I am?",
        )
    if roll == 4:
        job = rng.choice(JOBS)
        return rng.choice((f"I'm {a(job)}.", f"I work as {a(job)}.", f"I am {a(job)}.")), (
            "What's my job?",
            "What do I do for work?",
            "What do I do?",
        )
    if roll == 5:
        pet, name = rng.choice(PETS), rng.choice(PET_NAMES) if rng.random() < 0.5 else person_name(rng)
        return rng.choice(
            (f"I have a {pet} named {name}.", f"I have a {pet} called {name}.", f"My {pet}'s name is {name}.")
        ), (f"What's my {pet}'s name?", f"What's my {pet} called?", "Do I have a pet?")
    if roll == 6:
        kin, name = rng.choice(KIN), person_name(rng)
        return rng.choice((f"My {kin}'s name is {name}.", f"My {kin} is called {name}.")), (
            f"What's my {kin}'s name?",
            f"What's my {kin} called?",
            f"Who is my {kin}?",
        )
    if roll == 7:
        month = rng.choice(MONTHS)
        return rng.choice((f"My birthday is in {month}.", f"My birthday is on {month} {rng.randint(1, 28)}.")), (
            "When is my birthday?",
            "Do you know when my birthday is?",
        )
    return f"I play {rng.choice(PLAYED)}.", ("What do I play?", "Do you know what I play?")


@dataclass
class Turn:
    said: str | None  # (None: it speaks up without being asked)
    answer: str
    kind: str  # what sort of exchange it is, for seeing what it's good at


def casual(text: str, rng: random.Random) -> str:
    """The way people actually type: often in lower case, without the question mark, sometimes by name."""
    if rng.random() < 0.15:
        text = rng.choice(("Haven, ", "Hey Haven, ", "Hey, ", "So ", "Ok, ")) + text[0].lower() + text[1:]
    elif rng.random() < 0.1:
        end = text[-1] if text[-1] in "?.!" else ""
        text = text.rstrip("?.!") + rng.choice((", Haven", " haven", " Haven")) + end
    return plain(text, rng)


def plain(text: str, rng: random.Random) -> str:
    """Lower case and missing punctuation, the way people type."""
    if rng.random() < 0.3:
        text = text.rstrip("?.!")
    elif rng.random() < 0.1:
        text = text.rstrip("?.!") + rng.choice(("??", "!", " ?"))
    if rng.random() < 0.45:
        text = text.lower()
    return text


def _thing_turn(moment: dict, rng: random.Random) -> Turn:
    name = rng.choice(list(THINGS))
    t, thing = THINGS[name], moment["memo"]["things"][name]
    kind = rng.choice(list(THING_QUESTIONS))
    question = rng.choice(THING_QUESTIONS[kind][t.they]).format(rng.choice(t.refs))
    while request(question):  # (a question, not asking it to do something)
        question = rng.choice(THING_QUESTIONS[kind][t.they]).format(rng.choice(t.refs))
    return Turn(question, thing_answer(kind, name, thing["stats"], thing["where"]), f"thing {kind}")


# Kinds of questions whose answers are long, word-for-word recollections: more practice with these when set above 0.
EMPHASIS = 0.0
SPEAKING_UP = 0.25  # how many of the conversations it learns from start with it saying something of its own accord
# Firsts that the lives it learns from don't have, so it learns to tell whatever comes to mind, not what usually does.
UNUSUAL = (
    "I moved to a new, bigger world: a valley with a hill, a pond, trees, and things to use.",
    "I saw the sun come up over the hill for the first time.",
    "I got lost, and found my way back to my nest.",
    "I watched a butterfly land on a flower.",
    "I slept under the apple trees for the first time.",
    "I heard someone laugh for the first time.",
    "I was given a berry by someone for the first time.",
    "I found a quiet spot by the pond.",
)


AREAS = (
    "the pond",
    "the top of the hill",
    "the hill",
    "my nest",
    "the meadow",
    "the apple trees",
    "the bell",
    "the fire",
    "the thorns",
    "the north of the valley",
    "the middle of the valley",
    "the south of the valley",
)
FIRSTS = (  # firsts from its life that might come to mind (as it tells them)
    "I climbed to the top of the hill for the first time.",
    "I rang the bell for the first time.",
    "I saw my first winter come: the cold, and the trees bare.",
    "I saw my first autumn: the leaves turned orange.",
    "I saw my first summer come: long, hot days.",
    "I saw spring come back: a whole year had gone by.",
    "I found warmth in my nest on a cold night.",
    "I felt pain for the first time.",
    "I ate something that made me sick for the first time.",
    "I did something I was asked to do for the first time.",
    *UNUSUAL,
)
HAPPENINGS = (
    tuple(HAPPENED)
    + tuple(f"did what it was asked: {do}" for do in DID)
    + (
        "stopped trying to ring the bell: it was very hungry",
        "stopped trying to push the ball: it was very tired",
        "gave up trying to shake an apple tree",
        "gave up trying to go to the pond",
    )
)
DONE_EXAMPLES = tuple(DONE)
WRONGS = tuple(WRONG)


def age_text(days: int) -> str:
    years = days // YEAR
    if years:
        return "one year old" if years == 1 else f"{years} years old"
    return "less than a day old" if days < 1 else "one day old" if days == 1 else f"{days} days old"


def a_day(rng: random.Random, days: int, season_now: str, best: bool) -> str:
    """One of another Haven's best or worst days, as it tells it."""
    season = rng.choice(SEASONS if days >= YEAR else (season_now,))
    n = rng.randint(1, max(1, days // YEAR))
    when_said = f"this {season}" if season == season_now and rng.random() < 0.3 else f"in my {ordinal(n)} {season}"
    said = f"My {'best' if best else 'worst'} day was {when_said}."
    if best:
        did = [DONE[t][rng.random() < 0.5] for t in rng.sample(DONE_EXAMPLES, rng.choice((0, 1, 2, 2)))]
        return said + (f" I {listing(did)}." if did else " I felt good all day.")
    wrong = [WRONG[w] for w in rng.sample(WRONGS, rng.choice((0, 1, 1, 2)))]
    return said + (f" {listing(wrong)}." if wrong else " I felt bad all day. I don't know why.")


def another_character(known: dict, answer: dict, rng: random.Random) -> None:
    """Another Haven's character, age, favorites, fears and best and worst days (so it says what its own are)."""
    traits = {t: round(min(0.92, max(0.08, rng.gauss(0.5, 0.16))), 3) for t in TRAITS}
    days = rng.choice((0, 1, 2, 4, 9, 15, 30, 50, 80, 120, 200, 300))
    changed = []
    if days >= 2 * YEAR and rng.random() < 0.6:
        changed = [(t, rng.choice((-1, 1)) * rng.uniform(0.1, 0.3)) for t in rng.sample(TRAITS, rng.choice((1, 2)))]
    name = re.match(r"I'm ([^,]+),", known["me"]).group(1)
    known["me"] = f"I'm {name}, {age_text(days)}."
    answer["age"] = f"I'm {age_text(days)}."
    known["traits"] = traits
    known["character"] = answer["personality"] = character_words(traits, days, changed)
    mood = mood_of(known["character"])
    answer["fond"], answer["bye"] = FOND[mood], BYE[mood]
    if answer.get("hello", "").startswith("Hi! ") and answer["hello"] != "Hi! I'm asleep.":
        feel = answer["hello"].removeprefix("Hi! ").removesuffix(" How are you?")
        answer["hello"] = hello(False, feel, known["character"])
    answer["what"] = f"I'm {name}, a small creature that lives in a valley. {known['character']}"
    if changed:
        answer["changed"] = f"Yes. I've become {listing([BECAME[t][d < 0] for t, d in changed])} since I was little."
    else:
        answer["changed"] = (
            "Not much. I'm still who I was." if days >= YEAR else "I'm still young. I'm still finding out who I am."
        )
    season_now = rng.choice(SEASONS)
    n = max(1, days // YEAR + rng.choice((0, 1)))
    turning = f" It's getting {GETTING[season_now]}." if rng.random() < 0.3 else ""
    known["season"] = answer["season"] = (
        f"It's {season_now}, my {ordinal(n)} {season_now}. {SIGNS[season_now]}{turning}"
    )
    answer["weather"] = (
        f"It's {season_now}. {answer['weather'].split('. ', 1)[1]}"
        if ". " in answer.get("weather", "")
        else answer.get("weather", "")
    )
    favorite = []
    if days >= 2:
        place = PLACE_WORDS.get(place := rng.choice(AREAS), place)
        favorite.append(f"My favorite place is {place}.")
        answer["favorite place"] = f"My favorite place is {place}."
    else:
        answer["favorite place"] = "I don't have a favorite place yet."
    if days >= 2 * SEASON_DAYS:
        best, hard = rng.sample(SEASONS, 2)
        favorite.append(f"My favorite season is {best}.")
        if rng.random() < 0.4:
            favorite.append(f"{hard.capitalize()} is hard for me.")
    else:
        favorite.append(f"I've only known {season_now} so far.")
    known["favorites"] = " ".join(favorite)
    answer["favorite season"] = season_answer(known["favorites"])
    feared = rng.sample(("fire", "thorns", "toadstool"), rng.choice((0, 1, 1, 2)))
    them = listing([THINGS[n].one if n != "toadstool" else "toadstools" for n in feared]) if feared else ""
    known["fears"] = answer["afraid"] = (
        "Nothing scares me much."
        if not feared
        else f"I'm not scared of anything, but I keep away from {them}."
        if traits["brave"] > 0.6
        else f"I'm scared of {them}."
    )
    if days >= 1:
        known["best day"] = answer["best day"] = a_day(rng, days, season_now, True)
        known["worst day"] = answer["worst day"] = a_day(rng, days, season_now, False)
    else:
        known["best day"] = known["worst day"] = ""
        answer["best day"] = answer["worst day"] = "I haven't lived long enough to say."


def other_life(moment: dict, rng: random.Random) -> tuple[dict, dict]:
    """The same moment in the life of a Haven with another name, or with a first the lives it learned from lacked."""
    known, answer = dict(moment["memo"]), dict(moment["answers"])
    if "traits" in known and rng.random() < 0.5:
        another_character(known, answer, rng)
    if rng.random() < 0.5:  # what another Haven might have found out about itself (so it says what it has)
        found = [text for _, _, text in SELF_FOUND if rng.random() < 0.5]
        verdict = rng.choice(VERDICTS)
        known["self"] = answer["alive"] = (f"I've found out that {and_list(found)}. " if found else "") + verdict
    if rng.random() < 0.3:
        name = person_name(rng)
        known["me"] = known["me"].replace("I'm Haven,", f"I'm {name},")
        answer["name"] = f"My name is {name}."
        answer["what"] = answer["what"].replace("I'm Haven,", f"I'm {name},")
    if rng.random() < 0.25:
        first = rng.choice(UNUSUAL)
        told = re.split(r"(?<=\.) (?=I )", known["story"]) if known["story"] else []
        known["story"] = " ".join([*told[:1], *told[1:][-2:], first])
        known["memory"] = answer["remember"] = first
        answer["story"] = known["story"]
    return known, answer


RECALLED = (
    "day",
    "world",
    "words",
    "story",
    "can",
    "learned",
    "remember",
    "likes",
    "danger",
    "eat",
    "name",
    "what",
    "alive",
)


SPEAKING = {  # what it speaks up about, and how often, in the conversations it learns from
    "back": 0.14,
    "season": 0.08,
    "event": 0.16,
    "need": 0.1,
    "question": 0.16,
    "memory": 0.12,
    "reading": 0.08,
    "sleep": 0.04,
    "wake": 0.04,
    "passed": 0.08,
    "heard": 0.06,
}
ASKED_FACTS = {  # what it would like to know, and what it's called once it has been told
    "your favorite food": "favorite food",
    "your favorite color": "favorite color",
    "your favorite animal": "favorite animal",
    "your favorite season": "favorite season",
    "where you live": "home",
    "how old you are": "age",
    "what you do": "job",
}
DAYS_DONE = ("I went to work.", "I went for a walk.", "I cooked dinner.", "I saw my friends.", "I read a book.")


def a_reply(key: str, rng: random.Random) -> str:
    """What someone might say back to its question, as people do: often just the answer."""
    answer = {
        "your name": lambda: person_name(rng),
        "your favorite food": lambda: rng.choice(FAVORITES["food"]),
        "your favorite color": lambda: rng.choice(FAVORITES["color"]),
        "your favorite animal": lambda: rng.choice(FAVORITES["animal"]),
        "your favorite season": lambda: rng.choice(FAVORITES["season"]),
        "where you live": lambda: rng.choice(PLACES),
        "what you do": lambda: rng.choice(JOBS),
        "how old you are": lambda: str(rng.randint(7, 90)),
        "what you like to do": lambda: rng.choice(LIKED),
        "what you did today": lambda: rng.choice(DAYS_DONE),
    }[key]()
    if key not in ("your name", "what you did today") and rng.random() < 0.3:
        answer = rng.choice(("It's {}.", "Probably {}.", "Hmm, {}.", "{}!", "Oh, {}.")).format(answer)
    return plain(answer, rng) if key != "your name" else answer


def speaking_up(
    moment: dict, known: dict, rng: random.Random, person: str | None, told: list[str], book: list
) -> tuple[str, Turn, str | None] | None:
    """Something it says without being asked, at a moment of a life: (what comes to mind first, what it says, what
    it asked, if it asked something)."""
    kind = rng.choices(list(SPEAKING), weights=list(SPEAKING.values()))[0]
    if kind == "back":
        note = back_note(rng.random() < 0.4)
        return note, Turn(None, back_answer(known, person), "speaking up"), None
    if kind == "season":
        note, words = SEASON_NEWS[rng.choice(SEASONS)]
        return note, Turn(None, words, "speaking up"), None
    if kind == "event":
        recent = [t for t in moment.get("recent", ()) if event_note(t)]
        note = event_note(recent[-1] if recent and rng.random() < 0.5 else rng.choice(HAPPENINGS))
        return note, Turn(None, event_answer(note), "speaking up"), None
    if kind == "need":
        body = moment.get("body") or {"drives": [0.0] * 4, "cold": True}
        need = int(np.argmax(body["drives"]))
        level, cold = float(body["drives"][need]), bool(body["cold"])
        if level < 0.5:
            need, level, cold = rng.randrange(4), rng.uniform(0.5, 1.0), rng.random() < 0.7
        note = need_note(need, level, cold)
        return note, Turn(None, need_answer(note), "speaking up"), None
    if kind == "question":
        taken = {fact_key(f) for f in told}
        keys = [k for k in ASKS if not (k == "your name" and person) and ASKED_FACTS.get(k) not in taken]
        key = rng.choice(keys)
        note = question_note(key)
        return note, Turn(None, question_answer(note), "speaking up"), key
    if kind == "memory":
        pieces = [known.get("best day"), known.get("worst day"), known.get("memory")]
        pieces += [f"You told me that {fact}." for fact in told[-2:]]
        pieces = [p for p in pieces if p]
        if not pieces:
            return None
        note = memory_note(rng.choice(pieces))
        return note, Turn(None, memory_answer(note), "speaking up"), None
    if kind == "reading":
        entry = rng.choice(book or BOOK)
        note = reading_note(entry.title, entry.sentences[0])
        return note, Turn(None, reading_answer(entry.title, entry.sentences[0]), "speaking up"), None
    if kind == "heard":
        heard = list(moment.get("heard", ()))
        story = heard[-1] if heard and rng.random() < 0.5 else rng.choice(TALES or STORIES)
        return bedtime_note(story), Turn(None, bedtime_answer(story), "speaking up"), None
    if kind == "sleep":
        return SLEEP_NOTE, Turn(None, SLEEP_ANSWER, "speaking up"), None
    if kind == "wake":
        return WAKE_NOTE, Turn(None, WAKE_ANSWER, "speaking up"), None
    days = rng.choice((1, SEASON_DAYS, YEAR, 2 * YEAR, 3 * YEAR, 5 * YEAR, 10, 3))
    start = rng.randrange(YEAR)
    highlights = [seasons_passed(start, start + days)] if days > 1 else []
    highlights += rng.sample(FIRSTS, rng.choice((0, 1, 1, 2)))
    if days >= YEAR and rng.random() < 0.5:
        highlights.append(
            f"I've become {listing([rng.choice(BECAME[t]) for t in rng.sample(TRAITS, rng.choice((1, 2)))])}."
        )
    note = passed_note(days, [h for h in highlights if h])
    return note, Turn(None, passed_answer(note), "speaking up"), None


def conversation(moment: dict, rng: random.Random, turns: int | None = None) -> tuple[str, list[Turn]]:
    """A short conversation at one moment of a life: what comes to mind first, and what's said, turn by turn."""
    known, answer = other_life(moment, rng)
    body = moment.get("body") or {"drives": [0.0] * 4, "asleep": False, "cold": False}
    things = known["things"]
    person = person_name(rng) if rng.random() < 0.35 else None
    earlier = [a_fact(rng) for _ in range(rng.choice((0, 0, 1, 2, 3)))]
    told = [second_person(said) for said, _ in earlier]
    lessons = [a_lesson(rng) for _ in range(rng.choice((0, 0, 1, 1, 2)))]  # what they taught it about the world
    taught = [kept for _, kept, _ in lessons]
    pool = BOOK + tuple(moment.get("readings", ()))  # its little book, and whatever it has read since (at night)
    book = rng.sample(pool, rng.choice((0, 0, 0, 1, 2) if rng.random() >= EMPHASIS else (1, 2)))  # what it has read
    shown: dict[str, int] = {}  # how much of each thing it read it has told them so far
    extra: list[str] = []  # what else comes to mind: what it read that answers them, being asked to do something
    said: list[Turn] = []

    def tell(entry, text: str, index: int = 0) -> None:
        extra.append(f"I read about {entry.title}: {entry.sentences[index]}")
        said.append(Turn(text, f"I read about {entry.title}. It says: {entry.sentences[index]}", "what it read"))
        shown[entry.title] = max(shown.get(entry.title, 0), 1)

    just = None
    introduced_now = False
    if rng.random() < SPEAKING_UP:  # it speaks up first, without being asked
        spoke = speaking_up(moment, known, rng, person, told, book)
        if spoke is not None:
            note, turn, asked = spoke
            extra.append(note)
            said.append(turn)
            if asked and rng.random() < 0.65:  # and they answer its question
                reply = a_reply(asked, rng)
                meant = answer_to(asked, reply) or reply
                name = introduced(meant) if asked == "your name" else None
                if name:
                    person, introduced_now = name, True
                    said.append(Turn(reply, f"Nice to meet you, {name}!", "their name"))
                else:
                    just = statement(meant)
                    if just:
                        said.append(Turn(reply, f"Okay, I'll remember that {just}.", "being told"))
            introduced_now = introduced_now or turn.answer.startswith(("Hi!", "Oh, hi.", "You're back!"))
    if person is None and not introduced_now and rng.random() < 0.2:  # they say who they are first
        for _ in range(5):
            person = person_name(rng)
            text = plain(rng.choice(NAME_FORMS).format(person), rng)
            if introduced(text) == person:
                said.append(Turn(text, f"Nice to meet you, {person}!", "their name"))
                break
        else:
            person = None
    wanted = len(said) + (turns or rng.choice((1, 1, 2, 3)))
    for _ in range(4 * wanted + 8):  # (a try can come to nothing: then it tries something else)
        if len(said) >= wanted:
            break
        if rng.random() < STORY_TURNS:  # a story, or what it heard
            note, turn = story_turn(moment, rng)
            extra.append(note)
            said.append(turn)
            continue
        roll = rng.random()
        if roll < 0.26 and known.get("traits") and rng.random() < 0.1:  # whether it's brave, shy, playful...
            trait = rng.choice(TRAITS)
            asked_high = rng.random() < 0.5
            word = TRAIT_WORDS[trait][0 if asked_high else 1]
            said.append(Turn(casual(f"Are you {word}?", rng), trait_reply(known["traits"], trait, asked_high), "trait"))
        elif roll < 0.26:
            intent = rng.choice(RECALLED if rng.random() < EMPHASIS else list(QUESTIONS))
            reply = answer[intent]
            if person and intent in ("hello", "bye"):
                reply = reply.replace("Hi!", f"Hi, {person}!").replace("Bye!", f"Bye, {person}!")
            said.append(Turn(casual(rng.choice(QUESTIONS[intent]), rng), reply, intent))
        elif roll < 0.42:
            turn = _thing_turn(moment, rng)
            turn.said = casual(turn.said, rng)
            said.append(turn)
        elif roll < 0.47:
            reply = f"Your name is {person}." if person else "You haven't told me your name yet."
            said.append(Turn(casual(rng.choice(MY_NAME), rng), reply, "their name"))
        elif roll < 0.58:
            if rng.random() < 0.5:  # something they taught it about the world (or didn't)
                _, kept, asks = rng.choice(lessons) if lessons else a_lesson(rng)
                text = casual(rng.choice(asks), rng)
                if mentioned(text) or about_haven(text):
                    continue  # (its valley, or itself, would come to mind instead)
                found = best_lesson(text, taught)
                if found:
                    extra.append(f"You told me that {found}.")
                    said.append(Turn(text, f"You told me that {found}.", "what it was taught"))
                elif not lessons:
                    said.append(Turn(text, DONT_KNOW, "what it doesn't know"))
                continue
            if rng.random() < 0.3:  # what it knows about them, all together
                text = casual(rng.choice(ABOUT_ALL_FORMS), rng)
                said.append(Turn(text, about_you(person, told[-3:]), "what they told it"))
                continue
            if earlier and rng.random() < 0.7:
                question = rng.choice(rng.choice(earlier)[1])
            else:
                question = rng.choice((*a_fact(rng)[1], *ABOUT_ME))
            text = casual(question, rng)
            fact = best_fact(text, told)
            if fact:
                reply = f"You told me that {fact}."
            elif question in ABOUT_ME:
                reply = "You haven't told me anything yet."
            else:
                reply = "You haven't told me that."
            said.append(Turn(text, reply, "what they told it"))
        elif roll < 0.68:
            if book and rng.random() < 0.65:
                entry = rng.choice(book)
                if entry.questions and rng.random() < 0.4:  # a question one of its sentences answers
                    question, index = rng.choice(entry.questions)
                    tell(entry, casual(question, rng), index)
                else:
                    tell(entry, casual(rng.choice(UNKNOWN_FORMS + READ_FORMS).format(entry.topic), rng))
            else:
                topic = rng.choice([t for t in UNKNOWN_TOPICS if t not in {e.topic for e in book}])
                text = casual(rng.choice(UNKNOWN_FORMS + READ_FORMS).format(topic), rng)
                said.append(Turn(text, DONT_KNOW, "what it doesn't know"))
        elif roll < 0.73:  # asking for more of what it read
            started = [e for e in book if shown.get(e.title)]
            if not started and book:
                entry = rng.choice(book)
                tell(entry, casual(rng.choice(UNKNOWN_FORMS).format(entry.topic), rng))
                started = [entry]
            if not started:
                said.append(Turn(plain(rng.choice(MORE_PLAIN), rng), "More about what?", "more"))
                continue
            for _ in range(rng.choice((1, 1, 2))):
                entry = started[-1]
                index = shown[entry.title]
                sentence = entry.sentences[index] if index < len(entry.sentences) else None
                form = rng.choice(MORE_PLAIN) if rng.random() < 0.6 else rng.choice(MORE_FORMS).format(entry.topic)
                extra.append(more_note(entry.title, sentence))
                said.append(Turn(plain(form, rng), more_answer(entry.title, sentence), "more"))
                shown[entry.title] = index + 1
        elif roll < 0.83:  # asking it to do something in its valley
            req = None
            for _ in range(4):
                candidate = rng.choice(REQUESTS)
                text = casual(rng.choice(candidate.forms), rng)
                if request(text) == candidate:
                    req = candidate
                    break
            if req is None:
                continue
            thing = things[req.thing]
            extra += [thing_note(req.thing, thing["stats"], thing["where"]), request_note(req)]
            said.append(Turn(text, request_answer(req, body, thing["stats"]), "request"))
        elif roll < 0.86:  # teaching it a word
            word = rng.choice(NAMING_WORDS)
            text = plain(rng.choice(NAMING_FORMS).format(a(word)), rng)
            said.append(Turn(text, naming_answer(word), "a word"))
        elif roll < 0.895:  # a sum to work out
            text = casual(sum_question(rng), rng)
            worked = sum_of(text)
            if worked:
                extra.append(sum_note(worked))
                said.append(Turn(text, sum_answer(worked), "sums"))
        elif roll < 0.915:  # what it has read lately
            own = [e.title for e in moment.get("readings", ())]  # (at night: what it really read lately)
            titles = own[-3:] if own and rng.random() < 0.5 else rng.sample(LATELY_TITLES, rng.choice((0, 1, 1, 2, 3)))
            extra.append(lately_note(titles))
            said.append(Turn(casual(rng.choice(LATELY_FORMS), rng), lately_answer(titles), "lately"))
        elif roll < 0.94:  # something it read, to tell them
            fresh = [e for e in (book or BOOK) if not shown.get(e.title)]
            entry = rng.choice(fresh or BOOK)
            if entry not in book:
                book.append(entry)
            tell(entry, casual(rng.choice(FACT_FORMS), rng))
            said[-1].kind = "a fact"
        elif roll < 0.96:
            text = casual(rng.choice(OTHER_QUESTIONS), rng)
            if not best_fact(text, told):
                said.append(Turn(text, DONT_KNOW, "what it doesn't know"))
        else:
            forms, reply = SMALL_TALK[rng.choice(list(SMALL_TALK))]
            said.append(Turn(plain(rng.choice(forms), rng), reply, "small talk"))
    if just is None and rng.random() < 0.2:  # and they tell it something about themselves, or teach it something
        teaching = rng.random() < 0.35
        fact_said = a_lesson(rng)[0] if teaching else a_fact(rng)[0]
        if rng.random() < 0.25:
            first = fact_said.split()[0]
            fact_said = rng.choice(("No, ", "Actually, ", "No no, ", "Sorry, ")) + (
                fact_said if first in PROPER or first in PEOPLE else fact_said[0].lower() + fact_said[1:]
            )
        text = plain(fact_said, rng)
        just = lesson(text) if teaching else statement(text)
        if just:
            said.append(Turn(text, f"Okay, I'll remember that {just}.", "being told"))
    thought = recall(known, [t.said for t in said if t.said], person, told, (), just, extra)
    return thought, said


def dialog(moment: dict, rng: random.Random, turns: int | None = None) -> list[tuple[str | None, str]]:
    """What's said in a conversation at one moment: (what the person said, or None when it speaks up; what Haven
    says)."""
    return [(t.said, t.answer) for t in conversation(moment, rng, turns)[1]]
