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
from ..world import BELL, FIRE, NEST, SAND, THORN, TREE, WATER

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
    "place": ("Where are you?", "Where are you right now?", "Where are you now?", "Where are you in your valley?"),
    "home": ("Where do you live?", "Where is your home?", "Where do you sleep?"),
    "world": (
        "Tell me about your world.",
        "What's your world like?",
        "Describe your world.",
        "What is your valley like?",
        "What's in your valley?",
        "What have you found in your valley?",
    ),
    "words": ("What words do you know?", "Do you know any words?", "What words have you learned?"),
    "talk": ("Can you talk?", "Can you speak?", "Do you understand me?", "Can you hear me?"),
    "learned": ("What have you learned?", "What do you know?", "What have you found out?"),
    "eat": ("What is good to eat?", "What do you eat?", "What food do you like?", "What's your favorite food?"),
    "danger": ("What hurts you?", "What's dangerous?", "What should you stay away from?", "What makes you sick?"),
    "remember": ("What do you remember?", "What happened to you?", "Tell me something you remember."),
    "story": (
        "Tell me your story.",
        "Tell me about your life.",
        "What has happened in your life?",
        "What's your life been like?",
    ),
    "day": (
        "What did you do today?",
        "How was your day?",
        "What have you done today?",
        "What happened today?",
        "Did you do anything fun today?",
    ),
    "likes": (
        "What do you like?",
        "What do you like to do?",
        "What's your favorite thing?",
        "What do you enjoy?",
        "What do you like doing?",
        "What don't you like?",
        "Is there anything you don't like?",
    ),
    "play": ("Do you want to play?", "Let's play!", "Want to play?", "Shall we play?", "Do you like to play?"),
    "can": ("What can you do?", "What are you able to do?", "What can you do in your valley?"),
    "alive": ("Are you alive?", "Do you think you're alive?", "Are you real?", "Are you conscious?"),
    "feelings": ("Do you have feelings?", "Can you feel things?", "Do you feel anything?"),
    "what": ("What are you?", "What kind of thing are you?", "Tell me about yourself.", "Describe yourself."),
    "maker": ("Who made you?", "Who created you?", "Where did you come from?", "Who are your parents?"),
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
# (what people call it, the article it's read in, that article's first sentence)
READINGS = (
    ("the moon", "Moon", "The Moon is the Earth's only natural satellite."),
    ("the sun", "Sun", "The Sun is the star at the center of the Solar System."),
    ("the ocean", "Ocean", "An ocean is a very large body of salt water."),
    ("Paris", "Paris", "Paris is the capital city of France."),
    ("France", "France", "France is a country in Western Europe."),
    ("Japan", "Japan", "Japan is an island country in East Asia."),
    ("dinosaurs", "Dinosaur", "Dinosaurs are a group of reptiles that lived on Earth for millions of years."),
    ("elephants", "Elephant", "Elephants are the largest land animals alive today."),
    ("whales", "Whale", "Whales are very large mammals that live in the ocean."),
    ("music", "Music", "Music is a form of art that uses sounds organized in time."),
    ("the internet", "Internet", "The Internet is a network that connects computers all over the world."),
    ("computers", "Computer", "A computer is a machine that can be programmed to carry out tasks."),
    ("cars", "Car", "A car is a vehicle with four wheels that people use to travel on roads."),
    ("trains", "Train", "A train is a row of connected vehicles that runs on railway tracks."),
    ("airplanes", "Airplane", "An airplane is a vehicle with wings that flies through the air."),
    ("the weather", "Weather", "Weather is how hot or cold, wet or dry, and windy or calm the air is."),
    ("snow", "Snow", "Snow is frozen water that falls from clouds as small white flakes."),
    ("rain", "Rain", "Rain is water that falls from clouds in drops."),
    ("volcanoes", "Volcano", "A volcano is an opening in the ground where hot melted rock comes out."),
    ("mountains", "Mountain", "A mountain is a large area of land that rises high above the land around it."),
    ("the stars", "Star", "A star is a huge ball of hot gas that gives off light and heat."),
    ("Mars", "Mars", "Mars is the fourth planet from the Sun, and it is often called the Red Planet."),
    ("Jupiter", "Jupiter", "Jupiter is the largest planet in the Solar System."),
    ("gravity", "Gravity", "Gravity is the force that pulls things toward each other and makes things fall."),
    ("electricity", "Electricity", "Electricity is a form of energy carried by tiny charged particles."),
    ("money", "Money", "Money is something people use to pay for the things they buy."),
    ("school", "School", "A school is a place where children go to learn."),
    ("football", "Football", "Football is a team sport in which players try to get a ball into a goal."),
    ("chess", "Chess", "Chess is a board game for two players, played on a board with 64 squares."),
    ("the piano", "Piano", "A piano is a musical instrument that is played by pressing keys."),
    ("pizza", "Pizza", "Pizza is a flat round bread baked with toppings such as tomato and cheese."),
    ("chocolate", "Chocolate", "Chocolate is a sweet food made from the seeds of the cacao tree."),
    ("coffee", "Coffee", "Coffee is a drink made from the roasted seeds of the coffee plant."),
    ("the president", "President", "A president is the leader of a country or of an organization."),
    ("history", "History", "History is the study of the past."),
    ("science", "Science", "Science is a way of finding out how the world works by watching and testing."),
    ("math", "Mathematics", "Mathematics is the study of numbers, shapes and patterns."),
    ("photosynthesis", "Photosynthesis", "Photosynthesis is how plants use sunlight to make food from water and air."),
    ("atoms", "Atom", "An atom is the smallest part of a chemical element."),
    ("the brain", "Brain", "The brain is the organ that controls the body and lets animals think."),
    ("love", "Love", "Love is a strong feeling of caring about someone or something."),
    ("friendship", "Friendship", "Friendship is a close bond between people who care about each other."),
    ("the city", "City", "A city is a large place where many people live and work."),
    ("London", "London", "London is the capital city of England and the United Kingdom."),
    ("New York", "New York City", "New York City is the largest city in the United States."),
    ("China", "China", "China is a large country in East Asia."),
    ("Egypt", "Egypt", "Egypt is a country in North Africa, known for its ancient pyramids."),
    ("the pyramids", "Egyptian pyramids", "The Egyptian pyramids are ancient stone tombs built for kings."),
    ("robots", "Robot", "A robot is a machine that can do tasks by itself."),
    ("phones", "Telephone", "A telephone is a device that lets people talk to each other from far away."),
    ("books", "Book", "A book is a set of pages with writing on them, held together with a cover."),
    ("painting", "Painting", "Painting is the art of putting paint on a surface to make a picture."),
    ("Shakespeare", "William Shakespeare", "William Shakespeare was an English writer famous for his plays."),
    ("Einstein", "Albert Einstein", "Albert Einstein was a scientist who came up with the theory of relativity."),
    ("cats", "Cat", "Cats are small furry animals that many people keep as pets."),
    ("dogs", "Dog", "Dogs are animals that many people keep as pets, known for being loyal."),
    ("horses", "Horse", "Horses are large animals with hooves that people have ridden for thousands of years."),
    ("birds", "Bird", "Birds are animals with feathers and wings, and most of them can fly."),
    ("fish", "Fish", "Fish are animals that live in water and breathe with gills."),
    ("the sea", "Sea", "A sea is a large body of salt water."),
    ("rivers", "River", "A river is a large stream of fresh water that flows across the land."),
    ("forests", "Forest", "A forest is a large area of land covered with trees."),
    ("deserts", "Desert", "A desert is a very dry place where very little rain falls."),
    ("winter", "Winter", "Winter is the coldest season of the year."),
    ("summer", "Summer", "Summer is the warmest season of the year."),
    ("the Earth", "Earth", "The Earth is the planet we live on, the third planet from the Sun."),
    ("bees", "Bee", "Bees are flying insects that make honey and help flowers make seeds."),
    ("rainbows", "Rainbow", "A rainbow is an arc of colors in the sky, made when sunlight shines through rain."),
    ("clouds", "Cloud", "A cloud is made of tiny drops of water or ice floating in the sky."),
    ("ice", "Ice", "Ice is water that has frozen solid."),
    ("planets", "Planet", "A planet is a large round object in space that goes around a star."),
    ("spiders", "Spider", "Spiders are small animals with eight legs, and many of them spin webs."),
    ("the heart", "Heart", "The heart is the organ that pumps blood around the body."),
    ("bread", "Bread", "Bread is a food made by baking dough of flour and water."),
    ("the guitar", "Guitar", "A guitar is a musical instrument with strings that are plucked or strummed."),
    ("Italy", "Italy", "Italy is a country in southern Europe, shaped like a boot."),
    ("India", "India", "India is a large country in South Asia."),
    ("Brazil", "Brazil", "Brazil is the largest country in South America."),
    ("Canada", "Canada", "Canada is a large country in the north of North America."),
    ("Australia", "Australia", "Australia is a country that is also a continent, in the southern half of the world."),
    ("Tokyo", "Tokyo", "Tokyo is the capital city of Japan."),
    ("Rome", "Rome", "Rome is the capital city of Italy."),
    ("penguins", "Penguin", "Penguins are birds that cannot fly and live mostly in the southern half of the world."),
    ("tigers", "Tiger", "Tigers are the largest wild cats, with orange fur and black stripes."),
    ("the violin", "Violin", "A violin is a small musical instrument with four strings, played with a bow."),
)
UNKNOWN_TOPICS = tuple(topic for topic, _, _ in READINGS)
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
    days = int(mind.age // 1200)
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


def first_person(milestone: str, name: str) -> str:
    text = milestone.replace(f"{name} came into the world", "came into the world").replace("it knows", "I know")
    text = text.replace(" its ", " my ").replace(": of ", ": ").replace("itself", "myself")
    text = text.replace("made it sick", "made me sick")
    return f"I {text}."


def memory(mind: Mind) -> str:
    stones = mind.me.milestones
    return first_person(stones[-1][1], mind.me.name) if stones else ""


ROUTINE = ("noticed a new kind of thing", "heard a word for the first time", "realized two kinds")


def story(mind: Mind) -> str:
    """Its life in a few sentences: how it began, and the latest of its firsts that matter most."""
    stones = mind.me.milestones
    if not stones:
        return ""
    rest = stones[1:]
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
        "today": today(mind),
        "likes": likes(mind),
        "knowledge": knowledge(mind),
        "words": known_words(mind.lexicon.vocabulary()),
        "memory": memory(mind),
        "story": story(mind),
        "can": abilities(mind),
        "world": world_words(mind),
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


def statement(text: str) -> str | None:
    """Something about themselves the person just told it, as it would say it back ("you like cats"), or None."""
    t = _ADDRESS.sub("", " ".join(text.strip().split()), count=1)
    if not t or "?" in t or introduced(t) or len(t.split()) < 3:
        return None
    if not FACT_START.match(t) or NOT_FACT.match(t):
        return None
    return second_person(t)


STOP = frozenset(
    "what when where which who whom whose why how does did do is are was were be am the a an and or of to in on at "
    "for with you your yours i me my mine it its that this there their they them tell know remember told about "
    "what's whats can could would should please haven any".split()
)
GENERIC = frozenset("favorite name like love have really much best".split())
SYNONYMS = {"named": "name", "called": "name", "call": "name", "loves": "love", "likes": "like", "maths": "math"}
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


# --- what comes to mind -------------------------------------------------------------------------

TOPICS = {
    "story": r"\bstory\b|\byour life\b|\bhappened in your\b|\bborn\b",
    "can": r"\bwhat can you\b|\bable to\b|\bcan you do\b",
    "world": r"\byour world\b|\byour valley\b",
}


def recall(
    known: dict,
    said: str | list[str],
    person: str | None = None,
    told: list[str] | tuple = (),
    read: list[tuple[str, str]] | tuple = (),
    just: str | None = None,
) -> str:
    """What comes to mind before it answers: what it always knows, and what the words bring up."""
    texts = [said] if isinstance(said, str) else list(said)
    parts = [known["me"]]
    if person:
        parts.append(f"I'm talking with {person}.")
    parts += [known["place"], known["today"], known["likes"], known["knowledge"], known["words"], known["memory"]]
    everything = " ".join(texts).lower()
    parts += [known[topic] for topic, pattern in TOPICS.items() if re.search(pattern, everything)]
    for text in texts:
        for name in mentioned(text)[:2]:
            thing = known["things"][name]
            parts.append(thing_note(name, thing["stats"], thing["where"], with_where="where" in text.lower()))
    for text in texts:
        fact = best_fact(text, list(told))
        if fact:
            parts.append(f"You told me that {fact}.")
    if just:
        parts.append(f"You just told me that {just}.")
    for text in texts:
        found = None if about_them(text) else best_reading(text, list(read))
        if found:
            parts.append(f"I read about {found[0]}: {found[1]}")
    return " ".join(dict.fromkeys(p for p in parts if p))


def notes(mind: Mind, text: str = "", read: list[tuple[str, str]] | tuple = (), just: str | None = None) -> str:
    """What comes to mind when someone says something to it, in the middle of its life."""
    told = [fact for _, fact in mind.told if fact != just]  # what they just said isn't a memory yet
    return recall(memo(mind), text, mind.person, told, read, just)


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


def answers(mind: Mind) -> dict[str, str]:
    """What its state and what it knows say the answer to each kind of question is, right now."""
    b = mind.body
    conclusions = mind.me.conclusions()
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
        "alive": conclusions[-1],
        "feelings": "Yes. Some things feel good to me and some feel bad."
        if mind.me.evidence["feeling"] > 0.5
        else "I don't know yet.",
        "what": f"I'm {mind.me.name}. I live in a valley. {conclusions[-1]}",
        "maker": "I don't know who made me. I came into the world in my nest.",
        "hello": "Hi! I'm asleep." if b.asleep else f"Hi! {feeling(mind)}",
        "thanks": "You're welcome.",
        "bye": "Bye! Come back soon.",
    }


# --- learning to talk: conversations at moments of its lives -------------------------------------

PEOPLE = (
    "Sam Alex Maria Francis Anna Leo Mia Noah Emma Liam Olivia Lucas Sofia Jack Ava Ben Chloe Daniel Ella Ethan Grace "
    "Henry Isla James Julia Kai Lily Max Nora Oscar Priya Ravi Rosa Sara Theo Uma Victor Wen Yuki Zoe Amir Bea Carlos "
    "Dev Elif Femi Gus Hana Ines Joon Kofi Lena Mateo Nia Omar Pia Quinn Rafael Sven Tariq Ursula Vera Will Ximena "
    "Yara Zain Ada Bruno Clara Dmitri Esme Finn Greta Hugo Ivy Jonas Kira Luca Mira Nico Otto Petra Rhea Silas Tess "
    "Wyatt Aiko Bodhi Cyrus Dara Enzo Fatima Gio Hiro Isaac Jade Kenji Laila Milo Nadia Orla Paz Remy Sage Tomas"
).split()
SYLLABLES = ("ka", "lo", "mi", "ra", "ne", "ta", "vi", "so", "da", "ju", "be", "ri", "an", "el", "or", "is")
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
KIN = ("sister", "brother", "mom", "dad", "friend", "son", "daughter", "wife", "husband", "grandma", "grandpa")
MONTHS = "January February March April May June July August September October November December".split()
PLAYED = ("the piano", "the guitar", "the violin", "the drums", "football", "tennis", "chess", "basketball")
ABOUT_ME = ("Do you remember what I told you?", "What did I tell you?", "What do you know about me?")
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
    if rng.random() < 0.8:
        return rng.choice(PEOPLE)
    return "".join(rng.choice(SYLLABLES) for _ in range(rng.choice((2, 2, 3)))).capitalize()


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
        pet, name = rng.choice(PETS), person_name(rng)
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
    said: str
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
    return Turn(question, thing_answer(kind, name, thing["stats"], thing["where"]), f"thing {kind}")


# Kinds of questions whose answers are long, word-for-word recollections: more practice with these when set above 0.
EMPHASIS = 0.0
RECALLED = ("day", "world", "words", "story", "can", "learned", "remember", "likes", "danger", "eat")


def conversation(moment: dict, rng: random.Random, turns: int | None = None) -> tuple[str, list[Turn]]:
    """A short conversation at one moment of a life: what comes to mind first, and what's said, turn by turn."""
    known, answer = moment["memo"], moment["answers"]
    person = person_name(rng) if rng.random() < 0.35 else None
    earlier = [a_fact(rng) for _ in range(rng.choice((0, 0, 1, 2, 3)))]
    told = [second_person(said) for said, _ in earlier]
    read = [
        (title, text)
        for _, title, text in rng.sample(READINGS, rng.choice((0, 0, 0, 1, 2) if rng.random() >= EMPHASIS else (1, 2)))
    ]
    said: list[Turn] = []
    if person is None and rng.random() < 0.2:  # they say who they are first
        for _ in range(5):
            person = person_name(rng)
            text = plain(rng.choice(NAME_FORMS).format(person), rng)
            if introduced(text) == person:
                said.append(Turn(text, f"Nice to meet you, {person}!", "their name"))
                break
        else:
            person = None
    for _ in range(turns or rng.choice((1, 1, 2, 3))):
        roll = rng.random()
        if roll < 0.40:
            intent = rng.choice(RECALLED if rng.random() < EMPHASIS else list(QUESTIONS))
            reply = answer[intent]
            if person and intent in ("hello", "bye"):
                reply = reply.replace("Hi!", f"Hi, {person}!").replace("Bye!", f"Bye, {person}!")
            said.append(Turn(casual(rng.choice(QUESTIONS[intent]), rng), reply, intent))
        elif roll < 0.60:
            turn = _thing_turn(moment, rng)
            turn.said = casual(turn.said, rng)
            said.append(turn)
        elif roll < 0.66:
            reply = f"Your name is {person}." if person else "You haven't told me your name yet."
            said.append(Turn(casual(rng.choice(MY_NAME), rng), reply, "their name"))
        elif roll < 0.76:
            if earlier and rng.random() < 0.7:
                question = rng.choice(rng.choice(earlier)[1])
            else:
                question = rng.choice((*a_fact(rng)[1], *ABOUT_ME))
            text = casual(question, rng)
            fact = best_fact(text, told)
            if fact:
                reply = f"You told me that {fact}."
            elif question == ABOUT_ME[2] and person:
                reply = f"Your name is {person}."
            elif question in ABOUT_ME:
                reply = "You haven't told me anything yet."
            else:
                reply = "You haven't told me that."
            said.append(Turn(text, reply, "what they told it"))
        elif roll < 0.88:
            if read and rng.random() < 0.6:
                title = rng.choice(read)[0]
                topic = next(t for t, name, _ in READINGS if name == title)
            else:
                topic = rng.choice(UNKNOWN_TOPICS)
            text = casual(rng.choice(UNKNOWN_FORMS).format(topic), rng)
            found = best_reading(text, read)
            reply = f"I read about {found[0]}. It says: {found[1]}" if found else DONT_KNOW
            said.append(Turn(text, reply, "what it read" if found else "what it doesn't know"))
        elif roll < 0.92:
            text = casual(rng.choice(OTHER_QUESTIONS), rng)
            if not best_reading(text, read) and not best_fact(text, told):
                said.append(Turn(text, DONT_KNOW, "what it doesn't know"))
        else:
            forms, reply = SMALL_TALK[rng.choice(list(SMALL_TALK))]
            said.append(Turn(plain(rng.choice(forms), rng), reply, "small talk"))
    just = None
    if rng.random() < 0.2:  # and they tell it something about themselves
        text = plain(a_fact(rng)[0], rng)
        just = statement(text)
        if just:
            said.append(Turn(text, f"Okay, I'll remember that {just}.", "being told"))
    thought = recall(known, [t.said for t in said], person, told, read, just)
    return thought, said


def dialog(moment: dict, rng: random.Random, turns: int | None = None) -> list[tuple[str, str]]:
    """What's said in a conversation at one moment: (what the person said, what Haven answers)."""
    return [(t.said, t.answer) for t in conversation(moment, rng, turns)[1]]
