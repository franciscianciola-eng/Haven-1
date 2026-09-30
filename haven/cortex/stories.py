"""Stories Haven heard as it grew up, and the ones it hears now: it tells them in its own words.

Its language cortex heard hundreds of children's books while it grew (see hearing.py).
What it keeps in mind of a story is what it's called and how it begins. Asked for a
story, that comes to mind, and its cortex tells how it goes on, from what it learned: it
isn't reading it out, and it may tell it a little differently each time. At home, it hears
a bedtime story now and then, and the latest is the one it has most in mind.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Story:
    title: str
    opening: tuple[str, ...]  # how it begins: what comes to mind
    then: tuple[str, ...] = ()  # how it goes on (as it learned to tell it)
    book: int = 0  # Project Gutenberg's number for the book it's from


STORIES = (  # the beginnings of some stories it heard as it grew up
    Story(
        "The Tale of Peter Rabbit",
        (
            "Once upon a time there were four little Rabbits, and their names were Flopsy, Mopsy, Cotton-tail, and Peter.",
            "They lived with their Mother in a sand-bank, underneath the root of a very big fir-tree.",
        ),
        (
            "'Now my dears,' said old Mrs. Rabbit one morning, 'you may go into the fields or down the lane, but don't go into Mr. McGregor's garden: your Father had an accident there; he was put in a pie by Mrs. McGregor.'",
        ),
        14838,
    ),
    Story(
        "The Tale of Tom Kitten",
        ("Once upon a time there were three little kittens, and their names were Mittens, Tom Kitten, and Moppet.",),
        (
            "They had dear little fur coats of their own; and they tumbled about the doorstep and played in the dust.",
            "But one day their mother -- Mrs. Tabitha Twitchit -- expected friends to tea; so she fetched the kittens indoors, to wash and dress them, before the fine company arrived.",
        ),
        14837,
    ),
    Story(
        "The Tale of Squirrel Nutkin",
        (
            "This is a Tale about a tail -- a tail that belonged to a little red squirrel, and his name was Nutkin.",
            "He had a brother called Twinkleberry, and a great many cousins: they lived in a wood at the edge of a lake.",
        ),
        (
            "In the middle of the lake there is an island covered with trees and nut bushes; and amongst those trees stands a hollow oak-tree, which is the house of an owl who is called Old Brown.",
        ),
        14872,
    ),
    Story(
        "The Tale of Johnny Town-Mouse",
        ("Johnny Town-mouse was born in a cupboard.", "Timmy Willie was born in a garden."),
        (
            "Timmy Willie was a little country mouse who went to town by mistake in a hamper.",
            "The gardener sent vegetables to town once a week by carrier; he packed them in a big hamper.",
        ),
        15284,
    ),
    Story(
        "The Tale of Samuel Whiskers",
        ("Once upon a time there was an old cat, called Mrs. Tabitha Twitchit, who was an anxious parent.",),
        (
            "She used to lose her kittens continually, and whenever they were lost they were always in mischief!",
            "On baking day she determined to shut them up in a cupboard.",
        ),
        15575,
    ),
    Story(
        "The Adventures of Reddy Fox",
        (
            "Reddy Fox lived with Granny Fox.",
            "You see, Reddy was one of a large family, so large that Mother Fox had hard work to feed so many hungry little mouths and so she had let Reddy go to live with old Granny Fox.",
        ),
        (
            "Granny Fox was the wisest, slyest, smartest fox in all the country round, and now that Reddy had grown so big, she thought it about time that he began to learn the things that every fox should know.",
        ),
        1825,
    ),
    Story(
        "Whitefoot the Wood Mouse",
        ("In all his short life Whitefoot the Wood Mouse never had spent such a happy winter.",),
        (
            "Whitefoot is one of those wise little people who never allow unpleasant things of the past to spoil their present happiness, and who never borrow trouble from the future.",
            "Whitefoot believes in getting the most from the present.",
        ),
        4698,
    ),
    Story(
        "Blacky the Crow",
        (
            "Blacky the Crow is always watching for things not intended for his sharp eyes.",
            "The result is that he gets into no end of trouble which he could avoid.",
        ),
        (
            "In this respect he is just like his cousin, Sammy Jay.",
            "Between them they see a great deal with which they have no business and which it would be better for them not to see.",
        ),
        4979,
    ),
    Story(
        "The Adventures of Jerry Muskrat",
        (
            "What was it Mother Muskrat had said about Farmer Brown's boy and his traps?",
            "Jerry Muskrat sat on the edge of the Big Rock and kicked his heels while he tried to remember.",
        ),
        ("The fact is, Jerry had not half heeded.", "He had been thinking of other things."),
        5110,
    ),
    Story(
        "The Tale of Freddie Firefly",
        (
            "Nobody in Pleasant Valley ever paid any attention to Freddie Firefly in the daytime.",
            "But on warm, and especially on dark summer nights he always appeared at his best.",
        ),
        (
            "Then he went gaily flitting through the meadows.",
            "And sometimes he even danced right in Farmer Green's dooryard, together with a hundred or two of his nearest relations.",
        ),
        5727,
    ),
    Story(
        "Mrs. Peter Rabbit",
        (
            "Peter Rabbit had lost his appetite.",
            "Now when Peter Rabbit loses his appetite, something is very wrong indeed with him.",
        ),
        (
            "Peter has boasted that he can eat any time and all the time.",
            "In fact, the two things that Peter thinks most about are his stomach and satisfying his curiosity, and nearly all of the scrapes that Peter has gotten into have been because of those two things.",
        ),
        5791,
    ),
    Story(
        "The Adventures of Johnny Chuck",
        ("All the Green Meadows and all the Green Forest had heard the news.", "Peter Rabbit had seen to that."),
        (
            "And just as soon as each of the little meadow and forest folks heard it, he hurried out to listen for himself and make sure that it was true.",
            'And each, when he heard that sweet voice of Winsome Bluebird, had kicked up his heels and shouted "Hurrah!"',
        ),
        5844,
    ),
    Story(
        "The Tale of Tommy Fox",
        (
            "Tommy Fox was having a delightful time.",
            "If you could have come upon him in the woods you would have been astonished at his antics.",
        ),
        (
            "He leaped high off the ground, and struck out with his paws.",
            "He opened his mouth and thrust his nose out and then clapped his jaws shut again, with a snap.",
        ),
        5955,
    ),
    Story(
        "Happy Jack",
        (
            "Happy Jack Squirrel sat on the tip of one of the highest branches of a big hickory tree.",
            "Happy Jack was up very early that morning.",
        ),
        (
            "In fact, jolly, round, red Mr. Sun was still in his bed behind the Purple Hills when Happy Jack hopped briskly out of bed.",
            "He washed himself thoroughly and was ready for business by the time Mr. Sun began his climb up in the blue, blue sky.",
        ),
        13355,
    ),
    Story(
        "The Adventures of Grandfather Frog",
        (
            "Billy Mink ran around the edge of the Smiling Pool and turned down by the Laughing Brook.",
            "His eyes twinkled with mischief, and he hurried as only Billy can.",
        ),
        ("As he passed Jerry Muskrat's house, Jerry saw him.", '"Hi, Billy Mink!'),
        14375,
    ),
    Story(
        "The Tale of Cuffy Bear",
        (
            "Far up on the side of Blue Mountain lived Cuffy Bear with his father and mother and his little sister Silkie.",
        ),
        ("Mr. Bear's house was quite the finest for many miles around.",),
        15528,
    ),
    Story(
        "The Tale of Frisky Squirrel",
        ("Frisky Squirrel was a lively little chap.", "And he was very bold, too."),
        (
            "You see, he was so nimble that he felt he could always jump right out of danger -- no matter whether it was a hawk chasing him, or a fox springing at him, or a boy throwing stones at him.",
            "He would chatter and scold at his enemies from some tree-top.",
        ),
        18630,
    ),
    Story(
        "The Tale of Henrietta Hen",
        ("Henrietta Hen thought highly of herself.",),
        (
            'Not only did she consider herself a "speckled beauty" (to use her own words) but she had an excellent opinion of her own ways, her own ideas -- even of her own belongings.',
            'When she pulled a fat worm -- or a grub -- out of the ground she did it with an air of pride; and she was almost sure to say, "There!',
        ),
        18652,
    ),
    Story(
        "The Tale of Miss Kitty Cat",
        (
            "The rats and the mice thought that Miss Kitty Cat was a terrible person.",
            "She was altogether too fond of hunting them.",
        ),
        (
            "They agreed, however, that in one way it was pleasant to have her about the farmhouse.",
            "When she washed her face, while sitting on the doorsteps, they knew -- so they said! -- that it was going to rain.",
        ),
        21078,
    ),
    Story(
        "The Tale of Grandfather Mole",
        (
            "There was a queer old person that lived in Farmer Green's garden.",
            "Nobody knew exactly how long he had made his home there because his neighbors seldom saw him.",
        ),
        (
            "He might have been in the garden a whole summer before anybody set eyes on him.",
            "Those that were acquainted with him called him Grandfather Mole.",
        ),
        21203,
    ),
    Story(
        "The Tale of Turkey Proudfoot",
        (
            "All the hen turkeys thought Turkey Proudfoot a wonderful creature.",
            "They said he had the most beautiful tail on the farm.",
        ),
        (
            "When he spread it and strutted about Farmer Green's place the hen turkeys were sure to nudge one another and say, \"Ahem!",
            "Isn't he elegant?\"",
        ),
        21844,
    ),
    Story(
        "The Adventures of Buster Bear",
        (
            "Buster Bear yawned as he lay on his comfortable bed of leaves and watched the first early morning sunbeams creeping through the Green Forest to chase out the Black Shadows.",
        ),
        ("Once more he yawned, and slowly got to his feet and shook himself.",),
        22816,
    ),
    Story(
        "The Tale of Ferdinand Frog",
        ("There was something about Ferdinand Frog that made everybody smile.",),
        (
            "It may have been his amazingly wide mouth and his queer, bulging eyes, or perhaps it was his sprightly manner -- for one never could tell when Mr. Frog would leap into the air, or turn a somersault backward.",
            "Indeed, some of his neighbors claimed that he himself didn't know what he was going to do next -- he was so jumpy.",
        ),
        24590,
    ),
    Story(
        "The Tale of Kiddie Katydid",
        (
            "On warm, dry, midsummer nights the Katydids all made a terrific racket.",
            "But there wasn't one of them that outdid Kiddie.",
        ),
        (
            "He always had the best time when he was making the most noise.",
            "And since he liked to station himself in a tree near Farmer Green's house, his uproar often rose plainly above that of the other Katydids.",
        ),
        24608,
    ),
    Story(
        "The Tale of Jimmy Rabbit",
        (
            "Jimmy Rabbit wanted a new tail.",
            "To be sure, he already had a tail -- but it was so short that he felt it was little better than none at all.",
        ),
        (
            "Frisky Squirrel and Billy Woodchuck had fine, bushy tails; and so had all the other forest-people, except the Rabbit family.",
            "Jimmy had tried his hardest to get a handsome tail for himself.",
        ),
        24628,
    ),
    Story(
        "The Tale of Billy Woodchuck",
        (
            "One day, when Johnnie Green tramped over the fields toward the woods, he did not dream that he walked right over somebody's bedroom.",
            "The snow was deep, for it was midwinter.",
        ),
        (
            "And as Johnnie crossed his father's pasture he thought only of the fresh rabbit tracks that he saw all about him.",
        ),
        25090,
    ),
    Story(
        "The Adventures of Danny Meadow Mouse",
        (
            "Danny Meadow Mouse sat on his door-step with his chin in his hands, and it was very plain to see that Danny had something on his mind.",
        ),
        (
            'He had only a nod for Jimmy Skunk, and even Peter Rabbit could get no more than a grumpy "Good morning."',
            "It wasn't that he had been caught napping the day before by Reddy Fox and nearly made an end of.",
        ),
        25301,
    ),
    Story(
        "The Adventures of Chatterer the Red Squirrel",
        ("Chatterer the Red Squirrel had been scolding because there was no excitement.",),
        (
            "He had even tried to make some excitement by waking Bobby Coon and making him so angry that Bobby had threatened to eat him alive.",
            "It had been great fun to dance around and call Bobby names and make fun of him.",
        ),
        37952,
    ),
    Story(
        "Uncle Wiggily's Automobile",
        ("Once upon a time, a good many years ago, there was an old rabbit gentleman named Uncle Wiggily Longears.",),
        (
            "He was related to Johnnie and Billie Bushytail, the squirrels, as well as being an Uncle to Sammie and Susie Littletail, his rabbit nephew and niece.",
        ),
        60017,
    ),
    Story(
        "The Velveteen Rabbit",
        (
            "There was once a velveteen rabbit, and in the beginning he was really splendid.",
            "He was fat and bunchy, as a rabbit should be; his coat was spotted brown and white, he had real thread whiskers, and his ears were lined with pink sateen.",
        ),
        (
            "On Christmas morning, when he sat wedged in the top of the Boy's stocking, with a sprig of holly between his paws, the effect was charming.",
        ),
        11757,
    ),
    Story(
        "The Aesop for Children",
        (
            "There was once a little Kid whose growing horns made him think he was a grown-up Billy Goat and able to take care of himself.",
        ),
        (
            "So one evening when the flock started home from the pasture and his mother called, the Kid paid no heed and kept right on nibbling the tender grass.",
            "A little later when he lifted his head, the flock was gone.",
        ),
        19994,
    ),
    Story(
        "The Emperor's New Clothes",
        (
            "Many years ago, there was an Emperor, who was so excessively fond of new clothes, that he spent all his money in dress.",
        ),
        (
            "He did not trouble himself in the least about his soldiers; nor did he care to go either to the theatre or the chase, except for the opportunities then afforded him for displaying his new clothes.",
        ),
        1597,
    ),
    Story(
        "The Little Red Hen",
        (
            "One day the Little Red Hen found a Seed.",
            "It was a Wheat Seed, but the Little Red Hen was so accustomed to bugs and worms that she supposed this to be some new and perhaps very delicious kind of meat.",
        ),
        (
            "She bit it gently and found that it resembled a worm in no way whatsoever as to taste although because it was long and slender, a Little Red Hen might easily be fooled by its appearance.",
        ),
        18735,
    ),
    Story(
        "How the Whale Got His Throat",
        ("In the sea, once upon a time, O my Best Beloved, there was a Whale, and he ate fishes.",),
        (
            "He ate the starfish and the garfish, and the crab and the dab, and the plaice and the dace, and the skate and his mate, and the mackereel and the pickereel, and the really truly twirly-whirly eel.",
            "All the fishes he could find in all the sea he ate with his mouth -- so!",
        ),
        2781,
    ),
    Story(
        "Peter Pan",
        (
            "All children, except one, grow up.",
            "They soon know that they will grow up, and the way Wendy knew was this.",
        ),
        (
            "One day when she was two years old she was playing in a garden, and she plucked another flower and ran with it to her mother.",
            'I suppose she must have looked rather delightful, for Mrs. Darling put her hand to her heart and cried, "Oh, why can\'t you remain like this for ever!"',
        ),
        16,
    ),
    Story(
        "The Wonderful Wizard of Oz",
        (
            "Dorothy lived in the midst of the great Kansas prairies, with Uncle Henry, who was a farmer, and Aunt Em, who was the farmer's wife.",
        ),
        ("Their house was small, for the lumber to build it had to be carried by wagon many miles.",),
        55,
    ),
    Story(
        "The Secret Garden",
        (
            "When Mary Lennox was sent to Misselthwaite Manor to live with her uncle everybody said she was the most disagreeable-looking child ever seen.",
            "It was true, too.",
        ),
        (
            "She had a little thin face and a little thin body, thin light hair and a sour expression.",
            "Her hair was yellow, and her face was yellow because she had been born in India and had always been ill in one way or another.",
        ),
        113,
    ),
    Story(
        "Black Beauty",
        (
            "The first place that I can well remember was a large pleasant meadow with a pond of clear water in it.",
            "Some shady trees leaned over it, and rushes and water-lilies grew at the deep end.",
        ),
        (
            "Over the hedge on one side we looked into a plowed field, and on the other we looked over a gate at our master's house, which stood by the roadside; at the top of the meadow was a grove of fir trees, and at the bottom a running brook overhung by a steep bank.",
        ),
        271,
    ),
    Story(
        "The Wind in the Willows",
        ("The Mole had been working very hard all the morning, spring-cleaning his little home.",),
        (
            "First with brooms, then with dusters; then on ladders and steps and chairs, with a brush and a pail of whitewash; till he had dust in his throat and eyes, and splashes of whitewash all over his black fur, and an aching back and weary arms.",
        ),
        289,
    ),
    Story(
        "The Adventures of Pinocchio",
        ("Once upon a time there was a piece of wood.",),
        ("It was not an expensive piece of wood.",),
        500,
    ),
    Story(
        "The Water-Babies",
        ("Once upon a time there was a little chimney-sweep, and his name was Tom.",),
        (
            "That is a short name, and you have heard it before, so you will not have much trouble in remembering it.",
            "He lived in a great town in the North country, where there were plenty of chimneys to sweep, and plenty of money for Tom to earn and his master to spend.",
        ),
        25564,
    ),
    Story(
        "The Bobbsey Twins at the Seashore",
        ("The Bobbsey twins lived at Lakeport, where Mr. Richard Bobbsey had large lumber yards.",),
        (
            "The mother and father were quite young themselves, and so enjoyed the good times that came as naturally as sunshine to the little Bobbseys.",
        ),
        6950,
    ),
    Story(
        "Bunny Brown and His Sister Sue in the Sunny South",
        (
            "Bunny Brown and his sister Sue were in the backyard of their home, making a big man of snow.",
            "There had been quite a storm the day before, and many white flakes had fallen.",
        ),
        (
            "As soon as the storm stopped and the weather grew warm enough, Mrs. Brown let Bunny and Sue go out to play.",
        ),
        20309,
    ),
    Story(
        "The Golden Bird",
        ("A certain king had a beautiful garden, and in the garden stood a tree which bore golden apples.",),
        (
            "These apples were always counted, and about the time when they began to grow ripe it was found that every night one of them was gone.",
            "The king became very angry at this, and ordered the gardener to keep watch all night under the tree.",
        ),
        2591,
    ),
)

_SENTENCE = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"']))\s+(?=[\"'(]?[A-Z])")
_SHORT = re.compile(r"\b(Mr|Mrs|Dr|St|Mt)\.")


def sentences(text: str) -> list[str]:
    """A passage's sentences ("Mr. Rabbit" doesn't end one)."""
    flow = _SHORT.sub(r"\1<dot>", " ".join(text.split()))
    return [s.replace("<dot>", ".") for s in _SENTENCE.split(flow) if s.strip()]


def from_passage(title: str, passage: str, rng: random.Random | None = None, most: int = 70) -> Story | None:
    """A story as it has it in mind from a passage it heard: the first sentence or two, and how it goes on (a sentence
    or two). With `rng`, from the start of some paragraph of it that isn't someone talking."""
    paragraphs = [p for p in passage.split("\n\n") if p.strip()]
    starts = [i for i, p in enumerate(paragraphs) if not p.lstrip().startswith(('"', "'"))] or [0]
    start = rng.choice(starts[:-1] or starts) if rng else 0
    found = [s for s in sentences(" ".join(paragraphs[start:])) if 3 <= len(s.split()) <= 45]
    if len(found) < 2:
        return None
    n = 2 if rng is None or rng.random() < 0.5 else 1
    opening, then = found[:n], found[n : n + 2]
    while len(then) > 1 and sum(len(s.split()) for s in opening + then) > most:
        then = then[:-1]
    return Story(title, tuple(opening), tuple(then)) if then or rng is None else None
