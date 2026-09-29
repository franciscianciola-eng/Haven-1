"""The little book Haven is born having read: a few plain sentences about each of some everyday topics.

Each entry is the start of an encyclopedia article in simple words, and some questions
that one of its sentences answers. Haven learns to tell what it read by practising on
these; it can also recall them in conversation, alongside whatever it reads later.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Entry:
    topic: str  # what people call it when they ask
    title: str  # the article it's read in
    sentences: tuple[str, ...]
    questions: tuple[tuple[str, int], ...] = ()  # (a question, the sentence that answers it)


BOOK = (
    Entry(
        "the moon",
        "Moon",
        (
            "The Moon is the Earth's only natural satellite.",
            "It goes around the Earth about once a month.",
            "People first walked on the Moon in 1969.",
        ),
        (("When did people first walk on the moon?", 2), ("How long does the moon take to go around the Earth?", 1)),
    ),
    Entry(
        "the sun",
        "Sun",
        (
            "The Sun is the star at the center of the Solar System.",
            "It is a huge ball of hot gas that gives the Earth light and warmth.",
            "Light from the Sun takes about eight minutes to reach the Earth.",
        ),
        (("How long does light from the sun take to reach us?", 2),),
    ),
    Entry(
        "the ocean",
        "Ocean",
        (
            "An ocean is a very large body of salt water.",
            "Oceans cover about seventy percent of the Earth's surface.",
            "The Pacific Ocean is the largest and deepest ocean.",
        ),
        (("What is the biggest ocean?", 2), ("How much of the Earth is covered by oceans?", 1)),
    ),
    Entry(
        "Paris",
        "Paris",
        (
            "Paris is the capital city of France.",
            "The river Seine flows through it.",
            "The Eiffel Tower in Paris was built in 1889.",
        ),
        (("What river flows through Paris?", 1), ("When was the Eiffel Tower built?", 2)),
    ),
    Entry(
        "France",
        "France",
        (
            "France is a country in Western Europe.",
            "Its capital city is Paris.",
            "People in France speak French.",
        ),
        (("What is the capital of France?", 1), ("What language do people speak in France?", 2)),
    ),
    Entry(
        "Japan",
        "Japan",
        (
            "Japan is an island country in East Asia.",
            "Its capital city is Tokyo.",
            "Mount Fuji is the tallest mountain in Japan.",
        ),
        (("What is the capital of Japan?", 1), ("What is the tallest mountain in Japan?", 2)),
    ),
    Entry(
        "dinosaurs",
        "Dinosaur",
        (
            "Dinosaurs are a group of reptiles that lived on Earth for millions of years.",
            "Most of them died out about 66 million years ago.",
            "Birds are the living relatives of dinosaurs.",
        ),
        (("When did the dinosaurs die out?", 1), ("Are birds related to dinosaurs?", 2)),
    ),
    Entry(
        "elephants",
        "Elephant",
        (
            "Elephants are the largest land animals alive today.",
            "They use their long trunks to smell, drink and pick things up.",
            "Elephants live in Africa and Asia.",
        ),
        (("What do elephants use their trunks for?", 1), ("Where do elephants live?", 2)),
    ),
    Entry(
        "whales",
        "Whale",
        (
            "Whales are very large mammals that live in the ocean.",
            "They breathe air through a blowhole on top of their heads.",
            "The blue whale is the biggest animal that has ever lived.",
        ),
        (("What is the biggest animal?", 2), ("How do whales breathe?", 1)),
    ),
    Entry(
        "music",
        "Music",
        (
            "Music is a form of art that uses sounds organized in time.",
            "People make music by singing and by playing instruments.",
            "Music has rhythm, melody and harmony.",
        ),
        (("How do people make music?", 1),),
    ),
    Entry(
        "the internet",
        "Internet",
        (
            "The Internet is a network that connects computers all over the world.",
            "People use it to send messages, share pictures and read the news.",
            "The World Wide Web is one of the things that runs on the Internet.",
        ),
        (("What do people use the internet for?", 1),),
    ),
    Entry(
        "computers",
        "Computer",
        (
            "A computer is a machine that can be programmed to carry out tasks.",
            "Computers work with numbers made of ones and zeros.",
            "The first electronic computers were built in the 1940s.",
        ),
        (("When were the first computers built?", 2),),
    ),
    Entry(
        "cars",
        "Car",
        (
            "A car is a vehicle with four wheels that people use to travel on roads.",
            "Most cars have an engine that burns petrol, and some run on electricity.",
            "The first cars were made in the 1880s.",
        ),
        (("When were the first cars made?", 2), ("What makes cars go?", 1)),
    ),
    Entry(
        "trains",
        "Train",
        (
            "A train is a row of connected vehicles that runs on railway tracks.",
            "Trains carry people and goods from place to place.",
            "The first trains were pulled by steam engines.",
        ),
        (("What pulled the first trains?", 2),),
    ),
    Entry(
        "airplanes",
        "Airplane",
        (
            "An airplane is a vehicle with wings that flies through the air.",
            "Its wings are shaped so that moving air lifts it up.",
            "The Wright brothers flew the first airplane in 1903.",
        ),
        (("Who flew the first airplane?", 2), ("How do airplanes fly?", 1)),
    ),
    Entry(
        "the weather",
        "Weather",
        (
            "Weather is how hot or cold, wet or dry, and windy or calm the air is.",
            "Weather can change from day to day.",
            "People who study the weather are called meteorologists.",
        ),
        (("What are people who study the weather called?", 2),),
    ),
    Entry(
        "snow",
        "Snow",
        (
            "Snow is frozen water that falls from clouds as small white flakes.",
            "It falls when the air is cold enough to freeze the water.",
            "No two snowflakes look exactly alike.",
        ),
        (("When does snow fall?", 1),),
    ),
    Entry(
        "rain",
        "Rain",
        (
            "Rain is water that falls from clouds in drops.",
            "Rain gives plants and animals the water they need.",
            "Water rises from the sea into the sky and falls again as rain.",
        ),
        (("Why is rain important?", 1),),
    ),
    Entry(
        "volcanoes",
        "Volcano",
        (
            "A volcano is an opening in the ground where hot melted rock comes out.",
            "The melted rock is called lava when it reaches the surface.",
            "Some volcanoes have not erupted for thousands of years.",
        ),
        (("What is lava?", 1),),
    ),
    Entry(
        "mountains",
        "Mountain",
        (
            "A mountain is a large area of land that rises high above the land around it.",
            "Mount Everest is the highest mountain on Earth.",
            "It is colder at the top of a mountain than at the bottom.",
        ),
        (("What is the highest mountain?", 1), ("Is it cold at the top of a mountain?", 2)),
    ),
    Entry(
        "the stars",
        "Star",
        (
            "A star is a huge ball of hot gas that gives off light and heat.",
            "The Sun is the star closest to the Earth.",
            "At night we can see thousands of stars in the sky.",
        ),
        (("What is the closest star?", 1),),
    ),
    Entry(
        "Mars",
        "Mars",
        (
            "Mars is the fourth planet from the Sun, and it is often called the Red Planet.",
            "It looks red because of rust in its soil.",
            "Mars has two small moons.",
        ),
        (("Why is Mars red?", 1), ("How many moons does Mars have?", 2)),
    ),
    Entry(
        "Jupiter",
        "Jupiter",
        (
            "Jupiter is the largest planet in the Solar System.",
            "It is made mostly of gas.",
            "Jupiter has a giant storm called the Great Red Spot.",
        ),
        (("What is the biggest planet?", 0), ("What is the Great Red Spot?", 2)),
    ),
    Entry(
        "gravity",
        "Gravity",
        (
            "Gravity is the force that pulls things toward each other and makes things fall.",
            "It keeps the Moon going around the Earth.",
            "Isaac Newton wrote about gravity more than three hundred years ago.",
        ),
        (("Why do things fall?", 0), ("Who wrote about gravity?", 2)),
    ),
    Entry(
        "electricity",
        "Electricity",
        (
            "Electricity is a form of energy carried by tiny charged particles.",
            "It flows through wires to make lights, machines and computers work.",
            "Lightning is electricity in the sky.",
        ),
        (("What is lightning?", 2),),
    ),
    Entry(
        "money",
        "Money",
        (
            "Money is something people use to pay for the things they buy.",
            "It can be coins, paper notes or numbers in a bank.",
            "Before money, people traded things with each other.",
        ),
        (("What did people do before money?", 2),),
    ),
    Entry(
        "school",
        "School",
        (
            "A school is a place where children go to learn.",
            "Teachers help them learn to read, write and count.",
            "In many countries children must go to school.",
        ),
        (("What do children learn at school?", 1),),
    ),
    Entry(
        "football",
        "Football",
        (
            "Football is a team sport in which players try to get a ball into a goal.",
            "Each team has eleven players on the field.",
            "Only the goalkeeper may touch the ball with their hands.",
        ),
        (("How many players are on a football team?", 1),),
    ),
    Entry(
        "chess",
        "Chess",
        (
            "Chess is a board game for two players, played on a board with 64 squares.",
            "Each player starts with sixteen pieces.",
            "The game is won by trapping the other player's king.",
        ),
        (("How do you win at chess?", 2), ("How many pieces does each chess player have?", 1)),
    ),
    Entry(
        "the piano",
        "Piano",
        (
            "A piano is a musical instrument that is played by pressing keys.",
            "Most pianos have 88 black and white keys.",
            "When a key is pressed, a small hammer hits a string inside.",
        ),
        (("How many keys does a piano have?", 1),),
    ),
    Entry(
        "pizza",
        "Pizza",
        (
            "Pizza is a flat round bread baked with toppings such as tomato and cheese.",
            "It comes from Italy.",
            "Pizza is usually baked in a very hot oven.",
        ),
        (("Where does pizza come from?", 1),),
    ),
    Entry(
        "chocolate",
        "Chocolate",
        (
            "Chocolate is a sweet food made from the seeds of the cacao tree.",
            "The seeds are called cocoa beans.",
            "People in Central America were drinking chocolate thousands of years ago.",
        ),
        (("What is chocolate made from?", 0),),
    ),
    Entry(
        "coffee",
        "Coffee",
        (
            "Coffee is a drink made from the roasted seeds of the coffee plant.",
            "It has caffeine in it, which helps people feel awake.",
            "Coffee plants first grew in Ethiopia.",
        ),
        (("Why does coffee keep people awake?", 1), ("Where did coffee come from?", 2)),
    ),
    Entry(
        "the president",
        "President",
        (
            "A president is the leader of a country or of an organization.",
            "In many countries the people choose their president by voting.",
            "A president usually leads for a set number of years.",
        ),
        (("How is a president chosen?", 1),),
    ),
    Entry(
        "history",
        "History",
        (
            "History is the study of the past.",
            "People who study history are called historians.",
            "Historians learn about the past from old writings and objects.",
        ),
        (("What are people who study history called?", 1),),
    ),
    Entry(
        "science",
        "Science",
        (
            "Science is a way of finding out how the world works by watching and testing.",
            "Scientists make guesses and test them with experiments.",
            "Physics, chemistry and biology are kinds of science.",
        ),
        (("What do scientists do?", 1),),
    ),
    Entry(
        "math",
        "Mathematics",
        (
            "Mathematics is the study of numbers, shapes and patterns.",
            "People use it to count, measure and build things.",
            "Adding, taking away, multiplying and dividing are parts of mathematics.",
        ),
        (("What do people use math for?", 1),),
    ),
    Entry(
        "photosynthesis",
        "Photosynthesis",
        (
            "Photosynthesis is how plants use sunlight to make food from water and air.",
            "It happens in the green parts of plants.",
            "Photosynthesis puts oxygen into the air.",
        ),
        (("How do plants make food?", 0),),
    ),
    Entry(
        "atoms",
        "Atom",
        (
            "An atom is the smallest part of a chemical element.",
            "Everything around us is made of atoms.",
            "Atoms are much too small to see.",
        ),
        (("What are things made of?", 1),),
    ),
    Entry(
        "the brain",
        "Brain",
        (
            "The brain is the organ that controls the body and lets animals think.",
            "It is inside the head, protected by the skull.",
            "The brain sends messages to the body through nerves.",
        ),
        (("Where is the brain?", 1),),
    ),
    Entry(
        "love",
        "Love",
        (
            "Love is a strong feeling of caring about someone or something.",
            "People can love their family, their friends and their pets.",
            "Many songs and stories are about love.",
        ),
    ),
    Entry(
        "friendship",
        "Friendship",
        (
            "Friendship is a close bond between people who care about each other.",
            "Friends help each other and enjoy spending time together.",
            "Some friendships last for a whole life.",
        ),
    ),
    Entry(
        "the city",
        "City",
        (
            "A city is a large place where many people live and work.",
            "Cities have many buildings, roads and shops.",
            "Tokyo is one of the biggest cities in the world.",
        ),
    ),
    Entry(
        "London",
        "London",
        (
            "London is the capital city of England and the United Kingdom.",
            "The river Thames flows through it.",
            "Big Ben is a famous clock tower in London.",
        ),
        (("What river flows through London?", 1), ("What is Big Ben?", 2)),
    ),
    Entry(
        "New York",
        "New York City",
        (
            "New York City is the largest city in the United States.",
            "The Statue of Liberty stands in its harbor.",
            "It is sometimes called the Big Apple.",
        ),
        (("Where is the Statue of Liberty?", 1),),
    ),
    Entry(
        "China",
        "China",
        (
            "China is a large country in East Asia.",
            "Its capital city is Beijing.",
            "The Great Wall of China is thousands of kilometers long.",
        ),
        (("What is the capital of China?", 1),),
    ),
    Entry(
        "Egypt",
        "Egypt",
        (
            "Egypt is a country in North Africa, known for its ancient pyramids.",
            "Its capital city is Cairo.",
            "The river Nile flows through Egypt.",
        ),
        (("What is the capital of Egypt?", 1), ("What river flows through Egypt?", 2)),
    ),
    Entry(
        "the pyramids",
        "Egyptian pyramids",
        (
            "The Egyptian pyramids are ancient stone tombs built for kings.",
            "The biggest is the Great Pyramid of Giza.",
            "They were built more than four thousand years ago.",
        ),
        (("How old are the pyramids?", 2),),
    ),
    Entry(
        "robots",
        "Robot",
        (
            "A robot is a machine that can do tasks by itself.",
            "Robots are used in factories to build things like cars.",
            "Some robots are sent to explore other planets.",
        ),
        (("What are robots used for?", 1),),
    ),
    Entry(
        "phones",
        "Telephone",
        (
            "A telephone is a device that lets people talk to each other from far away.",
            "Alexander Graham Bell was one of the first people to make one.",
            "Many people now carry small phones in their pockets.",
        ),
        (("Who invented the telephone?", 1),),
    ),
    Entry(
        "books",
        "Book",
        (
            "A book is a set of pages with writing on them, held together with a cover.",
            "People read books to learn and to enjoy stories.",
            "A place where people borrow books is called a library.",
        ),
        (("What is a library?", 2),),
    ),
    Entry(
        "painting",
        "Painting",
        (
            "Painting is the art of putting paint on a surface to make a picture.",
            "The Mona Lisa is a famous painting by Leonardo da Vinci.",
            "People who paint are called painters.",
        ),
        (("Who painted the Mona Lisa?", 1),),
    ),
    Entry(
        "Shakespeare",
        "William Shakespeare",
        (
            "William Shakespeare was an English writer famous for his plays.",
            "He wrote Romeo and Juliet and Hamlet.",
            "He lived about four hundred years ago.",
        ),
        (("Who wrote Romeo and Juliet?", 1),),
    ),
    Entry(
        "Einstein",
        "Albert Einstein",
        (
            "Albert Einstein was a scientist who came up with the theory of relativity.",
            "He was born in Germany in 1879.",
            "He won the Nobel Prize in Physics in 1921.",
        ),
        (("Where was Einstein born?", 1),),
    ),
    Entry(
        "cats",
        "Cat",
        (
            "Cats are small furry animals that many people keep as pets.",
            "They are good at hunting mice.",
            "Cats sleep for much of the day.",
        ),
        (("Do cats sleep a lot?", 2),),
    ),
    Entry(
        "dogs",
        "Dog",
        (
            "Dogs are animals that many people keep as pets, known for being loyal.",
            "They have a very good sense of smell.",
            "Dogs came from wolves long ago.",
        ),
        (("Where did dogs come from?", 2),),
    ),
    Entry(
        "horses",
        "Horse",
        (
            "Horses are large animals with hooves that people have ridden for thousands of years.",
            "A young horse is called a foal.",
            "Horses can sleep standing up.",
        ),
        (("What is a baby horse called?", 1),),
    ),
    Entry(
        "birds",
        "Bird",
        (
            "Birds are animals with feathers and wings, and most of them can fly.",
            "They lay eggs, and many build nests for them.",
            "Penguins and ostriches are birds that cannot fly.",
        ),
        (("Which birds cannot fly?", 2),),
    ),
    Entry(
        "fish",
        "Fish",
        (
            "Fish are animals that live in water and breathe with gills.",
            "Most fish are covered in scales.",
            "Some fish live in the sea and some in rivers and lakes.",
        ),
        (("How do fish breathe?", 0),),
    ),
    Entry(
        "the sea",
        "Sea",
        (
            "A sea is a large body of salt water.",
            "Seas are smaller than oceans and are often partly closed in by land.",
            "The Mediterranean Sea lies between Europe and Africa.",
        ),
    ),
    Entry(
        "rivers",
        "River",
        (
            "A river is a large stream of fresh water that flows across the land.",
            "Rivers usually flow into a sea, a lake or another river.",
            "The Nile and the Amazon are the longest rivers on Earth.",
        ),
        (("What is the longest river?", 2),),
    ),
    Entry(
        "forests",
        "Forest",
        (
            "A forest is a large area of land covered with trees.",
            "Forests are home to many kinds of animals and plants.",
            "The Amazon rainforest is the largest rainforest on Earth.",
        ),
    ),
    Entry(
        "deserts",
        "Desert",
        (
            "A desert is a very dry place where very little rain falls.",
            "The Sahara is the largest hot desert.",
            "Camels can live in deserts because they need little water.",
        ),
        (("What is the biggest desert?", 1),),
    ),
    Entry(
        "winter",
        "Winter",
        (
            "Winter is the coldest season of the year.",
            "The days are short and the nights are long.",
            "In some places it snows in winter.",
        ),
    ),
    Entry(
        "summer",
        "Summer",
        (
            "Summer is the warmest season of the year.",
            "The days are long and the nights are short.",
            "Many people go on holiday in the summer.",
        ),
    ),
    Entry(
        "the Earth",
        "Earth",
        (
            "The Earth is the planet we live on, the third planet from the Sun.",
            "It takes a year to go around the Sun.",
            "It spins around once a day, which gives us day and night.",
        ),
        (("Why is there day and night?", 2), ("How long does the Earth take to go around the sun?", 1)),
    ),
    Entry(
        "bees",
        "Bee",
        (
            "Bees are flying insects that make honey and help flowers make seeds.",
            "They live together in big groups called colonies.",
            "Each colony has a queen bee.",
        ),
        (("How do bees help flowers?", 0),),
    ),
    Entry(
        "rainbows",
        "Rainbow",
        (
            "A rainbow is an arc of colors in the sky, made when sunlight shines through rain.",
            "Its colors are red, orange, yellow, green, blue, indigo and violet.",
            "You can see a rainbow when the Sun is behind you and rain is in front of you.",
        ),
        (("What colors are in a rainbow?", 1),),
    ),
    Entry(
        "clouds",
        "Cloud",
        (
            "A cloud is made of tiny drops of water or ice floating in the sky.",
            "Dark gray clouds often bring rain.",
            "Clouds are moved around by the wind.",
        ),
    ),
    Entry(
        "ice",
        "Ice",
        (
            "Ice is water that has frozen solid.",
            "Water turns to ice when it gets colder than zero degrees Celsius.",
            "Ice floats on water.",
        ),
        (("When does water freeze?", 1), ("Does ice float?", 2)),
    ),
    Entry(
        "planets",
        "Planet",
        (
            "A planet is a large round object in space that goes around a star.",
            "There are eight planets in the Solar System.",
            "The Earth is the only planet known to have life.",
        ),
        (("How many planets are there?", 1),),
    ),
    Entry(
        "spiders",
        "Spider",
        (
            "Spiders are small animals with eight legs, and many of them spin webs.",
            "They catch insects in their webs to eat.",
            "Spiders are not insects, because insects have six legs.",
        ),
        (("How many legs does a spider have?", 0),),
    ),
    Entry(
        "the heart",
        "Heart",
        (
            "The heart is the organ that pumps blood around the body.",
            "It beats about seventy times a minute when you are resting.",
            "The heart is a muscle.",
        ),
    ),
    Entry(
        "bread",
        "Bread",
        (
            "Bread is a food made by baking dough of flour and water.",
            "Yeast makes most bread rise and become soft.",
            "People have been making bread for thousands of years.",
        ),
        (("What makes bread rise?", 1),),
    ),
    Entry(
        "the guitar",
        "Guitar",
        (
            "A guitar is a musical instrument with strings that are plucked or strummed.",
            "Most guitars have six strings.",
            "Electric guitars need an amplifier to be heard.",
        ),
        (("How many strings does a guitar have?", 1),),
    ),
    Entry(
        "Italy",
        "Italy",
        (
            "Italy is a country in southern Europe, shaped like a boot.",
            "Its capital city is Rome.",
            "Pizza and pasta come from Italy.",
        ),
        (("What is the capital of Italy?", 1),),
    ),
    Entry(
        "India",
        "India",
        (
            "India is a large country in South Asia.",
            "Its capital city is New Delhi.",
            "More people live in India than in any other country.",
        ),
        (("What is the capital of India?", 1),),
    ),
    Entry(
        "Brazil",
        "Brazil",
        (
            "Brazil is the largest country in South America.",
            "People in Brazil speak Portuguese.",
            "Most of the Amazon rainforest is in Brazil.",
        ),
        (("What language do people speak in Brazil?", 1),),
    ),
    Entry(
        "Canada",
        "Canada",
        (
            "Canada is a large country in the north of North America.",
            "Its capital city is Ottawa.",
            "People in Canada speak English and French.",
        ),
        (("What is the capital of Canada?", 1),),
    ),
    Entry(
        "Australia",
        "Australia",
        (
            "Australia is a country that is also a continent, in the southern half of the world.",
            "Its capital city is Canberra.",
            "Kangaroos and koalas live in Australia.",
        ),
        (("What is the capital of Australia?", 1), ("Where do kangaroos live?", 2)),
    ),
    Entry(
        "Tokyo",
        "Tokyo",
        (
            "Tokyo is the capital city of Japan.",
            "More than thirteen million people live there.",
            "Tokyo was once called Edo.",
        ),
    ),
    Entry(
        "Rome",
        "Rome",
        (
            "Rome is the capital city of Italy.",
            "It was the center of the Roman Empire.",
            "The Colosseum in Rome is almost two thousand years old.",
        ),
        (("How old is the Colosseum?", 2),),
    ),
    Entry(
        "penguins",
        "Penguin",
        (
            "Penguins are birds that cannot fly and live mostly in the southern half of the world.",
            "They use their wings to swim fast under water.",
            "Many penguins live in cold Antarctica.",
        ),
        (("Can penguins fly?", 0),),
    ),
    Entry(
        "tigers",
        "Tiger",
        (
            "Tigers are the largest wild cats, with orange fur and black stripes.",
            "They live in forests in Asia.",
            "Every tiger has its own pattern of stripes.",
        ),
        (("Where do tigers live?", 1),),
    ),
    Entry(
        "the violin",
        "Violin",
        (
            "A violin is a small musical instrument with four strings, played with a bow.",
            "It is held under the chin.",
            "Violins are played in orchestras.",
        ),
        (("How many strings does a violin have?", 0),),
    ),
)
