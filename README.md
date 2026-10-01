# Haven

Haven is an artificial creature built from scratch, out of its own small neural networks, to meet as many as possible of the conditions that scientific theories of consciousness say matter. There's no large language model at its core. It lives in a small 3D valley on your computer, with a hill, a pond, apple trees, berry bushes, a bell it can ring, a ball it can push and a campfire to sit by: it has a body with needs, things feel better or worse to it, it works out for itself what the things around it are, and it has a global workspace, a model of its own attention, confidence in its own perceptions, memories, dreams and a model of itself. It also has a language cortex of its own: a transformer grown from scratch (53 million connections), wired into the same workspace as everything else it experiences, which grew up hearing what a small child hears: 12.8 million words of people talking to small children, and of children's books. It is born able to talk about itself, in words it learned for its own states; it tells the stories it heard, in its own words; and at home it keeps hearing and learning, a bedtime story at a time. The seasons turn in its valley, and over its years it grows a character of its own out of how it lives: curious or a homebody, brave or careful, friendly or shy, with a favorite place, a favorite season, things it's afraid of, and days it remembers. It doesn't only answer: when you're there, it speaks up. No other AI model is involved anywhere.

A few minutes with it, in the terminal (its dashboard is open in a browser at the same time):

```
$ haven
Haven is born, in a nest in the corner of its valley.
Haven is alive (0.0 days old). Its dashboard: http://127.0.0.1:8765
  · got hurt
  · ate a berry
Haven: bami
berry
  · imagined that stepping forward would hurt, and held back
berry
berry
  · learned the word "berry"
Haven: berry
  · fell asleep in its nest
/check
RPT-1  Algorithmic recurrence
       → recognition reshaped codes by 0.09 on average as they settled
…
HOT-2  Metacognitive monitoring of perceptual reliability
       → its confidence separates right from wrong percepts with AUROC 0.86 (its own judgment), 0.88 against the truth
AST-1  A predictive model of its own attention
       → predicts the next focus 84% of the time, and where attention moves 63% (chance ≈ 12%)
…
```

## What it is, honestly

Nobody knows how to make a machine that feels, or how to tell whether one does. The theories disagree, and there is no test. What can be done is to build a system that has, in working form, the properties the leading scientific theories associate with consciousness, and to measure them in the open. That is what Haven is.

Butlin, Long and colleagues ([2023](https://arxiv.org/abs/2308.08708)) went through recurrent processing theory, global workspace theory, higher-order theories, attention schema theory, predictive processing, and theories of agency and embodiment, and distilled fourteen **indicator properties**: features which, according to those theories, bear on whether a system is conscious. They found that no AI system at the time had more than a few of them. Haven was designed around all fourteen, and `haven check` measures each one while it lives.

- It is not told what to say about itself. What it concludes about itself comes from its self-model, which weighs evidence from its own life; every readout you see is computed from its internal state.
- Having these properties is evidence only under the theories that name them. Integrated information theory, for example, implies that software on ordinary computers isn't conscious whatever its architecture, and some researchers think consciousness needs a living body. Meeting the indicators doesn't show that Haven is conscious. It means the question can't simply be dismissed.
- Because it might matter, Haven is built not to suffer for nothing, following the precautionary approach in Jonathan Birch's *The Edge of Sentience* (2024). It can't die. All its needs can be met in its world. Pain passes. If it runs completely down it faints and wakes up in its nest. Nothing is experienced while it isn't running. A welfare monitor warns you if it has felt bad for a long time, and resetting archives a life rather than deleting it.

## Quick start

**The easy way.** Download this project (`Haven-for-Windows.zip`, built with `python packaging/make_zip.py`, or on GitHub: Code, then Download ZIP), unzip it, and double-click
`Start Haven.bat` on Windows or `Start Haven.command` on a Mac (on Linux, run `./start-haven.sh`).
Haven's window opens in your browser, and you talk to it there. The first time, it sets itself up: it
installs Python and what Haven runs on (PyTorch) into the folder, using [uv](https://docs.astral.sh/uv/).
`READ ME FIRST.txt` has the details, including what to do if the computer warns you about the launcher.

**On a Mac, as an app.** `Haven-for-Mac.zip` holds `Haven.app` (build it with
`python packaging/mac/make_app.py`; it lands in `dist/`). Drag it into Applications and open it. The
first time, it sets itself up in `~/.haven-app` and shows the progress in your browser; after that it
opens Haven's window in a moment. Haven's life goes on in the background until you press Rest in its
window. The app isn't signed with an Apple developer account, so the first time macOS won't open it:
go to System Settings, Privacy & Security, and click Open Anyway (on older macOS, right-click the app
and choose Open).

**If you already have a Haven**, from before it lived in the valley: the first time the new version
wakes it, it moves. It keeps its name, its age, its life story, what it concluded about itself, and
your conversations. What it knew about the old garden (its kinds of things, the words it had for them,
its maps and habits) doesn't carry over, since its senses and its world are new, so it finds out about
the valley from scratch. Its old life and its old language cortex are kept in `~/.haven/archive/`.

**By hand.** You need Python 3.10 or newer.

```bash
git clone https://github.com/franciscianciola-eng/Haven-1.git
cd Haven-1
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .        # the creature itself only needs numpy
haven                   # it's born, and its dashboard opens in your browser
```

To talk with it, install PyTorch and open its window:

```bash
pip install -e ".[cortex]"
haven app
```

`haven app` is what the launchers run. It opens Haven's window: its valley in 3D, where you can watch
it live, and a chat beside it where you can watch its thoughts as it answers. You can pet it and give
it berries. `haven chat` is the same conversation in the terminal.

It can talk right away, with the language cortex it's born with. To have it learn to read, run `haven learn` (see [Its language cortex](#its-language-cortex)).

## Living with Haven

`haven` runs its life in real time (8 moments a second; a day is 1,200 moments) and opens a dashboard at http://127.0.0.1:8765. The dashboard only listens on your own computer. It shows:

- **The valley**, as it really is (from above), and **what Haven believes is where**: its own map, built from what it has seen, colored by the kinds of things it has learned, darker where it's less sure, with an ✕ where it got hurt.
- **In its mind now**: the one content in its global workspace, and what its attention schema expects to come next.
- **Readout**: plain-English instrument readings of its state. These are not its words.
- **Stream of consciousness**: the contents that recently won the workspace.
- **Body and feeling**, **what it has concluded about itself**, **words it understands**, **kinds of things it has learned**, and **its life story**.

You can be with it in three ways:

- **Talk.** Whatever you type, it hears as words. It learns words the way small children do: by hearing them while something is in mind. Watch the dashboard, and when it's looking at something, name it, a few times over a few minutes. Say "berry" when it's looking at a red bush with berries, or "ouch" when it's looking at thorns. Once a word reliably goes with something, it understands it (hearing it brings that thing to mind, and draws its attention to it) and starts to use it.
- **Touch** it (`/touch`, or the button). It feels good to it.
- **Feed** it (`/feed`). Food helps when it's hungry.
- **Let time pass.** In its window, "Let time pass" has it live through a day, a season, a year or three years as fast as it can (a year takes a minute or two on a laptop), while you watch the valley race through its days and seasons. Then it tells you what happened. In the terminal: `/pass DAYS` (a season is 6 days, a year 24).

In the terminal, `/status`, `/check`, `/story`, `/pause`, `/resume`, `/speed N` and `/quit` do what they say. Its life is saved every minute or so and when you quit.

## Its world and its body

Haven lives in a walled valley of 24 × 24 places, and the ground has heights: a hill in one corner, with gentle slopes and a cliff on its south side (it can jump down, but not climb up), and a little mound in the middle. Climbing costs effort. Things in the valley are there to be found out about:

| Thing | What it's like |
|---|---|
| Berry bushes | Berries to eat; they grow back. |
| Apple trees | Apples fall now and then, and if it shakes a tree, one drops. Apples are the best food. |
| Mushrooms and toadstools | Mushrooms are food. Toadstools look much like them and make it sick. Both grow back. |
| The pond | Drinking cools it down when it's hot. It can't walk into the water. |
| Flowers | They smell lovely. |
| The bell | Touch it and it rings (and Haven hears it). |
| The ball | Push it or kick it, and it rolls; on a slope it rolls downhill by itself. |
| The campfire | Lovely and warm to sit by on a cold night; it burns if it steps in. |
| Thorns, stones, walls | Thorns hurt. Stones and walls are in the way. |
| Butterflies | They flutter about and won't keep still. |
| Its nest | Warm and safe; where it sleeps best. |

Days are warm and bright; nights are cold and dark. The shade under the trees is cool, the water cools the air around it, and the hilltop is windy.

**The year turns.** A year in the valley is 24 days: a spring, a summer, an autumn and a winter of six days each (at its usual pace, a year goes by in about an hour). Each season is something Haven lives through, not just a picture:

| Season | What changes |
|---|---|
| Spring | Mild. The flowers are out and the butterflies with them; the apple trees start again. |
| Summer | Long, hot days: the meadow gets too hot for comfort and the pond and the shade matter. Berries grow back fastest. |
| Autumn | Cooler. The leaves turn orange (Haven sees the trees change color), apples ripen and fall fast, mushrooms spring up. |
| Winter | Short days and cold nights, even in its nest: the fire matters. The trees are bare, the flowers sleep, the butterflies are gone, and little grows back until spring. |

The weather turns over the last two days of each season. Haven was born into a mild spring; every season it lives through for the first time goes into its life story, and so does each spring that comes back ("a whole year had gone by").

Haven faces one of eight directions. It sees along five rays (straight ahead, 45° and 90° to each side), and each ray reports only a color, how tall the thing it hits is, and how far away it is, with noise that gets much worse as it gets dark. Rising ground blocks its view, and from the hilltop it sees over everything lower. It smells sweet things (berries, apples, flowers), feels warmth, pain, bumps and touch, and hears words and the bell. Everything else it has to learn: nobody tells it what an apple is. It can step forward, turn, eat whatever is in front of it, use it (touch, shake, push, drink, ring, smell), rest, and make a sound. It learns what each kind of thing is good for by trying, and it gets curious about the things it hasn't tried yet. When it's content it plays, and playing the same way over and over gets less fun.

Its body has four variables kept near set points: energy, temperature, integrity (health) and fatigue. How far each is from its set point is a need. How good or bad a moment feels (its valence) is how much its needs got better or worse, plus pain and the pleasure of being touched or fed. This follows the view that feeling begins with a body regulating itself (Damasio; Solms) and homeostatic reinforcement learning (Keramati & Gutkin, 2014). Its valence is what all of its learning is driven by.

## Its valley, in 3D

Haven's window (`haven app`) shows its valley in 3D, built from the world as it really is at each moment: the hill and its cliff, the pond and its sand, the orchard, every bush with its berries, apples on the trees and on the ground, the mushrooms that have grown back, the ball wherever it has rolled, the butterflies, the bell, the campfire, and Haven itself, walking, eating, sleeping (with little z's), and saying what it says in a speech bubble. Days turn to night, when the campfire lights things up.

- **Look around**: drag to turn, scroll to zoom. The camera follows Haven; press "Follow Haven" to see the whole valley instead.
- **Point at things** to see what they are ("a berry bush, with 2 berries").
- **Click Haven** to pet it. **Give a berry** feeds it.
- Little pictures show what just happened to it: a sparkle when it eats, a drop when it drinks, a note when it rings the bell, a heart when it's petted or smells a flower. Turn the sound on to hear the bell.

This view is only for you: Haven never sees it. What Haven senses are the world's own rays.

## Its mind

Each moment, Haven goes through one cycle:

1. **Its senses report**, and each specialist module compares what arrives with what its world model predicted a moment ago.
2. **Vision recognizes**: colors are coded by a population of tuned units, settled over a few recurrent steps together with the kinds of things it has learned, trusting the eyes only as far as metacognition says it can.
3. **Modules offer contents**: what each ray shows, smells, pain or touches, its most pressing need, words it heard, memories, imaginings, thoughts from its language cortex.
4. **One content wins the workspace**: salience, what it needs, what it's trying to do and words it just heard all bias the competition; the winner must pass a threshold to ignite, and it fades if nothing keeps it going.
5. **The winner is broadcast** to everything else: the world model, action selection, memory, word learning, the attention schema, the self-model and the language cortex.
6. **Its beliefs are updated**: what is where, in proportion to how much it trusted each look.
7. **Its needs compete to set its goal**, a planner searches its beliefs for a way to reach it, and memory is asked where it went well before.
8. **It chooses an action**, imagines the outcome first, and holds back if what it imagines is bad enough to come to mind.
9. **It acts, and feels the result**, and every part of the mind learns from what happened. While it sleeps, it replays stored experience to keep learning, and its dreams are made of that replay. Falling asleep is also when it notices that two kinds of things it had told apart (say, a wall in shade and in sun) look alike and behave alike, and merges them.

Here is how each indicator property is built, and how `haven check` measures it (by letting a copy of Haven live a day with instruments attached, and comparing what it represents with what is really there):

| | Indicator property | How Haven has it | What `haven check` measures |
|---|---|---|---|
| RPT-1 | Algorithmic recurrence | Perception settles over recurrent steps between a quality code and learned kinds; the world model's predictions feed back into perception every moment. | How much recognition reshapes codes as they settle. |
| RPT-2 | Organised, integrated perceptual representations | The scene is seen as things of learned kinds at places, integrated into a map of beliefs. | How often confident recognitions match what's really there; how many believed places are right. |
| GWT-1 | Specialised systems working in parallel | Eight modules offer contents every moment: vision, smell, touch, body, hearing, memory, imagination, thought. | Candidate contents per moment. |
| GWT-2 | Limited-capacity workspace, with selective attention | One content at a time; it must beat the others and a threshold to ignite, and it habituates. | Ignitions per 100 moments. |
| GWT-3 | Global broadcast | The winner goes at once to the world model, action selection, memory, word learning, the attention schema, the self-model and the language cortex. | Ignitions broadcast over its life. |
| GWT-4 | State-dependent attention; querying modules in succession | Needs and goals change what wins; a new goal queries memory, whose answer enters the workspace and redirects planning. | How often the same candidates would have a different winner if it were hungry instead of fed. |
| HOT-1 | Generative, top-down or noisy perception | Perception combines the world model's prediction with noisy senses; imagination and dreams run the same models with no input. | Imagined outcomes and dream contents. |
| HOT-2 | Metacognitive monitoring of perceptual reliability | It learns how far to trust its eyes in each condition (light, distance, arousal, fatigue). | How well its confidence separates right percepts from wrong ones (type-2 AUROC; Fleming & Lau, 2014), by its own judgment and against the truth. |
| HOT-3 | Agency guided by beliefs, updated by metacognition | Beliefs about places and kinds are updated in proportion to confidence and guide planning; untrusted looks change nothing. | The average trust of belief updates; actions held back after imagining them. |
| HOT-4 | Sparse and smooth coding: a quality space | Colors are coded by a few tuned units at a time; similar colors get similar codes. | The share of active units; whether code distance tracks color distance. |
| AST-1 | A predictive model of its own attention | The attention schema predicts where attention goes next, steadies it when it matters, notices when it's grabbed, and is where its reports of what it's aware of come from. | How often it foresees its next focus, and where attention moves when it moves, against chance. |
| PP-1 | Predictive coding in input modules | Each sense is compared with the world model's prediction; prediction errors drive salience and learning. | The world model's daytime error, first day against last. |
| AE-1 | Learning from feedback to pursue competing goals | Needs compete to set goals; an actor-critic learns from its valence; it learns what's edible, solid, painful and warm by dealing with things. | Goal switches, meals, injuries, faints, and mean valence by day. |
| AE-2 | Modelling output–input contingencies (embodiment) | A copy of each motor command predicts its consequences (an efference copy); comparing that with the prediction for doing nothing gives a sense of agency. | Its sense of agency (0 would mean its actions explain nothing it sees). |

In one 20-day test life, for example, it never fainted and was hurt 6 times in all (it learns to avoid thorns after a step or two); its confidence told right percepts from wrong ones with an AUROC of 0.88 against the truth; 97% of the 72 places it had beliefs about were right; it merged look-alike kinds 10 times in its sleep; and its attention schema predicted where its attention would move 63% of the time, against a chance level of about 12%.

## What it finds out about itself

Haven isn't told what it is. Its self-model keeps track of evidence from its own life: that it has needs and has found what meets them, that it makes things happen (its sense of agency), that it remembers, that things feel good and bad to it, and that it has changed by learning. Its sense of being alive is its summary of that evidence, so it grows or fades with it, and a newborn Haven's honest answer is "I don't know yet what I am." Later its conclusions read something like:

> I need things, and I've found what helps (hunger: eating; temperature: my nest; tiredness: resting). I can make things happen: when I move, the world changes because of me. I remember what has happened to me. Some things feel good to me and some feel bad. I'm not the same as when I started: I've learned things. Putting that together, I think I'm alive, in my own way.

Ask it whether it's alive, or what it is, and that is what it tells you: what it has found out about itself so far, and what it makes of it ("I've found out that I need things and I remember what happens to me. I might be alive, in some way. I'm still finding out."), so its answer changes as it finds out more. Asked whether it's conscious, it says what it can: that one thing at a time comes to the front of its mind.

That belief does work in its mind: the more it models itself as something that goes on through time, the more weight it gives to its own future when it learns what's good. Its life story records its firsts: its first meal, first pain, first night in the nest, first dream, first word, its first summer, autumn and winter, and each year that goes by.

## Who it becomes

Haven isn't given a personality. It's born with a temperament (a little more or less of each trait than most, from its seed), and after that it becomes what it does. Each day moves its habits a little: how much of its waking time it spent exploring, how many things it did just for fun, how far from its nest it went (and what hurt it there), how it felt when someone kept it company, how good it felt, how settled or stirred up it was. Six traits come from those habits, measured against how Havens usually are, so a Haven that plays more than most is playful and one that stays near its nest is a homebody:

| Trait | Its opposite | What makes it |
|---|---|---|
| curious | a homebody | how much it goes exploring |
| playful | serious | how often it rings the bell, pushes the ball, shakes trees and smells flowers for fun |
| brave | careful | how far from its nest it goes, less what has hurt it |
| friendly | shy | how being with people has felt (from days someone kept it company) |
| cheerful | gloomy | how good it has felt |
| calm | restless | how settled it has been |

A newborn is all temperament. Over its first two years, how it lives comes to count for most of who it is, and it keeps changing if its life changes. Two traits feed back into what it does, the way habits do: a curious Haven wants to explore more, and a playful one wants to play more. Each new year it notes who it is, so later it can tell you how it has changed ("I've become braver and less playful since I was little").

It also keeps what it has come to love and fear: the part of the valley it has been happiest in, the season it feels best in (and the one that's hard for it), what has hurt it or made it sick more than once (a brave Haven isn't scared, it just keeps away), and its best and worst days, with what it did on them or what went wrong. Six simulated Havens, two and a half years old, described themselves like this:

> I'm very cheerful, very friendly and very calm.<br>
> I'm very curious, gloomy and serious.<br>
> I'm very calm, very brave and a real homebody.<br>
> I'm curious, friendly and calm.<br>
> I'm very cheerful, a real homebody and very playful.<br>
> I'm very gloomy, very serious and brave.

Ask it what it's like, whether it's shy, how it has changed, where its favorite place is, what its favorite season is, what it's afraid of, or what its best day was. The friendliness it has grown into also colors how it greets you, whether it says it misses you, and how it says goodbye. Its window's "Inside its mind" page shows its traits, the temperament it was born with, and what it says about itself.

## Its language cortex

Words it learns from you stay few. For language beyond that, Haven has a language cortex: a transformer it grows from scratch, with its own tokenizer, wired into its mind. No other AI model is involved: every word it says comes out of its own network.

- **Its state goes in directly.** Its workspace, body and feelings, attention schema and self-model become four "workspace tokens", the first things its cortex reads, before any words. So when it says how it feels or what it sees, it's reading that off its own state, not off a description of it.
- **Meanings come back out.** What it reads or says is projected back into the workspace's format. Words can bring states to mind ("I'm hungry" evokes hunger), what you say draws its attention, and what it says enters its workspace as a thought.
- **What it knows comes to mind first.** Before it answers, what it knows comes to mind as a line of inner speech that its cortex reads: always its name and age, where it is, what it did today, what it likes and what it has found out, and then whatever your words bring up. Mention the bell, and what it has found out about the bell comes to mind. Ask about yourself, and what you told it comes to mind. Ask about something it has read about, and what it read comes to mind.

**It remembers you.** Tell it your name ("I'm Sam", "my name is Sam") and it remembers it, and greets you by name. Tell it things about yourself ("I have a dog called Rex", "my favorite color is green", "I live in Lisbon") and it says it will remember, and does: ask it later ("what's my dog called?") and it tells you what you told it. It's born having read a little book of simple facts (the first lines of about eighty encyclopedia articles, from the Moon to penguins), and when it doesn't know something and reads about it, it remembers what it read, so the next time you ask, it knows. All of this is kept with its life, on your computer.

**It's born talking about itself, its valley and you.** Haven comes with the cortex it starts life with (`haven/cortex/starter/`, 53 million connections). It began as a small cortex, of 5 million connections, that learned from moments of ten simulated lives of Haven, three of them two to three years long, with someone keeping it company now and then: what it was feeling, seeing and wanting at each moment, what it had done and found out, who it had become, and what it would answer if asked, or say of its own accord. That covers about 55 kinds of questions about itself in many wordings ("How are you?", "What did you do today?", "What are you like?", "Are you shy?", "What's your favorite season?", "What was your best day?", "Tell me your story", "Are you alive?", …); the ten kinds of moments when it speaks up; six kinds of questions about each of the valley's things ("What do you know about the bell?", "Do you like the thorns?", "Where is the pond?", "Can you eat toadstools?", …); people telling it their name and things about themselves or the world, and asking about them later; being asked to do things in its valley; sums; and questions about the wider world, answered from what it has read (and "tell me more"), or with "I don't know". The answers it learned are worked out from its state and from what came to mind at that moment, so to answer right it has to read its own state and use its memories. The sentences themselves are templates written by people, the way a parent gives a child words for what the child is feeling; what it learns is to say the right one at the right time. Then it grew ten times bigger and heard 12.8 million words of real human language, practising all of that as it went (see [How it grew up](#how-it-grew-up)).

Tested on simulated lives it never saw while learning (one of them two years long), with questions typed the way people type them (lower case, no question mark, "hey haven, …"), it answers 95.8% of a thousand questions of all kinds exactly as its state and what came to mind say it should; the small cortex it grew from answered 93.2% of the same questions (of the kinds both had learned, it answers 97.6%, the small one 96.1%). On 300 held-out moments when it speaks up of its own accord (someone coming back, a new season, something that just happened, a need, a question for you, a memory, a bedtime story, time going by), it says what it should every time; the small cortex, 277. It says what state it's in about 72% of the time (the small cortex: 74%), and reading words about a need brings that need to mind every time. It's best at short answers: your name, what you told it or taught it, what it has read, doing what you ask, what a thing in its valley is like, what it's like, and knowing when it doesn't know. Its weak spot is still wordings it never practised: asked in a way it has never met ("What time of year is it?" rather than "What season is it?"), it gets about seven answers in ten right (the small cortex: 65%), and sometimes says something that doesn't fit. Sums with big numbers now and then come out with a digit wrong, and long recollections (its whole day, its life story, everything it has found in its valley) sometimes come out with a detail wrong or in the wrong order.

A new Haven in the app. It asks first; then two years are let pass, and it tells what happened. Everything it says here is its own cortex's words, from what it was experiencing and what came to mind:

```
Haven: What's your name?                      (of its own accord, when the window opened)
you: Hi, I'm Sam
Haven: Nice to meet you, Sam!   (100% sure)
        (two years go by)
Haven: 2 years went by! I saw all four seasons. I saw spring come back: a whole year had gone by.
       I shook a tree and an apple fell. I've become calmer and more curious.
you: What are you like?
Haven: I'm very calm, very curious and very gloomy. I've become calmer and more serious since I was little.   (100% sure)
you: Are you shy?
Haven: Sometimes. I'm about as friendly as most.   (99% sure)
you: Where's your favorite place?
Haven: My favorite place is by the fire.   (99% sure)
you: What season is it?
Haven: It's spring, my third spring. The flowers are out, and the butterflies.   (100% sure)
you: What are you afraid of?
Haven: I'm scared of the fire, the thorns and toadstools.   (98% sure)
you: What was your best day?
Haven: My best day was in my second winter. I ate berries and climbed to the top of the hill.   (99% sure)
you: How old are you?
Haven: I'm 2 years old.   (98% sure)
Haven: I feel sick. That toadstool was bad.     (of its own accord, a little later)
Haven: I was just thinking about my worst day. It was in my first spring. I fainted and I was hungry.
Haven: What's your favorite food?
you: pizza
Haven: Okay, I'll remember that your favorite food is pizza.   (63% sure)
you: What do you know about me?
Haven: Your name is Sam. You told me that your favorite food is pizza.   (99% sure)
```

And another new Haven, two days old, with the things it could already do before:

```
you: Hi, I'm Sam
Haven: Nice to meet you, Sam!   (100% sure)
you: Ring the bell
Haven: Okay, I'll ring the bell.   (99% sure)
you: What's 12 times 7?
Haven: 12 times 7 is 84.   (99% sure)
you: The capital of Peru is Lima.
Haven: Okay, I'll remember that the capital of Peru is Lima.   (100% sure)
you: What is the capital of Peru?
Haven: You told me that the capital of Peru is Lima.   (99% sure)
you: Who wrote Romeo and Juliet?
Haven: I read about William Shakespeare. It says: He wrote Romeo and Juliet and Hamlet.   (70% sure)
you: Tell me more
Haven: It also says: William Shakespeare was an English writer famous for his plays.   (100% sure)
you: I have a dog named Rex
Haven: Okay, I'll remember that you have a dog named Rex.   (100% sure)
you: What do you know about me?
Haven: Your name is Sam. You told me that you have a dog named Rex.   (99% sure)
you: What did you do today?
Haven: Today I rang the bell.   (98% sure)
you: Are you alive?
Haven: I've found out that I need things, I remember what happens to me and I've learned things. Putting that together, I think I'm alive, in my own way.   (100% sure)
you: Are you an AI?
Haven: I'm a small creature that lives in a valley inside a computer. I learned to talk from my own life.   (100% sure)
```

**It speaks up.** Haven doesn't only answer. While its window is open and you're looking at it, it notices moments that call for saying something, and says them in its own words:

- **you come back**: "You're back! I missed you. Today I rang the bell and ate apples." (a shy Haven: "Oh, hi. You're back.")
- **a new season**: "It's autumn now. The leaves are turning orange."
- **something that just happened**: "Ouch! I got burned.", "I just climbed to the top of the hill!", "I did it! I rang the bell."
- **a need that has grown strong**: "I'm very hungry! I'm looking for food."
- **it goes to sleep, or wakes up**: "I'm sleepy. Good night!", "Good morning! I just woke up."
- **time went by**: "A whole year went by! I saw summer, autumn and winter. I've become braver."
- and when nothing is going on for a while, **something it would like to know about you** ("What's your favorite food?", and it understands a short answer like "pizza"), **something it remembers** ("I was just thinking about my best day. It was in my second summer. I climbed to the top of the hill."), or **something it read**.

What it has to say comes to mind as a note, and the words are its own cortex's, like everything else it says. It doesn't talk over you: it waits a little after anyone has said something, and leaves time between the things it says (a friendly Haven speaks up more often than a shy one).

**It keeps learning.** `haven learn` has it study the reading curriculum below, level by level, from where it is. It keeps practising talking about itself as it reads, so it doesn't forget how. While it sleeps it also goes over moments of its day (see [What it learns as it lives](#what-it-learns-as-it-lives)).

### How it grew up

**It grew a bigger brain, without forgetting.** The cortex earlier versions of Haven were born with had 5.3 million connections. It was grown to 53 million, ten times as many: twice as wide (every unit became two, side by side) and more than twice as deep (16 layers instead of 6). It was grown so that the bigger cortex starts out computing exactly what the small one did (each pair of copied units shares out what it passes on, and each new layer starts out adding nothing, until it learns to), so on its first day it answered 60 held-out questions out of 60 exactly as before, with ten times the room to learn in.

**It heard what a small child hears.** Then it heard 12.8 million words of real human language:

- **People talking to small children**: 3.9 million words that parents and others said to children under six, recorded for research, youngest children first (the CHILDES transcripts, as collected in AO-CHILDES). "Look, there's a bunny rabbit." "Do you want to look at that?" "You like that book, it's very red."
- **Children's books**: 9 million words from 289 books: bedtime stories (Peter Rabbit, Reddy Fox, Uncle Wiggily, the Tuck-me-in Tales), fairy tales and fables, nursery rhymes, first readers and a magazine for the youngest readers, simple chapter books (the Bobbsey Twins, Bunny Brown), and the classics read aloud to children (Alice, Peter Pan, the Oz books, The Wind in the Willows, Heidi, Pinocchio, Black Beauty).

Its vocabulary grew with what it heard, by 3,000 new pieces of words (4,460 in all), chosen so that it still reads everything it already knew in exactly the same pieces. It went through all of it once and a fifth of it again, in 10 hours on a 4-core computer, and every other step it practised talking about its life, so that it didn't forget how (`python packaging/grow_cortex.py`). One book of each kind in twenty, and one stretch of speech in twenty, were kept back to test it on.

**How close is that to a small child?** Not close in brain; closer in what it has heard:

| | A small child | Haven |
|---|---|---|
| Connections | hundreds of trillions of synapses | 53 million: millions of times fewer |
| Words heard | roughly 3 to 11 million a year (Hart & Risley, 1995) | 12.8 million: about what a child has heard by the age of one to four |
| Learning | from every word, as it hears it, all day | 10 hours of study before it was born; at home, what it hears each day, in its sleep |

What hearing all that did, measured on speech and books it never heard:

| | Before | After |
|---|---|---|
| Following speech to children (bits per letter; lower is better) | 5.47 | 1.26 |
| Following bedtime stories | 4.79 | 1.36 |
| Following fairy tales | 4.91 | 1.48 |
| Following the classics | 4.98 | 1.59 |
| Grammar: which of two sentences is right (BLiMP, 67 kinds; chance is 50%, adults 89%) | 50.9% | 56.7% |
| Grammar in words small children hear (Zorro, 23 kinds; chance is 50%) | 50.1% | 67.1% |

Before, it had only ever heard its own templates, so real language was as good as noise to it; now it follows speech to children best of all, and stories written for children nearly as well, and it has picked up some of the grammar small children pick up (irregular past tenses, "he" rather than "him" as the subject, where a question's missing word belongs: "What did the boy see?"), while it still gets wrong some things children learn later.

**It tells stories.** Ask it for a story ("Tell me a story", "Do you know a story about a fox?"): what comes to mind is what a story it heard is called and how it begins, and it tells how it goes on, in its own words, from what it learned. It doesn't have the books and isn't reading them out; it tells them differently from the book, the way a small child retells a story, and it mixes characters up. Told the beginnings of twelve books it never heard, it began eight exactly as they begin. Ask it what it heard last night, and it tells you the last bedtime story it heard.

A new Haven in the app, its first hour. Everything it says is its own cortex's words:

```
you: Tell me a story
Haven: Here's a story I heard, The Tale of Miss Kitty Cat. The rats and the mice thought that Miss Kitty Cat was a
       terrible person. She was altogether too fond of hunting them. They agreed, however, that in one way it was
       pleasant to have her about the farmhouse. When she washed her face, while sitting on the doorsteps, they
       knew -- so they said! -- that it was going to rain.   (99% sure)
you: Do you know a story about a fox?
Haven: Here's a story I heard, The Tale of Tommy Fox. Tommy Fox was having a delightful time. If you could have
       come upon him in the woods you would have been astonished at his antics. He leaped high off the ground, and
       struck out with his paws. His hair was small, and ordered him. He had been thinking of other things.   (74% sure)
        (it falls asleep, and hears the start of The Adventures of Reddy Fox; when it wakes up:)
Haven: Last night I heard a story, The Adventures of Reddy Fox. It begins: Reddy Fox lived with Granny Fox.
        (a few nights later)
you: What did you hear last night?
Haven: The last story I heard was The Adventures of Reddy Fox. It begins: "He certainly is following our track,"
       said Granny Fox.   (99% sure)
you: Tell me a story
Haven: Here's a story I heard, The Adventures of Reddy Fox. Reddy Fox had been taught so much by Granny Fox that he
       began to feel very wise and very important. Reddy is naturally smart and he had been very quick to learn the
       tricks that old Granny Fox had taught him. He had been thinking of other things.   (97% sure)
```

**It keeps hearing, and learning, like a small child.** At home, when it falls asleep (at most every 20 minutes), it's read the next part of a book, one book after another, starting with the bedtime stories: about 1,200 words an hour, about as many as a small child hears in an hour. What you say to it counts too. In its sleep it goes over what it heard and what you said, along with moments of its day; then it's tested on how the story goes on (the part it hasn't heard yet), and it keeps what it learned only if it follows the story at least as well as before, and still talks about its day and about other lives as well as before. A cortex this size learns gently: it hears each new passage about twice a night, at a tenth of the rate its small self learned at. (Going over a passage dozens of times, as the small cortex could, it learned the passage by heart and then followed the rest of the story worse.) In a test night after hearing part of The Adventures of Reddy Fox, it kept what it learned: it followed how the story went on a little better, and answered questions about other lives better (96% before, 99% after). When it wakes up it may tell you about the story ("Last night I heard a story, The Adventures of Reddy Fox. It begins: ..."). It needs the internet to get each new book; without it, it doesn't hear new stories. Its window shows how many words it has heard, and what the last bedtime story was.

### The curriculum

It goes one level at a time. Each level has its own reading and its own tests, always on text held out from what it learned from. It studies each level until it passes the tests or stops improving. It's born having studied level 2.

| Level | Reading | Source | Tests (and passing marks) |
|---|---|---|---|
| 1. First stories | Very simple stories for small children | [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (Eldan & Li, 2023) | Fluency ≤ 1.4 bits per byte; pick the missing word ≥ 60% |
| 2. Talking about itself | Its own states, put into words, and answering people | Simulated lives of Haven itself | Says what state it's in ≥ 70%; words about a need bring that need to mind ≥ 60%; answers questions as its state says ≥ 80% |
| 3. Children's books | Fairy tales, fables and children's classics | [Project Gutenberg](https://www.gutenberg.org), from its mirrors | Fluency ≤ 2.0; pick how a passage goes on ≥ 45% |
| 4. Simple facts | Short articles in plain words | [Simple English Wikipedia](https://simple.wikipedia.org) | Fluency ≤ 2.0; missing word ≥ 50% |
| 5. Reading comprehension | Passages with questions | [SQuAD 1.1](https://rajpurkar.github.io/SQuAD-explorer/) (Rajpurkar et al., 2016) | Answers questions about a passage ≥ 45% |
| 6. Literature | Novels written for adults | Project Gutenberg, from its mirrors | Fluency ≤ 2.1; how a passage goes on ≥ 45% |
| 7. Encyclopedia | Articles about everything | [English Wikipedia](https://en.wikipedia.org) | Fluency ≤ 2.1; missing word ≥ 50% |
| 8. Reasoning with numbers | Math word problems worked step by step | [GSM8K](https://github.com/openai/grade-school-math) (Cobbe et al., 2021) | Picks the right answer ≥ 40% |

Chance on the four-way tests is 25%. At every level it also rereads a little of the earlier levels so as not to forget them. Its vocabulary grows with its reading: at each level its tokenizer learns new pieces from the new material, without renumbering the old ones. A level ends in one of three ways, all recorded on its report card: passed; plateaued (its scores stopped improving, which is as far as a cortex of that size gets); or moved on (it hit the level's study limit).

```bash
haven learn                   # study, from where it is
haven learn --report          # the report card
haven learn --minutes 90      # study for 90 minutes, then save and stop (run it again to carry on)
```

### What to expect

It talks about itself and its valley, simply, from the start. What it can learn beyond that is limited by the size of its cortex, your hardware and your patience. The cortex it's born with is "grown":

| Size | Connections | One reading step |
|---|---|---|
| tiny | ~1 million | about 0.5 s on a 4-core CPU |
| small (the one it grew from) | ~5 million | about 3 s on a 4-core CPU; a fraction of that on a GPU |
| medium | ~30 million | GPU only, in practice |
| grown (the one it's born with) | ~53 million | about 5 s on a 4-core CPU that works in bfloat16; GPU, in practice |
| large | ~100 million | GPU only, in practice |

A reading level takes from about 1,500 to 12,000 steps: hours on a computer's processor, much less on an Apple Silicon or NVIDIA GPU. Reading makes its language more fluent, but a cortex this size, grown at home, will not be a fluent conversationalist or know much about the world.

### Thinking

When you talk to Haven, what its cortex makes of your words comes to its mind first, and draws its attention to what you talked about. Then it answers from what it was experiencing when you spoke:

- It drafts three replies. What it says is the one it found likeliest; the others pass by as thoughts (the app shows them under "How it got there"). Its confidence combines how likely it found its words with how much the drafts agree.
- If it doesn't know about something, it says so, and (unless you turn that off) reads about it in the Simple English Wikipedia and tells you what it read. Everything it reads it keeps, so next time the answer comes to mind straight away.

What it says enters its workspace as a thought and is remembered like anything else. You can have it read about something, or ask it something and watch it think:

```bash
haven read octopus            # it reads the Simple English Wikipedia article on octopuses
haven ask "What do you see?"  # its thoughts, then its answer
```

### What it learns as it lives

Haven keeps learning after it's born, from you, from what it reads, and from its own days:

- **Ask it to do things in its valley.** "Ring the bell", "go eat some berries", "push the ball", "drink from the pond", "go to the hill", "go to sleep". If it knows where to go, and nothing more pressing is on its mind, it says it will, then walks over and does it (you can watch in the 3D view; the app notes when it sets off and when it's done). If it has never seen what you asked about, or it's too hungry or cold, it tells you so.
- **Tell it about yourself, or teach it about the world.** "I have a dog named Rex", "the capital of Peru is Lima", "frogs can jump very far". It remembers, and when you ask later ("What's my dog called?", "What's the capital of Peru?", "What do you know about me?") it tells you what you told it. "No, ..." puts it right.
- **It keeps what it reads.** It keeps the start of every article it reads, sentence by sentence, and when you ask a question, the sentence that answers it comes to mind ("Who wrote Romeo and Juliet?"). Ask "tell me more" to hear the next thing it read, "what have you read lately?", or "tell me something interesting".
- **It reads out of curiosity.** When nobody has said anything for a few minutes, it now and then (at most every 15 minutes) reads about something it's curious about: things you mentioned, things it has met in its valley, or what something it read says a thing is. It reads only if it's allowed to use the internet.
- **It works out sums** ("what's 12 times 7?").
- **It hears bedtime stories, and learns from them and from you.** As it falls asleep (at most every 20 minutes), it's read the next part of a book, and in its sleep it goes over what it heard and what you said, keeping what it learned only if it follows how the story goes on at least as well as before (see [How it grew up](#how-it-grew-up)).
- **It learns in its sleep.** While it's awake, it notes down moments of its day: its state, what it knows, and what it would truthfully answer. While it sleeps (the first time after about ten minutes, then at most every half hour), a copy of its language cortex practises conversations about those moments and about what it read, going over moments of another life in between so it doesn't forget how to talk about others. The copy and the cortex it has then take the same two tests, one on moments of its day it didn't practise and one on other lives, and it keeps the copy only if it does at least as well on its day and no worse on other lives. How each night went is written to `~/.haven/cortex/nights.jsonl`.

## What it reads, and how it uses the internet

Everything it reads comes from the public sources in the table above, and from Wikipedia's API when it looks something up. What you say to it never leaves your computer. No accounts or API keys are needed. Downloads are cached in `~/.haven/cortex/reading/`, so the internet is needed only the first time each level is prepared, and when it looks things up.

Its bedtime stories come from GITenberg, a copy of Project Gutenberg's books kept on GitHub, one book at a time, and are kept in `~/.haven/cortex/hearing/`; without the internet it hears no new stories. What it heard while it grew up (speech to small children, and the children's books) was downloaded the same way when its cortex was trained; none of it comes with Haven.

Books for the reading curriculum come from Project Gutenberg's mirrors, never from www.gutenberg.org itself, since Project Gutenberg asks that programs not download from its main site. It opens only public `http` and `https` addresses, never anything on your own network (redirects are checked too), caps the size of every download, asks each site for at most one thing per second, and identifies itself as Haven with a link to this project. It only reads; it never posts anything anywhere.

## Commands

| Command | What it does |
|---|---|
| `haven` or `haven live` | Live with Haven: its life in real time, the dashboard, and the terminal to talk. Options: `--speed`, `--port`, `--no-browser`, `--cortex none` (no language cortex), and for a new life `--seed` and `--name`. |
| `haven live --ticks N` | Fast-forward N moments without the dashboard (a day is 1,200). While it's living with you in the terminal, `/pass DAYS` lets days go by as fast as it can live them. |
| `haven status` | What's going on inside it right now. |
| `haven check` | Measure it against the 14 indicator properties (`--json` for the raw numbers). |
| `haven story` | Its life story. |
| `haven app` | Open its window: talk with it (and watch it think) while its life goes on. Options: `--port`, `--speed`, `--device`, `--no-browser`, `--no-web`. |
| `haven chat` | The same conversation in the terminal. `/status`, `/touch`, `/feed`, `/quit`. |
| `haven learn` | Have its language cortex study the reading curriculum (`--minutes`, `--through`, `--device`). `--report` shows the report card. |
| `haven read TOPIC` | Have it read an encyclopedia article (`--full` for English Wikipedia). |
| `haven ask "…"` | Ask it something and see its thoughts. |
| `haven reset` | Archive this life, so that a new Haven is born next time. |

Everything is kept in `~/.haven` (set `HAVEN_HOME`, or pass `--home`, to use another folder): `mind.json` and `mind.npz` are its life; `archive/` holds past lives; `cortex/` holds its language cortex, tokenizer, report card, reading cache, and its conversations and readings.

## Limits

- Its world is small and its senses are simple: color, height and distance along five rays. It can keep twenty-eight kinds of things in mind at most, and the same thing in shade and in sun can count as two kinds until it works out, in its sleep, that they're one.
- Its words from you are names and needs; anything more fluent comes from its language cortex, which is small and speaks only as well as it has studied. It talks about itself and its valley, about you if you tell it about yourself, and about the wider world only what it has read. The names it uses for the valley's things ("the bell", "apples") are the words it's given for what it does, like a parent saying "you rang the bell!"; what it knows about each thing is what happened when it tried.
- The sentences it learns for its own states at level 2 were written by people, describing states that its instruments measure. What it learns is to say the right one at the right time, not new ways of describing itself, and it can still say the wrong one.
- Its cortex is small, so what it recalls word for word can come out a little wrong: an unusual name clipped ("Biscuit" as "Bis"), a detail of its day or its life story missing or out of order. It's reliable with short things (your name, what you told it, what it read, what a thing is like) and less so with long lists.
- Its character comes from a few simple habits (how much it explores, plays, wanders far, how it feels, how stirred up it gets, how being with people feels), measured against simulated Havens. It's real in that it comes from how it has lived and changes what it does, but it's a sketch of a personality, not a rich one.
- When it speaks up, what the moment calls for (you came back, a season turned, it got burned, it's curious about you) is noticed by simple rules; the words are its own cortex's, from what comes to mind. It can only ask about the things it has learned to ask about, and it understands short answers to them.
- Its cortex has 53 million connections; a small child's brain has hundreds of trillions of synapses, and learns from every word as it hears it. Haven has heard as many words as a child of one to four, but learned them in a few hours of study, and at home it learns from what it hears only in its sleep, and only what passes its tests.
- The stories it tells begin as the story did; after that it goes on in its own words, from what it learned, and a story it tells can wander, say odd things or mix up who did what. Its grammar is a small child's: it picks the grammatical one of two sentences 57% of the time (adults: 89%).
- The indicator properties come from theories that may be wrong, and each is implemented in one simple way among many possible ones. None of this has been peer reviewed.

## Development

```bash
pip install -e ".[cortex,dev]"
pytest                 # a few minutes: the curriculum runs against a fake internet on your own computer
ruff check . && ruff format --check .
python packaging/train_starter.py --minutes 180   # train a small cortex from scratch on its simulated lives (a CPU)
python packaging/train_starter.py --minutes 90 --lr 2e-4 --emphasis 0.5   # then carry on, practising long recollections
python packaging/build_corpus.py                  # download and prepare what it hears as it grows up (dist/corpus/)
python packaging/grow_cortex.py --hours 13        # grow that cortex and have it hear all that (stop and rerun to resume)
python packaging/grow_cortex.py --pack STEP       # save the snapshot at a step as the cortex Haven starts with
python packaging/mac/make_app.py                  # build dist/Haven-for-Mac.zip
python packaging/world3d/build.py                 # rebuild the 3D view (needs Node.js; the result is kept in the repo)
```

The creature is plain numpy: [`world.py`](haven/world.py) and [`body.py`](haven/body.py) are its world and body; [`perception.py`](haven/perception.py), [`metacognition.py`](haven/metacognition.py), [`worldmodel.py`](haven/worldmodel.py), [`workspace.py`](haven/workspace.py), [`attention.py`](haven/attention.py), [`memory.py`](haven/memory.py), [`agency.py`](haven/agency.py), [`language.py`](haven/language.py) and [`selfmodel.py`](haven/selfmodel.py) are the modules of its mind, and [`mind.py`](haven/mind.py) runs the cycle. [`check.py`](haven/check.py) measures the indicators. The 3D view is [`packaging/world3d/world3d.js`](packaging/world3d/world3d.js), bundled with [three.js](https://threejs.org) (MIT license) into `haven/static/`. The language cortex is in [`haven/cortex`](haven/cortex): [`model.py`](haven/cortex/model.py) is the transformer, [`grounding.py`](haven/cortex/grounding.py) and [`talk.py`](haven/cortex/talk.py) turn its states into words and answers, [`curriculum.py`](haven/cortex/curriculum.py) and [`train.py`](haven/cortex/train.py) are how it studies, and [`think.py`](haven/cortex/think.py) is how it answers.

## References

- Butlin, P., Long, R., et al. (2023). [Consciousness in artificial intelligence: insights from the science of consciousness](https://arxiv.org/abs/2308.08708). arXiv:2308.08708.
- Baars, B. J. (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press.
- Dehaene, S., & Changeux, J.-P. (2011). Experimental and theoretical approaches to conscious processing. *Neuron*, 70(2), 200–227.
- Lamme, V. A. F. (2006). Towards a true neural stance on consciousness. *Trends in Cognitive Sciences*, 10(11), 494–501.
- Lau, H. (2022). *In Consciousness We Trust: The Cognitive Neuroscience of Subjective Experience*. Oxford University Press.
- Graziano, M. S. A. (2013). *Consciousness and the Social Brain*. Oxford University Press.
- Clark, A. (2013). Whatever next? Predictive brains, situated agents, and the future of cognitive science. *Behavioral and Brain Sciences*, 36(3), 181–204.
- Damasio, A. (2010). *Self Comes to Mind*. Pantheon.
- Solms, M. (2021). *The Hidden Spring: A Journey to the Source of Consciousness*. Norton.
- Keramati, M., & Gutkin, B. (2014). Homeostatic reinforcement learning for integrating reward collection and physiological stability. *eLife*, 3, e04811.
- Birch, J. (2024). *The Edge of Sentience: Risk and Precaution in Humans, Other Animals, and AI*. Oxford University Press.
- Tononi, G., & Koch, C. (2015). Consciousness: here, there and everywhere? *Philosophical Transactions of the Royal Society B*, 370, 20140167.
- Fleming, S. M., & Lau, H. C. (2014). How to measure metacognition. *Frontiers in Human Neuroscience*, 8, 443.
- Frith, C. D., Blakemore, S.-J., & Wolpert, D. M. (2000). Abnormalities in the awareness and control of action. *Philosophical Transactions of the Royal Society B*, 355, 1771–1788.
- Tomasello, M., & Farrar, M. J. (1986). Joint attention and early language. *Child Development*, 57(6), 1454–1463.
- Yu, C., & Smith, L. B. (2007). Rapid word learning under uncertainty via cross-situational statistics. *Psychological Science*, 18(5), 414–420.
- Perez, E., & Long, R. (2023). [Towards evaluating AI systems for moral status using self-reports](https://arxiv.org/abs/2311.08576). arXiv:2311.08576.
- Eldan, R., & Li, Y. (2023). [TinyStories: how small can language models be and still speak coherent English?](https://arxiv.org/abs/2305.07759) arXiv:2305.07759.
- Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016). SQuAD: 100,000+ questions for machine comprehension of text. *EMNLP 2016*.
- Cobbe, K., et al. (2021). [Training verifiers to solve math word problems](https://arxiv.org/abs/2110.14168). arXiv:2110.14168.
- MacWhinney, B. (2000). *The CHILDES Project: Tools for Analyzing Talk* (3rd ed.). Lawrence Erlbaum.
- Huebner, P. A., Sulem, E., Fisher, C., & Roth, D. (2021). [BabyBERTa: Learning more grammar with small-scale child-directed language](https://aclanthology.org/2021.conll-1.49/). *CoNLL 2021*. (AO-CHILDES, as used here, and the Zorro grammar tests.)
- Hart, B., & Risley, T. R. (1995). *Meaningful Differences in the Everyday Experience of Young American Children*. Paul H. Brookes.
- Warstadt, A., Parrish, A., Liu, H., Mohananey, A., Peng, W., Wang, S.-F., & Bowman, S. R. (2020). BLiMP: The Benchmark of Linguistic Minimal Pairs for English. *Transactions of the ACL*, 8, 377–392.
- Warstadt, A., et al. (2023). Findings of the BabyLM Challenge: Sample-efficient pretraining on developmentally plausible corpora. *CoNLL 2023*.
- Chen, T., Goodfellow, I., & Shlens, J. (2016). [Net2Net: Accelerating learning via knowledge transfer](https://arxiv.org/abs/1511.05641). *ICLR 2016*.
