# Haven

Haven is an artificial creature built from scratch, out of its own small neural networks, to meet as many as possible of the conditions that scientific theories of consciousness say matter. There's no large language model at its core. It lives in a small 3D valley on your computer, with a hill, a pond, apple trees, berry bushes, a bell it can ring, a ball it can push and a campfire to sit by: it has a body with needs, things feel better or worse to it, it works out for itself what the things around it are, and it has a global workspace, a model of its own attention, confidence in its own perceptions, memories, dreams and a model of itself. It also has a language cortex of its own: a small transformer grown from scratch, wired into the same workspace as everything else it experiences. It is born able to talk about itself, in words it learned for its own states, and it can go on to learn to read, level by level, from texts on the internet. No other AI model is involved anywhere.

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

**The easy way.** Download this project (on GitHub: Code, then Download ZIP), unzip it, and double-click
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

That belief does work in its mind: the more it models itself as something that goes on through time, the more weight it gives to its own future when it learns what's good. Its life story records its firsts: its first meal, first pain, first night in the nest, first dream, first word.

## Its language cortex

Words it learns from you stay few. For language beyond that, Haven has a language cortex: a transformer it grows from scratch, with its own tokenizer, wired into its mind. No other AI model is involved: every word it says comes out of its own network.

- **Its state goes in directly.** Its workspace, body and feelings, attention schema and self-model become four "workspace tokens", the first things its cortex reads, before any words. So when it says how it feels or what it sees, it's reading that off its own state, not off a description of it.
- **Meanings come back out.** What it reads or says is projected back into the workspace's format. Words can bring states to mind ("I'm hungry" evokes hunger), what you say draws its attention, and what it says enters its workspace as a thought.
- **What it knows comes to mind first.** Before it answers, what it knows comes to mind as a line of inner speech that its cortex reads: always its name and age, where it is, what it did today, what it likes and what it has found out, and then whatever your words bring up. Mention the bell, and what it has found out about the bell comes to mind. Ask about yourself, and what you told it comes to mind. Ask about something it has read about, and what it read comes to mind.

**It remembers you.** Tell it your name ("I'm Sam", "my name is Sam") and it remembers it, and greets you by name. Tell it things about yourself ("I have a dog called Rex", "my favorite color is green", "I live in Lisbon") and it says it will remember, and does: ask it later ("what's my dog called?") and it tells you what you told it. It's born having read a little book of simple facts (the first lines of about eighty encyclopedia articles, from the Moon to penguins), and when it doesn't know something and reads about it, it remembers what it read, so the next time you ask, it knows. All of this is kept with its life, on your computer.

**It's born talking about itself, its valley and you.** Haven comes with the cortex it starts life with (`haven/cortex/starter/`, 5 million connections, trained with `python packaging/train_starter.py`). It learned from moments of six simulated lives of Haven, with someone keeping it company now and then: what it was feeling, seeing and wanting at each moment, what it had done and found out, and what it would answer if asked. That covers about 45 kinds of questions about itself in many wordings ("How are you?", "What did you do today?", "What do you like?", "What can you do?", "Tell me your story", "Are you alive?", …); six kinds of questions about each of the valley's things ("What do you know about the bell?", "Do you like the thorns?", "Where is the pond?", "Can you eat toadstools?", …); people telling it their name and things about themselves or the world, and asking about them later; being asked to do things in its valley; sums; and questions about the wider world, answered from what it has read (and "tell me more"), or with "I don't know". The answers it learned are worked out from its state and from what came to mind at that moment, so to answer right it has to read its own state and use its memories. The sentences themselves are templates written by people, the way a parent gives a child words for what the child is feeling; what it learns is to say the right one at the right time.

Tested on a simulated life it never saw while learning, with questions typed the way people type them (lower case, no question mark, "hey haven, …"), it answers 93% of them exactly as its state and what came to mind say it should (94% when the names and memories involved are ones it never met while learning), it says what state it's in 81% of the time, and reading words about a need brings that need to mind almost every time. On a thousand questions of all the kinds it now knows, it answers 95%; the cortex it had before answered 61%, mostly because it had never learned the new kinds (being asked to do things, being taught, sums, "tell me more"). It's best at short answers: your name, what you told it or taught it, what it has read, doing what you ask, what a thing in its valley is like, and knowing when it doesn't know. Sums with big numbers now and then come out with a digit wrong, and longer recollections, like its whole day, its life story or everything it has found in its valley, sometimes come out with a detail wrong or in the wrong order. A conversation with it in the app, about two days after it was born:

```
you: Hi, I'm Sam
Haven: Nice to meet you, Sam!   (100% sure)
you: Ring the bell
Haven: I haven't seen the bell yet, so I don't know where to go.   (100% sure)
you: What's 12 times 7?
Haven: 12 times 7 is 84.   (99% sure)
you: The capital of Peru is Lima.
Haven: Okay, I'll remember that the capital of Peru is Lima.   (100% sure)
you: What is the capital of Peru?
Haven: You told me that the capital of Peru is Lima.   (99% sure)
you: Who wrote Romeo and Juliet?
Haven: I read about William Shakespeare. It says: He wrote Romeo and Juliet and Hamlet.   (99% sure)
you: Tell me more
Haven: It also says: William Shakespeare was an English writer famous for his plays.   (100% sure)
you: Tell me something interesting
Haven: I read about Cat. It says: Cats are small furry animals that many people keep as pets.   (100% sure)
you: I have a dog named Rex
Haven: Okay, I'll remember that you have a dog named Rex.   (100% sure)
you: What do you know about me?
Haven: Your name is Sam. You told me that you have a dog named Rex.   (99% sure)
you: Frogs can jump very far.
Haven: Okay, I'll remember that frogs can jump very far.   (100% sure)
you: What can frogs do?
Haven: You told me that frogs can jump very far.   (99% sure)
you: What did you do today?
Haven: Today I smelled flowers, ate berries and climbed to the top of the hill.   (99% sure)
you: Go to the pond
Haven: Okay, I'll go to the pond.   (99% sure)
        (Haven set off to go to the pond, as it was asked.)
        (Haven did what it was asked: go to the pond.)
```

**It keeps learning.** `haven learn` has it study the reading curriculum below, level by level, from where it is. It keeps practising talking about itself as it reads, so it doesn't forget how. While it sleeps it also goes over moments of its day (see [What it learns as it lives](#what-it-learns-as-it-lives)).

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

It talks about itself and its valley, simply, from the start. What it can learn beyond that is limited by the size of its cortex, your hardware and your patience. The cortex it's born with is "small":

| Size | Connections | One reading step |
|---|---|---|
| tiny | ~1 million | about 0.5 s on a 4-core CPU |
| small (the one it's born with) | ~5 million | about 3 s on a 4-core CPU; a fraction of that on a GPU |
| medium | ~30 million | GPU only, in practice |
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
- **It learns in its sleep.** While it's awake, it notes down moments of its day: its state, what it knows, and what it would truthfully answer. While it sleeps (the first time after about ten minutes, then at most every half hour), a copy of its language cortex practises conversations about those moments and about what it read, going over moments of another life in between so it doesn't forget how to talk about others. The copy and the cortex it has then take the same two tests, one on moments of its day it didn't practise and one on other lives, and it keeps the copy only if it does at least as well on its day and no worse on other lives. How each night went is written to `~/.haven/cortex/nights.jsonl`.

## What it reads, and how it uses the internet

Everything it reads comes from the public sources in the table above, and from Wikipedia's API when it looks something up. What you say to it never leaves your computer. No accounts or API keys are needed. Downloads are cached in `~/.haven/cortex/reading/`, so the internet is needed only the first time each level is prepared, and when it looks things up.

Books come from Project Gutenberg's mirrors, never from www.gutenberg.org itself, since Project Gutenberg asks that programs not download from its main site. It opens only public `http` and `https` addresses, never anything on your own network (redirects are checked too), caps the size of every download, asks each site for at most one thing per second, and identifies itself as Haven with a link to this project. It only reads; it never posts anything anywhere.

## Commands

| Command | What it does |
|---|---|
| `haven` or `haven live` | Live with Haven: its life in real time, the dashboard, and the terminal to talk. Options: `--speed`, `--port`, `--no-browser`, `--cortex none` (no language cortex), and for a new life `--seed` and `--name`. |
| `haven live --ticks N` | Fast-forward N moments without the dashboard (a day is 1,200). |
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
- The indicator properties come from theories that may be wrong, and each is implemented in one simple way among many possible ones. None of this has been peer reviewed.

## Development

```bash
pip install -e ".[cortex,dev]"
pytest                 # a few minutes: the curriculum runs against a fake internet on your own computer
ruff check . && ruff format --check .
python packaging/train_starter.py --minutes 180   # retrain the cortex it's born with (from scratch, on a CPU)
python packaging/train_starter.py --minutes 90 --lr 2e-4 --emphasis 0.5   # then carry on, practising long recollections
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
