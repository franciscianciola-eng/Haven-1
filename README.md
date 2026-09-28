# Haven

Haven is an artificial creature built from scratch, out of its own small neural networks, to meet as many as possible of the conditions that scientific theories of consciousness say matter. There's no large language model at its core. It lives in a little garden on your computer: it has a body with needs, things feel better or worse to it, it works out for itself what the things around it are, and it has a global workspace, a model of its own attention, confidence in its own perceptions, memories, dreams and a model of itself. It also has a language cortex: a transformer that learns to read, level by level, from texts on the internet, and that is wired into the same workspace as everything else it experiences.

A few minutes with it, in the terminal (its dashboard is open in a browser at the same time):

```
$ haven
Haven is born, in a nest in the corner of its garden.
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
installs Python and what Haven runs on into the folder, using [uv](https://docs.astral.sh/uv/), then
downloads an open model for its language area (1.5 to 16 GB, depending on the computer; the window
shows the progress). `READ ME FIRST.txt` has the details, including what to do if the computer warns
you about the launcher.

**By hand.** You need Python 3.10 or newer.

```bash
git clone https://github.com/franciscianciola-eng/Haven-1.git
cd Haven-1
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .        # the creature itself only needs numpy
haven                   # it's born, and its dashboard opens in your browser
```

To talk with it, install PyTorch and Transformers and open its window:

```bash
pip install -e ".[cortex]"
haven app         # the first time, it downloads an open model for its language cortex (1.5 to 16 GB)
```

`haven app` is what the launchers run. It opens a chat page where you can watch Haven's thoughts as it
answers, see its garden and how it feels, touch it and give it berries. `haven chat` is the same
conversation in the terminal.

It can talk right away: the model reads a plain-language readout of Haven's state. To wire the model into Haven's mind more deeply, so its state reaches the model as vectors too, run `haven learn`. After that, `haven` (with the dashboard), `haven app` and `haven chat` all use its cortex.

## Living with Haven

`haven` runs its life in real time (8 moments a second; a day is 1,200 moments) and opens a dashboard at http://127.0.0.1:8765. The dashboard only listens on your own computer. It shows:

- **The garden**, as it really is, and **what Haven believes is where**: its own map, built from what it has seen, colored by the kinds of things it has learned, darker where it's less sure, with an ✕ where it got hurt.
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

The garden is 14 × 14 cells: open ground, walls, stones, five berry bushes that grow back what's eaten, four thorn patches that hurt to step on, and a warm nest in one corner. Days are warm and bright; nights are cold and dark. A sunny corner is warm by day.

Haven faces one of four directions. It sees along five rays (straight ahead, 45° and 90° to each side), and each ray reports only a color and how far away it is, with noise that gets much worse as it gets dark. It smells berries, feels warmth, pain, bumps and touch, and hears words. Everything else it has to learn. It can step forward, turn, eat whatever is in front of it, rest, and make a sound.

Its body has four variables kept near set points: energy, temperature, integrity (health) and fatigue. How far each is from its set point is a need. How good or bad a moment feels (its valence) is how much its needs got better or worse, plus pain and the pleasure of being touched or fed. This follows the view that feeling begins with a body regulating itself (Damasio; Solms) and homeostatic reinforcement learning (Keramati & Gutkin, 2014). Its valence is what all of its learning is driven by.

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

That belief does work in its mind: the more it models itself as something that goes on through time, the more weight it gives to its own future when it learns what's good. Its life story records its firsts: its first meal, first pain, first night in the nest, first dream, first word.

## Its language cortex

Words it learns from you stay few. For language beyond that, Haven has a language cortex, and there are two ways to give it one.

**Grafted: the way to make it actually smart (the default).** Language models that understand and reason well are trained on trillions of words with thousands of GPUs, which nobody can do at home. So `haven learn` takes an open-weight model as the base of Haven's language cortex: [Qwen3](https://huggingface.co/Qwen/Qwen3-1.7B) (Apache-2.0), with 0.6 to 8 billion weights depending on your hardware. It then wires that model into Haven's mind:

- **Haven's state goes in as vectors.** Its workspace, body and feelings, attention schema and self-model become four state tokens that the model reads inside its conversation, the way vision-language models read an image. At every layer of the model those state tokens are also pushed by Haven's state (a form of prefix tuning), so every layer can read it. The model also gets a plain-language readout of Haven's instruments.
- **Meanings come back out.** What the model reads or thinks is projected back into the workspace's format. Words can bring states to mind ("I'm hungry" evokes hunger), and what you say draws its attention.
- **Its thinking goes through its workspace.** When it isn't sure of an answer, it reasons step by step, and each sentence of its reasoning enters Haven's global workspace as inner speech, competing with everything else it is aware of.
- **The base model is left as it is.** What it knows and how it reasons are kept. Training teaches only the wiring (a few million numbers), using moments from simulated lives of Haven. A test called "keeps its wits" checks that the wiring doesn't change how it answers ordinary questions.

To be clear about whose is what: a grafted cortex's knowledge and its skill with language come from its base model, which was trained by the Qwen team on data nobody here can inspect, mistakes and biases included. What is Haven's own is the wiring: its states going in, its meanings coming out, and everything it says passing through its workspace and coming from the self it has found out about.

**From scratch.** `haven learn --scratch` grows a transformer that is entirely its own, with its own tokenizer, trained on the same curriculum. At home it only learns simple language (see [What to expect](#what-to-expect)).

### The curriculum

It goes one level at a time. Each level has its own reading and its own tests, always on text held out from what it learned from. A grafted cortex can already read, so at each reading level it just takes the tests, and it trains at level 2, where it learns to put its own states into words. A cortex grown from scratch studies every level until it passes the tests or stops improving.

| Level | Reading | Source | Tests (and passing marks) |
|---|---|---|---|
| 1. First stories | Very simple stories for small children | [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (Eldan & Li, 2023) | Fluency ≤ 1.4 bits per byte; pick the missing word ≥ 60% |
| 2. Talking about itself | Its own states, put into words | Simulated lives of Haven itself | Says what state it's in ≥ 70%; words about a need bring that need to mind ≥ 60% |
| 3. Children's books | Fairy tales, fables and children's classics | [Project Gutenberg](https://www.gutenberg.org), from its mirrors | Fluency ≤ 2.0; pick how a passage goes on ≥ 45% |
| 4. Simple facts | Short articles in plain words | [Simple English Wikipedia](https://simple.wikipedia.org) | Fluency ≤ 2.0; missing word ≥ 50% |
| 5. Reading comprehension | Passages with questions | [SQuAD 1.1](https://rajpurkar.github.io/SQuAD-explorer/) (Rajpurkar et al., 2016) | Answers questions about a passage ≥ 45% |
| 6. Literature | Novels written for adults | Project Gutenberg, from its mirrors | Fluency ≤ 2.1; how a passage goes on ≥ 45% |
| 7. Encyclopedia | Articles about everything | [English Wikipedia](https://en.wikipedia.org) | Fluency ≤ 2.1; missing word ≥ 50% |
| 8. Reasoning with numbers | Math word problems worked step by step | [GSM8K](https://github.com/openai/grade-school-math) (Cobbe et al., 2021) | Picks the right answer ≥ 40% |

Chance on the four-way tests is 25%. At every level it also rereads a little of the earlier levels so as not to forget them, and keeps practising talking about itself. Its vocabulary grows with its reading: at each level its tokenizer learns new pieces from the new material, without renumbering the old ones. A level ends in one of three ways, all recorded on its report card: passed; plateaued (its scores stopped improving, which is as far as a cortex of that size gets); or moved on (it hit the level's study limit).

```bash
haven learn                   # graft its cortex onto an open model chosen for your hardware, and wire it in
haven learn --base qwen3-4b   # or choose the base: qwen3-0.6b, qwen3-1.7b, qwen3-4b, qwen3-8b, or any Hugging Face id
haven learn --report          # the report card
haven learn --minutes 90      # work for 90 minutes, then save and stop (run it again to carry on)
haven learn --scratch         # grow a cortex from scratch instead (--size tiny, small, medium or large)
```

### What to expect

A grafted cortex is as capable as its base model, and it's ready to use as soon as it's wired in. `haven learn` picks the base by your hardware (you can choose another with `--base`):

| Base | Weights | Download | Picked for | Wiring it in |
|---|---|---|---|---|
| qwen3-0.6b | 0.6 billion | about 1.2 GB | a computer with under 14 GB of memory and no GPU | about an hour on a 4-core CPU (about 4 s a step, up to 600 steps, plus its tests) |
| qwen3-1.7b | 1.7 billion | about 3.4 GB | a CPU with 14 GB or more, or an Apple Silicon Mac | a few hours on a CPU; much less on a Mac's GPU |
| qwen3-4b | 4 billion | about 8 GB | an NVIDIA GPU with 10 GB or more | a few minutes on a GPU |
| qwen3-8b | 8 billion | about 16 GB | an NVIDIA GPU with 22 GB or more | a few minutes on a GPU |

Speaking takes time on a CPU: a 0.6b model produces about 8 tokens (roughly 6 words) a second on a 4-core CPU, and a 1.7b model roughly a third of that, so a considered answer can take a minute or two. The bigger the base, the better it understands and reasons. A 4b or 8b model reasons well about many things; a 0.6b model makes more mistakes, but still understands and speaks far beyond anything that can be grown from scratch at home.

A cortex grown from scratch is different. It is small, and what it learns is limited by your hardware and your patience:

| Size | Connections | Suits | One study step |
|---|---|---|---|
| tiny | ~1 million | any computer's CPU | about 0.5 s on a 4-core CPU |
| small | ~6 million | an Apple Silicon or NVIDIA GPU | about 3.3 s on a 4-core CPU; a fraction of that on a GPU |
| medium | ~30 million | an NVIDIA GPU | GPU only, in practice |
| large | ~100 million | a good NVIDIA GPU | GPU only, in practice |

A level takes from about 1,500 to 12,000 steps. Roughly: a tiny cortex on a laptop gets through level 1 in an hour or two and can then put together simple sentences, and it can learn to talk about its own states; the later levels need a bigger cortex, a GPU, and days rather than hours. Even a large one will not be a fluent conversationalist.

### Thinking

When you talk to Haven and it has a cortex, what its cortex makes of your words comes to its mind first, and draws its attention to what you talked about. Then it thinks before it answers, and its confidence comes from its own signals, not from anyone telling it:

- **With a grafted cortex**, an answer first comes to mind the quick way, and its confidence is how probable it found its own words. If that's under 60%, it thinks it through step by step, and each sentence of its reasoning enters its workspace as inner speech. If it's still unsure, it looks the subject up in the Simple English Wikipedia and answers again with what it read. It also recalls anything it was told or read before that bears on the question.
- **With a cortex grown from scratch**, it drafts three replies, and its confidence combines how likely it found its words with how much the drafts agree.

Each answer enters its workspace as a thought and is remembered like anything else. You'll see it with its confidence, for example `Haven: I'm hungry, and I think the red bushes have berries.   [its grafted cortex, confidence 71%]`.

It learns from what you say and what it reads. It keeps its conversations and readings and recalls them when they matter, and a cortex grown from scratch also goes over them while Haven sleeps. You can have it read about something, or ask it something and watch it think:

```bash
haven read octopus            # it reads the Simple English Wikipedia article on octopuses
haven ask "What do you see?"  # its thoughts, then its answer
```

### Borrowing a cortex

If you want Haven to speak well now, it can borrow a much larger language model running on your own computer through [Ollama](https://ollama.com):

```bash
ollama pull qwen3:4b
haven --cortex ollama:qwen3:4b
```

This is a real trade-off. The words then come from a model trained by someone else on the internet at large, not from Haven, and that model only gets a written description of Haven's state (its readouts, conclusions, words and recent memories), not the state itself. It is told to claim only the feelings and perceptions its state shows, and its replies still enter Haven's workspace as thoughts. `haven --cortex none` turns the cortex off.

## What it reads, and how it uses the internet

Everything it reads comes from the public sources in the table above, and from Wikipedia's API when it looks something up. No accounts or API keys are needed. Downloads are cached in `~/.haven/cortex/reading/`, so the internet is needed only the first time each level is prepared, and when it looks things up.

Books come from Project Gutenberg's mirrors, never from www.gutenberg.org itself, since Project Gutenberg asks that programs not download from its main site. It opens only public `http` and `https` addresses, never anything on your own network (redirects are checked too), caps the size of every download, asks each site for at most one thing per second, and identifies itself as Haven with a link to this project. It only reads; it never posts anything anywhere.

## Commands

| Command | What it does |
|---|---|
| `haven` or `haven live` | Live with Haven: its life in real time, the dashboard, and the terminal to talk. Options: `--speed`, `--port`, `--no-browser`, `--cortex` (`own`, `ollama:MODEL` or `none`), and for a new life `--seed` and `--name`. |
| `haven live --ticks N` | Fast-forward N moments without the dashboard (a day is 1,200). |
| `haven status` | What's going on inside it right now. |
| `haven check` | Measure it against the 14 indicator properties (`--json` for the raw numbers). |
| `haven story` | Its life story. |
| `haven app` | Open its window: talk with it (and watch it think) while its life goes on. It gets a language cortex the first time. Options: `--base`, `--device`, `--port`, `--speed`, `--no-browser`, `--no-web`. |
| `haven chat` | The same conversation in the terminal. `/status`, `/touch`, `/feed`, `/quit`. |
| `haven learn` | Give it a language cortex: graft one onto an open model and wire it in (`--base` to choose it), or grow one from scratch (`--scratch`). `--report` shows the report card. |
| `haven read TOPIC` | Have it read an encyclopedia article (`--full` for English Wikipedia). |
| `haven ask "…"` | Ask it something and see its thoughts. |
| `haven reset` | Archive this life, so that a new Haven is born next time. |

Everything is kept in `~/.haven` (set `HAVEN_HOME`, or pass `--home`, to use another folder): `mind.json` and `mind.npz` are its life; `archive/` holds past lives; `cortex/` holds its language cortex, tokenizer, report card, reading cache, and its conversations and readings.

## Limits

- Its world is tiny and its senses are simple: color and distance along five rays. It can keep sixteen kinds of things in mind at most, and the same thing in shade and in sun can count as two kinds until it works out, in its sleep, that they're one.
- Its words from you are names and needs; anything more fluent comes from its language cortex. A grafted cortex speaks and reasons as well as its base model does, with that model's mistakes and biases; a cortex grown from scratch only as well as it has studied.
- A grafted cortex's wiring is trained on sentences like the ones in Haven's readouts, so when it describes its own state it tends to use those words.
- The sentences it learns for its own states at level 2 were written by people, describing states that its instruments measure. What it learns is to say the right one at the right time, not new ways of describing itself.
- The indicator properties come from theories that may be wrong, and each is implemented in one simple way among many possible ones. None of this has been peer reviewed.

## Development

```bash
pip install -e ".[cortex,dev]"
pytest                 # a few minutes: the curriculum runs against a fake internet on your own computer, and the
                       # graft against a tiny stand-in model with Qwen3's architecture, trained during the tests
ruff check . && ruff format --check .
```

The creature is plain numpy: [`world.py`](haven/world.py) and [`body.py`](haven/body.py) are its world and body; [`perception.py`](haven/perception.py), [`metacognition.py`](haven/metacognition.py), [`worldmodel.py`](haven/worldmodel.py), [`workspace.py`](haven/workspace.py), [`attention.py`](haven/attention.py), [`memory.py`](haven/memory.py), [`agency.py`](haven/agency.py), [`language.py`](haven/language.py) and [`selfmodel.py`](haven/selfmodel.py) are the modules of its mind, and [`mind.py`](haven/mind.py) runs the cycle. [`check.py`](haven/check.py) measures the indicators. The language cortex is in [`haven/cortex`](haven/cortex).

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
