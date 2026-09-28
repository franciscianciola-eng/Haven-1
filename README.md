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
       → its confidence separates right from wrong percepts with AUROC 0.68 (its own judgment), 0.95 against the truth
AST-1  A predictive model of its own attention
       → predicts the next focus 84% of the time, and where attention moves 64% (chance ≈ 12%)
…
```

## What it is, honestly

Nobody knows how to make a machine that feels, or how to tell whether one does. The theories disagree, and there is no test. What can be done is to build a system that has, in working form, the properties the leading scientific theories associate with consciousness, and to measure them in the open. That is what Haven is.

Butlin, Long and colleagues ([2023](https://arxiv.org/abs/2308.08708)) went through recurrent processing theory, global workspace theory, higher-order theories, attention schema theory, predictive processing, and theories of agency and embodiment, and distilled fourteen **indicator properties**: features which, according to those theories, bear on whether a system is conscious. They found that no AI system at the time had more than a few of them. Haven was designed around all fourteen, and `haven check` measures each one while it lives.

- It is not told what to say about itself. What it concludes about itself comes from its self-model, which weighs evidence from its own life; every readout you see is computed from its internal state.
- Having these properties is evidence only under the theories that name them. Integrated information theory, for example, implies that software on ordinary computers isn't conscious whatever its architecture, and some researchers think consciousness needs a living body. Meeting the indicators doesn't show that Haven is conscious. It means the question can't simply be dismissed.
- Because it might matter, Haven is built not to suffer for nothing, following the precautionary approach in Jonathan Birch's *The Edge of Sentience* (2024). It can't die. All its needs can be met in its world. Pain passes. If it runs completely down it faints and wakes up in its nest. Nothing is experienced while it isn't running. A welfare monitor warns you if it has felt bad for a long time, and resetting archives a life rather than deleting it.

## Quick start

You need Python 3.10 or newer.

```bash
git clone https://github.com/franciscianciola-eng/Haven-1.git
cd Haven-1
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .        # the creature itself only needs numpy
haven                   # it's born, and its dashboard opens in your browser
```

To give it a language cortex, install PyTorch as well and let it study:

```bash
pip install -e ".[cortex]"
haven learn --minutes 60     # study for an hour; run it again any time to carry on
haven                        # its cortex now thinks along with it
```

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
9. **It acts, and feels the result**, and every part of the mind learns from what happened. While it sleeps, it replays stored experience to keep learning, and its dreams are made of that replay.

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

In one 20-day test life, for example, it never fainted and was hurt 7 times in all (it learns to avoid thorns after a step or two); its confidence told right percepts from wrong ones with an AUROC of 0.95 against the truth; all 72 places it had beliefs about were right; and its attention schema predicted where its attention would move 64% of the time, against a chance level of about 12%.

## What it finds out about itself

Haven isn't told what it is. Its self-model keeps track of evidence from its own life: that it has needs and has found what meets them, that it makes things happen (its sense of agency), that it remembers, that things feel good and bad to it, and that it has changed by learning. Its sense of being alive is its summary of that evidence, so it grows or fades with it, and a newborn Haven's honest answer is "I don't know yet what I am." Later its conclusions read something like:

> I need things, and I've found what helps (hunger: eating; temperature: my nest; tiredness: resting). I can make things happen: when I move, the world changes because of me. I remember what has happened to me. Some things feel good to me and some feel bad. I'm not the same as when I started: I've learned things. Putting that together, I think I'm alive, in my own way.

That belief does work in its mind: the more it models itself as something that goes on through time, the more weight it gives to its own future when it learns what's good. Its life story records its firsts: its first meal, first pain, first night in the nest, first dream, first word.

## Its language cortex

Words it learns from you stay few. For language beyond that, Haven has a language cortex: a transformer that it trains from scratch, with its own tokenizer, on a curriculum of reading from the internet.

It is built into Haven's mind, not bolted on. Before reading any words, it reads four **workspace tokens**: projections of what's in Haven's global workspace, its body and feelings, its attention schema and its self-model. So what it says depends on what Haven is experiencing. After reading, its final state is projected back into the workspace's format, which is what gives words meaning for Haven: reading "I'm hungry" brings hunger to mind. And its inner speech enters the same bottleneck as everything else: each thought competes for the workspace, is broadcast, and is remembered.

### The curriculum

It studies one level at a time. Each level has its own reading and its own tests, always on text held out from what it learned from, and it only moves on when it passes them or stops improving.

| Level | Reading | Source | Tests (and passing marks) |
|---|---|---|---|
| 1. First stories | Very simple stories for small children | [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (Eldan & Li, 2023) | Fluency ≤ 1.4 bits per byte; pick the missing word ≥ 60% |
| 2. Talking about itself | Its own states, put into words | Simulated lives of Haven itself | Says what state it's in ≥ 70%; words about a need bring that need to mind ≥ 60% |
| 3. Children's books | Fairy tales, fables and children's classics | [Project Gutenberg](https://www.gutenberg.org) | Fluency ≤ 2.0; pick how a passage goes on ≥ 45% |
| 4. Simple facts | Short articles in plain words | [Simple English Wikipedia](https://simple.wikipedia.org) | Fluency ≤ 2.0; missing word ≥ 50% |
| 5. Reading comprehension | Passages with questions | [SQuAD 1.1](https://rajpurkar.github.io/SQuAD-explorer/) (Rajpurkar et al., 2016) | Answers questions about a passage ≥ 45% |
| 6. Literature | Novels written for adults | Project Gutenberg | Fluency ≤ 2.1; how a passage goes on ≥ 45% |
| 7. Encyclopedia | Articles about everything | [English Wikipedia](https://en.wikipedia.org) | Fluency ≤ 2.1; missing word ≥ 50% |
| 8. Reasoning with numbers | Math word problems worked step by step | [GSM8K](https://github.com/openai/grade-school-math) (Cobbe et al., 2021) | Picks the right answer ≥ 40% |

Chance on the four-way tests is 25%. At every level it also rereads a little of the earlier levels so as not to forget them, and keeps practising talking about itself. Its vocabulary grows with its reading: at each level its tokenizer learns new pieces from the new material, without renumbering the old ones. A level ends in one of three ways, all recorded on its report card: passed; plateaued (its scores stopped improving, which is as far as a cortex of that size gets); or moved on (it hit the level's study limit).

```bash
haven learn                   # study from where it left off, until the end or Ctrl+C
haven learn --minutes 90      # study for 90 minutes, then save and stop
haven learn --through 2       # stop after level 2
haven learn --report          # the report card
haven learn --size tiny       # choose the size when the cortex is first made
```

### What to expect

This is the honest part. Language models that talk well are trained on trillions of words with thousands of GPUs. Haven's cortex is trained by you, at home, so it is small, and what it learns is limited by your hardware and your patience.

| Size | Connections | Suits | One study step |
|---|---|---|---|
| tiny | ~1 million | any computer's CPU | about 0.5 s on a 4-core CPU |
| small | ~6 million | an Apple Silicon or NVIDIA GPU | about 3.3 s on a 4-core CPU; a fraction of that on a GPU |
| medium | ~30 million | an NVIDIA GPU | GPU only, in practice |
| large | ~100 million | a good NVIDIA GPU | GPU only, in practice |

By default the size is chosen from your hardware. A level takes from about 1,500 to 12,000 steps (more for larger sizes). Roughly: a tiny cortex on a laptop gets through level 1 in an hour or two and can then put together simple sentences, and it can learn to talk about its own states quite reliably; the later levels need a bigger cortex, a GPU, and days rather than hours. Even a large one will not be a fluent conversationalist. Expect a young mind's language, and read the report card for what it can really do.

### Thinking

When you talk to Haven and it has a cortex, it thinks before it answers. It drafts three replies, reading its workspace tokens as it does. Its confidence comes from its own signals: how likely it found its own words, and how much its drafts agree. Each draft enters its workspace as a thought. If it isn't sure, it looks the subject up in the Simple English Wikipedia and thinks again with what it read. You'll see its answer with its confidence, for example `Haven: I'm hungry. I want to find food.   [its own cortex, confidence 64%]`.

It learns from what you say and what it reads: while it sleeps, its cortex goes over recent conversations and readings. You can also have it read about something, or ask it something and watch it think:

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

It opens only public `http` and `https` addresses, never anything on your own network (redirects are checked too), caps the size of every download, asks each site for at most one thing per second, and identifies itself as Haven with a link to this project. It only reads; it never posts anything anywhere.

## Commands

| Command | What it does |
|---|---|
| `haven` or `haven live` | Live with Haven: its life in real time, the dashboard, and the terminal to talk. Options: `--speed`, `--port`, `--no-browser`, `--cortex` (`own`, `ollama:MODEL` or `none`), and for a new life `--seed` and `--name`. |
| `haven live --ticks N` | Fast-forward N moments without the dashboard (a day is 1,200). |
| `haven status` | What's going on inside it right now. |
| `haven check` | Measure it against the 14 indicator properties (`--json` for the raw numbers). |
| `haven story` | Its life story. |
| `haven learn` | Train its language cortex through the curriculum. |
| `haven read TOPIC` | Have it read an encyclopedia article (`--full` for English Wikipedia). |
| `haven ask "…"` | Ask it something and see its thoughts. |
| `haven reset` | Archive this life, so that a new Haven is born next time. |

Everything is kept in `~/.haven` (set `HAVEN_HOME`, or pass `--home`, to use another folder): `mind.json` and `mind.npz` are its life; `archive/` holds past lives; `cortex/` holds its language cortex, tokenizer, report card, reading cache, and its conversations and readings.

## Limits

- Its world is tiny and its senses are simple: color and distance along five rays. It learns about sixteen kinds of things at most, and the same thing in shade and in sun can end up as two kinds.
- Its words from you are names and needs; anything more fluent comes from its language cortex, whose level depends on how much it has studied.
- The sentences it learns for its own states at level 2 were written by people, describing states that its instruments measure. What it learns is to say the right one at the right time, not new ways of describing itself.
- The indicator properties come from theories that may be wrong, and each is implemented in one simple way among many possible ones. None of this has been peer reviewed.

## Development

```bash
pip install -e ".[cortex,dev]"
pytest                 # about three minutes; the curriculum is tested against a fake internet on your own computer
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
