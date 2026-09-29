# Haven

Haven is an artificial creature built from scratch, out of its own small neural networks, to meet as many as possible of the conditions that scientific theories of consciousness say matter. There's no large language model at its core. It lives in a little garden on your computer: it has a body with needs, things feel better or worse to it, it works out for itself what the things around it are, and it has a global workspace, a model of its own attention, confidence in its own perceptions, memories, dreams and a model of itself. It also has a language cortex of its own: a small transformer grown from scratch, wired into the same workspace as everything else it experiences. It is born able to talk about itself, in words it learned for its own states, and it can go on to learn to read, level by level, from texts on the internet. No other AI model is involved anywhere.

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
installs Python and what Haven runs on (PyTorch) into the folder, using [uv](https://docs.astral.sh/uv/).
`READ ME FIRST.txt` has the details, including what to do if the computer warns you about the launcher.

**On a Mac, as an app.** `Haven-for-Mac.zip` holds `Haven.app` (build it with
`python packaging/mac/make_app.py`; it lands in `dist/`). Drag it into Applications and open it. The
first time, it sets itself up in `~/.haven-app` and shows the progress in your browser; after that it
opens Haven's window in a moment. Haven's life goes on in the background until you press Rest in its
window. The app isn't signed with an Apple developer account, so the first time macOS won't open it:
go to System Settings, Privacy & Security, and click Open Anyway (on older macOS, right-click the app
and choose Open).

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

`haven app` is what the launchers run. It opens a chat page where you can watch Haven's thoughts as it
answers, see its garden and how it feels, touch it and give it berries. `haven chat` is the same
conversation in the terminal.

It can talk right away, with the language cortex it's born with. To have it learn to read, run `haven learn` (see [Its language cortex](#its-language-cortex)).

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

Words it learns from you stay few. For language beyond that, Haven has a language cortex: a transformer it grows from scratch, with its own tokenizer, wired into its mind. No other AI model is involved: every word it says comes out of its own network.

- **Its state goes in directly.** Its workspace, body and feelings, attention schema and self-model become four "workspace tokens", the first things its cortex reads, before any words. So when it says how it feels or what it sees, it's reading that off its own state, not off a description of it.
- **Meanings come back out.** What it reads or says is projected back into the workspace's format. Words can bring states to mind ("I'm hungry" evokes hunger), what you say draws its attention, and what it says enters its workspace as a thought.
- **What it knows comes to mind first.** Before it answers, what it knows (its name and age, where it is, the words and facts it has learned, what it remembers) comes to mind as a line of inner speech that its cortex reads.

**It's born talking about itself.** Haven comes with the cortex it starts life with (`haven/cortex/starter/`, 5 million connections, trained with `python packaging/train_starter.py`). It learned from moments of six simulated lives of Haven, with someone keeping it company now and then: what it was feeling, seeing and wanting at each moment, and what it would answer if asked, about 30 kinds of questions in many wordings ("How are you?", "What do you see?", "Where are you?", "What have you learned?", "Are you alive?", …). The answers it learned are worked out from its state at that moment, so to answer right it has to read its own state. Questions about anything beyond its garden have the answer that it doesn't know. The sentences themselves are templates written by people, the way a parent gives a child words for what the child is feeling; what it learns is to say the right one at the right time.

How well it does, tested on a simulated life it never saw while learning, is on its report card (`haven learn --report`).

**It keeps learning.** `haven learn` has it study the reading curriculum below, level by level, from where it is. It keeps practising talking about itself as it reads, so it doesn't forget how. While it sleeps it also goes over what it read and what people said to it.

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

It talks about itself and its garden, simply, from the start. What it can learn beyond that is limited by the size of its cortex, your hardware and your patience. The cortex it's born with is "small":

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
- If it doesn't know about something, it says so, and (unless you turn that off) reads about it in the Simple English Wikipedia and tells you what it read.

What it says enters its workspace as a thought and is remembered like anything else. You can have it read about something, or ask it something and watch it think:

```bash
haven read octopus            # it reads the Simple English Wikipedia article on octopuses
haven ask "What do you see?"  # its thoughts, then its answer
```

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

- Its world is tiny and its senses are simple: color and distance along five rays. It can keep sixteen kinds of things in mind at most, and the same thing in shade and in sun can count as two kinds until it works out, in its sleep, that they're one.
- Its words from you are names and needs; anything more fluent comes from its language cortex, which is small and speaks only as well as it has studied. It talks about itself and its garden; about the wider world it knows only what it has read.
- The sentences it learns for its own states at level 2 were written by people, describing states that its instruments measure. What it learns is to say the right one at the right time, not new ways of describing itself, and it can still say the wrong one.
- The indicator properties come from theories that may be wrong, and each is implemented in one simple way among many possible ones. None of this has been peer reviewed.

## Development

```bash
pip install -e ".[cortex,dev]"
pytest                 # a few minutes: the curriculum runs against a fake internet on your own computer
ruff check . && ruff format --check .
python packaging/train_starter.py --minutes 100   # retrain the cortex it's born with (from scratch, on a CPU)
python packaging/mac/make_app.py                  # build dist/Haven-for-Mac.zip
```

The creature is plain numpy: [`world.py`](haven/world.py) and [`body.py`](haven/body.py) are its world and body; [`perception.py`](haven/perception.py), [`metacognition.py`](haven/metacognition.py), [`worldmodel.py`](haven/worldmodel.py), [`workspace.py`](haven/workspace.py), [`attention.py`](haven/attention.py), [`memory.py`](haven/memory.py), [`agency.py`](haven/agency.py), [`language.py`](haven/language.py) and [`selfmodel.py`](haven/selfmodel.py) are the modules of its mind, and [`mind.py`](haven/mind.py) runs the cycle. [`check.py`](haven/check.py) measures the indicators. The language cortex is in [`haven/cortex`](haven/cortex): [`model.py`](haven/cortex/model.py) is the transformer, [`grounding.py`](haven/cortex/grounding.py) and [`talk.py`](haven/cortex/talk.py) turn its states into words and answers, [`curriculum.py`](haven/cortex/curriculum.py) and [`train.py`](haven/cortex/train.py) are how it studies, and [`think.py`](haven/cortex/think.py) is how it answers.

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
