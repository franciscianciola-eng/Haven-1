"""Haven's brain: spiking neurons wired into the regions a mammal's brain has.

Every region is made of the neurons in neurons.py, and everything it does, it does by
neurons firing and synapses changing:

- The thalamus relays what Haven senses to the cortex. Each thing it can sense (a kind
  of thing it sees, a word it hears, a place, a feeling, what it's doing) gets a few relay
  cells of its own the first time it comes along. Their synapses tire with use and recover
  with time (habituation, as in Kandel's Aplysia): what it has met a lot evokes less.
- The cortex is a sheet of minicolumns of a hundred cells: pyramidal cells of three kinds
  and two kinds of interneurons, as in a real column. Every pyramidal cell has thousands
  of synapses onto cells of its own column, its neighbours and columns far away (about
  as many as a human cortical neuron has). Each thing it senses excites a sparse set of
  cells in a few columns. Synapses between cells that fire together strengthen by
  spike-timing-dependent plasticity, so things it meets together come to call each other
  up (cell assemblies).
- The amygdala learns what to fear: what it senses when it's hurt or sick becomes able,
  by itself, to excite the central amygdala, which raises the alarm.
- The basal ganglia choose what to do. Each thing it could do has its own channel: medium
  spiny neurons of the direct pathway (D1, go) and the indirect pathway (D2, no-go), the
  globus pallidus, and the motor thalamus that is let go when a channel wins. Dopamine
  makes go-cells stronger and no-go cells weaker, and teaches the channels what tends to
  turn out well (three-factor learning: an eligibility trace waiting for dopamine).
- Five small nuclei of pacemaker cells set the brain's chemistry: dopamine (the ventral
  tegmental area: things turning out better than expected, and novelty), noradrenaline
  (the locus coeruleus: surprise, pain, alarm), serotonin (the raphe: being content and
  safe), acetylcholine (the basal forebrain: awake and attending; low in deep sleep) and
  oxytocin (the hypothalamus: someone kind being there, touch). Their levels are its moods,
  and they change how the rest of the brain works and learns.

In sleep the senses go quiet and the hippocampal replay of the day's important moments
plays through the cortex, which learns from it; synapses shrink a little overnight
(Tononi and Cirelli's synaptic homeostasis) and the senses recover from the day.

Haven's brain runs a short stretch of brain time (MOMENT ms) at every moment of its life.
How big it is depends on the computer: see SIZES and fit().
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .neurons import SIZES as LEVELS
from .neurons import Network, Projection

OPTIONS = (
    "food",
    "warmth",
    "healing",
    "sleep",
    "explore",
    "play",
    "watch",
    "sing",
    "dance",
    "chase",
    "visit",
    "company",
)
CHEMICALS = ("dopamine", "noradrenaline", "serotonin", "acetylcholine", "oxytocin")
# size: (cortical columns, target columns of each column, thalamic relay cells)
SIZES = {
    "tiny": (64, 12, 1024),  # 9,040 neurons, 6.8 million synapses (for tests)
    "small": (256, 24, 2048),  # 29,264 neurons, 51 million synapses
    "standard": (1024, 50, 4096),  # 108,112 neurons, 417 million synapses
    "large": (2048, 64, 8192),  # 214,608 neurons, 1.06 billion synapses
    "huge": (4096, 96, 16384),  # 427,600 neurons, 3.17 billion synapses
}
MOMENT = 12  # ms of brain time at each moment of its life
THREADS = 1  # cores its brain computes on, in the thread it lives in (more fight the language cortex for them)
COLUMN = (("RS", 64), ("IB", 8), ("CH", 8), ("FS", 15), ("LTS", 5))  # a minicolumn: 80 pyramidal cells, 20 interneurons
E_CELLS, CELLS = 80, 100
RELAYS = 8  # thalamic relay cells for each thing it senses
HOME_COLUMNS, HOME_CELLS = 8, 20  # a thing excites 20 pyramidal cells in each of 8 columns
LA_E, LA_I, CEA = 400, 80, 80  # amygdala: lateral (pyramidal-like and interneurons) and central
LA_HOME = 40  # lateral amygdala cells each thing it senses reaches
LA_FF = 8  # and lateral amygdala interneurons (feedforward inhibition)
MSNS = 24  # medium spiny neurons of each pathway, per channel
PALLIDUM, MOTOR = 6, 8  # globus pallidus (external, internal) cells and motor thalamus cells, per channel
STN = 40  # subthalamic nucleus cells
NUCLEUS = 40  # pacemaker cells in each neuromodulatory nucleus
UNIT = {  # nS of the largest synapse of each projection
    "cortex": 2.0,
    "cortex inhibition": 20.0,
    "thalamus": 2.5,
    "amygdala": 6.0,
    "fear": 3.0,
    "feedforward": 4.0,
    "alarm": 0.2,
    "striatum": 3.0,
    "pathways": 20.0,
    "pallidum": 5.0,
    "pallidal output": 40.0,
    "hyperdirect": 3.0,
    "collaterals": 12.0,
}
BACKGROUND = (0.1, 1.5)  # background synaptic bombardment: events per ms per cell, nS per 100 pF
TARGET = {"cortex E": 0.5, "cortex I": 4.0}  # spontaneous rates (Hz) homeostasis keeps them near
EPISODES = 240  # important moments of a day kept for replay in its sleep
REST_MSN = 260.0  # pA: the cortex's steady input to spiny cells (urges come on top of it)
LOOP = 0.25  # how much more urge the action being done gets, from its own loop back through cortex and thalamus
USUAL_RATE = 4.0  # Hz: how fast the nuclei's pacemaker cells fire at their usual level
NOVELTY_USUAL = 0.4  # how new an ordinary waking moment feels (the nuclei answer what's newer than that)
FEAR_STEP = 20.0  # sizes a synapse onto the lateral amygdala grows in a moment of hurt (fear can take one lesson)
EXTINCTION = 0.03  # the chance it shrinks a size in a moment of meeting it unharmed, close (fear fades, slowly)
CUES = ("see:", "word:", "event:", "place:")  # what fear can be learned for (not what it does, or how it feels)
STDP_WINDOW, STDP_STEP = 30, 3.0  # ms within which two spikes count as a pair; most sizes one pair moves a synapse
PACEMAKER = 58.0  # pA: the steady current that keeps them at it
# Habituation of relay synapses, per moment of use, and recovery per moment (a moment is about a minute of its day):
QUICK_TIRING, QUICK_RECOVERY = 0.035, 0.01  # bored of something in half an hour; over it in a couple of hours
SLOW_TIRING, SLOW_RECOVERY = 0.002, 0.0002  # and over days, what it meets all the time becomes familiar


def stable_hash(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")


@dataclass
class Sense:
    """Something it can sense, with the relay cells and cortical cells it has for it."""

    name: str
    relays: torch.Tensor  # thalamic cells (network numbers)
    home: torch.Tensor  # cortical pyramidal cells it excites (network numbers)
    met: int = 0  # moments it has sensed it
    last: int = 0  # the last moment it did
    novelty: float = 1.0  # how new it feels, lately (its cortical cells' answer against a new thing's)
    seen: int = 0  # the last moment its novelty was measured


@dataclass
class Reading:
    """What the brain makes of one moment."""

    novelty: float = 0.0  # how new what it senses is, 0 (all known and worn out) to 1 (all new)
    new: list[str] = field(default_factory=list)  # what it senses that's new or nearly so
    fear: float = 0.0  # alarm in the central amygdala, 0 to 1
    choice: str | None = None  # what the basal ganglia let it do
    channels: dict[str, float] = field(default_factory=dict)  # how strongly each channel's motor thalamus fired
    chemistry: dict[str, float] = field(default_factory=dict)  # levels, 1 = usual
    rates: dict[str, float] = field(default_factory=dict)  # Hz, by region
    spikes: int = 0
    ms: float = 0.0  # how long the moment took to compute (wall-clock ms)


def one_core() -> None:
    """Compute the brain on THREADS cores in this thread (the setting is the thread's own). Spread over every core,
    alongside the language cortex or anything else the computer is doing, its many small steps wait on each other and
    it slows down a hundredfold."""
    if torch.get_num_threads() != THREADS:
        torch.set_num_threads(THREADS)


class Brain:
    def __init__(self, seed: int = 0, size: str = "standard", fresh: bool = True):
        """A newborn brain (or, with fresh=False, the wiring only, for load() to fill in)."""
        one_core()
        self.seed, self.size = seed, size
        self.fresh = fresh
        columns, targets, relays = SIZES[size]
        self.ncols, self.C, self.nrelays = columns, targets, relays
        self.gen = torch.Generator().manual_seed(seed + 11)
        net = self.net = Network(seed + 12)
        self.cortex = net.population("cortex", list(COLUMN) * columns, align=True)
        self.thalamus = net.population("thalamus", [("TC", relays)])
        self.la = net.population("lateral amygdala", [("RS", LA_E), ("FS", LA_I)])
        self.cea = net.population("central amygdala", [("RS", CEA)])
        n = len(OPTIONS)
        self.d1 = net.population("striatum D1", [("MSN", MSNS * n)])
        self.d2 = net.population("striatum D2", [("MSN", MSNS * n)])
        self.gpe = net.population("globus pallidus externa", [("GP", PALLIDUM * n)])
        self.gpi = net.population("globus pallidus interna", [("GP", PALLIDUM * n)])
        self.motor = net.population("motor thalamus", [("TC", MOTOR * n)])
        self.stn = net.population("subthalamic nucleus", [("RS", STN)])
        self.nuclei = {
            chem: net.population(name, [("PM", NUCLEUS)])
            for chem, name in zip(
                CHEMICALS,
                ("ventral tegmental area", "locus coeruleus", "raphe nuclei", "basal forebrain", "hypothalamus"),
                strict=True,
            )
        }
        self._wire()
        net.background = BACKGROUND
        net.build()
        if fresh:
            self._settle()
        self.senses: dict[str, Sense] = {}
        self.free = list(range(relays // RELAYS))  # groups of relay cells not yet given to anything
        self.moments = 0
        self.levels = dict.fromkeys(CHEMICALS, 1.0)
        self.episodes: list[dict[str, float]] = []  # important moments of the day, for replay in sleep
        self.channel_rates = torch.zeros(n)
        self.current: str | None = None
        self.eligibility = torch.zeros_like(self.striatal.sizes, dtype=torch.float32)
        self.timing = 0.0  # running average of wall-clock ms per moment
        self._noise = None  # (for comparing two thoughts)
        self.changed = True  # (the cortex's synapses changed since they were last saved)

    # --- wiring --------------------------------------------------------------------------------------------

    def _wire(self) -> None:
        net, g = self.net, self.gen
        cols, C = self.ncols, self.C
        side = max(1, int(math.sqrt(cols)))
        xy = torch.stack([torch.arange(cols) % side, torch.arange(cols) // side], 1).float()
        # Each column reaches itself, its four nearest neighbours, and columns anywhere in the sheet.
        local = torch.cdist(xy, xy).argsort(1)[:, :5]
        far = torch.randint(0, cols, (cols, C - 5), generator=g)
        targets = torch.cat([local, far], 1)
        distance = torch.gather(torch.cdist(xy, xy), 1, targets)
        delays = (1 + distance * 15 / max(side, 1)).clamp(max=15).long()
        # Most synapses between pyramidal cells start weak, and many silent: what they come to carry is learned.
        if self.fresh:  # (with its own random numbers, so the rest of the wiring is the same either way)
            own = torch.Generator().manual_seed(self.seed + 13)
            sizes = torch.empty((cols * E_CELLS, C, CELLS), dtype=torch.uint8)
            for part in torch.split(sizes, 2048):  # (a part at a time, so growing takes little more memory than it)
                part.copy_(torch.randint(1, 7, part.shape, generator=own))
                part[torch.rand(part.shape, generator=own) < 0.6] = 0  # silent synapses
        else:
            sizes = torch.zeros((cols * E_CELLS, C, CELLS), dtype=torch.uint8)  # (load() fills them in)
        sizes[:, :, E_CELLS:] = torch.randint(
            14, 19, (cols * E_CELLS, C, CELLS - E_CELLS), dtype=torch.uint8, generator=g
        )
        self.recurrent = net.connect(
            Projection(
                "cortex",
                self.cortex,
                sizes,
                UNIT["cortex"],
                delays,
                columns=self.cortex.start // CELLS + targets,
                per_column=E_CELLS,
                stride=CELLS,
                width=CELLS,
            )
        )
        self.column_targets = targets
        # Interneurons inhibit their own column and, more sparsely, the four next to it.
        ni = cols * (CELLS - E_CELLS)
        rows = torch.full((self.cortex.size,), -1, dtype=torch.long)
        icells = (torch.arange(cols).unsqueeze(1) * CELLS + E_CELLS + torch.arange(CELLS - E_CELLS)).reshape(-1)
        rows[icells] = torch.arange(ni)
        own = icells // CELLS
        reach = [own.unsqueeze(1) * CELLS + torch.arange(CELLS)]
        for k in range(1, 5):
            reach.append(local[own, k].unsqueeze(1) * CELLS + torch.randint(0, CELLS, (ni, 50), generator=g))
        reached = torch.cat(reach, 1).int() + self.cortex.start
        net.connect(
            Projection(
                "cortex inhibition",
                self.cortex,
                torch.randint(8, 16, reached.shape, dtype=torch.uint8, generator=g),
                UNIT["cortex inhibition"],
                torch.ones(ni, dtype=torch.long),
                targets=reached,
                rows=rows,
                excitatory=False,
                plastic=False,
            )
        )
        # Thalamus to cortex: given out as it comes to sense things (see _sense); until then, nowhere.
        width = HOME_COLUMNS * (HOME_CELLS + 2)
        self.relay = net.connect(
            Projection(
                "thalamus",
                self.thalamus,
                torch.zeros((self.nrelays, width), dtype=torch.uint8),
                UNIT["thalamus"],
                torch.full((self.nrelays,), 2, dtype=torch.long),
                targets=torch.zeros((self.nrelays, width), dtype=torch.int32),
                plastic=False,
            )
        )
        self.slow = torch.ones(self.nrelays)  # long-term habituation of each relay cell (days)
        self.cue = torch.zeros(self.nrelays, dtype=torch.bool)  # relay cells of things fear can be learned for
        self.quick = torch.ones(self.nrelays)  # short-term (minutes to hours)
        # Thalamus to the lateral amygdala: each thing it senses reaches a few dozen cells there, weakly at first;
        # what goes with being hurt grows strong (given out with the senses, in sense_of).
        self.fearful = net.connect(
            Projection(
                "amygdala",
                self.thalamus,
                torch.zeros((self.nrelays, LA_HOME), dtype=torch.uint8),
                UNIT["amygdala"],
                torch.full((self.nrelays,), 3, dtype=torch.long),
                targets=torch.full((self.nrelays, LA_HOME), self.la.start, dtype=torch.int32),
            )
        )
        # ...and a few of its interneurons, strongly: they hold the lateral amygdala back in proportion to all it
        # senses (feedforward inhibition), so only what has come to be frightening gets through.
        self.feedforward = net.connect(
            Projection(
                "amygdala feedforward",
                self.thalamus,
                torch.zeros((self.nrelays, LA_FF), dtype=torch.uint8),
                UNIT["feedforward"],
                torch.full((self.nrelays,), 2, dtype=torch.long),
                targets=torch.full((self.nrelays, LA_FF), self.la.start + LA_E, dtype=torch.int32),
                plastic=False,
            )
        )
        net.connect(  # lateral to central amygdala
            Projection(
                "fear",
                self.la,
                torch.randint(12, 18, (LA_E, 40), dtype=torch.uint8, generator=g),
                UNIT["fear"],
                torch.full((LA_E,), 2, dtype=torch.long),
                targets=(self.cea.start + torch.randint(0, CEA, (LA_E, 40), generator=g)).int(),
                rows=torch.cat([torch.arange(LA_E), torch.full((LA_I,), -1)]),
                plastic=False,
            )
        )
        net.connect(  # the amygdala's interneurons keep it sparse
            Projection(
                "amygdala inhibition",
                self.la,
                torch.randint(10, 16, (LA_I, 60), dtype=torch.uint8, generator=g),
                UNIT["cortex inhibition"],
                torch.ones(LA_I, dtype=torch.long),
                targets=(self.la.start + torch.randint(0, LA_E, (LA_I, 60), generator=g)).int(),
                rows=torch.cat([torch.full((LA_E,), -1), torch.arange(LA_I)]),
                excitatory=False,
                plastic=False,
            )
        )
        net.connect(  # alarm: the central amygdala excites the locus coeruleus
            Projection(
                "alarm",
                self.cea,
                torch.randint(14, 20, (CEA, 20), dtype=torch.uint8, generator=g),
                UNIT["alarm"],
                torch.full((CEA,), 3, dtype=torch.long),
                targets=(self.nuclei["noradrenaline"].start + torch.randint(0, NUCLEUS, (CEA, 20), generator=g)).int(),
                plastic=False,
            )
        )
        # Thalamus to striatum: what it senses, to every channel's go- and no-go cells (learned by dopamine).
        msn = torch.cat([torch.arange(self.d1.start, self.d1.stop), torch.arange(self.d2.start, self.d2.stop)])
        self.striatal = net.connect(
            Projection(
                "striatum",
                self.thalamus,
                torch.randint(4, 9, (self.nrelays, 24), dtype=torch.uint8, generator=g),
                UNIT["striatum"],
                torch.full((self.nrelays,), 3, dtype=torch.long),
                targets=msn[torch.randint(0, msn.numel(), (self.nrelays, 24), generator=g)].int(),
            )
        )
        # The two pathways, channel by channel: D1 -| GPi; D2 -| GPe -| GPi; GPi -| motor thalamus.
        n = len(OPTIONS)
        channel = torch.arange(n).repeat_interleave(MSNS)
        pal = torch.arange(n).repeat_interleave(PALLIDUM)

        def onto(pop, per, owner) -> torch.Tensor:  # every cell of the same channel in `pop`
            return (pop.start + owner.unsqueeze(1) * per + torch.arange(per)).int()

        for name, pre, post, per, owner, unit in (
            ("direct pathway", self.d1, self.gpi, PALLIDUM, channel, UNIT["pathways"]),
            ("indirect pathway", self.d2, self.gpe, PALLIDUM, channel, UNIT["pathways"]),
            ("pallidum", self.gpe, self.gpi, PALLIDUM, pal, UNIT["pallidum"]),
            ("pallidum to thalamus", self.gpi, self.motor, MOTOR, pal, UNIT["pallidal output"]),
        ):
            reached = onto(post, per, owner)
            net.connect(
                Projection(
                    name,
                    pre,
                    torch.full(reached.shape, 20, dtype=torch.uint8),
                    unit,
                    torch.full((pre.size,), 2, dtype=torch.long),
                    targets=reached,
                    excitatory=False,
                    fast=True,
                    plastic=False,
                )
            )
        # The hyperdirect pathway: the subthalamic nucleus, stirred by every urge at once, excites the whole internal
        # pallidum, so only a channel whose go-cells fire hard enough gets through (Gurney, Prescott & Redgrave 2001).
        gpi = torch.arange(self.gpi.start, self.gpi.stop)
        net.connect(
            Projection(
                "hyperdirect pathway",
                self.stn,
                torch.full((STN, 40), 18, dtype=torch.uint8),
                UNIT["hyperdirect"],
                torch.full((STN,), 2, dtype=torch.long),
                targets=gpi[torch.randint(0, gpi.numel(), (STN, 40), generator=g)].int(),
                plastic=False,
            )
        )
        # Spiny cells' collaterals inhibit the go-cells of other channels: the channels compete.
        d1 = torch.arange(self.d1.start, self.d1.stop)
        own = torch.arange(n).repeat_interleave(MSNS)
        others = torch.randint(0, d1.numel(), (d1.numel(), 30), generator=g)
        others = torch.where(own[others] == own.unsqueeze(1), (others + MSNS) % d1.numel(), others)
        net.connect(
            Projection(
                "striatal collaterals",
                self.d1,
                torch.full((d1.numel(), 30), 18, dtype=torch.uint8),
                UNIT["collaterals"],
                torch.ones(d1.numel(), dtype=torch.long),
                targets=d1[others].int(),
                excitatory=False,
                fast=True,
                plastic=False,
            )
        )

    def _settle(self) -> None:
        """Development: each region's cells find their resting excitability, so that with nothing to sense they fire
        as their kind does in a resting brain: pacemakers tick over at a few Hz (each at its own pace), the pallidum
        fires fast and holds the motor thalamus back, spiny cells and relay cells are nearly silent. Then it learns
        what something new feels like: how strongly the cortex answers a thing never met before (the measure of
        novelty)."""
        net, bias = self.net, self.net.bias
        for pop in (*self.nuclei.values(), self.gpe, self.gpi, self.motor):
            net.noise[pop.slice] *= 0.0 if pop in self.nuclei.values() else 0.3  # (pacemakers keep their own time)
            net.v[pop.slice] = -70.0 + 30.0 * torch.rand(pop.size, generator=self.gen)
        for pop in self.nuclei.values():  # about 4 Hz (their f-I curve: 60 pA at rheobase, 0.22 Hz/pA above it)
            bias[pop.slice] = PACEMAKER + 6.0 * torch.randn(pop.size, generator=self.gen)
        jitter = 1.0 + 0.08 * torch.randn(self.gpe.size, generator=self.gen)  # (each cell at its own pace)
        bias[self.gpe.slice] = 100.0 * jitter  # about 50 Hz
        bias[self.gpi.slice] = 135.0 * jitter  # about 60 Hz, held back a little by the externa
        bias[self.motor.slice] = 220.0 * (
            1.0 + 0.15 * torch.randn(self.motor.size, generator=self.gen)
        )  # ~30 Hz if let
        rest = {"thalamus": 0.5, "lateral amygdala": 0.5, "central amygdala": 1.0, "subthalamic nucleus": 10.0}
        msn = torch.cat([torch.arange(self.d1.start, self.d1.stop), torch.arange(self.d2.start, self.d2.stop)])
        window = 50
        for rounds in range(24):
            net.begin()
            net.drive.zero_()
            net.drive[msn] = REST_MSN
            net.drive[self.stn.slice] = 60.0
            net.prepare()
            for _ in range(window):
                net.step()
            for name, hz in rest.items():
                pop = net.populations[name]
                rate = net.rate(pop, window)
                bias[pop.slice] += max(-30.0, min(30.0, (6.0 if rounds < 12 else 2.0) * (hz - rate)))
            self._homeostasis(window)
        net.drive.zero_()
        self.base_bias = bias.clone()
        # What something new feels like, at full strength and at half.
        self.reference = {}
        for intensity in (1.0, 0.5):
            probe = Sense("(new)", self.thalamus.start + torch.arange(RELAYS), torch.empty(0))
            self.senses = {}
            self.free = [0]
            evoked = []
            for _ in range(6):
                self.senses, self.free = {}, [0]
                self.quick[:RELAYS] = 1.0
                sense = self.sense_of("(calibration)")
                probe = sense
                net.begin()
                net.drive.zero_()
                net.drive[probe.relays] = 150.0 + 250.0 * intensity
                net.drive[msn] = REST_MSN
                net.prepare()
                for _ in range(MOMENT):
                    net.step()
                evoked.append(float(net.spikes[probe.home].float().mean()) * 1000.0 / MOMENT)
            self.reference[intensity] = max(1.0, sum(evoked[2:]) / len(evoked[2:]))
        net.drive.zero_()

    # --- senses ---------------------------------------------------------------------------------------------

    def sense_of(self, name: str) -> Sense:
        """The cells it has for something it senses (given the first time it comes along)."""
        found = self.senses.get(name)
        if found is not None:
            return found
        if not self.free:  # every relay cell is taken: the thing it sensed longest ago gives up its cells
            oldest = min(self.senses.values(), key=lambda s: s.last)
            del self.senses[oldest.name]
            group = int((oldest.relays[0] - self.thalamus.start) // RELAYS)
        else:
            group = self.free.pop(0)
        rng = torch.Generator().manual_seed(stable_hash(f"{self.seed}:{name}") % (2**63))
        columns = torch.randperm(self.ncols, generator=rng)[:HOME_COLUMNS]
        cells = torch.stack([torch.randperm(E_CELLS, generator=rng)[:HOME_CELLS] for _ in range(HOME_COLUMNS)])
        home = (self.cortex.start + columns.unsqueeze(1) * CELLS + cells).reshape(-1)
        basket = (self.cortex.start + columns.unsqueeze(1) * CELLS + E_CELLS + torch.arange(2)).reshape(-1)
        relays = self.thalamus.start + group * RELAYS + torch.arange(RELAYS)
        local = relays - self.thalamus.start
        self.relay.targets[local] = torch.cat([home, basket]).int().unsqueeze(0)
        self.relay.sizes[local] = torch.randint(
            18, 23, (RELAYS, home.numel() + basket.numel()), dtype=torch.uint8, generator=rng
        )
        self.slow[local] = 1.0
        self.quick[local] = 1.0
        self.cue[local] = name.startswith(CUES)
        la = self.la.start + torch.randperm(LA_E, generator=rng)[:LA_HOME]
        self.fearful.targets[local] = la.int().unsqueeze(0)
        self.fearful.sizes[local] = torch.randint(1, 5, (RELAYS, LA_HOME), dtype=torch.uint8, generator=rng)
        inter = self.la.start + LA_E + torch.randperm(LA_I, generator=rng)[:LA_FF]
        self.feedforward.targets[local] = inter.int().unsqueeze(0)
        self.feedforward.sizes[local] = torch.randint(16, 21, (RELAYS, LA_FF), dtype=torch.uint8, generator=rng)
        self.net.sync(self.relay, local, targets=True)
        self.net.sync(self.fearful, local, targets=True)
        self.net.sync(self.feedforward, local, targets=True)
        sense = Sense(name, relays, home)
        self.senses[name] = sense
        return sense

    def freshness(self, name: str) -> float:
        """How fresh something would feel: how much transmitter its relay cells have ready (1 = new or rested)."""
        sense = self.senses.get(name)
        if sense is None:
            return 1.0
        local = sense.relays - self.thalamus.start
        return float((self.quick[local] * self.slow[local]).mean())

    # --- one moment -----------------------------------------------------------------------------------------

    def moment(
        self,
        senses: dict[str, float],
        signals: dict[str, float] | None = None,
        urges: dict[str, float] | None = None,
        learn: bool = True,
        asleep: bool = False,
        ms: int = MOMENT,
        imagine: bool = False,
    ) -> Reading:
        """Run a moment of brain time with these senses (name: intensity 0 to 1).

        `signals`: what the rest of the mind reports, for the nuclei: "reward" (how much better than expected things
        turned out, -1 to 1), "surprise", "pain", "sick", "content", "social" (someone kind is there), "touch".
        `urges`: how strongly the body and mind push for each of the OPTIONS, 0 to 1.
        """
        started = time.perf_counter()
        one_core()
        signals = signals or {}
        net = self.net
        net.begin()
        self.moments += 1
        drive = net.drive
        drive.zero_()
        present = []
        for name, intensity in senses.items():
            if intensity <= 0:
                continue
            sense = self.sense_of(name)
            if not imagine:
                sense.met += 1
                sense.last = self.moments
            present.append((sense, float(min(intensity, 1.0))))
            drive[sense.relays] = 150.0 + 250.0 * min(intensity, 1.0)
        for home, current in getattr(self, "_top_down", {}).values():
            drive[home] = current
        local = None
        if present:
            local = torch.cat([s.relays for s, _ in present]) - self.thalamus.start
        net.release[self.thalamus.slice] = (self.quick * self.slow).clamp(min=0.02)
        # The body's hurt reaches the lateral amygdala directly (the unconditioned stimulus), and silences its
        # interneurons for the moment (VIP cells' disinhibitory gating: Krabbe et al. 2019), so it gets through.
        hurt = max(signals.get("pain", 0.0), signals.get("sick", 0.0), signals.get("fright", 0.0))
        drive[self.la.start : self.la.start + LA_E] = 600.0 * hurt
        drive[self.la.start + LA_E : self.la.stop] = -400.0 * hurt
        self._chemistry_drive(signals, asleep)
        self._channels(urges or {})
        net.prepare()
        for _ in range(ms):
            net.step()
        reading = self._read(present, ms, imagine)
        if learn and not asleep:
            self._habituate(local, present)
        if learn:
            self._learn(present, hurt, ms)
        if not asleep and not imagine:
            importance = reading.novelty + abs(signals.get("reward", 0.0)) + hurt + 0.5 * signals.get("social", 0.0)
            if present and importance > 0.6:
                self.episodes = [*self.episodes[-(EPISODES - 1) :], {s.name: i for s, i in present}]
        if not imagine:
            self._homeostasis(ms)
            reading.ms = (time.perf_counter() - started) * 1000
            self.timing = 0.95 * self.timing + 0.05 * reading.ms if self.timing else reading.ms
        return reading

    def _chemistry_drive(self, signals: dict[str, float], asleep: bool) -> None:
        """Input to the five nuclei from what the mind reports (their pacemaker bias keeps them ticking over)."""
        g = signals.get
        new = max(0.0, getattr(self, "_novelty", 0.0) - NOVELTY_USUAL) / (1.0 - NOVELTY_USUAL)  # (beyond the usual)
        inputs = {
            "dopamine": 40.0 * g("reward", 0.0) + 30.0 * new * g("curious", 1.0),
            "noradrenaline": 25.0 * g("surprise", 0.0) + 40.0 * g("pain", 0.0) + 10.0 * new - (30.0 if asleep else 0.0),
            "serotonin": 36.0 * (g("content", 0.5) - 0.5) - 50.0 * g("pain", 0.0) - 30.0 * g("sick", 0.0),
            "acetylcholine": (-40.0 if asleep else 0.0) + 25.0 * new,
            "oxytocin": 15.0 * g("social", 0.0) + 60.0 * g("touch", 0.0),
        }
        for chem, pop in self.nuclei.items():
            self.net.drive[pop.slice] = inputs[chem]

    def _channels(self, urges: dict[str, float]) -> None:
        """Each channel's go- and no-go cells are pushed by how strongly it's urged, against how strongly everything is
        (divisive normalization, as the cortex does with what it passes on); dopamine tips them toward go."""
        net, dopamine = self.net, self.levels["dopamine"]
        strongest = max([v for k, v in urges.items() if k in OPTIONS] + [0.15])
        for i, option in enumerate(OPTIONS):
            urge = float(min(max(urges.get(option, 0.0), 0.0), 1.5)) / strongest  # (against the strongest)
            urge = 0.9 * min(1.0, max(0.0, (urge - 0.55) / 0.45))  # (what isn't nearly as urgent stays quiet)
            go = slice(self.d1.start + i * MSNS, self.d1.start + (i + 1) * MSNS)
            stop = slice(self.d2.start + i * MSNS, self.d2.start + (i + 1) * MSNS)
            fresh = self.freshness(f"doing {option}") if option == self.current else 1.0  # (bored of it, or not)
            if option == self.current:  # the loop through cortex and thalamus keeps a chosen action going
                urge += LOOP
            net.drive[go] = REST_MSN + 450.0 * urge * fresh
            net.drive[stop] = REST_MSN + 150.0 * urge
        total = sum(min(max(v, 0.0), 1.5) for k, v in urges.items() if k in OPTIONS)
        net.drive[self.stn.slice] = 60.0 + 120.0 * total  # (the hyperdirect pathway: all urges at once)
        net.gain[self.d1.slice] = 1.0 + 0.4 * (dopamine - 1.0)
        net.gain[self.d2.slice] = max(0.2, 1.0 - 0.4 * (dopamine - 1.0))

    # --- reading the moment --------------------------------------------------------------------------------

    def _read(self, present: list[tuple[Sense, float]], ms: int, imagine: bool = False) -> Reading:
        net = self.net
        spikes = net.spikes
        r = Reading(spikes=int(spikes.sum()))
        r.rates = {
            "cortex": net.rate(self.cortex, ms),
            "thalamus": net.rate(self.thalamus, ms),
            "amygdala": net.rate(self.la, ms),
            "striatum": (net.rate(self.d1, ms) + net.rate(self.d2, ms)) / 2,
            "pallidum": net.rate(self.gpi, ms),
        }
        # Novelty: how strongly the cortical cells of what it senses fire, against how they fire to something new.
        news, weights = [], []
        for sense, intensity in present:
            evoked = float(spikes[sense.home].float().mean()) * 1000.0 / ms
            usual = self.reference[0.5] + (self.reference[1.0] - self.reference[0.5]) * (intensity - 0.5) / 0.5
            measured = min(1.0, evoked / max(usual, 1.0))
            if self.moments - sense.seen > 3 or sense.seen == 0:  # (back after a while: as fresh as its relays are)
                sense.novelty = self.freshness(sense.name)
            elif not imagine:
                sense.novelty = 0.7 * sense.novelty + 0.3 * measured
            if not imagine:
                sense.seen = self.moments
            fresh = sense.novelty if not imagine else measured
            news.append(fresh)
            weights.append(intensity)
            if fresh > 0.6 and sense.met <= 60:
                r.new.append(sense.name)
        r.novelty = sum(n * w for n, w in zip(news, weights, strict=True)) / sum(weights) if weights else 0.0
        if not imagine:
            self._novelty = r.novelty
        r.fear = min(1.0, max(0.0, net.rate(self.cea, ms) - 2.0) / 40.0)
        motor = spikes[self.motor.slice].float().view(len(OPTIONS), MOTOR).mean(1) * 1000.0 / ms
        if not imagine:
            self.channel_rates = 0.75 * self.channel_rates + 0.25 * motor
        r.channels = {o: float(v) for o, v in zip(OPTIONS, self.channel_rates, strict=True)}
        ranked = torch.sort(self.channel_rates, descending=True)
        best, runner_up = int(ranked.indices[0]), float(ranked.values[1])
        if self.channel_rates[best] > 3.0 and self.channel_rates[best] > 1.15 * runner_up:  # (a clear winner)
            r.choice = OPTIONS[best]
        for chem, pop in self.nuclei.items():
            if not imagine:
                self.levels[chem] = 0.9 * self.levels[chem] + 0.1 * net.rate(pop, ms) / USUAL_RATE
        r.chemistry = dict(self.levels)
        return r

    # --- learning ---------------------------------------------------------------------------------------------

    def _habituate(self, local: torch.Tensor | None, present: list[tuple[Sense, float]]) -> None:
        """Relay synapses tire with use and recover with time (minutes and days)."""
        self.quick.add_((1.0 - self.quick) * QUICK_RECOVERY)
        self.slow.add_((1.0 - self.slow) * SLOW_RECOVERY)
        if local is None:
            return
        fired = self.net.spikes[self.thalamus.start + local].float()
        use = (fired / 2.0).clamp(max=1.0)
        self.quick[local] -= QUICK_TIRING * use * self.quick[local]
        self.slow[local] -= SLOW_TIRING * use * self.slow[local]

    def _learn(self, present: list[tuple[Sense, float]], hurt: float, ms: int) -> None:
        net = self.net
        acetylcholine, dopamine = self.levels["acetylcholine"], self.levels["dopamine"]
        rate = 0.6 * min(2.0, 0.4 + 0.6 * acetylcholine) * min(2.0, 0.7 + 0.3 * dopamine)
        self._cortical_stdp(rate)
        # Fear: relay synapses onto lateral amygdala cells that fired with the hurt grow (Hebb, while it hurts);
        # without hurt, what it fears slowly fades (extinction).
        relays = (net.last[self.thalamus.slice] >= net.t - STDP_WINDOW).nonzero().squeeze(1)  # (fired lately)
        cued = relays[self.cue[relays]]  # (what it sees, hears, finds happening, where it is: not what it does)
        if cued.numel():
            post = net.last[self.fearful.targets[cued].long()] >= net.t - STDP_WINDOW
            sizes = self.fearful.sizes[cued].float()
            felt = torch.zeros(self.nrelays)  # (how strongly each thing is sensed now: what's near, most)
            for sense, intensity in present:
                felt[sense.relays - self.thalamus.start] = intensity
            if hurt > 0.1:  # (what stood out most, and what it didn't already know well: latent inhibition)
                step = FEAR_STEP * hurt * felt[cued] * (0.15 + 0.85 * self.slow[cued])
                sizes += post.float() * step.unsqueeze(1) + torch.rand(sizes.shape, generator=self.gen) * 0.5
            else:  # (met unharmed, close up, a synapse of what it fears now and then loses a size step)
                chance = EXTINCTION * felt[cued].unsqueeze(1)
                sizes -= post.float() * (torch.rand(sizes.shape, generator=self.gen) < chance).float()
            self.fearful.sizes[cued] = sizes.round().clamp(1, LEVELS - 1).to(torch.uint8)
            net.sync(self.fearful, cued)
        # Striatum: an eligibility trace for every synapse whose relay and spiny cell fired together; dopamine
        # turns it into learning (more go for what turned out well, more no-go for what didn't).
        self.eligibility.mul_(0.7)
        if relays.numel():
            post = net.spikes[self.striatal.targets[relays].long()] > 0
            self.eligibility[relays] += post.float()
        surprise = dopamine - 1.0
        if abs(surprise) > 0.05:
            d1 = (self.striatal.targets >= self.d1.start) & (self.striatal.targets < self.d1.stop)
            sign = torch.where(d1, 1.0, -1.0)
            change = 3.0 * surprise * sign * self.eligibility
            noise = torch.rand(change.shape, generator=self.gen)
            moved = torch.floor(change + noise)
            if moved.abs().sum() > 0:
                self.striatal.sizes.copy_((self.striatal.sizes.float() + moved).clamp(1, LEVELS - 1).to(torch.uint8))
                net.sync(self.striatal)

    def _cortical_stdp(self, rate: float) -> None:
        """Spike-timing-dependent plasticity between cortical cells that fired close together.

        A synapse grows when its cell fired just before the target did (it helped make it fire) and shrinks when it
        fired just after (Bi and Poo 1998), by up to STDP_STEP sizes, the closer the two spikes the more. Pairs
        count when either spike is from this moment and the other within STDP_WINDOW ms of it.
        """
        net, cortex = self.net, self.cortex
        now = net.t
        last = net.last[cortex.slice].view(self.ncols, CELLS)
        recent = last >= now - STDP_WINDOW  # fired lately (this moment or just before)
        fresh = last >= now - net.spikes_ms  # fired this moment
        pre = (recent[:, :E_CELLS]).nonzero()  # (column, cell) of pyramidal cells that fired lately
        if not pre.numel() or rate <= 0:
            return
        if pre.shape[0] > 4000:  # (a burst: learn from a sample of it)
            pre = pre[torch.randperm(pre.shape[0], generator=self.gen)[:4000]]
        counts = recent.sum(1)
        most = int(counts.max())
        order = torch.argsort((~recent).to(torch.uint8), dim=1, stable=True)[:, :most]  # cells that fired first
        valid = torch.arange(most).unsqueeze(0) < counts.unsqueeze(1)
        when = torch.gather(last, 1, order).float()
        new = torch.gather(fresh, 1, order)
        col, cell = pre[:, 0], pre[:, 1]
        targets = self.column_targets[col]  # (P, C)
        delays = self.recurrent.delays[col].float()
        t_pre = last[col, cell].float()
        pre_new = fresh[col, cell]
        post_cells = order[targets]  # (P, C, most)
        ok = valid[targets] & (new[targets] | pre_new.view(-1, 1, 1))  # (one of the two spikes is this moment's)
        dt = when[targets] - t_pre.view(-1, 1, 1) - delays.unsqueeze(2)  # post - (pre + delay)
        change = torch.where(dt >= 0, torch.exp(-dt / 20.0), -0.6 * torch.exp(dt / 20.0)) * (rate * STDP_STEP)
        keep = ok.nonzero(as_tuple=True)
        if not keep[0].numel():
            return
        rows = (col * E_CELLS + cell)[keep[0]]
        flat = (rows * self.C + keep[1]) * CELLS + post_cells[keep]
        flat_sizes = self.recurrent.sizes.view(-1)
        size = flat_sizes[flat].float()
        moved = torch.floor(change[keep] + torch.rand(size.shape, generator=self.gen))
        flat_sizes[flat] = (size + moved).clamp(0, LEVELS - 1).to(torch.uint8)
        self.changed = True

    def _homeostasis(self, ms: int) -> None:
        """Each cortical cell nudges its own excitability toward its usual rate (intrinsic plasticity)."""
        net, cortex = self.net, self.cortex
        spikes = net.spikes[cortex.slice].float().view(self.ncols, CELLS) * (1000.0 / ms)
        bias = net.bias[cortex.slice].view(self.ncols, CELLS)
        bias[:, :E_CELLS] += 0.02 * (TARGET["cortex E"] - spikes[:, :E_CELLS]).clamp(-20, 20)
        bias[:, E_CELLS:] += 0.01 * (TARGET["cortex I"] - spikes[:, E_CELLS:]).clamp(-40, 40)
        bias.clamp_(-200.0, 200.0)

    # --- sleep ------------------------------------------------------------------------------------------------

    def dream(self, ms: int = MOMENT) -> Reading:
        """A moment of sleep: the hippocampus replays one of the day's important moments, faintly, through the
        cortex (which learns from it), and the senses recover."""
        self.quick.add_((1.0 - self.quick) * 0.05)
        self.slow.add_((1.0 - self.slow) * 0.001)
        replay = {}
        if self.episodes:
            i = int(torch.randint(0, len(self.episodes), (1,), generator=self.gen))
            replay = {name: 0.6 * v for name, v in self.episodes[i].items()}
        return self.moment(replay, {}, {}, learn=True, asleep=True, ms=ms)

    def downscale(self, share: float = 0.03) -> None:
        """Overnight, cortical synapses shrink a little (synaptic homeostasis): a share of them lose a size step."""
        sizes = self.recurrent.sizes
        step = 1 << 22
        flat = sizes.view(-1)
        for i in range(0, flat.numel(), step):
            part = flat[i : i + step]
            shrink = (torch.rand(part.shape, generator=self.gen) < share) & (part > 0)
            part.sub_(shrink.to(torch.uint8))
        self.episodes = []
        self.changed = True

    # --- what it knows ----------------------------------------------------------------------------------------

    def probe(self, senses: dict[str, float], ms: int = 25, top_down: bool = False) -> Reading:
        """Imagine something without learning from it: what the brain does at the thought of it.

        Thinking of something doesn't come through tired senses: its relay cells answer as if rested, or (top_down)
        its cortical cells are driven directly, as the thought of a thing excites them from within.
        """
        one_core()
        net = self.net
        saved = [x.clone() for x in (net.v, net.u, net.ampa, net.nmda, net.gaba_a, net.gaba_b, net.arriving)]
        saved_recent, saved_t, saved_last = list(net.recent), net.t, net.last.clone()
        noise = net.gen.get_state()
        if self._noise is not None:  # (the same background as the thought it's compared with)
            net.gen.set_state(self._noise)
        saved_rates, saved_spikes = self.channel_rates.clone(), net.spikes.clone()
        saved_release = net.release.clone()
        cues = [self.sense_of(name) for name in senses]
        for sense in cues:
            net.release[sense.relays] = 1.0
        if top_down:
            self._top_down = {s.name: (s.home, 120.0 + 180.0 * senses[s.name]) for s in cues}
            reading = self.moment({}, {}, {}, learn=False, asleep=False, ms=ms, imagine=True)
            self._top_down = {}
        else:
            reading = self.moment(senses, {}, {}, learn=False, asleep=False, ms=ms, imagine=True)
        self.probed = net.spikes.clone()
        net.v, net.u, net.ampa, net.nmda, net.gaba_a, net.gaba_b, net.arriving = saved
        net.recent, net.t, net.last = saved_recent, saved_t, saved_last
        self.channel_rates, net.spikes, net.release = saved_rates, saved_spikes, saved_release
        net.gen.set_state(noise)
        return reading

    def associations(self, cue: dict[str, float], most: int = 5, ms: int = 25) -> list[tuple[str, float]]:
        """What comes to mind with something: the things whose cortical cells fire at the thought of it, beyond how
        they usually do."""
        known = [s for s in self.senses.values() if s.name not in cue and s.name != "(calibration)"]
        if not known:
            return []
        self._noise = self.net.gen.get_state()  # (both thoughts with the same background, to compare them)
        self.probe({}, ms)
        quiet = self._fire_counts(known)
        self.probe(cue, ms, top_down=True)
        self._noise = None
        lit = self._fire_counts(known)
        lift = (lit - quiet) * 1000.0 / ms  # Hz
        lift = lift - lift.median()  # (beyond what the thought of anything stirs up)
        order = torch.argsort(lift, descending=True)[:most]
        return [(known[i].name, float(lift[i])) for i in order if lift[i] > 2.0]

    def _fire_counts(self, senses: list[Sense]) -> torch.Tensor:
        homes = torch.stack([s.home for s in senses])
        return self.probed[homes].float().mean(1)

    def fear_of(self, senses: dict[str, float], ms: int = 60) -> float:
        """How afraid the thought of something makes it: the central amygdala's answer, beyond its resting rate."""
        self._noise = self.net.gen.get_state()
        rates = []
        for thought in ({}, senses):
            self.probe(thought, ms)
            rates.append(float(self.probed[self.cea.slice].float().mean()) * 1000.0 / ms)
        self._noise = None
        return min(1.0, max(0.0, rates[1] - rates[0]) / 20.0)

    # --- about it ---------------------------------------------------------------------------------------------

    def neurons(self) -> int:
        return sum(p.size for p in self.net.populations.values())

    def synapses(self) -> int:
        return self.net.synapses()

    def stats(self) -> dict:
        return {
            "size": self.size,
            "neurons": self.neurons(),
            "synapses": self.synapses(),
            "columns": self.ncols,
            "senses": len(self.senses),
            "chemistry": dict(self.levels),
            "ms per moment": round(self.timing, 1),
        }

    # --- keeping it -------------------------------------------------------------------------------------------

    def save(self, folder: Path, synapses: bool = True) -> None:
        """Keep the brain: its state in state.pt, and (if `synapses` and they changed) every cortical synapse's size
        in cortex.u8, one byte each."""
        folder.mkdir(parents=True, exist_ok=True)
        net = self.net
        state = {
            "seed": self.seed,
            "size": self.size,
            "bias": net.bias,
            "relay": (self.relay.targets, self.relay.sizes),
            "fearful": (self.fearful.targets, self.fearful.sizes),
            "feedforward": (self.feedforward.targets, self.feedforward.sizes),
            "striatal": self.striatal.sizes,
            "quick": self.quick,
            "slow": self.slow,
            "cue": self.cue,
            "senses": {
                name: (int((s.relays[0] - self.thalamus.start) // RELAYS), s.met, s.last, s.novelty, s.seen)
                for name, s in self.senses.items()
            },
            "free": self.free,
            "moments": self.moments,
            "levels": self.levels,
            "episodes": self.episodes,
            "reference": self.reference,
            "t": net.t,
        }
        partial = folder / "state.pt.part"
        torch.save(state, partial)
        partial.replace(folder / "state.pt")
        if synapses and (self.changed or not (folder / "cortex.u8").exists()):
            partial = folder / "cortex.u8.part"
            self.recurrent.sizes.numpy().tofile(partial)
            partial.replace(folder / "cortex.u8")
            self.changed = False

    @classmethod
    def load(cls, folder: Path) -> Brain:
        state = torch.load(folder / "state.pt", weights_only=False)
        brain = cls(state["seed"], state["size"], fresh=False)
        net = brain.net
        expected = brain.recurrent.sizes.numel()
        raw = torch.from_file(str(folder / "cortex.u8"), shared=False, size=expected, dtype=torch.uint8)
        brain.recurrent.sizes.copy_(raw.view_as(brain.recurrent.sizes))
        net.bias.copy_(state["bias"])
        brain.base_bias = net.bias.clone()
        brain.relay.targets.copy_(state["relay"][0])
        brain.relay.sizes.copy_(state["relay"][1])
        brain.fearful.targets.copy_(state["fearful"][0])
        brain.fearful.sizes.copy_(state["fearful"][1])
        brain.feedforward.targets.copy_(state["feedforward"][0])
        brain.feedforward.sizes.copy_(state["feedforward"][1])
        brain.striatal.sizes.copy_(state["striatal"])
        for projection in (brain.relay, brain.fearful, brain.feedforward, brain.striatal):
            net.sync(projection, targets=True)
        brain.quick.copy_(state["quick"])
        brain.slow.copy_(state["slow"])
        brain.cue.copy_(state["cue"])
        for pop in (*brain.nuclei.values(), brain.gpe, brain.gpi, brain.motor):
            net.noise[pop.slice] *= 0.0 if pop in brain.nuclei.values() else 0.3
        brain.senses = {}
        for name, (group, met, last, novelty, seen) in state["senses"].items():
            relays = brain.thalamus.start + group * RELAYS + torch.arange(RELAYS)
            home = brain.relay.targets[group * RELAYS, : HOME_COLUMNS * HOME_CELLS].long()
            brain.senses[name] = Sense(name, relays, home, met, last, novelty, seen)
        brain.free = list(state["free"])
        brain.moments = int(state["moments"])
        brain.levels = dict(state["levels"])
        brain.episodes = list(state["episodes"])
        brain.reference = dict(state["reference"])
        net.t = int(state["t"])
        net.last.fill_(net.t - 1000)
        brain.changed = False
        return brain


def memory_gb() -> float:
    """How much memory the computer has (GB), as well as can be told."""
    import os
    import sys

    try:
        if sys.platform == "win32":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong)] + [
                    (f"x{i}", ctypes.c_ulonglong) for i in range(6)
                ]

            status = Status()
            status.length = ctypes.sizeof(Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.total / 1e9
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (AttributeError, ValueError, OSError):
        return 8.0


def fit() -> str:
    """The biggest brain this computer can keep running alongside everything else, at every moment of Haven's life."""
    import os

    gb, cores = memory_gb(), os.cpu_count() or 2  # (it computes on one core, so many cores mean a fast computer)
    if gb >= 30 and cores >= 16:
        return "huge"
    if gb >= 15 and cores >= 12:
        return "large"
    if gb >= 6 and cores >= 4:
        return "standard"
    return "small"
