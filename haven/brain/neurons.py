"""Spiking neurons and their synapses, simulated a millisecond at a time.

Each neuron is Izhikevich's simple model of a real cell type, fitted to recordings of that
type (Izhikevich 2007, Dynamical Systems in Neuroscience, chapter 8): a membrane potential
that integrates input and fires a spike, and a recovery current that makes it adapt, burst
or chatter as its type does. Units are physiological: millivolts, milliseconds, picofarads,
picoamperes, nanosiemens.

Synapses are conductances with the receptors of real synapses (AMPA, NMDA with its
magnesium block, GABA-A and GABA-B), and each spike reaches its targets after the time it
takes to travel along the axon (1 to 15 ms). A neuron is either excitatory or inhibitory in
everything it sends (Dale's law). A synapse's strength is one of 26 sizes, about as many as
real synapses can be told apart by (Bartol et al. 2015, 4.7 bits), spread over the sixty-fold
range of their sizes, and size 0 is a silent synapse (one with no AMPA receptors yet).

Neurons also get the background of synaptic input real cortical neurons are bombarded
with, so they fire now and then on their own, as real neurons do, and can't fire again
for 2 ms after a spike (the absolute refractory period).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

# Cell types: (C pF, k nS/mV, rest mV, threshold mV, peak mV, a 1/ms, b nS, reset mV, d pA), Izhikevich (2007).
CELLS = {
    "RS": (100.0, 0.7, -60.0, -40.0, 35.0, 0.03, -2.0, -50.0, 100.0),  # regular-spiking pyramidal cell
    "IB": (150.0, 1.2, -75.0, -45.0, 50.0, 0.01, 5.0, -56.0, 130.0),  # intrinsically bursting pyramidal cell (layer 5)
    "CH": (50.0, 1.5, -60.0, -40.0, 25.0, 0.03, 1.0, -40.0, 150.0),  # chattering pyramidal cell (layer 2/3)
    "FS": (20.0, 1.0, -55.0, -40.0, 25.0, 0.2, 0.025, -45.0, 0.0),  # fast-spiking basket cell (inhibitory)
    "LTS": (
        100.0,
        1.0,
        -56.0,
        -42.0,
        40.0,
        0.03,
        8.0,
        -53.0,
        20.0,
    ),  # low-threshold spiking Martinotti cell (inhibitory)
    "TC": (200.0, 1.6, -60.0, -50.0, 35.0, 0.01, 0.0, -60.0, 10.0),  # thalamocortical relay cell (in relay mode)
    "MSN": (50.0, 1.0, -80.0, -25.0, 40.0, 0.01, -20.0, -55.0, 150.0),  # medium spiny neuron of the striatum
    "GP": (50.0, 1.0, -60.0, -45.0, 30.0, 0.1, 0.2, -55.0, 10.0),  # pallidal neuron: a fast autonomous pacemaker
    "PM": (100.0, 0.7, -60.0, -40.0, 35.0, 0.03, -2.0, -50.0, 100.0),  # a pacemaker (dopamine, serotonin... cells)
}
INHIBITORY = frozenset({"FS", "LTS"})
FS_KNEE = -55.0  # fast-spiking cells' recovery current only switches on above this (their cubic nullcline)

# Receptors: (time constant ms, reversal potential mV)
AMPA, NMDA, GABA_A, GABA_B = (5.0, 0.0), (150.0, 0.0), (6.0, -70.0), (150.0, -90.0)
NMDA_SHARE = 0.1  # NMDA conductance an excitatory spike adds, for each nS of AMPA
GABA_B_SHARE = 0.02  # and GABA-B, for each nS of GABA-A
MAGNESIUM = 1.0  # mM, outside the cell: what blocks NMDA channels until the cell is depolarized (Jahr & Stevens 1990)
REFRACTORY = 2  # ms

SIZES = 26  # distinguishable synapse sizes (Bartol et al. 2015)
RANGE = 60.0  # the largest synapse is about sixty times the smallest
STRENGTH = torch.tensor([0.0] + [RANGE ** ((s - (SIZES - 1)) / (SIZES - 2)) for s in range(1, SIZES)])


@dataclass
class Population:
    """A group of neurons of one or more cell types, numbered start..stop in the network."""

    name: str
    start: int
    stop: int
    kinds: list[tuple[str, int]]
    inhibitory: bool = False

    @property
    def size(self) -> int:
        return self.stop - self.start

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)


@dataclass
class Projection:
    """Synapses from one population to others: each presynaptic neuron's targets, sizes and delays.

    Two layouts: `targets` (pre, K) of neuron numbers, or column blocks, where every excitatory
    cell of a source column reaches every cell of each of its column's target columns
    (`columns` (source columns, C) and `width` cells per target column).
    """

    name: str
    pre: Population
    sizes: torch.Tensor  # uint8: (pre, K), or (pre, C, width) for blocks
    unit: float  # nS of the largest synapse
    delays: torch.Tensor  # int: (pre,) or (pre, K); for blocks (source columns, C)
    targets: torch.Tensor | None = None  # int32 (pre, K)
    columns: torch.Tensor | None = None  # int64 (source columns, C): target blocks (of `width` cells) in the network
    per_column: int = 0  # cells per source column that send (blocks: its excitatory cells, first in each column)
    stride: int = 0  # cells per source column in the presynaptic population (blocks)
    width: int = 0
    release: torch.Tensor | None = None  # (pre,): how much of its transmitter each axon has ready (habituation)
    rows: torch.Tensor | None = None  # (pre,): which row each presynaptic cell sends from, -1 for none (if not all do)
    plastic: bool = True
    excitatory: bool = True
    fast: bool = False  # inhibitory synapses with GABA-A receptors only (no slow GABA-B)
    count: int = field(default=0)

    def __post_init__(self):
        self.count = int(self.sizes.numel())


def strength(sizes: torch.Tensor) -> torch.Tensor:
    return STRENGTH[sizes.long()]


class Network:
    """Every neuron of the brain in one set of arrays, and the projections between them."""

    def __init__(self, seed: int = 0, delays: int = 16, block: int = 100):
        self.gen = torch.Generator().manual_seed(seed)
        self.D = delays
        self.block = block  # the width of a column, for block projections (block populations start on a multiple)
        self.populations: dict[str, Population] = {}
        self.projections: list[Projection] = []
        self.kinds: list[str] = []
        self.n = 0
        self.t = 0
        self.background = (0.0, 0.0)  # (events per ms per neuron, nS per 100 pF) of background excitatory input

    # --- building ----------------------------------------------------------------------------------

    def population(self, name: str, kinds: list[tuple[str, int]], align: bool = False) -> Population:
        if align and self.n % self.block:
            pad = self.block - self.n % self.block
            self.kinds += ["RS"] * pad  # (unconnected padding: they never fire)
            self.n += pad
        start = self.n
        for kind, count in kinds:
            self.kinds += [kind] * count
            self.n += count
        pop = Population(name, start, self.n, kinds, all(k in INHIBITORY for k, _ in kinds))
        self.populations[name] = pop
        return pop

    def connect(self, projection: Projection) -> Projection:
        self.projections.append(projection)
        return projection

    def build(self) -> None:
        """Allocate the neurons' state, once every population is in place."""
        n = self.n + (-self.n % self.block)
        self.n_padded = n
        params = torch.tensor([CELLS[k] for k in self.kinds] + [CELLS["RS"]] * (n - self.n))
        self.C, self.k, self.vr, self.vt, self.vpeak, self.a, self.b, self.c, self.d = params.T.contiguous()
        self.vpeak[self.n :] = 1e9  # (padding never fires)
        self.kc = 0.5 * self.k / self.C  # (folded into the half-millisecond steps)
        self.hc = 0.5 / self.C
        self.fs = torch.tensor([k == "FS" for k in self.kinds] + [False] * (n - self.n)).nonzero().squeeze(1)
        self.v = self.vr.clone()
        self.u = torch.zeros(n)
        self.ampa, self.nmda, self.gaba_a, self.gaba_b = (torch.zeros(n) for _ in range(4))  # conductances (nS)
        self.decays = [math.exp(-1.0 / tau) for tau in (AMPA[0], NMDA[0], GABA_A[0], GABA_B[0])]
        self.arriving = torch.zeros(3, self.D, n)  # excitatory, inhibitory, fast inhibitory conductance on its way
        self.bias = torch.zeros(n)  # steady current (pA): homeostasis, development
        self.drive = torch.zeros(n)  # input current (pA) from the senses, for the cells that get it
        self.steady = torch.zeros(n)  # bias + drive (set by prepare())
        self.gain = torch.ones(n)  # how strongly synaptic input moves each cell (neuromodulators)
        self.noise = self.C * (self.background[1] / 100.0)  # nS each background event brings each cell
        self.noise[self.n :] = 0.0
        self.spikes = torch.zeros(n, dtype=torch.int16)  # spikes in the current moment
        self.last = torch.full((n,), -1000, dtype=torch.int32)  # when each last fired (ms)
        self.trace: list[torch.Tensor] = []  # who fired at each ms of the current moment
        self.recent: list[torch.Tensor] = []  # who fired in the last REFRACTORY ms
        self.release = torch.ones(n)  # how much transmitter each cell's axon has ready (habituation)
        self.blocks = [p for p in self.projections if p.columns is not None]
        self._unify([p for p in self.projections if p.columns is None])

    def prepare(self) -> None:
        """After changing bias or drive: the steady current every cell gets this moment."""
        torch.add(self.bias, self.drive, out=self.steady)

    # --- running -----------------------------------------------------------------------------------

    def step(self) -> torch.Tensor:
        """One millisecond. Returns the neurons that fired."""
        n, slot = self.n_padded, self.t % self.D
        exc, inh, fast = self.arriving[0, slot], self.arriving[1, slot], self.arriving[2, slot]
        da, dn, dga, dgb = self.decays
        self.ampa.mul_(da).add_(exc)
        self.nmda.mul_(dn).add_(exc, alpha=NMDA_SHARE)
        self.gaba_a.mul_(dga).add_(inh).add_(fast)
        self.gaba_b.mul_(dgb).add_(inh, alpha=GABA_B_SHARE)
        exc.zero_()
        inh.zero_()
        fast.zero_()
        rate = self.background[0]
        if rate > 0:
            kicked = torch.randint(0, n, (max(1, int(n * rate)),), generator=self.gen)
            self.ampa.index_add_(0, kicked, self.noise[kicked])  # (in proportion to the cell's size)
        v, u = self.v, self.u
        # NMDA channels open as the magnesium block lifts with depolarization (Jahr & Stevens 1990)
        block = torch.sigmoid(v * 0.062 + math.log(3.57 / MAGNESIUM))
        excitation = torch.addcmul(self.ampa, self.nmda, block)  # (both reverse at 0 mV)
        current = excitation.mul_(v).neg_()
        current.addcmul_(self.gaba_a, torch.sub(GABA_A[1], v))
        current.addcmul_(self.gaba_b, torch.sub(GABA_B[1], v))
        current.mul_(self.gain).add_(self.steady).sub_(u)
        for _ in range(2):  # two half-millisecond steps, for stability
            # (below rest, a real membrane leaks back linearly: the model's parabola would pull far too hard)
            pull = torch.maximum(v, self.vr).sub_(self.vt).mul_(v - self.vr).mul_(self.kc)
            v.add_(pull.addcmul_(self.hc, current))
        v.clamp_(min=-90.0, max=60.0)  # (no lower than the most negative reversal potential, GABA-B's)
        if self.recent:  # the absolute refractory period
            held = torch.cat(self.recent)
            v[held] = self.c[held]
        fs = self.fs
        before = u[fs] if fs.numel() else None
        u.addcmul_(self.a, torch.sub(v, self.vr).mul_(self.b).sub_(u))
        if before is not None:  # fast-spiking cells' cubic recovery
            u[fs] = before + self.a[fs] * (0.025 * (v[fs] - FS_KNEE).clamp(min=0.0) ** 3 - before)
        fired = (v >= self.vpeak).nonzero().squeeze(1)
        if fired.numel():
            v[fired] = self.c[fired]
            u[fired] += self.d[fired]
            self.spikes[fired] += 1
            self.last[fired] = self.t
            self._send(fired)
        self.recent = [*self.recent[-(REFRACTORY - 1) :], fired] if REFRACTORY > 1 else [fired]
        self.trace.append(fired)
        self.t += 1
        return fired

    def _unify(self, sparse: list[Projection]) -> None:
        """Every synapse that isn't in a column block, in one table sorted by the cell that sends it, so a millisecond's
        spikes are delivered in one pass whatever projection they belong to."""
        n, D = self.n_padded, self.D
        pres, dests, sizes, delays, luts = [], [], [], [], []
        self.luts = torch.stack([STRENGTH * p.unit for p in sparse]) if sparse else torch.zeros(1, SIZES)
        offset = 0
        for pid, p in enumerate(sparse):
            rows, k = p.sizes.shape
            if p.rows is not None:
                senders = (p.rows >= 0).nonzero().squeeze(1)
                if senders.numel() != rows:
                    raise ValueError(f"{p.name}: {rows} rows of synapses for {senders.numel()} cells that send")
                pre = torch.empty(rows, dtype=torch.long)
                pre[p.rows[senders]] = p.pre.start + senders
            else:
                pre = p.pre.start + torch.arange(rows)
            pres.append(pre.repeat_interleave(k))
            p.inhibition = 0 if p.excitatory else (2 if p.fast else 1) * D * n
            dests.append(p.targets.reshape(-1).long() + p.inhibition)
            sizes.append(p.sizes.reshape(-1))
            delay = p.delays if p.delays.dim() == 2 else p.delays.unsqueeze(1).expand(rows, k)
            delays.append(delay.reshape(-1).long())
            luts.append(torch.full((rows * k,), pid, dtype=torch.long))
            p.index = torch.arange(offset, offset + rows * k).view(rows, k)
            offset += rows * k
        if not sparse:
            self.syn_pre = torch.zeros(0, dtype=torch.long)
            self.indptr = torch.zeros(n + 1, dtype=torch.long)
            return
        pre = torch.cat(pres)
        order = torch.argsort(pre, stable=True)
        position = torch.empty_like(order)
        position[order] = torch.arange(order.numel())
        for p in sparse:
            p.index = position[p.index]
        self.syn_pre = pre[order]
        self.syn_dest = torch.cat(dests)[order]
        self.syn_size = torch.cat(sizes)[order]
        self.syn_delay = torch.cat(delays)[order]
        self.syn_lut = torch.cat(luts)[order] * SIZES
        self.indptr = torch.searchsorted(self.syn_pre, torch.arange(n + 1))

    def sync(self, p: Projection, rows: torch.Tensor | None = None, targets: bool = False) -> None:
        """After a projection's synapses changed (learning, or new targets): copy them into the table."""
        index = p.index if rows is None else p.index[rows]
        self.syn_size[index.reshape(-1)] = (p.sizes if rows is None else p.sizes[rows]).reshape(-1)
        if targets:
            chosen = p.targets if rows is None else p.targets[rows]
            self.syn_dest[index.reshape(-1)] = chosen.reshape(-1).long() + p.inhibition

    def _send(self, fired: torch.Tensor) -> None:
        """Each spike sets off along its axon: its synapses' conductances arrive after their delays."""
        n, D = self.n_padded, self.D
        for p in self.blocks:  # column blocks: only a column's excitatory cells send them
            mine = fired[(fired >= p.pre.start) & (fired < p.pre.stop)] - p.pre.start
            mine = mine[mine % p.stride < p.per_column]
            if not mine.numel():
                continue
            source = mine // p.stride
            rows = source * p.per_column + mine % p.stride
            grams = STRENGTH[p.sizes[rows].long()].mul_(p.unit)  # (F, C, width)
            blocks = n // p.width
            where = ((self.t + p.delays[source]) % D) * blocks + p.columns[source]
            self.arriving[0].view(D * blocks, p.width).index_add_(0, where.reshape(-1), grams.reshape(-1, p.width))
        starts, stops = self.indptr[fired], self.indptr[fired + 1]
        counts = stops - starts
        total = int(counts.sum())
        if not total:
            return
        # every synapse of every cell that fired: its place in the table
        first = torch.repeat_interleave(starts - (torch.cumsum(counts, 0) - counts), counts)
        synapse = first + torch.arange(total)
        grams = self.luts.view(-1)[self.syn_lut[synapse] + self.syn_size[synapse].long()]
        grams.mul_(torch.repeat_interleave(self.release[fired], counts))
        where = self.syn_dest[synapse] + ((self.t + self.syn_delay[synapse]) % D) * n
        self.arriving.view(-1).index_add_(0, where, grams)

    def begin(self) -> None:
        """A new moment: forget whose spikes belonged to the last one."""
        self.spikes.zero_()
        self.trace = []
        self.began = self.t

    @property
    def spikes_ms(self) -> int:
        """How many ms the current moment has run."""
        return self.t - getattr(self, "began", 0)

    def rate(self, pop: Population, ms: int) -> float:
        """How fast a population fired in the moment just run (Hz)."""
        return float(self.spikes[pop.slice].float().mean()) * 1000.0 / max(ms, 1)

    def synapses(self) -> int:
        return sum(p.count for p in self.projections)
