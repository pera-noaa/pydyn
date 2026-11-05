"""A forward run from the input files alone: no dump set, the clock and a run configuration in
place of the dump headers.  `python3 -m pydyn.forward --days 14 --device metal`
(see --help for the input paths).

The same sequence as replay.FreeRun with an Ingestion (which a dump set schedules): per meteo
read interval of nread seconds

    Meteo_Setup_Mass   the raw fluxes of the instants around the interval, balanced against
                       its surface-pressure change (balance.setup_mass, the FFT Poisson)
    Check_CFL          on a copy of the run's mass, the fluxes interpolated to each step's
                       middle; it picks ndyn (and n steps of it)
    per step           the fluxes at the step's middle, Setup_MassFlow, Calc_Kzz's dkg (the
                       surface pressure at the step's start), the convection fields of the
                       3-hourly interval, then zoom.step -- x y z v s c | c s v z y x, an
                       m2phlb1 after each half -- the emission and the CH4 chemistry at their
                       half step's time (hh:00 and hh:30 for ndyn 3600)

on the reference (`--device cpu`) or a pydyn.gpu device.  The run starts from the istart-66
file on the first interval's air mass (Pressure_to_Mass), as initexit does.  One region (the
local archive's glb6x4, its reduced grid); the configuration's defaults are ch4_6x4_fwd.rc's,
the run of the repin_phys set, whose dumps this driver reproduces bitwise on the CPU
(tests/test_forward.py).

The host's reading of the next intervals runs while the device carries this one: with
`readers` = 0 on one thread (every archive read of a process goes through one thread: netCDF4
is not thread-safe); with readers > 0 (the CLI's default 3) in that many spawned processes,
each reading whole intervals, a 3-hour window of them at a time (the archive's instants then
read once a window: meteo.Archive.KEEP), and handing them over through a shared-memory block
(share / unshare) -- the same functions, the same bits (tests/test_forward.py).  With
`stage` (a node's local disk) a meteo.Stager copies the archive's next days there first.  On
an asynchronous device the intervals are speculated, two in flight (gpu.zoom.Pipeline).  A
script that makes a Forward with readers must guard its body with `if __name__ ==
"__main__"` (spawned processes import it).
"""

import argparse
import sys
import time
from collections import deque, namedtuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from . import balance, chem, runtime
from . import zoom as Z
from .inputs import Ingestion, seed
from .lapack import batched_inverse
from .redgrid import RedGrid
from .replay import engine_for

__all__ = ["Forward", "RunConfig", "main"]


@dataclass
class RunConfig:
    """What a run takes from its rc (the dumps' headers carry the same): ch4_6x4_fwd.rc's."""

    start: datetime = datetime(2000, 1, 1)
    grid: str = "glb6x4"
    im: int = 60
    jm: int = 45
    jred: tuple = (1, 2, 3, 45)
    clustsize: tuple = (60, 20, 10, 60)
    nread: int = 3600
    ndyn_max: int = 3600
    limits: bool = True
    nt: int = 1
    lmax_conv: int = 26
    # emission (emission_ch4: one do_add_2d per s, the month's field)
    l_level: int = 1
    i_tracer: int = 1
    nsrce: int = 3600
    mol_mass: float = 1.6042750000000002e01
    mol_mass_e: float = 1.6042750000000002e01
    efrac: float = 1.0
    # chemistry (ch4_loss)
    oh_scale: float = 9.0100000000000002e-01
    cl_scale: float = 1.0
    o1d_scale: float = 1.0
    trop_cl_sink: bool = True
    soil_sink: bool = True
    soil_rate: float = 2.0000000000000001e-04
    ch4_tracers: tuple = (1,)
    extra: dict = field(default_factory=dict)


def _zoom(cfg, meteo_root):
    """The run's one region, its half levels from the archive (meteo.hybrid_coefficients: the
    model's table)."""
    from .meteo import Archive

    grid = RedGrid(cfg.im, cfg.jm, cfg.jred, cfg.clustsize) if cfg.jred else None
    region = Z.Region(cfg.grid, cfg.im, cfg.jm, xcyc=1, touch_sp=1, touch_np=1, redgrid=grid)
    a, b = Archive(meteo_root).hybrid("mfuv", cfg.start)
    return Z.Zoom([region], len(a) - 1, a, b)


class Reader:
    """The host's work of a read interval, a function of its time alone (Meteo.mass's `first`,
    the initial air mass, is passed): in the run's process or in a worker's."""

    def __init__(self, cfg, zm, ing, fft="auto"):
        self.cfg, self.inputs = cfg, ing[1]
        self.M = self.inputs.meteo
        self.M.lmax_conv = cfg.lmax_conv
        self.solver = balance.poisson_solver(cfg.im, cfg.jm, runtime.fft_choice(fft))
        self.levi_db, self.area_m2 = np.diff(zm.bt), self.M.area_m2

    def interval(self, t, first):
        """The read interval at t: Meteo_Setup_Mass (`read`), the convection fields, and
        `steps`, its step's reads for ndyn = nread."""
        got = self.read(t, first)
        conv = self.M.convection(got["tread"])
        got["conv"] = {k: q[: self.cfg.im, : self.cfg.jm] for k, q in conv.items() if k != "cloud_base"}
        got["steps"] = self.step_reads(got["tread"], got["tread"])
        return got

    def read(self, t, first):
        """Meteo_Setup_Mass of the read interval at t: the raw fluxes balanced; if first, the
        air mass."""
        cfg, M = self.cfg, self.M
        tread = (t, t + timedelta(seconds=cfg.nread))
        M.first = first
        raw = M.mass(tread)
        out = {"tread": tread, "instants": raw["instants"]}
        if "m" in raw:
            out.update(m0=raw["m"], sp0=raw["sp"], phlb0=raw["phlb"])
        pairs = []
        for k in (1, 2):
            pu, pv, pw = (raw[f"{v}{k}"].copy() for v in ("mfu", "mfv", "mfw"))
            balance.setup_mass(pu, pv, pw, raw["sp1"], raw["sp2"], self.levi_db, self.area_m2,
                               float(cfg.nread), solver=self.solver)  # fmt: skip
            pairs.append((pu, pv))
        out["pairs"] = pairs
        return out

    def step_reads(self, tr, tread, surface=True):
        """The step's own reads: Calc_Kzz's inputs but the surface pressure and
        fluxes, the temperature the chemistry takes, and each half's emission and chemistry
        month fields."""
        cfg, M, ins = self.cfg, self.M, self.inputs
        a, b = M.levels(tr[0])
        out = {
            "spm": M.spm(tr, tread), "t": M.temperature(tr), "q": M.humidity(tr), "a": a, "b": b,
        }  # fmt: skip
        if surface:
            out["surface"] = M.surface(tr)
        half = timedelta(seconds=(tr[1] - tr[0]).total_seconds() / 2)
        if ins.emissions is not None:
            out["emis"] = [
                ins.emissions.injection(
                    tr[0] + h * half, cfg.nsrce, cfg.mol_mass, cfg.mol_mass_e, cfg.efrac, 1
                )
                for h in (0, 1)
            ]
        if ins.loss is not None:
            out["loss"] = []
            for h in (0, 1):
                mid = tr[0] + h * half + half / 2  # chem_month: the half step's middle
                out["loss"].append(ins.loss.month(mid.month, mid.year, cfg.trop_cl_sink, cfg.soil_sink))
        return out


_WORKER = None  # a reader process's Reader
WINDOW = 3 * 3600  # seconds of consecutive intervals a reader process reads: the instants' spacing


def _worker_init(cfg, meteo_root, emissions, chemistry, fft, libm, stage=None):
    global _WORKER
    from . import ops

    if libm:
        ops.use_libm(libm)
    zm = _zoom(cfg, meteo_root)
    _WORKER = Reader(cfg, zm, Ingestion(zm, meteo_root, emissions, chemistry, meteo_grid="model"), fft)
    _WORKER.M.archive.stage = stage


def _worker_interval(t, first, block=None):
    return share(_WORKER.interval(t, first), block)


# ------------------------------------------------------------------ results through shared memory
_SHARED_MIN = 1 << 16  # arrays from this size go through the block, the rest pickled
_Shared = namedtuple("_Shared", "offset shape dtype")
_ATTACHED = {}  # block name -> its SharedMemory, mapped once in this process


def _attach(name):
    shm = _ATTACHED.get(name)
    if shm is None:
        shm = _ATTACHED[name] = SharedMemory(name)
    return shm


def _walk(o, leaf):
    if isinstance(o, dict):
        return {k: _walk(v, leaf) for k, v in o.items()}
    if isinstance(o, (list, tuple)) and not isinstance(o, _Shared):
        return type(o)(_walk(v, leaf) for v in o)
    return leaf(o)


def share(obj, block=None):
    """(block name, skeleton): obj with its large arrays copied into a shared-memory block (C
    order, as pickling makes a strided array) and replaced by their places in it -- `block`
    (a name) if it is large enough, else a new one.  A pipe's pickling of an interval's ~11 MB
    is the receiving process's work, under its GIL; this is one copy."""
    arrays, total = [], 0

    def leaf(o):
        nonlocal total
        if not (isinstance(o, np.ndarray) and o.nbytes >= _SHARED_MIN):
            return o
        arrays.append((total, o))
        out = _Shared(total, o.shape, o.dtype.str)
        total += -(-o.nbytes // 64) * 64
        return out

    skeleton = _walk(obj, leaf)
    if not arrays:
        return None, skeleton
    shm = _attach(block) if block is not None else None
    if shm is None or shm.size < total:
        shm = SharedMemory(create=True, size=total)
        _ATTACHED[shm.name] = shm
    for off, a in arrays:
        np.ndarray(a.shape, a.dtype, shm.buf, off)[...] = a
    return shm.name, skeleton


def unshare(name, skeleton):
    """share's obj, its arrays copied out of the block (which the caller may give out again)."""
    if name is None:
        return skeleton
    buf = _attach(name).buf
    return _walk(skeleton, lambda o: np.ndarray(o.shape, np.dtype(o.dtype), buf, o.offset).copy()
                 if isinstance(o, _Shared) else o)  # fmt: skip


def _free(name):
    shm = _ATTACHED.pop(name, None)
    if shm is not None:
        shm.close()
        shm.unlink()


class Forward:
    """The run: `interval()` advances one read interval, `run(hours)` many."""

    def __init__(self, cfg, meteo_root, emissions=None, chemistry=None, initial=None, device="cpu",
                 fft="auto", lapack="auto", readers=0, pipeline=2, stage=None):  # fmt: skip
        self.cfg, self.E = cfg, engine_for(device)
        self.pipeline = pipeline  # intervals in flight on an asynchronous device (run)
        self.inverse = batched_inverse(runtime.lapack_choice(lapack))
        self.zm = _zoom(cfg, meteo_root)
        self.ing = Ingestion(self.zm, meteo_root, emissions, chemistry, initial, meteo_grid="model")
        self.host = Reader(cfg, self.zm, self.ing, fft)
        self.M = self.host.M
        self.stager = None  # stage: the archive's next days copied under it (meteo.Stager)
        if stage is not None:
            from .meteo import Stager

            self.stager = Stager(meteo_root, stage)
            self.M.archive.stage = self.stager.stage
        self.st = self.E.zeros(self.zm, cfg.nt)
        self.cfl_st = self.E.zeros(
            self.zm, 1
        )  # Check_CFL's copy, one for the run: its box programs are kept on it
        for s in (self.st, self.cfl_st):
            self.E.put(s[1]["dxyp"], self.M.dxyp)
        self.t = cfg.start
        self.seeded = self.m2phlb_seen = False
        self.lat = np.radians(self.M.ll.south + self.M.ll.dlat * (np.arange(cfg.jm) + 0.5))
        self.reader = ThreadPoolExecutor(1)  # every archive read of this process on this one thread
        # readers > 0: worker processes reading whole intervals, each a window of WINDOW
        # consecutive ones (the archive's 3-hourly instants read once a window, Archive.KEEP)
        self.pools = []
        if readers:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            from . import ops

            init = (cfg, meteo_root, emissions, chemistry, fft, ops.libm.path, self.M.archive.stage)
            ctx = mp.get_context("spawn")
            self.pools = [ProcessPoolExecutor(1, ctx, _worker_init, init) for _ in range(readers)]
        self.window = max(1, WINDOW // cfg.nread)
        self.blocks = deque()  # shared-memory blocks free for the next interval's reading
        self.lent = set()  # every block made, free or in use (close frees them)
        self.depth = max(1, readers * self.window)  # intervals read ahead
        self.pending = deque()  # (t, collect) of the intervals being read, in order
        self.stop = None  # no interval at or after it is read (run's end)
        self.timing = {"host_wait": 0.0, "device": 0.0, "intervals": 0}
        self.speculated = {"kept": 0, "redone": 0}

    # ---------------------------------------------------------------- the host's inputs
    def _interp(self, got, t):
        """Setup_MassFlow's TimeInterpolation of the balanced pairs to the instant t."""
        (pu1, pv1), (pu2, pv2) = got["pairs"]
        t1, t2 = got["instants"]
        a1, a2 = balance.interpol_fractions(t, t1, t2)
        return balance.time_interpolation(pu1, pu2, a1, a2), balance.time_interpolation(pv1, pv2, a1, a2)

    # ---------------------------------------------------------------- the run
    def _seed(self, got):
        """initexit's start: the istart-66 tracers on the first air mass (Ingestion.initial_state),
        m's periodic halos two deep (Pressure_to_Mass), as FreeRun._seed_initial."""
        ic = self.reader.submit(self.ing.initial_state, {1: got["m0"]}, self.cfg.nt).result()
        seed(self.zm, self.st, ic, self.E)
        self.sp0, self.phlb0 = got["sp0"], got["phlb0"]
        self.seeded = True

    def interval(self, ahead=True, pipe=None):
        """One read interval from self.t, the next ones' reading (ahead) under way; returns
        (ndyn, n) -- or, given a pipeline (E.pipeline), queues it there (None)."""
        cfg, E = self.cfg, self.E
        t0 = time.perf_counter()
        step = timedelta(seconds=cfg.nread)
        limit = self.t + (self.depth + 1) * step if ahead else self.t + step
        if self.stop is not None:
            limit = min(limit, self.stop)
        if self.stager is not None:
            self.stager.ahead(self.t)
        nxt = self.pending[-1][0] + step if self.pending else self.t
        while nxt < limit and len(self.pending) <= self.depth:  # this interval and depth more
            self.pending.append((nxt, self._submit(nxt)))
            nxt += step
        t_read, collect = self.pending.popleft()
        if t_read != self.t:
            raise RuntimeError(f"forward: read {t_read} for the interval at {self.t}")
        got = collect()
        if not self.seeded:
            self._seed(got)
        self.timing["host_wait"] += time.perf_counter() - t0
        t1 = time.perf_counter()
        mine = pipe is None  # one interval, settled before it returns
        if mine:
            pipe = E.pipeline(self.zm, self.st, 1)
        seen = self.m2phlb_seen

        def body():
            self.m2phlb_seen = seen  # repeatable: the host's own state reset
            return self._carry(got, got["steps"])

        pipe.submit(body)
        out = None
        if mine:
            (out,) = pipe.drain()
            self._tally(pipe)
            E.sync()
        self.timing["device"] += time.perf_counter() - t1
        self.timing["intervals"] += 1
        self.t += step
        return out

    def _submit(self, t):
        """The reading of the interval at t under way: () -> its dict."""
        first = not self.seeded and t == self.t  # the run's first interval brings its air mass
        if not self.pools:
            return self.reader.submit(self.host.interval, t, first).result
        k = int((t - self.cfg.start).total_seconds()) // self.cfg.nread // self.window
        block = self.blocks.popleft() if self.blocks else None
        got = self.pools[k % len(self.pools)].submit(_worker_interval, t, first, block)

        def collect():
            name, skeleton = got.result()
            out = unshare(name, skeleton)
            if name is not None:
                if block is not None and block != name:  # outgrown: the worker made a larger one
                    _free(block)
                    self.lent.discard(block)
                self.lent.add(name)
                self.blocks.append(name)
            elif block is not None:
                self.blocks.append(block)
            return out

        return collect

    def _tally(self, pipe):
        for k in ("kept", "redone"):
            self.speculated[k] += pipe.tally[k]

    def _carry(self, got, steps):
        """Check_CFL, then the interval's steps."""
        cfg, E, zm = self.cfg, self.E, self.zm
        tread = got["tread"]
        tmp = self.cfl_st  # Check_CFL rehearses on a copy (FreeRun.cfl)
        tmp.lib.assign(tmp, self.st, 1, ("m", "sp"))

        def interpolate(zoom, st, i, ndyn):
            pu, pv = self._interp(got, tread[0] + timedelta(seconds=(i - 0.5) * ndyn))
            E.put(st[1]["pu"], pu)
            E.put(st[1]["pv"], pv)

        ndyn, n, _ = Z.check_cfl_zoom(
            zm, tmp, cfg.ndyn_max, cfg.nread // cfg.ndyn_max, cfg.nread, interpolate
        )
        for s in range(n):
            tr = (tread[0] + timedelta(seconds=s * ndyn), tread[0] + timedelta(seconds=(s + 1) * ndyn))
            reads = (
                steps if ndyn == cfg.nread else self.reader.submit(self.host.step_reads, tr, tread).result()
            )
            self._step(got, tr, ndyn, reads)
        return ndyn, n

    def _step(self, got, tr, ndyn, reads):
        cfg, E, zm, st = self.cfg, self.E, self.zm, self.st
        g = zm.g(1)
        pu, pv = self._interp(got, tr[0] + (tr[1] - tr[0]) / 2)
        E.put(st[1]["pu"], pu)
        E.put(st[1]["pv"], pv)
        Z.setup_massflow(zm, st, ndyn)
        sp = st[1]["sp"][1 : g.im, 1 : g.jm, 1] if self.m2phlb_seen else self.sp0  # the step's start
        args = (reads["spm"], sp, pu, pv, reads["t"], reads["q"], reads["surface"], reads["a"], reads["b"],
                self.M.dxyp, cfg.lmax_conv, self.M.ll.dlon, self.M.ll.dlat, self.M.ll.south)  # fmt: skip
        dkg = E.calc_kzz(*args)
        si, sj = Z.scope(zm, 1)  # Calc_Kzz runs on the interior scope
        if (si.stop - si.start, sj.stop - sj.start) != (g.im, g.jm):
            raise NotImplementedError("forward: a region-1 scope smaller than the region")
        conv = {1: dict(got["conv"], dkg=dkg, lmax=cfg.lmax_conv)}

        def which_half(region):
            return zm.status[region] // Z.N_OPERATORS  # 0: xyzvsc, 1: csvzyx

        def sources(zoom, s, region):
            if "emis" in reads:
                s.lib.sources_region(
                    zoom, s, region, [(reads["emis"][which_half(region)], cfg.l_level, cfg.i_tracer)]
                )

        def chemistry(zoom, s, region):
            if "loss" not in reads:
                return
            h = which_half(region)
            if h == 1:
                self.m2phlb_seen = True  # the first half's m2phlb1 has run
            phlb = s[1]["phlb"][1 : g.im, 1 : g.jm, 1 : zm.lm + 1] if self.m2phlb_seen else self.phlb0
            settings = {k: getattr(cfg, k) for k in chem.SETTINGS}
            inputs = chem.loss_inputs(reads["t"], phlb, self.lat, reads["loss"][h], settings)
            _, decay = E.chemistry(inputs, ndyn / 2.0)
            s.lib.chemistry_region(zoom, s, region, decay, cfg.ch4_tracers)

        Z.step(
            zm, st, conv, ndyn, limits=cfg.limits, inverse=self.inverse, sources=sources, chemistry=chemistry
        )
        self.m2phlb_seen = True

    def run(self, hours, every=None):
        """hours of simulated time; every(self) after each interval, if given (the intervals
        in flight settled first).  On an asynchronous device `pipeline` intervals are in
        flight: the next is queued while the device runs this one (gpu.zoom.Pipeline)."""
        n = int(hours * 3600) // self.cfg.nread
        self.stop = self.t + timedelta(seconds=n * self.cfg.nread)
        pipe = self.E.pipeline(self.zm, self.st, max(1, self.pipeline))
        for _ in range(n):
            self.interval(pipe=pipe)
            if every is not None:
                pipe.drain()
                every(self)
        pipe.drain()
        self.E.sync()
        self._tally(pipe)

    def close(self):
        self.reader.shutdown(wait=True, cancel_futures=True)
        for pool in self.pools:
            pool.shutdown(wait=True, cancel_futures=True)
        if self.pools:
            for _, collect in self.pending:  # read ahead, never used: their blocks given back
                try:
                    collect()
                except Exception:  # cancelled, or failed: nothing was shared
                    pass
            self.pending.clear()
            for name in self.lent:
                _free(name)
            self.lent.clear()
            self.blocks.clear()
        if self.stager is not None:
            self.stager.close()


def main(argv=None):
    from pathlib import Path

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--meteo", required=True, metavar="ROOT")
    p.add_argument("--emissions")
    p.add_argument("--chemistry")
    p.add_argument("--initial")
    p.add_argument("--device", default="cpu", choices=("cpu", "emu", "cuda", "metal", "emu32"))
    p.add_argument("--start", default="2000-01-01")
    p.add_argument("--days", type=float, default=1.0)
    p.add_argument("--hours", type=float, default=None)
    p.add_argument("--lapack", default="auto")
    p.add_argument("--fft", default="auto")
    p.add_argument("--libm", metavar="PATH")
    p.add_argument("--readers", type=int, default=3, help="reader processes (0: a thread of this one)")
    p.add_argument("--pipeline", type=int, default=2, help="intervals in flight on an asynchronous device")
    p.add_argument("--stage", metavar="DIR", help="copy the archive's next days under DIR (local disk)")
    p.add_argument("--save", metavar="FILE.npz", help="the final state's arrays")
    a = p.parse_args(argv)
    runtime.apply(a)
    cfg = RunConfig(start=datetime.fromisoformat(a.start))
    fwd = Forward(cfg, a.meteo, a.emissions, a.chemistry, a.initial, a.device, a.fft, a.lapack, a.readers,
                  a.pipeline, a.stage)  # fmt: skip
    hours = a.hours if a.hours is not None else a.days * 24
    wall = time.perf_counter()
    fwd.run(hours)
    wall = time.perf_counter() - wall
    fwd.close()
    n, tm = fwd.timing["intervals"], fwd.timing
    per_hour, years = wall / hours * 1e3, hours * 3600 / wall / (365 * 86400) * 3600
    print(f"{a.device}: {hours:g} simulated hours from {a.start} in {wall:.1f} s -- "
          f"{per_hour:.1f} ms a simulated hour, {years:.2f} simulated years an hour")  # fmt: skip
    wait, dev = tm["host_wait"] / n * 1e3, tm["device"] / n * 1e3
    print(f"  host (reading, balancing; what the device waited for): {wait:.1f} ms an interval; "
          f"device: {dev:.1f} ms an interval")  # fmt: skip
    if fwd.speculated["kept"] or fwd.speculated["redone"]:
        print(f"  speculation: {fwd.speculated['kept']} intervals kept, {fwd.speculated['redone']} redone")
    if a.save:
        st = fwd.st
        np.savez(Path(a.save), **{k: fwd.E.host(v.data) for k, v in st[1].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
