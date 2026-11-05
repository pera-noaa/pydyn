"""Replay a `pydyn_dump` directory (rc key `pydyn.dump`) of a model run -- one region or a
zoom tree -- through the Python and compare bitwise with what the model produced next.

    python -m pydyn.replay <dump_dir> [--free-run] [--meteo <tm5-nc root>] [--emissions <dir>]
                           [--chemistry <input/ch4c13>] [--initial <istart-66 file>]
                           [--meteo-grid model|1x1] [--tags x,y,...] [--limit N] [--rank R]
                           [--workers N] [--fft …] [--lapack …] [--libm <libimf.so>] [-v]

The region tree is built from the dump header (`status`, split order, footprint, children;
tm5-zoom 611fb1ca9) and the replay refuses to run unless it is the model's; a single-region
run is a tree of one.  The arrays keep their Fortran bounds (zoom.FView) and are compared
whole, halos included.  By default every event is replayed from the model's own inputs, so a
difference cannot propagate:

    bef_x/y/z .. aft    zoom.advect_{x,y,z}_zoom with the header's status (which half of the
                        parent step, which block of the split order sets the scope); a
                        child's halo is in its bef_ event, the parent's put_x/yedges having
                        filled it
    bef_v .. aft_v      zoom.convec_region (bitwise only through the model's own LAPACK)
    bef_s .. aft_s      zoom.sources_region: each `emis` event's injection on the cells the
                        region owns
    bef_c .. aft_c      chem.loss_rates on the `chem` event (its `loss` compared, over the
                        scope isr:ier x jsr:jer) and zoom.chemistry_region's decay
    massflow            the TimeInterpolation of each region's data pairs, then
                        zoom.setup_massflow on every region at once
    cfl                 zoom.check_cfl_zoom on the dumped masses: the model's (ndyn, n)
    meteo               Meteo_Setup_Mass, region by region (a child against its parent's
                        meteo event of the same read interval); from the cfl events of a set
                        without meteo events (the census sets)
    m2phlb              m2phlb1 (region 1) / m2phlb (a child): sp and phlb

With --free-run (FreeRun) the Python carries every region's state from the initial field (or
each region's first bef_ dump) on -- through the operators, Check_CFL's ndyn, Setup_MassFlow,
update_parent and every m2phlb, and the put_x/yedges do_steps adds at a block end, including
blocks ended by the s / c no-ops of a transport-only build -- and every dump is a checkpoint:
`in:<name>` the state on entering an operator, the plain names the result.  The balanced
meteo is the Python's own (a child's against the Python's parent).  phlb is compared on the
cells m2phlb writes.

The inputs the model reads from files are the dumps' unless read from the files themselves
(inputs.Ingestion), each compared as `in:<name>` (`ic:<name>` the initial field) and, in a
free run, used:

    --meteo <root>       the tm5-nc archive: the raw mass fluxes, matched surface pressures
                         and initial air mass of every meteo event, the convection fields,
                         Calc_Kzz's dkg (over Meteo_Setup_Other's step, the step's starting
                         surface pressure: a parent's m2phlb1 after coarsen_region, a child's
                         last m2phlb) and the chemistry's temperature.  --meteo-grid: the set
                         the rc reads -- the model grid (ch4_6x4_fwd.rc; the default for one
                         region) or the 1x1, 137-level one regridded onto each region as tmm
                         does (the zoom rc; regrid.py)
    --emissions <dir>    the 1x1 monthly CH4 files, each region's field from them on its own
                         grid; a free run injects it with month2dt from the calendar
    --chemistry <root>   the CH4 input tree: the Bruehl columns, the OH, the Hossaini Cl and
                         the soil map of every chem event, per region
    --initial <file>     the istart-66 field, each region on its own grid and initial air
                         mass, then update_parent

With all four a free run takes no field from a dump: only the grid constants and scalars.
"""

import argparse
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import numpy as np

from . import balance, cfl, chem, emis, ops, runtime
from . import zoom as Z
from . import zoom_meteo as ZM
from .dump import _META, DumpError, FView, load_events
from .harness import compare
from .inputs import MASS_INPUTS, Ingestion, dumped_dkg, seed, times
from .lapack import batched_inverse
from .redgrid import RedGrid

__all__ = [
    "FreeRun", "Meteo", "build_tree", "ch4_tracers", "check_cfl", "chem_inputs", "conv_inputs", "copy_into",
    "dumped_injections", "dumped_step", "format_summary", "interpolated_fluxes", "preceding", "replay",
    "replay_operator", "setup_mass", "span", "step_inputs", "summarize", "tally",
]  # fmt: skip

INNER = {"s": "emis", "c": "chem"}  # the operators whose bef/aft pair encloses input events


# ------------------------------------------------------------------ the dumps
def _dumped_pair(ev, k):
    """The balanced meteo fluxes (pu, pv) of instant k (1 or 2) as the model dumped them."""
    return ev.interior(f"pu{k}"), ev.interior(f"pv{k}")


def interpolated_fluxes(ev, t, pairs=None):
    """pu, pv (pydyn interiors) interpolated to the instant t between the data1/data2 pairs
    (the dumped ones unless `pairs` = ((pu1, pv1), (pu2, pv2)) is given)."""
    t1, t2 = times(ev, "pu_t1", "pu_t2")
    (pu1, pv1), (pu2, pv2) = pairs if pairs is not None else (_dumped_pair(ev, 1), _dumped_pair(ev, 2))
    a1, a2 = balance.interpol_fractions(t, t1, t2)
    return balance.time_interpolation(pu1, pu2, a1, a2), balance.time_interpolation(pv1, pv2, a1, a2)


def conv_inputs(ev, ingested=None):
    """The convection parameters of a bef_v dump (or the `ingested` ones): kwargs of
    kernels.convec."""
    lmax = ev["entu"].upper[2]
    im, jm = ev.meta["im"], ev.meta["jm"]
    conv = {name: ev[name][1:im, 1:jm, 1:lmax] for name in ("entu", "detu", "entd", "detd")}
    conv.update({name: ev[name][1:im, 1:jm] for name in ("cloud_top", "cloud_lfs")})
    # dkg is not dumped by a without_diffusion build: no diffusion term then
    conv["dkg"] = dumped_dkg(ev, lmax)
    conv.update(ingested or {})
    return conv, lmax


def preceding(events, k, tag):
    """The last event tagged `tag` before events[k] in its region, None if there is none."""
    ev = events[k]
    return next((e for e in reversed(events[:k]) if e.region == ev.region and e.tag == tag), None)


def dumped_step(massflow, met):
    """What Calc_Kzz takes from the dynamic step itself, from its `massflow` and `meteo`
    dumps: {tr, pu, pv, sp} -- the step's interval and interpolated fluxes, the surface
    pressure advected to its start."""
    if massflow is None or met is None:
        return None
    return {
        "tr": times(massflow, "tr1", "tr2"),
        "pu": massflow.interior("pu"),
        "pv": massflow.interior("pv"),
        "sp": met.interior("sp"),
    }


def chem_inputs(ev, ingested=None):
    """The loss-rate inputs of a chem dump, those in `ingested` (the temperature, the
    climatologies, the soil map) taken from there instead: kwargs of chem.loss_rates."""
    im, jm, lm = ev.meta["im"], ev.meta["jm"], ev.meta["lm"]
    meta = ev.meta
    fields = {"oh_3d": ev["oh_3d"][1:im, 1:jm, 1:lm]}
    fields.update({f"strat_{name}": ev[f"strat_{name}"][1:jm] for name in ("press", "oh", "cl", "od")})
    fields.update({name: ev[name][1:im, 1:jm] for name in ("pl_trop", "cl_trop", "soil") if name in ev})
    temper = ev["temper"][1:im, 1:jm, 1:lm]
    inputs = chem.loss_inputs(temper, ev.interior("phlb"), ev["lat"][1:jm], fields, meta)
    inputs.update({k: v for k, v in (ingested or {}).items() if k in inputs})
    return inputs


def ch4_tracers(ev):
    tracers = ev.meta["ch4_tracers"]
    return tracers if isinstance(tracers, tuple) else (tracers,)


def dumped_injections(inner):
    """The do_add_2d calls of an s pair from its `emis` dumps: [(x, l_level, i_tracer)] -- the
    global tracer, the same as itracer_loc in a serial run and the right index into a
    multi-rank run's merged state (dump.merge_ranks)."""
    out = []
    for ev in inner:
        m = ev.meta
        x = emis.inject(ev["emis"].data, m["month2dt"], m["mol_mass"], m["mol_mass_e"], m["efrac"])
        out.append((x, m["l_level"], m["i_tracer"]))
    return out


def check_cfl(ev, m=None, pairs=None):
    """Check_CFL on the mass field m (the dumped one if None): (ndyn, n, iterations)."""
    meta = ev.meta
    t0 = balance.date(meta["tread1"])

    def interpolate(offset):
        return interpolated_fluxes(ev, t0 + timedelta(seconds=offset), pairs)

    m = ev.interior("m") if m is None else m
    return cfl.check_cfl(
        m, meta["ndyn_in"], meta["n_in"], meta["nread"], ev["bt"][1:], meta["tref"], interpolate,
        grid=RedGrid.from_meta(meta),
    )  # fmt: skip


def setup_mass(ev, solver, raw=None):
    """Meteo_Setup_Mass for the instants in ev: {k: (pu, pv, info)} with pu, pv balanced.
    `raw`: the unbalanced fluxes and surface pressures {mfu1, mfv1, mfw1, ..., sp1, sp2} to
    start from, the dumped ones if None."""
    t1, t2 = times(ev, "tread1", "tread2")
    dt_sec = (t2 - t1).total_seconds()
    raw = {name: ev.interior(name) for name in MASS_INPUTS if name in ev} if raw is None else raw
    sp1, sp2 = raw["sp1"], raw["sp2"]
    levi_db, area_m2 = ev["levi_db"][1:], ev["area_m2"][1:]
    out = {}
    for k in (1, 2):
        if f"mfu{k}" not in raw:
            continue
        pu, pv, pw = (raw[f"{name}{k}"].copy() for name in ("mfu", "mfv", "mfw"))
        info = balance.setup_mass(pu, pv, pw, sp1, sp2, levi_db, area_m2, dt_sec, solver=solver)
        _, rms, mav = balance.check_mass_balance(*_dumped_pair(ev, k), sp1, sp2, dt_sec, area_m2, 1e-7, 1e-6)
        out[k] = (pu, pv, {"tries": info["tries"], "python": info["post"], "fortran": (rms, mav)})
    return out


def span(events, k):
    """(aft_* event, inner input events) of the bef_* event events[k]: the inner ones are the
    operator's `emis` / `chem` dumps written between the pair."""
    ev = events[k]
    aft, inner_tag = "aft_" + ev.tag[4:], INNER.get(ev.tag[-1])
    inner = []
    for e in events[k + 1 :]:
        if e.region != ev.region:
            continue
        if e.tag == aft:
            return e, inner
        if e.tag == inner_tag:
            inner.append(e)
        elif e.tag == ev.tag:
            break
    raise ValueError(f"{ev}: no matching {aft} event")


# ------------------------------------------------------------------ the engine
STATE = ("m", "rm", "rxm", "rym", "rzm", "am", "bm", "cm")


def _as_tuple(v):
    return (v,) if isinstance(v, int) else tuple(v)


def build_tree(evs):
    """zoom.Zoom from the headers of one moment's events {region: event}, checked against them."""
    m1 = evs[1].meta
    im1, jm1 = m1["im"], m1["jm"]
    regions, west, south = [], {1: -180.0}, {1: -90.0}
    for r in sorted(evs):  # a parent is numbered before its children
        m = evs[r].meta
        common = {k: m[k] for k in ("xref", "yref", "zref", "tref", "parent", "xcyc",
                                     "touch_sp", "touch_np")}
        if r == 1:
            regions.append(Z.Region(m["grid_name"], m["im"], m["jm"], redgrid=RedGrid.from_meta(m),
                                    **common))
            continue
        p = evs[m["parent"]].meta
        dxp, dyp = 360.0 / (im1 * p["xref"]), 180.0 / (jm1 * p["yref"])
        dx, dy = 360.0 / (im1 * m["xref"]), 180.0 / (jm1 * m["yref"])
        # the footprint counts parent cells from the parent's own south-west corner
        xbeg = west[r] = west[m["parent"]] + (m["ibeg"] - 1) * dxp
        ybeg = south[r] = south[m["parent"]] + (m["jbeg"] - 1) * dyp
        regions.append(Z.Region(m["grid_name"], m["im"], m["jm"], xbeg=xbeg, xend=xbeg + m["im"] * dx,
                                ybeg=ybeg, yend=ybeg + m["jm"] * dy, **common))
    zm = Z.Zoom(regions, m1["lm"], evs[1]["at"].data, evs[1]["bt"].data)
    check_tree(zm, evs)
    return zm


def check_tree(zm, evs):
    """The model's own footprint, children and split order against the Python's."""
    bad = []
    for r, ev in evs.items():
        m = ev.meta
        if "splitorder" not in m and list(evs) == [1]:
            continue  # a single region before the zoom header: nothing to check (see infer_status)
        if "splitorder" not in m:
            raise SystemExit(f"region {r}: no zoom header (status, splitorder, ibeg..) in {ev.prefix}; "
                             "this set predates tm5-zoom 611fb1ca9")
        if zm.splitorderzoom[r] != m["splitorder"]:
            bad.append(f"region {r} splitorder: model {m['splitorder']!r}, python {zm.splitorderzoom[r]!r}")
        if sorted(zm.children[r]) != sorted(_as_tuple(m.get("children", ()))):
            bad.append(f"region {r} children: model {m.get('children')}, python {zm.children[r]}")
        if r > 1:
            for k in ("ibeg", "iend", "jbeg", "jend"):
                if getattr(zm, k)[r] != m[k]:
                    bad.append(f"region {r} {k}: model {m[k]}, python {getattr(zm, k)[r]}")
    if bad:
        raise SystemExit("the Python tree is not the model's:\n  " + "\n  ".join(bad))


def copy_into(view, arr):
    """The Fortran-bounded dump array `arr` over the bounds of the FView `view` (which the dump
    must cover) written into it."""
    for lo, hi, alo, ahi in zip(view.lower, view.upper, arr.lower, arr.upper, strict=True):
        if lo < alo or hi > ahi:
            raise ValueError(f"dump bounds {arr} do not cover {view}")
    view.data[...] = arr.data[arr.window(view)]


def _compare(view, arr, name):
    """The Python's array against the dump's, over the Python's bounds."""
    return compare(arr.data[arr.window(view)], view.data, name)


class HostEngine:
    """What a replay computes with: the reference, pydyn.zoom on NumPy arrays.  pydyn.gpu's
    DeviceEngine has the same members for a state on the device (`--device`).  With
    `fingerprint` set every comparison also records the sha1 of the Python's array, so that
    two replays of one set can be shown to have carried the same bits (pydyn.gpu.check)."""

    name = "cpu"
    fingerprint = False

    def zeros(self, zm, nt):
        return Z.ZoomState.zeros(zm, nt)

    def copy_into(self, view, arr):
        copy_into(view, arr)

    def put(self, view, values):
        """view.data[...] = values, a host array of the inputs."""
        view.data[...] = values

    def host(self, x, tag="verify"):
        """x as a host array (on the device: a counted download)."""
        return x

    def compare(self, view, arr, name):
        report = _compare(view, arr, name)
        if self.fingerprint:
            report["sha1"] = _sha1(view.data)
        return report

    def report(self, model, ours, name):
        """compare(model, ours) of an array that is not a state FView (loss, phlb)."""
        ours = self.host(ours)
        report = compare(model, ours, name)
        if self.fingerprint:
            report["sha1"] = _sha1(ours)
        return report

    @staticmethod
    def column_pressure(zm, m, dxyp):
        """balance.m2phlb1's sp of an air-mass view (a zoom parent's, for Calc_Kzz)."""
        return balance.m2phlb1(m, zm.at, zm.bt, dxyp)[0]

    @staticmethod
    def calc_kzz(*args):
        """diffusion.calc_kzz's dkg (the arguments of meteo.Meteo.kzz_inputs)."""
        from .diffusion import calc_kzz

        return calc_kzz(*args)[0]

    def chemistry(self, inputs, dt_sec):
        """chem.loss_rates and chem.decay: (loss, decay)."""
        loss = chem.loss_rates(**inputs)
        return loss, chem.decay(loss, dt_sec)

    asynchronous = False  # the host decides as it goes

    def sync(self):
        """Wait for what was queued (on the device; the host has nothing queued)."""

    def pipeline(self, zm, st, depth):
        """gpu.zoom.Pipeline's members for calls that run as they are submitted."""
        return _Now()


class _Now:
    def __init__(self):
        self.done, self.tally = [], {"kept": 0, "redone": 0}

    def submit(self, fn):
        self.done.append(fn())

    def drain(self):
        out, self.done = self.done, []
        return out


def _sha1(a):
    import hashlib

    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()


HOST = HostEngine()


def engine_for(device):
    """The engine of --device: None / 'cpu' the reference, 'emu' / 'cuda' / 'metal' / 'emu32'
    pydyn.gpu's."""
    if device in (None, "cpu"):
        return HostEngine()
    from .gpu.engine import DeviceEngine

    return DeviceEngine(device)


def physics(zm, st, bef, inner, phlb=None, ingestion=None, step_tr=None, engine=HOST):
    """s or c of region r on the ZoomState `st` from the pair's `inner` events (emis / chem);
    `phlb` the half levels the chemistry uses instead of its dump's.  With an `ingestion`
    (inputs.Ingestion) the emission, the chemistry's fields and its temperature over the step
    `step_tr` are read from the input files, compared and used.  Returns the reports of the
    operator's own intermediate comparisons."""
    r, op = bef.region, bef.tag[-1]
    region = None if ingestion is None else ingestion[r]
    if op == "s":
        if region is not None and region.emissions is not None:
            injections, reports = region.sources(inner)
        else:
            injections, reports = dumped_injections(inner), []
        st.lib.sources_region(zm, st, r, injections)
        return reports
    (ev,) = inner
    if ev.meta["isotopes"]:
        raise NotImplementedError("chem: the isotope path is not transcribed")
    fields, reports = (None, []) if region is None else region.chemistry(ev, step_tr)
    inputs = chem_inputs(ev, fields)
    if phlb is not None:
        inputs["phlb_python"] = phlb
    # ch4_loss_kernel runs on isr:ier x jsr:jer: every input cut to the scope
    si, sj = Z.scope(zm, r)
    for name in ("temper", "phlb", "phlb_python", "oh_3d", "pl_trop", "cl_trop", "soil"):
        if name in inputs:
            inputs[name] = inputs[name][si, sj]
    for name in ("lat", "strat_press", "strat_oh", "strat_cl", "strat_od"):
        inputs[name] = inputs[name][sj]
    if phlb is not None:
        reports.append(engine.report(inputs["phlb"], inputs["phlb_python"], "in:phlb"))
        inputs["phlb"] = inputs.pop("phlb_python")
    loss, decay = engine.chemistry(inputs, ev.meta["dt_sec"])
    reports.append(engine.report(ev["loss"].data, loss, "loss"))
    st.lib.chemistry_region(zm, st, r, decay, ch4_tracers(ev))
    return reports


def step_inputs(zm, events, k):
    """replay.dumped_step for events[k] of a zoom: the step's interval and mid-step pu, pv from its
    massflow event, and the region's sp at the step's start.  A parent's is m2phlb1 of its
    massflow-dumped mass, which Setup_MassFlow's coarsen_region rebuilt over the child.  A
    child's is its last m2phlb's before the step, before any the meteo read's."""
    mf, met = (preceding(events, k, tag) for tag in ("massflow", "meteo"))
    step = dumped_step(mf, met)
    if step is None:
        return None
    r = events[k].region
    if zm.children[r]:
        if r != 1:
            raise NotImplementedError("step_inputs: a parent below region 1 (m2phlb, not m2phlb1)")
        step["sp"], _ = balance.m2phlb1(mf.interior("m"), met["at"][1:], met["bt"][1:], met["dxyp"][1:])
        return step
    before = (e for e in reversed(events[:k]) if e.region == r and e.tag == "m2phlb" and e.seq < mf.seq)
    last = next(before, None)
    if last is not None:
        step["sp"] = last.interior("sp")
    return step


def replay_operator(zm, bef, aft, inverse, inner=(), ingested=None, ingestion=None, step_tr=None,
                    engine=HOST):  # fmt: skip
    """One bef_<op> .. aft_<op> pair of region r: the reports of every STATE array.  A v's
    convection inputs are the dump's; `ingested`, (inputs, reports) read from the archive, is
    only compared."""
    r, op = bef.region, bef.tag[-1]
    zm.status[r] = bef.meta["status"]
    if zm.splitorderzoom[r][zm.status[r]] != op:
        raise ValueError(f"{bef}: status {zm.status[r]} is {zm.splitorderzoom[r][zm.status[r]]!r} "
                         f"in the split order, not {op!r}")
    st = engine.zeros(zm, bef["rm"].data.shape[3])
    for k in STATE:
        engine.copy_into(st[r][k], bef[k])
    reports = [] if ingested is None else list(ingested[1])
    if op in Z.OPERATORS:
        Z.OPERATORS[op](zm, st, r, limits=bef.meta["limits"])
    elif op == "v":
        conv, lmax = conv_inputs(bef)
        st.lib.convec_region(zm, st, r, dict(conv, lmax=lmax), bef.meta["nconv"], inverse)
    elif op in "sc":
        reports += physics(zm, st, bef, inner, ingestion=ingestion, step_tr=step_tr, engine=engine)
    else:
        raise ValueError(f"{bef}: operator {op!r} is not replayed here")
    return reports + [engine.compare(st[r][k], aft[k], k) for k in STATE]


def initial_reports(zm, ic, bef):
    """The initial state (Ingestion.initial_state) against a region's first bef_ dump, `ic:<name>`:
    a child over its scope only -- its edge ring has had the parent's put_x/yedges by then -- and
    a parent's m off its children's footprints, which Setup_MassFlow's coarsen_region has rebuilt
    (summed in another order than update_parent; the free run, seeded before it, checks them)."""
    r, g = bef.region, zm.g(bef.region)
    cut = Z.scope(zm, r) if r > 1 else (slice(None), slice(None))
    reports = []
    for k in ("m", "rm", "rxm", "rym", "rzm"):
        dumped, ours = bef.interior(k)[cut], ic[r][k][1 : g.im, 1 : g.jm, 1 : zm.lm][cut]
        if k == "m":
            off = np.ones(dumped.shape[:2], dtype=bool)
            for c in zm.children[r]:
                off[zm.ibeg[c] - 1 : zm.iend[c], zm.jbeg[c] - 1 : zm.jend[c]] = False
            dumped, ours = dumped[off], ours[off]
        reports.append(compare(dumped, ours, f"ic:{k}"))
    return reports


def replay_massflow(zm, evs, engine=HOST):
    """Setup_MassFlow for all regions from their massflow events {region: event}: the
    TimeInterpolation of each region's dumped data pairs to the step's middle against its pu,
    pv, then dynam0 / coarsen on the dumped pu, pv against am, bm, cm."""
    st = engine.zeros(zm, 1)
    reports = {r: [] for r in evs}
    for r, ev in evs.items():
        if "pu1" in ev:
            t1, t2 = times(ev, "tr1", "tr2")
            pu, pv = interpolated_fluxes(ev, t1 + (t2 - t1) / 2)
            reports[r] += [compare(ev.interior("pu"), pu, "pu"), compare(ev.interior("pv"), pv, "pv")]
        for k in ("m", "pu", "pv"):
            engine.copy_into(st[r][k], ev[k])
        engine.copy_into(st[r]["dxyp"], ev["dxyp"])
    Z.setup_massflow(zm, st, evs[1].meta["ndyn"])
    for r, ev in evs.items():
        reports[r] += [engine.compare(st[r][k], ev[k], k) for k in ("am", "bm", "cm")]
    return reports


def cfl_interpolation(evs, pairs=None, engine=HOST):
    """Setup_MassFlow's TimeInterpolation of every region's (pu, pv) to the middle of
    Check_CFL's step i, from the cfl events' data pairs or `pairs` {region: ((pu1, pv1), (pu2,
    pv2))} (pydyn interiors), as check_cfl_zoom's `interpolate` takes it."""
    t0 = balance.date(evs[1].meta["tread1"])

    def interpolate(zm, st, i, ndyn):
        t = t0 + timedelta(seconds=(i - 0.5) * ndyn)
        for r, ev in evs.items():
            if pairs is not None and pairs.get(r) is not None:
                pu, pv = interpolated_fluxes(ev, t, pairs[r])
                engine.put(st[r]["pu"], pu)
                engine.put(st[r]["pv"], pv)
                continue
            t1, t2 = balance.date(ev.meta["pu_t1"]), balance.date(ev.meta["pu_t2"])
            a1, a2 = balance.interpol_fractions(t, t1, t2)
            for name in ("pu", "pv"):
                d1, d2 = ev[f"{name}1"], ev[f"{name}2"]
                mixed = balance.time_interpolation(d1.data, d2.data, a1, a2)
                engine.copy_into(st[r][name], type(d1)(mixed, d1.lower))

    return interpolate


def replay_cfl(zm, evs, caps=None, engine=HOST):
    """Check_CFL (zoom.check_cfl_zoom) of a read interval from its cfl events {region: event}:
    the dumped masses and data pairs, from the dumped (ndyn_in, n_in), under `caps` (the
    production ones if None).  (ndyn, n, iterations)."""
    st = engine.zeros(zm, 1)
    for r, ev in evs.items():
        for name in ("m", "sp", "dxyp"):
            engine.copy_into(st[r][name], ev[name])
    m = evs[1].meta
    return Z.check_cfl_zoom(zm, st, m["ndyn_in"], m["n_in"], m["nread"],
                                   cfl_interpolation(evs, engine=engine), caps=caps)


def _cfl_reports(ev, ndyn, n):
    return [compare(np.array([ev.meta["ndyn_out"]]), np.array([ndyn]), "ndyn"),
            compare(np.array([ev.meta["n_out"]]), np.array([n]), "n")]


def replay_m2phlb(zm, ev, dxyp, engine=HOST):
    """m2phlb1 (region 1) or m2phlb (a child, its edge ring rescaled to the parent's cells)
    on the event's air mass: sp and phlb.  `dxyp` {region: FArray} for the parent's."""
    st = engine.zeros(zm, 1)
    r, g = ev.region, zm.g(ev.region)
    engine.copy_into(st[r]["m"], ev["m"])
    for q, arr in dxyp.items():
        engine.copy_into(st[q]["dxyp"], arr)
    (st.lib.m2phlb1 if r == 1 else st.lib.m2phlb)(zm, st, r)
    interior = FView(st[r]["phlb"][1 : g.im, 1 : g.jm, 1 : zm.lm + 1].copy(), (1, 1, 1))
    return [engine.compare(st[r]["sp"], ev["sp"], "sp"), engine.compare(interior, ev["phlb"], "phlb")]


def meteo_mass(ev, solver, parent=None, poisson=None, raw=None):
    """Meteo_Setup_Mass of one region's meteo event: {k: (pu, pv, pw, info)} for the instants
    k = 1, 2, from the event's raw fluxes and matched surface pressures (or `raw`, the same
    keys as MASS_INPUTS).  A child needs `parent` = {k: (pu, pv, pw)}, its parent's
    balanced fluxes, and the `poisson` of its refinement."""
    raw = {n: ev.interior(n) for n in MASS_INPUTS} if raw is None else raw
    if ev.region == 1:
        balanced = setup_mass(ev, solver, raw)
        return {k: (pu, pv, raw[f"mfw{k}"], info) for k, (pu, pv, info) in balanced.items()}
    t1, t2 = times(ev, "tread1", "tread2")
    out = {}
    for k in (1, 2):
        fluxes = tuple(raw[f"{n}{k}"] for n in ("mfu", "mfv", "mfw"))
        rf = ZM.Refinement.from_meta(ev.meta, parent["meta"])  # = Refinement.of(zm, r): check_tree
        out[k] = ZM.setup_mass_child(fluxes, parent[k], raw["sp1"], raw["sp2"], ev["levi_db"][1:],
                                     ev["area_m2"][1:], (t2 - t1).total_seconds(), rf, poisson)
    return out


def _meteo_reports(ev, mass):
    return [compare(ev.interior(f"{n}{k}"), q, f"{n}{k}")
            for k, fluxes in mass.items() for n, q in zip(("pu", "pv", "pw"), fluxes[:3], strict=True)]


class Meteo:
    """Meteo_Setup_Mass for every region of a read interval, region by region in the model's
    order (a parent before its children), each child against its parent's result: the
    dumped one per event (`own=False`) or the Python's (`own=True`, the free run).  With an
    `ingestion` (inputs.Ingestion) the raw inputs are also read from the archive and compared;
    the free run then balances the ingested ones."""

    def __init__(self, zm, fft, own, ingestion=None):
        self.zm, self.fft, self.own, self.ingestion = zm, fft, own, ingestion
        self.solver, self.poisson, self.last = {}, {}, {}
        self.sp0 = {}  # region -> the run's initial surface pressure, the ingested one if read
        self.phlb0 = {}  # region -> the initial half levels (Pressure_to_Mass), if ingested
        self.m0 = {}  # region -> the initial air mass (Pressure_to_Mass), the ingested one if read

    def event(self, ev):
        r, zm = ev.region, self.zm
        raw, inputs = None, []
        if self.ingestion is not None and self.ingestion.has_meteo:
            raw, inputs = self.ingestion[r].mass(ev)
            if "sp" in raw:
                self.sp0.setdefault(r, raw["sp"])
            if "phlb" in raw:
                self.phlb0.setdefault(r, raw["phlb"])
            if "m" in raw:
                self.m0.setdefault(r, raw["m"])
            raw = {n: raw[n] for n in MASS_INPUTS} if self.own else None
        self.sp0.setdefault(r, ev.interior("sp"))
        self.m0.setdefault(r, ev.interior("m"))
        g = zm.g(r)
        if r not in self.solver:
            self.solver[r] = balance.poisson_solver(g.im, g.jm, self.fft)
        parent = poisson = None
        if r > 1:
            p = g.parent
            pev, pmass = self.last.get(p, (None, None))
            if pev is None or times(pev, "tread1", "tread2") != times(ev, "tread1", "tread2"):
                raise ValueError(f"{ev}: no meteo event of its parent {p} for the same read interval")
            if self.own:
                parent = {k: q[:3] for k, q in pmass.items()}
            else:
                parent = {k: tuple(pev.interior(f"{n}{k}") for n in ("pu", "pv", "pw")) for k in (1, 2)}
            parent["meta"] = pev.meta
            rf = ZM.Refinement.of(zm, r)
            if (rf.ri, rf.rj) not in self.poisson:
                solver = balance.poisson_solver(rf.ri, rf.rj, self.fft)
                self.poisson[rf.ri, rf.rj] = ZM.PoissonDirect(rf.ri, rf.rj, solver)
            poisson = self.poisson[rf.ri, rf.rj]
        mass = meteo_mass(ev, self.solver[r], parent, poisson, raw)
        self.last[r] = (ev, mass)
        notes = {f"data{k}": q[3]["tries"] for k, q in mass.items()}
        return inputs + _meteo_reports(ev, mass), notes

    def pairs(self, ev):
        """The Python's balanced (pu, pv) pairs for a massflow event's data instants, None if
        its region's last meteo event is not for them."""
        last = self.last.get(ev.region)
        if last is None:
            return None
        mev, mass = last
        if (mev.meta["pu_t1"], mev.meta["pu_t2"]) != (ev.meta["pu_t1"], ev.meta["pu_t2"]):
            return None
        return mass[1][:2], mass[2][:2]


class FreeRun:
    """The zoom run re-done by the Python from each region's first bef_ dump on; every later
    dump is a checkpoint, never a re-seed."""

    def __init__(self, zm, nt, inverse, dxyp, meteo=None, dumped="xyzv", ingestion=None, engine=HOST):
        self.zm, self.inverse, self.meteo, self.dumped, self.ingestion = zm, inverse, meteo, dumped, ingestion
        self.nt, self.engine = nt, engine
        self.st = engine.zeros(zm, nt)
        for r, arr in dxyp.items():
            engine.copy_into(self.st[r]["dxyp"], arr)
        self.seeded = set()
        self.done = dict.fromkeys(zm.regions(), 0)  # split-order positions the Python has passed
        self.step = {}  # region -> Calc_Kzz's step inputs, the Python's own (see massflow)
        self.m2phlb_seen = set()  # regions whose sp is the Python's m2phlb's
        self.ndyn = None  # Check_CFL's choice for the current interval, the Python's own

    def _advance(self, r, to):
        """Pass the undumped operators of r up to position `to` (the s / c no-ops of a
        transport-only build), with do_steps' put_x/yedges at every block end they close."""
        order = self.zm.splitorderzoom[r]
        if to < self.done[r]:  # a new dynamic step: finish the last one first
            self._advance(r, len(order))
            self.done[r] = 0
        while self.done[r] < to:
            op = order[self.done[r]]
            if op in "xyzv" or op in self.dumped:
                raise ValueError(f"region {r}: operator {op!r} at {self.done[r]} was not dumped "
                                 "(is the tag filter missing it?)")
            self._passed(r, op)

    def _passed(self, r, op):
        self.done[r] += 1
        if self.done[r] % Z.N_OPERATORS == 0 and op != "x":
            self.st.lib.put_xedges(self.zm, self.st, r)
            self.st.lib.put_yedges(self.zm, self.st, r)

    def operator(self, bef, aft, inner=()):
        zm, st, r, op, E = self.zm, self.st, bef.region, bef.tag[-1], self.engine
        status = bef.meta["status"]
        self._advance(r, status)
        if r not in self.seeded:
            for k in STATE:
                E.copy_into(st[r][k], bef[k])
            self.seeded.add(r)
            if zm.children[r]:  # the start's update_parent ended with the parent's m2phlb
                (st.lib.m2phlb1 if r == 1 else st.lib.m2phlb)(zm, st, r)
                self.m2phlb_seen.add(r)
                if r in self.step and self.step[r]["sp"] is None:
                    g = zm.g(r)
                    self.step[r]["sp"] = st[r]["sp"][1 : g.im, 1 : g.jm, 1].copy()
        reports = [E.compare(st[r][k], bef[k], f"in:{k}") for k in STATE]
        zm.status[r] = status
        if op in Z.OPERATORS:
            Z.OPERATORS[op](zm, st, r, limits=bef.meta["limits"])
        elif op == "v":
            ingested = None
            if self.ingestion is not None and self.ingestion.has_meteo:
                ingested, in_reports = self.ingestion[r].convection(bef, self.step.get(r), engine=E)
                reports += in_reports
            conv, lmax = conv_inputs(bef, ingested)
            st.lib.convec_region(zm, st, r, dict(conv, lmax=lmax), bef.meta["nconv"], self.inverse)
        else:
            g = zm.g(r)
            if r in self.m2phlb_seen:
                phlb = st[r]["phlb"][1 : g.im, 1 : g.jm, 1 : zm.lm + 1].copy()
            else:  # before the region's first m2phlb: the initial half levels, the Python's if read
                phlb = None if self.meteo is None else self.meteo.phlb0.get(r)
            step = self.step.get(r)
            tr = step and step["tr"]
            reports += physics(zm, st, bef, inner, phlb, self.ingestion, step_tr=tr, engine=E)
        self._passed(r, op)
        return reports + [E.compare(st[r][k], aft[k], k) for k in STATE]

    def _seed_initial(self):
        """The run's start from the istart-66 file (Ingestion.initial_state), every region at
        once, before the first Setup_MassFlow -- as initexit start does; then, as in
        operator's seeding, the parent's m2phlb1 that ends update_parent."""
        ic = self.ingestion.initial_state(self.meteo.m0, self.nt)
        self.m2phlb_seen.update(seed(self.zm, self.st, ic, self.engine))
        self.seeded.update(self.zm.regions())

    def _start(self):
        """The run's start from the initial field, if read, before its first Check_CFL or
        Setup_MassFlow (every region's initial air mass is known by then)."""
        if not self.seeded and self.ingestion is not None and self.ingestion.initial is not None:
            self._seed_initial()

    def cfl(self, evs):
        """Check_CFL of the interval on the Python's own masses and balanced fluxes (on a copy:
        the rehearsal's sp / phlb, which the model leaves behind, the real Setup_MassFlow's
        coarsen_region recomputes).  (reports of region 1's event, notes)."""
        self._start()
        pairs = {r: None if self.meteo is None else self.meteo.pairs(ev) for r, ev in evs.items()}
        E = self.engine
        st = E.zeros(self.zm, 1)
        for r, ev in evs.items():
            own = r in self.seeded
            for name in ("m", "sp"):
                if own:
                    st[r][name].data[...] = self.st[r][name].data
                else:
                    E.copy_into(st[r][name], ev[name])
            st[r]["dxyp"].data[...] = self.st[r]["dxyp"].data
        m = evs[1].meta
        ndyn, n, iterations = Z.check_cfl_zoom(self.zm, st, m["ndyn_in"], m["n_in"], m["nread"],
                                                 cfl_interpolation(evs, pairs, E))
        self.ndyn = ndyn
        seeded = len(self.seeded) < self.zm.nregions or any(q is None for q in pairs.values())
        notes = {"iterations": iterations, "ndyn": ndyn, "n": n, "seeded": seeded}
        return _cfl_reports(evs[1], ndyn, n), notes

    def massflow(self, evs):
        self._start()
        E = self.engine
        seeded = {}
        for r, ev in evs.items():
            pairs = None if self.meteo is None else self.meteo.pairs(ev)
            seeded[r] = pairs is None
            if seeded[r]:
                for k in ("pu", "pv"):
                    E.copy_into(self.st[r][k], ev[k])
                continue
            t1, t2 = times(ev, "tr1", "tr2")
            pu, pv = interpolated_fluxes(ev, t1 + (t2 - t1) / 2, pairs)
            E.put(self.st[r]["pu"], pu)
            E.put(self.st[r]["pv"], pv)
            self.step[r] = {"tr": (t1, t2), "pu": pu, "pv": pv}
        if self.ndyn is None:  # no Check_CFL seen: the dumped step
            seeded = dict.fromkeys(seeded, True)
        Z.setup_massflow(self.zm, self.st, self.ndyn or evs[1].meta["ndyn"])
        for r in (r for r in evs if r in self.step and self.step[r].get("tr") == times(evs[r], "tr1", "tr2")):
            g = self.zm.g(r)
            if self.zm.children[r]:  # a parent: m2phlb1 of its mass after coarsen_region (see step_inputs)
                if r in self.seeded:
                    sp = E.column_pressure(self.zm, self.st[r]["m"][1 : g.im, 1 : g.jm, 1 : self.zm.lm],
                                           self.st[r]["dxyp"][1 : g.jm])  # fmt: skip
                else:
                    sp = None  # set when the parent is seeded
            elif r in self.m2phlb_seen:
                sp = self.st[r]["sp"][1 : g.im, 1 : g.jm, 1].copy()
            else:
                sp = self.meteo.sp0[r]
            self.step[r]["sp"] = sp
        return {r: [E.compare(self.st[r][k], ev[k], k) for k in ("pu", "pv", "am", "bm", "cm")]
                for r, ev in evs.items()}, seeded

    def m2phlb(self, ev):
        zm, st, r, E = self.zm, self.st, ev.region, self.engine
        self.m2phlb_seen.add(r)
        # Proces_Region(r) ends after tref_ do_steps: r is at the end of its current
        # stretch of N_OPERATORS * tref_ positions
        q = Z.N_OPERATORS * zm.ref(r, "t")
        self._advance(r, -(-self.done[r] // q) * q)
        if r != 1:
            st.lib.update_parent(zm, st, r)
            st.lib.m2phlb(zm, st, r)
        else:
            st.lib.m2phlb1(zm, st, 1)
        g, lm = zm.g(r), zm.lm
        interior = FView(st[r]["phlb"][1 : g.im, 1 : g.jm, 1 : lm + 1].copy(), (1, 1, 1))
        reports = [E.compare(st[r][k], ev[k], k) for k in ("m", "sp")]
        return [*reports, E.compare(interior, ev["phlb"], "phlb")]


def _summary(reports):
    bad = [r for r in reports if not r["equal"]]
    if not bad:
        return "bitwise"
    return "  ".join(f"{r['name']} {r['max_ulp']} ulp ({r['n_diff']} cells)" for r in bad)


def infer_status(zm, events):
    """A single-region set from before the zoom header (the PYD-2 sets): each bef_ / aft_ pair's
    position in the split order, the next one holding its operator after the last (the s / c
    of a transport-only build are never dumped), written into the header as `status`."""
    pos = {}
    for ev in events:
        if not ev.tag.startswith(("bef_", "aft_")) or "status" in ev.meta:
            continue
        order, r = zm.splitorderzoom[ev.region], ev.region
        if ev.tag.startswith("aft_"):
            ev.meta["status"] = pos[r] % len(order)
            continue
        p = pos.get(r, -1) + 1
        while order[p % len(order)] != ev.tag[-1]:
            p += 1
        pos[r] = p
        ev.meta["status"] = p % len(order)


def _load(dump_dir, rank):
    """The events of one rank, or (rank None) every rank's put back together (dump.merge_ranks)."""
    if rank is None:
        return load_events(dump_dir, merge=True)
    return [e for e in load_events(dump_dir) if e.meta.get("rank", 0) == rank]


# events behind the one being replayed whose arrays are dropped (read again if looked back at)
RELEASE_AFTER = 32
READ_AHEAD = 16  # events ahead whose arrays a thread reads while the current one is replayed


def _read(ev, names):
    for name in names:
        try:
            ev[name]
        except Exception:  # the replay itself will meet it, and report it
            return


class Results(list):
    """replay's [(event, kind, reports, notes)]; `failures` is [(seq, tag, repr of the exception)]
    of every event that raised, which is in no result."""

    def __init__(self, items=(), failures=()):
        super().__init__(items)
        self.failures = list(failures)


class _Session:
    """What a replay sets up once -- the events, the tree, the inputs, the solvers -- and the
    loop over a range of its events (`run`).  The per-event loop carries state only within a
    read interval (a child's meteo against its parent's, the cfl and massflow groups), so a
    range that starts at a region-1 read (`boundaries`) is replayed alone as it is in the whole."""

    def __init__(self, dump_dir, rank, lapack, verbose, out, free_run, fft, meteo_root, emissions,
                 chemistry, initial_file, meteo_grid, tags, limit, device=None,
                 fingerprint=False):  # fmt: skip
        self.engine = engine_for(device)
        self.engine.fingerprint = fingerprint
        self.events = events = _load(dump_dir, rank)
        self.inverse = batched_inverse(runtime.lapack_choice(lapack))
        fft = runtime.fft_choice(fft)
        first = {}
        for e in events:
            first.setdefault(e.region, e)
            if len(first) == max(ev.region for ev in events):
                break
        self.zm = zm = build_tree(first)
        infer_status(zm, events)
        which = "merged ranks" if rank is None else f"rank {rank}"
        self.header = (f"{len(events)} events ({which}), regions "
                       + ", ".join(f"{r}:{zm.g(r).name}" for r in zm.regions())
                       + "; tree checked against the header")  # fmt: skip
        sources = (meteo_root, emissions, chemistry, initial_file)
        self.ingestion = None if sources == (None,) * 4 else Ingestion(zm, *sources, meteo_grid=meteo_grid)
        self.met = Meteo(zm, fft, own=free_run, ingestion=self.ingestion)
        self.nt = next(e for e in events if e.tag.startswith("bef_"))["rm"].data.shape[3]
        self.dxyp = {r: e["dxyp"] for r, e in first.items()}
        self.run_ = None
        if free_run:
            dumped = {e.tag[-1] for e in events if e.tag.startswith("bef_")}
            self.run_ = FreeRun(zm, self.nt, self.inverse, self.dxyp, self.met, dumped, self.ingestion,
                                self.engine)  # fmt: skip
        self.meteo_tag = "meteo" if any(e.tag == "meteo" for e in events) else "cfl"
        self.tags, self.limit, self.verbose, self.out = tags, limit, verbose, out

    def wanted(self, kind):
        return self.tags is None or kind in self.tags

    def boundaries(self):
        """The indices at which a region-1 read interval starts: where a range may begin."""
        return [k for k, e in enumerate(self.events) if e.tag == self.meteo_tag and e.region == 1]

    def run(self, lo=0, hi=None, with_ic=True):
        """Replay events[lo:hi]; the initial field is reported only `with_ic` (the first range)."""
        events, zm, met, run, ingestion = self.events, self.zm, self.met, self.run_, self.ingestion
        limit, wanted, out, nt = self.limit, self.wanted, self.out, self.nt
        hi = len(events) if hi is None else hi
        results, pending, ic = Results(), {"cfl": {}, "massflow": {}}, None
        ic_seen = set() if with_ic else set(zm.regions())
        if not with_ic and ingestion is not None:  # a later range: its first read is not the run's start
            for r in zm.regions():
                if ingestion[r].meteo is not None:
                    ingestion[r].meteo.first = False
        released, ahead, learned = lo, lo, {}  # learned: tag -> the arrays read of such an event
        reader = ThreadPoolExecutor(1)

        def emit(ev, kind, reports, notes=None):
            notes = notes or {}
            if run is not None and ingestion is not None and ingestion.initial is not None and not results:
                notes["initial"] = "istart 66"
            if wanted(kind) and (limit is None or len(results) < limit):
                results.append((ev, kind, reports, notes))
                if self.verbose or not all(x["equal"] for x in reports):
                    print(f"e{ev.seq:05d} {kind:>8} r{ev.region}  {_summary(reports)}", file=out)

        for k in range(lo, hi):
            ev = events[k]
            if limit is not None and len(results) >= limit:
                break
            for e in events[released : max(released, k - RELEASE_AFTER)]:
                e.release()
            released = max(released, k - RELEASE_AFTER)
            if k > lo:
                learned.setdefault(events[k - 1].tag, set()).update(events[k - 1].cached())
            ahead = max(ahead, k + 1)
            while ahead < min(hi, k + 1 + READ_AHEAD):
                names = learned.get(events[ahead].tag)
                if names:
                    reader.submit(_read, events[ahead], sorted(names))
                ahead += 1
            kind = ev.tag[4:] if ev.tag.startswith("bef_") else ev.tag
            try:
                if ev.tag == self.meteo_tag and (run is not None or wanted("meteo")):
                    reports, notes = met.event(ev)
                    emit(ev, "meteo", reports, notes)
                if ev.tag in pending and (run is not None or wanted(ev.tag)):
                    group = pending[ev.tag]
                    group[ev.region] = ev
                    if len(group) < zm.nregions:
                        continue
                    pending[ev.tag] = {}
                    if ev.tag == "cfl":
                        if run is not None:
                            reports, notes = run.cfl(group)
                        else:
                            ndyn, n, iterations = replay_cfl(zm, group, engine=self.engine)
                            reports = _cfl_reports(group[1], ndyn, n)
                            notes = {"iterations": iterations, "ndyn": ndyn, "n": n}
                        emit(group[1], "cfl", reports, notes)
                    else:
                        done, seeded = (run.massflow(group) if run
                                        else (replay_massflow(zm, group, self.engine), {}))  # fmt: skip
                        for r, reports in done.items():
                            notes = {"seeded": seeded.get(r, False)} if run else {}
                            emit(group[r], "massflow", reports, notes)
                    continue
                if ev.tag == "m2phlb":
                    if run is not None:
                        emit(ev, "m2phlb", run.m2phlb(ev))
                    elif wanted("m2phlb"):
                        emit(ev, "m2phlb", replay_m2phlb(zm, ev, self.dxyp, self.engine))
                    continue
                if not ev.tag.startswith("bef_") or kind not in "xyzvsc":
                    continue
                if run is None and not wanted(kind):
                    continue
                try:
                    aft, inner = span(events, k)
                except ValueError:  # the run ended inside the pair
                    continue
                if ingestion is not None and ingestion.initial is not None and ev.region not in ic_seen:
                    ic_seen.add(ev.region)
                    if ic is None:
                        ic = ingestion.initial_state(met.m0, nt)
                    emit(ev, "ic", initial_reports(zm, ic, ev))
                notes = {}
                if kind == "s":
                    region = None if ingestion is None else ingestion[ev.region]
                    ingested = region is not None and region.emissions is not None
                    notes = {"injections": len(inner), "ingested": ingested}
                if run is not None:
                    notes["seeded"] = ev.region not in run.seeded
                    emit(ev, kind, run.operator(ev, aft, inner), notes)
                    continue
                ingested = step = None
                if ingestion is not None:
                    step = step_inputs(zm, events, k)
                    if ev.tag == "bef_v" and ingestion.has_meteo:
                        ingested = ingestion[ev.region].convection(ev, step, engine=self.engine)
                tr = step and step["tr"]
                reports = replay_operator(zm, ev, aft, self.inverse, inner, ingested, ingestion, tr,
                                          self.engine)  # fmt: skip
                emit(ev, kind, reports, notes)
            except Exception as exc:  # recorded in results.failures; carry on with the next event
                results.failures.append((ev.seq, ev.tag, repr(exc)))
                print(f"e{ev.seq:05d} {ev.tag:>8}: FAILED {exc!r}", file=out)
        reader.shutdown(cancel_futures=True)
        return results


def replay(dump_dir, rank=None, lapack="auto", verbose=False, out=None, free_run=False, fft="auto",
                meteo_root=None, emissions=None, chemistry=None, initial_file=None, meteo_grid=None,
                tags=None, limit=None, workers=1, device=None, fingerprint=False):  # fmt: skip
    """Every replayable event of a dump set -- one region or a tree -- per event or (free_run)
    carried by the Python; the inputs read from the files with `meteo_root`, `emissions`,
    `chemistry`, `initial_file` (inputs.Ingestion).  Returns [(event, kind, reports, notes)], kind
    the operator of a bef_ pair or the event's tag (meteo, cfl, massflow, m2phlb) or `ic`.
    `tags` limits what is replayed per event, and what is returned of a free run (which runs
    everything); `limit` stops after that many results.  An event that raises is reported as
    FAILED, recorded in the returned Results' `failures`, and the replay carries on.  `out` is
    where the report goes (default: sys.stdout at call time).  `workers` > 1 replays per event
    in that many processes, each a run of read intervals (the free run and a `limit` stay in
    one); their events come
    back as dump.EventRef, the order and the reports the same.  `device` 'emu' or 'cuda' runs
    every operator through pydyn.gpu with the state on the device (pydyn.gpu.engine);
    `fingerprint` adds the sha1 of the Python's array to every report."""
    out = sys.stdout if out is None else out
    args = (dump_dir, rank, lapack, verbose, out, free_run, fft, meteo_root, emissions, chemistry,
            initial_file, meteo_grid, tags, limit, device, fingerprint)  # fmt: skip
    if workers > 1 and not free_run and limit is None:
        return _replay_parallel(args, workers)
    session = _Session(*args)
    print(session.header, file=out)
    results = session.run()
    if device not in (None, "cpu"):
        print(f"device transfers ({session.engine.name}):\n{session.engine.ledger.report()}", file=out)
    return results


def _ranges(boundaries, n_events, workers):
    """`workers` contiguous ranges of read intervals with about as many events each."""
    starts = sorted({0, *boundaries})
    cuts, target = [0], n_events / workers
    for b in starts[1:]:
        if b >= target * len(cuts) and len(cuts) < workers:
            cuts.append(b)
    return list(zip(cuts, [*cuts[1:], n_events], strict=True))


def _replay_parallel(args, workers):
    import concurrent.futures as cf
    import multiprocessing as mp

    out = args[4]
    session = _Session(*args)
    print(session.header, file=out)
    ranges = _ranges(session.boundaries(), len(session.events), workers)
    del session
    quiet = (*args[:4], None, *args[5:])
    jobs = [(quiet, lo, hi, i == 0, ops.libm.path) for i, (lo, hi) in enumerate(ranges)]
    with cf.ProcessPoolExecutor(len(jobs), mp_context=mp.get_context("spawn")) as pool:
        parts = list(pool.map(_replay_range, jobs))
    results = Results()
    for text, part, failures in parts:
        out.write(text)
        results += part
        results.failures += failures
    return results


def _replay_range(job):
    import io

    args, lo, hi, with_ic, libm_path = job
    if libm_path:
        ops.use_libm(libm_path)
    buf = io.StringIO()
    session = _Session(*args[:4], buf, *args[5:])
    results = session.run(lo, hi, with_ic)
    refs = [(ev.ref(), kind, reports, notes) for ev, kind, reports, notes in results]
    return buf.getvalue(), refs, results.failures


def tally(results):
    """(kind, region, bitwise?) -> count."""
    return Counter((kind, ev.region, all(x["equal"] for x in reports)) for ev, kind, reports, _ in results)


def summarize(results):
    """Per kind: number of events, worst ulp distance and relative difference, per array."""
    table = {}
    for _, kind, reports, _ in results:
        row = table.setdefault(kind, {"events": 0, "arrays": {}})
        row["events"] += 1
        for r in reports:
            a = row["arrays"].setdefault(r["name"], {"max_ulp": 0, "max_rel": 0.0, "n_diff": 0})
            a["max_ulp"] = max(a["max_ulp"], r["max_ulp"])
            a["max_rel"] = max(a["max_rel"], r["max_rel"])
            a["n_diff"] += r["n_diff"]
    return table


def format_summary(table):
    fmt = "{:>8} {:>6}  {:<6} {:<10} {:>8} {:>10}"
    lines = [fmt.format("step", "events", "array", "result", "max ulp", "max rel")]
    for kind, row in table.items():
        for name, a in row["arrays"].items():
            result = "bitwise" if a["n_diff"] == 0 else "differs"
            lines.append(fmt.format(kind, row["events"], name, result, a["max_ulp"], f"{a['max_rel']:.2e}"))
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="pydyn-replay",
        description="Replay a pydyn_dump directory of a TM5 run through the Python and compare it "
        "bitwise with what the model produced; exit 0 only if every event is bitwise and none raised.",
    )
    p.add_argument("dump")
    p.add_argument("--rank", type=int, default=None, help="one rank's events (default: every rank merged)")
    p.add_argument("--meteo-grid", choices=("model", "1x1"), default=None,
                   help="the archive's set the rc reads: on the model grid (default for one region) or 1x1")
    p.add_argument("--tags", help="comma-separated subset of x,y,z,v,s,c,meteo,cfl,massflow,m2phlb,ic")
    p.add_argument("--limit", type=int, help="stop after this many results")
    p.add_argument("--workers", type=int, default=1,
                   help="replay per event in this many processes, by read interval (default 1)")
    runtime.add_library_args(p)
    p.add_argument("--free-run", action="store_true", help="carry the state (FreeRun) instead of per event")
    runtime.add_input_args(p)
    p.add_argument("--device", choices=("cpu", "emu", "cuda", "metal", "emu32"), default="cpu",
                   help="where the operators run: the NumPy reference (cpu) or pydyn.gpu (emu, cuda; "
                        "metal and emu32 in float32)")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    if not Path(a.dump).is_dir():
        p.error(f"{a.dump}: not a directory")
    if not any(_META.match(f.name) for f in Path(a.dump).iterdir()):
        p.error(f"{a.dump}: no pydyn_dump events (*.meta)")
    runtime.apply(a)
    try:
        return _main(a)
    except DumpError as e:
        p.exit(2, f"pydyn-replay: {e}\n")


def _main(a):
    print(runtime.describe(a.fft, a.lapack))
    results = replay(a.dump, rank=a.rank, lapack=a.lapack, verbose=a.verbose, free_run=a.free_run,
                          fft=a.fft, meteo_root=a.meteo, emissions=a.emissions, chemistry=a.chemistry,
                          initial_file=a.initial, meteo_grid=a.meteo_grid,
                          tags=None if a.tags is None else set(a.tags.split(",")), limit=a.limit,
                          workers=a.workers, device=a.device)
    counts = tally(results)
    print("\nsummary (kind, region, bitwise): count")
    for key, n in sorted(counts.items(), key=str):
        print(f"  {key}: {n}")
    if a.verbose:
        print("\n" + format_summary(summarize(results)))
    if results.failures:
        print(f"\n{len(results.failures)} events FAILED (raised); first: {results.failures[0]}")
    return 0 if results and not results.failures and all(key[-1] for key in counts) else 1


if __name__ == "__main__":
    sys.exit(main())
