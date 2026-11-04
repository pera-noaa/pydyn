"""Throughput of a whole dynamics step on a device: `python3 -m pydyn.gpu.bench --device metal`.

The production tree (glb3x2 + nam1x1, 34 levels, convection over 26 as tropo34's archive has
it) on synthetic meteo; each timed step starts from the same state (restored on the device), so
every step is the same work.  Reports wall ms a step and what it means at ndyn = 3600 s in
simulated years an hour of wall clock; --profile adds every kernel's GPU time (Metal: GPU
timestamps an encoder a dispatch) and the CPU-side operations the driver did on device arrays.

What is timed is the operator sequence of a step (zoom.step) -- advection, convection, the
exchange and m2phlb -- not the meteo ingestion, Check_CFL or the I/O around it.
"""

import argparse
import sys
import time
from collections import defaultdict

import numpy as np

from .. import synth
from .. import zoom as Z
from ..redgrid import RedGrid

__all__ = ["main", "production_case", "time_steps"]

SECONDS_A_YEAR = 365.0 * 86400.0


# glb3x2's reduced grid (the production fixture's nred, jred, clustsize)
JRED = (1, 2, 3, 4, 5, 6, 7, 84, 85, 86, 87, 88, 89, 90)
CLUSTSIZE = (40, 8, 8, 4, 4, 4, 2, 2, 4, 4, 4, 8, 8, 40)


def production_case(lm=34, lmax=26, nt=1, cfl=0.4, seed=11, reduced=True):
    grid = RedGrid(120, 90, jred=JRED, clustsize=CLUSTSIZE) if reduced else None
    glb = Z.Region("glb3x2", 120, 90, xcyc=1, touch_sp=1, touch_np=1, redgrid=grid)
    nam = Z.Region("nam1x1", 72, 44, xbeg=-132, xend=-60, ybeg=20, yend=64, xref=3, yref=2, tref=2, parent=1)
    zm = Z.Zoom([glb, nam], lm, np.linspace(2000.0, 0.0, lm + 1), np.linspace(1.0, 0.0, lm + 1))
    rng = np.random.default_rng(seed)
    st = synth.zoom_meteo(zm, rng, nt=nt, cfl=cfl)
    return zm, st, synth.zoom_convection(zm, st, rng, lmax=lmax)


def _chem_inputs(rng, im, jm, lm, ns=12, ntrop=5):
    """chem.loss_rates' inputs on an (im, jm) scope (tests/test_gpu.py's, at this size)."""
    sp = rng.uniform(9e4, 1.02e5, (im, jm))
    return {
        "temper": rng.uniform(200.0, 300.0, (im, jm, lm)),
        "phlb": sp[..., None] * np.linspace(1.0, 0.0, lm + 1)[None, None] ** 1.5,
        "lat": np.radians(-90 + 180.0 / jm * (np.arange(jm) + 0.5)),
        "strat_press": np.sort(rng.uniform(0.5, 1000.0, (jm, ns)), axis=1)[:, ::-1],
        "strat_oh": rng.uniform(0, 1e-6, (jm, ns)),
        "strat_cl": rng.uniform(0, 1e-7, (jm, ns)),
        "strat_od": rng.uniform(0, 1e-8, (jm, ns)),
        "oh_3d": rng.uniform(1e5, 1e6, (im, jm, lm)),
        "pl_trop": np.ascontiguousarray(np.sort(rng.uniform(1e4, 1e5, (im, jm, ntrop)), axis=-1)[..., ::-1]),
        "cl_trop": rng.uniform(0, 1e3, (im, jm, ntrop)),
        "soil": rng.uniform(0, 1, (im, jm)),
        "oh_scale": 0.9,
        "cl_scale": 1.1,
        "o1d_scale": 1.0,
        "soil_rate": 1e-9,
    }


def physics(zm, nt, lmax, seed=5):
    """The rest of a run's interval on the device, production-sized synthetic inputs:
    (per_interval(), sources(zoom, st, region), chemistry(zoom, st, region)) -- Calc_Kzz on
    every region once an interval (its inputs uploaded, as read), an emission field added at
    every `s`, the loss rates computed and the decay applied at every `c` (as the replay's
    chemistry events do)."""
    from . import backend as B
    from . import diffusion as GD
    from . import kernels as G
    from . import zoom as gz

    be = B.backend()
    rng = np.random.default_rng(seed)
    kzz, emis, chem = {}, {}, {}
    for r in zm.regions():
        g = zm.g(r)
        kzz[r] = synth.kzz_inputs(rng, g.im, g.jm, lm=zm.lm, lmax_conv=lmax)
        emis[r] = be.to_device(rng.uniform(0.0, 1e3, (g.im, g.jm)), "input")
        si, sj = Z.scope(zm, r)
        host = ("lat", "strat_press")  # loss_rates derives the tropopause and edges from these on the host
        chem[r] = {
            k: be.to_device(v, "input") if isinstance(v, np.ndarray) and k not in host else v
            for k, v in _chem_inputs(rng, si.stop - si.start, sj.stop - sj.start, zm.lm).items()
        }

    def sources(zoom, st, region):
        gz.sources_region(zoom, st, region, [(emis[region], 1, t) for t in range(1, nt + 1)])

    def per_interval():
        for args, grid in kzz.values():
            GD.calc_kzz(*args, **grid)

    def chemistry(zoom, st, region):
        # every `c` computes its own rates, as a run does: the second of a step runs after the
        # half step's m2phlb, on other half levels
        _, decay = G.loss_rates(1800.0, **chem[region])
        gz.chemistry_region(zoom, st, region, decay, range(1, nt + 1))

    return per_interval, sources, chemistry


def time_steps(
    zm,
    st,
    conv,
    steps=20,
    ndyn=3600,
    profile=False,
    interval=False,
    phys=None,
    speculative=False,
    graph=False,
    inverse="crout",
):
    """(wall seconds a step [list], profile dict or None) of `steps` steps on the current
    backend; interval: each step preceded by what a meteo interval of nread = ndyn adds --
    Check_CFL (its mass-only passes) and Setup_MassFlow; phys: physics()'s three, Calc_Kzz an
    interval and the step's sources and chemistry."""
    from . import backend as B
    from . import zoom as gz

    be = B.backend()
    dev = gz.upload(st)
    conv = {  # uploaded once, as a run's ingestion does for its interval
        r: {
            k: be.to_device(v, "input", np.int32 if k.startswith("cloud") else None) if k != "lmax" else v
            for k, v in c.items()
        }
        for r, c in conv.items()
    }
    Z.setup_massflow(zm, dev, ndyn)
    # two start states whose masses differ (by 1e-6): consecutive steps never see the inputs of
    # the step before, as in a run -- convec's kept inverse serves only a step's own paired calls
    pristine = []
    for factor in (1.0, 1.0 + 1e-6):
        pristine.append({r: {k: dev[r][k].data.copy() for k in dev[r]} for r in zm.regions()})
        for r in zm.regions():
            pristine[-1][r]["m"][...] = pristine[-1][r]["m"] * factor
    which = [0]

    def restore():
        which[0] ^= 1
        for r in zm.regions():
            for k, v in pristine[which[0]][r].items():
                dev[r][k].data[...] = v

    per_interval, sources, chemistry = phys if phys is not None else (None, None, None)

    def body():
        if interval:
            Z.check_cfl_zoom(zm, dev, ndyn, 1, ndyn)
            Z.setup_massflow(zm, dev, ndyn)
        if per_interval is not None:
            per_interval()
        Z.step(zm, dev, conv, ndyn, sources=sources, chemistry=chemistry, inverse=inverse)

    pipe = gz.Pipeline(zm, dev, 1)

    def one():
        if speculative or graph:
            pipe.submit(body, repeat="interval" if graph else None)  # graph: every step the same work
            pipe.drain()
        else:
            body()

    one()  # compile and warm up
    be.sync()
    walls, gpu = [], []
    for _ in range(steps):
        restore()
        be.sync()
        g0 = be.gpu_seconds() if hasattr(be, "gpu_seconds") else 0.0
        t = time.perf_counter()
        one()
        be.sync()
        walls.append(time.perf_counter() - t)
        gpu.append((be.gpu_seconds() if hasattr(be, "gpu_seconds") else 0.0) - g0)
    time_steps.gpu = gpu  # the GPU's busy time a step (Metal: its command buffers' spans)
    time_steps.tally = pipe.tally  # the speculated steps kept and redone
    prof = None
    if profile and hasattr(be, "profile_begin"):
        prof = defaultdict(lambda: [0, 0.0])
        cpu_ops = getattr(be, "cpu_ops", {})  # Metal's count of host operations on device arrays
        cpu0 = dict(cpu_ops)
        for _ in range(3):
            restore()
            be.profile_begin()
            one()
            for k, (c, s) in be.profile_end().items():
                prof[k][0] += c
                prof[k][1] += s
        prof = {k: (c / 3, s / 3) for k, (c, s) in prof.items()}
        prof["_cpu_ops"] = {k: (v - cpu0.get(k, 0)) / 3 for k, v in cpu_ops.items() if v - cpu0.get(k, 0)}
    return walls, prof


def be_syncs(device):
    from . import backend as B

    be = B.backend()
    return getattr(be, "syncs", "-")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--device", default="metal", help="metal | emu32 | emu | cuda")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--nt", type=int, default=1, help="tracers")
    p.add_argument("--lm", type=int, default=34)
    p.add_argument("--lmax", type=int, default=26, help="convection levels")
    p.add_argument("--uniform", action="store_true", help="glb3x2 without its reduced grid")
    p.add_argument("--interval", action="store_true", help="add Check_CFL and Setup_MassFlow to every step")
    p.add_argument(
        "--physics",
        action="store_true",
        help="add Calc_Kzz (an interval) and the step's sources and chemistry",
    )
    p.add_argument("--speculate", action="store_true", help="speculate every interval (gpu.zoom.Pipeline)")
    p.add_argument("--graph", action="store_true", help="speculated, and replayed as a captured graph (CUDA)")
    p.add_argument("--profile", action="store_true")
    p.add_argument(
        "--inverse",
        default="crout",
        choices=("crout", "lapack"),
        help="convection's (lapack: the reference's)",
    )
    a = p.parse_args(argv)
    from . import backend as B

    B.set_backend(a.device)
    zm, st, conv = production_case(a.lm, a.lmax, a.nt, reduced=not a.uniform)
    phys = physics(zm, a.nt, a.lmax) if a.physics else None
    walls, prof = time_steps(
        zm,
        st,
        conv,
        a.steps,
        profile=a.profile,
        interval=a.interval,
        phys=phys,
        speculative=a.speculate,
        graph=a.graph,
        inverse=a.inverse,
    )
    w = np.array(walls)
    med = float(np.median(w))
    grid = "uniform" if a.uniform else "reduced"
    what = " + ".join(
        [
            *(["Check_CFL + Setup_MassFlow"] if a.interval else []),
            *(["Calc_Kzz"] if a.physics else []),
            "step" + (" (s, c)" if a.physics else ""),
        ]
    )
    print(f"{a.device}: glb3x2 ({grid}) + nam1x1, lm {a.lm}, lmax {a.lmax}, nt {a.nt}, {a.steps} x {what}")
    print(f"  wall a step: median {med * 1e3:.1f} ms, min {w.min() * 1e3:.1f}, max {w.max() * 1e3:.1f}")
    print(f"  at ndyn 3600 s: {3600.0 / med * 3600.0 / SECONDS_A_YEAR:.2f} simulated years an hour")
    if a.speculate or a.graph:
        t = time_steps.tally
        print(f"  speculation: {t['kept']} intervals kept, {t['redone']} redone, {t['replayed']} replayed")
    if any(time_steps.gpu):
        print(
            f"  GPU busy a step (unprofiled): median {np.median(time_steps.gpu) * 1e3:.1f} ms, "
            f"syncs a step {be_syncs(a.device)}"
        )
    if prof:
        cpu = prof.pop("_cpu_ops")
        gpu = sum(s for _, s in prof.values())
        print(f"  GPU time a step (sum of dispatches): {gpu * 1e3:.1f} ms")
        for k, (c, s) in sorted(prof.items(), key=lambda kv: -kv[1][1]):
            print(f"    {k:22s} {c:6.0f} dispatches {s * 1e3:8.2f} ms")
        print(f"  CPU operations on device arrays a step: {sum(cpu.values()):.0f}")
        for k, v in sorted(cpu.items(), key=lambda kv: -kv[1])[:10]:
            print(f"    {k:22s} {v:6.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
