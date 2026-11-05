"""Is the longer time step accurate, or only stable?

`Check_CFL` is a stability gate.  `docs/cfl_headroom.md` §8 showed that 85 % of the
intervals it shortens would pass at the full hour if the z direction were allowed a second
sub-cycle -- but passing the mass rehearsal is not the same as carrying the tracer field
acceptably, and nothing in that note measures the tracer at all.  This does.

The experiment, per meteo interval: take the tracer state the model had at the start of it,
and integrate the SAME interval at n = 1, 2, 4, ... sub-steps, so that n = 1 is one long
step and the larger n are refinements of it.  The fluxes for each sub-step are rebuilt the
way `Setup_MassFlow` rebuilds them -- interpolated to the middle of that sub-step and turned
into (am, bm, cm) at that dt -- which is verified bitwise against the dumped `massflow`
event before anything else runs.

Three things are then measured at each n:

  conservation   total tracer mass, which advection must preserve exactly.  Reported as a
                 relative drift from the starting total; anything above round-off is a defect
                 in the scheme, not a step-size effect.
  positivity     min(rm).  `slopes` with the limiter on must not go negative.
  departure      rm(n) against the finest n run.  Reported twice: relative to the field's
                 own spread, and -- because a relative number decides nothing -- as a mole
                 fraction in ppb, which can be held against the inversion's own assumed
                 model-data mismatch for CH4 (`ct.obs.mdm.default.ch4 : 15.0` ppb in
                 `ch4_3x2_cachegen_year.rc` and its sisters).  This is the
                 splitting-plus-diffusion error that grows with ndyn, and it is the number
                 the accuracy question turns on.

By default transport only: x, y and z, no convection, no sources, no chemistry.  That is
deliberate -- `Check_CFL` rehearses advection and nothing else, so the default measures the
error the checker could be said to bound.  `--convection` adds the 'v' operator where the
model puts it (xyzv | vzyx), from the set's `bef_v` events, so that what is measured is the
step's transport error and not only its advection error (`PYD-11` gap 3).  The convective
fluxes and cloud levels are constant over a meteo interval (aver3); `dkg` is not where the
model computes it live per dynamics step (`Calc_Kzz` on the step's own fluxes, as the glb3x2
rcs do), and a sub-step of the counterfactual has no dumped `dkg` of its own.  `--dkg` says
which of the model's per-step values a sub-step uses: `first` / `last` freeze the interval's
first or last, `step` takes the model step containing the sub-step's midpoint (so n = 2 at a
reduced interval reproduces production's sampling exactly).  The spread between the three is
the size of the question they leave open.  Every `bef_v` of an interval is compared with the
first and a difference is reported.

Usage:
    python3 -m analysis.step_accuracy <dump_dir> [--steps 1 2 4 8] [--intervals N]
                                      [--since YYYYMMDDHH] [--convection [--dkg first|last|step]]
                                      [--csv f]
"""

import argparse
import csv
import time
from datetime import timedelta

import numpy as np

from pydyn import inputs, kernels, replay
from pydyn.diffusion import XMAIR
from pydyn.driver import Model, State
from pydyn.dump import load_events
from pydyn.emis import XMCH4
from pydyn.kernels import CFLError
from pydyn.redgrid import RedGrid


def verify_flux_path(cfl, massflow):
    """The fluxes this script builds must be the model's own, bitwise, at the dumped ndyn."""
    t1, t2 = inputs.times(massflow, "tr1", "tr2")
    pu, pv = replay.interpolated_fluxes(cfl, t1 + (t2 - t1) / 2)
    ok = np.array_equal(pu, massflow.interior("pu")) and np.array_equal(pv, massflow.interior("pv"))
    am, bm, cm = kernels.fluxes_from_meteo(
        pu, pv, cfl["bt"][1:], massflow.meta["ndyn"] / (2.0 * massflow.meta["tref"])
    )
    for name, got in (("am", am), ("bm", bm), ("cm", cm)):
        ok = ok and (name not in massflow or np.array_equal(got, massflow.interior(name)))
    return ok


def convection_inputs(befs, mode):
    """For an interval's bef_v events: a function of time giving the kwargs of kernels.convec
    at that time under `--dkg mode`, and whether every bef_v carried the same inputs."""
    timed = [(replay.balance.date(b.meta["idate"]), *replay.conv_inputs(b)) for b in befs]
    constant = all(all(np.array_equal(timed[0][1][k], c[k]) for k in c) for _, c, _ in timed[1:])

    def at(t):
        if mode == "first":
            _, conv, lmax = timed[0]
        elif mode == "last":
            _, conv, lmax = timed[-1]
        else:  # step: the last bef_v at or before t, i.e. the model step t falls in
            _, conv, lmax = [x for x in timed if x[0] <= t][-1]
        return dict(conv, lmax=lmax)

    return at, constant


def integrate(cfl, state0, n, nread, tref, grid, limits, conv=None):
    """The interval run as n sub-steps of nread/n: x, y, z, and v too when `conv` (a function
    of the sub-step's midpoint giving convec's inputs) is given."""
    t0 = replay.balance.date(cfl.meta["tread1"])
    ndyn = nread / n
    model = Model(cfl["bt"][1:], ndyn, limits=limits, grid=grid, inverse="numpy")
    state = state0.copy()
    worst = 0.0
    for i in range(n):
        pu, pv = replay.interpolated_fluxes(cfl, t0 + timedelta(seconds=(i + 0.5) * ndyn))
        am, bm, cm = kernels.fluxes_from_meteo(pu, pv, cfl["bt"][1:], ndyn / (2.0 * tref))
        cv = None if conv is None else conv(t0 + timedelta(seconds=(i + 0.5) * ndyn))
        model.half_step(state, am, bm, cm, "xyzv", cv)
        model.half_step(state, am, bm, cm, "vzyx", cv)
        for op in "xyz":
            st = model.stats.get(op) or {}
            worst = max(worst, float(st.get("xi", 0.0)))
    return state, worst


def totals(state):
    return float(np.sum(state.rm)), float(np.min(state.rm))


def ppb(rm, m):
    """Mass mixing ratio to mole fraction in ppb, the units the tracer is judged in."""
    return rm[..., 0] / m * (XMAIR / XMCH4) * 1.0e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--steps", nargs="*", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--intervals", type=int, default=None, help="how many meteo intervals to do")
    ap.add_argument("--since", default=None, metavar="YYYYMMDDHH",
                    help="skip the intervals before this hour: a run started early for spin-up "
                         "is measured on the window it was started early for")
    ap.add_argument("--rank", type=int, default=0,
                    help="which MPI rank's events to read. The meteo is replicated across "
                         "ranks, but at ntracet=1 only one rank holds the tracer, so taking "
                         "'the first matching event' would silently pick an empty rm.")
    ap.add_argument("--convection", action="store_true",
                    help="add the model's 'v' operator (xyzv | vzyx) from the bef_v events")
    ap.add_argument("--dkg", choices=("first", "last", "step"), default="first",
                    help="with --convection: which of the model's per-step dkg a sub-step uses")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    steps = sorted(set(args.steps))
    events = [e for e in load_events(args.dump) if e.meta.get("rank", 0) == args.rank]
    cfls = [e for e in events if e.tag == "cfl"]
    if args.since:
        cfls = [e for e in cfls if "".join(f"{v:02d}" for v in e.meta["idate"][:4]) >= args.since]
    if args.intervals:
        cfls = cfls[: args.intervals]
    if not cfls:
        raise SystemExit(f"no cfl event in {args.dump}")

    # the first massflow inside the first measured interval, not the set's first
    t1, t2 = inputs.times(cfls[0], "tread1", "tread2")
    mf0 = next((e for e in events if e.tag == "massflow"
                and t1 <= inputs.times(e, "tr1", "tr2")[0] < t2), None)
    if mf0 is not None:
        ok = verify_flux_path(cfls[0], mf0)
        print(f"flux path vs the dumped massflow event: {'bitwise' if ok else 'DIFFERS -- stop'}")
        if not ok:
            raise SystemExit(1)
    else:
        print("no massflow event in this set; the flux path is unverified here")

    meta = cfls[0].meta
    print(f"{len(cfls)} intervals (rank {args.rank}), {meta['im']}x{meta['jm']}x{meta['lm']}, "
          f"reduced={meta.get('grid_reduced')}, nread={meta['nread']} s, ntracet={meta['ntracet']}")
    print(f"sub-steps compared: {steps}; the finest ({steps[-1]}) is the reference")
    befvs = [e for e in events if e.tag == "bef_v"]
    if args.convection and not befvs:
        raise SystemExit(f"--convection, but no bef_v event in {args.dump}")
    print("operators: " + (f"x y z v (xyzv | vzyx), dkg {args.dkg}" if args.convection
                             else "x y z (transport only)") + "\n")

    rows = []
    for k, cfl in enumerate(cfls, 1):
        meta = cfl.meta
        bef = next((e for e in events if e.tag == "bef_x" and e.region == cfl.region
                    and e.meta["idate"] == meta["idate"]), None)
        if bef is None:
            continue
        state0 = State(bef.interior("m"), bef.interior("rm"),
                       bef.interior("rxm"), bef.interior("rym"), bef.interior("rzm"))
        grid = RedGrid.from_meta(meta)
        conv, conv_constant = None, ""
        if args.convection:
            t1, t2 = inputs.times(cfl, "tread1", "tread2")
            here = [e for e in befvs if e.region == cfl.region
                    and t1 <= replay.balance.date(e.meta["idate"]) < t2]
            if not here:
                print(f"  [{k}/{len(cfls)}] no bef_v in this interval; skipped", flush=True)
                continue
            conv, conv_constant = convection_inputs(here, args.dkg)
            if not conv_constant:
                print(f"  [{k}/{len(cfls)}] convection inputs differ across the interval's "
                      f"{len(here)} bef_v events; dkg {args.dkg}", flush=True)
        m0, _ = totals(state0)
        out = {}
        for n in steps:
            t0 = time.monotonic()
            try:
                st, worst = integrate(cfl, state0, n, meta["nread"], meta["tref"], grid, meta["limits"],
                                      conv)
            except CFLError as exc:
                # not a bug: the REAL kernel refused this step size, having exceeded its own
                # max_nloop / iter_limit.  That is the strongest possible answer for this
                # interval -- the longer step is not merely less accurate, it is unavailable
                out[n] = (None, str(exc))
                print(f"  [{k}/{len(cfls)}] n={n:2d}  REFUSED by the kernel: {exc}", flush=True)
                continue
            out[n] = (st, worst)
            # per-interval progress, flushed: an interval whose Courant number is near 1 at
            # n = 1 makes the kernels sub-cycle hard, and those are the slow ones -- without
            # this the whole sweep is silent for over an hour and looks hung
            print(f"  [{k}/{len(cfls)}] n={n:2d}  {time.monotonic() - t0:6.1f} s  "
                  f"max Courant {worst:.4f}", flush=True)
        ref_state = out[steps[-1]][0]
        if ref_state is None:
            print(f"  [{k}/{len(cfls)}] the reference run itself was refused; interval skipped",
                  flush=True)
            continue
        ref = ref_state.rm
        spread = float(ref.max() - ref.min()) or 1.0
        ref_ppb = ppb(ref, ref_state.m)
        for n in steps:
            st, worst = out[n]
            if st is None:
                rows.append({
                    "idate": "".join(f"{v:02d}" for v in meta["idate"][:4]),
                    "model_ndyn": meta.get("ndyn_out", meta["ndyn"]),
                    "n": n, "ndyn": meta["nread"] // n,
                    "drift": float("nan"), "min_rm": float("nan"), "max_courant": float("nan"),
                    "departure": float("nan"), "rms": float("nan"),
                    "ppb_max": float("nan"), "ppb_rms": float("nan"),
                    "ppb_field": float("nan"), "refused": worst,
                    "worst_i": -1, "worst_j": -1, "worst_l": -1,
                    "ops": f"xyzv:{args.dkg}" if conv else "xyz", "conv_constant": conv_constant,
                })
                continue
            tot, lo = totals(st)
            d_ppb = np.abs(ppb(st.rm, st.m) - ref_ppb)
            # where the worst cell is, 1-based as the Fortran counts (l = 1 at the surface)
            wi, wj, wl = (int(v) + 1 for v in np.unravel_index(np.argmax(d_ppb), d_ppb.shape))
            rows.append({
                "idate": "".join(f"{v:02d}" for v in meta["idate"][:4]),
                "model_ndyn": meta["ndyn_out"] if "ndyn_out" in meta else meta["ndyn"],
                "n": n, "ndyn": meta["nread"] // n,
                "drift": (tot - m0) / m0,
                "min_rm": lo,
                "max_courant": worst,
                "departure": float(np.abs(st.rm - ref).max()) / spread,
                "rms": float(np.sqrt(np.mean((st.rm - ref) ** 2))) / spread,
                "ppb_max": float(d_ppb.max()),
                "ppb_rms": float(np.sqrt(np.mean(d_ppb ** 2))),
                "ppb_field": float(ref_ppb.max() - ref_ppb.min()),
                "refused": "",
                "worst_i": wi, "worst_j": wj, "worst_l": wl,
                "ops": f"xyzv:{args.dkg}" if conv else "xyz", "conv_constant": conv_constant,
            })

    print(f"{'n':>3} {'ndyn':>6}  {'mass drift':>12} {'min rm':>13} {'max Courant':>12} "
          f"{'departure':>11} {'rms':>10} {'ppb max':>10} {'ppb rms':>10}")
    for n in steps:
        sub = [r for r in rows if r["n"] == n and not r["refused"]]
        if not sub:
            print(f"{n:3d} {'':6}  every interval refused by the kernel at this step size")
            continue
        print(f"{n:3d} {sub[0]['ndyn']:6d}  "
              f"{max(abs(r['drift']) for r in sub):12.3e} "
              f"{min(r['min_rm'] for r in sub):13.6e} "
              f"{max(r['max_courant'] for r in sub):12.4f} "
              f"{max(r['departure'] for r in sub):11.3e} "
              f"{max(r['rms'] for r in sub):10.3e} "
              f"{max(r['ppb_max'] for r in sub):10.4f} "
              f"{max(r['ppb_rms'] for r in sub):10.4f}"
              + ("   <- reference" if n == steps[-1] else ""))
    print("\n(worst value over the intervals; departure and rms are relative to the "
          "reference run's own spread)")
    if rows:
        field = max(r["ppb_field"] for r in rows)
        print(f"the field itself spans {field:.1f} ppb; "
              f"the CH4 model-data mismatch the inversions assume is 15.0 ppb")

    for n in steps:
        ref_n = [r for r in rows if r["n"] == n and r["refused"]]
        if ref_n:
            print(f"n = {n} ({ref_n[0]['ndyn']} s): {len(ref_n)} interval(s) REFUSED outright "
                  f"by the real kernel -- the step is not available there at all, not merely "
                  f"less accurate")

    neg = [r for r in rows if not r["refused"] and r["min_rm"] < 0.0]
    print(f"\npositivity: {len(neg)} of {len(rows)} runs go negative"
          + (f", worst {min(r['min_rm'] for r in neg):.3e}" if neg else ""))

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
