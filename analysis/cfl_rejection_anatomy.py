"""Why Check_CFL rejected a time step, operator by operator.

`Check_CFL` answers with one bit per interval: this `ndyn` is not allowed.  The reference
can answer the question behind it — which of the six operators of the mass-only rehearsal
ran out of sub-cycles, in which cell, at what Courant number, and whether the violation is
real or an artifact of the rehearsal's own cell depletion.

It does that by patching two names inside `pydyn.cfl`:

  subcycles      a spy that forwards to the real one and records the |Courant| field of
                 every trial sub-step (the `seen` hook the kernels already expose)
  rehearse_step  a clone of the original, identical statement for statement, that names the
                 operator it is running and can pass per-direction sub-cycle limits

`check_cfl` itself, and the three mass kernels it drives, are the transcribed ones: the
baseline run below reproduces the dumped `(ndyn, n, iterations)` exactly, which is the
check that the instrumentation changed nothing.

The limits are the lever.  `advectm_cfl.F90` caps each direction's retries with
`cfl_max_nloop_{x,y,z}`; on `NOAA/ct-unified` they are 10, 1, 1, so y and z give up on the
first overshoot and the whole interval is retried at half the time step.  `--limits` runs
the same rehearsal at other caps and reports the `ndyn` each would have chosen.  That is a
counterfactual about the checker, not a parity claim, and it says nothing on its own about
whether the real kernels are stable at the larger step.

    python3 -m analysis.cfl_rejection_anatomy [<dump_dir>] [--limits 10,1,1 10,6,15 ...]
"""

import argparse
import os
from datetime import timedelta
from pathlib import Path

import numpy as np

import pydyn.cfl as C
from pydyn import balance, kernels, replay
from pydyn.dump import load_events

TM5ZOOM = Path(os.environ.get("PYDYN_TM5ZOOM", Path(__file__).resolve().parents[2] / "tm5-zoom"))
SCRATCH = TM5ZOOM / ".tm5-cloud-scratch"
DEFAULT_DUMP = SCRATCH / "pydyn_build" / "dumps" / "repin_cfl_745e94f70"
REAL_KERNEL_LIMITS = (50, 10, 15)  # dynamu / dynamv / dynamw, what the advection itself allows
SEQUENCE = "xyzzyx"

_real_subcycles = C.subcycles
_trace = []          # one dict per subcycles() call of the current rehearsal
_ctx = {"op": None, "call": 0}


def _spy(m, f, nmax, shrink, axis=0, periodic=False, seen=None):
    trials = []

    def record(c, ok):
        flat = int(np.argmax(c))
        trials.append({"max": float(c.max()), "where": np.unravel_index(flat, c.shape), "ok": ok})

    out = _real_subcycles(m, f, nmax, shrink, axis=axis, periodic=periodic, seen=record)
    _trace.append({"op": _ctx["op"], "call": _ctx["call"], "shape": tuple(m.shape),
                   "nloop": None if out is None else out[0], "trials": trials})
    _ctx["call"] += 1
    return out


def make_rehearse(limits):
    """`cfl.rehearse_step` with the operator named and the three caps injected."""
    lx, ly, lz = limits

    def rehearse_step(m, am, bm, cm, grid=None):
        operators = {"x": (C.dynamum, (am.copy(), grid), {"max_nloop": lx}),
                     "y": (C.dynamvm, (bm.copy(),), {"max_nloop": ly}),
                     "z": (C.dynamwm, (cm.copy(),), {"iter_limit": lz})}
        m0 = m.copy()
        for pos, op in enumerate(SEQUENCE):
            kernel, args, kw = operators[op]
            _ctx["op"], _ctx["call"] = op, 0
            if kernel(m, *args, **kw) is None:
                _trace.append({"failed": op, "pos": pos, "m_start": m0, "m_now": m.copy(),
                               "flux": args[0], "grid": grid, "call_rec": _trace[-1]})
                return False
        return True

    return rehearse_step


def run(ev, limits):
    """check_cfl with the given caps. Returns (ndyn, n, iterations, trace)."""
    global _trace
    _trace = []
    C.subcycles = _spy
    C.rehearse_step = make_rehearse(limits)
    try:
        return (*replay.check_cfl(ev), _trace)
    finally:
        C.subcycles = _real_subcycles
        C.rehearse_step = _real_rehearse


_real_rehearse = C.rehearse_step


def cell_of(rec, lm):
    """The (j, l) or layer the failing call belongs to, from its position in the kernel's loops."""
    if rec["op"] == "x":
        return f"row j={rec['call'] // lm}, level l={rec['call'] % lm}"
    if rec["op"] == "y":
        return f"level l={rec['call']}"
    return "whole region"


def used(trace):
    """Largest sub-cycle count each direction needed, over the whole rehearsal."""
    out = {}
    for r in trace:
        if "op" in r and r["nloop"] is not None:
            out[r["op"]] = max(out.get(r["op"], 1), r["nloop"])
    return ", ".join(f"{k}={out.get(k, 1)}" for k in "xyz")


def courant_z(m, cm):
    """|cm / m_upwind| on every interior z face, the quantity dynamwm tests."""
    faces = cm[..., 1:-1]
    up = np.where(faces >= 0.0, m[..., :-1], m[..., 1:])
    return np.abs(faces / up)


def report_failure(trace, lm):
    fail = next((r for r in trace if "failed" in r), None)
    if fail is None:
        return None
    rec = fail["call_rec"]
    worst = max(rec["trials"], key=lambda t: t["max"])
    print(f"  rejected by:   {fail['failed']}  (operator {fail['pos'] + 1} of {SEQUENCE})")
    print(f"  where:         {cell_of(rec, lm)}, |Courant| = {worst['max']:.4f} "
          f"at index {worst['where']}")
    print(f"  trial sub-steps evaluated before giving up: {len(rec['trials'])}")
    return fail


def artifact_test(fail):
    """Is a z rejection real, or made by the x/y passes that ran before it in the rehearsal?

    The rehearsal advances the mass field through x and y before z is tested, so z sees a
    cell that the earlier operators have already drained.  Both cm and m scale with ndyn,
    so a violation that only exists after that draining cannot be cured by halving ndyn —
    that is the scale-invariance argued in tm5-zoom's bugreport_cfl_y_substepping.md.
    """
    if fail is None or fail["failed"] != "z":
        return
    c_now = courant_z(fail["m_now"], fail["flux"]).max()
    c_start = courant_z(fail["m_start"], fail["flux"]).max()
    print(f"  z |Courant| vs the depleted mass it was tested against: {c_now:.4f}")
    print(f"  z |Courant| vs the mass at the start of the step:       {c_start:.4f}")
    if c_start < 1.0 <= c_now:
        print("  -> ARTIFACT: the column was only over CFL after the x/y passes drained it.")
    else:
        print("  -> REAL: over CFL against the undepleted mass too.")


def headroom(ev):
    """Largest |Courant| per direction at the rejected ndyn, against the undepleted mass:
    what the step actually demands, before any rehearsal bookkeeping."""
    meta = ev.meta
    m = ev.interior("m")
    # the fluxes of the first rehearsed step, built exactly as check_cfl builds them
    t0 = balance.date(meta["tread1"])
    pu, pv = replay.interpolated_fluxes(ev, t0 + timedelta(seconds=0.5 * meta["ndyn_in"]))
    am, bm, cm = kernels.fluxes_from_meteo(pu, pv, ev["bt"][1:], meta["ndyn_in"] / (2.0 * meta["tref"]))
    up_x = np.where(am[1:-1] >= 0, m[:-1], m[1:])
    up_y = np.where(bm[:, 1:-1] >= 0, m[:, :-1], m[:, 1:])
    print(f"  x: {np.abs(am[1:-1] / up_x).max():7.4f}   y: {np.abs(bm[:, 1:-1] / up_y).max():7.4f}"
          f"   z: {courant_z(m, cm).max():7.4f}   (|Courant| < 1 is one sub-step)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", nargs="?", default=str(DEFAULT_DUMP))
    ap.add_argument("--limits", nargs="*", default=["10,1,1", "10,1,2", "10,6,10", "50,10,15"],
                    help="x,y,z sub-cycle caps to try; the first is treated as the baseline")
    args = ap.parse_args()

    events = [e for e in load_events(args.dump) if e.tag == "cfl"]
    if not events:
        raise SystemExit(f"no cfl event in {args.dump}")

    for ev in events:
        meta = ev.meta
        print(f"\n=== cfl event, {meta['idate']}, {meta['im']}x{meta['jm']}x{meta['lm']}, "
              f"reduced={meta.get('grid_reduced')}, nread={meta['nread']} s")
        print(f"    the model chose ndyn {meta['ndyn_in']} -> {meta['ndyn_out']} s, "
              f"n {meta['n_in']} -> {meta['n_out']}")

        print("\n  Courant numbers the interval demands at the rejected ndyn "
              f"({meta['ndyn_in']} s), against the undepleted mass:")
        headroom(ev)

        for i, spec in enumerate(args.limits):
            limits = tuple(int(v) for v in spec.split(","))
            ndyn, n, iterations, trace = run(ev, limits)
            tag = "  <- production today" if i == 0 else ""
            if limits == REAL_KERNEL_LIMITS:
                tag = "  <- what the advection kernels themselves allow"
            print(f"\n  limits x,y,z = {limits}: ndyn {ndyn} s, n = {n}, "
                  f"{iterations} iteration(s){tag}")
            if i == 0:
                ok = (ndyn, n) == (meta["ndyn_out"], meta["n_out"])
                print(f"  baseline reproduces the dump: {ok}")
                if not ok:
                    raise SystemExit("instrumentation changed the answer — stop here")
            fail = report_failure(trace, meta["lm"])
            artifact_test(fail)
            if fail is None:
                print(f"  sub-cycles used: {used(trace)}")


if __name__ == "__main__":
    main()
