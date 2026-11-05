"""cfl_cap_replay.py for a region tree: does the nam1x1 child change what the caps cost?

`docs/cfl_headroom.md` §8-§10 measured `Check_CFL` on glb3x2 standalone, and production is
glb3x2 + nam1x1.  This replays `zoom.check_cfl_zoom` over the `cfl` events of a zoom dump set
(`tm5-zoom@397b5a2fa:proj/ct-unified/rc/ch4_3x2nam1x1_pydyn.rc`, `PYDYN_TAGS=cfl`), one
interval at a time, under each cap triple, and records the ndyn it chooses.

Per interval, from the rank-0 `cfl` event of every region:

  tree      zoom.Zoom built from the header -- im, jm, the refinements, xcyc and the pole
            flags, region 1's reduced grid, and each child's box from its footprint -- and
            then checked against the header: the footprint `ibeg..jend`, the children and the
            split order the model itself had must be what the Python derives.  A mismatch
            stops the run: it would be a different tree's answer.
  state     every region's m, sp and dxyp as dumped; am/bm/cm are Setup_MassFlow's.
  fluxes    each region's balanced (pu, pv) interpolated to the middle of step i, as the
            single-region replay does it (672 / 672 at glb3x2, §8).

The first triple is the control: under the caps the model ran, the replay must reproduce the
dumped (ndyn_out, n_out) on every interval, and it is reported first.

Usage:
    python3 -m analysis.cfl_cap_replay_zoom <dump_dir>... [--limits 10,1,1 10,1,2 ...]
                                            [--rank 0] [--csv out.csv] [--intervals N]
"""

import argparse
import csv
from collections import Counter, defaultdict

from analysis.common import parse_limits
from pydyn.dump import load_events
from pydyn.replay import build_tree, check_tree, replay_cfl


def replay_interval(evs, zm, limits):
    """check_cfl_zoom from the dumped (ndyn_in, n_in) under `limits`: (ndyn, n, iterations)."""
    return replay_cfl(zm, evs, caps=limits)


def intervals(dumps, rank):
    """{idate: {region: cfl event}} over the dump dirs, one rank."""
    out = defaultdict(dict)
    for d in dumps:
        for e in load_events(d):
            if e.tag == "cfl" and e.meta.get("rank", 0) == rank:
                out[tuple(e.meta["idate"])][e.region] = e
    return dict(sorted(out.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", nargs="+", help="one or more zoom dump dirs; intervals are pooled")
    ap.add_argument("--limits", nargs="*", default=["10,1,1", "10,1,2", "50,10,15"],
                    help="cap triples x,y,z; the FIRST is the control and must reproduce the dump")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--intervals", type=int, default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    limits = [parse_limits(s) for s in args.limits]
    table = intervals(args.dump, args.rank)
    if args.intervals:
        table = dict(list(table.items())[: args.intervals])
    if not table:
        raise SystemExit(f"no cfl event in {' '.join(args.dump)}")
    nreg = Counter(len(v) for v in table.values())
    if len(nreg) != 1:
        raise SystemExit(f"intervals with different region counts: {dict(nreg)}")

    first = next(iter(table.values()))
    zm = build_tree(first)
    desc = ", ".join(f"{r}:{zm.g(r).name} {zm.g(r).im}x{zm.g(r).jm} tref {zm.g(r).tref}"
                     for r in zm.regions())
    print(f"{len(table)} intervals, {zm.nregions} regions ({desc}); tree checked against the header")

    rows = []
    for k, (idate, evs) in enumerate(table.items(), 1):
        if k > 1:
            check_tree(zm, evs)
        m = evs[1].meta
        row = {"idate": "".join(f"{v:02d}" for v in idate[:4]),
               "model_ndyn": m["ndyn_out"], "model_n": m["n_out"]}
        for lim in limits:
            key = ",".join(map(str, lim))
            try:
                ndyn, n, iters = replay_interval(evs, zm, lim)
            except (RuntimeError, ValueError) as exc:  # no valid time step at all
                ndyn, n, iters = -1, -1, str(exc)
            row[f"ndyn[{key}]"], row[f"n[{key}]"], row[f"iters[{key}]"] = ndyn, n, iters
        rows.append(row)
        keys = [",".join(map(str, lim)) for lim in limits]
        print(f"  [{k}/{len(table)}] {row['idate']} model {m['ndyn_out']}/{m['n_out']}  "
              + "  ".join(f"({key}) {row['ndyn[' + key + ']']}" for key in keys), flush=True)

    ctl = ",".join(map(str, limits[0]))
    match = sum(1 for r in rows if (r[f"ndyn[{ctl}]"], r[f"n[{ctl}]"]) == (r["model_ndyn"], r["model_n"]))
    print(f"\ncontrol ({ctl}): reproduces the model's (ndyn, n) on {match} of {len(rows)} intervals"
          + ("" if match == len(rows) else "  <-- NOT a result until this is all of them"))
    full = m["ndyn_max"] if "ndyn_max" in m else max(r["model_ndyn"] for r in rows)
    reduced = [r for r in rows if r["model_ndyn"] < full]
    print(f"the model ran below {full} s on {len(reduced)} of {len(rows)} intervals")
    for lim in limits[1:]:
        key = ",".join(map(str, lim))
        back = sum(1 for r in reduced if r[f"ndyn[{key}]"] == full)
        pct = 100.0 * back / len(reduced) if reduced else float("nan")
        print(f"  ({key}): {back} of those pass at {full} s ({pct:.1f} %)")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
