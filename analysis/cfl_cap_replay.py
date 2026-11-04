"""How many of production's CFL reductions survive a larger z sub-cycle cap?

`cfl_rejection_anatomy.py` takes ONE rejection apart and names the operator that
refused it.  This runs the same rehearsal over every `cfl` event in a dump set and
counts, which is the question `docs/cfl_headroom.md` section 7 ends on: the census
says glb3x2 runs below its ndyn ceiling on 59.5 % of archived hours, and the
anatomy says one 6x4 rejection was z refused a second sub-cycle.  Nothing joined
the two.  A dump set from `proj/ct-unified/rc/ch4_3x2_pydyn_cfl.rc` does.

For each event the script runs `Check_CFL` under each cap triple and records the
ndyn it chooses.  The first triple is the control: under the caps the model itself
ran, the replay must reproduce the dumped (ndyn_out, n_out) exactly.  That check is
reported first and a mismatch is a broken transcription, not a result -- at 3x2 it
is also the first time this repo's Check_CFL has been run at that resolution or
against a reduced grid it did not itself dump.

Usage:
    python3 -m analysis.cfl_cap_replay <dump_dir>... [--limits 10,1,1 10,1,2 ...]
                                       [--csv out.csv] [--max-events N]
"""

import argparse
import csv
from collections import Counter

from analysis.cfl_rejection_anatomy import run, used
from analysis.common import parse_limits
from pydyn.dump import load_events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", nargs="+", help="one or more dump dirs; events are pooled")
    ap.add_argument("--limits", nargs="*", default=["10,1,1", "10,1,2", "50,10,15"],
                    help="cap triples x,y,z; the FIRST is the control and must reproduce the dump")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--max-events", type=int, default=None)
    args = ap.parse_args()

    limits = [parse_limits(s) for s in args.limits]
    events = [e for d in args.dump for e in load_events(d) if e.tag == "cfl"]
    if args.max_events:
        events = events[: args.max_events]
    if not events:
        raise SystemExit(f"no cfl event in {' '.join(args.dump)}")

    meta0 = events[0].meta
    print(f"{len(events)} cfl events, {meta0['im']}x{meta0['jm']}x{meta0['lm']}, "
          f"reduced={meta0.get('grid_reduced')}, nread={meta0['nread']} s, "
          f"ndyn_max={meta0['ndyn_max']} s")
    print(f"caps: {' '.join(args.limits)}  (first is the control)\n")

    rows, mismatches = [], []
    for ev in events:
        meta = ev.meta
        row = {"idate": "".join(f"{v:02d}" for v in meta["idate"][:4]),
               "model_ndyn": meta["ndyn_out"], "model_n": meta["n_out"]}
        for spec, lim in zip(args.limits, limits, strict=True):
            ndyn, n, iters, trace = run(ev, lim)
            row[f"ndyn[{spec}]"] = ndyn
            row[f"n[{spec}]"] = n
            row[f"iters[{spec}]"] = iters
            row[f"used[{spec}]"] = used(trace)
        if row[f"ndyn[{args.limits[0]}]"] != meta["ndyn_out"] or row[f"n[{args.limits[0]}]"] != meta["n_out"]:
            mismatches.append(row)
        rows.append(row)

    ctrl = args.limits[0]
    print(f"CONTROL: replay under {ctrl} vs the model's own choice")
    if mismatches:
        print(f"  {len(mismatches)} / {len(rows)} events DISAGREE -- the numbers below mean nothing")
        for r in mismatches[:5]:
            print(f"    {r['idate']}: model {r['model_ndyn']} s n={r['model_n']}, "
                  f"replay {r[f'ndyn[{ctrl}]']} s n={r[f'n[{ctrl}]']}")
    else:
        print(f"  {len(rows)} / {len(rows)} events agree exactly on (ndyn, n)")

    ceiling = meta0["ndyn_max"]
    print(f"\nChosen ndyn (ceiling {ceiling} s):")
    for spec in args.limits:
        dist = Counter(r[f"ndyn[{spec}]"] for r in rows)
        reduced = sum(v for k, v in dist.items() if k < ceiling)
        steps = sum(ceiling / r[f"ndyn[{spec}]"] for r in rows) / len(rows) - 1.0
        shown = "  ".join(f"{k} s x{v}" for k, v in sorted(dist.items(), reverse=True))
        print(f"  {spec:>10}: {reduced:5d} / {len(rows)} reduced ({100.0 * reduced / len(rows):5.1f} %), "
              f"steps +{100.0 * steps:5.1f} %   [{shown}]")

    if len(args.limits) > 1:
        base = args.limits[0]
        base_reduced = [r for r in rows if r[f"ndyn[{base}]"] < ceiling]
        print(f"\nOf the {len(base_reduced)} intervals {base} reduced:")
        for spec in args.limits[1:]:
            survive = sum(1 for r in base_reduced if r[f"ndyn[{spec}]"] < ceiling)
            recovered = len(base_reduced) - survive
            pct = 100.0 * recovered / len(base_reduced) if base_reduced else 0.0
            print(f"  {spec:>10}: {survive} still reduced, "
                  f"{recovered} recovered to the ceiling ({pct:.1f} %)")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
