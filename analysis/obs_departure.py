"""Would the CT-CH4 inversion see the longer step?  `PYD-11` item 2, as pre-registered.

Two forward runs identical but for `cfl_max_nloop_z` (1 = production, 2 = the proposal), each
with `output.flask : T`, write daily `flask_output.YYYYMMDD.nc` at every obs in the GV+ CH4
obspack.  The EnKF (`ch4_gcp_monthly_c12.rc`) assimilates all of them -- the obspack has no
`CT_assim`, the rc no exclusions -- with MDM per group from `mdm_lookup_ch4.csv`, else 15 ppb.

    Delta_k = H_k(z=2) - H_k(z=1),  k over obs in the scored window
    primary    per group g: d_g = mean Delta_k against s_g = MDM_g / sqrt(N_g)
    secondary  |Delta_k| / MDM_k: median, p95, max; sum (Delta_k / MDM_k)^2 / N

Decision: "would see" if any |d_g| > s_g; "would not" if none does and p95 |Delta_k|/MDM_k
< 0.1; otherwise "between", reported as such.  The R the EnKF actually uses adds the inline
slope-gradient term (`ct.obs.mdm.rep_inline : T`); leaving it out makes s_g smaller, so this
errs toward "would see".

Usage:
    python3 -m analysis.obs_departure <z1_outdir> <z2_outdir> --lookup mdm_lookup_ch4.csv
                                      [--since YYYYMMDD] [--until YYYYMMDD] [--csv f]
    python3 -m analysis.obs_departure --selftest
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DEFAULT_MDM_PPB = 15.0  # ct.obs.mdm.default.ch4, ch4_gcp_monthly_c12.rc
SEE, NOT, BETWEEN = "would see", "would not", "between"


def group_of(oid):
    """`apply_mdm_lookup_ch4`'s rule: the second '~' field, or None."""
    parts = oid.rstrip("-").strip().split("~")
    return parts[1] if len(parts) >= 2 else None


def mdm_for(groups, lookup, default=DEFAULT_MDM_PPB):
    return np.array([lookup.get(g, default) if g is not None else default for g in groups])


def score(groups, delta, mdm):
    """Per-group and per-obs statistics, and the pre-registered verdict."""
    by = defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    rows = []
    for g, idx in sorted(by.items(), key=lambda kv: str(kv[0])):
        idx = np.asarray(idx)
        n = len(idx)
        d = float(np.mean(delta[idx]))
        s = float(np.sqrt(np.mean(mdm[idx] ** 2)) / np.sqrt(n))
        rows.append({"group": g, "n": n, "mdm_ppb": float(np.sqrt(np.mean(mdm[idx] ** 2))),
                     "d_ppb": d, "s_ppb": s, "r": d / s,
                     "max_abs_delta_ppb": float(np.max(np.abs(delta[idx])))})
    z = np.abs(delta) / mdm
    stats = {
        "n_obs": len(delta), "n_groups": len(rows),
        "n_groups_over": sum(abs(r["r"]) > 1 for r in rows),
        "mean_r2": float(np.mean([r["r"] ** 2 for r in rows])) if rows else float("nan"),
        "z_median": float(np.median(z)), "z_p95": float(np.percentile(z, 95)),
        "z_max": float(np.max(z)), "frac_z_over_1": float(np.mean(z > 1)),
        "chi2_shift": float(np.mean(z ** 2)),
    }
    if stats["n_groups_over"] > 0:
        verdict = SEE
    elif stats["z_p95"] < 0.1:
        verdict = NOT
    else:
        verdict = BETWEEN
    return rows, stats, verdict


def selftest():
    """The scorer must be able to say both answers (a gate that cannot fail is not one)."""
    rng = np.random.default_rng(0)
    groups = ["a"] * 100 + ["b"] * 400
    mdm = np.array([20.0] * 100 + [10.0] * 400)
    ok = True
    _, _, v = score(groups, np.zeros(500), mdm)
    ok &= v == NOT
    delta = np.zeros(500)
    delta[:100] = 2 * 20.0 / np.sqrt(100)  # d_a = 2 s_a
    _, st, v = score(groups, delta, mdm)
    ok &= v == SEE and st["n_groups_over"] == 1
    delta = rng.normal(0, 3.0, 500)  # zero-mean noise, p95 |z| ~ 0.3-0.6: neither
    delta -= np.array([delta[:100].mean()] * 100 + [delta[100:].mean()] * 400)
    _, _, v = score(groups, delta, mdm)
    ok &= v == BETWEEN
    print("selftest", "PASS" if ok else "FAIL")
    return ok


def _chars(a):
    a = np.asarray(a)
    if a.dtype.kind == "S" and a.ndim == 2:
        return ["".join(c.decode() for c in row).strip("\x00 ") for row in a]
    return [(x.decode() if isinstance(x, bytes) else str(x)).strip("\x00 ") for x in a]


def read_dir(d, since, until):
    """{obspack_id: (ppb, nsamples, lat, lon, alt)} over the daily files in [since, until)."""
    from netCDF4 import Dataset

    out = {}
    files = sorted(Path(d).rglob("flask_output.*.nc"))
    for f in files:
        day = re.search(r"flask_output\.(\d{8})", f.name).group(1)
        if (since and day < since) or (until and day >= until):
            continue
        with Dataset(f) as nc:
            nc.set_auto_mask(False)
            ids = _chars(nc["obspack_id"][:])
            var = nc["flask"]  # MDF dims are Fortran order, so on disk (obs, member, species)
            v = np.moveaxis(np.asarray(var[:]), var.dimensions.index("obs"), 0).reshape(len(ids), -1)[:, 0]
            units = getattr(var, "units", "")
            ns = np.asarray(nc["nsamples"][:])
        if not units.startswith("mol"):
            sys.exit(f"{f}: flask units {units!r}, expected a mole fraction")
        scale = 1e9  # mol/mol -> ppb
        v = np.where(np.abs(v) < 1e30, v, np.nan)  # -1e34 sentinel
        fin = f.with_name(f.name.replace("flask_output", "flask_input"))
        loc = {}
        if fin.exists():
            with Dataset(fin) as nc:
                nc.set_auto_mask(False)
                fids = _chars(nc["obspack_id"][:])
                lat, lon, alt = (np.asarray(nc[k][:]) for k in ("latitude", "longitude", "altitude"))
                loc = {i: (lat[j], lon[j], alt[j]) for j, i in enumerate(fids)}
        for j, i in enumerate(ids):
            out[i] = (v[j] * scale, int(ns[j]), *loc.get(i, (np.nan,) * 3))
    return out, len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("z1", nargs="?")
    ap.add_argument("z2", nargs="?")
    ap.add_argument("--lookup", help="mdm_lookup_ch4.csv (group_name,mdm_ppb)")
    ap.add_argument("--since", default=None, metavar="YYYYMMDD")
    ap.add_argument("--until", default=None, metavar="YYYYMMDD")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if not selftest():
        sys.exit(2)
    if args.selftest:
        return
    if not (args.z1 and args.z2 and args.lookup):
        ap.error("z1, z2 and --lookup are required")

    lookup = {}
    with open(args.lookup) as fh:
        r = csv.DictReader(fh)
        if r.fieldnames != ["group_name", "mdm_ppb"]:
            sys.exit(f"{args.lookup}: header {r.fieldnames}")
        for row in r:
            lookup[row["group_name"]] = float(row["mdm_ppb"])

    a, na = read_dir(args.z1, args.since, args.until)
    b, nb = read_dir(args.z2, args.since, args.until)
    print(f"z1: {len(a)} obs in {na} files; z2: {len(b)} obs in {nb} files")
    if set(a) != set(b):
        sys.exit(f"GATE: obspack_id sets differ ({len(set(a) ^ set(b))} not in both)")
    ids = [i for i in sorted(a) if a[i][1] > 0 and b[i][1] > 0
           and np.isfinite(a[i][0]) and np.isfinite(b[i][0])]
    ns_diff = sum(a[i][1] != b[i][1] for i in a)
    print(f"sampled in both: {len(ids)}; unsampled dropped: {len(a) - len(ids)}; "
          f"nsamples differ (time step changed the averaging count): {ns_diff}")
    if not ids:
        sys.exit("GATE: nothing sampled")

    delta = np.array([b[i][0] - a[i][0] for i in ids])
    groups = [group_of(i) for i in ids]
    mdm = mdm_for(groups, lookup)
    n_def = sum(g not in lookup for g in groups)
    print(f"MDM: {len(ids) - n_def} obs from the lookup, {n_def} at the {DEFAULT_MDM_PPB} ppb default")
    if not np.any(delta):
        print("WARNING: Delta is identically zero -- check Gate 0 (did the z cap take?)")

    rows, st, verdict = score(groups, delta, mdm)
    loc = defaultdict(list)
    for i, g in zip(ids, groups):
        loc[g].append(a[i][2:])
    for r in rows:
        la, lo, al = np.nanmedian(np.array(loc[r["group"]], dtype=float), axis=0)
        r.update(lat=la, lon=lo, alt_m=al)

    print(f"\nobs {st['n_obs']}, groups {st['n_groups']}")
    print(f"primary   groups with |d_g| > s_g: {st['n_groups_over']}   mean (d_g/s_g)^2: {st['mean_r2']:.3f}")
    print(f"secondary |Delta|/MDM median {st['z_median']:.4f}  p95 {st['z_p95']:.4f}  "
          f"max {st['z_max']:.3f}  frac > 1 {st['frac_z_over_1']:.2e}  "
          f"sum (Delta/MDM)^2 / N {st['chi2_shift']:.2e}")
    print(f"VERDICT: {verdict}")
    worst = sorted(rows, key=lambda r: -abs(r["r"]))[:15]
    print("\n" + f"{'group':48s} {'n':>5s} {'MDM':>6s} {'d_g':>7s} {'s_g':>7s} {'d/s':>5s} "
          f"{'max|D|':>6s} {'lat':>6s} {'lon':>6s}")
    for r in worst:
        print(f"{str(r['group'])[:48]:48s} {r['n']:5d} {r['mdm_ppb']:6.1f} {r['d_ppb']:7.3f} "
              f"{r['s_ppb']:7.3f} {r['r']:5.2f} {r['max_abs_delta_ppb']:6.2f} "
              f"{r['lat']:6.1f} {r['lon']:6.1f}")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
