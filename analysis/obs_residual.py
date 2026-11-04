"""Which arm of the z-cap pair is closer to the observations?  `PYD-11` item 5, as pre-registered (`b61c488`).

Both arms share an undated `istart 66` field, run `without_chemistry` and carry the priors at
scaling 1.0, so H - y carries a common offset and drift far larger than the arms' difference, and
a change that only rescales a site's enhancement is indistinguishable from a flux error.  The
primary statistic is therefore blind to all three by construction:

    per group g (>= 48 obs on >= 7 UTC days): remove a least-squares line in time from y and from
    each arm's H, then  drho_g = corr(H'_z2, y') - corr(H'_z1, y')   (a partial correlation on t)

Uncertainty is a day-block bootstrap: October's UTC days resampled with replacement, one draw
shared by every group and both arms (as integer weights per obs), B = 2000, seed 0.

Decision on the population (section 12's 38 groups): "z2 closer" if the median drho_g has a 95 %
CI wholly above 0 and groups with CI above 0 outnumber those below at least 2:1; "z1 closer" the
mirror; "indistinguishable" if the median's CI contains 0; else "mixed".  Secondary, not in the
decision and confounded with the setup: the detrended MSE per arm, beta_g (z2 lowers it iff
beta_g > 1/2) and the raw mean residual per arm.

Usage:
    python3 -m analysis.obs_residual <z1_outdir> <z2_outdir> --lookup mdm_lookup_ch4.csv \
        --population results/obs_departure_3x2_2019-10f.csv [--ref <this pair's obs_departure csv>] \
        [--expect-n 117097] [--since 20191001] [--until 20191101] [--csv f]
    python3 -m analysis.obs_residual --selftest
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis.obs_departure import _chars, group_of, mdm_for, score

Z2, Z1, NONE, MIXED = "z2 closer", "z1 closer", "indistinguishable", "mixed"
MIN_OBS, MIN_DAYS = 48, 7
FP_FLOOR = 1e-9  # a CI bound inside this is zero: a pure rescale moves drho by rounding only


def partial_corr(w, t, h, y):
    """Weighted corr(h, y) after removing a weighted LS line in t from each; w is (B, n)."""
    sw = w.sum(1)
    mean = {k: (w @ v) / sw for k, v in (("t", t), ("h", h), ("y", y))}

    def cov(a, b, ka, kb):
        return (w @ (a * b)) / sw - mean[ka] * mean[kb]

    ctt, chh, cyy = cov(t, t, "t", "t"), cov(h, h, "h", "h"), cov(y, y, "y", "y")
    cht, cyt, chy = cov(h, t, "h", "t"), cov(y, t, "y", "t"), cov(h, y, "h", "y")
    num = chy - cht * cyt / ctt
    den = np.sqrt((chh - cht**2 / ctt) * (cyy - cyt**2 / ctt))
    return num / den


def detrend(t, x):
    a = np.vstack([np.ones_like(t), t]).T
    return x - a @ np.linalg.lstsq(a, x, rcond=None)[0]


def score_groups(groups, day, t, y, h1, h2, population, nboot=2000, seed=0):
    """Per-group statistics, the bootstrap, and the pre-registered verdict over `population`."""
    days = np.unique(day)
    dpos = {d: i for i, d in enumerate(days)}
    rng = np.random.default_rng(seed)
    counts = np.vstack([np.ones(len(days))] +
                       [np.bincount(rng.integers(0, len(days), len(days)), minlength=len(days))
                        for _ in range(nboot)])  # row 0 is the sample itself
    by = defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    rows, boot = [], {}
    for g, idx in sorted(by.items(), key=lambda kv: str(kv[0])):
        idx = np.asarray(idx)
        nd = len(np.unique(day[idx]))
        if len(idx) < MIN_OBS or nd < MIN_DAYS:
            continue
        w = counts[:, [dpos[d] for d in day[idx]]]
        tt = t[idx] - t[idx].mean()
        with np.errstate(invalid="ignore", divide="ignore"):
            r1 = partial_corr(w, tt, h1[idx], y[idx])
            r2 = partial_corr(w, tt, h2[idx], y[idx])
        dr = r2 - r1
        boot[g] = dr[1:]
        lo, hi = np.nanpercentile(dr[1:], [2.5, 97.5])
        yd, e1, e2 = (detrend(tt, v[idx]) for v in (y, h1, h2))
        dd = e2 - e1
        rows.append({
            "group": g, "n": len(idx), "ndays": nd, "in_population": g in population,
            "rho_z1": r1[0], "rho_z2": r2[0], "drho": dr[0], "drho_lo": lo, "drho_hi": hi,
            "mse_z1": float(np.mean((e1 - yd) ** 2)), "mse_z2": float(np.mean((e2 - yd) ** 2)),
            "beta": float(np.sum(dd * (yd - e1)) / np.sum(dd**2)) if np.any(dd) else float("nan"),
            "mean_resid_z1": float(np.mean(h1[idx] - y[idx])),
            "mean_resid_z2": float(np.mean(h2[idx] - y[idx])),
        })
    pop = [r for r in rows if r["in_population"]]
    if not pop:
        return rows, {"n_population": 0}, NONE
    med = np.nanmedian(np.vstack([boot[r["group"]] for r in pop]), axis=0)
    mlo, mhi = np.nanpercentile(med, [2.5, 97.5])
    up = sum(r["drho_lo"] > FP_FLOOR for r in pop)
    down = sum(r["drho_hi"] < -FP_FLOOR for r in pop)
    st = {"n_population": len(pop), "median_drho": float(np.median([r["drho"] for r in pop])),
          "median_lo": mlo, "median_hi": mhi, "groups_up": up, "groups_down": down}
    if mlo > FP_FLOOR and up >= 2 * down:
        verdict = Z2
    elif mhi < -FP_FLOOR and down >= 2 * up:
        verdict = Z1
    elif mlo <= FP_FLOOR and mhi >= -FP_FLOOR:
        verdict = NONE
    else:
        verdict = MIXED
    return rows, st, verdict


def naive_verdict(rows):
    """The scorer item 5 must not be: smaller |mean residual| wins.  Kept to prove the fixture bites."""
    better = sum(abs(r["mean_resid_z2"]) < abs(r["mean_resid_z1"]) for r in rows if r["in_population"])
    worse = sum(abs(r["mean_resid_z2"]) > abs(r["mean_resid_z1"]) for r in rows if r["in_population"])
    return Z2 if better > worse else Z1 if worse > better else NONE


def _synthetic(kind, ngroups=10, ndays=31, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(ndays * 24) / 24.0  # hourly, days
    groups, day, tt, y, h1, h2 = [], [], [], [], [], []
    for g in range(ngroups):
        walk = np.convolve(rng.normal(0, 6, len(t) + 48), np.ones(48) / 48 ** 0.5, "valid")[: len(t)]
        anom = 25 * np.sin(2 * np.pi * (t - 0.6)) + walk  # diurnal + synoptic
        truth = 1900 + anom
        obs = truth + rng.normal(0, 5, len(t))
        good = 1880 + 0.7 * anom + 0.5 * t  # low, damped, drifting: the setup's offset and flux scale
        late = np.roll(good, 3)  # the same, 3 h late: a timing error no flux scaling absorbs
        a, b = {"z1": (good, late), "z2": (late, good),
                "rescale": (good, good + 0.2 * (good - good.mean()) + 5)}[kind]
        groups += [f"g{g}"] * len(t)
        day += list(np.floor(t).astype(int))
        tt.append(t * 86400.0)
        y.append(obs), h1.append(a), h2.append(b)
    return (groups, np.asarray(day), np.concatenate(tt), np.concatenate(y),
            np.concatenate(h1), np.concatenate(h2))


def selftest():
    """Each answer must be reachable, and the confounded scorer must fail where this one holds."""
    ok = True
    pop = {f"g{g}" for g in range(10)}
    for kind, want in (("z1", Z1), ("z2", Z2), ("rescale", NONE)):
        rows, st, v = score_groups(*_synthetic(kind), pop, nboot=300)
        good = v == want
        print(f"  selftest {kind:8s}: {v:18s} (want {want}) median drho {st['median_drho']:+.4f} "
              f"[{st['median_lo']:+.4f}, {st['median_hi']:+.4f}]  {'ok' if good else 'FAIL'}")
        ok &= good
        if kind == "rescale":
            nv = naive_verdict(rows)
            bites = nv == Z2
            print(f"  selftest naive   : {nv:18s} on the rescale (must be wrong: {Z2})  "
                  f"{'ok' if bites else 'FAIL'}")
            ok &= bites
    print("selftest", "PASS" if ok else "FAIL")
    return ok


def read_dir(d, since, until):
    """{obspack_id: (H ppb, nsamples, y ppb, time s, lat, lon)} over daily files in [since, until)."""
    import re

    from netCDF4 import Dataset

    out = {}
    files = sorted(Path(d).rglob("flask_output.*.nc"))
    nf = 0
    for f in files:
        dd = re.search(r"flask_output\.(\d{8})", f.name).group(1)
        if (since and dd < since) or (until and dd >= until):
            continue
        nf += 1
        with Dataset(f) as nc:
            nc.set_auto_mask(False)
            ids = _chars(nc["obspack_id"][:])
            var = nc["flask"]
            h = np.moveaxis(np.asarray(var[:]), var.dimensions.index("obs"), 0).reshape(len(ids), -1)[:, 0]
            if not getattr(var, "units", "").startswith("mol"):
                sys.exit(f"{f}: flask units {var.units!r}")
            ns = np.asarray(nc["nsamples"][:])
        fin = f.with_name(f.name.replace("flask_output", "flask_input"))
        with Dataset(fin) as nc:
            nc.set_auto_mask(False)
            if not nc["value"].units.startswith("mol"):
                sys.exit(f"{fin}: value units {nc['value'].units!r}")
            src = {i: (nc["value"][j], nc["time"][j], nc["latitude"][j], nc["longitude"][j])
                   for j, i in enumerate(_chars(nc["obspack_id"][:]))}
        for j, i in enumerate(ids):
            v, tm, la, lo = src.get(i, (np.nan,) * 4)
            ok = lambda x: x * 1e9 if abs(x) < 1e30 else np.nan  # noqa: E731  -1e34 sentinel
            out[i] = (ok(h[j]), int(ns[j]), ok(v), float(tm), float(la), float(lo))
    return out, nf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("z1", nargs="?")
    ap.add_argument("z2", nargs="?")
    ap.add_argument("--lookup")
    ap.add_argument("--population", help="section 12's obs_departure csv: |r| > 1 defines the 38")
    ap.add_argument("--ref", help="this pair's own obs_departure csv, for Gate 0 (default: --population)")
    ap.add_argument("--expect-n", type=int, default=None)
    ap.add_argument("--since", default="20191001")
    ap.add_argument("--until", default="20191101")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--csv")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if not selftest():
        sys.exit(2)
    if a.selftest:
        return
    if not (a.z1 and a.z2 and a.lookup and a.population):
        ap.error("z1, z2, --lookup and --population are required")

    def over(path):
        with open(path) as fh:
            return {r["group"] for r in csv.DictReader(fh) if abs(float(r["r"])) > 1}

    population = over(a.population)
    ref = over(a.ref or a.population)
    lookup = {}
    with open(a.lookup) as fh:
        for r in csv.DictReader(fh):
            lookup[r["group_name"]] = float(r["mdm_ppb"])

    p, nfp = read_dir(a.z1, a.since, a.until)
    q, nfq = read_dir(a.z2, a.since, a.until)
    print(f"z1: {len(p)} obs in {nfp} files; z2: {len(q)} obs in {nfq} files; population {len(population)}")
    if set(p) != set(q):
        sys.exit(f"GATE 0: obspack_id sets differ ({len(set(p) ^ set(q))})")
    ids = [i for i in sorted(p)
           if p[i][1] > 0 and q[i][1] > 0 and np.isfinite(p[i][0]) and np.isfinite(q[i][0])]
    if a.expect_n is not None and len(ids) != a.expect_n:
        sys.exit(f"GATE 0: {len(ids)} obs sampled in both, expected {a.expect_n}")
    same_y = all(p[i][2] == q[i][2] or (np.isnan(p[i][2]) and np.isnan(q[i][2])) for i in ids)
    if not same_y:
        sys.exit("GATE 0: the arms' y differ")
    groups = [group_of(i) for i in ids]
    delta = np.array([q[i][0] - p[i][0] for i in ids])
    rows0, _, _ = score(groups, delta, mdm_for(groups, lookup))
    mine = {r["group"] for r in rows0 if abs(r["r"]) > 1}
    if mine != ref:
        sys.exit(f"GATE 0: groups over recomputed {len(mine)}, reference {len(ref)} "
                 f"({len(mine ^ ref)} differ)")
    print(f"GATE 0 passed: {len(ids)} obs, y identical in both arms, groups over = reference ({len(mine)})")

    keep = [k for k, i in enumerate(ids) if np.isfinite(p[i][2])]
    print(f"y missing (sentinel) on {len(ids) - len(keep)} obs, dropped")
    ids = [ids[k] for k in keep]
    groups = [groups[k] for k in keep]
    y = np.array([p[i][2] for i in ids])
    if not 1500 < np.median(y) < 2500:
        sys.exit(f"GATE 0: median y {np.median(y):.1f} ppb is not CH4 in ppb")
    tsec = np.array([p[i][3] for i in ids])
    t = tsec / 86400.0
    day = np.floor(t).astype(int)
    h1 = np.array([p[i][0] for i in ids])
    h2 = np.array([q[i][0] for i in ids])

    rows, st, verdict = score_groups(groups, day, t, y, h1, h2, population, nboot=a.nboot)
    loc = defaultdict(list)
    for i, g in zip(ids, groups, strict=True):
        loc[g].append(p[i][4:6])
    for r in rows:
        r["lat"], r["lon"] = np.nanmedian(np.array(loc[r["group"]]), axis=0)
    pop = [r for r in rows if r["in_population"]]
    m1 = np.median([r["mean_resid_z1"] for r in rows])
    print(f"\neligible groups {len(rows)}, of the population {st['n_population']} of {len(population)}")
    print(f"PRIMARY  median drho over the population {st['median_drho']:+.4f}  "
          f"95% CI [{st['median_lo']:+.4f}, {st['median_hi']:+.4f}]  "
          f"groups CI>0 {st['groups_up']}, CI<0 {st['groups_down']}")
    print(f"VERDICT: {verdict}")
    print("\nsecondary (descriptive, confounded with the setup):")
    print(f"  network median mean residual z1 {m1:+.1f} ppb (the setup offset, next to d_g 2-6 ppb)")
    closer = sum(abs(r["mean_resid_z2"]) < abs(r["mean_resid_z1"]) for r in pop)
    print(f"  population: median mse z1 {np.median([r['mse_z1'] for r in pop]):.2f}, "
          f"z2 {np.median([r['mse_z2'] for r in pop]):.2f} ppb^2; "
          f"median beta {np.nanmedian([r['beta'] for r in pop]):+.3f}; "
          f"groups with smaller |mean residual| at z2: {closer} of {len(pop)}")
    ra = [r for r in rows if not r["in_population"]]
    if ra:
        up, down = sum(r["drho_lo"] > FP_FLOOR for r in ra), sum(r["drho_hi"] < -FP_FLOOR for r in ra)
        print(f"  outside the population ({len(ra)}): median drho {np.median([r['drho'] for r in ra]):+.4f}, "
              f"CI>0 {up}, CI<0 {down}")
    print(f"\n{'group':46s} {'n':>4s} {'rho1':>6s} {'drho':>8s} {'lo':>8s} {'hi':>8s} {'beta':>6s} "
          f"{'m1':>7s} {'m2':>7s} {'lat':>5s}")
    for r in sorted(pop, key=lambda r: r["drho"]):
        print(f"{str(r['group'])[:46]:46s} {r['n']:4d} {r['rho_z1']:6.3f} {r['drho']:+8.4f} "
              f"{r['drho_lo']:+8.4f} {r['drho_hi']:+8.4f} {r['beta']:6.2f} "
              f"{r['mean_resid_z1']:+7.1f} {r['mean_resid_z2']:+7.1f} {r['lat']:5.1f}")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()
