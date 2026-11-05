"""Census of the CFL caches: how often does Check_CFL actually reduce ndyn?

Read-only, and it answers the question `docs/cfl_headroom.md` §6 left open — whether the
checker's sub-cycle caps cost production anything — from data that already exists. The
per-year cache files hold the `ndyn` Check_CFL chose for every meteo interval of 2000-2025,
and their global attributes carry the `max_nloop_{x,y,z}` the choice was made under, so the
census says which checker it is describing.

Runs where the caches are (Orion; `PYDYN_CFL_CACHE` names the directory elsewhere):

    python3 -m analysis.cfl_cache_census

Results and their reading: `docs/cfl_headroom.md` §7. `PYD-11`.
"""

import glob
import os
from collections import Counter

import numpy as np
from netCDF4 import Dataset

ROOT = os.environ.get("PYDYN_CFL_CACHE", "/work2/noaa/co2/METEO/tm5-nc/ec/ea/cfl/tropo34")
GRIDS = ["glb3x2", "glb3x2+nam1x1", "glb100x100"]

for grid in GRIDS:
    files = sorted(glob.glob(f"{ROOT}/*/cfl_ndyn3600_{grid}.nc"))
    ndyns, times, fingerprints, per_year = [], [], set(), {}
    for path in files:
        with Dataset(path) as ds:
            fingerprints.add(tuple(
                ds.getncattr(a) for a in
                ("ndyn_max", "nread", "nregions", "max_nloop_x", "max_nloop_y", "max_nloop_z")))
            nd = np.asarray(ds.variables["ndyn"][:])
            ndyns.append(nd)
            times.append(np.asarray(ds.variables["time6"][:]))
            year = os.path.basename(os.path.dirname(path))
            per_year[year] = (len(nd), int((nd < ds.getncattr("ndyn_max")).sum()))
    if not files:
        print(f"{grid}: no files")
        continue
    nd = np.concatenate(ndyns)
    t6 = np.concatenate(times)
    fp = sorted(fingerprints)
    ndyn_max = fp[0][0]
    reduced = nd < ndyn_max
    # every operator runs ndyn_max/ndyn times as often as it would at the ceiling
    cost = float(np.sum(ndyn_max / nd) / len(nd) - 1.0)

    print(f"\n=== {grid}: {len(files)} years, {len(nd)} cached intervals")
    for f in fp:
        print(f"    fingerprint ndyn_max={f[0]} nread={f[1]} nregions={f[2]} "
              f"max_nloop x,y,z = {f[3]},{f[4]},{f[5]}" + ("  [MIXED]" if len(fp) > 1 else ""))
    print(f"    reduced below ndyn_max: {int(reduced.sum())} ({100.0 * reduced.mean():.4f} %)")
    print(f"    extra dynamic steps over the whole archive: {100.0 * cost:+.4f} %")
    print(f"    ndyn values: {dict(sorted(Counter(nd.tolist()).items(), reverse=True))}")
    if reduced.any():
        print(f"    by month: {dict(sorted(Counter(int(r[1]) for r in t6[reduced]).items()))}")
        print(f"    by year:  {dict(sorted(Counter(int(r[0]) for r in t6[reduced]).items()))}")
    thin = {y: c for y, c in sorted(per_year.items()) if c[0] < 8000}
    if thin:
        print(f"    years with fewer than 8000 entries (incomplete): {thin}")
