#!/usr/bin/env python3
"""Upload the dump-set fixture to Zenodo as a draft; publishing stays a click on the site.

    ZENODO_TOKEN=... python3 runs/zenodo_upload.py DIR [--sandbox]

DIR holds the three `<set>.tar.gz` that `runs/fetch_fixture.py` reads back.  The token needs the
`deposit:write` scope (zenodo.org > Account > Applications); `--sandbox` runs the same against
sandbox.zenodo.org with a sandbox token.  Prints the draft's page and its reserved DOI.  A
draft can be deleted, a published record cannot: check the page before pressing Publish.
Standard library only.
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

SETS = ["repin_free_745e94f70", "repin_cfl_745e94f70", "zm_meteo_fixture"]

DESCRIPTION = """\
<p>Model dumps that let <a href="https://github.com/pera-noaa/pydyn">pydyn</a>, a NumPy
transcription of the TM5 transport core, be checked bitwise against the Fortran model without a
TM5 installation. Each set is the state and inputs of every operator call of a real model run,
written by the model's rc-gated <code>pydyn_dump</code> module: one
<code>.meta</code> text file per event (tag, date, grid scalars, and the bounds of each array) and
one raw little-endian, Fortran-order <code>.bin</code> per array.</p>
<ul>
<li><code>repin_free_745e94f70</code>: tm5-zoom <code>745e94f70</code>, glb6x4 (60 x 45 x 34
levels), one CH4 tracer, 2000-01-01 00-23 h, transport only
(<code>without_emission without_chemistry</code>); 483 events.
gfortran 16.1 <code>-O3 -march=native -ffp-contract=off</code>, Accelerate LAPACK, FFTW.</li>
<li><code>repin_cfl_745e94f70</code>: the same with a 10800 s step that Check_CFL rejects;
22 events.</li>
<li><code>zm_meteo_fixture</code>: tm5-zoom <code>397b5a2fa</code>, glb3x2 (reduced grid) +
nam1x1 zoom, 2019-01-01, the four meteo events of both regions on rank 0 of an MPI run (ifort
2021.12 at the production flags).</li>
</ul>
<p>Use: <code>python3 runs/fetch_fixture.py DIR</code> in the pydyn repository, then
<code>PYDYN_SETS=DIR uv run pytest -rs</code>. The replay and free-run tests on these sets are
bitwise; the provenance of each set is in pydyn's <code>runs/MANIFEST.md</code>.</p>
<p>Meteorology: ECMWF ERA5. Contains modified Copernicus Climate Change Service information
2026. Neither the European Commission nor ECMWF is responsible for any use that may be made of
the Copernicus information or data it contains.</p>
"""

METADATA = {
    "title": "pydyn fixture: TM5 transport dumps for bitwise checks",
    "upload_type": "dataset",
    "description": DESCRIPTION,
    "creators": [{"name": "Pera, Ash",
                  "affiliation": "CIRES, University of Colorado Boulder; NOAA Global Monitoring Laboratory"}],
    "license": "cc-by-4.0",
    "version": "1",
    "keywords": ["TM5", "atmospheric transport", "reference implementation", "bitwise reproducibility"],
    "related_identifiers": [{"relation": "isSupplementTo", "identifier": "https://github.com/pera-noaa/pydyn",
                             "resource_type": "software"}],
    "prereserve_doi": True,
}


def call(method, url, token, body=None, headers=None):
    req = urllib.request.Request(url, data=body, method=method,
                                 headers={"Authorization": f"Bearer {token}", **(headers or {})})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir", type=Path)
    ap.add_argument("--sandbox", action="store_true", help="sandbox.zenodo.org, to rehearse")
    args = ap.parse_args()
    token = os.environ.get("ZENODO_TOKEN") or sys.exit("set ZENODO_TOKEN")
    paths = [args.dir / f"{s}.tar.gz" for s in SETS]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        sys.exit(f"missing: {' '.join(missing)}")
    api = f"https://{'sandbox.' if args.sandbox else ''}zenodo.org/api/deposit/depositions"
    json_h = {"Content-Type": "application/json"}

    draft = call("POST", api, token, b"{}", json_h)
    print(f"draft {draft['id']}", file=sys.stderr)
    for p in paths:
        print(f"uploading {p.name} ({p.stat().st_size / 1e9:.2f} GB)", file=sys.stderr)
        with open(p, "rb") as f:
            call("PUT", f"{draft['links']['bucket']}/{p.name}", token, f,
                 {"Content-Type": "application/octet-stream", "Content-Length": str(p.stat().st_size)})
    done = call("PUT", f"{api}/{draft['id']}", token, json.dumps({"metadata": METADATA}).encode(), json_h)
    print(f"record id {done['id']}, DOI {done['metadata']['prereserve_doi']['doi']} (reserved)")
    print(f"review and publish: {done['links']['html']}")


if __name__ == "__main__":
    main()
