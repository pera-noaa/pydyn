#!/usr/bin/env python3
"""Upload the dump-set fixture to Zenodo as a draft; publishing stays a click on the site.

    python3 runs/zenodo_upload.py DIR [--sandbox]

DIR holds the `.tar.xz` archives of `runs/pack_fixture.py`, which `runs/fetch_fixture.py` reads
back.  The token needs the `deposit:write` scope (zenodo.org > Account > Applications) and is
read from $ZENODO_TOKEN, else from the macOS keychain item `zenodo` (`zenodo-sandbox` with
--sandbox), stored once with

    security add-generic-password -s zenodo -a "$USER" -w

which prompts for it.  `--sandbox` runs the same against sandbox.zenodo.org with a sandbox
token.  Prints the draft's page and its reserved DOI.  A draft can be deleted, a published record
cannot: check the page before pressing Publish.  Standard library only.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

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
<p>Every archive is a plain <code>.tar.xz</code> of whole files under its set's directory:
<code>for f in *.tar.xz; do tar -xJf "$f"; done</code> gives the three sets back, in any order.
<code>repin_free_745e94f70</code> is in seven, by field (<code>m</code>, <code>rm</code>,
<code>rxm</code>, <code>rym</code>, <code>rzm</code>, <code>am-bm-cm</code>, <code>other</code>), to keep each
near 100 MB. The arrays are near-duplicates of each other, which xz's large dictionary finds when
each field's events are next to each other: its 3.5 GB is 654 MB.</p>
<p>Each set also has a <code>&lt;set&gt;.config.tar.xz</code>, unpacking to
<code>&lt;set&gt;/config/</code>, with everything needed to make it again: the rc pycasso was
given and the rc the executable read (every key after includes and overrides: levels,
convection and <code>lmax_conv</code> macros, meteo, time steps), the build fingerprint
(compiler, flags, macros, overlay order), the linked libraries, the setup and build log, the
run's log, and the recipe script; a <code>README.txt</code> in each says how they fit. Two
changes from the files as the runs left them, both stated in that README: the rc lines naming
colleagues' directories that the run never read (the ObsPack path, with flask output off, and
three comments) are removed, and <code>zm_replay</code>'s rc is reconstructed from the run
directory's surviving rc, which a later run from the same build overwrote, with the
difference included. The model source is tm5-zoom at the commit named, not public yet; the
meteorology and the CH4 initial field are not included.</p>
<p>Use: <code>python3 runs/fetch_fixture.py DIR</code> in the pydyn repository, then
<code>PYDYN_SETS=DIR uv run pytest -rs</code>. The replay and free-run tests on these sets are
bitwise on macOS arm64 with Accelerate and Homebrew's FFTW, the libraries the glb6x4 sets were
made with; elsewhere the CPU replays differ in the last bits (pydyn's <code>docs/state.md</code>
section 3c). Each set's history is also in pydyn's <code>runs/MANIFEST.md</code>.</p>
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


def zenodo_token(service):
    """$ZENODO_TOKEN, else the macOS keychain's generic password `service`."""
    if os.environ.get("ZENODO_TOKEN"):
        return os.environ["ZENODO_TOKEN"]
    found = subprocess.run(["security", "find-generic-password", "-s", service, "-w"],
                           capture_output=True, text=True) if sys.platform == "darwin" else None
    if found is None or found.returncode or not found.stdout.strip():
        sys.exit(f"no Zenodo token: set ZENODO_TOKEN, or store it with "
                 f"`security add-generic-password -s {service} -a \"$USER\" -w`")
    return found.stdout.strip()


def call(method, url, token, body=None, headers=None):
    req = urllib.request.Request(url, data=body, method=method,
                                 headers={"Authorization": f"Bearer {token}", **(headers or {})})
    try:
        with urllib.request.urlopen(req) as r:
            text = r.read()
            return json.loads(text) if text else None
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {url}: {e.code} {e.read().decode(errors='replace')[:2000]}")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class _Progress:
    """A file read by http.client, reporting every tenth of the way."""

    def __init__(self, f, size, name):
        self.f, self.size, self.name, self.sent, self.t0, self.next = f, size, name, 0, time.time(), 0.1

    def read(self, n=-1):
        block = self.f.read(n)
        self.sent += len(block)
        if self.sent >= self.next * self.size:
            rate = self.sent / max(time.time() - self.t0, 1e-3) / 1e6
            print(f"  {self.name}: {self.sent / 1e6:.0f} of {self.size / 1e6:.0f} MB, {rate:.1f} MB/s",
                  file=sys.stderr, flush=True)
            self.next += 0.1
        return block


def upload(files_url, token, paths):
    """Put `paths` on the draft through Zenodo's files API: a file already there at the same md5
    is kept, every other file is removed, each upload is checked against the md5 Zenodo reports."""
    json_h = {"Content-Type": "application/json"}
    local = {p.name: p for p in paths}
    kept = set()
    for e in call("GET", files_url, token)["entries"]:
        key = e["key"]
        if key in local and e.get("status") == "completed" and e.get("checksum") == f"md5:{md5(local[key])}":
            kept.add(key)
            print(f"{key}: already there", file=sys.stderr)
        else:
            call("DELETE", f"{files_url}/{key}", token)
            print(f"{key}: removed", file=sys.stderr)
    for p in paths:
        if p.name in kept:
            continue
        size = p.stat().st_size
        print(f"uploading {p.name} ({size / 1e6:.0f} MB)", file=sys.stderr, flush=True)
        call("POST", files_url, token, json.dumps([{"key": p.name}]).encode(), json_h)
        with open(p, "rb") as f:
            call("PUT", f"{files_url}/{p.name}/content", token, _Progress(f, size, p.name),
                 {"Content-Type": "application/octet-stream", "Content-Length": str(size)})
        done = call("POST", f"{files_url}/{p.name}/commit", token)
        if done.get("checksum") != f"md5:{md5(p)}":
            sys.exit(f"{p.name}: Zenodo has {done.get('checksum')}, the file is md5:{md5(p)}")
        print(f"{p.name}: {done['checksum']}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir", type=Path)
    ap.add_argument("--sandbox", action="store_true", help="sandbox.zenodo.org, to rehearse")
    ap.add_argument("--draft", type=int, help="an existing draft's id, instead of a new draft")
    ap.add_argument("--metadata-only", action="store_true", help="set the metadata and upload nothing")
    args = ap.parse_args()
    token = zenodo_token("zenodo-sandbox" if args.sandbox else "zenodo")
    paths = sorted(args.dir.glob("*.tar.xz"))  # runs/pack_fixture.py's archives
    if not paths:
        sys.exit(f"no .tar.xz in {args.dir}")
    api = f"https://{'sandbox.' if args.sandbox else ''}zenodo.org/api"
    json_h = {"Content-Type": "application/json"}

    draft = args.draft or call("POST", f"{api}/deposit/depositions", token, b"{}", json_h)["id"]
    print(f"draft {draft}", file=sys.stderr)
    if not args.metadata_only:
        upload(f"{api}/records/{draft}/draft/files", token, paths)
    done = call("PUT", f"{api}/deposit/depositions/{draft}", token,
                json.dumps({"metadata": METADATA}).encode(), json_h)
    print(f"record id {done['id']}, DOI {done['metadata']['prereserve_doi']['doi']} (reserved)")
    print(f"review and publish: https://{'sandbox.' if args.sandbox else ''}zenodo.org/uploads/{draft}")


if __name__ == "__main__":
    main()
