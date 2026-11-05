#!/usr/bin/env python3
"""Upload the dump-set fixture to Zenodo as a draft; publishing stays a click on the site.

    python3 runs/zenodo_upload.py DIR [--sandbox] [--draft ID] [--prune] [--metadata-only]

DIR holds the `.tar.xz` archives of `runs/pack_fixture.py`, which `runs/fetch_fixture.py` reads
back.  The token needs the `deposit:write` scope (zenodo.org > Account > Applications) and is
read from $ZENODO_TOKEN, else from the macOS keychain item `zenodo` (`zenodo-sandbox` with
--sandbox), stored once with

    security add-generic-password -s zenodo -a "$USER" -w

which prompts for it.  `--sandbox` runs the same against sandbox.zenodo.org with a sandbox
token.  A file on the draft that DIR does not hold at the same md5 is removed only with
--prune; without it the run lists them and stops.  Prints the draft's page and its reserved
DOI.  A draft can be deleted, a published record cannot: check the page before pressing
Publish.  Standard library only.
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
TM5 installation. Each set holds the state and inputs of every operator call of a model run,
written by its <code>pydyn_dump</code> module: a <code>.meta</code> text file per event and a
raw little-endian, Fortran-order <code>.bin</code> per array.</p>
<ul>
<li><code>repin_free_745e94f70</code>: tm5-zoom <code>745e94f70</code>, glb6x4 (60 x 45 x 34
levels), one CH4 tracer, 2000-01-01 00-23 h, transport only; 483 events. gfortran 16.1
<code>-O3 -march=native -ffp-contract=off</code>, Accelerate LAPACK, FFTW.</li>
<li><code>repin_cfl_745e94f70</code>: the same with a 10800 s step that Check_CFL rejects;
22 events.</li>
<li><code>zm_meteo_fixture</code>: tm5-zoom <code>397b5a2fa</code>, glb3x2 + nam1x1 zoom,
2019-01-01, the four meteo events of both regions on rank 0 of an MPI run (ifort 2021.12,
production flags).</li>
</ul>
<p>Unpack every archive into one directory, in any order:
<code>for f in *.tar.xz; do tar -xJf "$f"; done</code>. <code>repin_free_745e94f70</code> is
split by field to keep each archive near 100 MB.</p>
<p>Each <code>&lt;set&gt;.config.tar.xz</code> (<code>&lt;set&gt;/config/</code>) holds what
is needed to make the set again: the rc as given and as the executable read it (every key,
<code>lmax_conv</code> included), the build fingerprint, linked libraries, setup and run logs,
and the recipe. Its <code>README.txt</code> states the two changes from the run's files:
unused rc lines naming colleagues' directories are removed, and <code>zm_replay</code>'s rc is
reconstructed, with the diff included. The rc files are TM5's configuration: this record's
CC-BY-4.0 licence does not cover them, and their terms are TM5's. The tm5-zoom source (not yet
public), the meteorology and the CH4 initial field are not included.</p>
<p>Use: <code>python3 runs/fetch_fixture.py DIR</code> in pydyn, then
<code>PYDYN_SETS=DIR uv run pytest -rs</code>. The tests are bitwise on macOS arm64 with
Accelerate and Homebrew's FFTW (measured on an Apple M4, macOS 15.8.1); elsewhere the CPU
replays differ in the last bits (pydyn's
<code>docs/state.md</code> section 3c). Each set's history is in pydyn's
<code>runs/MANIFEST.md</code>.</p>
<p>Meteorology: ECMWF ERA5. Contains modified Copernicus Climate Change Service information
2026. Neither the European Commission nor ECMWF is responsible for any use that may be made of
the Copernicus information or data it contains.</p>
"""

METADATA = {
    "title": "pydyn fixture: TM5 transport dumps for bitwise checks",
    "upload_type": "dataset",
    "description": DESCRIPTION,
    "creators": [{"name": "Pera, Ash", "orcid": "0009-0002-9019-279X",
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
        with urllib.request.urlopen(req, timeout=600) as r:
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


def upload(files_url, token, paths, prune=False):
    """Put `paths` on the draft through Zenodo's files API: a file already there at the same md5
    is kept, every other file is removed (only with `prune`: else they are listed and nothing
    changes), each upload is checked against the md5 Zenodo reports."""
    json_h = {"Content-Type": "application/json"}
    local = {p.name: p for p in paths}
    kept, stale = set(), []
    for e in call("GET", files_url, token)["entries"]:
        key = e["key"]
        if key in local and e.get("status") == "completed" and e.get("checksum") == f"md5:{md5(local[key])}":
            kept.add(key)
            print(f"{key}: already there", file=sys.stderr)
        else:
            stale.append(key)
    if stale and not prune:
        sys.exit("on the draft and not in DIR at the same md5 (pass --prune to remove them):\n  "
                 + "\n  ".join(stale))
    for key in stale:
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
    ap.add_argument("--prune", action="store_true",
                    help="remove draft files DIR does not hold at the same md5 (else: list them, stop)")
    args = ap.parse_args()
    if args.metadata_only and not args.draft:
        sys.exit("--metadata-only needs --draft")
    token = zenodo_token("zenodo-sandbox" if args.sandbox else "zenodo")
    paths = sorted(args.dir.glob("*.tar.xz"))  # runs/pack_fixture.py's archives
    if not paths and not args.metadata_only:
        sys.exit(f"no .tar.xz in {args.dir}")
    api = f"https://{'sandbox.' if args.sandbox else ''}zenodo.org/api"
    json_h = {"Content-Type": "application/json"}

    draft = args.draft or call("POST", f"{api}/deposit/depositions", token, b"{}", json_h)["id"]
    print(f"draft {draft}", file=sys.stderr)
    if not args.metadata_only:
        upload(f"{api}/records/{draft}/draft/files", token, paths, prune=args.prune)
    done = call("PUT", f"{api}/deposit/depositions/{draft}", token,
                json.dumps({"metadata": METADATA}).encode(), json_h)
    print(f"record id {done['id']}, DOI {done['metadata']['prereserve_doi']['doi']} (reserved)")
    print(f"review and publish: https://{'sandbox.' if args.sandbox else ''}zenodo.org/uploads/{draft}")


if __name__ == "__main__":
    main()
