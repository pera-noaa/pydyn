#!/usr/bin/env python3
"""Fetch the public dump-set fixture from Zenodo, check it and unpack it.

    python3 runs/fetch_fixture.py DIR        then   PYDYN_SETS=DIR uv run pytest -rs

Three sets (`runs/MANIFEST.md`): `repin_free_745e94f70` and `repin_cfl_745e94f70` (tm5-zoom
`745e94f70`, glb6x4, transport only) and `zm_meteo_fixture` (`397b5a2fa`, glb3x2 + nam1x1, four
meteo events).  An archive already in DIR with the record's md5 is not downloaded again, so a
CI cache of DIR costs one listing.  Standard library only.
"""

import argparse
import hashlib
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

RECORD = None  # the Zenodo record id, set when the fixture is published
API = "https://zenodo.org/api/records/{}"


def listing(record):
    """(name, url, md5) of every archive in the record."""
    with urllib.request.urlopen(API.format(record)) as r:
        files = json.load(r)["files"]
    return [(f["key"], f["links"]["self"], f["checksum"].split(":")[-1])
            for f in files if f["key"].endswith(".tar.gz")]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(files, out):
    out.mkdir(parents=True, exist_ok=True)
    for name, url, want in files:
        path = out / name
        if not (path.is_file() and md5(path) == want):
            print(f"downloading {name}", file=sys.stderr)
            urllib.request.urlretrieve(url, path)
            if md5(path) != want:
                sys.exit(f"{name}: md5 {md5(path)}, the record says {want}")
        with tarfile.open(path) as tar:
            if hasattr(tarfile, "data_filter"):
                tar.extractall(out, filter="data")
            else:  # Python < 3.10.12: the archive's members are plain files under one directory
                tar.extractall(out)
        print(f"{name}: md5 ok, unpacked", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir", type=Path, help="where the sets go (PYDYN_SETS)")
    ap.add_argument("--record", default=RECORD, help="Zenodo record id (default: the published fixture)")
    args = ap.parse_args()
    if args.record is None:
        ap.error("no published record yet: pass --record")
    fetch(listing(args.record), args.dir)
    print(f"export PYDYN_SETS={args.dir.resolve()}")


if __name__ == "__main__":
    main()
