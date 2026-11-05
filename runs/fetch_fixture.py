#!/usr/bin/env python3
"""Fetch the public dump-set fixture from Zenodo, check it and unpack it.

    python3 runs/fetch_fixture.py DIR        then   PYDYN_SETS=DIR uv run pytest -rs

Three sets (`runs/MANIFEST.md`): `repin_free_745e94f70` and `repin_cfl_745e94f70` (tm5-zoom
`745e94f70`, glb6x4, transport only) and `zm_meteo_fixture` (`397b5a2fa`, glb3x2 + nam1x1, four
meteo events), as `runs/pack_fixture.py` packs them: `.tar.xz` archives of whole files, the first
set split by field, plus a `<set>.config.tar.xz` each (`<set>/config/`: the rc as the model read
it, the build, the logs, the recipe), all unpacking into DIR in any order.  Every archive's md5
is checked before any is unpacked.  An archive already in DIR at the record's md5 is not
downloaded again, and one unpacked by an earlier run (DIR/.fetched.json, by size and mtime) is
neither hashed nor unpacked again, so a CI cache of DIR costs one listing; delete
DIR/.fetched.json to check and unpack everything anew.  Standard library only.
"""

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

RECORD = "23955885"  # 10.5281/zenodo.23955885
API = "https://zenodo.org/api/records/{}"
ARCHIVES = (".tar.xz", ".tar.gz")
TIMEOUT = 60  # seconds without a byte before a request is given up
STAMP = ".fetched.json"


def listing(record):
    """(name, url, md5) of every archive in the record."""
    url = API.format(record)
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
            files = json.load(r)["files"]
    except urllib.error.HTTPError as e:
        hint = " (not published yet, or no such record)" if e.code in (403, 404, 410) else ""
        sys.exit(f"{url}: HTTP {e.code}{hint}")
    except (urllib.error.URLError, TimeoutError) as e:
        sys.exit(f"{url}: {getattr(e, 'reason', e)}")
    return [(f["key"], f["links"]["self"], f["checksum"].split(":")[-1])
            for f in files if f["key"].endswith(ARCHIVES)]


def download(url, path):
    """`url` to `path` through a .part file, so an interrupted download never looks complete."""
    part = path.with_name(path.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
    except (urllib.error.URLError, TimeoutError) as e:
        sys.exit(f"{url}: {getattr(e, 'reason', e)}")
    part.replace(path)


def _key(path):
    st = path.stat()
    return [st.st_size, st.st_mtime_ns]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(files, out):
    out.mkdir(parents=True, exist_ok=True)
    try:
        done = json.loads((out / STAMP).read_text())
    except (OSError, ValueError):
        done = {}

    def unpacked(name, want):  # unpacked by an earlier run from this very file
        path = out / name
        return path.is_file() and done.get(name) == {"md5": want, "file": _key(path)}

    for name, url, want in files:
        path = out / name
        if unpacked(name, want) or (path.is_file() and md5(path) == want):
            continue
        print(f"downloading {name}", file=sys.stderr)
        download(url, path)
        if md5(path) != want:
            sys.exit(f"{name}: md5 {md5(path)}, the record says {want}")
    for name, _, want in files:
        if unpacked(name, want):
            print(f"{name}: unpacked already", file=sys.stderr)
            continue
        with tarfile.open(out / name) as tar:
            if hasattr(tarfile, "data_filter"):
                tar.extractall(out, filter="data")
            else:  # Python < 3.10.12: the archive's members are plain files under one directory
                tar.extractall(out)
        done[name] = {"md5": want, "file": _key(out / name)}
        (out / STAMP).write_text(json.dumps(done, indent=1))
        print(f"{name}: md5 ok, unpacked", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir", type=Path, help="where the sets go (PYDYN_SETS)")
    ap.add_argument("--record", default=RECORD, help="Zenodo record id (default: the published fixture)")
    args = ap.parse_args()
    fetch(listing(args.record), args.dir)
    print(f"export PYDYN_SETS={args.dir.resolve()}")


if __name__ == "__main__":
    main()
