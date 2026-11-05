#!/usr/bin/env python3
"""Fetch the public dump-set fixture from Zenodo, check it and unpack it.

    python3 runs/fetch_fixture.py DIR        then   PYDYN_SETS=DIR uv run pytest -rs

Three sets (`runs/MANIFEST.md`): `repin_free_745e94f70` and `repin_cfl_745e94f70` (tm5-zoom
`745e94f70`, glb6x4, transport only) and `zm_meteo_fixture` (`397b5a2fa`, glb3x2 + nam1x1, four
meteo events), as `runs/pack_fixture.py` packs them: `.tar.xz` archives of whole files, the first
set split by field, plus a `<set>.config.tar.xz` each (`<set>/config/`: the rc as the model read
it, the build, the logs, the recipe), all unpacking into DIR in any order.  Every archive's md5
is checked before any is unpacked, and for the published record its SHA-256 against the one
pinned here (SHA256), so the record cannot be swapped under this script.  Archive names are
plain file names and members plain files and directories inside DIR, or nothing is unpacked.
An archive already in DIR at the record's md5 is not
downloaded again, and one unpacked by an earlier run (DIR/.fetched.json, by size and mtime) is
neither hashed nor unpacked again, so a CI cache of DIR costs one listing; delete
DIR/.fetched.json to check and unpack everything anew.  Standard library only.
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

RECORD = "23955885"  # 10.5281/zenodo.23955885
SHA256 = {  # the published record's archives, from the files uploaded (runs/pack_fixture.py)
    "repin_cfl_745e94f70.config.tar.xz": "d0ab97523fecd64201e7bb024cb379c561ff976e819bf07452add1e3da9bb0dd",
    "repin_cfl_745e94f70.tar.xz": "912dfc986cca08d6a34f3d733919269c615f4e1ecb2216a85a02a4afe090169d",
    "repin_free_745e94f70.am-bm-cm.tar.xz":
        "d9f7c87ba859b805180935c84125ce8faf2013717334e0e5955a5dde5eac64e0",
    "repin_free_745e94f70.config.tar.xz": "9d4be50bd462e431377ecc6973af079a4296c80ec9eb17864f52111c642ef551",
    "repin_free_745e94f70.m.tar.xz": "785bf657d4d4a17ee68da09521237ca6510e393ff3a9b55415429519b622416f",
    "repin_free_745e94f70.other.tar.xz": "f01ac4a14c94f082711172e0206c21cd6aae6c6c064110464d57d4dce71f3221",
    "repin_free_745e94f70.rm.tar.xz": "8e3c86f88f3626c6928cef909ea8b422a0fbbff2b41c3309560b69c6741fd30f",
    "repin_free_745e94f70.rxm.tar.xz": "3c902cef8b08d21d76ab5177e2c2583b485e8aaf0409c80c09c4b21d47ad8ac2",
    "repin_free_745e94f70.rym.tar.xz": "a86efb313030262ef1a43e306446dd5c4376713997aa02edb6b9e936c05cfe7a",
    "repin_free_745e94f70.rzm.tar.xz": "a07667c394faf92596193a07ace931e092cd3917e14757c12b923efb6994ca39",
    "zm_meteo_fixture.config.tar.xz": "a32b33f978b1f40857d147005dcd1c2de265758d4c2b93eb4f8305f1221cfa35",
    "zm_meteo_fixture.tar.xz": "992799184853323db4fad5df4b3a2649b520b45e7d8ccd8c2c8809575f8b1a5c",
}
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
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
    found = [(f["key"], f["links"]["self"], f["checksum"].split(":")[-1])
             for f in files if f["key"].endswith(ARCHIVES)]  # fmt: skip
    for name, _, _ in found:
        if not NAME.match(name) or ".." in name:
            sys.exit(f"record {record}: refusing the file name {name!r}")
    if record == RECORD and sorted(n for n, _, _ in found) != sorted(SHA256):
        sys.exit(f"record {record}: its files are not the pinned ones (SHA256 in {__file__})")
    return found


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


def md5(path, algorithm="md5"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _members(tar, out):
    """The archive's members, refused unless each is a plain file or directory inside `out`:
    the check tarfile's "data" filter makes, done here too so that a Python without the
    filter (before 3.10.12) is no weaker."""
    root = out.resolve()
    members = tar.getmembers()
    for m in members:
        target = (root / m.name).resolve()
        if not (m.isfile() or m.isdir()) or m.name.startswith("/") or ".." in Path(m.name).parts \
                or not target.is_relative_to(root):
            sys.exit(f"{tar.name}: refusing the member {m.name!r}")
    return members


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
        if name in SHA256 and not unpacked(name, want) and md5(out / name, "sha256") != SHA256[name]:
            sys.exit(f"{name}: SHA-256 {md5(out / name, 'sha256')}, pinned {SHA256[name]}")
    for name, _, want in files:
        if unpacked(name, want):
            print(f"{name}: unpacked already", file=sys.stderr)
            continue
        with tarfile.open(out / name) as tar:
            members = _members(tar, out)
            if hasattr(tarfile, "data_filter"):
                tar.extractall(out, members=members, filter="data")
            else:  # Python < 3.10.12: _members made the same check
                tar.extractall(out, members=members)
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
