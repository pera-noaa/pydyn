#!/usr/bin/env python3
"""Pack dump sets as the Zenodo fixture: `.tar.xz` archives, members ordered by field.

    python3 runs/pack_fixture.py SETS_DIR OUT_DIR [--config-only] SET[:FIELDS,FIELDS,...] ...

A set's `config/` directory, if it has one (the rc as the model read it, the build fingerprint,
the setup and run logs, the recipe), is `<set>.config.tar.xz`; `--config-only` packs just those.
A plain SET is one `<set>.tar.xz`.  With fields, each comma-separated group (fields joined by +)
is its own `<set>.<group>.tar.xz` and the rest of the set `<set>.other.tar.xz`; every archive
holds whole files under `<set>/`, so unpacking them all into one directory gives the set back, in
any order.  The fixture is

    repin_free_745e94f70:m,rm,rxm,rym,rzm,am+bm+cm repin_cfl_745e94f70 zm_meteo_fixture

which keeps every archive near 100 MB: Zenodo dropped a 654 MB upload and a 2.7 GB one, while
archives of 70-120 MB went up.

A set's arrays are near-duplicates of each other (an operator's bef_ and aft_, one event and the
next), so the compression is in matching them, and gzip's 32 KB window never sees two of them.
xz -9's 64 MB dictionary does, more so with each field's events next to each other:
repin_free_745e94f70's 3.5 GB is 2.68 GB as tar.gz, 808 MB as tar.xz in name order, 654 MB in
field order; the tracer and its slopes are most of what is left (rm..rzm 100-110 MB each, the
mass fluxes 5% of raw).  `tar -xJf` and Python's tarfile both read it.  Run logs are left out,
and so are owner names (--numeric-owner).  Needs xz.
"""

import re
import subprocess
import sys
from pathlib import Path

DUMP = re.compile(r"e(\d+)_(.+?)_r(\d+)_p(\d+)(?:_(.+))?\.(bin|meta)$")


def field(f):
    """A dump file's field: its array's name, or "" for an event's .meta."""
    _, _, _, _, name, ext = DUMP.match(f.name).groups()
    return "" if ext == "meta" else name


def members(root, name):
    """The set's dump files, the .meta first, then field by field in event order."""
    files = [f for f in (root / name).iterdir() if DUMP.match(f.name)]
    return sorted(files, key=lambda f: (field(f), int(DUMP.match(f.name).group(1)), f.name))


def config(root, name):
    """The set's `config/` files: how it was made (rc as read, build, logs, recipe)."""
    conf = root / name / "config"
    return sorted(p for p in conf.rglob("*") if p.is_file()) if conf.is_dir() else []


def pack(root, target, name, files):
    listing = "".join(f"{name}/{f.relative_to(root / name).as_posix()}\n" for f in files).encode()
    with open(target, "wb") as f:
        tar = subprocess.Popen(["tar", "--numeric-owner", "-cf", "-", "-T", "-"], cwd=root,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        xz = subprocess.Popen(["xz", "-9", "-T0"], stdin=tar.stdout, stdout=f)
        tar.stdout.close()
        tar.stdin.write(listing)
        tar.stdin.close()
        if tar.wait() or xz.wait():
            sys.exit(f"{target.name}: tar or xz failed")
    print(f"{target.name}: {target.stat().st_size / 1e6:.0f} MB", file=sys.stderr)


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    args = [a for a in sys.argv[1:] if a != "--config-only"]
    only_config = len(args) < len(sys.argv) - 1
    root, out = Path(args[0]), Path(args[1])
    out.mkdir(parents=True, exist_ok=True)
    for spec in args[2:]:
        name, _, groups = spec.partition(":")
        if conf := config(root, name):
            pack(root, out / f"{name}.config.tar.xz", name, conf)
        elif only_config:
            sys.exit(f"{name}: no config/")
        if only_config:
            continue
        files = members(root, name)
        if not groups:
            pack(root, out / f"{name}.tar.xz", name, files)
            continue
        taken = set()
        for group in groups.split(","):
            fields = set(group.split("+"))
            part = [f for f in files if field(f) in fields]
            if not part:
                sys.exit(f"{name}: no field of {group}")
            taken |= fields
            pack(root, out / f"{name}.{group.replace('+', '-')}.tar.xz", name, part)
        pack(root, out / f"{name}.other.tar.xz", name, [f for f in files if field(f) not in taken])


if __name__ == "__main__":
    main()
