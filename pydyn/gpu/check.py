"""Is the device run the CPU reference's, bit for bit?  One dump set replayed twice -- on the
NumPy reference and through pydyn.gpu -- with the twin libraries on both (--lapack twin,
--libm twin: the open DGETRF/DGETRI and exp/log/pow the device runs), every comparison
fingerprinted (sha1 of the Python's array), and the two runs' fingerprints matched checkpoint
by checkpoint.  Equal fingerprints everywhere is the claim; the dumps are only the inputs.
The float32 Metal device is matched against its own reference, the host's float32 build
(--device metal --reference emu32), not the float64 one.

    python3 -m pydyn.gpu.check <dump_dir> [--device emu|cuda|metal] [--reference cpu|emu32]
                               [--free-run] [--meteo ...] [--emissions ...] [--chemistry ...]
                               [--initial ...] [--tags ...] [--limit N] [--lapack twin|crout]

Exit status 0 when every fingerprinted array of every checkpoint is equal and both runs
produced the same checkpoints.  The device run's transfer ledger is printed with it.
"""

import argparse
import io
import sys
from collections import defaultdict

from .. import ops, replay, runtime
from .backend import host_library

__all__ = ["match", "run_both"]


def run_both(dump_dir, device="emu", lapack="twin", reference="cpu", **kw):
    """(reference results, device results, device ledger text) of one set, the twin libm
    applied; the reference the CPU's (cpu) or a device's (emu32 for metal)."""
    ops.use_libm(str(host_library()))
    cpu = replay.replay(
        dump_dir, lapack=runtime.lapack_choice(lapack), out=io.StringIO(), fingerprint=True,
        device=None if reference == "cpu" else reference, **kw,
    )  # fmt: skip
    out = io.StringIO()
    dev = replay.replay(
        dump_dir, lapack=runtime.lapack_choice(lapack), out=out, device=device, fingerprint=True, **kw
    )
    return cpu, dev, out.getvalue()


def match(cpu, dev):
    """{(kind, array): [equal, differ]} over the fingerprinted reports, and the list of
    mismatched checkpoints (seq, kind, array)."""
    table, bad = defaultdict(lambda: [0, 0]), []
    for name, run in (("reference", cpu), ("device", dev)):
        if getattr(run, "failures", None):
            raise SystemExit(f"the {name} run raised on {len(run.failures)} events: {run.failures[:3]}")
    if [(e.seq, k) for e, k, _, _ in cpu] != [(e.seq, k) for e, k, _, _ in dev]:
        raise SystemExit("the two runs did not produce the same checkpoints")
    for (ev, kind, rc, _), (_, _, rd, _) in zip(cpu, dev, strict=True):
        for a, b in zip(rc, rd, strict=True):
            if a["name"] != b["name"]:
                raise SystemExit(f"e{ev.seq:05d} {kind}: report {a['name']} vs {b['name']}")
            if "sha1" not in a:
                continue
            same = a["sha1"] == b["sha1"]
            table[kind, a["name"]][0 if same else 1] += 1
            if not same:
                bad.append((ev.seq, kind, a["name"]))
    return dict(table), bad


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("dump")
    p.add_argument("--device", choices=("emu", "cuda", "metal"), default="emu")
    p.add_argument(
        "--reference",
        choices=("cpu", "emu32"),
        default=None,
        help="what the device is matched to (default: cpu; emu32 for metal)",
    )
    p.add_argument("--free-run", action="store_true")
    p.add_argument("--lapack", default="twin", choices=("twin", "crout"))
    p.add_argument("--tags")
    p.add_argument("--limit", type=int)
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--meteo-grid", choices=("model", "1x1"), default=None)
    runtime.add_input_args(p)
    a = p.parse_args(argv)
    kw = {
        "free_run": a.free_run,
        "meteo_root": a.meteo,
        "emissions": a.emissions,
        "chemistry": a.chemistry,
        "initial_file": a.initial,
        "tags": None if a.tags is None else set(a.tags.split(",")),
        "limit": a.limit,
        "rank": a.rank,
        "meteo_grid": a.meteo_grid,
    }
    reference = a.reference or ("emu32" if a.device == "metal" else "cpu")
    cpu, dev, log = run_both(a.dump, a.device, a.lapack, reference, **kw)
    table, bad = match(cpu, dev)
    print(f"{len(cpu)} checkpoints, device {a.device} against {reference}, lapack {a.lapack}, libm twin")
    fmt = "{:>9} {:<8} {:>8} {:>8}"
    print(fmt.format("step", "array", "equal", "differ"))
    for (kind, name), (eq, ne) in sorted(table.items()):
        print(fmt.format(kind, name, eq, ne))
    for seq, kind, name in bad[:20]:
        print(f"  differs: e{seq:05d} {kind} {name}")
    print(log[log.find("device transfers") :].rstrip() if "device transfers" in log else "")
    return 0 if table and not bad else 1


if __name__ == "__main__":
    sys.exit(main())
