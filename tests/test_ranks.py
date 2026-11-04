"""PYD-2: the ranks together equal the serial answer.

Dump sets made by `.tm5-cloud-scratch/pydyn_build/setup_pyd2.sh` from one clean tm5-zoom
clone -- the same ch4_6x4_fwd.rc run with four distinct tracers, serial and on 2, 3 and 5
MPI ranks (slabs 2+2, 1+1+2, and 1+1+1+1+0: a rank with no tracer), runs/MANIFEST.md --
read through `dump.merge_ranks`:

  replicated   every array that is not split by tracer is byte-identical on every rank
  union        the merged multi-rank run is, event for event and array for array, the
               serial run to the bit
  replay       the merged set replays bitwise, as a serial set does (docs/state.md §3b)

Each multi-rank set present is tested; skipped when the serial one is absent
($PYDYN_SERIAL_DUMP_DIR overrides the default, $PYDYN_MPI_DUMP_DIRS takes a colon-separated
list of multi-rank sets).
"""

import os

import numpy as np
import pytest

from pydyn import replay
from pydyn.dump import MergedEvent, load_events

from common import MPI_DUMP_DIRS, SERIAL_DUMP_DIR

SERIAL, MPI_SETS = SERIAL_DUMP_DIR, MPI_DUMP_DIRS


def _has_events(d):
    return d.is_dir() and any(p.suffix == ".meta" for p in d.iterdir())


PRESENT = [d for d in MPI_SETS if _has_events(d)]

pytestmark = pytest.mark.skipif(
    not (_has_events(SERIAL) and PRESENT), reason=f"PYD-2 dump sets absent ({SERIAL}, {MPI_SETS})"
)


@pytest.fixture(scope="module")
def serial():
    return load_events(SERIAL)


@pytest.fixture(scope="module", params=PRESENT or [None], ids=lambda d: d.name if d else "none")
def mpi_set(request):
    return request.param


@pytest.fixture(scope="module")
def merged(mpi_set):
    events = load_events(mpi_set, merge=True)
    assert any(isinstance(e, MergedEvent) for e in events), "the MPI set has a single rank"
    return events


def _key(e):
    return (e.tag, e.region, e.meta.get("i_tracer"))


def test_same_events(serial, merged):
    """Both runs dump the same sequence of events -- the per-tracer `emis` ones interleaved
    from the ranks by global tracer -- though the multi-rank `seq` numbers are per rank."""
    assert [_key(e) for e in serial] == [_key(e) for e in merged]


def test_tracers_are_split(merged):
    """The set really is decomposed: rm is split over more than one rank, and the tracers are
    not all alike (a misplaced slab would show)."""
    ev = [e for e in merged if isinstance(e, MergedEvent) and "rm" in e][-1]  # t = 0: one IC for all
    assert len(ev.ranks) > 1 and ev.decomposed("rm")
    rm = ev["rm"].data
    assert rm.shape[-1] == ev.meta["ntracet"]
    assert not any(np.array_equal(rm[..., a], rm[..., b]) for a in range(rm.shape[-1]) for b in range(a))
    ev.release()


def test_replicated_and_union_equals_serial(serial, merged):
    """Every array: replicated ones identical on all ranks (MergedEvent raises otherwise), and
    the merged run bitwise the serial one; the headers agree except for rank-local keys."""
    diffs = []
    for s, m in zip(serial, merged):
        for key, value in m.meta.items():
            if key in ("seq", "rank", "itracer_loc"):  # per rank by definition; i_tracer is compared
                continue
            if key in s.meta and s.meta[key] != value:
                diffs.append(f"e{s.seq:05d} {s.tag}: meta {key} serial {s.meta[key]!r} mpi {value!r}")
        for name in s.arrays:
            a, b = s[name], m[name]
            if a.lower != b.lower or a.data.shape != b.data.shape:
                where = f"e{s.seq:05d} {s.tag}: {name}"
                diffs.append(f"{where} bounds {a.lower}+{a.data.shape} vs {b.lower}+{b.data.shape}")
            elif a.data.tobytes() != b.data.tobytes():
                n = int(np.sum(a.data != b.data))
                diffs.append(f"e{s.seq:05d} {s.tag}: {name} differs in {n} values")
        s.release(), m.release()  # one event's arrays at a time: a zoom set is tens of GB
    assert not diffs, "\n".join(diffs[:40])


def test_merged_set_replays_bitwise(mpi_set):
    workers = int(os.environ.get("PYDYN_WORKERS", "1"))
    results = replay.replay(mpi_set, out=open(os.devnull, "w"), workers=workers)
    assert results, "nothing replayed"
    worst = {kind: max(r["max_ulp"] for r in reports) for _, kind, reports, _ in results}
    assert all(v == 0 for v in worst.values()), worst
