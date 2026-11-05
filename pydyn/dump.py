"""Reader for the state dumps written by the Fortran module `pydyn_dump` (rc key `pydyn.dump`).

One event is a pair `<dir>/e<seq:05d>_<tag>_r<region>_p<rank:03d>.meta` (text, `key = value`
lines; arrays are declared as `array NAME = lb1 ub1 lb2 ub2 ...`, integer arrays as `iarray`)
plus one `<prefix>_<NAME>.bin` per array (raw stream, Fortran order, 8-byte reals or 4-byte
integers).  `Event[name]` returns an `FArray`: the array with its Fortran bounds, so that
`ev["m"][1:60, 1:45, 1:34]` is `m(1:60,1:45,1:34)` and `ev["am"][0, 3, 5]` is `am(0,3,5)`.
"""

import re
from pathlib import Path

import numpy as np

from .layout import interior_section

__all__ = [
    "Event", "EventRef", "FArray", "FView", "MergedEvent", "RankMismatch", "load_events", "merge_ranks",
]  # fmt: skip

_META = re.compile(r"^e(\d{5})_(.+)_r(\d+)_p(\d{3})\.meta$")

class FArray:
    """A numpy array addressed with the inclusive Fortran bounds it was declared with."""

    def __init__(self, data, lower):
        self.data = data
        self.lower = tuple(lower)

    @property
    def upper(self):
        return tuple(lo + n - 1 for lo, n in zip(self.lower, self.data.shape, strict=True))

    def _index(self, item):
        if not isinstance(item, tuple):
            item = (item,)
        out = []
        for lo, ix in zip(self.lower, item, strict=False):  # a partial index: the rest whole
            if isinstance(ix, slice):
                if ix.step not in (None, 1):
                    raise IndexError("FArray: strided Fortran sections are not supported")
                start = None if ix.start is None else ix.start - lo
                stop = None if ix.stop is None else ix.stop - lo + 1
                out.append(slice(start, stop))
            else:
                out.append(ix - lo)
        return (*out, Ellipsis)

    def __getitem__(self, item):
        """Inclusive Fortran section: `[lo:hi]` is `(lo:hi)`; trailing axes are taken whole."""
        return np.ascontiguousarray(self.data[self._index(item)])

    def window(self, other):
        """The numpy index into self.data of the cells `other` (an FArray) spans."""
        spans = zip(other.lower, other.upper, self.lower, strict=True)
        return tuple(slice(lo - slo, hi - slo + 1) for lo, hi, slo in spans)

    def __repr__(self):
        bounds = ", ".join(f"{lo}:{hi}" for lo, hi in zip(self.lower, self.upper, strict=True))
        return f"{type(self).__name__}({bounds})"


class FView(FArray):
    """An FArray whose sections are numpy *views* and can be assigned to:
    `f[0:im, 1:jm, 1:lm]` is `f(0:im,1:jm,1:lm)`, inclusive, and writes go through."""

    def __getitem__(self, item):
        return self.data[self._index(item)]

    def __setitem__(self, item, value):
        self.data[self._index(item)] = value

    @classmethod
    def zeros(cls, *bounds, dtype=float):
        """`FView.zeros((-1, im + 2), (-1, jm + 2), (1, lm))`: bounds as (lower, upper) pairs."""
        shape = tuple(hi - lo + 1 for lo, hi in bounds)
        return cls(np.zeros(shape, dtype=dtype), [lo for lo, _ in bounds])

    def copy(self):
        return FView(self.data.copy(), self.lower)

    def at(self, *index):
        """Fancy indexing with Fortran index arrays (broadcast as numpy does)."""
        return self.data[tuple(np.asarray(ix) - lo for ix, lo in zip(index, self.lower, strict=False))]


def _parse_value(text):
    words = text.split()
    if len(words) == 1:
        w = words[0]
        if w in ("T", "F"):
            return w == "T"
        try:
            return int(w)
        except ValueError:
            try:
                return float(w)
            except ValueError:
                return w
    try:
        return tuple(int(w) for w in words)
    except ValueError:
        return text.strip()


class DumpError(ValueError):
    """A dump file that is not what pydyn_dump writes: a truncated or foreign `.meta`."""


REQUIRED = ("tag", "seq", "region", "im", "jm", "lm")  # every event's header has these


class Event:
    """One dump event: `meta` (dict of scalars/tuples), `tag`, `seq`, `region`, and the
    arrays by name (loaded lazily from the .bin files)."""

    def __init__(self, meta_path):
        self.path = Path(meta_path)
        self.prefix = self.path.with_suffix("")
        self.meta = {}
        self._bounds = {}
        self._dtype = {}
        self._cache = {}
        try:
            text = self.path.read_text()
        except (OSError, UnicodeDecodeError) as e:
            raise DumpError(f"{self.path}: not readable as a dump header ({e})") from e
        for n, line in enumerate(text.splitlines(), 1):
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key.startswith(("array ", "iarray ")):
                try:
                    kind, name = key.split()
                    bounds = [int(w) for w in value.split()]
                    pairs = list(zip(bounds[::2], bounds[1::2], strict=True))
                except ValueError as e:
                    raise DumpError(f"{self.path}:{n}: not an array line: {line.strip()[:80]}") from e
                if not pairs or any(hi < lo - 1 for lo, hi in pairs) or not name.isidentifier():
                    raise DumpError(f"{self.path}:{n}: bad array bounds or name: {line.strip()[:80]}")
                self._bounds[name] = pairs
                self._dtype[name] = "<i4" if kind == "iarray" else "<f8"
            else:
                self.meta[key] = _parse_value(value)
        missing = [k for k in REQUIRED if k not in self.meta]
        if missing:
            raise DumpError(f"{self.path}: no {', '.join(missing)} in the header (a truncated dump?)")
        self.tag = self.meta["tag"]
        self.seq = self.meta["seq"]
        self.region = self.meta["region"]

    @property
    def arrays(self):
        return tuple(self._bounds)

    def __contains__(self, name):
        return name in self._bounds

    def __getitem__(self, name):
        if name not in self._cache:
            bounds = self._bounds[name]
            shape = tuple(hi - lo + 1 for lo, hi in bounds)
            raw = np.fromfile(f"{self.prefix}_{name}.bin", dtype=self._dtype[name])
            if raw.size != int(np.prod(shape)):
                raise ValueError(f"{self.prefix}_{name}.bin: {raw.size} values for bounds {bounds}")
            self._cache[name] = FArray(raw.reshape(shape, order="F"), [lo for lo, _ in bounds])
        return self._cache[name]

    def interior(self, name):
        """The pydyn-layout interior of a state or flux array (see layout.py): the Fortran
        section `INTERIOR[name]` of the dumped array."""
        im, jm, lm = self.meta["im"], self.meta["jm"], self.meta["lm"]
        return self[name][interior_section(name, im, jm, lm)]

    def release(self):
        """Drop the arrays read so far (a later access reads them again): a set holds tens of GB."""
        self._cache.clear()

    def cached(self):
        """The names of the arrays read so far."""
        return list(self._cache)

    def ref(self):
        """The event without its arrays, small enough to return from a worker process."""
        return EventRef(self.seq, self.tag, self.region, dict(self.meta), repr(self))

    def __repr__(self):
        return f"Event(seq={self.seq}, tag={self.tag!r}, region={self.region})"


class EventRef:
    """An event's identity and header, no arrays (Event.ref)."""

    def __init__(self, seq, tag, region, meta, text):
        self.seq, self.tag, self.region, self.meta, self._text = seq, tag, region, meta, text

    def __repr__(self):
        return self._text


def load_events(dump_dir, tags=None, region=None, rank=None, merge=False):
    """All events in `dump_dir`, in sequence order; optionally only some tags / one region /
    one rank.  merge=True: a multi-rank run's events put back together (`merge_ranks`); a
    single-rank directory is returned as it is."""
    events = []
    for p in sorted(Path(dump_dir).iterdir()):
        mo = _META.match(p.name)
        if mo is None:
            continue
        if tags is not None and mo.group(2) not in tags:
            continue
        if region is not None and int(mo.group(3)) != region:
            continue
        if rank is not None and int(mo.group(4)) != rank:
            continue
        events.append(Event(p))
    events.sort(key=lambda e: (e.seq, e.meta.get("rank", 0)))
    if merge and len({e.meta.get("rank", 0) for e in events}) > 1:
        return merge_ranks(events)
    return events


# ------------------------------------------------------------------ the ranks of a parallel run
# TM5's ct-unified MPI decomposition is by tracer only: rank r holds the contiguous block of
# tracers after those of ranks 0..r-1 (mpi_comm.F90: offsetn = sum(ntracet_ar(0:r-1)) + 1),
# and everything else is replicated.  What that means for the dump, established on the
# 2026-09-24 two-rank set (docs/state.md section 3e):
TRACER_ARRAYS = ("rm", "rxm", "rym", "rzm")  # the only arrays split by tracer; slabs in rank order
RANK_LOCAL_TRACER_KEYS = ("ch4_tracers",)  # header lists of the rank's LOCAL tracer indices
PER_TRACER_TAGS = ("emis",)  # one event per tracer the rank holds: `i_tracer` is global


class RankMismatch(ValueError):
    """The ranks of one event disagree where the run's decomposition says they must not."""


class MergedEvent:
    """One event of a multi-rank run seen as the serial run would have dumped it (`PYD-2`).

    The TRACER_ARRAYS are the ranks' slabs concatenated in rank order; every other array must
    be byte-identical on every rank (RankMismatch otherwise).  The header is the ranks' common
    one without `rank`; a RANK_LOCAL_TRACER_KEYS list is made global with each rank's tracer
    offset and concatenated; any other key that differs raises."""

    def __init__(self, parts, offsets):
        parts = sorted(parts, key=lambda e: e.meta["rank"])
        first = parts[0]
        for e in parts[1:]:
            if (e.tag, e.region) != (first.tag, first.region):
                raise RankMismatch(f"e{first.seq:05d}: ranks disagree on the event: {first!r} vs {e!r}")
            if e.arrays != first.arrays:
                raise RankMismatch(f"e{first.seq:05d} {first.tag}: ranks dump different arrays")
        self.parts = parts
        self.ranks = [e.meta["rank"] for e in parts]
        self.tag, self.seq, self.region = first.tag, first.seq, first.region
        self.meta = {}
        # `seq` is each rank's own event counter: it drifts once the ranks' emis blocks differ
        for key in set().union(*(e.meta for e in parts)) - {"rank", "seq"}:
            values = [e.meta.get(key) for e in parts]
            if key in RANK_LOCAL_TRACER_KEYS:
                glob = []
                for rank, v in zip(self.ranks, values, strict=True):
                    local = v if isinstance(v, tuple) else (v,)
                    glob += [offsets[rank] + int(n) for n in local]
                self.meta[key] = tuple(glob)
            elif all(v == values[0] for v in values):
                self.meta[key] = values[0]
            else:
                raise RankMismatch(f"e{first.seq:05d} {first.tag}: header `{key}` differs by rank: {values}")
        self.path = first.path
        self._cache = {}

    @property
    def arrays(self):
        return self.parts[0].arrays

    def __contains__(self, name):
        return name in self.parts[0]

    def decomposed(self, name):
        """True if `name` is split over the ranks by tracer."""
        return name in TRACER_ARRAYS

    def __getitem__(self, name):
        if name not in self._cache:
            arrays = [e[name] for e in self.parts]
            if self.decomposed(name):
                data = np.concatenate([a.data for a in arrays], axis=-1)
                if data.shape[-1] != self.meta["ntracet"]:
                    raise RankMismatch(
                        f"e{self.seq:05d} {self.tag}: `{name}` slabs hold {data.shape[-1]} tracers, "
                        f"the run has {self.meta['ntracet']}"
                    )
                self._cache[name] = FArray(data, arrays[0].lower)
            else:
                ref = arrays[0]
                for rank, a in zip(self.ranks[1:], arrays[1:], strict=True):
                    same = a.lower == ref.lower and a.data.shape == ref.data.shape
                    if not same or a.data.tobytes() != ref.data.tobytes():
                        raise RankMismatch(
                            f"e{self.seq:05d} {self.tag}: `{name}` is not replicated -- rank {rank} "
                            f"differs from rank {self.ranks[0]}"
                        )
                self._cache[name] = ref
        return self._cache[name]

    def interior(self, name):
        im, jm, lm = self.meta["im"], self.meta["jm"], self.meta["lm"]
        return self[name][interior_section(name, im, jm, lm)]

    def release(self):
        self._cache.clear()
        for e in self.parts:
            e.release()

    def cached(self):
        return list(self._cache)

    def ref(self):
        return EventRef(self.seq, self.tag, self.region, dict(self.meta), repr(self))

    def __repr__(self):
        return f"MergedEvent(seq={self.seq}, tag={self.tag!r}, region={self.region}, ranks={self.ranks})"


def merge_ranks(events):
    """A multi-rank run's events as the serial run's sequence: shared events merged
    (MergedEvent), and each block of PER_TRACER_TAGS events -- rank r dumps one per tracer it
    holds -- taken from every rank and ordered by the global `i_tracer`, as plain Events."""
    per_rank = {}
    for e in events:
        per_rank.setdefault(e.meta.get("rank", 0), []).append(e)
    ranks = sorted(per_rank)
    for r in ranks:
        per_rank[r].sort(key=lambda e: e.seq)
    offsets = _tracer_offsets(per_rank, ranks)
    at = dict.fromkeys(ranks, 0)
    merged = []
    while any(at[r] < len(per_rank[r]) for r in ranks):
        heads = {r: per_rank[r][at[r]] for r in ranks if at[r] < len(per_rank[r])}
        if any(e.tag in PER_TRACER_TAGS for e in heads.values()):
            block = []
            for r in ranks:
                while at[r] < len(per_rank[r]) and per_rank[r][at[r]].tag in PER_TRACER_TAGS:
                    block.append(per_rank[r][at[r]])
                    at[r] += 1
            merged += sorted(block, key=lambda e: e.meta["i_tracer"])
            continue
        if len(heads) != len(ranks):
            raise RankMismatch(f"ranks {sorted(set(ranks) - set(heads))} ran out of events before the others")
        merged.append(MergedEvent(list(heads.values()), offsets))
        for r in ranks:
            at[r] += 1
    return merged


def _tracer_offsets(per_rank, ranks):
    """Each rank's tracer offset: the slab sizes of the ranks before it (from `rm`)."""
    for r0_event in per_rank[ranks[0]]:
        if "rm" in r0_event:
            seq = r0_event.seq
            sizes = []
            for r in ranks:
                e = next(x for x in per_rank[r] if x.seq == seq)
                lo, hi = e._bounds["rm"][-1]
                sizes.append(hi - lo + 1)
            offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(int).tolist()
            return dict(zip(ranks, offsets, strict=True))
    return dict.fromkeys(ranks, 0)
