"""Which loss rates did a zoom child's chemistry apply?  (`PYD-3` item 4)

In an MPI build, ch4_chemistry_step (chemistry_ch4.F90, chem_unified) binds the shared-memory
loss array with `c_f_pointer(ch4_loss_baseptr, ch4_loss, (/n_lon, n_lat, n_levels, nreac/))`,
so its lower bounds are 1, and then reads `ch4_loss(i_lon, j_lat, k, r)` with i_lon = isr..ier,
j_lat = jsr..jer.  The serial build allocates it (lon_start:lon_end, lat_start:lat_end, ...).
For region 1 (isr = jsr = 1) the two agree; for a child they do not.

For every bef_c / chem / aft_c triple of a dump set this replays the decay two ways from the
dumped loss rates (bitwise the model's, the replay's `loss` report) and counts, over the
cells the region owns, where each reproduces aft_c bit for bit:

    intended   cell (i, j) decays with the loss of (i, j)                 -- the serial build
    as bound   cell (i, j) with the element at the flat column-major offset of (i, j, k, r)
               in a 1-based (n_lon, n_lat, n_levels, nreac) array          -- the MPI build;
               an offset past the array's end reads memory the dump does not have, so those
               cells are counted, not compared

Usage: python3 -m analysis.ch4_loss_bounds <dump_dir> [--rank 0] [--expect bug|fixed]
Exit 0 when region 1 is bitwise both ways and every child is bitwise only "as bound" (bug, the
default: tm5-zoom before the fix) or only "intended" (fixed).
"""

import argparse
import sys

import numpy as np

from pydyn import replay
from pydyn import zoom as Z
from pydyn.dump import load_events
from pydyn.ops import libm, seqsum

NAMES = ("rm", "rxm", "rym", "rzm")


def decay_intended(loss, dt):
    return libm.exp(-(seqsum(loss, axis=-1) * dt))


def decay_as_bound(loss, dt, isr, jsr):
    """The decay of every scope cell when loss (n_lon, n_lat, nk, nr) is indexed from 1 with
    the region's own i, j: (decay, in_array mask), both (n_lon, n_lat, nk)."""
    nl, nj, nk, nr = loss.shape
    flat = loss.reshape(-1, order="F")
    i = np.arange(isr, isr + nl)[:, None, None]
    j = np.arange(jsr, jsr + nj)[None, :, None]
    k = np.arange(1, nk + 1)[None, None, :]
    base = (i - 1) + nl * (j - 1) + nl * nj * (k - 1)
    offsets = [base + nl * nj * nk * r for r in range(nr)]
    inside = offsets[-1] < flat.size
    total = flat[np.where(inside, offsets[0], 0)]
    for off in offsets[1:]:
        total = total + flat[np.where(inside, off, 0)]
    return libm.exp(-(total * dt)), inside


def score(zm, bef, aft, decay, valid=None):
    """(bitwise, compared) over the owned cells of the scope, every tracer array."""
    r = bef.region
    sc = Z.scope(zm, r)
    own = Z.owned(zm, r)[sc][..., None] & (True if valid is None else valid)
    hits = total = 0
    for name in NAMES:
        b, a = bef.interior(name)[sc][..., 0], aft.interior(name)[sc][..., 0]
        mask = np.broadcast_to(own, b.shape)
        hits += int(((b * decay == a) & mask).sum())
        total += int(mask.sum())
    return hits, total


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("dump")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--expect", choices=("bug", "fixed"), default="bug")
    a = p.parse_args(argv)
    events = [e for e in load_events(a.dump) if e.meta.get("rank", 0) == a.rank]
    first = {}
    for e in events:
        first.setdefault(e.region, e)
    zm = replay.build_tree(first)
    rows = []
    for k, ev in enumerate(events):
        if ev.tag != "bef_c":
            continue
        try:
            aft, (chem_ev,) = replay.span(events, k)
        except ValueError:  # the dump ends inside the pair
            continue
        loss, dt, r = chem_ev["loss"].data, chem_ev.meta["dt_sec"], ev.region
        good = score(zm, ev, aft, decay_intended(loss, dt))
        dec, inside = decay_as_bound(loss, dt, zm.isr[r], zm.jsr[r])
        bound = score(zm, ev, aft, dec, inside)
        beyond = int((Z.owned(zm, r)[Z.scope(zm, r)][..., None] & ~inside).sum())
        rows.append((r, good, bound, beyond))
        print(f"e{ev.seq:05d} r{r}  intended {good[0]}/{good[1]}  as bound {bound[0]}/{bound[1]}"
              f"  past the array {beyond} cells")
    ok = all(g[0] == g[1] and b[0] == b[1] for r, g, b, _ in rows if r == 1)
    for r, g, b, _ in rows:
        if r > 1:
            hit, miss = (b, g) if a.expect == "bug" else (g, b)
            ok &= hit[0] == hit[1] and miss[0] < miss[1]
    ok &= any(r > 1 for r, *_ in rows)
    which = "as-bound" if a.expect == "bug" else "intended"
    print("verdict:", f"children applied the {which} losses, region 1 both" if ok else "NOT as stated")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
