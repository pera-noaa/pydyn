"""The driver's small statements as batches of box operations (csrc/boxes.cuh).

A statement `dst[...] = op(src...)` over device views -- a copy, a scaling, a division,
ops.seqsum along an axis, a constant, ops.advance, red2uni's (a / b) * c -- becomes one entry of
a Batch; a Program is a list of batches (stages) run in order, one dispatch each.  Within a
stage no thread may read an element another thread writes: Batch.add checks it exactly
(element addresses) and refuses, so the builder stages.  Programs are built once per state and
call site and kept (`program(cache, key, build)`): the views' addresses, shapes and strides
are fixed for a run, so the steady state is a few dispatches and no work on the host.
"""

from collections import namedtuple

import numpy as np

from .backend import backend

__all__ = [
    "ADV",
    "COPY",
    "DIV",
    "DIVMUL",
    "MUL",
    "SEQSUM",
    "SET",
    "ZSUM",
    "ZSUMMUL",
    "Batch",
    "Raw",
    "program",
]

COPY, MUL, DIV, SEQSUM, SET, ADV, DIVMUL, ZSUM, ZSUMMUL = range(9)
_NSRC = {COPY: 1, MUL: 1, DIV: 1, SEQSUM: 1, SET: 0, ADV: 3, DIVMUL: 3, ZSUM: 1, ZSUMMUL: 1}
_SUMS = (SEQSUM, ZSUM, ZSUMMUL)
_W = 44
_CHUNK = 256  # csrc/boxes.cuh PD_BOX_CHUNK: entries start on these boundaries

Raw = namedtuple("Raw", "address shape strides")
Raw.__doc__ = "A strided source no view expresses (a row's clusters): address, shape, element strides."


def _pad(shape, strides):
    pad = 4 - len(shape)
    if pad < 0:
        raise ValueError(f"boxes: {len(shape)} axes (at most 4)")
    return (1,) * pad + tuple(shape), (0,) * pad + tuple(strides)


def _source(be, src, dshape, rep, sum_axes):
    """(address, strides, divisors, [(n, stride)] of the summed axes) of one source for dst's
    shape; sum_axes the source's axes summed, outer first."""
    if isinstance(src, Raw):
        addr, shape, st = src.address, list(src.shape), list(src.strides)
    else:
        addr, shape, st = be.address(src), list(src.shape), list(be.strides(src))
    sums = [(shape[a], st[a]) for a in sum_axes]
    for a in sorted(sum_axes, reverse=True):
        shape.pop(a)
        st.pop(a)
    shape, st = _pad(shape, st)
    rep = (1,) * (4 - len(rep)) + tuple(rep) if rep is not None else (1, 1, 1, 1)
    strides, divs = [], []
    for d, s, stride, r in zip(dshape, shape, st, rep, strict=True):
        if s == d and r == 1:
            strides.append(stride)
            divs.append(1)
        elif s == 1:  # broadcast: stride 0
            strides.append(0)
            divs.append(1)
        elif s * r == d:
            strides.append(stride)
            divs.append(r)
        else:
            raise ValueError(f"boxes: source {tuple(shape)} for destination {tuple(dshape)}, rep {rep}")
    return addr, strides, divs, sums


class Batch:
    def __init__(self):
        self.rows, self.scalars, self.total = [], [], 0
        self._dst, self._src = [], []  # each entry's element addresses, for the hazard checks

    def add(self, op, dst, *srcs, scalar=0.0, rep=None, sum_axis=None, check=True):
        """dst = op(srcs): a source's shape is dst's, an axis of 1 broadcast, or dst's divided
        by rep (one rep for every source, or a list of them); SEQSUM's source has one more
        axis, sum_axis, summed in order; ZSUM / ZSUMMUL's one or two more, sum_axis an int or
        (outer, inner)."""
        be = backend()
        if dst.size == 0:
            return
        if len(srcs) != _NSRC[op]:
            raise ValueError(f"boxes: op {op} takes {_NSRC[op]} sources, not {len(srcs)}")
        dshape, dstr = _pad(dst.shape, be.strides(dst))
        reps = rep if isinstance(rep, list) else [rep] * len(srcs)
        axes = () if op not in _SUMS else (sum_axis,) if isinstance(sum_axis, int) else tuple(sum_axis)
        if op == SEQSUM and len(axes) != 1:
            raise ValueError("boxes: SEQSUM sums one axis")
        cols, sums = [], [(1, 0), (1, 0)]
        for src, r in zip(srcs, reps, strict=True):
            addr, strides, divs, sm = _source(be, src, dshape, r, axes)
            cols.append((addr, strides, divs))
            if op in _SUMS:
                sums = [*sm, (1, 0)][:2]
        size = dst.itemsize
        lo = be.address(dst)
        (n1, st1), (n2, st2) = sums
        if not check:  # the caller vouches the entry is hazard-free
            return self._row(op, lo, dshape, dstr, cols, n1, st1, n2, st2, scalar)
        elems = _elements(lo, dshape, dstr, size)
        idx = np.indices(dshape).reshape(4, -1)
        reads = [np.zeros((elems.size, 0), np.int64)]
        terms = ((np.arange(n1)[:, None] * st1 + np.arange(n2)[None, :] * st2).ravel() * size)[None, :]
        for addr, strides, divs in cols:
            first = addr + sum((idx[a] // divs[a]) * strides[a] for a in range(4)) * size
            reads.append(first[:, None] + terms.astype(np.int64))
        reads = np.concatenate(reads, axis=1)
        _check(elems, reads, self._dst, self._src)
        self._dst.append(elems)
        self._src.append(reads.ravel())
        self._row(op, lo, dshape, dstr, cols, n1, st1, n2, st2, scalar)

    def _row(self, op, lo, dshape, dstr, cols, n1, st1, n2, st2, scalar):
        while len(cols) < 3:
            cols.append((0, [0] * 4, [1] * 4))
        count = int(np.prod(dshape))
        row = [op, 0, self.total, lo, *dshape, *dstr]
        for addr, strides, divs in cols:
            row += [addr, *strides, *divs]
        row += [n1, st1, n2, st2, count]
        self.rows.append(row)
        self.scalars.append(scalar)
        self.total += -(-count // _CHUNK) * _CHUNK
        if self.total >= 2**31:
            raise ValueError("boxes: a stage of 2^31 elements (the kernel's indices are 32-bit)")


def _check(dst, reads, dsts, srcs):
    """No element written twice, none read by a thread other than the one writing it."""
    if np.unique(dst).size != dst.size:
        raise ValueError("boxes: a destination box writes an element twice")
    for other in dsts:
        if np.isin(dst, other).any():
            raise ValueError("boxes: two destinations of one stage overlap (stage them)")
        if np.isin(reads, other).any():
            raise ValueError("boxes: an entry reads what another entry of its stage writes (stage them)")
    for other in srcs:
        if np.isin(dst, other).any():
            raise ValueError("boxes: an entry writes what another entry of its stage reads (stage them)")
    if reads.size:
        order = np.argsort(dst)
        pos = np.minimum(np.searchsorted(dst[order], reads), dst.size - 1)
        hit = dst[order][pos] == reads
        if (hit & (order[pos] != np.arange(dst.size)[:, None])).any():
            raise ValueError("boxes: a thread reads an element another thread of the entry writes (stage it)")


def _elements(base, shape, strides, itemsize):
    grids = np.ix_(*(np.arange(n, dtype=np.int64) * s * itemsize
                     for n, s in zip(shape, strides, strict=True)))
    return (base + sum(grids)).ravel()


class Program:
    def __init__(self, stages, keep=()):
        be = backend()
        self.keep = keep  # scratch arrays the program's addresses point into
        self.stages = []
        for b in stages:
            if not b.rows:
                continue
            rows = np.asarray(b.rows, dtype=np.int64).reshape(-1, _W)
            blk = np.repeat(np.arange(len(rows), dtype=np.int32), -(-rows[:, 43] // _CHUNK))
            hdr = be.to_device(rows, "seed")
            sc = be.to_device(np.asarray(b.scalars, dtype=np.float64), "seed")
            self.stages.append((b.total, be.to_device(blk, "seed"), hdr, sc))

    def run(self):
        be = backend()
        for total, blk, hdr, sc in self.stages:
            be.launch("pd_boxes", total, total, blk, hdr, sc)


def program(cache, key, build):
    """cache[key], built by build() -> (stages, keep) on first use, then run."""
    prog = cache.get(key)
    if prog is None:
        stages, keep = build()
        prog = cache[key] = Program(stages, keep)
    prog.run()
