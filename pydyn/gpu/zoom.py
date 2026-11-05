"""zoom.host's bodies for a device ZoomState (its `lib`): the same names and arguments, every
array of every region on the device (FViews whose data are device arrays), the arithmetic in
csrc/ and the glue -- halos, the reduced grid's copies, the exchange's copies and sums, the
compress / uncompress, the outer faces -- as box programs (exchange.py, boxes.py: built once a
state, a few dispatches a call).  pydyn.zoom's driver runs them: the host keeps the tree, the
split orders and the sub-cycle counts.

Scope: the reference's -- a global region (cyclic, reduced grid or not) and nested children
inside it in x, touching no pole (zoom/__init__.py).  Statement for statement, with one
kernel launch where the reference loops over rows, layers, levels or tracers to call a kernel.
"""

import sys
from collections import defaultdict, deque

import numpy as np

from .. import zoom as Z
from ..balance import GRAV
from ..dump import FView
from ..layout import bounds
from ..zoom.advect import _dynamu_child as _ref_dynamu_child
from ..zoom.advect import _dynamu_region1 as _ref_dynamu_region1
from ..zoom.advect import _dynamv_child as _ref_dynamv_child
from ..zoom.advect import _dynamw_child as _ref_dynamw_child
from ..zoom.state import FIELDS, ZoomState, _interior
from . import exchange
from . import kernels as G
from .backend import Uncapturable, backend
from .exchange import compress_xedges as _compress_xedges
from .exchange import compress_yedges as _compress_yedges
from .exchange import outer_x_faces_zeroed, outer_y_faces_zeroed, periodic_rows, put_xedges, put_yedges
from .exchange import uncompress_xedges as _uncompress_xedges
from .exchange import uncompress_yedges as _uncompress_yedges
from .kernels import CFLError, _default

__all__ = [  # zoom.host.BODIES, and the state's upload / download and the speculation
    "Pipeline",
    "_coarsen_block",
    "_dynamu_child",
    "_dynamu_region1",
    "_dynamum_child",
    "_dynamum_region1",
    "_dynamv_child",
    "_dynamv_region1",
    "_dynamvm_child",
    "_dynamvm_region1",
    "_dynamw_child",
    "_dynamw_region1",
    "_dynamwm",
    "_uncompress_xedges",
    "_uncompress_yedges",
    "assign",
    "chemistry_region",
    "coarsen_ambm",
    "convec_region",
    "download",
    "dynam0",
    "m2phlb",
    "m2phlb1",
    "mix_edges",
    "outer_x_faces_zeroed",
    "outer_y_faces_zeroed",
    "put_xedges",
    "put_yedges",
    "restore",
    "save",
    "scale",
    "sources_region",
    "update_parent",
    "upload",
    "zeros",
]


# ------------------------------------------------------------------ the state on the device
def zeros(zoom, nt):
    """ZoomState.zeros on the device."""
    xp, lm = backend().xp, zoom.lm
    fields = {}
    for r in zoom.regions():
        im, jm = zoom.g(r).im, zoom.g(r).jm
        fields[r] = {}
        for name in FIELDS:
            b = bounds(name, im, jm, lm, nt)
            fields[r][name] = FView(xp.zeros(tuple(hi - lo + 1 for lo, hi in b)), [lo for lo, _ in b])
    return ZoomState(fields, _LIB)


def upload(st, tag="seed"):
    be = backend()
    return ZoomState(
        {
            r: {k: FView(be.to_device(v.data, tag), v.lower) for k, v in f.items()}
            for r, f in st.fields.items()
        },
        _LIB,
    )


def download(st, tag="verify"):
    be = backend()
    return ZoomState(
        {r: {k: FView(be.to_host(v.data, tag), v.lower) for k, v in f.items()} for r, f in st.fields.items()}
    )


_LIB = sys.modules[__name__]


class _Consts:
    """What the device needs of a tree, uploaded once ("seed"): at, bt, and per region the
    reduced-grid row lengths, the owned-cell mask, the owned columns (flat i * jm + j), the
    `edge == -1` ring mix_edges averages and the ring m2phlb rescales."""

    def __init__(self, zoom):
        be = backend()
        self.at, self.bt = be.to_device(zoom.at, "seed"), be.to_device(zoom.bt, "seed")
        self.nrow, self.own, self.cols, self.ring, self.m2phlb_ring = {}, {}, {}, {}, {}
        for r in zoom.regions():
            g = zoom.g(r)
            imred = g.redgrid.imred if g.redgrid is not None else np.full(g.jm, g.im)
            self.nrow[r] = be.to_device(np.asarray(imred, dtype=np.int32), "seed")
            own = Z.owned(zoom, r)
            self.own[r] = be.to_device(own.astype(np.uint8), "seed")
            self.cols[r] = be.to_device(np.flatnonzero(own.ravel()).astype(np.int32), "seed")
            self.ring[r] = be.to_device((zoom.edge[r] == -1).astype(np.uint8), "seed")
            if r > 1:
                self.m2phlb_ring[r] = be.to_device(_m2phlb_ring(zoom, r).astype(np.uint8), "seed")


def _m2phlb_ring(zoom, region):
    """(im, jm) bool: the cells zoom.massflow.m2phlb rescales to the parent's cell area."""
    g = zoom.g(region)
    im, jm = g.im, g.jm
    xr, yr = zoom.ref(region, "x"), zoom.ref(region, "y")
    i = np.arange(1, im + 1)[:, None]
    j = np.arange(1, jm + 1)[None, :]
    inner_j = (yr < j) & (j < jm - yr + 1)
    skip = ((xr < i) & (i < im - xr + 1) & inner_j) | ((g.xcyc == 1) & inner_j)
    skip |= (g.touch_np == 1) & (j > jm - yr)
    skip |= (g.touch_sp == 1) & (j < yr + 1)
    return ~skip


def consts(zoom):
    c = getattr(zoom, "_pydyn_gpu", None)
    if c is None or c[0] is not backend():
        c = (backend(), _Consts(zoom))
        zoom._pydyn_gpu = c
    return c[1]


def _limit_box(s, rm):
    """s = kernels._limit(s, rm) in place on two equal-shape device views (3 or 4 axes)."""
    be = backend()
    v, r = (q if q.ndim == 4 else q[..., None] for q in (s, rm))
    be.launch("pd_limit_box", v.size, *v.shape, v, *be.strides(v), r, *be.strides(r))


def _repeat_rows(view, n, axis=0):
    """view repeated n times along `axis` -- the parent cells a child's index vector visits,
    each once per child cell (arange(m) // n)."""
    return backend().xp.repeat(view, n, axis=axis)


# ------------------------------------------------------------------ the reduced grid (redgrid.py)
# uni2red / red2uni around region 1's x advection: exchange.reduced_x_before / reduced_x_after


def _runs(rows, key):
    """Maximal runs of consecutive rows sharing key(row): [(first, last, key)]."""
    out = []
    for j in rows:
        k = key(j)
        if out and out[-1][1] == j - 1 and out[-1][2] == k:
            out[-1] = (out[-1][0], j, k)
        else:
            out.append((j, j, k))
    return out


# ------------------------------------------------------------------ the exchange (zoom.exchange)
# put_x/yedges, compress / uncompress and the outer faces: exchange.py's box programs


def _offset(fv, *index):
    """The element offset of Fortran index `index` from the start of fv.data."""
    s = backend().strides(fv.data)
    return sum((i - lo) * st for i, lo, st in zip(index, fv.lower, s, strict=False))  # a partial index


def mix_edges(zoom, st, region, tracers=True):
    g = zoom.g(region)
    xr, yr, zr = zoom.ref(region, "x"), zoom.ref(region, "y"), zoom.ref(region, "z")
    if zr != 1:
        raise ValueError("mix_edges: zooming in z direction not supported!")
    if (xr == 1 or g.im // g.xref == zoom.g(1).im) and region == 1:
        return
    cache = exchange._cache(st)
    key = ("mix_edges", region, tracers)
    if key not in cache:
        cache[key] = _mix_tables(zoom, st, region, tracers)
    t = cache[key]
    be = backend()
    be.launch(
        "pd_ring_means",
        t["nq"] * t["nb"],
        t["nq"],
        t["nb"],
        t["bp"],
        t["imp"],
        t["jmp"],
        xr,
        yr,
        consts(zoom).ring[region],
        g.jm,
        t["qa"],
        t["qs0"],
        t["qs1"],
        t["tp"],
    )
    be.launch(
        "pd_ring_spread",
        t["nq"] * t["nr"],
        t["nq"],
        t["nr"],
        t["rc"],
        g.im,
        g.jm,
        xr,
        yr,
        t["imp"],
        t["jmp"],
        *t["flags"],
        t["qa"],
        t["qs0"],
        t["qs1"],
        t["tp"],
    )


def _mix_tables(zoom, st, region, tracers):
    """mix_edges' level slices (every field's, by address), its ring cells and the parent
    blocks they read: the reference's statements are per slice and per cell, so these are
    all of them."""
    be = backend()
    g = zoom.g(region)
    xr, yr = zoom.ref(region, "x"), zoom.ref(region, "y")
    xedges = not (xr == 1 or g.im // g.xref == zoom.g(1).im)
    ysp, ynp = g.touch_sp != 1, g.touch_np != 1
    imr, jmr = g.im, g.jm
    imp, jmp = imr // xr, jmr // yr
    f, lm = st[region], zoom.lm
    qa, qs0, qs1 = [], [], []
    for name in ("m", "rm", "rxm", "rym", "rzm") if tracers else ("m",):
        q = f[name]
        s0, s1 = be.strides(q.data)[:2]
        index = (
            [(1, 1, lev) for lev in range(1, lm + 1)]
            if name == "m"
            else [(1, 1, lev, n) for n in range(1, st.nt + 1) for lev in range(1, lm + 1)]
        )
        for ix in index:
            qa.append(be.address(q.data) + _offset(q, *ix) * q.data.itemsize)
            qs0.append(s0)
            qs1.append(s1)
    i, j = np.meshgrid(np.arange(imr), np.arange(jmr), indexing="ij")
    wall, south, north = (i < xr) | (i >= imr - xr), j < yr, j >= jmr - yr
    cells = (xedges & wall) | (ysp & south) | (ynp & north)
    reads = []  # every block any strip of a ring cell reads (the last strip wins in the kernel)
    if xedges:
        reads += [(0 * jmp + j // yr)[i < xr], ((imp - 1) * jmp + j // yr)[i >= imr - xr]]
    if ysp:
        reads.append(((i // xr) * jmp)[south])
    if ynp:
        reads.append(((i // xr) * jmp + jmp - 1)[north])
    blocks = np.unique(np.concatenate(reads)) if reads else np.zeros(0, np.int64)
    rc = np.flatnonzero(cells.ravel())
    nq = len(qa)
    return {
        "nq": nq,
        "nr": rc.size,
        "nb": blocks.size,
        "imp": imp,
        "jmp": jmp,
        "flags": (int(xedges), int(ysp), int(ynp)),
        "rc": be.to_device(rc.astype(np.int32), "seed"),
        "bp": be.to_device(blocks.astype(np.int32), "seed"),
        "qa": be.to_device(np.asarray(qa, dtype=np.int64), "seed"),
        "qs0": be.to_device(np.asarray(qs0, dtype=np.int64), "seed"),
        "qs1": be.to_device(np.asarray(qs1, dtype=np.int64), "seed"),
        "tp": be.xp.zeros((nq, imp, jmp)),
    }


def update_parent(zoom, st, region, tracers=True):
    if region == 1:
        return
    be = backend()
    g, parent = zoom.g(region), zoom.g(region).parent
    xr, yr, zr = zoom.ref(region, "x"), zoom.ref(region, "y"), zoom.ref(region, "z")
    imp, jmp = g.im // xr, g.jm // yr
    ib, ie, jb, je = zoom.ibeg[region], zoom.iend[region], zoom.jbeg[region], zoom.jend[region]
    f, p, lm = st[region], st[parent], zoom.lm

    def sums(dst, src):
        be.launch(
            "pd_block_sums",
            imp * jmp * lm,
            imp,
            jmp,
            lm,
            xr,
            yr,
            src,
            *be.strides(src),
            dst,
            *be.strides(dst),
        )

    sums(p["m"][ib:ie, jb:je, 1:lm], f["m"][1 : g.im, 1 : g.jm, 1:lm])
    if zr != 1:
        raise ValueError("update_parent: zooming in z direction not supported")
    for n in range(1, st.nt + 1 if tracers else 1):
        sums(p["rm"][ib:ie, jb:je, 1:lm, n], f["rm"][1 : g.im, 1 : g.jm, 1:lm, n])


# ------------------------------------------------------------------ advection
def _stats(stats, nloop, xi):
    if stats is not None:
        be = backend()
        stats["nloop_max"] = int(be.to_host(nloop, "control").max())
        stats["xi"] = float(be.to_host(xi, "control").max())


def _dynamu_region1(zoom, st, js, je, limits, stats, max_nloop=None):
    max_nloop = _default(_ref_dynamu_region1, "max_nloop") if max_nloop is None else max_nloop
    g, f, lm = zoom.g(1), st[1], zoom.lm
    im, jm = g.im, g.jm
    fields = _interior(f, im, jm, lm)
    am = f["am"][0:im, 1:jm, 1:lm]
    grid = g.redgrid
    if grid is not None:
        exchange.reduced_x_before(zoom, st, fields, am)
    nloop, amax = G.advect_x_rows(*fields, am, consts(zoom).nrow[1], limits, max_nloop, deferred=True)
    imred = grid.imred if grid is not None else np.full(jm, im)
    periodic_rows(zoom, st, _runs(range(js, je + 1), lambda j: int(imred[j - 1])))
    if grid is not None:
        exchange.reduced_x_after(zoom, st, fields, am)
    _stats(stats, nloop, amax)


def _child_x_rows(zoom, st, region, js, je, ls, le, tracers, limits, max_nloop):
    """The kernel call of _dynamu_child / _dynamum_child: (nloop, amax, the device fail flag)."""
    be = backend()
    g, f = zoom.g(region), st[region]
    xr, imr = zoom.ref(region, "x"), g.im
    is_, ie = xr, imr - xr + 1
    m, am = f["m"][is_ - 1 : ie + 1, js:je, ls:le], f["am"][is_ - 1 : ie, js:je, ls:le]
    n, J, L = m.shape
    if tracers:
        q4 = [f[name][is_ - 1 : ie + 1, js:je, ls:le] for name in ("rm", "rxm", "rym", "rzm")]
        if limits:
            _limit_box(q4[1], q4[0])
        nt, tr = st.nt, (*q4, *G._tracer_strides(*q4))
    else:
        nt, tr = 0, G._no_tracers(m)
    nloop, amax, fail = G._scratch(J * L, np.int32), G._scratch(J * L), G._scratch(1, np.int32)
    be.launch(
        "pd_x_child_rows",
        J * L,
        n,
        J,
        L,
        nt,
        m,
        *be.strides(m),
        *tr,
        am,
        *be.strides(am),
        int(limits),
        max_nloop - 1,
        nloop,
        amax,
        fail,
    )
    return nloop, amax, fail


def _dynamu_child(zoom, st, region, js, je, ls, le, limits, stats, max_nloop=None):
    max_nloop = _default(_ref_dynamu_child, "max_nloop") if max_nloop is None else max_nloop
    _compress_xedges(zoom, st, region, js, je, ls, le)
    nloop, amax, fail = _child_x_rows(zoom, st, region, js, je, ls, le, True, limits, max_nloop)
    backend().check(fail, CFLError("dynuamu: nloop too high"), deferred=True)
    _stats(stats, nloop, amax)


def _dynamv_region1(zoom, st, limits, stats):
    g, f, lm = zoom.g(1), st[1], zoom.lm
    im, jm = g.im, g.jm
    bm_int = f["bm"][1:im, 1 : jm + 1, 1:lm]
    # dynamv scales and restores whole layers: the halo'd bm's levels 1..lm (index = level)
    G.advect_y(*_interior(f, im, jm, lm), bm_int, limits, stats=stats, full_bm=f["bm"].data, level_offset=1)


def _held_search(cells, b, nmax, message):
    """The per-layer search of a child's y columns to its fixpoint: (device, host) nloop."""
    be = backend()
    ni, n, L = cells.shape
    return G._search(
        lambda cand, nxt, bad: be.launch(
            "pd_held_search", ni * L, ni, n, L, cells, *be.strides(cells), b, *be.strides(b), cand, nmax, nxt
        ),
        L,
        nmax,
        message,
    )


def _scale_layers(arr, ls, nloop, restore):
    """pd_layer_scale over every face of the levels ls.. of a (n0, n1, levels) device array."""
    be = backend()
    L = nloop.size
    n0, n1 = arr.shape[0], arr.shape[1]
    be.launch("pd_layer_scale", n0 * n1 * L, n0, n1, ls, L, arr, *be.strides(arr), nloop, int(restore))


def _dynamv_child(zoom, st, region, is_, ie, ls, le, limits, stats, max_nloop=None):
    max_nloop = _default(_ref_dynamv_child, "max_nloop") if max_nloop is None else max_nloop
    be = backend()
    _compress_yedges(zoom, st, region, is_, ie, ls, le)
    g, f = zoom.g(region), st[region]
    yr, jmr = zoom.ref(region, "y"), g.jm
    js, je = yr, jmr - yr + 1
    q4 = [f[name][is_:ie, js - 1 : je + 1, ls:le] for name in ("rm", "rxm", "rym", "rzm")]
    if limits:
        _limit_box(q4[2], q4[0])
    cells, b = f["m"][is_:ie, js - 1 : je + 1, ls:le], f["bm"][is_:ie, js : je + 1, ls:le]
    nloop, _ = _held_search(cells, b, max_nloop - 1, "nloop too high in dynamv")
    full = f["bm"]
    lv = ls - full.lower[2]
    _scale_layers(full.data, lv, nloop, restore=False)  # the whole layer, as dynamv scales it
    ni, n, L = cells.shape
    bmax = G._scratch(L)
    be.launch(
        "pd_y_child_steps",
        ni * L,
        ni,
        n,
        L,
        st.nt,
        cells,
        *be.strides(cells),
        *q4,
        *G._tracer_strides(*q4),
        b,
        *be.strides(b),
        nloop,
        int(limits),
        bmax,
    )
    _scale_layers(full.data, lv, nloop, restore=True)
    _stats(stats, nloop, bmax)


def _dynamw_region1(zoom, st, limits, stats):
    g, f, lm = zoom.g(1), st[1], zoom.lm
    im, jm = g.im, g.jm
    G.advect_z(*_interior(f, im, jm, lm), f["cm"][1:im, 1:jm, 0:lm], limits, stats=stats)


def _z_scope_search(zoom, st, region, is_, ie, js, je, iter_limit):
    """column_trial's n_advz on the region with the mass advanced on the scope, as a device
    array: None past iter_limit (speculating: the device array, the limit checked later)."""
    be = backend()
    g, f, lm = zoom.g(region), st[region], zoom.lm
    m, cm = f["m"][1 : g.im, 1 : g.jm, 1:lm], f["cm"][1 : g.im, 1 : g.jm, 0:lm]
    im, jm = g.im, g.jm
    try:
        n, _ = G._search(
            lambda cand, nxt, bad: be.launch(
                "pd_z_scope_search",
                im * jm,
                im,
                jm,
                lm,
                m,
                *be.strides(m),
                cm,
                *be.strides(cm),
                is_ - 1,
                ie,
                js - 1,
                je,
                cand,
                iter_limit,
                nxt,
                bad,
            ),
            1,
            iter_limit,
            "dynamw",
        )
    except CFLError:
        return None
    return n


def _dynamw_child(zoom, st, region, is_, ie, js, je, limits, stats, iter_limit=None):
    iter_limit = _default(_ref_dynamw_child, "iter_limit") if iter_limit is None else iter_limit
    f, lm = st[region], zoom.lm
    cm = f["cm"]
    be = backend()
    flag = G._scratch(1, np.int32)
    for face in (cm[:, :, 0], cm[:, :, lm]):
        be.launch("pd_any_nonzero", face.size, *face.shape, face, *be.strides(face), flag)
    be.check(flag, ValueError("dynamw: found non-zero halo interfaces in cm"), deferred=True)
    rm, rxm, rym, rzm = (f[name] for name in ("rm", "rxm", "rym", "rzm"))
    if limits:
        _limit_box(rzm[is_:ie, js:je, 1:lm], rm[is_:ie, js:je, 1:lm])
    n_advz = _z_scope_search(zoom, st, region, is_, ie, js, je, iter_limit)
    if n_advz is None:
        raise CFLError("dynamw: too many CFL violations!")
    cells = (slice(is_, ie), slice(js, je), slice(1, lm))
    faces = (slice(is_, ie), slice(js, je), slice(0, lm))
    G.advect_z_substeps(f["m"][cells], *(q[cells] for q in (rm, rxm, rym, rzm)), cm[faces], n_advz, limits)
    if stats is not None:
        stats["nloop_max"], stats["xi"] = int(be.to_host(n_advz, "control")[0]), None


# ------------------------------------------------------------------ physics (zoom.physics)
def _inverse_kind(inverse):
    from ..lapack import LapackInverse

    if isinstance(inverse, LapackInverse):
        return "lapack"  # bitwise to the reference only when its library is the host twin
    if inverse in ("crout", "lapack"):
        return inverse
    raise ValueError(f"pydyn.gpu: inverse {inverse!r} has no device twin ('crout', or a LapackInverse)")


def convec_region(zoom, st, region, conv, ndyn, inverse="crout"):
    """zoom.convec_region: the owned columns of the scope; `conv`'s arrays host (uploaded as
    "input") or device."""
    g, f, be = zoom.g(region), st[region], backend()
    im, jm, lm = g.im, g.jm, zoom.lm
    dconv = {}
    for k in ("entu", "detu", "entd", "detd", "dkg", "cloud_top", "cloud_lfs"):
        v = conv[k]
        dtype = np.int32 if k.startswith("cloud") else be.real
        dconv[k] = v if be.is_device(v) and v.dtype == dtype else be.to_device(np.asarray(v), "input", dtype)
    fields = [f[name][1:im, 1:jm, 1:lm] for name in ("m", "rm", "rxm", "rym", "rzm")]
    dt = ndyn / (2.0 * g.tref)
    G.convec(*fields, dconv, dt, conv["lmax"], consts(zoom).cols[region], _inverse_kind(inverse))


def sources_region(zoom, st, region, injections):
    """zoom.sources_region: each (x (im, jm) kg, level, tracer) added on the owned cells."""
    g, f, be = zoom.g(region), st[region], backend()
    for x, level, tracer in injections:
        rm = f["rm"][1 : g.im, 1 : g.jm, level, tracer]
        rzm = f["rzm"][1 : g.im, 1 : g.jm, level, tracer]
        if be.strides(rm) != be.strides(rzm):
            raise ValueError("pydyn.gpu.sources_region: rm and rzm views differ in strides")
        dx = x if be.is_device(x) else be.to_device(np.asarray(x, dtype=float), "input")
        be.launch(
            "pd_add_2d", g.im * g.jm, g.im, g.jm, consts(zoom).own[region], dx, rm, rzm, *be.strides(rm)
        )


def chemistry_region(zoom, st, region, decay, tracers):
    """zoom.chemistry_region: rm and slopes of the (1-based) tracers times decay (the scope's,
    a device array) on the owned cells -- one launch for every tracer and field."""
    g, f, be, lm = zoom.g(region), st[region], backend(), zoom.lm
    si, sj = Z.scope(zoom, region)
    sni, snj = si.stop - si.start, sj.stop - sj.start
    tracers = tuple(tracers)
    if not tracers:
        return
    q4 = [f[name][1 : g.im, 1 : g.jm, 1:lm, :] for name in ("rm", "rxm", "rym", "rzm")]
    cache = exchange._cache(st)
    key = ("chem_tracers", tracers)
    if key not in cache:  # the tracer list on the device, once
        cache[key] = be.to_device(np.asarray([t - 1 for t in tracers], dtype=np.int32), "seed")
    be.launch(
        "pd_scale_owned_all", g.im * g.jm * lm, g.im, g.jm, lm, consts(zoom).own[region], si.start, sj.start,
        sni, snj, decay, *be.strides(decay), *q4, *G._tracer_strides(*q4), cache[key], len(tracers),
    )  # fmt: skip


# ------------------------------------------------------------------ mass flow (zoom.massflow)
def _column_pressure(zoom, m, dxyp):
    """m2phlb1's p (im, jm) of an (im, jm, lm) device view: a new device array."""
    be, c = backend(), consts(zoom)
    im, jm, lm = m.shape
    sp, phlb = be.xp.zeros((im, jm)), be.xp.zeros((im, jm, lm + 1))
    be.launch(
        "pd_m2phlb1",
        im * jm,
        im,
        jm,
        lm,
        GRAV,
        m,
        *be.strides(m),
        dxyp,
        c.at,
        c.bt,
        sp,
        *be.strides(sp),
        phlb,
        *be.strides(phlb),
    )
    return sp


def m2phlb1(zoom, st, region=1):
    g, f, be, lm = zoom.g(region), st[region], backend(), zoom.lm
    im, jm = g.im, g.jm
    c = consts(zoom)
    m, sp, phlb = f["m"][1:im, 1:jm, 1:lm], f["sp"], f["phlb"][1:im, 1:jm, 1 : lm + 1]
    exchange.zero(st, ("sp", region), sp.data)
    spv = sp[1:im, 1:jm, 1]
    be.launch(
        "pd_m2phlb1",
        im * jm,
        im,
        jm,
        lm,
        GRAV,
        m,
        *be.strides(m),
        f["dxyp"][1:jm],
        c.at,
        c.bt,
        spv,
        *be.strides(spv),
        phlb,
        *be.strides(phlb),
    )


def m2phlb(zoom, st, region):
    """zoom.massflow.m2phlb: m2phlb1, the ring's pressure rescaled to the parent cell's area."""
    g, f, lm, be = zoom.g(region), st[region], zoom.lm, backend()
    im, jm = g.im, g.jm
    xr, yr = zoom.ref(region, "x"), zoom.ref(region, "y")
    c = consts(zoom)
    jb = zoom.jbeg[region]
    m, spv, phlb = f["m"][1:im, 1:jm, 1:lm], f["sp"][1:im, 1:jm, 1], f["phlb"][1:im, 1:jm, 1 : lm + 1]
    exchange.zero(st, ("sp", region), f["sp"].data)
    be.launch(
        "pd_m2phlb_child", im * jm, im, jm, lm, GRAV, m, *be.strides(m), f["dxyp"][1:jm],
        st[g.parent]["dxyp"][jb : jb + (jm - 1) // yr], xr, yr, c.m2phlb_ring[region], c.at, c.bt,
        spv, *be.strides(spv), phlb, *be.strides(phlb),
    )  # fmt: skip


def _coarsen_block(zoom, st, region):
    """zoom.massflow._coarsen_block: the parent's am, bm, cm and m over the child footprint
    from the child's (contiguous footprints: a child inside its parent in x)."""
    parent = zoom.g(region).parent
    g = zoom.g(region)
    xr, zr = zoom.ref(region, "x"), zoom.ref(region, "z")
    imp, ib = g.im // xr, zoom.ibeg[region]
    if zr != 1:
        raise ValueError("coarsen_region: zooming in z direction not supported")
    if ib - 1 + imp >= zoom.g(parent).im + 1:
        raise NotImplementedError("pydyn.gpu.coarsen_region: a child across the parent's x seam")

    exchange.coarsen(st, region, lambda: _coarsen_batch(zoom, st, region))


def _coarsen_batch(zoom, st, region):
    """coarsen_region's four block sums (0 + the terms in the reference's loop order), each
    straight into the parent's footprint: one stage."""
    from .boxes import ZSUM, ZSUMMUL, Batch

    parent = zoom.g(region).parent
    g, f, p = zoom.g(region), st[region], st[parent]
    xr, yr, tr = (zoom.ref(region, a) for a in "xyt")
    imp, jmp, lmp = g.im // xr, g.jm // yr, zoom.lm
    ib, jb, lb = zoom.ibeg[region], zoom.jbeg[region], zoom.lbeg[region]
    lm = zoom.lm
    b = Batch()
    faces = f["am"][0 : g.im, 1 : g.jm, 1:lm][0 : imp * xr + 1 : xr]  # faces ip * xref_
    b.add(
        ZSUMMUL,
        p["am"].data[ib - 1 : ib + imp, jb : jb + jmp, lb : lb + lmp],
        faces.reshape(imp + 1, jmp, yr, lm),
        scalar=float(tr),
        sum_axis=2,
    )
    faces = f["bm"][1 : g.im, 1 : g.jm + 1, 1:lm][:, 0 : jmp * yr + 1 : yr]
    b.add(
        ZSUMMUL,
        p["bm"][ib : ib + imp - 1, jb : jb + jmp, lb : lb + lmp - 1],
        faces.reshape(imp, xr, jmp + 1, lm),
        scalar=float(tr),
        sum_axis=1,
    )
    b.add(
        ZSUMMUL,
        p["cm"][ib : ib + imp - 1, jb : jb + jmp - 1, lb : lb + lmp - 1],
        f["cm"][1 : g.im, 1 : g.jm, 1:lm].reshape(imp, xr, jmp, yr, lm),
        scalar=float(tr),
        sum_axis=(1, 3),
    )
    b.add(
        ZSUM,
        p["m"][ib : ib + imp - 1, jb : jb + jmp - 1, lb : lb + lmp - 1],
        f["m"][1 : g.im, 1 : g.jm, 1:lm].reshape(imp, xr, jmp, yr, lm),
        sum_axis=(3, 1),
    )  # b outer, a inner
    return [b], ()


def coarsen_ambm(zoom, st, region):
    """zoom.massflow.coarsen_ambm, its (j, l) and (i, l) loops as whole sections."""
    exchange.run(st, ("coarsen_ambm", region), lambda: ([_coarsen_ambm_batch(zoom, st, region)], ()))


def _coarsen_ambm_batch(zoom, st, region):
    from .boxes import DIV, Batch

    f, b = st[region], Batch()
    am, bm = f["am"], f["bm"]
    for child in zoom.children[region]:
        gc, c = zoom.g(child), st[child]
        xr, yr, zr = zoom.ref(child, "x"), zoom.ref(child, "y"), zoom.ref(child, "z")
        if zr != 1:
            raise NotImplementedError("pydyn.gpu.coarsen_ambm: zooming in z")
        ib, ie, jb, je = zoom.ibeg[child], zoom.iend[child], zoom.jbeg[child], zoom.jend[child]
        lb, le = zoom.lbeg[child], zoom.lend[child]
        amc, bmc = c["am"], c["bm"]
        lc = slice(1, (le - lb) * zr + 1)
        if gc.im // gc.xref != zoom.g(1).im:
            jc = slice(1, (je - jb) * yr + yr)
            b.add(DIV, amc[xr - 1, jc, lc], am[ib - 1, jb:je, lb:le], scalar=float(yr * zr), rep=(yr, 1))
            b.add(DIV, amc[gc.im - xr + 1, jc, lc], am[ie, jb:je, lb:le], scalar=float(yr * zr), rep=(yr, 1))
        ic = slice(1, (ie - ib) * xr + xr)
        if gc.touch_sp != 1:
            b.add(DIV, bmc[ic, yr, lc], bm[ib:ie, jb, lb:le], scalar=float(xr * zr), rep=(xr, 1))
        if gc.touch_np != 1:
            b.add(
                DIV, bmc[ic, gc.jm - yr + 2, lc], bm[ib:ie, je + 1, lb:le], scalar=float(xr * zr), rep=(xr, 1)
            )
    return b


def dynam0(zoom, st, region, ndyn):
    g, f, lm = zoom.g(region), st[region], zoom.lm
    im, jm = g.im, g.jm
    dt = ndyn / (2.0 * g.tref)
    exchange.run(
        st, ("dynam0_zero", region), lambda: ([_set_batch([f[k].data for k in ("am", "bm", "cm")])], ())
    )
    G.fluxes_from_meteo(
        f["pu"][0:im, 1:jm, 1:lm],
        f["pv"][1:im, 1 : jm + 1, 1:lm],
        consts(zoom).bt,
        dt,
        f["am"][0:im, 1:jm, 1:lm],
        f["bm"][1:im, 1 : jm + 1, 1:lm],
        f["cm"][1:im, 1:jm, 0:lm],
    )
    if g.xcyc == 1:
        exchange.run(st, ("dynam0_cyclic", region), lambda: ([_cyclic_am_batch(f["am"], im)], ()))


def _set_batch(views, value=0.0):
    from .boxes import SET, Batch

    b = Batch()
    for v in views:
        b.add(SET, v, scalar=value)
    return b


def _cyclic_am_batch(a, im):
    from .boxes import COPY, Batch

    b = Batch()
    b.add(COPY, a[0, :, :], a[im, :, :])
    b.add(COPY, a[im + 1, :, :], a[1, :, :])
    return b


# ------------------------------------------------------------------ Check_CFL (zoom.check_cfl)
def _dynamum_region1(zoom, st, js, je, max_nloop):
    """check_cfl._dynamum_region1: cfl.dynamum and its halos."""
    g, f, lm = zoom.g(1), st[1], zoom.lm
    im, jm = g.im, g.jm
    m_int, am = f["m"][1:im, 1:jm, 1:lm], f["am"][0:im, 1:jm, 1:lm]
    grid = g.redgrid
    if grid is not None:
        exchange.reduced_x_before(zoom, st, (m_int,), am, tracers=False)
    if not G.mass_x_rows(m_int, am, consts(zoom).nrow[1], max_nloop):
        return False
    if grid is not None:
        exchange.reduced_xm_after(zoom, st, m_int, am)
    exchange.mass_rows(zoom, st, js, je)
    return True


def _dynamum_child(zoom, st, region, js, je, ls, le, max_nloop):
    """check_cfl._dynamum_child: the rows mass only; uncompress and mix only when they pass."""
    _compress_xedges(zoom, st, region, js, je, ls, le, tracers=False)
    _, _, fail = _child_x_rows(zoom, st, region, js, je, ls, le, False, False, max_nloop)
    if not G.ok_flag(fail):  # Check_CFL decides on it (or assumes, speculating)
        return False
    _uncompress_xedges(zoom, st, region, js, je, ls, le, tracers=False)
    mix_edges(zoom, st, region, tracers=False)
    return True


def _dynamvm_region1(zoom, st, max_nloop):
    g, f, lm = zoom.g(1), st[1], zoom.lm
    im, jm = g.im, g.jm
    return G.mass_y(f["m"][1:im, 1:jm, 1:lm], f["bm"][1:im, 1 : jm + 1, 1:lm], max_nloop)


def _dynamvm_child(zoom, st, region, is_, ie, ls, le, max_nloop):
    """check_cfl._dynamvm_child: the scope's faces scaled by the search, the whole layer
    multiplied back."""
    be = backend()
    _compress_yedges(zoom, st, region, is_, ie, ls, le, tracers=False)
    g, f = zoom.g(region), st[region]
    yr = zoom.ref(region, "y")
    js, je = yr, g.jm - yr + 1
    cells, b = f["m"][is_:ie, js - 1 : je + 1, ls:le], f["bm"][is_:ie, js : je + 1, ls:le]
    try:
        nloop, _ = _held_search(cells, b, max_nloop, "dynamvm")
    except CFLError:
        return False
    _scale_layers(b, 0, nloop, restore=False)
    ni, n, L = cells.shape
    m = cells
    be.launch(
        "pd_y_child_steps",
        ni * L,
        ni,
        n,
        L,
        0,
        cells,
        *be.strides(cells),
        *G._no_tracers(m),
        b,
        *be.strides(b),
        nloop,
        0,
        G._scratch(L),
    )
    full = f["bm"]
    _scale_layers(full.data, ls - full.lower[2], nloop, restore=True)
    _uncompress_yedges(zoom, st, region, is_, ie, ls, le, tracers=False)
    mix_edges(zoom, st, region, tracers=False)
    return True


def _dynamwm(zoom, st, region, is_, ie, js, je, iter_limit):
    f, lm = st[region], zoom.lm
    n = _z_scope_search(zoom, st, region, is_, ie, js, je, iter_limit)
    if n is None:
        return False
    be = backend()
    m, cm = f["m"][is_:ie, js:je, 1:lm], f["cm"][is_:ie, js:je, 0:lm]
    ni, nj, _ = m.shape
    be.launch(
        "pd_z_steps",
        ni * nj,
        ni,
        nj,
        lm,
        0,
        m,
        *be.strides(m),
        *G._no_tracers(m),
        cm,
        *be.strides(cm),
        n,
        0,
        0,
    )
    return True


def assign(dst, src, region, names):
    pairs = [(dst[region][k].data, src[region][k].data) for k in names]
    exchange.run(dst, ("assign", id(src), region, names), lambda: ([_copy_batch(pairs)], ()))


def _copy_batch(pairs):
    from .boxes import COPY, Batch

    b = Batch()
    for d, s in pairs:
        b.add(COPY, d, s)
    return b


def save(st, key, names):
    exchange.save(st, key, [f[k].data for f in st.fields.values() for k in names])


def restore(st, key):
    exchange.restore(st, key)


def scale(st, names, factor):
    exchange.scale(st, (names, factor), [f[k].data for f in st.fields.values() for k in names], factor)


# ------------------------------------------------------------------ speculation
class Pipeline:
    """Calls fn() -- any sequence of the driver's on st -- speculated, up to `depth` in flight.
    Speculating, the host decides nothing on the way: every search takes its last queued round
    (kernels.SEARCH_ROUNDS), Check_CFL's operators pass, the fatal flags wait.  Each call is
    queued on the state its predecessor is assumed to leave (saved first) and checked, oldest
    first, once the device has run it (a mark waited for, not a sync) -- so the host queues the
    next while the device runs this one.  A call whose check fails, or that set a deferred
    error, is redone deciding as it goes on the state put back, and every later one is queued
    again on its result: the results are always that careful run's bits.  fn must be
    repeatable (its host-side state its own).  On a synchronous backend each call just runs.
    `tally`: the calls kept and redone, and of those queued the replayed graphs.

    submit(fn, repeat=key) declares more: that every call with that key queues the same work
    -- the same kernels with the same arguments, its inputs already in place, the same result
    and checks.  On a backend that captures (CUDA), a slot's second such call is captured as
    a graph and the later ones replay it, fn not called; a capture refused (an upload, read
    back, sync, mark or new scratch inside it: backend.Uncapturable) runs the call as usual
    and that key is not captured again on the slot."""

    def __init__(self, zoom, st, depth=2):
        self.zoom, self.st, self.depth = zoom, st, depth
        self.flight = deque()  # [fn, out, ok so far, assumptions, deferred flags, mark, slot, repeat]
        self.done = []  # the calls' results, in order, as they settle
        self.tally = {"kept": 0, "redone": 0, "replayed": 0}
        self.graphs = {}  # (repeat, slot) -> (graph, out, assumptions, deferred), or None: refused
        self.calls = defaultdict(int)  # (repeat, slot) -> its calls queued

    def submit(self, fn, repeat=None):
        be = backend()
        if not be.asynchronous:
            self.done.append(fn())
            return
        while len(self.flight) >= self.depth:
            self._settle()
        busy = {f[6] for f in self.flight}
        self.flight.append(self._queue(fn, min(set(range(self.depth)) - busy), repeat))

    def drain(self):
        """Settle every call in flight; the results so far."""
        while self.flight:
            self._settle()
        out, self.done = self.done, []
        return out

    def _queue(self, fn, slot, repeat=None):
        be = backend()
        views = [v.data for r in self.zoom.regions() for v in self.st[r].values()]
        exchange.save(self.st, ("speculate", slot), views)
        key = (repeat, slot)
        kept = self.graphs.get(key) if repeat is not None else None
        if kept is not None:
            graph, out, assumptions, deferred = kept
            be.replay(graph)
            self.tally["replayed"] += 1
            return [fn, out, True, list(assumptions), list(deferred), be.mark(), slot, repeat]
        capture = (
            repeat is not None and key not in self.graphs and self.calls[key] and hasattr(be, "capture_begin")
        )
        if repeat is not None:
            self.calls[key] += 1
        got = self._speculate(fn, slot, capture) if capture else None
        if got is None:  # not captured: queued as it goes
            if capture:
                self.graphs[key] = None
            got = self._speculate(fn, slot, False)
        out, ok, assumptions, deferred, graph = got
        if graph is not None:  # captured, nothing queued yet: the graph is the call
            if ok:
                self.graphs[key] = (graph, out, tuple(assumptions), tuple(deferred))
            be.replay(graph)
        return [fn, out, ok, assumptions, deferred, be.mark(), slot, repeat]

    def _speculate(self, fn, slot, capture):
        """(out, ok, assumptions, deferred flags, graph or None) of fn speculated on the slot's
        scratch; capturing, None where the backend refused what fn did."""
        be = backend()
        be.speculating, be._assumptions, ndef = True, [], len(be._deferred)
        be.slot_arena = be.arena_for((id(self), slot))  # free: the slot's last call is checked
        if be.slot_arena is not None:
            be.slot_arena.reset()
        out, ok, graph, refused = None, True, None, False
        if capture:
            be.capture_begin()
        try:
            out = fn()
        except (CFLError, ValueError):
            ok = False
        except Uncapturable:
            refused = True
        finally:
            be.speculating, be.slot_arena = False, None
            if capture:
                graph = be.capture_end()
        assumptions, be._assumptions = be._assumptions, []
        deferred = be._deferred[ndef:]
        del be._deferred[ndef:]
        if capture and (refused or graph is None):
            return None
        return out, ok, assumptions, deferred, graph

    def _settle(self):
        be = backend()
        fn, out, ok, assumptions, deferred, mark, slot, _ = self.flight.popleft()
        be.wait(mark)
        be.settled = True  # the verdicts read what ran before the mark: no sync
        try:
            ok = ok and not any(int(be.to_host(f, "control")[0]) for f, _ in deferred)
            ok = ok and all(verdict() for verdict in assumptions)
        except (CFLError, ValueError):
            ok = False
        finally:
            be.settled = False
        if ok:
            self.tally["kept"] += 1
            self.done.append(out)
            return
        self.tally["redone"] += 1
        later = [(f[0], f[7]) for f in self.flight]
        self.flight.clear()
        be.sync()  # what was queued on the wrong state has run
        exchange.restore(self.st, ("speculate", slot))
        self.done.append(fn())
        for f, repeat in later:
            busy = {g[6] for g in self.flight}
            self.flight.append(self._queue(f, min(set(range(self.depth)) - busy), repeat))
