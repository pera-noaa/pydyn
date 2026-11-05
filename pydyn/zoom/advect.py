"""advect{x,y,z}zoom: a region's advection, region 1 on the kernels, a child on its own.  The
entry points run on a state wherever it lives (the bodies through `st.lib`); the rest are the
NumPy bodies.  Part of pydyn.zoom (see its docstring)."""

from contextlib import contextmanager

import numpy as np

from ..kernels import (
    CFLError,
    _advect_rows_x,
    _along_slope_update,
    _limit,
    _slopes_fluxes,
    advect_y,
    advect_z,
    advect_z_substeps,
    subcycle_rows,
)
from ..ops import ONE, advance, fresh, progressive, upwind
from ..redgrid import red2uni, uni2red, uni2red_mf
from .exchange import (
    _compress_xedges,
    _compress_yedges,
)
from .state import _interior
from .tree import N_OPERATORS

__all__ = [
    "advect_x_zoom",
    "advect_y_zoom",
    "advect_z_zoom",
    "column_trial",
    "outer_x_faces_zeroed",
    "outer_y_faces_zeroed",
    "x_scope",
    "y_scope",
    "z_scope",
]


# ------------------------------------------------------------------ advection: scope of a call
def _block(zoom, region):
    """The operators of the current block of splitorderzoom and tref_ (advect*zoom's q)."""
    tref_ = zoom.ref(region, "t")
    q = zoom.status[region] // (N_OPERATORS * tref_)
    start = q * N_OPERATORS * tref_
    return zoom.splitorderzoom[region][start : start + N_OPERATORS], tref_


def _second_half(zoom, region):
    """mod(status, 6 tref_) >= 6: the coarse flux has crossed the interface already."""
    tref_ = zoom.ref(region, "t")
    return zoom.status[region] % (N_OPERATORS * tref_) >= N_OPERATORS


def x_scope(zoom, region, block):
    """advectxzoom's j range (js, je) for the block's order: the full column when y comes after
    x, else the rows inside the y refinement (region 1, yref_ = 0, keeps them all)."""
    g = zoom.g(region)
    yr = 0 if region == 1 else zoom.ref(region, "y")
    x_seen = False
    js = je = None
    for d in block:
        if d == "x":
            x_seen = True
        elif d == "y":
            if not x_seen:
                js, je = 1, g.jm
            else:
                js = 1 if g.touch_sp == 1 else yr + 1
                je = g.jm if g.touch_np == 1 else g.jm - yr
    return js, je


def y_scope(zoom, region, block):
    """advectyzoom's i range (is_, ie): the full row when x comes after y or the region is
    cyclic, else the columns inside the x refinement (region 1, xref_ = 0, keeps them all)."""
    g = zoom.g(region)
    xr = 0 if region == 1 else zoom.ref(region, "x")
    y_seen = False
    is_ = ie = None
    for d in block:
        if d == "x":
            if not y_seen or g.xcyc == 1:
                is_, ie = 1, g.im
            else:
                is_, ie = xr + 1, g.im - xr
        elif d == "y":
            y_seen = True
    return is_, ie


def z_scope(zoom, region, block):
    """advectzzoom's (is_, ie, js, je): each direction's full range when it comes after z."""
    g = zoom.g(region)
    xr, yr = (0, 0) if region == 1 else (zoom.ref(region, "x"), zoom.ref(region, "y"))
    z_seen = False
    is_ = ie = js = je = None
    for d in block:
        if d == "x":
            if not z_seen or g.xcyc == 1:
                is_, ie = 1, g.im
            else:
                is_, ie = xr + 1, g.im - xr
        elif d == "y":
            if not z_seen:
                js, je = 1, g.jm
            else:
                js = 1 if g.touch_sp == 1 else yr + 1
                je = g.jm if g.touch_np == 1 else g.jm - yr
        elif d == "z":
            z_seen = True
    return is_, ie, js, je


@contextmanager
def outer_x_faces_zeroed(zoom, st, region, second):
    """In the second half of the parent step a child's outer x interface fluxes are zeroed
    around its x advection and restored after it (not when the advection raises)."""
    g, xr = zoom.g(region), zoom.ref(region, "x")
    am = st[region]["am"]
    zero = second and g.im // g.xref < zoom.g(1).im
    if zero:
        am0, am1 = am[xr - 1, :, :].copy(), am[g.im - xr + 1, :, :].copy()
        am[xr - 1, :, :] = 0.0
        am[g.im - xr + 1, :, :] = 0.0
    yield
    if zero:
        am[xr - 1, :, :] = am0
        am[g.im - xr + 1, :, :] = am1


@contextmanager
def outer_y_faces_zeroed(zoom, st, region, second):
    """The same for the y interface fluxes of a region that does not touch the pole."""
    g, yr = zoom.g(region), zoom.ref(region, "y")
    bm = st[region]["bm"]
    if second:
        if g.touch_sp != 1:
            bm0 = bm[:, yr, :].copy()
            bm[:, yr, :] = 0.0
        if g.touch_np != 1:
            bm1 = bm[:, g.jm - yr + 2, :].copy()
            bm[:, g.jm - yr + 2, :] = 0.0
    yield
    if second:
        if g.touch_sp != 1:
            bm[:, yr, :] = bm0
        if g.touch_np != 1:
            bm[:, g.jm - yr + 2, :] = bm1


def advect_x_zoom(zoom, st, region, limits=True, stats=None):
    """advectxzoom(region): put_xedges to the children, then dynamu on the scope the block's
    order implies, the outer interface fluxes zeroed in the second half of the parent step."""
    lib = st.lib
    lib.put_xedges(zoom, st, region)
    lm = zoom.lm
    block, _ = _block(zoom, region)
    js, je = x_scope(zoom, region, block)
    ls, le = 1, lm  # zoom2D
    with lib.outer_x_faces_zeroed(zoom, st, region, _second_half(zoom, region)):
        if region == 1:
            lib._dynamu_region1(zoom, st, js, je, limits, stats)
        else:
            lib._dynamu_child(zoom, st, region, js, je, ls, le, limits, stats)
            lib._uncompress_xedges(zoom, st, region, js, je, ls, le)
            lib.mix_edges(zoom, st, region)


def advect_y_zoom(zoom, st, region, limits=True, stats=None):
    """advectyzoom(region): put_yedges to the children, then dynamv on its scope."""
    lib = st.lib
    lib.put_yedges(zoom, st, region)
    lm = zoom.lm
    block, _ = _block(zoom, region)
    is_, ie = y_scope(zoom, region, block)
    ls, le = 1, lm  # zoom2D
    with lib.outer_y_faces_zeroed(zoom, st, region, _second_half(zoom, region)):
        if region == 1:
            lib._dynamv_region1(zoom, st, limits, stats)
        else:
            lib._dynamv_child(zoom, st, region, is_, ie, ls, le, limits, stats)
            lib._uncompress_yedges(zoom, st, region, is_, ie, ls, le)
            lib.mix_edges(zoom, st, region)


def advect_z_zoom(zoom, st, region, limits=True, stats=None):
    """advectzzoom(region): dynamw on its scope (no edges are put before z)."""
    lib = st.lib
    block, _ = _block(zoom, region)
    is_, ie, js, je = z_scope(zoom, region, block)
    if region == 1:
        lib._dynamw_region1(zoom, st, limits, stats)
    else:
        lib._dynamw_child(zoom, st, region, is_, ie, js, je, limits, stats)
        lib.mix_edges(zoom, st, region)


def _dynamv_region1(zoom, st, limits, stats):
    """dynamv(1): the kernels.py y kernel on the interior, region 1's bm halo kept as dynamv
    leaves it."""
    g, f, lm = zoom.g(1), st[1], zoom.lm
    im, jm = g.im, g.jm
    layers = []
    bm_int = f["bm"][1:im, 1 : jm + 1, 1:lm]
    advect_y(*_interior(f, im, jm, lm), bm_int, limits, stats=stats, nloop_layers=layers)
    _rescale_layer_halo(f["bm"], im, jm, layers)


def _rescale_layer_halo(bm, im, jm, nloops):
    """dynamv scales the whole bm(:,:,l) at every retry and multiplies it back by nloop at the
    end -- not an exact round trip -- so region 1's faces outside bm(1:im, 1:jm+1, l), which
    the kernel's layout does not hold, drift by an ulp when a layer sub-cycles.  Nothing reads
    them; they are kept equal because the transcription is compared array by array."""
    for lev, nloop in enumerate(nloops, start=1):
        if nloop == 1:
            continue
        layer = bm[:, :, lev]
        outside = np.ones(layer.shape, dtype=bool)
        outside[1 : im + 1, 1 : jm + 2] = False  # data index = Fortran index (lower bound 0)
        halo = layer[outside]
        for n in range(2, nloop + 1):
            halo = progressive(halo, n, None)
        layer[outside] = halo * nloop


def _dynamw_region1(zoom, st, limits, stats):
    """dynamw(1): the kernels.py z kernel on the interior."""
    g, f, lm = zoom.g(1), st[1], zoom.lm
    im, jm = g.im, g.jm
    advect_z(*_interior(f, im, jm, lm), f["cm"][1:im, 1:jm, 0:lm], limits, stats=stats)


# ------------------------------------------------------------------ region 1 x: halos written
def _dynamu_region1(zoom, st, js, je, limits, stats, max_nloop=50):
    """dynamu(1): the kernels.py / redgrid.py row kernel on the interior, and then what those
    leave out because nothing in a single-region run reads it -- the periodic x halos of
    m, rm and the slopes, written (in the reduced representation, before red2uni) for every
    row js..je as dynamu's closing loop does.  A child's put_xedges can read them."""
    g, f, lm = zoom.g(1), st[1], zoom.lm
    im, jm = g.im, g.jm
    fields = _interior(f, im, jm, lm)
    am = f["am"][0:im, 1:jm, 1:lm]
    grid = g.redgrid
    if grid is not None:
        m_uni, am_uni = fields[0].copy(), am.copy()
        uni2red(grid, *fields)
        uni2red_mf(grid, am)
    nloop_max, xi = 1, 0.0
    for j in range(jm):  # every level of a latitude at once
        n = grid.imred[j] if grid is not None else im
        rows = (slice(0, n), j)
        nloop, alpha_max = _advect_rows_x(*(q[rows] for q in fields), am[0 : n + 1, j], limits, max_nloop)
        nloop_max, xi = max(nloop_max, int(nloop.max())), max(xi, float(alpha_max.max()))
    for j in range(js, je + 1):
        iimr = grid.imred[j - 1] if grid is not None else im
        for name in ("rm", "rxm", "rym", "rzm", "m"):
            q = f[name]
            q[0, j] = q[iimr, j].copy()
            q[iimr + 1, j] = q[1, j].copy()
    if grid is not None:
        red2uni(grid, *fields, advance(m_uni, am_uni))
        am[...] = am_uni
        # redgridZoom.F90 red2uni ends each reduced row with q(0,j,l) = q(im,j,l) on the
        # uniform row, for every reduced row, replacing the reduced-row value written above
        # (first seen bitwise on a glb3x2 + nam1x1 dump; tier 1 stubs the reduced grid)
        for j0 in grid.rows:
            for name in ("rm", "rxm", "rym", "rzm", "m"):
                q = f[name]
                q[0, j0 + 1] = q[im, j0 + 1].copy()
    if stats is not None:
        stats["nloop_max"], stats["xi"] = nloop_max, xi


# ------------------------------------------------------------------ child kernels
def _dynamu_child(zoom, st, region, js, je, ls, le, limits, stats, max_nloop=50):
    """advectx__slopes.F90 dynamu for a non-cyclic child: compress_xedges, then per (j, l) row
    cells is-1..ie+1 with is = xref_, ie = im-xref_+1, faces is-1..ie, the end cells held
    fixed (they are the parent's, put there by put_xedges)."""
    _compress_xedges(zoom, st, region, js, je, ls, le)
    g, f = zoom.g(region), st[region]
    xr, imr = zoom.ref(region, "x"), g.im
    is_, ie = xr, imr - xr + 1
    m, am = f["m"], f["am"]
    q4 = [f[name] for name in ("rm", "rxm", "rym", "rzm")]
    if limits:
        rm, rxm = q4[0], q4[1]
        rxm[is_ - 1 : ie + 1, js:je, ls:le] = _limit(
            rxm[is_ - 1 : ie + 1, js:je, ls:le], rm[is_ - 1 : ie + 1, js:je, ls:le]
        )
    g_nl, g_xi = 0, 0.0
    for j in range(js, je + 1):  # every level ls..le of the row at once (inclusive sections, views)
        cells = m[is_ - 1 : ie + 1, j, ls:le]  # (n, L)
        a = am[is_ - 1 : ie, j, ls:le]  # (n-1, L)
        rows = [q[is_ - 1 : ie + 1, j, ls:le] for q in q4]  # (n, L, nt)
        nloop, max_alpha = _rows_child(cells, a, rows, limits, max_nloop)
        g_nl, g_xi = max(g_nl, int(nloop.max())), max(g_xi, float(max_alpha.max()))
    if stats is not None:
        stats["nloop_max"], stats["xi"] = g_nl, g_xi


def _rows_child(m, a, q4, limits, max_nloop):
    """dynamu on non-periodic x rows of a child, all at once: m (n, R) cells with fixed ends,
    a (n-1, R) interior faces, q4 the four tracer arrays (n, R, nt).  Returns each row's
    (nloop, max_alpha): dynamu's nloop_max and xi.  Each row's arithmetic is the one it gets
    alone (kernels.subcycle_rows)."""
    found = subcycle_rows(m, a, max_nloop - 1, periodic=False)
    if found is None:
        raise CFLError("dynuamu: nloop too high")
    # l_xi = MAX(l_xi, max_alpha) is taken here, before the flux loop: the Courant numbers
    # the flux loop computes (with the reciprocal mass) never reach xi
    nloop, max_alpha = found
    for n in np.unique(nloop):
        rows = np.flatnonzero(nloop == n)
        if rows.size == m.shape[1]:  # every row alike: on the views themselves
            _substeps_child(m, a, q4, int(n), limits)
            continue
        parts = [q[:, rows] for q in (m, *q4)]
        _substeps_child(parts[0], a[:, rows], parts[1:], int(n), limits)
        for q, part in zip((m, *q4), parts, strict=True):
            q[:, rows] = part
    a[...] = a * nloop  # restore 'old' am
    return nloop, max_alpha


def _substeps_child(m, a, q4, nloop, limits):
    """nloop sub-steps of non-periodic rows: the end cells are held, the interior updated."""
    rm, rxm, rym, rzm = q4
    af = a[..., None]
    inner = slice(1, -1)
    for _ in range(nloop):
        mnew = m[1:-1] + a[:-1] - a[1:]
        m_inv = 1.0 / m  # m_inv(i) = one / m(i), i = is-1..iie+1
        alpha = af * upwind(af, m_inv[..., None])
        f, pf, fy, fz = _slopes_fluxes(af, alpha, *(upwind(af, q) for q in (rm, rxm, rym, rzm)))
        rm[inner] = rm[inner] - np.diff(f, axis=0)  # rm + (f(i-1) - f(i))
        m_old = m[inner, ..., None]
        rxm[inner] = _along_slope_update(rxm[inner], pf, af, rm[inner], f, m_old, mnew[..., None])
        if limits:
            rxm[inner] = _limit(rxm[inner], rm[inner])
        rym[inner] = rym[inner] - np.diff(fy, axis=0)
        rzm[inner] = rzm[inner] - np.diff(fz, axis=0)
        m[1:-1] = mnew


def _dynamv_child(zoom, st, region, is_, ie, ls, le, limits, stats, max_nloop=10):
    """advecty__slopes.F90 dynamv for a child touching neither pole: compress_yedges, then per
    layer cells js-1..je+1 with js = yref_, je = jm-yref_+1 over columns is..ie, faces
    js..je+1; one nloop per layer, the whole layer of bm scaled."""
    _compress_yedges(zoom, st, region, is_, ie, ls, le)
    g, f = zoom.g(region), st[region]
    yr, jmr = zoom.ref(region, "y"), g.jm
    js, je = yr, jmr - yr + 1
    m, bm = f["m"], f["bm"]
    rm, rxm, rym, rzm = (f[name] for name in ("rm", "rxm", "rym", "rzm"))
    if limits:
        rym[is_:ie, js - 1 : je + 1, ls:le] = _limit(
            rym[is_:ie, js - 1 : je + 1, ls:le], rm[is_:ie, js - 1 : je + 1, ls:le]
        )
    g_nl, g_xi = 0, 0.0
    for lev in range(ls, le + 1):
        cells = m[is_:ie, js - 1 : je + 1, lev]  # (ni, n)
        b = bm[is_:ie, js : je + 1, lev]  # (ni, n-1) interior faces of the scope
        nloop = 1
        while True:
            mx = cells.copy()
            ok = True
            for _ in range(nloop):
                beta = np.where(b >= 0.0, b / mx[:, :-1], b / mx[:, 1:])
                if np.any(np.abs(beta) >= ONE):
                    ok = False  # the trial carries on, but mx is not advanced
                    continue
                mx[:, 1:-1] = mx[:, 1:-1] + b[:, :-1] - b[:, 1:]
            if ok:
                break
            layer = bm[:, :, lev]
            layer[...] = progressive(layer, nloop + 1, None)  # the whole layer
            nloop += 1
            if nloop == max_nloop:
                raise CFLError("nloop too high in dynamv")
        g_nl = max(g_nl, nloop)

        for _ in range(nloop):
            mnew = cells[:, 1:-1] + b[:, :-1] - b[:, 1:]
            inv_mnew = (1.0 / mnew)[..., None]
            flow_conv = (b[:, :-1] - b[:, 1:])[..., None]
            flow_mean = (3.0 * (b[:, :-1] + b[:, 1:]))[..., None]
            mass_x3 = (3.0 * cells[:, 1:-1])[..., None]
            bf = b[..., None]
            beta = bf / np.where(bf >= 0.0, cells[:, :-1, None], cells[:, 1:, None])
            g_xi = max(g_xi, float(np.abs(beta).max()))
            q = [x[is_:ie, js - 1 : je + 1, lev] for x in (rm, rym, rxm, rzm)]  # (ni, n, nt)
            fl, pf, fx, fz = _slopes_fluxes(bf, beta, *(upwind(bf, x, axis=1) for x in q))
            rmq, rymq, rxmq, rzmq = (x[:, 1:-1] for x in q)
            rm_new = rmq + (fl[:, :-1] - fl[:, 1:])
            rmq[...] = rm_new
            f_sum = fl[:, :-1] + fl[:, 1:]
            rym_new = (
                rymq
                + ((pf[:, :-1] - pf[:, 1:]) - flow_conv * rymq + flow_mean * rm_new - mass_x3 * f_sum)
                * inv_mnew
            )
            rymq[...] = _limit(rym_new, rm_new) if limits else rym_new
            rxmq[...] = rxmq + (fx[:, :-1] - fx[:, 1:])
            rzmq[...] = rzmq + (fz[:, :-1] - fz[:, 1:])
            cells[:, 1:-1] = mnew
        layer = bm[:, :, lev]
        layer[...] = layer * nloop  # restore old bm's
    if stats is not None:
        stats["nloop_max"], stats["xi"] = g_nl, g_xi


def column_trial(full_m, full_c, si, sj, iter_limit):
    """dynamw's (and dynamwm's) sub-cycle search on a region: the Courant numbers of every
    column of the interior full_m (im, jm, lm) under full_c (im, jm, lm+1), the trial mass
    advanced only on the scope (si, sj).  Returns (n_advz, xi, the trial mass) -- n_advz None
    past iter_limit -- with xi the largest Courant number, in Fortran order up to the first
    violation."""
    xi, n_advz = 0.0, 1
    while True:
        c = fresh(None, n_advz, full_c)
        mx = full_m.copy()
        ok = True
        for _ in range(n_advz):
            ci = c[..., 1:-1]
            gamma = np.abs(np.where(ci >= 0.0, ci / mx[..., :-1], ci / mx[..., 1:]))
            flat = gamma.ravel(order="F")
            bad = flat >= ONE
            if bad.any():
                xi = max(xi, float(flat[: int(np.argmax(bad)) + 1].max()))
                ok = False
                break
            xi = max(xi, float(flat.max()))
            mx[si, sj] = advance(mx[si, sj], c[si, sj], axis=2)
        if ok:
            return n_advz, xi, mx
        n_advz += 1
        if n_advz > iter_limit:
            return None, xi, mx


def _dynamw_child(zoom, st, region, is_, ie, js, je, limits, stats, iter_limit=15):
    """advectz__slopes.F90 (speedup) dynamw for a child: the Courant search looks at every
    column of the region, but the trial and the real mass updates touch only the scope
    is..ie x js..je; the sub-steps are kernels.advect_z_substeps on that scope."""
    g, f, lm = zoom.g(region), st[region], zoom.lm
    im, jm = g.im, g.jm
    m, cm = f["m"], f["cm"]
    rm, rxm, rym, rzm = (f[name] for name in ("rm", "rxm", "rym", "rzm"))
    if np.any(cm[:, :, 0] != 0.0) or np.any(cm[:, :, lm] != 0.0):
        raise ValueError("dynamw: found non-zero halo interfaces in cm")
    if limits:
        rzm[is_:ie, js:je, 1:lm] = _limit(rzm[is_:ie, js:je, 1:lm], rm[is_:ie, js:je, 1:lm])

    si, sj = slice(is_ - 1, ie), slice(js - 1, je)  # the scope, 0-based within the interior
    n_advz, xi, _ = column_trial(m[1:im, 1:jm, 1:lm], cm[1:im, 1:jm, 0:lm], si, sj, iter_limit)
    if n_advz is None:
        raise CFLError("dynamw: too many CFL violations!")

    cells = (slice(is_, ie), slice(js, je), slice(1, lm))
    faces = (slice(is_, ie), slice(js, je), slice(0, lm))
    advect_z_substeps(
        m[cells],
        *(q[cells] for q in (rm, rxm, rym, rzm)),
        cm[faces],
        n_advz,
        limits,
    )
    if stats is not None:
        stats["nloop_max"], stats["xi"] = n_advz, xi
