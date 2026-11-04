"""Check_CFL's zoom path (advectm_cfl.F90): the rehearsal on a state wherever it lives (the
bodies through `st.lib`) and the NumPy bodies.  Part of pydyn.zoom (see its docstring)."""

import numpy as np

from ..cfl import dynamum, dynamvm, new_valid_timestep
from ..kernels import subcycle_rows
from ..ops import ONE, nint, progressive
from .advect import (
    column_trial,
    x_scope,
    y_scope,
    z_scope,
)
from .exchange import (
    _compress_xedges,
    _compress_yedges,
    _uncompress_xedges,
    _uncompress_yedges,
    mix_edges,
)
from .massflow import setup_massflow

# ------------------------------------------------------------------ Check_CFL's zoom path (advectm_cfl)
# advectm_cfl.F90 (proj/ct-unified/trunk) rehearses the coming steps with the air mass alone,
# through its own copy of the zoom machinery: the split order `xyzzyx` in blocks of three,
# mass-only exchange (putm_x/yedges, compressm/uncompressm, mixm_edges, updatem_parent -- the
# m statements of the tracer routines above, hence tracers=False), mass-only kernels with the
# sub-cycle caps cfl.py mirrors, and a failure that sets cfl_ok(region) instead of stopping.
CFL_SPLITORDER = "xyzzyx"  # advectm_cfl.F90 splitorder
CFL_N_OPERATORS = 3  # advectm_cfl.F90 n_operators (its own, not dims')


def cfl_splitorderzoom(zoom):
    """advectm_cfl.F90's define_splitorderzoom: zoom_mode 1 on blocks of three."""
    order = {1: CFL_SPLITORDER}
    for r in range(2, zoom.nregions + 1):
        tref_ = zoom.ref(r, "t")
        parent = order[zoom.g(r).parent]
        out = []
        for j0 in range(0, len(parent), CFL_N_OPERATORS):
            block = parent[j0 : j0 + CFL_N_OPERATORS]
            for i in range(1, tref_ + 1):
                out.append(block if i % 2 == 1 else block[::-1])
        order[r] = "".join(out)
    return order


class _CFLRun:
    """The module state of advectm_cfl during one Check_CFL: regionm_status, cfl_ok, and the
    CFL split order."""

    def __init__(self, zoom, caps=None):
        self.caps = caps  # (x, y, z) sub-cycle caps; None: the kernels' pinned defaults
        self.order = cfl_splitorderzoom(zoom)
        self.status = [0] * (zoom.nregions + 1)
        self.ok = [True] * (zoom.nregions + 1)

    def all_ok(self):
        return all(self.ok[1:])


def _cfl_block(zoom, run, region):
    tref_ = zoom.ref(region, "t")
    q = run.status[region] // (CFL_N_OPERATORS * tref_)
    start = q * CFL_N_OPERATORS * tref_
    second = run.status[region] % (CFL_N_OPERATORS * tref_) >= CFL_N_OPERATORS
    return run.order[region][start : start + CFL_N_OPERATORS], second


def _advectm_x(zoom, st, run, region, max_nloop=10):
    """advectmxzoom -> advectmx -> dynamum (mass only)."""
    lib = st.lib
    lib.put_xedges(zoom, st, region, tracers=False)
    lm = zoom.lm
    block, second = _cfl_block(zoom, run, region)
    js, je = x_scope(zoom, region, block)
    with lib.outer_x_faces_zeroed(zoom, st, region, second):
        if region == 1:
            ok = lib._dynamum_region1(zoom, st, js, je, max_nloop)
        else:
            ok = lib._dynamum_child(zoom, st, region, js, je, 1, lm, max_nloop)
        run.ok[region] = ok


def _dynamum_region1(zoom, st, js, je, max_nloop):
    """dynamum(1) through cfl.dynamum, plus the halos its closing loop and the CFL's own
    red2uni leave: m(0) = m(im) on every row js..je, m(im+1) = m(1) on the uniform rows (on a
    reduced row that write lands inside the row and red2uni overwrites it)."""

    g, f, lm = zoom.g(1), st[1], zoom.lm
    im, jm = g.im, g.jm
    m_int, am_int = f["m"][1:im, 1:jm, 1:lm], f["am"][0:im, 1:jm, 1:lm]
    if dynamum(m_int, am_int, g.redgrid, max_nloop=max_nloop) is None:
        return False
    m = f["m"]
    reduced = set(g.redgrid.rows) if g.redgrid is not None else set()
    for j in range(js, je + 1):
        m[0, j] = m[im, j].copy()
        if j - 1 not in reduced:
            m[im + 1, j] = m[1, j].copy()
    return True


def _dynamum_child(zoom, st, region, js, je, ls, le, max_nloop):
    """dynamum (mass only) of a child: compressm_xedges, per-row trial and update with the
    end cells held, then uncompressm_xedges and mixm_edges -- the last two skipped when a row
    fails, as the Fortran returns straight away."""
    _compress_xedges(zoom, st, region, js, je, ls, le, tracers=False)
    g, f = zoom.g(region), st[region]
    xr = zoom.ref(region, "x")
    is_, ie = xr, g.im - xr + 1
    m, am = f["m"], f["am"]
    for j in range(js, je + 1):  # every level at once; a failed rehearsal's state is discarded
        cells, a = m[is_ - 1 : ie + 1, j, ls:le], am[is_ - 1 : ie, j, ls:le]
        found = subcycle_rows(cells, a, max_nloop - 1, periodic=False)
        if found is None:
            return False
        nloop, _ = found
        for v in np.unique(nloop):
            cols = np.flatnonzero(nloop == v)
            part, faces = cells[:, cols], a[:, cols]
            for _ in range(int(v)):
                part[1:-1] = part[1:-1] + faces[:-1] - faces[1:]
            cells[:, cols] = part
        a[...] = a * nloop
    _uncompress_xedges(zoom, st, region, js, je, ls, le, tracers=False)
    mix_edges(zoom, st, region, tracers=False)
    return True


def _advectm_y(zoom, st, run, region, max_nloop=1):
    """advectmyzoom -> advectmy -> dynamvm (mass only)."""
    lib = st.lib
    lib.put_yedges(zoom, st, region, tracers=False)
    lm = zoom.lm
    block, second = _cfl_block(zoom, run, region)
    is_, ie = y_scope(zoom, region, block)
    with lib.outer_y_faces_zeroed(zoom, st, region, second):
        if region == 1:
            ok = lib._dynamvm_region1(zoom, st, max_nloop)
        else:
            ok = lib._dynamvm_child(zoom, st, region, is_, ie, 1, lm, max_nloop)
        run.ok[region] = ok


def _dynamvm_region1(zoom, st, max_nloop):
    """dynamvm(1) through cfl.dynamvm."""
    g, f, lm = zoom.g(1), st[1], zoom.lm
    im, jm = g.im, g.jm
    return dynamvm(f["m"][1:im, 1:jm, 1:lm], f["bm"][1:im, 1 : jm + 1, 1:lm], max_nloop=max_nloop) is not None


def _dynamvm_child(zoom, st, region, is_, ie, ls, le, max_nloop):
    """dynamvm of a child touching neither pole: one nloop per layer; on a failed trial the
    scope's faces bm(is:ie, js:je+1, l) are scaled (the whole layer is only restored)."""
    _compress_yedges(zoom, st, region, is_, ie, ls, le, tracers=False)
    g, f = zoom.g(region), st[region]
    yr = zoom.ref(region, "y")
    js, je = yr, g.jm - yr + 1
    m, bm = f["m"], f["bm"]
    for lev in range(ls, le + 1):
        cells, b = m[is_:ie, js - 1 : je + 1, lev], bm[is_:ie, js : je + 1, lev]
        nloop = 1
        while True:
            mx, ok = cells.copy(), True
            for _ in range(nloop):
                beta = np.abs(np.where(b >= 0.0, b / mx[:, :-1], b / mx[:, 1:]))
                if (beta >= ONE).any():
                    ok = False
                    break
                mx[:, 1:-1] = mx[:, 1:-1] + b[:, :-1] - b[:, 1:]
            if ok:
                break
            b[...] = progressive(b, nloop + 1, None)
            nloop += 1
            if nloop > max_nloop:
                return False
        for _ in range(nloop):
            cells[:, 1:-1] = cells[:, 1:-1] + b[:, :-1] - b[:, 1:]
        layer = bm[:, :, lev]
        layer[...] = layer * nloop
    _uncompress_yedges(zoom, st, region, is_, ie, ls, le, tracers=False)
    mix_edges(zoom, st, region, tracers=False)
    return True


def _advectm_z(zoom, st, run, region, iter_limit=1):
    """advectmzzoom -> advectmz -> dynamwm (mass only): the Courant search over the whole
    region, the mass updated on the scope, cm restored exactly; mixm_edges after."""
    block, _ = _cfl_block(zoom, run, region)
    is_, ie, js, je = z_scope(zoom, region, block)
    run.ok[region] = st.lib._dynamwm(zoom, st, region, is_, ie, js, je, iter_limit)
    if run.ok[region] and region != 1:
        st.lib.mix_edges(zoom, st, region, tracers=False)


def _dynamwm(zoom, st, region, is_, ie, js, je, iter_limit):
    """dynamwm: the mass advanced on the scope is_:ie x js:je if the search passes."""
    g, f, lm = zoom.g(region), st[region], zoom.lm
    im, jm = g.im, g.jm
    full_m = f["m"][1:im, 1:jm, 1:lm]
    si, sj = slice(is_ - 1, ie), slice(js - 1, je)
    n_advz, _, mx = column_trial(full_m, f["cm"][1:im, 1:jm, 0:lm], si, sj, iter_limit)
    if n_advz is None:
        return False
    full_m[si, sj] = mx[si, sj]
    return True


_CFL_OPERATORS = {"x": _advectm_x, "y": _advectm_y, "z": _advectm_z}


def _cfl_do_steps(zoom, st, run, region):
    """advectm_cfl.F90 do_steps: the next operators up to a block boundary (three), returning
    as soon as any region has failed."""
    for _ in range(CFL_N_OPERATORS):
        op = run.order[region][run.status[region]]
        if run.caps is None:
            _CFL_OPERATORS[op](zoom, st, run, region)
        else:
            _CFL_OPERATORS[op](zoom, st, run, region, run.caps["xyz".index(op)])
        if not run.all_ok():
            return
        run.status[region] += 1
        if run.status[region] % CFL_N_OPERATORS == 0:
            return


def determine_cfl_iter(zoom, st, run, region):
    """advectm_cfl.F90 determine_cfl_iter: the region's tref_ blocks, each followed by its
    children, then updatem_parent; any failure unwinds straight back to Check_CFL."""
    run.ok[region] = True
    for _ in range(zoom.ref(region, "t")):
        _cfl_do_steps(zoom, st, run, region)
        if not run.ok[region]:
            return
        for child in zoom.children[region]:
            if not run.all_ok():
                break
            determine_cfl_iter(zoom, st, run, child)
        if run.status[region] % CFL_N_OPERATORS != 0 and run.all_ok():
            _cfl_do_steps(zoom, st, run, region)
            if not run.ok[region]:
                return
    if region != 1:
        st.lib.update_parent(zoom, st, region, tracers=False)


def check_cfl_zoom(zoom, st, ndyn, n, nread, interpolate=None, max_iteration=100, caps=None):
    """advectm_cfl.F90 Check_CFL for a region tree, the CFL cache off.  For each of the n
    steps, Setup_MassFlow (after `interpolate(zoom, st, i, ndyn)` has set every region's
    pu/pv, if given) and the xyz and zyx rehearsals; on a failure the masses and fluxes are
    restored, ndyn goes to the next valid step (a multiple of 2 max tref) and the stored
    fluxes are scaled by ndyn_new / ndyn_old.  The state is left as the Fortran leaves it:
    m, am, bm, cm restored (and scaled), sp / phlb as the rehearsal's m2phlb wrote them.
    `caps` = (x, y, z) replaces the sub-cycle caps (production: cfl_max_nloop_{x,y,z} = 10, 1,
    1), as cfl_cap_replay asks of a single region.  Returns (ndyn, n, iterations)."""

    tref_max = max(zoom.g(r).tref for r in zoom.regions())
    for iteration in range(1, max_iteration + 1):
        run = _CFLRun(zoom, caps)
        st.lib.save(st, "check_cfl", ("m", "am", "bm", "cm"))
        for i in range(1, n + 1):
            if interpolate is not None:
                interpolate(zoom, st, i, ndyn)
            setup_massflow(zoom, st, ndyn)
            determine_cfl_iter(zoom, st, run, 1)
            if run.all_ok():
                determine_cfl_iter(zoom, st, run, 1)
            if not run.all_ok():
                break
            run.status = [0] * (zoom.nregions + 1)
        st.lib.restore(st, "check_cfl")
        if run.all_ok():
            return ndyn, n, iteration
        ndyn_old = ndyn
        ndyn = new_valid_timestep(ndyn, nread, nread, tref_max)
        n = nint(float(n * ndyn_old) / float(ndyn))
        fraction = float(ndyn) / float(ndyn_old)
        st.lib.scale(st, ("am", "bm", "cm"), fraction)
    raise RuntimeError("check_cfl_zoom: exceeded maximum number of time steps")
