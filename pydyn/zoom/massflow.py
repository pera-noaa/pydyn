"""m2phlb / m2phlb1, coarsen_region / coarsen_ambm, Setup_MassFlow's region loop, dynam0.
coarsen_region's recursion and Setup_MassFlow run on a state wherever it lives (the bodies
through `st.lib`).  Part of pydyn.zoom (see its docstring)."""

import numpy as np

from ..balance import GRAV
from ..kernels import (
    fluxes_from_meteo,
)
from ..ops import seqsum

__all__ = [
    "coarsen_ambm",
    "coarsen_region",
    "dynam0",
    "m2phlb",
    "m2phlb1",
    "setup_massflow",
]


# ------------------------------------------------------------------ m2phlb / m2phlb1 (advect_tools)
def m2phlb1(zoom, st, region=1):
    """advect_tools__speedup m2phlb1: sp from the column mass, phlb = at + bt sp.  The whole
    sp array, halos included, is zeroed first."""
    g, f = zoom.g(region), st[region]
    im, jm, lm = g.im, g.jm, zoom.lm
    p = seqsum(f["m"][1:im, 1:jm, 1:lm], axis=2) * GRAV / f["dxyp"][1:jm][None, :]
    f["sp"].data[...] = 0.0
    f["sp"][1:im, 1:jm, 1] = p
    f["phlb"][1:im, 1:jm, 1 : lm + 1] = zoom.at + zoom.bt * p[..., None]


def m2phlb(zoom, st, region):
    """advect_tools__speedup m2phlb (a child): as m2phlb1, then the ring cells' pressure is
    rescaled to the parent cell's area, p dxyp(j) xref_ yref_ / dxyp_parent(jpar)."""
    g, f = zoom.g(region), st[region]
    im, jm, lm = g.im, g.jm, zoom.lm
    xr, yr = zoom.ref(region, "x"), zoom.ref(region, "y")
    dxyp, dxy = f["dxyp"], st[g.parent]["dxyp"]
    p = seqsum(f["m"][1:im, 1:jm, 1:lm], axis=2) * GRAV / dxyp[1:jm][None, :]
    for j in range(1, jm + 1):
        for i in range(1, im + 1):
            if xr < i < im - xr + 1 and yr < j < jm - yr + 1:
                continue
            if g.xcyc == 1 and yr < j < jm - yr + 1:
                continue
            if g.touch_np == 1 and j > jm - yr:
                continue
            if g.touch_sp == 1 and j < yr + 1:
                continue
            jpar = zoom.jbeg[region] + (j - 1) // yr
            p[i - 1, j - 1] = p[i - 1, j - 1] * dxyp[j] * xr * yr / dxy[jpar]
    f["sp"].data[...] = 0.0
    f["sp"][1:im, 1:jm, 1] = p
    f["phlb"][1:im, 1:jm, 1 : lm + 1] = zoom.at + zoom.bt * p[..., None]


# ------------------------------------------------------------------ coarsen_region / coarsen_ambm
def coarsen_region(zoom, st, region):
    """zoom_tools.F90 coarsen_region: the parent's am, bm, cm and m over the child footprint
    rebuilt from the child's (fluxes times tref_), then m2phlb of the parent, recursively up."""
    parent = zoom.g(region).parent
    if parent <= 0:
        return
    lib = st.lib
    lib._coarsen_block(zoom, st, region)
    if parent != 1:
        lib.m2phlb(zoom, st, parent)
    else:
        lib.m2phlb1(zoom, st, parent)
    coarsen_region(zoom, st, parent)


def _coarsen_block(zoom, st, region):
    """coarsen_region's sums into the parent's footprint."""
    parent = zoom.g(region).parent
    g, f, p = zoom.g(region), st[region], st[parent]
    xr, yr, zr, tr = (zoom.ref(region, a) for a in "xyzt")
    imp, jmp, lmp = g.im // xr, g.jm // yr, zoom.lm // zr
    ib, jb, lb = zoom.ibeg[region], zoom.jbeg[region], zoom.lbeg[region]
    im_par = zoom.g(parent).im
    ipg = np.mod(ib - 2 + np.arange(0, imp + 1), im_par) + 1  # parent i of child block i = 0..imp
    lm = zoom.lm
    if zr != 1:
        raise ValueError("coarsen_region: zooming in z direction not supported")

    # am: faces ip*xref_ of the child, summed over the block's rows (j ascending)
    am = f["am"][0 : g.im, 1 : g.jm, 1:lm]
    faces = am[xr * np.arange(0, imp + 1)]  # (imp+1, jm, lm)
    blk = faces.reshape(imp + 1, jmp, yr, lm)
    top = np.zeros((imp + 1, jmp, lm))
    for b in range(yr):
        top = top + blk[:, :, b] * tr
    jpg = (jb + np.arange(jmp))[None, :, None]
    lpg = (lb + np.arange(lmp))[None, None, :]
    p["am"].data[ipg[:, None, None], jpg, lpg] = top  # lower bounds 0: data index == Fortran index

    # bm: faces 1 + (jp-1) yref_ of the child, summed over the block's columns (i ascending)
    bm = f["bm"][1 : g.im, 1 : g.jm + 1, 1:lm]
    faces = bm[:, yr * np.arange(0, jmp + 1)]  # (im, jmp+1, lm)
    blk = faces.reshape(imp, xr, jmp + 1, lm)
    top = np.zeros((imp, jmp + 1, lm))
    for a in range(xr):
        top = top + blk[:, a] * tr
    pb = p["bm"]
    pb.data[
        (ipg[1:] - pb.lower[0])[:, None, None],
        (jb - 1 + np.arange(1, jmp + 2) - pb.lower[1])[None, :, None],
        (lb + np.arange(lmp) - pb.lower[2])[None, None, :],
    ] = top

    # cm: summed over the block, i outer / j inner
    cm = f["cm"][1 : g.im, 1 : g.jm, 1:lm]  # interfaces l = 1..lm (lp * zref_)
    blk = cm.reshape(imp, xr, jmp, yr, lm)
    top = np.zeros((imp, jmp, lm))
    for a in range(xr):
        for b in range(yr):
            top = top + blk[:, a, :, b] * tr
    pc = p["cm"]
    pc.data[
        (ipg[1:] - pc.lower[0])[:, None, None],
        (jb + np.arange(jmp) - pc.lower[1])[None, :, None],
        (lb + np.arange(lmp) - pc.lower[2])[None, None, :],
    ] = top

    # m: summed over the block, j outer / i inner
    blk = f["m"][1 : g.im, 1 : g.jm, 1:lm].reshape(imp, xr, jmp, yr, lm)
    top = np.zeros((imp, jmp, lm))
    for b in range(yr):
        for a in range(xr):
            top = top + blk[:, a, :, b]
    pm = p["m"]
    pm.data[
        (ipg[1:] - pm.lower[0])[:, None, None],
        (jb + np.arange(jmp) - pm.lower[1])[None, :, None],
        (lb + np.arange(lmp) - pm.lower[2])[None, None, :],
    ] = top


def coarsen_ambm(zoom, st, region):
    """advect_tools__speedup coarsen_ambm: each child's outer-interface fluxes am(xref_-1),
    am(im-xref_+1), bm(yref_), bm(jm-yref_+2) set from the parent's face flux divided over the
    child faces it covers."""
    f = st[region]
    am, bm = f["am"], f["bm"]
    for child in zoom.children[region]:
        gc, c = zoom.g(child), st[child]
        xr, yr, zr = zoom.ref(child, "x"), zoom.ref(child, "y"), zoom.ref(child, "z")
        ib, ie, jb, je = zoom.ibeg[child], zoom.iend[child], zoom.jbeg[child], zoom.jend[child]
        lb, le = zoom.lbeg[child], zoom.lend[child]
        amc, bmc = c["am"], c["bm"]
        if gc.im // gc.xref != zoom.g(1).im:
            for j in range(jb, je + 1):
                for lev in range(lb, le + 1):
                    jc, lc = (j - jb) * yr + 1, (lev - lb) * zr + 1
                    cell1 = am[ib - 1, j, lev] / (yr * zr)
                    cellm = am[ie, j, lev] / (yr * zr)
                    amc[xr - 1, jc : jc + yr - 1, lc : lc + zr - 1] = cell1
                    amc[gc.im - xr + 1, jc : jc + yr - 1, lc : lc + zr - 1] = cellm
        for i in range(ib, ie + 1):
            for lev in range(lb, le + 1):
                ic, lc = (i - ib) * xr + 1, (lev - lb) * zr + 1
                cell1 = bm[i, jb, lev] / (xr * zr)
                cellm = bm[i, je + 1, lev] / (xr * zr)
                if gc.touch_sp != 1:
                    bmc[ic : ic + xr - 1, yr, lc : lc + zr - 1] = cell1
                if gc.touch_np != 1:
                    bmc[ic : ic + xr - 1, gc.jm - yr + 2, lc : lc + zr - 1] = cellm


# ------------------------------------------------------------------ Setup_MassFlow's region loop
def setup_massflow(zoom, st, ndyn):
    """advectm_cfl.F90 (proj/ct-unified/trunk) Setup_MassFlow after its TimeInterpolation of
    pu/pv (taken as they are; check_cfl_zoom's `interpolate` is where a caller varies them): dynam0 and
    coarsen_region region by region -- interleaved, so a child's coarsening overwrites its
    parent's footprint after the parent's own dynam0 -- then coarsen_ambm for every region."""
    lib = st.lib
    for r in zoom.regions():
        lib.dynam0(zoom, st, r, ndyn)
        coarsen_region(zoom, st, r)
    for r in zoom.regions():
        lib.coarsen_ambm(zoom, st, r)


# ------------------------------------------------------------------ dynam0 / convec for a region
def dynam0(zoom, st, region, ndyn):
    """advect.F90 dynam0 for any region: dt = ndyn / (2 tref); am, bm, cm zeroed whole first;
    a cyclic region gets its periodic x halo faces."""
    g, f = zoom.g(region), st[region]
    im, jm, lm = g.im, g.jm, zoom.lm
    dt = ndyn / (2.0 * g.tref)
    am, bm, cm = fluxes_from_meteo(f["pu"][0:im, 1:jm, 1:lm], f["pv"][1:im, 1 : jm + 1, 1:lm], zoom.bt, dt)
    for name in ("am", "bm", "cm"):
        f[name].data[...] = 0.0
    f["am"][0:im, 1:jm, 1:lm] = am
    f["bm"][1:im, 1 : jm + 1, 1:lm] = bm
    f["cm"][1:im, 1:jm, 1 : lm - 1] = cm[..., 1:lm]
    if g.xcyc == 1:
        a = f["am"]
        a[0, :, :] = a[im, :, :].copy()
        a[im + 1, :, :] = a[1, :, :].copy()
