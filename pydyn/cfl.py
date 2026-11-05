"""Check_CFL (advectm_cfl.F90, proj/ct-unified/trunk): the mass-only rehearsal of the coming
`n` dynamic steps that decides the time step `ndyn`.  One region; the same routine on a
region tree -- advectm_cfl's own mass-only zoom exchange and recursion -- is
zoom.check_cfl_zoom, which uses the three mass kernels here for region 1.

For every step: the meteo fluxes are interpolated to the middle of the step and turned into
am, bm, cm (Setup_MassFlow = TimeInterpolation + dynam0), then the mass field is pushed
through x, y, z and z, y, x with the same sub-cycling rules as the advection kernels, but
with tighter sub-cycle limits (10, 1, 1).  If any operator would exceed its limit the whole
interval is retried with the next smaller `ndyn` that divides `nread` (new_valid_timestep),
and the fluxes already computed for the interval are scaled by ndyn_new / ndyn_old.

Arrays are in the pydyn layout (layout.py).  The three mass kernels are `ops.subcycles`
followed by the sub-steps; they work in place on m and return the sub-cycle count, or None
when the limit was exceeded.
"""

import numpy as np

from .kernels import fluxes_from_meteo, subcycle_rows
from .ops import advance, fresh, nint, progressive, subcycles
from .redgrid import red2uni_mass, uni2red_mass, uni2red_mf

__all__ = ["check_cfl", "dynamum", "dynamvm", "dynamwm", "new_valid_timestep", "rehearse_step"]

MAX_NLOOP_X, MAX_NLOOP_Y, MAX_NLOOP_Z = 10, 1, 1
MAX_GLOBAL_ITERATION = 100


def dynamum(m, am, grid=None, max_nloop=MAX_NLOOP_X):
    """advectmx: x mass advection, every (j, l) row sub-cycled on its own (on the reduced grid
    when given).  The Fortran gives up when nloop *reaches* max_nloop."""
    _, jm, _ = m.shape
    if grid is not None:
        m_uni, am_uni = m.copy(), am.copy()
        uni2red_mass(grid, m)
        uni2red_mf(grid, am)
    nloop_max = 1
    for j in range(jm):  # every level of a latitude at once; a failed rehearsal's state is discarded
        n = grid.imred[j] if grid is not None else m.shape[0]
        rows, a = m[:n, j], am[: n + 1, j]
        found = subcycle_rows(rows, a, max_nloop - 1)
        if found is None:
            return None
        nloop, _ = found
        for v in np.unique(nloop):
            cols = np.flatnonzero(nloop == v)
            part, faces = rows[:, cols], a[:, cols]
            for _ in range(int(v)):
                part = advance(part, faces)
            rows[:, cols] = part
        a[...] = a * nloop  # restore old am
        nloop_max = max(nloop_max, int(nloop.max()))
    if grid is not None:
        red2uni_mass(grid, m, advance(m_uni, am_uni))
        am[...] = am_uni
    return nloop_max


def dynamvm(m, bm, max_nloop=MAX_NLOOP_Y):
    """advectmy, region 1: y mass advection, one sub-cycle count per layer."""
    nloop_max = 1
    for k in range(m.shape[2]):
        layer, b = m[:, :, k], bm[:, :, k]
        found = subcycles(layer, b, max_nloop, progressive, axis=1)
        if found is None:
            return None
        nloop, _ = found
        for _ in range(nloop):
            layer[...] = advance(layer, b, 1)
        b[...] = b * nloop
        nloop_max = max(nloop_max, nloop)
    return nloop_max


def dynamwm(m, cm, iter_limit=MAX_NLOOP_Z):
    """advectmz, region 1: z mass advection, one sub-cycle count for the whole region."""
    found = subcycles(m, cm, iter_limit, fresh, axis=2)
    if found is None:
        return None
    n_advz, c = found
    for _ in range(n_advz):
        m[...] = advance(m, c, 2)
    return n_advz


def rehearse_step(m, am, bm, cm, grid=None):
    """determine_cfl_iter twice (xyz then zyx) on a copy of the fluxes.  Returns True if
    every operator stayed within its sub-cycle limit."""
    operators = {"x": (dynamum, am.copy(), grid), "y": (dynamvm, bm.copy()), "z": (dynamwm, cm.copy())}
    for op in "xyzzyx":
        kernel, *args = operators[op]
        if kernel(m, *args) is None:
            return False
    return True


def new_valid_timestep(dtime, nread, cfl_outputstep, tref_max=1):
    """Largest dtime' < dtime dividing nread, cfl_outputstep and 2*tref."""
    for dt in range(dtime - 1, 0, -1):
        if nread % dt == 0 and cfl_outputstep % dt == 0 and dt % (2 * tref_max) == 0:
            return dt
    raise ValueError("new_valid_timestep: no valid time step")


def check_cfl(m, ndyn, n, nread, bt, tref, interpolate, grid=None, max_iteration=MAX_GLOBAL_ITERATION):
    """Check_CFL for one region.  `interpolate(t_offset_sec)` must return the meteo fluxes
    (pu, pv) at that offset from the start of the interval; the fluxes of step i are
    interpolated to the middle of [(i-1) ndyn, i ndyn].  Returns (ndyn, n, iterations)."""
    for iteration in range(1, max_iteration + 1):
        mt = m.copy()
        for i in range(n):
            pu, pv = interpolate((i + 0.5) * ndyn)
            fluxes = fluxes_from_meteo(pu, pv, bt, ndyn / (2.0 * tref))
            if not rehearse_step(mt, *fluxes, grid):
                break
        else:
            return ndyn, n, iteration
        ndyn_old = ndyn
        ndyn = new_valid_timestep(ndyn, nread, nread, tref)
        n = nint(float(n * ndyn_old) / float(ndyn))
    raise RuntimeError("check_cfl: exceeded maximum number of iterations")
