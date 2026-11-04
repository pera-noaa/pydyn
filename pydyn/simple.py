"""The TM5 dynamic core written for reading, not for bit-parity.

Same interfaces and layout as pydyn.kernels (see its docstring), same algorithm, but every
step is written in its simplest form and the fluxes are left untouched.  It agrees with
pydyn.kernels -- and therefore with the Fortran -- to a few ulp (tests/test_simple.py);
the differences are all rounding-level: one generic 1-D routine serves x, y and z,
sub-cycling divides the flux by n exactly, sums are numpy sums.  (One diagnostic differs:
stats['xi'] is the largest Courant number actually used, whereas dynamw's also counts the
failed trial sub-steps.)

The slopes scheme (Russell & Lerner 1981).  Inside a cell of air mass m the tracer mass rm
is distributed linearly: mixing ratio (rm + 2 s x) / m for x in [-1/2, 1/2], where s is the
slope along that axis (kg), so the two edges have mixing ratios (rm -+ s) / m.  Air flux a
through a face moves the fraction c = a / m_upwind of the upwind cell across it, with the
tracer and slopes that piece contains.  The slope is 6x the tracer's first moment about the
cell centre, and the slope update below is that moment budget written out: moment carried
by the pieces that left and arrived (pf), re-centred on the new cell.
"""

from itertools import product

import numpy as np

from .kernels import CFLError

__all__ = ["CFLError", "advect_line", "advect_x", "advect_y", "advect_z", "convec", "fluxes_from_meteo"]


def limit(s, rm):
    """Keep the edge mixing ratios (rm -+ s)/m non-negative: |s| <= |rm|."""
    return np.clip(s, -np.abs(rm), np.abs(rm))


# ------------------------------------------------------------------ 1-D slopes advection
def _upwind(pos, q):
    """q in the upwind cell of every interior face (cells along axis 0)."""
    return np.where(pos, q[:-1], q[1:])


def _substeps(m, flux, max_steps):
    """Smallest n such that flux/n keeps every face under |c| < 1 on all n mass sub-steps."""
    for n in range(1, max_steps + 1):
        a = flux / n
        mx = m.copy()
        for _ in range(n):
            c = a[1:-1] / _upwind(a[1:-1] >= 0, mx)
            if np.any(np.abs(c) >= 1):
                break
            mx = mx + a[:-1] - a[1:]
        else:
            return n
    raise CFLError(f"more than {max_steps} sub-steps needed")


def _substep(m, rm, s, t, u, a, limits, flat_ends):
    """One sub-step along axis 0.  m (n, ...); rm, s, t, u (n, ..., nt); a (n+1, ...) faces.

    s is the slope along the axis, t and u the two across.  flat_ends: the end cells carry no
    s or t (the polar caps of the y direction).  Everything is updated in place; returns
    the largest |c| seen.
    """
    inner = a[1:-1][..., None]  # interior faces, broadcasting over tracers
    pos = inner >= 0
    c = inner / _upwind(pos, m[..., None])

    s_src, t_src = s, t
    if flat_ends:
        s_src, t_src = s.copy(), t.copy()
        s_src[[0, -1]] = t_src[[0, -1]] = 0.0
        s_ends, t_ends = s[[0, -1]].copy(), t[[0, -1]].copy()

    # what the piece of the upwind cell swept through each face contains
    s_u = _upwind(pos, s_src)
    f = c * _upwind(pos, rm) + np.abs(c) * (1 - np.abs(c)) * s_u  # tracer mass
    pf = inner * (c * c * s_u - 3 * f)  # its first moment, for the slope budget
    ft = c * _upwind(pos, t_src)  # cross slopes are simply carried along
    fu = c * _upwind(pos, u)

    def closed(q):  # nothing crosses the two outer faces
        return np.pad(q, [(1, 1)] + [(0, 0)] * (q.ndim - 1))

    f, pf, ft, fu = map(closed, (f, pf, ft, fu))
    a_lo, a_hi = a[:-1][..., None], a[1:][..., None]
    m_old, m_new = m[..., None], (m + a[:-1] - a[1:])[..., None]

    rm += f[:-1] - f[1:]
    s[...] = (
        s
        + (pf[:-1] - pf[1:] - (a_lo - a_hi) * s + 3 * ((a_lo + a_hi) * rm - (f[:-1] + f[1:]) * m_old)) / m_new
    )
    if limits:
        s[...] = limit(s, rm)
    t += ft[:-1] - ft[1:]
    u += fu[:-1] - fu[1:]
    m[...] = m_new[..., 0]
    if flat_ends:
        s[[0, -1]], t[[0, -1]] = s_ends, t_ends
    return float(np.abs(c).max())


def advect_line(m, rm, s, t, u, flux, limits=True, max_steps=50, periodic=False, flat_ends=False):
    """Slopes advection along axis 0 with CFL sub-cycling.  In place; returns (n, max |c|).

    periodic: the line wraps (flux[0] == flux[-1] is the shared face).  Otherwise the outer
    faces must carry zero flux.
    """
    if limits:
        s[...] = limit(s, rm)

    def wrap(q, halo):  # one halo cell/face on each side for periodic lines
        return np.concatenate([q[-1 - halo : -halo or None], q, q[halo : halo + 1]]) if periodic else q

    n = _substeps(wrap(m, 0), wrap(flux, 1), max_steps)
    a = wrap(flux, 1) / n
    c_max = 0.0
    for _ in range(n):
        fields = [wrap(q, 0) for q in (m, rm, s, t, u)]
        c_max = max(c_max, _substep(*fields, a, limits, flat_ends))
        if periodic:
            for q, w in zip((m, rm, s, t, u), fields, strict=True):
                q[...] = w[1:-1]
    return n, c_max


# ------------------------------------------------------------------ the three directions
def _along(q, axis):
    return np.moveaxis(q, axis, 0)


def _record(stats, results):
    if stats is not None:
        stats["nloop_max"] = max(n for n, _ in results)
        stats["xi"] = max(c for _, c in results)


def advect_x(m, rm, rxm, rym, rzm, am, limits=True, max_nloop=50, stats=None):
    """Zonal: every (j, l) row is its own periodic line with its own sub-step count."""
    _, jm, lm = m.shape
    results = []
    for j, k in product(range(jm), range(lm)):
        row = (slice(None), j, k)
        line = (m[row], rm[row], rxm[row], rym[row], rzm[row], am[row])
        results.append(advect_line(*line, limits, max_nloop, periodic=True))
    _record(stats, results)


def advect_y(m, rm, rxm, rym, rzm, bm, limits=True, max_nloop=10, stats=None):
    """Meridional: every layer is one closed line from pole cell to pole cell."""
    lm = m.shape[2]
    results = []
    for k in range(lm):
        layer = (slice(None), slice(None), k)
        line = [_along(q[layer], 1) for q in (m, rm, rym, rxm, rzm, bm)]
        results.append(advect_line(*line, limits, max_nloop, flat_ends=True))
    _record(stats, results)


def advect_z(m, rm, rxm, rym, rzm, cm, limits=True, iter_limit=15, stats=None):
    """Vertical: the whole region is one closed line with a single sub-step count."""
    line = [_along(q, 2) for q in (m, rm, rzm, rxm, rym, cm)]
    _record(stats, [advect_line(*line, limits, iter_limit)])


# ------------------------------------------------------------------ dynam0
def fluxes_from_meteo(pu, pv, bt, dt):
    """Half-step air mass fluxes (am, bm, cm) from the meteo fluxes pu, pv (kg/s) and the
    hybrid coefficients bt (bt[0] = 1 at the surface, bt[lm] = 0).

    The vertical flux is diagnosed, not read: the horizontal convergence into a column
    changes its surface pressure, and each layer must take the share (bt[l] - bt[l+1]) of
    that change; whatever converges above interface l beyond that share flows down through it.
    """
    lm = pu.shape[2]
    conv = pu[:-1] - pu[1:] + pv[:, :-1] - pv[:, 1:]  # convergence into each cell
    above = np.cumsum(conv[..., ::-1], axis=-1)[..., ::-1]  # summed from the top down to l
    column = above[..., :1]
    cm = np.zeros((*conv.shape[:2], lm + 1))
    cm[..., 1:lm] = -dt * (above[..., 1:] - bt[1:lm] * column)
    return dt * pu, dt * pv, cm


# ------------------------------------------------------------------ convection
def column_matrix(m, entu, detu, entd, detd, top, lfs, dkg, dt):
    """Implicit vertical-mixing matrix A of one column: rm_after = inv(A) @ rm_before.

    f[k, j] is the fraction of layer j's tracer that passes upward through interface k
    (bottom of layer k) per second; A = I - dt * (f[k] - f[k+1]) is the implicit budget.
    m, entu, detu, entd, detd: per layer (kg, kg/s); dkg: per interface, dkg[k] for the
    interface below layer k+1; top, lfs: number of layers the updraft / downdraft spans.
    """
    lmax = m.size
    fu, fd = np.zeros((2, lmax + 1, lmax))
    amu, amd = np.zeros((2, lmax + 1))

    # updraft: air amu[k] and tracer fractions fu[k] rise through interface k, entraining
    # part of each layer on the way and keeping the fraction that is not detrained
    for i in range(top):
        inflow = amu[i] + entu[i]
        amu[i + 1] = max(inflow - detu[i], 0.0)
        keep = amu[i + 1] / inflow if inflow > 0 else 0.0
        fu[i + 1] = fu[i] * keep
        fu[i + 1, i] = entu[i] / m[i] * keep
    # downdraft: the same descending from the level of free sinking, fluxes negative
    for i in range(lfs - 1, 0, -1):
        inflow = amd[i + 1] - entd[i]
        amd[i] = min(inflow + detd[i], 0.0)
        keep = amd[i] / inflow if inflow < 0 else 0.0
        fd[i] = fd[i + 1] * keep
        fd[i, i] = -entd[i] / m[i] * keep

    f = fu + fd
    f[[0, -1]] = 0.0  # nothing crosses the surface or the model top (even if top == lmax)
    # compensating subsidence of the environment and turbulent diffusion, interior interfaces
    for k in range(1, lmax):
        f[k, k] -= amu[k] / m[k]
        f[k, k - 1] -= amd[k] / m[k - 1]
        f[k, k - 1] += dkg[k - 1] / m[k - 1]
        f[k, k] -= dkg[k - 1] / m[k]
    return np.eye(lmax) - dt * (f[:-1] - f[1:])


def convec(m, rm, rxm, rym, rzm, entu, detu, entd, detd, cloud_top, cloud_lfs, dkg, dt, lmax, inverse=None):
    """Implicit convection + diffusion, column by column.  In place.  Slopes are damped by
    the diagonal of the inverse only (`inverse` is accepted for interface compatibility)."""
    im, jm, _ = m.shape
    for i, j in product(range(im), range(jm)):
        col = (i, j, slice(0, lmax))
        A = column_matrix(
            m[col], entu[col], detu[col], entd[col], detd[col], cloud_top[i, j], cloud_lfs[i, j], dkg[col], dt
        )
        Ainv = np.linalg.inv(A)
        rm[col] = Ainv @ rm[col]
        for s in (rxm, rym, rzm):
            s[col] *= np.diag(Ainv)[:, None]
