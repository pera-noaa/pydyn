"""Dynamic-core kernels of TM5-zoom, single region, slopes advection, serial.

Each routine is a transcription of one Fortran kernel, keeping the order of floating-point
operations so that a comparison against the compiled Fortran can be made bit-for-bit
(see docs/state.md for what can and cannot be bit-identical).

Layout (0-based, no halos, C-contiguous NumPy):
    m               (im, jm, lm)          air mass per cell, kg
    rm, rxm, rym, rzm (im, jm, lm, nt)    tracer mass and its three slopes, kg
    am              (im+1, jm, lm)        air mass through x-faces per half-step, kg;
                                          am[k] flows from cell k-1 into cell k; am[0] == am[im]
    bm              (im, jm+1, lm)        y-faces, bm[:, 0] == bm[:, jm] == 0 (poles)
    cm              (im, jm, lm+1)        z-faces, cm[..., 0] == cm[..., lm] == 0

Fortran ↔ NumPy index map: am_t(i,j,l) = am[i, j-1, l-1]; bm_t(i,j,l) = bm[i-1, j-1, l-1];
cm_t(i,j,l) = cm[i-1, j-1, l]; cells (i,j,l) -> [i-1, j-1, l-1].

Fluxes are modified IN PLACE by the CFL subcycling exactly as the Fortran does: dynamu and
dynamv scale the flux row/layer by nloop/(nloop+1) per retry and multiply by nloop at the
end (not a bit-exact restore); dynamw divides a saved copy and restores it exactly.

The three advection kernels are the same 1-D slopes scheme applied along x, y and z.  They
are written on views with the advected axis first (`_along`), so "cells" are axis 0 of
length n, "faces" axis 0 of length n+1, and `q[:-1]` / `q[1:]` are the lower / upper
neighbours everywhere.  What differs between them is real: the Courant number is taken per
row (x), per layer (y) or per region (z); x is periodic, y has slope-free pole caps; and the
production overlays evaluate a few statements in different orders (marked "production").
"""

import numpy as np

from .ops import (
    ONE,
    advance,
    closed,
    convergence,
    courant,
    fresh,
    progressive,
    seqsum,
    subcycles,
    upwind,
    wrap,
)  # fmt: skip


class CFLError(RuntimeError):
    """Raised where the Fortran calls escape_tm: the subcycle limit was reached."""


# ------------------------------------------------------------------ shared pieces
def _limit(s, rm):
    """The slope limiter: max(min(s, abs(rm)), -abs(rm))."""
    return np.clip(s, -np.abs(rm), np.abs(rm))


def _along(q, axis):
    """View of q with the advected axis first; writes through to q."""
    return np.moveaxis(q, axis, 0)


def _slopes_fluxes(flux, c, rm_u, s_u, t_u, w_u):
    """Tracer fluxes through the interior faces of the slopes scheme.

    All inputs per face: the flux, its Courant number c, and the upwind-cell tracer mass
    rm_u, slope along the flow s_u and the two cross slopes t_u, w_u.  Returns f (tracer
    mass), pf (its first moment, drives the along-slope update) and the advected cross slopes.
    """
    f = np.where(flux >= 0.0, c * (rm_u + (1.0 - c) * s_u), c * (rm_u - (1.0 + c) * s_u))
    pf = flux * (c * c * s_u - 3.0 * f)
    return f, pf, c * t_u, c * w_u


def _along_slope_update(s, pf, flux, rm_new, f, m_old, m_new):
    """The slope along the flow after one sub-step (base form, used by dynamu and dynamw)."""
    lo, hi = flux[:-1], flux[1:]
    return (
        s + (pf[:-1] - pf[1:] - (lo - hi) * s + 3.0 * ((lo + hi) * rm_new - (f[:-1] + f[1:]) * m_old)) / m_new
    )


def _record(stats, nloop_max, xi):
    if stats is not None:
        stats["nloop_max"] = nloop_max
        stats["xi"] = xi


# ------------------------------------------------------------------ dynam0 (advect.F90)
def fluxes_from_meteo(pu, pv, bt, dt):
    """Per-half-step mass fluxes from the meteo mass fluxes (advect.F90 dynam0).

    pu (im+1, jm, lm) and pv (im, jm+1, lm) are the balanced horizontal mass fluxes in
    kg/s, face convention as in the module docstring; bt (lm+1,) are the hybrid b
    coefficients at the interfaces, bt[0] = 1 (surface) .. bt[lm] = 0; dt is the
    half-step length in seconds (ndyn/(2 tref)).

    Returns (am, bm, cm).  cm is re-derived from the horizontal convergence and the
    hybrid distribution of the column tendency; the meteo vertical flux is not used.
    """
    im, jm, lm = pu.shape[0] - 1, pu.shape[1], pu.shape[2]
    if pv.shape != (im, jm + 1, lm) or bt.shape != (lm + 1,):
        raise ValueError("fluxes_from_meteo: inconsistent shapes")
    conv = convergence(pu, pv)  # conv_adv
    pit = seqsum(conv, axis=2)  # column total, accumulated bottom-up
    # sd(l), l = lm-1 .. 1 (Fortran indices kept: sd[l] == sd_F(l), sd[0] unused)
    sd = np.zeros((im, jm, lm))
    sd[..., lm - 1] = conv[..., lm - 1] - (bt[lm - 1] - bt[lm]) * pit
    for lev in range(lm - 2, 0, -1):
        sd[..., lev] = sd[..., lev + 1] + conv[..., lev] - (bt[lev] - bt[lev + 1]) * pit
    cm = np.zeros((im, jm, lm + 1))
    cm[..., 1:lm] = (-dt) * sd[..., 1:lm]
    return dt * pu, dt * pv, cm


# ------------------------------------------------------------------ dynamu (advectx__slopes.F90)
def advect_x(m, rm, rxm, rym, rzm, am, limits=True, max_nloop=50, stats=None):
    """Zonal slopes advection, periodic in x, per-(j,l)-row CFL subcycling.  In place.

    stats (optional dict) receives 'nloop_max' and 'xi' (max Courant number) like the
    Fortran diagnostics nloop_max(region,1) / xi(region,1).
    """
    _, jm, _ = m.shape
    nloop_max, xi = 1, 0.0
    for j in range(jm):  # every level of a latitude at once: the rows are independent
        nloop, alpha_max = _advect_rows_x(m[:, j], rm[:, j], rxm[:, j], rym[:, j], rzm[:, j], am[:, j],
                                          limits, max_nloop)  # fmt: skip
        nloop_max, xi = max(nloop_max, int(nloop.max())), max(xi, float(alpha_max.max()))
    _record(stats, nloop_max, xi)


def _advect_rows_x(m, rm, rxm, rym, rzm, a, limits, max_nloop):
    """dynamu on R periodic rows of n cells at once, each with its own CFL subcycling, in place:
    m (n, R), rm..rzm (n, R, nt), a (n+1, R) with a[0] == a[n].  Every statement is the
    single row's, applied column by column (no reduction runs across rows), so each row's
    result is the one it gets alone.  Returns each row's (nloop, xi): the sub-cycle count and
    the largest Courant number of its accepted trial."""
    if limits:
        rxm[...] = _limit(rxm, rm)
    found = subcycle_rows(m, a, max_nloop - 1)
    if found is None:
        raise CFLError("dynamu: nloop too high")
    nloop, alpha_max = found
    for n in np.unique(nloop):
        rows = np.flatnonzero(nloop == n)
        if rows.size == m.shape[1]:  # every row alike: on the views themselves
            _substeps_x(m, rm, rxm, rym, rzm, a, int(n), limits)
            continue
        parts = [q[:, rows] for q in (m, rm, rxm, rym, rzm)]
        _substeps_x(*parts, a[:, rows], int(n), limits)
        for q, part in zip((m, rm, rxm, rym, rzm), parts, strict=True):
            q[:, rows] = part
    a[...] = a * nloop  # Fortran: "restore old am"
    return nloop, alpha_max


def subcycle_rows(m, a, nmax, periodic=True):
    """`ops.subcycles(..., progressive)` of R x rows at once, m (n, R): the smallest nloop for
    which nloop trial sub-steps of each row's mass stay under CFL, the row's faces `a` scaled in
    place by (nloop-1)/nloop per retry; and the largest |Courant| of its accepted trial.  A
    periodic row has n+1 faces and wraps; a child's row (`periodic=False`) n-1 interior faces
    and its end cells held.  None once a row would need more than nmax.  Every row still
    searching has tried the same nloop, so they retry together; each row's arithmetic is the
    one it gets alone."""
    rows = m.shape[1]
    nloop, alpha_max = np.ones(rows, dtype=int), np.zeros(rows)
    searching, n = np.arange(rows), 1
    while True:
        mx, f = m[:, searching].copy(), a[:, searching]
        alive, amax = np.ones(searching.size, dtype=bool), np.zeros(searching.size)
        for _ in range(n):
            c = np.abs(courant(wrap(mx) if periodic else mx, f))
            ok = np.all(c < ONE, axis=0)
            amax = np.where(alive & ok, np.maximum(amax, c.max(axis=0)), np.where(alive, 0.0, amax))
            alive &= ok
            if periodic:
                mx = advance(mx, f)
            else:
                mx[1:-1] = advance(mx[1:-1], f)
        nloop[searching], alpha_max[searching] = n, amax
        searching = searching[~alive]
        if searching.size == 0:
            return nloop, alpha_max
        n += 1
        a[:, searching] = progressive(a[:, searching], n, None)
        if n > nmax:
            return None


def _substeps_x(m, rm, rxm, rym, rzm, a, nloop, limits):
    """nloop sub-steps of the rows (n, R): the single-row dynamu body, columns alongside."""
    af = a[..., None]
    for _ in range(nloop):
        mnew = advance(m, a)
        # production dynamu (proj/ct-unified/trunk): m_inv(i) = one/m(i), alpha = am*m_inv(iup)
        alpha = courant(wrap(m)[..., None], af, reciprocal=True)
        f, pf, fy, fz = _slopes_fluxes(af, alpha, *(upwind(af, wrap(q)) for q in (rm, rxm, rym, rzm)))
        for q in (f, pf, fy, fz):  # periodic seam: the west face of cell 1 is the east face of cell im
            q[0] = q[-1]

        rm[...] = rm - np.diff(f, axis=0)  # advectx: rm + (f(i-1) - f(i))
        rxm[...] = _along_slope_update(rxm, pf, af, rm, f, m[..., None], mnew[..., None])
        if limits:
            rxm[...] = _limit(rxm, rm)
        rym[...] = rym - np.diff(fy, axis=0)
        rzm[...] = rzm - np.diff(fz, axis=0)
        m[...] = mnew


# ------------------------------------------------------------------ dynamv (advecty__slopes.F90)
def advect_y(m, rm, rxm, rym, rzm, bm, limits=True, max_nloop=10, stats=None, nloop_layers=None):
    """Meridional slopes advection with polar caps, per-layer CFL subcycling.  In place.

    Rows 0 and jm-1 are the pole cells: the flux leaving a pole cell carries no slope
    (f = beta*rm, pf = -3*bm*f, fx = 0), and the pole cells' rxm/rym are never updated.
    nloop_layers (optional list) receives the nloop of every layer: dynamv scales and
    restores its whole bm(:,:,l), halo faces included, and zoom.advect_y_zoom repeats that on
    the faces this layout does not hold.
    """
    _, jm, lm = m.shape
    if np.any(bm[:, 0] != 0.0) or np.any(bm[:, jm] != 0.0):
        raise ValueError("advect_y: non-zero flux through the polar outer faces")
    nloop_max, xi = 1, 0.0
    for k in range(lm):
        layer = (slice(None), slice(None), k)
        args = [_along(q[layer], 1) for q in (m, rm, rxm, rym, rzm, bm)]
        nloop, beta_max = _advect_layer_y(*args, limits, max_nloop)
        nloop_max, xi = max(nloop_max, nloop), max(xi, beta_max)
        if nloop_layers is not None:
            nloop_layers.append(nloop)
    _record(stats, nloop_max, xi)


def _advect_layer_y(m, rm, rxm, rym, rzm, b, limits, max_nloop):
    """One layer, y axis first: m (jm, im), rm.. (jm, im, nt), b (jm+1, im)."""
    if limits:
        rym[...] = _limit(rym, rm)  # Fortran: j = js-1 .. je+1, i.e. every row

    found = subcycles(m, b, max_nloop - 1, progressive)  # the whole layer shares one nloop
    if found is None:
        raise CFLError("dynamv: nloop too high")
    nloop, _ = found

    beta_max = 0.0
    bf = b[1:-1, :, None]
    for _ in range(nloop):
        mnew = advance(m, b)
        beta = courant(m[:, :, None], bf)
        beta_max = max(beta_max, float(np.max(np.abs(beta))))
        f, pf, fx, fz = _slopes_fluxes(bf, beta, *(upwind(bf, q) for q in (rm, rym, rxm, rzm)))
        # polar faces: flow OUT of a pole cell is slope-free (Fortran faces 2 and jmr)
        for face, cell, leaving in ((0, 0, bf[0] >= 0.0), (-1, -1, bf[-1] < 0.0)):
            f[face] = np.where(leaving, beta[face] * rm[cell], f[face])
            pf[face] = np.where(leaving, (-3.0 * bf[face]) * f[face], pf[face])
            fx[face] = np.where(leaving, 0.0, fx[face])
            fz[face] = np.where(leaving, beta[face] * rzm[cell], fz[face])
        F, PF, FX, FZ = (closed(q) for q in (f, pf, fx, fz))

        rym_pole, rxm_pole = rym[[0, -1]].copy(), rxm[[0, -1]].copy()
        # production dynamv (proj/ct-unified/trunk): per-cell factors hoisted out of the
        # tracer loop, reciprocal multiply instead of division
        inv_mnew = (1.0 / mnew)[:, :, None]
        flow_conv = (b[:-1] - b[1:])[:, :, None]
        flow_mean = (3.0 * (b[:-1] + b[1:]))[:, :, None]
        mass_x3 = (3.0 * m)[:, :, None]
        rm[...] = rm - np.diff(F, axis=0)  # advecty: rm + (f(j-1) - f(j))
        rym[...] = (
            rym
            + ((PF[:-1] - PF[1:]) - flow_conv * rym + flow_mean * rm - mass_x3 * (F[:-1] + F[1:])) * inv_mnew
        )
        if limits:
            rym[...] = _limit(rym, rm)
        rxm[...] = rxm - np.diff(FX, axis=0)
        rzm[...] = rzm - np.diff(FZ, axis=0)
        rym[[0, -1]], rxm[[0, -1]] = rym_pole, rxm_pole  # Fortran updates only rm, rzm at the poles
        m[...] = mnew

    b[...] = b * nloop
    return nloop, beta_max


# ------------------------------------------------------------------ dynamw (advectz__slopes.F90)
def advect_z(m, rm, rxm, rym, rzm, cm, limits=True, iter_limit=15, stats=None):
    """Vertical slopes advection, one CFL iteration count for the whole region.  In place.

    cm is divided by n_advz for the update and restored exactly afterwards.
    """
    lm = m.shape[2]
    if np.any(cm[..., 0] != 0.0) or np.any(cm[..., lm] != 0.0):
        raise ValueError("dynamw: non-zero halo interfaces in cm")
    if limits:
        rzm[...] = _limit(rzm, rm)

    gamma_max = 0.0

    def seen(gamma, ok):  # xi: every trial sub-step counts, failed ones included
        nonlocal gamma_max
        if not ok:  # the Fortran exits at the first violation in its loop order (i, j, l),
            gamma = gamma.ravel(order="F")  # so xi has only seen the values up to it
            gamma = gamma[: np.argmax(gamma >= 1.0) + 1]
        gamma_max = max(gamma_max, float(gamma.max()))

    found = subcycles(m, cm, iter_limit, fresh, axis=2, seen=seen)
    if found is None:
        raise CFLError("dynamw: too many CFL violations!")
    n_advz, _ = found

    advect_z_substeps(m, rm, rxm, rym, rzm, cm, n_advz, limits)
    _record(stats, n_advz, gamma_max)


def advect_z_substeps(m, rm, rxm, rym, rzm, cm, n_advz, limits=True):
    """dynamw after its CFL search: n_advz sub-steps with cm / n_advz, cm restored exactly.
    In place, on whatever (i, j) scope the arrays cover -- the whole region for region 1,
    the child's interior scope in zoom.advect_z."""
    cmold = cm.copy()
    cm[...] = cmold / float(n_advz)
    m, rm, rxm, rym, rzm, c = (_along(q, 2) for q in (m, rm, rxm, rym, rzm, cm))
    cf = c[1:-1, ..., None]
    for _ in range(n_advz):
        mnew = advance(m, c)
        gamma = courant(m[..., None], cf)
        f, pf, fx, fy = _slopes_fluxes(cf, gamma, *(upwind(cf, q) for q in (rm, rzm, rxm, rym)))
        F, PF, FX, FY = (closed(q) for q in (f, pf, fx, fy))

        rm[...] = advance(rm, F)  # advectz (branches/speedup): rm + f(l-1) - f(l), no parentheses
        rzm[...] = _along_slope_update(rzm, PF, c[..., None], rm, F, m[..., None], mnew[..., None])
        if limits:
            rzm[...] = _limit(rzm, rm)
        rxm[...] = advance(rxm, FX)
        rym[...] = advance(rym, FY)
        m[...] = mnew

    cm[...] = cmold


# ------------------------------------------------------------------ convec (convection.F90)
def convection_matrix(m, entu, detu, entd, detd, cloud_top, cloud_lfs, dkg, dt, lmax):
    """Implicit convection + diffusion matrices for a batch of columns, before inversion.

    m, entu, detu, entd, detd, dkg: (ncol, >= lmax); cloud_top, cloud_lfs: (ncol,) integer
    1-based level indices as stored in conv_dat (0 = no cloud).  Returns conv1 (ncol, lmax, lmax)
    such that rm_new = inv(conv1) @ rm_old over levels 0..lmax-1.

    fu/fd[:, k, :] hold, for interface k (top of layer k, 1-based), the fraction of each
    layer's tracer carried through it by the up-/downdraft; amu/amd the air mass flux.
    """
    n = m.shape[0]
    fu = np.zeros((n, lmax + 1, lmax))
    fd = np.zeros((n, lmax + 1, lmax))
    amu = np.zeros((n, lmax + 1))
    amd = np.zeros((n, lmax + 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        # updraft, k = 1 .. cloud_top: zxi = fraction of the plume that is not detrained here
        for k in range(1, lmax + 1):
            act = cloud_top >= k
            a = amu[:, k - 1] + entu[:, k - 1] - detu[:, k - 1]
            up = a > 0.0
            zxi = np.where(up, np.maximum(0.0, 1.0 - detu[:, k - 1] / (amu[:, k - 1] + entu[:, k - 1])), 0.0)
            amu[:, k] = np.where(act & up, a, 0.0)
            fu[:, k, : k - 1] = np.where(act[:, None], fu[:, k - 1, : k - 1] * zxi[:, None], 0.0)
            fu[:, k, k - 1] = np.where(act, entu[:, k - 1] / m[:, k - 1] * zxi, 0.0)
        # downdraft, k = cloud_lfs .. 2 (amd negative downward)
        for k in range(lmax, 1, -1):
            act = cloud_lfs >= k
            a = amd[:, k] - entd[:, k - 1] + detd[:, k - 1]
            dn = a < 0.0
            zxi = np.where(dn, np.maximum(0.0, 1.0 + detd[:, k - 1] / (amd[:, k] - entd[:, k - 1])), 0.0)
            amd[:, k - 1] = np.where(act & dn, a, 0.0)
            fd[:, k - 1, k:] = np.where(act[:, None], fd[:, k, k:] * zxi[:, None], fd[:, k - 1, k:])
            fd[:, k - 1, k - 1] = np.where(act, -entd[:, k - 1] / m[:, k - 1] * zxi, fd[:, k - 1, k - 1])
    # combine, compensating subsidence, diffusion for interfaces k = 1 .. lmax-1
    f = fd
    for k in range(1, lmax):
        f[:, k, :] = fu[:, k, :] + f[:, k, :]
        f[:, k, k] = f[:, k, k] - amu[:, k] / m[:, k]  # f(k,k+1) -= amu(k)/m(k+1)
        f[:, k, k - 1] = f[:, k, k - 1] - amd[:, k] / m[:, k - 1]  # f(k,k)   -= amd(k)/m(k)
        f[:, k, k - 1] = f[:, k, k - 1] + dkg[:, k - 1] / m[:, k - 1]  # f(k,k)   += dkg(k)/m(k)
        f[:, k, k] = f[:, k, k] - dkg[:, k - 1] / m[:, k]  # f(k,k+1) -= dkg(k)/m(k+1)
    conv1 = np.empty((n, lmax, lmax))
    for k in range(1, lmax + 1):
        conv1[:, k - 1, :] = -dt * (f[:, k - 1, :] - f[:, k, :])
        conv1[:, k - 1, k - 1] = conv1[:, k - 1, k - 1] + 1.0
    return conv1


def fastminv(a):
    """Batched inverse by Crout LU without pivoting, loop order as convection.F90 fastminv
    (the build without with_lapack).  a: (ncol, n, n)."""
    ncol, n, _ = a.shape
    lu = a.copy()
    for j in range(n):
        for k in range(j):
            lu[:, k + 1 :, j] = lu[:, k + 1 :, j] - lu[:, k + 1 :, k] * lu[:, k, j][:, None]
        lu[:, j + 1 :, j] = lu[:, j + 1 :, j] / lu[:, j, j][:, None]
    # forward substitution, all unit right-hand sides at once (columns of y)
    y = np.broadcast_to(np.eye(n), (ncol, n, n)).copy()
    for i in range(n):
        for j in range(i):
            y[:, i, :] = y[:, i, :] - lu[:, i, j][:, None] * y[:, j, :]
    # back substitution
    inv = np.empty_like(y)
    for i in range(n - 1, -1, -1):
        inv[:, i, :] = y[:, i, :]
        for j in range(i + 1, n):
            inv[:, i, :] = inv[:, i, :] - lu[:, i, j][:, None] * inv[:, j, :]
        inv[:, i, :] = inv[:, i, :] / lu[:, i, i][:, None]
    return inv


def convec(
    m, rm, rxm, rym, rzm, entu, detu, entd, detd, cloud_top, cloud_lfs, dkg, dt, lmax, inverse="crout"
):
    """Vertical mixing ('v' operator): implicit convection + diffusion per column.  In place.

    entu, detu, entd, detd, dkg: (im, jm, lm); cloud_top, cloud_lfs: (im, jm) int;
    dt = nconv/(2 tref).  Slopes are scaled by the diagonal of the inverse only.
    inverse = 'crout' reproduces the non-LAPACK fastminv; 'numpy' uses numpy.linalg.inv
    (a few ulp from the with_lapack build); a callable `inv(a)` on (ncol, n, n) is used as is
    (lapack.LapackInverse reproduces the with_lapack build bitwise).
    """
    im, jm, _ = m.shape

    def columns(q):  # (im, jm, ...) -> (ncol, ...), a view
        return q.reshape(im * jm, *q.shape[2:])

    meteo = (columns(q) for q in (entu, detu, entd, detd, cloud_top, cloud_lfs, dkg))
    conv1 = convection_matrix(columns(m)[:, :lmax], *meteo, dt, lmax)
    if inverse == "crout":
        C = fastminv(conv1)
    elif inverse in ("numpy", "lapack"):
        C = np.linalg.inv(conv1)
    else:
        C = inverse(conv1)

    diag = np.einsum("nii->ni", C)[:, :, None]
    for s in (rxm, rym, rzm):
        sc = columns(s)
        sc[:, :lmax] = diag * sc[:, :lmax]
    rmc = columns(rm)
    srm = rmc[:, :lmax].copy()
    zrm = np.zeros_like(srm)
    for k in range(lmax):  # zrm(l) = zrm(l) + conv1(l,k)*srm(k), accumulated in k
        zrm = zrm + C[:, :, k, None] * srm[:, k, None, :]
    rmc[:, :lmax] = zrm
