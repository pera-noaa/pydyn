"""Calc_Kzz: the vertical diffusion coefficients `dkg` (kg/s across each level interface) of
one dynamic step, from the surface fields and the meteo of the step (diffusion.F90 calc_kzz
with no dkg cache; diffusion_kernels.F90 dd_calc_ustar, bldiff; meteo.F90
Meteo_Produce_Gph).  Four stages:

    ustar         dd_calc_ustar: the friction velocity of every cell, land and sea blended
                  by the land fraction
    geopotential  Meteo_Produce_Gph: the height of the level interfaces above the
                  orography, hydrostatic in the virtual temperature
    bldiff        Holtslag-Boville / Vogelezang-Holtslag: the free-troposphere diffusivity
                  from the local Richardson number, the boundary-layer height from the bulk
                  Richardson number (two passes, the second with the convective excess),
                  the boundary-layer diffusivity with its Prandtl number and the entrainment
                  level; everything at the level interfaces l+1 for l = 1..lm-1
    dkg           kvh turned into a mass exchange rate, zero at lmax_conv

`calc_kzz` strings them together with the step's inputs: `phlb_t`, `m_t` from the surface
pressure at the middle of the step (spm), the step-interpolated pu / pv, T and Q.

The Fortran is branchless: every `if` is a 0/1 mask `jqif(x, y)` (1 where x >= y) that
multiplies both branches -- kept as such, with NumPy's own arithmetic on 0.0 / 1.0, so a
NaN or inf in the unused branch propagates as it does there.  The real powers `**2.` are
`x*x` (gfortran folds an integral real exponent), `**3` is `libm.cube`, the third roots go
through libm's pow, exp / log / cos through libm too (`ops.libm`); products are written in
the Fortran's left-to-right order.
"""

import numpy as np

from .balance import GRAV, PI
from .ops import libm

__all__ = ["bldiff", "calc_kzz", "dkg", "geopotential", "jqif", "mass_t", "ustar"]

# Binas
AE = 6.371e6
GTOR = PI / 180.0
VKARMAN = 0.4
CP_AIR = 1004.0
LVAP = 2.5e6
RGAS = 8.3144
XMAIR = 28.94

# dd_calc_ustar
ALFA_GARRET = 0.11
RHO_AIR = 1.2
ALFA_CHARNOCK = 0.018
NU_AIR = 1.5e-5

# bldiff
CKMIN = 1.0e-15
SFFRAC = 0.1
ONET = 1.0 / 3.0
RAIR = RGAS * 1.0e3 / XMAIR
EPSILO = RAIR / 461.51
APZERO = 100000.0
CCON = SFFRAC * VKARMAN
BETAH = 15.0
BINM = 15.0 * SFFRAC
FAK = 8.5
FAKN = 7.2
ZKAPPA = RAIR / CP_AIR
ZCRDQ = 1.0 / EPSILO - 1.0
TINY = 1.0e-9
ZACB = 100.0
RICR = 0.3
SSFRAC = 0.1
KVHMIN = 0.1


def jqif(x, y):
    """nint(0.5 + sign(0.5, x - y)): 1.0 where x - y is +0.0 or positive, else 0.0."""
    return np.where(np.copysign(1.0, x - y) > 0.0, 1.0, 0.0)


def ustar(lsm, sr, srols, u10m, v10m, ewss, nsss):
    """dd_calc_ustar: ustar_loc (im, jm) from the land fraction lsm (%), the ECMWF and the
    Olsson roughness lengths, the 10-m wind and the surface stresses."""
    xland = lsm / 100.0
    tau = np.sqrt(ewss * ewss + nsss * nsss)
    ustar_sea = np.maximum(0.001, np.sqrt(tau / RHO_AIR))
    wind10m = np.sqrt(u10m * u10m + v10m * v10m)
    with np.errstate(all="ignore"):
        sr_land = np.maximum(1e-2, srols)
        sr_help = np.minimum(sr, 0.03)
        land = VKARMAN * wind10m / libm.log(10.0 / sr_help) * libm.log(75.0 / sr_help)
        land = land / libm.log(10.0 / sr_land)
    ustar_land = np.where(xland > 0.0, land, 0.0)
    return xland * ustar_land + (1.0 - xland) * ustar_sea


def geopotential(oro, sp, t, q, at, bt):
    """Meteo_Produce_Gph: gph (im, jm, lm+1), the interface heights in m: the orography
    (oro in m^2/s^2) at the surface, hydrostatic steps up to lm, 200 km at the top."""
    im, jm, lm = t.shape
    gph = np.empty((im, jm, lm + 1))
    gph[..., 0] = oro / GRAV
    for k in range(lm - 1):
        tv = t[..., k] * (1.0 + 0.608 * q[..., k])
        pdown = at[k] + bt[k] * sp
        pup = at[k + 1] + bt[k + 1] * sp
        gph[..., k + 1] = gph[..., k] + tv * 287.05 * libm.log(pdown / pup) / GRAV
    gph[..., lm] = 200000.0
    return gph


def mass_t(phlb, dxyp):
    """calc_kzz's m_t: the air mass (im, jm, lm) under the half-level pressures phlb, with
    the region's dxyp -- (dp * dxyp) / g, not Pressure_to_Mass's order."""
    return (phlb[..., :-1] - phlb[..., 1:]) * dxyp[None, :, None] / GRAV


def _winds(pu, pv, m, jm, dx, dy, ybeg):
    """uwind, vwind (im, jm, lm) in m/s from the mass fluxes through the cell faces."""
    lat = ybeg + 0.5 * dy + dy * np.arange(jm)
    dxx = AE * dx * GTOR * libm.cos(lat * GTOR)
    dyy = AE * dy * GTOR
    uwind = dxx[None, :, None] * (pu[1:] + pu[:-1]) * 0.5 / m
    vwind = dyy * (pv[:, 1:] + pv[:, :-1]) * 0.5 / m
    return uwind, vwind


def _pbl_pass(pblh, whm, uwind, vwind, wthm, q, ustr, wtseff, reference, first):
    """One Richardson-number sweep up the columns: pblh where the bulk Richardson number
    against `reference` (wtseff, or the parcel temperature) first reaches RICR, interpolated
    linearly; whm(lm) where it never does."""
    lm = whm.shape[-1]
    jwork, wrino = np.ones_like(pblh), np.zeros_like(pblh)
    with np.errstate(all="ignore"):
        for k in range(0 if first else 1, lm):
            du, dv = uwind[..., k] - uwind[..., 0], vwind[..., k] - vwind[..., 0]
            vvk = np.maximum(du * du + dv * dv + ZACB * ustr * ustr, TINY)
            tkv = wthm[..., k] * (1.0 + ZCRDQ * q[..., k])
            dz = whm[..., k] - whm[..., 0]
            if first:
                zrino = GRAV * (tkv - wtseff) * dz / (wtseff * vvk)
            else:
                zrino = GRAV * (tkv - reference) * dz / (vvk * wtseff)
            zrino = zrino + np.copysign(1.0e-10, zrino)
            zrino = jwork * zrino
            jq = jqif(RICR, zrino)
            frac = (RICR - wrino) / (zrino - wrino)
            if k == 0:
                at_ricr = frac * whm[..., k]
            else:
                at_ricr = whm[..., k - 1] + frac * (whm[..., k] - whm[..., k - 1])
            pblh = jq * pblh + (1.0 - jq) * at_ricr
            jwork = jq * jwork
            wrino = zrino + (1.0 - jwork) * 0.1
    if first:
        return (1.0 - jwork) * pblh + whm[..., lm - 1] * jwork
    return (1.0 - jwork) * pblh + jwork * whm[..., lm - 1]


def bldiff(phlb, gph, t, q, m, pu, pv, ustar_loc, sshf, slhf, dx, dy, ybeg):
    """bldiff: the diffusivity kvh (im, jm, lm-1) at the interfaces l+1 and the boundary-layer
    height pblh (im, jm).  phlb (im, jm, lm+1), gph (im, jm, lm+1), t, q, m (im, jm, lm),
    pu (im+1, jm, lm), pv (im, jm+1, lm), the rest (im, jm); dx, dy, ybeg the region's."""
    im, jm, lm = t.shape
    lo, hi = phlb[..., :-1], phlb[..., 1:]
    z0 = gph[..., :1]

    # potential temperatures, layer-centre heights
    pf = (lo + hi) * 0.5
    wthm = t * libm.pow(APZERO / pf, ZKAPPA)
    zthvk = wthm * (1.0 + ZCRDQ * q)
    intfac = (lo - pf) / (lo - hi)
    whm = (gph[..., :-1] - z0) * (1.0 - intfac) + (gph[..., 1:] - z0) * intfac

    # surface fluxes in kinematic units, friction velocity
    dz1, dp1 = gph[..., 1] - gph[..., 0], phlb[..., 0] - phlb[..., 1]
    wheat = -sshf * dz1 * GRAV / dp1 / CP_AIR
    wqflx = -slhf * dz1 * GRAV / dp1 / LVAP
    ustr = np.maximum(ustar_loc, 0.01)

    uwind, vwind = _winds(pu, pv, m, jm, dx, dy, ybeg)

    # free-troposphere diffusivity from the local Richardson number, interfaces l+1
    with np.errstate(all="ignore"):
        z = gph[..., 1:lm] - z0
        jq = jqif(1000.0, z)
        zlambdac = jq * 300.0 + (1.0 - jq) * (30.0 + 270.0 * libm.exp(1.0 - z / 1000.0))
        cml2 = 1.0 / (1.0 / zlambdac + 1.0 / (VKARMAN * z))
        cml2 = cml2 * cml2
        du, dv = np.diff(uwind, axis=-1), np.diff(vwind, axis=-1)
        zdup2 = np.maximum(du * du + dv * dv, 1.0e-10)
        zdz = np.diff(whm, axis=-1)
        zdup2 = zdup2 / (zdz * zdz)
        log_pf = libm.log(pf)
        arg = (libm.log(phlb[..., 1:lm]) - log_pf[..., :-1]) / (log_pf[..., 1:] - log_pf[..., :-1])
        dthv = np.diff(zthvk, axis=-1)
        zthva = zthvk[..., :-1] + arg * dthv
        zsstab = GRAV / zthva * dthv / zdz
        zrinub = zsstab / zdup2
        zfunst = np.sqrt(np.maximum(1.0 - 18.0 * zrinub, 0.0))
        zfstabh = 1.0 / (1.0 + 10.0 * zrinub * (1.0 + 8.0 * zrinub))
        zkvn = cml2 * np.sqrt(zdup2)
        jq = jqif(zrinub, 0.0)
        wkvf = jq * np.maximum(CKMIN, zkvn * zfstabh) + (1.0 - jq) * np.maximum(CKMIN, zkvn * zfunst)

        # boundary-layer height: surface values, then the two Richardson sweeps
        wtseff = wthm[..., 0] * (1.0 + ZCRDQ * q[..., 0])
        wheatv = wheat + ZCRDQ * wthm[..., 0] * wqflx
        wobkl = -wtseff * libm.cube(ustr) / (GRAV * VKARMAN * (wheatv + np.copysign(1.0e-10, wheatv)))
        pblh = np.zeros((im, jm))
        pblh = _pbl_pass(pblh, whm, uwind, vwind, wthm, q, ustr, wtseff, None, first=True)
        jq = jqif(wheatv, 0.0)
        fmt = libm.pow(jq * (1.0 - BINM * pblh / wobkl) + (1.0 - jq), ONET)
        wsc = ustr * fmt
        zexcess = wheatv * FAK / wsc
        thvparc = wtseff + jq * zexcess
        pblh = _pbl_pass(pblh, whm, uwind, vwind, wthm, q, ustr, wtseff, thvparc, first=False)
        pblh = np.maximum(pblh, 100.0)

        # boundary-layer diffusivity, interfaces l+1
        P, W, HV, U, TS = (a[..., None] for a in (pblh, wobkl, wheatv, ustr, wtseff))
        jq = jqif(P, z)  # inside the boundary layer
        zh = jq * z / P
        zl = jq * z / W + (1.0 - jq)
        zzh = jq * ((1.0 - zh) * (1.0 - zh))
        jq1 = jqif(HV, 0.0)  # unstable
        jq2 = jq * (1.0 - jq1)  # stable boundary layer
        jq3 = jq * jq1  # unstable boundary layer
        jq6 = jq1 * jqif(P, whm[..., :-1]) * jqif(whm[..., 1:], P)  # the entrainment interface
        jq1 = jqif(SFFRAC, zh)  # surface layer
        jq4 = jq3 * jq1  # unstable surface layer
        jq5 = jq3 * (1.0 - jq1)  # unstable boundary layer above it
        cml2 = 1.0 / (1.0 / (VKARMAN * z) + 1.0 / 450.0)
        cml2 = cml2 * cml2
        jqq = jqif(zrinub, 0.0)
        zfstabh = jqq / (1.0 + 10.0 * zrinub * np.sqrt(1.0 + jqq * zrinub)) + (1.0 - jqq)
        zpblk = cml2 * np.sqrt(zdup2) * zfstabh
        zkvh = jq2 * np.maximum(zpblk, KVHMIN) + (1.0 - jq2) * wkvf
        zslask = 1.0 - BETAH * zl
        zslask = jq4 * zslask + (1.0 - jq4)
        zpblk = (1.0 - jq4) * zkvh + jq4 * (U * VKARMAN * z * zzh * libm.pow(zslask, ONET))
        zslask = 1.0 - SSFRAC * BETAH * P / W
        zslask = jq5 * zslask + (1.0 - jq5)
        fmt = libm.pow(zslask, ONET)
        wsc = U * fmt
        zpblk = (1.0 - jq5) * zpblk + jq5 * wsc * VKARMAN * z * zzh
        zwstr = libm.pow(np.abs(HV) * GRAV * P / TS, ONET)
        obukov = jq3 * W - (1.0 - jq3)
        stab = 1.0 - SSFRAC * BETAH * P / obukov
        term1 = libm.pow(stab, ONET)
        term2 = np.sqrt(stab)
        term3 = CCON * FAKN * zwstr / (U * term1)
        pr = jq3 * (term1 / term2 + term3) + (1.0 - jq3)
        kvh = jq3 * np.maximum(zpblk / pr, KVHMIN) + (1.0 - jq3) * zkvh
        thvgrad = dthv / zdz
        kvhentr = 0.2 * HV / thvgrad
        kvh = jq6 * kvhentr + (1.0 - jq6) * kvh
    return kvh, pblh


def dkg(kvh, m, gph, lmax_conv):
    """dkg (im, jm, lmax_conv): the exchange rate in kg/s across the interfaces l+1 for
    l < lmax_conv, zero at lmax_conv."""
    n = lmax_conv - 1
    dz = gph[..., 2 : n + 2] - gph[..., :n]
    out = np.zeros((*m.shape[:2], lmax_conv))
    out[..., :n] = np.maximum(0.0, kvh[..., :n]) * 2.0 * (m[..., 1 : n + 1] + m[..., :n]) / (dz * dz)
    return out


def calc_kzz(spm, sp, pu, pv, t, q, surface, at, bt, dxyp, lmax_conv, dx=6.0, dy=4.0, ybeg=-90.0):
    """Calc_Kzz of one dynamic step: dkg (im, jm, lmax_conv) and the intermediates
    {ustar, gph, pblh, kvh}.  spm: the surface pressure at the middle of the step, sp: the
    advected one at its start (for gph); pu, pv: the step's fluxes; `surface`: {oro, lsm,
    sr, srols, u10m, v10m, sshf, slhf, ewss, nsss} on the model grid."""
    ustar_loc = ustar(*(surface[k] for k in ("lsm", "sr", "srols", "u10m", "v10m", "ewss", "nsss")))
    gph = geopotential(surface["oro"], sp, t, q, at, bt)
    phlb = at + bt * spm[..., None]
    m = mass_t(phlb, dxyp)
    kvh, pblh = bldiff(phlb, gph, t, q, m, pu, pv, ustar_loc, surface["sshf"], surface["slhf"], dx, dy, ybeg)
    return dkg(kvh, m, gph, lmax_conv), {"ustar": ustar_loc, "gph": gph, "pblh": pblh, "kvh": kvh}
