"""Calc_Kzz written for reading, not for bit-parity.

Same interface and outputs as pydyn.diffusion (see its docstring), same algorithm, but every
`if` of the Fortran -- a 0/1 mask multiplying both branches there -- is a `np.where` here,
the unit conversions are written as the physics (an air density, a hypotenuse), the first
Richardson-number crossing is found on the whole profile at once instead of being carried
up the column, and the powers, logs and cube roots are NumPy's.  Two things are kept as the
Fortran has them because the ulps allowed here cannot absorb them: the layer centres are
interpolated in pressure to the rounded mid-layer pressure (1e-14 of the height), and the
two regularisations -- the bulk Richardson number held away from zero, the Obukhov length
away from infinity -- which are 1e-10 relative.

It agrees with pydyn.diffusion -- and therefore with the Fortran -- to a few ulp
(tests/test_simple.py, tests/test_production.py) except where the algorithm itself is ill
conditioned: in a convective column the boundary-layer height comes from the difference
between theta_v and the surface parcel's theta_v, nearly equal at the crossing, so an ulp
of the parcel's excess is 1e-13 of the height and of the diffusivities that depend on it
(2e-13 relative at worst over the production day, 0.02 % of the values beyond 8 ulp).
The only non-rounding difference is where the Fortran's unused branch is not finite (a
zero theta_v gradient away from the entrainment interface, two equal bulk Richardson
numbers): its 0 * inf is NaN, ours is the branch it does use.

The scheme (Holtslag & Boville 1993; Vogelezang & Holtslag 1996; Beljaars & Viterbo 1998).
At every interface between two layers a heat diffusivity K_h (m^2/s) is chosen:

    free troposphere     K = l^2 |dU/dz| f(Ri)          l from an asymptotic mixing length
                                                        and kappa z, f from the gradient
                                                        Richardson number
    stable boundary      K = max(l^2 |dU/dz| f_m(Ri), 0.1)   the same, with a 450 m
    layer                                               asymptotic length
    unstable boundary    K = max(w_m kappa z (1 - z/h)^2 / Pr, 0.1)   a K-profile on the
    layer                                               velocity scale w_m and the Prandtl
                                                        number of the convective layer
    entrainment          K = 0.2 (w'theta_v')_s / (dtheta_v/dz)   at the interface
                                                        straddled by the layer centres
                                                        around the boundary-layer top

where the boundary-layer height h is where the bulk Richardson number of a layer against
the surface first exceeds 0.3, after a second sweep with the surface parcel warmed by its
convective excess.  K becomes the mass exchange rate dkg (kg/s) through the interface:
K x (mass of the two layers) x 2 / (their combined thickness)^2.
"""

import numpy as np

from .balance import GRAV
from .diffusion import (
    AE,
    APZERO,
    BETAH,
    BINM,
    CCON,
    CKMIN,
    CP_AIR,
    FAK,
    FAKN,
    GTOR,
    KVHMIN,
    LVAP,
    RHO_AIR,
    RICR,
    SFFRAC,
    TINY,
    VKARMAN,
    ZACB,
    ZCRDQ,
    ZKAPPA,
)

__all__ = [
    "air_mass",
    "boundary_layer_height",
    "bulk_richardson",
    "calc_kzz",
    "diffusivity",
    "exchange_rate",
    "free_troposphere_kv",
    "friction_velocity",
    "interface_heights",
    "winds",
]


def friction_velocity(lsm, sr, srols, u10m, v10m, ewss, nsss):
    """u* (im, jm): over sea from the surface stress, over land from the 10-m wind taken up
    the log profile of the (clipped) ECMWF roughness to 75 m and back down that of the
    Olsson roughness; mixed cells blended by the land fraction lsm (%)."""
    land = lsm / 100.0
    u_sea = np.maximum(0.001, np.sqrt(np.hypot(ewss, nsss) / RHO_AIR))
    z0_ecmwf, z0_olsson = np.minimum(sr, 0.03), np.maximum(0.01, srols)
    with np.errstate(all="ignore"):  # sea cells have no roughness: masked out below
        u_land = VKARMAN * np.hypot(u10m, v10m) / np.log(10.0 / z0_ecmwf) * np.log(75.0 / z0_ecmwf)
        u_land = u_land / np.log(10.0 / z0_olsson)
    u_land = np.where(land > 0.0, u_land, 0.0)
    return land * u_land + (1.0 - land) * u_sea


def interface_heights(oro, sp, t, q, at, bt):
    """gph (im, jm, lm+1), m: the orography (oro in m^2/s^2) at the surface, then hydrostatic
    layers R_d T_v / g ln(p_below / p_above); the model top is put at 200 km."""
    phlb = at + bt * sp[..., None]
    tv = t[..., :-1] * (1.0 + 0.608 * q[..., :-1])  # the top layer (up to p = 0) is not needed
    thickness = 287.05 * tv * np.log(phlb[..., :-2] / phlb[..., 1:-1]) / GRAV
    top = np.full_like(oro, 200000.0)
    gph = np.cumsum(np.concatenate([oro[..., None] / GRAV, thickness], axis=-1), axis=-1)
    return np.concatenate([gph, top[..., None]], axis=-1)


def air_mass(phlb, dxyp):
    """m (im, jm, lm), kg: the pressure drop across each layer times the cell area over g."""
    return -np.diff(phlb, axis=-1) * dxyp[None, :, None] / GRAV


def winds(pu, pv, m, dx, dy, ybeg):
    """u, v (im, jm, lm), m/s: the mass flux through the two faces of a cell, averaged, times
    the cell's length across those faces over its mass."""
    jm = m.shape[1]
    lat = ybeg + dy * (np.arange(jm) + 0.5)
    dxx, dyy = AE * GTOR * dx * np.cos(lat * GTOR), AE * GTOR * dy
    u = dxx[None, :, None] * (pu[:-1] + pu[1:]) / 2.0 / m
    v = dyy * (pv[:, :-1] + pv[:, 1:]) / 2.0 / m
    return u, v


def free_troposphere_kv(z, shear2, ri):
    """Holtslag & Boville: K = l^2 |dU/dz| f(Ri) at height z above the surface, the mixing
    length l from kappa z and an asymptotic length of 300 m in the lowest kilometre decaying
    to 30 m above; f damps stable, enhances unstable stratification."""
    asymptotic = np.where(z <= 1000.0, 300.0, 30.0 + 270.0 * np.exp(1.0 - z / 1000.0))
    length = 1.0 / (1.0 / asymptotic + 1.0 / (VKARMAN * z))
    stable = 1.0 / (1.0 + 10.0 * ri * (1.0 + 8.0 * ri))
    unstable = np.sqrt(np.maximum(1.0 - 18.0 * ri, 0.0))
    f = np.where(ri > 0.0, stable, unstable)
    return np.maximum(CKMIN, length * length * np.sqrt(shear2) * f)


def bulk_richardson(thv, thv_ref, height, u, v, ustr):
    """Vogelezang & Holtslag: the bulk Richardson number of every layer against the surface
    layer, g (theta_v - theta_ref) dz / (theta_v,s |dU|^2), the wind difference augmented by
    the surface friction's shear production 100 u*^2.  Kept away from zero (+-1e-10)."""
    du, dv = u - u[..., :1], v - v[..., :1]
    wind2 = np.maximum(du * du + dv * dv + ZACB * ustr[..., None] ** 2, TINY)
    ri = GRAV * (thv - thv_ref[..., None]) * (height - height[..., :1]) / (thv[..., :1] * wind2)
    return ri + np.copysign(1.0e-10, ri)


def boundary_layer_height(ri, height):
    """The height at which the bulk Richardson number profile first exceeds RICR, linearly
    interpolated between that layer's centre and the one below; the top layer's centre if it
    never does."""
    crossed = ri > RICR
    first = np.argmax(crossed, axis=-1, keepdims=True)

    def at(a, k):
        return np.take_along_axis(a, k, axis=-1)[..., 0]

    with np.errstate(all="ignore"):  # columns that never cross: a meaningless interpolation
        fraction = (RICR - at(ri, first - 1)) / (at(ri, first) - at(ri, first - 1))
        h = at(height, first - 1) + fraction * (at(height, first) - at(height, first - 1))
    return np.where(crossed.any(axis=-1), h, height[..., -1])


def diffusivity(phlb, gph, t, q, m, pu, pv, ustar, sshf, slhf, dx, dy, ybeg):
    """bldiff: K_h (im, jm, lm-1) at the interfaces above layers 1..lm-1 and the boundary-layer
    height h (im, jm).  phlb, gph (im, jm, lm+1); t, q, m (im, jm, lm); pu (im+1, jm, lm),
    pv (im, jm+1, lm); ustar and the heat fluxes sshf, slhf (W/m^2, downward) (im, jm)."""
    pf = (phlb[..., :-1] + phlb[..., 1:]) / 2.0
    z_if = gph - gph[..., :1]  # heights above the surface: interfaces, layer centres
    # the centres are halfway up, but interpolated in pressure to pf as the Fortran does: the
    # rounding of pf is worth 1e-14 of the height there, far more than the ulps allowed here
    w = (phlb[..., :-1] - pf) / (phlb[..., :-1] - phlb[..., 1:])
    z_mid = z_if[..., :-1] * (1.0 - w) + z_if[..., 1:] * w
    z = z_if[..., 1:-1]  # the interfaces K_h is defined at
    dz = np.diff(z_mid, axis=-1)

    theta = t * (APZERO / pf) ** ZKAPPA
    thv = theta * (1.0 + ZCRDQ * q)  # virtual potential temperature
    thv_s = thv[..., 0]
    dthv = np.diff(thv, axis=-1)

    # the surface fluxes in kinematic units, the friction velocity, the Obukhov length
    rho = (phlb[..., 0] - phlb[..., 1]) / (GRAV * z_if[..., 1])
    wheat = -sshf / (rho * CP_AIR)  # K m/s
    wqflx = -slhf / (rho * LVAP)  # m/s
    wheatv = wheat + ZCRDQ * theta[..., 0] * wqflx  # the virtual heat flux
    ustr = np.maximum(ustar, 0.01)
    obukhov = -thv_s * ustr**3 / (GRAV * VKARMAN * (wheatv + np.copysign(1.0e-10, wheatv)))
    unstable = wheatv >= 0.0

    u, v = winds(pu, pv, m, dx, dy, ybeg)

    # the free troposphere: shear and gradient Richardson number at the interfaces, theta_v
    # there interpolated in log p between the layer centres
    shear2 = np.maximum(np.diff(u, axis=-1) ** 2 + np.diff(v, axis=-1) ** 2, 1.0e-10) / dz**2
    lpf = np.log(pf)
    thv_if = thv[..., :-1] + (np.log(phlb[..., 1:-1]) - lpf[..., :-1]) / np.diff(lpf, axis=-1) * dthv
    ri = GRAV / thv_if * dthv / dz / shear2
    kv_free = free_troposphere_kv(z, shear2, ri)

    # the boundary-layer height: a first sweep against the surface theta_v, then one against
    # a surface parcel warmed by the convective excess (unstable columns)
    h = boundary_layer_height(bulk_richardson(thv, thv_s, z_mid, u, v, ustr), z_mid)
    w_m = ustr * np.cbrt(np.where(unstable, 1.0 - BINM * h / obukhov, 1.0))
    thv_parcel = thv_s + np.where(unstable, FAK * wheatv / w_m, 0.0)
    ri_b = bulk_richardson(thv, thv_parcel, z_mid, u, v, ustr)
    ri_b[..., 0] = 0.0  # the second sweep starts above the surface layer
    h = np.maximum(boundary_layer_height(ri_b, z_mid), 100.0)

    # the boundary layer, at the interfaces below h: the local scheme when stable, the
    # K-profile over the Prandtl number when unstable
    h_, l_, hv_, us_, ts_ = (a[..., None] for a in (h, obukhov, wheatv, ustr, thv_s))
    in_bl, unstable_ = z <= h_, unstable[..., None]
    length = 1.0 / (1.0 / (VKARMAN * z) + 1.0 / 450.0)
    ri_stable = np.maximum(ri, 0.0)  # the momentum stability function: 1 when unstable
    f_m = 1.0 / (1.0 + 10.0 * ri_stable * np.sqrt(1.0 + ri_stable))
    kv_stable = np.maximum(length * length * np.sqrt(shear2) * f_m, KVHMIN)
    zh = z / h_
    with np.errstate(invalid="ignore"):  # stable columns (L > 0), where none of this is used
        w_m = us_ * np.cbrt(np.where(zh <= SFFRAC, 1.0 - BETAH * z / l_, 1.0 - SFFRAC * BETAH * h_ / l_))
        profile = w_m * VKARMAN * z * (1.0 - zh) ** 2
        stab = 1.0 - SFFRAC * BETAH * h_ / l_  # the stability function at the surface-layer top
        w_star = np.cbrt(np.abs(hv_) * GRAV * h_ / ts_)
        prandtl = np.cbrt(stab) / np.sqrt(stab) + CCON * FAKN * w_star / (us_ * np.cbrt(stab))
        kv_unstable = np.maximum(profile / prandtl, KVHMIN)
    kvh = np.where(in_bl, np.where(unstable_, kv_unstable, kv_stable), kv_free)

    # the entrainment interface: the one between the layer centres around h
    entrainment = unstable_ & (z_mid[..., :-1] <= h_) & (h_ <= z_mid[..., 1:])
    with np.errstate(divide="ignore"):  # a zero theta_v gradient elsewhere
        kvh = np.where(entrainment, 0.2 * hv_ / (dthv / dz), kvh)
    return kvh, h


def exchange_rate(kvh, m, gph, lmax_conv):
    """dkg (im, jm, lmax_conv), kg/s: K_h as the air mass exchanged per second across the
    interfaces above layers 1..lmax_conv-1; nothing crosses the one above lmax_conv."""
    n = lmax_conv - 1
    thickness = gph[..., 2 : n + 2] - gph[..., :n]  # of the two layers around the interface
    rate = np.maximum(kvh[..., :n], 0.0) * 2.0 * (m[..., :n] + m[..., 1 : n + 1]) / thickness**2
    return np.pad(rate, [(0, 0), (0, 0), (0, 1)])


def calc_kzz(spm, sp, pu, pv, t, q, surface, at, bt, dxyp, lmax_conv, dx=6.0, dy=4.0, ybeg=-90.0):
    """Calc_Kzz of one dynamic step: dkg (im, jm, lmax_conv) and the intermediates
    {ustar, gph, pblh, kvh}; the arguments are pydyn.diffusion.calc_kzz's."""
    ustar = friction_velocity(*(surface[k] for k in ("lsm", "sr", "srols", "u10m", "v10m", "ewss", "nsss")))
    gph = interface_heights(surface["oro"], sp, t, q, at, bt)
    phlb = at + bt * spm[..., None]
    m = air_mass(phlb, dxyp)
    kvh, pblh = diffusivity(phlb, gph, t, q, m, pu, pv, ustar, surface["sshf"], surface["slhf"], dx, dy, ybeg)
    return exchange_rate(kvh, m, gph, lmax_conv), {"ustar": ustar, "gph": gph, "pblh": pblh, "kvh": kvh}
