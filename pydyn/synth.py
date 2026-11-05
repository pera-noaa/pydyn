"""Synthetic, physically admissible inputs for the property and parity tests."""

import numpy as np

from .driver import State

__all__ = [
    "convection",
    "fluxes",
    "kzz_inputs",
    "state",
    "zoom_convection",
    "zoom_meteo",
    "zoom_state",
]


def state(rng, im, jm, lm, nt, uniform=False, slope_amp=0.3):
    """Random air mass ~1e9 kg and tracer masses with slopes of relative size `slope_amp`.

    uniform=True gives a uniform mixing ratio per tracer and zero slopes."""
    m = 1.0e9 * rng.uniform(0.8, 1.2, (im, jm, lm))
    if uniform:
        rm = np.stack([m * (k + 1) * 1e-6 for k in range(nt)], axis=-1)
        slopes = [np.zeros_like(rm) for _ in range(3)]
    else:
        rm = m[..., None] * rng.uniform(0.5, 1.5, (im, jm, lm, nt)) * 1e-6
        slopes = [slope_amp * rm * rng.uniform(-1, 1, rm.shape) for _ in range(3)]
    return State(m, rm, *slopes)


def fluxes(rng, m, cfl=0.4, amp=None):
    """Fluxes (am, bm, cm) for air mass m: along the periodic x lines a mostly uniform flow
    with Courant number ~cfl; along the closed y/z lines the flux that moves each line from m
    to a random positive target m' = m (1 + amp u), u in [-1, 1], same line total.
    Sub-stepping such a flux keeps every intermediate mass positive."""
    im, jm, lm = m.shape
    amp = min(0.6 * cfl, 0.9) if amp is None else amp
    scale = cfl * m.mean()
    sign = rng.choice([-1.0, 1.0], (1, jm, lm))
    am = scale * sign * rng.uniform(0.8, 1.0, (im + 1, jm, lm))
    am[im] = am[0]

    def closed(axis):
        target = m * (1.0 + amp * rng.uniform(-1, 1, m.shape))
        target *= m.sum(axis=axis, keepdims=True) / target.sum(axis=axis, keepdims=True)
        d = m - target  # mass leaving each cell through its upper face, cumulatively
        f = np.cumsum(d, axis=axis)
        shape = list(m.shape)
        shape[axis] = 1
        return np.concatenate([np.zeros(shape), f], axis=axis)

    bm = closed(1)
    bm[:, jm] = 0.0
    cm = closed(2)
    cm[..., lm] = 0.0
    return am, bm, cm


def convection(rng, m, lmax, dt, top_max=None, lfs_above_top=False, strong_detu=False):
    """Random en/detrainment profiles (kg/s per box) and cloud levels for `convec`."""
    im, jm, lm = m.shape
    top_max = lmax if top_max is None else top_max
    rate = 0.05 * m.mean() / dt  # ~5 % of a cell per step
    entu = rate * rng.uniform(0, 1, (im, jm, lm))
    detu = rate * rng.uniform(0, 1, (im, jm, lm))
    entd = 0.3 * rate * rng.uniform(0, 1, (im, jm, lm))
    detd = 0.3 * rate * rng.uniform(0, 1, (im, jm, lm))
    if strong_detu:
        detu *= 4.0  # detu > amu + entu somewhere: the zxi = 0 branch
    cloud_top = rng.integers(0, top_max + 1, (im, jm))
    if lfs_above_top:
        cloud_lfs = np.minimum(cloud_top + rng.integers(1, 3, (im, jm)), lmax)
    else:
        cloud_lfs = rng.integers(0, np.maximum(cloud_top, 1))
    dkg = 0.02 * m[..., :lmax].mean() / dt * rng.uniform(0, 1, (im, jm, lmax))
    return {
        "entu": entu,
        "detu": detu,
        "entd": entd,
        "detd": detd,
        "dkg": dkg,
        "cloud_top": cloud_top.astype(np.int32),
        "cloud_lfs": cloud_lfs.astype(np.int32),
    }


def kzz_inputs(rng, im, jm, lm=8, lmax_conv=6):
    """Inputs of diffusion.calc_kzz on a global im x jm grid: a warm surface under a stable
    lapse rate, a few convective columns (heat flux into the air).  Returns (args, grid)
    with `grid` the dx / dy / ybeg keywords."""
    from .diffusion import AE

    a, b = 1.0e4 * np.sin(np.linspace(0.0, np.pi, lm + 1)), np.linspace(1.0, 0.0, lm + 1)
    sp = rng.uniform(9.5e4, 1.02e5, (im, jm))
    t = 290.0 - 6.5e-3 * np.cumsum(np.full((im, jm, lm), 800.0), axis=-1) + rng.normal(0.0, 0.5, (im, jm, lm))
    q = rng.uniform(1e-4, 1e-2, (im, jm, lm))
    pu = rng.normal(0.0, 3e9, (im + 1, jm, lm))
    pv = rng.normal(0.0, 1e9, (im, jm + 1, lm))
    surface = {
        "oro": rng.uniform(0.0, 5e3, (im, jm)),
        "lsm": rng.choice([0.0, 100.0], (im, jm)),
        "sr": rng.uniform(1e-4, 0.5, (im, jm)),
        "srols": rng.uniform(1e-3, 1.0, (im, jm)),
        "u10m": rng.normal(0.0, 5.0, (im, jm)),
        "v10m": rng.normal(0.0, 5.0, (im, jm)),
        "sshf": rng.normal(-20.0, 60.0, (im, jm)),
        "slhf": rng.normal(-40.0, 40.0, (im, jm)),
        "ewss": rng.normal(0.0, 0.1, (im, jm)),
        "nsss": rng.normal(0.0, 0.1, (im, jm)),
    }
    dx, dy = 360.0 / im, 180.0 / jm
    dxyp = np.radians(dx) * np.diff(np.sin(np.radians(-90.0 + dy * np.arange(jm + 1)))) * AE**2  # cell areas
    args = (sp, sp, pu, pv, t, q, surface, a, b, dxyp, lmax_conv)
    return args, {"dx": dx, "dy": dy, "ybeg": -90.0}


# ------------------------------------------------------------------ zoom configurations (§3d)
def zoom_state(zoom, rng, nt=2, cfl=0.4):
    """A ZoomState with every array filled, halos included, so that any read is of an
    admissible value: masses ~1e9 kg per parent cell (split over the child cells), tracers
    and slopes as `state`, fluxes of Courant number up to ~cfl.  cfl <= 1: independent
    random signs, for single-routine calls.  cfl > 1: one sign per row / layer and closed
    column fluxes (and closed meridians in region 1), which sub-cycling can always pass."""
    from .zoom import ZoomState

    st = ZoomState.zeros(zoom, nt)
    lm = zoom.lm
    for r in zoom.regions():
        f, g = st[r], zoom.g(r)
        m = f["m"].data
        m[...] = 1e9 * rng.uniform(0.8, 1.2, m.shape) / (g.xref * g.yref)
        f["rm"].data[...] = m[..., None] * rng.uniform(0.5, 1.5, f["rm"].data.shape) * 1e-6
        for s in ("rxm", "rym", "rzm"):
            f[s].data[...] = 0.3 * f["rm"].data * rng.uniform(-1, 1, f["rm"].data.shape)
        mm = m.mean()
        if cfl <= 1.0:
            for s in ("am", "bm", "cm"):
                f[s].data[...] = cfl * mm * rng.uniform(-1, 1, f[s].data.shape)
        else:
            a, b = f["am"].data, f["bm"].data
            a[...] = cfl * mm * rng.choice([-1.0, 1.0], (1, *a.shape[1:])) * rng.uniform(0.9, 1.0, a.shape)
            b[...] = cfl * mm * rng.choice([-1.0, 1.0], (1, 1, b.shape[2])) * rng.uniform(0.9, 1.0, b.shape)
            mi = f["m"][1 : g.im, 1 : g.jm, 1:lm]

            f["cm"].data[...] = 0.0
            f["cm"][1 : g.im, 1 : g.jm, 1:lm] = _closed(rng, mi, 2)
            if r == 1:  # gentler: a meridian's cumulative transport grows along its 18+ cells
                f["bm"][1 : g.im, 2 : g.jm + 1, 1:lm] = _closed(rng, mi, 1, amp=0.65)
        f["cm"][:, :, 0] = 0.0
        f["cm"][:, :, lm] = 0.0
        if r == 1:  # closed poles, and the periodic face dynam0 leaves
            f["bm"][:, 1, :] = 0.0
            f["bm"][:, g.jm + 1, :] = 0.0
            f["am"][0, :, :] = f["am"][g.im, :, :].copy()
        f["pu"].data[...] = 1e7 * rng.uniform(-1, 1, f["pu"].data.shape)
        f["pv"].data[...] = 1e7 * rng.uniform(-1, 1, f["pv"].data.shape)
        f["dxyp"].data[...] = 1e10 * rng.uniform(0.5, 1.5, f["dxyp"].data.shape)
        f["sp"].data[...] = rng.uniform(9e4, 1e5, f["sp"].data.shape)
        f["phlb"].data[...] = rng.uniform(0, 1e5, f["phlb"].data.shape)
    return st


def _closed(rng, m, axis, amp=0.9):
    """The interior-face fluxes that move each line of m to a random positive target with the
    same line total (the construction of `fluxes`)."""
    target = m * (1.0 + amp * rng.uniform(-1, 1, m.shape))
    target *= m.sum(axis=axis, keepdims=True) / target.sum(axis=axis, keepdims=True)
    return np.cumsum(m - target, axis=axis)


def zoom_meteo(zoom, rng, nt=1, cfl=0.3, ndyn=3600):
    """zoom_state with pu / pv from one smooth global flow that every region sees: per level a
    zonal rate (one sign for all regions) and a meridional one vanishing at the poles, each
    region's face flux being its share of the cross-section (pu / yref, pv / xref), 5 % noise.
    This is what makes a sequence of operators admissible: independent random fluxes per
    region fight each other across the interfaces and drain cells.  Parent Courant ~cfl, a
    child's ~cfl xref / tref.  Run zoom.setup_massflow on it for am, bm, cm."""
    st = zoom_state(zoom, rng, nt=nt, cfl=0.1)
    g1, lm = zoom.g(1), zoom.lm
    m1 = st[1]["m"][1 : g1.im, 1 : g1.jm, 1:lm].mean()
    u = cfl * m1 / (ndyn / 2.0) * rng.choice([-1.0, 1.0], lm) * rng.uniform(0.8, 1.0, lm)
    for r in zoom.regions():
        f, g = st[r], zoom.g(r)
        dy = (g.yend - g.ybeg) / g.jm
        lat_c = g.ybeg + dy * (np.arange(1, g.jm + 1) - 0.5)
        lat_f = g.ybeg + dy * np.arange(0, g.jm + 1)
        noise = 1 + 0.05 * rng.uniform(-1, 1, f["pu"].data.shape)
        f["pu"].data[...] = u[None, None, :] * np.cos(np.radians(lat_c))[None, :, None] / g.yref * noise
        noise = 1 + 0.05 * rng.uniform(-1, 1, f["pv"].data.shape)
        merid = np.sin(np.pi * (lat_f + 90) / 180)[None, :, None]
        f["pv"].data[...] = 0.5 * np.abs(u)[None, None, :] * merid / g.xref * noise
        if r == 1:
            f["pv"][:, 1, :] = 0.0
            f["pv"][:, g.jm + 1, :] = 0.0
    return st


def zoom_convection(zoom, st, rng, dt=1800.0, lmax=None):
    """{region: convec inputs} in the kernels.convec layout, ~5 % of a cell per step; convection
    over the lowest lmax levels (default all; tropo34's archive has 26)."""
    out = {}
    lm = zoom.lm if lmax is None else lmax
    for r in zoom.regions():
        g = zoom.g(r)
        rate = 0.05 * st[r]["m"][1 : g.im, 1 : g.jm, 1 : zoom.lm].mean() / dt
        shape = (g.im, g.jm, lm)
        out[r] = {
            "entu": rate * rng.uniform(0, 1, shape),
            "detu": rate * rng.uniform(0, 1, shape),
            "entd": 0.3 * rate * rng.uniform(0, 1, shape),
            "detd": 0.3 * rate * rng.uniform(0, 1, shape),
            "dkg": rate * rng.uniform(0, 1, shape),
            "cloud_top": rng.integers(0, lm + 1, (g.im, g.jm)).astype(np.int32),
            "cloud_lfs": rng.integers(0, lm + 1, (g.im, g.jm)).astype(np.int32),
            "lmax": lm,
        }
    return out
