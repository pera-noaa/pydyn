"""Calc_Kzz (diffusion.py, simple_diffusion.py)."""


import numpy as np

from pydyn import (
    balance,
    diffusion,
    synth,
)

from common import (
    IM,
    JM,
    LM,
)


# ------------------------------------------------------------------ the diffusion coefficients
def test_jqif_masks_on_the_sign_bit():
    x = np.array([1.0, 0.0, -0.0, -1.0, 1e-300])
    assert diffusion.jqif(x, 0.0).tolist() == [1.0, 1.0, 0.0, 0.0, 1.0]  # -0.0 - 0.0 is -0.0: sign bit set
    assert diffusion.jqif(0.0, x).tolist() == [0.0, 1.0, 1.0, 1.0, 0.0]


def test_ustar_blends_land_and_sea_by_the_land_fraction():
    rng = np.random.default_rng(16)
    lsm = np.array([[0.0, 100.0, 40.0]])  # sea, land, mixed
    f = [rng.uniform(0.5, 1.0, (1, 3)) for _ in range(4)]
    # sr, srols, u10m, v10m, ewss, nsss
    fields = (0.01 * f[0], 0.5 * f[1], 3.0 * f[2], -2.0 * f[3], 0.1 * f[0], 0.2 * f[1])
    u = diffusion.ustar(lsm, *fields)
    sea = diffusion.ustar(np.zeros((1, 3)), *fields)
    land = diffusion.ustar(np.full((1, 3), 100.0), *fields)
    assert u[0, 0] == sea[0, 0] and u[0, 1] == land[0, 1]
    assert u[0, 2] == 0.4 * land[0, 2] + (1.0 - 0.4) * sea[0, 2]
    assert (sea >= 0.001).all() and (land > 0.0).all()


def test_geopotential_is_hydrostatic():
    rng = np.random.default_rng(17)
    sp = rng.uniform(9.0e4, 1.05e5, (IM, JM))
    a, b = 1.0e4 * np.sin(np.linspace(0.0, np.pi, LM + 1)), np.linspace(1.0, 0.0, LM + 1)
    t, q = np.full((IM, JM, LM), 250.0), np.zeros((IM, JM, LM))
    oro = rng.uniform(0.0, 1.0e4, (IM, JM))
    gph = diffusion.geopotential(oro, sp, t, q, a, b)
    assert gph.shape == (IM, JM, LM + 1)
    np.testing.assert_array_equal(gph[..., 0], oro / balance.GRAV)
    phlb = a + b * sp[..., None]
    steps = 250.0 * 287.05 * np.log(phlb[..., :-2] / phlb[..., 1:-1]) / balance.GRAV
    np.testing.assert_allclose(np.diff(gph[..., :-1], axis=-1), steps, rtol=1e-13)
    assert (gph[..., -1] == 200000.0).all()


def test_calc_kzz_is_finite_bounded_and_closed_at_lmax_conv():
    args, grid = synth.kzz_inputs(np.random.default_rng(18), IM, JM)
    lm, lmax, surface = args[4].shape[-1], args[-1], args[6]
    dkg, parts = diffusion.calc_kzz(*args, **grid)
    assert dkg.shape == (IM, JM, lmax) and np.isfinite(dkg).all() and (dkg >= 0.0).all()
    assert (dkg[..., lmax - 1] == 0.0).all() and (dkg[..., : lmax - 1] > 0.0).any()
    assert (parts["pblh"] >= 100.0).all() and np.isfinite(parts["kvh"]).all()
    assert parts["kvh"].shape == (IM, JM, lm - 1) and (parts["kvh"] >= diffusion.CKMIN).all()
    # the convective columns (heat flux into the air) get a deeper boundary layer than the same
    # columns with the flux reversed
    stable = dict(surface, sshf=np.abs(surface["sshf"]), slhf=np.abs(surface["slhf"]))
    _, cold = diffusion.calc_kzz(*args[:6], stable, *args[7:], **grid)
    heated = surface["sshf"] < 0.0
    assert np.mean(parts["pblh"][heated] >= cold["pblh"][heated]) > 0.9


def test_dkg_is_the_interface_exchange_rate():
    rng = np.random.default_rng(19)
    lm, lmax = 6, 4
    kvh = rng.uniform(-1.0, 10.0, (IM, JM, lm - 1))
    m = rng.uniform(1e12, 2e12, (IM, JM, lm))
    gph = np.cumsum(rng.uniform(300.0, 800.0, (IM, JM, lm + 1)), axis=-1)
    dkg = diffusion.dkg(kvh, m, gph, lmax)
    i, j, k = 3, 4, 2
    dz = gph[i, j, k + 2] - gph[i, j, k]
    want = max(0.0, kvh[i, j, k]) * 2.0 * (m[i, j, k + 1] + m[i, j, k]) / (dz * dz)
    assert dkg[i, j, k] == want and (dkg[..., lmax - 1] == 0.0).all() and dkg.shape == (IM, JM, lmax)
