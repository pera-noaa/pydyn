"""Property tests for the pydyn kernels (these are NOT the Fortran-parity tests; see README)."""

import numpy as np
import pytest

from pydyn import Model, State, advect_x, advect_y, advect_z, convec, fluxes_from_meteo, synth
from pydyn.kernels import CFLError, convection_matrix, fastminv

IM, JM, LM, NT = 8, 6, 5, 2


def _total(s):
    return s.rm.sum(axis=(0, 1, 2))


KERNELS = {
    "x": (advect_x, 0),
    "y": (advect_y, 1),
    "z": (advect_z, 2),
}


@pytest.mark.parametrize("which", list(KERNELS))
def test_uniform_mixing_ratio_preserved(which):
    rng = np.random.default_rng(1)
    s = synth.state(rng, IM, JM, LM, NT, uniform=True)
    fl = synth.fluxes(rng, s.m)
    kern, idx = KERNELS[which]
    q0 = s.mixing_ratio()
    kern(s.m, s.rm, s.rxm, s.rym, s.rzm, fl[idx])
    np.testing.assert_allclose(s.mixing_ratio(), q0, rtol=1e-12)
    assert np.all(s.m > 0)


@pytest.mark.parametrize("which", list(KERNELS))
def test_mass_conservation(which):
    rng = np.random.default_rng(2)
    s = synth.state(rng, IM, JM, LM, NT)
    fl = synth.fluxes(rng, s.m)
    kern, idx = KERNELS[which]
    m0, r0 = s.m.sum(), _total(s)
    f0 = fl[idx].copy()
    kern(s.m, s.rm, s.rxm, s.rym, s.rzm, fl[idx])
    np.testing.assert_allclose(s.m.sum(), m0, rtol=1e-13)
    np.testing.assert_allclose(_total(s), r0, rtol=1e-13)
    np.testing.assert_allclose(fl[idx], f0, rtol=1e-15)  # flux handed back (up to nloop scaling)


@pytest.mark.parametrize("which", list(KERNELS))
def test_positivity_with_limiter(which):
    rng = np.random.default_rng(3)
    s = synth.state(rng, IM, JM, LM, NT)
    # slopes as large as allowed
    for sl in (s.rxm, s.rym, s.rzm):
        sl[...] = s.rm * rng.choice([-1.0, 1.0], s.rm.shape)
    fl = synth.fluxes(rng, s.m, cfl=0.9)
    kern, idx = KERNELS[which]
    kern(s.m, s.rm, s.rxm, s.rym, s.rzm, fl[idx])
    assert np.all(s.rm >= -1e-12 * s.rm.max())


@pytest.mark.parametrize("which", list(KERNELS))
def test_subcycling_on_cfl_violation(which):
    rng = np.random.default_rng(4)
    s = synth.state(rng, IM, JM, LM, NT)
    fl = synth.fluxes(rng, s.m, cfl=2.5)  # some faces will exceed |a/m| >= 1
    kern, idx = KERNELS[which]
    r0 = _total(s)
    st = {}
    kern(s.m, s.rm, s.rxm, s.rym, s.rzm, fl[idx], stats=st)
    assert st["nloop_max"] > 1
    np.testing.assert_allclose(_total(s), r0, rtol=1e-12)
    assert np.all(s.m > 0)


def test_cfl_error():
    rng = np.random.default_rng(5)
    s = synth.state(rng, IM, JM, LM, NT)
    am, _, _ = synth.fluxes(rng, s.m, cfl=200.0)
    with pytest.raises(CFLError):
        advect_x(s.m, s.rm, s.rxm, s.rym, s.rzm, am, max_nloop=5)


def test_y_pole_slopes_untouched():
    rng = np.random.default_rng(6)
    s = synth.state(rng, IM, JM, LM, NT)
    _, bm, _ = synth.fluxes(rng, s.m)
    rxm_p = s.rxm[:, [0, JM - 1]].copy()
    rym_p = np.maximum(np.minimum(s.rym, abs(s.rm)), -abs(s.rm))[:, [0, JM - 1]]
    advect_y(s.m, s.rm, s.rxm, s.rym, s.rzm, bm)
    np.testing.assert_array_equal(s.rxm[:, [0, JM - 1]], rxm_p)
    np.testing.assert_array_equal(s.rym[:, [0, JM - 1]], rym_p)


def test_x_row_equals_z_column():
    """A single x-row with periodic halo cut by a zero flux is the same 1-D problem as a z-column."""
    rng = np.random.default_rng(7)
    n, nt = LM, NT
    m = 1e9 * rng.uniform(0.8, 1.2, n)
    rm = m[:, None] * rng.uniform(0.5, 1.5, (n, nt)) * 1e-6
    sl = [0.3 * rm * rng.uniform(-1, 1, rm.shape) for _ in range(3)]
    a = 0.4 * m.mean() * rng.uniform(-1, 1, n + 1)
    a[0] = a[n] = 0.0

    # x: shape (n, 1, 1); the along-axis slope is rxm
    sx = State(
        m.reshape(n, 1, 1),
        rm.reshape(n, 1, 1, nt),
        sl[0].reshape(n, 1, 1, nt),
        sl[1].reshape(n, 1, 1, nt),
        sl[2].reshape(n, 1, 1, nt),
    )
    advect_x(sx.m, sx.rm, sx.rxm, sx.rym, sx.rzm, a.reshape(n + 1, 1, 1).copy())
    # z: shape (1, 1, n); the along-axis slope is rzm
    sz = State(
        m.reshape(1, 1, n),
        rm.reshape(1, 1, n, nt),
        sl[1].reshape(1, 1, n, nt),
        sl[2].reshape(1, 1, n, nt),
        sl[0].reshape(1, 1, n, nt),
    )
    advect_z(sz.m, sz.rm, sz.rxm, sz.rym, sz.rzm, a.reshape(1, 1, n + 1).copy())

    np.testing.assert_array_equal(sx.m.ravel(), sz.m.ravel())
    np.testing.assert_allclose(sx.rm.reshape(n, nt), sz.rm.reshape(n, nt), rtol=1e-14)
    np.testing.assert_allclose(sx.rxm.reshape(n, nt), sz.rzm.reshape(n, nt), rtol=1e-13)
    np.testing.assert_allclose(sx.rym.reshape(n, nt), sz.rxm.reshape(n, nt), rtol=1e-14)


def test_dynam0_column_mass_consistency():
    """The vertical flux from dynam0 distributes the column convergence with bt, so that
    after x, y and z the new column mass equals the old plus the horizontal convergence."""
    rng = np.random.default_rng(8)
    s = synth.state(rng, IM, JM, LM, NT)
    bt = np.linspace(1.0, 0.0, LM + 1) ** 1.5
    dt = 900.0
    pu = 1e5 * rng.uniform(-1, 1, (IM + 1, JM, LM))
    pu[IM] = pu[0]
    pv = 1e5 * rng.uniform(-1, 1, (IM, JM + 1, LM))
    pv[:, 0] = pv[:, JM] = 0.0
    am, bm, cm = fluxes_from_meteo(pu, pv, bt, dt)
    assert np.all(cm[..., 0] == 0) and np.all(cm[..., LM] == 0)
    m0 = s.m.copy()
    conv = (am[:-1] - am[1:] + bm[:, :-1] - bm[:, 1:]).sum(axis=2)
    for kern, fl in ((advect_x, am), (advect_y, bm), (advect_z, cm)):
        kern(s.m, s.rm, s.rxm, s.rym, s.rzm, fl)
    np.testing.assert_allclose(s.m.sum(axis=2), m0.sum(axis=2) + conv, rtol=1e-12)
    # and the layer masses follow the hybrid distribution of the column change
    expected = (bt[:-1] - bt[1:])[None, None, :] * conv[..., None]
    np.testing.assert_allclose(s.m - m0, expected, atol=1e-6 * s.m.mean())


def _conv_inputs(rng, im=IM, jm=JM, lm=LM):
    m = 1e9 * rng.uniform(0.8, 1.2, (im, jm, lm))
    entu = 1e4 * rng.uniform(0, 1, (im, jm, lm))
    detu = 1e4 * rng.uniform(0, 1, (im, jm, lm))
    entd = 1e3 * rng.uniform(0, 1, (im, jm, lm))
    detd = 1e3 * rng.uniform(0, 1, (im, jm, lm))
    dkg = 1e3 * rng.uniform(0, 1, (im, jm, lm))
    cloud_top = rng.integers(0, lm + 1, (im, jm))
    cloud_lfs = rng.integers(0, lm + 1, (im, jm))
    return m, {
        "entu": entu,
        "detu": detu,
        "entd": entd,
        "detd": detd,
        "dkg": dkg,
        "cloud_top": cloud_top,
        "cloud_lfs": cloud_lfs,
        "lmax": lm,
    }


def test_fastminv_matches_numpy():
    rng = np.random.default_rng(9)
    m, c = _conv_inputs(rng)
    conv1 = convection_matrix(
        m.reshape(-1, LM),
        c["entu"].reshape(-1, LM),
        c["detu"].reshape(-1, LM),
        c["entd"].reshape(-1, LM),
        c["detd"].reshape(-1, LM),
        c["cloud_top"].ravel(),
        c["cloud_lfs"].ravel(),
        c["dkg"].reshape(-1, LM),
        900.0,
        LM,
    )
    np.testing.assert_allclose(fastminv(conv1), np.linalg.inv(conv1), rtol=1e-10, atol=1e-14)


def test_convection_conserves_column_mass_and_uniform_ratio():
    rng = np.random.default_rng(10)
    m, c = _conv_inputs(rng)
    rm = m[..., None] * rng.uniform(0.5, 1.5, (IM, JM, LM, NT)) * 1e-6
    rm[..., 1] = 3e-6 * m
    slopes = [0.3 * rm * rng.uniform(-1, 1, rm.shape) for _ in range(3)]
    col0 = rm.sum(axis=2)
    convec(m, rm, *slopes, dt=900.0, **c)
    np.testing.assert_allclose(rm.sum(axis=2), col0, rtol=1e-10)
    np.testing.assert_allclose(rm[..., 1] / m, 3e-6, rtol=1e-10)


def test_driver_symmetric_step():
    rng = np.random.default_rng(11)
    s = synth.state(rng, IM, JM, LM, NT)
    bt = np.linspace(1.0, 0.0, LM + 1) ** 1.5
    pu = 2e5 * rng.uniform(-1, 1, (IM + 1, JM, LM))
    pu[IM] = pu[0]
    pv = 2e5 * rng.uniform(-1, 1, (IM, JM + 1, LM))
    pv[:, 0] = pv[:, JM] = 0.0
    _, c = _conv_inputs(rng)
    model = Model(bt, ndyn=1800)
    r0 = _total(s)
    model.step(s, pu, pv, conv=c)
    np.testing.assert_allclose(_total(s), r0, rtol=1e-11)
    assert set(model.stats) == {"x", "y", "z"}
