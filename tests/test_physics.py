"""The sources and the CH4 chemistry (emis.py, chem.py) and convection's LAPACK inverse (lapack.py)."""


import numpy as np
import pytest

from pydyn import (
    chem,
    emis,
    grid,
    kernels,
    lapack,
    meteo,
    synth,
)
from pydyn.harness import STATE, compare

from common import (
    DT,
    IM,
    JM,
    LM,
    assert_bitwise,
    synth_state,
)


# ------------------------------------------------------------------ sources and chemistry
def _rebin_levels_fortran(ip_levels, ip_data, op_levels):
    """rebin_levels(ip_mix=.true.) as the Fortran loops it, one column, with its j_start skip
    and early exit."""
    nlay_ip, nlay_op = len(ip_data), len(op_levels) - 1
    op_data = np.zeros(nlay_op)
    j_start = 0
    for i in range(nlay_op):
        op_bot, op_top = op_levels[i], op_levels[i + 1]
        while j_start < nlay_ip and ip_levels[j_start + 1] >= op_bot:
            j_start += 1
        for j in range(j_start, nlay_ip):
            ip_bot, ip_top = ip_levels[j], ip_levels[j + 1]
            if ip_bot <= op_top:
                break
            thickness = min(op_bot, ip_bot) - max(op_top, ip_top)
            if thickness > 0.0:
                op_data[i] = op_data[i] + ip_data[j] * thickness
        if op_bot - op_top > 0.0:
            op_data[i] = op_data[i] / (op_bot - op_top)
    return op_data


def test_rebin_is_the_fortran_loop_bitwise():
    rng = np.random.default_rng(13)
    for _ in range(20):
        ip = np.sort(rng.uniform(0.0, 1000.0, rng.integers(2, 12)))[::-1]
        op = np.sort(rng.uniform(0.0, 1200.0, rng.integers(2, 40)))[::-1]
        ip[-1], op[-1] = 0.0, 0.0  # both end at the top of the atmosphere
        data = rng.uniform(0.0, 1.0, len(ip) - 1)
        np.testing.assert_array_equal(chem.rebin(ip, data, op), _rebin_levels_fortran(ip, data, op))


def test_rebin_broadcasts_and_rejects_unordered_levels():
    rng = np.random.default_rng(14)
    ip = np.array([1000.0, 800.0, 500.0, 0.0])
    data = rng.uniform(0.0, 1.0, (3, 3))
    op = np.linspace(1010.0, 0.0, 6)
    got = chem.rebin(ip[None], data, np.broadcast_to(op, (3, 6)))
    for k in range(3):
        np.testing.assert_array_equal(got[k], _rebin_levels_fortran(ip, data[k], op))
    with pytest.raises(ValueError):
        chem.rebin(ip[::-1], data[0], op)
    with pytest.raises(ValueError):
        chem.rebin(ip, data[0], op[::-1])


def test_use_libm_routes_every_function_through_the_named_library():
    """ops.use_libm swaps exp/log/pow/cos/sin for the named library's (ifort's libimf on an
    Intel build); with the system libm it is the default, value for value."""
    import ctypes.util

    from pydyn import ops

    path = ctypes.util.find_library("m") or ctypes.util.find_library("c")
    x = np.linspace(0.1, 3.0, 17)
    saved = {n: ops.libm.__dict__[n] for n in ("exp", "log", "pow", "cos", "sin", "path")}
    want = [ops.libm.exp(x), ops.libm.log(x), ops.libm.pow(x, 1.37), ops.libm.cos(x), ops.libm.sin(x)]
    try:
        ops.use_libm(path)
        assert ops.libm.path == path
        got = [ops.libm.exp(x), ops.libm.log(x), ops.libm.pow(x, 1.37), ops.libm.cos(x), ops.libm.sin(x)]
        for g, w in zip(got, want, strict=True):
            np.testing.assert_array_equal(g, w)
    finally:
        for n, f in saved.items():
            setattr(ops.libm, n, f)


def test_strat_edges_and_tropopause():
    press = np.array([[900.0, 700.0, 300.0]])
    np.testing.assert_array_equal(chem.strat_edges(press), [[100000.0, 80000.0, 50000.0, 0.0]])
    assert chem.tropopause(np.array([0.0]))[0] == 30000.0 - 21500.0
    assert chem.tropopause(np.array([np.pi / 2]))[0] == pytest.approx(30000.0)


def test_loss_rates_switch_at_the_tropopause_and_decay_all_moments():
    rng, s = synth_state(15)
    im, jm, lm = IM, JM, LM
    phlb = np.linspace(1.0e5, 0.0, lm + 1)[None, None] * np.ones((im, jm, 1))
    temper = rng.uniform(210.0, 300.0, (im, jm, lm))
    lat = np.linspace(-1.2, 1.2, jm)
    strat_press = np.linspace(500.0, 1.0, 8)[None] * np.ones((jm, 1))
    strat = {n: rng.uniform(1e-9, 1e-8, (jm, 8)) for n in ("strat_oh", "strat_cl", "strat_od")}
    oh_3d = rng.uniform(1e5, 1e6, (im, jm, lm))
    soil = rng.uniform(0.0, 1.0, (im, jm))
    loss = chem.loss_rates(temper, phlb, lat, strat_press, **strat, oh_3d=oh_3d, soil=soil, soil_rate=2e-4)
    assert loss.shape == (im, jm, lm, 4)
    stratospheric = phlb[..., :-1] < chem.tropopause(lat)[None, :, None]
    spiv = (2.45e-12 * chem.libm.exp(-1775.0 / temper)) * oh_3d
    np.testing.assert_array_equal(loss[..., 0][~stratospheric], spiv[~stratospheric])
    assert (loss[..., 0][stratospheric] < 1e-7).all()  # the rebinned zonal column, not Spivakovsky
    np.testing.assert_array_equal(loss[..., 0, 3], -(2e-4 * soil))
    assert not loss[..., 1:, 3].any()

    factor = chem.decay(loss, 1800.0)
    assert (factor[..., 1:] < 1.0).all() and (factor[..., 0] > factor[..., 1]).all()  # the soil is a source
    before = s.copy()
    state = {n: getattr(s, n) for n in STATE}
    chem.apply_decay(state, factor, tracers=(2,))
    np.testing.assert_array_equal(s.rm[..., 0], before.rm[..., 0])  # not a CH4 tracer
    for n in ("rm", "rxm", "rym", "rzm"):
        np.testing.assert_array_equal(getattr(s, n)[..., 1], getattr(before, n)[..., 1] * factor)
    np.testing.assert_array_equal(s.m, before.m)


def test_do_add_2d_adds_at_the_cell_floor():
    rng, s = synth_state(16)
    field = rng.uniform(0.0, 1.0, (IM, JM))
    x = emis.inject(field, month2dt=1800.0 / 2.6784e6, mol_mass=16.04, mol_mass_e=16.04)
    np.testing.assert_array_equal(x, (field * (1800.0 / 2.6784e6)) * 1.0)
    before = s.copy()
    emis.do_add_2d({n: getattr(s, n) for n in STATE}, x, level=1, tracer=2)
    np.testing.assert_array_equal(s.rm[..., 0, 1], before.rm[..., 0, 1] + x)
    np.testing.assert_array_equal(s.rzm[..., 0, 1], before.rzm[..., 0, 1] - x)
    np.testing.assert_array_equal(s.rm[..., 1:, :], before.rm[..., 1:, :])
    np.testing.assert_array_equal(s.rm[..., 0], before.rm[..., 0])
    for n in ("m", "rxm", "rym"):
        np.testing.assert_array_equal(getattr(s, n), getattr(before, n))


def test_month2dt_is_the_half_source_step_per_month():
    assert emis.month2dt(3600, 2678400.0) == 1800.0 / 2678400.0
    assert emis.month2dt(3600, 2678400.0, tref=2) == 900.0 / 2678400.0
    assert emis.XMCH4 == 16.04275


def test_monthly_field_is_area_times_seconds_summed_over_the_block():
    """A uniform 1x1 flux gives each model cell its area (the 6x4 1x1 areas, summed in the
    FillGrid order) times the month's seconds times the unit factor."""
    raw = np.full((360, 180), 2.0)
    sec_month = 2678400.0
    got = emis.monthly_field(raw, sec_month, xm=16.0)
    assert got.shape == (60, 45)
    dxy11 = meteo.dxyp(180, 1, 1, -90.0)
    per_cell = ((2.0 * (16.0 / 1.0e9)) * dxy11) * sec_month
    for j in range(45):
        res = 0.0
        for jx in range(j * 4, (j + 1) * 4):
            for _ in range(6):
                res = res + per_cell[jx] * 1.0
        assert np.all(got[:, j] == res)
    total = 2.0 * (16.0 / 1.0e9) * sec_month * 4 * np.pi * grid.AE**2
    np.testing.assert_allclose(got.sum(), total, rtol=1e-12)


# ------------------------------------------------------------------ LAPACK inverse
@pytest.mark.skipif(lapack.lapack_library() is None, reason="no LAPACK library found")
def test_lapack_inverse_agrees_with_numpy():
    rng = np.random.default_rng(11)
    a = np.eye(LM)[None] + 0.1 * rng.standard_normal((4, LM, LM))
    inv = lapack.LapackInverse(lapack.lapack_library())
    np.testing.assert_allclose(inv(a), np.linalg.inv(a), rtol=1e-12, atol=1e-12)
    assert lapack.batched_inverse("numpy") is np.linalg.inv


def test_convec_accepts_a_callable_inverse():
    rng, s = synth_state(12)
    conv = synth.convection(rng, s.m, LM, DT)
    a, b = s.copy(), s.copy()
    kernels.convec(a.m, a.rm, a.rxm, a.rym, a.rzm, dt=DT, lmax=LM, inverse="numpy", **conv)
    kernels.convec(b.m, b.rm, b.rxm, b.rym, b.rzm, dt=DT, lmax=LM, inverse=np.linalg.inv, **conv)
    assert_bitwise([compare(getattr(a, n), getattr(b, n), n) for n in STATE])
