"""Meteo_Setup_Mass / BalanceMassFluxes and m2phlb1 (balance.py)."""

from datetime import datetime

import numpy as np
import pytest

from pydyn import (
    balance,
    synth,
)

from common import (
    IM,
    JM,
    LM,
    synth_state,
)


# ------------------------------------------------------------------ Meteo_Setup_Mass
def _setup_mass_case(seed, solver=None):
    rng = np.random.default_rng(seed)
    area = 1.0e10 * rng.uniform(0.5, 1.0, JM)
    levi_db = -np.diff(np.linspace(1.0, 0.0, LM + 1))
    sp1 = 1.0e5 * rng.uniform(0.97, 1.03, (IM, JM))
    sp2 = sp1 * rng.uniform(0.999, 1.001, (IM, JM))
    m = np.abs(levi_db)[None, None, :] * sp1[..., None] / balance.GRAV * area[None, :, None]
    dt = 3600.0
    pu, pv, pw = (q / dt for q in synth.fluxes(rng, m, cfl=0.05))
    pw = pw + 0.0
    # a divergence error small enough for the 1e-4 pre-check, large enough to need balancing
    pu[1:IM] += 1.0e-5 * pu[1:IM].std() * rng.standard_normal(pu[1:IM].shape)
    return {
        "pu": pu, "pv": pv, "pw": pw, "sp1": sp1, "sp2": sp2,
        "levi_db": levi_db, "area_m2": area, "dt_sec": dt, "solver": solver,
    }  # fmt: skip


def test_interpol_fractions():
    t1, t2 = datetime(2000, 1, 1, 0), datetime(2000, 1, 1, 3)
    assert balance.interpol_fractions(datetime(2000, 1, 1, 1, 30), t1, t2) == (0.5, 0.5)
    assert balance.interpol_fractions(t1, t1, t2) == (1.0, 0.0)
    assert balance.interpol_fractions(t1, t1, t1) == (0.5, 0.5)


def test_poisson_correction_closes_the_divergence():
    rng = np.random.default_rng(7)
    resid = rng.standard_normal((IM, JM))
    resid -= resid.mean()  # only a zero-mean residual can be closed on a periodic domain
    u, v = balance.poisson_correction(resid)
    assert not u[0].any() and not v[:, 0].any()
    np.testing.assert_allclose(u[0], u[IM]) and np.testing.assert_allclose(v[:, 0], v[:, JM])
    div = u[:IM] - u[1:] + v[:, :JM] - v[:, 1:]
    np.testing.assert_allclose(div, -resid, atol=1e-12)


def test_setup_mass_balances_to_the_strict_tolerance():
    case = _setup_mass_case(8)
    ok0, *_ = balance.check_mass_balance(
        case["pu"], case["pv"], case["sp1"], case["sp2"], case["dt_sec"], case["area_m2"], 1.0e-7, 1.0e-6
    )
    assert not ok0
    info = balance.setup_mass(**case)
    assert (
        info["ok"]
        and info["tries"] == 1
        and info["solver"] == ("fftw" if balance.fftw_library() else "numpy")
    )
    assert info["post"][0] < 1.0e-7 and info["post"][1] < 1.0e-6
    ok, *_ = balance.check_mass_balance(
        case["pu"], case["pv"], case["sp1"], case["sp2"], case["dt_sec"], case["area_m2"], 1.0e-7, 1.0e-6
    )
    assert ok


@pytest.mark.skipif(balance.fftw_library() is None, reason="no libfftw3 found")
def test_fftw_and_numpy_poisson_agree():
    rng = np.random.default_rng(9)
    resid = rng.standard_normal((IM, JM))
    ref = balance.NumpyPoisson(IM, JM).psi(resid)
    got = balance.FFTWPoisson(IM, JM, balance.fftw_library()).psi(resid)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)


def test_m2phlb1_matches_the_hydrostatic_sum():
    rng, s = synth_state(10)
    area = 1.0e10 * rng.uniform(0.5, 1.0, JM)
    at, bt = np.linspace(0.0, 100.0, LM + 1), np.linspace(1.0, 0.0, LM + 1)
    sp, phlb = balance.m2phlb1(s.m, at, bt, area)
    sp_ref = np.sum(s.m, axis=2) * balance.GRAV / area[None, :]  # numpy's sum: to a few ulp
    phlb_ref = at[None, None, :] + bt[None, None, :] * sp_ref[..., None]
    np.testing.assert_allclose(sp, sp_ref, rtol=1e-14)
    np.testing.assert_allclose(phlb, phlb_ref, rtol=1e-14)
