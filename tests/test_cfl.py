"""Check_CFL (cfl.py, advectm_cfl.F90)."""


import numpy as np
import pytest

from pydyn import (
    cfl,
    kernels,
    redgrid,
    synth,
)

from common import (
    DT,
    IM,
    JM,
    LM,
    NDYN,
    synth_state,
)


# ------------------------------------------------------------------ Check_CFL
def test_new_valid_timestep():
    assert cfl.new_valid_timestep(3600, 3600, 3600) == 1800
    assert cfl.new_valid_timestep(1800, 3600, 3600) == 1200
    assert cfl.new_valid_timestep(900, 3600, 3600, tref_max=2) == 720
    with pytest.raises(ValueError):
        cfl.new_valid_timestep(1, 3600, 3600)


def test_mass_kernels_agree_with_advection_on_mass():
    """dynamum / dynamvm / dynamwm move the air mass exactly as the tracer kernels do."""
    rng, s = synth_state(4)
    am, bm, cm = synth.fluxes(rng, s.m, cfl=0.5)
    for op, flux, mass_kernel, kernel in (
        ("x", am, cfl.dynamum, kernels.advect_x),
        ("y", bm, cfl.dynamvm, kernels.advect_y),
        ("z", cm, cfl.dynamwm, kernels.advect_z),
    ):
        ref, f = s.copy(), flux.copy()
        kernel(ref.m, ref.rm, ref.rxm, ref.rym, ref.rzm, f)
        m, f2 = s.m.copy(), flux.copy()
        assert mass_kernel(m, f2) == 1, op
        np.testing.assert_allclose(m, ref.m, rtol=1e-14, err_msg=op)
        np.testing.assert_array_equal(f2, flux)  # fluxes handed back unchanged


def test_dynamum_reports_sub_cycles_and_limit():
    rng, s = synth_state(5)
    am = synth.fluxes(rng, s.m, cfl=2.5)[0]
    nloop = cfl.dynamum(s.m.copy(), am.copy(), max_nloop=50)
    assert nloop > 2
    assert cfl.dynamum(s.m.copy(), am.copy(), max_nloop=nloop + 1) == nloop
    assert cfl.dynamum(s.m.copy(), am.copy(), max_nloop=nloop) is None  # nloop == max_nloop fails


def test_check_cfl_keeps_or_reduces_ndyn():
    rng, s = synth_state(6)
    bt = np.linspace(1.0, 0.0, LM + 1)

    # meteo fluxes (kg/s) that dynam0 turns into am, bm of Courant number ~0.3 per half step
    am, bm, _ = synth.fluxes(rng, s.m, cfl=0.3)
    quiet = (am / DT, bm / DT)
    ndyn, n, it = cfl.check_cfl(s.m, NDYN, 1, NDYN, bt, 1, lambda t: quiet)
    assert (ndyn, n, it) == (NDYN, 1, 1)

    # a periodic x flow with Courant number ~12 needs more than MAX_NLOOP_X sub-cycles per
    # half step; halving ndyn halves the flux per step (the total over the interval is fixed)
    am_wild = synth.fluxes(rng, s.m, cfl=12.0)[0]
    am_wild[:] = am_wild[0]  # non-divergent, so the vertical flux dynam0 derives from it is zero
    wild = (am_wild / DT, bm / DT)
    ndyn, n, it = cfl.check_cfl(
        s.m, NDYN, 1, NDYN, bt, 1, lambda t: wild, grid=redgrid.RedGrid(IM, JM, (), ())
    )
    assert ndyn < NDYN and NDYN % ndyn == 0 and n == NDYN // ndyn and it > 1
    fl = kernels.fluxes_from_meteo(*wild, bt, ndyn / 2.0)
    assert cfl.rehearse_step(s.m.copy(), *fl)
    fl = kernels.fluxes_from_meteo(*wild, bt, NDYN / 2.0)
    assert not cfl.rehearse_step(s.m.copy(), *fl)


def test_check_cfl_rounds_the_step_count_as_fortran_nint(monkeypatch):
    """advectm_cfl.F90: n = nint(real(n*ndyn_old)/real(ndyn)), half away from zero. n = 3 at
    1800 s reduced to 1200 s is 4.5 steps: 5, where np.rint's half-to-even gives 4."""
    calls = []
    monkeypatch.setattr(cfl, "rehearse_step", lambda *a, **k: calls.append(1) or len(calls) > 1)
    m = np.ones((IM, JM, LM))
    zero = (np.zeros((IM + 1, JM, LM)), np.zeros((IM, JM + 1, LM)))
    bt = np.linspace(1.0, 0.0, LM + 1)
    assert cfl.check_cfl(m, 1800, 3, 3600, bt, 1, lambda t: zero) == (1200, 5, 2)
