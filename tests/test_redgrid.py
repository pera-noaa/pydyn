"""The reduced grid (redgrid.py, redgridZoom.F90)."""


import numpy as np

from pydyn import (
    kernels,
    redgrid,
    synth,
)
from pydyn.harness import STATE, compare

from common import (
    IM,
    JM,
    LM,
    NT,
    assert_bitwise,
    small_redgrid,
    synth_state,
)


# ------------------------------------------------------------------ reduced grid
def test_redgrid_from_meta_matches_production_rc():
    meta = {
        "grid_reduced": True,
        "nred": 4,
        "jred": (1, 2, 3, 45),
        "clustsize": (60, 20, 10, 60),
        "im": 60,
        "jm": 45,
    }
    g = redgrid.RedGrid.from_meta(meta)
    assert g.bands == [(0, 60, 1), (1, 20, 3), (2, 10, 6), (44, 60, 1)]
    assert g.imred[3] == 60 and g.imred[44] == 1
    assert redgrid.RedGrid.from_meta({"grid_reduced": False}) is None


def test_uni2red_conserves_and_red2uni_restores_uniform_ratio():
    g = small_redgrid()
    rng, s = synth_state(1)
    s_uni = synth.state(rng, IM, JM, LM, NT, uniform=True)
    for st in (s, s_uni):
        st0 = st.copy()
        redgrid.uni2red(g, st.m, st.rm, st.rxm, st.rym, st.rzm)
        for j, _, n in g.bands:
            np.testing.assert_allclose(st.m[:n, j].sum(0), st0.m[:, j].sum(0), rtol=1e-14)
            np.testing.assert_allclose(st.rm[:n, j].sum(0), st0.rm[:, j].sum(0), rtol=1e-14)
            assert not st.rm[n:, j].any() and st.m[n:, j].any()  # tracers zeroed beyond imredj, m not
        redgrid.red2uni(g, st.m, st.rm, st.rxm, st.rym, st.rzm, st0.m)
        np.testing.assert_array_equal(st.m, st0.m)
        if st is s_uni:  # uniform mixing ratio: splitting by mass is the exact inverse
            np.testing.assert_allclose(st.rm, st0.rm, rtol=1e-14)


def test_uni2red_mf():
    g = small_redgrid()
    am = np.arange(IM + 1, dtype=float)[:, None, None] * np.ones((1, JM, LM))
    redgrid.uni2red_mf(g, am)
    j, c, n = g.bands[1]
    np.testing.assert_array_equal(am[1 : n + 1, j, 0], c * np.arange(1, n + 1))
    assert am[0, j, 0] == am[n, j, 0]
    assert (am[n + 1 :, j] == -1.0).all()
    np.testing.assert_array_equal(am[:, 3], np.arange(IM + 1)[:, None] * np.ones((1, LM)))  # untouched row


def test_advect_x_reduced_without_bands_is_advect_x():
    rng, s = synth_state(2)
    am = synth.fluxes(rng, s.m, cfl=0.6)[0]
    ref, ref_am, ref_stats = s.copy(), am.copy(), {}
    kernels.advect_x(ref.m, ref.rm, ref.rxm, ref.rym, ref.rzm, ref_am, stats=ref_stats)
    got, got_am, got_stats = s.copy(), am.copy(), {}
    redgrid.advect_x_reduced(
        got.m, got.rm, got.rxm, got.rym, got.rzm, got_am, redgrid.RedGrid(IM, JM, (), ()), stats=got_stats
    )
    assert_bitwise(
        [compare(getattr(ref, n), getattr(got, n), n) for n in STATE] + [compare(ref_am, got_am, "am")]
    )
    assert got_stats == ref_stats


def test_advect_x_reduced_conserves_and_returns_flux():
    g = small_redgrid()
    rng, s = synth_state(3)
    am = synth.fluxes(rng, s.m, cfl=0.8)[0]
    am[:, g.rows] *= 0.3  # the merged rows see the summed flux of a cluster face: keep it admissible
    s0, am0 = s.copy(), am.copy()
    redgrid.advect_x_reduced(s.m, s.rm, s.rxm, s.rym, s.rzm, am, g, stats={})
    np.testing.assert_array_equal(am, am0)
    np.testing.assert_allclose(s.m.sum(0), s0.m.sum(0), rtol=1e-13)
    np.testing.assert_allclose(s.rm.sum(0), s0.rm.sum(0), rtol=1e-13)
    np.testing.assert_allclose(s.m, s0.m + am0[:-1] - am0[1:], rtol=1e-13)  # the uniform-grid mass update
    assert (s.rm > 0).all()
