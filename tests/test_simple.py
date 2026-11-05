"""pydyn.simple agrees with pydyn.kernels, and pydyn.simple_diffusion with pydyn.diffusion (both
bitwise with the Fortran), to 1e-12 relative or 1e-12 of the field's largest magnitude: not a
few ulp element by element, where a small slope (rxm, rym) differs by up to ~3e5 ulp, 6e-11
relative (measured over five seeds of this grid), because the two sum large terms differently."""

import numpy as np
import pytest

from pydyn import diffusion, kernels, simple, simple_diffusion, synth
from pydyn.harness import STATE, compare

IM, JM, LM, NT = 8, 6, 5, 2
DT = 1800.0
RTOL = 1e-12


def _assert_close(a, b, name):
    scale = np.abs(b).max() or 1.0
    np.testing.assert_allclose(a, b, rtol=RTOL, atol=RTOL * scale, err_msg=name)


def _run_both(kernel, s, flux, **kw):
    """Run kernel `kernel` from both modules on copies of the same input and compare."""
    out = {}
    for mod in (kernels, simple):
        st, fl, stats = s.copy(), flux.copy(), {}
        getattr(mod, kernel)(st.m, st.rm, st.rxm, st.rym, st.rzm, fl, stats=stats, **kw)
        out[mod] = (st, stats)
    (ref, ref_stats), (got, got_stats) = out[kernels], out[simple]
    for name in STATE:
        _assert_close(getattr(got, name), getattr(ref, name), name)
    assert got_stats["nloop_max"] == ref_stats["nloop_max"]
    if kernel != "advect_z" or ref_stats["nloop_max"] == 1:
        # dynamw's xi also counts the Courant numbers of the failed trial sub-steps (>= 1);
        # simple reports the largest one actually used
        assert got_stats["xi"] == pytest.approx(ref_stats["xi"], rel=RTOL)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("cfl", [0.4, 2.5])
@pytest.mark.parametrize("limits", [True, False])
def test_advect(axis, cfl, limits):
    rng = np.random.default_rng(int(10 * cfl) + limits)
    s = synth.state(rng, IM, JM, LM, NT)
    flux = synth.fluxes(rng, s.m, cfl=cfl)["xyz".index(axis)]
    _run_both(f"advect_{axis}", s, flux, limits=limits)


@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_advect_y_poles(direction):
    rng = np.random.default_rng(3)
    s = synth.state(rng, IM, JM, LM, NT)
    half = JM // 2
    bm = np.zeros((IM, JM + 1, LM))
    bm[:, 1:half] = -direction * 0.4 * s.m.mean() * rng.uniform(0.5, 1.0, (IM, half - 1, LM))
    bm[:, half + 1 : JM] = direction * 0.4 * s.m.mean() * rng.uniform(0.5, 1.0, (IM, JM - half - 1, LM))
    _run_both("advect_y", s, bm)


def test_cfl_error():
    rng = np.random.default_rng(13)
    s = synth.state(rng, IM, JM, LM, NT)
    with pytest.raises(simple.CFLError):
        simple.advect_x(s.m, s.rm, s.rxm, s.rym, s.rzm, np.full((IM + 1, JM, LM), 200.0 * s.m.mean()))


def test_dynam0():
    rng = np.random.default_rng(17)
    rate = 1e5
    pu = rate * rng.uniform(-1, 1, (IM + 1, JM, LM))
    pu[IM] = pu[0]
    pv = rate * rng.uniform(-1, 1, (IM, JM + 1, LM))
    pv[:, 0] = pv[:, JM] = 0.0
    bt = np.array([1.0, 0.9, 0.6, 0.3, 0.1, 0.0])
    for got, ref, name in zip(
        simple.fluxes_from_meteo(pu, pv, bt, DT), kernels.fluxes_from_meteo(pu, pv, bt, DT), "abc",
        strict=True,
    ):
        _assert_close(got, ref, name + "m")


@pytest.mark.parametrize("variant", ["random", "top_zero", "top_lm", "lfs_above_top", "strong_detu"])
def test_convec(variant):
    rng = np.random.default_rng(19)
    s = synth.state(rng, IM, JM, LM, NT)
    conv = synth.convection(
        rng, s.m, LM, DT, lfs_above_top=(variant == "lfs_above_top"), strong_detu=(variant == "strong_detu")
    )
    if variant == "top_zero":
        conv["cloud_top"][:] = 0
    if variant == "top_lm":
        conv["cloud_top"][:] = LM
    ref, got = s.copy(), s.copy()
    kernels.convec(ref.m, ref.rm, ref.rxm, ref.rym, ref.rzm, dt=DT, lmax=LM, **conv)
    simple.convec(got.m, got.rm, got.rxm, got.rym, got.rzm, dt=DT, lmax=LM, **conv)
    for name in STATE:
        _assert_close(getattr(got, name), getattr(ref, name), name)


@pytest.mark.parametrize("seed", [18, 19, 20])
def test_calc_kzz(seed):
    """Every intermediate and the dkg within a handful of ulp of the bitwise transcription."""
    args, grid = synth.kzz_inputs(np.random.default_rng(seed), IM, JM)
    dkg, parts = diffusion.calc_kzz(*args, **grid)
    got, got_parts = simple_diffusion.calc_kzz(*args, **grid)
    reports = [compare(parts[name], got_parts[name], name) for name in parts] + [compare(dkg, got, "dkg")]
    assert all(r["max_ulp"] <= 16 for r in reports), [(r["name"], r["max_ulp"]) for r in reports]
    assert compare(parts["gph"], got_parts["gph"])["equal"]  # the same sequential accumulation
