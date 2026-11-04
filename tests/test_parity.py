"""Bitwise parity of pydyn against the production Fortran kernels (docs/state.md section 3a).

Needs fortran/harness (`make -C fortran`); skipped otherwise.
Every case writes a seeded input to a tmp dir, runs the same kernel in Fortran and in
Python, and requires the state arrays, the returned flux and the CFL statistics to be
identical to the bit.
"""

import numpy as np
import pytest

from pydyn import advect_x, advect_y, advect_z, convec, fluxes_from_meteo, harness, synth
from pydyn.kernels import CFLError

pytestmark = pytest.mark.skipif(not harness.available(), reason="Fortran harness not built")

IM, JM, LM, NT = 8, 6, 5, 2
NDYN = 3600
DT = NDYN / 2.0  # one half step (tref = 1)
KERNEL = {"x": advect_x, "y": advect_y, "z": advect_z}


def _assert_equal(out, **python):
    """Every named array in `out` is bitwise equal to its Python counterpart."""
    reports = [harness.compare(out[name], arr, name) for name, arr in python.items()]
    assert all(r["equal"] for r in reports), harness.format_report(reports)


def _advect_both(tmp_path, axis, s, fluxes, limits=True):
    """Run kernel `axis` in Fortran and in Python on the same input; assert bitwise parity."""
    k = "xyz".index(axis)
    name = harness.FLUX[axis]
    harness.write_case(tmp_path, axis, s, ndyn=NDYN, limits=limits, **{name: fluxes[k]})
    out = harness.run(tmp_path)

    py, flux, stats = s.copy(), fluxes[k].copy(), {}
    KERNEL[axis](py.m, py.rm, py.rxm, py.rym, py.rzm, flux, limits=limits, stats=stats)
    _assert_equal(out, **{n: getattr(py, n) for n in harness.STATE}, **{name: flux})
    for key, val in stats.items():
        assert out["stats"][key][k] == val, f"{key}[{axis}]: fortran {out['stats'][key][k]!r} python {val!r}"
    return out


# --- advection ------------------------------------------------------------------------


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("cfl", [0.3, 0.9, 2.5])
def test_advect_courant(tmp_path, axis, cfl):
    rng = np.random.default_rng(100 * "xyz".index(axis) + int(10 * cfl))
    s = synth.state(rng, IM, JM, LM, NT)
    fl = synth.fluxes(rng, s.m, cfl=cfl)
    out = _advect_both(tmp_path, axis, s, fl)
    if cfl > 1.0:
        assert out["stats"]["nloop_max"]["xyz".index(axis)] > 1  # subcycling was exercised


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("slopes", ["zero", "saturated"])
def test_advect_slope_regimes(tmp_path, axis, slopes):
    rng = np.random.default_rng(7)
    s = synth.state(rng, IM, JM, LM, NT, uniform=(slopes == "zero"))
    if slopes == "saturated":  # slopes at the limiter boundary, both signs
        for n in ("rxm", "rym", "rzm"):
            setattr(s, n, s.rm * rng.choice([-1.0, 1.0], s.rm.shape))
    _advect_both(tmp_path, axis, s, synth.fluxes(rng, s.m, cfl=0.9))


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_advect_no_limits(tmp_path, axis):
    rng = np.random.default_rng(11)
    s = synth.state(rng, IM, JM, LM, NT)
    _advect_both(tmp_path, axis, s, synth.fluxes(rng, s.m, cfl=0.9), limits=False)


@pytest.mark.parametrize("direction", ["into", "out_of"])
def test_advect_y_poles(tmp_path, direction):
    """Meridional flow converging on both poles (or diverging from them)."""
    rng = np.random.default_rng(3)
    s = synth.state(rng, IM, JM, LM, NT)
    am, bm, cm = synth.fluxes(rng, s.m, cfl=0.5)
    sign = 1.0 if direction == "into" else -1.0
    half = JM // 2
    scale = 0.4 * s.m.mean()
    bm = np.zeros_like(bm)
    bm[:, 1:half, :] = -sign * scale * rng.uniform(0.5, 1.0, (IM, half - 1, LM))  # toward j = 1
    bm[:, half + 1 : JM, :] = sign * scale * rng.uniform(0.5, 1.0, (IM, JM - half - 1, LM))  # toward jm
    _advect_both(tmp_path, "y", s, (am, bm, cm))


def _meteo(rng, s):
    rate = 0.3 * s.m.mean() / DT  # kg/s giving Courant ~0.3 per half step
    pu = rate * rng.uniform(-1, 1, (IM + 1, JM, LM))
    pu[IM] = pu[0]
    pv = rate * rng.uniform(-1, 1, (IM, JM + 1, LM))
    pv[:, 0] = pv[:, JM] = 0.0
    bt = np.array([1.0, 0.9, 0.6, 0.3, 0.1, 0.0])
    return pu, pv, bt


def test_advect_z_from_dynam0(tmp_path):
    """cm as dynam0 derives it from the horizontal convergence and a hybrid bt."""
    rng = np.random.default_rng(5)
    s = synth.state(rng, IM, JM, LM, NT)
    pu, pv, bt = _meteo(rng, s)
    _advect_both(tmp_path, "z", s, fluxes_from_meteo(pu, pv, bt, DT))


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_advect_cfl_failure_parity(tmp_path, axis):
    """Both sides refuse the same hopeless flux: Fortran escape_tm, Python CFLError."""
    rng = np.random.default_rng(13)
    s = synth.state(rng, IM, JM, LM, NT)
    am, bm, cm = synth.fluxes(rng, s.m, cfl=0.4)
    big = 200.0 * s.m.mean()
    if axis == "x":
        am[:] = big
    elif axis == "y":
        bm[:, 1:JM] = big
    else:
        cm[..., 1:LM] = big
    flux = {"x": am, "y": bm, "z": cm}[axis]
    harness.write_case(tmp_path, axis, s, ndyn=NDYN, **{harness.FLUX[axis]: flux})
    with pytest.raises(harness.HarnessError, match=r"nloop|CFL"):
        harness.run(tmp_path)
    with pytest.raises(CFLError):
        KERNEL[axis](s.m, s.rm, s.rxm, s.rym, s.rzm, flux)


# --- dynam0 ---------------------------------------------------------------------------


def test_dynam0(tmp_path):
    rng = np.random.default_rng(17)
    s = synth.state(rng, IM, JM, LM, NT)
    pu, pv, bt = _meteo(rng, s)
    harness.write_case(tmp_path, "dynam0", s, pu=pu, pv=pv, bt=bt, ndyn=NDYN)
    am, bm, cm = fluxes_from_meteo(pu, pv, bt, DT)
    _assert_equal(harness.run(tmp_path), am=am, bm=bm, cm=cm)


# --- convection -----------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["random", "top_zero", "top_lm", "lfs_above_top", "strong_detu"])
def test_convec(tmp_path, variant):
    rng = np.random.default_rng(19)
    s = synth.state(rng, IM, JM, LM, NT)
    lmax = LM
    conv = synth.convection(
        rng, s.m, lmax, DT, lfs_above_top=(variant == "lfs_above_top"), strong_detu=(variant == "strong_detu")
    )
    if variant == "top_zero":
        conv["cloud_top"][:] = 0
    if variant == "top_lm":
        conv["cloud_top"][:] = lmax
    harness.write_case(tmp_path, "convec", s, conv=conv, ndyn=NDYN, lmax_conv=lmax)
    out = harness.run(tmp_path)
    py = s.copy()
    convec(py.m, py.rm, py.rxm, py.rym, py.rzm, dt=DT, lmax=lmax, **conv)
    _assert_equal(out, **{n: getattr(py, n) for n in harness.STATE})
