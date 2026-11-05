"""Meteo_Setup_Mass for a zoom child (`PYD-3` item 3): pydyn.zoom_meteo.

The property tests need nothing.  The dump tests read one parent/child pair of meteo events
of the glb3x2 + nam1x1 set (`zm_replay`, runs/MANIFEST.md); off Orion/Hercules the Poisson
matrices come from a different FFT than the model's MKL, so only the Match is bitwise here and
the balanced interior is checked to rounding.  The bitwise claim is the Hercules job's
(docs/state.md section 3f).
"""


import numpy as np
import pytest

from pydyn import balance, inputs
from pydyn import zoom_meteo as ZM
from pydyn.dump import load_events
from pydyn.grid import LL
from pydyn.ops import convergence

from common import ZOOM_METEO_DIR, needs

FIXTURE = ZOOM_METEO_DIR
needs_fixture = needs(FIXTURE, "PYDYN_ZOOM_METEO_DIR")

RF = ZM.Refinement(ri=3, rj=2, ci1=3, ci2=6, cj1=2, cj2=4)  # nam1x1's factors, 4 x 3 parent cells
IMP, JMP, LM = 8, 6, 3
IM, JM = 12, 6
TOL = 1e-13


def _rng():
    return np.random.default_rng(7)


def test_refinement_from_header():
    child = {"xref": 3, "yref": 2, "ibeg": 17, "iend": 40, "jbeg": 56, "jend": 77}
    rf = ZM.Refinement.from_meta(child, {"xref": 1, "yref": 1})
    assert (rf.ri, rf.rj, rf.ci1, rf.ci2, rf.cj1, rf.cj2) == (3, 2, 17, 40, 56, 77)
    assert len(list(rf.cells())) == 24 * 22


def test_match_u_v_cell_bring_parent_faces_to_the_parent():
    rng = _rng()
    ppu, pu = rng.normal(size=(IMP + 1, JMP, LM)), rng.normal(size=(IM + 1, JM, LM))
    ppv, pv = rng.normal(size=(IMP, JMP + 1, LM)), rng.normal(size=(IM, JM + 1, LM))
    ppw, pw = rng.normal(size=(IMP, JMP, LM + 1)), rng.normal(size=(IM, JM, LM + 1))
    u0, v0 = pu.copy(), pv.copy()
    ZM.match_u_sum(ppu, pu, RF)
    ZM.match_v_sum(ppv, pv, RF)
    ZM.match_cell_sum(ppw, pw, RF)
    for ci in range(RF.ci1 - 1, RF.ci2 + 1):
        fi = (ci - RF.ci1 + 1) * RF.ri
        for cj in range(RF.cj1, RF.cj2 + 1):
            fj = (cj - RF.cj1) * RF.rj
            np.testing.assert_allclose(pu[fi, fj : fj + RF.rj].sum(0), ppu[ci, cj - 1], rtol=TOL, atol=TOL)
    for cj in range(RF.cj1 - 1, RF.cj2 + 1):
        fj = (cj - RF.cj1 + 1) * RF.rj
        for ci in range(RF.ci1, RF.ci2 + 1):
            fi = (ci - RF.ci1) * RF.ri
            np.testing.assert_allclose(pv[fi : fi + RF.ri, fj].sum(0), ppv[ci - 1, cj], rtol=TOL, atol=TOL)
    for ci, cj, fi, fj in RF.cells():
        np.testing.assert_allclose(pw[fi : fi + RF.ri, fj : fj + RF.rj].sum((0, 1)), ppw[ci - 1, cj - 1],
                                   rtol=TOL, atol=TOL)
    # faces inside a parent cell are not the Match's
    inner_u = np.ones(IM + 1, bool)
    inner_u[:: RF.ri] = False
    inner_v = np.ones(JM + 1, bool)
    inner_v[:: RF.rj] = False
    assert np.array_equal(pu[inner_u], u0[inner_u]) and np.array_equal(pv[:, inner_v], v0[:, inner_v])


def test_match_area_aver_gives_the_parent_its_mean():
    rng = _rng()
    parea, tarea = rng.uniform(1, 2, JMP), rng.uniform(1, 2, JM)
    for cj in range(RF.cj1, RF.cj2 + 1):  # a parent row's area is its child rows' (x refine_i)
        fj = (cj - RF.cj1) * RF.rj
        parea[cj - 1] = RF.ri * tarea[fj : fj + RF.rj].sum()
    pg, tg = rng.normal(1e5, 1e3, (IMP, JMP)), rng.normal(1e5, 1e3, (IM, JM))
    ZM.match_area_aver(pg, tg, RF, parea, tarea)
    for ci, cj, fi, fj in RF.cells():
        mean = (tg[fi : fi + RF.ri, fj : fj + RF.rj] * tarea[None, fj : fj + RF.rj]).sum() / parea[cj - 1]
        assert mean == pytest.approx(pg[ci - 1, cj - 1], rel=1e-14)


def test_poisson_direct_is_the_fft_solve_and_keeps_the_block_faces():
    pd = ZM.PoissonDirect(3, 2, balance.NumpyPoisson(3, 2))
    m = _rng().normal(size=(3, 2, 4))
    m -= m.mean(axis=(0, 1))
    u, v = pd.solve(m)
    for k in range(m.shape[2]):
        uf, vf = balance.poisson_correction(m[..., k], balance.NumpyPoisson(3, 2))
        np.testing.assert_allclose(u[..., k], uf, atol=1e-14)
        np.testing.assert_allclose(v[..., k], vf, atol=1e-14)
    assert not u[[0, -1]].any() and not v[:, [0, -1]].any()
    with pytest.raises(ValueError):
        ZM.PoissonDirect(5, 5)


def _consistent(rng):
    """Child fluxes and a mass tendency whose residual sums to zero over every parent cell."""
    pu, pv = rng.normal(size=(IM + 1, JM, LM)), rng.normal(size=(IM, JM + 1, LM))
    pw = rng.normal(size=(IM, JM, LM + 1))
    noise = rng.normal(size=(IM, JM, LM))
    for _ci, _cj, fi, fj in RF.cells():
        noise[fi : fi + RF.ri, fj : fj + RF.rj] -= noise[fi : fi + RF.ri, fj : fj + RF.rj].mean(axis=(0, 1))
    dm_dt = convergence(pu, pv) + pw[..., :-1] - pw[..., 1:] - noise
    return pu, pv, pw, dm_dt


def test_balance_closes_every_cell_and_leaves_the_parent_faces():
    pu, pv, pw, dm_dt = _consistent(_rng())
    u0, v0 = pu.copy(), pv.copy()
    ZM.balance_mass_fluxes_child(pu, pv, pw, dm_dt, RF, ZM.PoissonDirect(3, 2, balance.NumpyPoisson(3, 2)))
    resid = convergence(pu, pv) + pw[..., :-1] - pw[..., 1:] - dm_dt
    assert np.abs(resid).max() < 1e-12
    assert np.array_equal(pu[:: RF.ri], u0[:: RF.ri]) and np.array_equal(pv[:, :: RF.rj], v0[:, :: RF.rj])


def test_balance_cannot_close_a_parent_cell_that_does_not_balance():
    """Known-bad: a residual with a non-zero parent-cell sum is beyond a correction that keeps
    the parent faces, so the residual check above can fail."""
    pu, pv, pw, dm_dt = _consistent(_rng())
    dm_dt[0, 0, 0] += 1.0
    ZM.balance_mass_fluxes_child(pu, pv, pw, dm_dt, RF, ZM.PoissonDirect(3, 2, balance.NumpyPoisson(3, 2)))
    resid = convergence(pu, pv) + pw[..., :-1] - pw[..., 1:] - dm_dt
    assert np.abs(resid).max() > 0.1


# ------------------------------------------------------------------ the zm_replay pair
def _pair():
    evs = {e.seq: e for e in load_events(FIXTURE)}
    return evs[1], evs[2]


@needs_fixture
def test_child_meteo_against_the_model():
    p, c = _pair()
    rf = ZM.Refinement.from_meta(c.meta, p.meta)
    t1, t2 = inputs.times(c, "tread1", "tread2")
    pd = ZM.PoissonDirect(rf.ri, rf.rj, balance.poisson_solver(rf.ri, rf.rj))
    for k in (1, 2):
        raw = tuple(c.interior(f"{n}{k}") for n in ("mfu", "mfv", "mfw"))
        par = tuple(p.interior(f"{n}{k}") for n in ("pu", "pv", "pw"))
        sp = c.interior("sp1"), c.interior("sp2")
        dt = (t2 - t1).total_seconds()
        pu, pv, pw, info = ZM.setup_mass_child(raw, par, *sp, c["levi_db"][1:], c["area_m2"][1:], dt, rf, pd)
        fu, fv = c.interior(f"pu{k}"), c.interior(f"pv{k}")
        assert info["ok"]
        assert np.array_equal(pw, c.interior(f"pw{k}"))  # Match_cell 'sum', nothing else touches pw
        assert np.array_equal(pu[:: rf.ri], fu[:: rf.ri]) and np.array_equal(pv[:, :: rf.rj], fv[:, :: rf.rj])
        np.testing.assert_allclose(pu, fu, rtol=0, atol=1e-9 * np.abs(fu).max())
        np.testing.assert_allclose(pv, fv, rtol=0, atol=1e-9 * np.abs(fv).max())


@needs_fixture
def test_child_meteo_known_bad_unmatched_fluxes_differ():
    """Known-bad: the raw child fluxes balanced without the Match are not the model's on the
    parent faces -- the face test above can fail."""
    p, c = _pair()
    rf = ZM.Refinement.from_meta(c.meta, p.meta)
    pu = c.interior("mfu1").copy()
    assert not np.array_equal(pu[:: rf.ri], c.interior("pu1")[:: rf.ri])


@needs_fixture
def test_replay_refuses_a_parent_of_another_interval():
    from pydyn import replay as RZ

    p, c = _pair()
    zm = RZ.build_tree({1: p, 2: c})
    meteo = RZ.Meteo(zm, "auto", own=False)
    with pytest.raises(ValueError, match="no meteo event of its parent"):
        meteo.event(c)
    meteo.event(p)
    reports, _ = meteo.event(c)
    assert {r["name"] for r in reports} == {"pu1", "pv1", "pw1", "pu2", "pv2", "pw2"}


# ------------------------------------------------------------------ regrid (the 1x1, 137-level ingestion)
def test_fill_levels_sums_each_layer_top_down_and_copies_half_levels():
    from pydyn.chemfields import ECHLEVS
    from pydyn.regrid import fill_levels

    rng = _rng()
    full = rng.normal(size=(2, 3, 137))  # bottom-up, as the file
    half = rng.normal(size=(2, 3, 138))
    out = fill_levels(full, "sum")
    assert out.shape == (2, 3, 34)
    np.testing.assert_allclose(out.sum(-1), full.sum(-1), rtol=1e-12)
    # layer 1 is the file's three lowest levels, accumulated from the third down
    assert np.array_equal(out[..., 0], ((0.0 + full[..., 2]) + full[..., 1]) + full[..., 0])
    w = fill_levels(half, "bottom")
    assert w.shape == (2, 3, 35)
    assert np.array_equal(w[..., 0], half[..., 0]) and np.array_equal(w[..., -1], half[..., -1])
    assert np.array_equal(w[..., 1], half[..., 137 - ECHLEVS[1]])


def test_fill_grid_combines_faces_and_cells_and_copies_a_window():
    from pydyn.regrid import fill_grid

    rng = _rng()
    u = rng.normal(size=(361, 180, 2))
    v = rng.normal(size=(360, 181, 2))
    n = rng.normal(size=(360, 180, 2))
    g32 = LL.region(-180.0, -90.0, 3.0, 2.0, 120, 90)
    fu = fill_grid(u, "u", "sum", g32)
    fv = fill_grid(v, "v", "sum", g32)
    fn = fill_grid(n, "n", "sum", g32)
    assert fu.shape == (121, 90, 2) and fv.shape == (120, 91, 2) and fn.shape == (120, 90, 2)
    assert np.array_equal(fu[5, 7], (0.0 + u[15, 14]) + u[15, 15])
    assert np.array_equal(fv[5, 7], ((0.0 + v[15, 14]) + v[16, 14]) + v[17, 14])
    np.testing.assert_allclose(fn.sum((0, 1)), n.sum((0, 1)), rtol=1e-12)
    w = rng.uniform(1, 2, 180)
    const = np.full((360, 180), 7.0)
    np.testing.assert_allclose(fill_grid(const, "n", "area-aver", g32, w), 7.0, rtol=1e-15)
    nam = LL.region(-132.0, 20.0, 1.0, 1.0, 72, 44)
    assert np.array_equal(fill_grid(n, "n", "sum", nam), n[48:120, 110:154])
    assert np.array_equal(fill_grid(u, "u", "sum", nam), u[48:121, 110:154])
    off = LL.region(-170.0, -60.0, 3.0, 2.0, 10, 5)  # a window whose stride does not divide the globe
    want = sum((0.0 + n[13 + i, 34 + j] for j in range(2) for i in range(3)), 0.0)  # j outer, i inner
    assert np.array_equal(fill_grid(n, "n", "sum", off)[1, 2], want)


def test_mass_aver_keeps_a_constant_field():
    from pydyn.regrid import fill_grid, fill_levels

    rng = _rng()
    a = np.concatenate([[0.0], np.cumsum(rng.uniform(50, 150, 137))])[::-1].copy()  # bottom-up, top 0
    a = a - a[-1]
    b = np.linspace(1.0, 0.0, 138)
    ps = rng.uniform(9e4, 1.02e5, (360, 180))
    T = np.full((360, 180, 137), 250.0)
    dp = ((a[1:] - a[:-1]) + (b[1:] - b[:-1]) * ps[..., None]) * rng.uniform(1, 2, 180)[None, :, None]
    g32 = LL.region(-180.0, -90.0, 3.0, 2.0, 120, 90)
    Th = fill_grid(T, "n", "weight", g32, dp)
    np.testing.assert_allclose(Th, 250.0, rtol=1e-14)
    from pydyn.chemfields import ECHLEVS

    idx = [ECHLEVS[0] - h for h in ECHLEVS]
    psm = fill_grid(ps, "n", "area-aver", g32, np.ones(180))
    Tv = fill_levels(Th, "mass-aver", ps=psm, levX=(a, b), lev=(a[idx], b[idx]))
    np.testing.assert_allclose(Tv, 250.0, rtol=1e-12)


# ------------------------------------------------------------------ Calc_Kzz's step (dkg)
def test_spm_is_the_read_interval_at_the_step_middle():
    """Meteo_Setup_Other: spm = alfa1 sp1 + alfa2 sp2, sp1 / sp2 at the read interval's ends,
    the weights at the middle of the step (tdyn, tdyn + ndyn), which Check_CFL may cut."""
    from datetime import datetime

    from pydyn import meteo

    met = meteo.Meteo.__new__(meteo.Meteo)
    sp = {datetime(2019, 1, 1, 1): np.full((2, 2), 1.0e5), datetime(2019, 1, 1, 2): np.full((2, 2), 1.01e5)}
    met.surface_pressure = sp.__getitem__
    read = (datetime(2019, 1, 1, 1), datetime(2019, 1, 1, 2))
    step = (datetime(2019, 1, 1, 1), datetime(2019, 1, 1, 1, 30))
    np.testing.assert_allclose(met.spm(step, read), 0.75e5 + 0.25 * 1.01e5)
    np.testing.assert_allclose(met.spm(read), 0.5e5 + 0.5 * 1.01e5)


def test_cube_defaults_to_gfortrans_expansion():
    from pydyn.ops import libm

    x = _rng().uniform(0.0, 3.0, 1000)
    if libm.path is None:
        assert np.array_equal(libm.cube(x), (x * x) * x)
    else:  # an Intel libimf: __powr8i4, within an ulp of it
        np.testing.assert_allclose(libm.cube(x), (x * x) * x, rtol=3e-16)
