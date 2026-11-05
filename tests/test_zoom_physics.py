"""s and c on a zoom tree and their inputs per region (`PYD-3` items 4-5): the cells a region
owns, the zp_replay physics events, and the emission, the initial field and the chemistry's
fields of each region from the input files."""


import numpy as np
import pytest

from pydyn import replay
from pydyn.dump import load_events
from pydyn.grid import LL, ONE_BY_ONE, block_sum

from common import ZOOM_PHYS_DIR, needs


def _rng():
    return np.random.default_rng(7)


# ------------------------------------------------------------------ s and c on a zoom tree (item 4)
def _tree_state():
    from pydyn import zoom as Z

    glb = Z.Region("glb", 24, 18, xcyc=1, touch_sp=1, touch_np=1)
    child = Z.Region("child", 24, 16, xbeg=-105, xend=15, ybeg=-30, yend=50, xref=3, yref=2, tref=2, parent=1)
    zm = Z.Zoom([glb, child], 3, np.linspace(2000.0, 0.0, 4), np.linspace(1.0, 0.0, 4))
    st = Z.ZoomState.zeros(zm, 1)
    rng = _rng()
    for r in zm.regions():
        for k in ("rm", "rxm", "rym", "rzm"):
            st[r][k].data[...] = rng.uniform(1, 2, st[r][k].data.shape)
    return Z, zm, st


def test_sources_and_chemistry_act_on_the_owned_cells_only():
    Z, zm, st = _tree_state()
    for r in zm.regions():
        g, own = zm.g(r), Z.owned(zm, r)
        before = {k: st[r][k].data.copy() for k in ("rm", "rxm", "rym", "rzm")}
        x = np.full((g.im, g.jm), 0.5)
        Z.sources_region(zm, st, r, [(x, 1, 1)])
        rm, rzm = st[r]["rm"][1 : g.im, 1 : g.jm, 1, 1], st[r]["rzm"][1 : g.im, 1 : g.jm, 1, 1]
        assert np.array_equal(rm[own], before["rm"][2:-2, 2:-2, 0, 0][own] + 0.5)
        assert np.array_equal(rm[~own], before["rm"][2:-2, 2:-2, 0, 0][~own])
        assert np.array_equal(rzm[~own], before["rzm"][2:-2, 2:-2, 0, 0][~own])
        sc = Z.scope(zm, r)
        shape = (sc[0].stop - sc[0].start, sc[1].stop - sc[1].start, zm.lm)
        mid = st[r]["rm"][1 : g.im, 1 : g.jm, 1 : zm.lm, 1].copy()
        Z.chemistry_region(zm, st, r, np.full(shape, 0.5), (1,))
        rm2 = st[r]["rm"][1 : g.im, 1 : g.jm, 1 : zm.lm, 1]
        assert np.array_equal(rm2[~own], mid[~own]) and np.array_equal(rm2[own], mid[own] * 0.5)
    # the parent owns nothing inside the child's footprint (blank_zoom_region leaves it the
    # footprint's ring), the child nothing on its edge ring
    assert not Z.owned(zm, 1)[zm.ibeg[2] : zm.iend[2] - 1, zm.jbeg[2] : zm.jend[2] - 1].any()
    assert Z.owned(zm, 1)[zm.ibeg[2] - 1, zm.jbeg[2] - 1]
    assert not Z.owned(zm, 2)[0].any() and not Z.owned(zm, 2)[:, -1].any()


# ------------------------------------------------------------------ the zp_replay triple (MPI physics run)
ZP = ZOOM_PHYS_DIR
needs_zp = needs(ZP, "PYDYN_ZOOM_PHYS_DIR")


def _zp():
    from pydyn import replay as RZ

    events = load_events(ZP)
    by = {e.seq: e for e in events}
    return RZ, events, RZ.build_tree({1: by[1], 2: by[2]})


@needs_zp
def test_child_sources_are_the_models():
    RZ, events, zm = _zp()
    k = next(i for i, e in enumerate(events) if e.tag == "bef_s" and e.region == 2)
    aft, inner = replay.span(events, k)
    reports = RZ.replay_operator(zm, events[k], aft, None, inner)
    assert all(r["equal"] for r in reports), [r["name"] for r in reports if not r["equal"]]


@needs_zp
def test_child_chemistry_applied_the_as_bound_losses():
    """The MPI build's child decays with ch4_loss indexed from 1 by its own i, j
    (analysis/ch4_loss_bounds.py); the intended losses do not reproduce it.  Off the model's
    libimf the exp is a few ulp off, hence the fractions."""
    from analysis.ch4_loss_bounds import decay_as_bound, decay_intended, score

    _, events, zm = _zp()
    k = next(i for i, e in enumerate(events) if e.tag == "bef_c" and e.region == 2)
    aft, (chem_ev,) = replay.span(events, k)
    loss, dt = chem_ev["loss"].data, chem_ev.meta["dt_sec"]
    good = score(zm, events[k], aft, decay_intended(loss, dt))
    dec, inside = decay_as_bound(loss, dt, zm.isr[2], zm.jsr[2])
    bound = score(zm, events[k], aft, dec, inside)
    assert bound[0] / bound[1] > 0.99 and good[0] / good[1] < 0.05


# ------------------------------------------------------------------ s, c and the IC per region (item 5)
NAM = LL.region(-132.0, 20.0, 1.0, 1.0, 72, 44)  # nam1x1


def _ic_file(nx, ny, nlev=3):
    """An istart-66 field on a global nx x ny grid, every cell distinct."""
    dlon, dlat = 360.0 / nx, 180.0 / ny
    lon = -180.0 + dlon / 2 + np.arange(nx) * dlon
    lat = -90.0 + dlat / 2 + np.arange(ny) * dlat
    field = 1.8 + _rng().uniform(0.0, 0.1, (nx, ny, nlev))
    a, b = np.linspace(0.0, 1.0e4, nlev + 1), np.linspace(1.0, 0.0, nlev + 1)
    return field, lon, lat, a, b


def test_initial_remap_relates_like_fillgrid():
    """Fill3D per region (initexit, istart 66): the glb3x2 file is copied onto glb3x2,
    distributed over nam1x1 (each file cell's value in its 3 x 2 children), combined onto
    glb6x4; a region off the file's cell edges is refused."""
    from pydyn import initial
    from pydyn.grid import AE, block_mean, cell_areas

    field, lon, lat, a, b = _ic_file(120, 90)
    g32 = LL.region(-180.0, -90.0, 3.0, 2.0, 120, 90)
    assert np.array_equal(initial.remap(field, lon, lat, a, b, a, b, g32), field)
    child = initial.remap(field, lon, lat, a, b, a, b, NAM)
    win = field[16:40, 55:77]  # parent cells ibeg..iend = 17..40, jbeg..jend = 56..77
    assert np.array_equal(child, np.repeat(np.repeat(win, 3, 0), 2, 1))
    w = cell_areas(120, 90, 3.0, 2.0, lat[0]) * AE**2
    coarse = initial.remap(field, lon, lat, a, b, a, b)
    assert np.array_equal(coarse, np.stack([block_mean(field[..., k], w, 2, 2) for k in range(3)], -1))
    with pytest.raises(NotImplementedError):
        initial.remap(field, lon, lat, a, b, a, b, LL.region(-131.5, 20.0, 1.0, 1.0, 72, 44))


def test_emission_window_is_the_regions_cells():
    """emission_ct onto lli(region): nam1x1 the 1x1 window itself ('copy'), glb3x2 the 3 x 2
    block sums, row by row."""
    from pydyn import emis

    raw = _rng().uniform(0.0, 1.0, (360, 180))
    whole = emis.monthly_field(raw, 2678400.0, ll=ONE_BY_ONE)
    child = emis.monthly_field(raw, 2678400.0, ll=NAM)
    assert np.array_equal(child, whole[48:120, 110:154])
    g32 = LL.region(-180.0, -90.0, 3.0, 2.0, 120, 90)
    assert np.array_equal(emis.monthly_field(raw, 2678400.0, ll=g32), block_sum(whole, 3, 2))
