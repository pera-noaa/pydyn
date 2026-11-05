"""Zoom nesting (`PYD-3`), tier 1: pydyn.zoom against the production zoom routines compiled
unmodified in fortran/harness_zoom (docs/state.md §3d).

Every case fills a multi-region state, runs one routine -- or Proces_Region, a whole step,
or Check_CFL -- in Fortran and in Python, and requires every array of every region, halos
included, to be identical to the bit, and the CFL diagnostics nloop_max / xi (or Check_CFL's
ndyn, n) too.  Needs `make -C fortran zoom` (harness_zoom2, harness_zoom3); the geometry tests
need nothing.
"""

import numpy as np
import pytest

from pydyn import harness, synth
from pydyn import zoom as Z
from pydyn.kernels import CFLError

needs_harness = pytest.mark.skipif(not harness.zoom_available(), reason="harness_zoom not built")

NDYN = 3600


# ------------------------------------------------------------------ configurations
def _levels(lm):
    return np.linspace(2000.0, 0.0, lm + 1), np.linspace(1.0, 0.0, lm + 1)


def two(lm=5):
    """Parent 24x18 (15 x 10 deg), one child: xref 3, yref 2, tref 2 -- nam1x1's factors."""
    glb = Z.Region("glb", 24, 18, xcyc=1, touch_sp=1, touch_np=1)
    child = Z.Region("child", 24, 16, xbeg=-105, xend=15, ybeg=-30, yend=50, xref=3, yref=2, tref=2, parent=1)
    return Z.Zoom([glb, child], lm, *_levels(lm))


def three(lm=4):
    """A grandchild: region 2 at 2x2 / tref 2, region 3 at 3x2 / tref 2 relative to it."""
    glb = Z.Region("glb", 24, 18, xcyc=1, touch_sp=1, touch_np=1)
    mid = Z.Region("mid", 16, 16, xbeg=-90, xend=30, ybeg=-40, yend=40, xref=2, yref=2, tref=2, parent=1)
    fine = Z.Region("fine", 30, 16, xbeg=-75, xend=0, ybeg=-20, yend=20, xref=6, yref=4, tref=4, parent=2)
    return Z.Zoom([glb, mid, fine], lm, *_levels(lm))


def production(lm=4):
    """glb3x2 + nam1x1 as base/trunk/rc/pycasso-tm5-regions-glb3x2.rc defines them (the
    reduced grid of glb3x2 left off: harness_zoom stubs redgridZoom)."""
    glb = Z.Region("glb3x2", 120, 90, xcyc=1, touch_sp=1, touch_np=1)
    nam = Z.Region("nam1x1", 72, 44, xbeg=-132, xend=-60, ybeg=20, yend=64, xref=3, yref=2, tref=2, parent=1)
    return Z.Zoom([glb, nam], lm, *_levels(lm))


def eur(lm=3):
    """base/trunk/src/dims_grid__eur.F90: glb6x4, eur300x200 (tref 2), eur100x100 (tref 4)."""
    glb = Z.Region("glb600x400", 60, 45, xcyc=1, touch_sp=1, touch_np=1)
    e3 = Z.Region("eur300x200", 30, 36, xbeg=-36, xend=54, ybeg=2, yend=74, xref=2, yref=2, tref=2, parent=1)
    e1 = Z.Region("eur100x100", 60, 54, xbeg=-21, xend=39, ybeg=12, yend=66, xref=6, yref=4, tref=4, parent=2)
    return Z.Zoom([glb, e3, e1], lm, *_levels(lm))


CONFIGS = {"two": two, "three": three}


# ------------------------------------------------------------------ plumbing
def _assert_equal(zoom, fields, st):
    reports = [
        harness.compare(fields[r][name].data, st[r][name].data, f"{name}_r{r}")
        for r in zoom.regions()
        for name in harness.ZOOM_OUT
    ]
    assert all(rep["equal"] for rep in reports), harness.format_report([r for r in reports if not r["equal"]])


def _both(tmp_path, zoom, st, kernel, target, python, conv=None):
    """Run `kernel` in Fortran and `python(py_state)` on a copy; assert bitwise parity."""
    harness.write_zoom_case(tmp_path, kernel, zoom, st, target, ndyn=NDYN, conv=conv)
    fields, stats = harness.run_zoom(tmp_path, zoom, st)
    py = st.copy()
    python(py)
    _assert_equal(zoom, fields, py)
    return stats


# ------------------------------------------------------------------ the region tree
def test_production_tree():
    zm = production()
    assert (zm.ibeg[2], zm.iend[2], zm.jbeg[2], zm.jend[2]) == (17, 40, 56, 77)
    assert (zm.isr[2], zm.ier[2], zm.jsr[2], zm.jer[2]) == (4, 69, 3, 42)
    assert zm.children == {0: [], 1: [2], 2: []}
    assert zm.splitorderzoom[2] == "xyzvsc" + "csvzyx" + "csvzyx" + "xyzvsc"
    # the parent owns its footprint's outermost ring; the child's ring is parent-fed
    assert (zm.zoomed[1] == 2).sum() == (40 - 17 - 1) * (77 - 56 - 1)
    ring = zm.edge[2] == -1
    assert ring.sum() == 72 * 44 - (72 - 6) * (44 - 4)


def test_eur_tree_is_the_survey_one():
    """dynamic_core_survey.md §2.7: eur300x200 at parent cells 25:39 x 24:41, eur100x100 at
    region-2 cells 6:25 x 6:32.  The grandchild's 48 operators: every 6-block of its
    parent's order followed by that block reversed (define_splitorderzoom, zoom_mode 1)."""
    zm = eur()
    assert (zm.ibeg[2], zm.iend[2], zm.jbeg[2], zm.jend[2]) == (25, 39, 24, 41)
    assert (zm.ibeg[3], zm.iend[3], zm.jbeg[3], zm.jend[3]) == (6, 25, 6, 32)
    fwd, rev = "xyzvsc", "csvzyx"
    assert zm.splitorderzoom[2] == fwd + rev + rev + fwd
    assert zm.splitorderzoom[3] == fwd + rev + rev + fwd + rev + fwd + fwd + rev


@pytest.mark.parametrize(
    ("child", "why"),
    [
        ({"touch_np": 1}, "pole-touching"),
        ({"xbeg": -180, "xend": -60, "im": 120}, "halo"),
    ],
)
def test_unsupported_trees_refuse(child, why):
    kw = {"im": 72, "jm": 44, "xbeg": -132, "xend": -60, "ybeg": 20, "yend": 64,
          "xref": 3, "yref": 2, "tref": 2, "parent": 1}
    kw.update(child)
    glb = Z.Region("glb3x2", 120, 90, xcyc=1, touch_sp=1, touch_np=1)
    with pytest.raises(NotImplementedError, match=why):
        Z.Zoom([glb, Z.Region("c", **kw)], 3)


# ------------------------------------------------------------------ the exchange routines
EXCHANGE = [
    ("put_xedges", Z.put_xedges),
    ("put_yedges", Z.put_yedges),
    ("mix_edges", Z.mix_edges),
    ("update_parent", Z.update_parent),
    ("m2phlb", Z.m2phlb),
    ("m2phlb1", Z.m2phlb1),
    ("coarsen_ambm", Z.coarsen_ambm),
    ("coarsen_region", Z.coarsen_region),
]


@needs_harness
@pytest.mark.parametrize("config", sorted(CONFIGS))
@pytest.mark.parametrize(("kernel", "fn"), EXCHANGE, ids=[k for k, _ in EXCHANGE])
def test_exchange(tmp_path, config, kernel, fn):
    zm = CONFIGS[config]()
    for target in zm.regions():
        if kernel in ("put_xedges", "put_yedges", "coarsen_ambm") and not zm.children[target]:
            continue
        if kernel in ("mix_edges", "update_parent", "m2phlb", "coarsen_region") and target == 1:
            continue
        if kernel == "m2phlb1" and target != 1:
            continue
        st = synth.zoom_state(zm, np.random.default_rng(10 * target))
        _both(tmp_path / f"r{target}", zm, st, kernel, target, lambda py, t=target: fn(zm, py, t))


@needs_harness
@pytest.mark.parametrize("config", sorted(CONFIGS))
def test_dynam0_and_convec(tmp_path, config):
    """dt = ndyn / (2 tref) per region; convec only on the region's own interior columns
    (compiled under with_zoom, which pycasso defines for every multi-region build)."""
    zm = CONFIGS[config]()
    for target in zm.regions():
        rng = np.random.default_rng(target)
        st = synth.zoom_state(zm, rng)
        _both(
            tmp_path / f"d{target}", zm, st, "dynam0", target, lambda py, t=target: Z.dynam0(zm, py, t, NDYN)
        )
        conv = synth.zoom_convection(zm, st, rng)[target]
        _both(
            tmp_path / f"c{target}", zm, st, "convec", target,
            lambda py, t=target, c=conv: Z.convec_region(zm, py, t, c, NDYN), conv=conv,
        )


# ------------------------------------------------------------------ advection at every slot
OPS = {"x": Z.advect_x_zoom, "y": Z.advect_y_zoom, "z": Z.advect_z_zoom}


def _slots(config):
    zm = CONFIGS[config]()
    return [
        (config, target, axis, status)
        for target in zm.regions()
        for axis in "xyz"
        for status, op in enumerate(zm.splitorderzoom[target])
        if op == axis
    ]


SLOTS = _slots("two") + _slots("three")


@needs_harness
@pytest.mark.parametrize(("config", "target", "axis", "status"), SLOTS, ids=lambda v: str(v))
@pytest.mark.parametrize("cfl", [0.4, 3.5])
def test_advection_slot(tmp_path, config, target, axis, status, cfl):
    """advect{x,y,z}zoom(target) at one position of its splitorderzoom: the scope the block
    order implies, the outer interface fluxes zeroed in the second half of the parent step,
    put_x/yedges to the children first, uncompress and mix_edges after.  At Courant 3.5 the
    kernels sub-cycle; where the Fortran refuses (escape_tm), the Python must refuse too."""
    zm = CONFIGS[config]()
    zm.status[target] = status
    st = synth.zoom_state(zm, np.random.default_rng(status + 100 * target), cfl=cfl)
    harness.write_zoom_case(tmp_path, axis, zm, st, target, ndyn=NDYN)
    py, stats = st.copy(), {}
    try:
        fields, fstats = harness.run_zoom(tmp_path, zm, st)
    except harness.HarnessError as err:
        assert "nloop too high" in err.stderr or "too many CFL" in err.stderr, err.stderr
        with pytest.raises(CFLError):
            OPS[axis](zm, py, target, stats=stats)
        return
    OPS[axis](zm, py, target, stats=stats)
    _assert_equal(zm, fields, py)
    k = "xyz".index(axis)
    assert fstats[target]["nloop_max"][k] == stats["nloop_max"]
    assert fstats[target]["xi"][k] == stats["xi"]


# ------------------------------------------------------------------ Setup_MassFlow and the driver
DRIVER = {"two": two, "three": three, "production": production, "eur": eur}


@needs_harness
@pytest.mark.parametrize("config", sorted(DRIVER))
@pytest.mark.parametrize("cfl", [0.3, 0.9])
def test_step(tmp_path, config, cfl):
    """Setup_MassFlow's region loop, then one Proces_Region(1), then a whole ndyn step from
    the same start: the recursion over the tree, do_steps' exits after a child-relevant xyz,
    put_x/yedges at block ends, update_parent and m2phlb -- every array of every region and
    the CFL diagnostics bitwise."""
    zm = DRIVER[config]()
    rng = np.random.default_rng(1)
    st = synth.zoom_meteo(zm, rng, cfl=cfl)
    _both(tmp_path / "massflow", zm, st, "massflow", 1, lambda py: Z.setup_massflow(zm, py, NDYN))
    Z.setup_massflow(zm, st, NDYN)
    conv = synth.zoom_convection(zm, st, rng)
    for kernel in ("proces_region", "step"):
        zm.status = [0] * (zm.nregions + 1)
        harness.write_zoom_case(tmp_path / kernel, kernel, zm, st, 1, ndyn=NDYN, conv=conv)
        fields, fstats = harness.run_zoom(tmp_path / kernel, zm, st)
        py, stats = st.copy(), {}
        if kernel == "step":
            Z.step(zm, py, conv, NDYN, stats=stats)
        else:
            Z.proces_region(zm, py, 1, conv, NDYN, stats=stats)
        _assert_equal(zm, fields, py)
        for r in zm.regions():
            assert list(fstats[r]["nloop_max"]) == stats[r]["nloop_max"], (kernel, r)
            assert list(fstats[r]["xi"]) == stats[r]["xi"], (kernel, r)
        expected = {"proces_region": 1, "step": 2}[kernel]
        assert zm.status[1] == 6 * expected
        assert all(zm.status[r] == 6 * expected * zm.g(r).tref for r in zm.regions())


# ------------------------------------------------------------------ Check_CFL on a region tree
CFL_CASES = [
    # (tree, Courant, ndyn, n, nread)
    ("two", 0.3, 3600, 1, 3600),
    ("two", 0.9, 3600, 1, 3600),
    ("two", 1.2, 3600, 3, 10800),
    ("three", 2.5, 3600, 1, 3600),  # no valid time step: both refuse
    ("three", 1.4, 3600, 1, 3600),
    ("production", 0.6, 3600, 1, 3600),
    ("production", 1.2, 3600, 1, 3600),
    ("eur", 0.9, 3600, 1, 3600),
]


@needs_harness
@pytest.mark.parametrize(("config", "cfl", "ndyn", "n", "nread"), CFL_CASES, ids=lambda v: str(v))
def test_check_cfl(tmp_path, config, cfl, ndyn, n, nread):
    """advectm_cfl.F90's Check_CFL (Init_CFL with the cache off) against zoom.check_cfl_zoom:
    the (ndyn, n) it settles on, and every array it leaves -- masses and fluxes restored and
    rescaled, sp / phlb as the rehearsal's m2phlb wrote them -- bitwise.  Where the Fortran
    finds no valid time step at all, the Python must refuse as well."""
    zm = DRIVER[config]()
    st = synth.zoom_meteo(zm, np.random.default_rng(7), cfl=cfl)
    harness.write_zoom_case(tmp_path, "check_cfl", zm, st, 1, ndyn=ndyn, n_steps=n, nread=nread)
    py = st.copy()
    try:
        fields, _ = harness.run_zoom(tmp_path, zm, st)
    except harness.HarnessError as err:
        assert "no valid timestep" in err.stderr, err.stderr
        with pytest.raises(ValueError):
            Z.check_cfl_zoom(zm, py, ndyn, n, nread)
        return
    got = Z.check_cfl_zoom(zm, py, ndyn, n, nread)
    assert fields["cfl"] == got[:2]
    _assert_equal(zm, fields, py)


def test_reduced_parent_x_halo_is_the_uniform_row():
    """redgridZoom.F90 red2uni ends every reduced row with q(0,j,l) = q(im,j,l) on the
    uniform row, over the reduced-row value dynamu wrote there.  Tier 1 stubs the reduced grid,
    so this was first seen (and the Python fixed) bitwise against a glb3x2 + nam1x1 dump
    (docs/state.md §3f); this pins it without one."""
    from pydyn.redgrid import RedGrid

    lm = 4
    grid = RedGrid(24, 18, (1, 2, 17, 18), (4, 2, 2, 4))
    glb = Z.Region("glb", 24, 18, xcyc=1, touch_sp=1, touch_np=1, redgrid=grid)
    child = Z.Region("child", 24, 16, xbeg=-105, xend=15, ybeg=-30, yend=50, xref=3, yref=2, tref=2, parent=1)
    zm = Z.Zoom([glb, child], lm, *_levels(lm))
    st = synth.zoom_meteo(zm, np.random.default_rng(11), cfl=0.4)
    Z.setup_massflow(zm, st, NDYN)
    Z.advect_x_zoom(zm, st, 1)
    for j0 in grid.rows:
        for name in ("m", "rm", "rxm", "rym", "rzm"):
            q = st[1][name]
            assert np.array_equal(q[0, j0 + 1], q[24, j0 + 1]), (name, j0 + 1)
