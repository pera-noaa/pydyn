"""pydyn.gpu against the reference, bitwise, on the emulation backend (the device's own source
compiled for the host with contraction off): every kernel on synthetic inputs, a whole step
and Check_CFL of a reduced-grid region, the three laptop sets replayed on both sides with
their fingerprints matched, and the checks failing when contraction is let back in.  The
`cuda` backend is the same tests with PYDYN_DEVICE=cuda where a GPU is (the default is emu)."""

import os
import platform
import shutil

import numpy as np
import pytest

from pydyn import chem, diffusion, kernels, lapack, ops, synth
from pydyn import zoom as Z
from pydyn.redgrid import RedGrid

from common import CFL_DUMP_DIR, DUMP_DIR, FREE_DUMP_DIR, ZOOM_PHYS_DIR, needs, twins

DEVICE = os.environ.get("PYDYN_DEVICE", "emu")
pytestmark = pytest.mark.skipif(
    DEVICE == "emu" and shutil.which(os.environ.get("CXX", "c++")) is None,
    reason="no C++ compiler for the emu backend",
)


@pytest.fixture(scope="module")
def be():
    from pydyn.gpu import backend

    return backend.set_backend(DEVICE)


@pytest.fixture
def twin_libm():
    """The reference's libm routed through the twin library for the test, then put back."""
    with twins():
        yield


def _equal(a, b):
    """Bitwise as harness.compare means it: == or both NaN."""
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and bool(((a == b) | (np.isnan(a) & np.isnan(b))).all())


# ------------------------------------------------------------------ residency
def test_emu_arrays_refuse_the_host(be):
    if be.name != "emu":
        pytest.skip("the emulation backend's own check")
    from pydyn.gpu.emu import ResidencyError

    d = be.to_device(np.ones(4))
    for bad in (lambda: d + np.ones(4), lambda: np.ones(4) * d, lambda: np.asarray(d)):
        with pytest.raises(ResidencyError):
            bad()
    with pytest.raises(ResidencyError):
        d[...] = np.zeros(4)
    d[...] = 2.0  # scalars are fine, as in CuPy
    assert _equal(be.to_host(d + 1.0), np.full(4, 3.0))


# ------------------------------------------------------------------ the twins
def test_the_device_has_every_body_the_driver_runs():
    """zoom's driver runs st.lib's bodies: pydyn.gpu.zoom must have each of zoom.host's."""
    from pydyn.gpu import zoom as gz
    from pydyn.zoom import host

    assert [n for n in host.BODIES if not callable(getattr(gz, n, None))] == []
    assert set(host.BODIES) <= set(gz.__all__)


def test_pmath_is_within_an_ulp_of_the_system_libm():
    import ctypes
    import math

    from pydyn.gpu.backend import host_library
    from pydyn.harness import _ulp_distance

    lib = ctypes.CDLL(str(host_library()))
    for name in ("exp", "log", "pow"):
        getattr(lib, name).restype = ctypes.c_double
        getattr(lib, name).argtypes = [ctypes.c_double] * (2 if name == "pow" else 1)
    rng = np.random.default_rng(0)
    cases = {
        "exp": [(x,) for x in rng.uniform(-700, 700, 3000)],
        "log": [(x,) for x in np.exp(rng.uniform(-700, 700, 3000))],
        "pow": [(x, y) for x, y in zip(rng.uniform(0.1, 10, 3000), rng.uniform(-5, 5, 3000))]
        + [(x, 1.0 / 3.0) for x in rng.uniform(1e-6, 1e3, 1000)],
    }
    for name, args in cases.items():
        ours = np.array([getattr(lib, name)(*a) for a in args])
        theirs = np.array([getattr(math, name)(*a) for a in args])
        assert _ulp_distance(ours, theirs).max() <= 1, name


def test_twin_lapack_inverts(be):
    from pydyn.gpu.backend import host_library

    rng = np.random.default_rng(1)
    a = np.eye(12)[None] + 0.1 * rng.standard_normal((20, 12, 12))
    inv = lapack.LapackInverse(str(host_library()))(a)
    np.testing.assert_allclose(
        np.einsum("nij,njk->nik", inv, a), np.broadcast_to(np.eye(12), a.shape), atol=1e-12
    )


# ------------------------------------------------------------------ kernels
IM, JM, LM, NT = 12, 9, 7, 3


@pytest.mark.parametrize("cfl", [0.4, 1.3, 3.2])
def test_advection_kernels(be, cfl):
    from pydyn.gpu import kernels as G

    rng = np.random.default_rng(int(cfl * 10))
    s = synth.state(rng, IM, JM, LM, NT)
    fluxes = synth.fluxes(rng, s.m, cfl=cfl)
    for axis, ref_kernel in enumerate((kernels.advect_x, kernels.advect_y, kernels.advect_z)):
        ref, f_ref = s.copy(), fluxes[axis].copy()
        try:
            ref_kernel(ref.m, ref.rm, ref.rxm, ref.rym, ref.rzm, f_ref)
        except kernels.CFLError:
            with pytest.raises(kernels.CFLError):
                _device_advect(be, G, axis, s, fluxes[axis])
            continue
        got = _device_advect(be, G, axis, s, fluxes[axis])
        for name, a, b in zip(("m", "rm", "rxm", "rym", "rzm", "flux"), got, (*_fields(ref), f_ref)):
            assert _equal(a, b), ("xyz"[axis], name)


def _fields(s):
    return (s.m, s.rm, s.rxm, s.rym, s.rzm)


def _device_advect(be, G, axis, s, flux):
    d = [be.to_device(q, "seed") for q in (*_fields(s), flux)]
    if axis == 0:
        G.advect_x_rows(*d, be.to_device(np.full(JM, IM, np.int32), "seed"))
    elif axis == 1:
        G.advect_y(*d)
    else:
        G.advect_z(*d)
    return [be.to_host(q) for q in d]


@pytest.mark.parametrize("inverse", ["crout", "lapack"])
def test_convec(be, inverse):
    from pydyn.gpu import kernels as G
    from pydyn.gpu.backend import host_library

    rng = np.random.default_rng(2)
    s = synth.state(rng, IM, JM, 12, NT)
    for lmax, kw in ((12, {}), (9, {"strong_detu": True}), (5, {"lfs_above_top": True})):
        conv = synth.convection(rng, s.m, lmax, 1800.0, **kw)
        ref = s.copy()
        ref_inverse = "crout" if inverse == "crout" else lapack.LapackInverse(str(host_library()))
        kernels.convec(*_fields(ref), **conv, dt=1800.0, lmax=lmax, inverse=ref_inverse)
        d = [be.to_device(q, "seed") for q in _fields(s)]
        dc = {
            k: be.to_device(v, "input", np.int32 if k.startswith("cloud") else None) for k, v in conv.items()
        }
        G.convec(*d, dc, 1800.0, lmax, be.to_device(np.arange(IM * JM, dtype=np.int32)), inverse=inverse)
        for a, b in zip(d, _fields(ref)):
            assert _equal(be.to_host(a), b), (inverse, lmax)


def test_dynam0_and_cluster_sum(be):
    from pydyn.gpu import kernels as G

    rng = np.random.default_rng(3)
    pu, pv = rng.normal(size=(IM + 1, JM, LM)) * 1e9, rng.normal(size=(IM, JM + 1, LM)) * 1e9
    bt = np.linspace(1.0, 0.0, LM + 1)
    ref = kernels.fluxes_from_meteo(pu, pv, bt, 900.0)
    got = [be.xp.zeros(x.shape) for x in ref]
    G.fluxes_from_meteo(be.to_device(pu), be.to_device(pv), be.to_device(bt), 900.0, *got)
    assert all(_equal(be.to_host(g), r) for g, r in zip(got, ref))
    q = rng.normal(size=(60, LM, NT))
    assert _equal(
        be.to_host(G.cluster_sum(be.to_device(q), 3, 20)), ops.seqsum(ops.clusters(q, 3, 20), axis=1)
    )


@pytest.mark.parametrize(("lm", "lmax_conv"), [(8, 6), (34, 26)])
def test_calc_kzz(be, twin_libm, lm, lmax_conv):
    from pydyn.gpu import diffusion as GD

    rng = np.random.default_rng(lm)
    args, grid = synth.kzz_inputs(rng, IM, JM, lm=lm, lmax_conv=lmax_conv)
    dkg, parts = diffusion.calc_kzz(*args, **grid)
    pblh = be.xp.zeros((IM, JM))
    assert _equal(be.to_host(GD.calc_kzz(*args, **grid, pblh=pblh)), dkg)
    assert _equal(be.to_host(pblh), parts["pblh"])


def _chem_inputs(rng, im, jm, lm, ns=12, ntrop=5):
    sp = rng.uniform(9e4, 1.02e5, (im, jm))
    phlb = sp[..., None] * np.linspace(1.0, 0.0, lm + 1)[None, None] ** 1.5
    press = np.sort(rng.uniform(0.5, 1000.0, (jm, ns)), axis=1)[:, ::-1]  # hPa, descending
    pl = np.sort(rng.uniform(1e4, 1e5, (im, jm, ntrop)), axis=-1)[..., ::-1]
    return {
        "temper": rng.uniform(200.0, 300.0, (im, jm, lm)),
        "phlb": phlb,
        "lat": np.radians(-90 + 180.0 / jm * (np.arange(jm) + 0.5)),
        "strat_press": press,
        "strat_oh": rng.uniform(0, 1e-6, (jm, ns)),
        "strat_cl": rng.uniform(0, 1e-7, (jm, ns)),
        "strat_od": rng.uniform(0, 1e-8, (jm, ns)),
        "oh_3d": rng.uniform(1e5, 1e6, (im, jm, lm)),
        "pl_trop": np.ascontiguousarray(pl),
        "cl_trop": rng.uniform(0, 1e3, (im, jm, ntrop)),
        "soil": rng.uniform(0, 1, (im, jm)),
        "oh_scale": 0.9,
        "cl_scale": 1.1,
        "o1d_scale": 1.0,
        "soil_rate": 1e-9,
    }


def test_loss_rates_and_decay(be, twin_libm):
    from pydyn.gpu import kernels as G

    rng = np.random.default_rng(4)
    inputs = _chem_inputs(rng, IM, JM, 20)
    loss = chem.loss_rates(**inputs)
    got_loss, got_decay = G.loss_rates(1800.0, **inputs)
    assert _equal(be.to_host(got_loss), loss)
    assert _equal(be.to_host(got_decay), chem.decay(loss, 1800.0))


# ------------------------------------------------------------------ a region
def _region1(reduced=True, lm=6):
    grid = RedGrid(24, 18, jred=(1, 2, 17, 18), clustsize=(24, 4, 4, 24)) if reduced else None
    glb = Z.Region("glb", 24, 18, xcyc=1, touch_sp=1, touch_np=1, redgrid=grid)
    return Z.Zoom([glb], lm, np.linspace(2000.0, 0.0, lm + 1), np.linspace(1.0, 0.0, lm + 1))


@pytest.mark.parametrize("reduced", [True, False])
def test_whole_step_of_a_region(be, reduced):
    """Setup_MassFlow, then a whole ndyn step (both half steps, x y z v on the reduced grid,
    m2phlb1): every array of the region, halos included."""
    from pydyn.gpu import zoom as gz

    zm, rng = _region1(reduced), np.random.default_rng(5)
    st = synth.zoom_meteo(zm, rng, nt=2, cfl=0.5)
    conv = synth.zoom_convection(zm, st, rng)
    dev = gz.upload(st)
    Z.setup_massflow(zm, st, 3600)
    Z.setup_massflow(zm, dev, 3600)
    Z.step(zm, st, conv, 3600)
    Z.step(zm, dev, conv, 3600)
    got = gz.download(dev)
    for name in st[1]:
        assert _equal(got[1][name].data, st[1][name].data), name


def test_check_cfl_of_a_region(be):
    """Check_CFL rejecting the step until it fits (the fluxes scaled up), and the state it
    leaves; ndyn, n, iterations and every array the same."""
    from pydyn.gpu import zoom as gz

    zm, rng = _region1(), np.random.default_rng(6)
    st = synth.zoom_meteo(zm, rng, nt=1, cfl=2.5)
    dev = gz.upload(st)
    ref = Z.check_cfl_zoom(zm, st, 3600, 1, 3600)
    got = Z.check_cfl_zoom(zm, dev, 3600, 1, 3600)
    assert got == ref and ref[2] > 1
    host = gz.download(dev)
    for name in st[1]:
        assert _equal(host[1][name].data, st[1][name].data), name


# ------------------------------------------------------------------ zoom trees
def _trees():
    from test_zoom import eur, production, three, two

    return {"two": two, "three": three, "production": production, "eur": eur}


@pytest.mark.parametrize("tree", ["two", "three", "production", "eur"])
def test_whole_step_of_a_tree(be, tree):
    """Setup_MassFlow (dynam0, coarsen_region, coarsen_ambm) and a whole step of every region of
    a zoom tree -- the exchange, the children's x y z, convec, update_parent, m2phlb."""
    from pydyn.gpu import zoom as gz

    zm, rng = _trees()[tree](), np.random.default_rng(11)
    st = synth.zoom_meteo(zm, rng, nt=2, cfl=0.4)
    conv = synth.zoom_convection(zm, st, rng)
    dev = gz.upload(st)
    Z.setup_massflow(zm, st, 3600)
    Z.setup_massflow(zm, dev, 3600)
    Z.step(zm, st, conv, 3600)
    Z.step(zm, dev, conv, 3600)
    got = gz.download(dev)
    for r in zm.regions():
        for name in st[r]:
            assert _equal(got[r][name].data, st[r][name].data), (r, name)


@pytest.mark.parametrize(("tree", "cfl"), [("two", 1.0), ("production", 1.0), ("three", 0.3)])
def test_check_cfl_of_a_tree(be, tree, cfl):
    from pydyn.gpu import zoom as gz

    zm = _trees()[tree]()
    st = synth.zoom_meteo(zm, np.random.default_rng(3), nt=1, cfl=cfl)
    dev = gz.upload(st)
    ref = Z.check_cfl_zoom(zm, st, 3600, 2, 7200)
    assert Z.check_cfl_zoom(zm, dev, 3600, 2, 7200) == ref
    got = gz.download(dev)
    for r in zm.regions():
        for name in st[r]:
            assert _equal(got[r][name].data, st[r][name].data), (r, name)


def _intervals(zm, st, conv, n, pipe=None):
    """n intervals (Check_CFL, Setup_MassFlow, a step) one after another on st, or submitted
    to pipe; Check_CFL's (ndyn, n) of each."""
    out = []

    def body():
        out.append(Z.check_cfl_zoom(zm, st, 3600, 1, 3600))
        Z.setup_massflow(zm, st, out[-1][0])
        Z.step(zm, st, conv, out[-1][0])

    for _ in range(n):
        body() if pipe is None else pipe.submit(body)
    if pipe is not None:
        pipe.drain()
    return out


def _speculated(tree, cfl):
    """The host's four intervals and the device's, two in flight (gz.Pipeline): (host state,
    device state, its tally)."""
    from pydyn.gpu import zoom as gz

    zm, rng = _trees()[tree](), np.random.default_rng(5)
    st = synth.zoom_meteo(zm, rng, nt=2, cfl=cfl)
    conv = synth.zoom_convection(zm, st, rng)
    dev = gz.upload(st)
    _intervals(zm, st, conv, 4)
    pipe = gz.Pipeline(zm, dev, 2)
    _intervals(zm, dev, conv, 4, pipe)
    return zm, st, gz.download(dev), pipe.tally


@pytest.mark.parametrize(("tree", "cfl"), [("production", 0.4), ("two", 1.0), ("two", 2.5)])
def test_speculated_intervals_are_the_reference(be, tree, cfl):
    """Four intervals following on from each other, two in flight on the device (each slot's
    scratch kept: Backend.scratch), bitwise to the host's -- those whose assumptions hold and
    those redone (Check_CFL cutting the step).  On emu the Pipeline just runs each call."""
    zm, st, got, tally = _speculated(tree, cfl)
    for r in zm.regions():
        for name in st[r]:
            assert _equal(got[r][name].data, st[r][name].data), (tree, cfl, r, name)
    test_speculated_intervals_are_the_reference.seen.update(k for k, v in tally.items() if v)


test_speculated_intervals_are_the_reference.seen = set()


def test_both_speculation_paths_ran_on_the_device(be):
    """The cases above kept intervals and redid some (an asynchronous backend's)."""
    seen = test_speculated_intervals_are_the_reference.seen
    if not be.asynchronous or not seen:
        pytest.skip("no speculation on a synchronous backend")
    assert seen == {"kept", "redone"}, seen


def test_the_slots_scratch_stays_where_it_is(be):
    """What replaying a captured interval needs: after the first interval of each slot, every
    later one is given the same arrays (no request moved, the same addresses)."""
    from pydyn.gpu import zoom as gz

    if be.arena_for(("probe",)) is None:
        pytest.skip("no scratch arena on this backend")
    zm, rng = _trees()["production"](), np.random.default_rng(5)
    st = synth.zoom_meteo(zm, rng, nt=1, cfl=0.4)
    conv = synth.zoom_convection(zm, st, rng)
    dev = gz.upload(st)
    pipe = gz.Pipeline(zm, dev, 2)
    _intervals(zm, dev, conv, 2, pipe)
    arenas = [be.arena_for((id(pipe), slot)) for slot in range(2)]
    before = [[be.address(a) for a in arena.arrays] for arena in arenas]
    _intervals(zm, dev, conv, 4, pipe)  # a redone one runs on fresh arrays, the slots' untouched
    assert all(arena.arrays and not arena.moved for arena in arenas)
    assert [[be.address(a) for a in arena.arrays] for arena in arenas] == before


def test_scratch_that_keeps_old_values_is_caught(be, monkeypatch):
    """The checks see the scratch: handed out uncleared (the last interval's search maxima
    still in it), every interval's assumptions fail and it is redone deciding as it goes --
    the bits the reference's, none of the intervals kept."""
    from pydyn.gpu import backend as B

    if be.arena_for(("probe",)) is None:
        pytest.skip("no scratch arena on this backend")
    take = B._Arena.take

    def stale(self, shape, dtype, fill):
        reused = self.at < len(self.arrays)
        a = take(self, shape, dtype, fill)
        if reused and not fill:
            a.fill(7)  # what an uncleared buffer could hold
        return a

    _, _, _, honest = _speculated("production", 0.4)
    monkeypatch.setattr(B._Arena, "take", stale)
    zm, st, got, tally = _speculated("production", 0.4)
    assert all(_equal(got[r][k].data, st[r][k].data) for r in zm.regions() for k in st[r])
    assert honest["redone"] == 0 and tally["redone"] > 0, (honest, tally)


def _replayed(tree, cfl, n=6, ndyns=None, inside=None):
    """The host's n intervals and the device's submitted repeatable (gz.Pipeline's repeat),
    two in flight: (zoom, host state, device state, tally, the pipeline).  ndyns: each
    interval's own time step (default 3600); inside(dev): run inside every device call."""
    from pydyn.gpu import zoom as gz
    from pydyn.gpu.backend import backend as be_

    zm, rng = _trees()[tree](), np.random.default_rng(5)
    st = synth.zoom_meteo(zm, rng, nt=2, cfl=cfl)
    conv = synth.zoom_convection(zm, st, rng)
    dev = gz.upload(st)
    ndyns = ndyns or [3600] * n
    for ndyn in ndyns:
        Z.setup_massflow(zm, st, ndyn)
        Z.step(zm, st, conv, ndyn)
    pipe = gz.Pipeline(zm, dev, 2)
    dconv = {  # in place on the device, as an interval's ingestion leaves them
        r: {k: v if k == "lmax" else be_().to_device(v, "input", np.int32 if k.startswith("cloud") else None)
            for k, v in c.items()}
        for r, c in conv.items()
    }  # fmt: skip

    def body(ndyn):
        if inside is not None:
            inside(dev)
        Z.setup_massflow(zm, dev, ndyn)
        Z.step(zm, dev, dconv, ndyn)

    for ndyn in ndyns:  # each call repeatable on its own: its time step bound to it
        pipe.submit(lambda ndyn=ndyn: body(ndyn), repeat="interval")
    pipe.drain()
    return zm, st, gz.download(dev), pipe.tally, pipe


def _same(zm, st, got):
    return all(_equal(got[r][k].data, st[r][k].data) for r in zm.regions() for k in st[r])


@pytest.mark.parametrize(("tree", "cfl"), [("production", 0.2), ("two", 0.3)])  # six steps stay valid
def test_replayed_intervals_are_the_reference(be, tree, cfl):
    """Six intervals following on from each other, declared repeatable: on CUDA each slot's
    second captured as a graph and the rest replayed (fn not called), bitwise to the host's."""
    zm, st, got, tally, _ = _replayed(tree, cfl)
    assert _same(zm, st, got), (tree, cfl, tally)
    if hasattr(be, "capture_begin"):
        assert tally["replayed"] >= 2 and tally["redone"] == 0, tally


def test_a_call_that_uploads_is_not_captured(be):
    """A call that uploads (its staging reused: a replay would copy whatever is there then) is
    refused capture, run as it goes every time, and is still the reference."""
    if not hasattr(be, "capture_begin"):
        pytest.skip("no graphs on this backend")
    probe = np.arange(4.0)
    zm, st, got, tally, pipe = _replayed("two", 0.3, inside=lambda dev: be.to_device(probe, "input"))
    assert _same(zm, st, got) and tally["replayed"] == 0, tally
    assert set(pipe.graphs.values()) == {None}, pipe.graphs


def test_a_call_wrongly_declared_repeatable_is_caught(be):
    """The gate can fail: intervals whose time steps differ, declared repeatable anyway, replay
    the step captured with the first one's -- not the host's."""
    if not hasattr(be, "capture_begin"):
        pytest.skip("no graphs on this backend")
    ndyns = [3600, 3600, 3600, 3600, 1800, 1800]  # warmed and captured at 3600, replayed for 1800
    zm, st, got, tally, _ = _replayed("two", 0.3, ndyns=ndyns)
    assert tally["replayed"] >= 2 and not _same(zm, st, got), tally


@needs(ZOOM_PHYS_DIR, "PYDYN_ZOOM_PHYS_DIR")
def test_production_zoom_operators_are_the_models(be):
    """The glb3x2 + nam1x1 model run's own x, y, z of both regions, its s, Setup_MassFlow and
    Check_CFL, per event on the device: bitwise to the model's dumps, no twin involved."""
    from pydyn import replay

    results = replay.replay(
        ZOOM_PHYS_DIR,
        device=DEVICE,
        tags={"x", "y", "z", "s", "massflow", "cfl"},
        out=__import__("io").StringIO(),
    )
    kinds = {(kind, ev.region) for ev, kind, _, _ in results}
    assert kinds >= {("x", 1), ("x", 2), ("y", 1), ("y", 2), ("z", 1), ("z", 2), ("s", 2), ("cfl", 1)}
    assert all(r["equal"] for _, _, reports, _ in results for r in reports)


def test_the_checks_fail_under_contraction(monkeypatch):
    """A gate must prove it can fail: the same source compiled with FMA contraction on (the
    failure --fmad=false prevents) must not pass the advection test."""
    from pydyn.gpu import backend as B
    from pydyn.gpu import kernels as G

    monkeypatch.setenv("PYDYN_EMU_CONTRACT", "1")
    be = B.set_backend("emu")
    try:
        rng = np.random.default_rng(7)
        s = synth.state(rng, IM, JM, LM, NT)
        am, _, _ = synth.fluxes(rng, s.m, cfl=0.9)
        ref, a_ref = s.copy(), am.copy()
        kernels.advect_x(*_fields(ref), a_ref)
        got = _device_advect(be, G, 0, s, am)
        caught = not all(_equal(a, b) for a, b in zip(got, _fields(ref)))
        if not caught and platform.machine() not in ("arm64", "aarch64"):
            pytest.skip(f"no FMA to contract into on {platform.machine()}")
        assert caught, "an FMA build passed the bitwise test"
    finally:
        monkeypatch.delenv("PYDYN_EMU_CONTRACT")
        B.set_backend(DEVICE)


# ------------------------------------------------------------------ the laptop sets
@pytest.mark.parametrize(
    "dump_dir",
    [
        pytest.param(d, marks=needs(d, env), id=d.name)
        for d, env in (
            (FREE_DUMP_DIR, "PYDYN_FREE_DUMP_DIR"),
            (DUMP_DIR, "PYDYN_DUMP_DIR"),
            (CFL_DUMP_DIR, "PYDYN_CFL_DUMP_DIR"),
        )
    ],
)
def test_free_run_on_the_device_is_the_reference(dump_dir, twin_libm):
    """A day carried by the reference and by the device, the twins on both: every array of
    every checkpoint has the same sha1; nothing of the state came down but to be compared."""
    from pydyn.gpu import check

    cpu, dev, log = check.run_both(dump_dir, DEVICE, free_run=True)
    table, bad = check.match(cpu, dev)
    assert table and not bad, bad[:10]
    assert sum(eq for eq, _ in table.values()) > 100
    assert "kzz" not in log


def _nvptx_clang():
    """A clang with the NVPTX back end (PYDYN_CLANG, Homebrew's llvm, or the PATH's), or None."""
    import subprocess

    for c in (os.environ.get("PYDYN_CLANG"), "/opt/homebrew/opt/llvm/bin/clang", shutil.which("clang")):
        if c and os.path.exists(c):
            r = subprocess.run([c, "-print-targets"], capture_output=True, text=True)
            if "nvptx" in r.stdout:
                return c
    return None


def _cuda_ptx(tmp_path, lmax, contract="off"):
    """(clang's result, the PTX) of csrc/all.cu + convec_gen's CUDA passes for sm_80, against
    tests/nvrtc_stub.h (what NVRTC provides), optimized as NVRTC does."""
    import subprocess
    from pathlib import Path

    from pydyn.gpu import convec_gen, coop
    from pydyn.gpu.backend import CSRC, kernel_signatures

    src = tmp_path / f"all_{contract}.cu"
    src.write_text(
        (CSRC / "all.cu").read_text()
        + convec_gen.convec_source(lmax, "cuda")
        + coop.source(kernel_signatures(), "cuda")
    )
    ptx = tmp_path / f"all_{contract}.ptx"
    stub = Path(__file__).parent / "nvrtc_stub.h"
    cmd = [_nvptx_clang(), "-x", "cuda", "--cuda-device-only", "--cuda-gpu-arch=sm_80", "-nocudainc",
           "-nocudalib", "-std=c++14", "-O2", f"-ffp-contract={contract}", "-Xclang", "-target-feature",
           "-Xclang", "+ptx70", "-Wall", "-Wno-unused-variable", "-Wno-unused-function",
           "-Wno-unused-but-set-variable", "-include", str(stub), f"-I{CSRC}", f"-DPD_CONVEC_N={lmax}",
           "-DPD_COOP_FACES=64", "-S", "-o", str(ptx), str(src)]  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r, ptx.read_text() if ptx.exists() else ""


def _pass_fmas(ptx, lmax):
    import re

    from pydyn.gpu import coop

    out = {}
    names = [f"pd_{name}_{lmax}" for name in ("cvc", "cvb", "cvl", "cvs", "cvg", "cvu", "cvf")]
    for name in names + [f"{k}_coop" for k in coop.COOP]:
        body = re.search(rf"\.entry {name}\b.*?\n}}", ptx, re.S)
        assert body is not None, name
        out[name] = body.group(0).count("fma.")
    return out


@pytest.mark.skipif(_nvptx_clang() is None, reason="no clang with the NVPTX back end")
@pytest.mark.parametrize("lmax", [5, 26, 32])
def test_the_cuda_source_compiles(tmp_path, lmax):
    """csrc/all.cu, convec_gen's CUDA passes (pd_cvf too) and coop's line kernels compile for sm_80 (clang's
    CUDA front end: no GPU, no CUDA SDK), no warning but unused ones, and no FMA in them
    (contraction off, as NVRTC's --fmad=false); a contracted build has them (it can fail)."""
    r, ptx = _cuda_ptx(tmp_path, lmax)
    assert r.returncode == 0 and "warning" not in r.stderr, r.stderr[-3000:]
    assert not any(_pass_fmas(ptx, lmax).values()), _pass_fmas(ptx, lmax)
    r, ptx = _cuda_ptx(tmp_path, lmax, contract="fast")
    fmas = _pass_fmas(ptx, lmax)
    assert r.returncode == 0 and fmas[f"pd_cvg_{lmax}"] > 0 and fmas[f"pd_cvf_{lmax}"] > 0
    assert fmas["pd_z_steps_coop"] > 0
