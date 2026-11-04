"""pydyn.gpu's float32 pair: the Metal GPU against the host float32 build (emu32), bitwise.

Apple GPUs have no double, so `metal` runs csrc/ with real = float; emu32 is the same source
on the CPU with FZ set (the GPU flushes subnormals).  Every operation of test_gpu.py is run on
both from the same float32 inputs and every output compared bit for bit; the float32 results
are also held near the float64 reference (they are not it), the float32 exp/log/pow near the
system's, and a contracted Metal build is shown to fail the comparison."""

import platform
import shutil
import sys

import numpy as np
import pytest

from pydyn import chem, diffusion, kernels, synth
from pydyn import zoom as Z

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64" or shutil.which("xcrun") is None,
    reason="needs an Apple-silicon Mac with the Metal toolchain",
)

IM, JM, LM, NT = 12, 9, 7, 3


@pytest.fixture(scope="module")
def pair():
    from pydyn.gpu import backend as B

    try:
        metal = B.set_backend("metal")
    except RuntimeError as e:  # no Metal device
        pytest.skip(str(e))
    emu32 = B.Emu32Backend()
    yield emu32, metal
    B.set_backend("emu")


def _both(pair, fn):
    """fn(be) -> host arrays, run on emu32 then metal; the two lists."""
    from pydyn.gpu import backend as B

    out = []
    for be in pair:
        B.set_backend(be)
        out.append([np.asarray(a) for a in fn(be)])
    return out


def _bits(a):
    a = np.asarray(a)
    return a.view(np.uint32) if a.dtype == np.float32 else a


def _assert_bitwise(e, m, what=""):
    assert len(e) == len(m)
    for k, (a, b) in enumerate(zip(e, m, strict=True)):
        assert a.shape == b.shape and a.dtype == b.dtype, (what, k)
        diff = np.flatnonzero((_bits(a) != _bits(b)).ravel())
        assert diff.size == 0, (what, k, diff.size, a.ravel()[diff[:3]], b.ravel()[diff[:3]])


# ------------------------------------------------------------------ the float32 math
def test_float_exp_log_pow_are_within_a_few_ulp():
    """csrc/pmath.cuh's PD_F32 branch on the host, against float64 libm rounded to float."""
    import ctypes

    from pydyn.gpu import backend as B

    lib = ctypes.CDLL(str(B.Emu32Backend().module()._name))
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.uniform(-87, 88, 20000), rng.uniform(-1, 1, 5000)]).astype(np.float32)
    lx = np.exp(rng.uniform(-87, 88, 20000)).astype(np.float32)
    px, py = rng.uniform(0.1, 10, 20000).astype(np.float32), rng.uniform(-5, 5, 20000).astype(np.float32)
    for name, args, ref, tol in (
        ("pd_test_exp", (x,), np.exp(x.astype(np.float64)), 1),
        ("pd_test_log", (lx,), np.log(lx.astype(np.float64)), 1),
        ("pd_test_pow", (px, py), np.power(px.astype(np.float64), py.astype(np.float64)), 2),
    ):
        f = getattr(lib, name)
        out = np.zeros(args[0].size, np.float32)
        f(
            *(a.ctypes.data_as(ctypes.c_void_p) for a in args),
            out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_longlong(out.size),
        )
        ref32 = ref.astype(np.float32)
        ulp = np.abs(out.view(np.int32).astype(np.int64) - ref32.view(np.int32).astype(np.int64))
        assert ulp.max() <= tol, (name, ulp.max())


# ------------------------------------------------------------------ kernels
@pytest.mark.parametrize("cfl", [0.4, 1.3, 3.2])
def test_advection_kernels(pair, cfl):
    from pydyn.gpu import kernels as G

    rng = np.random.default_rng(int(cfl * 10))
    s = synth.state(rng, IM, JM, LM, NT)
    fluxes = synth.fluxes(rng, s.m, cfl=cfl)
    for axis in range(3):

        def run(be, axis=axis):
            d = [be.to_device(q, "seed") for q in (s.m, s.rm, s.rxm, s.rym, s.rzm, fluxes[axis])]
            try:
                if axis == 0:
                    G.advect_x_rows(*d, be.to_device(np.full(JM, IM, np.int32), "seed"))
                elif axis == 1:
                    G.advect_y(*d)
                else:
                    G.advect_z(*d)
            except kernels.CFLError:
                return [np.array([-1])]
            return [be.to_host(q) for q in d]

        _assert_bitwise(*_both(pair, run), what="xyz"[axis])


@pytest.mark.parametrize("inverse", ["crout", "lapack"])
def test_convec(pair, inverse):
    from pydyn.gpu import kernels as G

    rng = np.random.default_rng(2)
    s = synth.state(rng, IM, JM, 12, NT)
    for lmax, kw in ((12, {}), (9, {"strong_detu": True}), (5, {"lfs_above_top": True})):
        conv = synth.convection(rng, s.m, lmax, 1800.0, **kw)

        def run(be, conv=conv, lmax=lmax):
            d = [be.to_device(q, "seed") for q in (s.m, s.rm, s.rxm, s.rym, s.rzm)]
            dc = {
                k: be.to_device(v, "input", np.int32 if k.startswith("cloud") else None)
                for k, v in conv.items()
            }
            G.convec(*d, dc, 1800.0, lmax, be.to_device(np.arange(IM * JM, dtype=np.int32)), inverse=inverse)
            return [be.to_host(q) for q in d]

        _assert_bitwise(*_both(pair, run), what=(inverse, lmax))


def test_convec_above_32_levels(pair):
    """lmax > 32: csrc/convec_metal.h's SIMD-group kernel, not convec_gen's."""
    from pydyn.gpu import kernels as G

    rng = np.random.default_rng(5)
    s = synth.state(rng, IM, JM, 41, NT)
    conv = synth.convection(rng, s.m, 40, 1800.0)

    def run(be):
        d = [be.to_device(q, "seed") for q in (s.m, s.rm, s.rxm, s.rym, s.rzm)]
        dc = {
            k: be.to_device(v, "input", np.int32 if k.startswith("cloud") else None) for k, v in conv.items()
        }
        G.convec(*d, dc, 1800.0, 40, be.to_device(np.arange(IM * JM, dtype=np.int32)))
        return [be.to_host(q) for q in d]

    _assert_bitwise(*_both(pair, run), what="lmax 40")


def _hard_matrices(rng, n, ncol=48):
    """Row-major n x n float32 matrices: dense (they pivot), conv1-like (diagonally dominant),
    sparse with zeros of both signs (DTRSM's u == 0 skip shows in a signed zero)."""
    dense = rng.standard_normal((ncol, n, n))
    dominant = np.eye(n) * (n + 1) + rng.uniform(-1, 0, (ncol, n, n))
    sparse = rng.standard_normal((ncol, n, n))
    sparse[rng.random(sparse.shape) < 0.6] = 0.0
    sparse[(sparse == 0) & (rng.random(sparse.shape) < 0.3)] = -0.0
    sparse[:, np.arange(n), np.arange(n)] += 3.0
    return np.concatenate([dense, dominant, sparse]).astype(np.float32)


def _inverses(pair, M):
    from pydyn.gpu import kernels as G

    return _both(pair, lambda be: [be.to_host(G.inverse_rows(be.to_device(M, "seed"))[0])])


@pytest.mark.parametrize("n", [2, 5, 9, 12, 26, 32])
def test_lapack_kind_pass(pair, n):
    """convec_gen's pd_cvg (DGETRF2 + DGETRI, a row a lane) against emu32's serial twin."""
    e, m = _inverses(pair, _hard_matrices(np.random.default_rng(n), n))
    _assert_bitwise(e, m, what=n)


@pytest.mark.parametrize(
    ("variant", "old", "new"),
    [
        ("division for DSCAL's reciprocal", "if (low) ar{k} = r * ar{k};", "if (low) ar{k} = ar{k} / piv;"),
        ("no DTRSM zero skip", " && !(p < {cut} && u == 0.0f)", ""),
    ],
)
def test_a_lapack_pass_without_its_rules_fails(pair, monkeypatch, variant, old, new):
    import inspect

    from pydyn.gpu import backend as B
    from pydyn.gpu import convec_gen, metal

    src = inspect.getsource(convec_gen._getrf_getri)
    assert src.count(old) == 1, variant
    ns = {}
    exec(compile(src.replace(old, new), "variant", "exec"), vars(convec_gen), ns)
    monkeypatch.setattr(convec_gen, "_getrf_getri", ns["_getrf_getri"])
    emu32, good = pair
    bad = metal.MetalBackend()
    try:  # three sizes, many matrices: the skip shows only where a signed zero meets it
        e, m = [], []
        for n in (5, 9, 12):
            ee, mm = _inverses((emu32, bad), _hard_matrices(np.random.default_rng(n), n, ncol=128))
            e, m = e + ee, m + mm
    finally:
        B.set_backend(good)
        metal.MetalArray.be = good
    assert any((_bits(a) != _bits(b)).any() for a, b in zip(e, m, strict=True)), f"{variant} passed"


def test_dynam0_and_cluster_sum(pair):
    from pydyn.gpu import kernels as G

    rng = np.random.default_rng(3)
    pu, pv = rng.normal(size=(IM + 1, JM, LM)) * 1e9, rng.normal(size=(IM, JM + 1, LM)) * 1e9
    bt = np.linspace(1.0, 0.0, LM + 1)
    q = rng.normal(size=(60, LM, NT))

    def run(be):
        got = [be.xp.zeros(x.shape) for x in kernels.fluxes_from_meteo(pu, pv, bt, 900.0)]
        G.fluxes_from_meteo(be.to_device(pu), be.to_device(pv), be.to_device(bt), 900.0, *got)
        return [be.to_host(g) for g in got] + [be.to_host(G.cluster_sum(be.to_device(q), 3, 20))]

    _assert_bitwise(*_both(pair, run))


@pytest.mark.parametrize(("lm", "lmax_conv"), [(8, 6), (34, 26)])
def test_calc_kzz(pair, lm, lmax_conv):
    from pydyn.gpu import diffusion as GD

    rng = np.random.default_rng(lm)
    args, grid = synth.kzz_inputs(rng, IM, JM, lm=lm, lmax_conv=lmax_conv)

    def run(be):
        pblh = be.xp.zeros((IM, JM))
        return [be.to_host(GD.calc_kzz(*args, **grid, pblh=pblh)), be.to_host(pblh)]

    e, m = _both(pair, run)
    _assert_bitwise(e, m)
    _, parts = diffusion.calc_kzz(*args, **grid)  # float32 near the float64 reference
    np.testing.assert_allclose(m[1], parts["pblh"], rtol=1e-3)


def test_loss_rates_and_decay(pair):
    from test_gpu import _chem_inputs

    from pydyn.gpu import kernels as G

    inputs = _chem_inputs(np.random.default_rng(4), IM, JM, 20)

    def run(be):
        loss, decay = G.loss_rates(1800.0, **inputs)
        return [be.to_host(loss), be.to_host(decay)]

    e, m = _both(pair, run)
    _assert_bitwise(e, m)
    np.testing.assert_allclose(m[1], chem.decay(chem.loss_rates(**inputs), 1800.0), rtol=1e-5)


# ------------------------------------------------------------------ regions and trees
def _trees():
    from test_gpu import _region1
    from test_zoom import eur, production, three, two

    return {
        "region": _region1,
        "region_uniform": lambda: _region1(False),
        "two": two,
        "three": three,
        "production": production,
        "eur": eur,
    }


def _step(tree, nt=2, cfl=0.4, seed=11):
    zm, rng = _trees()[tree](), np.random.default_rng(seed)
    st = synth.zoom_meteo(zm, rng, nt=nt, cfl=cfl)
    return zm, st, synth.zoom_convection(zm, st, rng)


def _arrays(zm, st):
    return [st[r][name].data for r in zm.regions() for name in sorted(st[r])]


def _download(zm, gz, dev):
    return _arrays(zm, gz.download(dev))


@pytest.mark.parametrize("tree", ["region", "region_uniform", "two", "three", "production", "eur"])
def test_whole_step(pair, tree):
    """Setup_MassFlow and a whole step of every region; the float32 state also within float32
    rounding of the float64 reference's."""
    from pydyn.gpu import zoom as gz

    zm, st, conv = _step(tree)

    def run(be):
        dev = gz.upload(st)
        Z.setup_massflow(zm, dev, 3600)
        Z.step(zm, dev, conv, 3600)
        return _download(zm, gz, dev)

    e, m = _both(pair, run)
    _assert_bitwise(e, m, tree)
    Z.setup_massflow(zm, st, 3600)
    Z.step(zm, st, conv, 3600)
    ref = _arrays(zm, st)
    names = [(r, name) for r in zm.regions() for name in sorted(st[r])]
    for (r, name), a, b in zip(names, m, ref, strict=True):
        if name in ("m", "rm"):
            np.testing.assert_allclose(
                a, b, rtol=2e-4, atol=1e-6 * np.abs(b).max(), err_msg=f"{tree} {r} {name}"
            )


@pytest.mark.parametrize(
    ("tree", "cfl"), [("region", 2.5), ("two", 1.0), ("production", 1.0), ("three", 0.3)]
)
def test_check_cfl(pair, tree, cfl):
    from pydyn.gpu import zoom as gz

    zm, st, _ = _step(tree, nt=1, cfl=cfl, seed=3)
    nread = 3600 if tree == "region" else 7200
    n = 1 if tree == "region" else 2

    def run(be):
        dev = gz.upload(st)
        got = Z.check_cfl_zoom(zm, dev, 3600, n, nread)
        return [np.frombuffer(repr(got).encode(), np.uint8), *_download(zm, gz, dev)]

    _assert_bitwise(*_both(pair, run), what=tree)


@pytest.mark.parametrize("tree", ["region", "two", "production", "eur"])
def test_every_cooperative_line_kernel(pair, monkeypatch, tree):
    """PYDYN_COOP=all: every line kernel a SIMD group a line (CUDA's default; Metal's runs some
    serial), a whole step and Check_CFL against emu32's serial kernels."""
    from pydyn.gpu import zoom as gz

    monkeypatch.setenv("PYDYN_COOP", "all")
    zm, st, conv = _step(tree)

    def run(be):
        dev = gz.upload(st)
        Z.setup_massflow(zm, dev, 3600)
        Z.step(zm, dev, conv, 3600)
        got = Z.check_cfl_zoom(zm, dev, 3600, 2, 7200) if tree != "eur" else ()  # eur's tref: no ndyn of 7200
        return [np.frombuffer(repr(got).encode(), np.uint8), *_download(zm, gz, dev)]

    _assert_bitwise(*_both(pair, run), what=tree)


# ------------------------------------------------------------------ the gate can fail
def test_a_contracted_metal_build_fails(pair, monkeypatch):
    """With -ffp-contract=fast the Metal compiler fuses a*b+c: the whole step differs."""
    from pydyn.gpu import backend as B
    from pydyn.gpu import metal
    from pydyn.gpu import zoom as gz

    zm, st, conv = _step("region")

    def run(be):
        dev = gz.upload(st)
        Z.setup_massflow(zm, dev, 3600)
        Z.step(zm, dev, conv, 3600)
        return _download(zm, gz, dev)

    emu32, good = pair
    monkeypatch.setattr(metal, "_CONTRACT", True)
    bad = metal.MetalBackend()
    try:
        e, m = _both((emu32, bad), run)
    finally:
        B.set_backend(good)
        metal.MetalArray.be = good
    assert any((_bits(a) != _bits(b)).any() for a, b in zip(e, m, strict=True)), "a contracted build passed"


# ------------------------------------------------------------------ speculation
@pytest.mark.parametrize(
    ("tree", "cfl"), [("region", 0.4), ("region", 1.3), ("region", 2.5), ("production", 0.4), ("two", 1.0)]
)
def test_a_speculated_interval_is_the_careful_one(pair, tree, cfl):
    """gpu.zoom.Pipeline around Check_CFL + Setup_MassFlow + a step: metal speculating against
    emu32 deciding as it goes, bitwise -- across intervals whose assumptions hold (kept) and
    whose do not (Check_CFL cutting the step, searches past their queued rounds: redone)."""
    from pydyn.gpu import zoom as gz

    zm, st, conv = _step(tree, nt=2, cfl=cfl, seed=7)
    tally = {}

    def run(be):
        dev = gz.upload(st)
        out = []

        def body():
            out.append(Z.check_cfl_zoom(zm, dev, 3600, 1, 3600))
            Z.setup_massflow(zm, dev, out[-1][0])
            Z.step(zm, dev, conv, out[-1][0])

        pipe = gz.Pipeline(zm, dev, 1)
        pipe.submit(body)
        pipe.drain()
        tally.update((k, v) for k, v in pipe.tally.items() if v)
        return [np.frombuffer(repr(out[-1]).encode(), np.uint8), *_download(zm, gz, dev)]

    _assert_bitwise(*_both(pair, run), what=(tree, cfl))
    test_a_speculated_interval_is_the_careful_one.seen.update(tally)


test_a_speculated_interval_is_the_careful_one.seen = set()


def test_both_speculation_paths_ran():
    """The cases above went through both: an interval kept and one redone."""
    seen = test_a_speculated_interval_is_the_careful_one.seen
    if not seen:
        pytest.skip("the speculation cases did not run")
    assert seen == {"kept", "redone"}, seen


@pytest.mark.parametrize("nt", [1, 3])
def test_an_interval_with_its_physics(pair, nt):
    """Check_CFL, Setup_MassFlow, Calc_Kzz and a step with emissions and the CH4 chemistry
    (bench.physics' inputs on the two-region tree), speculated: bitwise to emu32."""
    from pydyn.gpu import bench
    from pydyn.gpu import zoom as gz

    zm, st, conv = _step("two", nt=nt, cfl=0.4, seed=9)

    def run(be):
        per_interval, sources, chemistry = bench.physics(zm, nt, lmax=zm.lm - 1)
        dev = gz.upload(st)

        def body():
            Z.check_cfl_zoom(zm, dev, 3600, 1, 3600)
            Z.setup_massflow(zm, dev, 3600)
            per_interval()
            Z.step(zm, dev, conv, 3600, sources=sources, chemistry=chemistry)

        pipe = gz.Pipeline(zm, dev, 1)
        pipe.submit(body)
        pipe.drain()
        return _download(zm, gz, dev)

    _assert_bitwise(*_both(pair, run), what=nt)
