"""pydyn.forward, the run from the input files alone: on the CPU, the model's dumps bitwise (and
the comparison failing on one ulp of the OH scaling); on the emulation device, the CPU run with
the twins bitwise; read in worker processes, what one thread reads; on Metal, with either
convection inverse, emu32 bitwise and finite through the hours whose float32 Calc_Kzz once made
a NaN (kzz.cuh pd_blend, pd_resolved)."""

import platform
import shutil
import sys
from datetime import timedelta

import numpy as np
import pytest

from pydyn import dump, forward
from pydyn.harness import compare

from common import CHEM_ROOT, DUMP_DIR, EMIS_DIR, IC_FILE, METEO_ROOT, needs, twins

pytestmark = [
    needs(METEO_ROOT, "PYDYN_METEO_ROOT"),
    needs(EMIS_DIR, "PYDYN_EMIS_DIR"),
    needs(CHEM_ROOT, "PYDYN_CHEM_ROOT"),
    needs(IC_FILE, "PYDYN_IC_FILE"),
]
STATE = ("m", "rm", "rxm", "rym", "rzm", "am", "bm", "cm", "sp", "phlb")
HAS_METAL = sys.platform == "darwin" and platform.machine() == "arm64" and shutil.which("xcrun") is not None


def _run(device, cfg=None, **kw):
    return forward.Forward(cfg or forward.RunConfig(), METEO_ROOT, EMIS_DIR, CHEM_ROOT, IC_FILE, device, **kw)


def _against_dumps(cfg, hours):
    """Reports of each hour's last aft_x (every state array) and m2phlb (m, sp, phlb)."""
    evs = dump.load_events(DUMP_DIR)
    aft_x = [e for e in evs if e.tag == "aft_x"]
    m2 = [e for e in evs if e.tag == "m2phlb"]
    fwd, reports = _run("cpu", cfg), []
    g, lm = fwd.zm.g(1), fwd.zm.lm
    try:
        for k in range(hours):
            fwd.interval(ahead=k < hours - 1)
            st = fwd.st[1]
            for name in STATE[:8]:
                arr = aft_x[2 * k + 1][name]
                reports.append(compare(arr.data[arr.window(st[name])], st[name].data, f"{k + 1}:{name}"))
            ev = m2[2 * k + 1]
            for name in ("m", "sp"):
                arr = ev[name]
                reports.append(
                    compare(arr.data[arr.window(st[name])], st[name].data, f"{k + 1}:m2phlb:{name}")
                )
            phlb = st["phlb"][1 : g.im, 1 : g.jm, 1 : lm + 1]
            reports.append(compare(ev.interior("phlb"), phlb, f"{k + 1}:m2phlb:phlb"))
    finally:
        fwd.close()
    return reports


@needs(DUMP_DIR, "PYDYN_DUMP_DIR")
def test_the_cpu_run_is_the_models_dumps():
    reports = _against_dumps(forward.RunConfig(), 3)
    assert len(reports) == 33 and all(r["equal"] for r in reports), [
        r["name"] for r in reports if not r["equal"]
    ]


@needs(DUMP_DIR, "PYDYN_DUMP_DIR")
def test_the_comparison_fails_on_one_ulp_of_oh():
    cfg = forward.RunConfig()
    cfg.oh_scale = float(np.nextafter(cfg.oh_scale, 2.0))
    assert not all(r["equal"] for r in _against_dumps(cfg, 2))


def _hourly(f, hours):
    """Host copies of the state arrays after each of `hours` intervals (one device at a time:
    the backend is the process's)."""
    out = []
    try:
        for k in range(hours):
            f.interval(ahead=k < hours - 1)
            out.append({n: np.array(f.E.host(f.st[1][n].data)) for n in STATE})
    finally:
        f.close()
    return out


def _differ(a, b):
    """Each hour, the names of the arrays a and b differ in."""
    return [[n for n in STATE if not compare(x[n], y[n], n)["equal"]] for x, y in zip(a, b, strict=True)]


def test_the_device_is_the_cpu_run_with_its_twins():
    with twins():
        cpu = _hourly(_run("cpu", lapack="twin"), 3)
    off = _differ(cpu, _hourly(_run("emu"), 3))
    assert not any(off), off


def test_reader_processes_read_what_the_thread_reads():
    """7 hours: three windows of intervals over two processes, through shared memory."""
    off = _differ(_hourly(_run("cpu"), 7), _hourly(_run("cpu", readers=2), 7))
    assert not any(off), off


def test_a_staged_run_reads_the_copies_and_is_the_same(tmp_path):
    """meteo.Stager: the intervals' files opened from the stage's copies, every state array as
    read from the archive itself; close removes the copies."""
    f = _run("cpu", stage=tmp_path)
    stage, before = f.stager.stage, set(f.M.archive._open)
    f.stager.ahead(f.t)
    f.stager.wait()  # the first days copied before the first read
    staged, opened = [], set()
    try:
        for k in range(3):
            f.interval(ahead=k < 2)
            staged.append({n: np.array(f.E.host(f.st[1][n].data)) for n in STATE})
            opened |= {nc.filepath() for p, nc in f.M.archive._open.items() if p not in before}
    finally:
        f.close()
    assert opened and all(path.startswith(str(stage)) for path in opened), sorted(opened)[:3]
    off = _differ(_hourly(_run("cpu"), 3), staged)
    assert not any(off), off
    assert not stage.exists()


@pytest.mark.skipif(not HAS_METAL, reason="needs an Apple-silicon Mac with the Metal toolchain")
@pytest.mark.parametrize("lapack", ["auto", "twin"])  # float32's Crout; the reference's LAPACK kind
def test_metal_is_emu32_and_finite(lapack):
    from pydyn.gpu import backend as B

    try:
        metal = _hourly(_run("metal", lapack=lapack), 12)
    except RuntimeError as e:  # no Metal device
        pytest.skip(str(e))
    try:
        emu32 = _hourly(_run("emu32", lapack=lapack), 12)
    finally:
        B.set_backend("emu")
    off = _differ(emu32, metal)
    assert not any(off), off
    assert all(np.isfinite(h[n]).all() for h in metal for n in STATE)


def _final(f, hours, fault=None):
    """The state arrays after run(hours) (the pipeline in flight on an asynchronous device);
    fault: the interval (0-based) whose first speculation corrupts rm and assumes it failed."""
    if fault is not None:
        from pydyn.gpu import exchange
        from pydyn.gpu.backend import backend

        carry, seen = f._carry, []
        at = f.cfg.start + timedelta(seconds=fault * f.cfg.nread)

        def faulty(got, steps):
            out = carry(got, steps)
            be = backend()
            if be.speculating and got["tread"][0] == at and not seen:
                seen.append(1)
                exchange.scale(f.st, "fault", [f.st[1]["rm"].data], 1.001)
                be.assume(lambda: False)
            return out

        f._carry = faulty
    try:
        f.run(hours)
        return {n: np.array(f.E.host(f.st[1][n].data)) for n in STATE}, dict(f.speculated)
    finally:
        f.close()


@pytest.mark.skipif(not HAS_METAL, reason="needs an Apple-silicon Mac with the Metal toolchain")
@pytest.mark.parametrize("bad", [False, True])  # True: the redo without the state put back
def test_a_pipelined_run_is_emu32_through_a_failed_speculation(bad, monkeypatch):
    """Metal's run with two intervals in flight (gpu.zoom.Pipeline), the third's speculation
    corrupting the state and failing: redone, the fourth queued again on it, emu32 bitwise."""
    from pydyn.gpu import backend as B
    from pydyn.gpu import exchange

    try:
        emu32, _ = _final(_run("emu32"), 6)
    finally:
        B.set_backend("emu")
    if bad:
        monkeypatch.setattr(exchange, "restore", lambda st, key: None)
    try:
        metal, spec = _final(_run("metal"), 6, fault=2)
    except RuntimeError as e:  # no Metal device
        pytest.skip(str(e))
    assert spec["redone"] == 1 and spec["kept"] == 5, spec
    off = [n for n in STATE if not compare(emu32[n], metal[n], n)["equal"]]
    assert bool(off) == bad, off
