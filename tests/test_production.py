"""The replay of the production dump sets (docs/state.md section 3b) -- per event, free run and on
the model's input files -- when a `pydyn_dump` directory (and the input files) are at hand."""


import numpy as np
import pytest

from pydyn import (
    balance,
    diffusion,
    emis,
    inputs,
    lapack,
    meteo,
    replay,
    simple_diffusion,
)
from pydyn.dump import load_events
from pydyn.harness import _ulp_distance, compare

from common import (
    CFL_DUMP_DIR,
    DUMP_DIR,
    EMIS_DIR,
    FREE_DUMP_DIR,
    METEO_ROOT,
    needs,
)

# ------------------------------------------------------------------ the replay
# Bitwise needs the model's FFTW (meteo's Poisson solve) and LAPACK (convection's inverse):
# without them the replay falls back to NumPy's, and the claim is not tested at all rather
# than tested at a tolerance.  test_replay_on_numpy_fallbacks is the tolerance, by name.
needs_model_libraries = pytest.mark.skipif(
    not (balance.fftw_library() and lapack.lapack_library()),
    reason="needs the model's FFTW and LAPACK (PYDYN_FFTW, PYDYN_LAPACK); without them nothing is bitwise",
)


def _assert_replay_matches(results):
    """Every event replayed, none raised, every array bitwise."""
    assert results
    assert not results.failures, results.failures
    for kind, row in replay.summarize(results).items():
        for name, a in row["arrays"].items():
            assert a["n_diff"] == 0, f"{kind} {name}: {a}"


def _assert_free_run_matches(results):
    _assert_replay_matches(results)
    # nothing after the first dynamic step is seeded from the model: it is all the Python's own
    first_m2phlb = min(ev.seq for ev, kind, _, _ in results if kind == "m2phlb")
    seeded = [ev for ev, _, _, notes in results if notes.get("seeded")]
    assert all(ev.seq < first_m2phlb for ev in seeded), [str(e) for e in seeded]


def _kinds(results):
    return {kind for _, kind, _, _ in results}


@needs_model_libraries
@needs(DUMP_DIR, "PYDYN_DUMP_DIR")
def test_replay_of_the_model_dumps():
    results = replay.replay(DUMP_DIR)
    _assert_replay_matches(results)
    assert _kinds(results) >= {"x", "y", "z", "v", "s", "c", "meteo", "cfl", "massflow", "m2phlb"}
    assert "loss" in replay.summarize(results)["c"]["arrays"]


@needs_model_libraries
@needs(DUMP_DIR, "PYDYN_DUMP_DIR")
def test_free_run_of_the_model_dumps():
    """The whole production step sequence -- transport, sources, chemistry -- carried by the
    Python for a day; the chemistry on the Python's own phlb (`in:phlb`)."""
    results = replay.replay(DUMP_DIR, free_run=True)
    _assert_free_run_matches(results)
    assert sum(kind == "c" for _, kind, _, _ in results) >= 40
    assert "in:phlb" in replay.summarize(results)["c"]["arrays"]


@needs_model_libraries
@needs(DUMP_DIR, "PYDYN_DUMP_DIR")
@needs(METEO_ROOT, "PYDYN_METEO_ROOT")
def test_free_run_on_the_ingested_meteo():
    """The meteo read from the archive (meteo.Meteo) instead of the dumps: the raw fluxes,
    surface pressures, initial air mass, temperature, convection parameters and diffusion
    coefficients are the model's bitwise, and the run carried on them stays so."""
    results = replay.replay(DUMP_DIR, free_run=True, meteo_root=METEO_ROOT, limit=40)
    _assert_free_run_matches(results)
    arrays = {kind: row["arrays"] for kind, row in replay.summarize(results).items()}
    assert set(arrays["meteo"]) >= {f"in:{n}" for n in (*replay.MASS_INPUTS, "m")}
    conv = ("entu", "detu", "entd", "detd", "cloud_top", "cloud_lfs", "dkg")
    assert set(arrays["v"]) >= {f"in:{n}" for n in conv}
    assert "in:temper" in arrays["c"]
    assert all(a["n_diff"] == 0 for row in arrays.values() for n, a in row.items() if n.startswith("in:"))


@needs(DUMP_DIR, "PYDYN_DUMP_DIR")
@needs(EMIS_DIR, "PYDYN_EMIS_DIR")
def test_emission_field_is_the_dumped_one():
    """The month's field from the 1x1 file -- unit factor, accumulated-latitude cell areas,
    the month's seconds, the FillGrid block sums -- is what every `emis` dump carries,
    bitwise; and so is the injection from the calendar's month2dt."""
    events = load_events(DUMP_DIR, tags=["emis"])
    assert len(events) >= 40
    emissions = emis.Emissions(EMIS_DIR)
    for ev in events:
        m = ev.meta
        t = balance.date(m["idate"])
        assert compare(ev["emis"].data, emissions.field(t))["equal"]
        assert emissions.month(t)[1] == m["sec_month"]
        x = emis.inject(ev["emis"].data, m["month2dt"], m["mol_mass"], m["mol_mass_e"], m["efrac"])
        got = emissions.injection(t, m["nsrce"], m["mol_mass"], m["mol_mass_e"], m["efrac"], m["tref"])
        np.testing.assert_array_equal(got, x)
    assert len(emissions._fields) == 1  # one month, read once


@needs_model_libraries
@needs(DUMP_DIR, "PYDYN_DUMP_DIR")
@needs(METEO_ROOT, "PYDYN_METEO_ROOT")
@needs(EMIS_DIR, "PYDYN_EMIS_DIR")
def test_free_run_on_the_ingested_meteo_and_emissions():
    """The run carried on the archive's meteo and the file's emission: `in:emis` bitwise on
    every s step, the state after it too."""
    results = replay.replay(DUMP_DIR, free_run=True, meteo_root=METEO_ROOT, emissions=EMIS_DIR, limit=60)
    _assert_free_run_matches(results)
    arrays = replay.summarize(results)["s"]["arrays"]
    assert arrays["in:emis"]["n_diff"] == 0
    assert all(notes["ingested"] for _, kind, _, notes in results if kind == "s")


@needs_model_libraries
@needs(DUMP_DIR, "PYDYN_DUMP_DIR")
@needs(METEO_ROOT, "PYDYN_METEO_ROOT")
def test_calc_kzz_reproduces_every_dumped_dkg():
    """Calc_Kzz on the archive's surface fields, T, Q and the dumped step fluxes: every hourly
    dkg of the day bitwise (the two bef_v of a step share one)."""
    results = replay.replay(DUMP_DIR, tags={"v"}, meteo_root=METEO_ROOT)
    _assert_replay_matches(results)
    dkg = replay.summarize(results)["v"]["arrays"]["in:dkg"]
    assert dkg["n_diff"] == 0 and sum(kind == "v" for _, kind, _, _ in results) >= 40


def _production_steps(events):
    """(tr, step) of every dynamic step of the dump that has its massflow and meteo dumps."""
    seen = set()
    for k, ev in enumerate(events):
        if ev.tag != "bef_v":
            continue
        tr = inputs.read_interval(ev)
        mf, met = (replay.preceding(events, k, tag) for tag in ("massflow", "meteo"))
        step = replay.dumped_step(mf, met)
        if step is not None and tr not in seen:
            seen.add(tr)
            yield tr, step


@needs(DUMP_DIR, "PYDYN_DUMP_DIR")
@needs(METEO_ROOT, "PYDYN_METEO_ROOT")
def test_simple_calc_kzz_on_the_production_steps():
    """The readable Calc_Kzz on the day's steps: gph bitwise, everything else within a few
    ulp except the ill-conditioned boundary-layer height of convective columns (1e-13)."""
    met = meteo.Meteo(meteo.Archive(METEO_ROOT))
    steps = list(_production_steps(load_events(DUMP_DIR)))
    assert len(steps) >= 20
    within_8 = []
    for tr, step in steps:
        args = met.kzz_inputs(tr, step["sp"], step["pu"], step["pv"])
        dkg, parts = diffusion.calc_kzz(*args)
        got, got_parts = simple_diffusion.calc_kzz(*args)
        assert compare(parts["gph"], got_parts["gph"])["equal"]
        for name, a, b in [*((n, parts[n], got_parts[n]) for n in parts), ("dkg", dkg, got)]:
            assert compare(a, b, name)["max_rel"] < 1e-12, (name, tr)
        within_8.append(_ulp_distance(dkg, got) <= 8)
    assert np.mean(np.concatenate([w.ravel() for w in within_8])) > 0.9998


@needs_model_libraries
@needs(FREE_DUMP_DIR, "PYDYN_FREE_DUMP_DIR")
def test_replay_of_the_transport_only_dumps():
    results = replay.replay(FREE_DUMP_DIR)
    _assert_replay_matches(results)
    assert _kinds(results) == {"x", "y", "z", "v", "meteo", "cfl", "massflow", "m2phlb"}


@needs(FREE_DUMP_DIR, "PYDYN_FREE_DUMP_DIR")
def test_replay_on_numpy_fallbacks():
    """The tolerance mode: NumPy's FFT and inverse in place of the model's FFTW and LAPACK, per
    event, so only meteo and v move and every other kind stays bitwise.  Measured on this set
    (2026-11): meteo's fluxes up to 5.9e-10 relative to the element (3.1e6 ulp, in near-zero
    pu1), v's tracers 1.7e-15; the bounds are those with headroom, not a claim."""
    results = replay.replay(FREE_DUMP_DIR, fft="numpy", lapack="numpy")
    assert not results.failures, results.failures
    bound = {"meteo": 1e-9, "v": 1e-14}
    for kind, row in replay.summarize(results).items():
        for name, a in row["arrays"].items():
            if kind in bound:
                assert a["max_rel"] < bound[kind], f"{kind} {name}: {a}"
            else:
                assert a["n_diff"] == 0, f"{kind} {name}: {a}"
    assert _kinds(results) >= {"meteo", "v", "x"}


def test_a_raising_event_is_counted(tmp_path, monkeypatch):
    """A gate must prove it can fail: an operator that raises is in `failures`, in no result,
    and fails the command line."""
    if not FREE_DUMP_DIR.exists():
        pytest.skip(f"no {FREE_DUMP_DIR} (set PYDYN_FREE_DUMP_DIR)")
    real = replay.replay_operator

    def broken(zm, ev, *a, **k):
        if ev.tag == "bef_y":
            raise RuntimeError("injected")
        return real(zm, ev, *a, **k)

    monkeypatch.setattr(replay, "replay_operator", broken)
    results = replay.replay(FREE_DUMP_DIR, tags={"x", "y"}, out=open(tmp_path / "out", "w"))
    assert results.failures and all(tag == "bef_y" for _, tag, _ in results.failures)
    assert "y" not in _kinds(results) and "x" in _kinds(results)
    assert replay.main([str(FREE_DUMP_DIR), "--tags", "x,y"]) == 1


@needs_model_libraries
@needs(FREE_DUMP_DIR, "PYDYN_FREE_DUMP_DIR")
def test_free_run_of_the_transport_only_dumps():
    results = replay.replay(FREE_DUMP_DIR, free_run=True)
    _assert_free_run_matches(results)
    assert sum(kind == "x" for _, kind, _, _ in results) >= 40  # a day, not a smoke test


@needs_model_libraries
@needs(CFL_DUMP_DIR, "PYDYN_CFL_DUMP_DIR")
def test_free_run_through_a_reduced_timestep():
    """What the production timing cannot show: Check_CFL rejecting ndyn = nread and the step
    then advected at the reduced ndyn with the Python's own (am, bm, cm)."""
    results = replay.replay(CFL_DUMP_DIR, free_run=True)
    _assert_free_run_matches(results)
    (cfl,) = [notes for _, kind, _, notes in results if kind == "cfl"]
    nread = replay.load_events(CFL_DUMP_DIR)[0].meta["nread"]
    assert cfl["iterations"] > 1 and cfl["ndyn"] < nread and cfl["n"] * cfl["ndyn"] == nread
    assert {kind for _, kind, _, _ in results} >= {"x", "y", "z", "v", "massflow", "m2phlb"}
