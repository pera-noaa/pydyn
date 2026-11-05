"""PYD-4: the last inputs a free run took from the model's dumps -- the loss-rate
climatologies and soil map of the chemistry (chemfields.py) and the initial tracer field
(initial.py) -- read from the files the model reads, bitwise; and the free run that then
takes no field from a dump at all.  The file-backed tests skip when the input tree, the IC
file or the dump set is absent ($PYDYN_CHEM_ROOT, $PYDYN_IC_FILE, $PYDYN_DUMP_DIR)."""


import numpy as np
import pytest
from netCDF4 import Dataset

from pydyn import balance, chemfields, emis, initial, inputs, lapack, replay
from pydyn.dump import load_events
from pydyn.grid import LL
from pydyn.harness import STATE, compare

from common import CHEM_ROOT, DUMP_DIR, EMIS_DIR, IC_FILE, METEO_ROOT, needs


# ------------------------------------------------------------------ the regridding helpers
def _interpolate_lat_loop(lats_in, vals_in, lats_out):
    """read_OH_field interpolate_lat, statement by statement, for one column."""
    order = np.argsort(lats_in, kind="stable")
    lat, val = lats_in[order], vals_in[order]
    out = np.full(lats_out.size, np.nan)
    out[lats_out < lat[0]] = val[0]
    out[lats_out > lat[-1]] = val[-1]
    for i, x in enumerate(lats_out):
        for j in range(lat.size - 1):
            if lat[j + 1] >= x and lat[j] <= x:
                s1, s2 = lat[j + 1] - x, x - lat[j]
                out[i] = (val[j] * s1 + val[j + 1] * s2) / (s1 + s2)
    return out


def test_interpolate_lat_is_the_fortran_loop():
    """Clamped outside the input range, and on an input latitude the later of the two
    bracketing intervals wins -- the loop has no `exit`."""
    rng = np.random.default_rng(0)
    lats_in = np.sort(rng.uniform(-87.0, 87.0, 30))[::-1].copy()  # descending, as some files are
    lats_out = np.concatenate([np.arange(-90.0, 90.1, 4.0), lats_in[[3, 11, 29, 0]]])
    vals = rng.uniform(0.0, 1e-6, (30, 7))
    got = chemfields.interpolate_lat(lats_in, vals, lats_out)
    for k in range(vals.shape[1]):
        np.testing.assert_array_equal(got[:, k], _interpolate_lat_loop(lats_in, vals[:, k], lats_out))


def test_interpolation_maps_clamp_latitude_and_wrap_longitude():
    lat = np.linspace(-87.5, 87.5, 64)
    idx1, idx2, w1, w2 = chemfields.interpolation_map(lat, np.array([-89.0, -87.5, 0.1, 87.5, 89.0]))
    assert (idx1[[0, 1]] == 0).all() and (idx2[[0, 1]] == 0).all() and (idx1[[3, 4]] == 63).all()
    assert (w1[[0, 1, 3, 4]] == 1.0).all() and (w2[[0, 1, 3, 4]] == 0.0).all()
    assert idx2[2] == idx1[2] + 1 and 0.0 < w2[2] < 1.0

    lon = np.arange(128) * 2.8125  # 0 .. 357.1875, the Hossaini file's
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    out = np.array([-177.0, 0.0, 358.0 - 360.0, 179.0])
    idx1, idx2, w1, w2 = chemfields.interpolation_map(lon, out, cyclic=True)
    assert (idx1[2], idx2[2]) == (127, 0)  # past the last point: wraps onto the first
    assert w1[1] == 1.0 and w2[1] == 0.0 and idx1[1] == 0  # on a point: all of it
    np.testing.assert_allclose(w1 + w2, 1.0, rtol=1e-15)


def test_clump_levels_is_the_thickness_weighted_mean():
    sizes = [a - b for a, b in zip(chemfields.ECHLEVS[:-1], chemfields.ECHLEVS[1:], strict=True)]
    assert sum(sizes) == 137 and len(sizes) == 34
    rng = np.random.default_rng(1)
    dz = rng.uniform(10.0, 1000.0, (3, 2, 137))
    out = chemfields.clump_levels(np.full((3, 2, 137), 7.0), dz)
    np.testing.assert_allclose(out, 7.0, rtol=1e-15)
    f = rng.uniform(0.0, 1.0, (3, 2, 137))
    got = chemfields.clump_levels(f, dz)
    want = (f[..., :3] * dz[..., :3]).sum(-1) / dz[..., :3].sum(-1)  # the first clump: 137 - 134
    np.testing.assert_allclose(got[..., 0], want, rtol=1e-14)


# ------------------------------------------------------------------ the initial field
def _ic_file(path, nlon=12, nlat=6, lev=3, value=1.8, at=None, bt=None, dlon=None):
    dlon = 360.0 / nlon if dlon is None else dlon
    at = np.array([0.0, 5000.0, 10000.0, 0.0]) if at is None else at
    bt = np.array([1.0, 0.5, 0.0, 0.0]) if bt is None else bt
    with Dataset(path, "w") as ds:
        ds.createDimension("lon", nlon)
        ds.createDimension("lat", nlat)
        ds.createDimension("lev", lev)
        ds.createDimension("boundary", lev + 1)
        ds.createVariable("lon", "f8", ("lon",))[:] = -180.0 + dlon / 2.0 + np.arange(nlon) * dlon
        ds.createVariable("lat", "f8", ("lat",))[:] = -90.0 + 90.0 / nlat + np.arange(nlat) * 180.0 / nlat
        ds.createVariable("at", "f8", ("boundary",))[:] = at
        ds.createVariable("bt", "f8", ("boundary",))[:] = bt
        v = ds.createVariable("rm", "f8", ("lev", "lat", "lon"))
        v.units = "ppm"
        v[:] = value
    return at, bt


GRID = {"ll": LL.region(-180.0, -90.0, 60.0, 60.0, 6, 3)}


def test_initial_state_is_ppm_times_air_mass(tmp_path):
    a, b = _ic_file(tmp_path / "ic.nc")
    m = np.random.default_rng(2).uniform(1e9, 1e10, (6, 3, 3))
    st = initial.initial_state(tmp_path / "ic.nc", m, a, b, emis.XMCH4, ntracer=2, **GRID)
    field = initial.remap(*initial.read_field(tmp_path / "ic.nc"), a, b, **GRID)
    np.testing.assert_allclose(field, 1.8, rtol=1e-15)  # a weighted mean: not 1.8 to the bit
    want = ((field * initial.PPM) * m) / (initial.XMAIR / emis.XMCH4)
    for t in range(2):
        np.testing.assert_array_equal(st["rm"][..., t], want)
    assert all((st[k] == 0.0).all() and st[k].shape == (6, 3, 3, 2) for k in ("rxm", "rym", "rzm"))


def test_initial_state_flips_reversed_levels(tmp_path):
    at, bt = np.array([0.0, 5000.0, 10000.0, 0.0]), np.array([1.0, 0.5, 0.0, 0.0])
    _ic_file(tmp_path / "ic.nc", at=at[::-1].copy(), bt=bt[::-1].copy())
    with Dataset(tmp_path / "ic.nc", "a") as ds:
        ds["rm"][:] = np.arange(3.0)[:, None, None] + 1.0  # file level k holds k + 1
    field = initial.remap(*initial.read_field(tmp_path / "ic.nc"), at, bt, **GRID)
    np.testing.assert_allclose(field[..., 0], 3.0, rtol=1e-15)
    np.testing.assert_allclose(field[..., 2], 1.0, rtol=1e-15)


def test_initial_state_refuses_what_is_not_transcribed(tmp_path):
    a, b = _ic_file(tmp_path / "ic.nc", dlon=25.0)  # edges do not meet the model's
    with pytest.raises(NotImplementedError, match="longitudes"):
        initial.initial_state(tmp_path / "ic.nc", np.ones((6, 3, 3)), a, b, emis.XMCH4, **GRID)
    a, b = _ic_file(tmp_path / "ic2.nc")
    with pytest.raises(NotImplementedError, match="levels"):
        initial.initial_state(tmp_path / "ic2.nc", np.ones((6, 3, 3)), a + 50.0, b, emis.XMCH4, **GRID)
    _ic_file(tmp_path / "ic3.nc", value=1.8e-6)  # kg/kg-sized: the model's #9d guard
    with pytest.raises(ValueError, match="outside"):
        initial.initial_state(tmp_path / "ic3.nc", np.ones((6, 3, 3)), a, b, emis.XMCH4, **GRID)


# ------------------------------------------------------------------ against the model
@needs(DUMP_DIR, "PYDYN_DUMP_DIR")
@needs(CHEM_ROOT, "PYDYN_CHEM_ROOT")
def test_loss_fields_are_the_dumped_ones():
    """Every `chem` event's Bruehl columns, Spivakovsky OH, Hossaini Cl and soil map, from the
    files, bitwise; each file (each month of the OH) read once."""
    events = load_events(DUMP_DIR, tags=["chem"])
    assert len(events) >= 40
    fields = chemfields.LossFields(CHEM_ROOT)
    for ev in events:
        m = ev.meta
        t = inputs.midpoint(ev)
        got = fields.month(t.month, t.year, m["trop_cl_sink"], m["soil_sink"])
        want = {"strat_press", "strat_oh", "strat_cl", "strat_od", "oh_3d", "cl_trop", "pl_trop", "soil"}
        assert set(got) == want
        for name, q in got.items():
            assert compare(ev[name].data, q, name)["equal"], name
    assert len(fields._oh) == 1 and len(fields._soil) == 1


@needs(DUMP_DIR, "PYDYN_DUMP_DIR")
@needs(IC_FILE, "PYDYN_IC_FILE")
def test_initial_field_is_the_first_dumped_state():
    """istart 66 from the file and the run's initial air mass: the tracers of the first bef_*
    dump, bitwise."""
    events = load_events(DUMP_DIR, tags=["meteo", "bef_x"])
    met = next(e for e in events if e.tag == "meteo")
    bef = next(e for e in events if e.tag == "bef_x")
    m = bef.interior("m")
    assert compare(met.interior("m"), m)["equal"]  # nothing touches the air mass before it
    nt = bef.interior("rm").shape[-1]
    st = initial.initial_state(IC_FILE, m, met["at"].data, met["bt"].data, emis.XMCH4, nt)
    for name in STATE[1:]:
        assert compare(bef.interior(name), st[name], name)["equal"], name


@needs(DUMP_DIR, "PYDYN_DUMP_DIR")
@needs(METEO_ROOT, "PYDYN_METEO_ROOT")
@needs(EMIS_DIR, "PYDYN_EMIS_DIR")
@needs(CHEM_ROOT, "PYDYN_CHEM_ROOT")
@needs(IC_FILE, "PYDYN_IC_FILE")
@pytest.mark.skipif(not (balance.fftw_library() and lapack.lapack_library()),
                    reason="needs the model's FFTW and LAPACK")
def test_free_run_on_nothing_but_the_input_files():
    """The whole day carried by the Python from the initial field on, with the meteo, the
    emission, the loss-rate inputs and the initial tracers all read from the files: every
    checkpoint bitwise, and not one event seeded from a dump -- the first chemistry, which
    runs before any m2phlb, on the initial half levels of the ingested air mass."""
    results = replay.replay(
        DUMP_DIR, free_run=True, meteo_root=METEO_ROOT, emissions=EMIS_DIR, chemistry=CHEM_ROOT,
        initial_file=IC_FILE,
    )  # fmt: skip
    assert not results.failures, results.failures
    table = replay.summarize(results)
    bad = {(k, n): a for k, row in table.items() for n, a in row["arrays"].items() if a["n_diff"]}
    assert not bad, bad
    seeded = [str(ev) for ev, _, _, notes in results if notes.get("seeded")]
    assert not seeded, seeded
    assert results[0][3].get("initial") == "istart 66"
    names = {n for row in table.values() for n in row["arrays"]}
    ingested = {"in:emis", "in:temper", "in:oh_3d", "in:strat_oh", "in:cl_trop", "in:soil", "ic:rm", "in:m"}
    assert names >= ingested, ingested - names
    assert sum(kind == "c" for _, kind, _, _ in results) >= 40
