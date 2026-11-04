"""The meteo ingestion (meteo.py): the time rules, the surface-pressure match, the convection fields."""

from datetime import datetime, timedelta

import numpy as np

from pydyn import (
    balance,
    grid,
    meteo,
)

from common import (
    IM,
    JM,
    LM,
)


# ------------------------------------------------------------------ the meteo ingestion
def test_instants_and_interval_follow_the_3h_archive():
    d = datetime(2000, 1, 1)
    h = timedelta(hours=1)
    assert meteo.instants((d, d + h)) == (d, d + 3 * h)
    assert meteo.instants((d + 2 * h, d + 3 * h)) == (d, d + 3 * h)
    assert meteo.instants((d + 3 * h, d + 3 * h)) == (d + 3 * h, d + 3 * h)
    assert meteo.instants((d + 23 * h, d + 24 * h)) == (d + 21 * h, d + 24 * h)
    assert meteo.instants((d + timedelta(minutes=30), d + h), dth=1) == (d, d + h)
    assert meteo.interval(d + 23 * h + timedelta(minutes=30)) == (d + 21 * h, d + 24 * h)


def test_interpolate_returns_the_record_itself_at_a_coincident_instant():
    rng = np.random.default_rng(11)
    d = datetime(2000, 1, 1, 3)
    data = rng.uniform(1.0e4, 1.1e5, (IM, JM))
    assert (meteo.interpolate((d, d), d, data, d, data) == data).all()
    third = meteo.interpolate((d + timedelta(hours=1),) * 2, d, data, d + timedelta(hours=3), 4.0 * data)
    np.testing.assert_allclose(third, 2.0 * data, rtol=1e-15)




def test_match_global_mean_is_a_constant_shift_to_the_target():
    rng = np.random.default_rng(12)
    area = grid.cell_areas(IM, JM, 360.0 / IM, 180.0 / JM, -90.0 + 90.0 / JM)
    sp = rng.uniform(9.0e4, 1.05e5, (IM, JM))
    matched = meteo.match_global_mean(sp, area, grid.globe_area())
    shift = matched - sp
    assert np.ptp(shift) == 0.0
    np.testing.assert_allclose((matched * area).sum() / grid.globe_area(), meteo.SP_REGION0, rtol=1e-14)


def test_pressure_to_mass_is_the_inverse_of_m2phlb1():
    rng = np.random.default_rng(13)
    sp = rng.uniform(9.0e4, 1.05e5, (IM, JM))
    # a hybrid table: pure sigma at the surface, pure pressure (0 Pa) at the top
    a, b = 1.0e4 * np.sin(np.linspace(0.0, np.pi, LM + 1)), np.linspace(1.0, 0.0, LM + 1)
    area_m2 = 1.0e10 * rng.uniform(0.5, 1.0, JM)
    phlb, m = meteo.pressure_to_mass(sp, a, b, area_m2)
    assert phlb.shape == (IM, JM, LM + 1) and m.shape == (IM, JM, LM) and (m > 0.0).all()
    sp_back, phlb_back = balance.m2phlb1(m, a, b, area_m2)
    np.testing.assert_allclose(sp_back, sp, rtol=1e-14)
    np.testing.assert_allclose(phlb_back, phlb, rtol=1e-14)


def test_conv_unit_convert_closes_every_column_at_its_largest_entrainment():
    rng = np.random.default_rng(14)
    dxyp = 1.0e10 * rng.uniform(0.5, 1.0, JM)
    raw = [rng.uniform(0.0, 1.0e-3, (IM, JM, LM)) for _ in range(4)]
    raw[2][0, 0] = 0.0  # a column without downdraft entrainment
    entu, detu, entd, detd = meteo.conv_unit_convert(*raw, dxyp, czeta=0.5)
    for ent, det, ent_raw in ((entu, detu, raw[0]), (entd, detd, raw[2])):
        np.testing.assert_allclose(ent.sum(-1), det.sum(-1), rtol=1e-12)
        changed = ent != (ent_raw * dxyp[None, :, None]) * 0.5
        assert changed.sum(-1).max() == 1  # one level per column
    scaled = (raw[0] * dxyp[None, :, None]) * 0.5
    assert (np.argmax(entu != scaled, axis=-1) == np.argmax(raw[0], axis=-1)).all()
    assert np.argmax(entd[0, 0] != 0.0) == LM - 1 and detd[0, 0].sum() > 0.0


def test_cloud_dims_read_the_detrainment_profiles():
    detu = np.zeros((2, 1, LM))
    entd = np.zeros((2, 1, LM))
    detu[0, 0, [1, 3]] = 1.0  # levels 2 and 4 (1-based)
    entd[0, 0, 2] = 1.0
    top, lfs, base = meteo.cloud_dims(detu, entd)
    assert (top[:, 0].tolist(), lfs[:, 0].tolist(), base[:, 0].tolist()) == ([4, 0], [3, 0], [2, 0])


def test_month_and_the_surface_time_rules():
    d = datetime(2000, 1, 1)
    h = timedelta(hours=1)
    assert meteo.month(d + 23 * h) == (d, datetime(2000, 2, 1))
    assert meteo.month(datetime(2000, 12, 31, 23)) == (datetime(2000, 12, 1), datetime(2001, 1, 1))
    assert meteo.instants((d + 5 * h, d + 6 * h), dth=1) == (d + 5 * h, d + 6 * h)
    assert meteo.interval(d + 5 * h, dth=1) == (d + 5 * h, d + 6 * h)
