"""TM5's lon/lat grids and FillGrid (grid.py)."""


import numpy as np

from pydyn import (
    grid,
    meteo,
)

from common import (
    IM,
    JM,
)


# ------------------------------------------------------------------ grid.py
def test_cell_areas_tile_the_sphere():
    area = grid.cell_areas(IM, JM, 360.0 / IM, 180.0 / JM, -90.0 + 90.0 / JM)
    assert area.shape == (JM,) and (area > 0.0).all() and np.allclose(area, area[::-1])
    np.testing.assert_allclose(area.sum() * IM, 4.0 * np.pi, rtol=1e-14)
    np.testing.assert_allclose(grid.globe_area(), 4.0 * np.pi, rtol=1e-14)
    dxyp = meteo.dxyp(JM, 360.0 / IM, 180.0 / JM, -90.0)
    np.testing.assert_allclose(dxyp.sum() * IM, 4.0 * np.pi * grid.AE**2)


def test_block_mean_is_the_fillgrid_loop():
    rng = np.random.default_rng(15)
    nx, ny = 3, 2
    f = rng.uniform(0.0, 1.0, (IM * nx, JM * ny))
    w = rng.uniform(0.5, 1.0, JM * ny)
    got = grid.block_mean(f, w, nx, ny)
    assert got.shape == (IM, JM)
    for i in range(IM):
        for j in range(JM):
            res, resw = 0.0, 0.0
            for jx in range(j * ny, (j + 1) * ny):
                for ix in range(i * nx, (i + 1) * nx):
                    res = res + f[ix, jx] * w[jx]
                    resw = resw + w[jx]
            assert got[i, j] == res / resw
    flat = grid.block_mean(np.full((IM * nx, JM * ny), 2.5), w, nx, ny)
    np.testing.assert_allclose(flat, 2.5, rtol=1e-15)


def test_block_sum_is_the_fillgrid_loop():
    rng = np.random.default_rng(21)
    nx, ny = 3, 2
    f = rng.uniform(0.0, 1.0, (IM * nx, JM * ny))
    got = grid.block_sum(f, nx, ny)
    assert got.shape == (IM, JM)
    for i in range(IM):
        for j in range(JM):
            res = 0.0
            for jx in range(j * ny, (j + 1) * ny):
                for ix in range(i * nx, (i + 1) * nx):
                    res = res + f[ix, jx]
            assert got[i, j] == res
    np.testing.assert_allclose(got.sum(), f.sum(), rtol=1e-14)
