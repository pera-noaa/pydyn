"""The reduced grid (redgridZoom.F90, proj/ct-unified/trunk) and x advection on it.

Near the poles TM5 merges `clustsize` neighbouring cells of a latitude row into one before
the x advection and splits them again afterwards, so that the row is advected with
`imredj = im / clustsize` cells (advectx_work in advectx__slopes.F90):

    m_uni, am_uni = m, am
    uni2red      -- m, rm, rxm, rym, rzm summed over each cluster into cells 1..imredj
    uni2red_mf   -- am(i) = am(i*clustsize), am(0) = am(imredj), the rest set to -1
    dynamu       -- every row advected as a periodic row of imred(j) cells
    m_uni       += am_uni(0:im-1) - am_uni(1:im)          (the uniform-grid mass update)
    red2uni      -- rm.. of cluster cell i spread over its cells by m_uni(is:ie) / m(i),
                    m(is:ie) = m_uni(is:ie)
    am           = am_uni

Arrays are in the pydyn layout (layout.py); `RedGrid.imred[j]` is 0-based in j.
"""

import numpy as np

from .kernels import _advect_rows_x
from .ops import advance, clusters, seqsum

__all__ = ["RedGrid", "advect_x_reduced", "red2uni", "red2uni_mass", "uni2red", "uni2red_mass", "uni2red_mf"]


class RedGrid:
    """Reduced-grid description of one region: `bands` = [(j0, clustsize, imredj), ...]."""

    def __init__(self, im, jm, jred, clustsize):
        self.im, self.jm = im, jm
        self.bands = []
        self.imred = np.full(jm, im, dtype=int)
        for j, c in zip(jred, clustsize, strict=True):
            if im % c:
                raise ValueError(f"redgrid: im={im} not a multiple of clustsize {c}")
            self.bands.append((j - 1, c, im // c))
            self.imred[j - 1] = im // c

    @classmethod
    def from_meta(cls, meta):
        """From a dump event's header (nred, jred, clustsize); None when not reduced."""
        if not meta.get("grid_reduced") or not meta.get("nred"):
            return None
        nred = meta["nred"]
        jred, clust = meta["jred"], meta["clustsize"]
        if isinstance(jred, int):
            jred, clust = (jred,), (clust,)
        return cls(meta["im"], meta["jm"], jred[:nred], clust[:nred])

    @property
    def rows(self):
        return [b[0] for b in self.bands]


def uni2red(grid, m, rm, rxm, rym, rzm):
    """Sum every cluster into cells 0..imredj-1 of its row; tracers beyond are zeroed, m is not."""
    uni2red_mass(grid, m)
    for j, c, n in grid.bands:
        for q in (rm, rxm, rym, rzm):
            q[:n, j] = seqsum(clusters(q[:, j], n, c), axis=1)
            q[n:, j] = 0.0


def uni2red_mass(grid, m):
    for j, c, n in grid.bands:
        m[:n, j] = seqsum(clusters(m[:, j], n, c), axis=1)


def uni2red_mf(grid, am):
    """am[i] = am[i*clustsize] for i = 1..imredj, am[0] = am[imredj], rest -1 (never read)."""
    for j, c, n in grid.bands:
        row = am[:, j]
        row[1 : n + 1] = row[c * np.arange(1, n + 1)]
        row[n + 1 :] = -1.0
        row[0] = row[n]


def red2uni(grid, m, rm, rxm, rym, rzm, m_uni):
    """Split the cluster cells again, distributing the tracers with the uniform-grid masses."""
    for j, c, n in grid.bands:
        scale = clusters(m_uni[:, j], n, c) / m[:n, j, None]  # (n, c, lm): m_uni(is:ie) / m(i)
        for q in (rm, rxm, rym, rzm):
            q[: n * c, j] = (scale[..., None] * q[:n, j, None]).reshape(n * c, *q.shape[2:])
    red2uni_mass(grid, m, m_uni)


def red2uni_mass(grid, m, m_uni):
    for j, _, _ in grid.bands:
        m[:, j] = m_uni[:, j]


def advect_x_reduced(m, rm, rxm, rym, rzm, am, grid, limits=True, max_nloop=50, stats=None):
    """advectx_work with a reduced grid: kernels.advect_x by latitude, reduced rows with
    imred[j] cells.  In place (am is handed back unchanged, as in the Fortran)."""
    _, jm, _ = m.shape
    m_uni, am_uni = m.copy(), am.copy()
    uni2red(grid, m, rm, rxm, rym, rzm)
    uni2red_mf(grid, am)
    nloop_max, xi = 1, 0.0
    for j in range(jm):  # every level of a latitude at once
        n = grid.imred[j]
        rows = (slice(0, n), j)
        nloop, alpha_max = _advect_rows_x(*(q[rows] for q in (m, rm, rxm, rym, rzm)), am[0 : n + 1, j],
                                          limits, max_nloop)  # fmt: skip
        nloop_max, xi = max(nloop_max, int(nloop.max())), max(xi, float(alpha_max.max()))
    red2uni(grid, m, rm, rxm, rym, rzm, advance(m_uni, am_uni))
    am[...] = am_uni
    if stats is not None:
        stats["nloop_max"], stats["xi"] = nloop_max, xi
