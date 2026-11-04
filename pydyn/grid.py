"""TM5's lon/lat grids and FillGrid between them (grid_type_ll__speedup.F90, grid_tools).

`LL` is one grid as lli's Init takes it -- the first cell's centre, the spacing and the size
-- with the cell areas Init derives.  `relate` is Relate: which cells of a source grid a
target grid takes, and by which action -- 'copy' at the same spacing, 'combine' from a finer
grid, 'distribute' from a coarser one.  `fill` is FillGrid of a cell field ('n'):

    copy        the target's window of the source
    combine     the block of nx x ny source cells under each target cell, accumulated from 0.0
                row by row, west to east: 'sum', or 'area-aver' / 'weight' (each source cell
                weighted by its row's area_m2 / by a weight of its own, then divided by the
                sum of the weights)
    distribute  each source cell's value in every target cell under it ('aver'-type keys)

Everything that regrids a field onto a region goes through these: the 1x1 surface meteo
(meteo.py), tmm's records (regrid.py, which adds the staggered 'u' / 'v' faces), the emission
(emis.py), the chemistry's OH (chemfields.py) and the initial field (initial.py).  geometry.F90's
dxyp, accumulated latitude by latitude, is a different quantity and stays in meteo.py.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .balance import PI
from .ops import libm

__all__ = [
    "AE", "DEG2RAD", "GLB6X4", "LL", "ONE_BY_ONE", "Relation", "block_mean", "block_sum", "cell_areas", "cut",
    "fill", "globe_area", "ll_area", "relate",
]  # fmt: skip

AE = 6.371e6  # Binas: earth radius, m
DEG2RAD = PI / 180.0  # Binas


def ll_area(west, east, south, north):
    """grid_tools ll_area: the area (rad^2) of a lon/lat rectangle given in radians."""
    return (libm.sin(np.maximum(north, south)) - libm.sin(np.minimum(north, south))) * abs(east - west)


def cell_areas(im, jm, dlon_deg, dlat_deg, south_deg):
    """grid_type_ll Init: lli%area (jm,) in rad^2 of the latitude rows, from the first cell
    centre `south_deg` (blat_deg(j) = south + (j - 1/2) dlat, clipped to +-90)."""
    blat_deg = np.clip(south_deg + (np.arange(jm + 1) - 0.5) * dlat_deg, -90.0, 90.0)
    blat = blat_deg * DEG2RAD
    return ll_area(0.0, dlon_deg * DEG2RAD, blat[:-1], blat[1:])


def globe_area():
    """lli(0): the one-cell global grid Init(0.0, 360.0, 1, 0.0, 180.0, 1), area (rad^2)."""
    return cell_areas(1, 1, 360.0, 180.0, 0.0)[0]


@dataclass(frozen=True)
class LL:
    """lli: `im` x `jm` cells of `dlon` x `dlat` degrees, the first centred at (lon0, lat0)."""

    lon0: float
    lat0: float
    dlon: float
    dlat: float
    im: int
    jm: int
    west: float = None  # the corner a region was given by (xbeg, ybeg), kept as given
    south: float = None

    def __post_init__(self):
        if self.west is None:
            object.__setattr__(self, "west", self.lon0 - self.dlon / 2.0)
        if self.south is None:
            object.__setattr__(self, "south", self.lat0 - self.dlat / 2.0)

    @classmethod
    def region(cls, west, south, dlon, dlat, im, jm):
        """A model region by its south-west corner, as lli(region) is Init'ed."""
        return cls(west + dlon / 2.0, south + dlat / 2.0, dlon, dlat, im, jm, west, south)

    @property
    def lon_deg(self):
        return self.lon0 + np.arange(self.im) * self.dlon

    @property
    def lat_deg(self):
        return self.lat0 + np.arange(self.jm) * self.dlat

    def edges(self, axis):
        """The outer cell edges (blon_deg(0), blon_deg(n)) along x (0) or y (1, clipped to +-90)."""
        c0, d, n = (self.lon0, self.dlon, self.im) if axis == 0 else (self.lat0, self.dlat, self.jm)
        e = c0 + (np.array([0, n]) - 0.5) * d
        return np.clip(e, -90.0, 90.0) if axis == 1 else e

    @cached_property
    def area(self):
        """lli%area (jm,), rad^2."""
        return cell_areas(self.im, self.jm, self.dlon, self.dlat, self.lat0)

    @cached_property
    def area_m2(self):
        """lli%area_m2 (jm,), AreaOper '*' 'm2'."""
        return self.area * AE**2


ONE_BY_ONE = LL(-179.5, -89.5, 1.0, 1.0, 360, 180)  # the 1x1 grid of the surface fields and tmm's ml137 set
GLB6X4 = LL.region(-180.0, -90.0, 6.0, 4.0, 60, 45)  # the single-region production grid (ch4_6x4_fwd.rc)


@dataclass(frozen=True)
class Relation:
    """Relate's answer: `action`, the source's first cell (i0, j0), 0-based, and the factor
    -- source cells per target cell for 'combine', target cells per source cell for
    'distribute', 1 for 'copy'."""

    action: str
    i0: int
    j0: int
    nx: int
    ny: int


AXES = ("longitudes", "latitudes")


def _axis(dst, src, axis, what):
    (w, e), (wx, ex) = dst.edges(axis), src.edges(axis)
    d, dx = (dst.dlon, src.dlon) if axis == 0 else (dst.dlat, src.dlat)
    n, nX = (dst.im, src.im) if axis == 0 else (dst.jm, src.jm)
    r_w, r_e = abs(w - wx) / dx, abs(e - ex) / dx
    if not (wx <= w and ex >= e and r_w == round(r_w) and r_e == round(r_e)):
        raise NotImplementedError(f"{what}: the source's {AXES[axis]} do not cover the target's edges")
    i0, nsel = round(r_w), nX - round(r_w) - round(r_e)
    if dx == d:
        return i0, 1
    if dx < d:
        if nsel % n:
            raise NotImplementedError(f"{what}: the {AXES[axis]}: not whole source cells per target cell")
        return i0, -(nsel // n)
    if n % nsel:
        raise NotImplementedError(f"{what}: the {AXES[axis]}: not whole target cells per source cell")
    return i0, n // nsel


def relate(dst, src, what="Relate"):
    """Relate `dst` to `src`, the source covering the target with cell edges on the target's
    (a part-covered target is refused, as no caller needs it)."""
    i0, fx = _axis(dst, src, 0, what)
    j0, fy = _axis(dst, src, 1, what)
    if (fx, fy) == (1, 1):
        return Relation("copy", i0, j0, 1, 1)
    if fx <= 1 and fy <= 1:  # an equal axis combines one cell
        return Relation("combine", i0, j0, -min(fx, -1), -min(fy, -1))
    if fx >= 1 and fy >= 1:
        return Relation("distribute", i0, j0, fx, fy)
    raise NotImplementedError(f"{what}: the source is finer along one axis and coarser along the other")


def cut(field, rel, dst, nuv="n"):
    """The source cells (or faces) the target's window covers: field (nX[+1], mX[+1], ...) ->
    (im*nx [+1], jm*ny [+1], ...) for 'combine' and 'copy', (im/nx, jm/ny, ...) for 'distribute'."""
    if rel.action == "distribute":
        ni, nj = -(-dst.im // rel.nx), -(-dst.jm // rel.ny)
    else:
        ni, nj = dst.im * rel.nx, dst.jm * rel.ny
    return field[rel.i0 : rel.i0 + ni + (nuv == "u"), rel.j0 : rel.j0 + nj + (nuv == "v")]


def block_sum(f, nx, ny):
    """FillGrid 'combine' 'sum' (coarsen_emission_2d avg=0): each coarse cell the sum of its
    nx x ny fine cells of f (nx*im, ny*jm, ...), accumulated from 0.0 row by row, west to east."""
    res = 0.0
    for jx in range(ny):
        for ix in range(nx):
            res = res + f[ix::nx, jx::ny] * 1.0
    return res


def block_mean(f, weights, nx, ny):
    """FillGrid 'combine' 'area-aver' (`weights` the fine rows' area_m2, (ny*jm,)) or 'weight'
    (`weights` one per fine cell, f's shape): the weighted mean of each block, the weighted sum
    and the sum of the weights accumulated from 0.0 row by row, west to east."""
    if weights.ndim == 1:
        weights = weights.reshape((1, -1) + (1,) * (f.ndim - 2))
    res, resw = 0.0, 0.0
    for jx in range(ny):
        for ix in range(nx):
            w = weights[ix::nx, jx::ny] if weights.shape[0] > 1 else weights[:, jx::ny]
            res = res + f[ix::nx, jx::ny] * w
            resw = resw + w
    return res / resw


def fill(field, dst, src, combkey="area-aver", weights=None, rel=None):
    """FillGrid 'n' of a source cell field (src.im, src.jm, ...) onto `dst`: 'copy', 'combine'
    with `combkey` 'sum' / 'area-aver' (weights default the source rows' area_m2) / 'weight'
    (`weights` per source cell), or 'distribute' (each source value repeated)."""
    rel = rel or relate(dst, src)
    sub = cut(field, rel, dst)
    if rel.action == "copy":
        return sub.copy()
    if rel.action == "distribute":
        return np.repeat(np.repeat(sub, rel.nx, axis=0), rel.ny, axis=1)
    if combkey == "sum":
        return block_sum(sub, rel.nx, rel.ny)
    if weights is None:
        weights = src.area_m2
    w = cut(weights, rel, dst) if weights.ndim > 1 else weights[rel.j0 : rel.j0 + dst.jm * rel.ny]
    return block_mean(sub, w, rel.nx, rel.ny)
