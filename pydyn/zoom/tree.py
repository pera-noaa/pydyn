"""The region tree (initexit.F90): the dims_grid parameters and what initexit derives.  Part of
pydyn.zoom (see its docstring)."""

from dataclasses import dataclass

import numpy as np

from ..grid import LL
from ..ops import nint

__all__ = [
    "N_OPERATORS",
    "SPLITORDER",
    "Region",
    "Zoom",
]

SPLITORDER = "xyzvsccsvzyx"  # dims.F90 splitorder
N_OPERATORS = len(SPLITORDER) // 2


# ------------------------------------------------------------------ the region tree (initexit)
@dataclass
class Region:
    """One region as the pycasso `region.<name>.*` keys describe it.  Refinement factors are
    absolute (relative to the globe, as dims_grid has them); `parent` is a region number,
    0 for the global region."""

    name: str
    im: int
    jm: int
    xbeg: float = -180.0
    xend: float = 180.0
    ybeg: float = -90.0
    yend: float = 90.0
    xref: int = 1
    yref: int = 1
    zref: int = 1
    tref: int = 1
    parent: int = 0
    xcyc: int = 0
    touch_sp: int = 0
    touch_np: int = 0
    redgrid: object = None  # redgrid.RedGrid of the region's x advection, or None

    @property
    def ll(self):
        """lli(region): grid.LL from the corner and (xend - xbeg) / im, (yend - ybeg) / jm."""
        dx, dy = (self.xend - self.xbeg) / self.im, (self.yend - self.ybeg) / self.jm
        return LL.region(self.xbeg, self.ybeg, dx, dy, self.im, self.jm)


class Zoom:
    """The region tree: the dims_grid parameters plus what initexit derives from them --
    `children`, the child footprints `ibeg..lend`, the interior scopes `isr..jer`, the
    `zoomed`/`edge` maps of blank_zoom_region and the per-region `splitorderzoom`.  Regions
    are numbered from 1 as in the Fortran; `status[r]` is the position counter do_steps
    advances."""

    def __init__(self, regions, lm, at=None, bt=None, splitorder=SPLITORDER):
        self.region = [None, *regions]
        self.nregions = len(regions)
        self.lm = lm
        self.at = np.zeros(lm + 1) if at is None else np.asarray(at, dtype=float)
        self.bt = np.zeros(lm + 1) if bt is None else np.asarray(bt, dtype=float)
        self.status = [0] * (self.nregions + 1)
        self._determine_children_etc()
        self.zoomed, self.edge = {}, {}
        for r in self.regions():
            self.zoomed[r], self.edge[r] = self._blank_zoom_region(r)
        self.splitorderzoom = self._define_splitorderzoom(splitorder)
        self._check_scope()

    def regions(self):
        return range(1, self.nregions + 1)

    def g(self, r):
        """The Region of number r (r = 0 is the globe: all refinements 1)."""
        return self.region[r] if r > 0 else Region("globe", self.region[1].im, self.region[1].jm)

    def ref(self, r, axis):
        """xref_/yref_/zref_/tref_: refinement of r relative to its parent, as integer division."""
        g, p = self.g(r), self.g(self.g(r).parent)
        return getattr(g, axis + "ref") // getattr(p, axis + "ref")

    # determine_children_etc (initexit.F90)
    def _determine_children_etc(self):
        n = self.nregions
        self.children = {r: [] for r in range(0, n + 1)}
        for r in range(2, n + 1):
            self.children[self.g(r).parent].append(r)
        self.ibeg, self.iend, self.jbeg, self.jend = {}, {}, {}, {}
        self.lbeg, self.lend = {}, {}
        g1 = self.g(1)
        dx = (g1.xend - g1.xbeg) / g1.im * g1.xref  # dims dx, dy: the grid at refinement 1
        dy = (g1.yend - g1.ybeg) / g1.jm * g1.yref
        for r in range(2, n + 1):
            g, p = self.g(r), self.g(self.g(r).parent)
            self.ibeg[r] = nint((g.xbeg - p.xbeg) / (dx / p.xref)) + 1
            self.jbeg[r] = nint((g.ybeg - p.ybeg) / (dy / p.yref)) + 1
            self.iend[r] = nint((g.xend - p.xbeg) / (dx / p.xref))
            self.jend[r] = nint((g.yend - p.ybeg) / (dy / p.yref))
            self.lbeg[r], self.lend[r] = 1, self.lm  # zoom2D: no refinement in z
            xr, yr = self.ref(r, "x"), self.ref(r, "y")
            if self.iend[r] > self.ibeg[r]:
                icells = self.iend[r] - self.ibeg[r] + 1
            else:
                icells = p.im - self.ibeg[r] + 1 + self.iend[r]
            if icells * xr != g.im:
                raise ValueError(f"region {r}: check xbeg/xend, im and ibeg/iend")
            if (self.jend[r] - self.jbeg[r] + 1) * yr != g.jm:
                raise ValueError(f"region {r}: check ybeg/yend, jm and jbeg/jend")
        self.isr, self.ier, self.jsr, self.jer = {1: 1}, {1: g1.im}, {1: 1}, {1: g1.jm}
        for r in range(2, n + 1):
            g, xr, yr = self.g(r), self.ref(r, "x"), self.ref(r, "y")
            self.isr[r], self.ier[r] = xr + 1, g.im - xr
            self.jsr[r], self.jer[r] = yr + 1, g.jm - yr
            if g.xcyc != 0:
                self.isr[r], self.ier[r] = 1, g.im
            if g.touch_sp == 1:
                self.jsr[r] = 1
            if g.touch_np == 1:
                self.jer[r] = g.jm

    # blank_zoom_region (initexit.F90); 0-based numpy arrays (im, jm)
    def _blank_zoom_region(self, r):
        g = self.g(r)
        zoomed = np.full((g.im, g.jm), r, dtype=np.int32)
        edge = np.zeros((g.im, g.jm), dtype=np.int32)
        for c in self.children[r]:
            gc = self.g(c)
            ib, ie, jb, je = self.ibeg[c], self.iend[c], self.jbeg[c], self.jend[c]
            for i in range(1, g.im + 1):
                if (i <= ib or i >= ie) and gc.xcyc != 1:
                    continue
                for j in range(1, g.jm + 1):
                    if j <= jb and gc.touch_sp != 1:
                        continue
                    if j >= je and gc.touch_np != 1:
                        continue
                    zoomed[i - 1, j - 1] = c
            if gc.xcyc != 1:
                edge[ib - 1, jb - 1 : je] = 1
                edge[ie - 1, jb - 1 : je] = 1
            if gc.touch_sp != 1:
                edge[ib - 1 : ie, jb - 1] += 2
            if gc.touch_np != 1:
                edge[ib - 1 : ie, je - 1] += 2
        if g.parent != 0:
            xr, yr = self.ref(r, "x"), self.ref(r, "y")
            if g.xcyc != 1:
                edge[:xr, :] = -1
                edge[g.im - xr :, :] = -1
            if g.touch_sp != 1:
                edge[:, :yr] = -1
            if g.touch_np != 1:
                edge[:, g.jm - yr :] = -1
        return zoomed, edge

    # define_splitorderzoom (initexit.F90), zoom_mode = 1: even copies reversed
    def _define_splitorderzoom(self, splitorder):
        order = {1: splitorder}
        for r in range(2, self.nregions + 1):
            tref_ = self.ref(r, "t")
            if tref_ > 4 or tref_ == 3:
                raise ValueError(f"define_splitorderzoom: region {r}: tref_ = {tref_} not 1, 2 or 4")
            parent = order[self.g(r).parent]
            out = []
            for j0 in range(0, len(parent), N_OPERATORS):
                block = parent[j0 : j0 + N_OPERATORS]
                for i in range(1, tref_ + 1):
                    out.append(block if i % 2 == 1 else block[::-1])
            order[r] = "".join(out)
        return order

    def _check_scope(self):
        g1 = self.g(1)
        if not (g1.xcyc == 1 and g1.touch_sp == 1 and g1.touch_np == 1):
            raise NotImplementedError("region 1 must be global: cyclic and touching both poles")
        for r in range(2, self.nregions + 1):
            g, p = self.g(r), self.g(self.g(r).parent)
            if g.xcyc or g.touch_sp or g.touch_np:
                raise NotImplementedError(f"region {r}: cyclic or pole-touching children are not transcribed")
            if self.ibeg[r] >= self.iend[r]:
                raise NotImplementedError(f"region {r}: a child across the dateline is not transcribed")
            if self.ibeg[r] <= 1 or self.iend[r] >= p.im:
                raise NotImplementedError(f"region {r}: put_xedges would read the parent's halo")
            if self.jbeg[r] <= 1 or self.jend[r] >= p.jm:
                raise NotImplementedError(f"region {r}: put_yedges would read the parent's halo")
            if g.redgrid is not None:
                raise NotImplementedError(f"region {r}: a reduced grid in a child is not transcribed")
