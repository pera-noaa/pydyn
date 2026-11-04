"""Meteo_Setup_Mass for a zoom child (`PYD-3` item 3): the child's meteo matched to its parent's
and balanced cell by parent cell.

Transcribed from `tm5-zoom@745e94f70`, as the production chain resolves it:

    meteo__meteo_server.F90  Meteo_Setup_Mass, region n > 1: sp1 / sp2 Match 'area-aver'
                            against the parent's matched sp; the raw fluxes Match 'sum' 'u' /
                            'v' / 'n' against the parent's *balanced* pu / pv / pw, level by
                            level; CheckMassBalance, then BalanceMassFluxes against lli(parent)
                            in the Richardson loop (proj/ct-unified/branches/meteo-server)
    grid_type_ll__speedup   GetRefinement, Match_cell / Match_u / Match_v, BalanceMassFluxes_m
                            (with_lapack), EnsurePoissonCache, SolvePoissonDirect
                            (proj/ct-unified/branches/speedup)

Region 1's balance is balance.py: its parent is the one-cell globe, so the Poisson solve is
one FFT over the whole grid.  A child's parent cell holds refine_i x refine_j child cells
(3 x 2 for nam1x1 under glb3x2), at most POISSON_DIRECT_MAX of them, so each is solved by
EnsurePoissonCache's direct matrices: the FFT solve applied to every unit vector once, then
a matrix-vector product per parent cell and level.  The correction is periodic over the
parent cell and gauged to zero on its west and south faces, hence on all four, so the parent
faces keep the sums the Match gave them.

The ingestion (`ChildMeteo`) is meteo.Meteo on the child's archive (regrid.RegriddedArchive
for the production zoom) with one difference: a child's surface pressure is matched
'area-aver' to its parent's matched one, not to the global mean.

Layout is pydyn's (layout.py, as balance.py): pu (im+1, jm, lm) = pu(0:im), pv (im, jm+1, lm),
pw (im, jm, lm+1) = pw(0:lm), sp (im, jm).  `Refinement` is GetRefinement's answer, read off
the dump header: refine_i = xref(child) / xref(parent), cg_i1..cg_i2 = ibeg..iend in parent
cells, and the same in j.
"""

from dataclasses import dataclass

import numpy as np

from . import balance, meteo
from .ops import convergence, seqsum

__all__ = [
    "POISSON_DIRECT_MAX",
    "ChildMeteo",
    "PoissonDirect",
    "Refinement",
    "balance_mass_fluxes_child",
    "match_area_aver",
    "match_cell_sum",
    "match_u_sum",
    "match_v_sum",
    "setup_mass_child",
]

POISSON_DIRECT_MAX = 24  # grid_type_ll__speedup.F90: direct matrices up to this many cells


@dataclass(frozen=True)
class Refinement:
    """GetRefinement of a child against its parent: cells per parent cell and the parent
    cells covered, 1-based inclusive, as the Fortran has them."""

    ri: int
    rj: int
    ci1: int
    ci2: int
    cj1: int
    cj2: int

    @classmethod
    def from_meta(cls, child, parent):
        return cls(child["xref"] // parent["xref"], child["yref"] // parent["yref"],
                   child["ibeg"], child["iend"], child["jbeg"], child["jend"])

    @classmethod
    def of(cls, zoom, r):
        """Child r's refinement in the zoom.Zoom tree (whose footprint replay checks against the
        dump header)."""
        g, p = zoom.g(r), zoom.g(zoom.g(r).parent)
        return cls(g.xref // p.xref, g.yref // p.yref, zoom.ibeg[r], zoom.iend[r], zoom.jbeg[r], zoom.jend[r])

    def cells(self):
        """(ci, cj, fi1, fj1): every covered parent cell in the Fortran loop order (cj outer)
        and the 0-based first child cell under it."""
        for cj in range(self.cj1, self.cj2 + 1):
            for ci in range(self.ci1, self.ci2 + 1):
                yield ci, cj, (ci - self.ci1) * self.ri, (cj - self.cj1) * self.rj


# ------------------------------------------------------------------ Match
def match_cell_sum(pg, tg, rf):
    """Match_cell 'sum': each parent cell's value spread equally over the child cells under
    it, the child sum (column-major, i fastest) brought to the parent's.  pg (imp, jmp, ...),
    tg (im, jm, ...) in place; trailing axes (levels) are independent."""
    for ci, cj, fi, fj in rf.cells():
        block = tg[fi : fi + rf.ri, fj : fj + rf.rj]
        fsum = seqsum(block.reshape(rf.ri * rf.rj, *block.shape[2:], order="F"), axis=0)
        block[...] = block + ((pg[ci - 1, cj - 1] - fsum) / (rf.rj * rf.ri))


def match_u_sum(pg, tg, rf):
    """Match_u 'sum': the child's x faces on every parent x face (the child's west and east
    edges included) brought to the parent face's flux.  pg (imp+1, jmp, ...), tg (im+1, jm, ...)."""
    for cj in range(rf.cj1, rf.cj2 + 1):
        fj = (cj - rf.cj1) * rf.rj
        for ci in range(rf.ci1 - 1, rf.ci2 + 1):
            fi = (ci - (rf.ci1 - 1)) * rf.ri
            face = tg[fi, fj : fj + rf.rj]
            fsum = seqsum(face, axis=0)
            face[...] = face + (pg[ci, cj - 1] - fsum) / rf.rj


def match_v_sum(pg, tg, rf):
    """Match_v 'sum': the same on the y faces.  pg (imp, jmp+1, ...), tg (im, jm+1, ...)."""
    for cj in range(rf.cj1 - 1, rf.cj2 + 1):
        fj = (cj - (rf.cj1 - 1)) * rf.rj
        for ci in range(rf.ci1, rf.ci2 + 1):
            fi = (ci - rf.ci1) * rf.ri
            face = tg[fi : fi + rf.ri, fj]
            fsum = seqsum(face, axis=0)
            face[...] = face + (pg[ci - 1, cj] - fsum) / rf.ri


def match_area_aver(pg, tg, rf, parea, tarea):
    """Match_cell 'area-aver': each block of child cells shifted by the constant that makes its
    row-area-weighted mean the parent cell's value.  pg (imp, jmp), tg (im, jm) in place;
    parea (jmp,), tarea (jm,) lli%area in rad^2."""
    for ci, cj, fi, fj in rf.cells():
        fsum = 0.0
        for j in range(fj, fj + rf.rj):
            fsum = fsum + seqsum(tg[fi : fi + rf.ri, j], axis=0) * tarea[j]
        fsum = fsum / parea[cj - 1]
        tg[fi : fi + rf.ri, fj : fj + rf.rj] += pg[ci - 1, cj - 1] - fsum


# ------------------------------------------------------------------ Poisson, per parent cell
class PoissonDirect:
    """EnsurePoissonCache's direct matrices for an (im, jm) cell block: column k is the
    correction (u, v) of the unit divergence at cell k (column-major), each through the
    FFT solve of `solver` (balance.FFTWPoisson for bitwise)."""

    def __init__(self, im, jm, solver=None):
        n = im * jm
        if n > POISSON_DIRECT_MAX:
            raise ValueError(f"PoissonDirect: {im} x {jm} cells exceeds POISSON_DIRECT_MAX")
        solver = balance.poisson_solver(im, jm) if solver is None else solver
        self.im, self.jm = im, jm
        self.m_to_u = np.empty(((im + 1) * jm, n))
        self.m_to_v = np.empty((im * (jm + 1), n))
        for k in range(n):
            unit = np.zeros(n)
            unit[k] = 1.0
            u, v = balance.poisson_correction(unit.reshape(im, jm, order="F"), solver)
            self.m_to_u[:, k] = u.reshape(-1, order="F")
            self.m_to_v[:, k] = v.reshape(-1, order="F")

    def solve(self, m):
        """SolvePoissonDirect on m (im, jm, ...): (u (im+1, jm, ...), v (im, jm+1, ...)), the
        matmul accumulated over the cells in index order."""
        tail = m.shape[2:]
        vec = m.reshape(self.im * self.jm, *tail, order="F")
        u = np.zeros((self.m_to_u.shape[0], *tail))
        v = np.zeros((self.m_to_v.shape[0], *tail))
        ex = (slice(None),) + (None,) * len(tail)
        for k in range(vec.shape[0]):
            u = u + self.m_to_u[:, k][ex] * vec[k]
            v = v + self.m_to_v[:, k][ex] * vec[k]
        return (u.reshape(self.im + 1, self.jm, *tail, order="F"),
                v.reshape(self.im, self.jm + 1, *tail, order="F"))


def balance_mass_fluxes_child(pu, pv, pw, dm_dt, rf, poisson):
    """BalanceMassFluxes_m (with_lapack) against a parent with refinement rf: the residual of
    every level first, then per parent cell (cj outer, ci inner) the direct Poisson correction
    added to the faces of its block.  pu, pv in place."""
    resid = convergence(pu, pv) + pw[..., :-1] - pw[..., 1:] - dm_dt
    for _ci, _cj, fi, fj in rf.cells():
        cu, cv = poisson.solve(resid[fi : fi + rf.ri, fj : fj + rf.rj])
        pu[fi : fi + rf.ri + 1, fj : fj + rf.rj] += cu
        pv[fi : fi + rf.ri, fj : fj + rf.rj + 1] += cv


def setup_mass_child(raw, parent, sp1, sp2, levi_db, area_m2, dt_sec, rf, poisson,
                     max_tries=balance.MB_MAX_TRIES):
    """The balancing part of Meteo_Setup_Mass for a child and one meteo instant: the raw
    fluxes (pu, pv, pw) matched to the parent's balanced `parent` (pu, pv, pw), checked, and
    balanced in the Richardson loop.  sp1, sp2 are the child's matched surface pressures.
    Returns (pu, pv, pw, info) with info as balance.setup_mass."""
    pu, pv, pw = (q.copy() for q in raw)
    ppu, ppv, ppw = parent
    match_u_sum(ppu, pu, rf)
    match_v_sum(ppv, pv, rf)
    match_cell_sum(ppw, pw, rf)
    dm_dt = balance.fill_mass_change(sp1, sp2, levi_db, area_m2, dt_sec)
    ok0, rms0, mav0 = balance.check_mass_balance(pu, pv, sp1, sp2, dt_sec, area_m2, *balance.MB_TOL_PRE)
    if not ok0:
        raise ValueError(f"setup_mass_child: unbalanced matched fluxes (rms {rms0:.3e}, max {mav0:.3e})")
    for tries in range(1, max_tries + 1):  # noqa: B007
        balance_mass_fluxes_child(pu, pv, pw, dm_dt, rf, poisson)
        ok, rms, mav = balance.check_mass_balance(pu, pv, sp1, sp2, dt_sec, area_m2, *balance.MB_TOL_TIGHT)
        if ok:
            break
    return pu, pv, pw, {"tries": tries, "ok": ok, "pre": (rms0, mav0), "post": (rms, mav)}


# ------------------------------------------------------------------ ingestion
class ChildMeteo(meteo.Meteo):
    """meteo.Meteo for a child: its surface pressure at an instant is the archive's, matched
    'area-aver' to the `parent` Meteo's (Meteo_Setup_Mass, n > 1).  `rf` its Refinement, `ll` its
    grid (grid.LL); its surface fields are the 1x1 ones FillGrid'ed onto it, as meteo.Meteo's."""

    def __init__(self, archive, parent, rf, ll, **kwargs):
        super().__init__(archive, ll, **kwargs)
        self.parent, self.rf = parent, rf

    def surface_pressure(self, t):
        if t not in self._sp:
            sp = self._interp3("sp", "sp", (t, t))
            match_area_aver(self.parent.surface_pressure(t), sp, self.rf, self.parent.area, self.area)
            self._sp[t] = sp
        return self._sp[t]
