"""tmm's regridding of a tm5-nc record onto a region (`PYD-3` items 3, 4): what ReadField does
when the archive is not on the model grid -- the production zoom (`ch4_3x2nam1x1_pydyn.rc`)
reads every region's mass fluxes, surface pressure, convection, temperature and humidity from
the 1x1, 137-level ERA5 set and builds glb3x2 and nam1x1 at tropo34 from it in memory.

Transcribed from `tm5-zoom@745e94f70`:

    tmm.F90                 tmm_ReadField_2d/_3d: FillBuffer, Transform3Dh, Transform3Dv
    tmm_param.F90           GetCombineKeys: mfu, mfv 'sum' / 'sum'; mfw 'sum' / 'bottom';
                            sp 'area-aver'; eu du ed dd 'area-aver' / 'sum'; T, Q 'mass-aver' /
                            'mass-aver' (and Q clipped at 0, tmm_ReadField_3d)
    grid_type_ll__speedup   FillGrid 'copy' (a window at the source resolution) and 'combine'
                            (diX x djX source cells per model cell): 'u' faces summed over the
                            source rows of a face, 'v' over its columns, 'n' cells over both
                            (rows outer), 'area-aver' weighted by the source rows' area_m2,
                            'weight' (Transform3Dh's 'mass-aver') by each source cell's
                            (da + db ps) area_m2, ps the record's own; AreaOper
    grid_type_hyb.F90       levi_Init_select (flevs, hlevs of a selection), levi_Compare
                            (combine, reversed: the file is bottom-up, the parent table
                            top-down), levi_FillLevels_3d 'combine': 'sum' accumulates the
                            parent full levels le1..le2 of each layer, 'mass-aver' each weighted
                            by |da + db ps| / |da(l) + db(l) ps| on the model grid's ps, 'w'
                            copies half levels; levi_Init (da = a(l) - a(l-1))
    dims_levels__ml137_tropo34.F90   echlevs (chemfields.ECHLEVS)

The record itself is read as meteo.Archive reads one (float32 widened, unit factor 1 for
kg/s and Pa).  `RegriddedArchive` presents the result through meteo.Archive's interface,
so meteo.Meteo and zoom_meteo.ChildMeteo run on it unchanged.
"""

import numpy as np

from . import meteo
from .chemfields import ECHLEVS
from .grid import ONE_BY_ONE, cut, fill, relate

__all__ = ["ML137", "CachedArchive", "RegriddedArchive", "fill_grid", "fill_levels"]

# meteo-tm5-noaa.rc tmm.sourcekey.*.ml of the production zoom
ML137 = "ec/ea/h06h18tr3/ml137/glb100x100/{t:%Y}/{t:%m}/{p}_{t:%Y%m%d}_00p03.nc"
# GetCombineKeys (hcomb, vcomb) and the staggering ReadField asks for, per variable
COMBINE = {"mfu": ("sum", "sum", "u"), "mfv": ("sum", "sum", "v"), "mfw": ("sum", "bottom", "n"),
           "sp": ("area-aver", None, "n"), "T": ("mass-aver", "mass-aver", "n"),
           "Q": ("mass-aver", "mass-aver", "n"),
           **dict.fromkeys(("eu", "du", "ed", "dd"), ("area-aver", "sum", "n"))}  # fmt: skip
PARAMS = ("mfuv", "mfw", "sp", "t", "q", "convec")  # the parameter files regridded here


def fill_grid(llX, nuv, combkey, dst, weights=None, src=ONE_BY_ONE):
    """FillGrid of the source field llX (lon, lat, ...) on `src` onto the region `dst` (grid.LL):
    'n' cells through grid.fill ('copy', or 'combine' with `combkey` 'sum', 'area-aver' or
    'weight'), 'u' faces summed over the source rows of a face, 'v' over its columns, each sum
    from 0.0 in the Fortran's order."""
    rel = relate(dst, src, "regrid")
    if nuv == "n":
        return fill(llX, dst, src, combkey, weights, rel)
    sub = cut(llX, rel, dst, nuv)
    if rel.action == "copy":
        return sub
    di, dj = rel.nx, rel.ny
    res = 0.0
    if nuv == "u":
        for jx in range(dj):
            res = res + sub[::di, jx::dj]
        return res
    for ix in range(di):
        res = res + sub[ix::di, ::dj]
    return res


def fill_levels(llX, vcomb, echlevs=ECHLEVS, ps=None, levX=None, lev=None):
    """levi_FillLevels_3d 'combine', reversed, of a bottom-up source llX (..., nlevX[+1]) onto
    the selection `echlevs` (parent half levels, top-down, echlevs[0] the surface): 'sum' a
    layer's parent levels in ascending parent order, i.e. the file's levels top to bottom;
    'mass-aver' the same with each weighted by its pressure thickness over the layer's, on the
    model grid's `ps` (levX, lev: the (a, b) half levels of the file and of the model);
    'bottom' ('w') each model half level the file's."""
    nparent = echlevs[0]
    nlev = len(echlevs) - 1
    if vcomb == "bottom":
        return np.stack([llX[..., nparent - h] for h in echlevs], axis=-1)
    if vcomb == "mass-aver":
        (ax, bx), (am, bm) = levX, lev
        dax, dbx, dam, dbm = ax[1:] - ax[:-1], bx[1:] - bx[:-1], am[1:] - am[:-1], bm[1:] - bm[:-1]
    out = []
    for lv in range(1, nlev + 1):
        acc = 0.0
        for le in range(echlevs[lv] + 1, echlevs[lv - 1] + 1):
            k = nparent - le  # k = nlev_parent + 1 - le, 1-based
            if vcomb == "sum":
                acc = acc + llX[..., k]
            else:
                acc = acc + llX[..., k] * abs(dax[k] + dbx[k] * ps) / abs(dam[lv - 1] + dbm[lv - 1] * ps)
        out.append(acc)
    return np.stack(out, axis=-1)


class CachedArchive(meteo.Archive):
    """meteo.Archive keeping its last few records: every region regrids the same ones."""

    def __init__(self, root, sources=meteo.SOURCES, keep=8):
        super().__init__(root, sources)
        self.keep, self._records = keep, {}

    def read(self, param, t1, t2):
        """The record's variables and, for a 3-D field, the record's own `ps`."""
        key = (param, t1, t2)
        if key not in self._records:
            if len(self._records) >= self.keep:
                self._records.pop(next(iter(self._records)))
            out = super().read(param, t1, t2)
            nc, irec = self.record(param, t1, t2)
            if "ps" in nc.variables and param != "sp":
                out["ps"] = self._widen(nc["ps"][irec], param)
            self._records[key] = out
        return self._records[key]


class RegriddedArchive:
    """meteo.Archive's interface over the 1x1, 137-level set under `root`, regridded onto the
    region `ll` (grid.LL).  The records are cached in the `source` Archive, which every region
    may share."""

    def __init__(self, root, ll, source=None):
        self.source = source or CachedArchive(root, dict(meteo.SOURCES, **dict.fromkeys(PARAMS, ML137)))
        self.ll = ll
        self._cache = {}

    def _horizontal(self, nuv, hcomb, field, weights=None):
        return fill_grid(field, nuv, hcomb, self.ll, weights)

    def read(self, param, t1, t2):
        """A model-level parameter regridded onto the region; any other (the 1x1 surface
        fields, which meteo.Meteo puts on its grid itself) as the file has it."""
        if param not in PARAMS:
            return self.source.read(param, t1, t2)
        key = (param, t1, t2)
        if key not in self._cache:
            raw = self.source.read(param, t1, t2)
            out = {}
            for var, field in raw.items():
                if var == "ps":
                    continue
                hcomb, vcomb, nuv = COMBINE[var]
                if hcomb == "mass-aver":
                    levX = self.source.hybrid(param, t1)  # the file's, bottom-up
                    dp = (levX[0][1:] - levX[0][:-1]) + (levX[1][1:] - levX[1][:-1]) * raw["ps"][..., None]
                    dp = dp * ONE_BY_ONE.area_m2[None, :, None]  # AreaOper '*' 'm2'
                    field = self._horizontal(nuv, "weight", field, dp)
                    ps = self._horizontal("n", "area-aver", raw["ps"])
                    field = fill_levels(field, vcomb, ps=ps, levX=levX, lev=self.hybrid(param, t1))
                    out[var] = np.maximum(0.0, field) if var == "Q" else field
                    continue
                field = self._horizontal(nuv, hcomb, field)
                out[var] = field if vcomb is None else fill_levels(field, vcomb)
            if len(self._cache) >= 12:  # a read interval needs two records of three parameters
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = out
        return self._cache[key]

    def read_instant(self, param, t):
        return self.read(param, t, t)

    def read_const(self, param):
        return self.source.read_const(param)

    def hybrid(self, param, t):
        """(a, b) of the selection: levi_Init_select's parent a(hlevs), the parent table the
        file's half levels reversed."""
        a, b = self.source.hybrid(param, t)  # bottom-up, 0..137
        idx = [ECHLEVS[0] - h for h in ECHLEVS]
        return a[idx], b[idx]
