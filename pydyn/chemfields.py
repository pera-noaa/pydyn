"""The loss-rate inputs of the `c` operator, read from the files the model reads them from
(proj/ct-unified/branches/chem_unified/src/read_OH_field.F90 read_CH4_loss_rate,
Import_OH_Spivakovsky, Read_trop_Cl_Hossaini; chemistry_ch4.F90 ch4_chemistry_init for the
soil map) for the production CH4 run: the arrays `chem.loss_rates` takes, for one month,
on the model grid.  The rc's paths, relative to the CH4 input root (`input/ch4c13`):

    strat_*   chemistry/CH4_loss_rate/Bruehl_ECHAM_Lrate_CH4+{OH,Cl,OD}.nc   (ch4loss.inputdir)
              Lrate (and, from the OH file, pressure) (month, level, lat) float32, s-1 / hPa,
              each (level, month) column linearly interpolated in latitude onto the model's
              cell centres (interpolate_lat)                              -> (jm, nlev)
    oh_3d     chemistry/OH/OH_troposphere-Spivakovsky-2000-ml137-glb100x100.nc   (oh.file)
              field (time, alt, lat, lon) float32 molec/cm^3 on 137 levels, and the month's
              geopotential boundary heights chemistry/OH/gph/ml137/gph_2000MM.nc, group
              glb100x100, gph (times, boundaries, lat, lon)            (oh.gph.inputdir):
              the layers are clumped onto the model's 34 by `echlevs`, each clump the
              thickness-weighted mean sum(f dz) / sum(dz), then FillGrid 'area-aver' from
              the 1x1 grid over the region's window, or 'copy' of the window for a 1x1
              region (a zoom's nam1x1)                                 -> (im, jm, lm)
    cl_trop   chemistry/Cl/fin2_glo304_t042_2008_cl.nc                  (trop.cl.filename)
    pl_trop   CL_mm, p_mm, te_mm, pl_mm (time, niv, lat, lon) float32; CL_mm to number
              density (1e-9 Cl)(p/T)(Avog/Rgas)1e-6, then latitude and longitude maps
              (mod_emission_regrid, the longitude cyclic), levels flipped to ground -> top
                                                                        -> (im, jm, ntrop)
    soil      soil/soil_sink_map_<region>.nc                          (ch4.soil_sink.input.dir)
              'qianlai': group year_YYYY, sink (sub_period, lats, lons), the month's
              record as is                                              -> (im, jm)

`month` is the calendar month of the middle of the chemistry step (the chem dump's
`month`), `year` its year (the soil map only).  Every file value is widened float32 ->
float64 on read (MDF), then the arithmetic is the Fortran's, in its order.  The files'
fill values are data to the Fortran (MDF does not mask): read unmasked.

Refused rather than guessed: an OH source other than Spivakovsky, a tropospheric Cl other
than Hossaini, a soil map other than 'qianlai', OH IAV, and ties in a coordinate that the
Fortran's argsort would have to break.
"""

from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from .diffusion import RGAS  # binas Rgas, J/mol/K
from .grid import GLB6X4, LL, fill
from .ops import seqsum

__all__ = [
    "AVOG",
    "ECHLEVS",
    "MAP_EPS",
    "PER_CM3",
    "PPB",
    "RGAS",
    "LossFields",
    "cl_number_density",
    "clump_levels",
    "interpolate_lat",
    "interpolation_map",
]  # fmt: skip

AVOG = 6.02205e23  # binas Avog, mlc/mol
# Read_trop_Cl_Hossaini writes these as literals: ppb -> mole fraction, molec/m^3 -> /cm^3.
PPB, PER_CM3 = 1.0e-9, 1.0e-6
MAP_EPS = 1.0e-10  # mod_emission_regrid: the weights fall back to (1, 0) below this span
# dims_levels (ml137_tropo34): the ECMWF half levels the 34 model levels are cut at, bottom up.
ECHLEVS = (
    137, 134, 131, 126, 122, 117, 113, 109, 105, 102, 98, 96, 94, 92, 90, 88, 85, 83, 81, 78,
    75, 73, 70, 67, 64, 61, 57, 54, 51, 44, 34, 26, 20, 14, 0,
)  # fmt: skip

BRUEHL = "chemistry/CH4_loss_rate/Bruehl_ECHAM_Lrate_CH4+{}.nc"
OH_FILE = "chemistry/OH/OH_troposphere-Spivakovsky-2000-ml137-glb100x100.nc"
GPH_FILE = "chemistry/OH/gph/ml137/gph_2000{month:02d}.nc"
GPH_GROUP = "glb100x100"
CL_FILE = "chemistry/Cl/fin2_glo304_t042_2008_cl.nc"
SOIL_FILE = "soil/soil_sink_map_{region}.nc"


def _read(ds, name):
    """A variable as the Fortran sees it: unmasked, unscaled, widened to float64, the axes
    reversed to Fortran order."""
    v = ds.variables[name]
    v.set_auto_maskandscale(False)
    return np.asarray(v[:], dtype=np.float64).T


def _ascending(x, what):
    if not (np.diff(x) > 0.0).all():
        raise NotImplementedError(f"{what}: not strictly ascending once sorted (argsort would break ties)")


def _sorted(x, what):
    order = np.argsort(x, kind="stable")
    _ascending(x[order], what)
    return order, x[order]


# ------------------------------------------------------------------ the regridding helpers
def interpolate_lat(lats_in, vals_in, lats_out):
    """read_OH_field interpolate_lat: `vals_in` (nlat_in, …) on `lats_in` linearly onto
    `lats_out`, clamped outside the input range.  Where the Fortran's loop finds two
    bracketing intervals (an output latitude on an input one) the later wins."""
    order, lat = _sorted(lats_in, "interpolate_lat")
    vals = vals_in[order]
    n = lat.size
    j = np.minimum(np.searchsorted(lat, lats_out, side="right") - 1, n - 2)
    j = np.maximum(j, 0)
    s1 = lat[j + 1] - lats_out
    s2 = lats_out - lat[j]
    shape = (-1,) + (1,) * (vals.ndim - 1)
    s1, s2 = s1.reshape(shape), s2.reshape(shape)
    out = (vals[j] * s1 + vals[j + 1] * s2) / (s1 + s2)
    below = (lats_out < lat[0]).reshape(shape)
    above = (lats_out > lat[-1]).reshape(shape)
    out = np.where(below, vals[:1], out)
    return np.where(above, vals[-1:], out)


def interpolation_map(coords_in, coords_out, cyclic=None):
    """mod_emission_regrid setup_interpolation_map_lat (cyclic None) or _lon (cyclic True /
    False, both coordinates taken modulo 360 into [0, 360)): (idx1, idx2, w1, w2), 0-based."""
    if cyclic is not None:
        coords_in = np.where(coords_in < 0.0, coords_in + 360.0, coords_in)
        coords_out = np.where(coords_out < 0.0, coords_out + 360.0, coords_out)
    order, c = _sorted(coords_in, "interpolation_map")
    n = c.size
    idx1 = np.empty(coords_out.size, dtype=int)
    idx2 = np.empty_like(idx1)
    s1 = np.empty(coords_out.size)
    s2 = np.empty_like(s1)
    for i, x in enumerate(coords_out):
        if cyclic is None and (x <= c[0] or x >= c[-1]):
            k = 0 if x <= c[0] else n - 1
            idx1[i] = idx2[i] = order[k]
            s1[i], s2[i] = np.nan, np.nan  # clamped: w1 = 1, w2 = 0
            continue
        if cyclic is not None and x < c[0]:
            idx1[i], idx2[i] = (order[-1], order[0]) if cyclic else (order[0], order[0])
            s1[i], s2[i] = (c[0] - x, (x - c[-1]) + 360.0) if cyclic else (1.0, 0.0)
            continue
        if cyclic is not None and x > c[-1]:
            idx1[i], idx2[i] = (order[-1], order[0]) if cyclic else (order[-1], order[-1])
            s1[i], s2[i] = ((c[0] - x) + 360.0, x - c[-1]) if cyclic else (1.0, 0.0)
            continue
        lo, hi = 0, n - 1  # the binary search: c[lo] <= x, and c[hi] > x or hi the last
        while hi - lo > 1:
            j = (lo + hi) // 2
            if c[j] <= x:
                lo = j
            else:
                hi = j
        idx1[i], idx2[i] = order[lo], order[hi]
        s1[i], s2[i] = c[hi] - x, x - c[lo]
    total = s1 + s2
    ok = total > MAP_EPS  # False for NaN (the clamped latitudes) as well
    w1 = np.where(ok, s1 / np.where(ok, total, 1.0), 1.0)
    w2 = np.where(ok, s2 / np.where(ok, total, 1.0), 0.0)
    return idx1, idx2, w1, w2


def apply_map(m, vals, axis):
    """apply_interpolation along `axis`: vals(idx1) w1 + vals(idx2) w2."""
    idx1, idx2, w1, w2 = m
    shape = [1] * vals.ndim
    shape[axis] = -1
    return np.take(vals, idx1, axis) * w1.reshape(shape) + np.take(vals, idx2, axis) * w2.reshape(shape)


def clump_levels(field, dz, echlevs=ECHLEVS):
    """Import_OH_Spivakovsky's vertical regrid: `field` (…, lme) on the layers of thickness
    `dz` (…, lme), bottom up, onto the len(echlevs)-1 model layers, each the mean over its
    clump of echlevs(l-1) - echlevs(l) source layers weighted by thickness, summed in order."""
    sizes = [a - b for a, b in zip(echlevs[:-1], echlevs[1:])]
    out = np.empty((*field.shape[:-1], len(sizes)))
    start = 0
    for ilev, n in enumerate(sizes):
        ks = slice(start, start + n)
        out[..., ilev] = seqsum(field[..., ks] * dz[..., ks], axis=-1) / seqsum(dz[..., ks], axis=-1)
        start += n
    return out


def cl_number_density(cl, p, t):
    """Read_trop_Cl_Hossaini: ppbv -> molec/cm^3, left to right as written."""
    return (((PPB * cl) * (p / t)) * (AVOG / RGAS)) * PER_CM3


# ------------------------------------------------------------------ the fields of a month
class LossFields:
    """The loss-rate inputs of the chemistry from the CH4 input tree under `root`, on the
    region `ll` (grid.LL, its lli) named `region` (the soil map's file):
    `month(month, year)` -> the kwargs of chem.loss_rates other than temper, phlb, lat and
    the scale factors, i.e. {strat_press, strat_oh, strat_cl, strat_od, oh_3d, pl_trop,
    cl_trop, soil}.  Each file (each month of the OH) is read once."""

    def __init__(self, root, ll=GLB6X4, region="glb6x4"):
        self.root, self.region, self.ll = Path(root), region, ll
        self.lon_deg, self.lat_deg = ll.lon_deg, ll.lat_deg  # lli(region)'s cell centres
        self._bruehl = None
        self._oh = {}
        self._cl = None
        self._soil = {}

    def path(self, template, **kw):
        return self.root / template.format(**kw)

    def bruehl(self):
        """{press, oh, cl, od}: (jm, nlev, 12) on the model latitudes."""
        if self._bruehl is None:
            out = {}
            for species, key in (("OH", "oh"), ("Cl", "cl"), ("OD", "od")):
                with Dataset(self.path(BRUEHL.format(species))) as ds:
                    assert ds.variables["Lrate"].units == "s-1"
                    lat = _read(ds, "lat")
                    out[key] = interpolate_lat(lat, _read(ds, "Lrate"), self.lat_deg)
                    if key == "oh":
                        assert ds.variables["pressure"].units == "hPa"
                        out["press"] = interpolate_lat(lat, _read(ds, "pressure"), self.lat_deg)
            self._bruehl = out
        return self._bruehl

    def oh(self, month):
        """Import_OH_Spivakovsky: the month's OH (im, jm, lm), molec/cm^3."""
        if month not in self._oh:
            with Dataset(self.path(GPH_FILE, month=month)) as ds:
                gph = _read(ds.groups[GPH_GROUP], "gph")[..., 0]  # (lon, lat, lme+1)
            dz = gph[..., 1:] - gph[..., :-1]
            with Dataset(self.path(OH_FILE)) as ds:
                v = ds.variables["field"]
                v.set_auto_maskandscale(False)
                assert v.getncattr("unit") == "molec/cm^3"
                if v.shape[0] != 12 or v.shape[1] != dz.shape[-1]:
                    raise ValueError(f"OH field {v.shape}: expected 12 months of {dz.shape[-1]} levels")
                field = np.asarray(v[month - 1], dtype=np.float64).T  # (lon, lat, lme)
            dlon, dlat = 360.0 / field.shape[0], 180.0 / field.shape[1]
            src = LL(-180.0 + 0.5 * dlon, -90.0 + 0.5 * dlat, dlon, dlat, field.shape[0], field.shape[1])
            self._oh[month] = fill(clump_levels(field, dz), self.ll, src)
        return self._oh[month]

    def hossaini(self):
        """Read_trop_Cl_Hossaini: (cl_trop, pl_trop), each (im, jm, ntrop, 12), ground -> top."""
        if self._cl is None:
            with Dataset(self.path(CL_FILE)) as ds:
                units = {name: ds.variables[name].units for name in ("CL_mm", "p_mm", "te_mm", "pl_mm")}
                assert units == {"CL_mm": "ppbv", "p_mm": "Pa", "te_mm": "K", "pl_mm": "Pa"}, units
                cl = cl_number_density(_read(ds, "CL_mm"), _read(ds, "p_mm"), _read(ds, "te_mm"))
                pl = _read(ds, "pl_mm")
                lat, lon = _read(ds, "lat"), _read(ds, "lon")
            lon = np.where(lon > 180.0, lon - 360.0, lon)
            lat_map = interpolation_map(lat, self.lat_deg)
            lon_map = interpolation_map(lon, self.lon_deg, cyclic=True)
            out = []
            for q in (cl, pl):  # (lon, lat, niv, time): latitude first, then longitude
                q = apply_map(lon_map, apply_map(lat_map, q, 1), 0)
                out.append(q[:, :, ::-1, :])
            self._cl = tuple(out)
        return self._cl

    def soil_map(self, year, month):
        """The 'qianlai' soil sink multiplier (im, jm) of the year and month."""
        if (year, month) not in self._soil:
            with Dataset(self.path(SOIL_FILE, region=self.region)) as ds:
                grp = ds.groups[f"year_{year:04d}"]
                assert grp.variables["sink"].units == "1"
                self._soil[year, month] = _read(grp, "sink")[:, :, month - 1]
        return self._soil[year, month]

    def month(self, month, year=None, trop_cl=True, soil=True):
        """The month's fields; the tropospheric Cl only with `trop_cl` (cl.trop.sink.apply),
        the soil map only with `soil` (ch4.soil.sink.apply), which needs the `year`."""
        b = self.bruehl()
        out = {f"strat_{k}": b[k][:, :, month - 1] for k in ("press", "oh", "cl", "od")}
        out["oh_3d"] = self.oh(month)
        if trop_cl:
            cl, pl = self.hossaini()
            out["cl_trop"], out["pl_trop"] = cl[..., month - 1], pl[..., month - 1]
        if soil:
            out["soil"] = self.soil_map(year, month)
        return out
