"""The initial tracer field of a forward run, `istart 66` (sources_sinks.F90
start_from_field_fill3d, flat-file branch): one 3-D mixing-ratio field in ppm on a regular
lon/lat grid and hybrid levels of its own, remapped to the model grid and turned into
tracer mass for every tracer alike.

    file       istart.66.filename, variable istart.66.var_name (rm), (lev, lat, lon), plus
               lon, lat (cell centres) and at, bt (lev+1 half levels) -- the file's grid is
               Init(lon(1), lon(2)-lon(1), nlon, lat(1), lat(2)-lat(1), nlat)
    horizontal FillGrid 'area-aver' level by level onto the region (Fill3D, each region of a
               zoom on its own), Relate's action by the spacings: the file finer, each model
               cell the mean of the file's cells under it weighted by their rows' area_m2,
               rows outer, west to east ('combine'); the same spacing, a copy ('copy'); the
               file coarser, each file cell's value in every model cell under it
               ('distribute').  The file must cover the region with its cell edges on the
               region's (a part-covered region is refused)
    vertical   FillLevels 'mass-aver': the file's levels are the model's (|da|, |db| < 0.1
               level for level) -> a plain copy, or a copy upside down when they match
               reversed; anything else (interpolation) is refused here
    mass       rm = ((f * 1e-6) * m) / fscale,  fscale = xmair / ra = 28.94 / xm(tracer)
               with m the run's initial air mass; rxm = rym = rzm = 0

The field is checked the way the model checks it: the mean of the remapped field must lie in
[1e-3, 1e6] ppm (a kg/kg file is refused).  The grouped (4DVAR save) format is not
transcribed.
"""

import numpy as np
from netCDF4 import Dataset

from .diffusion import XMAIR  # binas xmair, g/mol: chem_param fscale = xmair / ra
from .grid import GLB6X4, LL, fill, relate

__all__ = ["IC_MEAN_RANGE", "PPM", "XMAIR", "fscale", "initial_state", "read_field", "remap"]

PPM = 1.0e-6  # ppm -> mole fraction, a literal in start_from_field_fill3d
IC_MEAN_RANGE = (1.0e-3, 1.0e6)  # the TODO #9d guard on the remapped field's mean, ppm



def fscale(xm):
    return XMAIR / xm


def read_field(path, var_name="rm"):
    """(field (nlon, nlat, nlev), lon, lat, at, bt) of an istart-66 file, as the Fortran
    reads them: float64, unmasked, Fortran axis order."""
    with Dataset(path) as ds:
        if any(name.startswith("glb") for name in ds.groups):
            raise NotImplementedError("istart 66: the grouped (4DVAR save) format is not transcribed")
        ds.set_auto_maskandscale(False)
        v = ds.variables[var_name]
        if getattr(v, "units", "ppm") != "ppm":
            raise ValueError(f"istart 66 expects ppm, the file's {var_name} is in {v.units}")
        field = np.asarray(v[:], dtype=np.float64).T
        lon_name = "lon" if "lon" in ds.variables else "longitude"
        lat_name = "lat" if "lat" in ds.variables else "latitude"
        coords = [np.asarray(ds.variables[n][:], dtype=np.float64) for n in (lon_name, lat_name, "at", "bt")]
    return (field, *coords)


def remap(field, lon, lat, at, bt, a, b, ll=GLB6X4):
    """Fill3D 'mass-aver' of the file's `field` onto the region `ll` (grid.LL) and its levels
    (a, b), the half levels bottom up: (im, jm, lm) in ppm."""
    nlonX, nlatX, nlevX = field.shape
    src = LL(lon[0], lat[0], lon[1] - lon[0], lat[1] - lat[0], nlonX, nlatX)  # Init(lon(1), dlon, nlon, ...)
    horizontal = fill(field, ll, src, "area-aver", rel=relate(ll, src, "istart 66"))

    lm = a.size - 1
    if nlevX == lm and (np.abs(at - a) < 0.1).all() and (np.abs(bt - b) < 0.1).all():
        return horizontal
    if nlevX == lm and (np.abs(at[::-1] - a) < 0.1).all() and (np.abs(bt[::-1] - b) < 0.1).all():
        return horizontal[..., ::-1]
    raise NotImplementedError("istart 66: the file's levels are not the model's; no interpolation")


def initial_state(path, m, a, b, xm, ntracer=1, var_name="rm", ll=GLB6X4):
    """The initial rm, rxm, rym, rzm (im, jm, lm, ntracer) of every tracer from the
    istart-66 file at `path`, with the initial air mass `m` (im, jm, lm), the model's half
    levels (a, b) and the tracers' molar mass `xm` (g/mol)."""
    field = remap(*read_field(path, var_name), a, b, ll)
    mean = field.sum() / field.size
    if not IC_MEAN_RANGE[0] <= mean <= IC_MEAN_RANGE[1]:
        raise ValueError(f"istart 66: IC field mean {mean:.4e} outside the physical [1e-3, 1e6] ppm")
    rm1 = ((field * PPM) * m) / fscale(xm)
    rm = np.repeat(rm1[..., None], ntracer, axis=-1)
    return {"rm": rm, "rxm": np.zeros_like(rm), "rym": np.zeros_like(rm), "rzm": np.zeros_like(rm)}
