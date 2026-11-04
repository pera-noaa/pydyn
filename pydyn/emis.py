"""The `s` operator: sources_sinks_apply -> emission_ct_apply -> do_add_2d
(proj/ct-unified/trunk/src/emission_data.F90) for the production CH4 run, where every
emission category is injected at one level with no partial fraction and no 3-D profile,
and the emission field it injects, read from the CarbonTracker-CH4 1x1 monthly files.

One do_add_2d call adds to one tracer, at one level, the month's flux field scaled to the
emission time step:

    x = emis(i,j) * month2dt * (mol_mass / mol_mass_e) * efrac        (left to right)
    rm(i,j,l) += x ;  rzm(i,j,l) -= x            (slopes: the injection is at the cell floor)

with month2dt = (nsrce / (2 tref)) / sec_month.  The factors other than month2dt are the
run's tracer table (which tracer, which level, the two molar masses) and are taken from the
`emis` dump event.

The field `emis` (kg CH4 per model cell per month) is built once per month
(emission_ch4_calc on newmonth / newsrun, emission_ct_calc, a forward run: the EnKF scaling
is the constant 1.0 and the one member's sum is the field itself, both exact):

    file       <dir>/flux1x1_all_source.YYYYMM.nc, `all_source_flux` (lon, lat) float64 in
               umol m-2 s-1, lon -179.5 .. 179.5, lat -89.5 .. 89.5 -- the model's own
               orientation, no roll                                       (file_src_read)
    units      f = raw * (xm / 1e9)                    kg CH4 m-2 s-1     (EU_MICROMOL)
    per cell   g(i,j) = f(i,j) * dxy11(j) * sec_month  kg per 1x1 cell per month
                                                              (flux_to_gridbox_per_month)
    coarsen    the 6x4 block sums of g, row by row, west to east    (FillGrid 'sum')

A zoom builds each region's field from g on its own (emission_ct coarsen_cat, toolbox.F90
coarsen_emission_2d: FillGrid onto lli(region)): the block sums over the region's window of
the 1x1 grid ('combine', glb3x2 3x2), or the window itself at 1x1 ('copy', nam1x1) -- the whole
region, the edge ring included; do_add_2d injects on the cells the region owns only.

dxy11 is calc_dxy's 1x1 cell area, its latitude accumulated cell by cell from -90 deg
(`meteo.dxyp`); with lat = j * dlat instead the field is off by thousands of ulp.
"""

from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from . import meteo
from .grid import GLB6X4, ONE_BY_ONE, fill

__all__ = ["XMCH4", "Emissions", "do_add_2d", "inject", "month2dt", "monthly_field"]

XMCH4 = 12.01115 + 1.0079 * 4.0  # chem_param: xmc + xmh * 4, g/mol
FILE = "flux1x1_all_source.{t:%Y%m}.nc"  # ct.emissions.rc: ch4.fname.prefix + YYYYMM
VARNAME = "all_source_flux"  # ct.emissions.rc: ch4.varname


def inject(emis, month2dt, mol_mass, mol_mass_e, efrac=1.0):
    """The mass added to each cell by one do_add_2d call."""
    return ((emis * month2dt) * (mol_mass / mol_mass_e)) * efrac


def do_add_2d(state, x, level, tracer):
    """Add the injection `x` (im, jm) to tracer `tracer` at level `level` (both 1-based, as
    in the Fortran) of the pydyn-layout state, in place."""
    at = (..., level - 1, tracer - 1)
    state["rm"][at] += x
    state["rzm"][at] -= x


def month2dt(nsrce, sec_month, tref=1):
    """do_add_2d's month -> emission-step factor: dtime / sec_month with dtime the half
    source step nsrce / (2 tref) of the region."""
    return (float(nsrce) / (2 * tref)) / sec_month


def monthly_field(raw, sec_month, xm=XMCH4, ll=GLB6X4):
    """The month's emission (im, jm) in kg per model cell of the region `ll` (grid.LL) from the
    1x1 flux `raw` (360, 180) in umol m-2 s-1: the unit factor, the 1x1 cell area and the
    month's seconds, then FillGrid 'sum' onto the region (its window copied at 1x1)."""
    dxy11 = meteo.dxyp(raw.shape[1], 1, 1, -90.0)
    f = raw * (xm / 1.0e9)
    g = (f * dxy11[None, :]) * sec_month
    return fill(g, ll, ONE_BY_ONE, "sum")


class Emissions:
    """The CH4 emission of each month from the 1x1 files under `root` (the rc's
    emis.ch4.input.dir), on the region `ll` (grid.LL)."""

    def __init__(self, root, xm=XMCH4, ll=GLB6X4, file=FILE, varname=VARNAME):
        self.root, self.xm, self.ll = Path(root), xm, ll
        self.file, self.varname = file, varname
        self._fields = {}  # month start -> (field, sec_month)

    def path(self, t):
        return self.root / self.file.format(t=t)

    def read(self, t):
        """The file's flux of the month of `t` as (lon, lat), float64."""
        with Dataset(self.path(t)) as ds:
            v = ds.variables[self.varname]
            assert v.units == "umol m-2 s-1", v.units
            lon, lat = ds.variables["lon"][:], ds.variables["lat"][:]
            assert lon[0] == -179.5 and lat[0] == -89.5, (lon[0], lat[0])  # the model's orientation
            return np.asarray(v[0], dtype=np.float64).T

    def month(self, t):
        """(field, sec_month) of the month of `t`, read once."""
        t1, t2 = meteo.month(t)
        if t1 not in self._fields:
            sec_month = (t2 - t1).total_seconds()
            field = monthly_field(self.read(t1), sec_month, self.xm, self.ll)
            self._fields[t1] = field, sec_month
        return self._fields[t1]

    def field(self, t):
        return self.month(t)[0]

    def injection(self, t, nsrce, mol_mass, mol_mass_e, efrac=1.0, tref=1):
        """The mass (im, jm) one do_add_2d call adds at `t` with the source step `nsrce`."""
        field, sec_month = self.month(t)
        return inject(field, month2dt(nsrce, sec_month, tref), mol_mass, mol_mass_e, efrac)
