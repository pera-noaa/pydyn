"""One-member scaling file for the re-pin sets: mean 1.0, deviation 0, so the CH4 tracer's
emission is the prior exactly, as in pydyn_dump_phys.  Usage: make_scaling_one.py <output dir>"""
import sys
from pathlib import Path

from netCDF4 import Dataset

NPARMS, NMEMBERS, NWEEKS = 2700, 1, 12
out = Path(sys.argv[1]) / "scaling.factor.60x45.2000-01-01.nc"
with Dataset(out, "w") as nc:
    nc.sf_convention = "center-1"
    nc.createDimension("one", 1)
    nc.createDimension("nweeks", NWEEKS)
    nc.createDimension("nmembers", NMEMBERS)
    nc.createDimension("parameters", NPARMS)
    nc.createVariable("scaling_factor", "f4", ("one", "nweeks", "parameters"))[:] = 1.0
    nc.createVariable("scaling_factor_deviations", "f4", ("one", "nweeks", "nmembers", "parameters"))[:] = 0.0
print(out)
