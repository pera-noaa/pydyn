"""pydyn -- serial NumPy reference of the TM5-zoom dynamic core (slopes).

Transcribed from base/trunk/src {advect.F90, advectx/y/z__slopes.F90, convection.F90}
as compiled by the production overlay chain (slopes, without_wet_deposition), for one
region; `zoom` adds the nesting -- the region tree, the parent<->child exchange, a child's
advection, Proces_Region and Check_CFL on a tree -- in its own halo-bounded layout.
The production steps around the kernels live in their own modules: `redgrid` (reduced
grid), `cfl` (Check_CFL), `balance` (Meteo_Setup_Mass, m2phlb1), `lapack` (the with_lapack
inverse), `emis` and `chem` (the `s` and `c` operators of the CH4 run: do_add_2d with the
month's field from the 1x1 emission files, and ch4_chemistry_step), `chemfields` (its
loss-rate climatologies and soil map from their files), `initial` (the istart-66 initial
field), `meteo` (the ingestion of the tm5-nc archive), `diffusion` (Calc_Kzz on the
ingested surface fields), `dump` + `replay` (replay of an instrumented model run).
`simple` and `simple_diffusion` are the core and Calc_Kzz written for reading, within 1e-12
of the bitwise transcriptions (tests/test_simple.py says relative to what).  The algorithm is
tm5-zoom@4dc2017c5:docs/dynamic_core_survey.md, in the model's repository; docs/state.md has
the layout conventions and the Fortran-parity harness.
"""

from .driver import Model, State
from .kernels import advect_x, advect_y, advect_z, convec, convection_matrix, fluxes_from_meteo

__all__ = [
    "Model",
    "State",
    "advect_x",
    "advect_y",
    "advect_z",
    "convec",
    "convection_matrix",
    "fluxes_from_meteo",
]
