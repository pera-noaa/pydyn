!
! Macro definitions for the pydyn kernel harness.
!
! Pycasso rewrites tm5.inc with every macro of the build and leaves no
! advect.h in the build directory.  Here the base/trunk/src/tm5.inc that
! base sources pick up (the directory of the including file is searched
! first) does `#include "advect.h"`, which resolves to this file through -I.;
! the harness's own tm5.inc does the same.  So all defines live here.
!
! Production defines of proj/ct-unified/rc/__zoom_1day.rc that matter to the
! dynamic core: my.advect.define = slopes, my.tm5.define has
! without_wet_deposition.  MPI, with_budgets, with_zoom, secmom,
! with_tendencies, without_diffusion are intentionally undefined.
! with_lapack is a Makefile choice (-Dwith_lapack); with_zoom is defined
! on the command line for harness_zoom only, as pycasso defines it for any
! build with more than one region.
!
#define slopes
#define without_wet_deposition
#define without_photolysis
#define without_dry_deposition
