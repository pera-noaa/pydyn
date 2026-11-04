/* The device module: every kernel of pydyn.gpu.  Compiled by NVRTC with --fmad=false for the
   GPU, and by the host compiler with -ffp-contract=off (plus host_twin.inc) for the emulation
   backend and the CPU reference's twin libraries. */
#include "pd.h"
#include "advect.cuh"
#include "convec.cuh"
#include "mass.cuh"
#include "chem.cuh"
#include "redgrid.cuh"
#include "kzz.cuh"
#include "child.cuh"
#include "boxes.cuh"
