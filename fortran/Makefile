# fortran/Makefile -- standalone build of the production dynamic-core kernels
# against stub modules (docs/state.md section 3a).  The kernel sources stay in
# tm5-zoom and are compiled from there, unmodified; see $(TM5ZOOM) below.
#
#   make                 # gfortran, strict FP (-O0 -ffp-contract=off), Crout fastminv
#   make WITH_LAPACK=1   # fastminv via DGETRF/DGETRI (Accelerate on macOS, -llapack elsewhere)
#   make FC=ifort        # ifort: -O0 -fp-model strict -r8
#   make OPT="-O3"       # measure what optimisation does to the bitwise comparison
#   make clean
#   make TM5ZOOM=/path/to/tm5-zoom   # kernels from another checkout (default ../../tm5-zoom)
#   make zoom            # harness_zoom2 and harness_zoom3: two and three regions, the real
#                        # zoom_tools, advect_tools__speedup, modelIntegration and advectm_cfl
#                        # (docs/state.md section 3d); objects in build-zoom2/, build-zoom3/
#
# The kernel sources are the ones the production build resolves through the
# overlay chain of proj/ct-unified (later wins, X__suffix.F90 competes as X.F90);
# they are compiled unmodified.  Convection is the exception: production
# compiles branches/speedup/src/convection__speedup.F90, which needs rcfile,
# shared-memory and MPI infrastructure; base/trunk/src/convection.F90 is the
# reference arithmetic it was derived from and is what is built here.

# TM5ZOOM is the tm5-zoom checkout the kernels are read from; nothing is written
# there.  Default: the sibling repo under ash-code.  Override on the command line
# or in the environment to build against another checkout or sha.
TM5ZOOM ?= ../../tm5-zoom

ROOT   := $(TM5ZOOM)
TRUNK  := $(ROOT)/proj/ct-unified/trunk/src
SPEEDUP:= $(ROOT)/proj/ct-unified/branches/speedup/src
BASE   := $(ROOT)/base/trunk/src

SRC_X    := $(TRUNK)/advectx__slopes.F90
SRC_Y    := $(TRUNK)/advecty__slopes.F90
SRC_Z    := $(SPEEDUP)/advectz__slopes.F90
SRC_ADV  := $(BASE)/advect.F90
SRC_CONV := $(BASE)/convection.F90
SRC_ZT   := $(TRUNK)/zoom_tools.F90
SRC_AT   := $(SPEEDUP)/advect_tools__speedup.F90
SRC_MI   := $(TRUNK)/modelIntegration.F90
SRC_DT   := $(TRUNK)/datetime.F90
SRC_CC   := $(TRUNK)/cfl_cache.F90
SRC_CFL  := $(TRUNK)/advectm_cfl.F90

FC   = gfortran
OPT  = -O0

ifeq ($(findstring ifort,$(FC))$(findstring ifx,$(FC)),)
  # gfortran
  FFLAGS = $(OPT) -g -ffp-contract=off -fdefault-real-8 -fdefault-double-8 -cpp -I. -ffree-line-length-none
else
  FFLAGS = $(OPT) -g -fp-model strict -r8 -fpp -I.
endif

DEFS =
LIBS =
ifdef WITH_LAPACK
  DEFS += -Dwith_lapack
  ifeq ($(shell uname),Darwin)
    LIBS += -framework Accelerate
  else
    LIBS += -llapack -lblas
  endif
endif

# Objects and .mod files of each build live in their own directory: gfortran looks for
# modules in the working directory before any -I, so none may be left in fortran/ itself.
BIN  := harness
BDIR := build
OBJS := $(addprefix $(BDIR)/,stubs.o advectx.o advecty.o advectz.o advect.o convection.o harness.o)
BFLAGS = -I$(BDIR) -J$(BDIR) $(FFLAGS)
ifneq ($(findstring ifort,$(FC))$(findstring ifx,$(FC)),)
  BFLAGS = -I$(BDIR) -module $(BDIR) $(FFLAGS)
endif

$(BIN): $(OBJS)
	$(FC) $(BFLAGS) -o $@ $(OBJS) $(LIBS)

$(BDIR):
	mkdir -p $@
$(BDIR)/stubs.o: stubs.F90 tm5.inc advect.h | $(BDIR)
	$(FC) $(BFLAGS) $(DEFS) -c -o $@ $<
$(BDIR)/advectx.o: $(SRC_X) $(BDIR)/stubs.o tm5.inc advect.h
	$(FC) $(BFLAGS) $(DEFS) -c -o $@ $<
$(BDIR)/advecty.o: $(SRC_Y) $(BDIR)/stubs.o tm5.inc advect.h
	$(FC) $(BFLAGS) $(DEFS) -c -o $@ $<
$(BDIR)/advectz.o: $(SRC_Z) $(BDIR)/stubs.o tm5.inc advect.h
	$(FC) $(BFLAGS) $(DEFS) -c -o $@ $<
$(BDIR)/advect.o: $(SRC_ADV) $(BDIR)/stubs.o tm5.inc advect.h
	$(FC) $(BFLAGS) $(DEFS) -c -o $@ $<
$(BDIR)/convection.o: $(SRC_CONV) $(BDIR)/stubs.o tm5.inc advect.h
	$(FC) $(BFLAGS) $(DEFS) -c -o $@ $<
$(BDIR)/harness.o: harness.F90 $(BDIR)/advectx.o $(BDIR)/advecty.o $(BDIR)/advectz.o $(BDIR)/advect.o $(BDIR)/convection.o
	$(FC) $(BFLAGS) $(DEFS) -c -o $@ $<

# --- harness_zoom<N>: the same kernels plus the zoom exchange, NREGIONS = N -------------
# One binary per region count: production's nregions is the true number of regions, and
# Check_CFL, Setup_MassFlow and store_masses loop over all of them -- a two-region case in
# a three-region binary would touch an unallocated region 3.  Each build has its own object
# directory, so no two builds' .mod files ever mix.  with_zoom is what
# pycasso defines whenever more than one region is configured (pycasso-tm5-expert*.rc);
# convec's `zoomed(i,j) /= region` skip is compiled only under it.  The stubs keep the
# reduced grid off (redgridZoom.F90 needs the rc machinery); every other zoom routine on
# the transport path is the production source.
NZ    ?= 3
ZDIR  := build-zoom$(NZ)
ZBIN  := harness_zoom$(NZ)
ZOBJS := $(addprefix $(ZDIR)/,stubs.o datetime.o advect_tools.o zoom_tools.o advectx.o advecty.o \
                              advectz.o advect.o convection.o modelIntegration.o cfl_cache.o \
                              advectm_cfl.o harness_zoom.o)
# modelIntegration.F90 (Proces_Region, do_steps) is built as the transport-only dump sets
# were (runs/MANIFEST.md): chemistry, emissions and the diffusion setup compiled out, so
# the `c` and `s` cases of do_steps are empty and Proces_Init/Update name only stubs.
MI_DEFS := -Dwithout_chemistry -Dwithout_emission -Dwithout_diffusion
ZFLAGS = -I$(ZDIR) -J$(ZDIR) $(FFLAGS) -DNREGIONS=$(NZ) -Dwith_zoom
ifneq ($(findstring ifort,$(FC))$(findstring ifx,$(FC)),)
  ZFLAGS = -I$(ZDIR) -module $(ZDIR) $(FFLAGS) -DNREGIONS=$(NZ) -Dwith_zoom
endif

zoom:
	$(MAKE) zoom-one NZ=2
	$(MAKE) zoom-one NZ=3
zoom-one: $(ZBIN)
$(ZBIN): $(ZOBJS)
	$(FC) $(ZFLAGS) -o $@ $(ZOBJS) $(LIBS)

$(ZDIR):
	mkdir -p $@
$(ZDIR)/stubs.o: stubs.F90 tm5.inc advect.h | $(ZDIR)
	$(FC) $(ZFLAGS) $(DEFS) -c -o $@ $<
$(ZDIR)/datetime.o: $(SRC_DT) $(ZDIR)/stubs.o
	$(FC) $(ZFLAGS) $(DEFS) -c -o $@ $<
$(ZDIR)/cfl_cache.o: $(SRC_CC) $(ZDIR)/stubs.o
	$(FC) $(ZFLAGS) $(DEFS) -c -o $@ $<
$(ZDIR)/advectm_cfl.o: $(SRC_CFL) $(ZDIR)/cfl_cache.o $(ZDIR)/datetime.o $(ZDIR)/zoom_tools.o \
                       $(ZDIR)/advect.o
	$(FC) $(ZFLAGS) $(DEFS) -c -o $@ $<
$(ZDIR)/advect_tools.o: $(SRC_AT) $(ZDIR)/stubs.o
	$(FC) $(ZFLAGS) $(DEFS) -c -o $@ $<
$(ZDIR)/zoom_tools.o: $(SRC_ZT) $(ZDIR)/advect_tools.o
	$(FC) $(ZFLAGS) $(DEFS) -c -o $@ $<
$(ZDIR)/advectx.o: $(SRC_X) $(ZDIR)/zoom_tools.o
	$(FC) $(ZFLAGS) $(DEFS) -c -o $@ $<
$(ZDIR)/advecty.o: $(SRC_Y) $(ZDIR)/zoom_tools.o
	$(FC) $(ZFLAGS) $(DEFS) -c -o $@ $<
$(ZDIR)/advectz.o: $(SRC_Z) $(ZDIR)/zoom_tools.o
	$(FC) $(ZFLAGS) $(DEFS) -c -o $@ $<
$(ZDIR)/advect.o: $(SRC_ADV) $(ZDIR)/stubs.o
	$(FC) $(ZFLAGS) $(DEFS) -c -o $@ $<
$(ZDIR)/convection.o: $(SRC_CONV) $(ZDIR)/datetime.o
	$(FC) $(ZFLAGS) $(DEFS) -c -o $@ $<
$(ZDIR)/modelIntegration.o: $(SRC_MI) $(ZDIR)/advectx.o $(ZDIR)/advecty.o $(ZDIR)/advectz.o \
                            $(ZDIR)/advect.o $(ZDIR)/convection.o $(ZDIR)/datetime.o
	$(FC) $(ZFLAGS) $(DEFS) $(MI_DEFS) -c -o $@ $<
$(ZDIR)/harness_zoom.o: harness_zoom.F90 $(ZDIR)/modelIntegration.o $(ZDIR)/advectm_cfl.o
	$(FC) $(ZFLAGS) $(DEFS) -c -o $@ $<

# A clear message instead of make's "No rule to make target" when TM5ZOOM is wrong.
$(SRC_X) $(SRC_Y) $(SRC_Z) $(SRC_ADV) $(SRC_CONV) $(SRC_ZT) $(SRC_AT) $(SRC_MI) $(SRC_DT) $(SRC_CC) $(SRC_CFL):
	@echo "no tm5-zoom checkout at $(TM5ZOOM) -- pass TM5ZOOM=<path to tm5-zoom>"; exit 1

clean:
	rm -rf $(BIN) harness_zoom* $(BDIR) build-zoom* *.o *.mod *.dSYM

.PHONY: clean zoom zoom-one
