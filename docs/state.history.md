<!-- author: Ash Pera (NOAA/GML CIRES) -->
<!-- footer-title: pydyn — state, history -->

> **History of `state.md`, verbatim as of pydyn `88b2ddc`.** The head (`state.md`) states what is
> currently true; this file keeps the dated measurements, job references and the order things
> were found in. Section numbers are the head's. Not read by default.

# pydyn: what is transcribed, what is bitwise, and how to reproduce it

The head of this repo; `README.md` is the status page. Model-side pin:
`tm5-zoom@745e94f70` on `NOAA/ct-unified` (`docs/sources.md`, which says which rows were read
at which sha and how the re-pin was tested, §7); the file table below cites `4dc2017c5`, where most of it was read. Paths below
that start with `base/`, `proj/` or `docs/` are in that repo; everything else is in this one.

## 0. Status

**Scope.** The production build variant — `slopes` advection, `without_wet_deposition`, no
secmom — for **one region, no OpenMP**, under the meteo-server driver (§3b); a multi-rank MPI run is covered by merging its ranks
(§3e), and zoom nesting both in the standalone harness (§3d) and on a real glb3×2 + nam1×1
run (§3f), through `pydyn/zoom.py`, with every region's meteo through `zoom_meteo.py` and
`regrid.py`. Inside the one
region, everything the step
does: the three advection kernels and `dynam0`, convection, the reduced grid, `Check_CFL`,
`Meteo_Setup_Mass` (BalanceMassFluxes) and `m2phlb1` around them, the meteo ingestion from
the tm5-nc archive and `Calc_Kzz` on it, and the two physics operators of the CH4 run — the
surface emission injection (`s`) and the CH4 chemistry (`c`: Spivakovsky OH, Bruehl
stratospheric OH/Cl/O(1D), Hossaini tropospheric Cl, soil sink). Algorithm:
`tm5-zoom@4dc2017c5:docs/dynamic_core_survey.md`; the file-by-file map is the table below.

**Bitwise, under two independent disciplines, and the flags are part of the claim.**

| tier | what it compares | build |
|---|---|---|
| §3a | the production kernels, compiled unmodified against stubs and driven from Python | gfortran 16, `-O0 -ffp-contract=off`, arm64 |
| §3b, per event | every x/y/z/v/s/c call, the chemistry's loss rates, Setup_MassFlow, `Check_CFL`'s `(ndyn, n)`, the balanced fluxes and m2phlb of a real `ch4_6x4_fwd.rc` run, each replayed from that run's own dumps | `-O3 -march=native -ffp-contract=off` |
| §3b, `--free-run` | the same 23 h re-integrated from the first operator on, the Python carrying its own state, fluxes, `ndyn` and `phlb`; the dumps only checkpoints | as above |
| §3b, ingestion | the mass fluxes, surface pressures, temperature, convection and `dkg` of all 23 hourly reads (`--meteo`), the month's emission field (`--emissions`), the chemistry's OH, Bruehl, Hossaini Cl and soil inputs (`--chemistry`) and the initial tracer field (`--initial`), read or computed from the files the model reads; with all four the free run takes no field from a dump | as above |
| §3e, MPI | a 2-, 3- and 5-rank run (one rank empty) merged, against the serial run of the same build: every array of every event, and the merged sets replayed | as §3b, Open MPI 5.0.9 |
| §3f, zoom | a glb3×2 + nam1×1 run on Orion, per region: every x/y/z/v event, Setup_MassFlow and `Meteo_Setup_Mass` (the child's matched to its parent and balanced per parent cell), the raw meteo read from the 1×1, 137-level archive and regridded as tmm does, per event and in a 6 h free run that also carries the parent/child exchange, `update_parent` and m2phlb on its own meteo; `Meteo_Setup_Mass` and its ingestion on 672 more intervals per region, a week in each season; `Check_CFL`'s `(ndyn, n)` on 672 | ifort, Orion's `-fp-model precise -no-fma`, MPI, the model's MKL |
| §3d, zoom | the zoom exchange, the advection of a child at every slot of its split order, `dynam0`/`convec` per region, `Proces_Region` / a whole step and `Check_CFL`'s decision on two- and three-level trees (glb3x2 + nam1x1 among them), `modelIntegration.F90` and `advectm_cfl.F90` compiled unmodified | as §3a, `with_zoom`, no reduced grid |

§3b is 3 dump sets, 1264 events, 391 free-run checkpoints: all of it bitwise, and again on the
same three sets regenerated at the pin (`sources.md` §7). `runs/MANIFEST.md`
has a row for every set; the Orion glb3×2 sets other than `zm_replay` (`cfl3x2_*`, `tr3x2_*`,
`zm_2019-*`) carry findings, not bitwise claims. Two conditions on
that word — the build needs `-ffp-contract=off` (§3c), and the FFT and LAPACK steps are
bitwise only when the Python calls the model's own libfftw3 and LAPACK through ctypes
(numpy's give 1e-11 and 1e-15 relative instead).

Constants are not covered by any of that and are checked on their own terms: §4.

**Two results from outside those conditions.** Every row above but §3f is glb6×4, gfortran,
arm64, NumPy 1.26.4. On glb3×2 dumps from an **ifort MPI** build on Orion, run under
NumPy 2.0.2: `Check_CFL` chose the same `(ndyn, n)` as the model on **672 of 672** intervals
(`cfl_headroom.md` §8), and the flux path `Setup_MassFlow` builds — (pu, pv) and
(am, bm, cm) — came out **bitwise equal** to the model's own dumped `massflow` event at
both resolutions (§9).

Neither widens the table. The first is an integer agreement; the second is bitwise but on
one derived quantity rather than the operator set. With §3f and `PYD-1` they are the evidence
that the transcription is not tuned to one grid, one compiler or one NumPy.

**Open.**

- ifort is settled for Orion's build and not for Hercules' (§3c, `PYD-1`): a day of the
  ifort model at Orion's production flags replays bitwise per event and in `--free-run`
  (408 / 408, with the model's own MKL and libimf); Hercules' bare `-O3` is a few ulp off
  in the kernels and 7e-10 in the Poisson balance. Production reads `dkg` from a cache that
  is 7e-6 rms from live `Calc_Kzz`; live `Calc_Kzz` from the archive is bitwise (§3c)
- the isotope and `with_budgets` paths of the chemistry are not transcribed (`PYD-5`), nor
  the grouped (4DVAR-save) initial-field format or any IC needing vertical interpolation
- zoom: the inputs of `s` and `c` (emission, chemistry fields, the initial field) are still
  the dumps' (`PYD-3` item 5); every operator of both regions, the meteo, the convection's
  inputs, `dkg` and the chemistry's temperature are bitwise on real glb3×2 + nam1×1 runs (§3f).
  An MPI build mis-indexes a child's CH4 loss rates (§3f) -- a tm5-zoom bug, fixed in `0385efe7e`
- the MPI tracer decomposition (slabs split across ranks) is settled on the laptop -- 2, 3 and
  5 ranks bitwise the serial run (§3e, `PYD-2`) -- but not under ifort; the ifort replays
  (§3c, §3f) ran beside IO-server ranks with the tracer on one rank (the zoom's 6 compute + 2 IO
  ranks bitwise its serial run, §3f). OpenMP is untested

```
python3 -m pytest                      # 430: 19 property + 30 parity + 24 simple + 50 production + 12 ranks + 9 inputs + 159 zoom + 17 zoom meteo + 9 zoom census + 101 constant
make -C fortran                        # builds the §3a harness; parity tests skip without it
make -C fortran zoom                   # builds §3d's harness_zoom2/3; the zoom parity tests skip without them
                                       # (kernels from ../../tm5-zoom; TM5ZOOM=<path> overrides)
python3 -m pydyn.replay <dump_dir>     # §3b: replay a pydyn_dump directory of a model run
python3 -m pydyn.replay <dump_dir> --free-run   # §3b: carry the Python's own state through it
python3 -m pydyn.replay <dump_dir> --free-run --meteo <my.meteo.dir>   # … on meteo read from the archive
python3 -m pydyn.replay <dump_dir> --free-run --meteo <my.meteo.dir> --emissions <emis.ch4.input.dir>
python3 -m pydyn.replay <dump_dir> --free-run --meteo … --emissions … --chemistry <input/ch4c13> --initial <istart.66.filename>
                                               # … on nothing but the model's input files
```

| file (this repo) | Fortran source in `tm5-zoom@4dc2017c5`, as resolved by `proj/ct-unified/rc/__zoom_1day.rc` |
|---|---|
| `pydyn/kernels.py` `fluxes_from_meteo` | `base/trunk/src/advect.F90` dynam0 |
| `pydyn/kernels.py` `advect_x/_y` | `proj/ct-unified/trunk/src/advectx__slopes.F90`, `advecty__slopes.F90` dynamu/dynamv |
| `pydyn/kernels.py` `advect_z` | `proj/ct-unified/branches/speedup/src/advectz__slopes.F90` dynamw |
| `pydyn/kernels.py` `convection_matrix`, `fastminv`, `convec` | `base/trunk/src/convection.F90` convec, fastminv (see §3a for why not `__speedup`) |
| `pydyn/simple.py` | the same core written for reading: one generic `advect_line` for x/y/z, `column_matrix`, `np.linalg.inv`; agrees with `kernels.py` to 1e-12 (`tests/test_simple.py`), fluxes left untouched, `xi` = largest Courant number actually used |
| `pydyn/driver.py` `Model.step` | `modelIntegration.F90` do_steps: `xyzvsc` then `csvzyx`, dt = ndyn/2. `Model(grid=…)` routes x through `redgrid.advect_x_reduced`, as every production region needs and `replay` already did per event |
| `pydyn/ops.py` | the array idioms the kernels and production modules are written in — `seqsum`, `advance`, `upwind`, `courant`, `convergence`, `clusters`, `wrap`/`closed` halos — each a named Fortran statement evaluated in the same order, and `subcycles`, the one CFL sub-cycle search all six advection operators share (`progressive` vs `fresh` flux shrinking: dynamu/v vs dynamw, not interchangeable bitwise); the docstring says where `np.diff` is and is not bitwise-safe |
| `pydyn/redgrid.py` | `proj/ct-unified/trunk/src/redgridZoom.F90` uni2red/red2uni/uni2red_mf and the reduced-grid wrapper of `advectx_work` |
| `pydyn/cfl.py` | `proj/ct-unified/trunk/src/advectm_cfl.F90` Check_CFL, dynamum/dynamvm/dynamwm, new_valid_timestep |
| `pydyn/balance.py` | `meteo.F90` Meteo_Setup_Mass, FillMassChange, CheckMassBalance; `grid_type_ll__speedup.F90` BalanceMassFluxes_m / SolvePoissonFFTW; `go_date` InterpolFractions; `advect_tools` m2phlb1 |
| `pydyn/lapack.py` | `convection__speedup.F90` `fastminv` (`with_lapack`: DGETRF + DGETRI) through the system LAPACK |
| `pydyn/emis.py` | `proj/ct-unified/trunk/src/emission_data.F90` do_add_2d (the whole of `sources_sinks_apply → emission_ct_apply` for a surface-injected CH4 run) and the month's field it injects: `emission_ch4_tracer.F90` emission_ch4_calc (`emis_unified`; `emission_units.F90` EU_MICROMOL, `emission_data.F90` flux_to_gridbox_per_month with `geometry.F90` calc_dxy = `meteo.dxyp`), `emission_ct.F90` emission_ct_calc's one-member sum at the constant scaling 1.0 (exact) and `toolbox.F90` coarsen_emission_2d = FillGrid 'sum' (`meteo.coarse_sum`) |
| `pydyn/chem.py` | `proj/ct-unified/branches/chem_unified/src/chemistry_ch4.F90` ch4_loss_kernel / ch4_chemistry_step (CH4-only path) and `emission_regrid.F90` rebin_levels, vectorised; `ops.libm` = the C library's exp/pow/cos/sin one value at a time, as the Fortran calls them |
| `pydyn/chemfields.py` | the chemistry's loss-rate inputs from their files: `read_OH_field.F90` read_CH4_loss_rate / interpolate_lat (the Bruehl columns), Import_OH_Spivakovsky (the `echlevs` clumps, then `grid_type_ll__speedup.F90` FillGrid 'area-aver' = `meteo.area_average`), Read_trop_Cl_Hossaini with `emission_regrid.F90` setup_interpolation_map_lat/_lon / apply_interpolation; `chemistry_ch4.F90` ch4_chemistry_init's soil map (qianlai); `dims_levels__ml137_tropo34.F90` echlevs |
| `pydyn/initial.py` | the initial tracer field, `istart 66`: `sources_sinks__ctuni.F90` start_from_field_fill3d (flat format), `grid_3d.F90` Fill3D 'mass-aver', `grid_type_ll__speedup.F90` Relate / FillGrid 'area-aver', `grid_type_hyb.F90` levi_Compare / FillLevels (the copy case); `chem_param.F90` fscale |
| `pydyn/meteo.py` | the meteo ingestion: `tmm_mf_tm5_nc.F90` GetTimeRecordNumber / mf_ReadRecord and the udunits factor (`Archive`, `SOURCES`, `FACTORS`), `meteo.F90` SetupSetup `interpN`/`averN`/`month` (`instants`, `interval`, `month`), `meteodata.F90` TimeInterpolation (`interpolate`), `grid_type_ll.F90` Init / Match 'area-aver' (`cell_areas`, `match_global_mean`) / FillGrid 'area-aver' of the 1×1 surface fields (`area_average`), `geometry.F90` dxyp, Pressure_to_Mass, Meteo_Produce_ConvUnitConvert / ConvCloudDim; `Meteo` drives them per read interval, `Meteo.diffusion` composing Calc_Kzz |
| `pydyn/diffusion.py` | `diffusion.F90` calc_kzz (`mass_t`, `dkg`), `diffusion_kernels.F90` dd_calc_ustar (`ustar`), bldiff (`bldiff`: free-troposphere and boundary-layer diffusivities, the two-pass boundary-layer height, `jqif` as 0/1 masks), `meteo.F90` Meteo_Produce_Gph (`geopotential`); pure arithmetic on (im, jm[, l]) arrays, no I/O |
| `pydyn/simple_diffusion.py` | Calc_Kzz written for reading: the Fortran's 0/1 masks as `np.where`, the boundary-layer height from the first crossing of the whole bulk-Richardson profile, the unit conversions as the physics; agrees with `diffusion.py` to a few ulp (`tests/test_simple.py`; on the production day, `test_production.py`: 2e-13 relative at worst, from the ill-conditioned boundary-layer height of convective columns) |
| `pydyn/dump.py`, `pydyn/replay.py` | reader for the `pydyn_dump` events, and `merge_ranks` for a multi-rank run (§3e); per-event replay and `FreeRun`, the Python integrating on its own with the dumps as checkpoints, `Ingestion` feeding it the meteo, emission, loss-rate and initial fields read from their files (§3b) |
| `pydyn/zoom.py` | the zoom tree and exchange (§3d): `initexit.F90` determine_children_etc / blank_zoom_region / define_splitorderzoom (`Zoom`); `advect{x,y}__slopes.F90` advectxzoom / advectyzoom, put_/compress_/uncompress_{x,y}edges, dynamu / dynamv of a child; `advectz__slopes.F90` (speedup) advectzzoom, dynamw of a child; `zoom_tools.F90` mix_edges, update_parent, coarsen_region; `advect_tools__speedup.F90` m2phlb, m2phlb1, coarsen_ambm; `advectm_cfl.F90` Setup_MassFlow's region loop; `advect.F90` dynam0 and `convection.F90` convec of any region; `modelIntegration.F90` do_steps, Proces_Region, and `tm5.F90`'s two half steps (`step`); `advectm_cfl.F90` Check_CFL's zoom path (`check_cfl_zoom`: store/restore_masses, determine_cfl_iter, its do_steps and split order, advectm{x,y,z}zoom, dynamum/vm/wm of a child); `emission_data.F90` do_add_2d and `chemistry_ch4.F90` ch4_chemistry_step's phase 2 on the cells a region owns (`owned`, `sources_region`, `chemistry_region`) |
| `pydyn/zoom_meteo.py` | Meteo_Setup_Mass for a child (§3f): `meteo__meteo_server.F90` (`n > 1`: sp Match 'area-aver' to the parent, the raw fluxes Match 'sum' to the parent's balanced ones, the Richardson loop); `grid_type_ll__speedup.F90` GetRefinement, Match_cell / Match_u / Match_v, BalanceMassFluxes_m against a parent cell by cell, EnsurePoissonCache / SolvePoissonDirect (`PoissonDirect`); `ChildMeteo`, `meteo.Meteo` with the child's sp match |
| `pydyn/regrid.py` | tmm's regridding of a record onto a region (§3f): `tmm.F90` ReadField / Transform3Dh / Transform3Dv, `tmm_param.F90` GetCombineKeys, `grid_type_ll__speedup.F90` FillGrid 'copy' / 'combine' / 'weight' (`fill_grid`), `grid_type_hyb.F90` levi_Init_select / levi_Compare / levi_FillLevels_3d 'combine' 'sum' / 'mass-aver' / 'w' (`fill_levels`); `RegriddedArchive` serves mfuv, mfw, sp, convec, t and q of the 1×1, 137-level set as `meteo.Archive` does a model-grid one |
| `analysis/ch4_loss_bounds.py` | which loss rates a zoom child's chemistry applied: the intended ones or ch4_loss indexed from 1 by the child's own i, j, as an MPI build's `c_f_pointer` binds it (§3f) |
| `pydyn/layout.py` | halo-layout ↔ pydyn conversions, stream-I/O readers |
| `pydyn/synth.py` | seeded admissible inputs shared by both test files |
| `pydyn/harness.py` | write a case, run `fortran/harness`, read back, ulp-level compare |
| `fortran/` | stub modules, the two drivers (`harness`, §3a; `harness_zoom`, built as `harness_zoom2`/`3`, §3d), Makefile; compiles the sources above unmodified and in place, from the tm5-zoom checkout at `$(TM5ZOOM)` |

## 1. Conventions

0-based, no halos, C-contiguous, float64.

| array | shape | meaning |
|---|---|---|
| `m` | `(im, jm, lm)` | air mass, kg |
| `rm, rxm, rym, rzm` | `(im, jm, lm, nt)` | tracer mass and slopes, kg |
| `am` | `(im+1, jm, lm)` | mass through x-faces per half-step; `am[k]` from cell k-1 into k; `am[0] == am[im]` |
| `bm` | `(im, jm+1, lm)` | y-faces; `bm[:,0] == bm[:,jm] == 0` |
| `cm` | `(im, jm, lm+1)` | z-faces; `cm[...,0] == cm[...,lm] == 0` |

Fortran ↔ NumPy: cell `(i,j,l)` → `[i-1, j-1, l-1]`; `am_t(i,j,l)` → `am[i, j-1, l-1]`;
`bm_t(i,j,l)` → `bm[i-1, j-1, l-1]`; `cm_t(i,j,l)` → `cm[i-1, j-1, l]`.
`bt[0] = 1` at the surface, `bt[lm] = 0`.

Everything is updated in place, including the fluxes: `advect_x`/`advect_y` scale the flux
row/layer by `nloop/(nloop+1)` per CFL retry and by `nloop` at the end (same non-exact
"restore" as the Fortran); `advect_z` restores `cm` exactly from a copy.

Floating-point evaluation order follows the Fortran statement by statement, including the
asymmetries that are only visible bitwise: `rm + (f_l - f_r)` in x and y but
`rm + f_l - f_r` in z; `pf = (-3*bm)*f` on the pole faces; the un-updated pole-row
`rxm`/`rym`; `conv1` accumulated in k order; the Crout LU without pivoting in `fastminv`.

```python
import numpy as np
from pydyn import Model, State, mass_from_pressure

model = Model(bt, ndyn=3600)          # bt (lm+1,), hybrid b at interfaces
state = State(m, rm)                  # slopes default to zero
model.step(state, pu, pv, conv=dict(entu=..., detu=..., entd=..., detd=..., dkg=...,
                                    cloud_top=..., cloud_lfs=..., lmax=lm))
model.stats                           # {'x': {'nloop_max', 'xi'}, 'y': ..., 'z': ...}
```

## 2. What the property tests cover

Per kernel: a uniform mixing ratio is preserved; air and tracer mass are conserved
(periodic x, closed y and z); positivity with the limiter at Courant 0.9; sub-cycling is
triggered by over-CFL fluxes and stays conservative; `CFLError` at the loop limit; the pole
rows keep their `rxm`/`rym`; an x-row and a z-column are the same 1-D problem under the
slope permutation (checks the three transcriptions against each other); the dynam0 vertical
flux reproduces the hybrid distribution of the column convergence; `fastminv` agrees with
`numpy.linalg.inv`; convection conserves column mass and preserves a uniform ratio; the
driver runs the symmetric sequence.

None of this proves parity with the Fortran. That needs §3.

## 3. Kernel-parity framework against the Fortran

The goal is a test per kernel that says "for these inputs the Python and the compiled
Fortran produce the same `m, rm, rxm, rym, rzm` (and returned flux)". Two ways to obtain the
Fortran side; they are complementary.

### 3a. Standalone harness: compile the real kernels against stub modules

`fortran/`:

```
Makefile     make [FC=ifort] [OPT=-O3] [WITH_LAPACK=1] [TM5ZOOM=<checkout>]; the kernel
             sources are read from that checkout by their overlay path, never copied
stubs.F90    GO go_system dims chem_param MeteoData tracer_data global_data ParTools
             omp_ParTools redgridZoom zoom_tools toolbox datetime
advect.h     the cpp defines of the production build (slopes, without_wet_deposition, ...)
tm5.inc      #include "advect.h" -- the base sources find base/trunk/src/tm5.inc first, which
             itself includes advect.h; -I. resolves that to ours, so one file holds all defines
harness.F90  reads <case>/case.nml + *.bin, calls one kernel for region 1, writes out_*.bin
harness_zoom.F90  the same for a region tree, built per region count as harness_zoom2/3 (§3d)
```

The kernels are compiled as-is. `dynamu/dynamv/dynamw` are `private`, so the driver enters
through the public `advectxzoom(1)/advectyzoom(1)/advectzzoom(1)` with `status(1) = 0` and
`splitorderzoom(1,1:6) = 'xyzvsc'`; for region 1 (`yref = zref = 0`, `touch_sp/np = 1`,
`zoom2D`) these reduce to `dynamu(1,1,jm,1,lm)`, `dynamv(1,1,im,1,lm)`, `dynamw(1,1,im,1,jm)`
and the edge bookkeeping (`put_xedges`, `compress_xedges`, `mix_edges`) is a no-op, so empty
stubs are faithful. Halo bounds are the production ones: `m_dat(-1:im+2,-1:jm+2,lm)`,
`rm_t/rxm_t/rym_t/rzm_t(-1:im+2,-1:jm+2,lm,nt)`, `am_t/bm_t/cm_t(0:im+1,0:jm+1,0:lm+1)`.
`escape_tm` prints and `error stop 2`, which `pydyn.harness.run` turns into `HarnessError`.

**Convection caveat.** Production resolves `convection.F90` to
`proj/ct-unified/branches/speedup/src/convection__speedup.F90`, which needs the rc file,
shared-memory windows and MPI; the harness compiles `base/trunk/src/convection.F90`, the
arithmetic it was derived from. Parity of the speedup overlay against base was an assumption
here; §3b settles it for the serial case — `ch4_6x4_fwd.rc` has `branches/speedup` in
`my.source.dirs`, so the `v` operator replayed bitwise is `convection__speedup.F90`, at
`par.ntask=1`. The shared-memory paths of that overlay are still untested.

Case format: one directory, each array a raw stream file (`layout.write_fortran_stream`,
halo layout via `layout.to_tm5`) plus `case.nml` (`kernel, im_, jm_, lm_, nt, ndyn_,
lmax_conv, limits_, debug`). `harness.write_case` / `harness.run` / `harness.compare`:

```python
from pydyn import harness, synth
harness.write_case(d, "y", state, bm=bm, ndyn=3600)
out = harness.run(d)                       # m, rm, rxm, rym, rzm, bm, stats{nloop_max, xi, nxi}
harness.format_report([harness.compare(out["rm"], rm_py, "rm")])
```

`compare` reports count, max ulp distance, max relative difference and the first differing
index — a 1-ulp difference is an evaluation order, a large one an index error. Three
order-of-evaluation details found this way and now mirrored in Python: dynamu multiplies by a
precomputed `1/m`; dynamv computes `rym` from hoisted `inv_mnew, flow_conv, flow_mean,
mass_x3`; dynamw's CFL trial loop exits at the first violation, so `xi(region,3)` only sees
the Courant numbers up to it.

`tests/test_parity.py` (seeded, `pydyn.synth` inputs): x/y/z at Courant 0.3, 0.9, 2.5
(sub-cycling), slopes zero and saturated at `±rm`, limiter off, flows into and out of both
poles, `cm` from dynam0 with a hybrid `bt`, CFL failure on both sides; dynam0; convection
with random clouds, `cloud_top = 0`, `cloud_top = lm`, `cloud_lfs > cloud_top`,
`detu > amu + entu` (the `zxi = 0` branch). All bitwise.

### 3b. Replay of an instrumented model run

`tm5-zoom:proj/ct-unified/trunk/src/pydyn_dump.F90` (module `pydyn_dump`, rc-gated, compiled
into every ct-unified build) writes the state around each dynamic-core call of a real run:

```
pydyn.dump             : T                  ! default F: no output, no cost
pydyn.dump.dir         : <existing dir>     ! default ./pydyn_dump
pydyn.dump.max_events  : 400
```

Hooks: `modelIntegration.F90` (`bef_x/aft_x`, `y`, `z`, `v`, `s`, `c` around every operator
call of `do_steps`; `m2phlb` after each `proces_region`), `tm5__meteo_server.F90` and
`initexit.F90` (`meteo` after Meteo_Setup_Mass, `cfl` around Check_CFL, `massflow` after
Setup_MassFlow), `emission_data.F90` (`emis`: one event per `do_add_2d` call, the field and
every factor) and `chemistry_ch4.F90` (`chem`, written through the module's generic
`pydyn_dump_begin/key/put/end` so that `pydyn_dump` itself depends on no chemistry overlay:
the month's loss-rate inputs — `temper`, `phlb`, `lat`, the zonal `strat_press/oh/cl/od`,
`oh_3d`, `pl_trop/cl_trop`, `soil` — the scale factors, `dt_sec`, `ch4_tracers`, and the
`loss(im,jm,lm,nreac)` the kernel produced). Every
event is `e<seq>_<tag>_r<region>_p<rank>.meta` (header: `idate, itau, ndyn, nconv, nread,
ntracet, im, jm, lm, xref..tref, xcyc, limits, grid_reduced, nred, jred, clustsize, conv_dt,
...`; `array NAME = lb ub ...` per array) plus one raw stream `.bin` per array **with its
Fortran halo bounds**: `m(-1:im+2,-1:jm+2,lm)`, `rm..(…,nt)`, `am/bm/cm(0:im+1,0:jm+1,0:lm+1)`,
meteo `pu/pv/mfu/mfv(0:im+1,0:jm+1,0:lm)`, `mfw(1:im,1:jm,0:lm)`, `sp1/sp2(-1:im+2,-1:jm+2,1)`,
`entu..dkg(im,jm,lmax_conv)`, the cloud levels as `iarray`. `pydyn.dump.Event` reads them
lazily; `ev["am"][0:im, 1:jm, 1:lm]` is the Fortran section, `ev.interior("am")` the pydyn
layout.

**The reference is defined against the meteo-server driver, `tm5__meteo_server.F90`
(`PYD-6`).** Every CH4 rc with a source list builds it. `base/trunk/src/tm5.F90` is not hooked and will not be:
the builds that take it are the `co2rc_*` family and five `drift_*` rc files (118 of the 594
ct-unified rc files with a source list, at tm5-zoom `c133fb437`), none of which pydyn has a
use for, and hooking it would mean editing `base/trunk` for builds nobody replays. Such a build
still compiles `pydyn_dump` and writes the operator, `m2phlb`, `emis` and `chem` events (their
hooks are in `ct-unified/trunk`) and the first interval's `meteo` and `cfl` (`initexit.F90`
`start`); it writes no `massflow` and no later `meteo` or `cfl`. So its operator events replay
per event, while the ingestion check and `--free-run` lack their inputs (the free run takes its
fluxes from `massflow` events). Read from the source, not run.

`python3 -m pydyn.replay <dir> [--tags x,y,z,v,s,c,meteo,cfl,massflow,m2phlb] [--fft auto|fftw|numpy]
[--lapack auto|lapack|numpy] [--free-run] [--meteo …] [--emissions …] [--chemistry …] [--initial …] [-v]` replays every event from the Fortran's own
inputs, so a difference cannot propagate to the next event, and prints a per-array table:

| event | Python | compared |
|---|---|---|
| `bef_x → aft_x` | `redgrid.advect_x_reduced` (or `kernels.advect_x` when the grid is not reduced) | `m rm rxm rym rzm am` |
| `bef_y → aft_y`, `bef_z → aft_z` | `kernels.advect_y`, `kernels.advect_z` | `… bm` / `… cm` |
| `bef_v → aft_v` | `kernels.convec` with `conv_dt`, `lmax_conv`, `inverse=lapack.LapackInverse` | `m rm rxm rym rzm` |
| `bef_s → [emis…] → aft_s` | `emis.do_add_2d` for each `emis` event in between | `m rm rxm rym rzm` |
| `bef_c → chem → aft_c` | `chem.loss_rates` on the `chem` event, then `chem.decay` / `apply_decay` | `loss`, then `m rm rxm rym rzm` |
| `massflow` | `balance.time_interpolation` at the step mid-point + `kernels.fluxes_from_meteo` | `pu pv am bm cm` |
| `cfl` | `cfl.check_cfl` with the same interpolation callback | `ndyn n` |
| `meteo` | `balance.setup_mass` on `mfu/mfv/mfw` (pre-check, BalanceMassFluxes, CheckMassBalance) | `pu1 pv1 pu2 pv2` |
| `m2phlb` | `balance.m2phlb1` | `sp phlb` |

`--free-run` (`replay.FreeRun`) is the opposite discipline: the state is taken from the first
`bef_*` dump and never again; the balanced pairs come from the Python's own `meteo` events
(keyed by the data instants `pu_t1/pu_t2`, not the read interval), `ndyn` from its own
`cfl`, `am/bm/cm` from its own `massflow`, `phlb` for the chemistry from its own `m2phlb`,
and each dump is only a checkpoint — `in:m`, `in:rm`, …, `in:am`, `in:phlb` are the carried
state on *entering* an operator against `bef_*` / `chem` (they would catch anything the
model does to the state between operators), the plain names the result against `aft_*`.
The inputs the model reads from files — the meteo, the emission fields, the OH / loss-rate
climatologies, the soil map, the initial tracer field — are taken from the dumps unless
ingested (below); they are inputs, not state. `seeded: True` in the notes marks the few
events before the first dynamic step that still had to use the dump (cfl, the first `x`,
and the first `c`, which precedes the first m2phlb); with `--meteo` and `--initial` there
are none — the state starts from the initial field at the first `meteo` event and the first
`c` runs on the initial `phlb` of the ingested air mass.

`--meteo <root>` (`replay.Ingestion`, `meteo.Meteo`) removes the meteo from that list: the
raw `mfu/mfv/mfw` at the two archive instants, `sp1/sp2` and the initial `m` of every
`meteo` event, `temper` of every `chem` event and `entu/detu/entd/detd/cloud_*` of every
`bef_v` are read from the tm5-nc files under `<root>` (the rc's `my.meteo.dir`; the
production sourcekeys in `meteo.SOURCES`) and compared with the dumped ones as
`in:<name>`; in a free run they are also the ones used. The model-level archive is already
on the model grid and levels, so its ingestion is the time rules and the surface-pressure
match; the surface fields Calc_Kzz needs (`u10m v10m sr` interp1, `sshf slhf ewss nsss`
aver1, `srols` monthly, `oro lsm` constant) are hourly 1×1 and go through FillGrid's
area-weighted mean (`meteo.py`'s docstring lists the six rules). `dkg` is then
`diffusion.calc_kzz` on them with `T`, `Q`, the step's interpolated `pu/pv` from the
preceding `massflow` dump (the Python's own in a free run), the step's starting surface pressure for
the geopotential (a parent's is m2phlb1 of its mass after Setup_MassFlow's coarsen_region; a child's its last `m2phlb` before the step, else the `meteo` dump's) and `spm`,
the read interval's matched `sp1/sp2` weighted to the step's middle — compared as `in:dkg`
on every `bef_v` (the two of a step share one).

`--emissions <dir>` (`emis.Emissions`) removes the emission field too:
the month's `flux1x1_all_source.YYYYMM.nc` under `<dir>` (the rc's `emis.ch4.input.dir`)
gives `emis` in kg per model cell per month as `raw * (xmch4 / 1e9)`, times the 1×1 cell
area of calc_dxy (the latitude accumulated cell by cell — as `j·dlat` the field is off by
thousands of ulp) and the month's seconds, then FillGrid's 6×4 block sums; the forward run's
EnKF scaling is the constant 1.0 and its one-member sum starts from 0.0, both exact. It is
compared as `in:emis` on every `bef_s`; a free run injects it with `month2dt` from the
calendar rather than the dump's scalar. Only the unscaled forward field is transcribed: the
PYD-2 sets (§3e) inject four members scaled by their EnKF factors and differ from it by
design, up to 40 %.

`--chemistry <root>` (`chemfields.LossFields`) removes the loss-rate inputs of every `chem`
event: under the CH4 input tree (`input/ch4c13`, where the rc's `ch4loss.inputdir`,
`oh.file`, `oh.gph.inputdir`, `trop.cl.filename` and `ch4.soil_sink.input.dir` all point)
it reads the three Bruehl files (`Lrate`, `pressure` — each (level, month) column linearly
interpolated in latitude by `interpolate_lat`, on an input latitude the later bracketing
interval winning), the Spivakovsky OH (137 levels clumped onto the model's 34 by
`echlevs`, each clump the mean weighted by the month's geopotential thicknesses, then
FillGrid's area-weighted mean from 1×1), the Hossaini Cl (converted to molec/cm³, the
latitude then the cyclic longitude map of `mod_emission_regrid`, levels flipped) and the
qianlai soil map of the step's year and month; compared as `in:strat_press`, `in:oh_3d`,
`in:cl_trop`, `in:soil`, … Every file value is widened float32 → float64 as MDF does, and
read unmasked (the Hossaini fill values are data to the Fortran).

`--initial <file>` (`initial.initial_state`) removes the initial tracer field, `istart 66`
(`start_from_field_fill3d`): the file's ppm field on its own 3×2 grid goes through Fill3D
'mass-aver' — FillGrid's area-weighted mean level by level, then FillLevels, which for a
file on the model's own 34 levels (|Δa|, |Δb| < 0.1) is a copy — and becomes
`rm = f · 1e-6 · m / fscale` for every tracer, slopes zero, with the initial air mass. It is
compared with the first `bef_*` as `ic:rm`, …; a free run starts from it.

**Result** on the local-mac build of `ch4_6x4_fwd.rc` (glb6x4 60×45×34, one CH4 tracer,
`nred = 4`, `with_lapack`, FFTW Poisson solver, one emission category injected at the surface,
`oh.scale 0.901`, tropospheric Cl and the qianlai soil sink on), `-O3 -march=native
-ffp-contract=off`, `pydyn_dump_phys` = 23 h of 2000-01-01 at the production timing (759
events, 4.8 GB; 46 calls of each of the six operators, 24 meteo reads): **everything
bitwise**, per event and in `--free-run` — x (reduced rows and the `nloop = 2` sub-cycling
included), y, z, v (Accelerate DGETRF/DGETRI), s, c (the 60×45×34×4 loss rates *and* the
decayed moments; `rebin_levels` vectorised as a sequential sum over every input layer,
`exp`/`pow`/`cos` through libm one value at a time), massflow, cfl (`ndyn = 3600, n = 1`, one
iteration), meteo (FFTW `-framework Accelerate`), m2phlb. In the free run the Python
integrates the day of the production model on its own — transport, injection, decay — and
all 391 checkpoints agree to the bit, `in:phlb` at every chemistry call included; with
`--meteo` on the local archive the 23 ingested meteo reads, 46 temperature and 46 convection
inputs and the 23 hourly `dkg` fields (96 % non-zero, up to 3.7e11 kg/s; 4 s for the day)
are bitwise too, as is the January 2000 emission field with `--emissions` (474 Tg CH4/yr;
the log's 475.6 counts the leap year), and the free run on them stays bitwise throughout.
With `--chemistry` and `--initial` as well, the 46 sets of loss-rate inputs and the initial
field are bitwise, and the free run on all four — the day integrated from the input files
alone, no event seeded from a dump — agrees at all 391 checkpoints. The same holds on
the four-tracer `pyd2_serial` set (§3e) with `--meteo --chemistry --initial`, 394
checkpoints. With
`--fft numpy` the balanced fluxes are within 1e-11 relative (numpy's FFT is not FFTW's
`FFTW_ESTIMATE` plan); with `--lapack numpy` the convection tracers are within 7 ulp
(`numpy.linalg.inv` is an LU solve, not DGETRI). `tests/test_production.py` runs the replay
when the phys set (by default the one made at the pin, below) or `$PYDYN_DUMP_DIR` exists and
asserts exactly that. Two more dump sets from a **transport-only build** (`without_emission
without_chemistry` added to `build.configure.macro.define` — overriding `my.tm5.define` on the
command line is too late, it has already been expanded), kept because they are the ones where
the transport can be studied without physics in between: `pydyn_dump_free`, 23 h of
2000-01-01 at the production timing (483 events, 3.3 GB, largest Courant number 0.998),
bitwise per event *and* in `--free-run`; and
`pydyn_dump_cfl`, the same build at `time.ntimestep = time.ndyn_max = 10800` with
`meteo.tinterp.*=interp3`: Check_CFL rejects 10800 s (`iterations = 2`) and the step runs at
5400 s, `n = 2`, x/y/z/v/massflow/m2phlb all bitwise in free-run mode (22 events; the run then
aborts on the hourly `aver1` surface fields, which `meteo.F90`'s second-interval guess cannot
fit into a 1.5 h step — hence no long run at that timing). Note that y/z sub-cycling
(`nloop > 1`) cannot be observed in the model: Check_CFL chooses ndyn so that it never
happens; only the §3a harness exercises it. That is not a property of the meteorology but
of the checker's own sub-cycle caps, which is what [`cfl_headroom.md`](cfl_headroom.md)
takes apart — the `pydyn_dump_cfl` rejection above is a z column needing two sub-cycles
where `Check_CFL` allows one. Three transcription details this tier caught
that §3a could not: gfortran's `sum`
intrinsic accumulates sequentially (`uni2red`, `m2phlb1`, `CheckMassBalance`) where
`np.sum` is pairwise; `X / fac` on a complex array is componentwise; and on the reduced grid
`m` is replaced by the uniform-grid mass update while only the tracers are re-split.

**Making the dumps (local mac).** In the tm5-zoom checkout, not here. Bash, not zsh (`tm5_env.sh` uses `BASH_SOURCE`); the
project directory and the dump directory must exist; pycasso overrides follow the rc file;
FFTW is not in `machine-local-mac.rc`, and `compiler-gfortran-local-mac.rc`'s `optim-fast` is
`-O3 -march=native` *without* `-ffp-contract=off`, so both are passed on the command line:

```
./setup_tm5 proj/ct-unified/rc/ch4_6x4_fwd.rc build.configure.flags=optim-fast \
    "compiler.lib.lapack.libs=-framework Accelerate -L/opt/homebrew/lib -lfftw3" \
    "compiler.flags.optim-fast.fflags=-O3 -march=native -ffp-contract=off" \
    par.mpi=F par.ntask=1 par.compute_ranks=1 par.io_ranks=0 par.io_server_ranks=0 \
    ct.ensemble.size=1 ct.ensemble.total_members=1 cfl.cache.read=F \
    "timerange.start=2000-01-01 00:00:00" "timerange.end=2000-01-01 23:00:00" \
    pydyn.dump=T "pydyn.dump.dir=$PWD/.tm5-cloud-scratch/pydyn_dump_phys" pydyn.dump.max_events=3000 \
    ... --allow-new-keys --no-submit -j 8
cd <rundir>; DYLD_LIBRARY_PATH=/opt/homebrew/lib ./<exe> <rc>      # ~30 s, 759 events, 4.8 GB
```

(The local meteo archive holds only 2000-01-01, and the hourly `u10m` of 2000-01-02 is
needed to step to 24:00, hence 23 h.)

The recipe as a script: `runs/setup_repin.sh <sha> phys|free|cfl`
clones tm5-zoom clean at `<sha>`, writes the paths and the end time into a copy of the rc
(pycasso expands them before command-line overrides apply), writes a one-member scaling
file at 1.0, and takes the transport-only macro list from the phys run's expanded rc. Since
tm5-zoom `5292746df` the build also needs `CONDA_PREFIX` exported (`/opt/anaconda3` here).
Run from scratch at `745e94f70` it reproduced the three sets' event counts and sizes.

For the transport-only sets add `"build.configure.macro.define=<the expanded list from the
run's rc> without_emission without_chemistry"`, and for `pydyn_dump_cfl`
`time.ntimestep=10800 time.ndyn_max=10800 "meteo.tinterp.*=interp3"`. `tests/test_production.py`
looks for the three sets made at the pin,
`$PYDYN_TM5ZOOM/.tm5-cloud-scratch/pydyn_build/dumps/repin_{phys,free,cfl}_745e94f70` (default
`../tm5-zoom`; `$PYDYN_DUMP_DIR`, `$PYDYN_FREE_DUMP_DIR`, `$PYDYN_CFL_DUMP_DIR`), and skips
what is absent; a 23 h replay or
free run takes ~20 s.

### 3c. What can and cannot be bit-identical

- `optim-fast` is not one thing: Orion's is `-O3 -xHost -g -fp-model precise -no-fma`
  (`base/trunk/rc/compiler-ifort-orion.rc:62`), Hercules' bare `-O3`
  (`compiler-ifort-hercules.rc:60`), which leaves ifort's default `-fp-model fast=1` on.
  **ifort, tier 1** (2026-09-24, Orion, ifort 2021.12.0, Xeon Gold 6148, NumPy 2.0.2; jobs
  `23887649`, `23887807`, `runs/MANIFEST.md`): `tests/test_parity.py` + `tests/test_zoom.py`
  (188 bitwise tests) against the kernels built five ways —

  | arm | flags | passed |
  |---|---|---|
  | O0 | `-O0 -fp-model strict` (the Makefile's ifort default) | 188 / 188 |
  | O3 | `-O3 -fp-model strict` | 188 / 188 |
  | Orion production | `-O3 -xHost -fp-model precise -no-fma` | 188 / 188 |
  | Hercules production | `-O3` (fast=1, generic SSE2 target) | 94 / 188 |
  | fast + FMA | `-O3 -xHost` (fast=1, AVX-512 FMA) | 36 / 188 |

  The failures look like gfortran's contraction did: `rm` at most 8 ulp (10 with `-xHost`),
  `sp`/`phlb` 1–23 ulp, `cm` up to 2048 ulp, and the slopes' near-zero values changing sign
  (relative 2.0). `Check_CFL`'s `(ndyn, n)` is the model's in every case, the fast arms
  included — only the arrays it leaves differ. So the Python is bitwise the Orion production
  kernels and a few ulp from Hercules'; the Hercules arm was built on Orion's CPU, which
  with no `-x` target generates the same code, but not with Hercules' own ifort.
  **ifort, tier 2** (2026-09-24, the same machine; `runs/MANIFEST.md`'s `pyd1-*` sets): the
  `ch4_6x4_fwd.rc` day — glb6x4, one CH4 tracer, 1 compute + 2 IO-server ranks, full physics,
  792 events, 408 checkpoints — built with Orion's `optim-fast` and replayed with the
  model's own libraries: MKL 2022.2.1 for both DGETRF/DGETRI and the `dfftw_*` Poisson
  solver (`-qmkl=sequential`; `libmkl_rt.so` through `$PYDYN_LAPACK` / `$PYDYN_FFTW`) and
  ifort's **libimf** for exp/log/pow/cos/sin (`--libm`, `ops.use_libm`). **Bitwise, per
  event and in `--free-run`, 408 / 408.** With glibc's libm instead, everything but the
  chemistry is still bitwise; its OH and Cl rates are 2–3 ulp off (libimf's exp/pow round
  differently), and the free run carries that to 1e-11 in the slopes. The ingestion on
  Orion's copies of the inputs is bitwise too — meteo, emission, the chemistry's
  climatologies, the initial field, `dkg` — and so is the all-files free run, 408 / 408.
  `dkg` needs ifort's integer power: it compiles `ustr**3` (wobkl) to libimf's `__powr8i4`,
  1 ulp from `(x*x)*x` on a quarter of values (`**2` it inlines), and that ulp grew to 557 in
  `dkg` and 21–27 in the free run's `rm` until `ops.libm.cube` called it. Two things the
  run showed about the model rather than the Python: `diffusion.cache.read` is retired —
  the `dkg` cache is read through `tmm.sourcekey.<region>.{ml,sfc}.diffus : dkg-cache:`, and
  a missing file falls back to live `Calc_Kzz` — and the production cache
  (`METEO/tm5-nc/ec/ea/dkg/NEW/tropo34/glb6x4`, written 2026-07-08) is not what live
  `Calc_Kzz` gives this build: 96 % of cells differ, 7e-6 rms relative per hour (at most
  1.4e-4), level means equal to four figures. The Hercules-flags build (`-O3`, fast=1)
  replays as tier 1 predicted: per event 171 of 408 checkpoints off — BalanceMassFluxes to
  7e-10, x/y/z a few ulp and near-zero slopes changing sign, the chemistry 4 ulp — with
  `massflow`, `cfl`, `v` and `m2phlb` still bitwise event by event. Earlier,
  with gfortran 16 on arm64: in the §3a harness `-O3`
  and `-O3 -ffp-contract=fast` are bitwise equal to `-O0` for x/y/z at Courant 2.5 and for
  convec — but the full model at `-O3 -march=native` (contraction on) replayed with the same
  Python is not: `phlb = at + bt*sp` moves 1 ulp, the dynam0 `cm` up to 9e-11 relative
  (a divergence of contracted products), and the slopes get sign flips of values near
  zero (`rel = 2`). `-ffp-contract=off` makes all of it bitwise; the generic
  `compiler-gfortran.rc` already has it, `compiler-gfortran-local-mac.rc` does not. That is
  a gfortran statement, not an ifort one; redo with `make FC=ifort OPT=-O3` on a machine
  that has it.
- NumPy elementwise ops are IEEE double per element (no reassociation of `a + b - c`
  within an expression; the parenthesisation in the source is what is evaluated), so
  matching order-of-operations statement by statement is sufficient on the Python side.
  `np.sum`/`np.dot` are *not* sequential; that is why `pit`, `zrm` and `fastminv` are
  written as explicit accumulation loops.
- `with_lapack` (`DGETRF/DGETRI`) vs the Crout branch: measured on the Accelerate build,
  the Fortran-vs-Fortran difference is ≤ 2 ulp on `rm` and the slopes (`m` equal);
  `inverse="crout"` is bitwise equal to the Crout build, `inverse="numpy"`
  (`numpy.linalg.inv`) within 7 ulp of the LAPACK build, and `inverse=LapackInverse(lib)`
  (`pydyn/lapack.py`: the same `DGETRF` + `DGETRI(lwork = n)` calls through ctypes into the
  library the model was linked with) bitwise equal to it.
- Library calls are reproducible only through the same library: FFTW's `FFTW_ESTIMATE` plan
  for 60×45 is not numpy's pocketfft (1e-11 relative on the balanced fluxes), and pyfftw's
  bundled FFTW gives a third answer; `balance.FFTWPoisson` therefore dlopens libfftw3 itself
  (`$PYDYN_FFTW`, `find_library`, homebrew) and issues the Fortran's `dfftw_plan_dft_2d(im,
  jm)` / `dfftw_execute_dft` calls. Likewise `$PYDYN_LAPACK` for the inverse.
- NumPy's vectorised transcendentals depend on the CPU as well as the version: on x86-64 with
  AVX-512, `np.log`/`np.exp`/`np.cbrt`/`**` dispatch to SVML kernels that are not the C
  library's. The transcription never calls them (`ops.libm` is scalar), so the bitwise tier is
  unaffected; the simple-vs-fast tier does, and on Orion it moved `dkg` 782 ulp. `conftest.py`
  turns the AVX-512 dispatch off on x86-64 and prints what was dispatched (`PYD-8`).
- The Fortran limiter `max(min(s, abs(rm)), -abs(rm))` and `np.maximum/np.minimum` agree
  except in NaN propagation, which should not occur.

### 3d. Zoom nesting, in the standalone harness (`PYD-3`)

**Bitwise, tier 1 only.** `pydyn/zoom.py` against the production zoom routines compiled
unmodified in `fortran/harness_zoom{2,3}` (`make -C fortran zoom`), gfortran 16, `-O0
-ffp-contract=off`, arm64. No zoom run has been dumped or replayed yet: §3b's discipline does
not reach this, and the flags caveat of §3c applies unchanged.

What the zoom binaries compile, beside §3a's kernels: `proj/ct-unified/trunk/src/zoom_tools.F90`
(`mix_edges`, `update_parent`, `coarsen_region`), `branches/speedup/src/advect_tools__speedup.F90`
(`m2phlb`, `m2phlb1`, `coarsen_ambm`), `trunk/src/modelIntegration.F90` (`Proces_Region`,
`do_steps`), `trunk/src/advectm_cfl.F90` (`Init_CFL`, `Check_CFL`, `Setup_MassFlow`, and the
mass-only zoom path behind them) with `trunk/src/cfl_cache.F90`, and `trunk/src/datetime.F90`
(`new_valid_timestep`, which keeps `ndyn` a multiple of `2 max(tref)`). modelIntegration is
built as the transport-only dump sets were, with `without_chemistry without_emission
without_diffusion`, so `do_steps`' `c` and `s` cases are empty and `Proces_Init/Update` name
only stubs. There is one binary per region count, `NREGIONS = 2` and `3`: production's
`nregions` is the true count and `Check_CFL`, `store_masses` and `Setup_MassFlow` loop over
all of it -- a two-region case in a three-region binary crashed on the phantom region. The
stubs carry the region tree as run-time variables where production has `dims_grid`
parameters. `with_zoom` is
defined, as pycasso defines it for every build with more than one region, and it matters:
`convec` skips a parent's columns under a child only under that macro — without it the parent
convects its footprint too, and the first zoom convec comparison differed in exactly those
36 columns. Two things are not production behaviour: the reduced grid stays stubbed
(`redgridZoom.F90` needs the rc machinery), and `TimeInterpolation` is a no-op, so pu/pv are
held over a `Check_CFL` interval (`zoom.check_cfl_zoom` takes an `interpolate` callback for a
caller that varies them). The CFL cache is compiled but off, its netCDF layer (`MDF`) a stub
that stops if reached.

The Python keeps the Fortran halo bounds (`zoom.FView`, a writable `dump.FArray`): the exchange
is halo arithmetic, and a child's halos hold its parent's boundary cells. Region 1 is advected
by §3a's kernels on interior views; a child by `_dynam{u,v,w}_child`, which mirror dynamu /
dynamv / dynamw's non-periodic, non-polar paths. The region tree (`Zoom`) is
`determine_children_etc`, `blank_zoom_region` and `define_splitorderzoom` (zoom_mode 1),
computed in Python and handed to the Fortran; it is checked against the production rc
(`glb3x2` + `nam1x1` at parent cells 17:40 x 56:77) and against `dims_grid__eur.F90` and the
survey's footprints, not against a compiled `initexit.F90`.

**Covered**, `tests/test_zoom.py` (150 cases, 7 s): the eight exchange routines; `dynam0` and
`convec` per region; x, y and z at **every** slot of every region's `splitorderzoom` in a
two-level (child at nam1x1's factors) and a three-level tree (the grandchild's 48-operator
order), at Courant 0.4 and 3.5 -- children sub-cycling to nloop 7 in x and y and 8 in z, and
both sides refusing the same inadmissible cases; and Setup_MassFlow, one `Proces_Region(1)` and
a whole `ndyn` step on four trees: the two above, glb3x2 + nam1x1 at production size and the
eur tree. Every array of every region, halos included, and `nloop_max` / `xi` per region and
axis, bitwise.

**`Check_CFL` on a tree** (`zoom.check_cfl_zoom`, 8 cases, 15 s): the mass-only rehearsal
with advectm_cfl's own `xyzzyx` order in blocks of three, its mass-only exchange (the m
statements of the tracer routines, `tracers=False`), its caps (10, 1, 1) and its
`cfl_ok`-and-unwind recursion (`determine_cfl_iter`). Compared: the `(ndyn, n)` it settles on
and every array it leaves -- masses and fluxes restored and rescaled, and sp / phlb as the
rehearsal's `coarsen_region -> m2phlb` wrote them, which is the only window onto the
rehearsed masses the private module allows. The cases keep `ndyn`, reduce it once, reduce it
12 and 19 times (to 80 and 144 s), and refuse outright (no valid step) on both sides. The
caps are a parameter (`caps=`), as `cfl_cap_replay` needs.

**Found on the way.**
- Region 1 in a zoom run needs two things a single-region run never shows: dynamu's closing
  loop writes the periodic x halos of m, rm and the slopes -- which a child's `put_xedges`
  can read -- and dynamv scales and restores the *whole* `bm(:,:,l)` when a layer sub-cycles,
  which leaves region 1's halo faces an ulp off. `zoom.py` does both; kernels.py need not.
- `update_parent`'s dateline branch (`ibeg > iend`) indexes the parent with the *child's*
  `im`; `put_xedges` reads `ibeg-1` with no wrap where `put_yedges` wraps with `mod`. Both are
  latent -- no configuration in use has a child at the dateline or at the parent's first or
  last column -- and `zoom.Zoom` refuses such trees rather than transcribe them untested, as it
  does pole-touching and cyclic children.
- `tm5-zoom@13149b4a1:docs/dynamic_core_survey.md` has two statements the code contradicts,
  not corrected there from here: §2.5 says a direction's scope is restricted when the other
  directions precede it -- it is the reverse (advectxzoom restricts x in j when x comes before
  y); and §3's `SETUP_MASSFLOW` runs every `dynam0` and then `coarsen_region` from the last
  region down, where the code interleaves `dynam0(n); coarsen_region(n)` for n = 1..nregions
  and then `coarsen_ambm` for every region.

**Not covered**: the reduced parent together with a child, in the transport and in
`Check_CFL`; pu/pv varying within a `Check_CFL` interval; the meteo of a child (`Meteo_Setup_Mass`'s surface-pressure matching,
BalanceMassFluxes per parent cell); `convection__speedup.F90` on a child; `s` and `c` on a
child. `PYD-3` orders them.

### 3e. The ranks together equal the serial answer (`PYD-2`)

**Claim**, decided 2026-09-24: every array a multi-rank run dumps, with the ranks' tracer slabs
put back in rank order, is bitwise the array the same build dumps serially. **Holds** on the
`pyd2_*` sets (`runs/MANIFEST.md`): one `ch4_6x4_fwd.rc` run with four distinct tracers, 23 h,
serial and on 2, 3 and 5 Open MPI ranks -- slabs 2+2, 1+1+2 and 1+1+1+1+0 -- all from one clean
tm5-zoom clone at `d5e682986`, gfortran 16, `-O3 -march=native -ffp-contract=off`.
`tests/test_ranks.py` (12 cases, ~9 min, skipped without the sets):

- every array that is not split by tracer is byte-identical on every rank, in every event --
  `m`, the fluxes, the balanced meteo pairs, `dkg`, `Check_CFL`'s decision, the chemistry
  inputs and loss rates, `sp`/`phlb`;
- the merged run is the serial run, event for event and array for array, to the bit -- 900
  events, BalanceMassFluxes under MPI included;
- the merged sets replay bitwise, as the serial one does (all 441 replayable events).

What "together" means is the Fortran's, read from `mpi_comm.F90` and then confirmed by the
sets: ct-unified decomposes by tracer only (the `'levels'` swap is `base/trunk`'s driver, not
production's), and rank r holds the contiguous block after ranks 0..r-1. In the dumps that is
three things, and `dump.merge_ranks` (used by `load_events(..., merge=True)`, which `replay`
now calls) handles exactly those:

- `rm, rxm, rym, rzm` are the only split arrays (`TRACER_ARRAYS`), concatenated in rank order.
  They must be named: with equal slabs every rank's bounds are identical, so no size rule tells
  a split array from a replicated one (`loss(...,1:nreac)` has `nreac = ntracet = 4` here);
- `ch4_tracers` in the `chem` header is the rank's *local* tracer list (`1 2` on both ranks
  of the 2-rank run), made global with each rank's offset;
- `emis` is dumped once per tracer the rank holds, so the ranks' event counts differ and their
  `seq` drifts; each injection's `emis` blocks are interleaved by the global `i_tracer`, the
  order the serial run writes them. `replay.sources` now injects at `i_tracer` rather than
  `itracer_loc` -- the same number serially, the right one in a merged state.

**Not covered**: ifort (`PYD-1`); IO-server ranks (production runs 2 beside its compute
ranks; these sets have none); OpenMP; zoom under MPI; and Orion's MPI. The MPI build drops
`with_hdf5_par with_netcdf4_par` for the laptop's serial netCDF -- file I/O only (`mdf.F90`,
`tm5_restart.F90`), and the serial build defines them inertly.

### 3f. Zoom, tier 2: a glb3x2 + nam1x1 run replayed per region (`PYD-3`)

**Claim**, 2026-09-29: the production zoom -- glb3×2 with its 14-band reduced grid and the
nam1×1 child -- run on Orion with ifort at Orion's production flags (`-O3 -xHost -fp-model
precise -no-fma`, 6 compute + 2 IO ranks, one CH4 tracer, transport only) is reproduced by
`pydyn/zoom.py` **per event and in a free run, bitwise, in every array including the halos**.
`pydyn/replay_zoom.py`, on the six hours of `zm_replay` (`runs/MANIFEST.md`), convec through
the model's own MKL:

| | per event | `--free-run` |
|---|--:|--:|
| x, y, z, v of the parent (20 each) | 80 / 80 | 80 / 80 |
| x, y, z, v of the child (40 each) | 160 / 160 | 160 / 160 |
| Setup_MassFlow, both regions (10 steps) | 20 / 20 | 20 / 20 |
| `update_parent` + m2phlb / m2phlb1 | -- | 40 / 40 |

The free run carries both regions' state from each region's first `bef_` event; the dumps are
checkpoints only (`in:<name>` on entering an operator, the result on leaving it). It therefore
also covers what no single event shows: the parent's `put_x/yedges` into the child, the
child's `update_parent` into the parent, and the `put_x/yedges` that `do_steps` adds at a block
end -- including blocks ended by the `s`/`c` no-ops of a transport-only build, which are never
dumped and are passed by following each region's split-order position from the header's
`status`. `phlb` is compared on the cells m2phlb writes: its halos are what initialisation
left (`at(l)` on region 1's x halos, from `mass_to_pressure`), m2phlb never rewrites them and
nothing reads them.

**What made it replayable** is a tm5-zoom change, `611fb1ca9` on branch `pydyn/zoom-dump`
(on `NOAA/ct-unified` as `84842782e` since 2026-10-07): the dump header now carries `status` (the region's position in
its split order: `splitorder[status]` is the operator of a `bef_`/`aft_` pair), `splitorder`,
the footprint `ibeg..lend` of a child and the region's `children`. A replay builds its
`zoom.Zoom` from the header and **refuses to run** unless the footprint, children and split
order it derives are the model's (`replay_zoom.build_tree` / `check_tree`). The change writes
only inside `open_event`, i.e. only under `pydyn.dump : T`; its control arm, local gfortran,
`ch4_6x4_fwd.rc` 00-06 h from clean clones of `13149b4a1` and `611fb1ca9`: dump off, the 06:00
restarts equal in all 13 variables and the region file byte-identical; dump on, all 3930 files'
arrays byte-identical and the `.meta` files different only by the three added keys.

**Found on the way, a transcription bug tier 1 could not see.** `redgridZoom.F90`'s `red2uni`
ends every reduced row with `q(0,j,l) = q(im,j,l)` on the *uniform* row, overwriting the
reduced-row value dynamu's closing loop wrote there. `zoom._dynamu_region1` kept the reduced
value: 476 halo cells (14 rows × 34 levels) off in every parent x event, nothing inside
`1..im`. The harness of §3d stubs the reduced grid, so only a real glb3×2 dump could show it;
fixed, and pinned without a dump by `test_zoom.py::test_reduced_parent_x_halo_is_the_uniform_row`
(verified to fail without the fix).

**`Check_CFL` with the child, 672 real intervals.** `analysis/cfl_cap_replay_zoom.py` runs
`zoom.check_cfl_zoom` over the `cfl` events of four week-long windows and reproduces the
model's `(ndyn, n)` on every one -- the reduced parent and the child together, at ifort. The
findings are in `cfl_headroom.md` §11.

**Every region's meteo, from the input files** (`PYD-3` item 3, 2026-10-13).
`replay_zoom` replays `Meteo_Setup_Mass` for both regions and, with `--meteo`, reads what it
balances from the archive the model read. Hercules, the model's MKL 2022.2.1 and libimf
(`runs/MANIFEST.md`):

| | per event | `--free-run` |
|---|--:|--:|
| `Meteo_Setup_Mass`, glb3×2 / nam1×1, `zm_replay` (6 reads each) | 6 / 6, 6 / 6 | 6 / 6, 6 / 6 |
| the same with Setup_MassFlow on the Python's balanced fluxes (10 steps each) | -- | 10 / 10, 10 / 10 |
| `Meteo_Setup_Mass` on the `cfl` events of `zm_2019-{01,04,07,10}` (168 reads each) | 672 / 672, 672 / 672 | -- |

A row counts an event bitwise only if every array is: the balanced `pu, pv, pw` of both
instants and, from the archive, the raw `mfu, mfv, mfw` of both, `sp1, sp2`, and at the cold
start `sp` and `m`. Three things make it so:

- *The child's balance* (`zoom_meteo.py`). Its raw fluxes are matched 'sum' to its parent's
  **balanced** ones on every parent face, then balanced per parent cell: each 3×2 block gets
  the Poisson correction of its own residual from `EnsurePoissonCache`'s direct matrices (the
  FFT solve of each unit vector, through the model's FFTW interface), which vanishes on the
  block's faces, so the Match survives. Region 1's parent is the globe, so it is §3b's
  balance unchanged. Its surface pressure is matched 'area-aver' to the parent's.
- *The ingestion* (`regrid.py`). The zoom rc does **not** read the coarsened `tropo34/glb300x200`
  and `nam100x100` sets: `tmm.sourcekey.*.ml` is the 1×1, 137-level ERA5 set, and tmm
  regrids it in memory. FillGrid combines 3×2 cells for the parent ('u' faces summed over
  rows, 'v' over columns, `mfw` over both, `sp` area-averaged) and copies a window for the
  child; levi_FillLevels_3d sums each tropo34 layer's ml137 levels, `reverse` because the
  file is bottom-up and the parent table top-down, so in descending file order, and copies
  `mfw`'s half levels. The coarsened sets hold the same fields rounded to float32: read
  instead, every raw flux is 2^28 ulp off, which is how the path was found.
- *The `cfl` events* of the census sets carry the same meteo arrays as `meteo` events
  (`pydyn_dump_cfl`'s put_meteo): on `zm_replay` 186 of 192 arrays are identical; the six
  that differ are region 1's `sp`, which Check_CFL recomputes from the restored mass, so it is
  not compared there.

The comparison can fail: the same events replayed on the laptop (numpy's or Homebrew's FFT
for the 3×2 direct matrices) are 512–32768 ulp off inside the parent cells and bitwise on
their faces, and the first ingestion, from the coarsened sets, failed every raw array.

**Emission, chemistry and convection on a child, and the convection's inputs from the
archive** (`PYD-3` item 4, 2026-10-13). Two Hercules runs of the zoom with its physics on
(`pyd4-phys.rc` = `ch4_3x2nam1x1_pydyn.rc` without `without_emission`/`without_chemistry`;
the CT-CH4 defaults), 6 h of 2019-01-01, every tag, Orion's flags: `zp_replay` (MPI, 6 compute
+ 2 IO ranks) and `zps_replay` (serial). `replay_zoom --meteo`, the model's MKL and libimf:

| | `zps_replay` per event | `zps_replay --free-run` | `zp_replay` per event |
|---|--:|--:|--:|
| s, glb3×2 / nam1×1 | 20 / 20, 40 / 40 | 20 / 20, 40 / 40 | 20 / 20, 40 / 40 |
| c | 20 / 20, 40 / 40 | 20 / 20, 40 / 40 | 20 / 20, **0 / 40** |
| v, convection inputs from the archive | 20 / 20, 40 / 40 | 20 / 20, 40 / 40 | 20 / 20, 40 / 40 |
| x, y, z | 60 / 60, 120 / 120 | 60 / 60, 120 / 120 | 60 / 60, 120 / 120 |
| Meteo_Setup_Mass, Setup_MassFlow on its own meteo, m2phlb | 6 / 6, 6 / 6 and 10 / 10, 10 / 10 | 6, 6; 10, 10; 20, 20 of as many | as `zm_replay` |

- *s and c on a zoom tree* (`zoom.sources_region`, `zoom.chemistry_region`, `zoom.owned`).
  do_add_2d and ch4_chemistry_step act on the interior scope isr:ier × jsr:jer and skip the
  cells a region does not own (`zoomed /= region`: a parent's cells under its child, a child's
  edge ring); the chemistry computes its loss rates on the scope only (a chem dump's `loss` is
  the scope's) and leaves the rest at decay 1.0. convec already worked so (§3f, item 2).
- *The convection's inputs* (`regrid.py`, `ZoomIngestion.convection`). eu..dd are the 1×1,
  137-level set area-averaged onto the region and summed per tropo34 layer, then
  ConvUnitConvert and the cloud levels as for glb6×4; T and Q (for Calc_Kzz and the chemistry)
  are 'mass-aver' both ways -- each source cell weighted by its own `(da + db ps) area`, `ps`
  the record's, then each layer by `|dp|` on the model grid's `ps` -- and Q is clipped at 0.
  Bitwise on both regions, every event; the zm_replay free run convects with them bitwise too.
- *Meteo_Setup_Other's interval.* The temperature and Calc_Kzz's fields are interpolated over
  the dynamics step (tdyn, tdyn + ndyn), not the read interval: they differ when Check_CFL
  cuts the step, which no glb6×4 set did. `replay.Ingestion` had the read interval; it now takes
  the step's (the glb6×4 results are unchanged, their steps are their read intervals).
- *A parent's first half levels.* The run's start ends with update_parent, whose last act is
  the parent's m2phlb1: before its first `m2phlb` event region 1's `phlb` is that of its
  coupled mass, not Pressure_to_Mass's (a child's is). The free run seeds it so.
- *dkg.* Calc_Kzz on the archive is bitwise on both regions, zero on the child's edge ring as
  in the model, once three things match it: `ops.libm.cube` (§3c); `spm` from the read
  interval's `sp1/sp2` weighted to the step's middle (meteo_server's alfa1/alfa2), not the
  step's ends; the step's starting `sp` for the geopotential, a parent's m2phlb1 of its mass
  after Setup_MassFlow's coarsen_region (`replay_zoom.step_inputs`). It is tallied as `in:dkg`,
  and the free run convects with it: zps_replay and zm_replay free runs bitwise, `in:dkg` 20/20
  and 40/40.

**A model bug, found here: an MPI build's zoom child decays CH4 with misplaced loss rates.**
ch4_chemistry_step (`chem_unified/src/chemistry_ch4.F90`) binds the per-node shared loss array
with `c_f_pointer(..., ch4_loss, (/n_lon, n_lat, n_levels, nreac/))` -- lower bounds 1 -- and
then reads `ch4_loss(i_lon, j_lat, k, r)` with `i_lon = isr..ier`, `j_lat = jsr..jer`. The
serial build allocates it `(lon_start:lon_end, lat_start:lat_end, ...)`. For region 1
(isr = jsr = 1) they agree; nam1×1 (isr = 4, jsr = 3) decays each cell with the losses of the
cell 3 east and 2 north, wrapping into the next row and level at the ends, and 135 of its
cells read past the array. `analysis/ch4_loss_bounds.py` replays every chemistry event both
ways from the dumped losses:

| set | region 1: intended / as bound | nam1×1: intended / as bound |
|---|---|---|
| `zp_replay` (MPI) | bitwise / bitwise | ≤ 1.2 % of values / **358 500 of 358 500** (135 cells past the array not compared) |
| `zps_replay` (serial) | bitwise / bitwise | **359 040 of 359 040** / ≤ 1.2 % |
| `zpfix_replay` (MPI, fixed) | bitwise / bitwise | **359 040 of 359 040** / ≤ 1.2 % |

So every MPI production run with a nam1×1 child has had its child chemistry displaced by
(3°, 2°) since the shared window was introduced (tm5-zoom `57933d3ca`, 2026-03-26). pydyn
transcribes the serial semantics; the MPI child's `c` fails per event for this reason alone.

**The fix** remaps the pointer to the serial bounds,
`ch4_loss(lon_start:, lat_start:, 1:, 1:) => loss_flat` after `c_f_pointer` into `loss_flat`
(tm5-zoom `0385efe7e`; `runs/MANIFEST.md`, `pyd5`). `zpfix_replay`, the `zp_replay` run rebuilt with it, equals the
serial `zps_replay` bitwise in every tracer array of all 720 operator events (`zp_replay`
differs from the child's first `aft_c` on), and replays bitwise per event and in the free run,
every operator of both regions. `ch4_loss_bounds.py --expect fixed` is its check.

**Not covered**: the inputs of s and c on a zoom (the 1×1 emission onto each region, the
chemistry's climatologies and soil map, the initial field -- `PYD-3` item 5); the three-level
eur tree outside tier 1; zoom under OpenMP.

## 4. The constants mirror

A third discipline, and not a bitwise one. §2 and §3 compare arrays — same inputs, same
outputs — which leaves two blind spots. A constant that only decides behaviour at the margin
is invisible when no run reaches the margin: `Check_CFL` picks an `ndyn` at which y and z
never sub-cycle, so `cfl_max_nloop_{y,z}` could hold any value and every array test would
still pass. And the physical constants that *are* covered are only covered where 8.3 GB of
dump sets is on disk.

`tests/test_constants.py` reads the declarations out of the checkout at `$PYDYN_TM5ZOOM` and
compares them with the Python's: 77 cases, 0.2 s, every module-level numeric constant in the
package. Keyed by the Python name and applied to every module that defines it, so two
modules disagreeing about `GRAV` is caught as well as either drifting from `binas`.
Declarations that are expressions over other parameters (`fak = fr_excess * 8.5`,
`rair = Rgas*1.e3/xmair`, `zcrdq = 1.0/epsilo - 1.0`) are resolved recursively across files,
so a change to a leaf surfaces in every value derived from it. Literals that are named on
neither side (`krate_oh = 2.45e-12 exp(-1775.0/T)`) are compared as the real literals of
that statement. A new module-level numeric with no row and no waiver fails the suite;
Fortran limits with no Python counterpart — `nredmax` is the one — are recorded in
`NO_COUNTERPART` rather than left unsaid.

It exists because `cfl_max_nloop_{y,z}` went 6, 10 → 1, 1 in `6f59f9f72`, inside a commit
whose subject is an `MPI_Allreduce` fix, and nothing here would have noticed
(`cfl_headroom.md`). It runs only where a checkout is present, so it protects a developer,
not CI.

---

> **§3e as the head had it at pydyn `ec6485f`, verbatim** (rewritten into a table the same day).

### 3e. The ranks together equal the serial answer (`PYD-2`)

**Claim**: every array a multi-rank run dumps, with the ranks' tracer slabs put back in rank
order, is bitwise the array the same build dumps serially. **Holds** on the `pyd2_*` sets
(`runs/MANIFEST.md`): one `ch4_6x4_fwd.rc` run with four distinct tracers, 23 h, serial and on 2,
3 and 5 Open MPI ranks — slabs 2+2, 1+1+2 and 1+1+1+1+0 — from one clean tm5-zoom clone at
`d5e682986`, gfortran 16, `-O3 -march=native -ffp-contract=off`; and on the `pyd2i_*` sets, the
same four tracers for 6 h at `745e94f70` under ifort 2024.1 at Orion's flags, serial and on 2, 3
and 5 compute ranks each beside 2 IO-server ranks (Hercules, `runs/hpc/ranks.slurm`, the model's
MKL and libimf); and in a zoom tree on the `zp4_*` sets, `zpfix_replay`'s glb3x2 + nam1x1 run
(emission, chemistry, the ch4_loss fix) with the same four tracers, serial and on the same three
layouts. `tests/test_ranks.py` (4 cases per set; ~9 min laptop, 19 min Hercules for the three
`pyd2i` sets, 53-100 min and up to 190 GB for each `zp4` set):

- every array not split by tracer is byte-identical on every rank, in every event — `m`, the
  fluxes, the balanced meteo pairs, `dkg`, `Check_CFL`'s decision, the chemistry inputs and loss
  rates, `sp`/`phlb`;
- the merged run is the serial run, event for event and array for array — 900 events,
  BalanceMassFluxes under MPI included;
- the merged sets replay bitwise, as the serial one does (all 441 replayable events).

ct-unified decomposes by tracer only (the `'levels'` swap is `base/trunk`'s driver), rank r
holding the contiguous block after ranks 0..r-1 (`mpi_comm.F90`). `dump.merge_ranks` (called by
`load_events(..., merge=True)`) handles the three things that means in the dumps:

- `rm, rxm, rym, rzm` are the only split arrays (`TRACER_ARRAYS`), concatenated in rank order.
  They must be named: with equal slabs every rank's bounds are identical, so no size rule tells a
  split array from a replicated one;
- `ch4_tracers` in the `chem` header is the rank's *local* tracer list, made global with each
  rank's offset;
- `emis` is dumped once per tracer the rank holds, so the ranks' event counts and `seq` differ;
  each injection's `emis` blocks are interleaved by the global `i_tracer`, the serial order.

| set | ranks, slabs | events per rank | merged | merged replays |
|---|---|---|---|---|
| `pyd2_mpi{2,3,5}` (gfortran) | 2+2, 1+1+2, 1+1+1+1+0 | 808; 762, 762, 808; 762 ×4, 716 | 900 = `pyd2_serial` | bitwise, 441 events |
| `pyd2i_mpi{2,3,5}` (ifort) | the same, + 2 IO-server ranks | 210; 198, 198, 210; 198 ×4, 186 | 234 = `pyd2i_serial` | bitwise |
| `zp4_mpi{2,3,5}` (ifort, zoom) | the same, + 2 IO-server ranks | 984; 924, 924, 984; 924 ×4, 864 | 1104 = `zp4_serial` | bitwise |

`zp4_serial` itself replays bitwise per event and in a free run on `--meteo --chemistry
--initial` (its emission is EnKF-scaled per member, so not `--emissions`): every operator of both
regions, `Check_CFL`, Setup_MassFlow, m2phlb and the initial field per region — the zoom engine
with more than one tracer. A first `zp4` attempt ran with the zoom rc's
`parameters.constant.value : 1.0`, which scales every member alike: four identical tracers, and
`test_tracers_are_split` failed on all three sets, as it exists to.

**Not covered**: OpenMP. The laptop MPI build drops `with_hdf5_par with_netcdf4_par` for its
serial netCDF, file I/O only.
