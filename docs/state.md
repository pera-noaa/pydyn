<!-- author: Ash Pera (NOAA/GML CIRES) -->
<!-- footer-title: pydyn — state -->

# pydyn: what is transcribed, what is bitwise, and how to reproduce it

The head of this repo; `README.md` is the status page. Model-side pin: `tm5-zoom@745e94f70`
on `NOAA/ct-unified` (`docs/sources.md`: which rows were read at which sha, and how the
re-pin was tested). Paths that start with `base/`, `proj/` or `docs/` are in that repo;
everything else is in this one. Dated measurements, job IDs and the order things were found
in: [`state.history.md`](state.history.md).

## 0. Status

**Scope.** The production build variant — `slopes` advection, `without_wet_deposition`, no
secmom, no OpenMP — under the meteo-server driver (§3b): one region, a multi-rank run by
merging its ranks (§3e), and a zoom tree (§3d, §3f). Everything the dynamic step does — the
three advection kernels and `dynam0`, convection, the reduced grid, `Check_CFL`,
`Meteo_Setup_Mass` and `m2phlb1`, the zoom exchange — and the two physics operators of the
CH4 run, the surface emission (`s`) and the CH4 chemistry (`c`), with every input the run
reads from files: the meteo, `Calc_Kzz`, the emission field, the chemistry's fields and the
initial tracer field. Algorithm: `tm5-zoom@4dc2017c5:docs/dynamic_core_survey.md`.

**Bitwise, and the flags are part of the claim.**

| § | what is compared | build |
|---|---|---|
| §3a | the production kernels, compiled unmodified against stubs and driven from Python | gfortran 16 `-O0 -ffp-contract=off`, arm64; ifort at Orion's flags (§3c) |
| §3b | a `ch4_6x4_fwd.rc` day, glb6×4: every operator call, the loss rates, Setup_MassFlow, `Check_CFL`'s `(ndyn, n)`, the balance and m2phlb, per event and in a free run that reads nothing but the model's input files | gfortran `-O3 -march=native -ffp-contract=off`; ifort at Orion's flags with the model's MKL and libimf (§3c) |
| §3e | a 2-, 3- and 5-rank run, its tracer slabs merged, against the serial run: every array of every event, on glb6×4 and on the glb3×2 + nam1×1 tree | as §3b, Open MPI; ifort 2024.1 at Orion's flags with 2 IO-server ranks, the model's MKL and libimf |
| §3d | the zoom exchange, a child's advection at every slot of its split order, a whole step and `Check_CFL` on two- and three-level trees, against the compiled zoom routines | as §3a, `with_zoom`, no reduced grid |
| §3f | a glb3×2 + nam1×1 run: every operator of both regions, both regions' `Meteo_Setup_Mass`, Setup_MassFlow, m2phlb and `update_parent`, per event and in a free run that reads nothing but the model's input files; the MPI run equal to the serial one; `Meteo_Setup_Mass` and `Check_CFL` on 672 more intervals | ifort 2021.12 / 2024.1 at Orion's flags, the model's MKL and libimf |

Two conditions on that word: the build must not contract (`-ffp-contract=off`, or ifort's
`-fp-model precise -no-fma`), and the FFT, LAPACK and libm calls are bitwise only through the
model's own libraries, which the Python calls through ctypes (§3c). Constants are checked on
their own terms (§4). `runs/MANIFEST.md` has a row for every dump set. `pydyn.gpu`, the same
core with the state on a GPU, is held bitwise to this reference (float64: `emu`, `cuda`) or to
its float32 host twin (Metal on Apple silicon, 15.6 simulated years an hour for a glb3x2 +
nam1x1 interval); `pydyn/forward.py` runs from the input files alone, bitwise these dumps on
the CPU: `docs/gpu_port.md` §9.

**Open.**

- Hercules' own flags (bare `-O3`, `-fp-model fast`) are a few ulp off in the kernels and
  7e-10 in the Poisson balance (§3c); every Hercules result here is built at Orion's flags
- not transcribed: the isotope and `with_budgets` chemistry (`PYD-5`), the grouped (4DVAR-save)
  initial field and one needing vertical interpolation, and the non-default rc paths of
  `sources.md` §7
- not tested: OpenMP, and the three-level tree outside §3d

```
python3 -m pytest                      # 488: 27 device + 29 metal + 19 property + 30 parity + 24 simple + 41 unit + 10 production + 12 ranks + 9 inputs + 159 zoom + 14 zoom meteo + 5 zoom physics + 9 zoom census + 100 constant
make -C fortran                        # builds the §3a harness; parity tests skip without it
make -C fortran zoom                   # builds §3d's harness_zoom2/3; the zoom parity tests skip without them
                                       # (kernels from ../../tm5-zoom; TM5ZOOM=<path> overrides)
python3 -m pydyn.replay <dump_dir>     # §3b: replay a pydyn_dump directory of a model run
python3 -m pydyn.replay <dump_dir> --free-run --meteo <my.meteo.dir> --emissions <emis.ch4.input.dir> \
        --chemistry <input/ch4c13> --initial <istart.66.filename>   # … on nothing but the input files
python3 -m pydyn.replay <dump_dir> --rank 0 --meteo-grid 1x1 …   # §3f: a zoom set, one rank's events, the 1x1 meteo
python3 -m pydyn.replay <dump_dir> --workers 6 …   # per event in 6 processes, a run of read intervals each
```

| file (this repo) | Fortran source, as `proj/ct-unified/rc/__zoom_1day.rc` resolves it (read at `tm5-zoom@4dc2017c5` unless `sources.md` says otherwise) |
|---|---|
| `pydyn/kernels.py` `fluxes_from_meteo` | `base/trunk/src/advect.F90` dynam0 |
| `pydyn/kernels.py` `advect_x/_y` | `proj/ct-unified/trunk/src/advectx__slopes.F90`, `advecty__slopes.F90` dynamu/dynamv |
| `pydyn/kernels.py` `advect_z` | `proj/ct-unified/branches/speedup/src/advectz__slopes.F90` dynamw |
| `pydyn/kernels.py` `convection_matrix`, `fastminv`, `convec` | `base/trunk/src/convection.F90` convec, fastminv (see §3a for why not `__speedup`) |
| `pydyn/simple.py` | the same core written for reading: one generic `advect_line` for x/y/z, `column_matrix`, `np.linalg.inv`; agrees with `kernels.py` to 1e-12 (`tests/test_simple.py`), fluxes left untouched, `xi` = largest Courant number actually used |
| `pydyn/driver.py` `Model.step` | `modelIntegration.F90` do_steps: `xyzvsc` then `csvzyx`, dt = ndyn/2. `Model(grid=…)` routes x through `redgrid.advect_x_reduced`. A single region outside the zoom driver, for the kernel tests and `analysis/step_accuracy.py` (any operator sequence) |
| `pydyn/ops.py` | the array idioms the kernels are written in — `seqsum`, `advance`, `upwind`, `courant`, `convergence`, `clusters`, `wrap`/`closed` halos — each a named Fortran statement evaluated in the same order; `subcycles`, the CFL sub-cycle search all six advection operators share (`progressive` vs `fresh` flux shrinking: dynamu/v vs dynamw); `libm`, the C library's (or ifort's libimf's) exp/log/pow/cos/sin and integer power on every value, through a small C loop that calls the library's own function (compiled on first use; without a compiler, one Python call per value). The docstring says where `np.diff` is and is not bitwise-safe |
| `pydyn/redgrid.py` | `proj/ct-unified/trunk/src/redgridZoom.F90` uni2red/red2uni/uni2red_mf and the reduced-grid wrapper of `advectx_work` |
| `pydyn/cfl.py` | `proj/ct-unified/trunk/src/advectm_cfl.F90` Check_CFL, dynamum/dynamvm/dynamwm, new_valid_timestep |
| `pydyn/balance.py` | `meteo.F90` Meteo_Setup_Mass, FillMassChange, CheckMassBalance; `grid_type_ll__speedup.F90` BalanceMassFluxes_m / SolvePoissonFFTW; `go_date` InterpolFractions; `advect_tools` m2phlb1 |
| `pydyn/lapack.py` | `convection__speedup.F90` `fastminv` (`with_lapack`: DGETRF + DGETRI) through the model's LAPACK |
| `pydyn/emis.py` | `emission_data.F90` do_add_2d and the month's field it injects: `emission_ch4_tracer.F90` emission_ch4_calc (`emission_units.F90` EU_MICROMOL, `emission_data.F90` flux_to_gridbox_per_month with `geometry.F90` calc_dxy), `emission_ct.F90` emission_ct_calc's one-member sum at the constant scaling 1.0, and `toolbox.F90` coarsen_emission_2d = FillGrid 'sum' onto each region's window of the 1×1 grid ('copy' for a 1×1 region) |
| `pydyn/chem.py` | `proj/ct-unified/branches/chem_unified/src/chemistry_ch4.F90` ch4_loss_kernel / ch4_chemistry_step (CH4-only path) and `emission_regrid.F90` rebin_levels, vectorised over the overlapping layers only (`rebin_many`: the three zonal columns share them) |
| `pydyn/chemfields.py` | the chemistry's inputs from their files, on any region's grid: `read_OH_field.F90` read_CH4_loss_rate / interpolate_lat (Bruehl), Import_OH_Spivakovsky (the `echlevs` clumps, then FillGrid 'area-aver', or 'copy' for a 1×1 region), Read_trop_Cl_Hossaini with `emission_regrid.F90`'s interpolation maps; `chemistry_ch4.F90`'s soil map (qianlai, one file per region); `dims_levels__ml137_tropo34.F90` echlevs |
| `pydyn/initial.py` | the initial tracer field, `istart 66`: `sources_sinks__ctuni.F90` start_from_field_fill3d (flat format), `grid_3d.F90` Fill3D 'mass-aver', `grid_type_ll__speedup.F90` Relate and FillGrid 'area-aver' as 'copy', 'combine' or 'distribute', `grid_type_hyb.F90` FillLevels (the copy case); `chem_param.F90` fscale |
| `pydyn/meteo.py` | the meteo ingestion: `tmm_mf_tm5_nc.F90` GetTimeRecordNumber / mf_ReadRecord and the udunits factor, `meteo.F90` SetupSetup `interpN`/`averN`/`month`, `meteodata.F90` TimeInterpolation, `grid_type_ll.F90` Init / Match 'area-aver' / FillGrid 'area-aver' of the 1×1 surface fields, `geometry.F90` dxyp, Pressure_to_Mass, ConvUnitConvert / ConvCloudDim; `Meteo` drives them per read interval, `Meteo.diffusion` composing Calc_Kzz with Meteo_Setup_Other's step interval |
| `pydyn/diffusion.py` | `diffusion.F90` calc_kzz, `diffusion_kernels.F90` dd_calc_ustar and bldiff (`jqif` as 0/1 masks), `meteo.F90` Meteo_Produce_Gph; arithmetic on (im, jm[, l]) arrays, no I/O |
| `pydyn/simple_diffusion.py` | Calc_Kzz written for reading; agrees with `diffusion.py` to a few ulp (2e-13 relative at worst on the production day, from the ill-conditioned boundary-layer height of convective columns) |
| `pydyn/dump.py` | reader for the `pydyn_dump` events (`FArray`, writable `FView`), arrays read on first use and dropped with `release`, and `merge_ranks` for a multi-rank run (§3e) |
| `pydyn/replay.py` | the replay of a dump set, one region or a tree (a single region is a tree of one): per event and `FreeRun` (§3b, §3f); per event in `--workers` processes, each a run of read intervals (a range starting at a region-1 read replays alone as in the whole); a thread reads the next events' arrays ahead and those 32 behind are dropped; `replay_zoom.py` is its former name, kept for the command lines on record |
| `pydyn/inputs.py` | the inputs read from the files: `RegionInputs` per region (meteo, emission, loss fields, initial field) and `Ingestion` over a tree (§3b, §3f) |
| `pydyn/grid.py` | TM5's lon/lat grids and FillGrid between them: `LL` (lli), `relate` (Relate: copy / combine / distribute), `block_sum` / `block_mean` (FillGrid's loops), `fill`; `grid_type_ll__speedup.F90`, `grid_tools` ll_area |
| `pydyn/runtime.py` | the FFT, LAPACK and libm a run calls (`$PYDYN_FFTW`, `$PYDYN_LAPACK`, `$PYDYN_LIBM`, the command line) and the options the replays share; `describe()` logs them |
| `pydyn/zoom/` | the zoom tree and exchange (§3d), a package -- `tree`, `state`, `exchange`, `massflow`, `physics`, `advect`, `steps`, `check_cfl`, `host` (the driver runs on any `ZoomState`, the bodies through its `lib`: `host` on NumPy, `gpu.zoom` on a device): `initexit.F90` determine_children_etc / blank_zoom_region / define_splitorderzoom (`Zoom`); `advect{x,y}__slopes.F90` advectxzoom / advectyzoom, put_/compress_/uncompress_{x,y}edges, a child's dynamu / dynamv; `advectz__slopes.F90` (speedup) advectzzoom, a child's dynamw; `zoom_tools.F90` mix_edges, update_parent, coarsen_region; `advect_tools__speedup.F90` m2phlb, m2phlb1, coarsen_ambm; `advectm_cfl.F90` Setup_MassFlow's region loop and Check_CFL's zoom path; `advect.F90` dynam0 and `convection.F90` convec of any region; `modelIntegration.F90` do_steps, Proces_Region and `tm5.F90`'s two half steps; do_add_2d and ch4_chemistry_step on the cells a region owns |
| `pydyn/zoom_meteo.py` | Meteo_Setup_Mass for a child (§3f): `meteo__meteo_server.F90` (`n > 1`), `grid_type_ll__speedup.F90` GetRefinement, Match_cell / Match_u / Match_v, BalanceMassFluxes_m per parent cell, EnsurePoissonCache / SolvePoissonDirect |
| `pydyn/regrid.py` | tmm's regridding of a record onto a region (§3f): `tmm.F90` ReadField / Transform3Dh / Transform3Dv, `tmm_param.F90` GetCombineKeys, FillGrid 'copy' / 'combine' / 'weight', `grid_type_hyb.F90` levi_FillLevels_3d 'sum' / 'mass-aver' / 'w' |
| `pydyn/layout.py`, `synth.py`, `harness.py` | the arrays' Fortran bounds and pydyn interiors (`DECLARED`, `INTERIOR`) and the conversions and stream I/O derived from them; seeded admissible test inputs; write a case, run `fortran/harness`, ulp-level compare |
| `pydyn/gpu/` | the device port (`docs/gpu_port.md`): `csrc/` the kernels, one source for CUDA (`--fmad=false`), Metal (float32, `xcrun metal -ffp-contract=off`) and the host (`-ffp-contract=off`, the `emu` / `emu32` backends); `kernels.py`, `diffusion.py` the reference's functions on device views, `zoom.py` zoom.host's bodies on them and the speculation (`Pipeline`); `exchange.py` / `boxes.py` the driver's statements as box programs; `metal.py`, `convec_gen.py` the Metal backend; `engine.py` the replay's `--device`; `check.py` the fingerprint match of two replays; `bench.py` an interval's throughput; `backend.py` the transfer ledger |
| `fortran/` | stub modules, the two drivers (`harness`, §3a; `harness_zoom`, built as `harness_zoom2`/`3`, §3d), Makefile; compiles the sources above unmodified and in place from `$(TM5ZOOM)` |
| `analysis/` | scripts that use the reference to answer a question (README's layout table); `ch4_loss_bounds.py` is §3f's |

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
from pydyn import Model, State

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
pydyn.dump.tags        : <comma list>       ! default: every tag
```

Hooks: `modelIntegration.F90` (`bef_x/aft_x`, `y`, `z`, `v`, `s`, `c` around every operator
call of `do_steps`; `m2phlb` after each `proces_region`), `tm5__meteo_server.F90` and
`initexit.F90` (`meteo` after Meteo_Setup_Mass, `cfl` around Check_CFL, `massflow` after
Setup_MassFlow), `emission_data.F90` (`emis`: one event per `do_add_2d` call, the field and
every factor) and `chemistry_ch4.F90` (`chem`, through the module's generic
`pydyn_dump_begin/key/put/end`, so `pydyn_dump` depends on no chemistry overlay: the loss-rate
inputs, the scale factors, `dt_sec`, `ch4_tracers` and the `loss(im,jm,lm,nreac)` the kernel
produced). Every event is `e<seq>_<tag>_r<region>_p<rank>.meta` (the header: dates, timing,
grid, limits, reduced grid, the zoom header of §3f; `array NAME = lb ub ...` per array) plus
one raw stream `.bin` per array **with its Fortran halo bounds** — `m(-1:im+2,-1:jm+2,lm)`,
`rm..(…,nt)`, `am/bm/cm(0:im+1,0:jm+1,0:lm+1)`, the meteo `pu/pv/mfu/mfv(0:im+1,0:jm+1,0:lm)`,
`mfw(1:im,1:jm,0:lm)`, `sp1/sp2(-1:im+2,-1:jm+2,1)`, `entu..dkg(im,jm,lmax_conv)`, the cloud
levels as `iarray`. `pydyn.dump.Event` reads them lazily; `ev["am"][0:im, 1:jm, 1:lm]` is the
Fortran section, `ev.interior("am")` the pydyn layout.

**The reference is defined against the meteo-server driver, `tm5__meteo_server.F90`
(`PYD-6`).** Every CH4 rc with a source list builds it. `base/trunk/src/tm5.F90` is not hooked:
the builds that take it are the `co2rc_*` family and five `drift_*` rc files (118 of the 594
ct-unified rc files with a source list, at tm5-zoom `c133fb437`). Such a build still writes the
operator, `m2phlb`, `emis` and `chem` events and the first interval's `meteo` and `cfl`, but no
`massflow` and no later `meteo` or `cfl`: its operator events replay per event, while the
ingestion check and `--free-run` lack their inputs.

`python3 -m pydyn.replay <dir> [--tags x,y,z,v,s,c,meteo,cfl,massflow,m2phlb] [--fft auto|fftw|numpy]
[--lapack auto|lapack|numpy] [--libm <libimf.so>] [--free-run] [--meteo …] [--emissions …]
[--chemistry …] [--initial …] [-v]` replays every event from the Fortran's own inputs, so a
difference cannot propagate to the next event, and prints a per-array table:

| event | Python | compared |
|---|---|---|
| `bef_x → aft_x` | `zoom.advect_x_zoom`: region 1 on `redgrid.advect_x_reduced` / `kernels.advect_x` | `m rm rxm rym rzm am bm cm`, halos included |
| `bef_y → aft_y`, `bef_z → aft_z` | `zoom.advect_{y,z}_zoom`: region 1 on `kernels.advect_y` / `advect_z` | the same |
| `bef_v → aft_v` | `zoom.convec_region` (`kernels.convec`, `inverse=lapack.LapackInverse`) | the same |
| `bef_s → [emis…] → aft_s` | `zoom.sources_region`: `emis.do_add_2d` for each `emis` event in between | the same |
| `bef_c → chem → aft_c` | `chem.loss_rates` on the `chem` event, then `zoom.chemistry_region`'s decay | `loss`, then the same |
| `massflow` | `balance.time_interpolation` of the data pairs at the step mid-point, then `zoom.setup_massflow` | `pu pv am bm cm` |
| `cfl` | `zoom.check_cfl_zoom` (region 1's mass kernels are `cfl.py`'s) | `ndyn n` |
| `meteo` | `balance.setup_mass` on `mfu/mfv/mfw` (pre-check, BalanceMassFluxes, CheckMassBalance) | `pu1 pv1 pw1 pu2 pv2 pw2` |
| `m2phlb` | `zoom.m2phlb1` | `sp phlb` |

`--free-run` (`replay.FreeRun`) is the opposite discipline: the state is taken from the first
`bef_*` dump and never again; the balanced pairs come from the Python's own `meteo` events
(keyed by the data instants `pu_t1/pu_t2`, not the read interval), `ndyn` from its own `cfl`,
`am/bm/cm` from its own `massflow`, `phlb` for the chemistry from its own `m2phlb`, and each
dump is only a checkpoint — `in:<name>` the carried state on *entering* an operator (which
would catch anything the model does between operators), the plain names the result. `seeded`
in the notes marks the events before the first dynamic step that still had to use the dump
(cfl, the first `x`, the first `c`); with `--meteo` and `--initial` there are none.

The inputs the model reads from files are taken from the dumps unless read from the files
themselves (`inputs.Ingestion`), each compared with the dumped ones as `in:<name>` and, in a
free run, used. With all four the free run takes no field from a dump.

| option | what is read, from where (the rc key) | how it reaches the model grid |
|---|---|---|
| `--meteo <root>` | the raw `mfu/mfv/mfw` at the two archive instants, `sp1/sp2`, the initial `m`, `temper`, the convection fields and Calc_Kzz's surface fields, from the tm5-nc archive (`my.meteo.dir`; `meteo.SOURCES`) | the time rules and the surface-pressure match; the hourly 1×1 surface fields by FillGrid's area-weighted mean (`meteo.py`'s docstring lists the six rules) |
| (with `--meteo`) `dkg` | `diffusion.calc_kzz` on those, with the step's `pu/pv`, the step's starting surface pressure for the geopotential and `spm`, the read interval's `sp1/sp2` weighted to the step's middle; T, Q and the surface fields over Meteo_Setup_Other's step interval (tdyn, tdyn + ndyn) | compared on every `bef_v` (`in:dkg`) |
| `--emissions <dir>` | the month's `flux1x1_all_source.YYYYMM.nc` (`emis.ch4.input.dir`) | `raw · (xmch4 / 1e9)` × the 1×1 cell area of calc_dxy (its latitude accumulated cell by cell; as `j·dlat` it is thousands of ulp off) × the month's seconds, then FillGrid's block sums; a free run injects it with `month2dt` from the calendar. Only the unscaled forward field is transcribed (the `PYD-2` sets inject EnKF-scaled members and differ by design) |
| `--chemistry <root>` | under `input/ch4c13`: the three Bruehl files, the Spivakovsky OH and its month's geopotential, the Hossaini Cl, the qianlai soil map (`ch4loss.inputdir`, `oh.file`, `oh.gph.inputdir`, `trop.cl.filename`, `ch4.soil_sink.input.dir`) | Bruehl linearly in latitude (on an input latitude the later bracketing interval wins); OH's 137 levels clumped onto 34 by `echlevs`, thickness-weighted, then FillGrid 'area-aver'; Cl to molec/cm³, then the latitude and cyclic longitude maps, levels flipped; every value widened from float32 and read unmasked, as MDF does |
| `--initial <file>` | the `istart 66` field (`istart.66.filename`), ppm on its own grid | Fill3D 'mass-aver': FillGrid level by level, then FillLevels, a copy for a file on the model's 34 levels; `rm = f · 1e-6 · m / fscale` for every tracer, slopes zero, compared with the first `bef_*` as `ic:<name>` |

**Result**, on the `ch4_6x4_fwd.rc` sets made at the pin (glb6×4 60×45×34, one CH4 tracer,
`nred = 4`, `with_lapack`, FFTW Poisson solver, `oh.scale 0.901`, tropospheric Cl and the soil
sink on), gfortran `-O3 -march=native -ffp-contract=off`, 23 h of 2000-01-01:

| set | build | events | per event | `--free-run` |
|---|---|--:|---|---|
| `phys` | full physics | 759 | every event, every array (x incl. reduced rows and `nloop = 2`; v through DGETRF/DGETRI; c's 60×45×34×4 loss rates) | 391 / 391 checkpoints; the same on the input files alone (`--meteo --emissions --chemistry --initial`) |
| `free` | transport only (`without_emission without_chemistry`) | 483 | every event | 299 / 299 |
| `cfl` | transport only, `ndyn_max = 10800`, `interp3` | 22 | every event; Check_CFL rejects 10800 s and runs 5400 s, `n = 2` | 14 / 14 |
| `pyd2_serial` (§3e) | four tracers | 900 | every event | 394 / 394 on `--meteo --chemistry --initial` |

With numpy's FFT the balanced fluxes are within 1e-11 relative (not FFTW's `FFTW_ESTIMATE`
plan); with `numpy.linalg.inv` the convection tracers within 7 ulp (an LU solve, not DGETRI).
`tests/test_production.py` runs the replay on the sets when present and asserts exactly this.
y/z sub-cycling (`nloop > 1`) cannot be observed in the model: `Check_CFL` chooses `ndyn` so
that it never happens, by its own sub-cycle caps ([`cfl_headroom.md`](cfl_headroom.md)); only
§3a exercises it. Three transcription traps this tier caught and §3a could not: gfortran's
`sum` intrinsic is sequential (`uni2red`, `m2phlb1`, `CheckMassBalance`) where `np.sum` is
pairwise; `X / fac` on a complex array is componentwise; on the reduced grid `m` takes the
uniform-grid update while only the tracers are re-split.

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

- **The flags.** `optim-fast` is not one thing: Orion's is `-O3 -xHost -g -fp-model precise
  -no-fma` (`base/trunk/rc/compiler-ifort-orion.rc:62`), Hercules' bare `-O3`
  (`compiler-ifort-hercules.rc:60`), which leaves ifort's `-fp-model fast=1` on. §3a's 188
  bitwise parity tests against the kernels built five ways with ifort 2021.12 (`runs/MANIFEST.md`,
  `pyd1_*` jobs):

  | arm | flags | passed |
  |---|---|---|
  | O0 | `-O0 -fp-model strict` (the Makefile's ifort default) | 188 / 188 |
  | O3 | `-O3 -fp-model strict` | 188 / 188 |
  | Orion production | `-O3 -xHost -fp-model precise -no-fma` | 188 / 188 |
  | Hercules production | `-O3` (fast=1, generic SSE2 target) | 94 / 188 |
  | fast + FMA | `-O3 -xHost` (fast=1, AVX-512 FMA) | 36 / 188 |

  The failures are a contraction's: `rm` at most 8–10 ulp, `sp`/`phlb` 1–23 ulp, `cm` up to
  2048 ulp, near-zero slopes changing sign; `Check_CFL`'s `(ndyn, n)` is the model's in every
  arm. The full model agrees: the `ch4_6x4_fwd.rc` day built at Orion's flags (1 compute + 2
  IO-server ranks, 792 events, 408 checkpoints; the `pyd1-*` sets) replays **bitwise per event
  and in `--free-run`, ingestion included**; built at Hercules' flags, 171 of 408 checkpoints
  are off (BalanceMassFluxes 7e-10, x/y/z a few ulp, the chemistry 4 ulp) while `massflow`,
  `cfl`, `v` and `m2phlb` stay bitwise. With gfortran the same holds for `-ffp-contract`: the
  full model at `-O3 -march=native` with contraction on moves `phlb = at + bt*sp` 1 ulp, the
  dynam0 `cm` up to 9e-11 relative and flips near-zero slopes; `-ffp-contract=off` makes it
  bitwise. The generic `compiler-gfortran.rc` has it, `compiler-gfortran-local-mac.rc` does not.
- **The model's own libraries.** Library calls are reproducible only through the same library:
  FFTW's `FFTW_ESTIMATE` plan for 60×45 is not numpy's pocketfft (1e-11 relative on the balanced
  fluxes), so `balance.FFTWPoisson` dlopens libfftw3 (`$PYDYN_FFTW`, MKL's FFTW interface under
  ifort) and issues the Fortran's `dfftw_plan_dft_2d` / `dfftw_execute_dft`; `$PYDYN_LAPACK` the
  same for DGETRF/DGETRI (the Crout branch is `inverse="crout"`, bitwise its build; the
  Fortran-vs-Fortran difference between the two is ≤ 2 ulp). Under ifort the transcendentals
  are libimf's, not glibc's (`--libm`, `$PYDYN_LIBM`, `ops.use_libm`): with glibc the
  chemistry's OH and Cl rates are 2–3 ulp off. ifort also compiles an integer power `x**3`
  (Calc_Kzz's `ustr**3`) to libimf's `__powr8i4`, 1 ulp from `(x*x)*x` on a quarter of values,
  which `dkg` amplified to 557 ulp; `ops.libm.cube` calls it (`**2` ifort inlines as `x*x`).
- **NumPy.** Elementwise ops are IEEE double per element with no reassociation, so matching the
  source's parenthesisation is enough; `np.sum`/`np.dot` are *not* sequential, so `pit`, `zrm`
  and `fastminv` are written as accumulation loops. Its vectorised transcendentals depend on the
  CPU: on x86-64 with AVX-512 they dispatch to SVML kernels that are not the C library's. The
  transcription never calls them (`ops.libm` is scalar); the simple-vs-fast tier does, and
  `conftest.py` turns that dispatch off and prints what was dispatched (`PYD-8`).
- The Fortran limiter `max(min(s, abs(rm)), -abs(rm))` and `np.maximum/np.minimum` agree except
  in NaN propagation, which should not occur.
- **About the model, not the Python.** `diffusion.cache.read` is retired: the `dkg` cache is read
  through `tmm.sourcekey.<region>.{ml,sfc}.diffus : dkg-cache:`, and a missing file falls back to
  live `Calc_Kzz`. The production glb6x4 cache (`METEO/tm5-nc/ec/ea/dkg/NEW/tropo34/glb6x4`) is
  not what live `Calc_Kzz` gives the current build: 96 % of cells differ, 7e-6 rms relative per
  hour, at most 1.4e-4, level means equal to four figures.

### 3d. Zoom nesting, in the standalone harness (`PYD-3`)

**Bitwise, tier 1.** `pydyn/zoom/` against the production zoom routines compiled unmodified in
`fortran/harness_zoom{2,3}` (`make -C fortran zoom`), gfortran 16, `-O0 -ffp-contract=off`, arm64.
Real zoom runs are §3f.

What the zoom binaries compile, beside §3a's kernels: `proj/ct-unified/trunk/src/zoom_tools.F90`
(`mix_edges`, `update_parent`, `coarsen_region`), `branches/speedup/src/advect_tools__speedup.F90`
(`m2phlb`, `m2phlb1`, `coarsen_ambm`), `trunk/src/modelIntegration.F90` (`Proces_Region`,
`do_steps`), `trunk/src/advectm_cfl.F90` (`Init_CFL`, `Check_CFL`, `Setup_MassFlow`, and the
mass-only zoom path behind them) with `trunk/src/cfl_cache.F90`, and `trunk/src/datetime.F90`
(`new_valid_timestep`, which keeps `ndyn` a multiple of `2 max(tref)`). modelIntegration is
built with `without_chemistry without_emission without_diffusion`, so `do_steps`' `c` and `s`
cases are empty. There is one binary per region count, `NREGIONS = 2` and `3`: production's
`nregions` is the true count and `Check_CFL`, `store_masses` and `Setup_MassFlow` loop over all
of it. The stubs carry the region tree as run-time variables where production has `dims_grid`
parameters. `with_zoom` is defined, as pycasso defines it for every multi-region build, and it
matters: `convec` skips a parent's columns under a child only under that macro. Not production
behaviour: the reduced grid stays stubbed (`redgridZoom.F90` needs the rc machinery), and
`TimeInterpolation` is a no-op, so pu/pv are held over a `Check_CFL` interval
(`zoom.check_cfl_zoom` takes an `interpolate` callback for a caller that varies them). The CFL
cache is compiled but off.

The Python keeps the Fortran halo bounds (`zoom.FView`, a writable `dump.FArray`): the exchange
is halo arithmetic, and a child's halos hold its parent's boundary cells. Region 1 is advected
by §3a's kernels on interior views; a child by `_dynam{u,v,w}_child`, which mirror dynamu /
dynamv / dynamw's non-periodic, non-polar paths. The region tree (`Zoom`) is
`determine_children_etc`, `blank_zoom_region` and `define_splitorderzoom` (zoom_mode 1),
computed in Python; on a real run it is checked against the dump header (§3f).

**Covered**, `tests/test_zoom.py`: the eight exchange routines; `dynam0` and `convec` per
region; x, y and z at **every** slot of every region's `splitorderzoom` in a two-level (child at
nam1x1's factors) and a three-level tree (the grandchild's 48-operator order), at Courant 0.4
and 3.5 — children sub-cycling to nloop 7 in x and y and 8 in z, and both sides refusing the
same inadmissible cases; Setup_MassFlow, one `Proces_Region(1)` and a whole `ndyn` step on four
trees (the two above, glb3x2 + nam1x1 at production size, the eur tree). Every array of every
region, halos included, and `nloop_max` / `xi` per region and axis, bitwise.

**`Check_CFL` on a tree** (`zoom.check_cfl_zoom`, 8 cases): the mass-only rehearsal with
advectm_cfl's `xyzzyx` order in blocks of three, its mass-only exchange (`tracers=False`), its
caps (10, 1, 1) and its `cfl_ok`-and-unwind recursion (`determine_cfl_iter`). Compared: the
`(ndyn, n)` it settles on and every array it leaves, including sp / phlb as the rehearsal's
`coarsen_region -> m2phlb` wrote them. The cases keep `ndyn`, reduce it once, 12 and 19 times,
and refuse outright on both sides. The caps are a parameter (`caps=`), as `cfl_cap_replay` needs.

**What the code does that the survey and the configurations do not show.**
- Region 1 in a zoom run needs two things a single-region run never shows: dynamu's closing loop
  writes the periodic x halos of m, rm and the slopes, which a child's `put_xedges` can read; and
  dynamv scales and restores the *whole* `bm(:,:,l)` when a layer sub-cycles.
- `update_parent`'s dateline branch (`ibeg > iend`) indexes the parent with the *child's* `im`;
  `put_xedges` reads `ibeg-1` with no wrap where `put_yedges` wraps with `mod`. Both latent — no
  configuration in use has a child at the dateline or at the parent's first or last column — and
  `zoom.Zoom` refuses such trees, as it does pole-touching and cyclic children.
- `tm5-zoom@13149b4a1:docs/dynamic_core_survey.md` §2.5 says a direction's scope is restricted
  when the other directions precede it; it is the reverse. Its §3 `SETUP_MASSFLOW` runs every
  `dynam0` and then `coarsen_region` from the last region down; the code interleaves
  `dynam0(n); coarsen_region(n)` for n = 1..nregions and then `coarsen_ambm` for every region.
  Not corrected there from here.

**Not covered here** (§3f covers them on real runs): the reduced parent with a child, a child's
meteo, `convection__speedup.F90`, `s` and `c` on a child. **Not covered anywhere**: pu/pv varying
within a `Check_CFL` interval of a tree.

### 3e. The ranks together equal the serial answer (`PYD-2`)

**Claim**: every array a multi-rank run dumps, with the ranks' tracer slabs put back in rank
order, is bitwise the array the same build dumps serially. **Holds** on three families of sets
(`runs/MANIFEST.md`), each one run with four distinct tracers made serially and on 2, 3 and 5
compute ranks — slabs 2+2, 1+1+2 and 1+1+1+1+0, the last a rank with no tracer:

| sets | run | build | events per rank (2; 3; 5 ranks) | merged = serial |
|---|---|---|---|---|
| `pyd2_*` | `ch4_6x4_fwd.rc`, 23 h, `tm5-zoom@d5e682986` | gfortran 16 `-O3 -march=native -ffp-contract=off`, Open MPI, no IO ranks | 808; 762, 762, 808; 762 ×4, 716 | 900 events |
| `pyd2i_*` | the same, 6 h, `745e94f70` | ifort 2024.1 at Orion's flags, + 2 IO-server ranks | 210; 198, 198, 210; 198 ×4, 186 | 234 |
| `zp4_*` | `zpfix_replay`'s glb3×2 + nam1×1 with emission and chemistry, 6 h, `745e94f70` + the ch4_loss fix | as `pyd2i_*` | 984; 924, 924, 984; 924 ×4, 864 | 1104 |

`tests/test_ranks.py`, 4 cases per multi-rank set (`runs/hpc/ranks.slurm` on Hercules, the
model's MKL and libimf; a `zp4` set one per job, `-c 6`):

- the tracers differ and the set is decomposed (`test_tracers_are_split`: a misplaced slab
  would show);
- every array not split by tracer is byte-identical on every rank, in every event — `m`, the
  fluxes, the balanced meteo pairs, `dkg`, `Check_CFL`'s decision, the chemistry inputs and
  loss rates, `sp`/`phlb`;
- the merged run is the serial run, event for event and array for array, BalanceMassFluxes
  under MPI included;
- the merged set replays bitwise, as the serial one does.

`zp4_serial` also replays bitwise per event and in a free run on `--meteo --chemistry --initial`:
the zoom engine with more than one tracer. None of these sets is `--emissions`-replayable: the
members' emission is EnKF-scaled. Four distinct members need the scaling file actually read —
the zoom rc's `parameters.constant.value` scales every member alike, so its run copies drop it.

ct-unified decomposes by tracer only (the `'levels'` swap is `base/trunk`'s driver), rank r
holding the contiguous block after ranks 0..r-1 (`mpi_comm.F90`); IO-server ranks dump nothing.
`dump.merge_ranks` (called by `load_events(..., merge=True)`) handles the three things that
means in the dumps:

- `rm, rxm, rym, rzm` are the only split arrays (`TRACER_ARRAYS`), concatenated in rank order.
  They must be named: with equal slabs every rank's bounds are identical, so no size rule tells a
  split array from a replicated one;
- `ch4_tracers` in the `chem` header is the rank's *local* tracer list, made global with each
  rank's offset;
- `emis` is dumped once per tracer the rank holds, so the ranks' event counts and `seq` differ;
  each injection's `emis` blocks (one region's, in a tree) are interleaved by the global
  `i_tracer`, the serial order.

**Not covered**: OpenMP. The laptop MPI build drops `with_hdf5_par with_netcdf4_par` for its
serial netCDF, file I/O only.

### 3f. Zoom, tier 2: a glb3x2 + nam1x1 run replayed per region (`PYD-3`)

**Claim**: the production zoom — glb3×2 with its 14-band reduced grid and the nam1×1 child,
built with ifort at Orion's flags — is reproduced by `pydyn/zoom/` **per event and in a free
run, bitwise, in every array including the halos**, and the free run reads nothing but the
model's input files. `python3 -m pydyn.replay <dir> --rank 0 [--free-run] --meteo … --emissions …
--chemistry … --initial …`, through the model's MKL and libimf:

| set (`runs/MANIFEST.md`) | build | per event | `--free-run` on the input files |
|---|---|---|---|
| `zm_replay` | Orion, ifort 2021.12, 6 compute + 2 IO ranks, transport only | x, y, z, v of both regions (80 / 80, 160 / 160); Setup_MassFlow 20 / 20; `Meteo_Setup_Mass` 6 / 6 per region; `Check_CFL` 6 / 6; m2phlb 20 / 20 per region | every checkpoint, with `Check_CFL`'s own `ndyn`, `update_parent`, m2phlb and `dkg` (`--meteo --initial`) |
| `zps_replay` | Hercules, ifort 2024.1, serial, emission and chemistry on | every operator of both regions incl. s and c (120 / 120, 240 / 240), `Check_CFL` 6 / 6, m2phlb, the initial field per region, every input `in:<name>` | every checkpoint (`--meteo --emissions --chemistry --initial`) |
| `zpfix_replay` | as `zps_replay`, 6 compute + 2 IO ranks, tm5-zoom's ch4_loss fix | as `zps_replay`; and equal to `zps_replay` in every tracer array of all 720 operator events | every checkpoint, as `zps_replay` |
| `zp_replay` | as `zpfix_replay`, without the fix | as `zps_replay` but the child's `c`, 0 / 40 (the model bug below) | — |
| `zm_2019-{01,04,07,10}` | Orion, ifort 2021.12, `cfl` events of four weeks | `Meteo_Setup_Mass` 672 / 672 per region with its ingestion; `Check_CFL`'s `(ndyn, n)` 672 / 672 (`analysis/cfl_cap_replay_zoom.py`, `cfl_headroom.md` §11) | — |

A replay builds its `zoom.Zoom` from the dump header — `status` (the region's position in its
split order: `splitorder[status]` is the operator of a `bef_`/`aft_` pair), `splitorder`, a
child's footprint `ibeg..lend` and the region's `children`, tm5-zoom `84842782e` — and **refuses
to run** unless the footprint, children and split order it derives are the model's
(`replay.build_tree` / `check_tree`). The header is written only under `pydyn.dump : T`; with
the dump off the restarts are identical, with it on the arrays are byte-identical.

The free run carries every region's state from the initial field on, through the operators,
Check_CFL (its own `ndyn`), Setup_MassFlow, `update_parent` and every m2phlb, and the `put_x/yedges` that `do_steps` adds at a
block end — including blocks ended by the `s`/`c` no-ops of a transport-only build, which are
never dumped and are passed by following each region's split-order position. `phlb` is compared
on the cells m2phlb writes; its halos are what initialisation left and nothing reads them. What it
takes, beyond §3d:

| | the model's rule | where |
|---|---|---|
| a reduced parent | `red2uni` ends every reduced row with `q(0,j,l) = q(im,j,l)` on the *uniform* row, overwriting the halo dynamu's closing loop wrote (476 halo cells per parent x event otherwise; pinned by `test_zoom.py::test_reduced_parent_x_halo_is_the_uniform_row`) | `zoom._dynamu_region1` |
| a child's meteo | the child's sp matched 'area-aver' to its parent's; its raw fluxes matched 'sum' to the parent's **balanced** ones on every parent face, then balanced per parent cell with the Poisson correction of each 3×2 block's own residual from `EnsurePoissonCache`'s direct matrices, which vanishes on the block's faces | `zoom_meteo.py` |
| the meteo's source | the zoom rc reads the 1×1, 137-level ERA5 set, not the coarsened `tropo34` sets (which hold the same fields rounded to float32, 2^28 ulp off): FillGrid combines 3×2 cells for the parent ('u' faces summed over rows, 'v' over columns, `mfw` over both, `sp` and the convection fields area-averaged, T and Q 'mass-aver' on each source cell's own `(da + db ps) area`) and copies a window for the child; FillLevels sums each tropo34 layer's ml137 levels in descending file order (T, Q weighted by `|dp|` on the model's `ps`, Q clipped at 0) and copies `mfw`'s half levels | `regrid.py` |
| who acts where | do_add_2d, ch4_chemistry_step and convec act on the interior scope isr:ier × jsr:jer and skip the cells a region does not own (`zoomed /= region`: a parent's cells under its child, a child's edge ring); the chemistry computes its loss rates on the scope only and leaves the rest at decay 1.0; Calc_Kzz runs on the scope, `dkg` zero on a child's ring | `zoom.owned`, `sources_region`, `chemistry_region`, `convec_region` |
| the physics' time | Meteo_Setup_Other interpolates T, Q and Calc_Kzz's fields over the dynamics step (tdyn, tdyn + ndyn), not the read interval; they differ when Check_CFL cuts the step | `meteo.Meteo`, `replay.Ingestion` |
| `dkg`'s pressures | `spm` is the read interval's `sp1/sp2` weighted to the step's middle; the geopotential's starting `sp` is a parent's m2phlb1 of its mass after Setup_MassFlow's `coarsen_region`, a child's from its last m2phlb | `replay.step_inputs`, `FreeRun.massflow` |
| the start | each region's initial field from the file on its own grid — the glb3×2 file copied onto glb3×2 and distributed over nam1×1 (each file cell's value in its 3×2 children) — on its own initial air mass, then `update_parent`, whose last act is the parent's m2phlb1; before its first m2phlb event region 1's `phlb` is that of its coupled mass; Pressure_to_Mass's `m` has periodic x halos | `inputs.Ingestion.initial_state`, `FreeRun._seed_initial` |
| the physics' inputs | emission_ct coarsens the 1×1 field onto each region on its own (3×2 block sums for glb3×2, the window for nam1×1, the edge ring included), and read_OH_field and the soil map do the same per region (OH 'copy' onto a 1×1 region; one soil map file per region) | `emis.Emissions`, `chemfields.LossFields`, `inputs.Ingestion` |

The comparisons can fail: the per-event `ic:m` of a parent skips the child's footprint, which
Setup_MassFlow's `coarsen_region` rebuilds in another summation order than `update_parent` (2 ulp,
the free run checks the order instead); the child's balance on the laptop's FFT is 512–32768 ulp
off inside the parent cells; the coarsened meteo fails every raw array; and `zp_replay` fails the
child's chemistry.

**A model bug, found here, fixed in tm5-zoom `0385efe7e`.** In an MPI build ch4_chemistry_step
bound the per-node shared loss array with `c_f_pointer(..., ch4_loss, (/n_lon, n_lat, n_levels,
nreac/))` — lower bounds 1 — and read `ch4_loss(i_lon, j_lat, k, r)` with `i_lon = isr..ier`,
`j_lat = jsr..jer`; the serial build allocates it `(lon_start:lon_end, lat_start:lat_end, ...)`.
Region 1 (isr = jsr = 1) was unaffected; nam1×1 (isr = 4, jsr = 3) decayed each cell with the
losses of the cell 3 east and 2 north, wrapping at the ends, 135 cells reading past the array —
in every MPI run with a nam1×1 child since tm5-zoom `57933d3ca` (2026-03-26). The fix remaps the
pointer to the serial bounds. `analysis/ch4_loss_bounds.py` replays each chemistry event both
ways from the dumped losses (`--expect bug|fixed`):

| set | region 1: intended / as bound | nam1×1: intended / as bound |
|---|---|---|
| `zp_replay` (MPI, before the fix) | bitwise / bitwise | ≤ 1.2 % of values / **358 500 of 358 500** |
| `zps_replay` (serial) | bitwise / bitwise | **359 040 of 359 040** / ≤ 1.2 % |
| `zpfix_replay` (MPI, fixed) | bitwise / bitwise | **359 040 of 359 040** / ≤ 1.2 % |

**Not covered**: the three-level eur tree outside §3d; zoom under OpenMP.

## 4. The constants mirror

A third discipline, and not a bitwise one. §2 and §3 compare arrays — same inputs, same
outputs — which leaves two blind spots. A constant that only decides behaviour at the margin
is invisible when no run reaches the margin: `Check_CFL` picks an `ndyn` at which y and z
never sub-cycle, so `cfl_max_nloop_{y,z}` could hold any value and every array test would
still pass. And the physical constants that *are* covered are only covered where the dump
sets are on disk.

`tests/test_constants.py` reads the declarations out of the checkout at `$PYDYN_TM5ZOOM` and
compares them with the Python's: 100 cases, 0.2 s, every module-level numeric constant in the
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
(`cfl_headroom.md` §5). It runs only where a checkout is present, so it protects a developer,
not CI.
