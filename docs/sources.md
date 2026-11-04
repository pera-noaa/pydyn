<!-- author: Ash Pera (NOAA/GML CIRES) -->
<!-- footer-title: pydyn — source register -->

# Source register

What this repo reads, transcribes or depends on in `tm5-zoom`, and what it does with it.
Pin: `tm5-zoom@745e94f70`, branch `NOAA/ct-unified`, re-pinned 2026-10-13 **by test**: tier 1
compiled from a clean export at that sha, the constant mirrors parsed from a clean clone of it,
and the three laptop dump sets regenerated from it and replayed bitwise (§7). The previous pin
was `13149b4a1` (2026-09-23). The kernel and doc rows below were read at `4dc2017c5`; the two
`ch4_3x2_pydyn_*.rc` files and `pydyn_dump.F90`'s tag filter arrived after it.
The `chemfields.py` and `initial.py` rows (`PYD-4`, 2026-09-24) were read from the copies the
PYD-2 build compiled (`d5e682986`) and are byte-identical at `f7ab87082`. Since the pin,
`0385efe7e` changes `chemistry_ch4.F90`'s MPI branch only (the zoom child's `ch4_loss` bounds,
found here, `state.md` §3f); the path `chem.py` transcribes is unchanged.

**Nothing in this table is copied into this repo.** There are no verbatim lifts and no md5
rows, which is the difference from the `ct-meteo` register: the Fortran is compiled where it
lives and the Python is a transcription, not a copy. Re-pin by re-reading, not by re-syncing.

**`tm5-zoom` is not public yet.** TM5's release under an open licence is in progress; until it
lands, a `tm5-zoom@<sha>:<path>` citation here or anywhere in this repo names a file outside
readers cannot open, and the tiers that need the Fortran (the compiled kernels, the constant
mirrors, the dump sets) run only where a checkout is. The shas stay as written: they resolve if
that history is published, and if TM5 is released from another tree this register is the map
from them to its files.

## 1. Compiled in place by `fortran/Makefile` (§3a)

Unmodified, through `$(TM5ZOOM)`. The overlay resolution is the production one
(`proj/ct-unified/rc/__zoom_1day.rc`; later wins, `X__suffix.F90` competes as `X.F90`).

| file | why this copy |
|---|---|
| `proj/ct-unified/trunk/src/advectx__slopes.F90` | production x advection (`dynamu`) |
| `proj/ct-unified/trunk/src/advecty__slopes.F90` | production y advection (`dynamv`) |
| `proj/ct-unified/branches/speedup/src/advectz__slopes.F90` | production z advection (`dynamw`) |
| `base/trunk/src/advect.F90` | `dynam0`, not overlaid |
| `base/trunk/src/convection.F90` | the arithmetic `convection__speedup.F90` was derived from; the speedup copy needs rc, shared memory and MPI, so it cannot be driven by a stub harness. See `state.md` §3a for how §3b settles the parity question anyway |

`make zoom` (`harness_zoom2`, `harness_zoom3`, `state.md` §3d) compiles the five above again
with `NREGIONS = 2` and `3` and `with_zoom`, plus:

| file | why this copy |
|---|---|
| `proj/ct-unified/trunk/src/zoom_tools.F90` | `mix_edges`, `update_parent`, `coarsen_region` |
| `proj/ct-unified/branches/speedup/src/advect_tools__speedup.F90` | `m2phlb`, `m2phlb1`, `coarsen_ambm` |
| `proj/ct-unified/trunk/src/modelIntegration.F90` | `Proces_Region`, `do_steps`; built with `without_chemistry without_emission without_diffusion`, as the transport-only dump sets were |
| `proj/ct-unified/trunk/src/advectm_cfl.F90` | `Init_CFL`, `Check_CFL`, `Setup_MassFlow` and the mass-only zoom path behind them; the CFL cache off |
| `proj/ct-unified/trunk/src/cfl_cache.F90` | the cache state `advectm_cfl` uses |
| `proj/ct-unified/trunk/src/datetime.F90` | `new_valid_timestep` (and `tau2date` for `Proces_Region`) |

`fortran/stubs.F90` supplies `GO`, `go_system`, `dims`, `chem_param`, `MeteoData`,
`tracer_data`, `global_data`, `ParTools`, `omp_ParTools`, `redgridZoom`, `zoom_tools` (single
region only), `toolbox`, `datetime` (single region only), `binas`, `misctools`, and for the
zoom builds `user_output`, `pydyn_dump`, `sources_sinks`, `MDF` and `cfl_cache_shared` (the
shared CFL cache, `advectm_cfl.F90`'s since `79b30df3b`, held off) as no-ops. `fortran/advect.h` + `tm5.inc`
reproduce the production cpp defines.

## 2. Transcribed into Python (never compiled here)

Each row is a Fortran routine the Python reproduces statement by statement.
`state.md` has the routine-level table; this one is the file-level index.

| file | Python |
|---|---|
| `proj/ct-unified/trunk/src/advectm_cfl.F90` | `pydyn/cfl.py` |
| `proj/ct-unified/trunk/src/redgridZoom.F90` | `pydyn/redgrid.py` |
| `base/trunk/src/{meteo,advect_tools,go_date}.F90` | `pydyn/balance.py`, `pydyn/meteo.py` |
| `proj/ct-unified/branches/speedup/src/grid_type_ll__speedup.F90` | `pydyn/balance.py` (BalanceMassFluxes, SolvePoissonFFTW) |
| `base/trunk/src/{grid_type_ll,geometry,meteodata,tmm_mf_tm5_nc}.F90` | `pydyn/meteo.py` |
| `base/trunk/src/diffusion.F90`, `proj/ct-unified/trunk/src/diffusion_kernels.F90` | `pydyn/diffusion.py`, `pydyn/simple_diffusion.py` |
| `proj/ct-unified/trunk/src/emission_data.F90`, `emission_ct.F90`, `base/trunk/src/toolbox.F90`, `proj/ct-unified/branches/emis_unified/src/{emission_ch4_tracer,emission_units}.F90` | `pydyn/emis.py` |
| `proj/ct-unified/branches/chem_unified/src/chemistry_ch4.F90`, `proj/ct-unified/trunk/src/emission_regrid.F90` | `pydyn/chem.py` |
| `proj/ct-unified/branches/chem_unified/src/read_OH_field.F90`, `proj/ct-unified/trunk/src/emission_regrid.F90` (the interpolation maps), `chemistry_ch4.F90` (the soil map), `proj/levels/ml137/tropo34/src/dims_levels__ml137_tropo34.F90` (`echlevs`), `grid_type_ll__speedup.F90` (FillGrid) | `pydyn/chemfields.py` |
| `proj/ct-unified/trunk/src/sources_sinks__ctuni.F90` (`start_from_field_fill3d`), `base/trunk/src/{grid_3d,grid_type_hyb}.F90`, `grid_type_ll__speedup.F90` (Relate, FillGrid), `chem_param.F90` (`fscale`) | `pydyn/initial.py` |
| `proj/ct-unified/branches/speedup/src/convection__speedup.F90` (`with_lapack` `fastminv`) | `pydyn/lapack.py` |
| `proj/ct-unified/trunk/src/modelIntegration.F90` (`do_steps`) | `pydyn/driver.py`; with the region recursion, `pydyn/zoom/` |
| `proj/ct-unified/trunk/src/initexit.F90` (`determine_children_etc`, `blank_zoom_region`, `define_splitorderzoom`) | `pydyn/zoom/` (`Zoom`) |
| `proj/ct-unified/trunk/src/advect{x,y}__slopes.F90`, `branches/speedup/src/advectz__slopes.F90` (the zoom paths), `trunk/src/zoom_tools.F90`, `branches/speedup/src/advect_tools__speedup.F90` (`m2phlb`, `coarsen_ambm`), `trunk/src/advectm_cfl.F90` (Setup_MassFlow's region loop; Check_CFL's zoom path) | `pydyn/zoom/` |
| `proj/ct-unified/branches/meteo-server/src/meteo__meteo_server.F90` (Meteo_Setup_Mass for a child), `branches/speedup/src/grid_type_ll__speedup.F90` (GetRefinement, Match_cell / Match_u / Match_v, BalanceMassFluxes_m against a parent, EnsurePoissonCache, SolvePoissonDirect) | `pydyn/zoom_meteo.py` |
| `base/trunk/src/{tmm,tmm_param,grid_type_hyb}.F90` (ReadField, Transform3Dh/v incl. 'mass-aver', GetCombineKeys, levi_Init, levi_Init_select, levi_Compare, levi_FillLevels_3d), `grid_type_ll__speedup.F90` (FillGrid 'copy' / 'combine' / 'weight', AreaOper), `branches/speedup/src/tmm_mf_tm5_nc__speedup.F90` (mf_ReadRecord) | `pydyn/regrid.py` |
| `proj/ct-unified/trunk/src/emission_data.F90` (do_add_2d's scope and `zoomed`), `branches/chem_unified/src/chemistry_ch4.F90` (ch4_chemistry_step's scope, with_zoom decay 1.0), `base/trunk/src/diffusion.F90` (Calc_Kzz's scope) | `pydyn/zoom/` (`owned`, `sources_region`, `chemistry_region`), `pydyn/replay.py` |
| `proj/ct-unified/trunk/src/emission_ct.F90` (coarsen_cat onto each region), `base/trunk/src/toolbox.F90` (coarsen_emission_2d's region loop), `branches/chem_unified/src/read_OH_field.F90` and `chemistry_ch4.F90` (each region's fields and soil map), `trunk/src/initexit.F90` `start` and `sources_sinks__ctuni.F90` (Fill3D per region, then update_parent), `grid_type_ll__speedup.F90` (FillGrid 'copy' and 'distribute') | `pydyn/emis.py`, `pydyn/chemfields.py`, `pydyn/initial.py`, `pydyn/inputs.py` (`Ingestion`) |

A change to any of these is a change to what this repo claims. There is no automatic check
for that: the tests catch it only when a dump set is regenerated, as at the re-pin (§7).

## 3. Read by `tests/test_constants.py`

Not compiled and not transcribed — *parsed*, so a constant that moves in the Fortran fails
a test here (`state.md` §4). These are declaration sites, and a restructuring that moves a
declaration out of them fails too, by design.

| file | declarations read |
|---|---|
| `base/trunk/src/binas.F90` | `grav`, `pi`, `ae`, `cp_air`, `lvap`, `Rgas`, `Avog`, `xmair`, `vkarman`, `p_global` |
| `base/trunk/src/dims.F90` | `one`, `gtor`, `n_operators`, `splitorder`, `zoom_mode` |
| `proj/ct-unified/trunk/src/diffusion_kernels.F90` | the 20-odd boundary-layer parameters of `bldiff`, and `kvhmin` as an assignment |
| `proj/ct-unified/trunk/src/chem_param.F90` | `xmch4` (`xmc + xmh*4`) |
| `proj/ct-unified/trunk/src/advectm_cfl.F90` | `cfl_max_nloop_{x,y,z}` (for `cfl.py` and `zoom/check_cfl.py`'s mass kernels), `max_global_iteration`, its own `n_operators` and `splitorder` |
| `proj/ct-unified/trunk/src/advect{x,y}__slopes.F90`, `branches/speedup/src/advectz__slopes.F90` | the kernels' own `max_nloop` / `iter_limit`, for `kernels.py` and the child kernels of `zoom/advect.py` |
| `base/trunk/src/meteo.F90` | `MB_MAX_TRIES`, and `tol_rms` / `tol_diff` as assignments |
| `proj/ct-unified/branches/chem_unified/src/chemistry_ch4.F90` | the rate literals of `krate_oh`, `krate_cl_trop` and `tropopause_pressure` |
| `proj/ct-unified/branches/chem_unified/src/read_OH_field.F90` | the Hossaini unit literals (`1.0E-9`, `1.0E-6`) |
| `proj/ct-unified/trunk/src/emission_regrid.F90` | the interpolation maps' span floor `1.0e-10` |
| `proj/levels/ml137/tropo34/src/dims_levels__ml137_tropo34.F90` | `echlevs` |
| `proj/ct-unified/trunk/src/sources_sinks__ctuni.F90` | the IC's ppm factor and its `fld_mean` guard |

## 4. Depended on, owned by tm5-zoom

| file | relationship |
|---|---|
| `proj/ct-unified/trunk/src/pydyn_dump.F90` | writes the events `pydyn/dump.py` reads. Added in `tm5-zoom@a4fc52de9`. Its `.meta` key list and array bounds are the wire format of §3b; under MPI each rank writes its own `_p<rank>` files with its tracer slab, and the `emis` header's `i_tracer` (global) / `itracer_loc` and the `chem` header's rank-local `ch4_tracers` are what `dump.merge_ranks` relies on (§3e) |
| `base/trunk/src/mpi_comm.F90` | the tracer decomposition `merge_ranks` assumes -- contiguous slabs in rank order (`:566`, `:724`); read, not compiled |
| `modelIntegration.F90`, `initexit.F90`, `emission_data.F90`, `branches/chem_unified/src/chemistry_ch4.F90`, `branches/meteo-server/src/tm5__meteo_server.F90` | the hook sites |
| `proj/ct-unified/rc/ch4_6x4_fwd.rc` | the rc the three laptop dump sets are made from, and in per-run copies the four-tracer `PYD-2` sets (`pyd2_*`, `pyd2i_*`) |
| `proj/ct-unified/rc/ch4_3x2_pydyn_tracer.rc` | the rc the Orion glb3x2 tracer dumps are made from (`cfl_headroom.md` §9). Its sister below with `ct.cachegen : F` so `ntracet = 1`, and the dump widened to `cfl,massflow,bef_x` |
| `proj/ct-unified/rc/ch4_3x2nam1x1_pydyn.rc` | the rc the zoom sets are made from (`zm_replay`, the `zm_2019-*` census weeks; `state.md` §3f), transport only; the physics sets (`zp*_replay`) use an untracked copy without `without_emission`/`without_chemistry`, and the four-tracer `zp4_*` a copy of that with `ct.ensemble.total_members 4` and no `parameters.constant.value`, which `emission_ct.F90` would apply to every member instead of reading the scaling file (`runs/MANIFEST.md`) |
| `proj/ct-unified/rc/ch4_3x2_pydyn_cfl.rc` | the rc the Orion glb3x2 `cfl` dumps are made from. Derived from `ch4_3x2_cachegen_year.rc` — the rc that wrote the caches §7 of `cfl_headroom.md` censused — with the cache reads and writes off and `pydyn.dump.tags : cfl` on |

Changing the dump format is a tm5-zoom commit **and** a `pydyn/dump.py` change **and** a
re-pin here. The old dump sets do not migrate; they are regenerated.

The zoom geometry tests type in two region definitions rather than parse them:
`base/trunk/rc/pycasso-tm5-regions-glb3x2.rc` (`glb3x2`, `nam1x1`) and
`base/trunk/src/dims_grid__eur.F90`.

## 5. Docs

| doc | disposition |
|---|---|
| `docs/dynamic_core_survey.md` | the algorithm this transcribes. Its §5 points back here. Cite by `tm5-zoom@<sha>:docs/dynamic_core_survey.md §N`; never move or renumber it. Two statements in it are contradicted by the code (§2.5's scope rule, §3's `SETUP_MASSFLOW` order; `state.md` §3d) and are not corrected from here |
| `docs/kernel_catalog.md` | per-kernel timing. Context for which kernels are worth a port |
| `docs/tm5mp_vs_tm5zoom.md` §5 | zoom vs mp transport differences. `PYD-2`'s rank merge is ct-unified's tracer decomposition; read this before assuming it of tm5-mp |
| `docs/draft_knmi_tm5_slopelimiter_report.md` | the limiter |
| `docs/adjoint_phase3_transport.md` | the adjoint of the same operators |
| `proj/ct-unified/trunk/doc/bugreport_cfl_y_substepping.md` | the `Check_CFL` y/z strictness question `state.md` §3b reports on |
| `base/trunk/doc/advection_perf_notes.md` | context |
| `proj/gpu-native/INDEX.md` and its tree | a separately owned workstream, a likely consumer (`PYD-7`). Read-only from here |

## 6. History

`tools/pydyn` in tm5-zoom, from its first commit `28e94d4f6` to `4dc2017c5`, where it became
this repo. `tools/pydyn/README.md` there is now a pointer. Anything before 2026-09-22 is in
that repo's `git log`, not in this one's.

## 7. Re-pin `13149b4a1` → `745e94f70` (2026-10-13, `PYD-13`)

13 of the files above changed between the two pins. Every change is either a new rc option
that defaults to the old behaviour, an MPI or shared-memory path the serial build does not
take, a log or error check, or a comment. The re-pin tests the default path rather than
trusting the commit messages:

| test, at `745e94f70` | result |
|---|---|
| §3a / §3d harnesses compiled from a clean export; parity, zoom, kernel, simple-vs-fast and constant tests | 341 / 341 |
| `pydyn_dump_{phys,free,cfl}` regenerated from a clean clone by `state.md` §3b's recipe (`runs/MANIFEST.md`, `repin_*`) | 759, 483, 22 events, as the originals |
| `replay` per event, phys | 391 / 391 events bitwise, 47 / 47 arrays |
| `replay --free-run` on the input files alone (`--meteo --emissions --chemistry --initial`), phys | 391 / 391 checkpoints bitwise |
| `replay --free-run`, free and cfl | 299 / 299 and 14 / 14 bitwise |
| the whole suite on those sets and harnesses (rank tests excepted: they replay the PYD-2 MPI sets) | 400 / 400, none skipped |

| file | commits | what changed | on the transcribed path? |
|---|---|---|---|
| `base/trunk/src/meteo.F90` | `6e9117dfe` | the `kzz_dat` comment (ECMWF K placement, CH4-46) | no: pydyn computes `Calc_Kzz`, and no ct-unified rc reads ECMWF K |
| `branches/chem_unified/src/chemistry_ch4.F90` | `4873ac9fe`, `189b103aa`, `0639bb644`, `71a34b977`, `d7944a3bb` | `ch4loss.strat.source : transcom` (default `bruehl`); a `with_budgets` reduce guard; an MPI node barrier; a shared-memory fallback; `rebin_levels` status checked | default path yes, replayed bitwise (`c`); `transcom` and `with_budgets` are not transcribed |
| `branches/chem_unified/src/read_OH_field.F90` | `4873ac9fe` | the TRANSCOM OH/Cl/O(1D) reader, new | no, not transcribed |
| `branches/speedup/src/convection__speedup.F90` | `014e6b911`, `e4382ea21`, `33c5080eb`, `14a097b3b`, `4b2776ab9`, `71a34b977` | `convection.slopes.full_matrix` (default F) and its `rzm` sub-keys, `convection.slopes.rzm.subsidence` and `convection.diag.outrates` (default F); an MPI barrier; shared memory | default path yes, replayed bitwise (`v`); the `full_matrix` options are not transcribed |
| `branches/speedup/src/grid_type_ll__speedup.F90` | `71a34b977` | shared-memory fallback | MPI only; `meteo` bitwise |
| `trunk/src/advectm_cfl.F90` | `79b30df3b`, `155d43ba0` | `cfl.cache.shared` (default F), a log gate | yes, `cfl` bitwise; the shared cache is stubbed off in the zoom harness |
| `trunk/src/diffusion_kernels.F90` | `11c9f506a` | `bldiff` exported only where compiled | no arithmetic; constants unchanged |
| `trunk/src/emission_regrid.F90` | `d7944a3bb` | `rebin_levels` zeroes its output before an error return | only on a failing column |
| `trunk/src/initexit.F90` | `9574924ce` | `proces.advection.limits` (default T, the old behaviour) | default yes; limiter off is not transcribed |
| `trunk/src/modelIntegration.F90` | `11c9f506a` | `convection.apply` (default T) | default yes; `F` (no `v` step) is not transcribed |
| `trunk/src/pydyn_dump.F90` | `84842782e` | the zoom header | this repo's own (`state.md` §3f) |
| `trunk/src/sources_sinks__ctuni.F90` | `8f95cc69a` | `Check_Initial_Tracers`' prefix tokens | no: `start_from_field_fill3d` is unchanged |
| `trunk/src/zoom_tools.F90` | `155d43ba0` | a log gate | no arithmetic |

Outside the source tree: `base/trunk/rc/machine-local-mac.rc` takes `MAC_PREFIX` from
`${CONDA_PREFIX}` since `5292746df`, so a local build now needs `CONDA_PREFIX` exported
(`/opt/anaconda3` for every set in `runs/MANIFEST.md`).

**New non-default paths, not transcribed:** `ch4loss.strat.source : transcom`,
`proces.advection.limits : F`, `convection.apply : F`, `convection.slopes.full_matrix : T`.
A dump made with any of them on cannot be replayed here.
