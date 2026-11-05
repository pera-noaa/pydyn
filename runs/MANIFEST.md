# Run manifest: the dump sets pydyn is verified against

One row per `pydyn_dump` set. Each is produced by a TM5 run with `pydyn.dump : T`, pinned to
the model sha, the build flags and the rc it came from; the flags are part of the claim,
because `-ffp-contract=off` is what makes the comparison bitwise (`docs/state.md` §3c).

**Published:** `repin_free_745e94f70`, `repin_cfl_745e94f70` and `zm_meteo_fixture` are the
Zenodo fixture, [10.5281/zenodo.23955885](https://doi.org/10.5281/zenodo.23955885), each with
a `config/` folder (the rc the model read, the build fingerprint, the logs, the recipe):
`runs/pack_fixture.py` made it, `runs/fetch_fixture.py` fetches it. No other set below is
published.

**The dumps themselves are scratch and are not in this repo.** The laptop sets live under
`$PYDYN_TM5ZOOM/.tm5-cloud-scratch/` (default `../tm5-zoom`) and are regenerated from the recipe
in `docs/state.md` §3b rather than archived; the Orion and Hercules sets under
`/work2/noaa/co2/ash/pydyn/dumps/`, the file system the two machines share. A set made on
another machine needs its own row.

**Recipes.** The laptop sets are made by `runs/setup_repin.sh` (it stops on a failed model run,
except the cfl set's known abort). The Orion and Hercules sets were made by job scripts that
are not in this repo (`run_model.slurm`, `step_accuracy.slurm`, `run_pyd*.sh`); the rows below
name them and the `/work2` directory that keeps them. A published set carries its own recipe in
its `config/`.

| set | made | host | model sha | rc + overrides | build flags | events | size | what it proves |
|---|---|---|---|---|---|---|---|---|
| `pydyn_dump_phys` | 2026-09 | local mac (arm64, gfortran 16) | `tm5-zoom` pre-`4dc2017c5`, `NOAA/ct-unified` | `ch4_6x4_fwd.rc`, 23 h of 2000-01-01, glb6x4 60×45×34, 1 CH4 tracer, `nred = 4`, `with_lapack`, FFTW Poisson, `par.mpi=F par.ntask=1`, `oh.scale 0.901` | `-O3 -march=native -ffp-contract=off` | 759 | 4.8 GB | the whole production path: x y z v s c, massflow, cfl, meteo, m2phlb — per event and in `--free-run` (391 checkpoints), and with `--meteo`/`--emissions`/`--chemistry`/`--initial` the ingestion too: the free run on the input files alone (IC `istart 66` from `.tm5-cloud-scratch/ic/ch4_init_glb3x2_mmr.nc`: the lost rc's path is not on record, but this file reproduces the set's first state bitwise) |
| `pydyn_dump_free` | 2026-09 | local mac | same | same plus `build.configure.macro.define=… without_emission without_chemistry` | same | 483 | 3.3 GB | transport alone, no physics in between; largest Courant number 0.998 |
| `pydyn_dump_cfl` | 2026-09 | local mac | same | transport-only build at `time.ntimestep = time.ndyn_max = 10800`, `meteo.tinterp.*=interp3` | same | 22 | 155 MB | `Check_CFL` rejecting 10800 s and the step running at 5400 s, `n = 2`; the run then aborts on the hourly `aver1` surface fields |

**These three were moved to the Trash on 2026-10-13**, after the re-pin sets below were found
byte-identical to them: all 21,968 `.bin` files equal, the `.meta` files different only by the
three header keys `84842782e` added. Their `run.log`, the one thing unique to them, is kept in
each re-pin set as `run.log.pre-4dc2017c5`. So the unknown "pre-`4dc2017c5`" build wrote
exactly what `745e94f70` writes.

`pydyn_dump/` (440 MB) in the same directory is an earlier scratch set, superseded by the
three above and not asserted on by any test.

**The run directory these three sets came from was overwritten on 2026-09-24.** A PYD-2 setup
passed `my.gitroot`, `my.basedir` and `unoptimized.runid` on the pycasso command line; the
values were recorded but the paths derived from them were already expanded from the rc, so the
build compiled the live checkout into `.tm5-cloud-scratch/ash/unified-testing/ch4u-6x4-test/`
and replaced its executable, `run/ch4u-6x4-test.rc`, `Makefile`, `submit_tm5` and
`build_local-mac_gfortran_default_real8_optim-fast/` (including `build_fingerprint.txt`). The
dump sets are separate directories and untouched, as are `run/tm5.log` and `run/tm5.ok`. What
the lost rc said, as read before the overwrite: `my.gitroot $HOME/tm5-zoom`,
`MAC_PREFIX /opt/anaconda3`, HDF4 from `$HOME/Documents/local-tm5/.tm5-local/hdf4`,
`optim-fast = -O3 -march=native -ffp-contract=off`, `compiler.lib.lapack.libs = -framework
Accelerate -L/opt/homebrew/lib -lfftw3`, macros `with_hdf4 with_hdf5_par with_netcdf4
with_netcdf4_par with_udunits2 with_io_server_staging slopes with_pycasso with_tmm_tm5
without_photolysis without_dry_deposition without_wet_deposition with_lapack`,
`ct.ensemble.size 1`, `par.mpi F`, `diffusion.cache.read T` from
`METEO/tm5-nc/ec/ea/dkg/NEW/tropo34`, `cfl.cache.read F`, `restart.write T`, timerange
2000-01-01 00:00 to 23:00, `pydyn.dump.dir .../pydyn_dump_phys`, `max_events 3000`. The
model sha is not recoverable from it; the table's "pre-`4dc2017c5`" stands. Anything that
sets paths for a local TM5 build now does it in a copy of the rc
(`.tm5-cloud-scratch/pydyn_build/setup_pyd2.sh`).

## The re-pin sets: the three above, regenerated at `745e94f70` (laptop)

Made 2026-10-13 for `PYD-13` (`docs/sources.md` §7) by
`runs/setup_repin.sh 745e94f70 phys|free|cfl`, from a clean clone
(`pydyn_build/ctl-745e94f70`, nothing edited), under `pydyn_build/dumps/repin_<kind>_745e94f70`.
Build and run as the rows above: `ch4_6x4_fwd.rc` copy with the paths and
`timerange.end 2000-01-01 23:00:00` in the file, glb6x4, 1 CH4 tracer (`ct.ensemble.size = 1`,
one-member scaling file at 1.0 from `runs/make_scaling_one.py`), IC from
`.tm5-cloud-scratch/ic/ch4_init_glb3x2_mmr.nc`, Homebrew gfortran 16.1, `CONDA_PREFIX` =
`MAC_PREFIX` = `/opt/anaconda3`, `-O3 -march=native -ffp-contract=off`, Accelerate LAPACK and
Homebrew FFTW, serial, CFL cache read and write off.

| set | build | events | size | what it proves |
|---|---|---|---|---|
| `repin_phys_745e94f70` | macros from the rc (`with_hdf4 with_hdf5_par with_netcdf4 with_netcdf4_par with_udunits2 with_io_server_staging slopes with_pycasso with_tmm_tm5 without_photolysis without_dry_deposition without_wet_deposition with_lapack`) | 759 | 4.8 GB | per event 391/391 bitwise; `--free-run` on the input files alone 391/391; the 50 production and 9 input tests |
| `repin_free_745e94f70` | the same plus `without_emission without_chemistry` | 483 | 3.3 GB | `--free-run` 299/299 bitwise |
| `repin_cfl_745e94f70` | as free, `time.ntimestep = time.ndyn_max = 10800`, `meteo.tinterp.*=interp3` | 22 | 155 MB | Check_CFL rejecting 10800 s; `--free-run` 14/14 bitwise; the run then aborts on the hourly surface fields, as the original did |

These are the tests' defaults (`tests/test_production.py`, `tests/test_inputs.py`) and
`analysis/cfl_rejection_anatomy.py`'s. The re-pin itself also ran the suite with
`PYDYN_TM5ZOOM` at the clean clone `pydyn_build/ctl-745e94f70`, so the constant mirrors were
read at the pin too.

## The PYD-2 sets: one run, serial and on 2, 3 and 5 ranks (laptop)

Made 2026-09-24 to test `PYD-2`'s claim that the ranks together equal the serial answer
(`docs/state.md` §3e, `tests/test_ranks.py`). All four come from one clean clone of
tm5-zoom at **`d5e682986`** (`.tm5-cloud-scratch/pydyn_build/tm5-zoom`, cloned from the
local checkout, nothing edited), built and run by
`.tm5-cloud-scratch/pydyn_build/setup_pyd2.sh`, which writes a per-run copy of
`proj/ct-unified/rc/ch4_6x4_fwd.rc` beside the original with `my.gitroot`, `my.basedir`, the
runid and the local IC path set **in the file** -- pycasso expands those before command-line
overrides apply (see the note above). Everything lives under `.tm5-cloud-scratch/pydyn_build/`
(`runs/<set>/`, `dumps/<set>/`), none of it in git.

Common to all four: `ch4_6x4_fwd.rc`, glb6x4 60×45×34, `ct.ensemble.total_members = 4`
tracers, 23 h of 2000-01-01 (the `timerange.end` override is recorded but not used either,
so each run steps to 23:00 and then stops on the next day's `u10m`, as the older sets do),
IC `istart 66` from `.tm5-cloud-scratch/ic/ch4_init_glb3x2_mmr.nc`, the four members' emission
scaling from `pydyn_build/make_scaling.py` -- mean 1.0, member deviations 0, +0.2, a -0.3 sine
and a seeded ±0.4 random field -- so the **four tracers differ** and a misplaced slab cannot
pass. Homebrew gfortran 16.1 (the serial toolchain), `MAC_PREFIX /opt/anaconda3`,
`-O3 -march=native -ffp-contract=off`, Accelerate LAPACK and Homebrew FFTW, `with_lapack`.

| set | build | ranks, tracer slabs | events | size | what it proves |
|---|---|---|---|---|---|
| `pyd2_serial` | `par.mpi=F`; macros as the older sets (with the inert `with_hdf5_par with_netcdf4_par`) | 1: 4 | 900 | 10 GB | replays bitwise, all 441 replayable events -- the reference; the four-tracer free run on `--meteo --chemistry --initial` too (394 checkpoints; its emission is EnKF-scaled, so not `--emissions`) |
| `pyd2_mpi2` | `par.mpi=T`, Open MPI 5.0.9 (Homebrew, wrapping the same gfortran); `with_hdf5_par with_netcdf4_par` **dropped** (serial netCDF/HDF5; the two macros only touch `mdf.F90` and `tm5_restart.F90`, i.e. file I/O) | 2: 2+2 | 808 per rank | 13 GB | merged, bitwise the serial set event for event; replays bitwise |
| `pyd2_mpi3` | the `pyd2_mpi2` binary, `mpirun -np 3` | 3: 1+1+2 | 762, 762, 808 | 16 GB | the same, with uneven slabs |
| `pyd2_mpi5` | the `pyd2_mpi2` binary, `mpirun -np 5 --oversubscribe` | 5: 1+1+1+1+0 | 762 ×4, 716 | 22 GB | the same, with a rank holding no tracer (`ntracetloc = 0`) |

Per rank the counts differ by the `emis` events: one per tracer the rank holds per injection
(46 steps). No IO-server ranks (`par.io_ranks = 0`), no OpenMP, one region; Orion's production
layouts have IO ranks: the ifort `pyd2i_*` sets below cover them.

Left beside them, not asserted on and safe to delete: `pyd2_serial.failed-ic`,
`.failed-scaling`, `.failed-scaling-dims`, `.failed-scaling-type` and `pyd2_mpi2.failed-restart`,
`.failed-restart2` -- the partial events of attempts that stopped on setup errors (an Orion IC
path, the scaling file's shape and type, parallel-netCDF restart writes), moved aside rather
than deleted. So is `pydyn_build/tm5-zoom/proj/ct-unified/rc/pyd2_*.rc`, untracked in the clone.

## The zoom-header control arm (laptop)

Made 2026-09-28 for the tm5-zoom rule that a model change is proven bit-identical when off:
`611fb1ca9` (the zoom header in `pydyn_dump.F90`) against its parent `13149b4a1`, each built
from a clean clone (`.tm5-cloud-scratch/pydyn_build/ctl-<sha>`) by `pydyn_build/setup_pyd3ctl.sh`:
serial `ch4_6x4_fwd.rc`, 2000-01-01 00:00-06:00 (the end time set in the rc copy, where pycasso
honours it), gfortran 16, `-O3 -march=native -ffp-contract=off`, four ensemble members.

| set | build | dump | result |
|---|---|---|---|
| `ctl_base_off`, `ctl_patch_off` | `13149b4a1`, `611fb1ca9` | off | the 06:00 restarts equal in all 13 variables, the 00:00 remapped restart too, `region_glb6x4.nc` byte-identical |
| `ctl_base_on`, `ctl_patch_on` | same | on | 3930 files each, the same list; every `.bin` byte-identical; the `.meta` files differ only by the added `status`, `splitorder`, `children` (no footprint: one region) |

## Orion and Hercules

| date | what | host | path | written? |
|---|---|---|---|---|
| 2026-09-22 | CFL cache census, 26 years x 3 grids (`analysis/cfl_cache_census.py`) | Orion | `/work2/noaa/co2/METEO/tm5-nc/ec/ea/cfl/tropo34/` | no — read-only; the script was staged at `/work2/noaa/co2/ash/cfl_cache_census.py` and nothing else was touched. Result: `docs/cfl_headroom.md` §7 |
| 2026-09-22 | the `cfl3x2_2019-*` dump sets below — first model run from this repo | Orion | wrote only under `/work2/noaa/co2/ash/pydyn/` | yes, and **nothing outside that directory**. The rc sets `cfl.cache.read/write : F` and `diffusion.cache.write : F` and points `cfl.cache.dir` into the run dir, so the job has no path to `/work2/noaa/co2/METEO/`; `diffusion.cache.read : T` reads it. Result: `docs/cfl_headroom.md` §8 |
| 2026-09-23 | the `tr3x2_2019-01` dump set below, and the `step_accuracy` batch job that read it (`23883000`) | Orion | same, `/work2/noaa/co2/ash/pydyn/` only | yes, same isolation — the tracer rc inherits every cache key from the `cfl` one. Result: `docs/cfl_headroom.md` §9 |
| 2026-09-24 | `PYD-1` step 1: the §3a/§3d harnesses compiled with ifort five ways and the parity tests run on each (batch jobs `23887649`, `23887807`) — no model run, no dump | Orion | `/work2/noaa/co2/ash/pydyn/pyd1/` only | yes: `pydyn-cf19f3a.tgz` (`git archive` of this repo), `tm5-zoom-kernels-baa64104f.tgz` (the 11 files `fortran/Makefile` compiles, `git archive` at `baa64104f`, md5-equal to the laptop's; the older clone beside it has different `zoom_tools`/`advectm_cfl` and was left alone), one `arm-<flags>/` tree per arm with its build and `pytest.log`, the two job scripts and their `.out`. Result: `docs/state.md` §3c |
| 2026-09-24 | `PYD-1` step 2: three ifort model runs of the `ch4_6x4_fwd.rc` day (the `pyd1-*` sets below) and their replays | Orion | `/work2/noaa/co2/ash/pydyn/pyd1/` only | yes: a fresh clone `pyd1/tm5-zoom` at `baa64104f` (cloned from the §8 clone, advanced by `bundles/tm5-zoom-a9a41f346..baa64104f.bundle`; the §8 clone untouched), per-run rc copies `proj/ct-unified/rc/pyd1-{precise,fast}.rc` in it (untracked), builds and run dirs under `pyd1/runs/`, the dumps under `pyd1/dumps/`, a one-member scaling file per run (`make_scaling1.py`), `pydyn-3a811ed+libm` (this repo's working tree with `use_libm`), the job scripts and outputs. **Caches:** `cfl.cache.read/write : F`, `cfl.cache.dir` into the run dir; the production `dkg` cache was *read* by the first two runs (see below) and nothing was written under `/work2/noaa/co2/METEO/`. Result: `docs/state.md` §3c |
| 2026-09-28 | `PYD-11` gaps 2 and 3: the `tr3x2_2019-10` and `tr3x2_2019-09s` dump sets below (model jobs `23920196`, `23921488`) and five `step_accuracy` batch jobs on them (`23920198`, `-199`, `23920654`, `-655`, `23921523`); and `PYD-8`: `test_simple.py` + `test_kernels.py` on the login node (seconds) with and without NumPy's AVX-512 dispatch | Orion | `/work2/noaa/co2/ash/pydyn/` only: `run_pyd11_oct.sh`, `run_pyd11_spinup.sh`, `pyd11/` (the job scripts, `pydyn-f947f5d+conv.tgz` = this repo's tracked files with the `--convection` edit, unpacked as `pyd11/pydyn`, and `pyd11/pydyn-dkg` with the `--dkg` edit), `runs/ch4u-3x2-pydyntr-2019-{10,09s}/`, `dumps/tr3x2_2019-{10,09s}/` | yes, same isolation as the other 3x2 runs. The rc is an **untracked copy** in the §8 clone, `proj/ct-unified/rc/pyd11-tracer-conv.rc` = `ch4_3x2_pydyn_tracer.rc` at `a9a41f346` with `pydyn.dump.tags : cfl,massflow,bef_x,bef_v` and `restart.write : T` (the fix of `13149b4a1`); no tm5-zoom commit. Result: `docs/cfl_headroom.md` §10 |
| 2026-09-29 | `PYD-3` item 2 and `PYD-11` gap 1: five glb3x2 + nam1x1 model runs (`zm_replay`, `zm_2019-{01,04,07,10}`, jobs `23921802`, `23921818`, `23921953`, `23922016`, `23922059`), the per-event and free-run replays of `zm_replay` (`23921843` failed on the reduced-grid halo, `23921950` after the fix; free run `23922013`, `23922049` superseded, `23922110`) and five census jobs (`23921817`, `23921952`, `23922014`, `23922058`, `23922118`); `PYD-11` gap 4: the `tr3x2_2019-10e` run (`23921774`) and its two `step_accuracy` jobs (`23921799`, `23921800`) | Orion | `/work2/noaa/co2/ash/pydyn/` only: `pyd3/` (a fresh clone `pyd3/tm5-zoom` of the §8 clone advanced by `bundles/tm5-zoom-a9a41f346..397b5a2fa.bundle` to branch `pydyn/zoom-dump`, with the ignored `setup_tm5` symlink recreated; `run_pyd3.sh`, the slurm scripts, `pydyn/` = this repo's tracked files plus the new zoom files, the census csv and job output), `runs/ch4u-3x2nam1x1-pydyn-pyd3/` (one build, every set after the first `--no-compile`), `dumps/zm_*`, and for the sources run `run_pyd11_emis.sh`, `runs/ch4u-3x2-pydyntr-2019-10e/`, `dumps/tr3x2_2019-10e/` and the untracked rc copy `pyd11-tracer-emis.rc` in the §8 clone | yes, same isolation: `ch4_3x2nam1x1_pydyn.rc` inherits every cache key of the tracer rc. Nothing pushed to any remote. Results: `docs/state.md` §3f, `docs/cfl_headroom.md` §11 |
| 2026-09-29 | `PYD-11` where: two `step_accuracy` jobs on `tr3x2_2019-10e` with the worst cell's `(i, j, l)` (`23922289` `--dkg first`, `23922290` `--dkg step`, both `COMPLETED`, EXIT=0) | Orion | `/work2/noaa/co2/ash/pydyn/pyd11/` only: `step_accuracy_emis.slurm`, the two `.out`, the two csv (the earlier `first` csv kept as `*.noloc.csv`); run from `pyd11/pydyn-dkg`, whose `step_accuracy.py` is md5-equal to this repo's | no model run; reads the dump set only. Result: `docs/cfl_headroom.md` §11 |
| 2026-10-09 | `PYD-11` item 2: the paired flask run, two forward runs `2019-10f-z1` / `-z2` (jobs `24009053`, `24009063`), 2019-09-26 .. 2019-11-01, identical but `cfl_max_nloop_z` 1 vs 2; then `obs_departure` scoring: `24009281` (crashed on the flask axis order, fixed in `3b231d0`), `24009288` (the result), and the exploratory `nsamples` split `24009295` (`pyd11/obs_departure_split.py`, untracked) | Orion | `/work2/noaa/co2/ash/pydyn/` only: `run_pyd11_flask.sh`, `runs/ch4u-3x2-pydyntr-2019-10f-z{1,2}/` (flask output under `output/`), `pyd11/pydyn-flask` (= `git archive 3b231d0`), `pyd11/obs_departure.slurm`; in the §8 clone, untracked: `proj/ct-unified/rc/pyd11-flask-z{1,2}.rc` (= `pyd11-tracer-emis.rc` with `pydyn.dump : F`, `output.flask : T`, 4 h walltime; z2 adds the overlay last in `my.source.dirs`) and the overlay `proj/pyd11-zcap2/src/advectm_cfl.F90` (one line, `cfl_max_nloop_z = 2`; checked in both build trees) | yes, same isolation as the other 3x2 runs: CFL cache read and write off, cache dir in the run. Nothing pushed to any remote. Result: `docs/cfl_headroom.md` §12 |
| 2026-10-09 | `PYD-11` item 2, §13: the point-sampled rerun, `2019-10f-z1p` / `-z2p` (jobs `24009391`, `24009407`), identical to the §12 pair but for a sampler overlay; scoring `24009490` | Orion | `/work2/noaa/co2/ash/pydyn/` only: `run_pyd11_point.sh`, `runs/ch4u-3x2-pydyntr-2019-10f-z{1,2}p/`, `pyd11/obs_departure_point.slurm` (writes `obs_departure_2019-10f-point.csv`); in the §8 clone, untracked: `proj/ct-unified/rc/pyd11-point-z{1,2}.rc` (= the `pyd11-flask` rcs with `proj/pyd11-point` last in `my.source.dirs`) and the overlay `proj/pyd11-point/src/user_output_flask.F90` (the effective flask-branch file, strategy 4 `itau_start = itau_end = itau_center`) | yes, as §12: CFL cache read and write off, cache dir in the run. Nothing pushed to any remote. Result: `docs/cfl_headroom.md` §13 |
| 2026-10-13 | `PYD-11` item 5: which arm is closer to the obs -- `analysis/obs_residual.py` (pydyn `cf592e4`, pre-registered at `b61c488`) on the §12 pair and the §13 pair, read-only on their flask output; job `10103585` | Hercules (Orion refused connections that day; same `/work2`) | `/work2/noaa/co2/ash/pydyn/pyd11/` only: `pydyn-cf592e4` (= `git archive cf592e4`), `obs_residual.slurm`, `obs_residual_10103585.out`, `obs_residual_2019-10f{,-point}.csv` | yes, those files only; no model run, nothing pushed. Result: `docs/cfl_headroom.md` §14 |
| 2026-10-13 | `PYD-3` item 3: `replay_zoom --meteo` (every region's Meteo_Setup_Mass, its raw meteo from the 1x1, 137-level set) on `zm_replay` per event (`10105047`) and `--free-run` (`10105048`), and on the `cfl` events of `zm_2019-{01,04,07,10}` (`10105085`..`88`); the four-event fixture runs before them (`10104912` failed: libimf's libintlc not on the path; `10104916` the coarsened-set ingestion, every raw array off; `10105039` bitwise), `10105053` (failed import) and `10105055`, the `cfl`-vs-`meteo` array check; `10105049`..`52` cancelled for a fix, outputs in `pyd3m/superseded/`. No model run | Hercules (Orion was down; `/work2` is shared) | `/work2/noaa/co2/ash/pydyn/pyd3m/` only: `meteo_zoom.slurm` (the model's MKL and libimf from the Orion executable's RUNPATH), `pydyn-wt.tgz` = this repo's tracked files plus `zoom_meteo.py`, `regrid.py`, `tests/test_zoom_meteo.py`, unpacked as `pyd3m/pydyn`, `fixture/` (symlinks to four `zm_replay` meteo events, also copied to the laptop as `pydyn_build/dumps/zm_meteo_fixture`), `cfl_meteo_same.py`, the job outputs | yes, `pyd3m/` only; the dumps and `/work2/noaa/co2/METEO/` read-only. Result: `docs/state.md` §3f |
| 2026-10-13 | `PYD-3` item 4: two glb3x2 + nam1x1 model runs **with emission and chemistry**, `zp_replay` (MPI, 6 compute + 2 IO ranks, job `10105727`, 3.5 min) and `zps_replay` (serial, `10105887`, 3.4 min), 2019-01-01 00-06 h, every tag; their replays (`10105786`/`87` first pass, the free run crashed on the child's unmaintained edge phlb; `10105904`, `10106104`, `10106116`, `10106294` per event; `10106117`, `10106295`, `10106557` free runs that crashed or seeded phlb wrongly; `10106678` the bitwise one), `zm_replay` again with the convection ingestion (`10105788`/`89`, `10105905`, `10106679`), `analysis/ch4_loss_bounds.py` on both sets (`10105903`, `10106111`) and four debug jobs (`pyd4-dbg`) on the temperature and phlb | Hercules | `/work2/noaa/co2/ash/pydyn/pyd4/` only: a fresh clone `pyd4/tm5-zoom` of `pyd3/tm5-zoom` advanced by `bundles/tm5-zoom-397b5a2fa..745e94f70.bundle` to `745e94f70` (branch `pin`), with the untracked `proj/ct-unified/rc/pyd4-phys.rc` (= `ch4_3x2nam1x1_pydyn.rc` without `without_emission` / `without_chemistry`) and the `setup_tm5` link; `run_pyd4.sh` / `run_pyd4s.sh` (Hercules modules, Orion's optim-fast fflags on the command line, serial: `par.mpi=F`, no IO ranks), `replay_zoom.slurm`, `bounds.slurm`, `pydyn/` (this repo's tracked files + `analysis/ch4_loss_bounds.py`), the job outputs; runs under `runs/ch4u-3x2nam1x1-pydyn-pyd4{,s}/`, dumps `dumps/zp_replay`, `dumps/zps_replay` | yes: the rc inherits every cache key of the zoom pydyn rc (cfl cache read/write off, dir in the run dir); `/work2/noaa/co2/METEO/` and the inputs read-only. A 27-event fixture of `zp_replay` copied to the laptop as `pydyn_build/dumps/zp_fixture`. Result: `docs/state.md` §3f |
| 2026-10-13 | `dkg` under ifort: `Calc_Kzz` from the archive against the model's, on `zps_replay`, `zp_replay`, `zm_replay` per event and the `pyd1-precise-nocache` all-files free run (`replay6x4.slurm`, `10107250`: 408 / 408, `in:dkg` 48 / 48); jobs `10107032`..`36`, `10107043`, `10107245`..`49` (the passes as each cause was fixed), the zoom free runs `10107408` (zps) and `10107409` (zm), and `pyd4-dbg` debug jobs (`dbg_dkg*.py`, `dbg_shim.py`, `dbg_powr.py`, `dbg_temper*.py`, `dbg_phlb.py`) | Hercules | `/work2/noaa/co2/ash/pydyn/pyd4/` only: the scripts above, `replay6x4.slurm`, `dbg.slurm`, `pydyn/` refreshed from this repo's tracked files, and `shim/libmathshim.so` -- Calc_Kzz's own Fortran built with ifort `-r8 -O3 -xHost -g -fp-model precise -no-fma -static-intel`, loaded with `RTLD_DEEPBIND` so its libimf calls are not glibc's; it showed `ustr**3` is `__powr8i4` | yes, `pyd4/` only; no model run. Result: `docs/state.md` §3c, §3f |
| 2026-10-13 | The MPI ch4_loss bounds fix, and the zoom free runs on the Python's own `dkg`: the `zp_replay` model run rebuilt with the fix (`10107625`, `run_pyd5.sh` = `run_pyd4.sh` on the `pyd5/tm5-zoom` clone, refusing any diff but the patch); `pyd5/mpi_vs_serial.py` (`cmp.slurm`) on `zp_replay` (`10107626`: differs from the child's first `aft_c` -- the control) and `zpfix_replay` (`10107738`: 720/720 bitwise); `replay_zoom --meteo` per event (`10107739`) and `--free-run` (`10107740`) on it; `ch4_loss_bounds.py` (`10107741` default, `10107779` `--expect fixed`, `10107780` on `zp_replay`); free runs `10107623` (zps), `10107624` (zm) after the coarsen_region sp fix, both bitwise with `in:dkg` 20/20, 40/40; `pyd4-dbg-freesp` (`10107578`, cancelled once it showed the parent's mass equal and its sp not) | Hercules | `/work2/noaa/co2/ash/pydyn/pyd5/` only: `tm5-zoom` (a clone of `pyd4/tm5-zoom`, branch `fix` at `745e94f70`, the patch uncommitted, `pyd4-phys.rc` copied, the `setup_tm5` link), `ch4_loss_bounds.patch`, `run_pyd5.sh`, `mpi_vs_serial.py`, `cmp.slurm`, the job outputs; run under `runs/ch4u-3x2nam1x1-pydyn-pyd5/`, dumps `dumps/zpfix_replay`; `pyd4/bounds.slurm` now passes `$ARGS` | yes, as `pyd4`'s; `/work2/noaa/co2/METEO/` and the inputs read-only. Result: `docs/state.md` §3f |
| 2026-10-13 | `PYD-3` item 5: `replay_zoom --emissions --chemistry --initial` (with `--meteo`) on `zps_replay` per event (`10109108`: crashed on a name clash; `10109151`) and free run (`10109152`: region 1's `m` x halos unseeded; `10109305` bitwise), on `zpfix_replay` per event (`10109306`) and free run (`10109307`), on `zm_replay`'s free run with `--initial` (`10109308`), all bitwise; debug jobs `10109002` (the child's first state against the parent's) and `10109293` (region 1's `m` halos at the start) | Hercules | `/work2/noaa/co2/ash/pydyn/pyd4/` only: `pydyn/` refreshed from this repo, `dbg_ic.py`, `dbg_mhalo.py`, the job outputs; read `/work2/noaa/co2/input/ch4c13/` and `/work/noaa/co2/input/ch4_IC_2026/ch4_init_glb3x2_mmr.nc` | no model run; nothing outside `pyd4/` written. Result: `docs/state.md` §3f |
| 2026-10-14 | `PYD-14`, the restructure: the zoom battery (`runs/hpc/regress_zoom.sh`: `zps`, `zpfix` per event and free run on every input file, `zm` with `--meteo --initial`) on the tree after each step -- r1 runtime/nint `10110135`..`40`, r2 grid `10110226`..`31`, r3 zoom package `10110729`..`34`, r4 input layer `10110839`..`44`, r5 bounds/tests/analysis `10111197`..`202`, r6 (cancelled for a fix) `10111941`..`46`, r7 the one replay `10112072`..`77` -- all 6 / 6 bitwise | Hercules | `/work2/noaa/co2/ash/pydyn/restructure/r1`..`r7` only: each an unpacked copy of this repo's tree and its `regress/` logs | no model run; the dumps and inputs read-only |
| 2026-10-14 | `PYD-2` under ifort: four model runs of the four-tracer `ch4_6x4_fwd.rc` 6 h, serial (`10115792`) and on 2, 3, 5 compute + 2 IO-server ranks (`10115809`, `10115811`, `10115812`), all EXIT=0; then `tests/test_ranks.py` on them (`runs/hpc/ranks.slurm`, `10115856`: 12 / 12, 19 min, 26.8 GB peak; `10115841` stopped on `tm5_312` lacking pytest, `10115855` cancelled before it ran). `pytest` 9.1.1 (with pluggy, iniconfig, Pygments) pip-installed into the shared `miniconda3/envs/tm5_312` env; nothing else in it changed | Hercules | `/work2/noaa/co2/ash/pydyn/pyd2i/` (a clone `tm5-zoom` at `745e94f70`, branch `pin`, with untracked `proj/ct-unified/rc/pyd2i_*.rc` and the `setup_tm5` link; `setup_pyd2i.sh`, `make_scaling.py`, `run_model.slurm`, `submit_when_built.sh`, the setup logs and run outputs; runs under `pyd2i/runs/`), `dumps/pyd2i_*`, `restructure/r8` (this repo's tree for the test job) | yes: `cfl.cache.read/write : F`, its dir in the run; `/work2/noaa/co2/METEO/` and the inputs read-only. Nothing pushed. Result: `docs/state.md` §3e |
| 2026-10-14 | `PYD-2` in a zoom tree: `zpfix_replay`'s run with four tracers, serial and on 2, 3, 5 compute + 2 IO-server ranks. First attempt (`10116180`, `10116214`, `10116216`, `10116235`) with the rc's `parameters.constant.value : 1.0`: four identical tracers; its replays and rank tests (`10116377`..`81`) passed all but `test_tracers_are_split`, which failed on all three; the probe `10117370` (`pyd6/same_tracers.py`) showed every tracer pair equal. Rerun without the key (`10117386`, `10117392`, `10117408`, `10117418`, each `read_parameters` from the run's own file): serial replays `10117848` (per event) and `10117849` (free run), bitwise; `ranks.slurm` with `--mem=400g` per set, `10117850`..`52`, 4 / 4 each (53, 68, 100 min; 190 GB peak on mpi5) | Hercules | `/work2/noaa/co2/ash/pydyn/pyd6/` (clone `tm5-zoom` of `pyd5`'s, `745e94f70` + `ch4_loss_bounds.patch`, untracked `proj/ct-unified/rc/pyd4-phys.rc` and per-run `pyd6-zp4_*.rc`, the `setup_tm5` link; `setup_pyd6.sh`, `make_scaling.py` (date argument), `check.sh`, `same_tracers.py`, setup logs, `first-attempt/`), runs `runs/ch4u-3x2nam1x1-pydyn-zp4_*/`, dumps `dumps/zp4_*`; the first attempt's dumps kept as `dumps/zp4_*.failed-constant` (155 GB, not asserted on, safe to delete); job logs in `restructure/r8/log/` | yes: the zoom rc's cache keys write nothing; the scaling file in each run's `output/` (`parameters.input.dir` pointed there, not at `input/ct-uni/scaling`); `/work2/noaa/co2/METEO/` and the inputs read-only. Nothing pushed. Result: `docs/state.md` §3e |
| 2026-10-14 | `PYD-15`, speed: a cProfile of `zps_replay` per event and free run on every input file with the r8 code (`10118928`, `10118929`; `10118926`/`27` cancelled before they ran, `replay.slurm` cannot take a module with arguments: `pyd6/prof.slurm` can) -- 871 s, 59 % `np.fromfile`, 22 % `chem.rebin`; then the zoom battery on the faster code (`regress_zoom.sh`, per-event jobs `-c 6`: `10119174`..`79`, 6 / 6 bitwise), `zp4_serial` per event `-c 6` (`10119180`) and free run (`10119181`), bitwise, and `ranks.slurm` on `zp4_mpi5` with `-c 6` (`10119182`, `--mem=64g`: 4 / 4, 61 min, at the 64 GB limit with a 256-event release window; `10119618`, `--mem=32g` with 32: 4 / 4, 63 min, MaxRSS at its 32 GB cap, not killed) | Hercules | `/work2/noaa/co2/ash/pydyn/restructure/r8/log/prof-*` and `restructure/r9` (this repo's tree with the speed changes, `replay.py` then updated in place to the 32-event window) with its `regress/` and `log/`; `pyd6/prof.slurm`, `pyd6/top.py` | no model run; the dumps and inputs read-only |
| 2026-10-14 | `PYD-15`, x advection by latitude: the zoom battery (`10119836`..`41`, 6 / 6 bitwise) and `zp4_serial` per event (`10119842`) and free run (`10119843`), bitwise, on the batched x rows | Hercules | `/work2/noaa/co2/ash/pydyn/restructure/r10` (this repo's tree) with its `regress/` and `log/` | no model run; the dumps and inputs read-only |
| 2026-10-17 | `PYD-16`, the GPU port on the zoom sets: `pydyn.gpu.check` (the reference and the `emu` device replay, `--lapack twin --libm twin` on both, every checkpoint's arrays fingerprinted) through `runs/hpc/gpu_check.slurm`: free runs on every input file of `zps_replay` (`10150491`, 440 checkpoints, 9 min, 15 GB), `zpfix_replay` (`10150492`, 440, 38 min, 31 GB), `zp4_serial` with `--meteo --chemistry --initial` (`10150494`, 440, 12 min), `zm_replay` with `--meteo --initial` (`10150493`, 320, 21 min), and `zps_replay` per event (`10150495`, 440, 7 min); 0 differing fingerprints in each. Also on the login node, compile only: `all.cu` by nvcc 12.1.1 for sm_80, `--fmad=false` and default, to count FMA in the PTX (0 and 388) | Hercules | `/work2/noaa/co2/ash/pydyn/gpu1/` only: `tree/` (this repo's working tree before its lint pass, which only reflowed and renamed), `log/`, `all_fmad_{false,true}.ptx` | no model run; the dumps and inputs read-only; the `emu` library built under `tree/pydyn/gpu/__pycache__` |
| 2026-10-17 | `PYD-16`: thirteen more days of the laptop meteo for longer local runs -- 2000-01-02..14 of the archive's `h06h18tr3/tropo34/glb600x400` (mfuv mfw sp t q convec tsp), `h06h18tr1/sfc/glb100x100` (u10m v10m sshf slhf ewss nsss) and `an0tr1/sfc/glb100x100` (sr): the files `pydyn.meteo` reads. Packed per day on Hercules, 13 x 65 MB, each md5-verified in transit, unpacked into the laptop's `.tm5-cloud-scratch/METEO/tm5-nc` without overwriting (the day-2 model-level files already there are byte-identical); the 91 surface files match their `.md5` sidecars. All 333 hourly read intervals 2000-01-01 00 h .. 01-14 21 h balance (Meteo_Setup_Mass, one try each) with every field finite. No model run | Hercules | `/work2/noaa/co2/ash/pydyn/gpu1/meteo_pull/` (the 13 tars); `/work2/noaa/co2/METEO/` read-only | yes, `gpu1/meteo_pull/` only |
| 2026-10-18 | `PYD-16`: how fast the meteo archive reads on Hercules, for `meteo.Stager` -- `probe.py` (job `10158109`, hercules-01-01): a day's files (13: mfuv mfw sp t q convec, six 1x1 surface fields, sr) read record by record as `pydyn.meteo` does, cold from `/work2` 7.0 s (glb6x4, 65 MB) and 2.4-2.7 s (glb3x2, 102 MB), again 2.8 / 1.1-1.3 s; another day copied whole to the node's `/local` 1.2 / 1.6 s (53-62 MB/s; `/dev/shm` 77 MB/s), read from the copies 0.45 / 1.0 s. `copies.py` (`10158158`): glb3x2 days copied by 1, 4, 8, 16 threads at 90, 409, 617, 1013 MB/s. No model run | Hercules | `/work2/noaa/co2/ash/pydyn/gpu1/stage_probe/` (the two scripts, their job files and logs); `/work2/noaa/co2/METEO/` read-only, the copies on the node's local disk removed | yes, `gpu1/stage_probe/` only |

### The glb3x2 CFL dump sets

| set | made | host | model sha | rc + overrides | build flags | events | size | what it proves |
|---|---|---|---|---|---|---|---|---|
| `cfl3x2_2019-{01,04,07,10}` | 2026-09-22 | Orion, ifort, 6 MPI compute + 2 IO ranks | `tm5-zoom@0e72e4cf8`, `NOAA/ct-unified` | `ch4_3x2_pydyn_cfl.rc`, 4 x 7 days of 2019, glb3x2 120×90×34, `nred = 14`, `ct.cachegen : T` so `ntrace = ntracet = 0`, `time.ntimestep = ndyn_max = 3600`, `pydyn.dump.tags : cfl` | `optim-fast` = `-O3 -xHost -g -fp-model precise -no-fma` (`compiler-ifort-orion.rc:62`) | 168 each, 672 total | 6.4 GB each | `Check_CFL` reproduced exactly on 672/672 intervals, and what other sub-cycle caps would have chosen. `analysis/cfl_cap_replay.py`; results in `results/cfl3x2_2019_cap_replay.csv` |

| `tr3x2_2019-01` | 2026-09-23 | Orion, ifort, 6 MPI compute + 2 IO ranks | `tm5-zoom@a9a41f346`, `NOAA/ct-unified` | `ch4_3x2_pydyn_tracer.rc`, 2019-01-01..02, glb3x2 120×90×34, **`ct.cachegen : F` so `ntracet = 1`**, `without_emission without_chemistry`, `pydyn.dump.tags : cfl,massflow,bef_x` | as above | 48 cfl, 80 massflow, 160 bef_x (rank 0) | 12 GB | that the longer step conserves mass, stays positive, and costs 12.9 ppb in the worst cell. `analysis/step_accuracy.py`; results in `results/step_accuracy_3x2.csv` |
| `tr3x2_2019-10` | 2026-09-28 | Orion, ifort, 6 MPI compute + 2 IO ranks (job `23920196`, 4 min, `COMPLETED`) | `tm5-zoom@a9a41f346` + the untracked rc copy above | `pyd11-tracer-conv.rc`, 2019-10-01..02 (the hardest two days of §8's October window: 47 of 48 intervals reduced, 18 of those recovered at `(10,1,2)`), otherwise as `tr3x2_2019-01`; `pydyn.dump.tags : cfl,massflow,bef_x,bef_v`, `max_events 3000` | as above | 48 cfl, 95 massflow, 190 bef_x, 190 bef_v (rank 0; the 95 steps are §8's census for these days exactly) | not measured: all six ranks kept, `du` does not finish in 5 min there | the §9 measurement on October, transport-only and with convection. `dkg` in its `bef_v` events changes from one dynamics step to the next inside a reduced interval (up to half its maximum), as live `Calc_Kzz` on each step's fluxes would -- the rc sets a `dkg-cache` sourcekey only for glb100x100 -- while the convective fluxes and cloud levels do not change |
| `tr3x2_2019-09s` | 2026-09-28 | same (job `23921488`) | same | the row above with `PYDYN_TAG=2019-09s`, 2019-09-30 00:00 .. 2019-10-01 06:00 (`run_pyd11_spinup.sh`) | as above | 30 cfl, 60 massflow, 120 bef_x, 120 bef_v (rank 0; every interval at 1800 s) | not measured | the spin-up control for `tr3x2_2019-10`'s first hour: 2019-10-01 00:00 measured 24 h after the initial field instead of at it, `docs/cfl_headroom.md` §10 |
| `tr3x2_2019-10e` | 2026-09-29 | same (job `23921774`, 7 min) | same | `pyd11-tracer-emis.rc` = `pyd11-tracer-conv.rc` with `without_emission` dropped (constant scaling 1.0 on the CT-CH4 priors), 2019-09-29 00:00 .. 2019-10-03 00:00, `run_pyd11_emis.sh` | as above | 96 cfl, 1026 in all (rank 0) | not measured | `PYD-11` gap 4: the same 48 October intervals as `tr3x2_2019-10`, on a field with 48 h of real sources (tracer mass +7.5 Tg over the 4 days); measured with `--since 2019100100` |
| `zm_replay` | 2026-09-29 | Orion, ifort, 6 MPI compute + 2 IO ranks (job `23921802`, 4 min) | `tm5-zoom@397b5a2fa` (`pydyn/zoom-dump`) | `ch4_3x2nam1x1_pydyn.rc`, glb3x2 (reduced, 14 bands) + nam1x1 (tref 2), 2019-01-01 00:00-06:00, `PYDYN_TAGS=cfl,massflow,meteo,m2phlb,bef_x,aft_x,bef_y,aft_y,bef_z,aft_z,bef_v,aft_v` | as above | 564 (rank 0; all six ranks kept) | not measured | the zoom tier 2: every operator event and Setup_MassFlow per region, per event and in the free run (`docs/state.md` §3f) |
| `zp_replay` | 2026-10-13 | Hercules, ifort 2024.1, 6 MPI compute + 2 IO ranks (job `10105727`, 3.5 min) | `tm5-zoom@745e94f70` + the untracked `pyd4-phys.rc` | glb3x2 (reduced) + nam1x1, emission and chemistry on (CT-CH4 defaults), 2019-01-01 00:00-06:00, every tag incl. `bef_s/aft_s/bef_c/aft_c/emis/chem` | Orion's `-O3 -xHost -g -fp-model precise -no-fma` on the command line, MKL 2022.2.1 | 924 (rank 0; all ranks kept) | not measured | `PYD-3` item 4 per event; the child's `c` carries the MPI ch4_loss bounds bug (`docs/state.md` §3f) |
| `zps_replay` | 2026-10-13 | Hercules, ifort 2024.1, serial (job `10105887`, 3.4 min) | as `zp_replay` | as `zp_replay`, `par.mpi=F`, no IO ranks | as `zp_replay` | 924 | not measured | `PYD-3` item 4: every operator of both regions bitwise per event and in the free run, the control of the bounds analysis |
| `zpfix_replay` | 2026-10-13 | Hercules, ifort 2024.1, 6 MPI compute + 2 IO ranks (job `10107625`, 3.7 min) | `tm5-zoom@745e94f70` + `pyd5/ch4_loss_bounds.patch` (the MPI ch4_loss pointer remapped to the serial bounds) + the untracked `pyd4-phys.rc` | as `zp_replay` | as `zp_replay` | 924 | not measured | the fix: bitwise `zps_replay` in every tracer array of 720 operator events, every operator bitwise per event and in the free run |
| `zm_2019-{01,04,07,10}` | 2026-09-29 | same (jobs `23921818`, `23921953`, `23922016`, `23922059`, 8-9 min each) | same | the same rc, `PYDYN_TAGS=cfl`, the four windows of `cfl3x2_2019-*` | as above | 336 each (168 intervals × 2 regions, rank 0; ranks 1-5 pruned after checking one event per window md5-equal on all six) | not measured | `Check_CFL` with the child: the census of `cfl_headroom.md` §11 |

**`tr3x2_2019-01`'s job ended `FAILED` and its data is complete anyway.** The model reached
its end date and then escaped on `io_save: requires with_hdf4` — an inherited
`restart.write : F` that stops being right at `ntracet > 0`, fixed in
`tm5-zoom@13149b4a1`, which is why the row above pins `a9a41f346` and not the fix. Verified
before use: 48 distinct hourly `idate`s covering both days, and all 5152 rank-0 arrays
exactly the size their `.meta` declares. Only rank 0 holds the tracer here — the other five
have `ntracetloc = 0` and a zero-length `rm` — so `--rank 0` is not an optimisation, it is
correctness.

`cfl3x2_smoke` (934 MB, 2019-01-01 only) sits beside them: the one-day run that checked the
rc and the tag filter before the windows were launched. It is a subset of `cfl3x2_2019-01`
and nothing asserts on it.

These live at `/work2/noaa/co2/ash/pydyn/dumps/` on Orion, not on this laptop. Only rank 0
is kept: every compute rank writes a byte-identical `cfl` event (md5 of `m.bin` equal across
all six, same `ndyn_out`), so `_p001`..`_p005` are deleted as soon as a window finishes and
a window is 6.4 GB instead of 38 GB. The tree that produced them is
`/work2/noaa/co2/ash/pydyn/tm5-zoom`, a clone of `~/tm5-zoom` with this repo's tm5-zoom
commits fetched from a bundle — it is not `~/tm5-zoom` and nothing was pushed to a shared
branch.

### The ifort `ch4_6x4_fwd.rc` sets (`PYD-1` step 2)

| set | made | host | model sha | rc + overrides | build flags | events | size | what it proves |
|---|---|---|---|---|---|---|---|---|
| `pyd1-precise` | 2026-09-24 | Orion, ifort 2021.12.0, 1 MPI compute + 2 IO-server ranks (job `23888162`) | `tm5-zoom@baa64104f` | `ch4_6x4_fwd.rc` copied as `pyd1-precise.rc`: 2000-01-01 00:00 to 01-02 00:00, `ct.ensemble.total_members/size 1`, `par.compute_ranks 1`, `cfl.cache.read F`, `pydyn.dump.max_events 3000`; the rc's `dkg-cache:` sourcekeys left on, so `dkg` is the production cache | Orion's `optim-fast` = `-O3 -xHost -g -fp-model precise -no-fma`, `-qmkl=sequential` (MKL 2022.2.1), `with_lapack` | 792 | 5.0 GB | the ifort model replays bitwise per event and in `--free-run` (408 / 408) given MKL and libimf; its `dkg` is the cache's |
| `pyd1-precise-nocache` | 2026-09-24 | same executable (job `23888224`) | same | `pyd1-precise-nocache.rc`: the above with both `tmm.sourcekey.glb6x4.*.diffus` pointed at an empty directory, so live `Calc_Kzz` runs (24 cache misses) | same | 792 | 5.0 GB | the same 408 / 408, and the all-files free run bitwise in every input but `dkg` (557 ulp); with the set above, cache vs live `dkg` |
| `pyd1-fast` | 2026-09-24 | same host type (job `23888163`) | same | as `pyd1-precise`, runid `pyd1-fast` | Hercules' `optim-fast`: `-O3` (ifort `-fp-model fast`), same MKL | 792 | 5.0 GB | what the fast flags cost: 171 of 408 per-event checkpoints off, BalanceMassFluxes to 7e-10 |

All three run the executable directly (`run_model.slurm`), as the laptop sets were: pycasso's
job first runs the EnKF cold start, which refuses fewer than three members (job `23888071`
failed there), and a one-member run still reads a scaling-factor file (job `23888113`
stopped on its absence after 12 events, which were deleted). A first `nocache` attempt
(`23888196`) set `diffusion.cache.read : F`, which is retired and changed nothing; its dumps
were identical to `pyd1-precise`'s in `dkg` and were replaced.

### The ifort `PYD-2` sets (`pyd2i_*`)

Made 2026-10-14 by `pyd2i/setup_pyd2i.sh <label> serial|mpi N`: a per-run copy of
`ch4_6x4_fwd.rc` with 2000-01-01 00:00 to 06:00, `ct.ensemble.total_members 4`, the CFL cache off
and its dir in the run, built by `setup_tm5` with Hercules' modules (ifort 2024.1, Intel MPI
2021.12, MKL 2022.2.1) and Orion's `optim-fast` fflags `-O3 -xHost -g -fp-model precise -no-fma`
on the command line, `pydyn.dump.max_events 3000`; the four members' scaling from
`make_scaling.py`, as the laptop sets, so the tracers differ. Run directly (`run_model.slurm`,
`srun` of the executable), as the `pyd1-*` sets were.

| set | ranks, slabs | events | size | what it proves |
|---|---|---|---|---|
| `pyd2i_serial` | `par.mpi=F`: 4 | 234 | 2.7 GB | the reference |
| `pyd2i_mpi2` | 2 compute (2+2) + 2 IO-server | 210 per rank | 3.5 GB | merged, bitwise the serial set event for event; replays bitwise |
| `pyd2i_mpi3` | 3 (1+1+2) + 2 IO | 198, 198, 210 | 4.2 GB | the same, uneven slabs |
| `pyd2i_mpi5` | 5 (1+1+1+1+0) + 2 IO | 198 ×4, 186 | 5.8 GB | the same, a compute rank with no tracer |

Six steps of 1 h: 12 of each operator call, 48 `emis` (12 per tracer), 6 `cfl`/`massflow`/`meteo`.
The IO-server ranks dump nothing; the per-rank counts differ by the 12 `emis` events per tracer
held.

### The ifort zoom `PYD-2` sets (`zp4_*`)

Made 2026-10-14 by `pyd6/setup_pyd6.sh <label> serial|mpi N`: `zpfix_replay`'s recipe
(`pyd4-phys.rc`, glb3x2 + nam1x1, emission and chemistry, 2019-01-01 00-06 h, every tag, the
`ch4_loss` fix) in a per-run rc copy with `ct.ensemble.total_members 4`, `parameters.input.dir`
in the run tree and **no `parameters.constant.value`** (with it every member is scaled alike),
the scaling file from `pyd6/make_scaling.py` (`pyd2i`'s, for 2019-01-01). Hercules' modules and
Orion's `optim-fast` fflags; submitted by pycasso.

| set | ranks, slabs | events | size | what it proves |
|---|---|---|---|---|
| `zp4_serial` | `par.mpi=F`: 4 | 1104 | 26 GB | the reference; replays bitwise per event and free run on `--meteo --chemistry --initial` |
| `zp4_mpi2` | 2 (2+2) + 2 IO | 984 per rank | 33 GB | merged, bitwise the serial set; replays bitwise |
| `zp4_mpi3` | 3 (1+1+2) + 2 IO | 924, 924, 984 | 41 GB | the same, uneven slabs |
| `zp4_mpi5` | 5 (1+1+1+1+0) + 2 IO | 924 ×4, 864 | 55 GB | the same, a compute rank with no tracer |

Per rank the counts differ by 60 `emis` events per tracer held (both regions' injections).

### Analysis jobs

| job | what | wall | why batch |
|---|---|--:|---|
| `23887649` | `pyd1_parity.slurm`: ifort builds at `-O0`/`-O3 -fp-model strict` and Orion's `optim-fast`, `test_parity.py` + `test_zoom.py` on each — 188/188 bitwise three times | 11 min | built on the compute node so `-xHost` is its CPU's |
| `23887807` | `pyd1_fast.slurm`: the same at `-O3` (Hercules' `optim-fast`) and `-O3 -xHost`, `-fp-model fast` — 94 and 152 of 188 fail, by a few ulp | 9 min | same |
| `23888174`, `23888175` | `replay.slurm` on `pyd1-precise` / `pyd1-fast`, glibc's libm: per event, `--free-run`, all-files free run. Precise: everything bitwise but the chemistry (2–3 ulp) | 8, 9 min | three passes over 5 GB |
| `23888207`–`09`, `23888243` | `replay_imf.slurm` (`--libm` = the model's libimf) on the three sets: `pyd1-precise` and `-nocache` 408 / 408 per event and free run | 7–11 min | same |
| `23883000` | `analysis/step_accuracy.py` over `tr3x2_2019-01`, 48 intervals x 4 step sizes | 50 min | `step_accuracy.slurm` in the scratch tree. An earlier attempt ran 100 min on a login node — antisocial, and a symptom: it was reading all six MPI ranks when only rank 0 holds the tracer |

## Alpine (CU Boulder; copies, no sets made)

No model runs on Alpine: it holds byte-identical copies of four laptop sets, each matched by
file count and an md5 of every file (2026-10-20), and 2 days of glb3x2 meteo, used for the
device tests on an A100 slice and a whole A100 (`runs/alpine/README.md`).

| copy | of | files | jobs | result |
|---|---|---|---|---|
| `/scratch/alpine/$USER/pydyn/dumps/repin_{phys,free,cfl}_745e94f70`, `zp_fixture` | the laptop sets above | 13,686 / 9,132 / 417 / 508 | hold 34705941 (c3gpu-c2-u13, `a100_3g.20gb`) | `test_gpu.py` 27 passed, 4 skipped (no NVPTX clang; the emu check); the three device free runs bitwise |
| the four sets above, as `/projects/$USER/pydyn/stage` tars | the copies above | -- | `full.slurm` 34711564 (c3gpu-a9-u31-1, a whole A100 80GB PCIe) | `test_gpu.py` 37 passed, 4 skipped; the free runs bitwise; `pydyn.gpu.bench` in `docs/gpu_port.md` §5 |
| `/scratch/alpine/$USER/pydyn/METEO/tm5-nc` | 2000-01-01..02, the January monthly file, lsm, oro (+ `.md5`) | 60 | same | input to the above |
| — | — | — | 34701177, and the held run | `test_production`'s meteo / `v` checks fail at 1e-12..2e-10 relative: they are bitwise to the Mac's FFTW and LAPACK, not Alpine's |

Not on Alpine: the emissions, the chemistry inputs, the IC file, a tm5-zoom checkout, the Fortran
harnesses.

## What is still gfortran

The laptop gfortran sets carry the transport-only `free` and `cfl` sets; the `PYD-2` rank merge is
on both compilers (`pyd2_*` here, `pyd2i_*` under ifort with IO-server ranks, `zp4_*` in a zoom tree). The ifort arm is the `pyd1-*` glb6x4 sets (Orion's production flags,
bitwise with the model's MKL and libimf) and the zoom sets (`zm_replay`, `zps_replay`,
`zpfix_replay`; Hercules' ifort 2024.1 at Orion's flags, `zpfix_replay`'s 6 compute + 2 IO ranks
equal to its serial run). The ifort sets other than `pyd2i_*` have one tracer; and
Hercules' own flags (bare `-O3`) are measured only in the `pyd1-fast` set, where they are not
bitwise. The census rows (`cfl3x2_*`, `tr3x2_*`, `zm_2019-*`) carry findings and `Check_CFL`'s
integer decisions, not array-level bitwise claims.
