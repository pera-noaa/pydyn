# TODO: pydyn, the serial Python reference of the TM5 dynamic core

Items use the prefix `PYD-N`. Nothing here was forwarded from a tm5-zoom item: the work ran
through commits in `tools/pydyn` and never had a number. If one is opened there that overlaps,
keep the origin in the title, `PYD-n. (was tm5-zoom #N) ...`, and leave the tm5-zoom item alone.

Source repo for the pointers below: tm5-zoom, branch `NOAA/ct-unified`. Most were
read at `4dc2017c5` and are cited that way; anything added since carries its own sha, and the
repo's current pin is `745e94f70` (`docs/sources.md`). Cite the sha you read at, never `HEAD`
— that repo's history gets rewritten.

Tags are the tm5-zoom convention: `[U:urgency F:confidence E:S|M|L]`. An item moves to
**Landed** when it is done and to **Parked** when it is deliberately not being done; nothing
is deleted, and a landed item keeps whatever a future reader needs to not redo the thinking
— the mechanics live in the code it landed, not here, and the full text an item had when it
landed in `TODO.history.md`.

## Open

### PYD-17. The public release  [U:3 F:4 E:M]

`pera-noaa/pydyn` (private until the owner says otherwise) is the release, one commit per
export of this repo's HEAD, made by `runs/export_public.sh`; its header says what it leaves
out and what it refuses. Apache-2.0 (`LICENSE`, `NOTICE`), `CITATION.cff`, `CONTRIBUTING.md`; installed with uv,
not on PyPI. The three dump sets a reader without TM5 can test against are the Zenodo fixture
[10.5281/zenodo.23955885](https://doi.org/10.5281/zenodo.23955885) (`runs/MANIFEST.md`), each
with the configuration that made it. Open, in order:

1. Publish the Zenodo record (the owner's click), then fetch it back with
   `runs/fetch_fixture.py` and diff against the sources, and drop "a draft until the v0.1.0
   release" from `README.md` ("What runs where") and `docs/state.md` §3b.
2. CI on the fixture: macOS arm64 with Homebrew FFTW running the 8 fixture tests and failing on
   any skip (the platform they are bitwise on); Linux running the device free runs.
3. Recreate `pera-noaa/pydyn` with one fresh commit (an earlier export's history named a
   cluster username, and GitHub keeps pull-request refs the owner cannot delete): delete the
   repository, create it empty, push the prepared single-commit `main`.
4. Make the repository public (the owner's decision); turn on private vulnerability reporting
   (`SECURITY.md` points at it); tag `v0.1.0`, date the `CHANGELOG.md` entry, add
   `date-released` to `CITATION.cff`, and a GitHub release (a software DOI if Zenodo's GitHub
   integration is on).

Undecided, the owner's: whether `analysis/` and `results/` (tm5-zoom investigations, two tests
use them) ship publicly or stay in the lab; whether outside readers get GitHub Issues in place
of this file; the default paths in `tests/common.py` assume the maintainer's layout
(`.tm5-cloud-scratch`), which the `PYDYN_*` variables override.

### PYD-5. The isotope and `with_budgets` chemistry paths  [U:2 F:3 E:M]

`chem.py` is the CH4-only path of `ch4_chemistry_step`. The d13C / δD tracers and the budget
accumulation are not transcribed, so a dump from `ct-ch4-compare`'s dual-tracer runs cannot be
replayed. Scope it against that project's rc before starting; it may be cheaper to wait until
its chemistry stops changing.

### PYD-7. Make the reference usable by a port without reading its tests  [U:3 F:2 E:M]

The two plausible consumers are the GPU-native transport work (`tm5-zoom@4dc2017c5:proj/gpu-native/`,
a separately owned workstream — coordinate, do not edit it) and the tm5-mp port. Both want the
same thing and neither wants pytest: a golden-case format (inputs, expected outputs, tolerances)
that any language can read, generated from the dump sets. `pydyn.harness.write_case` and
`pydyn.layout` are most of it already. Ask the owner what shape they want before building it.

**Proposal, 2026-10-14 (not agreed; the questions below are the owner's).** The dump format is
already language-neutral: one text `.meta` per event (`tag`, `idate`, grid scalars, and an
`array <name> <bounds>` / `iarray` line per array) beside one raw little-endian Fortran-order
`.bin` per array. So a golden case is a `bef_<op>` / `aft_<op>` event pair cut from a set, plus a
`case.toml` naming the operator, the arrays that are inputs and outputs, and the tolerance;
`harness.write_case` already writes the kernel-call half for x, y, z, dynam0 and convec. A
generator `python3 -m pydyn.cases <set> --op v --events N --out DIR` (filter, copy, manifest,
md5) is a few hundred lines; no new numerics. Questions:

1. Which operators: `gpu-native` Phase 1 is K4 (the `convec` inverse), so `bef_v`/`aft_v`; tm5-mp
   would want whole steps.
2. Which shapes: glb6x4 (laptop sets), glb3x2 + nam1x1 (`zps_replay`, real 1x1 columns), or
   production 1x1 (not dumped yet).
3. What "equal" means: bitwise for `CONVEC_GPU_K4_SHARED`, and for the cuBLAS path the
   residual and column-sum identity its `ROADMAP.md` Phase 1 judges by -- which needs the matrix itself
   dumped, not just `rm` before and after.
4. Where they live: `/work2` (a case is ~1-50 MB) or a tarball in the consumer's repo.

### PYD-16. Run the GPU port on a GPU, and time it  [run and timed 2026-10-21: A100, MI100]  [U:3 F:3 E:S]

Done: float64 bitwise to the reference on an NVIDIA A100 (a 3g.20gb MIG slice and the whole
card) and an AMD MI100 (CuPy for HIP, `-ffp-contract=off`): `tests/test_gpu.py` with the three
free runs. The production interval as a graph runs at 41.2 (slice), 60.1 (whole A100) and 29.6
(MI100) simulated years an hour (`docs/gpu_port.md` §5, §10). The full earlier text, with the
recipe that was run, is in `TODO.history.md`. Open:

1. A run from the files at production size (§7): glb3x2 + nam1x1 meteo is not local; the
   reading runs in `--readers` processes (§6), on an HPC node from `--stage` copies on its
   local disk (`/work2` read directly: 2.4-7 s a simulated day, copied by 8 threads 0.17 s;
   §6), and the device's interval is GPU-bound there (box dispatches, convection's passes, z,
   loss rates: §7). float32 chemistry's `rm * decay` rounding (-4.8e-8 of the tracer a day,
   §8) is measured, not fixed.
2. `forward.py`'s intervals as graphs: only a graph uses a whole A100 (`docs/gpu_port.md` §7).

## Landed

### PYD-15. Speed up the replay without touching its arithmetic  [DONE 2026-10-14]  [U:2 F:4 E:M]

**Done.** Each change moves data or calls the same functions differently, never reorders a sum,
and is checked against the dumps: the local suite (432), sequential and `--workers 4` per-event
replays identical in every report, and the Hercules zoom battery 6 / 6 bitwise.

| change | where | effect |
|---|---|---|
| per event in `--workers` processes, a run of read intervals each | `replay._Session`, `replay.slurm` `-c N` | zoom battery per-event jobs 10.9 -> 2.0-2.5 min |
| arrays dropped 32 events behind, a thread reading 16 ahead | `dump.Event.release`, `replay.run` | the dump reads (59 % of a zoom replay on `/work2`) overlap the arithmetic; zoom free runs 10.9 -> 2.8-6.9 min |
| rank tests compare one event at a time | `tests/test_ranks.py` | a `zp4` set 190 GB / 100 min -> fits 32 GB, ~1 h; the laptop's three sets 9 -> 4 min |
| `np.pad` halos -> preallocated slices | `ops.wrap`, `ops.closed` | 6 µs -> 1 µs a call, 500 k calls a day |
| `rebin` over the overlapping layers only, shared by the three zonal columns | `chem.rebin_many` | `loss_rates` 0.20 -> 0.07 s an event |
| libm over whole arrays through a C loop calling the library's own function | `ops.libm` | 10-30x per call; NaN, -inf and 0-d results as before |
| x advection by latitude: every level of a row at once, the rows' sub-cycle searches together, then each group of one `nloop` stepped together (region 1, a child, and Check_CFL's mass-only passes) | `kernels.subcycle_rows`, `_advect_rows_x`, `zoom.advect._rows_child`, `cfl.dynamum`, `zoom.check_cfl` | x 12.3 -> 1.2 s and Check_CFL's x 1.7 -> 0.15 s in a glb6x4 day; identical to the per-row code with 1 to 41 sub-cycles a row and on the limit's error |

Measured from a cProfile of `zps_replay` on Hercules (`runs/MANIFEST.md`): 871 s, of which
`np.fromfile` 59 %, `rebin` 22 %, the meteo reads 6 %, libm 4 %. A glb6x4 day free-run on every
input file, laptop: 42 -> 10.7 s. On Hercules the zoom jobs are bound by `/work2` reads, whose
speed varies run to run (the same zps free run 2:48 and 6:49). Jobs: `runs/MANIFEST.md`.

**Residue, not done:** the dump reads themselves, which only fewer or larger files would shrink.

### PYD-14. Restructure the codebase  [DONE 2026-10-14]  [U:2 F:4 E:L]

**Done**, every step pure code motion and checked against the model's own dumps (the local suite
and the Hercules zoom battery `runs/hpc/regress_zoom.sh`, bitwise 6 / 6 after each step):
- **One replay.** `pydyn/replay.py` is the engine for one region and for a tree; a single region
  is a tree of one (a set from before the zoom header gets its split-order positions inferred).
  The single-region engine, `FreeRun` and per-event replay it replaced are gone; the zoom free
  run now runs Check_CFL itself. `replay_zoom.py` is a shim for the command lines on record.
- **One input layer** (`inputs.py`): `RegionInputs` per region, `Ingestion` over a tree; the
  meteo set the rc reads (`--meteo-grid model|1x1`).
- **One FillGrid** (`grid.py`: `LL`, `relate`, `block_sum`/`block_mean`, `fill`), one library
  selection (`runtime.py`, `--libm` on every replay), one bounds table (`layout.DECLARED`).
- **`pydyn/zoom/`** a package of eight modules; the tracer and Check_CFL advection share their
  scope selection and sub-cycle searches.
- **Tests** by subject, every path in `tests/common.py`; **analysis/** a package.
- Found on the way: `cfl.check_cfl` rounded the step count half-to-even (Fortran `nint`, fixed,
  unreachable in practice); `regrid.fill_grid`'s stride bug for a coarse window off the dateline
  (latent, fixed); `netCDF4` missing from the dependencies; ruff's 19 findings.

### PYD-3. Zoom nesting  [DONE 2026-10-13]  [U:3 F:3 E:L]

**Done** (`docs/state.md` §3d, §3f): the production glb3x2 + nam1x1 zoom, built with ifort at
Orion's flags, is bitwise per event and in a free run that reads nothing but the model's input
files, in every array including the halos. In the order `PYD-11` needed it:

1. `Check_CFL`'s zoom path: tier 1 on four trees, then the model's `(ndyn, n)` on 672 real
   intervals (`cfl_headroom.md` §11).
2. A zoom dump set and a replay that learns regions: the dump header's `status`, `splitorder`,
   footprint and `children` (tm5-zoom `84842782e`, `20ffeaed0`), `pydyn/replay_zoom.py`, which
   refuses a tree that is not the model's. Found a transcription bug tier 1 could not see (the
   reduced rows' x halo after `red2uni`).
3. A child's meteo (`zoom_meteo.py`, `regrid.py`): the zoom rc reads the 1x1, 137-level set and
   regrids it in memory, and a child is balanced per parent cell.
4. Convection, `s` and `c` on a child, `dkg` included. **Found a model bug**: an MPI build's child
   decayed CH4 with the loss rates of the cell (3, 2) away (`c_f_pointer` bounds); fixed in
   tm5-zoom `0385efe7e`, verified on a rebuilt MPI run equal to the serial one.
5. The inputs of `s` and `c` and the initial field per region (`ZoomInputs`): the free run takes
   no field from a dump.

Refused by `zoom.Zoom` and not planned: pole-touching, cyclic and dateline-crossing children,
and children at the parent's first or last column -- no configuration in use has one, and two
of them hit latent Fortran quirks (`state.md` §3d). Not covered: the three-level tree outside
tier 1 and OpenMP.

### PYD-6. Hook `base/trunk/src/tm5.F90`, or say why not  [DONE 2026-10-13: not hooked]  [U:2 F:3 E:S]

Decided not to: `docs/state.md` §3b now says the reference is defined against the
meteo-server driver, and what a base-driver build would and would not dump. The premise here
("cannot be dumped at all") was wrong: such a build writes every operator event and the first
interval's `meteo`/`cfl`, and lacks only `massflow` and the later intervals' `meteo`/`cfl`,
which is what the free run and the ingestion check need. The builds concerned are the
`co2rc_*` family and five `drift_*` rc files, 118 of 594 at tm5-zoom `c133fb437`.

### PYD-13. Re-pin to tm5-zoom `745e94f70`  [DONE 2026-10-13]  [U:3 F:4 E:S]

The pin `13149b4a1` was three weeks old and 13 transcribed files had changed since. Each
commit message claims the default path is unchanged; the re-pin tests that instead of
trusting it (`docs/sources.md` §7):
- **Tier 1**: the zoom harness no longer built, because `advectm_cfl.F90` now uses
  `cfl_cache_shared` (`79b30df3b`). A cache-off stub in `fortran/stubs.F90` (`16aa560`) fixed
  it. Then 341/341 of the parity, zoom, kernel, simple and constant tests passed against a clean
  export.
- **Tier 2**: all three laptop dump sets were regenerated at `745e94f70` (759 / 483 / 22 events,
  as before). The phys set replays 391/391 events per event and 391/391 checkpoints in a free
  run fed from the input files alone; free 299/299, cfl 14/14; every array bitwise. The suite
  on those sets: 400/400 passed, none skipped.

Not covered: the rank-merge sets (`PYD-2`) were not regenerated, so MPI parity still rests on
`d5e682986` (since 2026-10-14 also on the ifort `pyd2i_*` and zoom `zp4_*` sets at `745e94f70`, `PYD-2`). Four new non-default paths are not transcribed: `ch4loss.strat.source : transcom`,
`proces.advection.limits : F`, `convection.apply : F` and `convection.slopes.full_matrix : T`.

### PYD-9. The dump sets are 8.3 GB of unreplicated laptop scratch  [DONE 2026-10-13, by PYD-13]  [U:2 F:3 E:S]

The recipe had never been re-run from scratch. `PYD-13` re-ran it for all three laptop sets,
on clean directories from a clean clone, and they came back with the originals' event counts
and sizes and replayed bitwise. What the docs had not said, now written into `state.md` §3b
and `runs/MANIFEST.md`: `CONDA_PREFIX` must be exported (since tm5-zoom `5292746df`); one
ensemble member needs a one-member scaling file; and the end time goes in the rc copy, not on
the command line. The recipe is now a script, `runs/setup_repin.sh`. The sets are still
unreplicated, but they are now reproducible, and that is the backup story.

### PYD-11. Check_CFL's z sub-cycle cap: stable and zoom-proof, and the inversion would see the longer step  [DONE 2026-10-13]  [U:4 F:3 E:M]

**Answered** (`docs/cfl_headroom.md`; the status block there has the numbers):

| question | answer | where |
|---|---|---|
| stable? | the replay reproduces production 672/672; 405 of 476 reduced glb3x2 intervals pass 3600 s at z = 2, worth 19 % of TM5 root | §8 |
| accurate on a spun-up field? | conserving, never negative, worst cell 12.9 ppb, the same as intervals production already runs at 3600 s; the real kernels refuse exactly the intervals the `(10,1,2)` rehearsal keeps reduced (3/3, then 29/29) | §9, §10 |
| zoom? | the nam1x1 child changes nothing, 672/672, 411 of 482 released | §11 |
| fresh sources? | with convection, the worst cell of a released interval moves about twice what it does at 1800 s, in the surface layer over strong sources | §11 |
| would the CT-CH4 EnKF see it? | **yes**: 38 of 284 groups over MDM/√N, all upward, continuous NH in-situ | §12 (pre-registered at `53ac94e`) |
| transport or the sampler? | **transport**: point-sampled, all 38 still over, ×1.49–1.80 | §13 (pre-registered at `d983045`) |
| which arm is closer to the obs? | **production, marginally**, in the timing and shape of the variability (median Δρ −0.0031, CI [−0.0058, −0.0003]; the ±0 s pair agrees but its CI reaches 0). The mean residual cannot decide: the setup sits ~100 ppb low | §14 (pre-registered at `b61c488`, below) |

The old items 3 (`s` in the sub-steps) and 4 (`dkg` for a sub-step never taken) were closed by
§12's forward runs, which inject sources and compute `dkg`.

**Done: 5. Which arm is closer to the observations** (§14). §12–§13 say the step moves H(x) by
more than the inversion resolves, not toward or away from the truth.
The pre-registration (the trap of a setup offset, Gate 0, the detrended-correlation statistic,
the day-block bootstrap, the decision rule and the selftest) is at commit `b61c488` and verbatim
in `TODO.history.md`; the run and its result are `cfl_headroom.md` §14.

**Not this repo's to close.** Whether to change production is a model decision
(tm5-zoom #111a, the per-grid cap rc key). If anyone pursues it, two costs are unchanged since
§5: the caps are in the CFL cache fingerprint, so changing one invalidates 26 years of cached
`ndyn`; and the bug report's y-cap claim still needs someone who remembers the 1x1 campaign
(§9 makes that sharper rather than softer: the real kernel's refusals are all in **y**).
Not answerable from §12's pair: whether the *mean* shift is toward the truth (§14).

### PYD-8. Package the repo properly  [DONE 2026-09-28; CI 2026-11-03]  [U:2 F:4 E:S]

**The Orion failures were NumPy's SIMD dispatch, not its version.** `test_calc_kzz` at seeds
18-20 failed on Orion (NumPy 2.0.2, Xeon Gold 6148) by up to 782 ulp in `dkg`, where the
laptop (NumPy 1.26.4, arm64) is within 16. With `NPY_DISABLE_CPU_FEATURES` switching off the
AVX-512 paths, the same NumPy 2.0.2 passes all three; with it emptied they fail again. The
cause is `np.log` in `simple_diffusion` (first difference in `ustar`) dispatched to an SVML
kernel that is not the C library's `log`, amplified through `Calc_Kzz`. The bitwise tier
never sees it: its transcendentals go through `ops.libm`, scalar. So neither a looser
tolerance nor a pin: `conftest.py` turns the AVX-512 dispatch off on x86-64 before NumPy is
imported (overridable by setting the variable), and prints NumPy's version and dispatched SIMD
paths in the pytest header, so every run records the condition it held under. Orion,
2026-09-28: `test_simple.py` + `test_kernels.py` 43 / 43 under `-W error`.

`pyproject.toml`: `pip install -e .` works (checked in a throwaway venv), `numpy>=1.26`, no
upper pin, `testpaths`; `ruff.toml` stays the lint config.

**CI** (`.github/workflows/tests.yml`, `ubuntu-24.04`): `ruff check` and the suite on every
push and pull request, on Python 3.10, 3.12 and 3.14 at `uv.lock`'s versions and on 3.10 at
`--resolution lowest-direct` (NumPy 1.26.0, the declared floor). With no dumps, no tm5-zoom
checkout and no compiled harness, only the tests that need none of those run (the constants
accounting among them); the rest skip by `tests/common.py`'s rules. Green on `pera-noaa/pydyn`.
Dependabot proposes the Actions and `uv.lock` updates, each a PR that also needs the full
suite run locally (CI has no Fortran or dump test). The sdist carries `tests/` and what they
import or load (`MANIFEST.in`).

### PYD-1. Redo the optimisation and FMA measurement with ifort  [DONE 2026-09-24, Orion's build]  [U:4 F:3 E:M]

**The claim now holds for Orion's production binary, not only a laptop build**
(`docs/state.md` §3c). Orion's `optim-fast` is `-O3 -xHost -g -fp-model precise -no-fma`
(`compiler-ifort-orion.rc`), Hercules' bare `-O3` (`compiler-ifort-hercules.rc`), which
leaves ifort's `-fp-model fast` on -- so the two machines sit on opposite sides of the line
`-ffp-contract=off` draws for gfortran.

**Tier 1:** the §3a/§3d harnesses built with ifort 2021.12 five ways, 188 bitwise tests each:
`-O0` / `-O3` under `-fp-model strict` and Orion's flags 188 / 188; Hercules' flags 94 fail,
with `-xHost` 152 -- `rm` by 8-10 ulp, near-zero slopes changing sign, `Check_CFL`'s
`(ndyn, n)` never different.

**Tier 2:** the `ch4_6x4_fwd.rc` day built with ifort on Orion (1 compute + 2 IO-server ranks,
one tracer, full physics) replays **bitwise per event and in `--free-run`, 408 / 408**, once
the Python calls the model's own libraries: MKL for LAPACK and the FFTW-interface Poisson
solver, and **libimf** for exp/pow/cos (`--libm`, `ops.use_libm`, new) -- with glibc the
chemistry's OH and Cl rates are 2-3 ulp off and nothing else is. The Hercules-flags build
replays as tier 1 said: 7e-10 in BalanceMassFluxes, ulps and sign flips in x/y/z.
Sets and jobs: `runs/MANIFEST.md`, `pyd1-*`.

**Residue, not done:** Hercules' own flags (bare `-O3`) built on Hercules; the zoom runs there
(`PYD-3` items 3-4, `state.md` §3f) are Hercules' ifort 2024.1 at Orion's flags, bitwise. Found on the way, about the model: `diffusion.cache.read` is
retired, `dkg` comes through the `tmm.sourcekey.<region>.*.diffus : dkg-cache:` keys, and the
production glb6x4 cache is 7e-6 rms from live `Calc_Kzz` in 96 % of cells.

### PYD-4. Transcribe the loss-rate ingestion and the initial condition  [DONE 2026-09-24]  [U:2 F:3 E:M]

**Done:** a free run needs no field from the model, only the rc's paths and scalars
(`docs/state.md` §3b). `pydyn/chemfields.py` reads the Bruehl columns, the Spivakovsky OH,
the Hossaini Cl and the qianlai soil map from the CH4 input tree (`replay --chemistry`);
`pydyn/initial.py` builds the `istart 66` field from its file and the initial air mass
(`--initial`). Both are bitwise on every `chem` event and the first `bef_*` of
`pydyn_dump_phys` and `pyd2_serial`, the first go. The free run on all four ingestions --
meteo, emission, loss rates, IC -- is bitwise at all 391 checkpoints of `pydyn_dump_phys` with
no event seeded from a dump: the first chemistry, which precedes any `m2phlb`, now runs on the
initial `phlb` of the ingested air mass. `tests/test_inputs.py`.

**Residue, not done:** the grouped (4DVAR-save) IC format and an IC on other levels than the
model's (FillLevels' interpolation) -- both refused, neither in use; OH IAV, other OH sources,
Wang Cl and the 'lori' soil map; the EnKF-scaled ensemble emission field (on the PYD-2 sets
`--emissions` differs by design). The grid constants and scalars still come from the dumps.

### PYD-2. The ranks together equal the serial answer  [DONE 2026-09-24; ifort and zoom 2026-10-14]  [U:3 F:2 E:L]

**It holds** (`docs/state.md` §3e): every array a multi-rank run dumps, with the ranks' tracer
slabs put back in rank order, is bitwise the array the same build dumps serially -- on a
four-tracer run serial and on 2, 3 and 5 compute ranks (slabs 2+2, 1+1+2, 1+1+1+1+0), for the
laptop gfortran glb6x4 day (`pyd2_*`), the same under ifort at Orion's flags beside 2 IO-server
ranks (`pyd2i_*`), and the glb3x2 + nam1x1 tree with emission and chemistry (`zp4_*`).
`tests/test_ranks.py`, `runs/hpc/ranks.slurm`; the sets and jobs are in `runs/MANIFEST.md`.

**Where it lives.** `dump.merge_ranks` / `load_events(merge=True)` (used by `replay`), with the
three things the dumps showed a merge must know: the split arrays by name (`rm..rzm`; with
equal slabs no size rule can tell them from replicated ones), the rank-local `ch4_tracers` of
the `chem` header, and the per-tracer `emis` events interleaved by global `i_tracer`.
`replay.sources` injects at `i_tracer`.

**What a rerun must know.** pycasso expands paths before command-line overrides apply, so they
go in a copy of the rc (`pydyn_build/setup_pyd2.sh`, `pyd2i/setup_pyd2i.sh`). The zoom rc's
`parameters.constant.value` scales every member alike and the scaling file is never read: four
identical tracers, which `test_tracers_are_split` catches (`pyd6/setup_pyd6.sh` drops the key
and refuses an rc that keeps it). A zoom set's rank test runs one set per job, `-c 6`.

**Residue, not done:** OpenMP. The hazard recorded for the mp port stands: matching routine
names hide a different decomposition, so none of this carries to tm5-mp.

### PYD-12. Pin the mirrored Fortran control parameters, so drift fails a test  [DONE 2026-09-22]  [U:3 F:4 E:S]

**What it guards.** The transcription copies Fortran constants that the array tests cannot
see: sub-cycle caps, iteration limits, convergence tolerances and the physical parameters of
`bldiff` and the CH4 chemistry. A cap only binds at the margin and no observed run reaches
the margin, so `cfl_max_nloop_{y,z}` could hold any value at all and the rest of the suite
would still pass — which is not hypothetical, they went 6, 10 → 1, 1 in `6f59f9f72`, inside
a commit about an `MPI_Allreduce`. The physical constants are covered by §3b, but only where
8.3 GB of dump sets is on disk.

**Where it lives.** `tests/test_constants.py`, 99 cases, 0.2 s; `docs/state.md` §4 for what
it does and `docs/sources.md` §3 for which declaration sites it parses. The file's docstring
is the reference for how — the expression resolver, the expression literals, the accounting
test — and is not repeated here.

**Verified to bite, not just to pass.** Against a doctored checkout with `fr_excess` 1 → 2,
`Rgas` 8.3144 → 8.3145, the OH pre-factor 2.45e-12 → 2.99e-12 and `kvhmin` 0.1 → 0.2, 14
cases fail naming old and new values, every derived constant included. A removed source file
fails; a constant added to a module with no row fails.

**`cfl_one`, settled on the way.** The y-substepping report's item 3 did land and was
withdrawn four days later (`5b8aaedda` → `a3d6cbac5` → `f43779a8a` → `ec7025cf2`,
2026-03-27..31), because `volatile :: ndyn` removed the cause it was padding over. So
`ops.ONE = 1.0` mirrors the right thing. Corrected in tm5-zoom at `f8abba4c5`, doc-only,
with a second stale claim in the same list; the table is in `docs/cfl_headroom.md` §5.

**Residue.** The guard runs only where a tm5-zoom checkout is present, so it protects a
developer, not CI: CI has no checkout (`PYD-8`).

## Parked

### PYD-10. The `secmom` and wet-deposition variants  [U:1 F:3 E:L]  [PARKED: not in the production build]

Second-order moments exist only in the `TM/trunk/TM5/cy2|cy3` reference trees, not on
`NOAA/ct-unified`, and the production define is `without_wet_deposition`. Transcribing either
means transcribing code this programme does not run. Unpark if a build starts using them.
