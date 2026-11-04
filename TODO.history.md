# TODO history: pydyn

> **`TODO.md`, verbatim as of pydyn `977896d`.** The head (`TODO.md`) keeps what a reader needs to
> not redo the thinking; this file keeps the full text the items had when they landed, including
> the PYD-11 item 5 pre-registration as it was run. Not read by default.


Items use the prefix `PYD-N`. Nothing here was forwarded from a tm5-zoom item: the work ran
through commits in `tools/pydyn` and never had a number. If one is opened there that overlaps,
keep the origin in the title, `PYD-n. (was tm5-zoom #N) ...`, and leave the tm5-zoom item alone.

Source repo for the pointers below: `ash-code/tm5-zoom`, branch `NOAA/ct-unified`. Most were
read at `4dc2017c5` and are cited that way; anything added since carries its own sha, and the
repo's current pin is `745e94f70` (`docs/sources.md`). Cite the sha you read at, never `HEAD`
— sister sessions commit to that repo concurrently and its history gets rewritten.

Tags are the tm5-zoom convention: `[U:urgency F:confidence E:S|M|L]`. An item moves to
**Landed** when it is done and to **Parked** when it is deliberately not being done; nothing
is deleted, and a landed item keeps whatever a future reader needs to not redo the thinking
— the mechanics live in the code it landed, not here.

## Open

### PYD-3. Zoom nesting  [U:3 F:3 E:L]

**Tier 1 landed 2026-09-24, `docs/state.md` §3d.** `pydyn/zoom.py` is bitwise against the
production zoom routines compiled unmodified in `fortran/harness_zoom` -- the exchange
(`put_x/yedges`, `mix_edges`, `update_parent`, `coarsen_region`, `coarsen_ambm`, `m2phlb`),
the advection of a child at every slot of its split order, and `Proces_Region` / a whole
`ndyn` step through the real `modelIntegration.F90` -- on two- and three-level trees,
glb3x2 + nam1x1 at production size among them. gfortran, arm64, `-O0`; the §3c caveats hold.
The tree is a Python computation (`Zoom`) checked against the production rc and
`dims_grid__eur.F90`, not a compiled `initexit.F90`.

**What is left, in the order `PYD-11` needs it** (its gap 1 is "production is glb3x2 +
nam1x1"; what it asks of zoom is whether the checker's decisions change with a child):

1. ~~**`Check_CFL`'s zoom path.**~~ Landed 2026-09-24, tier 1 (`state.md` §3d):
   `zoom.check_cfl_zoom` against the compiled `advectm_cfl.F90`, decision and every array
   bitwise on four trees, with a `caps=` override for `cfl_cap_replay`. What it cannot say yet
   is what the child does to production's decisions -- that needs item 2's dumps and meteo.
2. ~~**A zoom dump set, and replay learning regions.**~~ Landed 2026-09-29 (`state.md` §3f).
   The header gained `status`, `splitorder`, the footprint and `children` (tm5-zoom
   `611fb1ca9`, branch `pydyn/zoom-dump`, control arm in `runs/MANIFEST.md`), and
   `ch4_3x2nam1x1_pydyn.rc` (`397b5a2fa`) makes the sets. `pydyn/replay_zoom.py` replays a
   glb3x2 + nam1x1 run on Orion per event and in a 6 h free run, bitwise in every array, halos
   included, at ifort -- the reduced parent with a child, which tier 1 stubs out, and which
   turned up one transcription bug (the reduced rows' x halo after `red2uni`, fixed).
   On `origin/NOAA/ct-unified` as `84842782e`, `20ffeaed0` (cherry-picked 2026-10-07).
3. ~~**A child's meteo.**~~ Landed 2026-10-13 (`state.md` §3f). `pydyn/zoom_meteo.py`
   (the child's sp matched to its parent's, its raw fluxes matched to the parent's balanced
   ones, BalanceMassFluxes per parent cell through EnsurePoissonCache's direct matrices) and
   `pydyn/regrid.py` (the zoom rc reads the 1x1, 137-level set, not the coarsened ones: tmm's
   FillGrid and FillLevels onto each region). `replay_zoom --meteo` is bitwise on Hercules
   per event and in the free run on `zm_replay`, now on its own meteo, and on the `cfl`
   events of the four `zm_2019-*` weeks. The child's convection ingestion is item 4's.
4. ~~**`convection__speedup.F90`, `s` and `c` on a child.**~~ Landed 2026-10-13 (`state.md`
   §3f). Two Hercules runs of the zoom with its physics (`zp_replay` MPI, `zps_replay`
   serial): s, c, v, x, y, z of both regions bitwise per event and in the free run, with the
   convection's inputs and the chemistry's temperature read from the 1x1 set (`regrid.py`:
   'area-aver'/'sum' and 'mass-aver' both ways). dkg is bitwise too (`state.md` §3c, §3f).
   **Found: an MPI build's child decays CH4 with misplaced loss rates** -- ch4_chemistry_step
   binds the shared loss array from 1 and indexes it from isr/jsr (`analysis/ch4_loss_bounds.py`,
   358 500 of 358 500 per event); a tm5-zoom bug. Fixed by a pointer remap (tm5-zoom `0385efe7e`), verified on a rebuilt
   MPI run (`zpfix_replay`: bitwise the serial run and the Python, `state.md` §3f). Also fixed in
   pydyn: Meteo_Setup_Other interpolates over the dynamics step, not the read interval.
5. **The inputs of s and c on a zoom.** The 1x1 CH4 emission onto each region
   (coarsen_emission_2d 'sum' for glb3x2, the window for nam1x1), the chemistry's climatologies
   and soil map regridded per region (read_OH_field, the Hossaini maps, Bruehl in latitude),
   and the istart-66 field onto the child. With those the zoom free run takes no field from a
   dump.

Refused by `zoom.Zoom` and not planned: pole-touching, cyclic and dateline-crossing children,
and children at the parent's first or last column -- no configuration in use has one, and two
of them hit latent Fortran quirks (`state.md` §3d).

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

## Landed

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
`d5e682986`. Four new non-default paths are not transcribed: `ch4loss.strat.source : transcom`,
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
more than the inversion resolves, not toward or away from the truth. The pre-registration
follows, as run.

*The trap.* Both arms share an undated `istart 66` field that is not a 2019 state, run
`without_chemistry` (so CH4 accumulates with no sink), and carry the CT-CH4 priors at scaling
1.0. Each of those shifts H(x) − y at every site by a common amount far larger than d_g's
2–6 ppb. So "smaller mean residual" would score z = 2 closer wherever the setup happens to be
low and farther wherever it is high, whatever the transport. And a change that only rescales
a site's enhancement is indistinguishable from a flux error; the inversion would absorb it
into the fluxes, which is what §12 measured.

**Pre-registered 2026-10-13, before any residual was computed:**
- *Data*: §12's pair (`2019-10f-z1`/`-z2`, ±900 s, the EnKF's operator), October 2019, the
  same 117,097 obs; y is `flask_input`'s `value`. Robustness only: §13's ±0 s pair.
- *Gate 0*: both arms carry the same `obspack_id` set with identical y; the obs count equals
  §12's; the 38 groups are recomputed from Δ and equal `results/obs_departure_3x2_2019-10f.csv`'s.
- *Primary statistic*: per group g with ≥ 48 obs on ≥ 7 distinct UTC days, remove a
  least-squares line in time from y and from each arm's H separately, then
  Δρ_g = corr(H'_z2, y') − corr(H'_z1, y'). Correlation of the detrended series ignores an
  offset, a linear drift and a multiplicative rescaling of the model's anomaly, so it responds
  only to the *timing and shape* of the variability, the part a flux scaling cannot absorb.
- *Uncertainty*: a day-block bootstrap: resample October's UTC days with replacement, one
  draw shared by every group and both arms; B = 2000, seed 0, 95 % percentile intervals.
- *Population*: §12's 38 groups, the only ones where the arms differ by more than the
  inversion resolves. All eligible groups are reported as secondary.
- *Decision* on the 38: **z = 2 closer** if the median Δρ_g has a CI wholly above 0 and
  groups with CI above 0 outnumber those below by at least 2 to 1; **z = 1 closer** is the
  mirror image; **indistinguishable** if the median's CI contains 0; anything else **mixed**,
  reported as such.
- *Secondary, descriptive, labelled confounded*: the detrended MSE per arm and its projection
  β_g = Σ Δ'(y' − H'_z1) / Σ Δ'² (z = 2 lowers the MSE iff β_g > ½; this is confounded with
  flux magnitude); and the raw mean residual m_g per arm beside the network median of m_g(z1),
  to show how large the setup offset is next to d_g. Neither enters the decision.
- *Selftest* (`--selftest`, must PASS before scoring): synthetic groups where z1 is a scaled,
  offset copy of the truth and z2 that copy shifted 3 h must score z = 1 closer; swapped,
  z = 2; z2 = 1.2·z1 + 5 must score indistinguishable, where a naive mean-residual scorer
  says z = 2. The test checks that the naive scorer gets that case wrong, so the fixture is
  shown to be able to fail.

**Not this repo's to close.** Whether to change production is a model decision
(tm5-zoom #111a, the per-grid cap rc key). If anyone pursues it, two costs are unchanged since
§5: the caps are in the CFL cache fingerprint, so changing one invalidates 26 years of cached
`ndyn`; and the bug report's y-cap claim still needs someone who remembers the 1x1 campaign
(§9 makes that sharper rather than softer: the real kernel's refusals are all in **y**).
Not answerable from §12's pair: whether the *mean* shift is toward the truth (§14).

### PYD-8. Package the repo properly  [DONE 2026-09-28, except CI]  [U:2 F:4 E:S]

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

**Residue, not done: CI.** The repo is on GitHub (`Ash-Pera/pydyn`) with no workflow. The part of the suite
that needs neither dumps nor a Fortran build (property, simple-vs-fast, zoom geometry) runs
anywhere; the constants mirror (`PYD-12`) needs a tm5-zoom checkout beside it, and the parity
cases gfortran and `make -C fortran` / `make -C fortran zoom`.

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

### PYD-2. The ranks together equal the serial answer  [DONE 2026-09-24, compute ranks]  [U:3 F:2 E:L]

**The claim** (decided 2026-09-24): every array a multi-rank run dumps, with the ranks' tracer
slabs put back in rank order, is bitwise the array the same build dumps serially. **It holds**,
`docs/state.md` §3e: one `ch4_6x4_fwd.rc` run with four distinct tracers, serial and on 2, 3
and 5 Open MPI ranks (slabs 2+2, 1+1+2, 1+1+1+1+0), one clean tm5-zoom clone at `d5e682986` --
every replicated array identical on every rank, the merged run the serial run event for event
(900 events, BalanceMassFluxes under MPI included), and the merged sets replay bitwise.
`tests/test_ranks.py`; the sets are `runs/MANIFEST.md`'s `pyd2_*`.

**Where it lives.** `dump.merge_ranks` / `load_events(merge=True)` (used by `replay`), with the
three things the dumps showed a merge must know: the split arrays by name (`rm..rzm`; with
equal slabs no size rule can tell them from replicated ones), the rank-local `ch4_tracers` of
the `chem` header, and the per-tracer `emis` events interleaved by global `i_tracer`.
`replay.sources` injects at `i_tracer` now.

**Settled on the way.** §8's meteo half (the `cfl` events byte-identical on six ifort ranks) was
a decision; this is every array. And the laptop can make MPI dump sets at all, which it had
never done: `pydyn_build/setup_pyd2.sh` is the recipe, with the pycasso override trap it
records.

**Residue, not done:** a multi-rank tracer merge under ifort, and OpenMP. IO-server ranks and
zoom under MPI are covered as far as one tracer reaches: the ifort zoom `zpfix_replay` (6 compute
+ 2 IO ranks, slabs 1+0+0+0+0+0) equals its serial run in every array of rank 0 (`state.md` §3f),
but with one tracer there is no slab to merge. The hazard recorded for
the mp port stands: matching routine names hide a different decomposition, so none of this
carries to tm5-mp.

### PYD-12. Pin the mirrored Fortran control parameters, so drift fails a test  [DONE 2026-09-22]  [U:3 F:4 E:S]

**What it guards.** The transcription copies Fortran constants that the array tests cannot
see: sub-cycle caps, iteration limits, convergence tolerances and the physical parameters of
`bldiff` and the CH4 chemistry. A cap only binds at the margin and no observed run reaches
the margin, so `cfl_max_nloop_{y,z}` could hold any value at all and the rest of the suite
would still pass — which is not hypothetical, they went 6, 10 → 1, 1 in `6f59f9f72`, inside
a commit about an `MPI_Allreduce`. The physical constants are covered by §3b, but only where
8.3 GB of dump sets is on disk.

**Where it lives.** `tests/test_constants.py`, 77 cases, 0.2 s; `docs/state.md` §4 for what
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

**Residue.** The guard runs only where a checkout is present, so it protects a developer,
not CI. Folded into `PYD-8`, which is where a CI story would start.

## Parked

### PYD-10. The `secmom` and wet-deposition variants  [U:1 F:3 E:L]  [PARKED: not in the production build]

Second-order moments exist only in the `TM/trunk/TM5/cy2|cy3` reference trees, not on
`NOAA/ct-unified`, and the production define is `without_wet_deposition`. Transcribing either
means transcribing code this programme does not run. Unpark if a build starts using them.

---

> **`PYD-2` as `TODO.md` had it at pydyn `ec6485f`, verbatim** (condensed the same day).

### PYD-2. The ranks together equal the serial answer  [DONE 2026-09-24; ifort 2026-10-14]  [U:3 F:2 E:L]

**The claim** (decided 2026-09-24): every array a multi-rank run dumps, with the ranks' tracer
slabs put back in rank order, is bitwise the array the same build dumps serially. **It holds**,
`docs/state.md` §3e: one `ch4_6x4_fwd.rc` run with four distinct tracers, serial and on 2, 3
and 5 Open MPI ranks (slabs 2+2, 1+1+2, 1+1+1+1+0), one clean tm5-zoom clone at `d5e682986` --
every replicated array identical on every rank, the merged run the serial run event for event
(900 events, BalanceMassFluxes under MPI included), and the merged sets replay bitwise.
`tests/test_ranks.py`; the sets are `runs/MANIFEST.md`'s `pyd2_*`.

**Where it lives.** `dump.merge_ranks` / `load_events(merge=True)` (used by `replay`), with the
three things the dumps showed a merge must know: the split arrays by name (`rm..rzm`; with
equal slabs no size rule can tell them from replicated ones), the rank-local `ch4_tracers` of
the `chem` header, and the per-tracer `emis` events interleaved by global `i_tracer`.
`replay.sources` injects at `i_tracer` now.

**Settled on the way.** §8's meteo half (the `cfl` events byte-identical on six ifort ranks) was
a decision; this is every array. And the laptop can make MPI dump sets at all, which it had
never done: `pydyn_build/setup_pyd2.sh` is the recipe, with the pycasso override trap it
records.

**Under ifort, 2026-10-14.** The same four-tracer run for 6 h at `745e94f70`, Hercules ifort 2024.1
at Orion's flags, serial and on 2, 3 and 5 compute ranks each beside 2 IO-server ranks (the
`pyd2i_*` sets, `runs/MANIFEST.md`): `tests/test_ranks.py` 12 / 12 through the model's MKL and
libimf (`runs/hpc/ranks.slurm`, job `10115856`, 19 min). So the merge holds under the production
compiler and IO-server layout, and at the current pin.

**In a zoom tree, 2026-10-14.** `zpfix_replay`'s glb3x2 + nam1x1 run with the same four tracers,
serial and on the same three layouts (the `zp4_*` sets): 4 / 4 on each set (jobs `10117850`-`52`),
and `zp4_serial` replays bitwise per event and in a free run. The first attempt's four tracers
were identical -- the zoom rc's `parameters.constant.value : 1.0` scales every member alike and
the file is never read -- and `test_tracers_are_split` failed on all three sets; the rc copies
now drop the key and the setup refuses one that keeps it.

**Residue, not done:** OpenMP. The hazard recorded for
the mp port stands: matching routine names hide a different decomposition, so none of this
carries to tm5-mp.

## PYD-16 before the A100 and MI100 runs (`TODO.md` at pydyn `0db6e6d`)

### PYD-16. Run the GPU port on a GPU, and time it  [U:3 F:3 E:S]

`pydyn.gpu` puts every operator on the device with the state resident and is bitwise to the
reference on the emulation backend: 27 tests, the three laptop sets' free runs fingerprinted
checkpoint by checkpoint, the production zoom fixture per event against the model's own dumps
(`docs/gpu_port.md` §5). Its CUDA module compiles for sm_80 with no FMA (§2). On the laptop's
Apple M4 (Metal, float32) it runs bitwise to its float32 host twin and a production-sized
interval takes 24.4 ms on the device: 16.9 simulated years an hour (§8); from the input files
(`pydyn/forward.py`, glb6x4) 24.0, 14 days bitwise emu32, the tracer total 6.6e-7 from float64
(§9). What is not known: that NVRTC's float64 build on a real card gives the reference's bits,
and what it costs there.

1. Alpine (the only FP64 NVIDIA hardware within reach; a Duo push): `pip install
   cupy-cuda12x` in a venv, copy a laptop set, `PYDYN_DEVICE=cuda python3 -m pytest
   tests/test_gpu.py`, then `python3 -m pydyn.gpu.check <set> --device cuda --free-run`.
2. Time an interval with `python3 -m pydyn.gpu.bench --device cuda --interval --physics
   --speculate --inverse lapack`, and `pydyn.forward` from files. The CUDA path has what made
   Metal fast (convec_gen's passes, the line kernels a warp a line, the asynchronous queue with
   two intervals in flight; §8), compiled for sm_80 and its host side tested against a
   stand-in cupy, not run; `PYDYN_COOP=0` / `all` compare the line kernels. Not done (§7):
   32-bit indices, an interval as one CUDA graph. A change for speed is checked by the same
   fingerprints.
3. A run from the files at production size (§7): glb3x2 + nam1x1 meteo is not local; the
   reading runs in `--readers` processes (§6), on an HPC node from `--stage` copies on its
   local disk (`/work2` read directly: 2.4-7 s a simulated day, copied by 8 threads 0.17 s;
   §6), and the device's interval is GPU-bound there (box dispatches, convection's passes, z,
   loss rates: §7). float32 chemistry's `rm * decay` rounding (-4.8e-8 of the tracer a day,
   §8) is measured, not fixed.
