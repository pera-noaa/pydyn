<!-- author: Ash Pera (NOAA/GML CIRES) -->
<!-- footer-title: pydyn — Check_CFL headroom -->

# What rejected the time step, and was it the physics or the checker?

**Status (2026-10-13).** `Check_CFL`'s z sub-cycle cap is what shortens production's time
step and costs about 19 % of TM5 root. Raising it to 2 is stable, zoom-proof and conserving,
but **the CT-CH4 EnKF would see it**: in a paired October 2019 forward run, 38 of 284 obs
groups move by more than MDM/√N, all upward, all continuous NH sites (§12). **It is transport,
not the obs operator**: with both arms sampled at one instant, all 38 still exceed and every
one grows, by ×1.5–1.8 (§13). **And it is not closer to the observations**: in the timing and
shape of the variability, the part a flux error cannot mimic, production fits marginally better
(§14). Nothing here licenses a change to production.

| | what it establishes | evidence |
|---|---|---|
| §1–§4 | one rejection was **z** refused a second sub-cycle — a real CFL violation, not the depletion artifact, and one the real `dynamw` (cap 15) would have carried | 1 interval, glb6×4, gfortran, serial |
| §5 | where the caps came from, and two costs any change must pay | git history |
| §7 | production runs below its `ndyn` ceiling on **59.5 %** of hours, **+60.8 %** dynamic steps | 227,928 archived hours, 2000–2025 |
| §8 | the replay reproduces the model **672/672**; of 476 reduced intervals, **405 (85.1 %)** pass the full hour at `z=2`, all needing exactly two. Worth **19 % of root**, ~225 s | 672 glb3×2 intervals, ifort, MPI |
| §9 | mass conserved to 6e-16, **0 of 192 runs negative**, worst cell 12.9 ppb against a 15 ppb assumed mismatch — **the same as intervals production already runs at 3600 s**. The 3 intervals that refuse 3600 s are *exactly* the 3 `(10,1,2)` holds back | 48 glb3×2 intervals with a CH4 field |
| §10 | October's hardest days: the **29** intervals that refuse 3600 s are *exactly* the 29 `(10,1,2)` holds back. With convection in the step, released intervals move **≤ ~4 ppb** once the field is spun up (the first 10–12 h after `istart 66` are spin-up, shown by a control started a day earlier) | 48 + 30 glb3×2 intervals, ifort, with `bef_v` |
| §11 | the nam1×1 child changes nothing (**672/672**, 411 of 482 released); with 48 h of sources convection's worst cell at 3600 s is median **20.5** / worst **67** ppb vs 9.4 / 32 at 1800 s, in the surface layer over strong sources | 4 zoom windows; 48 intervals with sources |
| §12 | paired forward run, z cap 1 vs 2, at the EnKF's 117,097 October obs: **would see** (pre-registered rule). 38 of 284 groups over MDM/√N, all positive, 2–6 ppb, continuous in-situ 38–71°N; none where the obs operator's sample count is unchanged | 2 × 36 days glb3×2, ifort, sources on |
| §13 | both arms rerun with the sampler's window at ±0 s: **transport** (pre-registered rule). All 38 of §12's groups still over, d_g ×1.49–1.80 (FNE 9.8 ppb, 11.4 s); 47 of 209 over; the 75 groups lost are discrete samples at an instant one arm has no state for | 2 × 36 days glb3×2, ifort, sources on |
| §14 | which arm is closer to the obs, on §12's 38 groups: **z1, marginally** (pre-registered rule). Detrended correlation with the obs, median Δρ −0.0031, 95 % CI [−0.0058, −0.0003]; 3 groups resolved worse at z = 2, 1 better. The ±0 s pair agrees in sign and size but its CI reaches 0. The mean residual cannot decide: the setup sits ~100 ppb low, so z = 2's upward shift "improves" 37 of 38 groups whatever the transport | §12 and §13 pairs, October 2019 |

**Open, and not answerable from this pair:** whether the *mean* shift is toward the truth.
That needs a run whose mean is not set by its setup (a 2019 initial field, chemistry on,
posterior fluxes), and even then the posterior was fitted with z = 1 transport. The 15 ppb
default MDM covers 3,756 obs, and R's inline gradient term is left out, so §12's s_g errs
small.

## 1. The measurement

`Check_CFL` rehearses the coming step on the mass field alone, pushing it through
`xyzzyx` with per-direction caps on how many sub-cycles each operator may use. If any
operator exceeds its cap the whole interval is retried at the next smaller `ndyn`.

The script patches two names inside `pydyn.cfl` — `subcycles` (a spy on the `seen` hook the
kernels already expose) and `rehearse_step` (a statement-for-statement clone that names the
operator and can pass other caps). `check_cfl` and the three mass kernels are the
transcribed ones, and the baseline run reproduces the dumped `(ndyn, n, iterations)` =
`(5400, 2, 2)` exactly. That is the check that the instrumentation changed nothing.

Event: 2000-01-01 00:00, glb6×4 60×45×34, reduced grid, `nread = 10800 s`,
`time.ndyn_max = 10800`, `meteo.tinterp.* = interp3`.

## 2. What the interval demanded

Largest |Courant| at the rejected `ndyn` = 10800 s, against the undepleted mass:

| direction | max \|Courant\| | sub-steps needed | checker cap | real kernel cap |
|---|--:|--:|--:|--:|
| x | 7.8541 | 5 (as rehearsed) | 10 | 50 |
| y | 0.7957 | 1 | **1** | 10 |
| z | 1.3722 | 2 | **1** | 15 |

x is the direction with by far the worst Courant number and it passes comfortably. z is
over 1 by 37 % and is refused, because one sub-cycle is all it gets.

## 3. What each cap would have chosen

Same event, same fluxes, only the caps changed:

| caps (x, y, z) | chosen `ndyn` | iterations | sub-cycles used | |
|---|--:|--:|---|---|
| 10, 1, 1 | 5400 s, n = 2 | 2 | — | production today; reproduces the dump |
| 10, 1, 2 | **10800 s, n = 1** | 1 | x=5, y=1, z=2 | the smallest possible change |
| 10, 6, 10 | 10800 s, n = 1 | 1 | x=5, y=1, z=2 | the values in the tree on 2026-04-14 |
| 50, 10, 15 | 10800 s, n = 1 | 1 | x=5, y=1, z=2 | what the advection kernels themselves allow |

Every relaxed cap lands on the same answer by the same route: z sub-cycles twice and
nothing else changes. What that is worth is not estimated here — §7 counts how often
production is reduced at all and §8 measures how much of it a second z sub-cycle returns.

## 4. It is not the depletion artifact

`tm5-zoom@4dc2017c5:proj/ct-unified/trunk/doc/bugreport_cfl_y_substepping.md` argues that
z violations inside the rehearsal are typically artifacts: the x and y passes drain the cell
before z is tested, and because both `cm` and `m` scale with `ndyn`, such a violation is
scale-invariant and halving cannot cure it. The script tests exactly that, and here it
comes out the other way:

```
z |Courant| vs the depleted mass it was tested against: 1.5138
z |Courant| vs the mass at the start of the step:       1.3722
-> REAL: over CFL against the undepleted mass too.
```

The x/y passes made it worse (1.37 → 1.51) but did not create it. So this rejection is a
genuine CFL violation at 10800 s, correctly detected — and correctly handled by
sub-cycling, which the checker refuses to do and the real kernel would have done.

## 5. Where the caps came from

At the pin, `advectm_cfl.F90:104-106` has `cfl_max_nloop_{x,y,z} = 10, 1, 1`.

| commit | date | caps |
|---|---|---|
| `fb8eb246d` | 2026-04-14 | 10, 6, 10 — the constants hoisted to module scope and written into the CFL cache fingerprint (item 68) |
| `6f59f9f72` | 2026-04-19 | **10, 1, 1** — headline change is the `MPI_Allreduce` on `cfl_ok` that fixed a 1×1 silent hang; the caps moved in the same commit |
| HEAD | 2026-09-22 | 10, 1, 1 |

Two things to settle before anyone touches this, neither of which this repo can settle:

- **The bug report is stale in three places, and only one of them is still open.** Its
  "additional findings" list reads as current state and is not; checked against the tree
  2026-09-22, and the first two corrected in it (`tm5-zoom@f8abba4c5`):

  | claim | tree |
  |---|---|
  | `cfl_one = 1.0 + 1e-6` added to `dims.F90` | withdrawn 2026-03-31, four days after the report; `f43779a8a` reverted it to 1.0, `ec7025cf2` deleted the symbol. Every CFL comparison is `>= one`, `one = 1.0`. It was padding over ifort hoisting `ndyn`, which `volatile :: ndyn` fixed properly |
  | `max_global_iteration` increased to 32 | 100 since `fb70b5f4b` (2026-03-29), labelled "temporary: … for 1x1 spinup", still labelled that |
  | do not reduce the y cap below 10 | y = 1 since `6f59f9f72` (2026-04-19) — **still open** |

  The y cap is the one this note turns on, and it is the one a document cannot settle.
  `6f59f9f72`'s own comment says 1×1 with `cfl_max_nloop_y = 1` produced the rank desync it
  was fixing, so y = 1 was deliberate; the report says it causes `ndyn` spirals at 1×1.
  Someone who remembers that campaign should say which. Read the report knowing that two of
  the three things next to that claim had already gone stale.
- **The caps are in the CFL cache fingerprint.** Changing one invalidates every cached
  `ndyn` (`advectm_cfl.F90:634-658`), which is 2000–2025 of glb3×2 per `ct-meteo`. That is a
  real cost and it belongs in the decision.

## 6. Four things this note is not saying

- **The rc sets the ceiling; the checker sets the step under it.** `time.ndyn_max : 3600`
  in both `__zoom_1day.rc` and `ch4_6x4_fwd.rc`, so no interval can ever run *longer* than
  an hour however the caps are set. §3's jump to 10800 s is therefore **not** available to
  production, and the win demonstrated there is not the win on offer. What is on offer is
  the 59.5 % of hours that currently run *shorter* than the ceiling (§7).
- **The checker is a stability gate, not an accuracy gate — and that stays true even though
  §9 found the accuracy acceptable.** `Check_CFL` never looks at the tracer. §9's finding is
  not that the checker is protecting accuracy badly; it is that the checker is not
  protecting accuracy at all, in either direction, which is why the intervals it accepts
  carry the same tracer error as the ones it refuses.
- **The laptop dump sets show none of this.** On the 23 hourly intervals of
  `pydyn_dump_phys` and `pydyn_dump_free`, `Check_CFL` reduced nothing — 23/23 accepted
  3600 s first try. One quiet day at glb6×4 is not the archive, and §7 through §9 are why
  the 3×2 dumps had to be made.
- **`PYD-3` is on the critical path, not beside it.** Everything measured here is glb3×2
  standalone. Production is glb3×2+nam1×1, and §7's own census gives that configuration a
  different reduction rate (61.2 % against 59.5 %). Until the zoom arm is run, none of these
  numbers is a statement about the configuration that actually runs. Answered by §11: the
  child changes nothing (672 / 672, 411 of 482 released).

## 7. Production pays on three hours in five

Measured 2026-09-22 from the CFL caches on Orion,
`/work2/noaa/co2/METEO/tm5-nc/ec/ea/cfl/tropo34/<year>/cfl_ndyn3600_<grid>.nc`, 2000–2025.
Read-only; `analysis/cfl_cache_census.py` is the instrument. The files hold the `ndyn`
Check_CFL chose for every meteo interval, one per hour, and their fingerprint attributes
carry the caps it chose under — **`max_nloop x,y,z = 10,1,1` on all three grids**, so this
is a census of the checker §3 is about, not of an older one.

| grid | intervals | reduced below 3600 s | extra dynamic steps |
|---|--:|--:|--:|
| glb3×2 | 227,928 | 135,523 (**59.5 %**) | **+60.8 %** |
| glb3×2+nam1×1 | 227,928 | 139,397 (61.2 %) | +62.5 % |
| glb100×100 | 86,400 | 86,400 (**100 %**) | +194.6 % |

At 1×1 the ceiling is *never* reached. That tree is Q1-only for most years
(`ash-code/ct-meteo:docs/state.md` §5), so its number is winter-weighted and not comparable with
the other two; the 100 % is not.

Three things in the distribution matter more than the headline.

**Almost every reduction is a single step down.** glb3×2 chose 3600 s on 92,405 intervals,
1800 s on 132,580, 1200 s on 2,890 and 900 s on 53. So **97.8 % of all reductions are one
halving** — the same shape as the rejection §2 takes apart, where one extra z sub-cycle was
the whole difference.

**It is not weather.** By month, glb3×2 reductions run 14,001 in January to 9,652 in April:
a 1.45× seasonal range on a 60 % base. A checker tripped by polar jets and winter storms
would not fire on three hours in five in April. By year it is flat too, 4,525 (2022) to
6,153 (2025), across 26 years of different meteorology. That is the signature of a
structural limit, not of the atmosphere.

**The ceiling on what it could be worth.** Advection and convection are 58.7 % of TM5 root
(`tm5-zoom@4dc2017c5:docs/kernel_catalog.md` §1: `advectx` 12.8, `advectz` 11.9, `advecty`
9.0, `conv/apply` 22.0, `conv/invert` 2.4, `conv/matrix` 0.6) and scale linearly with the
number of dynamic steps. At +60.8 % more steps than the ceiling would need, removing *every*
reduction would save 22 % of TM5 root — about 262 s of the 1178.95 s root, 11.5 % of the
2280 s full step of the profiled configuration.

**That last number is a ceiling, not an estimate.** It assumes every reduction is avoidable,
which nothing in this section shows — the 135,523 reductions at 3×2 could all be genuine.
Its job is to establish that the question is worth answering at all: the thing being argued
about is on the order of a tenth of the model's runtime, not a rounding error.

**§8 replaces the assumption with a measurement**, and the ceiling with 19 %. Worth noting
how: it did not need the 3×2 meteo, which is what this section assumed the next step would
cost. The `cfl` event carries everything the replay wants, so the dump was the whole
interface.

## 8. On real 3x2 intervals: 85 % of the reductions are the z cap

Measured 2026-09-22 on Orion. Four seven-day windows of 2019 — January, April, July,
October — run from `tm5-zoom@0e72e4cf8:proj/ct-unified/rc/ch4_3x2_pydyn_cfl.rc`, which is
the rc that wrote the caches (§7) with three keys changed: `cfl.cache.read : F` so the
checker actually runs, both cache writes off, and `pydyn.dump : T` restricted to the `cfl`
tag. 672 intervals, glb3×2 120×90×34, reduced grid with 14 bands, ifort `optim-fast`,
six MPI compute ranks. `analysis/cfl_cap_replay.py` is the instrument; the per-interval
results are `results/cfl3x2_2019_cap_replay.csv`.

**The control comes first, because without it the rest is a different model's answer.**
Under the caps the model itself ran, the replay reproduces its `(ndyn, n)` on **672 of 672**
intervals. That is the first time this repo's `Check_CFL` has run at 3×2 at all, against a
reduced grid it did not dump itself, on a field an ifort MPI build produced — and it is the
first ifort agreement of any kind in this repo (`PYD-1`).

| caps (x, y, z) | intervals reduced | extra dynamic steps | of the 476, recovered |
|---|--:|--:|--:|
| 10, 1, 1 | 476 / 672 (**70.8 %**) | **+74.6 %** | — (production today) |
| 10, 1, 2 | 71 / 672 (10.6 %) | +10.6 % | **405 (85.1 %)** |
| 50, 10, 15 | 39 / 672 (5.8 %) | +5.8 % | 437 (91.8 %) |

**Every recovered interval uses two z sub-cycles, and nothing else moves.** Of the 405,
402 rehearse at exactly `x=2, y=1, z=2` and the other three at `x=3, y=1, z=2`.

The stronger version of that is what the whole sample says. Across all 672 intervals, under
all three cap settings including the one that allows x fifty sub-cycles and y ten:

| | x used | y used | z used |
|---|--:|--:|--:|
| 10, 1, 1 | max 3 | max 1 | max 1 (capped) |
| 10, 1, 2 | max 3 | max 1 | max 2 (capped) |
| 50, 10, 15 | max 3 | max 1 | max 4 |

x never exceeds 3 against a cap of 10, and y never exceeds 1 against a cap of 1 — not even
when given ten. Neither is near binding anywhere in the sample. z is the only direction that
ever reaches its cap, and at `(10, 1, 1)` it reaches it on 476 of 672 intervals. So the
question "what is `Check_CFL` costing production" has a one-line answer on this evidence: the
z cap, by one sub-cycle.

Read that table with one qualification, which §9 makes concrete. Each row reports the
sub-cycles the rehearsal used at the `ndyn` it finally *chose*, so for an interval that stayed
reduced the count is the one at 1800 s, not at 3600 s. "y never exceeds 1" is therefore a
statement about the steps that were taken, not a promise that y would stay quiet if 3600 s
were forced everywhere. §9 forces it, and on three intervals y is exactly what fails.

**The residue is real and it is a minority.** 71 intervals are still reduced at `(10, 1, 2)`.
32 of those come back under `(50, 10, 15)` — 26 need `z=3`, six need `z=4` — and **39 are
refused by every cap tried**, which is the count of genuine CFL violations in the sample:
5.8 % of intervals, not 70.8 %.

**It varies by season, and the variation is in the residue, not the mechanism.**

| window | reduced | recovered by z=2 | by 50, 10, 15 |
|---|--:|--:|--:|
| 2019-01 | 104 / 168 | 96 (92.3 %) | 101 (97.1 %) |
| 2019-04 | 142 / 168 | 142 (**100.0 %**) | 142 (100.0 %) |
| 2019-07 | 127 / 168 | 100 (78.7 %) | 127 (100.0 %) |
| 2019-10 | 103 / 168 | 67 (65.0 %) | 67 (65.0 %) |

April's week reduces the most (84.5 % of its intervals) and recovers all of it; October's
reduces the least and recovers the least. Note that this ordering is not the census's —
§7 has January high and April low across 26 years, and one week is not a month. What does
not vary is which cap is responsible.

**What it is worth.** §7's 22 % was a ceiling that assumed every reduction avoidable. This
replaces the assumption with a measurement. Carrying the sample's cut in *excess* steps
(+74.6 % → +10.6 %, an 85.8 % cut) onto the archive's own excess (+60.8 % → +8.6 %), and
holding advection and convection at 58.7 % of TM5 root scaling linearly with step count:

| | excess steps | saving |
|---|--:|--:|
| this sample, directly | 0.746 → 0.106 | 21.5 % of root, 254 s, 11.1 % of the 2280 s full step |
| archive, projected | 0.608 → 0.086 | **19.0 % of root, 225 s, 9.8 % of the full step** |

The two agree to within the sample's own excess being higher than the archive's, which is
what you would expect from four weeks against 26 years.

**All of this is still the mass rehearsal.** `Check_CFL` never looks at the tracer, so
nothing above says a step that passes at `z=2` carries the tracer field acceptably. §9 puts
the real tracer kernels on these same intervals and asks.

## 9. Is the longer step accurate, or only stable?

Measured 2026-09-23. §8 is a stability result: `Check_CFL` rehearses the air-mass field, so
"would have passed at 3600 s" says nothing about the tracer. This does, on the 48 glb3×2
intervals of 2019-01-01..02 — inside §8's January window — dumped with `ntracet = 1` from
`tm5-zoom@13149b4a1:proj/ct-unified/rc/ch4_3x2_pydyn_tracer.rc` and replayed by
`analysis/step_accuracy.py`. Each interval is re-integrated at n = 1, 2, 4 and 8 sub-steps,
so n = 1 is one 3600 s step and the rest refine it. Transport only — x, y, z, no convection
or physics — because advection is the operator set the checker rehearses. Results:
`results/step_accuracy_3x2.csv`, and `results/step_accuracy_6x4.csv` for the glb6×4 arm.

The flux path is verified before anything else runs: the (pu, pv) and (am, bm, cm) this
script rebuilds are bitwise equal to the model's own dumped `massflow` event at the dumped
`ndyn`, at both resolutions. Without that the refinement would be measuring its own
reconstruction.

**Conservation and positivity are not the problem.** Worst relative drift in total tracer
mass is 6.0e-16 at every step size, which is round-off. Not one of the 192 runs goes
negative. Whatever the case against a longer step is, it is not that `slopes` stops
conserving or starts producing negative mass.

| n | ndyn | mass drift | worst cell | rms | glb6×4 worst cell | glb6×4 rms |
|---|--:|--:|--:|--:|--:|--:|
| 1 | 3600 s | 6.0e-16 | **12.88 ppb** | 0.132 ppb | 5.14 ppb | 0.103 ppb |
| 2 | 1800 s | 6.0e-16 | 2.86 ppb | 0.046 ppb | 2.30 ppb | 0.042 ppb |
| 4 | 900 s | 6.0e-16 | 0.83 ppb | 0.014 ppb | 0.78 ppb | 0.013 ppb |
| 8 | 450 s | 6.0e-16 | reference | reference | reference | reference |

For scale: the field spans 2594 ppb, and the model-data mismatch these inversions assume for
CH4 is **15.0 ppb** (`ct.obs.mdm.default.ch4`). So the worst single grid cell in the worst
interval moves by most of one MDM, and the field as a whole moves by about 1 % of one.

**The finding that matters is what happens when you split by what production actually did.**

| | intervals | worst cell at 3600 s | rms at 3600 s |
|---|--:|--:|--:|
| production reduced these to 1800 s | 32 | 12.70 ppb | 0.096 ppb |
| production ran these at 3600 s already | 16 | **12.88 ppb** | **0.132 ppb** |

The intervals `Check_CFL` *accepted* at the full hour carry the same error — slightly more —
as the intervals it refused. The checker is not sorting intervals by tracer accuracy, and
there is no reason it should: it is a stability gate and it does not look at the tracer at
all. Whatever accuracy objection can be raised against running the reduced intervals at
3600 s applies already, today, to the third of intervals production runs at 3600 s.

**Three intervals cannot take the step at all, and §8 named them in advance.** On three of
the 48, forcing 3600 s does not degrade the answer — it produces no answer: the real
`dynamv` exceeds its own `max_nloop = 10` and raises. Those three are
2019-01-01 17:00, 18:00 and 23:00, and they are *exactly* the three that §8's checker at
`(10, 1, 2)` keeps reduced, and exactly the three that even `(50, 10, 15)` keeps reduced.
Set equality, 3 of 3, with no false positives and no false negatives in 48 intervals.

That is the strongest validation §8's counterfactual could have got, and it was not designed
in: the rehearsal with one extra z sub-cycle releases precisely the intervals the real
kernels can carry and holds back precisely the ones they cannot.

Note which direction fails, though. §8 found z to be the only binding cap in the *rehearsal*;
here the refusal is in **y**, in the real kernel, against `advecty__slopes.F90`'s own
`max_nloop = 10` rather than the checker's `cfl_max_nloop_y`. The two are consistent — §8's
y counts were taken at the `ndyn` each interval finally ran at — but anyone reading §8 as
"y is irrelevant" would be wrong about the residual. The y cap the bug report argues over
(§5) is not settled by any of this.

### What this licenses, and what it does not

It supports a narrow claim: on this evidence, raising the checker's z cap from 1 to 2 would
let production take the full hour on the intervals whose tracers behave no worse than the
ones it already runs at the full hour, while continuing to refuse the ones the kernels
genuinely cannot take.

It does not support acting on that yet. Four things stand between here and a model change:

- **Two days, one January, one grid, transport only.** §8's seasonal spread (Apr 100 %
  recovery, Oct 65 %) says the residual varies; this sample is from the January week.
- **Convection is not in the integration**, and in the real step it sits between the
  advection operators. This bounds the advection error, not the step's total error.
- **The 3×2 field here is transport-only from the initial condition** — real CH4 structure,
  but two days without sources, so without the sharp fresh gradients emissions create. The
  6×4 arm does have the physics and reports a smaller error, which is reassuring rather than
  conclusive.
- **Production is glb3×2+nam1×1**, and this is glb3×2 standalone — §6's last bullet, and
  the reason `PYD-3` sits on this item's critical path rather than beside it.

And the two costs from §5 are unchanged: the caps are in the CFL cache fingerprint, so
changing one invalidates 26 years of cached `ndyn`, and the y-cap question still needs
someone who remembers the 1×1 campaign.

## 10. October, and convection in the step

Measured 2026-09-28 on Orion, closing two of §9's four gaps. `tr3x2_2019-10` is the §9 run
moved to **2019-10-01..02** -- the hardest two days of §8's October window, 47 of 48
intervals reduced by the production caps and only 18 of those released at `(10, 1, 2)` --
with `bef_v` added to the dump tags so that the convection inputs the model used are on
disk (`runs/MANIFEST.md`). `analysis/step_accuracy.py` gained `--convection`: the model's
`v` operator where do_steps puts it, `xyzv | vzyx`, with `conv_dt = ndyn/2` because
`Check_CFL` sets `nconv = ndyn` (`advectm_cfl.F90`), so a shorter step shortens convection
too and refining it is what production would do. Results:
`results/step_accuracy_3x2_2019-10_{xyz,xyzv-first,xyzv-last,xyzv-step}.csv`, and
`results/step_accuracy_6x4_xyzv.csv` for the glb6×4 arm with convection.

**The prediction held on a window where it had something to predict.** In January §8 kept
3 of 48 intervals reduced and the real kernels refused exactly those 3. Here the rehearsal
at `(10, 1, 2)` keeps **29** reduced, and forcing 3600 s through the real kernels is refused
on **exactly those 29** -- set equality, no false positive or negative in 48, every refusal
`dynamv` exceeding its own `max_nloop = 10`, so in **y** as in §9. Adding convection changes
none of them. The checker with one extra z sub-cycle does not merely count the right number
of intervals; it names the ones the kernels can carry.

**Transport only, October is milder than January.** The 18 released intervals move at most
**9.2 ppb** in the worst cell at 3600 s (January 12.9), mass drifts 8e-16, no run of 192 goes
negative.

**Convection is the larger error, and it has one outlier hour.**

| released intervals, worst cell | n = 1 (3600 s, the proposal) | n = 2 (1800 s, production today) |
|---|--:|--:|
| transport only (x y z) | 9.2 ppb | 2.9 ppb |
| with convection, 17 intervals, hour 0 excluded | **3.1 – 4.0 ppb** | 1.6 – 1.9 ppb |
| with convection, 2019-10-01 00:00, the run's first hour | 30 – 50 ppb | 13 – 14 ppb |
| with convection, the same hour 24 h after the initial field | **3.4 ppb** | 0.6 ppb |

(The ranges are the three `--dkg` modes below. The transport-only 9.2 is also hour 0; without
it the worst is 3.1.)

Two things are in that table. For all but one released interval, the step production would
take at `(10, 1, 2)` moves the worst grid cell by about 4 ppb, under a third of the 15 ppb
MDM, against the ~2 ppb production's 1800 s step moves it today. And the run's first hour is
an order of magnitude worse at *every* step size, including the 1800 s production already
runs: its n = 2 error (14 ppb) is the largest of all 48 intervals, and the next four largest
are also in the first eight hours.

**It is spin-up, and the control shows it.** `tr3x2_2019-09s` is the same run started a day
earlier, 2019-09-30 00:00 to 2019-10-01 06:00 (`results/step_accuracy_3x2_2019-09s_xyzv-first.csv`).
Measured 24 h after the initial field, 2019-10-01 00:00 moves **3.4 ppb** at 3600 s instead
of 32.5, and the new first hour, 2019-09-30 00:00, takes the outlier's place at **31.7**. The
excess decays over the first 10–12 hours (hour 3: 17.1, hour 6: 10.9, hour 10: 3.9); from hour
10 on, no accepted interval exceeds 4.0 ppb at 3600 s. The refusals of 10-01 01:00–05:00 are
the same in both runs, as they must be -- they depend on the meteo, not the tracer. So the
first hours after an `istart 66` field are not evidence about the step size, and every accuracy
number here and in §9 should be read from spun-up intervals: §9's worst (12.9 ppb, January)
is 40 h in, and all 17 ordinary October intervals are at least 15 h in.

**What `dkg` does and does not change.** The convective fluxes and cloud levels are constant
over a meteo interval (aver3), but `dkg` is not: it changes from one dynamics step to the next
inside a reduced interval, by up to half its maximum, as live `Calc_Kzz` on each step's own
fluxes would -- the rc sets a `dkg-cache` sourcekey only for glb100x100. A sub-step of the
counterfactual has no dumped `dkg` of its own, and computing one means ingesting the
ml137 / 1×1 archive this run reads and remapping it, which pydyn did not do when this was run
(it does since, bitwise, `state.md` §3f). So `--dkg` brackets it instead: `first` / `last` freeze the interval's first or
last per-step value, `step` gives each sub-step the value of the model step containing its
midpoint, so that n = 2 reproduces production's own sampling exactly. The 17 ordinary
intervals are insensitive to the choice (3.1 – 4.0 ppb); hour 0 is not (30 – 50). The
refusals and positivity are identical in all three.

**The 6×4 arm puts a number on "production already pays it".** In January every interval of
`pydyn_dump_phys` ran at 3600 s, so it has no reduced intervals to release, but it says what
a production configuration carries at the full hour with convection in: worst cell
**14.9 ppb**, median 6.8, over 23 intervals. The October 3×2 released intervals at 3600 s
(≤ 4 ppb once spun up) sit well inside that. §9's in-window comparison is not available
here: only one of the 48 October intervals ran at 3600 s (3.8 ppb).

### What this adds to §9's licence, and what is still open

The narrow claim of §9 now holds on a second season, on the hardest days that season offers,
and with convection in the step: raising the checker's z cap from 1 to 2 releases exactly the
intervals the real kernels can carry (29 of 29 refusals predicted), and on a spun-up field the
released intervals move the worst grid cell by at most ~4 ppb with convection, about a quarter
of the 15 ppb MDM and inside what production's glb6×4 already carries at 3600 s.

Still open, in order: **zoom** (glb3×2+nam1×1; `PYD-3` item 2, which needs a tm5-zoom commit
and a new rc), **fresh sources** (both 3×2 arms are transport from the initial field), and
`dkg` for a sub-step that the model never took (bracketed here, not computed). Nothing here
licenses a production change on its own; the two costs of §5 stand.

All three are answered since: zoom and fresh sources in §11, `dkg` by §12's forward runs.

## 11. The nam1×1 child, and a field with sources

Measured 2026-09-29 on Orion; the last two of §10's open items.

**The child changes nothing about the cap question.** The same four week-long windows as §8,
run as production is -- glb3×2 + nam1×1, from `tm5-zoom@397b5a2fa:proj/ct-unified/rc/ch4_3x2nam1x1_pydyn.rc`
-- and replayed with `zoom.check_cfl_zoom` (`analysis/cfl_cap_replay_zoom.py`, which builds the
region tree from the dump header and refuses to run unless it is the model's). The control
reproduces the model's `(ndyn, n)` on **672 of 672** intervals, the reduced parent and the child
together, at ifort. Results: `results/cfl3x2nam1x1_2019-{01,04,07,10}_cap_replay.csv`.

| window | reduced, standalone → zoom | released at `(10,1,2)`, standalone → zoom |
|---|--:|--:|
| 2019-01 | 104 → 110 | 96 → 102 |
| 2019-04 | 142 → 142 | 142 → 142 |
| 2019-07 | 127 → 127 | 100 → 100 |
| 2019-10 | 103 → 103 | 67 → 67 |
| all | 476 → **482** | 405 → **411 (85.3 %)** |

The child adds six reductions -- 2019-01-07 07:00 to 12:00 -- and all six are released at
`z=2`: the child is refused by the same one-sub-cycle z cap as the parent. No interval is kept
reduced because of the child. So §8's 85 % and its 19 % of TM5 root carry over to the
production tree unchanged; what §9/§10 found about the refusals (set equality with the real
kernels) was not re-tested with the child, because the tracer replay under zoom is the §3f
machinery and has not been turned into a step-accuracy experiment.

**Fresh sources change the accuracy answer, through convection.** `tr3x2_2019-10e` is §10's
October window re-run with `without_emission` dropped and started on 2019-09-29, so that the
same 48 intervals -- same meteo, same refusals, 29 of 29 again -- carry a field with 48 h of real
CH4 sources (+7.5 Tg over the four days). Results:
`results/step_accuracy_3x2_2019-10e_{xyz,xyzv-first}.csv`.

| worst cell, the 18 released intervals | transport only | with convection |
|---|--:|--:|
| IC-only field (§10), at 3600 s | ≤ 3.1 ppb (hour 0 aside) | 3.1 – 4.0 ppb (hour 0 aside) |
| **with sources, at 3600 s** (the proposal) | ≤ 6.1 ppb | median **20.5**, worst **67** ppb |
| **with sources, at 1800 s** (production today) | ≤ 1.2 ppb | median **9.4**, worst **32** ppb |

Advection alone barely notices the sources. Convection acting on them does: at the step
production already takes, the worst grid cell of a typical released interval is 9 ppb from a
450 s reference, and the step `(10,1,2)` would take roughly doubles it (n=1 / n=2 ratio median 2.2,
at most 2.6: first order). The 29 intervals that stay reduced carry the same
at 1800 s (median 9.1). Field-wide the error stays small (rms ≤ 0.27 ppb at 3600 s): these are
local hot spots.

**Where: the surface layer over strong sources, not aloft.** Two more runs on the same set
with the worst cell's `(i, j, l)` recorded (jobs `23922289` `--dkg first`, `23922290` `--dkg step`;
the `first` csv equals the one above in every shared column). On the 18 released intervals the
worst cell at 3600 s is at `l = 1` on 15 and `l = 2` on 3 (`step`: 14, 3 and one `l = 3`), and
in the same column as the worst cell at 1800 s on 18 of 18 (`step`: 13). Nine columns in all:

| column `(i, j)` | cell centre | released intervals (`first`) |
|---|---|--:|
| (90, 58) | 88.5°E 25°N, Ganges–Brahmaputra delta | 6, incl. the 67 ppb |
| (76–78, 58–61) | 46.5–52.5°E 25–31°N, around the Persian Gulf | 6 |
| (95, 44) | 103.5°E 3°S, Sumatra | 2 |
| (96, 51), (21, 64), (44, 35) | Mekong; central California; south-east Brazil | 1 each |

At 3600 s the worst cell is above 15 ppb on 16 of the 18; at 1800 s on 4. `dkg` sampling
moves single intervals (per-interval difference median 2.8, at most 20.6 ppb: 2019-10-02 05:00,
61.3 → 40.7) but not the answer: `step` gives median 20.0 / worst 66.6 at 3600 s and 10.5 /
31.7 at 1800 s, production's own sampling. Results:
`results/step_accuracy_3x2_2019-10e_xyzv-{first,step}.csv`.

### What this does to §9's licence

§9 and §10 licensed a narrow claim: raising the z cap to 2 releases exactly the intervals the
kernels can carry, at an accuracy no worse than production already accepts. The zoom half of
that survives intact (above). The accuracy half does not survive fresh sources unqualified.
The *relative* statement holds -- the longer step costs about twice the shorter one, on every
interval, as a first-order scheme should -- but on a field with sources the absolute cost of the
longer step is a worst cell above the 15 ppb MDM on most released intervals, where at 1800 s it
is below it on most. Whether that matters to an inversion depends on where those cells are and
whether observations sample them, which is the next measurement, not a conclusion.

This is also a finding about production as it stands, independent of the caps: at 1800 s,
convection's splitting error near fresh sources already reaches 32 ppb in the worst cell.

## 12. Would the inversion see it? A paired forward run at the obs

Measured 2026-10-09 on Orion, pre-registered in `TODO.md` `PYD-11` item 2 (`53ac94e`) before
either run. Two forward runs of `pyd11-tracer-emis.rc` (sources on, scaling 1.0,
`without_chemistry`), 2019-09-26 .. 2019-11-01, `output.flask : T`, CFL cache off, identical
but for `cfl_max_nloop_z`: 1 (production; job `24009053`) and 2 (an untracked one-line
overlay last in `my.source.dirs`; job `24009063`), checked in both build trees. The scorer is
`analysis/obs_departure.py`; its `--selftest` fails an always-"see" and an always-"not"
scorer. Results: `results/obs_departure_3x2_2019-10f.csv`.

**Gate 0 passed.** Of the 864 hourly intervals, z1 reduced 510 (14 a day, as §8's October
week) and z2 39. Wall clock 23:25 → 20:37 (−12 %, the whole job including IO). Both runs
sampled the same 117,099 obs; 2 unsampled in both were dropped.

**The target system** is the CT-CH4 EnKF, `tm5-zoom:proj/ct-unified/rc/ch4_gcp_monthly_c12.rc`:
the GV+ CH4 v8.0 obspack has no `CT_assim` and the rc no exclusions, so every obs assimilates,
with MDM per group from `mdm_lookup_ch4.csv` (113,341 obs) else 15 ppb (3,756).

| October 2019, Δ = H(z=2) − H(z=1) | value |
|---|--:|
| obs, groups | 117,097, 284 |
| **groups with \|d_g\| > MDM_g/√N_g** | **38** (all d_g > 0) |
| mean (d_g / s_g)² over groups | 1.10 |
| \|Δ\|/MDM median, p95, max | 0.010, 0.158, 1.97 |
| Σ (Δ/MDM)² / N | 6.4e-3 |

By the pre-registered rule the inversion **would see it**. The 38 are all continuous in-situ
(22 tower, 16 surface; 0 of 105 flask, PFP, aircraft or ship groups), all between 38° and
71°N, all positive: d_g 2.3–5.8 ppb against s_g ≈ 0.8–1.3 ppb over ~740 hourly obs. Largest:
Fort Nelson (FNE, 6.8 s), Carnsore Point (CRP, 6.3), Lindenberg at all three heights (3.9–4.4),
Turkey Point (TPD) and Hanlan's Point (HNP) in Ontario, Lägern-Hottwil, Beromünster at all three heights, Karlsruhe.
Field-wide the shift is small and one-signed in the north: median d_g +0.26 ppb at 30–50°N,
+0.22 north of 50°N, −0.01 in the tropics and the south. This is not §11's hot spots (Ganges,
Persian Gulf, Sumatra): no obs samples those cells.

**Exploratory, not pre-registered: where the signal lives.** The sampler averages every model
state in an obs's window, so the step changes its count: `nsamples` (z1, z2) is (3, 1) on
59,830 obs, (1, 1) on 50,275, (3, 3) on 6,446.

| subset | obs | groups over | mean Δ | p95 \|Δ\|/MDM | rule alone |
|---|--:|--:|--:|--:|---|
| `nsamples` equal | 56,756 | 0 of 266 | +0.20 | 0.033 | would not |
| `nsamples` differ | 60,341 | 41 of 274 | +1.53 | 0.233 | would see |

That the count differs is the signature of an hour the change releases, so this split does
not separate transport from sampling: it says the visible part is confined to the released
hours, and that the accumulated difference carried into other hours is below what the
inversion resolves. A symmetric window average cannot by itself make a one-signed +1.5 ppb,
while §11's convection result predicts exactly a raised surface layer over sources; that is
an argument, not a measurement. §13 is the measurement.

Rides along: the run closes items 3 and 4 of `PYD-11` by construction (sources injected,
`dkg` live) and the bias is measured where it accumulates; chemistry is off, and with a CH4
lifetime of about nine years it cannot move a one-month paired difference measurably.

## 13. Transport, not the sampler: both arms sampled at one instant

Measured 2026-10-09 on Orion, pre-registered in `TODO.md` `PYD-11` item 2 (`d983045`) before
the run. Both arms of §12 rerun unchanged (jobs `24009391` z1, `24009407` z2) but for one more
untracked overlay last in `my.source.dirs`: `user_output_flask.F90` with strategy 4's window
at ±0 s instead of ±900 s, so each obs takes the single model state at its own instant. In-situ
obs are stamped at :30, where both arms have a state. Same scorer, same rule (job `24009490`).
Results: `results/obs_departure_3x2_2019-10f-point.csv`.

The method changed from the one first proposed, hourly instantaneous 3D fields: `output.mix`
averages over its interval too, so it carries the same confound, and the sampler change writes
about a hundredth of the bytes.

**Gate 0 passed.** Reductions 510 (z1) and 39 (z2), as in §12, so the same two trajectories.
The overlay took in both builds: obs whose `nsamples` differ between the arms fell from 60,342
to 18. 823 obs had no state at their instant in one arm or both and were dropped. That removes
75 groups, mostly discrete samples whose time is not on a half step (45 surface-flask,
13 aircraft-PFP, 12 surface-PFP, 5 other), none of them among §12's 38.

| October 2019, Δ = H(z=2) − H(z=1) | ±900 s (§12) | ±0 s |
|---|--:|--:|
| obs, groups | 117,097, 284 | 116,274, 209 |
| **groups with \|d_g\| > MDM_g/√N_g** | **38** | **47** (all d_g > 0) |
| §12's 38 still over | — | **38 of 38** |
| d_g(±0 s) / d_g(±900 s) over the 38 | — | median 1.67, range 1.49–1.80 |
| mean d_g over the 209 common groups | +0.73 ppb | +1.18 ppb |
| mean (d_g / s_g)² over groups | 1.10 | 4.17 |
| \|Δ\|/MDM median, p95, max | 0.010, 0.158, 1.97 | 0.012, 0.269, 2.63 |
| Σ (Δ/MDM)² / N | 6.4e-3 | 1.6e-2 |

By the pre-registered reading (≥ 19 of the 38 still over) the signal is **transport**. Fort
Nelson moves 9.8 ppb (11.4 s), Carnsore Point 9.5, Turkey Point and Hanlan's Point about 6,
Lindenberg, Lägern-Hottwil and Beromünster 5.8. The 9 new groups are again NH continuous
in-situ (BLK, BRA, CBW, DEM, GHG06, IPR, JAR, SMR, STE).

The sampler's averaging did not make the signal; it **diluted** it. In §12, z1's in-situ value
was the mean of its :15, :30 and :45 states and z2's was its :30 state alone, and removing that
average raises every one of the 38. That is consistent with the difference peaking at the
instant z2 samples, but this run does not test it. One caveat, named rather than removed: at a
shared instant the two arms' states sit at different points in the split operator sequence
(z2's 3600 s step is half done at :30). That is part of what a longer step is, and here it is
counted as transport. §12's verdict is unaffected either way, since the EnKF samples at ±900 s.

## 14. Which arm is closer to the observations?

Measured 2026-10-13 on Hercules (Orion refused connections that day; the run output is on the
shared `/work2`), pre-registered in `TODO.md` `PYD-11` item 5 (`b61c488`; verbatim in `TODO.history.md`)
before any residual was computed. The scorer is `analysis/obs_residual.py` (`cf592e4`, job `10103585`, 4.5 min). Its
`--selftest` reaches all three verdicts on synthetic groups, and it shows that a mean-residual
scorer gets a pure rescale wrong. Results: `results/obs_residual_3x2_2019-10f{,-point}.csv`.

**Why not the mean residual.** Both arms share an undated `istart 66` field that is not a 2019
state, run `without_chemistry` and carry the priors at scaling 1.0. The network median of the
mean residual H − y is **−99.8 ppb** at z = 1 (−96.3 over the 38), 17 to 50 times d_g.
Against that offset, z = 2's 2–6 ppb upward shift reduces |mean residual| at **37 of 38**
groups. A naive scorer would call z = 2 closer, and all that would reflect is the sign of the
setup's offset.

**The statistic.** Per group, remove a least-squares line in time from y and from each arm's H.
Then Δρ_g = corr(H'_z2, y') − corr(H'_z1, y'), with a day-block bootstrap (one draw of
October's days shared by all groups and both arms, B = 2000). This ignores an offset, a drift
and a rescaling of the model's anomaly, so it judges only the timing and shape of the variability.

**Gate 0 passed.** 117,097 obs, y identical in both arms, the 38 groups recomputed from Δ equal
to §12's. 2 obs with no y were dropped. 173 groups are eligible (≥ 48 obs on ≥ 7 days),
including all 38.

| October 2019, the 38 groups | ±900 s (§12 pair, decides) | ±0 s (§13 pair, robustness) |
|---|--:|--:|
| median Δρ_g | **−0.0031** | −0.0036 |
| its 95 % CI | **[−0.0058, −0.0003]** | [−0.0079, +0.0003] |
| groups whose CI is above 0 / below 0 | 1 / 3 | 1 / 2 |
| verdict by the rule | **z1 closer** | indistinguishable |
| the other 135 groups: median Δρ_g, CI above / below 0 | −0.0004, 12 / 6 | −0.0004, 15 / 7 |
| *secondary:* detrended MSE, median (z1 → z2) | 3963 → 4042 ppb² | 3973 → 4045 ppb² |
| *secondary:* median β_g (z = 2 lowers the MSE iff > ½) | −2.5 | −1.3 |

By the pre-registered rule **production's step is closer, marginally**. The CI clears zero by
0.0003. Four of the 38 groups are resolved individually: Karlsruhe 200 m, Heidelberg and LMU
worse at z = 2, Fort Nelson better. The point-sampled pair agrees in sign and size, but its
CI reaches zero. The groups' correlations are modest (median ρ 0.46 at z = 1), and the longer
step takes about 0.003 off them. The secondary numbers agree in direction: the detrended MSE
rises a median 1.8 %, and β_g < 0 says the time-varying part of the shift points away from the
obs anomaly. Both are confounded with the flux magnitude and neither enters the verdict.

**What this licenses.** The longer step has no accuracy case at the obs: where the inversion
would see it (§12), it does not improve the timing or shape of what the model puts there, and
if anything it worsens it. The case for z = 2 stays what it was, the 19 % of root (§8), bought
with a bias the inversion resolves (§12) and with no sign that the bias is toward the truth.

**What it does not say.** Whether the mean shift is toward the truth. That is not identifiable
from this pair (the setup offset), and with surface obs alone a transport change that rescales
a site's enhancement is degenerate with a flux error. It is the part the inversion would
absorb into its fluxes.
