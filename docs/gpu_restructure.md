<!-- author: Ash Pera (NOAA/GML CIRES) -->
<!-- footer-title: pydyn — restructuring the CUDA path -->

# Restructuring the CUDA path: a scope

A findings note. What a restructured device path -- fewer, larger kernels --
would buy on CUDA, measured against the path as it is (`docs/gpu_port.md` §5), and in what
order to build it. Nothing here is built yet. Measurements: Alpine's `a100_3g.20gb` slice
(42 SMs), `pydyn.gpu.bench --interval --graph` (glb3x2 + nam1x1, 34 levels, convection over 26,
one tracer), traced by Nsight Systems 2025.5 with `--cuda-graph-trace=node`.

## 0. Status

- **Today**: a production interval replayed as a CUDA graph takes 10.0 ms, 41.2 simulated
  years an hour on the slice (13.8 ms traced; the trace's own cost is ~1.4x), 6.8 ms, 60.1 on
  a whole A100 (`gpu_port.md` §5). The estimates below are the slice's.
- **Where it goes** (traced, an interval): the main kernels 7.2 ms; everything around them --
  ~470 operations under 10 us, memsets, gaps -- 4.8 ms; the Pipeline's verdict reads after
  the interval 1.8 ms (§1).
- **A restructure that keeps the bits** removes most of the 4.8 and the 1.8: **~5.5-6 ms,
  ~70 simulated years an hour** on the slice (§3). With the algorithms' freedom (convection
  without an explicit inverse, parallel reductions, FMA) **~4.3 ms, ~95**. On a whole A100
  the restructured interval would be ~3.5 ms (§3).
- **Not reachable this way**: the ~2 ms an interval that the state's memory traffic alone
  would allow. The main kernels are bound by each line's chain of dependent operations, not
  by bandwidth -- region 1's y steps move ~32 MB of state a sub-step in 256 us, about a
  sixth of the slice's bandwidth at one sub-step -- so fusing launches does not shorten them.
- **Next**: one fused operator as a spike, to calibrate §2's estimates (§4).
- **At hundreds of tracers none of this is the bound** (§6): the tracers' advection is.
  Today's line kernels hold at most 16 tracers a thread (`PD_NT_MAX`) and cost more per
  tracer as they hold more (1.2 ms a tracer to 4, 5.8 from 8 to 16); kernels parallel over
  tracers would be bound by the state's traffic, ~0.4 s an interval on the slice at 600. One
  path serves both: lanes split between tracers and faces by the count (§6).

## 1. An interval's anatomy

Every operation of the replayed graph between two "main" kernels is charged to the main
kernel after it: the searches, scalings, edge copies (box programs), flags and memsets that
prepare it. Traced, per interval:

| main kernel | an interval | its own time (us) | ops before it | their time (us) | gaps (us) | ms: main + before |
|---|--:|--:|--:|--:|--:|---|
| `pd_y_steps_coop` (region 1) | 4 | 256 | 12 | 118 | 20 | 1.02 + 0.55 |
| `pd_y_child_steps_coop` | 8 | 107 | 16 | 77 | 21 | 0.85 + 0.78 |
| `pd_x_rows_coop` (region 1) | 4 | 178 | 8 | 94 | 6 | 0.71 + 0.40 |
| `pd_x_child_rows_coop` | 8 | 68 | 10 | 45 | 7 | 0.54 + 0.41 |
| `pd_z_steps_coop` | 12 | 58 | 11 | 82 | 18 | 0.70 + 1.20 |
| `pd_cvf_26` (convection) | 6 | 501 | 3 | 24 | 5 | 3.01 + 0.18 |
| `pd_dynam0_cm` (Setup_MassFlow; Check_CFL's mass-only passes before it) | 4 | 42 | 11 | 258 | 27 | 0.17 + 1.14 |
| `pd_m2phlb1`, `pd_m2phlb_child` | 6 | 34 | 3 | 21 | 0 | 0.20 + 0.13 |
| **total** | | | | | | **7.21 + 4.79** |

The operations themselves, by kind (an interval): 145 box-program dispatches (12.6 us each,
1.8 ms), 154 memsets, 24 device copies, 24 each of `pd_layer_scale`, `pd_ring_spread` and
`pd_ring_means`, the searches' 46 dispatches (0.95 ms), and 68 verdict reads, each a 2 us copy
with the host's turn around it (1.8 ms of gap at `bench`'s one interval in flight; hidden
behind the next interval at `forward.py`'s two, but still host time).

The step itself fixes what can overlap: tm5.F90's two half steps, each region 1's xyz, then
the child's two sub-steps on that, `update_parent`, then region 1's v, s, c
(`zoom/steps.py`). The graph is one chain; a region's operators cannot run beside another's.

## 2. The pieces

**P1. One verdict read an interval.** The speculation's checks (`kernels._converged`, the
`ok` flags, the deferred fatal flags) evaluated on the device into one word, read once.
Saves ~1.8 ms traced at one interval in flight, ~68 host round trips at any depth. Small:
`Pipeline._settle`, `kernels._converged`, `ok_flag`. Decisions only; no state's bits move.

**P2. A directional operator in one or two kernels.** Today x / y / z each run as: the edge
copies to the children, `SEARCH_ROUNDS` search rounds (each with memsets), `pd_layer_scale`,
the steps, `pd_layer_scale` back, the ring averages, `mix_edges`. Fused, by the scope of the
sub-cycle count:

| axis | nloop's scope | fused as |
|---|---|---|
| x | a row | one kernel: the warp that owns the row finds its nloop and steps it |
| y | a layer (`dynamv`) | the search to its fixpoint in one kernel (a layer's maximum over blocks: atomics, grid sync), then the steps with the layer scaling folded in |
| z | the region (`dynamw`) | as y, one maximum for the region |

The in-kernel fixpoint is the sequential search's answer exactly, as the rounds are now
(`gpu_port.md` §2: a maximum of integers crosses threads, nothing else), and it needs no
speculation, so no verdict for it. Saves most of the 3.3 ms before the advection kernels
(x 0.81, y 1.33, z 1.20). Large: six operator variants (region 1 and child per axis) in
`csrc`, and the mass-only ones of P4.

**P3. Fewer box dispatches.** The 145 are copies, scalings and ordered sums at fixed
addresses, staged only where one entry reads another's writes (`gpu/boxes.py`). Two
consecutive stages from different call sites with no hazard between them can be one; the
edge copies before an operator can be its kernel's prologue (P2). ~145 -> ~30 dispatches,
~1.4 ms traced. Medium: a merging pass over a captured interval's stages. Bitwise: each entry's
operations unchanged.

**P4. Check_CFL's passes.** Its mass-only x, y, z (1.14 ms before `dynam0`) take P2's fused
operators with the tracers off.

**P5. No memsets.** A kernel that accumulates a maximum initialises its own output in its
first round instead of a memset before it: 154 operations and their gaps.

**P6. The main kernels.** What keeps the bits leaves little: convection's 163 registers
spilled to shared memory instead of local, or two columns a warp interleaved, for
occupancy (~1.3x on 3.0 ms); the line kernels' occupancy and loads (~1.2x). The bigger gains
there are the algorithms' (convection ~3-5x without the explicit inverse) or a larger GPU.

## 3. Estimates

From §1's traced shares, scaled to the untraced 10.0 ms:

| stage | slice interval | simulated years an hour |
|---|--:|--:|
| today (graph replay) | 10.0 ms | 41 |
| + P1 | ~9 | ~46 |
| + P2-P5 | ~6.5 | ~63 |
| + P6 | ~5.5-6 | ~70 |
| + the algorithms' freedom (convection, FMA, parallel reductions) | ~4.3 | ~95 |

A whole A100 (108 SMs, 2.6 times the slice's) replays today's interval in 6.8 ms: the main
kernels, latency-bound, roughly halve, while §1's 4.8 ms of small operations, latency per
dispatch, do not shrink with SMs. With those removed the restructured interval there would be
~3.5 ms.

## 4. Order and verification

1. **Spike: region 1's y fused** (search, scaling, steps, rescaling) and timed against
   today's sequence for it (1.57 ms traced): calibrates P2 before the other five variants.
2. **P1**, independent of the rest and small.
3. **P2 + P4**, then **P3** (P2's prologues change which box stages remain), **P5** with each.
4. **P6** last, measured per kernel.

Each fused kernel is checked against the kernels it replaces, kept as the control
(`PYDYN_FUSED=0`), bitwise, on `emu` locally and on CUDA (`runs/alpine/arun`); `test_gpu.py`'s
whole steps, free runs and speculated / replayed intervals stand unchanged.

Not in this scope but needed for a production run to see any of it: `forward.py`'s intervals
as graphs (their uploads into fixed buffers before the graph, `gpu_port.md` §7).

## 5. Risks

- **Register pressure**: a fused operator holds a search and the steps' state at once;
  `pd_cvf` reached 163 registers, and launch bounds that spill were no faster.
- **Grid-wide synchronisation** (P2's y and z) needs a cooperative launch inside a CUDA graph
  on a MIG slice; Metal has none, so Metal keeps today's path or fuses x alone.
- **Three implementations of an operator** (the NumPy reference, today's kernels as the
  control, the fused ones) against a minimum-perplexity codebase: the fused kernels should be
  built from the same `csrc` device functions, as `convec_gen` builds its passes, not
  written again.

## 6. Many tracers

Measured on the slice, `bench --interval --graph --nt N` (the slice's day slower: 12.7 ms at
one tracer against §0's 10.0):

| tracers | interval | the state on the device |
|--:|--:|--:|
| 1 | 12.7 ms | 46 MB |
| 2 | 13.7 ms | |
| 4 | 16.3 ms | |
| 8 | 28.5 ms | |
| 16 | 74.8 ms | 295 MB |

At 16 the x and y line kernels of region 1 take 36 ms of the 75: each thread holds a line's
face values for every tracer (`PD_NT_MAX`, the next power of two), which spills as the count
grows, and the kernels refuse more than 16. So at 600 tracers:

- **The bound is the tracers' memory traffic**, not launches: four arrays a tracer (rm and
  its slopes, 16.6 MB a tracer measured), read and written by some thirteen passes an
  interval (both halves' x, y, z of region 1, the child's two sub-steps of each, convection's
  update, sub-cycles) -- ~240 GB an interval at 600, ~0.4 s on the slice, ~0.17 s on a whole
  A100 40 GB (~1 and ~2.4 simulated years an hour). The arithmetic (~200 GFLOP) is below it.
- **The kernels to write are parallel over tracers**: the tracer axis is the fastest in the
  state's layout, so lanes over tracers read every line direction contiguously, and a line's
  mass and flux arithmetic is done once and shared by its tracers. Each tracer's operations
  stay in the reference's order -- tracers never mix -- so this keeps the bits.
- **Convection's inverse** is per column, shared by every tracer; its update becomes a
  26 x 26 matrix times 26 x 600 a column, memory-bound at ~3.5 GB an interval.
- **Speculation's saved state** is a copy of every tracer a slot (~10 GB each at 600): at
  ~0.2-0.4 s an interval the 34 syncs it avoids cost ~1%, so a many-tracer run should not
  speculate (and does not need graphs).
- **Memory**: ~10 GB of state fits the slice's 20 GB without speculation; a whole A100 80GB
  is the target.
- **The inputs**: hundreds of tracers' own emissions read from files would bound the run on
  the host (the Metal run from files was bound by its readers at one tracer,
  `gpu_port.md` §9) -- unless they are made on the device from a few shared fields (an
  ensemble's perturbations, say).

**One path for one tracer and for hundreds.** The two regimes need different mappings and
different orchestration, not different codes; both are chosen per launch and per run, and
neither moves an element's operations, so the bits hold in each.

- *The mapping*: a line's warp splits its 32 lanes between tracers and faces by the count --
  min(nt, 32) lanes over tracers, the rest over the line's faces. One tracer is today's warp
  a line; 8 are 8 x 4; 32 or more is every lane a tracer, walking its line in the reference's
  order. The line's mass and flux arithmetic (tracer-free) is done once into shared memory
  and read by every tracer's lanes; no thread holds more than its own tracers' face values,
  which removes today's spilling between 4 and 16.
- *The orchestration*: speculation, graphs and §2's fusion pay where an interval is ~10 ms
  (one or a few tracers); where it is hundreds, the syncs they avoid are ~1% and the saved
  state gigabytes, so the run goes without. `gz.Pipeline` is already optional and a graph
  opt-in (`submit(fn, repeat=key)`): a switch on the count in `forward.py`.
- *The work*: the five face kernels (`pd_x_rows`, `pd_y_steps`, `pd_z_steps`, the children's
  x and y) and their cooperative forms around the split; convection's update looping over
  tracers a column, nearly as now. Checked bitwise against today's kernels at 1, 2, 16 and
  64 tracers, counts that do not divide 32 among them.
- *Expected, not measured*: one tracer as today (or §3's), 600 near the traffic bound above.

