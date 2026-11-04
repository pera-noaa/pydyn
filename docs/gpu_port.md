<!-- author: Ash Pera (NOAA/GML CIRES) -->
<!-- footer-title: pydyn — the GPU port -->

# pydyn.gpu: the reference with its state on the device

A findings note (a new subject is a new file). The package docstring of
`pydyn/gpu/__init__.py` is the map of the code; this note is what is true about it and how it
was shown. Model pin as `docs/state.md`: `tm5-zoom@745e94f70`.

## 0. Status

- **Every operator the reference runs** runs on the device, the state resident: advection of
  both regions with their sub-cycle searches, convection, Setup_MassFlow, m2phlb, the zoom
  exchange, Check_CFL, emission, the CH4 loss rates, Calc_Kzz (§4). The state comes down
  only to be compared (§1).
- **float64, bitwise to the reference** on the emulation backend (`emu`): the laptop sets'
  free runs, five glb3x2 + nam1x1 checks on Hercules, the production zoom model's own dumps
  per event (§5). The CUDA build has no FMA (§2); closed libraries have open twins (§3).
- **float32 on an Apple M4 (Metal)**, bitwise to the same source's float32 host build
  (`emu32`): a free-run day and 14 days from the input files, 0 differ; after 14 days the
  tracer total is 6.6e-7 from float64 (§8, §9).
- **Speed (Metal)**: a production glb3x2 + nam1x1 interval in 24.4 ms, **16.9 simulated
  years an hour** (synthetic meteo, §8); from the files (glb6x4, `pydyn/forward.py`) **24.0**,
  bound by the reader processes' CPU (§9); on an HPC node read from staged copies (§6).
- **CUDA on Alpine's A100s**, bitwise: `tests/test_gpu.py` with the three free runs, on a
  3g.20gb MIG slice and a whole A100 80GB. The production interval replayed as a CUDA graph
  in 10.0 ms, **41.2 simulated years an hour** on the slice, 6.8 ms, **60.1** on the whole GPU;
  speculated without a graph, as `forward.py` runs it, 37.7 and 39.1: the whole GPU is then
  bound by the host's issue rate (§5).
- **AMD (an MI100, ROCm 6.1, the `cuda` backend through CuPy for HIP)**: bitwise, the same
  `tests/test_gpu.py` with graphs; the interval replayed in 13.9 ms, **29.6 simulated years an
  hour**, bound by the GPU (convection 5.2 ms of 14.3, §10).
- **Open**: `forward.py`'s intervals as graphs (§7); a run from the files at production size
  (§7); float32 chemistry's product rounding, -4.8e-8 of the tracer a day (§8).

## 1. Residency, and how it is shown

TM5's transport is bound by memory traffic (tm5-zoom `docs/memory_bandwidth_floor.md`, every
landed speedup moved bytes; advection's arithmetic intensity 0.5-0.8). A port that copied the
state over PCIe around each operator would be bound by the copy. So the unit of the port is
the array's home, not the kernel: `gpu.zoom.zeros` / `upload` allocate every region's `m,
rm..rzm, am, bm, cm, pu, pv, sp, phlb, dxyp` on the device, with their Fortran bounds and
halos (`FView` over a device array), and every function the replay calls keeps them there.

Two mechanisms make that checkable rather than claimed:

| mechanism | what it catches |
|---|---|
| `emu.EmuArray`: the emulated device's arrays refuse what CuPy refuses -- arithmetic with a host array, `np.asarray`, assigning a host array -- raising `ResidencyError` | an implicit transfer anywhere in the device code; a run on `emu` is a residency check |
| `backend.Ledger`: every transfer goes through `to_device` / `to_host` with a tag | what crossed, how often, how many bytes; `state_downloads` is the non-verify, non-control total |

A glb6x4 day on every input file (`repin_phys`, `--free-run --meteo --emissions --chemistry
--initial`), emu:

| direction | tag | transfers | MB | what |
|---|---|---:|---:|---|
| h2d | seed | 5 | 0.014 | the tree's constants (at, bt, row lengths, owned masks) |
| h2d | input | 1180 | 389 | what the model reads from files, when it is read |
| h2d / d2h | control | 184 / 460 | 0.01 / 0.01 | sub-cycle candidates and counts, error flags |
| d2h | verify | 4806 | 4107 | every array of every checkpoint, compared and fingerprinted |

Nothing else comes down. The verify traffic is the replay's comparison, not the model: a run
that compared nothing would move only the inputs up and the counts down.

## 2. Why the device's bits are the reference's

One C source (`pydyn/gpu/csrc/`) is compiled for each device: by NVRTC with `--fmad=false`
(CUDA, through CuPy), by the host compiler with `-ffp-contract=off` (the emulation backends),
and, in float32, by `xcrun metal -ffp-contract=off` (§8). The kernels are transcriptions of
the NumPy, statement for statement, so each value is the same sequence of IEEE operations.
For the float64 pair (`cuda`, `emu`) three conditions make "the same operations" mean "the
same bits":

1. **No contraction.** An `a*b + c` fused into an FMA rounds once instead of twice. Shown
   statically: the module compiled with nvcc 12.1 for sm_80 (Hercules, `gpu1/`) has
   `fma.rn.f64` 0 times with `--fmad=false` and 388 times without; every double
   instruction is a `.rn` form (`add`, `sub`, `mul`, `div`, `rcp`, `sqrt`), no `approx`, no
   device-libm transcendental (`ex2`, `lg2`: 0). `rcp.rn.f64` is the correctly rounded
   `1.0 / x`.
2. **No order chosen by the scheduler.** A line's work (an x row, a y or z column, a
   convection column, a Calc_Kzz column) is one thread, sequential in the reference's order;
   where a kernel spreads a line over threads (Metal's convection, §8) each element's own
   operations keep that order.
   What crosses threads is order-free: the max of integers (sub-cycle counts), the max of
   non-negative doubles through their bit patterns (Courant maxima), an or of flags.
   NumPy's own reductions the reference uses are reproduced as loops (`seqsum`, the
   reduced grid's cluster sums, `mix_edges`' block means, `update_parent`'s block sums), in
   the reference's index order.
3. **NumPy's semantics where they are not IEEE's.** `np.maximum` / `np.minimum` propagate NaN
   and order -0 below +0; `np.clip` is its own ufunc, `min(max(x, lo), hi)` with NaN tests
   (`csrc/pd.h`). `==` (the repo's bitwise) does not see a zero's sign; the helpers match it
   anyway.

The sub-cycle searches are the one place the device does not follow the reference's loop:
dynamv's per-layer and dynamw's region-wide nloop are found by a fixpoint of per-column
searches (`kernels._search`): each column reports the smallest nloop >= the candidate it
passes, the maximum becomes the next candidate, until a round returns it. That is the
sequential search's answer exactly -- every nloop below the fixpoint is failed by the column
that raised it, and each column's trial at a given nloop is a function of that nloop alone
(the faces scaled from the originals by the same sequence of operations).

The `emu` backend is these kernels as plain loops on the CPU. Its agreement with the
reference is therefore the device's, given that NVRTC with `--fmad=false` emits what nvcc
emitted for the PTX above; the CUDA runs of §5 are bitwise.

**A gate that can fail.** `PYDYN_EMU_CONTRACT=1` compiles the same source with
`-ffp-contract=fast -march=native`; on arm64 the advection test fails on it, and `pydyn.gpu.check`
on the transport set reports differences in every operator from the first x step
(`tests/test_gpu.py::test_the_checks_fail_under_contraction`).

## 3. The twins

The reference is bitwise to the model "through the model's own libraries" (`docs/state.md`
§3c): its LAPACK for convection's inverse (MKL under ifort, Accelerate on the laptop) and its
libm for the chemistry's rates and Calc_Kzz (libimf, glibc, Apple's). Those cannot run on a
GPU, so the port carries open twins compiled from the same source into the host library
(`backend.host_library()`), which the reference loads with `--lapack twin --libm twin`:

| twin | source | against the library it stands in for |
|---|---|---|
| `dgetrf_` / `dgetri_` | `csrc/convec.cuh`: reference LAPACK as `lapack.LapackInverse` calls it (lwork = n: DGETRF2's recursion, DGETRI's unblocked sweep; reference BLAS loops, no FMA) | Accelerate: 0.2-8.5 % of `rm` values differ after one convec of synthetic columns |
| `exp` / `log` / `pow` | `csrc/pmath.cuh`: fdlibm (`e_exp.c`, `e_log.c`, `e_pow.c`, `s_scalbn.c`; Java's StrictMath) | Apple's libm: at most 1 ulp on 134 000 arguments; 9 % of `exp`, 4 % of `log`, 9 % of `pow` differ |
| `cos` / `sin` | the system libm's, through the twin's exports | identical: the geometry the device uses is computed on the host |

So the claim has two links. Device = reference-with-twins, bitwise (§5). Reference-with-twins =
reference-with-the-model's-libraries to within those ulps, which is the same kind of
distance `docs/state.md` §3c already records between libimf and glibc. The arithmetic that
calls no library -- advection, the CFL searches, the convection matrix, Crout, dynam0,
m2phlb, the exchange, emission -- needs neither link: it is bitwise to the model's own dumps
on the device (§5, the zoom fixture).

The Crout path (`--lapack crout`, the build without `with_lapack`) is open on both sides and
needs no twin.

## 4. What is ported

| reference | device | kernel (`csrc/`) |
|---|---|---|
| `kernels.advect_x` / `_advect_rows_x`, `subcycle_rows` | `gpu.kernels.advect_x_rows` | `pd_x_rows` (every (j, l) row, its own nloop; reduced rows by their length) |
| `kernels.advect_y` + `zoom._rescale_layer_halo` | `gpu.kernels.advect_y` | `pd_y_search` (fixpoint), `pd_layer_scale`, `pd_y_steps` |
| `kernels.advect_z` | `gpu.kernels.advect_z` | `pd_z_search`, `pd_z_steps` |
| `redgrid.uni2red` / `red2uni` / `uni2red_mf` | `gpu.exchange` | `pd_boxes` (box programs, §8) |
| `zoom.advect._dynamu_child` / `_dynamv_child` / `_dynamw_child` | `gpu.zoom` | `pd_x_child_rows`, `pd_held_search`, `pd_y_child_steps`, `pd_z_scope_search`, `pd_limit_box` |
| `zoom.exchange` put_x/yedges, compress / uncompress, mix_edges, update_parent; `zoom.advect`'s outer faces and halos | `gpu.exchange`, `gpu.zoom` | `pd_boxes`, `pd_ring_means` + `pd_ring_spread`, `pd_block_sums` |
| `zoom.massflow` dynam0, coarsen_region, coarsen_ambm, m2phlb(1) | `gpu.zoom` | `pd_scale_copy`, `pd_dynam0_cm`, `pd_m2phlb1`, `pd_m2phlb_child`, `pd_boxes` |
| `kernels.convec` (Crout, LAPACK) | `gpu.kernels.convec` | `pd_convec` (a column per thread, PD_LMAXC levels); for lmax <= 32 `convec_gen`'s passes, a SIMD group / warp a column (§8): four kernels on Metal, one on CUDA (`pd_cvf_<n>`, the matrix in shared memory) |
| `zoom.check_cfl`'s bodies and `cfl.dynamum/vm` | `gpu.zoom` | the x / y / z kernels with nt = 0 |
| `zoom.physics` sources / chemistry | `gpu.zoom` | `pd_add_2d`, `pd_scale_owned_all` |
| `chem.loss_rates`, `chem.decay` | `gpu.kernels.loss_rates` | `pd_loss_rates` (rebin_many per column, fdlibm) |
| `diffusion.calc_kzz` | `gpu.diffusion.calc_kzz` | `pd_calc_kzz` (a column per thread, PD_LMK levels) |

The driver is written once. `zoom.steps` (do_steps, Proces_Region, step), the advect*zoom and
Check_CFL entry points, coarsen_region's recursion and Setup_MassFlow run on any `ZoomState`
and call the bodies above through its `lib`: `zoom.host` for NumPy arrays, `gpu.zoom` for a
state made on the device (`tests/test_gpu.py` checks `gpu.zoom` has every one). The rest a
run needs comes through an engine (`replay.HostEngine` / `gpu.engine.DeviceEngine`):
uploads, comparisons, Calc_Kzz, the loss rates, `sync` and `pipeline` (on the host each call
runs as it is submitted). `FreeRun`, the per-event replays and `pydyn.forward` take one and
have no device branches of their own.

Not ported: dynamw's `xi` diagnostic (the largest Courant number up to the Fortran's first
violation; stats only, never compared by the replay) -- `stats["xi"]` is None for z on the
device.

## 5. Results

Emulation backend, laptop (arm64, clang 17, NumPy 1.26.4), `--lapack twin --libm twin` on
both sides; `pydyn.gpu.check` matches the sha1 of every compared array of every checkpoint.

| what | checkpoints | arrays fingerprinted | result |
|---|---:|---:|---|
| `tests/test_gpu.py` (kernels, a reduced-grid region's whole step and Check_CFL, four zoom trees' whole steps and Check_CFL, the twins, residency, the contraction gate, the CUDA source compiled) | -- | -- | 36 passed, 5 skipped (speculation, the scratch arena and graphs are CUDA's) |
| `repin_free_745e94f70` free run (x y z v, massflow, cfl, m2phlb) | 299 | all | equal |
| `repin_phys_745e94f70` free run on the dumped inputs (+ s, c) | 391 | all | equal |
| `repin_phys_745e94f70` free run on the input files alone (+ Calc_Kzz, `in:dkg` 46 / 46) | 392 | all | equal |
| `repin_cfl_745e94f70` free run through a reduced time step | -- | all | equal |
| `zp_fixture` (glb3x2 + nam1x1, ifort, Orion's flags) per event against **the model's dumps**: x y z of both regions, s, massflow, cfl | 10 | -- | bitwise, no twin |

Hercules (x86-64, GCC 11.3; `runs/hpc/gpu_check.slurm`, jobs in `runs/MANIFEST.md`), the
glb3x2 + nam1x1 sets of `docs/state.md` §3e / §3f, both regions:

| set | run | checkpoints | arrays fingerprinted | result |
|---|---|---:|---:|---|
| `zps_replay` | free run on every input file (meteo, Calc_Kzz, emission, chemistry, initial field) | 440 | 6 160 | equal |
| `zps_replay` | per event on every input file | 440 | 3 140 | equal |
| `zpfix_replay` | free run on every input file (the MPI run, ranks merged) | 440 | 6 160 | equal |
| `zp4_serial` | free run, four tracers, `--meteo --chemistry --initial` | 440 | 6 160 | equal |
| `zm_replay` | free run, `--meteo --initial` | 320 | 4 120 | equal |

CUDA on Alpine (CuPy 13.6, CUDA 12.9 runtime, NVRTC `--fmad=false`; `runs/alpine/`, the
copies in `runs/MANIFEST.md`), on an `a100_3g.20gb` MIG slice (42 SMs, `hold.slurm`) and a
whole A100 80GB PCIe (108 SMs, `full.slurm`): `tests/test_gpu.py` 37 passed, 4 skipped (no
NVPTX clang; the emulator's own check) on each, the `repin_*` free runs and `zp_fixture` per
event bitwise. `pydyn.gpu.bench`, glb3x2 + nam1x1, wall a step:

| what | slice | years an hour | whole A100 | years an hour |
|---|--:|--:|--:|--:|
| step | 9.3 ms | 44.1 | 7.9 ms | 52.3 |
| Check_CFL + Setup_MassFlow + step (an interval) | 14.9 ms | 27.6 | 13.0 ms | 31.5 |
| the interval speculated (`gz.Pipeline`, `forward.py`'s) | 10.9 ms | 37.7 | 10.5 ms | 39.1 |
| the interval replayed as a CUDA graph (`bench --graph`) | 10.0 ms | 41.2 | 6.8 ms | 60.1 |
| a step replayed | 8.3 ms | 49.2 | 4.3 ms | 94.6 |

What each piece removed (on the slice unless named):

| change | what it removed |
|---|---|
| speculation (`gz.Pipeline`, §8) | 34 syncs an interval (the sub-cycle searches' and Check_CFL's reads), the GPU idle ~6 ms |
| launches through `cuLaunchKernel` with kept parameter buffers (~3 us a launch, CuPy's call ~20) | the host's issue: 14.9 -> 9.5 ms of host time an interval |
| scratch from the Pipeline slot's arena (`Backend.scratch`), at fixed addresses | 178 allocations and 24 uploads an interval; host 8.4 ms |
| the interval as a CUDA graph: `Pipeline.submit(fn, repeat=key)` captures a slot's second call of a key and replays it; an upload, read back, sync, mark or new scratch inside refuses the capture (`backend.Uncapturable`) | the remaining issue: 10.9 -> 10.0 ms; on the whole A100, host-bound without it (10.5 ms against 6.6 of dispatches), 10.5 -> 6.8 |

What remains on the slice is the GPU's own span: the box programs' 145 dispatches ~2 ms,
convection (`pd_cvf`) 2.5 ms -- latency-bound: 163 registers, 12 warps an SM, and launch
bounds that spill are no faster -- and the gaps between ~350 launches. `docs/gpu_restructure.md`
scopes fewer, larger kernels.

## 6. What stays on the host

The ingestion -- reading the meteo archive, the emission and chemistry files and the
initial field, regridding the 1x1 meteo onto each region, and Meteo_Setup_Mass, whose
Poisson balance is an FFT -- runs where the files are read, once per read interval, and its
products go up as "input". The FFT is the one computation of the reference whose bits are
fixed by FFTW's codelets; cuFFT would not reproduce them, and an open twin of FFTW's plan is
not a small thing. The step's time interpolation of pu / pv between the balanced pairs is
done on the host too (pu and pv go up once per step).

Its cost, measured on the local glb6x4 archive (14 days, `runs/MANIFEST.md`): reading and
balancing a read interval -- both meteo instants balanced against the interval's
surface-pressure change, as the model does every interval -- and its convection, temperature,
humidity and surface fields take ~22 ms on one CPU thread, 14.6 of them in 18 netCDF records.
Nine of those records are distinct: the 3-hourly model-level instants are shared by an
interval and its neighbours, the seven hourly 1x1 surface fields are not (each ~1.3 ms to
inflate).

`pydyn/forward.py` reads in `--readers` spawned processes (default 3), each doing whole
intervals:

| what | why |
|---|---|
| a reader takes a 3-hour window of consecutive intervals, and `meteo.Archive` keeps each parameter's last 3 records (read-only) | the model-level instants read once a window, not twice an interval |
| an interval comes back through a shared-memory block (`forward.share` / `unshare`), the blocks reused | a pipe's pickling of its 10.7 MB is the main process's work under its GIL: 2.9 ms an interval to copy out of a new block, 1.3 out of a reused one |
| `--stage DIR`: `meteo.Stager` copies the archive's next two days under DIR (a node's local disk) with 8 threads; an Archive reads a copy once it is complete | the parallel file system's small reads (below) |

On Hercules (`runs/MANIFEST.md`, jobs `10158109`, `10158158`), a day of the archive read
record by record as `pydyn.meteo` does:

| a simulated day | glb6x4 (65 MB) | glb3x2 (102 MB) |
|---|---:|---:|
| netCDF reads from `/work2`, cold | 7.0 s | 2.4-2.7 s |
| the same, read again | 2.8 s | 1.1-1.3 s |
| the day's files copied to the node's local disk, one thread | 1.2 s | 1.6 s |
| the same, 8 threads / 16 threads | | 0.17 / 0.10 s |
| netCDF reads from the copies | 0.45 s | 1.0 s |

A day on the M4's device takes ~0.3 s (§9), so reading `/work2` directly would bound the run
eight to twenty times over; staged, the copies keep ahead and the reading is the readers'
CPU, spread over the processes. glb3x2 + nam1x1 has about five times the cells (§7).

## 7. Not yet known

- **`forward.py`'s intervals as graphs.** Only a graph uses a whole A100 (§5), and forward's
  intervals upload the interpolated winds, Calc_Kzz's, the emissions' and the chemistry's
  inputs each step, which refuses the capture: they would have to land in fixed device
  buffers before the graph. Also not done on CUDA: 32-bit indices.
- **A run from the files at production size.** §9 is glb6x4, the only local archive; at
  glb3x2 + nam1x1 the device's interval is §8's (GPU-bound: the box programs' 145 dispatches
  4.9 ms, convection's passes 7.3, z 2.9, the loss rates 2.4, Calc_Kzz 1.5 of 23.7), the
  reading (§6) about five times §9's, for more reader processes, and on an HPC node from
  `--stage` copies (§6's measurements; the forward driver itself not yet run there).
- **float32 over years.** §8's 14 days: the total drifts linearly at 4.8e-8 a day, all of
  it the chemistry's product rounding; a year's extrapolation is 1.8e-5 (~0.03 ppb of CH4).
  Whether the cells' 1.2e-5 keeps growing as √t is not measured.

## 8. Metal: float32 on an Apple GPU, and the throughput

Apple GPUs have no double, so the `metal` backend (`gpu/metal.py`, `csrc/metal_shim.mm`)
runs csrc/ with `real = float` (`PD_F32`). Its bits are not the reference's; they are those
of `emu32`, the same source built on the host with `PD_F32`, and the claim tested is that
pair, bitwise.

**Why the pair is bitwise.** On 4 M random float32 pairs, including extremes, the M4's add,
mul, div, sqrt, fma, reciprocal and int conversion equal the CPU's bit for bit once the CPU's
FPCR.FZ is set: the GPU flushes subnormal inputs and outputs. `emu32` sets FZ for every
kernel's duration (pd.h). The Metal compiler contracts `a*b+c` unless told not to, and only
the offline `xcrun metal -fno-fast-math -ffp-contract=off` can be (the runtime compiler has
no such option), so the library is built offline. exp / log are msun's float algorithms
(within 1 ulp), pow is exp(y log x) with log carried as hi + lo through `fma` (within 2 ulp).
`tests/test_metal.py` (42): every `test_gpu.py` case Metal against emu32, convection's passes
and above 32 levels (`convec_metal.h`), six trees' whole steps, Check_CFL, every line kernel
cooperative, an interval with Calc_Kzz, emissions and chemistry, speculation kept and
redone, the float math's ulps, and a contracted build caught. On real data:

| set | run | checkpoints | fingerprints | differ |
|---|---|---:|---:|---:|
| `repin_free_745e94f70` | `pydyn.gpu.check --device metal --reference emu32 --free-run --lapack crout` | 299 | 3 197 | 0 |
| 2000-01-01 .. 14 21:00 from the files | `pydyn.forward --device metal` against `--device emu32` (§9) | 14 | every state array | 0 |

**Drift from the reference** (the same day, Metal against the model's float64 dumps, the
largest absolute difference over the largest value, per checkpoint):

| | first checkpoint | median | after 24 h |
|---|---:|---:|---:|
| m | 3.3e-8 | 1.3e-6 | 1.8e-6 |
| rm | 3.1e-8 | 2.9e-6 | 4.7e-6 |
| total tracer mass (relative) | 1.5e-10 | 3.0e-7 | 5.9e-7 |

**float32's own failures**, found on §9's 14 days and fixed in the float build only (`#ifdef
PD_F32`; the double's arithmetic is the Fortran's, unchanged):

| | what float does | fix | before | after |
|---|---|---|---|---|
| Calc_Kzz NaN (`kzz.cuh`) | two adjacent virtual potential temperatures round to one float (the double's difference, 5.4e-6 K, is under half an ulp at 300 K): `HV / thvgrad` is inf, and the Fortran's `jq * a + (1 - jq) * b` makes 0 * inf = NaN when that branch is unused; in the PBL-height pass two equal Richardson numbers do the same | `pd_blend`: a select for those two blends; `pd_resolved`: a difference under half an ulp of its operand taken at that size | NaN at hour 3, the whole tracer field by hour 16 | finite over 14 days |
| convection's column mass (`convec.cuh`, `convec_gen.py`, `convec_metal.h`) | conv1's diagonal is 1 + a small outflow whose low bits the float drops while the off-diagonal inflows keep theirs: its column sums miss 1 by an error of one sign, and the inverse, kept for the 3-h interval, applies it every step | `pd_column_deficit`: the column's tracer before minus after, each TwoSum-summed, added to its largest level (the deficit is under an ulp of every level, so no scaling can carry it) | -1.19e-8 of the tracer a call, every call | 6e-12, either sign |

Over §9's 14 days, Metal against the float64 CPU run (the model's bits, §9):

| hour | 24 | 96 | 168 | 240 | 333 |
|---|---:|---:|---:|---:|---:|
| tracer total, before the convection fix | -6.3e-7 | -2.5e-6 | -4.5e-6 | -6.5e-6 | -9.0e-6 |
| tracer total | -4.9e-8 | -1.9e-7 | -3.4e-7 | -4.8e-7 | -6.6e-7 |
| 99th-percentile cell | 4.1e-6 | 7.8e-6 | 9.4e-6 | 1.0e-5 | 1.2e-5 |
| air mass total | 1.5e-9 | 9.6e-10 | 5.0e-10 | -5.0e-11 | -3.0e-9 |

The remaining linear -4.8e-8 a day is the chemistry: by 72-hour runs with each process off,
transport and emission drift 1e-9 with no trend; the float decay itself is unbiased (2.2e-11
a call, rm-weighted), while the rounding of `rm * decay` loses 1.0e-9 a call, 48 calls a day.
The chemistry conserves nothing a column could be restored to, so it is left measured.

Memory is one shared-storage arena that the GPU addresses and the CPU sees as NumPy views
(`MetalArray`, EmuArray's residency rules); Apple silicon has no separate device memory, so
"resident" means in the arena, and the ledger counts uploads and downloads as on any backend.

**What made it fast.** `python3 -m pydyn.gpu.bench --device metal`: glb3x2 with its reduced
grid + nam1x1, 34 levels, convection over 26 (tropo34's), synthetic meteo, production-sized
Calc_Kzz, emission and chemistry inputs (the rates computed at every `c`); every timed step
starts from one of two saved states (masses 1e-6 apart, so no step reuses the last one's
convection). At ndyn = 3600 s a step is a simulated hour.

| change | what it removed |
|---|---|
| convection as `convec_gen.py`'s four unrolled passes per lmax (build / right-looking LU / sweeps / update), each element's operations in pd_crout's order; the divisions only one lane needs done once by that lane | thread-local 13 KB arrays in device memory: 26 -> 3.3 ms a 10 800-column call |
| the LAPACK kind as one more pass, `pd_cvg` (below) | the generic kernel's 3.9 ms a call at glb6x4 -> 0.60 |
| `pd_cvc`: each column's inputs compared bitwise with the copy kept, the inverse reused where equal | half of region 1's inverses (a step's two convec calls see the same mass) |
| box programs (`boxes.py`, `csrc/boxes.cuh`, `exchange.py`): the exchange, compress / uncompress, halos, outer faces, the reduced grid, coarsen, Check_CFL's save / restore, staged and checked exactly for read / write hazards, built once a state | 1 044 -> 0 NumPy operations on device arrays a step, each a sync |
| `mix_edges` as two ring kernels over every field's slices by address | 60 whole-region launches a step |
| 32-bit indices on Metal; PD_NT_MAX per launch | 64-bit integer division; 16-tracer face arrays in local memory |
| command buffers committed every 8 dispatches | the GPU idle while Python queued |
| speculation (`gpu.zoom.Pipeline`): searches take their last queued round, Check_CFL's operators pass, the fatal flags wait; one sync at the end checks it all; on a failure the state is restored and the interval rerun deciding as it goes | ~34 syncs an interval (all but the last) |
| `Backend.put` on an asynchronous device: the upload to an inbox and a COPY box (below); the loss rates' level-order flag a `check` | a sync at every pu / pv upload and every chemistry call |
| the line kernels (x / y / z sub-steps, a child's x / y, the nloop searches) a SIMD group a line (`csrc/advect_coop.cuh`, `coop.py`, below) | z's 10 800 sequential columns: pd_z_steps 2.94 -> 0.56 ms an interval |
| the launch and the queue (below) | 28 -> 17 µs a launch, every sync but a run's last |

**Convection's LAPACK kind as a pass** (`pd_cvg_<n>`, convec_gen's `_getrf_getri`): DGETRF2 +
DGETRI of convec.cuh on a SIMD group / warp a column, lane p holding row p in registers. DGETRF2's
recursion gives every element `c - u * l` in ascending k (DTRSM's form and DGEMM's `c + (-1 *
u) * l` are one IEEE operation), then the leaf's scaling by 1 / pivot: a right-looking LU. The
one difference the recursion makes is kept: DTRSM skips u == 0 and DGEMM does not, which shows
in a signed zero (conv1 holds -0), so each (k, column) update knows from the recursion which
rows were DTRSM's. Pivoting is IDAMAX's first largest |x| over the lanes, rows swapped whole;
DTRTI2's columns and DGETRI's sweep are broadcasts; the column interchanges move registers.
`tests/test_metal.py`: bitwise to emu32's serial twin (`pd_inverse_rows`) on dense matrices
(they pivot), conv1-like ones and sparse ones with zeros of both signs, n 2 .. 32; with DSCAL's
reciprocal replaced by a division, or DTRSM's skip dropped, it fails (253 346 and 8-19
elements); a forward run of 12 hours bitwise with either kind. Cost on the M4:

| | glb6x4, a call | glb3x2 + nam1x1, an interval's convection (the bench, `--inverse`) |
|---|---:|---:|
| Crout (`pd_cvl` + `pd_cvs`) | 0.21 ms | 2.3 ms |
| LAPACK (`pd_cvg`) | 0.60 ms | 5.1 ms |
| LAPACK, the generic one-thread-a-column kernel | 3.9 ms | |

Crout against LAPACK in float64 (the CPU reference, 24 hours of §9's run): m identical, rm at
most 35 ulp apart (4e-15, the 99th percentile 10 ulp), the tracer total equal to 2e-16; the
slopes, differences of near-equal values, up to 1e-5 of themselves. conv1 is column diagonally
dominant, so DGETRF's pivoting never swaps and the two are one factorization in another
order. LAPACK is the default (the reference's bits); `--lapack crout` the option.

**The line kernels a SIMD group / warp a line** (`csrc/advect_coop.cuh`, `pydyn/gpu/coop.py`).
Within a sub-step every face is a function of the line's old cells and every cell of its own
two faces, so 32 lanes compute the faces (strided, one tracer at a time into group memory),
then the cells, then the mass, with a barrier between: each element's operations are the
serial kernel's in its order. An x or held-ends line's nloop search is a trial per face, the
row's verdict the lanes' AND and its largest Courant number their maximum (exact, as any
max); a y line's |beta| maximum is kept per sub-step, so it holds with no tracers. `coop.py`
writes the Metal and CUDA kernels (`<kernel>_coop`, the serial kernel's arguments) around the
bodies; emu and emu32 keep the serial kernels, which the tests compare against. Which run
cooperatively is the backend's (`PYDYN_COOP=0` / `all` override it): CUDA every one; Metal
the four an M4 runs faster so, since it holds ~20 000 threads and x and y rows already give it
that many (glb3x2 + nam1x1, ms an interval):

| kernel | serial | a SIMD group a line | Metal's default |
|---|---:|---:|---|
| pd_z_steps | 2.94 | 0.56 | cooperative |
| pd_x_child_rows | 0.77 | 0.51 | cooperative |
| pd_y_search | 0.52 | 0.31 | cooperative |
| pd_held_search | 0.39 | 0.21 | cooperative |
| pd_x_rows | 0.86 | 1.02 | serial |
| pd_y_steps | 0.46 | 1.08 | serial |
| pd_z_scope_search | 0.43 | 0.57 | serial |
| pd_y_child_steps, pd_z_search | 0.33, 0.21 | 0.32, 0.25 | serial |

`tests/test_metal.py`: the defaults in every test, and every kernel cooperative
(`PYDYN_COOP=all`, CUDA's choice) in a whole step and Check_CFL of four trees, bitwise to
emu32; a reordered sum in each kernel, or a search one nloop off, fails them. The CUDA kernels
compile for sm_80 without FMA (`tests/test_gpu.py`).

**The launch and the queue.** The host's share, per call on the M4 in §9's run:

| change | before | after |
|---|---:|---:|
| `Backend.launch`: a kernel's checks once (`_plan`), EmuArray's dtype / shape / strides as properties; Metal's argument struct a kept `struct.Struct` and an array's address kept on it; CUDA's function and scalar converters kept; only `ROUTED` kernels routed | 28.4 µs a launch | 16.6 |
| `Backend.put`: the upload into an inbox of the view's whose last COPY has run, its COPY program kept | 170 µs a put | 75 |
| Calc_Kzz's strided `sp` by `Backend.contiguous` (a kept COPY into a kept scratch) | a sync an interval | none |
| `zoom.Pipeline` (`forward.run`, `--pipeline 2`): interval i + 1 queued on the state i is assumed to leave; i checked once a mark after it has run, its flags read without a sync; a failed i restored and redone carefully, i + 1 queued again | 2 syncs an interval | 1 a run |

A mark is a committed command buffer on Metal (`pdm_mark` / `pdm_wait`), an event on CUDA.
Every reuse waits for the mark past it rather than for a sync: the arena's frees, put's
inboxes and CUDA's pinned staging chunks carry the queue's position when released. CUDA
queues on a non-blocking stream of its own, an upload a copy into pinned memory and a
`copy_from_host_async`, the pipeline's flag reads on a second stream.

| claim | test | result |
|---|---|---|
| two intervals in flight are emu32 through a failed speculation | `test_forward.py`: Metal, 6 hours, the third's speculation corrupting rm and failing | bitwise, 5 kept and 1 redone; 7 of 10 state arrays differ without the state put back, or with the fourth kept from its speculation |
| CUDA's uploads, marks and flags | `test_cuda_host.py` (6): a cupy whose copies run only at a synchronize | staging held until its mark, a wait that runs nothing after its mark, deferred flags raised at the sync or a download; a staging block reused early fails |

Results, the median of 40 steps, the LAPACK kind:

| what an interval does | tracers | ms | simulated years an hour |
|---|---:|---:|---:|
| baseline: the step on `emu` (one CPU thread) | 1 | 640 | 0.6 |
| baseline: the step, the first Metal build | 1 | 262 | 1.6 |
| the step, careful | 1 | 18.1 | 22.7 |
| the step, speculated | 1 | 15.4 | 26.8 |
| Check_CFL + Setup_MassFlow + Calc_Kzz + the step with emission and chemistry, speculated | 1 | 24.4 | 16.9 |
| the same | 4 | 34.7 | 11.8 |

These are the device's interval (GPU busy 23.6 of the 24.4 ms); the host's ingestion (§6) is
not in them. §9 has a run from the files.

## 9. The forward run from the files

`pydyn/forward.py` (`python3 -m pydyn.forward --meteo ROOT --emissions DIR --chemistry DIR
--initial FILE --device metal --days 13.875`) runs replay.FreeRun's sequence with no dump set:
the clock and a `RunConfig` (ch4_6x4_fwd.rc's values) in place of the dump headers. Per read
interval: Meteo_Setup_Mass on the host, Check_CFL on a copy, then Setup_MassFlow, Calc_Kzz and
the step, the emission and chemistry at their half steps; on an asynchronous device the
interval is speculated, two in flight (`gpu.zoom.Pipeline`, §8), and the next intervals are read
while the device carries these (§6: `--readers` processes, default 3; `--stage`). The
convection inverse is the reference's LAPACK kind (`pd_cvg`, §8), `--lapack crout` the option
(the 14-day float32 drift -6.5e-7 with it, -6.6e-7 with LAPACK). `tests/test_forward.py` (9),
`tests/test_share.py` (2):

| claim | test | result |
|---|---|---|
| the CPU run is the model's | 3 hours against `repin_phys_745e94f70`'s aft_x / m2phlb, every state array; 23 hours by hand | bitwise, 0 of 253 arrays differ |
| the comparison can fail | the OH scaling one ulp up | differs from hour 2 |
| emu is the CPU run with the twins | 3 hours, every state array; 24 hours by hand (across day 2's files) | bitwise |
| reader processes read what the thread reads | 7 hours (three windows over two processes, shared memory), readers 2 against 0, every state array | bitwise |
| a staged run reads the copies, and the same bits | 3 hours with `stage`, the intervals' files opened from the copies | bitwise; fails if the Archive reads the originals |
| an interval crosses whole | share / unshare of nested results, strided arrays, a block reused and one outgrown | equal; a block written again before it was read differs |
| Metal is emu32, and finite, with either inverse | 12 hours (through hours 3 and 11, the Kzz cases); 333 hours by hand, 14 snapshots | bitwise; non-finite without the Kzz fix |
| two intervals in flight are emu32 through a failed speculation | 6 hours (§8) | bitwise; fails without the restore or the requeue |

Throughput on the M4, glb6x4, one tracer, 333 hours (2000-01-01 .. 14 21:00; the next
interval needs 15 January's fluxes), `python3 -m pydyn.forward --hours 333`, the start
included:

| what | s | ms a simulated hour | simulated years an hour |
|---|---:|---:|---:|
| one reader thread, LAPACK's generic kernel | 11.1 | 33.4 | 12.3 |
| one reader thread, Crout | 9.9 | 29.7 | 13.9 |
| two reader processes, Crout | 6.6 | 19.9 | 20.7 |
| two reader processes, LAPACK (`pd_cvg`), one interval in flight | 6.8 | 20.5 | 20.0 |
| the same, two in flight, §8's launch and queue | 6.7 | 20.0 | 20.6 |
| three reader processes, whole intervals in 3-hour windows, shared memory (§6) | 5.7 | 17.1 | **24.0** |

The run is bound by the readers: a steady interval is 11-14 ms, the GPU busy ~7 of them and
the main thread's CPU ~7, and the main thread waits ~7 ms an interval for the next reading.
Each reader takes 30-36 ms of CPU an interval with three or four running (17 ms alone): the
M4's ten cores are shared with the main process, the GPU driver and the machine's own load.
emu32 (the float32 host build, one CPU thread) takes 108 ms a simulated hour, the float64 CPU
reference 282.

## 10. AMD: an MI100 under ROCm

Alpine's `ami100` (AMD Instinct MI100, gfx908, 120 CUs, 64-lane wavefronts, 32 GB). CuPy 13.6
built from source for HIP against ROCm 6.1 (`runs/alpine/env_rocm.slurm`; PyPI's ROCm wheel
needs ROCm 5.0, which Alpine lacks). The `cuda` backend runs it unchanged but for three things:

| what | on HIP |
|---|---|
| no contraction | hiprtc's `-ffp-contract=off` for NVRTC's `--fmad=false` (`CudaBackend.hip`) |
| the 32-lane groups (`coop.py`'s lines, `convec_gen`'s columns) | half a 64-lane wavefront: pd.h's `pd_shfl*` at width 32, `pd_any` / `pd_all` a ballot masked to the half, `pd_group_sync` a wavefront fence and barrier; on CUDA the same names are the `*_sync` intrinsics and `__syncwarp` |
| graphs | HIP's `hipStreamBeginCapture` / `EndCapture` / `GraphInstantiate` / `GraphLaunch` by ctypes (`backend._HipGraph`): CuPy refuses stream capture on HIP |

hiprtc defines `__HIPCC_RTC__`, not `__CUDACC_RTC__`, so pd.h names both for its CUDA branch.
Verified (`runs/alpine/rocm_probe.py`, job 34721100; the suite on a held MI100, `hold.slurm`
with `--partition=ami100`):

| claim | test | result |
|---|---|---|
| hiprtc honours `-ffp-contract=off` | `a*b + c`, 2^20 doubles over 1e-17 .. 1e17 and subnormals, against NumPy | 0 differ (92 299 with `fast`) |
| double division and sqrt are correctly rounded | `a / b`, `sqrt(|a|)`, the same arguments | 0 differ |
| a 32-lane group in a 64-lane wavefront | width-32 broadcast, shuffle up, xor maximum; the half-masked ballot's any | each group's NumPy answer |
| the device is the reference | `tests/test_gpu.py`: kernels, whole steps, Check_CFL, the three free runs, `zp_fixture` per event, speculated and replayed intervals | 37 passed, 4 skipped (as CUDA's), bitwise |
| the suite can fail here | the same with `-ffp-contract=fast` | 29 fail, every operator |

`pydyn.gpu.bench`, glb3x2 + nam1x1, against §5's whole A100:

| what | MI100, wall a step | simulated years an hour | whole A100 |
|---|--:|--:|--:|
| step | 12.7 ms | 32.3 | 7.9 ms |
| an interval | 19.0 ms | 21.6 | 13.0 ms |
| the interval speculated | 14.6 ms | 28.1 | 10.5 ms |
| the interval replayed as a graph | 13.9 ms | 29.6 | 6.8 ms |
| a step replayed | 9.8 ms | 41.9 | 4.3 ms |

Replayed, it is bound by the GPU: 14.3 ms of dispatches an interval, of which `pd_cvf_26`
5.2 ms (the A100's 1.0: the warp-tuned kernel, its shared-memory pitch and 163 registers, on
a wavefront twice as wide), the box programs' 145 dispatches 3.2 ms (1.5), the line kernels
about as on the A100. Not tuned for AMD.
