# pydyn: a serial Python reference of the TM5 dynamic core

A NumPy transcription of the transport core of TM5, the
atmospheric chemistry-transport model, that is **bitwise equal to the production Fortran**,
plus the tiers of tests that prove it. A port, an optimisation or a GPU kernel can be checked
against it: the reference answers "what should this array be". Apache-2.0 (`LICENSE`,
`NOTICE`); cite it with `CITATION.cff`; `CONTRIBUTING.md` before a change.

## Status

<!-- ≤ 20 lines, what is true now, no dated entries. Rewrite; do not append. -->

- **Open:** a port-facing golden-case format (`PYD-7`: proposed, four questions for its consumer); the isotope and `with_budgets` chemistry (`PYD-5`); a GPU run from the input files at production size (`PYD-16`). `docs/state.md` is the technical head.
- **Scope**: the production build variant (`slopes`, `without_wet_deposition`, no secmom, no OpenMP) for one region, a multi-rank run, and a zoom tree. Everything the dynamic step does, the CH4 run's two physics operators (surface injection, chemistry), and the reading of every input from the model's own files: meteo, `Calc_Kzz`, emission, the chemistry's fields, the initial tracer field.
- **Tier 1** (`fortran/`): the production kernels compiled unmodified against stubs, driven from Python; bitwise at gfortran `-O0 -ffp-contract=off` and at ifort with Orion's production flags.
- **Tier 2** (`pydyn/replay.py`, one replay for a single region or a zoom tree): real model runs replayed from the rc-gated `pydyn_dump` events they wrote, per event and in a free run in which the Python carries the state itself and reads nothing but the model's input files. Bitwise for a glb6×4 day (gfortran, and ifort with the model's MKL and libimf), for its 2-, 3- and 5-rank MPI runs merged (gfortran, and ifort beside IO-server ranks), and for the production glb3×2 + nam1×1 zoom with emission and chemistry, serial, MPI and four tracers merged over ranks (ifort, Orion's flags).
- **A bitwise claim is a claim about flags**: no contraction (`-ffp-contract=off`, ifort `-fp-model precise -no-fma`), and the FFT, LAPACK and libm calls through the model's own libraries. Hercules' bare `-O3` is a few ulp off (`docs/state.md` §3c).
- **What it has found.** `Check_CFL`'s z sub-cycle cap is what shortens production's time step; raising it is stable, worth 19 % of TM5 root, and the CT-CH4 EnKF would see it, as transport, without fitting the observations better (`docs/cfl_headroom.md`, `PYD-11`). And an MPI zoom child decayed its CH4 with the loss rates of a cell 3° east and 2° north, fixed in tm5-zoom `0385efe7e` (`docs/state.md` §3f).
- Model pin: tm5-zoom `NOAA/ct-unified` at `745e94f70` (`docs/sources.md`). The dump module and its hooks are model code and live there, off unless `pydyn.dump : T`. `make -C fortran` reads that checkout in place (`TM5ZOOM=<path>`, default `../../tm5-zoom`) and writes nothing back.
- `uv run pytest`: 537 tests, ~8 min with `PYDYN_WORKERS=4`: 41 device (`pydyn.gpu` on `emu`; 4 need the dump sets, 3 compile the CUDA source with an NVPTX clang), 42 Metal (`metal` against `emu32`; Apple silicon only), 9 forward (`pydyn.forward` from the input files: against the dumps, emu, reader processes, a staged run, Metal with either inverse, Metal's pipelined run through a failed speculation), 6 CUDA host side (`CudaBackend` against a stand-in cupy), 2 shared-memory hand-off, 19 property, 30 parity, 159 zoom (158 parity), 14 zoom meteo, 5 zoom physics, 9 zoom-census plumbing, 24 simple-vs-fast, 41 unit (one file per module), 5 fixture-fetch, 1 README example, 10 production replays, 12 rank-merge, 9 input-file, 99 constant mirrors. The 10 and the 12 need the dump sets, 4 of the 9 the model's input files too, the 99 the tm5-zoom checkout, 5 of the zoom ones the two zoom fixtures, the parity ones `make -C fortran` / `make -C fortran zoom`; they skip when absent, so read the count, not the colour. CI (`.github/workflows/tests.yml`: Python 3.10, 3.12 and 3.14 at `uv.lock`'s versions, and 3.10 at the oldest `pyproject.toml` admits) runs `ruff` and the 162 that need none of it (24 of them device tests, which need a C++ compiler; 6 the CUDA host side). `tests/common.py` names every path a test looks for and its `PYDYN_*` variable.
- **`pydyn.gpu`: the same core with every region's state resident on a GPU** (TM5 is bandwidth-bound, so the arrays live on the device and only comparisons come down). Every operator is ported -- x/y/z of region 1 and of zoom children with their sub-cycle searches, convection, Check_CFL, Setup_MassFlow, the zoom exchange, emission, the CH4 loss rates, Calc_Kzz -- from one C source compiled for CUDA (`--fmad=false`; AMD through CuPy for HIP, `-ffp-contract=off`), for Metal in float32 (`xcrun metal -ffp-contract=off`) and for the host (`-ffp-contract=off`, the `emu` and `emu32` backends). **float64 bitwise to the reference** on `emu`, on an NVIDIA A100 and on an AMD MI100 (`tests/test_gpu.py` with the free runs); the closed libraries have open twins both sides load (`--lapack twin --libm twin`). A production-sized glb3x2 + nam1x1 interval replayed as a CUDA graph runs at 60.1 simulated years an hour on a whole A100 (41.2 on a 3g.20gb slice) and 29.6 on the MI100. On an Apple M4 (Metal, float32: bitwise to its host twin `emu32`, not to the reference) the device carries the same interval at 16.9 (`python3 -m pydyn.gpu.bench`; synthetic meteo), and `python3 -m pydyn.forward` runs from the input files alone at 24.0 (glb6x4, three reader processes, `--stage` for a slow file system; bitwise the model's dumps on the CPU, 14 days bitwise `emu32` on Metal). `docs/gpu_port.md`.
- **Outside a TM5 installation** the package, the property, unit, simple-vs-fast and synthetic zoom tests and `pydyn.gpu` on `emu` need only NumPy, netCDF4 and a C++ compiler. Tier 1 and the constant mirrors need a tm5-zoom checkout, which is not distributed. Three of the dump sets are, as a 2.9 GB Zenodo fixture (`python3 runs/fetch_fixture.py DIR`, then `PYDYN_SETS=DIR`): they add 8 bitwise tests against the model's own dumps -- the glb6×4 transport replay per event and as a free run, a free run through a rejected time step, both on the device (`emu`), and the zoom child's meteo. The rest of Tier 2 needs the model's input files.
- **The dump sets are scratch, not in git**: 8.3 GB of glb6×4 sets on the maintainer's laptop (the tests' defaults, regenerable by `runs/setup_repin.sh`) and the glb3×2 and zoom sets under `/work2/noaa/co2/ash/pydyn/dumps/` on Orion and Hercules. `runs/MANIFEST.md` has a row for each, and for every job that touched those machines.

## Install and run

```bash
uv sync             # Python 3.12 (.python-version) with uv.lock's NumPy, netCDF4, pytest, ruff
uv run pytest -rs   # what cannot run here skips and says why
make -C fortran     # Tier 1: gfortran and a tm5-zoom checkout (TM5ZOOM=<path>)
```

`uv sync --extra cuda` adds CuPy for `pydyn.gpu`. Without uv: `pip install -e . --group dev`
(pip >= 25.1).

```python
import numpy as np
from pydyn import kernels, simple, synth

rng = np.random.default_rng(0)
s = synth.state(rng, 8, 6, 5, 2)                  # im, jm, lm, tracers: m, rm, rxm, rym, rzm
am, bm, cm = synth.fluxes(rng, s.m, cfl=0.4)      # mass fluxes through the x, y and z faces
ref, got = s.copy(), s.copy()
kernels.advect_x(ref.m, ref.rm, ref.rxm, ref.rym, ref.rzm, am)  # bitwise with TM5's advectx
simple.advect_x(got.m, got.rm, got.rxm, got.rym, got.rzm, am)   # the same scheme, written to be read
```

## Inherits from tm5-zoom

- code: the kernels themselves are read in place, never copied — `base/trunk/src/{advect,convection}.F90`, `proj/ct-unified/trunk/src/{advectx__slopes,advecty__slopes,advectm_cfl,redgridZoom}.F90`, `branches/speedup/src/advectz__slopes.F90`. `fortran/stubs.F90` stands in for everything they `use`.
- docs: `tm5-zoom@4dc2017c5:docs/dynamic_core_survey.md` (the algorithm, with its own §-web), and `kernel_catalog.md`, `tm5mp_vs_tm5zoom.md` §5, `draft_knmi_tm5_slopelimiter_report.md`, `adjoint_phase3_transport.md` as context. Registered in `docs/sources.md`, not copied.
- model code this repo depends on but does not own: `pydyn_dump.F90` and its hook sites. Changing them is a tm5-zoom commit; this repo then re-pins.
- history: `tools/pydyn` in tm5-zoom up to `4dc2017c5`, where this started. `git log` there for anything before 2026-09-22.

tm5-zoom is not public yet (`docs/sources.md`). Existing tm5-zoom docs are not moved. Cite them as `tm5-zoom@<sha>:docs/<file>.md §N`.
This README is the head.

## Layout

| Dir | Holds |
|---|---|
| `pydyn/` | the package: kernels, driver, the production steps around them, zoom nesting, the dump reader and replay; `pydyn/gpu/` the device port (`csrc/` its kernels) |
| `tests/` | property, parity (single region and zoom), simple-vs-fast, production-replay and constant tests |
| `fortran/` | stubs, the two drivers and the Makefile that compile tm5-zoom's kernels (§3a) and its zoom routines (§3d) unmodified |
| `docs/` | `state.md` (head; `state.history.md` its verbatim archive), `sources.md` (register of what is inherited), `cfl_headroom.md`, `gpu_port.md` and any later findings note |
| `runs/` | `MANIFEST.md`: one row per dump set, pinned to a model sha, build flags and rc; `setup_repin.sh` (+ `make_scaling_one.py`) regenerates the three laptop sets at a given sha; `hpc/` the Orion/Hercules jobs: `replay.slurm` (`-c N` replays per event in N processes), `ranks.slurm`, `regress_zoom.sh` + `check_zoom.sh` (the zoom battery) |
| `analysis/` | scripts that use the reference to answer a question rather than test it, run as `python3 -m analysis.<name>` from the repo root: `cfl_rejection_anatomy.py` (which operator refused one step, in detail), `cfl_cache_census.py` (how often production is reduced at all), `cfl_cap_replay.py` (over a whole dump set, how many reductions a different cap recovers), `step_accuracy.py` (whether the longer step conserves, stays positive and stays accurate), `obs_departure.py` (whether two forward runs differ by more than an inversion resolves at its obs), `obs_residual.py` (which of two forward runs better follows the obs' variability), `cfl_cap_replay_zoom.py` (`Check_CFL` with the child, over the census sets), `ch4_loss_bounds.py` (which loss rates a zoom child's chemistry applied) |
| `results/` | small tracked csv and figures; dumps and large arrays are ignored |
| `TODO.md` | project items, prefixed `PYD-N` (never bare `#N`); `TODO.history.md` the full text landed items had |
