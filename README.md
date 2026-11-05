# pydyn: a serial Python reference of the TM5 dynamic core

pydyn is a NumPy transcription of the transport core of TM5, the atmospheric
chemistry-transport model. It is **bitwise equal to the production Fortran under stated
conditions**: no FMA contraction, and the model's own FFTW, LAPACK and libm (below and
`docs/state.md` §3c). A port, an optimisation or a GPU kernel can be checked against it,
because it answers "what should this array be".

To check the claim yourself, `uv run pytest -rs` runs whatever your machine can and names
everything it skips. The Zenodo fixture adds replays of the model's own dumps, and a tm5-zoom
checkout at the pin adds the compiled Fortran (see "What runs where").

Apache-2.0 (`LICENSE`, `NOTICE`); cite it with `CITATION.cff`; read `CONTRIBUTING.md` before a
change, `SECURITY.md` for what it trusts and how to report a vulnerability, `CHANGELOG.md` for
releases.

## Status

<!-- ≤ 20 lines, what is true now, no dated entries. Rewrite; do not append. -->

- **Open:** a port-facing golden-case format (`PYD-7`), the isotope and `with_budgets` chemistry (`PYD-5`), a GPU run from the input files at production size (`PYD-16`), the public release (`PYD-17`). `docs/state.md` is the technical head.
- **Scope**: the production build variant (`slopes`, `without_wet_deposition`, no secmom, no OpenMP) for one region, a multi-rank run and a zoom tree. Everything the dynamic step does, the CH4 run's two physics operators (surface injection, chemistry), and the reading of every input from the model's own files.
- **Tier 1** (`fortran/`): the production kernels compiled unmodified against stubs, driven from Python. Bitwise at gfortran `-O0 -ffp-contract=off` and at ifort with Orion's production flags. Convection is `base/trunk`'s `convection.F90`; production's `convection__speedup` is covered by Tier 2.
- **Tier 2** (`pydyn/replay.py`): real model runs replayed from the `pydyn_dump` events they wrote, per event and as a free run in which the Python carries the state and reads only the model's input files. Bitwise for a glb6×4 day, its 2-, 3- and 5-rank MPI runs, and the production glb3×2 + nam1×1 zoom with emission and chemistry (ifort, Orion's flags).
- **The flags are part of the claim**: no contraction (`-ffp-contract=off`, ifort `-fp-model precise -no-fma`), and the FFT, LAPACK and libm calls through the model's own libraries. Hercules' bare `-O3` is a few ulp off (`docs/state.md` §3c).
- **What it has found.** `Check_CFL`'s z sub-cycle cap is what shortens production's time step; raising it is stable and worth 19 % of TM5's root timer, its whole run (`docs/cfl_headroom.md`). An MPI zoom child decayed its CH4 with another cell's loss rates, fixed in tm5-zoom `0385efe7e` (`docs/state.md` §3f).
- **`pydyn.gpu`**: the same core with the state resident on a GPU, from one C source for CUDA, HIP, Metal (float32) and the host (`emu`). It is float64-bitwise to the reference on `emu`, an A100 and an MI100. A production-sized zoom interval runs at 60.1 simulated years an hour on an A100, 29.6 on the MI100 and 16.9 on an Apple M4 (Metal, synthetic meteo: bitwise to its float32 host twin) (`docs/gpu_port.md`).
- Model pin: tm5-zoom `NOAA/ct-unified` at `745e94f70` (`fortran/TM5ZOOM_PIN`, `docs/sources.md`). The dump module and its hooks are model code and live there, off unless `pydyn.dump : T`.
- **The dump sets are scratch, not in git**: glb6×4 sets on the maintainer's laptop (regenerable by `runs/setup_repin.sh`) and glb3×2 and zoom sets on Orion and Hercules. Three are published (below). `runs/MANIFEST.md` has a row for each set and each job.

## What runs where

`uv run pytest --collect-only -q` counts the tests; a test that cannot run here skips and says
why, so read the count, not the colour. `tests/common.py` names every path a test looks for
and its `PYDYN_*` variable.

| Needs | Tests | Bitwise against |
|---|---|---|
| NumPy, netCDF4 and a C++ compiler (CI: Python 3.10, 3.12, 3.14 and the oldest admitted versions) | property, unit, simple-vs-fast, synthetic zoom, the README example, `pydyn.gpu` on `emu` and its CUDA host side, the constants accounting, fixture-fetch and export plumbing | the NumPy reference (`emu`); simple-vs-fast to 1e-12 of each field's largest magnitude |
| Apple silicon | Metal (`metal` against `emu32`) | its float32 host twin |
| a tm5-zoom checkout at the pin, `make -C fortran` and `make -C fortran zoom` | Tier 1 parity, zoom parity, the constant mirrors | the compiled production kernels and declarations |
| the Zenodo fixture ([10.5281/zenodo.23955885](https://doi.org/10.5281/zenodo.23955885); `python3 runs/fetch_fixture.py DIR`, then `PYDYN_SETS=DIR`; a draft until the v0.1.0 release) | 8: the glb6×4 transport replay per event and as a free run, a free run through a rejected time step, both on the device (`emu`), the zoom child's meteo | the model's own dumps, on macOS arm64 with Accelerate and Homebrew's FFTW (an Apple M4, macOS 15.8.1). On Linux the device free runs stayed bitwise; the CPU replays' meteo and convection were 1e-12 to 2e-10 off (`docs/state.md` §3c) |
| the unpublished sets and the model's input files | the physics replays, input-file free runs, rank merges, zoom physics | the model's own dumps |

Each fixture set has a `config/` folder with the rc the model read, the build fingerprint, the
logs and the recipe. Without the model's FFTW and LAPACK the bitwise replays skip, and
`test_replay_on_numpy_fallbacks` measures the NumPy fallbacks against a stated tolerance
instead.

## Install and run

```bash
git clone https://github.com/pera-noaa/pydyn && cd pydyn
uv sync             # Python 3.12 (.python-version) with uv.lock's NumPy, netCDF4, pytest, ruff
uv run pytest -rs   # what cannot run here skips and says why
```

As a dependency, install from the repository: `uv pip install "git+https://github.com/pera-noaa/pydyn@v0.1.0"`.
The distribution is `tm5-pydyn` and the import is `pydyn`; it is not on PyPI, and the `pydyn`
there is an unrelated project. The public surface is `pydyn` itself, `kernels`, `simple`,
`synth`, `dump`, `replay` and `pydyn.gpu`; each module's `__all__` lists its names.

Tier 1 and the constant mirrors need gfortran and a tm5-zoom checkout at the pin (not public
yet). `make -C fortran` refuses any other commit unless `UNPINNED=1`:

```bash
git -C ../tm5-zoom worktree add ../tm5-zoom-pin $(cat fortran/TM5ZOOM_PIN)
make -C fortran TM5ZOOM=../../tm5-zoom-pin && make -C fortran zoom TM5ZOOM=../../tm5-zoom-pin
PYDYN_TM5ZOOM=../tm5-zoom-pin uv run pytest -rs
```

`python3 -m pydyn.replay DIR` (or `pydyn-replay DIR`) replays a `pydyn_dump` directory and
exits non-zero if any event differs or raises.

`uv sync --extra cuda` adds CuPy for `pydyn.gpu`. Without uv: `pip install -e . --group dev`
(pip >= 25.1). What pydyn compiles at run time (a C loop for libm, the emulation kernels, the
Metal shim) is cached in `${XDG_CACHE_HOME:-~/.cache}/pydyn`, a directory it uses only while it
is yours alone; it is never pruned, and deleting it is always safe.

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
This README is the front page; `docs/state.md` is the technical head.

## Layout

| Dir | Holds |
|---|---|
| `pydyn/` | the package: kernels, driver, the production steps around them, zoom nesting, the dump reader and replay; `pydyn/gpu/` the device port (`csrc/` its kernels) |
| `tests/` | property, parity (single region and zoom), simple-vs-fast, production-replay and constant tests |
| `fortran/` | stubs, the two drivers and the Makefile that compile tm5-zoom's kernels (§3a) and its zoom routines (§3d) unmodified |
| `docs/` | `state.md` (head; `state.history.md` its verbatim archive), `sources.md` (register of what is inherited), `cfl_headroom.md`, `gpu_port.md` and any later findings note |
| `runs/` | `MANIFEST.md`: one row per dump set, pinned to a model sha, build flags and rc; `setup_repin.sh` (+ `make_scaling_one.py`) regenerates the three laptop sets at a given sha; `pack_fixture.py`, `fetch_fixture.py`, `zenodo_upload.py` the Zenodo fixture; `export_public.sh` + `leak_scan.py` the export to this public repo. **Maintainer job scripts**, which name the maintainer's accounts and paths and are kept as the record of how results were made: `hpc/` for NOAA's Orion and Hercules clusters (`replay.slurm`, `ranks.slurm`, `regress_zoom.sh` + `check_zoom.sh`, the zoom battery) and `alpine/` for CU Boulder's Alpine (the GPU tests) |
| `analysis/` | scripts that use the reference to answer a question rather than test it, run as `python3 -m analysis.<name>` from the repo root: `cfl_rejection_anatomy.py` (which operator refused one step, in detail), `cfl_cache_census.py` (how often production is reduced at all), `cfl_cap_replay.py` (over a whole dump set, how many reductions a different cap recovers), `step_accuracy.py` (whether the longer step conserves, stays positive and stays accurate), `obs_departure.py` (whether two forward runs differ by more than an inversion resolves at its obs), `obs_residual.py` (which of two forward runs better follows the obs' variability), `cfl_cap_replay_zoom.py` (`Check_CFL` with the child, over the census sets), `ch4_loss_bounds.py` (which loss rates a zoom child's chemistry applied) |
| `results/` | small tracked csv the `analysis/` scripts wrote (`results/README.md`: which script, which section); dumps and large arrays are ignored |
| `TODO.md` | project items, prefixed `PYD-N` (never bare `#N`); `TODO.history.md` the full text landed items had |
