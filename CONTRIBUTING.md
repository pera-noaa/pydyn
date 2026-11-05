# Contributing to pydyn

pydyn exists to say what TM5's arrays should be, so a change is judged by one question: is the
answer still the Fortran's, bit for bit? Read `README.md`, then `docs/state.md` (what is
transcribed and under which flags) and `docs/sources.md` (every TM5 file it depends on).

## Running the tests

```bash
uv sync
uv run pytest -rs
```

Tests that need what is not on your machine skip and say why, so read the count, not the
colour: a green run without the Fortran harness, a tm5-zoom checkout or the dump sets says
nothing about bitwise. `PYDYN_SETS=DIR` with the Zenodo fixture (`runs/fetch_fixture.py DIR`)
adds the transport replays; `tests/common.py` lists every path a test looks for and its
`PYDYN_*` variable. Tier 1 and the constant mirrors read a tm5-zoom checkout, and the claims are
made at `fortran/TM5ZOOM_PIN`: `git -C ../tm5-zoom worktree add DIR $(cat fortran/TM5ZOOM_PIN)`,
then `make -C fortran TM5ZOOM=DIR` and `PYDYN_TM5ZOOM=DIR`. The pytest header says which commit
each of them is at.

The fixture's tests are bitwise on macOS arm64 with Accelerate and Homebrew's FFTW, the
libraries the sets were made with. On Linux, `test_production`'s meteo and convection (`v`)
checks **fail** at 1e-12 to 2e-10 relative on an unchanged tree, because the replay loads that
system's FFTW and LAPACK (`docs/state.md` §3c); the device free runs stay bitwise. Compare
against a run of the same tests before your change, not against zero failures.

A Dependabot pull request that bumps `uv.lock` is a NumPy (or netCDF4) change. CI runs no
Fortran or dump-set test, so run the full suite on the branch before merging, and read the test
count and the `numpy ... SIMD dispatched` line in the pytest header, not only the colour.

## Rules

- **Bitwise is a claim about flags.** Any statement that Python and Fortran agree carries the
  compiler, the optimisation level and `-ffp-contract`. Never write "matches" where you mean
  "agrees to 1e-12": say the tolerance and why it is not zero.
- **Evaluation order is the source code.** The Python mirrors the Fortran statement by
  statement, including asymmetries only visible bitwise. `np.sum` and `np.dot` are not
  sequential; gfortran's `sum` is. When a kernel needs an accumulation, write the loop
  (`pydyn/ops.py` says where `np.diff` is and is not safe).
- **A difference is an index error until proven an evaluation order.** `harness.compare`
  reports the ulp distance and the first differing index: 1 ulp is an ordering, a large one is
  a bug. Do not add a tolerance to make a test pass.
- **`pydyn/gpu/` is held to the reference as the reference is to the Fortran.** A change to a
  reference kernel changes its device twin in `pydyn/gpu/csrc/` in the same commit; the device
  tests on the `emu` backend fail otherwise, which is the intended reminder.
- **A constant copied from the Fortran gets a name and a row in `tests/test_constants.py`.** A
  new module-level number with neither a row nor a waiver fails the suite, in CI too.
- **TM5 is not edited from here.** The kernels are compiled in place from a tm5-zoom checkout
  and nothing is written back; cite the model by the sha you read it at.
- `ruff check .` is clean (`ruff.toml`); items in `TODO.md` are `PYD-N`.

By contributing you agree that your contribution is licensed under Apache-2.0 (`LICENSE`).
