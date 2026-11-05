# Changelog

One entry per release. Commit history is the detail; this is what a user needs to know.

## 0.1.0 (unreleased)

First public release.

- A serial NumPy transcription of TM5's dynamic core (tm5-zoom `745e94f70`, `NOAA/ct-unified`):
  the slopes advection, convection, the reduced grid, `Check_CFL`, `Meteo_Setup_Mass` and
  `m2phlb1`, the zoom exchange, the CH4 run's emission and chemistry, and the readers of every
  input file. Bitwise to the production Fortran under the conditions in `docs/state.md` §3c.
- `pydyn.replay` / `pydyn-replay`: replay a model run's `pydyn_dump` events per event or as a
  free run, and exit non-zero on any difference or failure.
- `pydyn.gpu`: the same core resident on a GPU (CUDA, HIP, Metal in float32, a host emulation).
- The Zenodo fixture [10.5281/zenodo.23955885](https://doi.org/10.5281/zenodo.23955885): three
  dump sets with the configuration that made them (`runs/fetch_fixture.py`).
