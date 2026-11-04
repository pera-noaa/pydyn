# Alpine (CU Boulder Research Computing): the device tests on an A100

pydyn's suite with `PYDYN_DEVICE=cuda` on one `a100_3g.20gb` MIG slice (`aa100`, QOS
`gpu-testing`: 1 h, one job per user), or on a whole A100 (QOS `gpu-normal`, a longer queue).
Paths are under `/projects/$USER/pydyn` (the code) and `/scratch/alpine/$USER/pydyn` (the dump
sets and meteo copied from the laptop). Results: `docs/gpu_port.md` §5 (CUDA), §10 (the MI100).

| script | what | where |
|---|---|---|
| `env.slurm` | the conda env (numpy 1.26, netCDF4, FFTW, CuPy 13 / CUDA 12) — once | `acompile` |
| `pack.slurm` | the dump sets and meteo as tar chunks in `/projects/$USER/pydyn/stage` — after a copy | `acompile` |
| `hold.slurm` | holds a slice, the tars unpacked onto its SSD, `env.sh` written; an MI100 with `--partition=ami100 --gres=gpu:mi100:1` (the ROCm env) | `aa100`, `ami100` |
| `arun [pytest args]` | from the laptop: rsync the tree, run pytest on the held slice | laptop |
| `tests.slurm` | the suite as one batch job, dumps read in place (slow) | `aa100` |
| `full.slurm` | a whole A100 80GB: `test_gpu.py` and `pydyn.gpu.bench`'s modes, the tars staged as `hold.slurm` | `aa100` |
| `env_rocm.slurm` | the env for the MI100s: CuPy 13.6 built from source for HIP, ROCm 6.1, gfx908 (~5 min) | `acompile` |
| `rocm_probe.slurm` | `rocm_probe.py` on an MI100: hiprtc, contraction, div / sqrt against numpy, 32-lane groups, pydyn's source compiled | `ami100` |

The loop: `sbatch runs/alpine/hold.slurm` on Alpine, then `runs/alpine/arun` here as often as
needed for its hour; `tests/test_gpu.py` takes about 2 minutes.

- **`/scratch/alpine` reads a small file in ~80 ms** from a compute node, a tar from `/projects`
  at ~350 MB/s; the replays read thousands of files, so in place a run takes 30+ minutes.
- **`$SLURM_SCRATCH` is emptied at the end of every job step**, so `hold.slurm` stages into its
  own `/scratch/local/pydyn-$SLURM_JOB_ID` and removes it on exit or cancel.
- GPU partitions refuse a job without `--nodes`; the `miniforge` and `rocm` modules are not
  visible to batch jobs, so the scripts call `conda` and ROCm (`/curc/sw/install/rocm/6.1.0`) by
  path.
- `acompile` allows 8 CPUs a user and gives a job a CPU per ~3.75 GB it asks for: 32 GB is 9
  CPUs and never starts.
- PyPI's one ROCm wheel (`cupy-rocm-5-0`) needs ROCm 5.0's symbols, which none of Alpine's ROCm
  5.2-5.6 exports; hence the source build.
- `test_production`'s meteo and `v` checks are bitwise to the Mac's FFTW and LAPACK, which made
  the dumps; on Alpine they differ by 1e-12 to 2e-10 relative, so run them on the Mac.
