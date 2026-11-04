#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=48g
#SBATCH -p hercules
#SBATCH -q batch
#SBATCH -A lpdm
#
# One replay of a dump set on Orion or Hercules, through the model's own libraries: MKL (FFTW
# interface + DGETRF/DGETRI) and libimf, both taken from the model executable with ldd, as
# docs/state.md section 3c requires.  Submit from anywhere:
#
#   ROOT=<pydyn copy> X=<model .x> D=<dump dir> MOD=pydyn.replay ARGS="--free-run ..." \
#       sbatch --export=ALL -J <name> -o <log> [-p orion] runs/hpc/replay.slurm
#
# MOD defaults to pydyn.replay; the log's first lines name the host, the libraries and a
# hash of the package, so a result can be tied to the code that produced it.  `sbatch -c N`
# replays per event in N processes (--workers; a free run stays in one).
set -u
unset PYTHONPATH
PY=${PY:-/work2/noaa/co2/miniconda3/envs/tm5_312/bin/python3}
MK=$(dirname "$(ldd "$X" | awk '/libmkl_core/{print $3}')")
export PYDYN_FFTW=$MK/libmkl_rt.so PYDYN_LAPACK=$MK/libmkl_rt.so
export PYDYN_LIBM=$(ldd "$X" | awk '/libimf/{print $3}')
export LD_LIBRARY_PATH=$(dirname "$PYDYN_LIBM"):$MK${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export MKL_THREADING_LAYER=SEQUENTIAL MKL_INTERFACE_LAYER=LP64 OMP_NUM_THREADS=1
cd "$ROOT"
echo "host $(hostname)  exe $X  dump $D"
echo "pydyn $(cat pydyn/*.py pydyn/zoom/*.py | sha1sum | cut -c1-12) at $ROOT"
"$PY" -m "${MOD:-pydyn.replay}" "$D" --rank 0 --lapack lapack --fft fftw --libm "$PYDYN_LIBM" \
    --workers "${SLURM_CPUS_PER_TASK:-1}" ${ARGS:-}
echo "EXIT=$?"
