#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=96g
#SBATCH -p hercules
#SBATCH -q batch
#SBATCH -A lpdm
#
# PYD-2's rank merge on any serial set and its multi-rank twins (tests/test_ranks.py), through
# the model's own MKL and libimf taken from the executable (as replay.slurm):
#
#   ROOT=<pydyn copy> X=<model .x> SERIAL=<dump dir> MPI=<dir>:<dir>:... \
#       sbatch --export=ALL -J <name> -o <log> runs/hpc/ranks.slurm
#
# `sbatch -c N` replays the merged sets in N processes.
set -u
unset PYTHONPATH
PY=${PY:-/work2/noaa/co2/miniconda3/envs/tm5_312/bin/python3}
MK=$(dirname "$(ldd "$X" | awk '/libmkl_core/{print $3}')")
export PYDYN_FFTW=$MK/libmkl_rt.so PYDYN_LAPACK=$MK/libmkl_rt.so
export PYDYN_LIBM=$(ldd "$X" | awk '/libimf/{print $3}')
export LD_LIBRARY_PATH=$(dirname "$PYDYN_LIBM"):$MK${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export MKL_THREADING_LAYER=SEQUENTIAL MKL_INTERFACE_LAYER=LP64 OMP_NUM_THREADS=1
export PYDYN_SERIAL_DUMP_DIR=$SERIAL PYDYN_MPI_DUMP_DIRS=$MPI
export PYDYN_WORKERS=${SLURM_CPUS_PER_TASK:-1}  # the merged sets replay in this many processes
cd "$ROOT"
echo "host $(hostname)  exe $X  serial $SERIAL  mpi $MPI"
echo "pydyn $(cat pydyn/*.py pydyn/zoom/*.py | sha1sum | cut -c1-12) at $ROOT"
"$PY" -m pytest -q -rs -p no:cacheprovider tests/test_ranks.py
echo "EXIT=$?"
