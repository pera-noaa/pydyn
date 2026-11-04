#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=64g
#SBATCH -p hercules
#SBATCH -q batch
#SBATCH -A lpdm
#
# pydyn.gpu.check on one dump set: the reference and the device (DEVICE=emu here, cuda on a
# GPU node) with the twin libraries, every checkpoint's fingerprints matched.  No model
# library is involved, so none is loaded.  Submit from anywhere:
#
#   ROOT=<pydyn copy> D=<dump dir> ARGS="--free-run ..." sbatch --export=ALL -J <name> -o <log> \
#       runs/hpc/gpu_check.slurm
set -u
unset PYTHONPATH
PY=${PY:-/work2/noaa/co2/miniconda3/envs/tm5_312/bin/python3}
export OMP_NUM_THREADS=1
cd "$ROOT"
echo "host $(hostname)  dump $D  device ${DEVICE:-emu}  cxx $(${CXX:-c++} --version | head -1)"
echo "pydyn $(cat pydyn/*.py pydyn/zoom/*.py pydyn/gpu/*.py pydyn/gpu/csrc/* | sha1sum | cut -c1-12) at $ROOT"
"$PY" -m pydyn.gpu.check "$D" --device "${DEVICE:-emu}" ${ARGS:-}
echo "EXIT=$?"
