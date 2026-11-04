#!/bin/bash
#SBATCH --job-name=pydyn-env
#SBATCH --partition=acompile
#SBATCH --qos=compile
#SBATCH --account=ucb-general
#SBATCH --ntasks=4
#SBATCH --time=01:00:00
#SBATCH --output=/projects/%u/pydyn/runs/alpine/env_%j.log
# The conda env for pydyn's tests on Alpine: numpy, netCDF4, pytest, FFTW, CuPy with CUDA 12 (NVRTC from
# conda-forge).  Once: sbatch runs/alpine/env.slurm
set -euo pipefail
CONDA=/curc/sw/install/miniforge3/24.11.3-0  # Alpine's miniforge (the module is not visible to batch jobs)
eval "$("$CONDA/bin/conda" shell.bash hook)"
ENV=/projects/$USER/software/anaconda/envs/pydyn
conda create -y -p "$ENV" -c conda-forge python=3.12 numpy=1.26 netcdf4 pytest fftw cupy "cuda-version=12.4"
"$ENV/bin/python" -c 'import numpy, netCDF4, cupy, pytest; print("numpy", numpy.__version__, "cupy", cupy.__version__, "netCDF4", netCDF4.__version__)'
