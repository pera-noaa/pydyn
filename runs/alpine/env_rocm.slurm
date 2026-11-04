#!/bin/bash
#SBATCH --job-name=pydyn-env-rocm
#SBATCH --partition=acompile
#SBATCH --qos=compile
#SBATCH --account=ucb-general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=22G
#SBATCH --time=03:00:00
#SBATCH --output=/projects/%u/pydyn/runs/alpine/env_rocm_%j.log
# The conda env for pydyn on Alpine's MI100s: env.slurm's, with CuPy built from source for HIP against
# ROCm 6.1 (gfx908). PyPI's one ROCm wheel (cupy-rocm-5-0) needs ROCm 5.0's C++-mangled symbols, which
# none of Alpine's ROCm 5.2-5.6 exports.  sbatch runs/alpine/env_rocm.slurm
set -euo pipefail
CONDA=/curc/sw/install/miniforge3/24.11.3-0
eval "$("$CONDA/bin/conda" shell.bash hook)"
ENV=/projects/$USER/software/anaconda/envs/pydyn-rocm
[ -x "$ENV/bin/python" ] || conda create -y -q -p "$ENV" -c conda-forge python=3.12 numpy=1.26 netcdf4 pytest fftw
"$ENV/bin/python" -m pip uninstall -y -q cupy-rocm-5-0 2>/dev/null || true
"$ENV/bin/python" -m pip install -q "cython>=3,<3.1" fastrlock
R=/curc/sw/install/rocm/6.1.0
export ROCM_HOME=$R ROCM_PATH=$R HIP_PATH=$R HCC_AMDGPU_TARGET=gfx908 CUPY_INSTALL_USE_HIP=1 CUPY_NUM_BUILD_JOBS=6
export PATH=$R/bin:$R/llvm/bin:$PATH LD_LIBRARY_PATH=$R/lib:$R/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
t0=$SECONDS
"$ENV/bin/python" -m pip install -v --no-build-isolation --no-binary cupy "cupy==13.6.0" > "$ENV.build.log" 2>&1 \
  || { tail -60 "$ENV.build.log"; exit 1; }
echo "built in $(( SECONDS - t0 )) s"
"$ENV/bin/python" -c 'import numpy, netCDF4, pytest; print("numpy", numpy.__version__, "netCDF4", netCDF4.__version__)'
"$ENV/bin/python" -m pip show cupy | sed -n 1,3p
