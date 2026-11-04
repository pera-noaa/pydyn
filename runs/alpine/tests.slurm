#!/bin/bash
#SBATCH --job-name=pydyn-cuda-tests
#SBATCH --partition=aa100
#SBATCH --qos=gpu-testing
#SBATCH --account=ucb-general
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:59:00
#SBATCH --output=/projects/%u/pydyn/runs/alpine/tests_%j.log
# pydyn's suite with the device tests on CUDA (PYDYN_DEVICE=cuda), one A100 MIG slice, as a batch job
# reading the dumps from /scratch/alpine (slow: ~80 ms a file; hold.slurm + arun stage them instead).
#   TESTS="tests/test_gpu.py -v" PYDYN_DUMP_DIR=... sbatch --export=ALL runs/alpine/tests.slurm
set -uo pipefail
CONDA=/curc/sw/install/miniforge3/24.11.3-0  # Alpine's miniforge (the module is not visible to batch jobs)
eval "$("$CONDA/bin/conda" shell.bash hook)"
conda activate /projects/$USER/software/anaconda/envs/pydyn
cd /projects/$USER/pydyn
echo "pydyn $(cat COMMIT) on $(hostname), $(date)"
nvidia-smi -L
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -c 'import cupy; d = cupy.cuda.Device(0); print("cupy", cupy.__version__, "runtime", cupy.cuda.runtime.runtimeGetVersion(), "driver", cupy.cuda.runtime.driverGetVersion(), "sm", d.compute_capability)'
export PYDYN_DEVICE=cuda PYDYN_WORKERS=8 CXX=g++
D=/scratch/alpine/$USER/pydyn/dumps  # copied from the laptop's .tm5-cloud-scratch/pydyn_build/dumps
export PYDYN_DUMP_DIR=${PYDYN_DUMP_DIR:-$D/repin_phys_745e94f70} \
       PYDYN_FREE_DUMP_DIR=${PYDYN_FREE_DUMP_DIR:-$D/repin_free_745e94f70} \
       PYDYN_CFL_DUMP_DIR=${PYDYN_CFL_DUMP_DIR:-$D/repin_cfl_745e94f70} \
       PYDYN_ZOOM_PHYS_DIR=${PYDYN_ZOOM_PHYS_DIR:-$D/zp_fixture} \
       PYDYN_METEO_ROOT=${PYDYN_METEO_ROOT:-/scratch/alpine/$USER/pydyn/METEO/tm5-nc}
python -m pytest -q -rs -p no:cacheprovider ${TESTS:-tests}
echo "pytest exit $?"
