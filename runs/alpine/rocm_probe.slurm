#!/bin/bash
#SBATCH --job-name=pydyn-rocm
#SBATCH --partition=ami100
#SBATCH --qos=gpu-testing
#SBATCH --account=ucb-general
#SBATCH --gres=gpu:mi100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/projects/%u/pydyn/runs/alpine/rocm_%j.log
# What CuPy on an MI100 gives pydyn (rocm_probe.py): hiprtc, contraction off, double div / sqrt against numpy, 32-lane
# groups in a 64-lane wavefront, pydyn's source compiled as is. After env_rocm.slurm.  sbatch runs/alpine/rocm_probe.slurm
set -uo pipefail
R=/curc/sw/install/rocm/6.1.0  # what CuPy was built against (the rocm modules are not visible to batch jobs)
export ROCM_HOME=$R ROCM_PATH=$R HIP_PATH=$R
export LD_LIBRARY_PATH=$R/lib:$R/lib64:$R/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
eval "$(/curc/sw/install/miniforge3/24.11.3-0/bin/conda shell.bash hook)"
conda activate /projects/$USER/software/anaconda/envs/pydyn-rocm
cd /projects/$USER/pydyn
python runs/alpine/rocm_probe.py
echo "exit $?"
