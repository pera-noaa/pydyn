#!/bin/bash
#SBATCH --job-name=pydyn-a100
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --account=ucb-general
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=/projects/%u/pydyn/runs/alpine/full_%j.log
# A whole A100 (not a MIG slice): test_gpu.py on CUDA and pydyn.gpu.bench's modes, the dumps staged
# onto the node's SSD as hold.slurm does.  sbatch runs/alpine/full.slurm (--gres=gpu:a100-40gb:1 for
# the 40 GB model); the code as last synced (arun, or rsync).
set -uo pipefail
W=/scratch/local/pydyn-$SLURM_JOB_ID L=$W/stage
trap 'rm -rf $W' EXIT; trap 'exit 143' TERM
mkdir -p $L && cd $L
mkdir -p repin_cfl_745e94f70 zp_fixture repin_free_745e94f70 repin_phys_745e94f70 tm5-nc
ls /projects/$USER/pydyn/stage/*.tar | xargs -P 8 -n 1 tar xf
E=/projects/$USER/software/anaconda/envs/pydyn
eval "$(/curc/sw/install/miniforge3/24.11.3-0/bin/conda shell.bash hook)"
conda activate $E
export PYDYN_DEVICE=cuda PYDYN_WORKERS=8 CXX=g++ OPENBLAS_NUM_THREADS=1
export PYDYN_FFTW=$E/lib/libfftw3.so.3 PYDYN_LAPACK=$E/lib/liblapack.so
export PYDYN_DUMP_DIR=$L/repin_phys_745e94f70 PYDYN_FREE_DUMP_DIR=$L/repin_free_745e94f70
export PYDYN_CFL_DUMP_DIR=$L/repin_cfl_745e94f70 PYDYN_ZOOM_PHYS_DIR=$L/zp_fixture PYDYN_METEO_ROOT=$L/tm5-nc
cd /projects/$USER/pydyn
echo "pydyn $(cat COMMIT) on $(hostname), $(date)"
nvidia-smi -L
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv,noheader
python -m pytest -q -rs -p no:cacheprovider tests/test_gpu.py
echo "pytest exit $?"
for m in "" "--interval" "--interval --speculate" "--interval --graph" "--graph"; do
  echo "== bench $m"
  python -m pydyn.gpu.bench --device cuda --steps 40 $m | grep -E "wall|years|replayed"
done
echo "== bench --interval --speculate --profile"
python -m pydyn.gpu.bench --device cuda --steps 10 --interval --speculate --profile | sed -n '/GPU time/,$p'
