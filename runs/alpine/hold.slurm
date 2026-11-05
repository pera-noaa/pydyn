#!/bin/bash
#SBATCH --job-name=pydyn-hold
#SBATCH --partition=aa100
#SBATCH --qos=gpu-testing
#SBATCH --account=ucb-general
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:59:00
#SBATCH --output=/projects/%u/pydyn/runs/alpine/hold_%j.log
# One A100 slice held for iterations (`srun --jobid=<this> --overlap`, see arun): the dumps and meteo
# unpacked from pack.slurm's tars onto the node's SSD, the env in $W/env.sh ($W, not $SLURM_SCRATCH:
# Slurm empties that at the end of every step, the first srun's included).  sbatch runs/alpine/hold.slurm;
# an MI100 instead: sbatch --partition=ami100 --gres=gpu:mi100:1 runs/alpine/hold.slurm (env_rocm.slurm's env)
set -euo pipefail
W=/scratch/local/pydyn-$SLURM_JOB_ID L=/scratch/local/pydyn-$SLURM_JOB_ID/stage
trap 'rm -rf $W' EXIT; trap 'exit 143' TERM  # Slurm's cancel and time limit
mkdir -p $L && cd $L
mkdir -p repin_cfl_745e94f70 zp_fixture repin_free_745e94f70 repin_phys_745e94f70 tm5-nc  # parallel tars race to make these
t0=$SECONDS
ls /projects/$USER/pydyn/stage/*.tar | xargs -P 8 -n 1 tar xf
echo "staged $(du -sh $L | cut -f1) in $(( SECONDS - t0 )) s"
E=/projects/$USER/software/anaconda/envs/pydyn ROCM=
if [ "$SLURM_JOB_PARTITION" = ami100 ]; then  # CuPy built for HIP against this ROCm (not a module in batch)
  E=$E-rocm R=/curc/sw/install/rocm/6.1.0
  ROCM="export ROCM_HOME=$R ROCM_PATH=$R HIP_PATH=$R LD_LIBRARY_PATH=$R/lib:$R/lib64:$R/llvm/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
fi
cat > $W/env.sh <<ENV
eval "\$(/curc/sw/install/miniforge3/24.11.3-0/bin/conda shell.bash hook)"
conda activate $E
$ROCM
export PYDYN_DEVICE=cuda PYDYN_WORKERS=8 CXX=g++ OPENBLAS_NUM_THREADS=1
export PYDYN_FFTW=$E/lib/libfftw3.so.3 PYDYN_LAPACK=$E/lib/liblapack.so
export PYDYN_DUMP_DIR=$L/repin_phys_745e94f70 PYDYN_FREE_DUMP_DIR=$L/repin_free_745e94f70
export PYDYN_CFL_DUMP_DIR=$L/repin_cfl_745e94f70 PYDYN_ZOOM_PHYS_DIR=$L/zp_fixture PYDYN_METEO_ROOT=$L/tm5-nc
cd /projects/$USER/pydyn
ENV
echo ready
sleep infinity & wait
