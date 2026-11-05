#!/bin/bash
#SBATCH --job-name=pydyn-pack
#SBATCH --partition=acompile
#SBATCH --qos=compile
#SBATCH --account=ucb-general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/%u/pydyn/runs/alpine/pack_%j.log
# The dump sets and meteo as tar chunks in /projects for hold.slurm: /scratch/alpine reads small files
# at ~80 ms each, a tar from /projects at ~350 MB/s.  After copying a set: sbatch runs/alpine/pack.slurm
set -euo pipefail
OUT=/projects/$USER/pydyn/stage
rm -rf $OUT.new && mkdir -p $OUT.new
pack() {  # pack <parent> <dir> <name>: <name>.NN.tar of 1500 files each, 16 at once
  cd "$1" && find "$2" -type f | LC_ALL=C sort | split -l 1500 -d -a 2 - $OUT.new/$3.list.
  ls $OUT.new/$3.list.* | xargs -P 16 -I{} sh -c 'tar cf "$(echo {} | sed s/.list//).tar" -T {}'
  rm $OUT.new/$3.list.*
}
D=/scratch/alpine/$USER/pydyn/dumps
for s in repin_cfl_745e94f70 zp_fixture repin_free_745e94f70 repin_phys_745e94f70; do
  t0=$SECONDS; pack $D $s $s; echo "$s $(( SECONDS - t0 )) s"
done
t0=$SECONDS; pack /scratch/alpine/$USER/pydyn/METEO tm5-nc meteo; echo "meteo $(( SECONDS - t0 )) s"
rm -rf $OUT && mv $OUT.new $OUT
ls $OUT | wc -l; du -sh $OUT
