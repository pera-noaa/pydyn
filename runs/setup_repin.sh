#!/usr/bin/env bash
# Re-pin sets: pydyn_dump_{phys,free,cfl} regenerated from a clean clone at <sha>, by the
# recipe of pydyn docs/state.md section 3b (1 CH4 tracer, scaling 1.0, 23 h of 2000-01-01).
# Usage: runs/setup_repin.sh <sha> <phys|free|cfl>   -> .tm5-cloud-scratch/pydyn_build/dumps/repin_<kind>_<sha>
# Derived from setup_pyd3ctl.sh; paths and the end time are set in the rc copy, because
# pycasso expands those before command-line overrides apply.  The transport-only macro list
# is read from the phys run's expanded rc, so phys must be built first.
set -euo pipefail
SHA=$1; KIND=$2; LABEL=repin_${KIND}_$SHA
# PYDYN_TM5ZOOM: the tm5-zoom checkout (default ../tm5-zoom, as tests/common.py); TM5_ENV: the
# script that sets up its Mac build environment
TM5ZOOM=${PYDYN_TM5ZOOM:-$(cd "$(dirname "$0")/../.." && pwd)/tm5-zoom}
SCR=$TM5ZOOM/.tm5-cloud-scratch
B=$SCR/pydyn_build
CLONE=$B/ctl-$SHA
if [ ! -d $CLONE ]; then
  git clone -q --no-checkout "$TM5ZOOM" $CLONE
  git -C $CLONE checkout -q $SHA
fi
[ "$(git -C $CLONE rev-parse HEAD | cut -c1-9)" = "$SHA" ] || { echo "clone not at $SHA" >&2; exit 1; }
[ -z "$(git -C $CLONE status --porcelain --untracked-files=no)" ] || { echo "clone is dirty" >&2; exit 1; }
RC=$CLONE/proj/ct-unified/rc/$LABEL.rc
mkdir -p $B/runs/$LABEL/output $B/runs/$LABEL/restart $B/dumps/$LABEL
sed -E -e "s#^my\.gitroot +: .*#my.gitroot : $CLONE#" \
    -e "s#^my\.basedir +: .*#my.basedir : $B/runs#" \
    -e "s#^(unoptimized|forward|inverse)\.runid +: .*#\1.runid : $LABEL#" \
    -e "s#^istart\.66\.filename +: .*#istart.66.filename : $SCR/ic/ch4_init_glb3x2_mmr.nc#" \
    -e "s#^timerange\.end +: .*#timerange.end : 2000-01-01 23:00:00#" \
    $CLONE/proj/ct-unified/rc/ch4_6x4_fwd.rc > $RC
for key in "my.gitroot : $CLONE" "my.basedir : $B/runs" "unoptimized.runid : $LABEL" \
           "timerange.end : 2000-01-01 23:00:00"; do
  grep -qxF "$key" $RC || { echo "setup_repin: rc copy lacks '$key'" >&2; exit 1; }
done
python3 "$(dirname "$0")/make_scaling_one.py" $B/runs/$LABEL/output
EXTRA=()
if [ $KIND != phys ]; then
  PRC=$B/runs/repin_phys_$SHA/run/repin_phys_$SHA.rc
  M=$(grep -E '^build\.configure\.macro\.define +:' $PRC | sed -E 's/^[^:]*: *//')
  [ -n "$M" ] || { echo "no macro list in $PRC" >&2; exit 1; }
  EXTRA+=("build.configure.macro.define=$M without_emission without_chemistry")
fi
if [ $KIND = cfl ]; then
  EXTRA+=(time.ntimestep=10800 time.ndyn_max=10800 "meteo.tinterp.*=interp3")
fi
export TM5_GITROOT=$CLONE CARBONTRACKER=$SCR
export CONDA_PREFIX=/opt/anaconda3   # machine-local-mac.rc reads it since 5292746df; the older sets used this prefix
source "${TM5_ENV:-$HOME/Documents/local-tm5/tm5_env.sh}"
cd $CLONE
base/trunk/bin/pycasso_setup_tm5 $RC \
  build.configure.flags=optim-fast \
  "compiler.lib.lapack.libs=-framework Accelerate -L/opt/homebrew/lib -lfftw3" \
  "compiler.flags.optim-fast.fflags=-O3 -march=native -ffp-contract=off" \
  par.mpi=F par.ntask=1 par.compute_ranks=1 par.io_ranks=0 par.io_server_ranks=0 \
  ct.ensemble.size=1 ct.ensemble.total_members=1 cfl.cache.read=F cfl.cache.write=F \
  pydyn.dump=T pydyn.dump.dir=$B/dumps/$LABEL pydyn.dump.max_events=3000 \
  ${EXTRA[@]+"${EXTRA[@]}"} --allow-new-keys --no-submit -j 8
cd $B/runs/$LABEL/run
DYLD_LIBRARY_PATH=/opt/homebrew/lib ./$LABEL.ml137.local-mac.x $LABEL.rc > tm5.out 2>&1 || true
echo "$LABEL: $(ls $B/dumps/$LABEL/*.meta | wc -l) events; tail of tm5.out:"; tail -4 tm5.out
