#!/bin/bash
# The zoom regression battery (docs/state.md section 3f): every zoom set the reference is
# bitwise on, per event and in a free run on the model's input files alone.  Run on Orion or
# Hercules with ROOT = an unpacked copy of this repo (`git archive HEAD | tar -x -C $ROOT`);
# writes $OUT/<name>_<job>.out and prints the job ids.  check_zoom.sh reads them back.
# The defaults are the NOAA co2 project's paths; PYDYN_HPC_ROOT (the model runs) and the
# tests' PYDYN_METEO_ROOT, PYDYN_CHEM_ROOT and PYDYN_IC_FILE override them.
set -eu
P=${PYDYN_HPC_ROOT:-/work2/noaa/co2/ash/pydyn}
ROOT=${ROOT:?ROOT = the pydyn copy to test}
OUT=${OUT:-$ROOT/regress}
M=${PYDYN_METEO_ROOT:-/work2/noaa/co2/METEO/tm5-nc}
I=${PYDYN_CHEM_ROOT:-/work2/noaa/co2/input/ch4c13}
IC=${PYDYN_IC_FILE:-/work/noaa/co2/input/ch4_IC_2026/ch4_init_glb3x2_mmr.nc}
ALL="--meteo $M --emissions $I/all_source --chemistry $I --initial $IC"
XS=$P/runs/ch4u-3x2nam1x1-pydyn-pyd4s/run/ch4u-3x2nam1x1-pydyn-pyd4s.ml137.hercules.x
X5=$P/runs/ch4u-3x2nam1x1-pydyn-pyd5/run/ch4u-3x2nam1x1-pydyn-pyd5.ml137.hercules.x
XO=$P/runs/ch4u-3x2nam1x1-pydyn-pyd3/run/ch4u-3x2nam1x1-pydyn-pyd3.ml137.orion.x
mkdir -p "$OUT"
here=$(cd "$(dirname "$0")" && pwd)
sub() {  # name exe set args: a per-event job replays its 6 read intervals in 6 processes
    case $4 in --free-run*) c=1 ;; *) c=6 ;; esac
    ROOT=$ROOT X=$2 D=$P/dumps/$3 ARGS=$4 sbatch --parsable --export=ALL -c $c -J "$1" -o "$OUT/%x_%j.out" \
        "$here/replay.slurm"
}
sub zps-ev "$XS" zps_replay "$ALL"
sub zps-free "$XS" zps_replay "--free-run $ALL"
sub zpfix-ev "$X5" zpfix_replay "$ALL"
sub zpfix-free "$X5" zpfix_replay "--free-run $ALL"
sub zm-ev "$XO" zm_replay "--meteo $M --initial $IC"
sub zm-free "$XO" zm_replay "--free-run --meteo $M --initial $IC"
