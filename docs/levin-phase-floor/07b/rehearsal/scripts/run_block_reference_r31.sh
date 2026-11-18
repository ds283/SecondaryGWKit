#!/bin/bash
# prompt 07b r31: R30's run_block_reference_r30.sh (4a2bd2c3) for R31: block_reference_r30.py (R30's, unchanged; no
# --reference) on R31's four block captures at the tree h (0a103b5a), one process each, at most four at once. Ray-free.
Z=/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07b/rehearsal
PY=/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python
T=/Users/ds283/Documents/Code/SecondaryGWKit-lpf07b-0a103b5
mkdir -p $Z/out/block_reference
cd $T || exit 9
one() { f=$1; b=$(basename $f .pkl)
  env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 PYTHONPATH=$T:$Z/scripts/capture $PY -B $Z/scripts/block_reference_r30.py $T $f $Z/out/block_reference/$b.json > $Z/out/block_reference/$b.txt 2>&1
  echo "$b exit $? $(date -u +%H:%M:%S)"; }
export -f one; export Z PY T
date -u
ls $Z/state/inputs/J4-h-*.pkl $Z/state/inputs/Y4-h-*.pkl | xargs -P 4 -L 1 bash -c 'one "$0"'
date -u
