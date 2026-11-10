#!/bin/bash
# prompt 07 r29: block_reference.py for the four blocks at R27's (p), R28's (o) and R29's (g) trees, the reference on g's
# captures; at most six processes at once (xargs -P 6). Ray-free.
Z=/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07/rehearsal/r29
PY=/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python
mkdir -p $Z/out/block_reference
cd $Z/state || exit 9
one() { t=$1; f=$2; ref=$3
  case $t in p) T=/Users/ds283/Documents/Code/SecondaryGWKit-lpf07-91472db;; o) T=/Users/ds283/Documents/Code/SecondaryGWKit-lpf07-7460035;; g) T=/Users/ds283/Documents/Code/SecondaryGWKit-lpf07-e182bc7;; esac
  b=$(basename $f .pkl)
  env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 PYTHONPATH=$T:$Z/scripts/capture $PY -B $Z/scripts/block_reference.py $T $f $Z/out/block_reference/$b.json $ref > $Z/out/block_reference/$b.txt 2>&1
  echo "$b exit $? $(date -u +%H:%M:%S)"; }
export -f one; export Z PY
date -u
{ for f in $Z/state/inputs/J4-g-*.pkl $Z/state/inputs/Y4-g-*.pkl; do echo "g $f --reference"; done
  for f in $Z/state/inputs/J4-p-*.pkl $Z/state/inputs/Y4-p-*.pkl; do echo "p $f -"; done
  for f in $Z/state/inputs/J4-o-*.pkl $Z/state/inputs/Y4-o-*.pkl; do echo "o $f -"; done; } | xargs -P 6 -L 1 bash -c 'one "$0" "$1" "$([ "$2" = - ] || echo "$2")"'
date -u
