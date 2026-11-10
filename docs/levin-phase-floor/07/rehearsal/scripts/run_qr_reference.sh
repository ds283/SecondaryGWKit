#!/bin/bash
# prompt 07 rehearsal: qr_reference.py for the three q-r calls, at g (with the reference) and at p; six processes at once.
R=/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07/rehearsal
P=/Users/ds283/Documents/Code/SecondaryGWKit-lpf07-91472db
G=/Users/ds283/Documents/Code/SecondaryGWKit-lpf07-7460035
PY=/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python
cd $R/state || exit 9
run() { t=$1; id=$2; ref=$3; if [ "$t" = p ]; then T=$P; else T=$G; fi
  env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 PYTHONPATH=$T:$R/scripts/capture $PY -B $R/scripts/qr_reference.py $T $R/state/inputs/qmr-$t-$id.pkl $R/out/qr_reference/qmr-$t-$id.json $ref > $R/out/qr_reference/qmr-$t-$id.txt 2>&1
  echo "qmr-$t-$id exit $? $(date -u +%H:%M:%S)"; }
date -u
run g 3-182 --reference & run g 0-10266 --reference & run g 3-52 --reference &
run p 3-20389 "" & run p 0-10036 "" & run p 3-20141 "" &
wait
date -u
