#!/bin/sh
# Calibration (scratch, not a record): B2's R and E cells at HEAD, one at a time.
cd /Users/ds283/Documents/Code/SecondaryGWKit
S=/private/tmp/claude-35086/-Users-ds283--claude-spares-spare-nIjlMh/cff1227b-41d0-4487-a706-efd137a71485/scratchpad/calib
git rev-parse HEAD > $S/head.txt
for obj in R E; do
 for cell in "together 980" "T-first 980" "q-smooth 980" "together 10000" "T-first 10000" "q-smooth 10000" "together 100000" "T-first 100000" "q-smooth 100000" "together 1e+06" "T-first 1e+06" "q-smooth 1e+06" "together 1e+07" "T-first 1e+07" "q-smooth 1e+07" "together 1e+08"; do
  set -- $cell
  timeout 1200 ./venv/bin/python -B docs/handover/b2/scripts/b2_cells.py --cell $1 $2 $obj --out "$S/$obj-$1-$2.json" > "$S/$obj-$1-$2.log" 2>&1
  echo "$obj $1 $2 rc=$?" >> $S/done.txt
 done
done
echo ALLDONE >> $S/done.txt
