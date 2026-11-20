#!/bin/sh
# R|together at 980 and 1e8 at HEAD with T's Levin phase derivative from the phase (cell_phase.py).
cd "$(dirname "$0")/../../../.."
D=docs/handover/b2-at-head
git rev-parse HEAD > $D/cells-phase/head.txt
for x in 980 1e+08; do
  PYTHONPATH=. ./venv/bin/python -B $D/scripts/cell_phase.py --cell together $x R --out $D/cells-phase/R-together-$x.json
  echo "R together $x rc=$?" >> $D/cells-phase/done.txt
done
