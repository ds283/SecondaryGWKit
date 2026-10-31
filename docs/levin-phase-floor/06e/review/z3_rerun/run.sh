#!/bin/bash
# Orchestrator's Z3 re-run (06e review): sample 2 at a resolution pair the agent did not use,
# P = 96 with 80/112 nodes, six workers, by orch_ref.py (8a0d291d) at the tree g.
cd /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06e/census/references
export PYTHONPATH=/Users/ds283/Documents/Code/SecondaryGWKit-lpf06e-6d81fad:/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06e/census/scripts PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
O=/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06e/review/z3_rerun
echo "$(date -u +%FT%TZ) start sh pid $$" >> $O/progress.txt
/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python -B orch_ref.py inputs/s02.pkl --P 96 --n1 80 --n2 112 --workers 6 > $O/ref-s02-P96.txt 2>&1
rc=$?
echo "$(date -u +%FT%TZ) exit $rc" >> $O/progress.txt
exit $rc
