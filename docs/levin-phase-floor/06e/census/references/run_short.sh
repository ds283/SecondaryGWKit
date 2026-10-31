#!/bin/bash
# levin-phase-floor prompt 06e section 5, Z3: the short references (samples 2, 8, 10, 13, 14, 22, 23
# and 27), in-session and unregistered, strictly one at a time: for each sample given, orch_ref.py
# (06c's orchestrator's, copied unchanged) at P = 128 with 64/96 nodes, then at P = 64 with 96/128
# nodes, each with six workers, the next starting only when the last has exited. New in 06e.
# Environment: the tree g on PYTHONPATH; no bytecode written; git's optional locks off (orch_ref.py
# runs `git status`, which would otherwise refresh the worktree's index); one BLAS/OpenMP thread per
# process (six workers, six cores). Output: out/ref-sNN-P128.txt and out/ref-sNN-P64.txt, and one
# line per invocation in out/short_progress.txt.
# Usage (from references/): bash run_short.sh 02 08 10 13 14 22 23 27
set -u
cd "$(dirname "$0")"
G=/Users/ds283/Documents/Code/SecondaryGWKit-lpf06e-6d81fad
PY=/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python
# The census scripts follow the tree on PYTHONPATH: the captured group calls' pickles name
# harness._rebuild_lru_cache (06c's census harness, as depth20-tiling's commands put it there).
export PYTHONPATH=$G:/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06e/census/scripts PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
rc_all=0
for n in "$@"; do
  for spec in "128 64 96" "64 96 128"; do
    read -r P N1 N2 <<< "$spec"
    out="out/ref-s${n}-P${P}.txt"
    if [ -e "$out" ]; then echo "$(date -u +%FT%TZ) s$n P$P: $out exists, not run" >> out/short_progress.txt; rc_all=1; continue; fi
    echo "$(date -u +%FT%TZ) s$n P$P start (bash pid $$)" >> out/short_progress.txt
    "$PY" -B orch_ref.py "inputs/s${n}.pkl" --P "$P" --n1 "$N1" --n2 "$N2" --workers 6 > "$out" 2>&1
    rc=$?
    [ $rc -ne 0 ] && rc_all=$rc
    echo "$(date -u +%FT%TZ) s$n P$P exit $rc" >> out/short_progress.txt
  done
done
echo "$(date -u +%FT%TZ) all done, exit $rc_all" >> out/short_progress.txt
exit $rc_all
