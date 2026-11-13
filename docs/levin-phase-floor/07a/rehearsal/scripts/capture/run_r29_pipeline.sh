#!/bin/sh
# prompt 07 r29: the replay of R29's located (and R28-changed) calls at g, its check, the four-block check's pass 1 at
# p, o and g, its capture spec, pass 2 (the captures), and the block check. Ray-free; at most six worker processes at
# once; under the held rehearsal lock; not a registered run. Each stage's output goes to r29/out/; the pipeline stops at
# the first failing stage.
set -u
Z=/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07/rehearsal/r29
PY=/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python
S=$Z/scripts/capture/replay07_r29.py
cd $Z/state || exit 9
export PYTHONDONTWRITEBYTECODE=1
stage() { name=$1; shift; echo "== stage $name start $(date -u +%Y-%m-%dT%H:%M:%SZ)"; "$@" > $Z/out/$name.txt 2>&1; rc=$?; echo "== stage $name exit $rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"; return $rc; }
stage replay_run $PY -I -B $S run $Z/state/replay_spec.json || exit 1
stage replay_check $PY -I -B $S check $Z/state/capture/replay $Z/state/replay_rows.json $Z/out/replay_calls.csv || exit 2
stage blocks1_run $PY -I -B $S run $Z/state/blocks1_spec.json || exit 3
stage blocks2_spec $PY -I -B $S spec blocks2 $Z/state/replay_rows.json $Z/state/blocks2_spec.json || exit 4
stage blocks2_run $PY -I -B $S run $Z/state/blocks2_spec.json || exit 5
stage blockcheck $PY -I -B $S blockcheck $Z/state/replay_rows.json $Z/out/blocks.json || exit 6
echo "== pipeline done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
