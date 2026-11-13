#!/bin/sh
# prompt 07a r30: R29's run_r29_pipeline.sh (984105f7) for R30: the replay of R30's located rows (and P1's, the w rows,
# the block rows and the 49 abserr-only records' rows) at h, its check against R30, the four-block check's capture
# spec, pass 2 (the captures), and the block check. Pass 1 (blocks1) ran in session before this, as the pilot.
# Ray-free; at most six worker processes at once; under the held rehearsal lock; not a registered run. Each stage's
# output goes to rehearsal/out/; the pipeline stops at the first failing stage.
set -u
Z=/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07a/rehearsal
PY=/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python
S=$Z/scripts/capture/replay07a_r30.py
cd $Z/state || exit 9
export PYTHONDONTWRITEBYTECODE=1
stage() { name=$1; shift; echo "== stage $name start $(date -u +%Y-%m-%dT%H:%M:%SZ)"; "$@" > $Z/out/$name.txt 2>&1; rc=$?; echo "== stage $name exit $rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"; return $rc; }
stage replay_run $PY -I -B $S run $Z/state/replay_spec.json || exit 1
stage replay_check $PY -I -B $S check $Z/state/capture/replay $Z/state/replay_rows.json $Z/out/replay_calls.csv || exit 2
stage blocks2_spec $PY -I -B $S spec blocks2 $Z/state/replay_rows.json $Z/state/blocks2_spec.json || exit 4
stage blocks2_run $PY -I -B $S run $Z/state/blocks2_spec.json || exit 5
stage blockcheck $PY -I -B $S blockcheck $Z/state/replay_rows.json $Z/out/blocks.json || exit 6
echo "== pipeline done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
