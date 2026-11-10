#!/bin/sh
# prompt 07 rehearsal: the replay of R28's located calls (pass 1 at g, 224 rows), its check, pass 1 at p (P1's two rows
# and the three q-r rows), the capture spec, the captures (S2's two G+q+r calls at p; the three q-r calls at p and g)
# and their check. Ray-free; at most six worker processes at once; run under the held rehearsal lock; not a
# registered run (the user's terms). Each stage's output goes to out/; the pipeline stops at the first failing stage.
set -u
R=/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07/rehearsal
PY=/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python
S=$R/scripts/capture/replay07.py
cd $R/state || exit 9
export PYTHONDONTWRITEBYTECODE=1
stage() { name=$1; shift; echo "== stage $name start $(date -u +%Y-%m-%dT%H:%M:%SZ)"; "$@" > $R/out/$name.txt 2>&1; rc=$?; echo "== stage $name exit $rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"; return $rc; }
stage replay_run $PY -I -B $S run $R/state/replay_spec.json || exit 1
stage replay_check $PY -I -B $S check $R/state/capture/replay $R/state/replay_rows.json $R/out/replay_calls.csv || exit 2
stage pass1p_run $PY -I -B $S run $R/state/pass1p_spec.json || exit 3
stage capture_spec $PY -I -B $S spec capture $R/state/replay_rows.json $R/state/capture_spec.json || exit 4
stage capture_run $PY -I -B $S run $R/state/capture_spec.json || exit 5
stage capture_check $PY -I -B $S capcheck $R/state/replay_rows.json $R/out/captures.json || exit 6
echo "== pipeline done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
