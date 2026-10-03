#!/bin/bash
# The interrupter's dummy test for a `--build` (a3-v2-readiness prompt 06a, B3), beside
# `docs/datastore-integrity-verification/interrupter_dummy_test.sh`, which is unchanged and covers
# a `scoped_pipeline_run.py` run. `interrupt_when.py` checks that the pid it signals is still the
# run's own, and takes the driver's name from the run's manifest `argv[0]`. A `--build` run's
# `argv[0]` is `docs/handover/quadsource_atol_sweep.py` (`RunRegistry.begin` records `sys.argv`
# before `run_build` replaces it). Fake run directories whose status.json names a `sleep` child of
# this shell, named for a driver with `exec -a`.
# Case A: a manifest whose argv[0] is the sweep script, a child named for it (must signal).
# Case B: the same manifest, a child named for scoped_pipeline_run.py (must refuse, child alive).
# Case C: a manifest whose argv[0] is empty (must exit without signalling, child alive).
# It polls every 1 s, not 15.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SCR="${1:?usage: interrupter_build_dummy_test.sh <work dir outside var/>}"
PY=$REPO/venv/bin/python  # the repository's venv
INT=$REPO/docs/datastore-integrity-verification/interrupt_when.py
[ -e "$SCR" ] && { echo "$SCR exists; give a new directory" >&2; exit 2; }; mkdir -p "$SCR"

write_status() {  # dir pid stage heartbeat-offset-seconds
  "$PY" - "$1" "$2" "$3" "$4" <<'EOF'
import json, os, sys
from datetime import datetime, timedelta, timezone
d, pid, stage, offset = sys.argv[1], int(sys.argv[2]), sys.argv[3], float(sys.argv[4])
beat = (datetime.now(timezone.utc) - timedelta(seconds=offset)).astimezone().isoformat(timespec="seconds")
tmp = os.path.join(d, "status.json.tmp")
json.dump({"state": "running", "pid": pid, "stage": stage, "heartbeat": beat, "exit_code": None}, open(tmp, "w"))
os.replace(tmp, os.path.join(d, "status.json"))
EOF
}
write_manifest() {  # dir argv0 slug
  printf '{"argv": ["%s", "--build", "--register", "%s"]}\n' "$2" "$3" > "$1/manifest.json"
}
GK='CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT'
SWEEP=docs/handover/quadsource_atol_sweep.py

echo "=== case A: a --build manifest (argv[0] the sweep script), a child named for it: signals"
A=$SCR/case-a; mkdir -p "$A"; write_manifest "$A" "$SWEEP" dummy-build-a
bash -c 'exec -a "python docs/handover/quadsource_atol_sweep.py --build --register dummy-build-a" sleep 120' &
CA=$!
write_status "$A" $CA "$GK, 45.00% (11 of 20 work items remaining)" 0
"$PY" -u "$INT" "$A" "^$GK, " 40 --interval 1 &
IA=$!
wait $IA; echo "interrupter exit=$?"
wait $CA; echo "child exit=$? (143 = killed by SIGTERM)"
echo "--- $A/interrupter.log"; cat "$A/interrupter.log"

echo "=== case B: the same manifest, a child named for scoped_pipeline_run.py: refuses"
B=$SCR/case-b; mkdir -p "$B"; write_manifest "$B" "$SWEEP" dummy-build-b
bash -c 'exec -a "python docs/gktk-remedial/scoped_pipeline_run.py --build --register dummy-build-b" sleep 30' &
CB=$!
write_status "$B" $CB "$GK, 45.00% (11 of 20 work items remaining)" 0
"$PY" -u "$INT" "$B" "^$GK, " 40 --interval 1
echo "interrupter exit=$?"
kill -0 $CB && echo "child $CB still alive (not signalled)"; kill $CB; wait $CB 2>/dev/null
echo "--- $B/interrupter.log"; cat "$B/interrupter.log"

echo "=== case C: a manifest whose argv[0] is empty: exits without signalling"
C=$SCR/case-c; mkdir -p "$C"; write_manifest "$C" "" dummy-build-c
bash -c 'exec -a "python docs/handover/quadsource_atol_sweep.py --build --register dummy-build-c" sleep 30' &
CC=$!
write_status "$C" $CC "$GK, 45.00% (11 of 20 work items remaining)" 0
"$PY" -u "$INT" "$C" "^$GK, " 40 --interval 1
echo "interrupter exit=$?"
kill -0 $CC && echo "child $CC still alive (not signalled)"; kill $CC; wait $CC 2>/dev/null
echo "--- $C/interrupter.log"; cat "$C/interrupter.log"
