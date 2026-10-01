#!/bin/bash
# The interrupter's dummy test (prompt 10 §1.4): fake run directories whose status.json names a
# `sleep` child of this shell. Case A: stale heartbeat at a matching stage (must wait), then a
# fresh one below the percentage (must wait), then at it (must signal). Case B: the child dies
# before the stage matches (must exit without signalling). Case C: the pid's command line does
# not name the driver and the slug (must exit without signalling). Case D: --hold at a bare
# banner (must hold, read again, then signal). Case E: --hold, and the stage completes during the
# hold (must not signal, and must exit when the child dies). It polls every 1 s, not 15.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SCR="${1:?usage: interrupter_dummy_test.sh <work dir outside var/>}"
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
write_manifest() {  # dir slug
  printf '{"argv": ["docs/gktk-remedial/scoped_pipeline_run.py", "--register", "%s"]}\n' "$2" > "$1/manifest.json"
}
GK='CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT'

echo "=== case A: signals when the stage matches"
A=$SCR/case-a; mkdir -p "$A"; write_manifest "$A" dummy-a
bash -c 'exec -a "python docs/gktk-remedial/scoped_pipeline_run.py --register dummy-a" sleep 120' &
CA=$!
write_status "$A" $CA "$GK, 45.00% (11 of 20 work items remaining)" 1000
"$PY" -u "$INT" "$A" "^$GK, " 40 --interval 1 &
IA=$!
sleep 3
write_status "$A" $CA "$GK, 10.00% (18 of 20 work items remaining)" 0
sleep 3
write_status "$A" $CA "$GK, 45.00% (11 of 20 work items remaining)" 0
wait $IA; echo "interrupter exit=$?"
wait $CA; echo "child exit=$? (143 = killed by SIGTERM)"
echo "--- $A/interrupter.log"; cat "$A/interrupter.log"

echo "=== case B: exits silently when the child dies first"
B=$SCR/case-b; mkdir -p "$B"; write_manifest "$B" dummy-b
bash -c 'exec -a "python docs/gktk-remedial/scoped_pipeline_run.py --register dummy-b" sleep 4' &
CB=$!
write_status "$B" $CB "$GK, 10.00% (18 of 20 work items remaining)" 0
"$PY" -u "$INT" "$B" "^$GK, " 40 --interval 1 &
IB=$!
wait $CB; echo "child exit=$? (0 = sleep ended by itself)"
wait $IB; echo "interrupter exit=$?"
echo "--- $B/interrupter.log"; cat "$B/interrupter.log"

echo "=== case C: refuses a pid whose command line does not name the run"
C=$SCR/case-c; mkdir -p "$C"; write_manifest "$C" dummy-c
sleep 30 &
CC=$!
write_status "$C" $CC "$GK, 45.00% (11 of 20 work items remaining)" 0
"$PY" -u "$INT" "$C" "^$GK, " 40 --interval 1
echo "interrupter exit=$?"
kill -0 $CC && echo "child $CC still alive (not signalled)"; kill $CC; wait $CC 2>/dev/null
echo "--- $C/interrupter.log"; cat "$C/interrupter.log"

echo "=== case D: --hold 4 at a bare banner (0%): holds, reads again, then signals"
E=$SCR/case-d; mkdir -p "$E"; write_manifest "$E" dummy-d
bash -c 'exec -a "python docs/gktk-remedial/scoped_pipeline_run.py --register dummy-d" sleep 120' &
CD=$!
write_status "$E" $CD "$GK" 0
"$PY" -u "$INT" "$E" "^$GK\$" 0 --interval 1 --hold 4
echo "interrupter exit=$?"
wait $CD; echo "child exit=$? (143 = killed by SIGTERM)"
echo "--- $E/interrupter.log"; cat "$E/interrupter.log"

echo "=== case E: --hold 4, but the stage completes during the hold: no signal; exits when the child dies"
F=$SCR/case-e; mkdir -p "$F"; write_manifest "$F" dummy-e
bash -c 'exec -a "python docs/gktk-remedial/scoped_pipeline_run.py --register dummy-e" sleep 9' &
CE=$!
write_status "$F" $CE "$GK" 0
"$PY" -u "$INT" "$F" "^$GK" 0 --interval 1 --hold 4 &
IE=$!
sleep 2
write_status "$F" $CE "$GK, all work items complete" 0
wait $CE; echo "child exit=$? (0 = sleep ended by itself)"
wait $IE; echo "interrupter exit=$?"
echo "--- $F/interrupter.log"; cat "$F/interrupter.log"
