#!/bin/bash
# Pre-launch check, in R3's prelaunch_checks_03.txt form. Usage: prelaunch.sh "<label>"; run from the repo root.
echo "== $(date '+%Y-%m-%dT%H:%M:%S%z') pre-launch check before $1"
echo "HEAD $(git rev-parse HEAD)"
echo "git status --short: [$(git status --short)]"
echo "RAY_ADDRESS=[${RAY_ADDRESS-unset}]"
uptime
echo "-- tools.rehearsal_lock status"
PYTHONPATH=. ./venv/bin/python -m tools.rehearsal_lock status; echo "rc=$?"
echo "-- tools.rehearsal_lock ray-processes"
PYTHONPATH=. ./venv/bin/python -m tools.rehearsal_lock ray-processes; echo "rc=$?"
echo "-- runregistrykit list (whole output; stderr separately)"
./venv/bin/python -m runregistrykit list 2>var/handover-06/.list_err; echo "rc=$?"
echo "stderr: [$(cat var/handover-06/.list_err 2>/dev/null)]"; rm -f var/handover-06/.list_err
echo "-- journals under var/datastores"
find var/datastores \( -name '*-journal' -o -name '*-wal' -o -name '*-shm' \) 2>/dev/null; echo "(end)"
df -h . | tail -1
