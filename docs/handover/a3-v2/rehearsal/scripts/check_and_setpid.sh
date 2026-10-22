#!/bin/bash
# The one check after a launch, then set-pid and status, recorded. Usage: check_and_setpid.sh <slug> [<interrupter pid>]
cd "$(dirname "$0")/../../.."
P=var/handover-06/out/prelaunch_checks_06.txt; L=var/handover-06/out/lock_06.txt
d=$(ls -d var/runs/handover-06-$1-* | head -1)
{ echo "== $(date '+%Y-%m-%dT%H:%M:%S%z') one check after $1 launch"; ./venv/bin/python var/handover-06/scripts/after_launch.py "$1" $2; [ -f "$d/interrupter.log" ] && cat "$d/interrupter.log"; uptime; } > var/handover-06/.check 2>&1
cat var/handover-06/.check >> $P; echo >> $P
pid=$(./venv/bin/python -c "import json; print(json.load(open('$d/status.json'))['pid'])")
for c in set-pid status; do
  if [ $c = set-pid ]; then args=(set-pid "$pid"); else args=(status); fi
  echo "== $(date '+%Y-%m-%dT%H:%M:%S%z') \$ PYTHONPATH=. ./venv/bin/python -m tools.rehearsal_lock ${args[*]}" >> $L
  PYTHONPATH=. ./venv/bin/python -m tools.rehearsal_lock "${args[@]}" >> $L 2>&1; echo "exit=$?" >> $L
done
grep -E 'True|False|state|kill -0|liveness|list rc|run\(s\)' var/handover-06/.check | cut -c1-200
grep -nE '^recorded|exit=|^owner pid|^Ray processes' $L | tail -4
echo "PID=$pid"
