#!/bin/bash
# After a run's pid died: no Ray process of this checkout left. Usage: after_run.sh <slug> <sh pid> <driver pid> [<interrupter pid>]
cd "$(dirname "$0")/../../.."
P=var/handover-06/out/prelaunch_checks_06.txt
d=$(ls -d var/runs/handover-06-$1-* | head -1)
rp=$(sed 's/\x1b\[[0-9;]*m//g' $d/stdout.log $d/stderr.log var/runs/handover-06-$1.launch.log | grep -oE 'pid=[0-9]+' | grep -oE '[0-9]+' | sort -u | paste -sd, -)
all=$2,$3${4:+,$4}${rp:+,$rp}
{ echo "== $(date '+%Y-%m-%dT%H:%M:%S%z') after $1: no Ray process left"
  echo "HEAD $(git rev-parse HEAD)"; echo "git status --short: [$(git status --short)]"; uptime
  echo "-- tools.rehearsal_lock ray-processes"; PYTHONPATH=. ./venv/bin/python -m tools.rehearsal_lock ray-processes; echo "rc=$?"
  echo "-- Ray pids its logs name: $rp"; echo "-- ps -p $all"; ps -p $all; echo "rc=$?"
  ./venv/bin/python -m runregistrykit list | tail -1
  echo "-- launch log last line: $(tail -1 var/runs/handover-06-$1.launch.log)"
  echo "-- files in run dir: $(ls -A $d | tr '\n' ' ')"
  [ -f $d/interrupter.log ] && { echo "-- interrupter.log"; cat $d/interrupter.log; }
  echo "-- stderr.log"; cat $d/stderr.log; df -h . | tail -1; echo; } >> $P 2>&1
sed -n "/after $1: no Ray/,/-- files in run dir/p" $P | cut -c1-200
echo "ALLPIDS=$all"
