#!/bin/bash
# Pre-launch check, launch, then the interrupter, detached, once the run's status.json exists.
#   bash launch_with_interrupter.sh <slug> <store> <resume yes|no> "<purpose>" "<regex>" <hold> "<previous pids>"
cd "$(dirname "$0")/../../.."
P=var/handover-06/out/prelaunch_checks_06.txt
bash var/handover-06/scripts/prelaunch.sh "$1" >> $P 2>&1
{ echo "-- ps -p $7 (previous run's pids)"; ps -p "$7"; echo "rc=$?"; echo; } >> $P
bash var/handover-06/scripts/launch.sh "$1" "$2" "$3" "$4" >> $P 2>&1
d=
for i in $(seq 1 20); do d=$(ls -d var/runs/handover-06-$1-* 2>/dev/null | head -1); [ -n "$d" ] && [ -f "$d/status.json" ] && break; sleep 1; done
if [ -n "$5" ]; then
  nohup ./venv/bin/python -u docs/datastore-integrity-verification/interrupt_when.py "$d" "$5" 0 --interval 5 --hold "$6" > /dev/null 2>&1 &
  ip=$!; disown
  echo "interrupter pid $ip at $(date +%H:%M:%S): interrupt_when.py $d '$5' 0 --interval 5 --hold $6" >> $P
fi
tail -6 $P | cut -c1-200
