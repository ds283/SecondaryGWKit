#!/bin/bash
# R3's launch command (rrk-v020-adoption verification §1.1), with prompt 06's names. From the repo root:
#   bash var/handover-06/scripts/launch.sh <slug> <store: reference|resumed> <resume: yes|no> "<purpose>"
slug=$1; store=$2; resume=$3; purpose=$4
R=""; [ "$resume" = yes ] && R=" --resume"
echo "== $(date '+%Y-%m-%dT%H:%M:%S%z') launch $slug (log var/runs/handover-06-$slug.launch.log)"
echo "git status --short: [$(git status --short)]"
uptime
nohup sh -c "PYTHONPATH=. ./venv/bin/python -u RunDrivers/quadsource_atol_sweep.py \\
    --build --rehearsal$R \\
    --database var/datastores/handover-06-$store.sqlite \\
    --job-name handover-06 --atol 1e-32 --rtol 1e-7 --cpus 6 \\
    --register $slug --purpose \"\$0\" --campaign handover --prompt 06; \\
  echo \"** driver exit status \$?\"" "$purpose" \
  > var/runs/handover-06-$slug.launch.log 2>&1 &
echo "sh pid $!"
disown
