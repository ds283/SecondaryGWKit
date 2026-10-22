#!/bin/bash
# The seven roots, serially, at HEAD, each by CLAUDE.md's command; full outputs to var/handover-06/roots/.
cd "$(dirname "$0")/../../.."
mkdir -p var/handover-06/roots
echo "== $(date '+%Y-%m-%dT%H:%M:%S%z') the seven roots at HEAD $(git rev-parse HEAD)"
for r in AdaptiveLevin ComputeTargets CosmologyModels LiouvilleGreen ObjectFactories RunDrivers tools; do
  f=var/handover-06/roots/$r.txt
  echo "\$ PYTHONPATH=. ./venv/bin/python -m unittest discover -s $r/tests -t . > $f 2>&1; grep -E '^(Ran|OK|FAILED)|^(FAIL|ERROR):' $f"
  echo "   load before: $(uptime | sed 's/.*load averages*: //')   start $(date '+%H:%M:%S')"
  PYTHONPATH=. ./venv/bin/python -m unittest discover -s $r/tests -t . > $f 2>&1
  echo "   exit $?   end $(date '+%H:%M:%S')"
  grep -E '^(Ran|OK|FAILED)|^(FAIL|ERROR):' $f
  echo
done
