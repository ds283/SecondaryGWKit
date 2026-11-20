#!/bin/bash
# Orchestrator's stop-2 re-runs (07c Part C), with the census agent's harness. Outputs under review/stop2/.
set -u
cd /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07c/census
O=/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07c/review/stop2
T=/Users/ds283/Documents/Code/SecondaryGWKit-lpf07c-cc3368c
P=/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python
mkdir -p "$O"
date -u
printf '%s\n' '{"units": [{"uid": "g-1e-8-0-2108"}, {"uid": "g-1e-8-3-14505"}]}' > "$O/units_parent.json"
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$T" "$P" -B scripts/run_k0.py parent "$O/units_parent.json" "$O/parent" --workers 2 --tree h > "$O/reruns_parent.log" 2>&1; echo "parent exit $?"
printf '%s\n' '{"units": [{"uid": "g-1e-8-0-33929"}]}' > "$O/units_acts.json"
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$T" "$P" -B scripts/run_k0.py tree "$O/units_acts.json" "$O/h1" --workers 1 --tree h > "$O/reruns_h1.log" 2>&1; echo "h1 exit $?"
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$T" "$P" -B scripts/run_k0.py tree "$O/units_acts.json" "$O/h2" --workers 1 --tree h > "$O/reruns_h2.log" 2>&1; echo "h2 exit $?"
date -u
cd "$T"; git status --short --ignored | wc -l
