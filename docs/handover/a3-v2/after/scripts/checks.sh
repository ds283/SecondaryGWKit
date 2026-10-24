#!/bin/bash
# R4's checks.sh, on the A3 v2 primary alone.
cd "$(dirname "$0")/../../.."
S=var/datastores/handover-A3-v2-lambdacdm.sqlite; O=var/handover-08/out
bash var/handover-08/scripts/storefiles.sh > var/handover-08/state/store_files_before_checks.txt
{ echo "\$ PYTHONPATH=. ./venv/bin/python docs/a3-v2-readiness/rehearsal/closed_store_state.py $S"; PYTHONPATH=. ./venv/bin/python docs/a3-v2-readiness/rehearsal/closed_store_state.py $S; echo "rc=$?"; } > $O/closed_store_state_08.txt 2>&1
{ echo "\$ PYTHONPATH=. ./venv/bin/python docs/datastore-integrity-audit/replicated_serial_census.py $S"; PYTHONPATH=. ./venv/bin/python docs/datastore-integrity-audit/replicated_serial_census.py $S; echo "rc=$?"; } > $O/census_08.txt 2>&1
{ echo "\$ PYTHONPATH=. ./venv/bin/python docs/dk-and-rpk-migration/rehearsal/store_soundness.py $S"; PYTHONPATH=. ./venv/bin/python docs/dk-and-rpk-migration/rehearsal/store_soundness.py $S; echo "rc=$?"; } > $O/store_soundness_08.txt 2>&1
bash var/handover-08/scripts/storefiles.sh > var/handover-08/state/store_files_after_checks.txt
if diff -q var/handover-08/state/store_files_before.txt var/handover-08/state/store_files_after_checks.txt >/dev/null; then echo "STORE FILES: identical to store_files_before.txt after the checks"; else echo "STORE FILES CHANGED"; diff var/handover-08/state/store_files_before.txt var/handover-08/state/store_files_after_checks.txt; fi
