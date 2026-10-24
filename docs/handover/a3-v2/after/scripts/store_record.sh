#!/bin/bash
cd "$(dirname "$0")/../../.."
S=var/datastores/handover-A3-v2-lambdacdm.sqlite
echo "== $(date '+%Y-%m-%dT%H:%M:%S%z') §2.1"
echo "\$ ./venv/bin/python -m runregistrykit store create --purpose \"A3 v2: ...\" $S"
./venv/bin/python -m runregistrykit store create --purpose "A3 v2: the LambdaCDM store rebuilt from scratch at QuadSourceIntegral atol 1e-32 rtol 1e-7, 9280 items, 8 k over 1e5-3e8/Mpc, zend 0.1; built by handover-07-a3-v2-20261022T211155 at b90752d on the Mac Studio. Kept; archive on the Mac Studio." $S; echo "rc=$?"; echo
echo "== $(date '+%Y-%m-%dT%H:%M:%S%z') §2.2"
echo "\$ ./venv/bin/python -m runregistrykit store fingerprint --write $S"
./venv/bin/python -m runregistrykit store fingerprint --write $S; echo "rc=$?"; echo
echo "-- compare_fingerprints(the run's recorded fingerprint, the sidecar's)"
./venv/bin/python - <<'PY'
import json
from runregistrykit import stores
s = json.load(open("var/runs/handover-07-a3-v2-20261022T211155/status.json"))["fingerprint"]
m = json.load(open("var/datastores/handover-A3-v2-lambdacdm.manifest.json"))["fingerprint"]
print("run digest    ", s["digest"]); print("sidecar digest", m["digest"])
print("digest equal to 0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56:", m["digest"] == "0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56")
cf = getattr(stores, "compare_fingerprints", None)
if cf is None:
    import runregistrykit; cf = getattr(runregistrykit, "compare_fingerprints", None)
if cf is None:
    from datastorekit.store_fingerprint import compare_fingerprints as cf
d = cf(s, m); print(f"compare_fingerprints(run, sidecar): {len(d)} difference(s)", d if d else "")
PY
echo "rc=$?"; echo
echo "== $(date '+%Y-%m-%dT%H:%M:%S%z') §2.3"
echo "\$ ./venv/bin/python -m runregistrykit store show $S"
./venv/bin/python -m runregistrykit store show $S; echo "rc=$?"; echo
echo "== §2.4 every store file again"
bash var/handover-08/scripts/storefiles.sh | tee var/handover-08/state/store_files_after_record.txt
if diff -q var/handover-08/state/store_files_before.txt var/handover-08/state/store_files_after_record.txt >/dev/null; then echo "store files: identical to state/store_files_before.txt (SHA-256, size, mtime); no journal"; else echo "STORE FILES CHANGED"; diff var/handover-08/state/store_files_before.txt var/handover-08/state/store_files_after_record.txt; fi
