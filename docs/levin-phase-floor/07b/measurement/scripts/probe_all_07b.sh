#!/bin/bash
# levin-phase-floor 07b Part A, section 2.2: run leafprobe_07b.py on every leaf the specs name (state/leafspecs/v4_*.json
# and attr_*.json), one probe at a time: (a)'s leaves with --eighths and --perturb D, D the ulp of the raw phase at the
# leaf's window edge (the call's largest |theta| at a window edge, or at x_span's ends if it has no window); b4's and
# b8's leaves plain. New in 07b. Usage (cwd the tree, PYTHONPATH the tree): probe_all_07b.sh
R=/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07b/measurement
PY=/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python
for spec in $R/state/leafspecs/v4_*.json $R/state/leafspecs/attr_*.json; do
  sid=$($PY -c "import json;print(json.load(open('$spec'))['sid'])")
  for v in tree 07 b4 b8; do
    L=$($PY -c "import json;s=json.load(open('$spec'));print(' '.join(str(n['i']) for n in s['named'] if n['variant']=='$v'))")
    [ -z "$L" ] && continue
    rep=$R/out/replays/$v/$sid.json
    out=$R/out/probe/${v}_${sid}.json
    [ -e "$out" ] && { echo "exists $out"; continue; }
    if [ "$v" = "tree" ] || [ "$v" = "07" ]; then
      D=$($PY -B -c "
import sys; sys.path.insert(1, '$R/scripts')
import json, numpy as np, hashlib, ray.cloudpickle as cp
d=json.load(open('$rep')); c=d['header']['capture']
f=open(c['path'],'rb'); f.seek(c['offset']); rec=cp.loads(f.read(c['length']))
xs=[float.fromhex(w['lo']) for w in d['windows']]+[float.fromhex(w['hi']) for w in d['windows']] or [float.fromhex(x) for x in d['x_span']]
print(float(max(np.spacing(abs(rec['theta']['theta'](x))) for x in xs)).hex())")
      $PY -B $R/scripts/leafprobe_07b.py $rep $out --leaves $L --eighths --perturb $D | grep -v "^wrote"
    else
      $PY -B $R/scripts/leafprobe_07b.py $rep $out --leaves $L | grep -v "^wrote"
    fi
  done
done
