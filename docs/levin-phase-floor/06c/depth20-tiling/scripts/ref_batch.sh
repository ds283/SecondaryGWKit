#!/bin/bash
# depth20-tiling step 4: the references, one rec.sh command per call (each estimated under 45 min
# by out/16_ref_size.txt), in sample order; stops at the first reference that does not converge
# (|S384 - S192| > min(abserr_census, abserr_pinned) / 100), per the brief's stop rule.
set -u
D=/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06c/depth20-tiling
PY=/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python
cd $D
export PYTHONPATH=/Users/ds283/Documents/Code/SecondaryGWKit-lpf06c-e5764a6:$D/scripts
i=17
for n in "$@"; do
  nn=$(printf %02d $n)
  scripts/rec.sh ${i}_ref_${nn} $PY -B scripts/refq.py run state/captured.json $n > /dev/null 2>&1
  $PY -B -c "
import json,sys
r=json.load(open('state/ref/$nn.ref.json')); p=json.load(open('state/runs/pinned/$nn'+'_pinned.json'))
e=min(float.fromhex(r['census_abserr']), float.fromhex(p['result']['abserr']))
c=float.fromhex(r['conv_diff'])
ok=c<=e/100
print('sample $nn conv_diff',c,'min abserr',e,'ratio',c/e,'converged',ok)
sys.exit(0 if ok else 3)
" >> out/ref_batch_progress.txt 2>&1 || { echo "STOP: sample $nn reference did not converge" >> out/ref_batch_progress.txt; exit 3; }
  i=$((i+1))
done
echo "batch done" >> out/ref_batch_progress.txt
