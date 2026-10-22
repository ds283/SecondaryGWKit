#!/bin/bash
# Prompt 07 §2.2, verbatim but for <SHA> = b90752d and <hours> = 39-44. Run from the repository root by bash.
nohup sh -c 'PYTHONPATH=. ./venv/bin/python -u RunDrivers/quadsource_atol_sweep.py --build \
    --database var/datastores/handover-A3-v2-lambdacdm.sqlite \
    --job-name handover-A3-v2 --atol 1e-32 --rtol 1e-7 --cpus 6 \
    --register a3-v2 \
    --purpose "handover prompt 07: A3 v2, the LambdaCDM store rebuilt from scratch at QuadSourceIntegral atol 1e-32 rtol 1e-7 (main.py 9280-item work list, 8 k over 1e5-3e8/Mpc, zend 0.1, 6 cpus) on the Mac Studio at b90752d; R4 (handover 06) reproduced 39809dca on this host; estimated 39-44 h" \
    --campaign handover --prompt 07; \
  echo "** driver exit status $?"' \
  > var/runs/handover-07-a3-v2.launch.log 2>&1 < /dev/null &
echo "sh pid $!"
disown
