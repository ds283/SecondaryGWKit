#!/bin/bash
# usage: e3_run.sh <n> ; one E3 run of the shipped reader on 07's resumed store, read-only
cd /Users/ds283/Documents/Code/SecondaryGWKit || exit 1
n=$1; V=var/a3-v2-readiness-07a
{
echo "RAY_ADDRESS=${RAY_ADDRESS-<unset>}"
echo "this venv's ray processes before: $(ps -axo pid,command | grep 'SecondaryGWKit/venv' | grep -i ray | grep -v grep | wc -l | tr -d ' ')"
$V/snapshot.sh "before run-$n"
start=$(date +%s)
PYTHONPATH=. ./venv/bin/python extract_tensor_source_data.py \
    --database var/datastores/a3-v2-readiness-07-resumed.sqlite \
    --ray-address local --models LambdaCDM \
    --output var/a3-v2-readiness-07a/extract/run-$n/ \
    > var/a3-v2-readiness-07a/extract/run-$n.log 2>&1
rc=$?
echo "run-$n exit $rc, wall $(( $(date +%s) - start )) s"
$V/snapshot.sh "after run-$n"
echo "this venv's ray processes after:"; ps -axo pid,ppid,command | grep 'SecondaryGWKit/venv' | grep -i ray | grep -v grep | cut -c1-200
} 2>&1 | tee $V/state/run-$n.txt
