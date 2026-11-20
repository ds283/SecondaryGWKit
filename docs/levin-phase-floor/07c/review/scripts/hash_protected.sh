#!/bin/bash
# Usage: hash_protected.sh TAG   (run from the main checkout). Read-only on everything it hashes.
set -u
cd /Users/ds283/Documents/Code/SecondaryGWKit
R=var/levin-phase-floor-07c/review/checks
TAG=$1
shasum -a 256 -c $R/protected_expected.sha256 > $R/protected_${TAG}_check.txt 2>&1; echo "expected-160 exit $?" > $R/protected_${TAG}_summary.txt
ls var/runs/levin-phase-floor-07b-census-20261117T010200 | wc -l | sed 's/^/07b census run dir files on disk: /' >> $R/protected_${TAG}_summary.txt
{ ls var/datastores/levin-phase-floor-07-r29* var/datastores/levin-phase-floor-07a-r30*;
  find var/runs/levin-phase-floor-07a-census-* -type f;
  find var/levin-phase-floor-07c/measurement -type f;
  find var/levin-phase-floor-07b/census/out/stop3 -type f -name 'compare_census_calls.calls.jsonl';
  find var/datastores/handover-A3-v2-lambdacdm* -type f 2>/dev/null; } | LC_ALL=C sort | while IFS= read -r f; do shasum -a 256 "$f"; done > $R/extra_${TAG}.sha256
wc -l < $R/extra_${TAG}.sha256 | sed 's/^/extra files hashed: /' >> $R/protected_${TAG}_summary.txt
find var/datastores var/levin-phase-floor-06/wider/state/a3v2_copy -name '*-wal' -o -name '*-shm' | sed 's/^/WAL-or-SHM: /' >> $R/protected_${TAG}_summary.txt
echo done >> $R/protected_${TAG}_summary.txt
