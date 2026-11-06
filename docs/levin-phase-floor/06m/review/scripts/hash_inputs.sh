#!/bin/sh
# Orchestrator's hashes of 06m Part B's read-only inputs. Usage: sh hash_inputs.sh TAG (before|after)
M=/Users/ds283/Documents/Code/SecondaryGWKit
R=$M/var/levin-phase-floor-06m/review
T=$1
D=$R/hashes_$T
mkdir -p $D
cd $M || exit 1
PY="$M/venv/bin/python -I -B $R/scripts/hash_tree.py"
$PY $D/a3v2_copy.sha256 var/levin-phase-floor-06/wider/state/a3v2_copy
$PY $D/r23_r24_stores.sha256 var/datastores/levin-phase-floor-06g-r23.sqlite var/datastores/levin-phase-floor-06g-r23-shard0000.sqlite var/datastores/levin-phase-floor-06g-r23-shard0001.sqlite var/datastores/levin-phase-floor-06g-r23-shard0002.sqlite var/datastores/levin-phase-floor-06g-r23-shard0003.sqlite var/datastores/levin-phase-floor-06g-r24.sqlite var/datastores/levin-phase-floor-06g-r24-shard0000.sqlite var/datastores/levin-phase-floor-06g-r24-shard0001.sqlite var/datastores/levin-phase-floor-06g-r24-shard0002.sqlite var/datastores/levin-phase-floor-06g-r24-shard0003.sqlite
$PY $D/r23_r24_rundirs.sha256 var/runs/levin-phase-floor-06g-r23-parent-20261101T110641 var/runs/levin-phase-floor-06g-r24-leafvalues-20261101T111215 var/runs/levin-phase-floor-06g-r23.launch.log var/runs/levin-phase-floor-06g-r24.launch.log
$PY $D/census_routeB.sha256 var/runs/levin-phase-floor-06l-census-routeB-20261105T100126
$PY $D/census_06g.sha256 var/runs/levin-phase-floor-06g-census-20261101T114315
ls -la var/levin-phase-floor-06/wider/state/a3v2_copy/ var/datastores/ | grep -E 'journal|-wal|-shm' > $D/journals.txt
echo "journal/wal/shm lines: $(wc -l < $D/journals.txt)"
