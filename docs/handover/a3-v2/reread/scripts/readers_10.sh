#!/bin/bash
# Prompt 10 §1 step 5: the six readers, then extract_Gk_data with nomodels, on the A3 v2 primary,
# as 08 ran them (docs/handover/a3-v2/after/out/readers_08.txt). Output: out/readers_10.txt.
# run_readers.sh itself records each reader's exit, wall time, file count and the store files'
# state before and after; this wrapper adds the load, each invocation's rc, and du -sk per reader.
set -u
cd /Users/ds283/Documents/Code/SecondaryGWKit
DB=var/datastores/handover-A3-v2-lambdacdm.sqlite
EXT=var/handover-10/extract
{
echo "== $(date '+%Y-%m-%dT%H:%M:%S%z') readers; HEAD $(git rev-parse HEAD); load: $(uptime)"
echo "\$ bash docs/dk-and-rpk-migration/rehearsal/run_readers.sh $DB $EXT"
bash docs/dk-and-rpk-migration/rehearsal/run_readers.sh "$DB" "$EXT"
echo "rc=$?"
echo "\$ bash docs/dk-and-rpk-migration/rehearsal/run_readers.sh $DB $EXT nomodels extract_Gk_data"
bash docs/dk-and-rpk-migration/rehearsal/run_readers.sh "$DB" "$EXT" nomodels extract_Gk_data
echo "rc=$?"
echo "== $(date '+%Y-%m-%dT%H:%M:%S%z') readers done; load: $(uptime)"
echo "-- output size per reader (du -sk), and file count"
for d in "$EXT"/*/; do printf '%s\t%s\t%s files\n' "$(du -sk "$d" | cut -f1)" "$d" "$(find "$d" -type f | wc -l | tr -d ' ')"; done
du -sh "$EXT"
} > var/handover-10/out/readers_10.txt 2>&1
