#!/bin/bash
# SHA-256, size, mtime (epoch and ISO) of the A3 v2 primary and of every shard the primary's own
# `shards` table names (sqlite3, mode=ro), and any -journal/-wal/-shm beside them. Then, on a line
# of its own marked SIDECAR, the same for the sidecar manifest (not a store file; recorded to show
# it too is unwritten). Format follows docs/handover/a3-v2/after/scripts/storefiles.sh (08).
# BSD /usr/bin/stat by full path.
set -u
cd /Users/ds283/Documents/Code/SecondaryGWKit
DB=var/datastores/handover-A3-v2-lambdacdm.sqlite
FILES=("$DB")
for name in $(sqlite3 "file:$DB?mode=ro" "SELECT filename FROM shards ORDER BY serial"); do
  FILES+=("var/datastores/$(basename "$name")")
done
for f in "${FILES[@]}"; do
  printf '%s  %s\n' "$(shasum -a 256 "$f" | cut -c1-64)" "$(/usr/bin/stat -f '%z %m %Sm %N' -t '%Y-%m-%dT%H:%M:%S' "$f")"
done
for f in "${FILES[@]}"; do
  for s in -journal -wal -shm; do [ -e "$f$s" ] && echo "JOURNAL $f$s"; done
done
f=var/datastores/handover-A3-v2-lambdacdm.manifest.json
printf 'SIDECAR %s  %s\n' "$(shasum -a 256 "$f" | cut -c1-64)" "$(/usr/bin/stat -f '%z %m %Sm %N' -t '%Y-%m-%dT%H:%M:%S' "$f")"
true
