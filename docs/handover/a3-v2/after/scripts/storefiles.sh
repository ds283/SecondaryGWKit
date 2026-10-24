#!/bin/bash
# SHA-256, size and mtime of the A3 v2 primary and its four shards, and any journal beside them.
cd "$(dirname "$0")/../../.."
for f in var/datastores/handover-A3-v2-lambdacdm.sqlite var/datastores/handover-A3-v2-lambdacdm-shard000{0,1,2,3}.sqlite; do
  printf '%s  %s\n' "$(shasum -a 256 "$f" | cut -c1-64)" "$(/usr/bin/stat -f '%z %m %Sm %N' -t '%Y-%m-%dT%H:%M:%S' "$f")"
done
ls var/datastores | grep -E 'A3-v2.*(-journal|-wal|-shm)$' | sed 's/^/JOURNAL /'
true
