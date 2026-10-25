#!/bin/bash
# SHA-256, size and mtime of A3 v2's primary, four shards and sidecar. Read-only.
#   bash var/levin-phase-floor-01/scripts/hash_store.sh > var/levin-phase-floor-01/state/store_files_{before,after}.txt
D=var/datastores
echo "# hash_store.sh, run $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname); script sha256 $(shasum -a 256 "$0" | cut -d' ' -f1)"
echo "# sha256  size_bytes  mtime(UTC, /usr/bin/stat -f %Sm)  path"
for f in "$D/handover-A3-v2-lambdacdm.sqlite" "$D"/handover-A3-v2-lambdacdm-shard000[0-3].sqlite "$D/handover-A3-v2-lambdacdm.manifest.json"; do
  h=$(shasum -a 256 "$f" | cut -d' ' -f1)
  s=$(/usr/bin/stat -f %z "$f")
  m=$(TZ=UTC /usr/bin/stat -f %Sm -t %Y-%m-%dT%H:%M:%SZ "$f")
  echo "$h  $s  $m  $f"
done
echo "# other files matching the store's name (wal/shm/journal would show here):"
ls -la "$D" | grep handover-A3-v2-lambdacdm
