#!/bin/bash
# usage: storefiles.sh  -- sha256, size, mtime of A3 v2's store files and sidecar. Read-only.
cd /Users/ds283/Documents/Code/SecondaryGWKit
for f in var/datastores/handover-A3-v2-lambdacdm*; do
  echo "$(shasum -a 256 "$f" | cut -d' ' -f1)  $(/usr/bin/stat -f '%z bytes  mtime %Sm' -t '%Y-%m-%dT%H:%M:%S%z' "$f")  $f"
done
