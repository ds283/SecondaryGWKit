#!/bin/bash
# usage: storefiles.sh  -- sha256, size, mtime of the store files, the sidecar and stdout.log
cd /Users/ds283/Documents/Code/SecondaryGWKit
for f in var/datastores/handover-A3-v2-lambdacdm*; do
  echo "$(shasum -a 256 "$f" | cut -d' ' -f1)  $(/usr/bin/stat -f '%z bytes  mtime %Sm' -t '%Y-%m-%dT%H:%M:%S%z' "$f")  $f"
done
f=var/runs/handover-07-a3-v2-20261022T211155/stdout.log
echo "$(shasum -a 256 "$f" | cut -d' ' -f1)  $(/usr/bin/stat -f '%z bytes  mtime %Sm' -t '%Y-%m-%dT%H:%M:%S%z' "$f")  $f"
