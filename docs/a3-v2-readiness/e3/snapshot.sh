#!/bin/bash
# usage: snapshot.sh <label> ; prints SHA-256, mtime and size of 07's resumed store, and any journal
cd /Users/ds283/Documents/Code/SecondaryGWKit/var/datastores || exit 1
echo "## $1  $(date '+%Y-%m-%dT%H:%M:%S%z')"
for f in a3-v2-readiness-07-resumed.sqlite a3-v2-readiness-07-resumed-shard000{0,1,2,3}.sqlite; do
  echo "$(shasum -a 256 "$f" | cut -d' ' -f1)  $(/usr/bin/stat -f '%m %z' "$f")  $f"
done
echo "journals: $(ls a3-v2-readiness-07-resumed*-journal a3-v2-readiness-07-resumed*-wal a3-v2-readiness-07-resumed*-shm 2>/dev/null | tr '\n' ' ')"
