#!/bin/bash
# levin-phase-floor 06a: compare the data lines (sha256 size mtime_ns mtime_utc name) of two
# hash_store.py outputs; prints both and "EQUAL" or "DIFFER". New in 06a.
set -u
a=$(grep -E '^[0-9a-f]{64} ' "$1")
b=$(grep -E '^[0-9a-f]{64} ' "$2")
echo "== $1"; echo "$a"; echo "== $2"; echo "$b"
if [ "$a" == "$b" ]; then echo "EQUAL"; exit 0; else echo "DIFFER"; diff <(echo "$a") <(echo "$b"); exit 1; fi
