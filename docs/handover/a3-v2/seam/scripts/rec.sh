#!/bin/bash
# usage: rec.sh OUTFILE 'command string'  -- appends header, command, output, exit status
out="$1"; shift
cd /Users/ds283/Documents/Code/SecondaryGWKit
{
  echo "### $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname -s)"
  echo "\$ $*"
} >> "$out"
bash -c "$*" >> "$out" 2>&1
rc=$?
echo "[exit $rc]" >> "$out"
echo >> "$out"
exit $rc
