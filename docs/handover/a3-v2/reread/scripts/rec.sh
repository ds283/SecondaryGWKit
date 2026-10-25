#!/bin/bash
# Usage: bash var/handover-10/scripts/rec.sh OUTFILE -- command args...
# Runs the command from the repository root, appending "$ command", its stdout+stderr,
# and its exit status to OUTFILE. Prints the same to the terminal afterwards.
set -u
out="$1"; shift
[ "$1" = "--" ] && shift
cd /Users/ds283/Documents/Code/SecondaryGWKit
{
  echo "### $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname -s)"
  printf '$'; printf ' %q' "$@"; echo
} >> "$out"
"$@" >> "$out" 2>&1
rc=$?
echo "[exit $rc]" >> "$out"
echo >> "$out"
exit $rc
