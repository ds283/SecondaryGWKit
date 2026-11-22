#!/bin/bash
# levin-phase-floor 07c Part A: run one command and record it with its output.
# Copied from 07b Part A's scripts/ext/rec.sh (var/levin-phase-floor-07b/measurement/scripts/ext/rec.sh), changed
# (prompt 07c) only in these comments and its paths: records in var/levin-phase-floor-07c/measurement/out/cmd/ and
# out/commands.log.
# Usage: rec.sh NAME CMD [ARGS...]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/../out/cmd"
mkdir -p "$OUT"
NAME="$1"
shift
F="$OUT/$NAME.txt"
{
  printf '$'
  printf ' %q' "$@"
  printf '\n'
  echo "# cwd $(pwd)"
  echo "# host $(hostname)"
  echo "# start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# ----"
} > "$F"
"$@" >> "$F" 2>&1
RC=$?
{
  echo "# ----"
  echo "# end $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# exit $RC"
} >> "$F"
{
  echo "=================== $NAME (out/cmd/$NAME.txt)"
  cat "$F"
} >> "$HERE/../out/commands.log"
cat "$F"
exit $RC
