#!/bin/bash
# levin-phase-floor 05 §6 (wider): run one command and record it with its output.
# Usage: rec.sh NAME CMD [ARGS...]
# Writes var/levin-phase-floor-05/wider/out/cmd/NAME.txt: the command line, cwd, host, start and
# end UTC times, the combined stdout+stderr, and the exit code. Also echoes the output.
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
cat "$F"
exit $RC
