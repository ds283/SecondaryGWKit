#!/bin/bash
# levin-phase-floor 06 §6 (wider): run one command and record it with its output.
# Copied from 05a's scripts/rec.sh (docs/levin-phase-floor/05a/wider/scripts/rec.sh), changed
# (prompt 06) only in these comments' prompt and paths. 05a's note: copied from 05's
# scripts/rec.sh; changed (prompt 05a): its comments' paths, and every record is also appended to
# out/commands.log.
# Usage: rec.sh NAME CMD [ARGS...]
# Writes var/levin-phase-floor-06/wider/out/cmd/NAME.txt: the command line, cwd, host, start and
# end UTC times, the combined stdout+stderr, and the exit code. Appends the same to
# out/commands.log. Also echoes the output.
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
