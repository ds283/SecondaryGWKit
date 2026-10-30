#!/bin/bash
# depth20-tiling (levin-phase-floor 06c follow-up, Mac Studio): run one command and record it with
# its output. Copied from the census's scripts/rec.sh (var/levin-phase-floor-06c/census/scripts/
# rec.sh, a record, read only); changed here: the output goes to out/NAME.txt (numbered by NAME),
# and the header also records the tree g's HEAD and `git status --short` (read-only git).
# Usage: rec.sh NAME CMD [ARGS...]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/../out"
TREE=/Users/ds283/Documents/Code/SecondaryGWKit-lpf06c-e5764a6
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
  echo "# tree g $TREE HEAD $(git --no-optional-locks -C $TREE rev-parse HEAD)"
  echo "# tree g status --short: [$(git --no-optional-locks -C $TREE status --short)]"
  echo "# PYTHONPATH ${PYTHONPATH:-}"
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
  echo "=================== $NAME (out/$NAME.txt)"
  cat "$F"
} >> "$OUT/commands.log"
cat "$F"
exit $RC
