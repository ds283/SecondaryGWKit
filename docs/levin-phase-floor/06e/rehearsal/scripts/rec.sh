#!/bin/sh
# rec.sh NAME COMMAND...  -- run COMMAND through sh -c, recording the command
# text, its start/end UTC time, its output (stdout+stderr) and its exit code in
# out/NAME.txt.  Used for read-only checks; never for lock commands or launches.
R=/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06e/rehearsal
name="$1"; shift
f="$R/out/$name.txt"
{
  echo "# host: $(hostname)"
  echo "# start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# cwd: $(pwd)"
  echo "# command:"
  printf '%s\n' "$*"
  echo "# output:"
} > "$f"
sh -c "$*" >> "$f" 2>&1
rc=$?
{
  echo "# exit: $rc"
  echo "# end: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "$f"
cat "$f"
exit $rc
