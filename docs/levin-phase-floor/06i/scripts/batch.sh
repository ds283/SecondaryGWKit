#!/bin/sh
# levin-phase-floor 06i: run readings one after another, appending to ../out/run_log.txt.
# Usage: sh batch.sh JOBFILE   (one "script.py args..." per line)
HERE=$(cd "$(dirname "$0")" && pwd)
LOG="$HERE/../out/run_log.txt"
while IFS= read -r job; do
  [ -z "$job" ] && continue
  echo "$(date -u +%FT%TZ) START $job" >> "$LOG"
  sh "$HERE/run.sh" $job >> "$LOG" 2>&1 < /dev/null
  echo "$(date -u +%FT%TZ) END($?) $job" >> "$LOG"
done < "$1"
