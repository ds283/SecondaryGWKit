#!/bin/bash
# The readers on a closed store, read-only (dk-and-rpk-migration README §4.1, L3; a3-v2-readiness
# prompt 07 §2.3). Before and after each reader: the SHA-256, the mtime and the size of the primary
# and of every shard, and any -journal, -wal or -shm file beside them. The shards are read from the
# primary's own `shards` table (sqlite3, mode=ro), never assumed. mtime and size come from BSD
# /usr/bin/stat: a GNU stat earlier on PATH reads `-f` as "file system" (07, deviation 4).
#
# Usage, from anywhere (paths relative to the repository root, or absolute):
#     bash docs/dk-and-rpk-migration/rehearsal/run_readers.sh <primary> <output dir>
#         the six extract_*.py readers, each with --ray-address local --models LambdaCDM
#     bash docs/dk-and-rpk-migration/rehearsal/run_readers.sh <primary> <output dir> nomodels <script>
#         one reader, <script> without its .py (e.g. extract_Gk_data), with --ray-address local
#         and no --models
#
# Each reader's output goes to <output dir>/<label>.log and its files to <output dir>/<label>/,
# where <label> is the script's name (with -no-models appended in the second form). The summary
# for each reader is printed on stdout. Exit status: 0 if every reader exited 0 and left every file
# of the store as it was; 1 otherwise; 2 on a usage error.
set -u
cd "$(dirname "$0")/../../.."
DB="${1:?usage: run_readers.sh <primary> <output dir> [nomodels <script>]}"
OUT="${2:?usage: run_readers.sh <primary> <output dir> [nomodels <script>]}"
MODE="${3:-six}"
[ -f "$DB" ] || { echo "no primary at $DB" >&2; exit 2; }
DIR=$(dirname "$DB")
SHARDS=$(sqlite3 "file:$DB?mode=ro" "SELECT filename FROM shards ORDER BY serial") || { echo "cannot read the shards table of $DB" >&2; exit 2; }
FILES=("$DB")
for name in $SHARDS; do FILES+=("$DIR/$(basename "$name")"); done
mkdir -p "$OUT"
STATUS=0

state() {  # the hash, mtime and size of each file of the store, and any journal beside it
  local f s
  for f in "${FILES[@]}"; do
    printf '%s  %s\n' "$(shasum -a 256 "$f" | cut -d' ' -f1)" "$(/usr/bin/stat -f '%Sm %z %N' -t '%Y-%m-%dT%H:%M:%S' "$f")"
  done
  for f in "${FILES[@]}"; do
    for s in -journal -wal -shm; do [ -e "$f$s" ] && echo "JOURNAL $f$s"; done
  done
  return 0
}

run_one() {  # label program extra-args...
  local label=$1 program=$2; shift 2
  local before after start end code
  before=$(state)
  start=$(date +%s)
  echo "=== $label: $(date '+%Y-%m-%d %H:%M:%S %z'): PYTHONPATH=. ./venv/bin/python $program --database $DB --ray-address local $* --output $OUT/$label/"
  PYTHONPATH=. ./venv/bin/python "$program" --database "$DB" --ray-address local "$@" --output "$OUT/$label/" > "$OUT/$label.log" 2>&1
  code=$?
  end=$(date +%s)
  after=$(state)
  echo "exit $code; wall $((end - start)) s; files written $(find "$OUT/$label" -type f 2>/dev/null | wc -l | tr -d ' ')"
  [ "$code" -eq 0 ] || STATUS=1
  if [ "$before" == "$after" ]; then
    echo "store files: identical before and after (SHA-256, mtime, size); no journal"
  else
    STATUS=1
    echo "STORE FILES CHANGED"; echo "before:"; echo "$before" | sed 's/^/  /'
  fi
  echo "after:"; echo "$after" | sed 's/^/  /'
  echo "Ray processes from this checkout's venv after: $(ps -axo pid=,command= | grep 'SecondaryGWKit/venv' | grep ray | grep -v grep | wc -l | tr -d ' ')"
  grep -nE 'ReadOnlyMiss|ReadOnlyWrite|Traceback|SKIPPING|reading run|RUNNING PIPELINE|Queue summary' "$OUT/$label.log" | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-220
}

case "$MODE" in
  six)
    for s in extract_Gk_data extract_GkWKB_data extract_GkSource_data extract_TkWKB_data extract_tensor_source_data extract_QuadSourceIntegral_data; do
      run_one "$s" "$s.py" --models LambdaCDM
    done ;;
  nomodels)
    script="${4:?usage: run_readers.sh <primary> <output dir> nomodels <script>}"
    run_one "$script-no-models" "$script.py" ;;
  *) echo "unknown mode $MODE (six, the default, or nomodels <script>)" >&2; exit 2 ;;
esac
exit $STATUS
