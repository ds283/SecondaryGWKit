#!/bin/bash
# Prompt 07 §2.3: readers on the resumed store, read-only. Before and after each: the SHA-256, the
# mtime and the size of the primary and every shard (BSD /usr/bin/stat: a GNU stat earlier on PATH
# reads `-f` as "file system", which the first pass of this script fell into), and any -journal,
# -wal or -shm file beside them. Each reader's output goes to
# var/a3-v2-readiness-07/extract/<label>.log, its files to var/a3-v2-readiness-07/extract/<label>/.
# Run from anywhere:
#     bash docs/a3-v2-readiness/rehearsal/run_readers.sh six          # the six, --models LambdaCDM
#     bash docs/a3-v2-readiness/rehearsal/run_readers.sh nomodels <script>   # one, without --models
#     bash docs/a3-v2-readiness/rehearsal/run_readers.sh sorted       # the sorted-sample wrapper
set -u
cd "$(dirname "$0")/../../.."
DB=var/datastores/a3-v2-readiness-07-resumed.sqlite
OUT=var/a3-v2-readiness-07/extract
MODE="${1:?mode: six | nomodels <script> | sorted}"
mkdir -p "$OUT"

state() {  # the hash, mtime and size of each file of the store, and any journal beside it
  for f in var/datastores/a3-v2-readiness-07-resumed.sqlite var/datastores/a3-v2-readiness-07-resumed-shard000{0,1,2,3}.sqlite; do
    printf '%s  %s\n' "$(shasum -a 256 "$f" | cut -d' ' -f1)" "$(/usr/bin/stat -f '%Sm %z %N' -t '%Y-%m-%dT%H:%M:%S' "$f")"
  done
  ls var/datastores | grep -E '^a3-v2-readiness-07-resumed.*-(journal|wal|shm)$' | sed 's/^/JOURNAL /'
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
  echo "exit $code; wall $((end - start)) s"
  if [ "$before" == "$after" ]; then
    echo "store files: identical before and after (SHA-256, mtime, size); no journal"
  else
    echo "STORE FILES CHANGED"; echo "before:"; echo "$before"
  fi
  echo "after:"; echo "$after" | sed 's/^/  /'
  echo "Ray processes from this checkout's venv after: $(ps -axo pid=,command= | grep 'SecondaryGWKit/venv' | grep ray | grep -v grep | wc -l | tr -d ' ')"
  grep -nE 'ReadOnlyMiss|ReadOnlyWrite|Traceback|SKIPPING|reading run|RUNNING PIPELINE|NOT THE SHIPPED|Queue summary' "$OUT/$label.log" | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-220
}

case "$MODE" in
  six)
    for s in extract_Gk_data extract_GkWKB_data extract_GkSource_data extract_TkWKB_data extract_tensor_source_data extract_QuadSourceIntegral_data; do
      run_one "$s" "$s.py" --models LambdaCDM
    done ;;
  nomodels)
    run_one "${2:?script}-no-models" "${2}.py" ;;
  sorted)
    run_one "extract_tensor_source_data-sorted-sample" docs/a3-v2-readiness/rehearsal/tensor_source_sorted_sample.py --models LambdaCDM ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
