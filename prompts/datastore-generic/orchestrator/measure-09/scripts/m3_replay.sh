#!/bin/bash
# M3 step 2: replay each mutation diff on the rough tree with git apply, run the guard, revert.
# Usage: m3_replay.sh <worktree> <diff-dir> <out-file>
set -u
WT=$1; D=$2; OUT=$3
cd "$WT" || exit 1
: > "$OUT"
for f in "$D"/*.diff; do
  n=$(basename "$f" .diff)
  echo "=== $n" >> "$OUT"
  git apply --check "$f" || { echo "apply --check FAILED" >> "$OUT"; continue; }
  git apply "$f"
  git apply -R --check "$f" && echo "apply -R --check ok" >> "$OUT"
  PYTHONPATH=. ./venv/bin/python -m unittest Datastore.tests.test_layer_is_generic > "$D/$n.run.txt" 2>&1
  grep -E '^(Ran|OK|FAILED)|^(FAIL|ERROR):|^Datastore/|^RayTools/|^AssertionError|Items in the first set' "$D/$n.run.txt" >> "$OUT"
  git apply -R "$f"
  # a new file added by the diff is left untracked-and-absent after -R; the tree must be clean
  if [ -n "$(git diff)" ] || [ -n "$(git status --porcelain --untracked-files=all | grep '^??')" ]; then
    echo "TREE NOT CLEAN AFTER REVERT" >> "$OUT"
  else
    echo "tree clean after revert" >> "$OUT"
  fi
done
