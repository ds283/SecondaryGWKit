#!/bin/bash
# M2: main.py --inventory and --inventory --inventory-verbose, from both trees, against one copy of
# one build_full_store primary (built once, with the base tree's fixtures). Store files are
# checksummed before and after every run. usage: m2_report.sh <S>
set -u
S=$1
OUT=$S/m08/out/m2
rm -rf $OUT; mkdir -p $OUT/store
cd $S/m08/base
PYTHONPATH=. ./venv/bin/python -c "
import sys
from pathlib import Path
from Datastore.tests.real_store_fixtures import build_full_store
store = build_full_store(Path(sys.argv[1]))
print(store.primary)
" $OUT/store > $OUT/primary.txt 2>/dev/null
PRIMARY=$(tail -1 $OUT/primary.txt)
echo "primary: $PRIMARY"
sums() { (cd $OUT/store && find . -type f | sort | xargs shasum -a 256); }
sums > $OUT/sums-before.txt
for tree in base wt08; do
  if [ $tree = base ]; then T=$S/m08/base; else T=$S/wt08; fi
  cd $T
  for mode in plain verbose; do
    if [ $mode = plain ]; then extra=""; else extra="--inventory-verbose"; fi
    PYTHONPATH=. ./venv/bin/python main.py --database "$PRIMARY" --inventory $extra \
      > $OUT/$tree-$mode.stdout 2> $OUT/$tree-$mode.stderr
    echo $? > $OUT/$tree-$mode.exit
    sums > $OUT/sums-after-$tree-$mode.txt
  done
done
cd $OUT
for mode in plain verbose; do
  for f in stdout stderr exit; do
    if cmp -s base-$mode.$f wt08-$mode.$f; then r=identical; else r=DIFFER; fi
    echo "$mode $f: $r ($(wc -c < base-$mode.$f) / $(wc -c < wt08-$mode.$f) bytes; sha256 $(shasum -a 256 < base-$mode.$f | cut -c1-16))"
  done
done
for f in sums-after-*.txt; do
  if cmp -s sums-before.txt $f; then echo "$f: store files unchanged"; else echo "$f: STORE FILES CHANGED"; fi
done
echo "exit codes: $(cat base-plain.exit) $(cat base-verbose.exit) $(cat wt08-plain.exit) $(cat wt08-verbose.exit)"
