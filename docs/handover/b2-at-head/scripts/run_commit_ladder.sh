#!/bin/sh
# The commit ladder: B2's R cells together 1e8, together 1e7 and T-first 1e7 on each code tree from
# B2's (d88faec) to levin-phase-floor 07c's (67b5af74), one cell at a time, to find which commit
# raised R|together|1e8's cost from 286 regions to 20,131.
#
# Usage: run_commit_ladder.sh TREES_DIR OUT_DIR
#
# Each tree is `git archive <sha>` of everything but docs/ and prompts/, with HEAD's harness added:
# docs/handover/realistic_large_x.py, docs/handover/b2/scripts/b2_cells.py and the three
# docs/radiation-oracle/ modules realistic_large_x.py imports, none changed since d88faec (b2_cells.py
# finds its repository root from its own location, so it imports the tree's code). No worktree is
# made and the repository is not changed. 89c368f2 (docstrings only) is skipped. Before its cells,
# each tree's import check writes which AdaptiveLevin and QuadSourceIntegral files were imported
# and their SHA-256 against the commit's blobs.
set -u
REPO=/Users/ds283/Documents/Code/SecondaryGWKit
PY=$REPO/venv/bin/python
mkdir -p "$1" "$2"
# absolute, since each cell runs from inside its tree
TREES=$(cd "$1" && pwd)
OUT=$(cd "$2" && pwd)
cd "$REPO" || exit 2
git rev-parse HEAD > "$OUT/launch_head.txt"
HARNESS="docs/handover/realistic_large_x.py docs/handover/b2/scripts/b2_cells.py docs/radiation-oracle/eq22_rounding.py docs/radiation-oracle/kt_verification.py docs/radiation-oracle/large_x.py"
for sha in d88faec8 4ddc3c2c ad9a5f48 3d70db93 f4d8e173 7172a8f5 3ae6f3a4 48cf87c0 ad732887 adffa21b 1ec734d7 5e19c4b6 67b5af74; do
  T=$TREES/$sha
  O=$OUT/$sha
  mkdir -p "$O"
  if [ ! -f "$T/.extracted" ]; then
    rm -rf "$T"; mkdir -p "$T/docs/handover/b2/scripts" "$T/docs/radiation-oracle"
    git archive "$sha" -- . ':!docs' ':!prompts' | tar -x -C "$T"
    for f in $HARNESS; do git show "HEAD:$f" > "$T/$f"; done
    touch "$T/.extracted"
  fi
  (
    cd "$T" && env -u PYTHONPATH "$PY" -B -c "
import hashlib, importlib, subprocess, sys
sys.path.insert(0, '.')
L = importlib.import_module('AdaptiveLevin.levin_quadrature')
Q = importlib.import_module('ComputeTargets.QuadSourceIntegral')
for m, p in ((L, 'AdaptiveLevin/levin_quadrature.py'), (Q, 'ComputeTargets/QuadSourceIntegral.py')):
    blob = subprocess.run(['git', '-C', '$REPO', 'show', '$sha:' + p], capture_output=True).stdout
    here = open(m.__file__, 'rb').read()
    print(p, m.__file__, 'matches $sha' if here == blob else 'DIFFERS FROM $sha', hashlib.sha256(here).hexdigest())
"
  ) > "$O/import_check.txt" 2>&1
  for cell in "together 1e+08" "together 1e+07" "T-first 1e+07"; do
    set -- $cell
    ( cd "$T" && env -u PYTHONPATH timeout 2400 "$PY" -B docs/handover/b2/scripts/b2_cells.py --cell "$1" "$2" R --out "$O/R-$1-$2.json" ) > "$O/R-$1-$2.log" 2>&1
    echo "$sha R $1 $2 rc=$? $(date -u +%FT%TZ)" >> "$OUT/done.txt"
  done
done
echo "ALLDONE $(date -u +%FT%TZ)" >> "$OUT/done.txt"
