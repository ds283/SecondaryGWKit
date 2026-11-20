#!/bin/bash
# levin-phase-floor 07c section 6, Gate Y0 (the census agent): the parent's AdaptiveLevin/levin_quadrature.py, taken by
# git show from cff4237f into state/variants/ (never into the worktree), and checked: its SHA-256 equals the blob's
# content at cff4237f and at 5e19c4b6 (07b's code commit) and 0a103b5a (07b's census tree), and 07b's census workers'
# module 8956c7b6...; its git blob id (hash-object, no -w: nothing written) equals the tree entries'.
set -u
H=/Users/ds283/Documents/Code/SecondaryGWKit-lpf07c-cc3368c
G="git --no-optional-locks -C $H"
F=state/variants/levin_quadrature_parent.py
if [ -e "$F" ]; then echo "$F exists; not overwritten"; exit 1; fi
$G show cff4237f:AdaptiveLevin/levin_quadrature.py > "$F"
chmod a-w "$F"
echo "== $F"; ls -la "$F"; shasum -a 256 "$F"
echo "== git show cff4237f / 5e19c4b6 / 0a103b5a, sha256"
for c in cff4237f 5e19c4b6 0a103b5a; do echo "$c $($G show $c:AdaptiveLevin/levin_quadrature.py | shasum -a 256)"; done
echo "== blob ids: the tree entries, and the file's (git hash-object, no -w)"
for c in cff4237f 5e19c4b6 0a103b5a; do echo "$c $($G rev-parse $c:AdaptiveLevin/levin_quadrature.py)"; done
echo "file $(git --no-optional-locks hash-object --no-filters "$F")"
echo "== 07b's census workers' module (07b's census log: 8956c7b6...)"; ls /Users/ds283/Documents/Code/SecondaryGWKit/var/runs/levin-phase-floor-07b-census-20261117T010200/ | grep hello | head -2; grep -o '"levin_quadrature_sha256": "[0-9a-f]*"' $(ls /Users/ds283/Documents/Code/SecondaryGWKit/var/runs/levin-phase-floor-07b-census-20261117T010200/*hello*.json | head -1)
echo "== cmp with git show 5e19c4b6 (byte-identical: no output, rc 0)"; $G show 5e19c4b6:AdaptiveLevin/levin_quadrature.py | cmp - "$F"; echo "rc $?"
echo "== cmp with the tree's file (expected to differ)"; cmp "$H/AdaptiveLevin/levin_quadrature.py" "$F" | head -1; echo "rc ${PIPESTATUS[0]}"
