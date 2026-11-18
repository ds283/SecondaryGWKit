#!/bin/bash
# levin-phase-floor 07b section 6, Gate U0 (the census agent): the parent's AdaptiveLevin/levin_quadrature.py, taken by
# git show from 7b82a0cd into state/variants/ (never into the worktree), and checked: its SHA-256 equals the blob's
# content at 7b82a0cd and at 1ec734d7 (07a's code commit) and 4bf50d34 (07a's census tree), and 07a's census workers'
# module 4ea1948b...; its git blob id (hash-object, no -w: nothing written) equals the tree entries'.
set -u
H=/Users/ds283/Documents/Code/SecondaryGWKit-lpf07b-0a103b5
G="git --no-optional-locks -C $H"
F=state/variants/levin_quadrature_parent.py
if [ -e "$F" ]; then echo "$F exists; not overwritten"; exit 1; fi
$G show 7b82a0cd:AdaptiveLevin/levin_quadrature.py > "$F"
chmod a-w "$F"
echo "== $F"; ls -la "$F"; shasum -a 256 "$F"
echo "== git show 7b82a0cd / 1ec734d7 / 4bf50d34, sha256"
for c in 7b82a0cd 1ec734d7 4bf50d34; do echo "$c $($G show $c:AdaptiveLevin/levin_quadrature.py | shasum -a 256)"; done
echo "== blob ids: the tree entries, and the file's (git hash-object, no -w)"
for c in 7b82a0cd 1ec734d7 4bf50d34; do echo "$c $($G rev-parse $c:AdaptiveLevin/levin_quadrature.py)"; done
echo "file $(git --no-optional-locks hash-object --no-filters "$F")"
echo "== 07a's census workers' module (07a's census log: 4ea1948b...)"; grep -o '"levin_quadrature_sha256": "[0-9a-f]*"' /Users/ds283/Documents/Code/SecondaryGWKit/var/runs/levin-phase-floor-07a-census-20261112T132744/worker_1e-7-k0_101_hello.json
echo "== cmp with git show 1ec734d7 (byte-identical: no output, rc 0)"; $G show 1ec734d7:AdaptiveLevin/levin_quadrature.py | cmp - "$F"; echo "rc $?"
echo "== cmp with h's file (expected to differ)"; cmp "$H/AdaptiveLevin/levin_quadrature.py" "$F" | head -1; echo "rc ${PIPESTATUS[0]}"
