#!/bin/bash
# levin-phase-floor 07a section 6, Gate W0 (the census agent): the parent's AdaptiveLevin/levin_quadrature.py, taken by
# git show from dadd63a0 into state/variants/ (never into the worktree), and checked: its SHA-256 equals the blob's
# content at dadd63a0 and at e182bc7a (07's census tree), its git blob id (hash-object, no -w: nothing written) equals
# the tree entries' at both commits.
set -u
H=/Users/ds283/Documents/Code/SecondaryGWKit-lpf07a-4bf50d3
G="git --no-optional-locks -C $H"
F=state/variants/levin_quadrature_parent.py
if [ -e "$F" ]; then echo "$F exists; not overwritten"; exit 1; fi
$G show dadd63a0:AdaptiveLevin/levin_quadrature.py > "$F"
chmod a-w "$F"
echo "== $F"; ls -la "$F"; shasum -a 256 "$F"
echo "== git show dadd63a0 / e182bc7a / 1ec734d7, sha256"
for c in dadd63a0 e182bc7a 1ec734d7; do echo "$c $($G show $c:AdaptiveLevin/levin_quadrature.py | shasum -a 256)"; done
echo "== blob ids: the tree entries, and the file's (git hash-object, no -w)"
for c in dadd63a0 e182bc7a 1ec734d7; do echo "$c $($G rev-parse $c:AdaptiveLevin/levin_quadrature.py)"; done
echo "file $(git --no-optional-locks hash-object --no-filters "$F")"
echo "== cmp with git show e182bc7a (byte-identical: no output, rc 0)"; $G show e182bc7a:AdaptiveLevin/levin_quadrature.py | cmp - "$F"; echo "rc $?"
echo "== cmp with h's file (expected to differ)"; cmp "$H/AdaptiveLevin/levin_quadrature.py" "$F" | head -1; echo "rc ${PIPESTATUS[0]}"
