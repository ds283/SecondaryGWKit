#!/bin/bash
# levin-phase-floor 07a section 6 (census agent): the trees, read only (git --no-optional-locks), and the modules' hashes.
set -u
H=/Users/ds283/Documents/Code/SecondaryGWKit-lpf07a-4bf50d3
M=/Users/ds283/Documents/Code/SecondaryGWKit
G="git --no-optional-locks"
for t in "$M" "$H"; do
  echo "== $t"; $G -C "$t" rev-parse HEAD
  echo "status --short --ignored: [$($G -C "$t" status --short --ignored | tr '\n' ';')]"
  echo "status --short: [$($G -C "$t" status --short | tr '\n' ';')]"
done
echo "== worktree list"; $G -C "$M" worktree list
echo "== git log --oneline -6 at h"; $G -C "$H" log --oneline -6
echo "== h's parents: 4bf50d34^ and 1ec734d7^"; $G -C "$H" rev-parse 4bf50d34^ 1ec734d7^
echo "== S0': git diff --stat e182bc7a dadd63a0 -- . ':!docs' ':!prompts'"; $G -C "$H" diff --stat e182bc7a dadd63a0 -- . ':!docs' ':!prompts'
echo "== git diff --stat dadd63a0 1ec734d7 -- . ':!docs' ':!prompts'"; $G -C "$H" diff --stat dadd63a0 1ec734d7 -- . ':!docs' ':!prompts'
echo "== git diff --stat 1ec734d7 4bf50d34 -- . ':!docs' ':!prompts' (expected empty)"; $G -C "$H" diff --stat 1ec734d7 4bf50d34 -- . ':!docs' ':!prompts'; echo "(end)"
echo "== git diff --stat e182bc7a 4bf50d34 -- ComputeTargets ObjectFactories RunDrivers main.py tools config CosmologyModels Quadrature requirements.txt StoreFixtures (expected empty)"
$G -C "$H" diff --stat e182bc7a 4bf50d34 -- ComputeTargets ObjectFactories RunDrivers main.py tools config CosmologyModels Quadrature requirements.txt StoreFixtures; echo "(end)"
echo "== module hashes (file in h; git show at each commit)"
shasum -a 256 "$H/AdaptiveLevin/levin_quadrature.py" "$H/ComputeTargets/QuadSourceIntegral.py" "$H/LiouvilleGreen/three_bessel_integrals.py" "$H/LiouvilleGreen/bessel_phase.py" "$H/AdaptiveLevin/__init__.py"
for c in e182bc7a dadd63a0 1ec734d7 4bf50d34; do echo "levin_quadrature.py at $c: $($G -C "$H" show $c:AdaptiveLevin/levin_quadrature.py | shasum -a 256)  blob $($G -C "$H" rev-parse $c:AdaptiveLevin/levin_quadrature.py)"; done
for c in e182bc7a dadd63a0 1ec734d7; do echo "AdaptiveLevin/__init__.py at $c: $($G -C "$H" show $c:AdaptiveLevin/__init__.py | shasum -a 256)"; done
for c in e182bc7a dadd63a0 1ec734d7; do echo "QuadSourceIntegral.py at $c: $($G -C "$H" show $c:ComputeTargets/QuadSourceIntegral.py | shasum -a 256)"; done
echo "== __pycache__ in h (expected none)"; find "$H" -name __pycache__ -not -path '*/.git/*' | head; echo "(end)"
echo "== python"; "$M/venv/bin/python" -B -c "import sys; print(sys.version)"
echo "== load"; uptime
