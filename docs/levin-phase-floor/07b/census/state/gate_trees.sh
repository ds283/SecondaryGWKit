#!/bin/bash
# levin-phase-floor 07b section 6 (census agent): the trees, read only (git --no-optional-locks), and the modules' hashes; Gate U0 (1)'s diffs.
set -u
H=/Users/ds283/Documents/Code/SecondaryGWKit-lpf07b-0a103b5
M=/Users/ds283/Documents/Code/SecondaryGWKit
G="git --no-optional-locks"
for t in "$M" "$H"; do
  echo "== $t"; $G -C "$t" rev-parse HEAD
  echo "status --short --ignored: [$($G -C "$t" status --short --ignored | tr '\n' ';')]"
  echo "status --short: [$($G -C "$t" status --short | tr '\n' ';')]"
done
echo "== worktree list"; $G -C "$M" worktree list
echo "== git log --oneline -9 at h"; $G -C "$H" log --oneline -9
echo "== parents: 0a103b5a^ 5e19c4b6^ 7b82a0cd^"; $G -C "$H" rev-parse 0a103b5a^ 5e19c4b6^ 7b82a0cd^
echo "== U0 (1), S0': git diff --stat 1ec734d7 7b82a0cd -- . ':!docs' ':!prompts'"; $G -C "$H" diff --stat 1ec734d7 7b82a0cd -- . ':!docs' ':!prompts'
echo "== the same, --name-status"; $G -C "$H" diff --name-status 1ec734d7 7b82a0cd -- . ':!docs' ':!prompts'
echo "== git diff --stat 7b82a0cd 5e19c4b6 -- . ':!docs' ':!prompts' (the code commit)"; $G -C "$H" diff --stat 7b82a0cd 5e19c4b6 -- . ':!docs' ':!prompts'
echo "== git diff --stat 5e19c4b6 0a103b5a -- . ':!docs' ':!prompts' (expected empty)"; $G -C "$H" diff --stat 5e19c4b6 0a103b5a -- . ':!docs' ':!prompts'; echo "(end)"
echo "== git diff --stat 1ec734d7 0a103b5a -- ComputeTargets ObjectFactories RunDrivers main.py tools config CosmologyModels Quadrature requirements.txt StoreFixtures LiouvilleGreen"
$G -C "$H" diff --stat 1ec734d7 0a103b5a -- ComputeTargets ObjectFactories RunDrivers main.py tools config CosmologyModels Quadrature requirements.txt StoreFixtures LiouvilleGreen; echo "(end)"
echo "== module hashes (file in h; git show at each commit)"
shasum -a 256 "$H/AdaptiveLevin/levin_quadrature.py" "$H/ComputeTargets/QuadSourceIntegral.py" "$H/LiouvilleGreen/three_bessel_integrals.py" "$H/LiouvilleGreen/bessel_phase.py" "$H/AdaptiveLevin/__init__.py"
for c in 1ec734d7 7b82a0cd 5e19c4b6 0a103b5a; do echo "levin_quadrature.py at $c: $($G -C "$H" show $c:AdaptiveLevin/levin_quadrature.py | shasum -a 256)  blob $($G -C "$H" rev-parse $c:AdaptiveLevin/levin_quadrature.py)"; done
for c in 1ec734d7 7b82a0cd 5e19c4b6; do echo "AdaptiveLevin/__init__.py at $c: $($G -C "$H" show $c:AdaptiveLevin/__init__.py | shasum -a 256)"; done
for c in 1ec734d7 7b82a0cd 5e19c4b6; do echo "QuadSourceIntegral.py at $c: $($G -C "$H" show $c:ComputeTargets/QuadSourceIntegral.py | shasum -a 256)"; done
for f in LiouvilleGreen/three_bessel_integrals.py LiouvilleGreen/bessel_phase.py; do for c in 1ec734d7 7b82a0cd 5e19c4b6; do echo "$f at $c: $($G -C "$H" show $c:$f | shasum -a 256)"; done; done
echo "== __pycache__ in h (expected none)"; find "$H" -name __pycache__ -not -path '*/.git/*' | head; echo "(end)"
echo "== python"; "$M/venv/bin/python" -B -c "import sys; print(sys.version)"
echo "== load"; uptime
