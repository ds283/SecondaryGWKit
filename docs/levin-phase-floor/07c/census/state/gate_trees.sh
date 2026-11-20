#!/bin/bash
# levin-phase-floor 07c section 6 (census agent): the trees, read only (git --no-optional-locks), and the modules' hashes; Gate Y0 (1)'s diffs.
set -u
H=/Users/ds283/Documents/Code/SecondaryGWKit-lpf07c-cc3368c
M=/Users/ds283/Documents/Code/SecondaryGWKit
G="git --no-optional-locks"
for t in "$M" "$H"; do
  echo "== $t"; $G -C "$t" rev-parse HEAD
  echo "status --short --ignored: [$($G -C "$t" status --short --ignored | tr '\n' ';')]"
  echo "status --short: [$($G -C "$t" status --short | tr '\n' ';')]"
done
echo "== worktree list"; $G -C "$M" worktree list
echo "== git log --oneline -9 at the tree"; $G -C "$H" log --oneline -9
echo "== parents: cc3368ce^ 67b5af74^ cff4237f^"; $G -C "$H" rev-parse cc3368ce^ 67b5af74^ cff4237f^
echo "== Y0 (1), S0': git diff --stat 5e19c4b6 cff4237f -- . ':!docs' ':!prompts'"; $G -C "$H" diff --stat 5e19c4b6 cff4237f -- . ':!docs' ':!prompts'; echo "(end)"
echo "== the same, --name-status"; $G -C "$H" diff --name-status 5e19c4b6 cff4237f -- . ':!docs' ':!prompts'; echo "(end)"
echo "== git diff --stat cff4237f 67b5af74 -- . ':!docs' ':!prompts' (the code commit)"; $G -C "$H" diff --stat cff4237f 67b5af74 -- . ':!docs' ':!prompts'
echo "== git diff --stat 67b5af74 cc3368ce -- . ':!docs' ':!prompts' (expected empty)"; $G -C "$H" diff --stat 67b5af74 cc3368ce -- . ':!docs' ':!prompts'; echo "(end)"
echo "== git diff --stat 5e19c4b6 cc3368ce -- ComputeTargets ObjectFactories RunDrivers main.py tools config CosmologyModels Quadrature requirements.txt StoreFixtures LiouvilleGreen"
$G -C "$H" diff --stat 5e19c4b6 cc3368ce -- ComputeTargets ObjectFactories RunDrivers main.py tools config CosmologyModels Quadrature requirements.txt StoreFixtures LiouvilleGreen; echo "(end)"
echo "== 07b's census tree (0a103b5a) against the parent cff4237f, outside docs and prompts"; $G -C "$H" diff --stat 0a103b5a cff4237f -- . ':!docs' ':!prompts'; echo "(end)"
echo "== module hashes (file in the tree; git show at each commit)"
shasum -a 256 "$H/AdaptiveLevin/levin_quadrature.py" "$H/ComputeTargets/QuadSourceIntegral.py" "$H/LiouvilleGreen/three_bessel_integrals.py" "$H/LiouvilleGreen/bessel_phase.py" "$H/AdaptiveLevin/__init__.py"
for c in 5e19c4b6 0a103b5a cff4237f 67b5af74 cc3368ce; do echo "levin_quadrature.py at $c: $($G -C "$H" show $c:AdaptiveLevin/levin_quadrature.py | shasum -a 256)  blob $($G -C "$H" rev-parse $c:AdaptiveLevin/levin_quadrature.py)"; done
for c in 5e19c4b6 cff4237f 67b5af74; do echo "AdaptiveLevin/__init__.py at $c: $($G -C "$H" show $c:AdaptiveLevin/__init__.py | shasum -a 256)"; done
for c in 5e19c4b6 cff4237f 67b5af74; do echo "QuadSourceIntegral.py at $c: $($G -C "$H" show $c:ComputeTargets/QuadSourceIntegral.py | shasum -a 256)"; done
for f in LiouvilleGreen/three_bessel_integrals.py LiouvilleGreen/bessel_phase.py; do for c in 5e19c4b6 cff4237f 67b5af74; do echo "$f at $c: $($G -C "$H" show $c:$f | shasum -a 256)"; done; done
echo "== __pycache__ in the tree (expected none)"; find "$H" -name __pycache__ -not -path '*/.git/*' | head; echo "(end)"
echo "== python"; "$M/venv/bin/python" -B -c "import sys; print(sys.version)"
echo "== load"; uptime
