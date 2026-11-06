#!/bin/bash
# levin-phase-floor 06m census agent: the Gate's tree checks, read-only git only.
M=/Users/ds283/Documents/Code/SecondaryGWKit
T=/Users/ds283/Documents/Code/SecondaryGWKit-lpf06m-38f6a82
P=/Users/ds283/Documents/Code/SecondaryGWKit-lpf06m-5ff8a5e
for d in $M $T $P; do
  echo "== $d"
  git --no-optional-locks -C $d rev-parse HEAD
  echo "status --short --ignored: [$(git --no-optional-locks -C $d status --short --ignored | tr '\n' ';')]"
  echo "status --short: [$(git --no-optional-locks -C $d status --short | tr '\n' ';')]"
done
echo "== worktree list"; git --no-optional-locks -C $M worktree list
echo "== module hashes"
shasum -a 256 $T/AdaptiveLevin/levin_quadrature.py $P/AdaptiveLevin/levin_quadrature.py $T/RunDrivers/quadsource_atol_sweep.py
echo "== git diff --stat 5ff8a5e4 38f6a828 -- . ':!docs' ':!prompts'"
git --no-optional-locks -C $T diff --stat 5ff8a5e44be775df299af0f1754123578905b86e 38f6a828d17b41893574f7344a543ebfb1e01e85 -- . ':!docs' ':!prompts'
echo "== git diff --stat 48cf87c0 38f6a828 -- . ':!docs' ':!prompts'"
git --no-optional-locks -C $T diff --stat 48cf87c0 38f6a828d17b41893574f7344a543ebfb1e01e85 -- . ':!docs' ':!prompts'
echo "== git diff --stat 48cf87c0 38f6a828 -- AdaptiveLevin/levin_quadrature.py (empty if the module is the code commit's)"
git --no-optional-locks -C $T diff --stat 48cf87c0 38f6a828d17b41893574f7344a543ebfb1e01e85 -- AdaptiveLevin/levin_quadrature.py
echo "== git log --oneline -6 at T"; git --no-optional-locks -C $T log --oneline -6
echo "== module at 38f6a828 by git show"; git --no-optional-locks -C $T show 38f6a828d17b41893574f7344a543ebfb1e01e85:AdaptiveLevin/levin_quadrature.py | shasum -a 256
echo "== 06l's committed scripts dir (the copy's source) at T, by git: tracked file count"
git --no-optional-locks -C $T ls-files docs/levin-phase-floor/06l/scripts | wc -l
echo "== __pycache__ in T or P"; find $T $P -name __pycache__ -not -path '*/.git/*' | head
