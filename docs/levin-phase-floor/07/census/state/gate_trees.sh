#!/bin/bash
# levin-phase-floor 07 census agent (Part B, prompt 07 section 6): the trees' checks, read-only git only.
# After 06m's census state/gate_trees.sh, rewritten for 07's three trees.
M=/Users/ds283/Documents/Code/SecondaryGWKit
G=/Users/ds283/Documents/Code/SecondaryGWKit-lpf07-e182bc7
P=/Users/ds283/Documents/Code/SecondaryGWKit-lpf07-91472db
O=/Users/ds283/Documents/Code/SecondaryGWKit-lpf07-7460035
for d in $M $G $P $O; do
  echo "== $d"
  git --no-optional-locks -C $d rev-parse HEAD
  echo "status --short --ignored: [$(git --no-optional-locks -C $d status --short --ignored | tr '\n' ';')]"
  echo "status --short: [$(git --no-optional-locks -C $d status --short | tr '\n' ';')]"
done
echo "== worktree list"; git --no-optional-locks -C $M worktree list
echo "== module hashes"
shasum -a 256 $G/AdaptiveLevin/levin_quadrature.py $P/AdaptiveLevin/levin_quadrature.py $O/AdaptiveLevin/levin_quadrature.py \
  $G/ComputeTargets/QuadSourceIntegral.py $P/ComputeTargets/QuadSourceIntegral.py $O/ComputeTargets/QuadSourceIntegral.py \
  $G/RunDrivers/quadsource_atol_sweep.py $P/RunDrivers/quadsource_atol_sweep.py
echo "== module at e182bc7a by git show"; git --no-optional-locks -C $G show e182bc7a827142cd4f42594f1a9d8cf85ccc3a5f:AdaptiveLevin/levin_quadrature.py | shasum -a 256
echo "== module at adffa21b by git show"; git --no-optional-locks -C $G show adffa21b:AdaptiveLevin/levin_quadrature.py | shasum -a 256
echo "== module at 91472dbe by git show"; git --no-optional-locks -C $P show 91472dbee1daba8374b0018db1195cf6c7898bbb:AdaptiveLevin/levin_quadrature.py | shasum -a 256
echo "== module at 48cf87c0 (06m's code commit) by git show"; git --no-optional-locks -C $P show 48cf87c0:AdaptiveLevin/levin_quadrature.py | shasum -a 256
echo "== git log --oneline -8 at G"; git --no-optional-locks -C $G log --oneline -8
echo "== git diff --stat 74600350 e182bc7a -- . ':!docs' ':!prompts'"
git --no-optional-locks -C $G diff --stat 746003508bff868fc1b256b29cbd5035609bed2b e182bc7a827142cd4f42594f1a9d8cf85ccc3a5f -- . ':!docs' ':!prompts'
echo "== git diff --stat 91472dbe e182bc7a -- . ':!docs' ':!prompts'"
git --no-optional-locks -C $G diff --stat 91472dbee1daba8374b0018db1195cf6c7898bbb e182bc7a827142cd4f42594f1a9d8cf85ccc3a5f -- . ':!docs' ':!prompts'
echo "== git diff --stat adffa21b e182bc7a -- . ':!docs' ':!prompts' (empty if the code is adffa21b's)"
git --no-optional-locks -C $G diff --stat adffa21b e182bc7a827142cd4f42594f1a9d8cf85ccc3a5f -- . ':!docs' ':!prompts'
echo "== 06m's census scripts at G, by git: tracked file count"
git --no-optional-locks -C $G ls-files docs/levin-phase-floor/06m/census/scripts | wc -l
echo "== __pycache__ in G, P or O"; find $G $P $O -name __pycache__ -not -path '*/.git/*' | head
