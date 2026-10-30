# Orchestrator — prompt 06e, exact end nodes and a node term that measures the displacement

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code**, and the deliberate breakages are the agent's, not yours.

**The prompt:** [`prompts/levin-phase-floor/06e-exact-end-nodes.md`](../06e-exact-end-nodes.md)
**Board item:** 06e · **Closes, in its records commit, if §4 holds and Z0, Z2 and Z4 hold:**
`[06c-chebyshev-end-nodes-miss-the-span]` and `[06d-cc-node-term-is-pessimistic]`. Otherwise ask
the user. Z3 undecided, or Z5 failing, is a finding, not by itself a reason to keep them open.
**Model:** Opus.

This note follows [`prompt-06c.md`](prompt-06c.md), whose machinery it shares: Part A as 06d's
note §A, Part B as 06c's note §B.

## 0. The order

1. **07's work is set aside a fourth time** (done 2026-10-30). Its §1, rebased onto 06d by its
   agent at its fourth stop, was uncommitted in the main checkout. It is saved:
   - as a patch and a tarball in `var/levin-phase-floor-07/wip/`
     (`07-part-a-rebased-on-06d-*`), recorded in `SHA256SUMS.txt`;
   - as `stash@{0}`, `19eb784e…`, naming 07, 06e and its base `f5def2f`.

   The earlier three stashes are kept.
2. **06c's records commit lands**, on the user's word: the prompt's Gate. It carries the census
   harness that §5 copies, and the `depth20-tiling` and `y1-references` records the prompt cites.
3. **Part A** is the code on the laptop, one commit, reviewed here.
4. **Part B**, R19, R20 and the census, runs on the Mac Studio after Part A is pushed.
5. **07 resumes** on 06e's code, under its §9 point 8: rebase, §2 re-run to its planned stop, then
   the amendment that writes QUADPACK's rule for window leaves.

## A. Part A — the code, on the laptop

### A.0 Facts, checked by the orchestrator

- **The code is 06d's** (`3d70db9`). The seven roots at 06d's records: AdaptiveLevin 60,
  ComputeTargets 780, CosmologyModels 39, LiouvilleGreen 148 (1 skipped), ObjectFactories 280,
  RunDrivers 127, tools 58. Re-take them at the dispatch SHA.
- **The end nodes.** `chebyshev_matrices` forms `x = (a + b) / 2.0 + (b - a) / 2.0 * x_base`, with
  `x_base[0] = 1` and `x_base[-1] = -1` exactly (`_chebyshev_base`'s W&R sine form). The ends miss
  on 1,249 of 4,000 random spans.
- **The midpoint.** `_adaptive_levin` bisects at `c = (b + a) / 2.0`, computed once
  (`levin_quadrature.py:2492` at 06d). Both halves receive that float.
- **`math.fma`** is in the venv's Python 3.13.
- **The node term's inputs today:** `_cc_node_error_bound(x_span)` gives one $\delta$;
  `theta_prime_bound` is $\max|\theta'|$ over the coarse sample; `f_prime_sample` is per fine node.
- **The known flake** is `test_tk_wkb_phase … test_wall_time_per_object`; re-run it alone. It is not
  a stop.

### A.1 Before you dispatch

1. 06c's records commit has landed; `git status --short` is empty.
2. Commit any note amendments and push. That is the dispatch SHA.
3. `touch` a marker in the scratchpad.

### A.2 Dispatch

One fresh-context subagent, on **Opus**, in the main checkout. It is given the prompt, the dispatch
SHA, the baselines and §A.0, and it does not read `orchestrator/`. As 06d's §A.2, with these paths:
- `AdaptiveLevin/levin_quadrature.py`;
- `AdaptiveLevin/tests/test_levin_end_nodes.py`, new;
- existing test files under `AdaptiveLevin/tests/` and `LiouvilleGreen/tests/`, only as prompt §2
  test 8 allows, each listed;
- `docs/levin-phase-floor/06e/calibration/`;
- `prompts/levin-phase-floor/logs/06e-exact-end-nodes.md` and the board's 06e row;
- `docs/OPEN_ISSUES.md`, only if it opens an issue.

Any other path is a stop. The agent never runs `git stash`, never touches
`var/levin-phase-floor-07/`, uses no Ray and no store, writes nothing under `var/`, and makes one
commit, not pushed.

### A.3 The review — on the laptop

1. **Scope:** `git show --stat` lists only §A.2's paths; the checkout is clean; the four stashes and
   `var/levin-phase-floor-07/wip/` are unchanged by SHA-256; nothing under `var/` is newer than the
   marker.
2. **Nothing else moved,** by AST without docstrings. Only these change:
   - `chebyshev_matrices`;
   - `_cc_roundoff_floor`, or its node-term helper;
   - `_adaptive_levin_subregion_cc`, where it forms the term's arguments;
   - the new private helpers and constants.

   `_CC_NODE_ROUNDING_COUNT` and `_cc_node_error_bound` are removed or still used, never left dead.
   The result keys are unchanged.
3. **The construction, read** against prompt §0.2:
   - only the two end nodes move, after the affine map;
   - $\Delta_k$ is exact to a few $u$ of itself, against double-double base nodes;
   - $\theta'$ is evaluated per fine node;
   - both parts of $g'$ are present;
   - each of $R_g$, $R_2$ and $R_c$ is present, and each $\eta$ has a stated basis;
   - there is no $1/\theta'$.
4. **The roots, re-run here.** Six equal the baselines unless an allowed assertion changed;
   AdaptiveLevin rises by the new module's methods.
5. **The tests, read and run:**
   - tests 1–8 are named in the log;
   - **re-run tests 1, 3 and 4 yourself at the parent and at the commit;**
   - for test 4's window-like cells, recompute the true node error yourself in `mpmath` on two
     cells;
   - read every changed assertion against prompt §2 test 8;
   - re-run the LiouvilleGreen draws in a throwaway worktree and compare with the committed output,
     timings aside.
6. **The breakages:** (i)–(iii), each named with its catching test; **re-apply (i) and (ii)
   yourself.**
7. **Determinism:** test 3's problem twice in fresh processes, bit-identical.
8. **The records:**
   - the log has README §5.1's headings, with every deviation classified;
   - P1 is scored;
   - §4's and §5's predictions are copied verbatim and marked not yet scored;
   - `black` holds;
   - the commit message follows `CLAUDE.md`.
9. **Nothing else landed unaccounted:** `git log --oneline <dispatch SHA>..HEAD`.

### A.4 Landing

If the review passes, push the code commit, then commit and push Part B's dispatch addendum (§B.1),
whose commit is R20's and $g$'s tree.

## B. Part B — R19, R20, the census and the references, on the Mac Studio

As 06c's note §B, through an SSH session on the Mac Studio (`ssh macstudio-tunnel`,
`--cpus 6`), with these differences:
- **The scope** is prompt §4 and §5. The worktrees are `SecondaryGWKit-lpf06e-<parent>` and
  `SecondaryGWKit-lpf06e-<dispatch>`.
- **Z1 is re-scored** by your own `mode=ro` reader over every `QuadSourceIntegral` row; no row is
  expected to be bit-identical.
- **Z0's first clause** is against 06c's Part B dispatch tree. **06c's census run** is the
  "before", read from its `checkpoint.jsonl` as recorded in 06c's records.
- **Z3's references** run in one registered, detached run for the long two (samples 3 and 15) and
  in-session for the rest, each under 45 minutes. The three already run (1, 9, 25) are re-scored,
  not re-run. **Re-run one short reference yourself** at a different resolution pair.
- **Z2, Z4 and Z5 are re-scored** from the two checkpoints by a scratch reader.
- **The census's checkpoint and unit pickles stay on the Mac Studio** in full; 07's §6 reads them.
- **A3 v2's scratch copy** is read-only and hashed before and after. Nothing in its directory is
  moved, retired or deleted.

### B.1 Part B's dispatch addendum

Written after Part A lands. It records the code commit, its parent, the trees by SHA, and what
Part B must carry. **The cost should fall**: the depth-20 sample's wall time fell ×26 when pinned.
A census longer than 06c's (3.21 h and 5.77 h of unit wall at `1e-7` and `1e-8`) is itself a finding.

## 5. Stop and ask the user

As 06c's note §5, with prompt 06e's §7 in place of 06c's. **Never touch 07's stashes or saved
work.** 07 is restored from `stash@{0}` only when the user resumes it.
