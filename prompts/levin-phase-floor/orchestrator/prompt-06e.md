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
2. **06c's records commit lands** (done 2026-10-30), on the user's word: the prompt's Gate. It carries the census
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

## A.5 Addendum, 2026-10-30: the stop, the review and the landing

**Dispatched** at `50277f3`, the roots re-taken there by the orchestrator: AdaptiveLevin 60,
ComputeTargets 780, CosmologyModels 39, LiouvilleGreen 148 (1 skipped), ObjectFactories 280,
RunDrivers 127, tools 58, all OK. Run all at once, `test_tk_wkb_phase … test_wall_time_per_object`
and `test_background_tau … test_build_cost` failed under the load and passed re-run alone; the
agent was told to run the roots one at a time.

**The stop.** Part A stopped before committing under prompt §7: 06b's fallback-branch test 3 (b)
asserted that the premise check at least halves the declared error, which no longer holds under
06e (on 72 problems of the family, halved on 12 under 06d and on none under 06e) and is of no
allowed kind. Relayed verbatim; the user: *"Looks like (a) to me. The halving assertion looks to me
like it was just made up, so no reason to worry about it."* Sent to the agent verbatim; (b) was
restated as bit-identity across the bypass with honesty kept (the log's deviation 18).

**The review of `f4d8e17`**, the nine checks of §A.3, run by the orchestrator:
1. **Scope:** 17 paths, all §A.2's, `test_levin_roundoff_exit.py` on the ruling; the checkout
   clean; the four stashes and `var/levin-phase-floor-07/wip/` unchanged by SHA-256; nothing under
   `var/` newer than the marker.
2. **By AST without docstrings:** only `chebyshev_matrices`, `_cc_roundoff_floor` and
   `_adaptive_levin_subregion_cc` change; added `_cc_node_term_parts`,
   `_chebyshev_base_corrections`, `_chebyshev_node_displacements`, `_two_sum`, `_two_product`, six
   constants, and two `_Basis_SinCos` methods (deviation 7, as 06d's one); `_CC_NODE_ROUNDING_COUNT`
   and `_cc_node_error_bound` removed. `_adaptive_levin` unchanged, so the result keys are.
3. **The construction** read against prompt §0.2: only x[0] and x[−1] move, after the map; Δ_k by
   TwoSum and Dekker's TwoProduct against 160-bit base corrections, cached per order, with a stated
   bound (2u|Δ| + an absolute part of order u²(M + h)); θ′ per fine node (the caller's at the odd
   nodes, the coarse sample at the even); both parts of g′; R_g, R_2 and R_c, each η with its basis
   (η_θ = 2⁻¹⁰ supplied, measured on closed forms only; 1 spectral; η_f = 1); no 1/θ′.
4. **The roots:** AdaptiveLevin 68 (+8, the new module), the other six equal, all OK, no flake.
5. **The tests**, re-run by the orchestrator's own script: test 1 on its own 4,000 spans, the parent
   missing on 2,029 and the commit on none, interior nodes and D identical; test 3 at the parent
   depth 16, 639 regions, unconverged, honest, and at the commit depth 0, converged, honest; test 4
   on two window cells (c = 10⁶ depth 12 leaf −2, c = 10⁸ ω = 1 depth 14 leaf +0), the true node
   error recomputed in `mpmath` at 50 digits: true/φ_node 0.249 and 0.976, 06d's term over φ_node
   2,308 and 432. The changed assertions read against prompt §2 test 8 (two counts, one rule, the
   removed helper inlined) and the ruling. The 20 draws re-run in a throwaway worktree at
   `f4d8e17`: 160 units, 8,640 calls, identical to the committed output timings aside.
6. **The breakages:** (i) re-applied, caught by tests 1, 2 and 3; (ii) re-applied, caught by test 4
   on all 122 cells, the reduced-phase cells among them; (iii)'s record, 83 failures in test 4.
7. **Determinism:** test 3's problem bit-identical in two fresh processes (value, abserr, regions).
8. **The records:** README §5.1's headings, 18 deviations classified; Z0–Z5 verbatim by
   substring, not yet scored; P1 holds (median 848); `black` clean; the commit message as
   `CLAUDE.md`.
9. `50277f3..HEAD` held only `f4d8e17`.

Reported to the user with: 7 near-singular draws newly flagged converged while dishonest (431 → 438
of 560, none honest → dishonest); 06d dishonest where the offset hid (bears on Z3); η_θ unmeasured
on production's θ′; the test module's `git show` at test time; a fallback solve about twice as
costly with θ′ supplied. **The user: *"Yes, please land it"*.** `f4d8e17` pushed; this addendum's
commit is the residue (the board's Decisions, **06e lands**).

### B.2 Addendum, 2026-10-30: Part B's dispatch (§B.1)

**Part A** is the code commit **`f4d8e17`**, whose parent is **`50277f3`**, reviewed (§A.5) and
pushed.

**The trees, by SHA:**
- **R19 and Z0's reproduction** at **`50277f3`**, in the parent worktree. Its code is 06d's, which
  06c's census ran: `git diff --stat e5764a6 50277f3 -- . ':!docs' ':!prompts'` shows only
  `CLAUDE.md` (checked on the laptop), so Z0's first clause holds against 06c's Part B dispatch
  tree `e5764a6`.
- **R20 and $g$** at **this addendum's commit**, in the code worktree. Its code is `f4d8e17`'s:
  the commit adds only records.
- **06c's census** `levin-phase-floor-06c-census-20261029T130214` is the "before", its
  `checkpoint.jsonl` `b61cb034…` (`docs/levin-phase-floor/06c/KEPT_ON_MAC_STUDIO.md`).

**What Part B must carry:**
- **The cost should fall.** On the laptop's draws depth-20 calls fell 101 → 0 and solves
  3.38 M → 0.47 M; the depth-20 sample's wall time fell ×26 with the ends pinned. A census longer
  than 06c's (3.21 h and 5.77 h of unit wall at `1e-7` and `1e-8`) is itself a finding.
- **A fallback solve costs about twice as much** with θ′ supplied (97 → 186 µs, N − 1 extra θ′
  evaluations); both production callers supply θ′.
- **η_θ = 2⁻¹⁰ was measured on closed-form θ′ only.** Production's θ′ is the first test of its
  margin; a Z3 or Z4 failure on a Clenshaw–Curtis-dominated call should be read against it.
- **Z3:** 06d was dishonest wherever the end-node offset happened to hide (06e's log, Observations
  2), so the census side, $|v_{06c} - {\rm ref}|/{\rm abserr}_{06c}$, may exceed 1 on references
  where $g$ is honest.
- **The near-singular class** may surface converged and dishonest if A3 v2 reaches it
  (`[06c-budget-can-stop-on-a-pre-asymptotic-residual]`, widened by 06e).

**Not yet done:** §B.0's read-only re-check on the Mac Studio, and the worktrees
`SecondaryGWKit-lpf06e-50277f3` and `SecondaryGWKit-lpf06e-<this commit's short SHA>`, made after
this commit is pushed. Part B is dispatched on the user's word.

### B.3 Addendum, 2026-10-31: Part B's run, the review and the records

**Dispatched** 2026-10-30 to a Mac Studio session on the user's word, after §B.2's commit (`6d81fad`)
was pushed and the worktrees `SecondaryGWKit-lpf06e-50277f3` and `SecondaryGWKit-lpf06e-6d81fad`
made. That session ran §B.0's preflight (`docs/levin-phase-floor/06e/review/preflight/`),
dispatched a rehearsal agent and a census agent, re-ran sample 2's reference at P = 96 with 80/112
nodes, and wrote its review (`docs/levin-phase-floor/06e/review.md`): **passes**, no §5 stop.

**Interruptions.** The laptop restarted once and slept twice during the 9 h 35 min references run;
the Mac Studio kept running. Its session's wait on the driver was re-armed once by message; its turn
when the run ended failed on an expired login and was resumed by message, with nothing lost.

**What Part B carried** (§B.2): the cost fell (unit wall ×0.711 and ×0.515); Z3's census side
exceeded 1 only on sample 3 (1.197); the near-singular class did not surface. Z3's reference passes
took 2–3.5× the prompt's estimates, which were sized for `refq.py`, not `orch_ref.py` at three
resolutions (the log's Part B record).

**The orchestrator's checks on the laptop:** the records copied back by `rsync` without pickles,
the readers' extract or files over 2 MB; the Z2 call read from the census checkpoint over SSH (its
floor is 06e's node term, the declared-phase term 0); sample 3's ratio recomputed from the recorded
doubles (1.24505); D3's hypothesis checked by replaying samples 2, 3, 8 and 9 at `6d81fad`
(`docs/levin-phase-floor/06e/investigations/d3-leaf-values/`).

**The user's rulings:** both header issues close, *"OK. Close both header issues. We need to open a
new issue for D4."*; D3 opened once its mechanism was checked; R19 and R20 retired by the user,
after fingerprints were written from a checkout at `6d81fad` (the main checkout's `28ebaf0` cannot
read a store with `theta_residual`); both worktrees removed. The census run directory stays on the
Mac Studio for 07, whose "before" it is. 07 resumes from `stash@{0}` on the user's word.
