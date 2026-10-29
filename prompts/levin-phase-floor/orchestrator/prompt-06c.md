# Orchestrator — prompt 06c, global error control in the Levin driver

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code**, and the deliberate breakages are the agent's, not yours.

**The prompt:** [`prompts/levin-phase-floor/06c-global-error-control.md`](../06c-global-error-control.md)
**Board item:** 06c · **Closes, when §4 and §5 hold:** `[07-per-region-acceptance-misses-a-cancelling-total]`
and `[06b-resolved-region-keeps-a-floor-bisection-would-cut]` · **Model:** Opus, as the prompt
recommends, for every part.

This note follows [`prompt-06b.md`](prompt-06b.md), whose machinery it shares, and through it
[`prompt-07.md`](prompt-07.md) §B. **Where this note is silent, 06b's note holds, with 06c's
names:**
- `var/levin-phase-floor-06c/`;
- R17 and R18 (slugs `r17-parent` and `r18-global`);
- the tree $g$;
- X0–X5 in place of W0–W4, and Y1 for the rehearsals.

The user's word to dispatch, 2026-10-28: *"Yes, please dispatch 06c"*, after choosing to land 06c
before diagnosing `[06b-moved-calls-declare-less-than-their-error]`: *"Isn't it worth landing 06c
first and then looking again to see if the issue is there?"*

## 0. The order, and why

1. **07's Part A is set aside a second time.** Its §1, rebased on 06b with §2 run, was uncommitted in
   the main checkout and edits the same file. Before this dispatch the laptop saved it twice:
   - as `07-part-a-rebased-on-06b-tracked.patch` (`dbaebb92…`) and
     `07-part-a-rebased-on-06b-untracked.tar.gz` (`7e68849f…`) in `var/levin-phase-floor-07/wip/`,
     recorded in its `SHA256SUMS.txt`, beside the first set-aside's files;
   - as `stash@{0}` (`1278f4e5…`), message naming 07, 06c and the base `22234a8`. The first
     set-aside's stash is now `stash@{1}` (`a89ac9be…`), kept.

   The checkout was then clean. 07's agent is idle and stays so until 06c's code lands.
2. **Part A (prompt §1–§3)** is the code on the laptop: a fresh-context subagent in the main
   checkout, one commit. This note's §A.
3. **Part B (prompt §4–§5)** runs on the Mac Studio as 06b's did: one session, two agents, the
   rehearsals first and the census after them. It must be over before 07's Part B, whose census
   reads 06c's.
4. **07's Part A resumes on 06c's code commit,** with its §2 re-run (07 §9 point 6).

## A. Part A — the code, on the laptop

### A.0 Facts, checked by the orchestrator at `22234a8`

Pass them on.
- **The checkout.** `handover-remedial` at `22234a8`, clean once 07's work is set aside, one
  worktree. The code is 06b's: `git diff --stat 4ddc3c2 22234a8 -- . ':!docs' ':!prompts'` shows
  only `CLAUDE.md`. So 06b's baselines hold: AdaptiveLevin 38, ComputeTargets 780, CosmologyModels
  39, LiouvilleGreen 148 (1 skipped), ObjectFactories 280, RunDrivers 127, tools 58, all OK.
- **The driver, at `HEAD`,** in `AdaptiveLevin/levin_quadrature.py`:
  - `_local_atol` at `:1743`; `_adaptive_levin`'s work loop `while len(regions) > 0:` at `:1888`,
    popping with `regions.pop()` (depth first);
  - the fallback branch `if data.get("is_direct", False):` at `:2054`, the Levin branch after it;
  - `used_regions.sort` at `:2407`; `converged = abserr_total <= requested_total` at `:2464`;
  - `used_interval.total_err` is $\max(\text{abserr}, \text{phase\_err}) + \text{truncation}$.
- **Callers that describe the length share of `atol`** in comments include
  `ComputeTargets/QuadSourceIntegral.py` (`atol_sub`, "matching how adaptive_levin_sincos itself
  distributes atol"). The prompt allows comment-only corrections there.
- **Test 3's parent numbers** are on the board's `[06b-resolved-region-keeps-a-floor-bisection-would-cut]`:
  $\theta = 1000e^x$ on $[0, 20]$, $f = (1, 0)$, `rtol` $10^{-7}$, declared
  $9.7\times10^{-7}$ against a true $2.1\times10^{-11}$, `converged` False.
- **The known flake** is `test_tk_wkb_phase … test_wall_time_per_object`. Re-run it alone and record
  it; it is not a stop.

### A.1 Before you dispatch

1. Set 07's work aside (§0 item 1), and check `git status --short` is empty. *Done 19:14 UTC.*
2. Commit this note and its README row, and push. That commit is the **dispatch SHA**.
3. **The marker:** `touch` a file in the scratchpad immediately before the dispatch.

### A.2 Dispatch

One fresh-context subagent, on **Opus**, in the main checkout. Give it the prompt, the dispatch SHA,
the baselines and §A.0. It does not read `orchestrator/`. Tell it plainly:
- **`HEAD` is not its own.** Before staging, it runs `git log --oneline <dispatch SHA>..HEAD`, and
  merges any shared file by reading it, never by overwriting it.
- **The stashes are not its own.** It never runs `git stash` in any form, and never reads or
  touches `var/levin-phase-floor-07/`.
- **Its scope is prompt §1–§3.** It copies §4's Y1 and §5's predictions into its log unchanged,
  marked not yet scored, and scores nothing.
- **One commit, staged by explicit path,** not pushed. It may cover:
  - `AdaptiveLevin/levin_quadrature.py`;
  - `AdaptiveLevin/tests/test_levin_global_budget.py`, new;
  - existing test files under `AdaptiveLevin/tests/` and `LiouvilleGreen/tests/`, **only** for
    assertions of the kinds prompt §2 test 9 allows, each listed in the log;
  - comment-only changes in callers, per prompt §0.3;
  - `docs/levin-phase-floor/06c/calibration/`;
  - `prompts/levin-phase-floor/logs/06c-global-error-control.md` and the board;
  - `docs/OPEN_ISSUES.md`, if it opens an issue.

  **Any other path is a stop.**
- **No Ray and no store.** It writes nothing under `var/`.
- **Every deviation is numbered and classified** before it reports. It **stops and asks** at prompt
  §7's conditions that apply to §1–§3.

### A.3 The review — nine checks, on the laptop

1. **Scope.**
   - `git show --stat <code commit>` lists only §A.2's paths.
   - `git status --short` is clean.
   - `find var -newer <marker>` lists nothing of the agent's.
   - Both stashes and `var/levin-phase-floor-07/wip/` are as you left them, by SHA-256 and by
     `git stash list`.
2. **Nothing else moved,** by AST with docstrings removed, at the dispatch SHA and the code commit.
   - In `levin_quadrature.py`: `_local_atol` is gone; `_adaptive_levin` and `used_interval` change;
     any new helper is the loop's. Every other function and constant is unchanged, in particular
     `_roundoff_floor`, `_levin_G0_G1`, `_roundoff_exit_premise_holds`, `_LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO`,
     the three subregion functions and `DEFAULT_LEVIN_MAX_DEPTH`.
   - **In every caller file touched, the AST without comments and docstrings is unchanged.**
   - The result dict's keys are today's.
3. **The construction, read** against prompt §0.2 and §1:
   - the stop test is $\sum e_i \le \max(\texttt{atol}, \texttt{rtol}|\sum v_i|)$ with
     $e_i$ = `total_err`, and nothing else accepts a leaf;
   - the leaf chosen is the refinable one with the largest $e_i$, ties by the smaller start;
   - not refinable: `depth_max`; floor-limited (06b's premise, checked only for the leaf chosen).
     No stall rule (prompt §8);
   - the returned `value` and `abserr` are summed in ascending $x$, and `converged` keeps its
     formula;
   - the counters count each solve once; the carry-forward of halves is kept;
   - the docstrings and comments prompt §1 item 8 names are corrected.
4. **The roots, re-run here.** Six equal the baselines, unless an existing test file changed, in
   which case each difference is accounted for by the log's list. `AdaptiveLevin` rises by exactly
   the new module's test methods. All OK, with the flake re-run alone.
5. **The tests, read and run.**
   - Tests 1–11 each exist, named in the log.
   - **Tests 1 and 3's comments** record the parent's numbers. Re-run each problem at the
     parent yourself, from the module extracted from the dispatch SHA, and compare.
   - **Every changed existing assertion** is of prompt §2 test 9's kinds. Read each.
   - **Test 10's output** shows 20 seeds at both codes with no new failure. Re-run its harness
     yourself and compare field by field, timings aside.
   - Run the new module alone, with `-v`, and keep its numbers.
6. **The breakages.** (i)–(iii) each name the test that failed, and were reverted. **Re-apply (i)
   yourself in any case**, and (ii) or (iii) if its record is thin. Revert with `git checkout
   -- <file>`; the tree is clean afterwards.
7. **Determinism, by you:** run test 3's problem twice in fresh processes and compare value, abserr
   and the region list by bit pattern.
8. **The records.**
   - The log has README §5.1's headings, and classifies every deviation.
   - Its **Predictions and verdicts** holds prompt §4's Y1 and §5's X1–X5 word for word (check by
     substring), marked not yet scored.
   - The board's 06c row says the code landed and §4–§5 are pending. The header issues have not
     moved.
   - `black --check` is clean outside `calibration/scripts/`. The commit message follows
     `CLAUDE.md`.
9. **Nothing else landed unaccounted.** `git log --oneline <dispatch SHA>..HEAD`.

### A.4 Landing

If the review passes, push the code commit. Then make one residue commit, and push it. It holds:
- the code commit's hash on the board's 06c row and in the log;
- §B.5's addendum, written then.

**Then 07's Part A resumes:**
1. Restore its work from `stash@{0}`. If the stash will not apply cleanly, apply the patch with
   `git apply --3way`. Leave any conflict markers for its agent: resolving them is code, and not
   yours.
2. Send 07's agent 07 §9 point 6 and the user's words on 06c, verbatim, with 06c's two commits.

### A.5 Addendum, 2026-10-28: Part A's stop, and the amendment

Part A's agent stopped before committing, on prompt §7's "the stall rule fires where refinement
works", with the checkout returned clean at `3f2a36d` and its driver kept as a patch in the
orchestrator's scratchpad (`06c-driver-wip.patch`, `fed7097a…`). The orchestrator verified:
- the checkout clean, both stashes at `1278f4e5…` and `a89ac9be…`, `var/` untouched since the marker;
- **SincIntegral under the rule as written,** by applying the patch in a throwaway worktree (since
  removed): 4 leaves, 2 stalled, depth 2, unconverged, true error $6.55\times10^{-6}$ against a
  declared $4.85\times10^{-6}$;
- **(b2), by re-running the agent's measurement script:** every figure in prompt §8.

The user chose (b2) and X6 (prompt §8, the board's Decisions, **06c stall rule**). The agent is
resumed with the amendment commit and the user's words verbatim, and continues §1–§3 from its
patch. §A.3's checks stand, with prompt §2 test 9's widened allowance and X6 among the predictions
to copy.

### A.6 Addendum, 2026-10-28: the second stop, and the stall rule withdrawn

The resumed agent stopped again before committing, on the same condition: (b2) stalls on lines
through a stationary point, where Levin leaves' error estimates rise under bisection until the
fallback takes over (prompt §8). The orchestrator applied its patch (`06c-b2-wip.patch`,
`5ff7fd4e…`) in a throwaway worktree, since removed, and reproduced two cases against its own erf
reference at $c = 10^5$, `rtol` $10^{-7}$: at $x_s = 0.382$ the parent converges with true error
$1.07\times10^{-12}$ and (b2) stalls at depth 3, true $4.27\times10^{-3}$ against declared
$4.11\times10^{-3}$ (dishonest); at $x_s = 0.724$, (b2) stalls at depth 6, true $4.82\times10^{-5}$.

The user chose (A): *"(A). Let's remeasure the stall question after 07"*. The stall rule, its key,
test 12, breakage (iv), test 11's stall check and X6 are withdrawn (prompt §8). §A.3's checks are
adjusted to match: tests 1–11, breakages (i)–(iii), X1–X5. The agent is resumed with the user's words
and the amendment commit, and continues from its patch with the stall rule removed.

### A.7 Addendum, 2026-10-28: the review of `79ae8c7`, and the trust check

The agent's code commit `79ae8c7` (not pushed) passed all nine checks of §A.3:
- scope and stashes as left; nothing under `var/`;
- by AST, only `_adaptive_levin`, its two progress helpers and the loop's new helpers changed,
  `_local_atol` and `_levin_interval` went, the result keys are unchanged, both caller files are
  comment-only;
- the seven roots: AdaptiveLevin 48 (+10, the new module), the other six equal, all OK;
- tests 1 and 3's parent numbers re-run by the orchestrator, equal to the docstrings;
- breakage (i) re-applied in a throwaway worktree: tests 1, 2, 3, 5 and 6 fail;
- test 3 bit-identical across two fresh processes;
- the 20 LiouvilleGreen draws re-run at the commit in a throwaway worktree: 160 units, 130,720
  compared values, 0 differences (timings aside);
- the log's predictions copied word for word; `black` clean; the commit message as `CLAUDE.md`.

Its records also showed the budget dishonest where the parent was honest (prompt §8, the third
amendment), which the orchestrator reproduced on the Gaussian bump. The user chose a trust check one
level down, folded into 06c before anything is pushed: *"Yes, please do that. We want the Part B
census to be measuring the final scheme."*

**So the branch was moved back.** `79ae8c7` was saved as `git format-patch` output in the
orchestrator's scratchpad (`06c-79ae8c7.patch`, `0ddc2726…`), and stays reachable by its hash; the
branch, never pushed past `cf7913b`, was reset to it, and the amendment committed on top. The agent
rebuilds one code commit from `79ae8c7` with the check added.

**The review of the rebuilt commit** is §A.3 again, with:
- in check 2, `used_interval` may change by the trust check's fields, and the result keys gain
  `num_untrusted_regions`;
- in check 3, the check read against prompt §8: $\Delta_1$, $\Delta_2$, $r$; the widening
  $\max(\epsilon_P, \Delta_1)/(1-r)$; untrusted leaves refined first, the loop not stopping on one;
  floor-limited leaves exempt; every returned leaf checked, floor-limited or at the cap;
- in check 5, tests 12a–12e; **re-run 12a, 12c and the bump's parent figures yourself**;
- in check 6, **re-apply (iv) yourself** as well as (i);
- the LiouvilleGreen draws re-run again, and 12e's table of calls honest at the parent and
  dishonest under the new code read in full.

### A.8 Addendum, 2026-10-29: the review of `c85f753`, and the noise level

The rebuilt commit `c85f753` (not pushed) passed §A.3 with §A.7's additions:
- scope and stashes as left; by AST only the loop, its helpers and `used_interval` changed, and the
  result keys gained `num_untrusted_regions` alone;
- the seven roots: AdaptiveLevin 53 (+15), the other six equal, all OK;
- the orchestrator's re-runs: tests 1 and 3 as recorded; the Gaussian bump honest (declared
  $8.5641\times10^{-11}$ against a true $8.5626\times10^{-11}$); 07 §2's (R) at $c = 10^5$, both
  $x_s$, converged and honest;
- breakages (i) (7 tests fail) and (iv) (12a, 12d, 12e fail) re-applied; test 3 bit-identical
  across fresh processes;
- the LiouvilleGreen draws re-run at the commit: 160 units, 165,280 values, 0 differences.

Its records' findings went to the user (prompt §8, the fourth amendment): 5 `test_YJJ_log_singularity`
cases honest at the parent and dishonest under the new code, dishonest-and-converged 44 → 228 there,
and round-off read as non-convergence. The user: *"Let's try (b) first and see what it does to item
1."* `c85f753` was saved as a patch (`06c-c85f753.patch`, `f38f0c53…`) and stays reachable; the
branch, never pushed past `9486375`, was reset to it and the amendment committed on top. The agent
rebuilds one commit from `c85f753` with the noise level.

**The review of that commit** is §A.3 with §A.7's additions and: in check 2 `used_interval` gains
`noise_limited` and the keys `num_noise_limited_regions`; in check 3 the noise level read against
prompt §8 ($\nu = 8uM$, max of both differences); in check 5 test 12f and the item-1 table, and
re-run 12f yourself; in check 6 re-apply (v) yourself.

### A.9 Addendum, 2026-10-29: the noise level withdrawn

The agent stopped before writing code on the fourth amendment: $\nu = 8uM$ falls $10^5$–$10^6$ short
of 12f's residual (phase-argument rounding), which the orchestrator reproduced. A phase-scaled
variant (b′), measured in scratch and never committed, fixed 12f but made five converged calls
unconverged and left item 1 at 225 dishonest-and-converged; the orchestrator reproduced its 12f and
SinIntegral figures. The user chose (ii): *"Yes, please go ahead with (ii)."* 06c ships `c85f753`'s
code, re-landed unchanged on top of the fifth amendment. **Its review** is check 1 (scope), check 2
extended to *the code files' ASTs equal `c85f753`'s*, check 4 (the seven roots), check 8 (the log
and board record both noise attempts), and check 9; the rest of `c85f753`'s review (§A.8) carries
over.

## B. Part B — R17, R18 and the census, on the Mac Studio

As 06b's note §B, with these differences:
- **The scope** is prompt §4 and §5. The worktrees are `SecondaryGWKit-lpf06c-<parent>` and
  `SecondaryGWKit-lpf06c-<dispatch>`.
- **Y1 is re-scored** by your own `mode=ro` reader over every `QuadSourceIntegral` row; no row is
  expected to be bit-identical.
- **X0's first clause** is against 06b's census tree `92bc62f`. **06b's census run**
  `levin-phase-floor-06b-census-20261028T161212` is the "before", read from its `checkpoint.jsonl`
  (`f934ea24…`, `docs/levin-phase-floor/06b/KEPT_ON_MAC_STUDIO.md`).
- **X1's reference** is recomputed by you on two calls of the 24, from the saved inputs.
- **X2 and X3 are re-scored** from the two checkpoints by a scratch reader.
- **The census's checkpoint and unit pickles stay on the Mac Studio** in full. 07's census reads
  them.
- **The header issues close** in the records commit only if §4 holds and X0–X4 hold. Otherwise, ask
  the user. X5 failing is a finding, not by itself a reason to keep them open.

### B.5 Addendum: Part B's dispatch

*(Written at Part A's landing: the code commit and its parent; §B.0 re-checked on the Mac Studio;
anything Part A's review found that Part B must carry.)*

## 5. Stop and ask the user

As 06b's note §5, with prompt 06c's §7 in place of 06b's. **Never touch 07's stashes or its saved
work** except to restore `stash@{0}` as §A.4 says.
