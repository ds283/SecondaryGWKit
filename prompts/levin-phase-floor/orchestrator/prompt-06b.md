# Orchestrator — prompt 06b, the round-off-limited exit checks its premise

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code**, and the deliberate breakages are the agent's, not yours.

**The prompt:** [`prompts/levin-phase-floor/06b-roundoff-exit-premise.md`](../06b-roundoff-exit-premise.md)
**Board item:** 06b · **Closes, when §4 and §5 hold:** `[07-roundoff-exit-assumes-bisection-cannot-help]`
· **Model:** Opus, as the prompt recommends, for every part.

This note follows [`prompt-07.md`](prompt-07.md), whose machinery it shares: the dispatch rules,
the Mac Studio's worktrees, and the review's form. **Where this note is silent, 07's note holds,
with 06b's names:**
- `var/levin-phase-floor-06b/`;
- R15 and R16 (slugs `r15-parent` and `r16-premise`);
- the tree $b$;
- W0–W4 in place of V0–V5.

## 0. The order, and why

1. **07's Part A is set aside first.** Its §1 is uncommitted in the main checkout and edits the
   same file. Before 06b's dispatch the laptop saves it twice:
   - as a patch and a tarball of its untracked files, in
     `var/levin-phase-floor-07/wip/` with their SHA-256;
   - as a `git stash` with a message naming 07.

   It then returns the checkout to `HEAD`. 07's agent is idle and stays so until 06b's code lands.
   It then restores the work and rebases it (07 §9 point 1).
2. **Part A (prompt §1–§3)** is the code on the laptop: a fresh-context subagent in the main
   checkout, one commit. This note's §A.
3. **Part B (prompt §4–§5)** runs on the Mac Studio as 07's note §B describes, one session and two
   agents. It may run beside 07's Part A once that resumes, since the two share no tree. It must be
   over before 07's Part B, whose census reads 06b's.
4. **07's Part A resumes on 06b's code commit**, not on Part B. Its §2 needs 06b's code, not 06b's
   census.

## A. Part A — the code, on the laptop

### A.0 Facts, checked by the orchestrator at `9afcfd0`

Pass them on.
- **The checkout.** `handover-remedial` at `9afcfd0`, clean once 07's work is set aside, one
  worktree. No code has changed since `93d5cb1`: `git diff --stat 93d5cb1 9afcfd0 -- . ':!docs'
  ':!prompts'` is empty. So 07's baselines hold: AdaptiveLevin 32, ComputeTargets 780,
  CosmologyModels 39, LiouvilleGreen 148 (1 skipped), ObjectFactories 280, RunDrivers 127, tools 58,
  all OK.
- **The exit, at `HEAD`,** in `AdaptiveLevin/levin_quadrature.py`:
  - **The fallback branch** (`if data.get("is_direct", False):`, about `:1980–2010`). It forms
    `phase_limited` from its own nested-pair `abserr` and `data["phase_err"]`. It computes no
    halves.
  - **The Levin branch** (about `:2130–2190`). `dataL` and `dataR` are computed for the step-(4)
    residual, each from `_adaptive_levin_subregion`, which returns `"phase_err"`. The test is
    `np.isfinite(phase_err) and phase_err > local_atol and phase_err > rtol * relerr_denom and
    abserr <= phase_err`, then `and not resolved`.
  - **`"phase_err"`** is `round_off_err + declared_err` on both branches (`:1547`, `:1663`).
- **The census's counts** (06a, `out/census_summary.txt`): phase-limited Levin calls 1,213 at `1e-7`
  (585 rows) and 1,720 at `1e-8` (745 rows), of 37,325.
- **The stored flags.** `QuadSourceIntegral` keeps `total_phase_limited` (the WKB Levin calls,
  `:798`), each group record's `phase_limited` (`:962`), and the analytic blocks' `[J, Y]`
  `phase_limited` (`:1331`). Where each lands in the store is for the Part B agent to establish.
- **The known flake** is `test_tk_wkb_phase … test_wall_time_per_object`. Re-run it alone and record
  it; it is not a stop.

### A.1 Before you dispatch

1. Set 07's work aside (§0 item 1), and check `git status --short` is empty.
2. Commit the prompt, this note, 07's §9 amendment, the board, the READMEs and
   `docs/OPEN_ISSUES.md`, and push. That commit is the **dispatch SHA**.
3. **The marker:** `touch` a file in the scratchpad immediately before the dispatch.

### A.2 Dispatch

One fresh-context subagent, on **Opus**, in the main checkout. Give it the prompt, the dispatch SHA,
the baselines and §A.0. It does not read `orchestrator/`. Tell it plainly:
- **`HEAD` is not its own.** Before staging, it runs `git log --oneline <dispatch SHA>..HEAD`, and
  merges any shared file by reading it, never by overwriting it.
- **The stash is not its own.** It never runs `git stash` in any form, and never reads or touches
  `var/levin-phase-floor-07/`.
- **Its scope is prompt §1–§3.** It copies §5's predictions into its log unchanged, marked not yet
  scored, and scores nothing.
- **One commit, staged by explicit path,** not pushed. It may cover:
  - `AdaptiveLevin/levin_quadrature.py`;
  - `AdaptiveLevin/tests/test_levin_roundoff_exit.py`, new;
  - `docs/levin-phase-floor/06b/calibration/`;
  - `prompts/levin-phase-floor/logs/06b-roundoff-exit-premise.md` and the board;
  - `docs/OPEN_ISSUES.md`, if it opens an issue.

  **Any other path is a stop.** No existing test file is edited.
- **No Ray and no store.** It writes nothing under `var/`.
- **Every deviation is numbered and classified** before it reports. It **stops and asks** at prompt
  §7's conditions that apply to §1–§3.

### A.3 The review — eight checks, on the laptop

1. **Scope.**
   - `git show --stat <code commit>` lists only §A.2's paths.
   - `git status --short` is clean, apart from nothing.
   - `find var -newer <marker>` lists nothing of the agent's.
   - The stash and `var/levin-phase-floor-07/wip/` are as you left them, by SHA-256 and by
     `git stash list`.
2. **Nothing else moved,** by AST with docstrings removed, at the dispatch SHA and the code commit.
   Every function but `_adaptive_levin` is unchanged, as are all constants but the new one. In
   `_adaptive_levin`, read the diff: the change is confined to the two `phase_limited` tests, the
   halves' floors on the fallback branch, and the bypass keyword.
3. **The construction, read** against prompt §0.2 and §1:
   - the premise is $\texttt{phase\_err}_L + \texttt{phase\_err}_R \ge \rho\,\texttt{phase\_err}$,
     with $\rho = \tfrac12$ a named constant;
   - it is evaluated only when the existing conditions hold;
   - on the fallback branch the halves' floors come from the functions that make a fallback
     region's own floor, on the halves' own samples, with no evaluation when the exit is not a
     candidate;
   - `+inf` halves behave as prompt §1 item 6 says;
   - the bypass is private, test-only, off by default, and not exposed by `adaptive_levin_sincos`;
   - the comments and docstrings prompt §1 item 5 lists are corrected.
4. **The roots, re-run here.** Six equal the baselines. `AdaptiveLevin` is up by exactly the new
   module's test methods. All OK, with the flake re-run alone.
5. **The tests, read and run.**
   - Tests 1–6 each exist, named in the log.
   - **Test 1's comment** records the parent's failure. Run its problem with the bypass yourself and
     compare.
   - **Test 4** compares by bit pattern, evaluation count included.
   - **Test 5's output** shows 20 seeds at both codes with no new failure. Re-run its harness
     yourself, and compare the outputs field by field (timings aside).
   - Run the new module alone, with `-v`, and keep its numbers.
6. **The breakages.** (i)–(iii) each name the test that failed, and were reverted. **Re-apply (i)
   yourself in any case**, and (ii) or (iii) if its record is thin. Revert with `git checkout --
   <file>`; the tree is clean afterwards.
7. **The records.**
   - The log has README §5.1's headings, and classifies every deviation.
   - Its **Predictions and verdicts** holds prompt §5's predictions word for word (check by
     substring), marked not yet scored.
   - The board's 06b row says the code landed and §4–§5 are pending, with no commit hash. The header
     issue has not moved.
   - `black --check` is clean outside `calibration/scripts/`. The commit message follows
     `CLAUDE.md`.
8. **Nothing else landed unaccounted.** `git log --oneline <dispatch SHA>..HEAD`.

### A.4 Landing

If the review passes, push the code commit. Then make one residue commit, and push it. It holds:
- the code commit's hash on the board's 06b row and in the log;
- §B.5's addendum, written then.

**Then 07's Part A resumes:**
1. Restore its work from the stash. If the stash will not apply cleanly, apply the patch with
   `git apply --3way`. Leave any conflict markers for its agent: resolving them is code, and not
   yours.
2. Send 07's agent the user's words on §2 and the LiouvilleGreen stop, verbatim, with 07 §9 and the
   two commits.

## B. Part B — R15, R16 and the census, on the Mac Studio

As 07's note §B, with these differences:
- **The scope** is prompt §4 and §5. The worktrees are `SecondaryGWKit-lpf06b-<parent>` and
  `SecondaryGWKit-lpf06b-<dispatch>`.
- **"Phase-limited in R15"** is read from every flag prompt §4.3 names, as the agent establishes
  them. The review re-derives that set by its own `mode=ro` reader before checking bit-identity.
- **W3 is re-scored** from the harness's recorded halves' floors, by your own reader, for every
  region $b$ accepted as phase-limited at `1e-7`.
- **W2's reference** is recomputed by you on two calls of the 24, from the saved inputs.
- **The census's checkpoint and unit pickles stay on the Mac Studio** in full. 07's census reads
  them, and `KEPT_ON_MAC_STUDIO.md` lists them.
- **The header issue closes** in the records commit if §4 holds and W0–W3 hold. Otherwise, ask the
  user. W4 failing is a finding, not by itself a reason to keep it open.

### B.5 Addendum, 2026-10-28: Part A landed

**Part A** is the code commit **`4ddc3c2`**, whose parent is **`da5c24d`**. It passed all eight
checks of §A.3 on the laptop, and is pushed.
- **Scope.** Only §A.2's paths. By AST without docstrings, the module changed only in
  `_adaptive_levin`, plus `_LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO = 0.5` and
  `_roundoff_exit_premise_holds`. `adaptive_levin_sincos` is unchanged. The stash and
  `var/levin-phase-floor-07/wip/` are intact by SHA-256.
- **The roots,** re-run by the laptop at `4ddc3c2`: AdaptiveLevin 38 (32 + the new module's 6),
  ComputeTargets 780, CosmologyModels 39, LiouvilleGreen 148 (1 skipped), ObjectFactories 280,
  RunDrivers 127, tools 58, all OK. The flake did not fire.
- **Test 1, re-run by the laptop against the parent's own module,** extracted from `da5c24d`. Its
  value `0x1.18d9a35c68fc4p-7` and abserr `0x1.047898c70661ep-20` (1 region, phase-limited) are
  bit-identical to the bypass and to the test's comment. Under the new code there are 2 regions,
  the call converges, and the error is $1.5\times10^{-11}$.
- **Test 5's 20 seeded draws, re-run by the laptop** in a scratch worktree at `4ddc3c2`, equal the
  committed `lg_seeded_draws.jsonl` on all 160 units, timings aside. Every recorded SHA-256 matches
  its committed file.
- **Breakage (i), re-applied by the laptop** (the premise always holds): tests 1 and 3 fail, and
  so does the infinite-half contract test. It was reverted, and the tree is clean.

**What Part B must carry:**
- **The census's W4 may fail.** One seeded LiouvilleGreen call, seed 15 `test_JJJ`, runs to
  depth 20 under the new code: from depth 5 and 23 solves to 2,295 solves, with abserr unchanged
  and the call unconverged in both. Its step-(4) residual is flat at 1–2e-15 while the eq. (151)
  floor falls to 8e-20 (`[06b-premise-check-can-chase-a-flat-residual-to-depth-max]`). W4 is
  scored as written, and a failure is a finding for the user.
- **Evaluation counters** differ only in calls that had a phase-limited **fallback** region at the
  parent. There the halves are now solved. Such a row is phase-limited in R15, so §4.3's
  bit-identity rule does not reach it.
- **The trees, by SHA:** R15 and W0's reproduction at `da5c24d`, in the parent worktree. R16 and
  $b$ at the dispatch commit of Part B, whose code is `4ddc3c2`'s.

**Part B starts on the user's word.** §B.0's Mac Studio facts are re-checked by the laptop at that
point, and added here.

## 5. Stop and ask the user

As 07's note §5, with prompt 06b's §7 in place of 07's §8. **Never touch 07's stash or its saved
work** except to restore it as §A.4 says.
