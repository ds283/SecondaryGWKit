# Orchestrator — prompt 01, the production quadrature `rtol`

Read [`../README.md`](../README.md) first: §0, §1, §2, §5, §6 and §7. Then read
[`prompt-00.md`](prompt-00.md) §0 and this board's *Orchestrator review of prompt 00*. Together they
are the conventions this note does not repeat.

**You do not write code.** You may:
- run the suites, the T1 search and `substitute_main_py`;
- replay the log's deliberate-breakage diffs with `git apply`, and revert them;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`01-set-the-production-quadrature-rtol.md`](../01-set-the-production-quadrature-rtol.md)
**Board items:** T1–T2 · **Closes:** `[00-quadsource-extraction-reads-at-the-default-rtol]`, on this
board's §4 · **Adds dated lines to:** `handover`'s
`[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]` and `levin-refactor`'s
`[06-quadrature-atol-is-inert-and-rtol-is-the-binding-half]`. Neither is closed.
**Model:** Sonnet.
**Gate:** none. `HEAD` is the commit that adds this note, or a later one that touches no code. One
prompt at a time in this checkout: 05 and 06 may land before or after 01, but not while it runs.

## 0. What makes this prompt unusual

**It is the smallest change in the campaign, and the one most likely to be over-done.** A single
constant moves, `DEFAULT_QUADRATURE_RTOL`, from `1e-8` to `1e-7`. The risk is not the value. It is
that the agent "corrects" things that are records, or history, or out of scope.

**The facts, checked by the orchestrator on 2026-10-02 at `9de2af3`.** Pass them on; the agent need
not rediscover them.
- **The constant has three production consumers:** `main.py` (the tolerance lookup the sweep's
  `RTOL_SITE` substitutes), `extract_QuadSourceIntegral_data.py` (its tolerance lookup), and
  `ComputeTargets/QuadSourceIntegral.py`.
  - In `QuadSourceIntegral.py` it is the default of `rtol` in `evaluate_QuadSource_integral` and
    `compute_QuadSource_integral`, and it appears in two comments.
  - **No test runs quadrature at the default.** `test_quadsource_integral.py`'s `run()` passes
    `rtol=1e-8` explicitly, and `test_main_plumbing.py` pins the constant's *name*, not its value.
    So the prompt's first stop condition is not expected to fire. If it does, it is real.
- **The figures for the new comment** are in `docs/handover/QUADSOURCE-TOLERANCE-SWEEP.md`. They
  give the worst deviation of `1e-7` from each band's tightest rung:
  - mid band **1.9e-06** (§2 table);
  - low band **2.84e-06** (§5, against `1e-8`);
  - seam band **8.26e-07** (§4, "1e-7 vs 1e-8").
  - `1e-6` is **1.1e-02** wrong in the mid band (§0 answer 3, §2.1).
  - Prompt 01 as first written gave the seam figure as 9.97e-07, which is `1e-6` against `1e-7`.
    It was corrected in the commit that adds this note. The review checks the comment against the
    sweep document, not against the prompt.
- **The T1 search's expected hits in production code.** The review compares the log's list
  against these. A reasonable agent may judge the history and "leave" rows differently, but each
  must appear in the log with a reason.

  | Where | What it says | Expected action |
  |---|---|---|
  | `config/defaults.py`, the constant | `= 1e-8` | **changed** to `1e-7`, with its new comment **above** it |
  | `config/defaults.py`, the comment below the constant | `atol`'s history: "rtol was confirmed not to bind (1e-8 -> 1e-11 …)" | **left**: the prompt says so, and standing note 26 and `[06-quadrature-atol-is-inert-…]` record that it is stale |
  | `QuadSourceIntegral.py`, the comment on the deleted `LEVIN_RELERR = 1e-8` | history of deleted module constants | **left**: history, not a statement of the current value |
  | `QuadSourceIntegral.py`, the `MIN_SUBINTERVAL_LOG_WIDTH` comment | "1e-7/11.4 ~ 1e-8 of the integral -- at the quadrature rtol" | **corrected**: ~1e-8 is now a tenth of the quadrature `rtol`, not equal to it. Name the constant, or say "a tenth of". Whether the argument still holds, that a discarded sub-interval is negligible, is the agent's to state in the log |
  | `QuadSourceIntegral.py`, `analytic_integral`, "with 1e-21 / 1e-8 hardwired" | history of the literals that call replaced | **left** |
  | `QuadSourceIntegral.py`, `analytic_integral`, "the pipeline supplies … 1e-25 and … 1e-8" | the current pair, wrong in both halves | **corrected**: name both constants rather than their values |
- **The layout of `config/defaults.py`.** The comment directly **below**
  `DEFAULT_QUADRATURE_RTOL = 1e-8` ("was 1e-25; tightened per …") is `DEFAULT_QUADRATURE_ATOL`'s.
  That file puts each comment above its constant, so the new `rtol` comment goes **above** the
  `rtol` line. The `atol` comment and line stay byte-identical.
- **The overlap with `[06-quadrature-atol-is-inert-and-rtol-is-the-binding-half]`**
  (`levin-refactor` board; index row in `docs/OPEN_ISSUES.md` §1.3). That issue names both
  stale comments: `defaults.py`'s `atol` comment, and `QuadSourceIntegral.py`'s "1e-25 and 1e-8".
  01 fixes the second and leaves the first. Prompt 01 was amended to add a dated line to that
  entry, saying exactly that, and to keep the issue open. The agent must not close it, and must not
  edit the `atol` comment to close it.
- **`substitute_main_py(filter_cases=False)`** matched each of its four sites once at `9de2af3`. To
  check it:

  ```bash
  PYTHONPATH=. ./venv/bin/python -c "import importlib.util as u; s=u.spec_from_file_location('sweep','docs/handover/quadsource_atol_sweep.py'); m=u.module_from_spec(s); s.loader.exec_module(m); m.substitute_main_py(filter_cases=False)"
  ```

  It prints four `replaced 1 occurrence` lines. Importing the module has no side effects (it has a
  `__main__` guard).
- **Records that read the default at their own tree.** 01 names these and does not touch them:
  - `docs/tolerance-convergence/quadsource_readonly.py` and `QUADSOURCE-READONLY.md`;
  - `docs/tolerance-convergence/inventory.py` and `TOLERANCE-INVENTORY.md`;
  - `docs/source-remediation-verification/run_quadsource_integrals.py`, whose `--rtol` defaults to
    the constant (added to the prompt with this note).

  `docs/a3-v2-readiness/reader_writes_probe.py` also imports the constant. It builds its stand-in
  store from the constant and deletes by it, so it is self-consistent at either value. It needs no
  mention beyond the log's list.
- **What the move does to stores.** The A3 baseline holds **no** `QuadSourceIntegral` rows (its
  fingerprint: `QuadSourceIntegral: 0`, audit S4), so the extractor loses nothing there. A3 v2 does
  not exist yet. After 01, a store built at any pair other than `(1e-32, 1e-7)` is unreadable by
  the extractor. The prompt's §6 makes that an *Observation not acted on*.

**What the review exists to establish.**
- **(B1) Only the value moved.** `config/defaults.py`'s AST differs from its parent's only in the
  value of `DEFAULT_QUADRATURE_RTOL`. `QuadSourceIntegral.py` and every other production file 01
  touches have ASTs identical to their parents': comments only. A docstring change shows in the AST
  and must be one the log lists.
- **(B2) The comment is true.** Every number in the new comment is in the sweep document, at the
  place the comment says, for the comparison it says.
- **(B3) Records are additive.** `TOLERANCE-PROVENANCE.md`, the `handover` board and the
  `levin-refactor` board each show additions only (`--numstat`). The provenance entry's heading,
  "`DEFAULT_QUADRATURE_RTOL = 1e-08` … unchanged", is not rewritten. A new subsection beneath it
  says what changed.
- **(B4) The pins catch what they claim.** The log's breakage diffs replay with `git apply`, and each
  fails the test the log names.

**Carried from audit 00's review.** All bind the agent, and the brief says so.
- **Staging and reporting.**
  - Stage by explicit path only, never `git add -A`, `.` or `commit -a`. Show
    `git diff --cached --name-only` before committing.
  - **Count, then paste.**
  - A log states only what the diff shows. Line numbers come from `git show`, not a working-tree
    view.
- **Suites.**
  - README §7's command now writes each suite to a file and greps for the verdict. `| tail -N`
    can hide `AdaptiveLevin`'s.
  - Under heavy load `test_tk_wkb_phase.TestCost.test_wall_time_per_object`,
    `test_background_tau.test_build_cost` and `test_numeric_phase_cut.test_the_print_is_gated_both_ways`
    fail. Re-run a failure alone before calling it real.
- **Residue** found in review is the orchestrator's, fixed in a follow-up commit (the user,
  2026-09-29).
- **Scratch.** The agent works in its own subdirectory of the session scratchpad, never its top
  level.
- **Usage limits.** Resume a stopped agent with `SendMessage`, saying exactly where it stopped. If
  it commits but sends no report, review from the commit and its log.

## 1. Before you dispatch

1. **The tree.** Record the branch and `HEAD`.
   - `git status --short` and `git diff --cached` are both empty.
   - `git worktree list` shows this checkout alone.
   - `git log --oneline 9de2af3..HEAD` touches no code. If it does, re-check §0's facts that it
     touches.
2. **The registry.** `./venv/bin/python -m RunRegistry list`: nothing `running`. At `9de2af3` there
   were 12 runs, 10 finished and 2 `unknown` (`a3-pilot`, `a3-pilot-resume`). Report any change,
   without acting on it. 01 launches nothing.
3. **Baselines.** At `e320338` (the same code as `9de2af3`), measured 13:28–13:34 on 2026-10-02:
   AdaptiveLevin 32, ComputeTargets 665, CosmologyModels 39, Datastore 551, LiouvilleGreen 148
   (1 skipped), RunRegistry 199. Re-measure all six on `HEAD`, serially, with README §7's command.
   Record the load average.
4. **The index.** `grep -c '^| \`\[' docs/OPEN_ISSUES.md`. It is **107** at `9de2af3`. Tell the
   agent the number.
5. **The pre-image.** Record `git rev-parse HEAD:config/defaults.py` and
   `HEAD:ComputeTargets/QuadSourceIntegral.py`, for check 2.

No snapshot of `var/` is needed: 01 opens no store. Check 1 confirms that nothing under `var/`
changed in the diff.

## 2. Dispatch

One fresh-context subagent, on **Sonnet**. Give it:
- the prompt;
- the campaign README;
- this board;
- `HEAD`, the baselines, and the index count;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad.
  The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the end.** Its log is at `logs/01-set-the-production-quadrature-rtol.md`, in
  README §5.1's form, with the prompt's §7 addition: every T1 search hit, with what was done and
  why.
- **The files it may add or change:**
  - `config/defaults.py`;
  - comment-only edits in production files the T1 search finds (expected: `ComputeTargets/QuadSourceIntegral.py` alone);
  - `docs/TOLERANCE-PROVENANCE.md`, a new subsection;
  - `prompts/handover/IMPLEMENTATION_STATE.md` and `prompts/levin-refactor/IMPLEMENTATION_STATE.md`,
    one dated line each, added and nothing rewritten;
  - one new test module under `ComputeTargets/tests/`;
  - the log; this board; `docs/OPEN_ISSUES.md` (the closed row deleted, `[06-quadrature-atol-…]`'s
    hook, the count and the date).

  **Nothing else.** No `main.py`, no sweep script, no existing test, nothing under `docs/` besides
  the provenance file, nothing under `orchestrator/`.
- **The comment's numbers come from the sweep document**, at §0's places, and the log quotes each
  line it took them from.
- **Stop and ask** on either of the prompt's §5 conditions.

## 3. The review — eight checks

1. **Scope.**
   - `git diff HEAD~1 HEAD --stat` touches only §2's list.
   - `git diff HEAD~1 HEAD -- main.py docs/handover/ '*/tests/*' ':!ComputeTargets/tests/<the new module>'`
     is empty, and so is the diff over `var/`.
   - `--numstat` on the provenance file and both other boards shows no deletion.
2. **(B1) Only the value moved.**
   - Compare `ast.dump(ast.parse(...))` of each touched production file at `HEAD~1` and `HEAD`.
   - `config/defaults.py`: the only difference is the `Constant` under `DEFAULT_QUADRATURE_RTOL`,
     `1e-08` → `1e-07`.
   - Every other production file: identical, or differing only in a docstring the log lists.
   - `DEFAULT_QUADRATURE_ATOL`'s line and its comment are byte-identical (`git diff -U0` shows them
     untouched).
3. **(B2) The comment.**
   - Read it against §0's figures and the sweep document. Every number has its band and its
     comparison right, and no claim goes beyond the document. The "converged" claim is the
     document's §0 answer 3, not a new one.
   - It names the old value, D1's one-rung-tighter check (`1e-8`), and the document.
   - It sits above the constant.
4. **The T1 list.**
   - Run the search yourself:
     `grep -rnE '1e-0?8\b' --include='*.py' main.py extract_*.py config ComputeTargets AdaptiveLevin LiouvilleGreen CosmologyModels Datastore RayTools RunRegistry tools | grep -v /tests/ | grep -iE 'quad|rtol|QuadSource|levin|tol'`.
   - Every hit of §0's table is in the log, with its action and reason.
   - The `MIN_SUBINTERVAL_LOG_WIDTH` and `analytic_integral` comments are corrected. The deleted
     constant's history and the `atol` comment are left.
5. **(B4) The pins.**
   - Replay each of the log's breakage diffs (i)–(iii) with `git apply`, and run the named test.
     It fails. Revert with `git apply -R`, and check `git status` is clean after each.
   - (iii) fails in `test_main_plumbing`, as the prompt says it will.
6. **`substitute_main_py`.** Run §0's command. Four `replaced 1 occurrence` lines.
7. **(B3) The records.**
   - `[00-quadsource-extraction-…]` is in this board's §4, with a closing note naming the commit.
   - Its index row is gone. The index count is 106 by counting, and matches the header.
   - The provenance subsection is dated and cites the sweep and D1, and the entry above it is
     unchanged.
   - The `handover` line says the issue stays open, and why.
   - The `levin-refactor` line says the `QuadSourceIntegral.py` comment is fixed, that the live run
     its next step asked for is the sweep, and that the `atol` comment keeps the issue open. Its
     index hook says the same.
   - The log's *Observations not acted on* has the other-pair-is-unreadable note, and the five
     records of §0.
8. **Suites.** All six serially, at `HEAD`.
   - ComputeTargets is 665 plus the new module's tests; every other count is unchanged.
   - `black --check` is clean on every touched Python file.

## 4. After the review

- **Record it on this board**, after the prompt-00 review and in its form, with `HEAD` at dispatch,
  the checks, and any remnants. Fix small residue in a follow-up commit of its own.
- **Report to the user:**
  - the constant moved;
  - what the T1 search corrected and what it left, and why;
  - that `[00-quadsource-extraction-…]` is closed;
  - that `[06-quadrature-atol-is-inert-…]` stays open only on `config/defaults.py`'s `atol`
    comment. Offer it as a one-line follow-up for the user to decide: either rewrite that sentence
    or add a dated note to it.
- **What 01 hands on.** 03 is now unblocked on 01's side; it still waits on 02. 07's rehearsal at
  `(1e-32, 1e-7)` can use `main.py`'s own default for `rtol`. Whether 07 still substitutes is 07's
  to decide, when it is written.
