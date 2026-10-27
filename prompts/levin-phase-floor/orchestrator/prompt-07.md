# Orchestrator — prompt 07, the stationary-point guard

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code**, and the deliberate breakages are the agent's, not yours.

**The prompt:** [`prompts/levin-phase-floor/07-stationary-point-guard.md`](../07-stationary-point-guard.md)
**Board item:** 07 · **Closes, when §5 and §6 hold:** `[06-levin-accepts-a-region-across-a-stationary-point]`
· **Model:** Opus, as the prompt recommends, for every part. You, the orchestrator, review on
whatever model you were started with.

The prompt runs in three parts. As 06's note did, this note has two:
- **Part A (prompt §1–§4), the code,** on the laptop. A fresh-context subagent works in the **main
  checkout at `HEAD`** and makes one commit. **Prompt §2 is a planned stop** inside it. This note's
  §A.
- **Part B (prompt §5 and §6), R15 and R16 under the lock, and 06a's census re-run,** on the Mac
  Studio. One session orchestrates it there and dispatches two agents. This note's §B. Its facts
  are re-checked, and its trees named, in an addendum written at its dispatch (§B.5).

## 0. The order, and why

1. **Part A first.** R16 and §6's tree $g$ are the code commit. Nothing on the Mac Studio starts
   until Part A has passed its review here and been pushed.
2. **Part A stops once by design**, at prompt §2. The agent reports its two tables and the rung each
   rule picks. You relay them to the user **as the agent wrote them**, and add nothing of your own
   but the state of the files (§A.2). The agent resumes only on the user's word, with the user's
   words passed to it verbatim.
3. **Every tree on the Mac Studio is a worktree**, as in 06 and 06a. The main checkout there does
   not move. It supplies the venv, the stores, `var/` and the registry.
4. **R15 and R16 run one after the other, under the lock.** Each is R8's build, about 3.5 minutes
   at `--cpus 6`. §6 takes no lock and uses no Ray, so its agent may run beside them. 06a took
   45 minutes on six workers. 07's windows add work only on the 720 stationary-point rows.

### 0.1 What 05–06a taught, which this note carries

- **Every deviation is numbered and classified before an agent reports,** failed attempts
  included.
- **A question relayed to the user carries the state of the numbers.** Before relaying, check what
  has already been written, and say so. In 05a the laptop told the user that no number existed when
  one did.
- **No fact in this note is to be trusted over the code or the store.** 06's note took a fact from
  a comment, and it was wrong. Each agent checks what it relies on.
- **"Value-identical" means the computed values.** It excludes timings, labels, `parents_digest`
  and work counters (`[05-store-value-diff-compares-volatile-columns]`). For R15 and R16 it also
  excludes the four new counts, which only R16 has (prompt §5.3).

### 0.2 Three things the prompt leaves open, found in writing this note

None of them blocks Part A. Each is put to the user, and nobody adjudicates it here.
1. **The analytic columns also call the integrator with `theta_deriv`.**
   - `LiouvilleGreen/three_bessel_integrals.py:509` calls `adaptive_levin_sincos` for each of four
     sum-and-difference groups. `BesselPhaseGroup.levin_theta()` (`:431–446`) supplies
     `theta_deriv`.
   - That rate is $Kx$ plus the residual log-derivatives (`:374–410`). Its docstring cites a
     resonant case with $d\theta/d\log x = 6.7\times10^{-12}$ at $x = 10^{12}$. **So location may
     fire there.** It may find a root where $K$ and the residual terms have opposite signs, or a
     shallow minimum.
   - **What follows for Part A.** Prompt §3 test 10 and §8 make any move in `LiouvilleGreen/tests/`
     a stop. The agent records whether location fires on any call those tests make, and on what.
   - **What follows for Part B.** Prompt §5.3 calls a row "located nothing" from its **group
     records**. Those cover the `phase_group_Levin_integral` calls only, not the analytic ones. A
     row whose analytic column moves because a three-Bessel call located a point would then read as
     §8's stop. **The user rules on this before Part B is dispatched** (§B.5).
2. **Gate V0's first clause does not hold as written.** Prompt §6 asks that
   `git diff --stat e05c34f <code commit's parent> -- . ':!docs' ':!prompts'` show only
   `CLAUDE.md`, or files whose ASTs without docstrings equal `e05c34f`'s. At `3921899` it also
   shows:
   - `ComputeTargets/primitive_phase.py` and `main.py`: a docstring and a comment, which the clause
     allows;
   - **five test modules** under `ComputeTargets/tests/`, changed by `handover` B2 (`d88faec`):
     `test_g_phase_reanchor.py`, `test_phase_groups.py`, `test_primitive_phase.py`,
     `test_quadsource_integral.py` and `test_tk_source_functions.py`. Their ASTs differ.

   06a's harness imports `ComputeTargets.tests.test_main_plumbing`, which is unchanged since
   `e05c34f`, and none of the five. So the clause's purpose holds, that 06a's run is the parent's.
   Its letter does not. **The user rules on it before Part B** (§B.5). V0's second clause, the
   bit-for-bit reproduction of 06a's units, is unaffected.
3. **§2's references may be impractical as written.** Prompt §2 asks for `mpmath.quad` at 30
   digits, split at every zero of $\theta'$ and at $x_0$. At $c = 10^8$ family (R) varies by about
   $10^7$ rad over $[0, 1]$, and at $b = 10^8$, $\lambda = 1024$ family (C) by about the same. The
   agent times the reference on the largest case first. If that is impractical, it **stops and
   asks**, giving the timing and a proposed alternative; it does not substitute one silently. For
   example, family (R) with these amplitudes has a closed form in the complex error function.

---

## A. Part A — the code, on the laptop

### A.0 Facts checked by the orchestrator at `3921899`

Pass them on.
- **The checkout.** `handover-remedial` at `3921899`, `git status --short` empty, one worktree
  (itself). The laptop's registry shows nothing `running` or stale: 61 runs, 59 finished and 2
  `unknown` from before the registry. The newest, `handover-13-residue-20261027T110832`, is `done`.
  Part A launches no run, starts no Ray and opens no production store.
- **The libraries:** `mpmath` 1.3.0, `scipy` 1.16.3 (`optimize.brentq`, `minimize_scalar` with
  `method="bounded"`), `numpy` 2.2.4.
- **The line references** in prompt "Read first", all in `AdaptiveLevin/levin_quadrature.py`
  (2,886 lines):
  - `_levin_G0_G1` `:322`; `_roundoff_floor` `:344`; `used_interval` `:450`;
  - `_sample_vectorized` `:890`;
  - `_Basis_SinCos` `:938`, whose `build_Levin_data` `:998` takes `theta_prime_Cheb` from
    `theta_deriv` at `:1056`, or by spectral differentiation at `:1062`;
  - `_adaptive_levin_subregion_impl` `:1194`, whose C2 gate `phase_span < SIX_PI` is `:1271`;
  - `_adaptive_levin_subregion_cc` `:1554`; `_local_atol` `:1672`;
  - `_adaptive_levin` `:1694`. Its two acceptance tests (`resolved or phase_limited or
    current_region.depth >= depth_max`) are at `:2009` and `:2189`, with `phase_limited` formed at
    `:2001` and `:2168`;
  - `adaptive_levin_sincos` `:2708`, which returns `_adaptive_levin(...)` at `:2873`.
- **The tests named in "Read first",** in `AdaptiveLevin/tests/test_levin_quadrature.py`:
  - `_TwoIndependentSinCosBasis` `:22`;
  - `test_stationary_phase_gate_uses_total_variation` `:507`;
  - `test_fallback_region_bisects_on_missed_tolerance` `:535`;
  - `test_roundoff_floor_G0_zero_is_unbounded` `:654` and `..._noise_is_unbounded` `:660`;
  - `test_generic_m_component_basis_two_decoupled_phases` `:839`.

  `AdaptiveLevin/tests/` holds that one module.
- **The production callers of `adaptive_levin_sincos`** are two:
  - `ComputeTargets/QuadSourceIntegral.py:935`, in `phase_group_Levin_integral` (`:884`), with
    `theta_deriv` supplied since `LEVIN_USE_THETA_DERIV = True` (`:102`). The group record that
    prompt §1.5 extends is the dict at `:957–967`;
  - `LiouvilleGreen/three_bessel_integrals.py:509`, with `theta_deriv` supplied. See §0.2 item 1.
- **No test in `ComputeTargets/tests/` pins a group record's exact key set,** as far as a grep for
  `simple_regions` shows. Prompt §3's one allowed change to an assertion may therefore have nothing
  to apply to. If the agent finds such a pin, it names it.
- **P1's records** are `docs/levin-phase-floor/06/p1/`: `report.md`, with the reference method in
  §5 and the root diagnosis in §6, and `scripts/reference.py` among twelve scripts. 06a's scan is
  `docs/levin-phase-floor/06a/scripts/spscan.py`. Its X1 and X2 are in
  `docs/levin-phase-floor/06a/issues.md`.
- **The known flake** in the `ComputeTargets` root is `test_tk_wkb_phase …
  test_wall_time_per_object`, a wall-clock assertion. Re-run it alone and record it; it is not a
  stop. Record `uptime` beside each roots run.

### A.1 Before you dispatch

1. Commit this note and this directory's README row for 07, and push. That commit is the
   **dispatch SHA**.
2. **Baselines:** run the seven roots serially in the main checkout, at the dispatch SHA, into the
   session scratchpad.
3. **The marker:** `touch` a file in the scratchpad immediately before the dispatch.

### A.2 Dispatch

Dispatch one fresh-context subagent, on **Opus**, in the main checkout. Give it the prompt, the
dispatch SHA, the baselines, §0.2 and §A.0 of this note. It does not read `orchestrator/`. Tell it
plainly:
- **`HEAD` is not its own.** Other sessions may commit to this branch while it works. Before staging,
  it runs `git log --oneline <dispatch SHA>..HEAD`. If anything has landed, it reads it, and merges
  any shared file by reading it, never by overwriting it.
- **Its scope is prompt §1–§4.** §5 and §6 run later on the Mac Studio, and are not its work. It
  copies §6's predictions into its log's **Predictions and verdicts** unchanged, marked not yet
  scored, and scores nothing.
- **§2 is a planned stop.** It builds §1, runs §2, and reports. It writes no test of §3 before the
  user's word.
  - **Its report** gives both tables in full, the rung each rule picks, the check on (C)'s cases
    below $L$, and any case that is not honest or not converged.
  - **It also gives** what location did on the `LiouvilleGreen/tests/` calls (§0.2 item 1), and the
    reference's cost on the largest case (§0.2 item 3).
  - **§2's records** go in `docs/levin-phase-floor/07/calibration/`: its scripts in `scripts/`,
    formatted by `black` before the run that is recorded, and its outputs in `out/`.
  - **Each output records:** the SHA-256 of its script and of `AdaptiveLevin/levin_quadrature.py`
    as it was measured, the `HEAD`, and that the tree was dirty.
- **After the user's word,** it sets the two constants as the user settled them, and goes on to §3
  and §4.
  - **If `levin_quadrature.py` changes after §2's recorded run** in anything but the two constants'
    values and docstrings, §2 is re-run on the final code.
  - **If a rung then moves,** that is a stop.
  - The review compares the committed file's SHA-256 with the one recorded.
- **One commit, at the end, staged by explicit path**, not pushed. It may cover:
  - `AdaptiveLevin/levin_quadrature.py`;
  - `AdaptiveLevin/tests/test_levin_stationary.py`, new;
  - `AdaptiveLevin/tests/test_levin_quadrature.py`: the two interface methods on
    `_TwoIndependentSinCosBasis` (prompt §1.0), and nothing else;
  - `ComputeTargets/QuadSourceIntegral.py`: the four counts in the group record (prompt §1.5), and
    nothing else;
  - in `ComputeTargets/tests/`: only a group-record key-set pin, under prompt §3's one exception;
  - `docs/levin-phase-floor/07/calibration/`;
  - `prompts/levin-phase-floor/logs/07-stationary-point-guard.md` and this campaign's board;
  - `docs/OPEN_ISSUES.md`, if the commit opens an issue.

  **A path outside that list is a stop.** In particular, these are not touched:
  `ComputeTargets/phase_groups.py`, every other file of `ComputeTargets/`, `LiouvilleGreen/`,
  `ObjectFactories/`, `config/`, `tools/`, `StoreFixtures/`, `AdaptiveLevin/__init__.py` and
  `main.py`.
- **No Ray and no store** in its tests and its checks. It writes nothing under `var/`.
- **The header's issue does not move in this commit** (prompt §4).
- **Every deviation is numbered and classified** in the log before it reports, at §2's stop too.
- **Stop and ask** at any of prompt §8's conditions that apply to §1–§4, and at §0.2 item 3.

**At §2's stop, before you relay:** list the files the agent has written. Check that its outputs'
recorded SHA-256 of `levin_quadrature.py` matches the working file. Then relay its report. If what
it reports and what its outputs hold disagree, say so.

### A.3 The review — ten checks, on the laptop

1. **Scope.**
   - `git show --stat <code commit>` lists only §A.2's paths.
   - `git diff --stat <dispatch SHA> <code commit> --` on §A.2's untouched list is empty.
   - `git status --short` is clean, and `find var -newer <marker>` lists nothing of the agent's.
2. **Nothing else moved**, by AST with docstrings removed, compared at the dispatch SHA and the
   code commit:
   - `adaptive_levin_sincos`'s body is unchanged. Location is not in the facade;
   - `_roundoff_floor`, `_levin_G0_G1`, `_local_atol` and `_adaptive_levin_subregion_cc` are
     unchanged, or each change is named in the log and is not to the eq. (151) formula;
   - `DEFAULT_LEVIN_MAX_DEPTH`, `DEFAULT_LEVIN_RELTOL`, `SIX_PI` and the `resolved` test are
     unchanged;
   - in `QuadSourceIntegral.py`, the only change is the four keys in the group record.

   `git diff <dispatch SHA> <code commit> -- LiouvilleGreen` is empty.
3. **The construction, read** against prompt §1:
   - **The interface:**
     - two new basis methods, the phases and their rates, with `_Basis_SinCos` reading
       `theta["theta"]` and `theta["theta_deriv"]`;
     - "cannot" without `theta_deriv`;
     - a basis without the methods treated as "cannot".
   - **Location:**
     - runs in `_adaptive_levin`, once per phase;
     - samples 4097 points through `_sample_vectorized`;
     - refines roots by `brentq` with `xtol` $\le 4\varepsilon\max(|x_a|, |x_b|)$;
     - refines minima by bounded Brent, with $\lambda = |\theta'|^{3/2}/\sqrt{|\theta'''|/2}$ and
       $\theta'''$ by a central difference of the rate;
     - applies the end and merge rules;
     - records each point's phase.
   - **Windows:**
     - edges by `brentq` on the raw phase at $|\theta - \theta(x_*)| = \Phi_w$, and merged on
       overlap;
     - pieces of span at most $2\pi$, the span estimated on the scan and taken as the largest over
       the phases;
     - each piece by the nested Clenshaw–Curtis pair, bisected on a missed tolerance, with
       `phase_limited` barred;
     - seeding of `x_span` minus the windows at depth 0;
     - `_local_atol` by length share, unchanged.
   - **The guard:**
     - a sign change of any rate on the region's own Chebyshev nodes;
     - never accepted on the Levin branch, and bisected while the span is $\ge 6\pi$;
     - the fallback below $6\pi$, with `phase_limited` barred;
     - at `depth_max`, the nested pair whatever the span, the abserr its estimate,
       `guard_at_depth_max` flagged and `converged` False.
   - **Nothing located and nothing guarded:** the region list is today's single region, and the
     path is today's.
   - **The two constants** are named module constants, each with a comment citing §2's
     measurement, and their values are the ones the user settled.
   - **The bypass switch** is private, named as test-only, off by default, and read nowhere but
     the location.
   - **The result** carries `stationary_points`, the three counts, and `used_interval`'s new type
     and `guarded` flag.
   - **The docstrings** prompt §1.4 lists say what is new, and `adaptive_levin_sincos`'s mentions
     the location.
4. **The roots, re-run here.**
   - Six roots equal their baselines.
   - `AdaptiveLevin` is up by exactly the test methods the new module adds, counted from the diff.
   - All OK, with the flake re-run alone.
5. **The tests, read.** Prompt §3's tests 1–10 each exist and are named in the log.
   - **References:** closed forms or `mpmath` at 30 digits, split at every stationary point.
   - **Honesty:** every test asserts both that the value is within `abserr` and that
     `abserr` $\ge$ the true error.
   - **Test 1's comment** records the parent's value, abserr and region count. Run test 1's problem
     at the dispatch SHA yourself, in a scratch worktree with no venv of its own, and compare with
     the comment.
   - **Test 8** compares by bit pattern through the bypass switch, over every stationary-free
     problem of `test_levin_quadrature.py`.
   - **Test 9** goes through `_adaptive_levin` directly.
   - **Run the new module alone** yourself, with `-v`, and keep its printed numbers.
6. **The breakages.** (i)–(vii) each name the test that failed, and each was reverted.
   - **Re-apply (i) and (iii) yourself in any case.** They are the defect and the round-off exit
     the header's issue is about.
   - **If a record is thin,** re-apply that breakage too.
   - Each time: in the working tree, run the one module, and revert with `git checkout -- <file>`.
     `git status --short` is clean afterwards.
7. **§2, re-scored.** From `docs/levin-phase-floor/07/calibration/out/`, by a scratch reader of your
   own:
   - each rule applied as written picks the rung the log says;
   - the check on (C)'s cases below $L$ holds;
   - the constants equal what the user settled;
   - the recorded SHA-256 of `levin_quadrature.py` is the committed file's, or §2 was re-run on the
     final code and its rungs did not move.
8. **The records.**
   - The log has README §5.1's headings, and classifies every deviation.
   - Its **Predictions and verdicts** holds §2's rules, tables and constants, and prompt §6's
     predictions word for word (check by substring), marked not yet scored.
   - It states what location did on the `LiouvilleGreen/tests/` calls (§0.2 item 1).
   - The board's 07 row says the code landed and §5–§6 are pending, with no commit hash yet.
   - The header's issue has not moved. `docs/OPEN_ISSUES.md` changed only if an issue was opened,
     with its count and date correct by counting.
   - `black --check` is clean on every `.py` file touched outside `calibration/scripts/`, whose
     files are as they ran.
   - The commit message follows `CLAUDE.md`'s form.
9. **What the agent touched in `ComputeTargets/tests/`,** if anything, is a key-set pin and
   nothing else: no other `assert`/`self.assert` line moved.
10. **Nothing else landed unaccounted.** `git log --oneline <dispatch SHA>..HEAD` shows the code
    commit, and anything another session landed, each accounted for.

### A.4 Landing

If the review passes, push the code commit. Then make one residue commit of your own, and push it.
It holds:
- the code commit's hash on the board's 07 row and in the log;
- §B.5's addendum, written then.

That commit is Part B's dispatch SHA. Part B is started only on the user's word, after §0.2 items 1
and 2 are ruled.

---

## B. Part B — R15, R16 and the census, on the Mac Studio

### B.0 When it starts

Part B starts only when all of these hold, each checked by the laptop over `ssh macstudio-tunnel`,
read-only:
- Part A has landed, and the residue commit is pushed;
- the user has ruled on §0.2 items 1 and 2, and §B.5 records the rulings;
- on the Mac Studio:
  - the main checkout's `git status --short` is empty;
  - `runregistrykit list` shows nothing `running` or stale;
  - the lock is free;
  - there is no `var/levin-phase-floor-07/` and no `var/datastores/levin-phase-floor-07-*`;
- the scratch copy `var/levin-phase-floor-06/wider/state/a3v2_copy/` matches its certification
  `var/levin-phase-floor-06/wider/state/copy_files_certified.txt` file for file;
- 06a's run `var/runs/levin-phase-floor-06a-census-20261027T023143/` is intact.

Then the laptop, which is the only party that writes git on the Mac Studio, does three things:
1. It fetches in the main checkout (`git fetch origin`), which moves no working tree.
2. It makes two detached worktrees, with no venv in either:
   - `git worktree add --detach ../SecondaryGWKit-lpf07-<parent short SHA> <code commit's parent>`;
   - `git worktree add --detach ../SecondaryGWKit-lpf07-<dispatch short SHA> <Part B dispatch SHA>`.
3. It starts the Mac Studio orchestrator, a Claude Code session over the saved `MacStudio` SSH
   connection. The session works from this note's §B and §B.5, which it reads in the code worktree.

### B.1 Facts, from 06a's note and run, to be re-checked at Part B's dispatch

- **The main checkout** was `handover-remedial` at `28ebaf0` when 06a ran.
- **Worktrees kept on the user's word:** `SecondaryGWKit-lpf06a-e05c34f` and 06's two. Part B
  neither reads nor removes them.
- **The scratch copy's certification:** primary `1542e05c…`; shards `bb007a4c…`, `aae63ccd…`,
  `5e2bda9b…` and `ecc3330d…`. Its fingerprint and A3 v2's are both
  `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`.
- **06a's run:** 18,560 units, 02:31:43Z–03:17:10Z, none over budget or in error. Its harness is
  `docs/levin-phase-floor/06a/scripts/` in the repository, byte-identical to the run.
- **R8's fingerprint,** which R15 and R16 must both give, is
  `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`. R9–R14 gave it.
- **R13 and R14's launches** are recorded in `docs/levin-phase-floor/06/rehearsal/`. 05's
  `classify_non_qsi.py` and `tabulate_qsi.py` are in `docs/levin-phase-floor/05/rehearsal/scripts/`.
  The readers are `docs/dk-and-rpk-migration/rehearsal/run_readers.sh`.

### B.2 What the Mac Studio orchestrator is told

**Where things are.**
- **The main checkout,** `/Users/ds283/Documents/Code/SecondaryGWKit`, holds the venv, the stores,
  `var/` and the registry. The records go in its `var/levin-phase-floor-07/`. **No tree runs here,
  and it is not moved.**
- **The parent worktree,** `…/SecondaryGWKit-lpf07-<parent>`, runs R15 and V0's reproduction of
  06a.
- **The code worktree,** `…/SecondaryGWKit-lpf07-<dispatch>`, runs R16, $g$, the value comparison
  and the readers. This note and the prompt are read here.
- **In both worktrees:**
  - every command calls the main checkout's `venv/bin/python` by its absolute path, with
    `PYTHONPATH` set to that worktree and `-B` or `PYTHONDONTWRITEBYTECODE=1`;
  - nothing is written in either.
- **The stores' paths are absolute,** into the main checkout's `var/datastores/`. Run directories
  are in the main checkout's `var/runs/`.
- **No git command that writes,** in any tree, `fetch` included. Git is read with
  `--no-optional-locks`.

**Two agents, each fresh-context, on Opus.**
- **What each is given:** the prompt, with its own section as its scope; Part B's dispatch SHA;
  both worktrees' paths; §B.1; and the addendum.
- Neither reads `orchestrator/`. They may run at the same time.

**The rehearsal agent: prompt §5.**
- **Where it writes:** `var/levin-phase-floor-07/rehearsal/`; the stores
  `var/datastores/levin-phase-floor-07-r15*` and `-r16*`; their run directories and launch logs;
  the lock; and `/tmp/ray-sgk`.
- **The launches** are R13's and R14's, with these arguments:
  - `--database`, `--job-name levin-phase-floor-07`;
  - `--register` with slugs `r15-parent` and `r16-guard`, and
    `--campaign levin-phase-floor --prompt 07`;
  - a `--purpose` that names the host, the tree's SHA and the code commit, and says that the
    fingerprint must equal `1b332ffa…`;
  - `--cpus 6`, never more.
- **The lock:**
  - `acquire` once before R15;
  - `set-pid` with each driver's Python pid;
  - held through the comparison and the readers;
  - `release` once `ray-processes` is empty.

  Each lock command is a shell command of its own, with output by `>>`, never a pipe.
- **The comparison** is `tools.store_value_diff` with `--json`, run from the code worktree.
  - 05's `classify_non_qsi.py` is copied unchanged, and recorded by SHA-256 beside 05's.
  - The per-row tabulation inside `QuadSourceIntegral` (prompt §5.3) is a new script of the
    agent's, saved in `scripts/`. It sets the four new counts aside, and reads "located nothing"
    as §B.5 records the user's ruling.
- **P1's two rows** (prompt §5.4) are read from R16 by a scratch reader opened `mode=ro`, their
  totals compared with P1's references and their located points with $z = 10211.3$ and $275.0$.
- **The readers** run on R16, with the store's files hashed before and after.

**The census agent: prompt §6.**
- **Where it writes:** only `var/levin-phase-floor-07/census/` and its own run directory. It uses
  no Ray and takes no lock.
- **The harness** is 06a's, copied from `docs/levin-phase-floor/06a/scripts/`. The diff against it
  is saved in `out/`, and each change is a deviation.
- **V0 runs before the launch,** all three clauses. The first is read as §B.5 records the user's
  ruling. The other two are compared by bit pattern.
- **The predictions** go in `state/predictions.md`, after V0 and before the launch:
  - prompt §6's text, unchanged;
  - V4's selection rule and its 24 calls.

  The mtime is the review's evidence.
- **Every checkpoint record** carries its script's SHA-256, its tree's `HEAD` and
  `git status --short`, and each unit's evidence of where its modules were loaded from.
- **At most six worker processes.** A Ray session under `/tmp/ray-sgk` appearing during the run is
  a stop.
- **The budget** is 300 s per unit, enforced from the dispatcher, with 06a's over-budget pass.
- **V4's reference** is P1's method, built from `docs/levin-phase-floor/06/p1/scripts/reference.py`
  and copied with its diff recorded.
- **`convergence_kinds.py`** is 06a's changed copy, run under $g$ as 06a ran it.

**Both agents:**
- **`HEAD` is not theirs.** They save every command and its output; the review reads the outputs,
  not the log.
- **Launch, one check, end the turn.** Each reports the run id, the run directory, the detached
  Python pid, the `sh` pid and the first checkpoint line. You wait on each pid once, in the
  background, and resume the agent with the final `status.json`.
- **Every deviation is numbered and classified** before an agent reports.
- **They stop and ask** at any of prompt §8's conditions.
- **Each agent's records** are:
  - `log.md`, its section, for appending to the code log;
  - `verification.md`, its part of `VERIFICATION.md` §7 "The stationary-point guard", with its
    links in the `07/…` form;
  - `issues.md`;
  - `out/`, `scripts/` and `state/`.

**The marker:** `touch` a file in the session scratchpad immediately before the first dispatch.

### B.3 The review — on the Mac Studio, after both agents report

Use scratch readers of your own, in your scratchpad, and never the agents' scripts, unless a check
says to re-run one. Compare floats by bit pattern (`float.hex`), never `isclose`.
1. **Scope.**
   - In all three trees, `git status --short` is empty and each `HEAD` is as dispatched.
   - `find var -newer <marker>` lists only §B.2's paths.
   - `find <each worktree> -newer <marker> -not -path '*/__pycache__/*'` lists nothing.
   - A3 v2, the scratch copy, 06's and 06a's `var/` directories and their run directories are
     unchanged.
2. **The runs.**
   - `runregistrykit list` shows R15, R16 and `census` `done`.
   - Each manifest names `levin-phase-floor`, `07` and its slug, with `git_dirty: false`.
   - Each names the SHA of the tree it ran from: R15 the parent, R16 and `census` the dispatch SHA.
3. **The fingerprints.**
   - R15 and R16's `status.json` both record `1b332ffa…`.
   - Re-run `store fingerprint` on each yourself, without `--write`.
4. **The values.**
   - Re-run `tools.store_value_diff` on R15 and R16 yourself, with `--json`. It must reproduce the
     saved JSON.
   - Outside `QuadSourceIntegral`, every value table is identical, every other difference is of the
     four volatile kinds, and no row is in one store only.
   - **Inside it, by your own reader:**
     - every row that located and guarded nothing, read as §B.5 records, is bit-identical in
       `total`, `total_abserr`, the analytic columns and every group record with the four counts
       set aside;
     - the agent's tabulation of the rest is complete.
   - Spot-check one moved row's `total` in both stores in `float.hex`.
5. **P1's rows, the readers and the lock.**
   - P1's two rows are honest under R16, by your own reader, and each locates its point.
   - The readers all exit 0, and R16's files are identical before and after.
   - The lock transcript shows `acquire` 0, `set-pid` 0 with each driver's pid, and `release` 0.
   - `ray-processes --ray-temp-dir /tmp/ray-sgk` lists nothing and `status` says free. Run each
     yourself, as a command of its own.
6. **V0 and the store.**
   - V0's three clauses are in the saved outputs, and V0 ran before the launch.
   - **Independently:** compare `ctrl-low-1`'s parent unit at `1e-7` with 06a's census unit, from
     the two `checkpoint.jsonl`s. Compare `total`, `total_abserr`, and every call's value and
     abserr, in `float.hex`.
   - The copy's files have the same hashes before and after, with no journal, `-wal` or `-shm`
     file beside them.
   - `store fingerprint`, run by you on the copy with the agent's copied `fingerprint_copy.py`,
     gives `0e32f24a…`.
7. **The trees, read.**
   - The parent's units import from the parent worktree, and $g$'s from the code worktree. Check
     the module evidence in one saved unit of each.
   - $g$'s results carry `stationary_points` and the three counts.
8. **The predictions.**
   - `state/predictions.md` is later than V0's output and earlier than the run's `manifest.json`.
   - It quotes prompt §6 word for word, and fixes V4's 24 calls.
   - The harness diff against 06a's copy matches the deviations.
9. **The verdicts, re-scored** from the checkpoint by a scratch reader:
   - **V1:** the bit-for-bit comparison against 06a's run, at both rtols;
   - **V2:** `spscan.py`'s count of accepted Levin regions across a sign change;
   - **V3:** both directions of the agreement with 06a's scan;
   - **V4:** the 24 calls against the reference;
   - **V5:** the regions at `depth_max` in calls that handled a point, with over-budget units
     counted as false.

   Each verdict is holds, fails or undecided against prompt §6's rule as written, with its numbers.
   The descriptive items are present.
10. **The records.**
    - Each `log.md` follows campaign README §5.1, and classifies every deviation, the worktrees
      among them.
    - Each `verification.md` is written as part of `VERIFICATION.md` §7.
    - Each `issues.md` exists.
    - The run directories' sizes are recorded.

Write your verdict on each check to `var/levin-phase-floor-07/review.md`, with the commands you ran
and what they printed. End the turn with a report whose first line is the verdict, pass or stop.
Then give:
- R15 and R16's fingerprints;
- the value comparison's summary: how many rows moved, how many located or guarded, and the
  largest change against `total_abserr`;
- P1's two rows;
- V0;
- V1–V5, with their numbers;
- the depth distribution at both rtols against 06a's;
- the over-budget units;
- the census's wall time against 06a's.

### B.4 After the review — on the laptop

The laptop copies `var/levin-phase-floor-07/` back with `rsync -a`, without pickles or any file over
2 MB, except a census CSV if one is kept. It then re-checks two things over `ssh macstudio-tunnel`
itself:
- check 4's spot-check, on one other moved row;
- check 6's comparison with 06a, on one stationary-point row at `1e-8`.

Then it makes **one records commit**, pushed, containing:
- the records, in `docs/levin-phase-floor/07/rehearsal/` and `…/07/census/`:
  - scripts byte-identical and unformatted, their directories added to `CLAUDE.md`'s `black`
    paragraph if any output records their SHA-256;
  - a `KEPT_ON_MAC_STUDIO.md` listing what stayed behind, with SHA-256;
- both `log.md`s, appended to `logs/07-stationary-point-guard.md`, with the verdicts in its
  Predictions and verdicts;
- both `verification.md`s, as `docs/levin-phase-floor/VERIFICATION.md` §7;
- the board: 07's row, and anything `issues.md` opens, on the user's word;
- **the header's issue:**
  - if §5 holds and V0–V4 hold, it is closed, on the board's §4, and its row is deleted from
    `docs/OPEN_ISSUES.md`;
  - if any of them fails or is undecided, **ask the user** before closing it;
  - V5 failing does not by itself keep it open. It reopens G2's L2 for the user (Decisions,
    **G2 L2, decided**), and is recorded as a finding;
- `docs/OPEN_ISSUES.md` to match, with the count and date corrected by counting;
- this directory's README row for 07, and the campaign README's.

R15 and R16's stores are the user's to keep or retire; ask. The worktrees are removed with `git
worktree remove` on the Mac Studio once the records are committed, and only on the user's word.
The scratch copy of A3 v2 stays for 08.

### B.5 Addendum: Part B's dispatch

*(Written at Part A's landing: the code commit and its parent; the rulings on §0.2 items 1 and 2;
§B.0 and §B.1 re-checked; anything Part A's review found that Part B must carry.)*

---

## 5. Stop and ask the user

Relay verbatim; do not adjudicate. Stop at:
- the prompt's §8 conditions;
- §0.2 item 3;
- any failed check in §A.3 or §B.3: report it, do not repair it.

**Never write to A3 v2 or its scratch copy,** or to 05's, 05a's, 06's or 06a's records on the
Mac Studio. **Never move the Mac Studio's main checkout,** and never end another session's run.
