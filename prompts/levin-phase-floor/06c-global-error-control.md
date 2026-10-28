# Prompt 06c — global error control in the Levin driver

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-28, on the user's decision the same day (the board's Decisions, **07 §2
stop**), when 07's §2 found that no rung of $\Phi_w$ passes its rule:
- *"The fully global scheme sounds like the right problem to solve, for the CC algorithm around the
  stationary point, and the Levin scheme in general … Otherwise we are just treating the symptoms
  rather than dealing with the actual cause."*
- *"Yes, please go ahead with 06c. I realise all existing computed values will move under this
  change."*

It lands after 06b's records commit and before 07's §2. 07's §1, built and uncommitted, is set aside
before 06c runs, as it was for 06b, and rebased onto 06c's code afterwards (07 §9 point 6).

**Where it runs:** in three parts, as 06b does.
- **§1–§3, the code:** on the laptop, by a fresh-context subagent, in the main checkout at `HEAD`.
  One commit.
- **§4, rehearsals R17 and R18:** on the Mac Studio, after the code commit is pushed, under the
  lock. R17 runs at the code commit's parent and R18 at the code commit, each in a worktree. (07's
  rehearsals become R19 and R20; 07 §9 point 6.)
- **§5, the census:** on the Mac Studio, Ray-free, with no lock. 06b's census is re-run at the code
  commit, on 06's certified scratch copy of A3 v2, with 06b's run as the "before". **This run is
  also 07's "before"** (07 §9 point 6).

The records of §4 and §5 are committed on the laptop afterwards, in 06c's records commit.

**Closes, when §4 and §5 hold:** `[07-per-region-acceptance-misses-a-cancelling-total]` and
`[06b-resolved-region-keeps-a-floor-bisection-would-cut]`.

**Recommended model:** **Opus**.

**Read first:**
1. The board's §3 entries `[07-per-region-acceptance-misses-a-cancelling-total]`,
   `[06b-resolved-region-keeps-a-floor-bisection-would-cut]` and
   `[06b-premise-check-can-chase-a-flat-residual-to-depth-max]`.
2. `AdaptiveLevin/levin_quadrature.py`:
   - the module docstring;
   - `_adaptive_levin` in full: the work list, both acceptance branches, the carry-forward of
     `dataL`/`dataR` and of the fallback's halves, the aggregate error and `converged`, and the
     returned dict;
   - `used_interval`, in particular `total_err`; `_levin_interval`; `_local_atol`;
     `_roundoff_exit_premise_holds` and the constant above it;
   - `_adaptive_levin_subregion_impl` (the p-mode gate and `abserr_truncation`) and
     `_adaptive_levin_subregion_cc`;
   - `adaptive_levin_sincos`'s docstring, for `atol`, `rtol`, `converged` and `phase_limited`.
3. `AdaptiveLevin/tests/test_levin_quadrature.py` and `AdaptiveLevin/tests/test_levin_roundoff_exit.py`,
   for every assertion on `num_regions`, `max_depth`, `converged`, `phase_limited` and the bypass.
4. `ComputeTargets/QuadSourceIntegral.py`, `compute_QuadSource_integral` and
   `phase_group_Levin_integral`, for how a caller shares `atol` and `rtol` across its Levin calls.
   **This prompt does not change them** (§0.4).
5. `logs/06b-roundoff-exit-premise.md`, and `docs/levin-phase-floor/06b/`, for 06b's tests,
   LiouvilleGreen draws and census, which this prompt re-runs.
6. `docs/levin-phase-floor/07/calibration/out/calibrate_tables.md` and `parent_compare.json`, when
   07 lands; until then the board's 07 §2 entry quotes the numbers this prompt needs.

**Gate:**
- 06b's records commit has landed (the board's 06b row is ✅);
- the main checkout's `git status --short` is empty;
- `runregistrykit list` shows nothing `running` or stale on the laptop;
- the seven roots' counts are taken at the dispatch `HEAD` (`CLAUDE.md`).

---

## 0. What this prompt is

### 0.1 The defect

`_adaptive_levin` keeps a stack of regions and decides each one alone. It pops a region, and
accepts it if its own test passes, otherwise it bisects it:
- **resolved:** its error estimate is below `_local_atol` (`atol` times its share of the span's
  length) or below `rtol` times **its own value**;
- **round-off-limited**, under 06b's premise;
- at `depth_max`.

Only after the stack is empty does it add the accepted regions' errors and compare the sum with
`max(atol, rtol |value|)`, to report `converged`. **Nothing ever compares a region with the
call's total.**

When the regions' values cancel in the sum, $\sum|v_i| \gg |\sum v_i|$, per-region relative
accuracy does not add up to the same relative accuracy in the total. Each accepted region may carry
up to `rtol`$\,|v_i|$, so the summed allowance is `rtol`$\sum|v_i|$, against a request of
`rtol`$\,|\sum v_i|$. Two ways it happens:
- **Physical cancellation.** 07's §2, family (R) at $x_s = 0.7236$, $\omega = 0$: the
  stationary-phase term is $\propto f_{\cos}(x_s) + f_{\sin}(x_s) = -0.566 + 0.485$, and $|I|$ is
  14–17× smaller than at $x_s = 0.382$. The pieces of the window do not see it: over a finite piece
  $\int\cos\theta$ and $\int\sin\theta$ are increments of the Fresnel integrals, which differ.
- **Telescoping.** A Levin region's value is a difference of boundary terms,
  $\sum_i[p_i(b)w_i(b) - p_i(a)w_i(a)]$. Adjacent regions share an endpoint with opposite signs,
  so $n$ regions sum to about their two outer ends while $\sum|v_i|$ grows like $n$.

**The evidence.**
- **07's §2** (rtol $10^{-10}$): every (R) case is honest at every rung of $\Phi_w$, yet 1–3 cases
  per rung report unconverged, all at $x_s = 0.7236$, $\omega = 0$. The declared error is
  1.0–6.7× the request while the **true error meets the request in every case** (at $c = 10^4$,
  $\Phi_w = 2\pi$: $1.4\times10^{-14}$ against $1.16\times10^{-13}$).
- **The parent passes the same cases narrowly:** declared/requested is 0.88, 0.30 and 0.74 at
  $x_s = 0.7236$ against 0.27, 0.02 and 0.05 at $x_s = 0.382$ ($c = 10^4, 10^6, 10^8$). 07's
  windows add pieces, each with its own allowance, and push the sum over the line.
- **In production,** a measurement made writing 05a (`handover` board,
  `[a3-baseline-unconverged-rows-are-stored-and-cannot-be-removed]`): of 67 unconverged Levin calls
  at `1e-7`, 18 fail on resolution because their regions cancel.
- **The same structure keeps an inflated floor:** a region accepted as resolved keeps its eq. (151)
  floor in `abserr` even where its halves would cut it by orders
  (`[06b-resolved-region-keeps-a-floor-bisection-would-cut]`: $\theta = 1000e^x$ on $[0, 20]$,
  declared $9.7\times10^{-7}$ against a true $2.1\times10^{-11}$, `converged` False). No test asks
  whether that region is the one holding the total back.

### 0.2 The remedy: one budget for the call

**Global adaptivity**, as QUADPACK's QAG. The driver keeps every **leaf** of the current partition,
each with its value and its error, and decides only against the whole call:
1. Let $E = \sum_i e_i$ and $V = \sum_i v_i$ over the leaves. If $E \le \max(\texttt{atol},
   \texttt{rtol}\,|V|)$, stop: converged.
2. Otherwise choose the **refinable** leaf with the largest $e_i$, and replace it by its two halves.
3. If no leaf is refinable, stop: unconverged.

**What a leaf carries is today's.** Its value $v_i$ is what today's accepted region contributes
(the region's own solve on the Levin branch, $CC_{2N-1}$ on the fallback). Its error $e_i$ is
`used_interval.total_err`: $\max(\text{residual}, \text{floor}) + \text{truncation}$. So **the loop
stops on exactly the quantity the call reports as `abserr`**, and `converged` keeps its formula.

**A leaf is refinable** unless one of these holds:
- **it is at `depth_max`;**
- **it is floor-limited:** its floor is finite, its residual is at most its floor, and 06b's premise
  holds, so its halves' floors sum to at least $\rho = \tfrac12$ of its own. Bisection would not
  lower its error materially. The premise is checked as 06b checks it: from `dataL`/`dataR` on the
  Levin branch, from halves solved on demand and carried forward on the fallback branch, and **only
  for the leaf chosen**, so a leaf never chosen costs nothing extra.

A leaf found not refinable is marked and never chosen again. **`phase_limited` keeps its meaning,
made exact:** a leaf the loop chose while the budget was unmet, and found floor-limited.

**What follows, by construction:**
- **Cancellation is handled.** The target is `rtol` times the current total, not each leaf's own
  value.
- **An inflated floor is chased.** A resolved leaf whose floor dominates the sum is the largest
  $e_i$; 06b's premise fails where its halves cut the floor, so it is bisected.
- **An infinite floor is chased.** A leaf with $e_i = +\infty$ (a stationary point at a node) is
  always chosen first, and is never floor-limited, so it is bisected until its floor is finite or
  it reaches `depth_max`. Today such a region can be accepted as resolved and make the call's
  `abserr` infinite.
- **The budget goes where the error is,** not by length. `_local_atol` goes.
- **Termination:** every bisection deepens a leaf, and a leaf at `depth_max` is never chosen.

### 0.3 What this changes, and what it does not

- **It changes** `_adaptive_levin`'s loop and acceptance; `_local_atol`, removed; and the docstrings
  and comments that describe per-region acceptance, the length share of `atol`, `phase_limited` and
  `converged` (module docstring, `adaptive_levin_sincos`, `used_interval`, the loop's comments).
- **It does not change:**
  - the eq. (151) formula, `_levin_G0_G1`, `_declared_endpoint_phase_err`, or what `"phase_err"`
    holds;
  - 06b's premise, $\rho$, or the test-only bypass, which is read only at the floor-limited test;
  - `_adaptive_levin_subregion`, `_adaptive_levin_subregion_impl`, `_adaptive_levin_subregion_cc`,
    the C2 gate, the p-mode `rtol` gate and `abserr_truncation`;
  - `DEFAULT_LEVIN_MAX_DEPTH`, the Chebyshev order, the input validation;
  - the result dict's keys and types, and the meanings of the solve and region counters;
  - any caller's code. **One exception:** a comment in a caller that describes how
    `adaptive_levin_sincos` shares `atol` by length is corrected, comment-only, and the module's
    AST without comments and docstrings is checked unchanged.
- **Every call can move.** Leaves are chosen in a different order and the loop stops on a different
  test, so no call is expected to be bit-identical. The user has accepted that all computed values
  move.

### 0.4 What it does not fix

- **A caller's own per-call shares.** `QuadSourceIntegral` splits one integral into sub-intervals
  and phase groups, gives each Levin call a length share of `atol` and the full `rtol`, and sums
  them; `total_converged` is the AND of the calls' flags. Groups and sub-intervals cancel too (its
  own comment: `numeric_quad` and `WKB_Levin` cancel by up to 5×). That is the same defect one level
  up, in the caller, and is `[07-quadsource-applies-rtol-per-call-against-a-cancelling-total]`, for
  G3's Q4 (v).
- **Stationary points.** A region accepted across one has an error estimate that is wrong, which no
  budget can repair. That is 07.
- **A residual flat in $h$** (`[06b-premise-check-can-chase-a-flat-residual-to-depth-max]`). A
  leaf whose residual does not fall is chased to `depth_max` as it is today, if it holds the total
  back. §2 test 10 and §5 report how often.
- **The p-mode gate's truncation.** It enters $e_i$ honestly, but bisection need not reduce it. §5
  reports its share.
- **G3's Q4 (iv)**, the dead length-share branch, is overtaken: the length share is removed. Q4 (iii)
  and (v) and Q5 stay G3's.

## 1. The code

1. **The leaves.** A structure holding, per leaf: its interval, depth, value, error, its carried
   halves' estimates, its histories (for the depth-18 log), and whether it is refinable. The heap
   or list is the agent's choice, recorded.
2. **The order.** The leaf chosen is the refinable one with the largest $e_i$, ties broken by the
   smaller start. **The order must be deterministic**, and two runs bit-identical. A NaN error is
   handled explicitly, by a rule the agent states and tests if it is reachable.
3. **The sums.** The loop's test may use running sums. **The returned `value` and `abserr` are
   summed over the leaves in ascending $x$**, and `converged` is computed from those, by today's
   formula. The log records whether the loop's running test and the returned flag ever disagree
   in §2's tests and the LiouvilleGreen draws.
4. **Bisection.** A Levin leaf's halves are its `dataL`/`dataR`, carried forward as today, each
   then solving its own halves for its residual. A fallback leaf's halves are evaluated fresh, or
   carried from a premise check. The counters count each solve once, as today.
5. **Floor-limited and capped leaves.** As §0.2. Each is recorded with its flag, and the loop never
   chooses it again.
6. **Removed:** `_local_atol` and every use of it; the per-region `resolved` test as an acceptance
   test. Each region's `relerr` stays as a diagnostic, its error over its own value, with the
   denominator floored at `atol`.
7. **Unchanged around the loop:** the progress notification (with $V$ and the leaf count), the
   diagnostics dump (`_write_progress_data`, handed the leaves), the depth-18 history log, the two
   health warnings, the returned `regions` sorted by start.
8. **Docstrings and comments,** per §0.3: each says what the budget is, that leaves are chosen by
   error against the call's total, what refinable means, and what `phase_limited` and `converged`
   now mean. Search the repository for "length share", `local_atol` and "share of atol" in comments
   that describe this module, and correct them, comment-only.

## 2. Tests

A new module, `AdaptiveLevin/tests/test_levin_global_budget.py`, with no Ray and no datastore.
References are closed forms, or `mpmath` at 30 digits by a method that converges at that precision
(07 §9 point 2), with their own error reported. **Every test asserts both that the value is within
its declared `abserr` of the reference, and that `abserr` is at least the true error.**

**The parent** is run from its own module, extracted from the code commit's parent with `git show`
and imported under another name, as 06b's review did. A test that needs the parent's numbers records
them in its comment, with the parent's SHA.
1. **Cancellation on the fallback branch.**
   - A stationary-phase problem whose leading term cancels. One choice: 07 §2's family (R),
     $\theta = \tfrac{c}{2}(x - x_s)^2$, with $f_{\sin} = -f_{\cos}$, so the integrand is
     $\sqrt2 f_{\cos}\cos(\theta + \pi/4)$ and the stationary-phase term vanishes.
   - Chosen so that **at the parent** $\sum|v_i| \ge 100\,|I|$ and the call misses: `converged`
     False with a true error within `rtol`$\,|I|$, or `converged` True with a true error above it.
   - **Under the new code:** `converged`; the true error within $\max(\texttt{atol},
     \texttt{rtol}|I|)$; honest.
2. **Cancellation on the Levin branch.** The same, on a problem with no stationary point whose
   Levin leaves telescope to a small total. **If none can be built** in which the parent misses,
   the log records the attempts; that is not a stop, and breakage (i) is caught by test 1.
3. **The inflated floor** (`[06b-resolved-region-keeps-a-floor-bisection-would-cut]`):
   $\theta = 1000e^x$ on $[0, 20]$, $f = (1, 0)$, `rtol` $10^{-7}$.
   - The parent: accepted resolved at depth 0, declared $\approx 9.7\times10^{-7}$ against a true
     $\approx 2.1\times10^{-11}$, `converged` False. Re-measure and record.
   - **Under the new code:** `converged`, `abserr` within `rtol`$\,|I|$, honest.
4. **The floor still stops refinement where bisection cannot help.** 06b's test 2 problem.
   - At least one leaf is flagged `phase_limited`; no leaf is at `depth_max`; honest.
5. **An unattainable request terminates.** A problem and `depth_max` (small, e.g. 6) for which the
   request cannot be met.
   - The call returns; `converged` False; the leaves holding the total back are at `depth_max` or
     flagged; honest.
6. **An infinite floor is chased.** A driven problem in which a region's stationary point falls on
   one of its nodes, so that eq. (151) is $+\infty$. `test_roundoff_floor_G0_*` test the floor
   function on such inputs; the comment in `test_roundoff_floor_G0_noise_is_unbounded` names a C2
   stationary-phase oracle that produced one. Show the parent's `abserr` infinite.
   - **Under the new code:** `abserr` finite, unless a leaf is at `depth_max`, which the test then
     shows.
7. **A total near zero.** An integrand whose integral is zero by symmetry.
   - `abserr` at most `atol`, `converged`, and the value within `atol` of 0.
8. **Determinism.** Two runs of tests 1 and 3: value, abserr and the region list bit-identical.
9. **Existing tests.** Every test in `AdaptiveLevin/tests/` and `LiouvilleGreen/tests/` passes.
   **An existing assertion may change only if it asserts the local scheme's structure**: a region
   count, a depth, a region list, an evaluation count, or bit-identity across the bypass. Never an
   honesty, accuracy, convergence or value assertion. Each change is listed in the log with the
   assertion before and after and the numbers under both codes.
10. **The LiouvilleGreen draws.** `test_3bessel_analytic.py`'s tests over 20 seeded draws, seeds
    0–19, as 06b's test 5, at the parent and the new code.
    - **No test fails under the new code** that passes at the parent.
    - The log tabulates, per seed, regions, solves, maximum depth and `converged` per call under
      both, and the calls that reach `depth_max` under either (06b found one: seed 15).
    - The harness lives in `docs/levin-phase-floor/06c/calibration/scripts/`, its output in `out/`.
11. **Cost, descriptive.** On every problem `test_levin_quadrature.py` drives: regions, solves and
    maximum depth under both codes. Cost is not an argument (README §5 rule 4).

**Deliberate breakages,** each recorded in the log with the test that catches it, then reverted:
- **(i)** a leaf leaves the candidate set once its own relative error is below `rtol` (the local
  test, restored inside the loop): test 1 must fail;
- **(ii)** floor-limited leaves stay refinable: test 4 must fail;
- **(iii)** the stop test uses the residual alone, without the floor: test 3 must fail.

**The seven roots,** before and after. A fall is a stop. A rise equals the tests added.

## 3. Records of the code

- `black` on the files changed, outside `calibration/scripts/`.
- `logs/06c-global-error-control.md`, in README §5.1's form. Its **Predictions and verdicts** holds
  §4's and §5's predictions, copied before they run and marked not yet scored.
- The board's 06c row.
- **One commit.** The header's issues move only in 06c's records commit, after §4 and §5.

## 4. Rehearsals R17 and R18 — on the Mac Studio, after the code commit is pushed

As 06b §4. Records go under `var/levin-phase-floor-06c/rehearsal/`.
1. **The lock:** `acquire`, `set-pid` at each launch, `release` after the readers. Each lock
   command is a shell command of its own (`CLAUDE.md`).
2. **R17** runs at the code commit's parent and **R18** at the code commit, each in a worktree.
   - Each runs `RunDrivers/quadsource_atol_sweep.py --build --rehearsal` as R15 and R16 ran it.
   - Each is registered with `--campaign levin-phase-floor --prompt 06c`, with slugs `r17-parent`
     and `r18-global`.
   - **Both fingerprints must equal `1b332ffa…`.** The fingerprint certifies identities, tags and
     row counts, not values (`CLAUDE.md` rule 7), and this prompt changes none of those.
3. **The value comparison** is `tools.store_value_diff` on R17 and R18, with `--json`, with 05's
   `classify_non_qsi.py` copied unchanged.
   - **Outside `QuadSourceIntegral`:** every value table is identical, every other difference is of
     the four volatile kinds of `[05-store-value-diff-compares-volatile-columns]`, and no row is in
     one store only.
   - **Inside `QuadSourceIntegral`, every row is tabulated:** $|{\rm total}_{18} - {\rm
     total}_{17}|$ against `total_abserr` in each, the region counts, maximum depths and
     `total_converged` in each.
   - **Y1 — consistency.** **Rule:** every row with finite `total_abserr` in both stores has
     $|{\rm total}_{18} - {\rm total}_{17}| \le {\rm total\_abserr}_{17} + {\rm
     total\_abserr}_{18}$. Rows with an infinite `total_abserr` in either are counted separately.
     **Holds** if no row violates it. A violation is a finding for the user, not by itself a stop.
4. **The readers** (`docs/dk-and-rpk-migration/rehearsal/run_readers.sh`) run on R18. Every one
   exits 0, and the store is byte-identical before and after.
5. **Release the lock.** `ray-processes --ray-temp-dir /tmp/ray-sgk` lists nothing.
6. **The stores are kept** until the user says otherwise.

## 5. The census — on the Mac Studio, Ray-free, on A3 v2's scratch copy

06b's census, re-run at the code commit, against 06b's run as the "before".
- **The harness** is 06b's census copy, from `docs/levin-phase-floor/06b/census/scripts/`, copied
  into `var/levin-phase-floor-06c/census/scripts/`. Change only:
  - the tree and its paths;
  - what X1–X5 need recorded per call;
  - the reference for X1.

  Save the diff against 06b's copy, and record each change as a deviation.
- **The tree** $g$ is a worktree at Part B's dispatch SHA, whose code is the code commit's.
- **The store** is `var/levin-phase-floor-06/wider/state/a3v2_copy/`, re-checked against its
  certification and fingerprint, read-only, and hashed before and after.
- **The units:** every row at `1e-7` and at `1e-8`, as in 06a and 06b, in one registered, detached
  run with slug `census`.
- **The budget:** 300 s per unit, with 06a's over-budget pass.
- **Its checkpoint and per-unit records are kept in full on the Mac Studio.** 07's §6 compares
  against them.

**Gate X0, before the run:**
- **The parent's production code is 06b's census's.** `git diff --stat <06b's Part B dispatch
  commit> <code commit's parent> -- . ':!docs' ':!prompts'` shows only `CLAUDE.md` and files whose
  ASTs without docstrings equal the dispatch commit's. So 06b's run is the parent's.
- **The harness reproduces 06b.** At the parent it reproduces 06b's units **bit for bit** on
  `low-G+r-1` and `ctrl-low-1` at `1e-7`, and on `0-1971` at `1e-8`.
- **The new code is deterministic.** Two runs of one $g$ unit are bit-identical.

**Predictions, fixed before the run** (README §5 rule 3). Copy them into the log unchanged.
"Moved" means the call's value, abserr, region count or maximum depth differs from 06b's.
- **X1 — honest where it moved.**
  - **The sample:** 24 calls at `1e-7` that moved, with no sign change of $\theta'$ in 06a's scan
    (`spscan.py`): 12 that 06b reported unconverged with no region at `depth_max`, none
    phase-limited and a finite abserr (or all such, if fewer), and the rest stratified by band
    (low, mid) and by $k$. The selection rule is fixed in `state/predictions.md` before the run.
  - **The reference:** W2's, Levin forced to $2^9$ and to $2^{10}$ equal pieces, with each piece's
    own floor reported. The two must agree to within a hundredth of $g$'s `abserr`; a call where
    they do not is reported, and dropped from the rule.
  - **The rule:** $|{\rm value}_g - {\rm ref}| \le$ `abserr`$_g$ for every call left in the sample.
- **X2 — budget misses converge.**
  - **The rule:** every Levin call that 06b reported unconverged, with no region at `depth_max`, no
    phase-limited region and a finite abserr, is `converged` under $g$, outside 06a's stationary-
    point rows.
  - **Holds** if true at both rtols.
- **X3 — no infinite declared error.**
  - **The rule:** no call under $g$ declares an infinite abserr unless one of its leaves is at
    `depth_max`.
  - **Holds** if true at both rtols.
- **X4 — the rows agree.**
  - **The rule:** outside 06a's 720 stationary-point rows, every row with finite `total_abserr`
    under both has $|{\rm total}_g - {\rm total}_{06b}| \le {\rm total\_abserr}_{06b} + {\rm
    total\_abserr}_g$.
  - **Holds** if true at both rtols.
- **X5 — the cap.**
  - **The rule:** at both rtols, no call that moved reaches `depth_max`. An over-budget unit counts
    as false.
  - A failure is a finding for the user, not a fault. It bears on G2's L2 (Decisions, **G2 L2,
    decided**).

**Descriptive, not scored:**
- **Convergence:** converged and unconverged calls and rows under 06b and $g$ at both rtols, with
  `convergence_kinds.py` (06a's changed copy) under $g$, and the `total_converged` counts.
- **Depth:** the distribution of row depth under $g$ at both rtols, against 06b's.
- **Cost:** regions, solves and wall time against 06b. Cost is not an argument (README §5 rule 4).
- **Where the budget went:** per call, the share of $E$ from residual, floor and truncation, and
  how many calls stop with truncation the largest share.
- **Flat residuals:** calls that reach `depth_max` with a residual flat over their last five levels,
  as `[06b-premise-check-can-chase-a-flat-residual-to-depth-max]`.
- **Totals:** how they moved against 06b, relative and against `total_abserr`$_{06b}$, by band and
  $k$.
- **Stationary-point rows:** how the 720 rows of 06a's scan moved. They are 07's, and not scored
  here.

## 6. Acceptance

1. **§1–§3 as written:**
   - every test passing, the three breakages caught;
   - every changed existing assertion of the kinds §2 test 9 allows, and listed;
   - the LiouvilleGreen draws with no new failure;
   - the seven roots rising only by the tests added.
2. **§4:** R17 and R18 both `done` with `1b332ffa…`; outside `QuadSourceIntegral`, every value table
   identical; inside it every row tabulated and Y1 scored; the readers exit 0; the lock released.
3. **§5:** X0 holds; X1–X5 are each scored holds, fails or undecided against the rule as written,
   with their numbers; the store is unchanged.
4. **Records:**
   - `logs/06c-global-error-control.md`, with §4 and §5's Mac Studio record appended by the
     orchestrator;
   - `docs/levin-phase-floor/VERIFICATION.md` §6c, "Global error control";
   - the records in `docs/levin-phase-floor/06c/`.

## 7. Stop conditions — stop and ask

- **The Gate** fails.
- **A test cannot be built as written:** the parent does not show test 1's or test 3's miss, or
  test 4 cannot find a leaf flagged under the new code.
- **The construction reaches outside its file.** It needs a change outside
  `AdaptiveLevin/levin_quadrature.py` beyond §0.3's comment-only exception: to `ComputeTargets/`,
  `LiouvilleGreen/`, the eq. (151) formula, what `"phase_err"` holds, 06b's premise or $\rho$, the
  subregion solvers, the C2 gate, the p-mode gate, `DEFAULT_LEVIN_MAX_DEPTH`, or the result dict's
  keys.
- **A test moves beyond §2 test 9's allowance:**
  - an honesty, accuracy, convergence or value assertion has to change;
  - a test in `LiouvilleGreen/tests/` fails, or a seeded draw fails under the new code that passes
    at the parent;
  - a seven-roots count falls.
- **The rehearsals:**
  - R17 or R18 does not give `1b332ffa…`;
  - outside `QuadSourceIntegral` a value table differs.
- **The census:** X0 fails, or a unit ends in an error (not over budget).
- **Any store file** changes.
- **The lock** is held, or will not release.
