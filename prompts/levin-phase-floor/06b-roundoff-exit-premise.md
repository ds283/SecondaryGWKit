# Prompt 06b — the round-off-limited exit checks its premise

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-28, on the user's decision the same day (the board's Decisions, **07 LG
stop**), when 07's Part A stopped before its §2:
- *"Fix it now. A prompt that runs before 07's §2."*

It lands before 07's code commit. 07's §1, built and uncommitted when this prompt was written, is
set aside before 06b runs, and rebased onto 06b's code afterwards (07 §9).

**Where it runs:** in three parts, as 05–07 do.
- **§1–§3, the code:** on the laptop, by a fresh-context subagent, in the main checkout at `HEAD`.
  One commit.
- **§4, rehearsals R15 and R16:** on the Mac Studio, after the code commit is pushed, under the
  lock. R15 runs at the code commit's parent and R16 at the code commit, each in a worktree. (07's
  rehearsals become R17 and R18; 07 §9.)
- **§5, the census:** on the Mac Studio, Ray-free, with no lock. 06a's census is re-run at the code
  commit, on 06's certified scratch copy of A3 v2, with 06a's run as the "before". **This run is
  also 07's "before"** (07 §9).

The records of §4 and §5 are committed on the laptop afterwards, in 06b's records commit.

**Closes, when §4 and §5 hold:** `[07-roundoff-exit-assumes-bisection-cannot-help]`.

**Recommended model:** **Opus**.

**Read first:**
1. The board's §3 entry `[07-roundoff-exit-assumes-bisection-cannot-help]`.
2. `AdaptiveLevin/levin_quadrature.py`:
   - the module docstring;
   - `_levin_G0_G1`, `_roundoff_floor` and `_declared_endpoint_phase_err`, and the constants
     above them;
   - `_adaptive_levin`: both acceptance branches, the nested Clenshaw–Curtis fallback (the
     `is_direct` branch) and the Levin branch. In particular, the computation of `dataL` and
     `dataR` for the step-(4) residual, and the comments above each `phase_limited`;
   - `_adaptive_levin_subregion_impl` and `_adaptive_levin_subregion_cc`, for what each puts in
     `"phase_err"`.
3. `AdaptiveLevin/tests/test_levin_quadrature.py`, in particular the two `test_roundoff_floor_G0_*`
   tests, and every test that asserts on `phase_limited` or `num_phase_limited_regions`.
4. `LiouvilleGreen/three_bessel_integrals.py`, `_Levin_3bessel` and `BesselPhaseGroup.theta_deriv`,
   for a production call whose $\theta'$ spans many decades. Then
   `LiouvilleGreen/tests/test_3bessel_analytic.py`. It draws its wavenumbers from `random.uniform`,
   unseeded.
5. `docs/levin-phase-floor/06a/out/census_summary.txt`, for how often the exit fires on A3 v2's
   rows, and `docs/levin-phase-floor/06a/scripts/`, the census harness that §5 re-runs.

**Gate:**
- the main checkout's `git status --short` is empty;
- `runregistrykit list` shows nothing `running` or stale on the laptop;
- the seven roots' counts are taken at the dispatch `HEAD` (`CLAUDE.md`).

---

## 0. What this prompt is

### 0.1 The defect

`_adaptive_levin` accepts a region without bisecting it in three ways:
- it is **resolved**: its step-(4) residual against its two halves is under tolerance;
- it is **round-off-limited** (`phase_limited`): its eq. (151) floor is finite and above
  tolerance, and the residual is already down at the floor;
- it is at `depth_max`.

The second exit rests on one premise, stated in the code's own comment: *"bisecting cannot help --
the children inherit essentially the same round-off floor"*. **Nothing checks it.**

The floor is roughly $\varepsilon\|f\|(h/2)(1 + G_1/G_0 + \max(G_1, k^2)/G_0)$, with
$G_0 = \min|\theta'|$ and $G_1 = \max|\theta'|$ on the rescaled region. **The premise fails where
$G_1/G_0$ is large.** There the floor is set by fitting one region over many decades of frequency,
not by the integrand. Halving the region cuts that ratio, roughly to its square root when $\theta'$
grows exponentially, so the halves' floors are far smaller than the parent's. The exit then reports
a precision limit that does not exist, and freezes a region that bisection would resolve.

**How it was found.** 07's Part A, 2026-10-27, on a seeded draw of `test_3bessel_analytic.py`'s
`test_JJJ`. The draw is $(k, q, s) = (3.302, 0.885, 2.651)$, at its own seed 10.
- **Under 07's §1** a window took one group's root, near the lower end of its span, out of the Levin
  range.
- **What remained,** $\log x \in [6.08, 27.63]$, has $|\theta'|$ from 102 to about $10^{11}$:
  $G_1/G_0 \approx 10^9$.
- **Its floor was $9\times10^{-7}$** against a residual of $6.6\times10^{-8}$, so it was accepted at
  depth 0 through the exit.
- **The result was honest but loose:** an error of $6.7\times10^{-8}$, a relative error of
  $2.6\times10^{-7}$ against the test's $10^{-7}$.
- **The parent code bisected** only because its region held the root, where $G_0 \approx 0$ makes
  the floor infinite, and the exit requires a finite floor.

The records are 07's, in `docs/levin-phase-floor/07/calibration/` when 07 lands.

**How far it reaches without 07.** In 06a's census at the current code:
- 1,213 of 37,325 Levin calls are phase-limited at `1e-7`, on 585 rows;
- 1,720 are phase-limited at `1e-8`, on 745 rows.

How many of those took the exit against its premise is not known. §5 measures it. Every
three-Bessel call is also exposed: its $\theta' \approx Kx$ grows exponentially in $\log x$.

**Why the information is already there.** On the Levin branch the step-(4) residual integrates both
halves, and each half's result carries its own floor: `dataL["phase_err"]` and
`dataR["phase_err"]`. The exit reads only the parent's (`phase_err = data.get("phase_err", ...)`).

### 0.2 The remedy

**The exit checks its premise.** A region is accepted as round-off-limited only if every existing
condition holds **and** bisection would not lower the floor materially:

$$\texttt{phase\_err}_L + \texttt{phase\_err}_R \;\ge\; \rho\,\texttt{phase\_err}_{\rm parent},
\qquad \rho = \tfrac12 .$$

**Why $\rho = \tfrac12$.**
- **Where the floor is genuinely round-off,** it does not fall under bisection. The floor's
  docstring records that the summed floor is roughly invariant under subdivision. Where the
  $k^2$ term dominates, it grows.
- **Where the floor comes from a wide range of $\theta'$,** the halves' sum falls like the ratio's
  square root. That is a factor of about $10^4$ at $G_1/G_0 = 10^9$.
- **$\tfrac12$ separates the two with a margin either way.**
  - $G_1/G_0 = 3$ gives a halves' sum of about 0.6 of the parent's, and the exit fires.
  - $G_1/G_0 = 10$ gives about 0.35, and it bisects.
- **Why not $\rho = 1$.** It would never let the exit fire while $G_1/G_0 > 1$, however slowly the
  floor fell. That drives every such region on towards `depth_max`.
- $\rho$ is a named module constant, with this argument in its comment.

**On both branches.**
- **Levin:** the halves' floors are `dataL["phase_err"]` and `dataR["phase_err"]`, which are
  already computed.
- **The nested Clenshaw–Curtis fallback:** it has no halves. Their floors are computed from the
  same functions that make a fallback region's own `"phase_err"`, on each half's own samples. The
  mechanism is the agent's choice, recorded in the log.
- **On both,** the premise is checked **only when the exit would otherwise fire.** A call in which
  no region reaches the exit does no extra work, makes no extra evaluation, and gives a
  bit-identical result.

**What the exit's flag means after the change.** `phase_limited` still marks a region accepted on
its floor. That is now a region whose floor bisection would not lower by half.

### 0.3 What this changes, and what it does not

- **It changes** `_adaptive_levin`'s two `phase_limited` tests, the comments and docstrings that
  state the premise, and one new constant.
- **It does not change:**
  - the eq. (151) formula, `_levin_G0_G1`, `_declared_endpoint_phase_err` or what `"phase_err"`
    holds;
  - the `resolved` test, `_local_atol`, `DEFAULT_LEVIN_MAX_DEPTH`, `rtol` or the C2 gate;
  - the result dict's keys, or any caller. `ComputeTargets/` and `LiouvilleGreen/` are not
    touched.
- **Only regions the parent accepted through the exit can move.** The new condition is strictly
  stronger. A region the parent accepted as resolved or at `depth_max` is accepted the same way. So
  **a call with no phase-limited region at the parent is bit-identical** (§2 test 4, §4, §5 W1).
- **It is not 07's fix.** Across a stationary point a region's halves keep a small $G_0$, so the
  exit can still fire there. 07's rule (ii) bars it in window pieces and guarded regions.

## 1. The code

1. **The constant:** `_LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO = 0.5` (or a like name), with §0.2's
   argument in its comment.
2. **The Levin branch:** `phase_limited` gains the premise, from `dataL` and `dataR`. It is
   evaluated after the existing conditions, and only if they hold.
3. **The fallback branch:** the same, with the halves' floors computed as §0.2 says, only if the
   existing conditions hold.
4. **The bypass:** a private keyword on `_adaptive_levin` that restores the old test. It is
   test-only, named as such, off by default, and read nowhere but the two `phase_limited` tests.
   `adaptive_levin_sincos` does not expose it.
5. **Docstrings and comments:**
   - the comment above each `phase_limited` test;
   - `_roundoff_floor`'s docstring, where it says the floor "cannot itself freeze a region that
     genuine further bisection could still improve";
   - the module docstring's account of the exit.

   Each says what the premise is, that it is now checked, and why $\rho = \tfrac12$.
6. **Non-finite halves' floors.** If either half's `"phase_err"` is `+inf`, the sum is `+inf`, the
   premise holds and the exit may fire. That is today's behaviour, since a finite parent floor is
   still required. The log states it.

## 2. Tests

A new module, `AdaptiveLevin/tests/test_levin_roundoff_exit.py`, with no Ray and no datastore.
References are closed forms, or `mpmath` at 30 digits by a method that converges at that precision.
Report its own error estimate. **Every test asserts both that the value is within its declared
`abserr` of the reference, and that `abserr` is at least the true error.**
1. **The exit against its premise, on the Levin branch.**
   - A phase with no stationary point, whose $\theta'$ spans at least eight decades over the span.
     $\theta = Ae^{x}$ is one choice.
   - The parameters are chosen so that **the parent code accepts a region through the exit at
     depth 0, with a true error above the requested tolerance.** Show this at the parent, and
     record the parent's value, abserr, region count and phase-limited count in the test's
     comment.
   - **Under the new code:**
     - the call is `converged`;
     - no region is phase-limited;
     - the true error is within the requested tolerance;
     - the value is honest.
2. **The exit still fires where its premise holds.**
   - A problem with modest $G_1/G_0$ (below 3 on every region) at an `rtol` tight enough that the
     floor binds. The existing `test_roundoff_floor_*` problems are candidates.
   - **Under the new code:**
     - at least one region is phase-limited;
     - no region reaches `depth_max`;
     - the result is bit-identical to the bypassed call.
3. **The fallback branch.**
   - A region on the fallback branch (phase span below $6\pi$), with no stationary point, that the
     parent accepts through the exit against its premise.
   - Under the new code it bisects, and the value is honest.
   - **If no such problem can be built,** the log records the attempts and the reason. That is not
     a stop: breakage (iii) below is then caught by a direct test of the fallback branch's premise
     check, which the agent writes.
4. **Nothing else moves.**
   - Run every problem in `test_levin_quadrature.py` with and without the bypass.
   - Where the bypassed call has no phase-limited region: value, abserr, region list and evaluation
     count are bit-identical.
   - Where it has one, the log tabulates both results.
5. **The LiouvilleGreen draws.**
   - Run `test_3bessel_analytic.py`'s tests over 20 seeded draws, seeds 0–19, seeding `random`
     before each test, at the parent (by the bypass) and at the new code.
   - **No test fails under the new code** that passes at the parent.
   - The log tabulates, per seed, the calls whose phase-limited regions changed.
   - The harness lives in `docs/levin-phase-floor/06b/calibration/scripts/`, its output in `out/`.
     The test module is not edited.
6. **Existing tests.** Every test in `AdaptiveLevin/tests/` and `LiouvilleGreen/tests/` passes
   unchanged.

**Deliberate breakages,** each recorded in the log with the test that catches it, then reverted:
- **(i)** the premise check removed, on both branches: test 1 must fail;
- **(ii)** the exit removed altogether: test 2 must fail;
- **(iii)** the premise checked on the Levin branch only: test 3 must fail.

**The seven roots,** before and after. A fall is a stop. A rise equals the tests added.

## 3. Records of the code

- `black` on the files changed, outside `calibration/scripts/`.
- `logs/06b-roundoff-exit-premise.md`, in README §5.1's form. Its **Predictions and verdicts**
  holds §5's predictions, copied before §5 runs and marked not yet scored.
- The board's 06b row.
- **One commit.** The header's issue moves only in 06b's records commit, after §4 and §5.

## 4. Rehearsals R15 and R16 — on the Mac Studio, after the code commit is pushed

As 06 §5. Records go under `var/levin-phase-floor-06b/rehearsal/`.
1. **The lock:** `acquire`, then `set-pid` at each launch, and `release` after the readers. Each
   lock command is a shell command of its own (`CLAUDE.md`).
2. **R15** runs at the code commit's parent and **R16** at the code commit, each in a worktree.
   - Each runs `RunDrivers/quadsource_atol_sweep.py --build --rehearsal` as R13 and R14 ran it.
   - Each is registered with `--campaign levin-phase-floor --prompt 06b`, with slugs `r15-parent`
     and `r16-premise`.
   - **Both fingerprints must equal `1b332ffa…`.**
3. **The value comparison** is `tools.store_value_diff` on R15 and R16, with `--json`, with 05's
   `classify_non_qsi.py` copied unchanged.
   - **Outside `QuadSourceIntegral`:** every value table is identical, every other difference is of
     the four volatile kinds of `[05-store-value-diff-compares-volatile-columns]`, and no row is in
     one store only.
   - **Inside `QuadSourceIntegral`, per row.** "Phase-limited in R15" means any phase-limited flag
     R15's row stores: `total_phase_limited`, each group record's `phase_limited`, and the analytic
     blocks' `phase_limited`. The agent establishes from the code and the store where each is
     stored, and records it.
     - **A row with no phase-limited flag in R15 is bit-identical** in `total`, `total_abserr`, the
       analytic columns and every group record.
     - **Any other row is tabulated:** the change against R15's and R16's `total_abserr`, and the
       phase-limited flags and region counts in each.
4. **The readers** (`docs/dk-and-rpk-migration/rehearsal/run_readers.sh`) run on R16. Every one
   exits 0, and the store is byte-identical before and after.
5. **Release the lock.** `ray-processes --ray-temp-dir /tmp/ray-sgk` lists nothing.
6. **The stores are kept** until the user says otherwise.

## 5. The census — on the Mac Studio, Ray-free, on A3 v2's scratch copy

06a's census, re-run at the code commit, against 06a's run as the "before".
- **The harness** is 06a's, from `docs/levin-phase-floor/06a/scripts/`, copied into
  `var/levin-phase-floor-06b/census/scripts/`. Change only:
  - the tree and its paths;
  - what §4's and W2's comparisons need recorded per call;
  - the reference for W2.

  Save the diff against 06a's copy, and record each change as a deviation.
- **The tree** $b$ is a worktree at Part B's dispatch SHA, whose code is the code commit's.
- **The store** is `var/levin-phase-floor-06/wider/state/a3v2_copy/`, re-checked against its
  certification and fingerprint, read-only, and hashed before and after.
- **The units:** every row at `1e-7` and at `1e-8`, as in 06a, in one registered, detached run with
  slug `census`.
- **The budget:** 300 s per unit, with 06a's over-budget pass.
- **Its checkpoint and per-unit records are kept in full on the Mac Studio.** 07's §6 compares
  against them.

**Gate W0, before the run:**
- **The parent's production code is 06a's.**
  `git diff --stat e05c34f <code commit's parent> -- . ':!docs' ':!prompts'` shows only:
  - `CLAUDE.md`;
  - files whose ASTs without docstrings equal `e05c34f`'s;
  - test modules that the copied harness does not import. Check this by listing every module the
    harness imports from the tree, in a dry run at the parent, and showing each changed test module
    absent from that list.

  So 06a's run is the parent's.
- **The harness reproduces 06a.** At the parent it reproduces 06a's units **bit for bit** on
  `low-G+r-1` and `ctrl-low-1` at `1e-7`, and on 06a's `0-1971` at `1e-8`.
- **The new code is deterministic.** Two runs of one $b$ unit on a row with a phase-limited call in
  06a are bit-identical.

**Predictions, fixed before the run** (README §5 rule 3). Copy them into the log unchanged.
- **W1 — nothing else moves.**
  - **The rule:** every Levin call with no phase-limited region in 06a equals 06a's call bit for
    bit, in value, abserr, region count and maximum depth. Every row all of whose calls are such
    equals 06a's row in `total` and `total_abserr`.
  - **Holds** if true at both rtols.
- **W2 — honest where it moved.**
  - **The sample:** 24 calls at `1e-7` that were phase-limited in 06a and whose result moved under
    $b$, with no sign change of $\theta'$ in 06a's scan (`spscan.py`). It is stratified by band
    (low, mid) and by $k$. The selection rule is fixed in `state/predictions.md` before the run.
  - **The reference:** Levin forced to $2^9$ and to $2^{10}$ equal pieces, with each piece's own
    floor reported. The two must agree to within a hundredth of $b$'s `abserr`; a call where they
    do not is reported, and dropped from the rule.
  - **The rule:** $|{\rm value}_b - {\rm ref}| \le$ `abserr`$_b$ for every call left in the sample.
- **W3 — the exit now fires only where bisection would not help.**
  - **The rule:** at `1e-7`, every region $b$ accepts as phase-limited has a halves' floor sum at
    least $\rho$ times its own. Recompute it from the halves' samples in the harness. Do not trust
    the code's flag.
  - **Holds** if no region violates it.
- **W4 — the cap.**
  - **The rule:** at both rtols, no call that moved reaches `depth_max`. An over-budget unit counts
    as false.
  - A failure is a finding for the user, not a fault. It bears on G2's L2 (Decisions, **G2 L2,
    decided**).

**Descriptive, not scored:**
- **How often the exit fired:** phase-limited calls and rows under 06a and under $b$ at both rtols,
  and how many of 06a's would have failed the premise.
- **Depth:** the distribution of row depth under $b$ at both rtols, against 06a's.
- **The sample's errors before:** for W2's sample, $|{\rm value}_{06a} - {\rm ref}|$ against 06a's
  abserr.
- **Totals:** how they moved against 06a, relative and against `total_abserr`$_{06a}$, by band
  and $k$.
- **Stationary-point rows:** how the 720 rows of 06a's scan moved. They are 07's, and not scored
  here.
- **Convergence:** `convergence_kinds.py` (06a's changed copy) under $b$, and the
  `total_converged` counts.
- **Cost:** wall time against 06a. Cost is not an argument (README §5 rule 4).

## 6. Acceptance

1. **§1–§3 as written:**
   - every test passing and the three breakages caught;
   - the LiouvilleGreen draws with no new failure;
   - the seven roots rising only by the tests added.
2. **§4:**
   - R15 and R16 both `done` with `1b332ffa…`;
   - outside `QuadSourceIntegral`, every value table identical;
   - inside it, every row with no phase-limited flag in R15 bit-identical, and the rest tabulated;
   - the readers exit 0, and the lock is released.
3. **§5:** W0 holds; W1–W4 are each scored holds, fails or undecided against the rule as written,
   with their numbers; and the store is unchanged.
4. **Records:**
   - `logs/06b-roundoff-exit-premise.md`, with §4 and §5's Mac Studio record appended by the
     orchestrator;
   - `docs/levin-phase-floor/VERIFICATION.md` §6b, "The round-off-limited exit";
   - the records in `docs/levin-phase-floor/06b/`.

## 7. Stop conditions — stop and ask

- **The Gate** fails.
- **A test cannot be built as written:** the parent does not show test 1's failure, or test 2 cannot
  find a problem on which the exit fires under the new code.
- **The construction reaches outside its file.** It needs a change outside
  `AdaptiveLevin/levin_quadrature.py`: to `ComputeTargets/`, `LiouvilleGreen/`, the eq. (151)
  formula, what `"phase_err"` holds, `resolved`, `_local_atol`, `DEFAULT_LEVIN_MAX_DEPTH`, `rtol`
  or the result dict's keys.
- **A test moves:**
  - an existing assertion has to change;
  - a test in `LiouvilleGreen/tests/` moves, or a seeded draw fails under the new code that passes
    at the parent;
  - a seven-roots count falls.
- **The rehearsals:**
  - R15 or R16 does not give `1b332ffa…`;
  - outside `QuadSourceIntegral` a value table differs;
  - inside it, a row with no phase-limited flag in R15 is not bit-identical.
- **The census:** W0 fails.
- **Any store file** changes.
- **The lock** is held, or will not release.
