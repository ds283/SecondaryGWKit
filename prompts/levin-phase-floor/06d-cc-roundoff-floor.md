# Prompt 06d — a round-off floor for the Clenshaw–Curtis rule

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-29, on the user's decisions the same day (the board's Decisions, **07 third
stop**), after 07's Part A, rebased onto 06c, stopped at §2:
- on the floor: *"I assumed we had updated the error model, not just taken over the Levin model
  intact. As you say this needs fixing for every CC solve."*
- on the order: *"06d prompt before Part B"*, and *"Yes, go with that ordering."*

It lands after 06c's code (`ad9a5f4`, records `ad63e8a`) and **before 06c's Part B**, whose
rehearsals and census then measure 06c and 06d together. It has no Part B of its own. 07's work,
rebased onto 06c and uncommitted, is set aside before 06d runs, and rebased onto 06d afterwards
(07 §9 point 7).

**Where it runs:** on the laptop only, by a fresh-context subagent, in the main checkout at
`HEAD`. One commit.

**Closes, when 06c's §4 and §5 hold on 06d's code:** `[07-cc-solves-use-the-levin-roundoff-floor]`
and `[07-depth18-diagnostic-divides-by-zero]`.

**Recommended model:** **Opus**.

**Read first:**
1. The board's §3 entries `[07-cc-solves-use-the-levin-roundoff-floor]` and
   `[07-depth18-diagnostic-divides-by-zero]`.
2. `AdaptiveLevin/levin_quadrature.py`:
   - `_roundoff_floor` and its docstring, `_levin_G0_G1`, `_LEVIN_ROUNDOFF_G0_NOISE_FLOOR`,
     `_declared_endpoint_phase_err`, `_roundoff_exit_premise_holds`;
   - `_adaptive_levin_subregion_cc` in full, in particular how it forms `"phase_err"`;
   - `_Basis_SinCos.eval_basis`, for the argument actually handed to sin and cos;
   - `_adaptive_levin`: where `"phase_err"` enters a leaf's error, the floor-limited test, the
     trust check's exemption for floor-limited leaves, and the depth-18 history log in
     `_make_leaf` (`abs_improvement = prev_abs / this_abs`).
3. `prompts/levin-refactor/logs/04-roundoff-floor.md`: why eq. (151) became the fallback's floor
   in that campaign's prompt 04, and its Item 2 (the ω-ladder, raw against reduced phase).
4. `prompts/levin-phase-floor/06c-global-error-control.md` §0.2 and §8, and
   `logs/06c-global-error-control.md`, for the loop this floor now feeds.
5. 07's third stop: `prompts/levin-phase-floor/logs/07-stationary-point-guard.md`, "third stop",
   and `docs/levin-phase-floor/07/calibration/out/stop3_diagnose.json` (in 07's saved work; the
   board quotes the numbers).

**Gate:**
- the main checkout's `git status --short` is empty;
- `runregistrykit list` shows nothing `running` or stale on the laptop;
- the seven roots' counts are taken at the dispatch `HEAD` (`CLAUDE.md`).

---

## 0. What this prompt is

### 0.1 The defect

**The Clenshaw–Curtis fallback carries the Levin rule's round-off floor.** Since `levin-refactor`
04, `_adaptive_levin_subregion_cc` forms its `"phase_err"` from eq. (151):
$\varepsilon\|f\|(h/2)(1 + G_1/G_0 + \max(G_1, k^2)/G_0)$, with $G_0 = \min|\theta'|$. That bound
models the conditioning of Levin's collocation system, $p' + i\theta' p = f$: its $1/G_0$ is the
solve's sensitivity where $\theta'$ is small. **A fallback region solves no such system.** It is a
weighted sum of $2N-1$ samples of $\sum_i f_i\,w_i(\theta)$, whose rounding has no $1/\theta'$ in it.

**Where it bites.** On a fallback region holding a stationary point, $G_0$ is $\theta'$ at the
node nearest the point, so the "floor" **grows** as the region shrinks round it.
- 07's §2 on 06c's code (`stop3_diagnose.json`): the window leaf holding $x_s$ ($c = 10^6$,
  $x_s = 0.7236$) has a floor of $6.6\times10^{-15}$ at depth 2 and $2.2\times10^{-13}$ at depth 8,
  while its nested-pair estimate is at round-off, often exactly 0. Under 06c's budget the call is
  never met, and the loop chases every window leaf towards depth 20.
- At 06c's code alone, the stationary-phase oracle (`test_stationary_phase_gate_uses_total_variation`)
  refines to 96,941 leaves; its leaves at $x = 0.5$ carry an infinite floor.

**The second defect, found on the way.** The depth-18 diagnostic divides an ancestor's `abserr`
by the next one's (`prev_abs / this_abs`). Before 06c a region with zero error was accepted and
never split, so the division was unreachable. Under 06c a leaf with a zero residual can be split
(the trust check, or a dominating floor), and 07's rebased tree raised `ZeroDivisionError` on a
§2 case and on seed 0's `test_JJJ`, which passes under 06c. 06c's census could reach it on A3 v2.

### 0.2 The remedy

**A round-off floor derived from what the rule computes.** On a fallback solve with fine-rule
weights $w_k$ ($k = 1 \dots n$, $n = 2N-1$), amplitudes $f_{ik} = f_i(x_k)$, basis values
$b_{ik}$ (sin and cos of the argument $a_k$ that `eval_basis` hands them), and $u = 2^{-53}$:

$$\phi_{\rm CC} = (n + 4)\,u\,S \;+\; u\,\sum_k |w_k|\,|a_k| \sum_i |f_{ik}|,
\qquad S = \sum_k |w_k| \sum_i |f_{ik}\,b_{ik}| .$$

- **The first term** bounds the rounding of the computed value: the $n$-term dot product
  contributes $n u S$ (the standard $\gamma_n$ bound), and each term's product, its weight's
  computation, the basis evaluation and the amplitude's last bit about one $u$ each. The constant is
  that count; it is **derived, not fitted**, and the log states the count term by term.
- **The second term** is the rounding of the argument handed to sin and cos: an error $u|a_k|$ in
  the argument moves $\sin$ or $\cos$ by at most that much. With `theta_mod_2pi` supplied, as both
  production callers supply it, $|a_k| < 2\pi$ and the term is negligible; with a raw phase it is
  $u|\theta(x_k)|$, real, and absent from eq. (151).
- **No $1/\theta'$** appears, so a stationary point inside or at the edge of a fallback region does
  not make its floor grow or become infinite.
- **The declared-phase term is unchanged:** `_declared_endpoint_phase_err` with
  $\tfrac12\|f\|h$ per endpoint already bounds $\sum_k |w_k|\,|f_k|\,\delta\theta$ from above
  (the weights are positive and sum to $h$). `"phase_err"` stays $\phi_{\rm CC} +$ declared.

**What the reduced phase does not carry.** If a caller's `theta_mod_2pi` is itself computed by
rounding a large raw phase, the second term understates; that error belongs in the caller's
`theta_abserr`, as the module's contract already says. Levin's 04 found the old *endpoint* model
optimistic by up to $8.1\times10^9$ on **Levin** regions, where the true floor is the solve's
conditioning; that does not arise on a fallback region, but §2 test 2 checks it directly.

**The guard.** The depth-18 log computes each "improvement" only where the denominator is non-zero
and finite, and prints it as not defined otherwise. Logging only: no value, error or decision moves.

### 0.3 What this changes, and what it does not

- **It changes** `_adaptive_levin_subregion_cc`'s round-off floor; a new private helper for it if
  the agent wants one; the depth-18 log; the docstrings and comments that say the fallback uses
  eq. (151) (that function's, `_roundoff_floor`'s, the module docstring, `adaptive_levin_sincos`'s
  `abserr_roundoff` description).
- **It does not change:** eq. (151) or `_roundoff_floor` on the Levin branch; `_levin_G0_G1`;
  `_LEVIN_ROUNDOFF_G0_NOISE_FLOOR` (still used by the Levin branch); the declared-phase term;
  06b's premise or $\rho$; 06c's loop, budget or trust check; the C2 gate; the result dict's keys;
  any caller.
- **What moves:** every fallback leaf's `"phase_err"`, so its `total_err`, so the call's `abserr`;
  and through them 06b's floor-limited test, 06c's budget and the trust check's floor-limited
  exemption, wherever a fallback leaf's floor matters. **Levin leaves are untouched**, so a call
  with no fallback leaf is bit-identical to 06c's (§2 test 6).

### 0.4 What it does not fix

- **07's window pieces** are fallback solves, so they inherit this floor when 07 rebases (07 §9
  point 7). Whether that lets 07's §2 run is 07's to show.
- **An error common to a leaf and its descendants** (`[06b-moved-calls-declare-less-than-their-error]`)
  and **input conditioning** (`[06c-budget-can-stop-on-a-pre-asymptotic-residual]`, item 1) are
  outside any round-off floor.

## 1. The code

1. **The floor:** $\phi_{\rm CC}$ as §0.2, in `_adaptive_levin_subregion_cc`, from the fine-grid
   samples it already holds. The arguments $a_k$ are those `eval_basis` uses; obtain them without
   a second evaluation of the phase where the basis can return them, otherwise by one evaluation
   per node, recorded.
2. **The constant:** `u` as `2.0 ** -53`, named; the $(n + 4)$ count stated in a comment term by
   term.
3. **The guard** in the depth-18 log.
4. **Docstrings and comments,** per §0.3, each saying what the fallback's floor now models and why
   eq. (151) does not apply to it.

## 2. Tests

A new module, `AdaptiveLevin/tests/test_levin_cc_floor.py`, no Ray, no datastore. References as in
06c (closed forms or `mpmath` at 30 digits). **Every test asserts honesty: the value within its
declared `abserr` of the reference, and `abserr` at least the true error.** The parent is
06c's module (`git show <dispatch SHA>:AdaptiveLevin/levin_quadrature.py`, imported under another
name), and a test that quotes its numbers records them in its comment.
1. **No $1/\theta'$.** A fallback solve whose region has a stationary point at a node (e.g.
   $\theta = c(x - x^2)$ on a short span with $x = 0.5$ a node): the parent's floor is $+\infty$;
   06d's is finite, and of order $u S$.
2. **The floor bounds round-off, raw and reduced.** Fallback-only calls (phase span below $6\pi$
   per region), with $|\theta|$ from 1 to $10^{12}$ on a short span, each with `theta` alone and with
   `theta_mod_2pi`, at a request below what round-off allows: honest in every cell; the log tables
   true error, $\phi_{\rm CC}$ and the parent's eq. (151) floor per cell, as 04's Item 2 did.
3. **A floor that no longer grows.** A fallback leaf holding a stationary point, bisected three
   levels by hand (`_adaptive_levin_subregion_cc` on the nested spans): the parent's floor rises
   level on level; 06d's halves' floors sum to at most a few times the parent leaf's.
4. **The stationary-phase oracle** (`test_stationary_phase_gate_uses_total_variation`'s problem,
   `atol = rtol = 10^{-10}`): honest, `abserr` finite. The log records leaves, depth and solves
   against 06c's (96,941 leaves) and whether it converges.
5. **The guard.** A call that logs a depth-18 history containing a zero `abserr` completes. If no
   natural problem reaches it, the history log is factored into a private helper and that helper
   is tested with a zero entry; the log says which.
6. **Nothing else moves.** Every problem `test_levin_quadrature.py` drives: where 06c's call has
   no fallback leaf, value, abserr and region list are bit-identical; otherwise the log tabulates
   both, with honesty.
7. **The LiouvilleGreen draws.** `test_3bessel_analytic.py` over seeds 0–19, as 06c's test 10, at
   06c's code and 06d's. No test fails under 06d that passes under 06c. The log carries the
   `test_YJJ_log_singularity` honesty table (dishonest, dishonest-and-converged, worst ratio) under
   both, beside 06c's. The harness lives in `docs/levin-phase-floor/06d/calibration/scripts/`.
8. **Existing tests.** Every test in `AdaptiveLevin/tests/` and `LiouvilleGreen/tests/` passes.
   An existing assertion may change only if it is of 06c §2 test 9's kinds (a region count, a
   depth, a region list or which rule its regions take, an evaluation count, bit-identity across
   the bypass), or **a phase-limited count on a fallback region**, since this prompt changes the
   fallback's floor. Never an honesty, accuracy or value assertion. **A convergence assertion may
   change only from unconverged to converged, with the result honest.** Each change is listed with
   its numbers under both codes. 06c's `test_levin_global_budget` docstrings may update quoted
   numbers that move; its assertions do not change.

**Deliberate breakages,** each recorded with the test that catches it, then reverted:
- **(i)** the fallback's floor back to eq. (151): test 1 must fail;
- **(ii)** the argument term removed: test 2 must fail on a raw-phase cell; if no cell can be made
  to fail, the log records why, and (ii) is caught by a direct test of the floor's value instead;
- **(iii)** the guard removed: test 5 must fail.

**The seven roots,** before and after. A fall is a stop. A rise equals the tests added.

## 3. Records

- `black` on the files changed, outside `calibration/scripts/`.
- `logs/06d-cc-roundoff-floor.md`, in README §5.1's form. Its **Predictions and verdicts** says
  that 06d has none of its own: 06c's §4 Y1 and §5 X1–X5 score 06c and 06d together.
- The board's 06d row. **One commit.** The header's issues move only in 06c's records commit.

## 4. Stop conditions — stop and ask

- **The Gate** fails.
- **A test cannot be made to hold:** in particular, $\phi_{\rm CC}$ fails to bound a cell of test 2,
  or test 1's floor is not finite.
- **The construction reaches outside its file,** beyond test files and comments: to
  `ComputeTargets/`, `LiouvilleGreen/`, eq. (151) on the Levin branch, the declared term, 06b's
  premise, 06c's loop or trust check, or the result dict's keys.
- **A test moves beyond §2 test 8's allowance,** a LiouvilleGreen test fails under 06d that passes
  under 06c, or a seven-roots count falls.
