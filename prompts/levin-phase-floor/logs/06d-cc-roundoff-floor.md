# levin-phase-floor 06d — a round-off floor for the Clenshaw–Curtis rule

> Written by the executing agent on the laptop. 06d has no Part B of its own: 06c's Part B (R17,
> R18 and the census) runs after it and measures 06c and 06d together.

**Subject:** prompt 06d §1–§3 with its §5 amendment, on the laptop, in the main checkout
`/Users/ds283/Documents/Code/SecondaryGWKit`, branch `handover-remedial`.
- **Dispatch SHA:** `4bbeade`; the code there is 06c's (`ad9a5f4`). One orchestrator commit landed
  while the work ran: `da043b9`, the prompt's §5 amendment, after this agent's stop (below). It
  touches only prompt and board files. `git log --oneline 4bbeade..HEAD` before staging shows it
  and nothing else. The stashes were not touched, and `var/levin-phase-floor-07/` was not read.
- **The gate** passed at dispatch: `git status --short` was empty, and `runregistrykit list` showed
  nothing `running` or stale (61 runs: 59 finished, 2 `unknown`, pre-registry). The seven roots'
  counts at the dispatch SHA were the dispatch's (AdaptiveLevin 53, ComputeTargets 780,
  CosmologyModels 39, LiouvilleGreen 148 (1 skipped), ObjectFactories 280, RunDrivers 127, tools
  58). The gate was not re-run on resumption (§5).
- No Ray was started, no run was registered, no store was opened, and nothing was written under
  `var/`.

**Commit:** `3d70db9` (recorded by the orchestrator in the residue commit), reviewed and pushed
2026-10-29; landed by the user with the near-singular cases recorded on
`[06c-budget-can-stop-on-a-pre-asymptotic-residual]` (the board's Decisions, **06d lands**).

**Result:** every fallback (Clenshaw–Curtis) solve carries a round-off floor derived from what the
rule computes, three terms with no $1/\theta'$, in place of eq. (151); the depth-18 history log no
longer divides by zero. Tests 1–8 hold; four breakages are each caught; the 20 seeded
LiouvilleGreen draws pass under both codes. One stop on the way (§5's amendment). One finding for
the user, not a stop: 203 near-singular `test_YJJ_log_singularity` cases that 06c reported
unconverged behind its inflated floor are now reported converged and dishonest (Observations 1).

## What shipped

**The floor** (`AdaptiveLevin/levin_quadrature.py`):

$$\phi_{\rm CC} = (n + 4)\,u\,S \;+\; u\sum_k |w_k| \sum_i |f_{ik}|\,|a_{ik}|
\;+\; \sum_k |w_k|\,\delta \sum_i \bigl(\max|\theta'|\,|f_{ik}| + |f'_{ik}|\bigr),
\qquad S = \sum_k |w_k| \sum_i |f_{ik} b_{ik}|,$$

with $u = 2^{-53}$ (`_UNIT_ROUNDOFF`), $n = 2N - 1$ the fine rule's points.
- `_cc_roundoff_floor(weights, f, basis, arguments, node_error, theta_prime_bound, f_prime)` forms
  it; `_adaptive_levin_subregion_cc` calls it from the samples it already holds, and adds the
  declared-phase term unchanged. `"phase_err"` is $\phi_{\rm CC}$ + declared.
- **The count $(n+4)$** (`_CC_ROUNDOFF_PER_TERM_COUNT = 4`, comment term by term): the $n$-term
  dot product's $\gamma_n$, then one $u$ each for the product of weight and integrand value, the
  weight's own computation, the basis evaluation and the amplitude's last bit.
- **The argument term:** the argument handed to sin and cos, from a new
  `_Basis_SinCos.eval_basis_and_arguments(x)`, which returns the basis values (bit-identical to
  `eval_basis`'s) and one argument per component from a single evaluation of the phase.
- **The node term** (§5 point 1). $\delta$ = `_cc_node_error_bound(x_span)` =
  $c_x\,u\,(\max(|a|,|b|) + h)$, $c_x$ = `_CC_NODE_ROUNDING_COUNT` = 5. The count, as the comment
  states it, from how `chebyshev_matrices` and `_chebyshev_base` form
  $\hat x_k = {\rm fl}({\rm fl}({\rm fl}(a+b)/2) + {\rm fl}({\rm fl}({\rm fl}(b-a)/2)\,\hat t_k))$,
  $\hat t_k = {\rm fl}(\sin({\rm fl}({\rm fl}(\pi j)/(2(N-1)))))$:
  - $\hat t_k$: $\pi$'s representation, the product and the quotient, $3u$ relative on an
    argument $|z| \le \pi/2$, so $\le 4.7u$ absolute, plus the sine's 1 ulp ($2u$): $\le 7u$;
  - the midpoint: $u\,|a+b|/2 \le uM$ (halving exact); the half-width: $u\,h/2$;
  - the product $(h/2)\hat t_k$: $uh/2 + 7uh/2 + uh/2 = 4.5\,uh$;
  - the final sum: $u|x_k| \le uM$;
  - total $2uM + 4.5uh \le 5u(M + h)$.

  $|\theta'|$ is bounded by its maximum over the coarse `theta_prime_Cheb` (the conservative side,
  §5's choice); $f'$ is the fine grid's own Chebyshev differentiation matrix applied to the samples
  in hand, per node. The $f'$ part is kept; Section F of `parent_demo.txt` shows its size (below).
- **No $1/\theta'$** anywhere: $\theta'$ enters only as a factor of the node term.

**The guard.** The depth-18 history log is factored into `_log_depth_history`, which takes each
improvement through `_history_improvement(previous, current)`: `None` when the denominator is zero
or not finite, logged as "not defined". Logging only.

**Docstrings and comments** (§0.3): the module docstring, `_LEVIN_ROUNDOFF_SAFETY`'s comment,
`_roundoff_floor` (Levin regions only since 06d, and why), `used_interval`'s comment, `phase_err`
and `abserr_roundoff`, `_adaptive_levin_subregion_cc`'s docstring and comments,
`adaptive_levin_sincos`'s `theta_mod_2pi` and `abserr_roundoff` descriptions.

**Unchanged** (§0.3): eq. (151), `_roundoff_floor`, `_levin_G0_G1`,
`_LEVIN_ROUNDOFF_G0_NOISE_FLOOR` on the Levin branch; the declared term; 06b's premise and
$\rho$; 06c's loop, budget and trust check; the C2 gate; the result dict's keys; every caller.
`_adaptive_levin_subregion_cc`'s signature is unchanged; its `theta_prime_Cheb` now feeds the node
term.

**Tests.** A new module, `AdaptiveLevin/tests/test_levin_cc_floor.py`, 7 methods (tests 1–6).
Two existing tests changed, both by §5: 06b's test 3 and the three-Bessel refinement test (listed
under Verification).

**Records.** `docs/levin-phase-floor/06d/calibration/`:
- `scripts/prov.py` (06c's, with the parent moved to `4bbeade`), `scripts/parent_demo.py` →
  `out/parent_demo.txt`, `scripts/lg_seeded_draws.py` → `out/lg_seeded_draws.jsonl` and
  `out/lg_seeded_draws_report.txt`, `scripts/breakages.py` → `out/breakages.txt`,
  `scripts/stop_node_rounding.py` → `out/stop_node_rounding.txt` (the stop's measurement, re-run
  on the final tree: its two-term $\phi$ is computed in the script, so it reproduces the stop's
  table exactly).
- The commands, from the repository root:
  ```bash
  PYTHONPATH=. ./venv/bin/python docs/levin-phase-floor/06d/calibration/scripts/parent_demo.py
  PYTHONPATH=. ./venv/bin/python docs/levin-phase-floor/06d/calibration/scripts/lg_seeded_draws.py run --processes 8
  PYTHONPATH=. ./venv/bin/python docs/levin-phase-floor/06d/calibration/scripts/lg_seeded_draws.py report
  PYTHONPATH=. ./venv/bin/python docs/levin-phase-floor/06d/calibration/scripts/breakages.py <a clean copy of AdaptiveLevin/levin_quadrature.py>
  ```
- Each output opens with its provenance (script and input hashes, `HEAD`, the dirty tree). The
  parent is the dispatch SHA's own module, read with `git show` (`prov.load_parent`).

## Predictions and verdicts

**06d has none of its own.** 06c's §4 Y1 and §5 X1–X5 score 06c and 06d together, in 06c's Part B,
which runs after this commit. Nothing is scored here.

## The stop, and §5's amendment

Part A first built §1 as §0.2 wrote it (two terms) with the guard and tests 1, 2, 3 and 5, and
**stopped under §4 before committing**: $\phi_{\rm CC}$ failed to bound test 2's reduced-phase
cells. On $\theta = cx$ on $[1, 1+10/c]$, one fallback solve, the reduced phase exact in `mpmath`,
true/$\phi_{\rm CC}$ was 6.7 at $c = 10^3$, $2.4\times10^3$ at $10^6$, $6.4\times10^5$ at $10^9$
and $3.4\times10^9$ at $10^{12}$ ($f = 1$); eq. (151) was short by the same amounts. The cause
was measured: the rule evaluated in `mpmath` at the float nodes against the exact nodes reproduced
each observed error to every printed digit (e.g. $+9.275\times10^{-17}$ both, at $c = 10^{12}$),
while the exact-node rule's own error was $10^{-19}$–$10^{-28}$. §0.2 omitted the rounding of the
abscissae. A control family ($\theta = \Theta + 10x$ on $[0, 1]$) was honest raw and reduced. It
also showed at test 3's level 2 (true $1.76\times10^{-18}$ against pair $1.38\times10^{-18}$ and
$\phi_{\rm CC}$ $5.1\times10^{-19}$). At the same stop, the three-Bessel refinement test failed
under 06d (it passes under 06c only because the inflated floor froze the value), and 06b's test 3
no longer fired its exit. Changes were left uncommitted; the user agreed all three proposals
(*"I agree with all three"*), written into the prompt as §5 (`da043b9`), and the work resumed on
them: the node term, the three-Bessel test restated, 06b's test 3 moved, and test 3's parent
assertion withdrawn.

## Deviations from the prompt

1. **The argument comes from a new basis method,** `eval_basis_and_arguments`, returning one
   argument per component, so the argument term is $u\sum_k|w_k|\sum_i|f_{ik}||a_{ik}|$; for
   (sin, cos) it equals §0.2's $u\sum_k|w_k||a_k|\sum_i|f_{ik}|$. It evaluates the phase once per
   node where `eval_basis` evaluated it twice. **IMPLEMENTATION CHOICE.**
2. **$\delta$ takes §5's single-constant form, $5u(M + h)$,** although the count gives
   $2uM + 4.5uh$: conservative by 2.5 on the $M$ part. **IMPLEMENTATION CHOICE.**
3. **$|\theta'|$ as its maximum over the coarse sample, $f'$ per node.** §5 left the choice; the
   maximum is the conservative side and is the only $\theta'$ the solve holds at the odd fine nodes.
   **IMPLEMENTATION CHOICE.**
4. **Test 1's "of order $uS$" became "bounded by the three terms".** With the node term,
   $\phi_{\rm CC}$ on test 1's region is $5.21\times10^{-15}$, of which the node term is
   $5.21\times10^{-15}$ and $(n+4)uS$ $5.3\times10^{-18}$. The test asserts finite, at least
   $(n+4)uS$, and at most the three terms with their arguments bounded from outside.
   **STRUCTURALLY REQUIRED** by §5 point 1.
5. **Test 2 has a second family,** (O) $\theta = \Theta + 10x$ on $[0, 1]$, $\Theta = 1$…$10^{12}$,
   beside (L) $\theta = cx$ on $[1, 1 + 10/c]$. With the node term, (L)'s raw cells are bounded by
   the node term alone ($5uc$ against the argument term's $u\,c\,x$), so breakage (ii) could not
   fail on them; (O)'s raw cells have large $|\theta|$ and small $\theta'x$, and do.
   **IMPLEMENTATION CHOICE.**
6. **Test 2 is asserted at two levels:** one solve, true error against the floor alone; and the call
   at `atol` $10^{-40}$, `rtol` $10^{-17}$, honest. The parent is tabulated at the solve level
   only: at the call level 06c's infinite floors on (L) would refine towards $2^{20}$ leaves per
   cell. **IMPLEMENTATION CHOICE.**
7. **Test 3's "a few times" is 2×** (measured 0.67–0.97 of the leaf's). **IMPLEMENTATION CHOICE.**
8. **Test 5 tests the factored helper.** No natural depth-18 problem was searched for.
   **IMPLEMENTATION CHOICE** (the prompt's stated alternative).
9. **The parent is reconstructed in-process in the tests** (`_parent_cc`: the working tree's
   fallback solve with the parent's eq. (151) floor plus the declared term, through the unchanged
   helpers). `parent_demo.txt` section A checks it equal, by `_call_signature`, to the parent's own
   module on all 26 driven problems that do not raise. **IMPLEMENTATION CHOICE.**
10. **Test 6's set of parent calls with a fallback leaf is written into the test** (five problems,
    from section A), so that the suite does not run 06c's 96,941-leaf oracle (20–34 s). Those five are
    asserted honest under 06d and tabulated against the parent; the other 21 are asserted
    bit-identical to the parent. **IMPLEMENTATION CHOICE.**
11. **06b's test 3 (b) is inside the same test method as (a),** with its constants beside test 3's;
    no new test method in that module. **IMPLEMENTATION CHOICE.**
12. **The draws ran 1,335 s and were not registered** with `runregistrykit`: the dispatch forbids
    writing under `var/`, where the registry records. They ran in this session, in the background.
    **STRUCTURALLY REQUIRED** by the dispatch.
13. **The breakages were applied by a script run from the session scratchpad,** copied
    byte-identical to `scripts/breakages.py`; its output's provenance header was added by hand,
    since the script writes none; likewise `stop_node_rounding.py` (the stop's scratch script,
    re-run on the final tree). **IMPLEMENTATION CHOICE.**
15. **The stop's other scratch measurements and the exit search for 06b's test 3 (b) are not
    kept:** the call-level two-term cells and the node-term variant run before §5 (whose numbers
    went to the orchestrator and into §5), and the exit search (its parameters and outcomes are
    listed under test 8). Each depended on code that no longer exists, or was superseded by
    `parent_demo.txt`. **UNINTENDED DRIFT** against README §5 rule 5's provenance, for exploratory
    runs only; every number this log relies on is in `calibration/out/`, except the exit search's
    ratios listed under test 8.
14. **The three-Bessel test's "it improved" is $|{\rm err}(10^{-8})| > 10\,|{\rm err}(10^{-14})|$**
    (§5 point 2 says "improves from `rtol` $10^{-8}$ to the tightest rung"). **IMPLEMENTATION
    CHOICE.**

## Verification performed

**Test 1** (`test_floor_finite_at_a_stationary_node`): $\theta = -10^6(x-\tfrac12)^2$ on
$[\tfrac12, \tfrac12 + 2^{-8}]$, $\theta'$ exact. The parent's floor is $+\infty$; 06d's is
$5.21\times10^{-15}$ = $5.3\times10^{-18}$ (sum) + $5.9\times10^{-19}$ (argument) +
$5.21\times10^{-15}$ (node, $\theta'$) + $6.7\times10^{-19}$ (node, $f'$). Honest (true
$2.8\times10^{-17}$, pair $1.18\times10^{-8}$). **Passes.**

**Test 2** (`test_floor_bounds_a_single_solve`, `test_calls_below_roundoff_are_honest`), from
`parent_demo.txt` section B; true/$\phi$ for one solve, and for the call:

| family, phase | scale | true | $\phi_{\rm CC}$ | eq. (151) | true/$\phi$ | true/(151) | call: true/abserr |
|---|---|---|---|---|---|---|---|
| L raw | 1 | 5.6e-17 | 9.8e-15 | 9.8e-15 | 0.0057 | 0.0057 | 0.015 |
| L raw | 1e3 | 9.1e-17 | 2.5e-15 | 9.8e-18 | 0.037 | **9.3** | 0.018 |
| L raw | 1e6 | 1.9e-17 | 2.5e-15 | 9.8e-21 | 0.0079 | **2.0e3** | 0.0079 |
| L raw | 1e9 | 2.0e-18 | 2.5e-15 | 9.8e-24 | 0.00082 | **2.1e5** | 0.00082 |
| L raw | 1e12 | 7.5e-17 | 2.5e-15 | 9.8e-27 | 0.030 | **7.6e9** | 0.030 |
| L reduced | 1 | 5.6e-17 | 9.8e-15 | 9.8e-15 | 0.0057 | 0.0057 | 0.015 |
| L reduced | 1e3 | 7.1e-17 | 2.1e-15 | 9.8e-18 | 0.034 | **7.3** | 0.018 |
| L reduced | 1e6 | 2.4e-17 | 2.0e-15 | 9.8e-21 | 0.012 | **2.4e3** | 0.012 |
| L reduced | 1e9 | 6.6e-18 | 2.0e-15 | 9.8e-24 | 0.0033 | **6.8e5** | 0.0033 |
| L reduced | 1e12 | 3.4e-17 | 2.0e-15 | 9.8e-27 | 0.017 | **3.5e9** | 0.017 |
| O raw | 1 | 1.4e-17 | 9.7e-15 | 2.7e-15 | 0.0014 | 0.0052 | 0.0084 |
| O raw | 1e3 | 1.7e-15 | 8.0e-14 | 2.7e-15 | 0.022 | 0.66 | 0.0087 |
| O raw | 1e6 | 4.7e-12 | 7.0e-11 | 2.7e-15 | 0.066 | **1.7e3** | 0.066 |
| O raw | 1e9 | 4.2e-9 | 7.0e-8 | 2.7e-15 | 0.060 | **1.6e6** | 0.060 |
| O raw | 1e12 | 1.4e-6 | 7.0e-5 | 2.7e-15 | 0.020 | **5.3e8** | 0.020 |
| O reduced | 1–1e12 | 4.2e-17–5.6e-17 | 9.4e-15–9.6e-15 | 2.7e-15 | ≤ 0.0059 | ≤ 0.021 | ≤ 0.0065 |

Every cell is bounded, raw and reduced; the parent's eq. (151) understated 11 of the 20 cells.
In (L) the node term carries the floor (raw and reduced alike: about $2.0\times10^{-15}$); in (O)
raw the argument term does ($u\Theta\sum|w||f|$). Calls stop at depth $\le 2$, every leaf
floor-limited. **Passes.**

**Test 3** (`test_nested_leaves_round_a_stationary_point`), section C:

| level | span | parent eq. (151) | $\phi_{\rm CC}$ | halves / leaf | pair | true |
|---|---|---|---|---|---|---|
| 0 | [0.72265625, 0.7265625] | 1.98e-16 | 4.55e-15 | 0.668 | 9.1e-10 | 2.3e-17 |
| 1 | [0.72265625, 0.724609375] | 9.04e-17 | 7.73e-16 | 0.966 | 3.3e-16 | 2.7e-18 |
| 2 | [0.72265625, 0.7236328125] | 6.63e-16 | 3.60e-16 | 0.741 | 1.4e-18 | 1.8e-18 |
| 3 | [0.72314453125, 0.7236328125] | 5.31e-16 | 8.69e-17 | 0.732 | 2.6e-19 | 1.0e-19 |

The parent's floors are recorded, not asserted (§5 point 4). 06d's halves sum to at most 0.97 of
the leaf's, and every solve is honest; at level 2 the node term is what makes it so. **Passes.**

**Test 4** (`test_stationary_phase_oracle`), section D. At the parent: 96,941 leaves (96,935
fallback), depth 20, 218,735 solves, 12,427 phase-limited, unconverged, abserr
$4.27\times10^{-9}$, true $2.7\times10^{-15}$, 20 s (34 s on the first run). Under 06d: **76 leaves (12 fallback), depth 14,
607 solves, 8 phase-limited, converged**, abserr $2.52\times10^{-11}$, true $1.9\times10^{-14}$,
0.1 s. Honest, abserr finite. **Passes.**

**Test 5** (`test_zero_abserr_in_history`): a depth-18 history with a zero and an infinite entry
completes; both improvements are logged "not defined". **Passes.**

**Test 6** (`test_nothing_else_moves`), section A. The in-process parent equals the parent's own
module on all 26 problems. 21 problems have no fallback leaf at the parent and are bit-identical
under 06d (value, abserr, evaluations, every region). Five have one:

| problem | parent: leaves, depth, solves, abserr, true, converged | 06d: the same |
|---|---|---|
| GRZIntegral(10) | 11, 4, 87, 3.08e-13, 5.6e-17, True | 11, 4, 87, 3.00e-13, 5.6e-17, True |
| GRZIntegral(100) | 102, 7, 407, 1.50e-14, 1.2e-16, False | 92, 7, 367, 4.06e-14, 5.2e-17, False |
| GRZIntegral(1000) | 382, 11, 1527, 6.48e-15, 7.3e-17, False | 228, 10, 911, 1.53e-13, 4.6e-16, False |
| stationary_phase | 96,941, 20, 218,735, 4.27e-9, 2.7e-15, False | 76, 14, 607, 2.52e-11, 1.9e-14, True |
| mode_filter_tight | 1,279, 15, 10,231, 4.81e-15, 4.0e-17, True | 32, 5, 127, 7.35e-14, 5.1e-16, **False** |

All honest under both. **Passes.**

**Test 7, the LiouvilleGreen draws** (`lg_seeded_draws_report.txt`; 160 units, wall time 1,335 s
on 8 processes; parent = `4bbeade`'s module, 06c's code).
- **Outcomes:** all 80 (seed, test) pairs pass under both. **No stop.**
- **Calls:** 4,320 under each, 475 bit-identical. Converged (parent, 06d): T,T 3,254; **T,F 0**;
  F,T 556; F,F 510. Regions 1,774,775 → 804,043; solves 5,936,892 → 3,375,592. Calls reaching
  depth 20: 230 → 101 (every one of the 101 also at the parent). Running-test disagreements 0.
- **The honesty table** at the integral level (each printed case against its closed form), beside
  06c's own (06c's log, test 10, where the parent was 06b's code):

  | test | cases | 06b (06c's parent) | 06c | **06d** |
  |---|---|---|---|---|
  | `test_YJJ_log_singularity`: dishonest | 800 | 259 | 235 | **438** |
  | — dishonest and converged | | 44 | 228 | **431** |
  | — worst true/reported (all; among converged) | | — | 5.46e5; 5.46e5 | **8.13e5; 8.13e5** |
  | `test_JJJ`: dishonest; worst ratio | 100 | 0 | 0; 0.336 | 0; 0.336 |
  | `test_YJJ`: dishonest; worst ratio | 40 | 0 | 0; 0.388 | 0; 0.410 |

  203 cases are honest at 06c and dishonest under 06d, all flagged converged under 06d; none goes
  the other way. All 203 are $(Y_0, J_0, J_0)$ near the logarithmic singularity: $\epsilon =
  10^{-4}$ (1), $10^{-5}$ (4), $10^{-6}$ (40), $10^{-7}$ (38), $10^{-8}$ to $10^{-10}$ (40 each).
  true/reported under 06d: median 260, 90th percentile $4.2\times10^4$, max $9.6\times10^4$; at 06c
  median $2.6\times10^{-4}$, max 0.014. **The true error barely moves** (seed 0 #10:
  $1.283\times10^{-8}$ at 06c, $1.301\times10^{-8}$ under 06d): 06c declared $2.2\times10^{-5}$ and
  `converged` False, the inflated floor; 06d declares $6.7\times10^{-9}$, `converged` True. See
  Observations 1.
- `test_abserr_bounds_truth` bounds on 7 of 7 under both. Largest printed relative error:
  `test_JJJ` $1.71\times10^{-11}$ / $2.90\times10^{-11}$; `test_YJJ` $3.39\times10^{-8}$ /
  $3.37\times10^{-8}$; `test_YJJ_log_singularity` $2.93\times10^{-4}$ both.

**Test 8, existing tests.** Every test in `AdaptiveLevin/tests/` and `LiouvilleGreen/tests/`
passes. Two changed, both by §5:

1. `test_levin_roundoff_exit.test_exit_against_its_premise_fallback_branch` (§5 point 3), on its
   problem (a), $\theta = 10^{-6}e^x$ on $[0, 6]$ (section E):

   | | 06c: bypass | 06c: premise | 06d: bypass and premise (bit-identical) |
   |---|---|---|---|
   | regions (all fallback) | 1 | 4 | 2 |
   | depth | 0 | 2 | 1 |
   | phase-limited | 1 | 4 | 1 |
   | abserr; converged | 1.03e-5; False | 4.59e-7; False | 1.98e-12; True |
   | true | 0 | 0 | 1.4e-17 |

   Assertions changed: the bypass's `num_regions` 1 → 2 and `num_simple_regions` 1 → 2 (region
   count, which rule), `regions[0].type == "direct"` → every region direct (which rule),
   `regions[0].depth == 0` → `max_depth == 1` (depth), `regions[0].phase_limited` →
   `num_phase_limited_regions == 1` (phase-limited count on a fallback region); the new code's
   depth, rule and premise assertions → `_call_signature` equal to the bypass's (bit-identity
   across the bypass). Honesty kept for both. The assertion that the premise check at least halves
   the declared error **moves to problem (b)**: $\theta = 0.025\,e^{x - X_0}$,
   $f = (e^{-2(x-X_0)}, 0)$ on $[X_0, X_0 + 6]$, $X_0 = 10^6$, `atol` $10^{-30}$, `rtol`
   $10^{-16}$, where 06d's node term binds. Under 06d the bypass accepts the root at depth 0
   (abserr $3.35\times10^{-9}$, true $9.7\times10^{-14}$); the premise check bisects, and the exit
   fires at depth 2 on 3 leaves whose premise holds (abserr $6.96\times10^{-10}$, 0.21 of the
   bypass's, true $2.1\times10^{-13}$); both honest. (At 06c, (b) refines to 160 leaves, depth 12,
   identically with and without the bypass.) The search, all with $\theta = A e^{x - X_0}$ and
   $f = e^{-\gamma(x - X_0)}$ on $[X_0, X_0 + L]$: at $X_0 = 10^3$ and $10^4$ ($A = 0.04$, $\gamma
   = 2, 3$, $L = 6$; $A = 0.3$, $\gamma = 3$, $L = 4$) the exit fired only below the root, where the
   premise held, identically across the bypass; at $X_0 = 10^6$ and $10^8$ with $A = 0.025$,
   $\gamma = 2$, $L = 6$ the root's pair estimate is below its floor and the premise fails there
   (abserr 0.21 of the bypass's at both); with $A = 0.04$, $\gamma = 3$ or $\gamma = 1$ the ratio was
   0.84–1.0. The search scripts were scratch and are not kept; section E records the chosen
   problem.
2. `test_three_bessel.TestPhaseGroups.test_quadrature_refinement_is_not_a_certificate_of_phase_accuracy`
   (§5 point 2), section G, true error / abserr per rung:

   | rtol | 06c | 06d |
   |---|---|---|
   | 1e-8 | 1.28e-10 / 1.66e-9 | 1.28e-10 / 1.42e-9 |
   | 1e-10 | 1.01e-14 / 2.33e-10 | 2.88e-12 / 1.88e-11 |
   | 1e-12 | 1.01e-14 / 2.33e-10 | 3.72e-14 / 2.80e-13 |
   | 1e-13 | 1.01e-14 / 2.33e-10 | 9.74e-15 / 4.20e-14 |
   | 1e-14 | 1.01e-14 / 2.33e-10 | 1.01e-14 / 3.82e-14 |

   `phase_limited` at every rung under both. Removed: `coarse_gap > 10·floor_gap` and
   `floor_gap < 1e-14·|analytic|` (the second fails under 06d: spread $4.7\times10^{-14}$ against
   $1.7\times10^{-15}$). Added: abserr ≥ true error at every rung, and the true error at
   $10^{-8}$ above ten times that at $10^{-14}$. Kept: `phase_limited` at $10^{-14}$ (it holds),
   and true error at $10^{-14}$ above $10^{-14}|{\rm analytic}|$. Docstring rewritten.

06c's `test_levin_global_budget` passes unchanged. Its docstrings quote 06c's own measurements at
its commit, and were left as written (§2 test 8 allows, does not require, updating them).

**Breakages** (`out/breakages.txt`), each applied, `test_levin_cc_floor` run, then reverted
(the module's SHA-256 checked equal afterwards):
- **(i)** the fallback's floor back to eq. (151): caught by **test 1** (floor $+\infty$), and by
  tests 2 (12 solve cells, 6 call cells) and 3.
- **(ii)** the argument term removed: caught by **test 2 on raw-phase cells**, (O) at
  $\Theta = 10^6, 10^9, 10^{12}$ (true $4.7\times10^{-12}$ against a floor of
  $9.2\times10^{-15}$ at $10^6$).
- **(iii)** the guard removed: caught by **test 5** (`ZeroDivisionError`).
- **(iv)** the node term removed: caught by **test 2 on reduced-phase cells**, (L) at
  $c = 10^3$…$10^{12}$, solve and call, and by test 3 at level 2.

**The $f'$ part** (section F), over 437 fallback solves of the problems above: its share of the
node term has median 0.0017, and of $\phi_{\rm CC}$ median 0.0013. It dominates where $\theta'$ is
tiny and $f$ steep: 06b's test 3 (a) ($f = e^x$, $\theta' \sim 10^{-6}e^x$), where it is the whole
floor ($6.4\times10^{-14}$ against $1.3\times10^{-18}$ for the $\theta'$ part), and 06b's (b)
($5.5\times10^{-10}$ of $6.9\times10^{-10}$). Kept, as §5 requires.

**The seven roots** after the change (before: the dispatch's):

| root | before | after |
|---|---|---|
| AdaptiveLevin | 53 | 60 |
| ComputeTargets | 780 | 780 |
| CosmologyModels | 39 | 39 |
| LiouvilleGreen | 148 (1 skipped) | 148 (1 skipped) |
| ObjectFactories | 280 | 280 |
| RunDrivers | 127 | 127 |
| tools | 58 | 58 |

All seven OK. AdaptiveLevin rises by 7, the methods of `test_levin_cc_floor.py`; the others are
unchanged. `parent_demo.txt` is its second run, on the final test files; it differs from the first
only in wall times.

`black` is clean on the four changed or added files outside `calibration/scripts/`.

## Observations not acted on

1. **Near-singular three-Bessel cases are now reported converged and dishonest.** In the draws,
   203 `test_YJJ_log_singularity` $(Y_0, J_0, J_0)$ cases at $\epsilon \le 10^{-4}$ go from honest
   and unconverged at 06c to converged and dishonest under 06d (median 260×, max
   $9.6\times10^4$); dishonest-and-converged rises 228 → 431 of 800. Their true errors do not
   move: 06c covered them with the inflated eq. (151) floor ($10^{-5}$–$10^{-4}$), not with an
   estimate of the error. The error is outside the quadrature: §0.4 puts input conditioning
   outside any round-off floor, and `[06c-budget-can-stop-on-a-pre-asymptotic-residual]` already
   records these cases as input conditioning (44 → 228 under 06c). No issue opened or amended
   here, since this prompt's board edits are its own row; **for the orchestrator:** the count in
   that issue's record moves to 431, and the `converged` flag on these cases is no longer guarded
   by an accidental floor.
2. **The node term is a worst-case bound, and can unconverge a call.** `mode_filter_tight` goes
   from converged (1,279 leaves, abserr $4.8\times10^{-15}$, true $4.0\times10^{-17}$) to
   unconverged (32 leaves, abserr $7.4\times10^{-14}$, true $5.1\times10^{-16}$): its leaves are
   floor-limited on a node term about 100× its realised error, since the per-node rounding errors
   largely cancel in the sum. Test 2's true/$\phi_{\rm CC}$ runs 0.0008–0.066. Summed over leaves
   the term is about $5u\,|x|\,\|f\|\,{\rm TV}(\theta)$; where $|x|$ is large (production's
   $\log(1+z)$ up to ~35), it may bind in 06c's census. Not changed: §5 fixes the form.
3. **The three-Bessel draws got cheaper:** solves 5.94 M → 3.38 M and 230 → 101 calls at depth 20,
   converged 3,254 → 3,810, no call converged → unconverged. Cost is not an argument (README §5
   rule 4); recorded for 06c's Part B, which will see the same direction.
4. **`prov.load_parent` writes the parent's source to the system temporary directory** (Python's
   `tempfile`), as 06c's did; not under `var/` and not in the tree.
5. **`_TwoIndependentSinCosBasis`** (a test stand-in in `test_levin_quadrature.py`) has no
   `eval_basis_and_arguments`; it never reaches the fallback (its own comment says so), so this is
   not exercised. A future basis with a fallback path must supply the method.

## State handed on

- **This commit** holds the code, the tests and these records. 06c's Part B (R17, R18, the census)
  runs next on the Mac Studio and measures 06c and 06d together; Y1 and X1–X5 are scored there.
- **For Part B:** every fallback leaf's floor moves, so every call with one can move; expect calls
  whose leaves 06c chased towards depth 20 at stationary points to stop much earlier, and some that
  converged on a fine partition to be floor-limited on the node term (Observations 2). Where
  $\theta'$ never vanishes and no fallback leaf exists, nothing moves.
- **For 07's rebase:** its window pieces are fallback solves and inherit $\phi_{\rm CC}$; on 07's
  window leaf the floor no longer grows towards $x_s$ (test 3).
- **The header issues** `[07-cc-solves-use-the-levin-roundoff-floor]` and
  `[07-depth18-diagnostic-divides-by-zero]` are not moved here: they move in 06c's records commit
  (§3).
