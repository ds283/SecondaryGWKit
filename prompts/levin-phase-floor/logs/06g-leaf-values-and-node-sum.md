# levin-phase-floor 06g — the trusted leaf's value, the node errors signed, and the round-off exit's premise

> **§1–§3 only** (the code, the tests, the Filon tool and these records), written by the executing
> agent on the laptop. §4 (rehearsals R23 and R24) and §5 (the census and the references) run later
> on the Mac Studio; the orchestrator appends their record to this log.

**Subject:** prompt 06g §1–§3, on the laptop, in the main checkout
`/Users/ds283/Documents/Code/SecondaryGWKit`, branch `handover-remedial`.
- **Dispatch SHA:** `4f4f9ed`; the code under `AdaptiveLevin/` there is 06e's (`f4d8e17`).
  `git log --oneline 4f4f9ed..HEAD` was empty at the start and before staging. The stashes were not
  touched, and `var/levin-phase-floor-07/` was not read.
- **The gate** held, as the orchestrator checked it at the dispatch SHA (06h's row ✅,
  `git status --short` empty, `runregistrykit list` nothing `running` or stale; re-listed at the
  start: 61 runs, 59 finished, 2 `unknown`, pre-registry). The seven roots at `4f4f9ed` are the
  orchestrator's (Verification).
- No Ray was started, no run was registered, no store was opened, and nothing was written under
  `var/`.

**Commit:** the code commit of this log (one commit, not pushed).

**Result:** the four remedies of §0.2 (a)–(d) are in `_adaptive_levin`, and the Filon tool of (e) is
`tools/filon_reference.py`. A trusted leaf returns its quarters sum; a leaf is trusted only at
$r < 1/4$, or $r < 1$ with $\Delta_1$ within its floor; the fallback leaves' node errors are carried
as $|\sum E|$, with a node-limited stop; the round-off exit's halving test leaves the declared term
out and does not fire where the split could meet the budget. Tests 1–8 pass, the five breakages are
caught, and the twenty LiouvilleGreen draws pass under both codes. Two existing assertions changed,
each of an allowed kind. **One deviation is of substance and is the first thing to review
(deviation 1, the budget clause).** One issue opened,
`[06g-node-limited-stop-can-end-a-call-refinement-would-converge]`.

## What shipped

**`AdaptiveLevin/levin_quadrature.py`.**
1. **(a), the finest value.** `_check_leaf` sets a trusted leaf's value to
   $Q = (V_{LL} + V_{LR}) + (V_{RL} + V_{RR})$, summed in that order, and moves the running value sum
   (`_ExactSum`) by $Q - V_P$ exactly (`_set_value`: subtract the old, add the new).
   `_exempt_floor_limited` restores $V_P$ the same way. The declared resolution is unchanged,
   $\max(\epsilon_P, \Delta_1)/(1-r)$. `used_interval` gains `value` (the returned value), `value_kind`
   (`"own"` or `"quarters"`) and `value_own` ($V_P$). An untrusted leaf, a floor-limited leaf and an
   unchecked leaf keep $V_P$.
2. **(b), slow leaves.** `_LEVIN_TRUST_MAX_RATIO = 0.25`, with the prompt's basis in its comment.
   Trusted iff $r < 1/4$, or $r < 1$ and $\Delta_1 \le \phi_P$ (`phase_err`, the whole floor). At
   `depth_max` an untrusted leaf declares $\max(\epsilon_P,\Delta_1)/(1-r)$ if $r < 1$ and
   $\max(\epsilon_P, \Delta_1+\Delta_2)$ if not.
3. **(c), the node errors signed.** The fallback solve returns `node_error_signed` ($E$),
   `node_error_bound` ($R_g + R_2 + R_c$) and `phase_err_less_node`, its floor formed with
   $R_g + R_2 + R_c$ alone plus the declared term (deviation 3). `used_interval.budget_err` is
   $e'_i$: `total_err` on a Levin leaf, $\max(\text{resolution}, \phi_i - |E_i|)$ + truncation on a
   fallback leaf. `_levin_leaf.error` is $e'_i$, so the running error sum, the candidates' order
   (deviation 2) and `_sum_leaves_ascending` use it. A running `_ExactSum` of $E$ is updated as
   leaves are added and removed. The call's error is $\sum e'_i + |\sum E|$, in the stop and the
   returned `abserr`. `_sum_leaves_ascending` returns $\sum e'_i$ (plain, ascending) and $\sum E$
   (`math.fsum`, ascending). The node-limited stop: when $\sum e'_i$ meets the request, every leaf is
   checked or floor-limited and no refinable leaf is untrusted, the call stops; it is converged if
   $\sum e'_i + |\sum E|$ meets the request too, else unconverged with `node_limited` True. The
   floor-limited test still compares the residual with the whole floor, $|E_i|$ included. New
   result keys: `node_error_signed_sum`, `node_error_abs_sum`, `abserr_node_signed`,
   `node_limited`. `abserr_roundoff` and the other breakdown keys are unchanged.
4. **(d), the premise.** Both branches return `phase_err_declared` and `phase_err_roundoff` (the
   floor less the declared term, as the solve added it, deviation 3).
   `_roundoff_exit_premise_holds(phase_err, phase_err_L, phase_err_R, split_could_meet_budget=False)`
   now takes the round-off parts and the budget clause. `_split_could_meet_budget` recomputes the
   call's error exactly (`_ExactSum.total_with`), with the leaf's $e'$ replaced by its halves'
   floors (`phase_err_less_node` on a fallback half, `phase_err` on a Levin half) and $\sum E$ by
   the halves' $E$. **It returns False when the call meets its request already (deviation 1).** An
   infinite half floor, or a non-finite error elsewhere, makes the recomputed error infinite, so the
   clause holds. The bypass skips both parts. The constant's comment, `phase_limited`'s docstring and
   the bypass comment say what the premise tests and why.
5. **Docstrings.** `_adaptive_levin`'s (the error formula, the four stops, the trust check and the
   value returned), `_check_leaf`'s ("Its value is unchanged" is gone), `used_interval`'s
   (`total_err`, `budget_err`, `trusted`, `declared_resolution`, the new fields),
   `_sum_leaves_ascending`'s, the fallback solve's return, and `adaptive_levin_sincos`'s return
   keys.

**`tools/filon_reference.py`** (new). 06i's `filon.py`, `route_a.py`, `route_b.py` and
`samples.py` as a module, its docstring stating the method, both routes, the count rule and what it
does not do. Its functions:
- `reading(call, impl, res, route=None, edges=None, workers=1, call_source=None)`;
- `reference(call, tolerance, ...)`, which returns the four readings, their scatter, `counts` and
  the FCC res-2 value;
- `counts`, `scatter`, `route_of`, `production_edges` (06i's closure walk);
- `linear_phase_integral(g, L, R, omega, impl, res)`, for a phase that is linear already;
- a command line, `python -m tools.filon_reference PICKLE --tolerance T [--unpickler
  ray.cloudpickle] [--workers N]`.

It imports neither Ray nor a datastore, and nothing under `ComputeTargets/`, `LiouvilleGreen/` or
`AdaptiveLevin/` imports it.

**Tests.** No Ray, no datastore. The parent is `git show 4f4f9ed:AdaptiveLevin/levin_quadrature.py`,
executed as `levin_quadrature_parent_4f4f9ed` (`test_levin_leaf_values.load_parent()`).

| Test | Where | Cell(s) and what it asserts | Key numbers (parent → 06g) |
|---|---|---|---|
| 1 | `test_levin_leaf_values.TestTrustedLeafReturnsQuarters.test_trusted_leaf_returns_quarters` | Seven cells, four Levin and three fallback, leaves to depth 4. On each: every trusted leaf's value is $Q$ bit for bit, recomputed from four fresh quarter solves, and `value_own` is $V_P$; other leaves return $V_P$; the call's value is the ascending sum. On the three cells where (b), (c) and (d) do not apply, `abserr`, `converged` and every region's span, depth and `total_err` equal the parent's. Both codes honest everywhere; the true error is at most the parent's everywhere. | True error: exp_sin_1e3 5.42e-19 → 3.25e-19; si_1000_exp 1.095e-14 → 1.096e-15; sinc_1e3 1.08e-18 → 5.42e-19; **gaussian_bump 8.563e-11 → 1.24e-13 (691×)**; **sqrt(1e-8) 3.06e-11 → 2.91e-14 (1,051×)**; sqrt(1e-10) 2.91e-14 → 0; runge 0 → 0 |
| 1 (extra) | `…test_noise_level_cell` | A cell where the true error rises, at rounding noise: the value moves by at most the trusted leaves' $\sum\Delta_1$; both honest | 0 → 5.55e-17 (1 ulp) against abserr 1.07e-13 (observation 3) |
| 2 | `test_levin_leaf_values.TestValueSumFollowsLeaves` (3 tests) | The running value sum (`_ExactSum` instance 2 of the driver's 3, recorded) equals `math.fsum` of the returned values, and the value is the ascending sum: (a) the bump, whose checked leaves are bisected; (b) a stand-in with a trusted root later bisected; (c) a stand-in with a trusted root later found floor-limited, its value reverting to $V_P$ | (b) root trusted, value `quarters`, bisected, 2 leaves returned; (c) 1 leaf, floor-limited, $r = 0.125$, value $1 + K$ |
| 3 | `test_levin_leaf_values.TestSlowLeaves` (3 tests) | SincIntegral (06c's 12b): the parent's leaf [1.38671875, 1.7734375] at $r = 0.9107$ with $\Delta_1 = 1.155\times10^{-15} > \phi_P = 1.317\times10^{-16}$ is trusted at the parent, untrusted under 06g, and refined before every trusted leaf (every other parent leaf returned unchanged; the replacements inside it); honest. A noise leaf at $r = 1/3$, $\Delta_1 = 1.04\times10^{-17} \le \phi_P = 4.04\times10^{-16}$ trusted. Every checked leaf obeys the trust test. The bump honest | 9 leaves, abserr 2.159e-13, true 5.2e-14 → 17 leaves (8 fallback), depth 12, 135 solves, abserr 2.074e-13, true 2.1e-15. Bump: abserr 8.564e-11 vs true 8.563e-11 → 1.24e-13 |
| 4 | `test_levin_node_sum.TestSignedNodeSum.test_signed_node_sum` | GRZIntegral(1000): `node_error_signed_sum` = `fsum(E)`, `node_error_abs_sum` = `fsum(|E|)`, `abserr` = ascending $\sum e'_i$ + $|\sum E|$, each $e'_i$ restated from the region's fields | 238 leaves, 232 fallback, 113 E > 0 / 119 E < 0; $\sum|E| = 1.297\times10^{-15}$, $|\sum E| = 5.20\times10^{-18}$ (250×); the parent's combination on the same partition 4.3015e-14 against 4.1724e-14 (1.031), and the parent's own run returns the same 238 leaves at 4.3015e-14 |
| 5 | `…test_node_limited_stop` | $\sin(0.9(x - 4\times10^7))$ on $[x_0, x_0+3.6]$, rtol 1e-9: node-limited, unconverged, below `depth_max`, $\sum e'_i$ within the request and the call not; honest. The parent runs deeper | 06g: depth 4, 5 leaves, 39 solves, $\sum e' = 1.278\times10^{-9}$, request 2.217e-9, $|\sum E| = 5.366\times10^{-9}$, abserr 6.645e-9, true 4.655e-9. **Parent: depth 11, 12 leaves (all floor-limited), 47 solves**, abserr 4.825e-9, unconverged. $|V - \sum E - {\rm ref}| = 7.1\times10^{-10}$ |
| 6 | `…test_floor_limited_keeps_own_node_error` | $\sin(1.25(x - 3\times10^8))$ on $[x_0, x_0+1.25]$, rtol 1e-10: the root's residual 7.997e-10 is within its whole floor 1.181e-9 and above its floor less $|E|$, 1.446e-11; it is floor-limited | abserr 1.181e-9 → 1.966e-9 (residual + $|E|$), true 1.167e-9; $|V - \sum E - {\rm ref}| = 1.1\times10^{-16}$ |
| 7 | `test_levin_roundoff_exit.TestRoundoffExitPremise06g` (3 tests, added) | (a) $\theta = 10^4 e^x$ on [0, 2], `theta_abserr` $= c\,e^{-5x}$, declared term 0.2500 of the root's floor, request 0.53 of it: the parent fires at depth 0 and misses; 06g bisects and converges; with no `theta_abserr` the two codes identical. (b) $\theta = 10^4 e^x$ on [0, 1], halves' floors 0.668 of the root's, request 0.8 of it: the parent fires and misses; 06g bisects and converges. (c) 06b's 2a and 2b: the exit fires on the root under both, results identical, values included. 06b's six tests unchanged and passing | (a) root floor 2.104e-15 (eq. 151 1.578e-15, declared 5.26e-16); halves whole 0.557, eq. 151 0.408, declared 1.005; request 1.115e-15; parent 1 leaf abserr 2.104e-15 unconverged → 3 leaves at depth 2, abserr 1.065e-15, converged. (b) parent 1 leaf, 3.218e-16 → 2 leaves, 2.149e-16, converged |
| 8 | `tools/tests/test_filon_reference.py` (5 tests) | R0's 16 cells at $\omega = 10^6, 10^9$, both implementations at both resolutions, within $10^{-13}\max|g|/\omega$; Route A against $s_f A[{\rm Ci}(|K|\eta_b) - {\rm Ci}(|K|\eta_a)]$ on an all-orders-½ stand-in with $K_{\rm lo} = -9.3\times10^{-12}$; Route B on $\Omega(e^x - 1)$ with jitter of rms $10^{-8}$ rad; the count rule; determinism (fresh process, and two spawned workers) | R0 worst 0.0722 of the tolerance (06i: 0.0875 over all 32 cells); Ci worst 0.03 of tolerance; Route B errors 0.25σ and 1.04σ of the declared 1σ fit noise (3.6e-12, 3.8e-12), every segment's σ within 2× the injected jitter |
| 9 | `docs/levin-phase-floor/06g/calibration/scripts/lg_seeded_draws.py` | Seeds 0–19, parent and 06g (below) | — |
| 10 | the seven roots | (below) | — |

**Two existing assertions changed**, each of §2 test 10's third kind, a count that (b) moves; the
numbers under both codes are in the comment beside each:
1. `test_levin_cc_floor.py` test 6 (`TestNothingElseMoves`): `SincIntegral` and `vectorized_sinc`
   move into `PARENT_HAS_FALLBACK_LEAF`. They had no fallback leaf at the parent and were compared
   bit for bit with the 06c-floor variant; under 06g the slow leaf is refined into fallback leaves
   (`num_simple_regions` 0 → 8; regions 9 → 17), so they are asserted to have one and to be honest,
   as the rest of that set are.
2. `test_levin_global_budget.py` test 12b (`test_trust_check_before_resolution`): "every region
   trusted, and every r < 1" became "exactly one region floor-limited, every other trusted, every
   trusted r < 1". This is a count of exit firings: 0 at the parent (9 regions, all trusted) and 1
   under 06g, the fallback leaf [1.652587890625, 1.6767578125] ($r = 2$, $\Delta_1 = 6.5\times10^{-19}$,
   rounding noise), found floor-limited and so exempt. 16 of 17 are trusted.

No convergence assertion moved.

**Calibration** (`docs/levin-phase-floor/06g/calibration/`):
- **`scripts/prov.py`.** 06h's, with the parent moved to `4f4f9ed` and `levin_quadrature.py` loaded.
- **`scripts/breakages.py`** → `out/breakages.txt`.
- **`scripts/cells.py`** → `out/cells.txt`: every cell of tests 1–7 at both codes, region by region.
  It also holds test 3's table over tests 1 and 3's cells, the budget clause as written against as
  built (deviation 1), and the convergence losses of the issue opened.
- **`scripts/lg_seeded_draws.py`** → `out/lg_seeded_draws.jsonl` and `out/lg_seeded_draws_report.txt`.
  This is 06h's harness, with the parent mode moved back to 06e's (the parent's driver) and 06g's
  keys and checked-leaf classes recorded.

## Predictions and verdicts

§4's and §5's predictions, copied before they run, **not yet scored** (they run on the Mac Studio).
Nothing in §1–§3 is a scored prediction.

- **V1 — consistency.** **Rule:** every row outside 06a's 720 stationary-point rows with finite
  `total_abserr` in both stores has $|{\rm total}_{24} - {\rm total}_{23}| \le {\rm
  total\_abserr}_{23} + {\rm total\_abserr}_{24}$, and the same on the oracle. Rows with an infinite
  error in either are counted separately; the stationary-point rows are tabulated, not scored.
  **Holds** if no row violates either. A violation is a finding for the user, not by itself a stop.
  — *not yet scored.*

**Gate V, before scoring:** — *not yet scored.*
- **The parent reproduces 06h.** At the parent the harness reproduces 06h's units **bit for bit**
  on `low-G+r-1` and `ctrl-low-1` at `1e-7`, and on `0-1971` at `1e-8`.
- **The integrand is 06h's.** `git diff --stat <06h's census tree c1ec3b2> <code commit> -- .
  ':!docs' ':!prompts'` touches only `AdaptiveLevin/`, `tools/`, test files and `CLAUDE.md`.
- **The new code is deterministic.** Two runs of one $g$ unit are bit-identical.

**Predictions, fixed before the run** (README §5 rule 3). "The stationary-point rows" are 06a's 720
(`spscan.py`).
- **V2 — the rows agree.** **Rule:** outside the stationary-point rows, every row with finite
  `total_abserr` under both has $|{\rm total}_g - {\rm total}_{06h}| \le {\rm
  total\_abserr}_{06h} + {\rm total\_abserr}_g$, and the same on the oracle. **Holds** if true at
  both rtols. — *not yet scored.*
- **V3 — honest where referenced.** **Rule:** $|v_g - {\rm ref}| \le {\rm abserr}_g$ for every
  counted reference, Z3's and the random sample's. $|v_{06h} - {\rm ref}|/{\rm abserr}_{06h}$ is
  reported beside it. **Holds** if no counted reference violates it; fewer than 45 of the 90 counted is
  **undecided**. — *not yet scored.*
- **V4 — the trusted leaf's value.** **Rule:** on Z3's samples 2, 8, 9 and 14,
  $|v_g - {\rm ref}|/{\rm abserr}_g \le 0.01$ (06e's breakdown, with the quarters: 0.0012, 0.00046,
  0.0015). **Holds** if true on all four. — *not yet scored.*
- **V5 — the cap.** **Rule:** outside the stationary-point rows no call under $g$ reaches
  `depth_max`, at either rtol. An over-budget unit counts as false. **Holds** if true at both.
  — *not yet scored.*
- **V6 — Z2's call.** **Rule:** `g-1e-8-0-33995` call 5 ends below `depth_max`, converged or
  `node_limited`. **Holds** if so; its $\sum E$, $\sum|E|$, regions and depth are reported.
  — *not yet scored.*
- **V7 — H4's call.** **Rule:** `g-1e-8-3-1072` call 13 converges. **Holds** if so.
  — *not yet scored.*
- **V8 — no convergence lost.** **Rule:** outside the stationary-point rows, no call that 06h
  reported converged is unconverged under $g$. **Holds** if true at both rtols. A failure is a
  finding for the user: each such call is reported with its cause ((b) on a non-noise leaf,
  `node_limited`, or other). — *not yet scored.* (Two LiouvilleGreen calls lost convergence in
  test 9, both by (b) on a non-noise leaf; and the issue opened shows synthetic fallback cells
  where the node-limited stop loses it. V8 may fail.)

## Deviations from the prompt

1. **The budget clause holds when the call meets its request already.** STRUCTURALLY REQUIRED.
   - *What changed.* `_split_could_meet_budget` returns False — the split "could not meet the
     budget", so the clause holds — when the call's error $\sum e' + |\sum E|$ is within its
     request now. The premise is then decided by the halving test alone, as before 06g.
   - *Why.* Such a leaf is chosen only because the trust check found it untrusted. As written, the
     recomputed error is then always within the request, so the clause would always fail and the
     leaf would always be bisected. A noise-level untrusted leaf ($r \ge 1$ on rounding noise),
     which 06b's premise lets the parent exempt as floor-limited, would be split on towards
     `depth_max`. Its children are checked again, and those at $r \ge 1$ are split again. That
     contradicts §0.2 (d)'s own termination argument, which assumes the call is above its request
     ("the recomputed error is the call's error now, which is above the request").
   - *Measured* (`cells.txt`, "the budget clause as the prompt writes it"; test 7b's family,
     $\theta = 10^4e^x$ on $[0, L]$, request $\kappa$ × the root's floor). Built this way, every
     case converges on 2 leaves at depth 1, except $L = 0.6$, $\kappa = 0.75$, unconverged at depth
     0 under both. As written: $L = 0.8$, $\kappa = 0.9$ is **unconverged on 48 leaves at depth
     11** (323 solves), $L = 0.6$, $\kappa = 0.9$ unconverged on 17 at depth 10, and the converged
     cases take 6–35 leaves to depth 4–7.
   - *For the user.* This is the deviation to review. The prompt's sentence "A leaf whose split
     might be all the call needs is bisected" is kept: when the call needs no split for its budget,
     no split "is all the call needs".
2. **The candidates are ordered by $e'_i$**, the leaf's contribution to the call's error, not by
   `total_err`. IMPLEMENTATION CHOICE. The prompt leaves the order unstated. Ordering by `total_err`
   would choose leaves for their $|E_i|$, which bisection does not lower systematically. On every
   Levin leaf the two are equal.
3. **The floor's parts are formed as they are added, not by subtraction.** IMPLEMENTATION CHOICE.
   - $\phi_i - |E_i|$ is `phase_err_less_node`: the fallback floor recomputed with node term
     $R_g + R_2 + R_c$, plus the declared term.
   - $\phi - \phi_{\rm decl}$ is `phase_err_roundoff`: the round-off part as the solve added it,
     returned beside `phase_err_declared`.
   - Subtracting carries a cancellation error of order $u\,\phi_{\rm decl}$ (or $u|E|$), which
     matters when the declared term (or $|E|$) dominates. `phase_err` itself is unchanged bit for
     bit.
4. **`used_interval` carries more than the prompt lists.** IMPLEMENTATION CHOICE. Besides the kind
   and $V_P$, it carries `value` (the returned value; test 1 needs it), `phase_err_declared`,
   `phase_err_roundoff`, `node_error_signed`, `phase_err_less_node` and the property `budget_err`
   ($e'_i$). The fallback solve's dict gains `node_error_bound`.
5. **The budget clause's details.** IMPLEMENTATION CHOICE.
   - The halves enter by their floors alone, without their truncation terms: a lower bound still,
     and more permissive of the split.
   - The request is today's $\max(\texttt{atol}, \texttt{rtol}|V|)$ with the running $V$.
   - A non-finite error on another leaf makes the clause hold.
6. **Test 7's first case is not on 06b's test 1's span, and its `theta_abserr` is a callable.**
   STRUCTURALLY REQUIRED.
   - *Why not 06b's span.* On $\theta = 100e^x$ over [0, 20] the halves' eq. (151) floors sum to
     $10^{-4}$ of the root's, so no declared term of 15–30 % can make the parent's halving test hold.
     The cell is $\theta = 10^4e^x$ on [0, 2] (ratio 0.408), the same shape.
   - *Why not a scalar.* A scalar `theta_abserr` on the Levin branch charges every endpoint,
     interior ones twice, so the halves' declared terms are 1.648× the root's (a development scan,
     not kept). On that span no request lets the parent fire and 06g converge. With
     $c\,e^{-5x}$ they are 1.005× (observation 2).
   - *The window.* It is narrow. The request must be below the halves' whole floors (0.557 of the
     root's), so the budget clause holds at the root and only the halving test decides there. It
     must also be above what two levels reach (0.505). Converged-and-caught held for 0.51–0.55;
     0.53 is used. This is deterministic, and breakage (iii) is caught there.
7. **Test 1's "at most the parent's" is asserted on seven cells, and a noise-level cell where it
   does not hold is a separate test.** IMPLEMENTATION CHOICE. The separate test,
   `test_noise_level_cell`, asserts §0.3's bound instead (the value moves by at most the trusted
   leaves' $\sum\Delta_1$) and honesty. It is reported as observation 3, not left out.
8. **Test 2 uses stand-ins for two of its three cases.** IMPLEMENTATION CHOICE.
   - Stand-ins (the prompt allows them): a trusted leaf later bisected, and a trusted leaf later
     found floor-limited. A scan of rtol (1e-6 to 1e-17) over four real families during development,
     in the scratchpad and not kept, found neither. Each stand-in fixes solves at $h + Kh^4$ ($r = 1/8$) with
     a chosen floor, and says so.
   - The real cell is the bump, whose checked leaves are bisected.
   - The running value sum is read by recording the driver's three `_ExactSum` instances, the
     second being the values.
9. **Test 3's descriptive table is in the calibration, not the test module.** IMPLEMENTATION
   CHOICE. It is in `cells.txt` (tests 1 and 3's cells) and in the draws' report (test 9).
   "Every checked leaf" includes leaves checked and later bisected, seen through a recorder on
   `used_interval`. The table is in Verification.
10. **Test 8's Route B reads "recovered within the declared fit-noise bound" as within three times
    the declared 1σ.** IMPLEMENTATION CHOICE.
    - The bound is a 1σ estimate, not a bound.
    - The jitter's size is 06i's in radians ($10^{-8}$; 06i measured up to $2.2\times10^{-8}$ per
      segment), not 06i's $\kappa = 4.6\times10^{-17}|\theta'|$. At the stand-in's $|\theta'|$ that
      $\kappa$ would sit under the double rounding of $\theta$ itself.
    - Each segment's $\sigma$ is asserted within 2× the injected rms.
    - The measured errors are 0.25σ and 1.04σ.
11. **The Filon tool's mechanics.** IMPLEMENTATION CHOICE. 06i's numerics are ported as written,
    with these mechanics:
    - Work is in chunks of a fixed size (8 or 256 panels, 16 segments), independent of the
      worker count, summed in order. 06i summed per-worker chunks with `fsum`.
    - With `workers > 1` the chunks run in spawned processes that rebuild the call from
      `call_source`.
    - Route B's default edges are 06i's closure walk (`production_edges`), and `edges=` overrides
      it.
    - `linear_phase_integral` serves R0's cells. It uses FCC with 4 and 7 panels, and QAWO with
      panels at most a sixteenth of the span, as 06i's `r0_closed_forms.py` did.
    - There is a command line.
    - Determinism is also checked across two workers.
12. **The Filon tool was not run on 06i's 13 captured calls.** STRUCTURALLY REQUIRED. Their
    pickles need the census tree's harness on the path and Ray's cloudpickle, and a worktree is
    outside this commit's paths. §5's "a reading disagrees with 06i's on one of its 13" checks the
    port. R0's 16 cells reproduce 06i's numbers (worst 0.0722 against 06i's 0.0875 over 32).
13. **Breakage (ii) is three replacements.** IMPLEMENTATION CHOICE. To carry $\sum|E|$ again, the
    breakage makes a fallback leaf's $e'$ its whole floor and drops $|\sum E|$ from the running
    sums and from the ascending sums. Breakage (i) labels $V_P$ as `"quarters"`, so test 1 fails
    on the value and not on the label.
14. **Test 9's harness swaps the driver, not LiouvilleGreen.** IMPLEMENTATION CHOICE. This is 06e's
    approach, since 06g changes the driver. It also records the new keys and the checked-leaf
    classes.
15. **Execution slips, none in a record.** UNINTENDED DRIFT.
    - A first scratch check of R0's cells passed the right-hand limit computed at mpmath's default
      15 digits, and read up to 594× the tolerance. It was the scratch script's error, corrected
      before the test was written.
    - The first draws run, and the first run of the seven roots, were made before a comment-only
      and a docstring-only edit to the module. The calibration outputs and the seven roots were
      re-run at the final module (the outputs' headers record its SHA-256, `d89c959e…`).

## Verification performed

- **The seven roots**, each by its own command (`CLAUDE.md`), at the dispatch SHA (the
  orchestrator's) and after the change. The rise is the tests added: AdaptiveLevin +14 (8 in
  `test_levin_leaf_values`, 3 in `test_levin_node_sum`, 3 added to `test_levin_roundoff_exit`) and
  tools +5 (`test_filon_reference`).

  | Root | Before | After |
  |---|---|---|
  | AdaptiveLevin | 68 | 82 |
  | ComputeTargets | 780 | 780 |
  | CosmologyModels | 39 | 39 |
  | LiouvilleGreen | 156 (1 skipped) | 156 (1 skipped) |
  | ObjectFactories | 280 | 280 |
  | RunDrivers | 127 | 127 |
  | tools | 58 | 63 |

  All OK after. The ComputeTargets flake did not occur.
- **The breakages** (`out/breakages.txt`), each applied, its test run, and the module restored and
  checked by SHA-256. All five were caught:
  - (i) $V_P$ returned: test 1 fails on all seven cells, on the value;
  - (ii) $\sum|E|$ again: test 4 fails, `node_error_signed_sum` 0.0 against $-5.20\times10^{-18}$;
  - (iii) declared term in the ratio: test 7a fails, the call unconverged;
  - (iv) $r_{\rm trust} = 1$: test 3 fails twice (the slow leaf trusted; no noise leaf in the
    $[1/4, 1)$ band);
  - (v) $K$ rounded once: test 8's Ci case fails, $2.16\times10^{-19}$ against the tolerance
    $1.20\times10^{-21}$.

  None is applied in the tree.
- **Test 3's descriptive table**, every checked leaf (returned or later bisected), by $r$ band and
  $\Delta_1$ against the leaf's floor, as counts $\Delta_1 > \phi_P$ / $\Delta_1 \le \phi_P$:

  | | $r < 1/16$ | $[1/16, 1/4)$ | $[1/4, 1)$ | $\ge 1$ | total |
  |---|---|---|---|---|---|
  | tests 1 and 3's cells, parent | 9 / 18 | 3 / 2 | 3 / 2 | 2 / 9 | 48 |
  | tests 1 and 3's cells, 06g | 11 / 22 | 8 / 3 | 7 / 5 | 2 / 10 | 68 |
  | test 9's draws, parent | 6,954 / 19,448 | 488 / 2,463 | 670 / 10,033 | 543 / 19,642 | 60,241 |
  | test 9's draws, 06g | 7,036 / 20,837 | 545 / 3,075 | 1,078 / 13,041 | 862 / 24,200 | 70,674 |

  In the draws, 36 % of checked leaves under 06g have $r \ge 1$ ($\ge 1$ with $\Delta_1 \le \phi_P$:
  24,200), nearly all within the floor: the noise question of
  `[06c-budget-can-stop-on-a-pre-asymptotic-residual]`, counted, not changed.
- **The LiouvilleGreen draws** (`out/lg_seeded_draws_report.txt`), 160 units, seeds 0–19 × both
  codes × four tests.
  - **No test fails under 06g; every unit passes under both.**
  - Of 4,320 Levin calls, converged parent → 06g: T,T 3,828; T,F 2; F,T 0; F,F 490. None at
    depth 20 under either. **No call is node-limited.**
  - Calls with an exit firing: 3,394 → 3,480. Regions: 61,126 → 73,464. Solves: 431,180 → 504,212.
    Under 06g, 43,685 regions return their quarters sum. Of the 1,080 three-Bessel results, convergence is unchanged (884 converged under both).
  - **The two calls that lost convergence** are `test_JJJ` seed 14 #9 (phase2, rtol 1e-10) and
    `test_YJJ` seed 14 #6 (phase3, rtol 1e-10): 13 → 43–44 leaves, abserr 1.62e-12 → 2.07e-12
    against a request of 1.66e-12, and 2.81e-12 → 3.53e-12 against 2.99e-12. Each is **(b) on a
    non-noise leaf**: a Levin leaf at $r = 0.405$ (resp. 0.512) with $\Delta_1$ above its floor,
    trusted at the parent, is refined under 06g into fallback leaves. Their floors do not fall, and
    every one ends floor-limited. Both results are honest, and both tests pass.
  - **Honesty.** Outside the near-singular cases nothing is dishonest under either code (worst
    true/reported 0.30 under 06g, 0.55 at the parent).
  - **The near-singular `test_YJJ_log_singularity` cases** ($\epsilon \le 10^{-4}$): dishonest
    436 → 432 of 560, all flagged converged, worst $7.95\times10^5 \to 7.97\times10^5$.
    - 6 are honest at the parent and dishonest under 06g, each flagged converged: true error moved
      by 1.3–3.5×, at 1.06–1.28× its declaration.
    - 10 move the other way.
    - This is the input-conditioning class recorded on
      `[06c-budget-can-stop-on-a-pre-asymptotic-residual]`.
  - The fixed triple's true errors fall on 5 of 7 oracles: J000 2.9e-12 → 1.5e-13, Y000
    1.8e-11 → 4.6e-14.
- **`black`** on every file changed outside `calibration/scripts/`, which are left as run.
  `black --check AdaptiveLevin tools` is clean.
- **Git.** `git log --oneline 4f4f9ed..HEAD` was empty before staging; staged by explicit path.

## Observations not acted on

1. **Under 06g a fallback call can lose a convergence the parent had.** The cause is the node-limited
   stop and the $e'$ combination. Opened as
   `[06g-node-limited-stop-can-end-a-call-refinement-would-converge]`; the numbers are in
   `cells.txt`'s last section. Two mechanisms, both as §0.2 (c) specifies:
   - *The combination.* $e'_i + |E_i| = \max(\text{res}, \phi_i - |E_i|) + |E_i|$ exceeds
     $\max(\text{res}, \phi_i)$ when the residual lies between $\phi_i - |E_i|$ and $\phi_i$. Where the
     $E_i$ do not cancel the call's error then rises. On test 6's cell it goes from 1.18e-9 to
     1.97e-9, against a true error of 1.17e-9.
   - *The stop.* It assumes bisection does not lower $|\sum E|$. On these cells it does.
     $\sin(x - 10^8)$ on $[10^8, 10^8 + 2]$ at rtol 1e-9 converges at the parent (abserr 1.32e-9
     against 1.42e-9) and stops node-limited under 06g at depth 0 (2.15e-9). At rtol 3e-10 the same
     call refines to depth 2 and $|\sum E|$ falls tenfold.
   - Four such cells are in `cells.txt`.
   - None was found in the LiouvilleGreen draws (no node-limited call). §5's V8 will count it on
     the census.
2. **On the Levin branch a scalar `theta_abserr` charges every endpoint, so its declared term
   rises under bisection.** Interior endpoints are counted twice, and the term grows with the
   number of leaves: on test 7's span the halves' term is 1.648× the root's. The prompt's
   "the halves' terms sum to about the parent's" holds for the fallback's proxy and for a declared
   error concentrated at one end; it does not hold in general on the Levin branch.
   - A shared endpoint's phase error enters its two neighbours with opposite signs, so it largely
     telescopes, and the term overstates it. This is the Levin-branch twin of
     `[06e-cc-declared-phase-term-is-a-crude-proxy]`.
   - No production caller declares a scalar. `BesselPhaseGroup.theta_abserr` grows with $x$.
3. **At rounding noise $Q$ is not closer than $V_P$.** On `test_noise_level_cell`
   (`cells.txt`) the true error rises from 0 to 1 ulp, within §0.3's bound and 1,900× inside
   abserr. Test 1's "at most the parent's" is a statement about resolution-limited values.
4. **The correction by $-\sum E$ works on the two node-dominated cells.**
   $|V - {\rm ref}|$ goes from 1.17e-9 to 1.1e-16 (test 6) and from 4.66e-9 to 7.1e-10 (test 5).
   This is for §5's descriptive measurement; nothing is applied.
5. **The budget clause and an infinite error elsewhere.** If a leaf with an infinite error is still
   refinable when an untrusted leaf is chosen, the clause holds whatever the split would do.
   Bisection order makes this rare: infinite-error trusted leaves come after untrusted ones. 06g
   never fires the exit where the parent would not.
6. **A census harness that wraps `_roundoff_exit_premise_holds` must pass its new keyword through.**
   06h's census counted premise checks by wrapping it, and the function now takes
   `split_could_meet_budget=` (§5's harness change, for the census agent).

## State handed on

- One code commit on `handover-remedial`, **not pushed**. §4 (R23/R24) and §5 run at it on the
  Mac Studio. The header issues are not moved; they move in 06g's records commit.
- **For §5's references:** `python -m tools.filon_reference PICKLE --tolerance T --unpickler
  ray.cloudpickle --workers 6`, run with the census tree $g$ and its harness on `PYTHONPATH` (the
  captures name the harness's functions). Route B's default edges walk the captured integrand's
  closures as 06i did; a capture whose closures differ raises, and its edges must be passed.
- **Deviation 1 is for the orchestrator and the user** before §4 runs.
- The issue opened bears on V8.
- Nothing was written under `var/`, and no run was registered.
