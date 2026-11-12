# levin-phase-floor 07a — the window's early stop, and the phase variation the C2 gate routes on

> **Part B, §1, §3 and §4** (the code, its tests and these records), written by the executing agent on the
> laptop, on the prompt as amended by its §9 and, after Part B's stop, its §10 (2026-11-12). **Part A, §2** (the measurement, on the Mac Studio)
> is recorded here in brief; the orchestrator appends Part A's log in full. **Part C, §5 and §6** (R30 and the
> census) runs later on the Mac Studio; the orchestrator appends its record.

**Subject:** prompt 07a §1, §3 and §4, as amended by §9 and §10 (Part B), on the laptop, in the main checkout
`/Users/ds283/Documents/Code/SecondaryGWKit`, branch `handover-remedial`.
- **Dispatch SHA:** `a9f6abc3` (pushed). The parent module is `a9f6abc3:AdaptiveLevin/levin_quadrature.py`,
  07's code (`adffa21b`'s; `git diff adffa21b a9f6abc3 -- AdaptiveLevin ComputeTargets LiouvilleGreen` empty).
  The parent was run from a temporary detached worktree at `a9f6abc3` outside the repository (in the session
  scratchpad, the main checkout's `venv` symlinked into it), removed afterwards.
- `git log --oneline a9f6abc3..HEAD` was empty at the start. **Stopped before the commit** (§8, a LiouvilleGreen test
  failing under the commit that passes under the parent; below, **The stop**), and **resumed on §10** at `dadd63a0`,
  the orchestrator's amendment, the only commit to land meanwhile (it also records the stop on the board and the
  $6\pi$ finding on `[07-c2-gate-threshold-was-never-measured]`); the board was merged by reading.
- `runregistrykit list` at the start: 88 runs, 86 finished, 2 `unknown` (pre-registry), nothing `running` or
  stale. No Ray was started, no store was opened, no run was registered, and nothing was written under `var/`.

**Commit:** the code commit of this log (one commit, not pushed), on `dadd63a0`.

**Result:**
- **§1.1 (O9) and §1.2 (the window's count, Part A's variant (d)) are in `AdaptiveLevin/levin_quadrature.py`.**
  §1.2 equals Part A's reference implementation bit for bit on eight window problems (Verification 4).
- **§3 and §9.4:** a new module `AdaptiveLevin/tests/test_levin_phase_variation.py` (tests 1–4, 7 and 8), and
  in `test_levin_stationary.py` test 13 rewritten (test 6; its (b) is test 5) and test 14 (a), (b) and (d) added,
  14 (c) as one assertion added to test 10 (b). Every new test passes; **every breakage caught**
  ((i)–(x), (x) in two forms, each by the test the prompt names).
- **Existing assertions changed: three,** all in test 13 (a), (b) and (d), each one the prompt rewrites (§3 test
  6); and one LiouvilleGreen test replaced under §10.3. No other existing assertion moved: in particular tests
  10 (a)–(c) are bit-identical to the parent's, so 14 (c) holds without a stop.
- **§10:** `LiouvilleGreen/tests/test_bessel_compatibility.py`'s `test_the_raw_phase_is_never_evaluated` rewritten
  and renamed `test_the_raw_phase_enters_no_value_or_error`, a constant-shift invariant ($C = 10^6 + 0.5$); the
  docstrings of `BesselPhaseFunction.raw_theta` and the group's `theta` corrected, each module's AST without
  docstrings unchanged; breakage (x) caught in two forms.
- **The seven roots:** AdaptiveLevin 107 → 116, the rise equal to the tests added; the other six equal
  (LiouvilleGreen 156, skipped 1, one test renamed).
- **The LiouvilleGreen draws:** no test fails under the commit that passes under the parent; 2,973 of 2,973 calls
  with no window and no region whose route or LU choice differs are bit-identical; three calls go from converged to
  unconverged at the $6\pi$ threshold (Observation 4, on `[07-c2-gate-threshold-was-never-measured]`).

## Part A, §2 (in brief)

Run on the Mac Studio at `f93aafd0` (07's code) through an orchestrator session on the laptop, in the worktree
`SecondaryGWKit-lpf07a-f93aafd`, 2026-11-10 to 2026-11-12. Records in
[`docs/levin-phase-floor/07a/measurement/`](../../../docs/levin-phase-floor/07a/measurement/): its log
(`log.md`), the report at the planned stop (`out/stop_2_report.md`), the tables (`out/tables_final_rg.md`) and
the attribution (`out/attr/`); prompt §9 holds the rulings and what Part A measured. The orchestrator appends
Part A's log in full below Part C's record.

- Every replay at 07's code reproduces 07's census record bit for bit, 430 of 430; the four new pair captures
  check, 9 of 9 calls; a call with no window is unchanged under every variant, 5 of 5.
- **No variant moves V4's five failures** (v08, v09, v12, v17, v22 at 3.16, 2.21, 3.08, 2.38, 2.53 of abserr):
  converged and dishonest under 07's code, (a), (c), (d), (e) and (f); unconverged under (b2) and (g2), at
  764,283 regions against (a)'s 2,386.
- **Where the error is: H3.** In all five the whole excess is in the Levin leaves outside the window, almost
  all of it in one leaf at the window's upper edge, its true error 2.66–7.44× its $e'$; the window's parts
  (H1, H2) are $10^{-9}$ to $5\times10^{-7}$ of abserr. O9 reaches none of them (120 regions checked).
- **Why (c) and (d) barely act:** all 382 window bisections of items 1–2 have an untrusted parent; `dqage`'s skip
  and raw estimates never act on those calls; (d)'s value test acts only on the 396 (58 calls move, by at most
  $1.8\times10^{-7}$ of their abserr, no flag changes).
- **The user's choice** (prompt §9.1): the orchestrator first recommended leaving 07's rule unchanged; the user:
  *"isn't there a risk we stop too early, though? `dqage` made its choice of stop rule for a reason. Why should
  we not keep the `dqage` stop rule defensively, even though it barely makes any difference here?"*; on the
  revised recommendation, *"Yes, write §1.2 on (d)"*. `dqage`'s outcome (`ier = 2`) is not adopted (§9.3).
  *"H3 defect: yes, this is an issue to fix in a separate prompt"*:
  `[07a-levin-leaf-beside-a-window-understates-its-error]` opened at the stop.

## What shipped

### §1.1, O9 (`AdaptiveLevin/levin_quadrature.py`)

1. **The location's first step** (`_locate_stationary_points`, step 1, in place of Rule A): a phase whose rate
   is exactly 0.0 at every finite scan sample (`not np.any(finite & (d != 0.0))`; `-0.0` is zero) is not
   located, and is reported in `location_skipped_phases` as `{"phase": i, "reason": "zero_rate"}` (the entry's
   form, the agent's choice; it no longer carries a variation). Every other phase runs steps 2–4 unchanged.
2. **The guard's handling of a skipped phase:** unchanged code (`guard_sign_change_phases`, `_nodal_guard`); it
   now applies to an identically zero rate only. Comments updated.
3. **$V$ in `_Basis_SinCos.build_Levin_data`:** if `_changes_sign(theta_prime_Cheb)` is false,
   $V = |\theta(b) - \theta(a)|$ with the raw phase at the two end nodes (from `theta_Cheb` where the method
   samples it already, i.e. without `theta_deriv` or without `theta_mod_2pi`; otherwise sampled at the two ends
   through `_sample_vectorized` under the same cache key); else $V$ is the node-partition sum, the new
   module-level `_node_partition_variation(theta_nodes)`, $\sum_j|\theta(x_{j+1})-\theta(x_j)|$ summed by
   `math.fsum` (correctly rounded, so independent of the nodes' order), the raw phase sampled on the nodes only in
   this case (or reused). A non-finite $V$ raises `ValueError` after a warning, as a non-finite $\theta'$ does
   (deviation 4).
   - **How each basis combines its phases:** `_Basis_SinCos` has one phase; $V$ is that phase's.
     `_TwoIndependentSinCosBasis` (test_levin_quadrature.py) and `_TwoPhaseQuadraticBasis` (test_levin_stationary.py)
     summed their phases' estimates, $(\overline{|\theta_1'|} + \overline{|\theta_2'|})\,|b-a|$; each now sums
     its phases' $V_i$, each by the rule above (a shared helper `_phase_variation` in test_levin_quadrature.py).
   - **The precision.** The largest raw phase this Part could measure is the LiouvilleGreen draws':
     $|\theta|$ at region ends up to $1.021\times10^{13}$ (`test_JJJ`, seed 0), ulp $1.95\times10^{-3}$ rad;
     `test_abserr_bounds_truth` $5.1\times10^{12}$, ulp $9.8\times10^{-4}$. Both negligible against $6\pi$. The
     largest raw phase in production (`QuadSourceIntegral`) is not measurable on the laptop (no store); it is
     Part C's (Observation 3).
4. **`phase_span` is $V$ everywhere.** Every use, at the commit's lines:

   | Line | Use | Receives now |
   |---|---|---|
   | `levin_quadrature.py` 2467, 2478, 2513 | `_Basis_SinCos.build_Levin_data` computes and returns it | $V$ (node-partition sum at 2467, $\|\theta(b)-\theta(a)\|$ at 2478); checked finite at 2480 |
   | 2769 | `_adaptive_levin_subregion_impl` takes it from `BasisData.build_Levin_data` | $V$ of the basis |
   | 2802 | passed to `_oriented_cc` (`force_direct`: window pieces, guarded leaves at `depth_max`) | $V$; only reported in the dict |
   | 2809 | the C2 gate, `phase_span < SIX_PI` | $V$ |
   | 2815 | passed to `_adaptive_levin_subregion_cc` | $V$; only reported in the dict |
   | 2904 | the LU choice, `phase_span > _LEVIN_DIRECT_SOLVE_PHASE_SPAN` | $V$ |
   | 3094 | the Levin solve's result dict, `"phase_span"` | $V$ |
   | 3114 (parameter), 3308 | `_adaptive_levin_subregion_cc`: reported in its result dict, `"phase_span"`; nothing else reads it | $V$ |
   | 3447 (parameter), 3472, 3481, 3494 | `_oriented_cc`: passed through on an ascending span; on a reversed span `phase_span_ascending` from `build_Levin_data` on the ascending span, passed to the pair | $V$ (of the ascending span on a reversed one) |
   | `test_levin_quadrature.py` 114, 124 | `_TwoIndependentSinCosBasis.build_Levin_data` | $V_1 + V_2$ |
   | `test_levin_stationary.py` 485, 498 | `_TwoPhaseQuadraticBasis.build_Levin_data` | $V_1 + V_2$ |
   | `test_levin_end_nodes.py` 471–476, 497–502 | from `build_Levin_data` into `_oriented_cc` (via the test's helpers) | $V$; no assertion moved (only reported) |
   | `test_levin_cc_floor.py` 86–102, 143–149 | from `build_Levin_data` into `_parent_cc` / `_oriented_cc` | $V$; no assertion moved (only reported) |
   | `test_levin_leaf_values.py` 471 | a dict literal's key | unchanged (100.0) |
   | `test_levin_phase_variation.py` | the new tests read it as $V$ | $V$ |

   `_adaptive_levin`'s result has no `phase_span` key; the regions it returns do not carry one. Comments only,
   untouched (outside §10.3's three LiouvilleGreen changes): `LiouvilleGreen/bessel_phase.py:932–935`,
   `LiouvilleGreen/three_bessel_integrals.py:589–590`, `ComputeTargets/tests/test_quadsource_integral.py:2121–2122`
   (Observation 1).

5. **Docstrings and comments rewritten** (§1.1 item 5), at the commit's lines: the module docstring's C2 item (84–99: what is computed, why the estimate was replaced, $6\pi$ unchanged and
unmeasured), its stationary-point item (131–138, Rule A replaced), the basis interface (152–153, 161–163); the comment
above `SIX_PI` (254–263); the comment above `_LEVIN_DIRECT_SOLVE_PHASE_SPAN` (495–501); `_node_partition_variation`
(2251, new); `build_Levin_data`'s docstring (2357–2377, 2393) and the comment at $V$ (2456–2460);
`_adaptive_levin_subregion_impl`'s docstring at `guard_sign_change_phases` (2706–2712) and at the gate (2714–2724,
2732–2735), the comment at sampling the phase (2752–2757), at the gate (2778–2787) and at the LU choice (2888–2889);
`_adaptive_levin_subregion_cc`'s `phase_span` parameter (3137–3139); `_nodal_guard`'s docstring (3418–3422);
`_locate_stationary_points`' docstring, step 1 (3551–3567) and `skipped_phases` (3612–3614), and the comment at step
1 (3651–3652); `_adaptive_levin`'s docstring item 1 (4030–4042), item 4 (4082–4088) and the result's
`location_skipped_phases` (4099–4102), the comment at `guard_sign_change_phases` (4219–4221) and at the result key
(5468–5470); `adaptive_levin_sincos`'s docstring on `theta_deriv` (5830–5836) and on `location_skipped_phases`
(5973–5979). Each says what is computed, why the estimate was replaced (§0.2), and, where the gate is described,
that $6\pi$ is unchanged and unmeasured.

### §1.2, the window's count

1. **The count** is the new module-level `_window_bisection_counts(value_parent, value_halves, error_parent,
   error_left, error_right, unresolved_left, unresolved_right, rtol)`: counts iff neither half is unresolved,
   $|(V_L+V_R)-V_P| \le \min(10^{-5},{\rm rtol})\,|V_L+V_R|$, and $e_L+e_R \ge 0.99\,e_P$, each inclusive. Called at
   the window bisection in `_adaptive_levin` with each leaf's own value (`value_own`), its own error
   `_own_error(region)` (`used_interval.own_err`, new: `max(abserr, floor)` plus truncation, the floor less
   $|E|$ on a Clenshaw–Curtis region, before any trust-check declaration; Part A's `_raw_err_07a`), and
   `_cc_leaf_unresolved(abserr, resasc)` (new: $200\,|{\rm CC}_{2N-1}-{\rm CC}_N| \ge {\rm resasc}$, both finite
   and positive; Part A's `_unresolved_07a`).
2. **resasc** is formed in `_adaptive_levin_subregion_cc` on the fine rule (Part A's formula, the same
   expressions), returned as the solve dict's `"resasc"` key, and kept on the region (`used_interval` gains a
   `resasc` constructor argument, `_resasc`, and a read-only `resasc` property). Nothing else reads it; it enters
   no value or error, and no result key of `_adaptive_levin` (deviation 5).
3. **Unchanged:** `_floor_limit_stopped_window_leaf`; `fell_at_last_bisection` and `leaves_falling_at_stop`, still
   on $e'$; the windows and pieces; every leaf outside a window; `converged`.
4. **The constants:** `_LEVIN_WINDOW_IROFF1_VALUE_RTOL` = 1e-5 (now the cap), `_ERROR_RATIO` = 0.99, `_LIMIT` = 6,
   unchanged; new `_DQK21_SATURATION_FACTOR = 200.0`, commented with `dqk21.f`'s statement
   `abserr = resasc*dmin1(0.1d+01,(0.2d+03*abserr/resasc)**1.5d+00)`. The comment above them is rewritten:
   `dqage`'s count with its skip and raw estimates; the value test no looser than the request and why;
   `ier = 2` not adopted and why (§9.3); where Part A's measurement is.
5. **Docstrings and comments rewritten** (§1.2 item 5), at the commit's lines: the comment above the constants (539–578; the constants 579–581) and the new `_DQK21_SATURATION_FACTOR`'s (583–591);
`_cc_leaf_unresolved`, `_own_error` and `_window_bisection_counts` (1071–1123, new); `used_interval`'s `_resasc`
comment (1380–1383), `phase_limited`'s description (1476–1483), `own_err` and `resasc` (1616–1634, new);
`_levin_leaf`'s description of `fell_at_last_bisection` (1772–1779: still on $e'$, and why the count is not);
`_adaptive_levin_subregion_cc`'s docstring (3156–3158) and the comment at resasc (3294–3297); `_adaptive_levin`'s
item 2a (4050–4068) and the result's `iroff1_count` (4102); the comment at the count (5089–5096); and the comment at
`stationary_windows`' `iroff1_count` (5446–5450).
6. **§1.3:** no new result key; `stationary_windows`' `iroff1_count` reports the count above.

### §3 and §9.4, the tests

All with no Ray and no datastore. References: closed forms (test 2's and 14 (d)'s by
`docs/levin-phase-floor/07a/code/test_references_07a.py`, mpmath at 40 and 60 digits, agreeing to $4.2\times10^{-42}$
and $7.2\times10^{-39}$ relative), or 07's references already in `test_levin_stationary.py`. Every honesty assertion
is `_Honest.assertHonest`: the value within its abserr of the reference, and abserr at least the true error. The
parent's figures quoted in the docstrings were measured by running `a9f6abc3`'s module on the same problems
(scratch scripts in the session; the figures are in the docstrings and below).

**O9, in `AdaptiveLevin/tests/test_levin_phase_variation.py` (new, 6 tests):**

| § 3 test | name | what it asserts | the parent |
|---|---|---|---|
| 1 | `TestVariationExact.test_1_V_exact_on_monotone_regions` | $V = |\theta(b)-\theta(a)|$ bit for bit on five regions (one reversed, one narrow at $x\approx5$) for `_Basis_SinCos` with `theta_deriv`, with `theta_deriv` and `theta_mod_2pi`, and spectral; on $\theta=x^3$ over [0, 1] (a zero rate at a node, no strict sign change); for `_TwoIndependentSinCosBasis` (five regions) and `_TwoPhaseQuadraticBasis` (three regions off its root) | the estimate, not $V$ |
| 2 | `TestVariationExact.test_2_end_weighted_shape` | $\theta=e^{\alpha x}$, $\alpha=\log(1+0.98\cdot6\pi)$: $V=18.4726<6\pi$ bit for bit; the parent's estimate, recomputed in the test, 21.9403 $>6\pi$; the root region's solve is Clenshaw–Curtis; the call (f = (1, 0), rtol $10^{-10}$) honest | Levin (`is_direct` False, `phase_span` 21.940294566299677); same driver result |
| 3 | `TestMonotoneUnderBisection.test_3_no_half_of_a_cc_region_goes_to_levin` | on the §0.2 bump ($|\theta'|$ Gaussian of width 0.05 at 1/2, $V=10\pi$), $e^{4x}$ and a left-weighted Lorentzian rate ($V=12\pi$), every region the driver solves (recorded by wrapping `lq._adaptive_levin_subregion`): no half of a Clenshaw–Curtis region goes to Levin, and at least one Clenshaw–Curtis region has its halves visited; the parent's estimates on the bump recomputed (4.7308 $\pi$ on [0, 1], 11.5877 $\pi$ on its half); the commit routes [0, 1] to Levin and its halves to Clenshaw–Curtis | the bump violates it: [0, 1] Clenshaw–Curtis at 4.7308 $\pi$, both halves Levin at 11.5877 $\pi$ (2 pairs among 159 regions visited); the other shapes do not |
| 4 | `TestNonMonotoneRegion.test_4_node_partition_sum` | P1 without rates on [0, 16]: spectral $\theta'$ changes sign on the nodes; $V$ is the node-partition sum bit for bit (math.fsum in the test), at most the true variation and above the net change; the call honest, no Levin leaf returned guarded | — |
| 7 | `TestMinimaStayLocated.test_7_nearly_coalescing_pair_located` | test 3's family (C) at $b=10^4$ and $10^8$: the rate positive on every scan sample, nothing skipped, the minimum located (kind "minimum", $\lambda<\Lambda_c$) and windowed, honest | the same (test 3 passes at the parent) |
| 8 | `TestLUChoice.test_8_lu_choice_follows_V` | the bump at $V=21\pi$: estimate 9.9348 $\pi<20\pi$, $V>20\pi$, LU (`direct_solve` 1, no `lstsq_solve`); the reverse, $e^{\alpha x}$ with $V=19.6\pi$: estimate 25.9678 $\pi>20\pi$, least squares | bump: least squares; reverse: LU |

**O9, in `AdaptiveLevin/tests/test_levin_stationary.py` (test 13 rewritten, prompt §3 test 6):**
- **13 (a)** `test_13a_zero_rate`: the zero rate still skipped, now `[{"phase": 0, "reason": "zero_rate"}]`; the rest
  as before (bit-identical to the bypassed call, converged, honest).
- **13 (b) = test 5** `test_13b_small_phase_with_a_root`: $\theta=35(x-x_0)^2$ on [0, 1]: nothing skipped, one root
  at $x_0$ within $4\epsilon x_0$, one window, pieces, no leaf guarded, converged, honest. At the parent: skipped
  (scan estimate 18.475242613612156), one guarded leaf, value `0x1.58f28f18e1696p-3`, abserr $1.962\times10^{-13}$,
  5 regions. Under the commit: value `0x1.58f28f18e1690p-3`, abserr $4.215\times10^{-12}$, 6 regions, max depth 2,
  $|v-{\rm ref}|/{\rm abserr}=3.29\times10^{-5}$.
- **13 (c)** unchanged (docstring notes it).
- **13 (d)** `test_13d_several_phases`: phase 2 skipped as `[{"phase": 1, "reason": "zero_rate"}]`, no leaf guarded,
  converged, honest; value and abserr bit-identical to the parent's (`0x1.a309197cde8e0p-5`, $1.965\times10^{-14}$).

**The window's count, in `test_levin_stationary.py` (`TestWindowCount`, §9.4 test 14):**
- **14 (a)** `test_14a_the_count_directly`: on `_window_bisection_counts` with constructed inputs: at rtol
  $10^{-8}$ a change of $10^{-6}$ relative is not counted, and is counted at rtol $10^{-3}$ (the cap $10^{-5}$,
  07's form), a change of $2\times10^{-5}$ not; either half or both unresolved: not counted; raw estimates flat
  while $e'$ falls (a parent with a widened declaration, built as `used_interval`s and read through
  `lq._own_error`): counted; raw falling while $e'$ flat: not counted; the value boundary at rtol $2^{-20}$ and the
  error boundary at 0.99, both inclusive, the next double outside not counted.
- **14 (b)** `test_14b_unresolved`: a smooth Clenshaw–Curtis leaf (F_23, $\theta=10x$, [0, 1]: nested-pair estimate
  $4.1\times10^{-9}>0$, 200 times it $1.3\times10^{-6}$ of resasc) is resolved; a peak of width $10^{-3}$ on an odd
  fine node (the coarse rule misses it) is unresolved, its $200\,|{\rm CC}_{2N-1}-{\rm CC}_N|$ 105 times resasc and
  $|{\rm CC}_{2N-1}-{\rm CC}_N|$ alone 0.53 of it; each leaf's resasc equals $\sum_j|w_j||g_j-\bar g|$ evaluated in the
  test to $10^{-13}$ relative; `_cc_leaf_unresolved` inclusive at 200, false with either number zero or missing.
- **14 (c)**: one assertion added to `test_10b_untrusted_leaves_at_noise_stop`, `iroff1_count` at the limit; 10 (a)
  already asserts it, 10 (c) a count of zero. Tests 10 (a)–(c) are bit-identical to the parent's (value, abserr,
  regions, windows: 10 (a) count 6 stopped after split 44, 50 regions; 10 (b) 6 after 97, 145 regions; 10 (c) 0, 36
  regions).
- **14 (d)** `test_14d_value_test_against_the_request`: $\theta=10^8+(c/2)(x-x_s)^2$, $c=10^6$, $x_s=0.38197$, f = (1,
  0), rtol $10^{-9}$, depth_max 12. The raw phase near $10^8$ puts the window leaves' floor above the request, so
  their own errors are flat ($e_L+e_R=e_P$ exactly at every window bisection) and the first four bisections change
  the value by $2.8\times10^{-9}$ and $3.8\times10^{-9}$ relative, between rtol and $10^{-5}$. **At the parent** the
  window stops after the call's 18th bisection: 28 regions, value `0x1.0895691a63805p-10`, abserr
  $1.977\times10^{-10}$, unconverged, honest at 0.0238. **Under the commit** they are not counted, and the window
  stops after the 22nd: 32 regions, unconverged, honest at 0.0126. Asserted: the stop, split 22, count 6, 32
  regions, unconverged; at least one bisection recorded (by wrapping `lq._window_bisection_counts`) with its value
  change in (rtol, $10^{-5}$], both halves resolved, raw error flat, and not counted; the honesty. The commit with
  the value test at a fixed $10^{-5}$ (breakage (vi)) reproduces the parent's stop at split 18 and 28 regions.
  **§3's case that leaves a call converged and dishonest at the parent was not built:** on every case tried (this
  family at $\theta_0 = 10^6$ and $10^8$, rtol $10^{-8}$ and $10^{-9}$, f = (1, 0) and (0, 1)) both codes are
  honest, as Part A found on production calls (§9.2: the stopped windows' declarations at or above their true
  error; the five failures' excess outside the window). 14 (a) asserts the rule directly.

**Tests added: 9** (6 in the new module, 3 in `TestWindowCount`); the AdaptiveLevin root goes from 107 to 116.

### The stop, and §10

**The stop** (2026-11-12, before the commit). The first "after" run of the seven roots failed two subtests of
`LiouvilleGreen/tests/test_bessel_compatibility.py`, `TestLevinConsumption.test_the_raw_phase_is_never_evaluated`,
spans (10, 60) and (1000, 1100): the test runs each span with a `theta` that raises and requires the same output,
because "``theta`` is a *required key* that is never called" when `theta_mod_2pi` and `theta_deriv` are supplied,
while §1.1 item 3 samples `theta["theta"]` on every gated region. It passes at the parent. Prompt §8 makes that a
stop, and LiouvilleGreen could not change, so the agent stopped with its work uncommitted and reported. The
orchestrator reproduced it; the user ruled *"Option 1, and record the 6π finding on that issue"*, written as §10
(`dadd63a0`): §1.1 unchanged, the test rewritten to its purpose, two docstrings corrected.

**§10.3, as built:**
1. **`test_bessel_compatibility.py`:** `test_the_raw_phase_is_never_evaluated` is replaced by
   `test_the_raw_phase_enters_no_value_or_error`. Each span (the class's `NEAR_SPAN` and `TAIL_SPAN`) is run with the
   supplied `raw_theta` and with `raw_theta(x) + C`, `C = RAW_PHASE_SHIFT = 1e6 + 0.5` (a class attribute; not a
   multiple of $2\pi$, asserted). Every region solve is recorded by wrapping
   `levin_quadrature._adaptive_levin_subregion` (`mock.patch.object`): its span (hex), whether it went to the
   Clenshaw–Curtis rule, whether forced, and `direct_solve` / `lstsq_solve`. Asserted: the two decision lists equal
   (so no route or LU choice moved), the value and abserr equal by hex. **$C = 10^6 + 0.5$ moves no decision on
   either span**, so no other $C$ was needed. The spans exercise both routes: (10, 60) 39 solves, 36 Clenshaw–Curtis,
   3 least squares; (1000, 1100) 7 solves, 1 LU, 6 least squares; no window, no point located. The docstring says
   what it protects and why, cites §1.1 and §10, and leaves `RECONCILIATION.md` as written. The class docstring's
   last clause, "the raw phase is never evaluated", is amended to match (deviation 17). The module gains
   `import AdaptiveLevin.levin_quadrature as levin_quadrature` and `from unittest import mock`.
2. **`LiouvilleGreen/bessel_phase.py`, `BesselPhaseFunction.raw_theta`'s docstring:** "Levin never *evaluates* it
   when ``theta_mod_2pi`` and ``theta_deriv`` are both supplied (``:1038``)" becomes: Levin samples it for routing
   only, through the difference $|\theta(b)-\theta(a)|$ its gate and LU choice read, never for a value or an error;
   until 07a it was never evaluated.
3. **`LiouvilleGreen/three_bessel_integrals.py`, the group's `theta` docstring:** the same correction.

   **The AST check:** for each of the two modules, `ast.dump` of the module parsed from `git show HEAD:` and from the
   working tree, each with every module, class and function docstring removed, compare equal (both True; the
   source differs). Nothing else under `LiouvilleGreen/` changed.

### Records

- `docs/levin-phase-floor/07a/code/test_references_07a.py` and `out/test_references_07a.json`: test 2's and
  test 14 (d)'s references, closed forms in mpmath at 40 and 60 digits (deviation 7).
- `docs/levin-phase-floor/07a/code/lg_draws_07a.py`, `out/parent/`, `out/commit/` (one JSON per unit, and
  `run.json` with the provenance), `out/lg_draws_report.txt` and `.json`, `out/lg_run_stdout.txt`: the
  LiouvilleGreen draws.
- `docs/levin-phase-floor/07a/code/breakages_07a.py` and `out/breakages_summary.txt`: the breakages, (x) included
  (re-run in full after §10).
- `docs/levin-phase-floor/07a/code/lg_draws_07a_changes.py` and `out/lg_draws_changes.txt`: the draws' changes,
  call by call.
- Each script is `black`-formatted before it ran, and its outputs record its SHA-256; no `CLAUDE.md` change
  is needed.

## Predictions and verdicts

### Part A, §2.4's tables (from `docs/levin-phase-floor/07a/measurement/out/tables_final_rg.md`, items 1–4)

#### 4. Qualification and cost

| variant | qualifies (items 1-2) | conv. & dishonest V4 | conv. & dishonest 396 | pairs failing | 396 scored / undecided | unconverged V4 / pairs / 396 | regions | wall s | regions 1-2 | wall s 1-2 | windows stopped | leaves falling at stop | missing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tree | no | v08, v09, v12, v17, v22 | 0 | p14501, p35006 | 0 / 396 | 1 / 0 / 4 | 29555 | 615 | 2386 | 48 | 408 | 4325 | 0 |
| a | no | v08, v09, v12, v17, v22 | 0 | p14501, p35006 | 0 / 396 | 1 / 0 / 4 | 29555 | 625 | 2386 | 48 | 408 | 4325 | 0 |
| b2 | yes | - | n/a | - | 0 / 396 | 25 / 4 / not run | 764283 | 50555 | 764283 | 50555 | 0 | 0 | 0 |
| c | no | v08, v09, v12, v17, v22 | 0 | p14501, p35006 | 0 / 396 | 1 / 0 / 4 | 29555 | 628 | 2386 | 48 | 408 | 4325 | 0 |
| d | no | v08, v09, v12, v17, v22 | 0 | p14501, p35006 | 0 / 396 | 1 / 0 / 4 | 29827 | 629 | 2386 | 48 | 408 | 4543 | 0 |
| e | no | v08, v09, v12, v17, v22 | 0 | - | 0 / 396 | 2 / 2 / 39 | 29555 | 617 | 2386 | 49 | 408 | 4325 | 0 |
| f | no | v08, v09, v12, v17, v22 | 0 | - | 0 / 396 | 2 / 2 / 39 | 29555 | 619 | 2386 | 49 | 408 | 4325 | 0 |
| g2 | yes | - | n/a | - | 0 / 396 | 25 / 4 / not run | 764283 | 51494 | 764283 | 51494 | 0 | 0 | 0 |

#### 1. Honesty on V4's 25: |v - ref| / abserr (* converged and dishonest; u unconverged)

| call | tree | a | b2 | c | d | e | f | g2 |
|---|---|---|---|---|---|---|---|---|
| cal_z12 | 0.718 | 0.718 | 0.718 u | 0.718 | 0.718 | 0.718 | 0.718 | 0.718 u |
| v00 | 0.0197 | 0.0197 | 0.0197 u | 0.0197 | 0.0197 | 0.0197 | 0.0197 | 0.0197 u |
| v01 | 0.0325 | 0.0325 | 0.0325 u | 0.0325 | 0.0325 | 0.0325 | 0.0325 | 0.0325 u |
| v02 | 0.00572 | 0.00572 | 0.00572 u | 0.00572 | 0.00572 | 0.00572 | 0.00572 | 0.00572 u |
| v03 | 0.00175 | 0.00175 | 0.00175 u | 0.00175 | 0.00175 | 0.00175 u | 0.00175 u | 0.00175 u |
| v04 | 0.138 u | 0.138 u | 0.138 u | 0.138 u | 0.138 u | 0.138 u | 0.138 u | 0.138 u |
| v05 | 0.00124 | 0.00124 | 0.00124 u | 0.00124 | 0.00124 | 0.00124 | 0.00124 | 0.00124 u |
| v06 | 0.000278 | 0.000278 | 0.000278 u | 0.000278 | 0.000278 | 0.000278 | 0.000278 | 0.000278 u |
| v07 | 0.000481 | 0.000481 | 0.000482 u | 0.000481 | 0.000481 | 0.000481 | 0.000481 | 0.000482 u |
| v08 (failure) | 3.16* | 3.16* | 3.16 u | 3.16* | 3.16* | 3.16* | 3.16* | 3.16 u |
| v09 (failure) | 2.21* | 2.21* | 2.21 u | 2.21* | 2.21* | 2.21* | 2.21* | 2.21 u |
| v10 | 0.192 | 0.192 | 0.192 u | 0.192 | 0.192 | 0.192 | 0.192 | 0.192 u |
| v11 | 0.00762 | 0.00762 | 0.00762 u | 0.00762 | 0.00762 | 0.00762 | 0.00762 | 0.00762 u |
| v12 (failure) | 3.08* | 3.08* | 3.08 u | 3.08* | 3.08* | 3.08* | 3.08* | 3.08 u |
| v13 | 0.309 | 0.309 | 0.309 u | 0.309 | 0.309 | 0.309 | 0.309 | 0.309 u |
| v14 | 0.00102 | 0.00102 | 0.00102 u | 0.00102 | 0.00102 | 0.00102 | 0.00102 | 0.00102 u |
| v15 | 0.00204 | 0.00204 | 0.00205 u | 0.00204 | 0.00204 | 0.00204 | 0.00204 | 0.00205 u |
| v16 | 0.0382 | 0.0382 | 0.0382 u | 0.0382 | 0.0382 | 0.0382 | 0.0382 | 0.0382 u |
| v17 (failure) | 2.38* | 2.38* | 2.38 u | 2.38* | 2.38* | 2.38* | 2.38* | 2.38 u |
| v18 | 0.529 | 0.529 | 0.529 u | 0.529 | 0.529 | 0.529 | 0.529 | 0.529 u |
| v19 | 0.0462 | 0.0462 | 0.0462 u | 0.0462 | 0.0462 | 0.0462 | 0.0462 | 0.0462 u |
| v20 | 0.0141 | 0.0141 | 0.0141 u | 0.0141 | 0.0141 | 0.0141 | 0.0141 | 0.0141 u |
| v21 | 0.00862 | 0.00862 | 0.00863 u | 0.00862 | 0.00862 | 0.00862 | 0.00862 | 0.00863 u |
| v22 (failure) | 2.53* | 2.53* | 2.53 u | 2.53* | 2.53* | 2.53* | 2.53* | 2.53 u |
| v23 | 0.767 | 0.767 | 0.767 u | 0.767 | 0.767 | 0.767 | 0.767 | 0.767 u |

#### 2. The pairs, scored on r_g = |v_7 - v_8| / e_7, the cross-check's; in brackets the prompt's r = |v_7 - v_8| / (e_7 + e_8), for the record (u: a member unconverged; * converged and r_g > 1)

| pair | tree | a | b2 | c | d | e | f | g2 |
|---|---|---|---|---|---|---|---|---|
| p14501 | r_g 1.39 (r 0.727)* | r_g 1.39 (r 0.727)* | r_g 1.39 (r 0.727) u | r_g 1.39 (r 0.727)* | r_g 1.39 (r 0.727)* | r_g 1.39 (r 0.727) u | r_g 1.39 (r 0.727) u | r_g 1.39 (r 0.727) u |
| p35006 | r_g 1.08 (r 0.794)* | r_g 1.08 (r 0.794)* | r_g 1.08 (r 0.794) u | r_g 1.08 (r 0.794)* | r_g 1.08 (r 0.794)* | r_g 1.08 (r 0.794) u | r_g 1.08 (r 0.794) u | r_g 1.08 (r 0.794) u |

#### 3. The 396 against (b): undecided by construction ((b) not run on the 396, the user's ruling at stop 1; (b2) converged on none of the 29 window calls of items 1-2)

| variant | scored | undecided | honest | converged & dishonest | unconverged & dishonest | max ratio |
|---|---|---|---|---|---|---|
| tree | 0 | 396 | 0 | 0 | 0 | nan |
| a | 0 | 396 | 0 | 0 | 0 | nan |
| b2 | 0 | 396 | 0 | 0 | 0 | nan |
| c | 0 | 396 | 0 | 0 | 0 | nan |
| d | 0 | 396 | 0 | 0 | 0 | nan |
| e | 0 | 396 | 0 | 0 | 0 | nan |
| f | 0 | 396 | 0 | 0 | 0 | nan |
| g2 | 0 | 396 | 0 | 0 | 0 | nan |

Item 5, the attribution, per call: `docs/levin-phase-floor/07a/measurement/out/attr/` (summarised in prompt
§9.2 and in **Part A** above). Per call and variant, value, abserr, `converged`, regions, depth, windows stopped,
leaves falling and wall: the same file's "Per call and variant".

**The user's choice:** variant (d), *"Yes, write §1.2 on (d)"* (prompt §9.1), with `dqage`'s outcome not adopted
(§9.3).

### Part B's acceptance (§7 item 2)

| Rule | Number | Verdict |
|---|---|---|
| §1 as written and amended | §1.1 items 1–5, §1.2 items 1–5, §1.3; §1.2 bit-identical to variant (d) on eight window problems | holds |
| every test passing | the seven roots OK (Verification 1) | holds |
| the breakages caught | (i)–(x), (x) in two forms, each by the test named (Verification 3) | holds |
| every changed existing assertion listed | three, test 13 (a), (b), (d) (Verification 2) | holds |
| the LiouvilleGreen draws with no new failure | 0 of 80 units newly failing | holds |
| the seven roots rising only by the tests added | AdaptiveLevin +9 = tests added; the other six equal (after §10) | holds |

### §5's predictions (R30), copied before they run — *not yet scored*

As 07 §5, with records under `var/levin-phase-floor-07a/rehearsal/`, slug `r30-07a`, `--prompt 07a`,
`--job-name levin-phase-floor-07a`, and the fingerprint `1b332ffa…`. There is no new parent
rehearsal: **R29** (`levin-phase-floor-07-r29-guard-rule-a-20261108T123429`, the store
`var/datastores/levin-phase-floor-07-r29*`, kept) was built at `e182bc7a`, whose code is the code
commit's parent's.
1. **S0′ — R29 is the parent.** `git diff --stat e182bc7a <code commit's parent> -- . ':!docs'
   ':!prompts'` lists nothing the build imports. **Holds** if so; if not, a parent rehearsal R31 is
   run first and this prompt's comparisons use it.
2. **The lock:** `acquire`, `set-pid` at the launch, `release` after the readers, each a shell
   command of its own.
3. **R30 against R29,** by `tools.store_value_diff` with `--json` and 05's `classify_non_qsi.py`
   (with the job names normalised in a copy, as 07 did):
   - **outside `QuadSourceIntegral`:** every value table identical, every other difference of the
     four volatile kinds, no row in one store only;
   - **inside it, every moved row tabulated:** the change against R29's and R30's `total_abserr`,
     regions, maximum depth, located points, window pieces, guarded regions and `total_converged`;
     and the rows whose `total_converged` changes, in each direction.
4. **S3 — consistency.** Every row with finite `total_abserr` in both has
   $|{\rm total}_{30} - {\rm total}_{29}| \le {\rm total\_abserr}_{29} + {\rm total\_abserr}_{30}$.
   **Holds** if no row violates it. A violation is a finding, not by itself a stop.
5. **S1 and S2** re-scored on R30 as 07 §5 items 5 and 6, against 06g's references and 07's
   `p1_references`.
6. **S4 — O9's four blocks.** R29's four guarded analytic J4/Y4 blocks: in R30 each locates its
   root, guards no leaf, and is honest against R29's block references
   (`docs/levin-phase-floor/07/rehearsal/r29/out/block_reference/`).
7. **The readers** run on R30, exit 0, and the store is byte-identical before and after. **The
   lock** is released and `ray-processes` lists nothing. **R29 and R30 are kept.**

### §6's predictions (the census), copied before they run, with §9.5's additions — *not yet scored*

**Gate W0, before the run:**
- **The parent's code is 07's:** as S0′.
- **The harness reproduces 07's census at the parent,** bit for bit by 07's comparison, on 06k's
  100 K0 units. One registered run, slug `k0`.
- **The instrument changes nothing:** one unit with a window and one with a route change, at $h$,
  run with and without it, bit-identical.
- **The new code is deterministic:** two runs of one $h$ unit with a window, bit-identical.

**Predictions, fixed before the run** (README §5 rule 3), copied into `state/predictions.md`
unchanged, with §9's additions:
- **W1 — nothing else moves.** Every call with no window at 07 or at $h$, and no region whose route
  or LU choice differs, equals 07's call bit for bit, with `phase_span` set aside. Every row all of
  whose calls are such equals 07's in `total` and `total_abserr`. **Holds** if true at both rtols.
- **W2 — no Levin leaf is returned across a stationary point** (07's V2): the scan finds no returned
  Levin leaf holding a sign change of $\theta'$.
- **W3 — every phase with something to locate is located.** The location agrees with the scan both
  ways (07's V3), with no phase set aside except one whose rate is identically zero; every skipped
  phase is listed with its call, and each is of that kind.
- **W4 — honest where it is referenced.** V4's 33 calls (z00, z12, Z3's seven and the 24),
  re-scored from the census against the readings 07 scored them on, by 07's rule. **Holds** if
  every scored call is honest, or is unconverged under a rule of §9's amendment.
- **W5 — the gate is consistent, and the guard is a backstop.** In every call: no Clenshaw–Curtis
  region has a half sent to the Levin solve; no region takes the node-partition sum; no region is
  guarded (`num_guarded_regions` 0). **Holds** if all three are zero. A non-zero count is a finding,
  with its calls listed.
- **W6 — the pairs.** The cross-check's $r$ re-computed at $h$ over all 2,484 pairs: every pair has
  $r \le 1$, or a member unconverged under §9's rule. `3-14501` and `3-35006` are listed.

§9.5's additions:

- **W4 is predicted to fail** on exactly v08, v09, v12, v17 and v22, at 3.16, 2.21, 3.08, 2.38 and
  2.53 to three figures, and to hold on the other 28 calls.
  - Basis: §9.2. Neither §1.2 nor O9 reaches the leaf that carries the excess
    (`[07a-levin-leaf-beside-a-window-understates-its-error]`).
  - A move of any of the five is reported with the regions it changed.
- **W6 is scored on $r_g$** (§9.1). It is predicted to fail on `3-14501` ($r_g$ 1.39) and `3-35006`
  (1.08), as at 07, and to hold on the other 2,482 pairs.
- **W7 — the stop's change is small.** The rule covers every call that has a window at 07 and at
  $h$, no region whose route or LU choice differs (the diagnostics), and no newly located phase.
  - Each such call moves from 07's value by at most $10^{-5}$ of 07's abserr, and keeps 07's
    `converged`.
  - **Holds** if true at both rtols.
  - Basis: (d) against (a) on 430 calls, largest move $1.76\times10^{-7}$ of abserr and no flag
    changed. The threshold is 57 times that.

## Deviations from the prompt

1. **`STRUCTURALLY REQUIRED`** — `AdaptiveLevin/tests/test_levin_quadrature.py` changes beyond an assertion:
   `_TwoIndependentSinCosBasis.build_Levin_data` now returns $V$ (§1.1 item 3: "The two test bases ... follow the
   same rule"), through a helper `_phase_variation` added there and imported by `test_levin_stationary.py` (for
   `_TwoPhaseQuadraticBasis`) and the new module. The dispatch's file list admits another module of
   `AdaptiveLevin/tests/` "only for an assertion prompt §3 allows"; the prompt's §1.1 item 3 and §8 ("the test
   modules") require this one. No assertion in it changed.
2. **`IMPLEMENTATION CHOICE`** — the skipped phase's entry is `{"phase": i, "reason": "zero_rate"}`. A phase with no
   finite scan sample at all is skipped too ("exactly 0.0 at every finite scan sample" holds vacuously); Rule A
   skipped it as well, with variation 0.
3. **`IMPLEMENTATION CHOICE`** — the node-partition sum is summed correctly rounded (`math.fsum`), so its bits do
   not depend on the nodes' order; where `build_Levin_data` samples the raw phase on the nodes already (no
   `theta_deriv`, or no `theta_mod_2pi`), the end values are taken from that sample, otherwise the two ends are
   sampled through `_sample_vectorized` under the same cache key `("theta", id(self))`.
4. **`IMPLEMENTATION CHOICE`** — a non-finite $V$ (a non-finite raw phase at a sampled node) raises `ValueError`
   after a warning, as a non-finite $\theta'$ already does. Before 07a the production path (`theta_deriv` and
   `theta_mod_2pi` supplied) never evaluated the raw phase in `build_Levin_data`; a NaN $V$ would otherwise send the
   region to the Levin solve and to least squares silently.
5. **`IMPLEMENTATION CHOICE`** — the names: `_window_bisection_counts`, `_cc_leaf_unresolved`, `_own_error`,
   `used_interval.own_err` and `.resasc`, `_DQK21_SATURATION_FACTOR`, `_node_partition_variation`. The
   Clenshaw–Curtis solve's dict (per region, internal) gains a `"resasc"` key; no key of `_adaptive_levin`'s result
   changes. `used_interval._combine` is split into `_combine` and `_combine_terms(resolution, floor)` so that
   `own_err` shares its arithmetic; `total_err` and `budget_err` are unchanged (every test that records them
   bit for bit passes, test 8 of `test_levin_stationary.py` and test 13 (a) among them).
6. **`IMPLEMENTATION CHOICE`** — 14 (c) is one assertion added to test 10 (b) (`iroff1_count` at the limit), not a
   separate test: 10 (a) already asserts the limit and 10 (c) a count of zero.
7. **`IMPLEMENTATION CHOICE`** — the records written beside the LiouvilleGreen harness in
   `docs/levin-phase-floor/07a/code/`: the reference script for tests 2 and 14 (d), and the breakage script. The
   dispatch named that directory for the draws' harness and outputs; nothing is written under `measurement/`.
8. **`IMPLEMENTATION CHOICE`** — test 5 is test 13 (b) rewritten in place (§3 test 6: "(b) becomes test 5"); the
   class keeps its name `TestRuleA`; the constant `T13B_V`, read only by the old 13 (b), is removed.
9. **`IMPLEMENTATION CHOICE`** — test 3's two other shapes, $e^{4x}$ and a Lorentzian rate weighted to the left
   end; the regions the driver visits are recorded by `mock.patch.object(lq, "_adaptive_levin_subregion", ...)`.
   Tests 2, 3 and 8 recompute the parent's estimate in the test (`_parent_estimate`, mean $|\theta'|$ on the nodes
   times the width) to show where it and $V$ disagree; the parent's routes are quoted from runs of `a9f6abc3`.
10. **`IMPLEMENTATION CHOICE`** — how the breakages are applied (`docs/levin-phase-floor/07a/code/breakages_07a.py`):
    (i) and (v) put the estimate's expression back at the gate and at the LU choice; (ii) puts Rule A's code back
    at step 1; (iii) skips a phase without samples of both strict signs; (iv) forces the $|\theta(b)-\theta(a)|$
    branch; (vi) sets the value tolerance to $10^{-5}$; (vii) removes the skip; **(viii) makes `_own_error()`
    return `budget_err`** (the leaf's $e'$), the one place the count reads the leaves' errors, since the count
    function takes them as arguments; (ix) sets the factor to 1.
11. **`STRUCTURALLY REQUIRED`** — §3's window test as first written asks for a stop that leaves a call converged
    and dishonest at the parent "or, if no such case can be built, says why". §9.4's 14 (d) specifies the case
    actually built (07 counts a bisection whose value change lies between rtol and $10^{-5}$, the commit does not);
    no converged-and-dishonest case was found (both codes honest on every case tried), and 14 (d)'s docstring says
    why; 14 (a) asserts the rule directly.
12. **`IMPLEMENTATION CHOICE`** — the LiouvilleGreen harness is a cut-down copy of 07's
    (`docs/levin-phase-floor/07/calibration/scripts/lg_seeded_draws.py`), not 07's run as is: 07's loads its
    parent by `git show` and carries 06x modes and recorders; here each tree runs its own module, the parent from
    the worktree the dispatch asks for. It adds the route instrument (a wrapper on `build_Levin_data` that compares
    the gate's and the LU choice's decisions under the estimate and under $V$ from the same $\theta'$), and its
    result digest sets aside `location_skipped_phases`, whose entry form changes by design.
13. **`UNINTENDED DRIFT`, corrected before the commit** — 14 (b)'s smooth leaf as first written ($\theta = 3x$ on
    [0, 1/4]) had a nested-pair estimate exactly 0, so it was "resolved" by the zero clause, not by being in its
    asymptotic regime; changed to $\theta=10x$ on [0, 1] (estimate $4.1\times10^{-9}$) with an assertion that it is
    positive.
14. **`UNINTENDED DRIFT`, corrected** — the first "before" run of the seven roots was started in the main checkout,
    where the code was about to be edited; it was stopped at once (AdaptiveLevin only had started) and the
    baseline re-run in the parent worktree, untouched for the whole session.
15. **`STRUCTURALLY REQUIRED`** — **the stop.** §8: "a LiouvilleGreen test fails under the commit that passes under
    the parent". `test_the_raw_phase_is_never_evaluated` forbade evaluating `theta["theta"]`, which §1.1 item 3
    requires, and `LiouvilleGreen/` could not change; the agent stopped before the commit, with its work in place,
    and resumed on the amendment §10 (`dadd63a0`), on the user's ruling.
16. **`IMPLEMENTATION CHOICE`** — §10.3 item 1's choices: the name `test_the_raw_phase_enters_no_value_or_error`;
    $C = 10^6 + 0.5$ as the class attribute `RAW_PHASE_SHIFT`; the decisions recorded by wrapping
    `levin_quadrature._adaptive_levin_subregion` (a helper method `_run_recording_decisions`), each region solve's
    span, Clenshaw–Curtis or Levin, forced or not, and LU or least squares; the value and abserr compared by hex.
17. **`IMPLEMENTATION CHOICE`** — beyond §10.3's letter, in the same test module: `TestLevinConsumption`'s class
    docstring ended "and the raw phase is never evaluated", the claim the rewritten test replaces; it now says the
    raw phase's absolute value enters no value or error (since 07a §10). A docstring of the test class, in the file
    §10.3 item 1 rewrites. The `bessel_phase` function's own docstring (`bessel_phase.py:932–935`), which makes the
    same claim, is **not** changed: §10.3 item 2 names `raw_theta`'s docstring only (Observation 1).
18. **`IMPLEMENTATION CHOICE`** — breakage (x) in two forms, both caught by the rewritten test: **x-cc**, the
    Clenshaw–Curtis rule's $\sin$ and $\cos$ of the raw phase in place of `theta_mod_2pi`
    (`eval_basis_and_arguments`), and **x-levin**, the Levin solve's endpoint basis $w_0$, $w_k$ from the raw phase
    (`build_Levin_data`). Both are run against `TestLevinConsumption`; (i)–(ix) against the two AdaptiveLevin
    modules as before. The whole breakage script was re-run after §10 (the record's SHA-256 is the new script's).
19. **`IMPLEMENTATION CHOICE`** — the two corrected LiouvilleGreen docstrings also replace their stale line references
    (`levin_quadrature.py:948-952`, `:1038`) by `_Basis_SinCos.__init__`, the function that raises, and cite 07a §1.1
    and §10.

## Verification performed

1. **The seven roots,** each run serially by `CLAUDE.md`'s command; "before" in the parent worktree at `a9f6abc3`
   (untouched), "after" in the main checkout with the change; `uptime` at the start of each root:

   | Root | Before (parent worktree, `a9f6abc3`) | After, first run (stopped on) | After, §10 (main checkout) |
   |---|---|---|---|
   | AdaptiveLevin | 107 OK (01:27, load 6.97) | 116 OK (01:53, 6.05) | 116 OK (10:56, 15.63) |
   | ComputeTargets | 781 OK (01:27, 6.68) | 781 OK (01:54, 6.00) | — (not re-run: nothing it imports moved but two docstrings) |
   | CosmologyModels | 39 OK (01:33, 4.93) | 39 OK (02:01, 5.95) | — |
   | LiouvilleGreen | 156 OK, skipped 1 (01:33, 4.78) | 156, **2 failures** (the stop), skipped 1 (02:01, 5.71) | 156 OK, skipped 1 (10:57, 12.66); one test renamed |
   | ObjectFactories | 280 OK (01:34, 4.55) | 280 OK (02:02, 5.65) | — |
   | RunDrivers | 127 OK, skipped 1 (01:35, 4.35) | 127 OK (02:03, 5.51) | — |
   | tools | 64 OK (01:35, 4.95) | 64 OK (02:03, 6.21) | — |

   The `levin_quadrature.py` of the §10 run is the first run's, byte for byte (SHA-256 `4ea1948b…`); §10 changed only
   the LiouvilleGreen test and two LiouvilleGreen docstrings (the AST check, §10.3 above), so ComputeTargets,
   CosmologyModels, ObjectFactories, RunDrivers and tools were not re-run.

   The rise is AdaptiveLevin's alone, +9, the tests added (6 in `test_levin_phase_variation.py`, 3 in
   `TestWindowCount`); the "before" counts equal the orchestrator's baselines.
2. **Existing assertions changed** (§3's allowance: a test whose problem has a region whose route changes, or a
   window; each is test 13, which §3 test 6 rewrites):

   | Test | Assertion | Parent | Commit |
   |---|---|---|---|
   | `test_13a_zero_rate` | `location_skipped_phases` | `[{"phase": 0, "variation": 0.0}]` | `[{"phase": 0, "reason": "zero_rate"}]`; value, abserr and regions unchanged (`0x1.815a092491c78p-5`, $2.472\times10^{-15}$, 1 region) |
   | `test_13b_small_phase_with_a_root` (now test 5) | nothing located, the phase skipped with its variation within the trapezoid's bound, `num_guarded_regions` 1, bit-identical to the bypassed call | skipped (18.475242613612156), guarded 1, value `0x1.58f28f18e1696p-3`, abserr $1.962\times10^{-13}$, 5 regions, depth 3 | one root located, one window, guarded 0, nothing skipped; value `0x1.58f28f18e1690p-3`, abserr $4.215\times10^{-12}$, 6 regions, depth 2; converged and honest under both ($1.4\times10^{-4}$ and $3.3\times10^{-5}$ of abserr) |
   | `test_13d_several_phases` | `location_skipped_phases` | `[{"phase": 1, "variation": 0.0}]` | `[{"phase": 1, "reason": "zero_rate"}]`; value and abserr bit-identical (`0x1.a309197cde8e0p-5`, $1.965\times10^{-14}$) |

   And, under §10.3 item 1 (not §3's allowance): `LiouvilleGreen/tests/test_bessel_compatibility.py`'s
   `test_the_raw_phase_is_never_evaluated` (each span with a raising `theta`: value and abserr equal) is replaced by
   `test_the_raw_phase_enters_no_value_or_error` (each span with `theta + C`: value, abserr and every region's
   decision equal). At the parent the old test passes; under the commit the old one fails on both spans (the stop)
   and the new one passes.

   Added, not changed: one assertion in `test_10b_untrusted_leaves_at_noise_stop` (14 (c); `iroff1_count` 6 under
   both codes). Docstrings rewritten: `TestRuleA` and tests 13 (a)–(d), 10 (b)'s comment. No convergence
   assertion moves; no honesty assertion fails. Tests 10 (a)–(c), and every other test of the seven roots, pass
   unchanged; tests 10 (a)–(c) were also run at the parent and are bit-identical in value, abserr, regions and
   windows.
3. **The breakages** (`docs/levin-phase-floor/07a/code/breakages_07a.py`, `out/breakages_summary.txt`; each applied
   to the module, the two test modules run, the module's bytes written back; its SHA-256 `4ea1948b…` the same
   before and after):

   | Breakage | Must fail | Failing (tests) |
   |---|---|---|
   | (i) the gate on the mean-of-nodes estimate | tests 2 and 3 | test 2, test 3 |
   | (ii) Rule A back | test 5 | test 13 (b) = test 5; also 13 (a), 13 (d) (the entry's form) |
   | (iii) a phase skipped without a strict sign change | test 7 | test 7; also `test_3_minimum_located_and_windowed`, 13 (d) |
   | (iv) $\|\theta(b)-\theta(a)\|$ on every region | test 4 | test 4 |
   | (v) the LU choice on the old estimate | test 8 | test 8 (both subtests) |
   | (vi) the value test at a fixed $10^{-5}$ | 14 (a) | 14 (a); also 14 (d) |
   | (vii) the skip removed | 14 (a) | 14 (a) (three subtests) |
   | (viii) the count on $e'$ (`_own_error` returning `budget_err`) | 14 (a) | 14 (a) |
   | (ix) the factor 200 replaced by 1 | 14 (b) | 14 (b) |
   | (x-cc) the Clenshaw–Curtis rule's sin and cos of the raw phase | the rewritten LiouvilleGreen test | `test_the_raw_phase_enters_no_value_or_error`; also `test_the_declared_phase_error_dominates_the_reported_abserr` |
   | (x-levin) the Levin endpoint basis from the raw phase | the rewritten LiouvilleGreen test | `test_the_raw_phase_enters_no_value_or_error` (both spans) |

   The first run of (i)–(ix) from the session scratchpad, and the recorded run before the stop, gave the same
   failures; the record is the run after §10.
4. **§1.2 against Part A's reference implementation.** Part A's `levin_quadrature_07a_d.py` (`_VARIANT_07A = "d"`,
   `_MEASURE_07A` None) and the commit, through `adaptive_levin_sincos`, on eight problems with a window: tests 10
   (a), 10 (b), 10 (c), test 5 of 07 (P5), test 13 (c), and 14 (d)'s problem at rtol $10^{-9}$ and, with atol
   $10^{-30}$ and depth_max 14, at $10^{-7}$ and $10^{-8}$: value, abserr, `converged`, every region's span, depth,
   abserr, value, value_own, budget_err, trusted, phase_limited and window, and each window's count, stop and split,
   **bit-identical in all eight**. None of them has a region whose route changes.
5. **The LiouvilleGreen draws:** below.
6. **`black`:** `--check` clean on every Python file changed or added
   (`AdaptiveLevin/levin_quadrature.py`, the three AdaptiveLevin test modules, the three LiouvilleGreen files, the four scripts under
   `docs/levin-phase-floor/07a/code/`); each script was formatted before it ran.
7. **§10's AST check:** `LiouvilleGreen/bessel_phase.py` and `LiouvilleGreen/three_bessel_integrals.py`, parsed at
   `HEAD` and in the working tree with every docstring removed: `ast.dump` equal for both.
8. **Scope:** `git status` shows only the files listed in the commit; nothing under `LiouvilleGreen/` beyond §10.3's
   three,
   `ComputeTargets/`, `var/` or `docs/levin-phase-floor/07a/measurement/` changed; `SIX_PI`, $20\pi$, $\Phi_w$,
   $\Lambda_c$, the trust rule, $\nu_B$, `DEFAULT_LEVIN_MAX_DEPTH` and `rtol` are untouched.

**The LiouvilleGreen draws** (`docs/levin-phase-floor/07a/code/out/lg_draws_report.txt`, `lg_draws_changes.txt`):
`test_3bessel_analytic.py`'s four default tests over seeds 0–19, 80 units per code, each in its own process (limit
1,200 s), the parent from the worktree at `a9f6abc3` (module SHA-256 `4e5da371…`), the commit from the main checkout
(`4ea1948b…`); harness `lg_draws_07a.py` (`33edc1fa…`).

| | parent | commit |
|---|---|---|
| units passing | 80 of 80 | 80 of 80 |
| **tests failing under the commit that pass under the parent** | — | **0** |
| Levin calls, paired by unit and index | 4,320 | 4,320 (none unpaired) |
| regions gated (not `force_direct`) | — | 493,516 |
| regions whose C2 route differs under 07's estimate | 0 (by construction) | 434, all to the Clenshaw–Curtis rule; $V$ 2.54–6.00 $\pi$, 0.31–0.999 of the estimate |
| regions whose LU choice differs (Levin-routed) | 0 | 771 |
| regions whose $V$ took the node-partition sum | — | 0 |
| calls with a guarded leaf | 0 | 0 |
| calls newly located (a phase Rule A skipped) | — | 0 |
| windows stopped | 83 | 83 |

| Levin calls | calls | not bit-identical | `converged` (parent → commit) | largest $\|v_c - v_p\|/{\rm abserr}_p$ |
|---|---|---|---|---|
| no window at either code, no region whose route or LU choice differs | 2,973 | **0** | — | 0 |
| a region whose route or LU choice differs | 1,024 | 1,024 | T→T 933, F→F 88, **T→F 3** | 0.0233 |
| a window (at either code) | 423 | 104 | T→T 419, F→F 4 | $3.33\times10^{-5}$ |

Honesty at the integral level (each three-Bessel result against its closed form, $|v-{\rm analytic}|\le{\rm abserr}$),
by the Levin calls it holds:

| three-Bessel results holding | results | honest, parent | honest, commit | dishonest and converged, commit | worst $\|{\rm err}\|/{\rm abserr}$, parent → commit |
|---|---|---|---|---|---|
| a call with a region whose route or LU choice differs | 674 | 426 | 427 | 247 | $1.2\times10^6$ → $4.86\times10^6$ |
| a call with a window (and no changed region) | 251 | 94 | 94 | 157 | $1.35\times10^6$ → $1.35\times10^6$ |
| neither | 155 | 123 | 123 | 32 | $5.35\times10^4$ → $5.35\times10^4$ |

The dishonest results are dishonest at both codes (the known phase-fit error of the three-Bessel oracles, which the
Levin abserr does not carry; the tests' tolerances pass). **No result goes from honest to dishonest;** one goes from
dishonest to honest (seed 15 Y000, 1.18 → 0.212). The three calls going from converged to unconverged and the
results whose ratio rises: Observations 4 and 5.

## Observations not acted on

1. **Comments made stale outside this prompt's files.** `LiouvilleGreen/bessel_phase.py:932–935` (at `HEAD`, 929–931), the `bessel_phase`
   function's docstring, still says `theta` "is never *evaluated* when the other two accessors are present" and
   that `phase_span` "is computed from" `theta_deriv` (`:1090`); `LiouvilleGreen/three_bessel_integrals.py`'s
   `residual_log_deriv` docstring (:589–590) and `ComputeTargets/tests/test_quadsource_integral.py:2121–2122` say
   Levin's subdivision (`phase_span`) consumes $\theta'$. Since 07a $V$ is computed from the raw phase. §10.3
   corrects only `raw_theta`'s and the group `theta`'s docstrings, so these are left (deviation 17). Proposed for the
   board in 07a's records commit.
2. **The raw phase now routes.** The gate and the LU choice read $|\theta(b)-\theta(a)|$ from the caller's raw
   phase, which until 07a the production path never evaluated in `build_Levin_data`. $V$'s error is therefore the
   raw phase's own error at the two ends, not only its rounding: a raw phase whose fit or accumulation error
   reaches a fraction of a radian at $|\theta|\sim10^{13}$ would move $V$ by that much. Not measured here (the
   draws' 434 route-changed regions have $V$ 0.31–0.999 of the estimate, all towards the Clenshaw–Curtis rule, as
   §0.2's end-weighting predicts, which says nothing about the raw phase's error). Part C's census sees the
   production phases.
3. **The largest raw phase in production** (§0.2's precision statement) is not measurable on the laptop. The
   LiouvilleGreen draws reach $|\theta| = 1.02\times10^{13}$ (ulp $1.95\times10^{-3}$ rad); `test_levin_stationary`'s
   P1, $1.5\times10^{9}$ (ulp $2.4\times10^{-7}$). Part C's census can record it.
4. **Three LiouvilleGreen Levin calls go from converged to unconverged**, recorded on
   **`[07-c2-gate-threshold-was-never-measured]`** (prompt §10.4; the orchestrator entered them on the board and in
   `docs/OPEN_ISSUES.md`). No test fails; each three-Bessel result they belong to passes. All at rtol $10^{-10}$, no
   window, each with one region whose route changed from Levin to Clenshaw–Curtis at the threshold:

   | call | region | $V$ | 07's estimate | regions | abserr | value moved / parent's abserr |
   |---|---|---|---|---|---|---|
   | seed 3 `test_YJJ` call 4 | [1.76079, 2.16501] | 18.832 | 18.914 | 11 → 42 | $5.38\times10^{-13}$ → $1.39\times10^{-12}$ | $6.8\times10^{-4}$ |
   | seed 8 `test_JJJ` call 4 | [1.0055, 1.83755] | 18.725 | 19.065 | 17 → 45 | $2.41\times10^{-12}$ → $4.64\times10^{-12}$ | $5.3\times10^{-3}$ |
   | seed 17 `test_JJJ` call 11 | [2.75185, 3.14675] | 18.816 | 18.892 | 9 → 27 | $1.04\times10^{-12}$ → $5.67\times10^{-12}$ | $8.1\times10^{-3}$ |

   The Clenshaw–Curtis floor on such a region ($1.5$–$1.7\times10^{-13}$ per leaf at depth 9–10 in seed 3's call) sums
   above the request ($6.8\times10^{-13}$), its leaves are found floor-limited, and the call stops unconverged at
   about twice its request. Evidence that, with the gate reading the true variation, $6\pi$ is high for
   Clenshaw–Curtis at tight tolerances: a question for that issue, not acted on here
   (`out/lg_draws_changes.txt`).
5. **Abserr falls on some near-singular three-Bessel results** (`test_YJJ_log_singularity`, rtol $10^{-8}$): 8
   results with $|{\rm err}|/{\rm abserr}$ more than doubled, the largest seed 17 Y022 from $1.03\times10^6$ to
   $4.86\times10^6$ and seed 7 Y000 from 3.02 to 7.21; each already dishonest at the parent, its true error (the
   phase fit near the log singularity, $2.4\times10^{-8}$–$4.4\times10^{-5}$) unchanged. One result goes from
   dishonest to honest (seed 15 Y000, 1.18 to 0.212). None goes from honest to dishonest. The test's tolerances
   (`SINGULARITY_*`) pass under both.
6. **§3's converged-and-dishonest window case was not built** (deviation 11): the evidence (Part A §9.2, and every
   case tried here) is that a stopped window's declarations cover its error.
7. **No region took the node-partition sum** in the LiouvilleGreen draws (493,516 regions gated): with
   `theta_deriv` supplied every phase is located, as W5 predicts for the census; the node-partition branch is
   exercised by test 4 (no rates) and by tests 1's spectral variant only where $\theta'$ has one sign.
8. **Cost.** The production path now evaluates the raw phase at two points per gated region solve; the draws'
   summed wall is 762 s against the parent's 692 s (on a shared machine). Recorded, not an argument (README §5
   rule 4).

## State handed on

- The code commit, not pushed. The orchestrator reviews it and pushes on the user's word; Part C (§5, R30 against
  R29; §6, the census against 07's, scored W0–W7) then runs on the Mac Studio at the pushed SHA.
- **For §6's diagnostics:** `docs/levin-phase-floor/07a/code/lg_draws_07a.py`'s route instrument (a wrapper on
  `_Basis_SinCos.build_Levin_data` beside one on `_adaptive_levin_subregion_impl` for `force_direct`) computes,
  from the same $\theta'$ and nodes, the gate's and the LU choice's decisions under 07's estimate, and counts the
  node-partition sums, changing no value; a Clenshaw–Curtis region with a half sent to the Levin solve needs the
  region-pairing of test 3 in addition.
- **W1 on the draws:** 2,973 of 2,973 calls with no window at either code and no changed region are bit-identical
  to the parent's, `location_skipped_phases` (whose entry form changed) set aside; 07's census comparison sets
  aside `phase_span` and will need to set that key aside too, or compare it by its phase indices.
- **Issues:** none moved by this commit (prompt §4: in 07a's records commit after Part C). Observation 4 is already
  on `[07-c2-gate-threshold-was-never-measured]` (the orchestrator, `dadd63a0`); §10.4 adds to §6's descriptive items
  the census's list of such calls. Observation 1 (the comments §10 did not correct) is proposed for the board then.
- The parent worktree used for the parent runs was removed (`git worktree list` shows the main checkout only).
