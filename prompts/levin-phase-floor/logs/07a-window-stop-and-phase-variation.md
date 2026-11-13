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

### §5's predictions (R30), copied before they run — scored in Part C, 2026-11-12

As 07 §5, with records under `var/levin-phase-floor-07a/rehearsal/`, slug `r30-07a`, `--prompt 07a`,
`--job-name levin-phase-floor-07a`, and the fingerprint `1b332ffa…`. There is no new parent
rehearsal: **R29** (`levin-phase-floor-07-r29-guard-rule-a-20261108T123429`, the store
`var/datastores/levin-phase-floor-07-r29*`, kept) was built at `e182bc7a`, whose code is the code
commit's parent's.
1. **S0′ — R29 is the parent.** `git diff --stat e182bc7a <code commit's parent> -- . ':!docs'
   ':!prompts'` lists nothing the build imports. **Holds** if so; if not, a parent rehearsal R31 is
   run first and this prompt's comparisons use it.
   — **Holds.** The diff lists `CLAUDE.md` only (`07a/rehearsal/out/00-gate.txt`).
2. **The lock:** `acquire`, `set-pid` at the launch, `release` after the readers, each a shell
   command of its own.
   — **Held, with one slip.** Nine lock commands, each exit 0; released, the lock free and
   `ray-processes` listing nothing (`state/lock.log`). One look-only `status` shared its shell
   command with a `shasum` and an `ls` (the rehearsal agent's deviation 15).
3. **R30 against R29,** by `tools.store_value_diff` with `--json` and 05's `classify_non_qsi.py`
   (with the job names normalised in a copy, as 07 did):
   - **outside `QuadSourceIntegral`:** every value table identical, every other difference of the
     four volatile kinds, no row in one store only;
   - **inside it, every moved row tabulated:** the change against R29's and R30's `total_abserr`,
     regions, maximum depth, located points, window pieces, guarded regions and `total_converged`;
     and the rows whose `total_converged` changes, in each direction.
   — **Holds, and tabulated.** R30 `levin-phase-floor-07a-r30-07a-20261112T113806`, `1b332ffa…`.
   Outside `QuadSourceIntegral`, 360,462 value-table rows identical and 0 unclassified (timing,
   `label`, `parents_digest` 291 and `rho_reuse` 19). Inside it, 380 of 864 rows moved, the largest
   $|\Delta{\rm total}|$ 0.0470 of R29's `total_abserr`; located rows 170 → 202, guarded analytic
   blocks 4 → 0; `total_converged` changes 0 in either direction. 49 records moved in abserr only,
   each through one auxiliary trust-check solve whose route or LU choice changed (the replay, 493
   of 493 records bit for bit).
4. **S3 — consistency.** Every row with finite `total_abserr` in both has
   $|{\rm total}_{30} - {\rm total}_{29}| \le {\rm total\_abserr}_{29} + {\rm total\_abserr}_{30}$.
   **Holds** if no row violates it. A violation is a finding, not by itself a stop.
   — **Holds:** 864 rows scored, 0 violations, the largest 0.0288.
5. **S1 and S2** re-scored on R30 as 07 §5 items 5 and 6, against 06g's references and 07's
   `p1_references`.
   — **Both hold.** S1: 12 of 12 locate the root at $z = 10211.31789$ and are honest, the worst
   0.2176 (w11), equal to R29's bit for bit. S2: 0.0202 and 0.263 of their abserr.
6. **S4 — O9's four blocks.** R29's four guarded analytic J4/Y4 blocks: in R30 each locates its
   root, guards no leaf, and is honest against R29's block references
   (`docs/levin-phase-floor/07/rehearsal/r29/out/block_reference/`).
   — **Holds:** each of the four locates one root at the reference's, guards nothing, and is
   4.37e-5, 5.81e-4, 2.06e-4 and 6.17e-4 of its abserr from the reference.
7. **The readers** run on R30, exit 0, and the store is byte-identical before and after. **The
   lock** is released and `ray-processes` lists nothing. **R29 and R30 are kept.**
   — **Holds:** 6 of 6 readers exit 0, R30 identical before and after; R29 unchanged; both kept.

### §6's predictions (the census), copied before they run, with §9.5's additions — scored in Part C, 2026-11-12

**Gate W0, before the run:**
- **The parent's code is 07's:** as S0′.
- **The harness reproduces 07's census at the parent,** bit for bit by 07's comparison, on 06k's
  100 K0 units. One registered run, slug `k0`.
- **The instrument changes nothing:** one unit with a window and one with a route change, at $h$,
  run with and without it, bit-identical.
- **The new code is deterministic:** two runs of one $h$ unit with a window, bit-identical.

— **W0 holds**, all four clauses: the K0 workers loaded no file S0′'s diff lists; run
`levin-phase-floor-07a-k0-20261112T130135` reproduces 07's census on 100 of 100 K0 units (2,110
calls, 32,956 leaves); `h-1e-8-1-2174` and `h-1e-7-1-11214` with and without the instrument, and
`h-1e-8-1-2174` twice, bit-identical.

**Predictions, fixed before the run** (README §5 rule 3), copied into `state/predictions.md`
unchanged, with §9's additions:
- **W1 — nothing else moves.** Every call with no window at 07 or at $h$, and no region whose route
  or LU choice differs, equals 07's call bit for bit, with `phase_span` set aside. Every row all of
  whose calls are such equals 07's in `total` and `total_abserr`. **Holds** if true at both rtols.
  — **Holds**, on the user's readings at stop 2 (*"Yes, A for Q1 and call's outputs for Q2"*:
  every solve counts as a region whose route or LU choice may differ, and "bit for bit" compares
  the call's outputs). 258,824 untouched calls, 0 differing in their outputs; 13,382 differ only in
  the skipped-phase record (07's `{phase, variation}` against 07a's none or `zero_rate`); 9,726 rows
  made only of untouched calls equal 07's in `total` and `total_abserr`, in hex.
- **W2 — no Levin leaf is returned across a stationary point** (07's V2): the scan finds no returned
  Levin leaf holding a sign change of $\theta'$.
  — **Holds:** 182,066 calls scanned, 0 returned Levin leaves holding a sign change.
- **W3 — every phase with something to locate is located.** The location agrees with the scan both
  ways (07's V3), with no phase set aside except one whose rate is identically zero; every skipped
  phase is listed with its call, and each is of that kind.
  — **Holds:** 0 violations either way; 784 skipped phases, all `zero_rate`, none with a sign
  change.
- **W4 — honest where it is referenced.** V4's 33 calls (z00, z12, Z3's seven and the 24),
  re-scored from the census against the readings 07 scored them on, by 07's rule. **Holds** if
  every scored call is honest, or is unconverged under a rule of §9's amendment.
  — **Fails, as §9.5 predicts:** 28 of 33 honest; v08, v09, v12, v17 and v22 at 3.16, 2.21,
  3.08, 2.38 and 2.53. All 33 calls are bit-identical to 07's.
- **W5 — the gate is consistent, and the guard is a backstop.** In every call: no Clenshaw–Curtis
  region has a half sent to the Levin solve; no region takes the node-partition sum; no region is
  guarded (`num_guarded_regions` 0). **Holds** if all three are zero. A non-zero count is a finding,
  with its calls listed.
  — **Holds on gated solves**, on the user's ruling (*"Yes, W5 gated only, and take references
  for both calls"*): 0 Clenshaw–Curtis regions with a half sent to Levin, 0 gated solves taking
  the node-partition sum, 0 guarded regions. **Descriptive:** 18,236 forced solves (window pieces
  whose end node is the root, and their halves and quarters) take the node-partition sum, 3–10
  per call, in 4,416 of the 4,804 root-located calls; their $V$ routes nothing but is reported as
  `phase_span`.
- **W6 — the pairs.** The cross-check's $r$ re-computed at $h$ over all 2,484 pairs: every pair has
  $r \le 1$, or a member unconverged under §9's rule. `3-14501` and `3-35006` are listed.
  — **Fails, as §9.5 predicts:** see W6's addition below.

§9.5's additions:

- **W4 is predicted to fail** on exactly v08, v09, v12, v17 and v22, at 3.16, 2.21, 3.08, 2.38 and
  2.53 to three figures, and to hold on the other 28 calls.
  - Basis: §9.2. Neither §1.2 nor O9 reaches the leaf that carries the excess
    (`[07a-levin-leaf-beside-a-window-understates-its-error]`).
  - A move of any of the five is reported with the regions it changed.
  - — **As predicted, exactly:** the five at 3.16, 2.21, 3.08, 2.38 and 2.53; none moved.
- **W6 is scored on $r_g$** (§9.1). It is predicted to fail on `3-14501` ($r_g$ 1.39) and `3-35006`
  (1.08), as at 07, and to hold on the other 2,482 pairs.
  — **As predicted:** $r_g \le 1$ in 2,482 of 2,484 pairs; `3-14501` at 1.385 and `3-35006` at
  1.084, both members converged, as at 07.
- **W7 — the stop's change is small.** The rule covers every call that has a window at 07 and at
  $h$, no region whose route or LU choice differs (the diagnostics), and no newly located phase.
  - Each such call moves from 07's value by at most $10^{-5}$ of 07's abserr, and keeps 07's
    `converged`.
  - **Holds** if true at both rtols.
  - Basis: (d) against (a) on 430 calls, largest move $1.76\times10^{-7}$ of abserr and no flag
    changed. The threshold is 57 times that.
  - — **Fails, not predicted.** 4,774 calls covered. At `1e-7` every one moves by at most
    $1.80\times10^{-7}$ of 07's abserr. At `1e-8`, **25 calls move by 0.0066–0.145** of it: every
    one a lone-`G` group call (group index 0) with minima windows (6 or 19 minima), at
    $k = 10^5$ (19) and $3.14\times10^5$ (6), whose window stopped at 07 (`iroff1` 6) and does not
    stop under 07a's value test at $\min(10^{-5}, {\rm rtol})$ (`iroff1` 4–5). Their abserr and
    `converged` are unchanged, and their regions change by 0 or 1. Part A's 430 calls, the
    threshold's basis, were all `G+q+r` and held no minima window. No reference was taken for
    them, so whether 07 or 07a is nearer is not known.

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

## Part C: the Mac Studio record (appended by the orchestrator, 2026-11-13)

Part C ran on 2026-11-12 on the Mac Studio (`mps026298`), through an orchestrator session there, from the worktree
`SecondaryGWKit-lpf07a-4bf50d3` at Part C's dispatch SHA `4bf50d34`, whose code is the code commit `1ec734d7`'s (the
orchestrator note's §C.5). The Mac Studio orchestrator dispatched a rehearsal agent (R30), then a census agent, and
held them at the note's planned stops 1, 2 and 3. A fourth stop came at scoring, on W5's wording. It checked each
stop with readers of its own, then reviewed Part C in the note's §C.2 checks, **all passing**. The records are in
`docs/levin-phase-floor/07a/`:
- `rehearsal/` and `census/`: each agent's log, verification, issues, scripts and outputs. The two logs are appended
  below as written on the Mac Studio;
- `review/`: the Mac Studio orchestrator's stop records, scripts and hashes, and its review (`review/review.md`).

What stays on the Mac Studio is in `07a/KEPT_ON_MAC_STUDIO.md`. `VERIFICATION.md` §7a is the summary. Part A's log
follows the census agent's, in full, with its headings moved down one level to sit under this log's own.

**The verdicts on the predictions above** (each entered against its rule there too):

| | Verdict | Numbers |
|---|---|---|
| §5, S0′ | **holds** | `e182bc7a..dadd63a0` lists `CLAUDE.md` only; R29 is the parent |
| §5, the fingerprint | **holds** | R30 `1b332ffa…`, its `classes` equal R29's |
| §5, R30 against R29 | **holds, and tabulated** | outside `QuadSourceIntegral` 360,462 rows identical; inside it 380 of 864 rows moved, the largest 0.0470 of R29's abserr, no `total_converged` change |
| §5, S3 | **holds** | 0 violations in 864 rows; the largest 0.0288 |
| §5, S1 | **holds** | 12 of 12 honest, worst 0.2176 (w11), equal to R29's bit for bit |
| §5, S2 | **holds** | 0.0202 and 0.263 |
| §5, S4 | **holds** | the four blocks each locate one root, guard nothing, and are honest at 4.4e-5 to 6.2e-4 |
| §5, the readers and the lock | **hold** | 6 of 6 exit 0, R30 unchanged; the lock released; one look-only `status` shared a shell command |
| §6, Gate W0 | **holds** | 100 of 100 K0 units equal 07's census; the instrument and the code deterministic |
| §6, W1 | **holds** | on the user's readings: 258,824 untouched calls, 0 differing in their outputs; 9,726 rows equal in hex |
| §6, W2 | **holds** | 0 returned Levin leaves hold a sign change, 182,066 calls scanned |
| §6, W3 | **holds** | 0 violations; 784 skipped phases, all `zero_rate` |
| §6, W4 | **fails, as predicted** | 28 of 33 honest; v08 3.16, v09 2.21, v12 3.08, v17 2.38, v22 2.53, all bit-identical to 07's |
| §6, W5 | **holds on gated solves** | 0, 0 and 0; 18,236 forced solves take the node-partition sum (descriptive, by the user's ruling) |
| §6, W6 | **fails, as predicted** | $r_g \le 1$ in 2,482 of 2,484; `3-14501` 1.385, `3-35006` 1.084 |
| §9.5, W7 | **fails, not predicted** | 25 lone-`G` minima-window calls at `1e-8` move by 0.0066–0.145 of abserr; `converged` unchanged |

**R30.** `levin-phase-floor-07a-r30-07a-20261112T113806`, about 9.7 min, store primary `e5f8b443…`. 64 analytic
blocks newly locate their root, and the four blocks R29 guarded now locate and guard nothing. 49 records moved in
abserr alone, by −19 % to +180 %. The replay (493 of 493 records bit for bit) puts each on one auxiliary
trust-check solve, a quarter or a comparison half, whose route or LU choice changed: quarter route 19, quarter LU
22, half LU 7, half route 1. No region's own solve changed in them.

**The census.** `levin-phase-floor-07a-census-20261112T132744`, checkpoint `461df2ec…`, 18,560 of 18,560 units in
23,284 s (07: 22,708 s, +2.5 %), none over budget.
- **Classes** (both rtols): untouched 258,824; aux_only 15,847; own 7,511; window_both 4,940; window_h_only 2,360;
  window_07_only 0.
- **Convergence:** no `converged` flag changes in any class at either rtol. §10.4's list, of calls losing convergence
  through a route change, is empty.
- **Route changes:** 10,014 C2 decisions changed, every one towards Clenshaw–Curtis (07's estimate over $V$
  1.00001–2.29), and 14,545 LU decisions, every one towards least squares. The smallest unchanged-route margin to
  $6\pi$ is $4.8\times10^{-8}$ rad, to $20\pi$ $3.0\times10^{-4}$. The largest $|\theta|$ at a gated region's end is
  $8.87\times10^{12}$ (analytic) and $4.12\times10^{12}$ (group).
- **Windows:** 3,650 per rtol at 07a against 2,470 at 07; 1,180 calls per rtol newly located, all converged before
  and after. Stopped windows 1,247 against 1,157 (`1e-7`) and 1,236 against 1,195 (`1e-8`).

**The user's references** (findings, not verdicts), by 07's V4 method, on the user's word at the W5 stop: rows
`1-34599` and `3-678`, group calls 2 and 3, both rtols. 16 of 16 readings, FCC r2 the reference, the readings'
spread at most $1.3\times10^{-6}$ of the smaller abserr.
- **`1-34599`** (own): 07a honest at 0.020 (`1e-7`) and 0.015 (`1e-8`); **07 not honest, at 4.03.** One region with
  $V$ just under $6\pi$, which 07's estimate sent to Levin, now goes to Clenshaw–Curtis.
- **`3-678`** (aux_only): 07a honest at 0.491 of an abserr 11× smaller than 07's (one region against 12); 07 at
  0.135.

**The laptop's re-checks** (2026-11-13), from the copies, against committed records the agents did not write: W4 on
z12, v08 and v12 equal 07's committed census values bit for bit (0.7181, 3.16, 3.08); S1 on w03, w07 and w11
(0.0726, 0.00297, 0.2176) against 06g's committed references; an untouched rehearsal row ($k = 10^5$, $z = 250$)
equal in hex in R29 and R30, and the most-moved row ($k = 3.162\times10^5$, $z = 1.918\times10^8$) at 0.04703 of R29's
abserr, its R29 values equal to 07's committed table.

**Deviations in Part C.** The rehearsal agent logged 21: 12 `IMPLEMENTATION CHOICE`, 3 `STRUCTURALLY REQUIRED`
and 6 `UNINTENDED DRIFT`. The census agent logged 29: 15 `IMPLEMENTATION CHOICE`, 6 `STRUCTURALLY REQUIRED` and 9
`UNINTENDED DRIFT`, item 3 carrying two classes. The Mac Studio orchestrator accepted each. The drift is cosmetic, or
a failed attempt re-run with nothing used from it. The structurally required ones: the job-name prefix normalised
for S0; the classifier run on the raw JSON too; a capture recording its own run; the parent run as a variant module
in tree h, 07's worktrees being removed; 06g's `rows.json` read from 07's copy (same SHA-256), and 06m's values taken
from 07's cross-check output, those records being gone from the Mac Studio; the references, on the user's ruling;
and the references' projection refitted on 07's own readings.

**The user's rulings** (relayed by the laptop orchestrator):
- stop 1: *"Yes, resume the rehearsal agent with the replay"*;
- stop 2: *"Yes, A for Q1 and call's outputs for Q2; launch the census"*;
- stop 3: *"Yes, resume the census agent for scoring"*;
- the W5 stop: *"Yes, W5 gated only, and take references for both calls"*;
- at the records commit, on the orchestrator's four proposals: *"Yes to 1–4; keep R29 for now; make the records
  commit."* So:
  1. `[07-iroff1-window-stop-can-leave-a-call-dishonest]` stays open, narrowed to W7's 25 calls, its next step
     references for two or three of them;
  2. `[07-c2-gate-threshold-was-never-measured]` loses O9 and gains the census's evidence, `1-34599` among it;
  3. `[07a-trust-check-auxiliary-route-moves-the-declared-error]` is opened from the census agent's item 5 and the
     rehearsal agent's item 1;
  4. `[07a-levin-leaf-beside-a-window-understates-its-error]` stands, with W4 and W6 reproduced.

  R29 and R30 are kept.

**Observations not acted on, and not opened as issues** (the agents' `issues.md`):
- **Retention:** 06g's census records are gone from the Mac Studio, and 06m's census checkpoint was trimmed on
  2026-11-10 (`TRIMMED_2026-11-10.txt`). 07's worktrees are removed. A later prompt reading those meets the same
  absence.
- **Analytic block records store no value**, so a rehearsal that must score a block's value plans a replay, as S4's
  did.
- **The readers draw random subsamples** of $k$ and $z$ (`extract_Gk_data.py`, `extract_GkSource_data.py`), so their
  output is not comparable file for file across rehearsals.
- **`route_07a` over the replay**, 427,437 solves: 312 differ in C2 route and 430 in LU choice, mostly trust-check
  quarters and comparison halves; descriptive.
- **The forced solves' node-partition sums** (W5's descriptive count) route nothing, but each is reported as its
  region's `phase_span`.
- **Whether other own-class calls are like `1-34599`** (converged and dishonest at 07, with no window) is not
  measured: 5,314 own calls moved. It is recorded on `[07-c2-gate-threshold-was-never-measured]`.
- The comments made stale by §1.1, which the rehearsal agent met, are corrected in `89c368f2`, a docstring-only commit
  after Part C's dispatch SHA that touches nothing Part C ran.

## Part C §5: rehearsal R30 on the Mac Studio

**Prompt:** [`07a-window-stop-and-phase-variation.md`](../07a-window-stop-and-phase-variation.md) §5 ("as 07 §5"), under
Part C's dispatch (`4bf50d34`) and the user's word at stop 1: *"Yes, resume the rehearsal agent with the replay"*.

**What was measured:** the code commit `1ec734d7`: the C2 gate on the raw phase, Rule A replaced, and the window's count
as `dqage`'s. Its parent is `dadd63a0`. The parent rehearsal is R29, built at `e182bc7a`, whose code is the parent's.

**Host and trees:** mps026298.
- Every build, replay, score and reader ran in the worktree `SecondaryGWKit-lpf07a-4bf50d3`, detached at `4bf50d34`.
  - Its key files: `levin_quadrature.py` `4ea1948b`; `QuadSourceIntegral.py` `33cb161e`; the launcher `b0135218`;
    `main.py` `a5d47f5f`; `store_value_diff.py` `7997af85`. The last four are R29's tree's.
- Nothing was written in the worktree; `status --short --ignored` was empty throughout.
- The main checkout stayed at `28ebaf07`.

**Model:** Claude Opus 5.5, the rehearsal subagent, in four turns:
1. the gate and R30's launch;
2. the end check, the comparisons, and stop 1;
3. the replay's set-up and launch;
4. the trust reading, S1–S4, the readers, the release and these records.

**Date:** 2026-11-12, 11:37–12:35 UTC. **Commit:** none; the laptop commits these records.

**Result:** COMPLETE.
- **R30 is `done` with `1b332ffa…`.** S0′ holds.
- **Outside `QuadSourceIntegral` every value table equals R29's.**
- Inside it, 380 of 864 rows moved, and no `total_converged` changed in either direction.
- **S3, S1, S2 and S4 hold.** The readers exit 0 with the store unchanged. The lock is released.
- The 49 abserr-only records are explained, by the replay, as a route or LU change in one auxiliary trust-check solve
  each (`issues.md` item 1).

### What shipped

No code. The records under `var/levin-phase-floor-07a/rehearsal/` (committed by the laptop as
`docs/levin-phase-floor/07a/rehearsal/`): `log.md`, `verification.md`, `issues.md`, `KEPT_ON_MAC_STUDIO.txt`, and:
- **`scripts/`:**
  - adapted:
    - `rec.sh` (`065afcc6`; R29's, path by sed, `out/rec_sh.diff`);
    - `normalise_jobname_labels.py` (`41da5c72`; 07's, `out/normalise_jobname_labels.diff`);
    - `block_reference_r30.py` (`efac125e`; R29's, `out/block_reference_r30.diff`);
    - `s1_score_r30.py` (`0f2a0469`) and `s2_score_r30.py` (`6fc3fb30`), both R29's, changes marked;
  - unchanged copies:
    - `classify_non_qsi.py` (`26690e61`, 05's);
    - `tabulate_qsi_guard_r29.py` (`47e18cbe`, kept as the source of the verbatim reader);
    - `run_readers_worktree.sh` (`ec96cf09`);
  - new:
    - `tabulate_qsi_r29_r30.py` (`adff0052`) and `qsi_summary_r29_r30.py` (`cc7bc729`);
    - `s3_score.py` (`f12ebafd`);
    - `trust_reading.py` (`49f1456c`);
    - `run_block_reference_r30.sh` (`4a2bd2c3`) and `block_score_r30.py` (`7c92da59`);
    - `capture_compare.py` (`f240a7c2`);
    - `replay_summary_r30.py` (`4ea0b46c`);
    - `peek_qsi_metadata.py` (`e33881f0`, deviation 8);
  - **`capture/`:** R29's on-disk harness (hashes in `state/capture_harness_r29_sha256.txt`), with `prov.py`
    (`b4cd02b9`) and `worker.py` (`63c28b49`) changed where marked "prompt 07a r30". It adds the new `replay07a_r30.py`
    (`c7ff4d85`) and `run_r30_pipeline.sh` (`c4b10450`). Diffs are in `out/capture_harness_r30_diff.txt` and
    `out/run_pipeline_r30.diff`.
- **`out/`:**
  - `commands.log`;
  - the gate, launch, end, stop-1 and final checks;
  - `r30_status.json`;
  - the diff JSON (kept, over 2 MB), its normalised copy, and the classifier outputs;
  - the QSI tabulation, its CSVs and its summary;
  - the replay's stage outputs: `replay_rows.txt`, `replay_run.txt`, `replay_check.txt`, `replay_calls.csv`,
    `replay_calls.checks.json`, `blocks1_run.txt`, `blocks2_*.txt`, `blockcheck.txt`, `blocks.json`,
    `replay_summary.txt`, `r30_pipeline.log`;
  - `trust_reading.txt`;
  - `s1_scores.*`, `s2_scores.*`, `s3_scores.*`;
  - `block_reference/`, `block_reference_run.txt`, `block_scores.*`, `capture_compare.txt`;
  - `readers.txt`, `readers_check.txt`.
- **`state/`:** `lock.log`, the launch lines and markers, every store-hash file, the replay's rows and specs, and the
  store links.
- **Not for copying back:** see `KEPT_ON_MAC_STUDIO.txt` (the diff JSONs, `extract/`, `state/capture/`,
  `state/inputs/`).
- **The store is kept:** `var/datastores/levin-phase-floor-07a-r30*`.

### Predictions and verdicts

| Rule (prompt 07a §5, as 07 §5) | Scored on | Verdict |
|---|---|---|
| 1. S0′: `git diff --stat e182bc7a dadd63a0 -- . ':!docs' ':!prompts'` lists nothing the build imports | `CLAUDE.md` only | **holds** |
| R30 with 07's command line (`--job-name levin-phase-floor-07a`, slug `r30-07a`, `--prompt 07a`, `--cpus 6`) gives `1b332ffa…` | `done`, exit 0, `1b332ffa…`, `classes` equal R29's; 9.7 min | **holds** |
| 2. The lock: `acquire`, `set-pid` at the launch, `release` after the readers, each a shell command of its own | acquired 11:37:54Z, `set-pid` 75161, released 12:31Z; one look-only `status` shared its shell with hashing (deviation 15) | **holds** (with deviation 15) |
| 3. Outside `QuadSourceIntegral`: every value table identical; every other difference of the four volatile kinds; no row in one store only | 360,462 value-table rows identical; 0 unclassified on the normalised copy | **holds** |
| 3. Inside it: every moved row tabulated, against both stores' `total_abserr`, regions, maximum depth, located points, window pieces, guarded regions and `total_converged`; the `total_converged` changes each way | 380 rows tabulated (`out/qsi_rows_r29_r30.csv`); 1,234 records (`out/qsi_records_r29_r30.csv`); `total_converged` changes 0 up, 0 down | tabulated |
| 4. S3: \|t30 − t29\| ≤ e29 + e30 on every row with finite `total_abserr` in both | 864 scored exactly; 0 violations; largest 0.0288 | **holds** |
| 5. S1: V1's twelve locate their point at z ≈ 10211 and are honest against 06g's references | 12 of 12, largest 0.2176 (w11); equal to R29 | **holds** |
| 5. S2: P1's two rows locate within one spacing and are honest against `p1_references` | both; 0.0202 and 0.263 | **holds** |
| 6. S4: R29's four guarded J4/Y4 blocks each locate their root, guard no leaf, and are honest against R29's block references | 4 of 4: the root bit-equal to the reference root, 0 guarded, ratios 4.4e-5 to 6.2e-4 | **holds** |
| 7. The readers on R30 exit 0, store byte-identical; lock released, `ray-processes` empty; R29 and R30 kept | 6 of 6; identical; free; nothing listed; R29 equal to its KEPT hashes | **holds** |

**The inside-QSI tabulation** (`out/qsi_tabulation_r29_r30.txt`, `out/qsi_summary_r29_r30.txt`):
- **Rows:** 484 unmoved and 380 moved.
  - `total` moved in 25 rows, `total_abserr` in 60, `analytic_rad` in 168 and `analytic.abserr` in 339.
  - The other 163 moved only in analytic-block fields or counters.
- **Largest changes:**
  - \|Δtotal\|/abserr is 0.0470 against R29's abserr and 0.0745 against R30's, at (3.162e5)³, z 1.918e8 (regions
    69 → 73).
  - The largest group-record \|Δv\|/abserr is 0.336 (R29) and 0.753 (R30), on G+q+r at (1e6)³, z 2.099e7: regions
    1 → 68, `phase_limited` False → True.
  - \|Δanalytic_rad\|/analytic.abserr is at most 5.5e-5.
- **Regions:** summed, 114,460 → 114,245. They change in 17 rows. Maximum depth changes in 2 rows (17 → 9 and 11 → 15).
  Five rows are at depth 20 in each store.
- **Convergence:** `total_converged` is 862 of 864 in both stores, with 0 changes either way. Group and analytic-block
  `converged` also have 0 changes either way.
- **Location:**

  | | Group records: points / window pieces / guarded | Analytic blocks: points / window pieces / guarded | Rows located | Rows guarded |
  |---|---|---|---|---|
  | R29 | 44 / 562 / 0 | 336 / 2,172 / 4 | 170 | 2 |
  | R30 | 44 / 562 / 0 | 400 / 2,515 / 0 | 202 | 0 |

  - The new points are 64 2pt5 J4/Y4 blocks in 32 rows, which R29's Rule A skipped. Each locates one root, with 2–13
    window pieces.
- **The replay** (`out/replay_summary.txt`): every located record holds exactly one root. There are 44 G+q+r, 200 J4 and
  200 Y4, with 3,077 window pieces and no record over 64 points. The four q-r calls of the replayed rows are skipped with
  reason `zero_rate`.

### Deviations from the prompt

1. **`rec.sh` is R29's** (`b5279dd9`) with its record path changed by sed. IMPLEMENTATION CHOICE.
2. **A GNU-style `stat -f` format string failed** in a read-only gate check. The hashes and `ls -la` carry the same
   information. UNINTENDED DRIFT.
3. **`normalise_jobname_labels.py` is 07's** (`b02fd3fd`), with its two prefixes (`07-` → `07a-`) and its docstring
   changed. IMPLEMENTATION CHOICE.
4. **`classify_non_qsi.py` was run on both the raw and the normalised JSON.** The raw run gives 5,148 unclassified, every
   one a `label` whose job-name prefix differs; the normalised run gives 0. The verdict is the normalised copy's, as 07's
   S0. STRUCTURALLY REQUIRED: the brief sets R30's job name.
5. **The inside-QSI tabulation is a new script**, `tabulate_qsi_r29_r30.py`.
   - R29's script classes a row "untouched" by the newer store's four counts, against a store that lacks them. That
     cannot apply when both stores carry the counts.
   - A row here is "moved" if and only if a value, counter or guard field differs.
   - The reader and the field classification are copied verbatim.
   - IMPLEMENTATION CHOICE.
6. **`qsi_summary_r29_r30.py`**, a summary of the two CSVs for the stop-1 report. IMPLEMENTATION CHOICE.
7. **The classifier prints no provenance header,** so the shell wrote one (script and input SHA-256, the tree's HEAD).
   IMPLEMENTATION CHOICE.
8. **Four read-only exploratory commands at stop 1 printed output that was not saved:** a peek at R30's metadata
   structure (`peek_qsi_metadata.py`; a first one-liner failed on sqlite quoting) and three inline analyses of the CSVs.
   `qsi_summary_r29_r30.py` re-derives and saves their content. UNINTENDED DRIFT.
9. **The capture harness is R29's on-disk copy** (hashes equal to R29's records). `prov.py` was changed to the one tree
   h and R30's store links, with the parents assertion one level shallower. IMPLEMENTATION CHOICE.
10. **`worker.py`: `guard_07`'s skipped-phase entries also record `reason`.** 07a's entry form carries no variation.
    STRUCTURALLY REQUIRED.
11. **`worker.py`: a route diagnostic, `route_07a`, added for the orchestrator's reading.**
    - Pass-throughs at `_adaptive_levin_subregion` (the existing one, extended) and at
      `_Basis_SinCos.build_Levin_data` record, for each solve: V, 07's estimate (the parent's expression on the same
      `theta_prime_Cheb`), `is_direct` and the caller.
    - It returns exactly what it is given. The replay reproducing R30 bit for bit, 493 of 493 records, shows it changed
      no value.
    - IMPLEMENTATION CHOICE.
12. **`replay07a_r30.py` is R29's driver run at one tree.**
    - The 49 abserr-only records' rows are added.
    - The check reports `route_07a`.
    - The blockcheck compares each capture with R29's.
    - IMPLEMENTATION CHOICE.
13. **A first `rows` run failed with a KeyError.** A comment I inserted had commented out the rest of a dict line. It was
    corrected before any output existed or was used. UNINTENDED DRIFT.
14. **Pass 1 of the block check (`blocks1`) ran in session, as the pilot, before the detached pipeline ran the rest.**
    The pipeline is not a registered run, as R29's was not. IMPLEMENTATION CHOICE.
15. **One look-only lock `status` ran in a shell command that had first run `shasum` and `ls`.** That is against "each
    lock command … the only process that command starts". Those processes had ended before it ran, and it changed
    nothing. UNINTENDED DRIFT.
16. **The block captures are not byte-identical to R29's,** as the dispatch's comparison asked.
    - They differ only in the unit id, the tree tag and the call's own result fields (`out/capture_compare.txt`).
    - The integrand's equality is shown instead: 1001-point samples bit-identical, with equal `x_span` and `kw`
      (`out/block_scores.txt`).
    - STRUCTURALLY REQUIRED: a capture records its own run.
17. **S4 used two scripts.**
    - `block_reference_r30.py` is R29's, with the skipped-phase form changed and the located points recorded. It was
      run at h without `--reference`.
    - `block_score_r30.py` is new.
    - R29's references were reused, as §5 item 6 says.
    - IMPLEMENTATION CHOICE.
18. **S3 is scored by a new script, exactly, in `Fraction`s.** IMPLEMENTATION CHOICE.
19. **S1 and S2 are R29's scripts, adapted. Some names were left as R29's:**
    - S1 keeps its key names `v29`/`abserr29`, which hold R30's numbers, so its JSON is byte-identical to R29's
      (`ae704a3c`). R30's twelve calls equal R29's bit for bit.
    - S2 still prints "replay reproduces R29" for R30's check, and its `R30_equals_R29` expression carries a dead
      conditional.
    - Cosmetic. UNINTENDED DRIFT.
20. **`trust_reading.py` and `replay_summary_r30.py` are new.** IMPLEMENTATION CHOICE.
21. **Two read-only looks printed output that was not saved:** the pilot's `emit.jsonl` and the block replays' JSON
    fields. The pipeline's check and `block_score_r30.py` save the same content. UNINTENDED DRIFT.

### Verification performed

- **Gate** (`out/00-gate.txt`, `out/01-*`):
  - the worktree at 4bf50d34, clean; the main checkout at 28ebaf07, clean;
  - S0′ as written;
  - `1ec734d7..4bf50d34` outside docs/prompts: empty;
  - `e182bc7a..4bf50d34` under ComputeTargets, ObjectFactories, RunDrivers, main.py, tools, config, CosmologyModels and
    requirements.txt: empty;
  - 73 runs, all finished; no lock; no Ray process; `auth_token` mode 600;
  - R29 equal to its KEPT hashes.
- **R30** (`out/r30_launch_check.txt`, `out/r30_end_check.txt`):
  - launched 11:38:05Z (sh 75153, driver 75161); manifest at 4bf50d34, clean, launcher `b0135218`;
  - `done` at 11:47:48Z; no Traceback, WARNING or ERROR.
- **Stores:**
  - R29 is byte-identical before, during and after everything (`state/r29_store_sha256_*`, `state/store_sha256_*`,
    `state/stores_sha256_final.txt`).
  - R30 is unchanged after its build: across the comparisons, the replay and the readers.
- **The comparisons:** `tools.store_value_diff` R29 against R30 (15.9 s, exit 1, `f6de2ef1`), run from the worktree.
- **The replay** (`out/replay_check.txt`, `out/blockcheck.txt`):
  - 493 of 493 records reproduce, with Ray uninitialised in all 223 workers;
  - the four block calls reproduce in pass 1 and pass 2;
  - wall time: replay 325 s, blocks 52.5 s each pass.
- **The readers** (`out/readers_check.txt`): 6 of 6 exit 0; R30 identical; 0 Tracebacks; the worktree clean.
- **The lock** (`state/lock.log`): nine commands, each exit 0: ray-processes, acquire, set-pid, ray-processes, status,
  ray-processes, release, status (free), ray-processes (nothing).
- **Final** (`out/final_check.txt`):
  - 74 runs, all finished; `~/.raypoolkit` empty; no Ray process;
  - the worktree and the main checkout clean;
  - `measurement/`, `var/levin-phase-floor-07/`, R29's run directory, A3 v2 and its scratch copy all untouched;
  - no `__pycache__` in the records.

### Observations not acted on

1. **The 49 abserr-only records** (`issues.md` item 1).
   - In each, exactly one auxiliary trust-check solve changes route or LU choice: a quarter in 41, a comparison half
     in 8.
   - No region's own solve changes, so neither does the value.
   - This refines the orchestrator's reading.
   - It bears on §6 W1's diagnostics: those must count the trust check's solves, or W1 will see calls move that it
     predicts are bit-identical.
2. **R30's four S4 block values equal R28's bit for bit.** R30's S1 and S2 calls equal R29's (and R28's) bit for bit.
3. **`route_07a` over 4,614 replayed calls:** 312 solves change C2 route and 430 change LU choice, of 427,437
   (`issues.md` item 3).
4. **The readers subsample k and z at random,** so their file counts differ from R29's (`issues.md` item 4).
5. **The stale comments the dispatch named** were met in passing and not touched (`issues.md` item 5).

### State handed on

- **R30:** `levin-phase-floor-07a-r30-07a-20261112T113806`, `done`, `1b332ffa…`. Its store is kept
  (`KEPT_ON_MAC_STUDIO.txt`).
- **R29** is unchanged and kept.
- **The lock is released.** Nothing of this agent runs. The worktree and the main checkout are clean and unmoved.
- §6, the census, was not touched.

## Part C §6: the census at 07a's code (Mac Studio), the census agent

**Subject:** prompt 07a §6, as amended by §9.5 and §10.4: the census re-run at the code commit `1ec734d7`, against 07's
census as the "before". On the Mac Studio (`mps026298`), 2026-11-12, by Part C's census agent. Covers Gate W0, the
predictions, the census, stop 3's comparison, W1–W7, the descriptive items and the records. Ray-free, no lock, no store
built.
- **The tree h:** the worktree `SecondaryGWKit-lpf07a-4bf50d3`, detached at Part C's dispatch SHA `4bf50d34`, whose code is
  `1ec734d7`'s.
  - `AdaptiveLevin/levin_quadrature.py` `4ea1948b…`; `ComputeTargets/QuadSourceIntegral.py` `33cb161e…`.
  - Clean with `--ignored` throughout. The main checkout stayed at `28ebaf07`, clean.
- **The parent:** there is no parent worktree. W0's K0 run ran in h with the parent's module installed in place of h's by
  `variant_loader.py`.
  - The module is `state/variants/levin_quadrature_parent.py`, from `git show dadd63a0`, `4e5da371…`.
  - It is byte-identical to 07's census tree `e182bc7a`'s.
- **The "before":** 07's census, `levin-phase-floor-07-census-20261108T171137`, checkpoint `03dfe786…`.
- **The store:** 06's certified scratch copy of A3 v2, read only.

**Commit:** none here. The orchestrator commits these records from the laptop as `docs/levin-phase-floor/07a/census/`.

**The user's rulings, relayed by the laptop orchestrator:**
- At STOP 2: *"Yes, A for Q1 and call's outputs for Q2; launch the census"*. The orchestrator's reading is recorded in
  `state/ruling_stop2.md`:
  - Q1, reading A: every solve counts as "a region whose route or LU choice differs" in W1 and W7.
  - Q2: "bit for bit" compares the call's outputs. The `location` and `harness` categories are set aside as `phase_span`
    is.
- At STOP 3: *"Yes, resume the census agent for scoring"*.
- At the W5 stop: *"Yes, W5 gated only, and take references for both calls"*. The orchestrator's reading is recorded in
  `state/ruling_w5.md`:
  - W5's counts are made over gated solves only. The forced solves' count is descriptive.
  - `state/predictions.md` part 5's wording ("every solve … forced included") is superseded by this ruling. The file is not
    edited.
  - References are taken, by 07's V4 method, for rows `3-678` and `1-34599`. The ruling overrides §C.1's "no new references
    are taken" for these two calls only. The results are findings, not W verdicts.

**Result:**
- **Gate W0 holds**, all four clauses.
- **The census** `levin-phase-floor-07a-census-20261112T132744` is `done`: 18,560 of 18,560 units, 0 in error, 0 over budget,
  wall 23,284 s (07: 22,708 s). Checkpoint `461df2ec…`.
- **W1 holds** under the ruled readings: 0 of 258,824 untouched calls differ in the call's outputs, and all 9,726 units made
  only of such calls have `total` and `total_abserr` equal to 07's.
- **W2 holds** (0 Levin leaves listed). **W3 holds** (0 violations; the 784 skipped phases are all `zero_rate`).
- **W4 fails, as §9.5 predicted:** exactly v08, v09, v12, v17 and v22, at 3.16, 2.21, 3.08, 2.38 and 2.53. All 33 calls are
  bit-identical to 07's.
- **W5 holds** (gated solves, as ruled): 0 node-partition sums, 0 Clenshaw–Curtis regions with a Levin half, 0 guarded
  regions. Descriptive: 18,236 forced solves took the node-partition sum, 3–10 per call, all within the root-located calls
  (4,416 of the 4,804). Their V routes nothing.
- **W6 fails, as §9.5 predicted:** 2,482 of 2,484 pairs have r_g ≤ 1. The two that do not are `3-14501` (1.39) and
  `3-35006` (1.08), as at 07.
- **W7 fails:** at 1e-7, 2,407 covered calls move by at most 1.80e-7 of abserr_07. At 1e-8, 25 of 2,367 move by
  0.0066–0.145 of abserr_07, beyond the 1e-5 threshold.
  - All 25 are lone-`G` group calls with minima windows (6 or 19 minima).
  - At 07 one window in each stopped (iroff1 6). At h none did (iroff1 4–5), because the value test is now at rtol 1e-8.
  - Their abserr and `converged` are unchanged.
  - Part A's basis (430 calls, all `G+q+r`) held no such call.
- **The store and 07's census are unchanged**, before, after the census and at the end.
- **The user's references** (findings, not verdicts):

  | row | class | h | 07 |
  |---|---|---|---|
  | `1-34599` (`G±q∓r`) | own | **honest**: 0.020 (1e-7) and 0.015 (1e-8) of abserr_h | **not honest**: 4.03× abserr_07 |
  | `3-678` (`G±q∓r`) | aux_only | **honest**: 0.491 of its abserr, 11× smaller than 07's and from 1 region | honest: 0.135 |

  - At `1-34599` the own C2 route change corrected a dishonest call of 07.
  - At `3-678` the shrunken declaration, the product of an auxiliary trust-check route change, is honest.
  - In every case the readings' spread is at least 7.8e5 below both abserrs.

### What shipped

No code. The records are in `var/levin-phase-floor-07a/census/`:
- **The records:**
  - `log.md` (this file), `verification.md`, `issues.md`, `KEPT_ON_MAC_STUDIO.txt`;
  - `state/predictions.md`: SHA-256 `3ec7b53d…`, mtime 2026-11-12T13:16:44.807Z, read-only, written after W0 and before the
    launch;
  - `state/ruling_stop2.md`.
- **`scripts/`:** 07's census harness, copied from h's `docs/levin-phase-floor/07/census/scripts/`. It is byte-identical to
  07's Mac Studio copy (`out/cmd/00`). Changes are marked "prompt 07a". The diffs are `out/scripts_diff_before_k0.txt`,
  `out/scripts_diff_at_stop2.txt` and `out/scripts_diff_at_end.txt`.
  - **Changed:**
    - `prov.py`: the tree h, the parent's module and 07's census;
    - `worker.py`: tree h; section 6's diagnostics `route_07a` with its on/off switch; the post-call tallies;
    - `run_k0.py`: variant `parent`, `--no-instrument`;
    - `run_census.py`: tree h;
    - `refs_common.py`: the ROWS path.
  - **New:**
    - `hash_runs_07a.py`, `ast_lg_check_07a.py`, `index_07_07a.py`, `compare_07a.py`;
    - `select_w0_units_07a.py`, `w0a_imports_07a.py`, `launch_07a.py`, `launch_census_07a.py`, `census_projection_07a.py`;
    - `write_predictions_07a.py`, `stop3_classes_07a.py`, `score_v4_07a.py`, `xcheck_07a.py`, `xcheck_06m_from07_07a.py`;
    - the references: `refs_select_07a.py`, `x1_capture_07a.py`, `check_captures_07a.py`, `refs_integrand_07a.py`,
      `refs_slice_07a.py`, `refs_projection_07a.py`, `score_refs_07a.py`;
    - `score_07a.py`, `summarise_07a.py`, `w7_detail_07a.py`, `moves_beyond_07a.py`, `w5_forced_detail_07a.py`,
      `kept_list_07a.py`.
- **`out/`:** every command and its output (`out/cmd/`, `out/commands.log`), with these results:
  - W0: `out/w0/`, `out/k0/compare_w0b.json`, `out/w0c3/compare_instr.json`, `out/w0c4/compare_det.json`,
    `out/k0/compare_k0_h_calls.json`;
  - stop 3: `out/stop3/compare_census_calls.json`, `.calls.jsonl` and `classes.json`;
  - the scores: `out/scores/summary.json`, `w4_scores.json`, `w4_report.md`, `w7_detail.json`, `moves_beyond.json`,
    `w5_forced.json`;
  - the cross-checks: `out/xcheck/`;
  - the user's references: `out/refs/` (captures, checks, integrands, projection, readings, `scores.json`, `report.md`);
  - before and after: `out/runs_hashed_{before,after,end}.txt`, and `out/census_projection.json`.
- **Three registered runs, all `done`:**
  - `levin-phase-floor-07a-k0-20261112T130135`: 11 of 11 commands, checkpoint `97731e2b…`;
  - `levin-phase-floor-07a-census-20261112T132744`: 18,560 of 18,560, exit 0, checkpoint `461df2ec…`;
  - `levin-phase-floor-07a-references-20261112T203555`: 16 of 16 readings, exit 0.
- **Kept on the Mac Studio** (`KEPT_ON_MAC_STUDIO.txt`):
  - the census's checkpoint and region records in full (6.7 GB, 08's "before");
  - the k0 and references run directories;
  - every pickle and every file over 2 MB under `census/`.

### Predictions and verdicts

`state/predictions.md` holds the prompt's §6 W1–W6 and descriptive items (lines 525–556), §9.5 (716–731) and §10.4's item
(823–825), each verbatim by line range. Then come this agent's descriptive additions (part 4), how each rule is read
(part 5, with Q1 and Q2 left to the user), and V4's 33 calls (part 6). Verdicts are as ruled at STOP 2.

| Prediction | Verdict | The numbers it was scored on |
|---|---|---|
| **W0 (1)** the parent's code is 07's (S0′) | **holds** | `git diff --stat e182bc7a dadd63a0` (outside docs and prompts) lists `CLAUDE.md` only. The K0 workers loaded 101 files of h. None is in S0′'s diff. Of the parent→h diff only the two LiouvilleGreen modules were loaded, both AST-equal without docstrings. h's own `levin_quadrature.py` was never loaded (`out/w0/w0a_imports.txt`, `out/w0/w0a_ast_lg.txt`) |
| **W0 (2)** the harness reproduces 07's census at the parent | **holds** | 100 of 100 K0 units equal 07's bit for bit: 2,110 calls, 2,110 region records, 32,956 leaves. Only 07a's four keys are set aside. 0 of 07a's keys in 07's records; `phase_span` 0 times as a key in either (`out/k0/compare_w0b.json`) |
| **W0 (3)** the instrument changes nothing | **holds** | `h-1e-8-1-2174` (a root window) and `h-1e-7-1-11214` (an own solve's C2 route changed; chosen by `select_w0_units_07a.py`'s fixed rule), diagnostics on and off: IDENTICAL (`out/w0c3/compare_instr.json`) |
| **W0 (4)** the new code is deterministic | **holds** | `h-1e-8-1-2174` run twice: identical on every key, 07a's included (`out/w0c4/compare_det.json`) |
| **W1** nothing else moves | **holds** (reading A; the call's outputs) | 1e-7: 129,774 untouched calls; 123,083 strictly equal; 6,691 differ only in `location`; 0 in `call` or `harness`. 1e-8: 129,050; 122,359; 6,691; 0. Rows made only of untouched calls: 4,891 and 4,835, totals equal in all. Strictly: fails (13,382 `location`-only calls). Under reading B: fails (15,847 aux_only calls differ in `call`; totals differ in 217 + 207 rows) |
| **W2** no Levin leaf across a stationary point | **holds** | 182,066 calls scanned (37,325 group at each rtol, 107,416 analytic at 1e-7), 0 scan errors. Listed leaves: 0 Levin; window leaves 1,682 (analytic, 1e-7), 526 (group, each rtol) |
| **W3** every phase with something to locate is located | **holds** | 181,282 scanned calls scored, 0 violations. 784 set aside (392 group calls per rtol, the q = r rate): every one `reason: zero_rate`, no scan sign change, min/max \|θ′\| undefined (rate zero) |
| **W4** honest where referenced | **fails** (as §9.5 predicted) | 28 of 33 honest. v08 3.16, v09 2.21, v12 3.08, v17 2.38, v22 2.53, to three figures as predicted. All 33 calls' value and abserr equal 07's in hex (26 window_both, 6 untouched, 1 own: z24). Tolerance min(abserr_07, abserr_h)/100. Every call's readings agree |
| **W5** the gate is consistent; the guard a backstop | **holds** (gated solves, as ruled at the W5 stop) | Gated solves: 0 Clenshaw–Curtis regions with a half sent to Levin (also 0 on 07's estimate); 0 node-partition sums; 0 guarded regions. Descriptive: 18,236 forced solves took the node-partition sum (`out/scores/w5_forced.json`), 3–10 per call, all within the root-located calls (4,416 of the 4,804; 388 root-located group calls, 194 per rtol, have none). In each such call this is the window piece whose end node is the root, and its forced halves and quarters; their V routes nothing. Before the ruling it was undecided, because it fails if forced solves count (part 5's superseded text) |
| **W6** the pairs, on r_g | **fails** (as §9.5 predicted) | 2,482 of 2,484 pairs r_g ≤ 1, 0 unpaired. `3-14501` 1.385 and `3-35006` 1.084, as at 07, all members converged. h's own population: 3,650 pairs, 3,648 ≤ 1, the same two |
| **W7** the stop's change is small | **fails** | Reading A: at 1e-7, 2,407 covered calls, 0 violating, max move 1.80e-7 of abserr_07. At 1e-8, 2,367 covered, 25 violating (0.0066–0.145 of abserr_07), no `converged` change. Reading B: also fails (38 and 97) |

### The user's references (findings, not W verdicts; `out/refs/`)

Method: 07's V4 reference method, ruled at the W5 stop.
- **The tool and readings:** `tools/filon_reference.py` (`d37e633b`, as 07's). QAWO r1 and r2 and FCC r1 and r2 on each call;
  each QAWO reading projected at 30 minutes or less. The reference is FCC r2. 07's reading-usability check passed on all 16.
- **The calls:** the eight flagged calls (row × group calls 2 and 3 × 1e-7 and 1e-8).
- **The captures:** each captured at h, and checked bit for bit against 07a's census (`out/refs/check_captures_h.json`,
  8 of 8). Each captured again in h with the parent's module, and checked against 07's census
  (`out/refs/check_captures_07.json`, 8 of 8).
- **The integrands:** equal by bit pattern at 1001 points between the two captures and across the rtols
  (`out/refs/integrands.json`). So one reference per row and index serves both rtols.
- **The run:** `levin-phase-floor-07a-references-20261112T203555`, `done`, 16 of 16 readings, wall about 13 minutes
  (projected 737 s).
- **The stated error:** the readings' spread.

| call | ref (FCC r2) | stated error | ÷ min abserr | abserr_h | abserr_07 | \|v_h − ref\|/abserr_h | \|v_07 − ref\|/abserr_07 | regions h / 07 |
|---|---|---|---|---|---|---|---|---|
| `1-34599` #2, #3, 1e-7 (own) | −3.162788408760542e-26 | 5.7e-40 | 1.3e-6 | 1.524e-33 | 4.462e-34 | **0.0201** | **4.03** | 16 / 6 |
| `1-34599` #2, #3, 1e-8 (own) | (the same) | 5.7e-40 | 1.3e-6 | 1.118e-33 | 4.462e-34 | **0.0150** | **4.03** | 17 / 6 |
| `3-678` #2, #3, 1e-7 and 1e-8 (aux_only) | 1.431843199210653e-30 | 1.0e-45 | 9.2e-8 | 1.088e-38 | 1.214e-37 | **0.491** | 0.135 | 1 / 12 |

- Every call's stated error is at least 10× below both abserrs, so every call is scored. The readings also agree within
  07's own tolerance (min/100).
- Calls #2 (G+q−r) and #3 (G−q+r) give the same values, since q = r.
- `1-34599`: h is honest, and 07 was 4.03× its abserr from the reference.
- `3-678`: h's eleven-times-smaller declaration is honest, at 0.491 of it.

### Descriptive (not scored)

- **Route changes:**
  - 24,559 solves changed decision, of 22,945,838 recorded:
    - 10,014 C2 route changes, every one towards Clenshaw–Curtis (V < 6π ≤ estimate);
    - 14,545 LU changes, every one towards least squares (V ≤ 20π < estimate).
  - By class:

    | class | to CC | to least squares |
    |---|---|---|
    | own | 3,007 | 4,689 |
    | comparison half | 3,864 | 5,251 |
    | trust-check quarter | 3,143 | 4,605 |

  - Estimate/V on the changed solves: 1.00001–2.29 (median 1.064) for C2 changes, 1.00001–2.39 (median 1.150) for LU
    changes. 07's estimate was always above V on a changed solve.
  - By rtol and kind: analytic 3,402 + 5,776 (1e-7) and 3,695 + 6,224 (1e-8); group 1,449 + 1,260 and 1,468 + 1,285.
  - Every band and k is touched (`out/scores/summary.json`).
  - Calls: own 1,380 analytic + 1,906 group (1e-7) and 2,186 + 2,039 (1e-8); aux_only 7,371 + 660 and 7,251 + 565.
- **Moves:** 3,091 units' `total` or `total_abserr` differ from 07's.
  - Relative: at most 3.3e-4 (90th percentile 2.3e-7).
  - Against total_abserr_07: at most 0.749, 0 above 1.
  - No `total_converged` and no call's `converged` changed.
  - **Calls moved beyond an abserr** (`out/scores/moves_beyond.json`):
    - **beyond both codes' abserrs:** `1-34599` `G+q-r`/`G-q+r` at 1e-7 (3.96× abserr_07, 1.16× abserr_h) and 1e-8 (3.99×,
      1.59×), class own (one own C2 route change; regions 16/17 against 6);
    - **beyond abserr_07 only:** `1-2179` `G±q∓r` (4.66× abserr_07, 0.34× abserr_h; abserr_h 13.6× abserr_07), class own;
      `3-843` analytic J2 (1.09×, 0.58×) and `3-13508` `G±q∓r` (1.08×, 0.60×), class aux_only.
- **The aux_only calls** (reading A sets them outside W1): 15,847 calls (`out/scores/summary.json`, `aux_only`).
  - The changed solves are comparison halves in 8,878 calls and quarters in 7,500.
  - abserr equal in 1,609, smaller at h in 6,697, larger in 7,541.
    - The ratio h/07 runs 0.090–5.10, median 1.0, 99th percentile 1.28.
    - It shrank by more than 2× in **47** calls and grew by more than 2× in 70.
  - Value: unchanged in 8,792. The move over abserr_07 is at most 1.09, 90th percentile 3.3e-5.
  - 4 calls moved by more than their own abserr at h: `3-678` `G+q-r` and `G-q+r`, at 1e-7 and at 1e-8.
    - Their abserr shrank to 0.0897 of 07's and their regions went 12 → 1.
    - Their value moved by 2.00× abserr_h (0.179× abserr_07).
    - These are among the 47.
  - No `converged` change; regions changed in 432 calls.
  - **Referenced aux_only calls: none.** V4's 33 calls are window_both (26), untouched (6) and own (1). The cross-check's
    2,484 pairs are all calls with a window. So no shrinking declaration among them is tested against a reference.
- **The newly located:** 2,360 calls (window_h_only; at each rtol 1,012 group plus 84 J4 and 84 Y4).
  - Points: 2,024 minima and 336 roots. Pieces 2–59, median 14.
  - All converged at both codes.
  - 07's Rule A variation on them was 0.009–18.58.
  - No call locates more than 64 points (the most is 64 pieces, in a call of the old population).
- **Windows:**
  - 3,650 windows per rtol at h (07: 2,470).
  - Stopped: 1,247 at 1e-7, 1,243 with a leaf still falling (07: 1,157/1,153); 1,236 at 1e-8, 1,233 falling (07:
    1,195/1,192).
  - Calls with a stopped window left unconverged: 7 (1e-7) and 9 (1e-8).
- **Convergence and the flag:** unconverged group calls at h equal 07's in every class.
  - 1e-7: 30 in the 720, 52 outside. 1e-8: 34 and 62.
  - No analytic call is unconverged.
  - Row depth 20: 77 (1e-7) and 83 (1e-8) at both codes.
- **The disagreement list:**
  - h against 07: 0 listed calls, on 07's population and on h's.
  - h against 06m (06m's values from 07's cross-check output; 06m's checkpoint was trimmed on 2026-11-10): 843 listed calls,
    class (c) 786, (a) 31, (e) 26, all `G+q+r`, each in the same class as at 07.
- **§10.4's item:** **0** calls go from converged to unconverged with a solve whose route changed and no window, under
  either reading.
- **The largest |θ| at a gated region's end:**
  - analytic: 8.874e12 (ulp 1.95e-3 rad), `h-*-2-35099` J1, both rtols;
  - group: 4.119e12 (ulp 4.9e-4 rad), the same row's `G+q+r`.
- **The smallest margins among unchanged solves:**
  - |V − 6π|: 4.77e-8 (group, a trust-check quarter, `*-1-14543`, V 18.8495560, estimate 18.8496054); analytic 1.38e-3
    (1e-7) and 3.10e-4 (1e-8).
  - |V − 20π| on the Levin route: 3.02e-4 (group, `*-3-34129`, an own solve); analytic 4.53e-3 and 2.94e-3.
- **Cost:**
  - Census wall 23,284 s against 22,708 s.
  - Unit walls 24,952 / 28,513 s against 24,203 / 27,479 s.
  - Group-call regions 952,395 / 1,065,828 against 952,275 / 1,065,288 (1e-7 / 1e-8).
  - The harness's own re-solves after the timed section: 06h's counterfactual made 6,906,084 solves, 7,097 with a C2 change
    and 11,999 with an LU change (`route_07a_post_unit`). They are not the call's.

### Deviations from the prompt

1. **The harness changes,** each within §6's list (`out/scripts_diff_at_end.txt`). STRUCTURALLY REQUIRED.
   - The tree and its paths (`prov.py`, `refs_common.py`, `run_census.py`, `run_k0.py`).
   - The diagnostics (`worker.py`'s `route_07a` and its on/off switch).
   - The comparison with 07's census (`compare_07a.py`).
   - V4's re-scoring (`score_v4_07a.py`).
2. **The parent ran as a variant from `state/variants/` in h**, not as a parent worktree, as the dispatch directs.
   STRUCTURALLY REQUIRED.
3. **06g's `rows.json` is no longer on this host.** 07's copy (same SHA-256) is read. The first copy command failed
   (`out/cmd/07`). STRUCTURALLY REQUIRED; the failed attempt is UNINTENDED DRIFT.
4. **New scripts** (listed under What shipped). IMPLEMENTATION CHOICE.
5. **Development smoke tests** (`state/smoke/`, `out/cmd/11`–`17`), not scored. IMPLEMENTATION CHOICE.
6. **K0 at h, its call comparison, the unit selection and W0's clauses 3 and 4 ran inside the `k0` run**, as 07 did.
   IMPLEMENTATION CHOICE.
7. **W0 (3)'s units:** z12's row fixed by name; the route unit chosen by a rule fixed before the run. IMPLEMENTATION CHOICE.
8. **The diagnostics also tally the harness's own re-solves after the timed section** (`route_07a_post`). IMPLEMENTATION
   CHOICE.
9. **The comparison puts each differing path in one category** (`call`, `location`, `harness`). It was put to the user as
   Q2 and ruled. IMPLEMENTATION CHOICE.
10. **`predictions.md` gained parts 4 and 5.** Part 6's column labelled "30 digits" holds 07's stored reading strings (17
    digits). The file is fixed and was not rewritten. The label is UNINTENDED DRIFT (cosmetic).
11. **The first edit of `prov.py` ran through the system `python3`** (an editing script; no repository code). UNINTENDED
    DRIFT (cosmetic).
12. **Redundant background waits on the k0 driver** were started and stopped. UNINTENDED DRIFT (cosmetic).
13. **No lock taken, and no lock `status` or `ray-processes` run**, as the dispatch says. IMPLEMENTATION CHOICE.
14. **The A3 v2 fingerprint was not recomputed;** the copy is checked by SHA-256 against its certification, as 07 did.
    IMPLEMENTATION CHOICE.
15. **The stop-3 comparison and every scoring script ran in the session through `rec.sh`**, not as registered runs. The
    dispatch names only the `k0` and `census` slugs. The longest took 103 s. IMPLEMENTATION CHOICE.
16. **`score_v4_07a.py`'s first run failed on a broken docstring** (`out/cmd/50`); it was fixed and re-run (`50b`).
    UNINTENDED DRIFT (a failed attempt).
17. **A first loop of the three cross-checks did not run** (zsh does not word-split `set -- $args`; `rec.sh` could not open
    its output). It was re-run under `bash`. Nothing was written but the shell's errors. UNINTENDED DRIFT (a failed attempt).
18. **The disagreement list against 06m used 07's cross-check output for 06m's values,** because 06m's census checkpoint was
    trimmed on 2026-11-10. The first attempt, against 06m's checkpoint, failed (`out/cmd/51_xcheck_h_vs_06m_pop07`).
    STRUCTURALLY REQUIRED.
19. **`xcheck_06m_from07_07a.py`'s output carries a vestigial empty key** `class_h_vs_07`. The real table is
    `class_07_to_h`. UNINTENDED DRIFT (cosmetic).
23. **References were taken for rows `3-678` and `1-34599`,** on the user's word at the W5 stop. This overrides §C.1's "no new
    references are taken" for these two calls only. STRUCTURALLY REQUIRED (by the ruling).
24. **The reference harness is 07's V4 method, copied with changes marked "prompt 07a"** (`out/scripts_diff_at_end.txt`).
    IMPLEMENTATION CHOICE.
    - `x1_capture_07a.py`: tree h, the parent variant, the match to 07's record.
    - `check_captures_07a.py`: the census named on the command line; `route_07a*` set aside.
    - `refs_slice_07a.py`: the 1e-7 captures only.
    - `launch_07a.py`'s `references` plan.
    - `one_reading.py` and `run_06j.py` are unchanged.
    - New: `refs_select_07a.py`, `refs_integrand_07a.py`, `score_refs_07a.py`.
25. **The projection was refitted on 07's own 74 readings,** because 07's calibration (06g's readings) is gone from this host.
    07's feature function and model form are unchanged (`refs_projection_07a.py`). STRUCTURALLY REQUIRED.
26. **One reference per row and index serves both rtols,** on the integrands' shown identity, as 07's cross-check did.
    IMPLEMENTATION CHOICE.
27. **The parent captures (`out/refs/captures_parent.json`) were made so that each call could be checked against 07's census.**
    IMPLEMENTATION CHOICE.
28. **The stated error of a reference is the readings' spread** (07's agreement measure). IMPLEMENTATION CHOICE.
29. **Two failed attempts, each re-run.** UNINTENDED DRIFT (failed attempts).
    - The first captures (`out/cmd/71`, `72`) could not write their JSON because `out/refs/` did not exist; re-run as `71b` and
      `72b` into new directories.
    - A zsh `$H:s…` substitution error stopped one command; re-run with the literal path.
20. **W4's tolerance** is min(abserr_07, abserr_h)/100: 07's rule with "the before" now 07 (part 5). 07's own scores are
    shown beside it. IMPLEMENTATION CHOICE.
21. **W3's "scored" count** is every scanned call less the 784 set aside. `score_07a.py` keeps in detail only calls with a
    sign change, a point or a listed leaf; the others satisfy V3 vacuously. IMPLEMENTATION CHOICE.
22. **`kept_list_07a.py` was made from 07's by `sed`.** A substitution miswrote its own source's name in its docstring; this
    was corrected before it ran. UNINTENDED DRIFT (cosmetic).

### Verification performed

`verification.md` has the account. In brief:
- **The trees:** h at `4bf50d34`, clean with `--ignored` at every check and at the end, with no `__pycache__`. The main
  checkout is at `28ebaf07`, clean.
- **Every census worker:** module `4ea1948b`, no variant, h clean at its SHA, no Ray, no foreign module.
- **The scratch copy:** EQUAL to its certification before, after the census and at the end. Before equals end.
- **07's census:** 150 of 150 files equal its kept list. Its 850 working-record files are byte-identical before, after and at
  the end.
- **The registry:** 76 runs, all finished. No lock was taken.

### Observations not acted on

1. **W7's 25 calls:** at 1e-8, minima windows of lone-`G` calls that stopped at 07 no longer stop under 07a's value test
   (min(1e-5, rtol)). Values move by up to 0.145 of their abserr, abserr and `converged` unchanged. §9.5's basis (Part A's
   430 `G+q+r` calls) did not sample them.
2. **W5's forced partition sums** (descriptive under the ruling): in a root-located call, the window piece whose end node is
   the root takes the node-partition sum.
   - θ′ changes sign at that node.
   - So do its forced halves and quarters there.
   - V routes nothing on a forced solve, but it is the `phase_span` reported for that region.
3. **Shrinking declarations on auxiliary-only calls:** 47 aux_only calls' abserr shrank by more than 2×.
   - The one the user referenced, `3-678`, is honest at 0.491 of its eleven-times-smaller abserr.
   - The other 46 are unreferenced.
4. **`1-34599`, where 07 and h disagreed beyond both errors:** the reference shows h honest (0.020 and 0.015) and 07 not
   (4.03). An own C2 route change corrected it.
5. **06m's census checkpoint was trimmed on 2026-11-10,** and 06g's census records are gone from this host (`issues.md`).
6. **§10.4's set is empty** in the census: no call loses convergence with a route change and no window.

### State handed on

- **Nothing is running.** All three registered runs are `done`: k0, census and references. No lock was taken.
- **Kept on the Mac Studio** (`KEPT_ON_MAC_STUDIO.txt`):
  - the census run directory in full (149 files, 6.7 GB; 08's "before");
  - the k0 and references run directories;
  - every pickle and every file over 2 MB under `census/` (the K0, W0 and smoke runs' region pickles; the stop-3 and
    scoring outputs).
- **Read only and unchanged:** the scratch copy and 07's census, by hash.
- **Issues:** proposed in `issues.md`, for the laptop and the user.

## levin-phase-floor 07a, Part A (section 2): the measurement — log of the measuring agent

**Subject:** prompt 07a Part A, the replays of section 2.1's calls at 07's code and at variants of the window's stop,
on the Mac Studio (mps026298), worktree `/Users/ds283/Documents/Code/SecondaryGWKit-lpf07a-f93aafd` at the dispatch
SHA `f93aafd0173bd5de1cdc4107656dbe07c4c1c2a3` (code 07's), records in
`/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07a/measurement/`.
**Commit:** none (records committed by the orchestrator, on the laptop).
**Result (2026-11-11): the planned stop of section 2.5.** Stop 1 (2026-11-10) asked two cost questions. The user's
rulings (deviations 16, 17, 20, 21) were applied, and every part of section 2 they left in scope is done:
- the Gate (Mac Studio clauses) holds before and after;
- section 2.1 item 2's captures check, 9 of 9;
- replays at 07's code reproduce 07's census, 430 of 430 bit for bit;
- variants (a), (c)-(f) ran on all 430 calls, and (b) and (g) (as b2 and g2) on items 1-2 and the no-window calls;
- the no-window check passes, 5 of 5 for every variant;
- all 24 attributions are done, and every one closes;
- the O9 check covers the five failures;
- the final tables are written.

**No variant changes any of the five failures.** In every failure the whole excess error is in H3, the Levin leaves
outside the windows. In three of them it sits in a single seed region beside the window that was never bisected.
(b2) and (g2) qualify only by leaving all 29 window calls unconverged. Item 3 is undecided by construction.

*Stop 1's result line, kept:* stopped before section 2.5 with a question on section 2.2's cost rule (the attribution
projects over 12 h) and on the cost of variants (b) and (g), which the prompt does not bound.

### What shipped

Nothing in the tree (its `git status --short --ignored` empty throughout; the main checkout at `28ebaf07`, clean).
Records only, under `measurement/`:
- `scripts/`: 07's census harness copied from `docs/levin-phase-floor/07/census/scripts/` (diff in
  `out/scripts_diff_against_07.txt`: `prov.py` and `refs_common.py` only, each change marked "prompt 07a"), and 07a's
  own scripts (`*_07a.py`, `capture_pairs_07a.py`, `gate_hashes.py`, `launch_07a.py`, `launch_attr_07a.py`);
- `scripts/variants/levin_quadrature_07a_{a,b,c,d,e,f,g,b2,g2}.py`, each with its diff against the tree in
  `out/variants/<v>.diff` and its SHA-256 in `out/variants/variants.json` (the first index kept as
  `variants_v1.json`);
- `state/`: `census_refs.pkl` (07's census records of every call), `calls_07a.json` (the manifest),
  `pairs_capture/` (section 2.1 item 2's captures), the plans;
- `out/`: the replays (`replays/<variant>/<sid>.json`, instrument records `*.instrument.json` for items 1-2), the
  interim tables (`tables_interim_1.{json,md}`), the attribution (`attr/`), the (b) cost probe (`b_scaling/`), the
  captures' check (`pairs_capture/`), the Gate's hashes (`gate_hashes_*.json`);
- registered runs `levin-phase-floor-07a-x396-fast-20261110T215229` (done, 2,376 of 2,376),
  `levin-phase-floor-07a-attr-1-20261110T215423` (done, 8 of 8), `levin-phase-floor-07a-bg2-items12-20261110T223735`
  (done, 68 of 68) and `levin-phase-floor-07a-attr-2-20261111T102500` (done, 15 of 15, 10 h 55 min);
- after stop 1: `out/tables_final.{json,md}` (`tables_07a_final.py`), `out/tables_final_rg.{json,md}`
  (`tables_07a_final_rg.py`, items 2 and 4 on r_g: **the tables of record**), `out/attr/attr_tables_final.{json,md}`
  (all 24 calls), `out/o9/` (`o9_check_07a.py`, `o9_table_07a.py`: the O9 check), `out/gate_hashes_after_final.json`,
  and `out/stop_2_report.md` (the section 2.5 report).

### Predictions and verdicts

Part A has no predictions (section 2.4's tables are descriptive; section 2.5 recommends nothing). The checks it
carries:

| check (prompt) | scored on | verdict |
|---|---|---|
| Gate: code is 07's | `git diff --stat adffa21b f93aafd0 -- AdaptiveLevin ComputeTargets LiouvilleGreen tools` empty | holds |
| Gate: 07's captures match `KEPT_ON_MAC_STUDIO.txt` | 78 of 78 lines (V4 `C00`-`C24`, cross-check `X00`-`X09`), before and after | holds |
| Gate: the scratch copy matches its certification | 5 of 5 files, SHA-256, size and mtime, no extra file, before and after | holds |
| Gate: nothing `running` or stale | Mac Studio registry, 69 runs finished at start (the laptop's: the orchestrator's check) | holds |
| 2.1 item 2: each capture checks against 07's census | 9 of 9 (four pair calls, five no-window calls) by 07's `check_captures.py`: value, abserr, label, converged, regions bit for bit | holds |
| 2.1: a replay at 07's code reproduces its census record | 430 of 430 at the tree and 430 of 430 at (a): value, abserr, converged, num_regions, max_depth, every region array, section 1.5's entries | holds |
| 2.3: a variant does not change a call with no window | five calls, (a), (b2), (c), (d), (e), (f), (g2): 5 of 5 each, bit for bit | holds |
| 2.2: the attribution's closure check, |sum ref_i - ref| <= tol | all 24 calls with a stopped window: 6.2e-6 to 3.3e-3 of tol | holds (no call reported unattributed) |

### Deviations from the prompt

1. **`STRUCTURALLY REQUIRED`** — "07's `make_capture.py`" is 06b's writer of a capture list from 06a's checkpoint; 07's
   capture runner is `xcheck_capture_07.py` (after 06c's `x1_capture.py`: the census `worker.py` re-runs the unit with
   `capture` in its spec). Section 2.1 item 2's captures were made by `capture_pairs_07a.py`, a copy of
   `xcheck_capture_07.py` changed only to merge several calls of one unit into one run of it (07's made one unit entry
   per call), and checked by 07's `check_captures.py` unchanged.
2. **`STRUCTURALLY REQUIRED`** — the harness's trees and paths: `prov.py` and `refs_common.py` name this worktree as
   `g`; `refs_common.ROWS` is 07's census copy of `rows.json` (`var/levin-phase-floor-07/census/state/rows.json`, the
   same SHA-256 `aa523d28...`), since 06g's was removed in the housekeeping of 2026-11-10.
3. **`STRUCTURALLY REQUIRED`** — section 2.3's five calls with no window: every call of section 2.1 has one, so five
   were captured from the pair units in the same two worker runs as item 2 (`g-1e-8-3-14501` calls 0, 2, 4;
   `g-1e-7-3-35006` calls 3 and 7, the last an analytic J2 block), each checked bit for bit like the pairs. Which five:
   **`IMPLEMENTATION CHOICE`**.
4. **`STRUCTURALLY REQUIRED`** — the 396: 07 captured 390 of them at 1e-7 and 6 at 1e-8 (one call per row; the integrand
   is the same at both rtols). Section 2.1 item 3 names the calls at 1e-8, so a 1e-7 capture is replayed with the 1e-8
   census call's `rtol` and `atol` and compared with that call's record; all 396 reproduce it bit for bit.
5. **`IMPLEMENTATION CHOICE`** — the variants are one instrumented copy of the module, differing only in the constant
   `_VARIANT_07A`; (a) is that copy, and 07's code itself is replayed too (`tree`), so the instrument is shown to change
   nothing (430 of 430 at both). The instrument records section 2.2's quantities at each window bisection and stop, and
   every returned leaf at the end; it is on for items 1-2 only (at (a), (c)-(f)).
6. **`IMPLEMENTATION CHOICE`** — the raw estimate e of a leaf is its own solve's error, `used_interval`'s budget error
   with no trust-check declaration; **the analogue of `defab.eq.error`** is `200 |CC_{2N-1} - CC_N| >= resasc`, with
   dqk21's `resasc` formed on the fine Clenshaw-Curtis rule (`sum_j |w_j| |g_j - gbar|`), both positive. The
   Clenshaw-Curtis solve returns `resasc_07a` in its dict and the region keeps it; nothing reads them but the window code
   and the instrument (the no-window check).
7. **`IMPLEMENTATION CHOICE`** — (d)'s value test: `|ΔV| <= min(1e-5, rtol) |V_L + V_R|`.
8. **`IMPLEMENTATION CHOICE`** — (e)'s "request unmet": the call's running error above `max(atol, rtol |V|)` once the
   halves of the bisection that reached the count are in, before the window's leaves are floor-limited (dqage's
   `errsum > errbnd`, tested before it sets `ier = 2`).
9. **`IMPLEMENTATION CHOICE`** — the two extra variants: (f), dqage in full, (c)'s count with (e)'s outcome (section
   0.1's default); (g), 07's count with no count while another leaf of the window fell at its last bisection.
10. **`UNINTENDED DRIFT`**, corrected — in (b) and (g) as first written, the list of the window's other falling leaves
    was formed at every window bisection, a pass over every leaf; read only by (g) and the instrument, it changes no
    value, but makes (b) quadratic in its leaves. `b2` and `g2` (`make_variants_07a_bg2.py`) form it only where read;
    `b2` equals `b` bit for bit at depth_max 12 (v08) at the same wall, so the driver's own passes dominate. (b) and (g)
    are to be run as `b2` and `g2`; no result of `b` or `g` is used.
11. **`IMPLEMENTATION CHOICE`** — the attribution's method: `tools.filon_reference` (Route B, 07's tool `d37e633b`),
    one reading per resolution with every leaf end added to the production edges, each leaf's reference the in-order
    sum of its own jobs (`attr_07a.py`'s docstring); window leaves fall to the tool's direct `mpmath` Gauss-Legendre
    path. Reference: resolution 2; stated accuracy `max(|ref_2 - ref_1|, fit-noise 1 sigma)`.
12. **`STRUCTURALLY REQUIRED`** — section 2.2's "stated accuracy at least 100x finer than the leaf's declared error"
    cannot be met on leaves whose e' is below about 100x the integrand's own double-precision noise (the readings agree
    to about 1e-15 of the local magnitude): 80 of v22's 124 leaves, 32 of v21's 62, 83 of v03's 161. Such leaves are
    listed per call; the parts' summed accuracies are given beside each part.
13. **`IMPLEMENTATION CHOICE`** — part H3 ("the Levin leaves outside every window") holds every leaf outside every
    window, Clenshaw-Curtis fallback leaves included (listed by type); each part's declared error is `sum e' + |sum E|`
    over its leaves, as the call's abserr is formed.
14. **`IMPLEMENTATION CHOICE`** — the (b) cost probe (`b_scaling_07a.py`: v08 and v12 at variant (b) with `depth_max`
    6-18, scratch, not a section 2.4 table), run to project (b)'s cost before committing it.
15. **`STRUCTURALLY REQUIRED`** — the projection rule as written passed (the pilot, v22, projects 9.4 h) but the next two
    calls cost 2.6-2.7x what the pilot's ratio predicts (07's own readings' CPU is the better guide: about 25 h). The
    20 remaining attributions were held, and reported with the calls to drop (stop 1).
16. **`STRUCTURALLY REQUIRED`** — the user's ruling on stop 1's first question, relayed by the orchestrator on
    2026-11-11 (the user's words: "run all the calls needed for the H3 attribution in full"): every remaining
    attribution of section 2.2 is run, v13, v18 and v23 included, past section 2.2's twelve-hour rule (projection
    about 25 h on six workers). Run as `levin-phase-floor-07a-attr-2-20261111T102500` (15 calls, serial, six spawn
    workers each); the earlier `attr-1` ran eight calls with one worker each beside the (b2)/(g2) run.
17. **`STRUCTURALLY REQUIRED`** — the user's ruling on stop 1's second question ("no (b) or (g) on the 396 class-c
    captures"): (b2) and (g2) are run on items 1-2 and the no-window calls only (68 of 68 ok), not on the 396.
    Section 2.4 item 3 is therefore **undecided by construction** for every variant; (b2) converged on none of the 29
    window calls of items 1-2 (only on the five no-window calls, where it equals the tree). Qualification (item 4) is
    judged on items 1-2 with item 3 undecided.
18. **`IMPLEMENTATION CHOICE`** — the final tables are written by `tables_07a_final.py`, a copy of `tables_07a.py`
    (left as run for the interim tables) changed only for deviation 17 and for the pairs' r_g (deviation 19), each
    change marked "07a final"; it reads b2 and g2 as (b) and (g) (deviation 10).
19. **`IMPLEMENTATION CHOICE`** — at the orchestrator's instruction (2026-11-11; the pairs' formula is a question still
    with the user), item 2 tabulates the prompt's r = |v_7 - v_8| / (e_7 + e_8) and, labelled beside it, the
    cross-check's r_g = |v_7 - v_8| / e_7.
20. **`STRUCTURALLY REQUIRED`** — the user's ruling of 2026-11-11 on the pairs' formula, relayed by the orchestrator
    ("happy to follow your recommendation of using `r_g`"): items 2 and 4 are scored on the cross-check's
    r_g = |v(1e-7) - v(1e-8)| / abserr(1e-7) (07's `xcheck_rule_20261109.md`); the prompt's r is kept beside it,
    labelled. The prompt's formula contradicted its own "as the cross-check scored them" (the orchestrator's error, in
    its words). Written by `tables_07a_final_rg.py`, a copy of `tables_07a_final.py` (left as run) changed only for
    this, each change marked "07a r_g". Deviation 19's tables are superseded for items 2 and 4, and kept.
21. **`STRUCTURALLY REQUIRED`** — an addition at the user's word (2026-11-11, on the orchestrator's proposal): the O9
    check on the H3 leaves of the five failures, a measurement only, at 07's code as it is
    (`scripts/o9_check_07a.py`, `out/o9/<sid>.{json,md}`): for each leaf, its parent and its sibling, 07's gate
    estimate `phase_span` from `_Basis_SinCos.build_Levin_data` on the region's own grid, |theta(b) - theta(a)| from
    the raw phase, whether theta' changes strict sign on the nodes, the node-partition sum, and the route each would
    take against `SIX_PI` and `_LEVIN_DIRECT_SOLVE_PHASE_SPAN`.
22. **`IMPLEMENTATION CHOICE`** — the O9 check's details. (i) The grid is the call's own `chebyshev_order` (24 on
    v22), not 16 as the instruction said: "the leaf's own nodes" are the ones its solve used. (ii) The leaves
    checked: every H3 leaf whose |true error| exceeds its declared e' + |E|, and, labelled, the three H3 leaves of
    largest |true error|. (iii) The node-partition sum is given for every region; it is the true variation used for
    the route only where theta' changes sign, else |theta(b) - theta(a)|. (iv) Parent and sibling are found by
    descending from the leaf's seed region (the range between windows that holds it) by `break_point`, the driver's
    own split, and the leaf must be reached exactly at its depth. (v) It is run for each failure once the
    attribution run has ended, not as each call lands: watching for each would mean polling a long run (CLAUDE.md
    rule 4); its result does not depend on when it runs.

### Verification performed

- The Gate's hash clauses before and after (`out/gate_hashes_before.json`, `out/gate_hashes_after_interim_1.json`):
  78 of 78 captures, 5 of 5 store files; 07's census checkpoint `03dfe786...` before and after; R29's store and 07's
  run directories not opened.
- 9 of 9 captures by `check_captures.py` (`out/pairs_capture/check_captures.json`).
- 430 of 430 replays reproduce 07's census at the tree and at (a) (`out/tables_interim_1.md`).
- The no-window check, (a), (c)-(f): 5 of 5 each.
- `b2` = `b` bit for bit on v08 at depth_max 12.
- The attribution's closure check on all 24 calls: |sum ref_i - ref| / tol between 6.2e-6 (v09) and 3.3e-3 (v08); holds
  on every call (`out/attr/attr_tables_final.md`).
- After the runs (`out/gate_hashes_after_final.json`): 78 of 78 captures and 5 of 5 store files match again. 07's
  census checkpoint is still `03dfe786...` (hashed read-only).
- The O9 check: each region was reached by descent at exactly its own depth; no family error on any of the five
  calls.
- The worktree's `git status --short --ignored` empty; the main checkout at `28ebaf07`, clean; no Ray initialised in
  any replay (`ray_initialized` false in every output).

### Observations not acted on

- The requests of several calls are atol-dominated (v08: `atol` 1.86e-33 against `rtol |V|` 3.4e-38; abserr 1.94e-36):
  converged is far inside the request; the honesty failures are of the declared error, not of the request.
- In every window of items 1-2 the bisections are driven by untrusted leaves (issues.md item 2).
- On v22 the excess error is in one Levin leaf beside the window (issues.md item 1).
- (d) changes 58 of the 396 (value, abserr or regions); (e) and (f) leave 35 more of the 396 unconverged, and v03 and
  the two 1e-8 pair members; no variant run so far changes a V4 failure.
- 16 of the 396 have a window that never stopped.
- (b2) and (g2) are equal bit for bit on all 34 calls: (g)'s rule never stops a window here, because at every
  bisection that reaches the count another leaf of the window also fell. Both leave all 29 window calls unconverged:
  764,283 regions and about 50,000 s, against (a)'s 2,386 regions and 48 s on items 1-2.
- In three failures (v08, v12, v17) the leaf carrying all of the excess is a **depth-0 seed region**, the range from
  the window's upper edge to the end of the span (about [9.23, 27-28]). It was solved as one Levin leaf, with an
  estimated phase span of 1.3e8-4.1e8, and never bisected; its true error is 2.9-7.4x its e'. In v09 the leaf is at
  depth 1 and in v22 at depth 5 (issues.md item 1).
- The O9 check finds no route change and no strict sign change of theta' on any leaf, parent or sibling checked.
  07's estimate of the phase span overstates the true variation on the big seed leaves (true / estimate 0.73-0.75),
  but both sides are far above 20 pi.
- The H2 part's |true| / declared exceeds 1 on two controls, v02 (1.31) and v10 (1.04). Both parts are about 1e-7 of
  their call's abserr.

### State handed on

At the section 2.5 stop: nothing running. All four of 07a's registered runs are `done`, and the lock was not used.
The scratch copy, 07's census and its captures are unchanged. The worktree's `git status --short --ignored` is empty,
and the main checkout is at `28ebaf07`, clean. `KEPT_ON_MAC_STUDIO.txt` lists 61 files: every pickle, and every file
over 2 MB, chiefly the (b2)/(g2) replay records.
*At stop 1 (kept):* `attr-1` was running, and nothing else.
