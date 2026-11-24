# levin-phase-floor 07d — the $6\pi$ gate, the Clenshaw–Curtis estimate and the integrand's noise

> **Part B, §1, §3 and §4** (the code, its tests and these records), written by the executing agent on the laptop, on
> the prompt as amended by its §9.4 (2026-11-24). **Part A, §2** and **Part A′, §9.2** (the measurements, on the Mac
> Studio) are recorded in `docs/levin-phase-floor/07d/measurement/` and `docs/levin-phase-floor/07d/measurement-a2/`;
> prompt §9 holds the rulings. **Part C, §5 and §6** (R33 and the census) runs later on the Mac Studio; the orchestrator
> appends its record.

**Subject:** prompt 07d §1, §3 and §4, as amended by §9.4 (Part B), on the laptop, in the main checkout
`/Users/ds283/Documents/Code/SecondaryGWKit`, branch `handover-remedial`.
- **Dispatch SHA:** `ca1d8cfa` (pushed), the amendment at Part A′'s stop. The parent module is
  `ca1d8cfa:AdaptiveLevin/levin_quadrature.py` (SHA-256 `d141a80b…`, 6,459 lines), 07c's code. The parent was run from
  a temporary detached worktree at `ca1d8cfa` in the session scratchpad, the main checkout's `venv` symlinked into it;
  V93T10 from a second such worktree with the variant copied over its module. Both were removed afterwards.
- The reference implementation is V93T10, `docs/levin-phase-floor/07d/measurement-a2/scripts/variants/
  levin_quadrature_a2_V93T10.py` (SHA-256 `f85976fc…`, checked before each use), with its diff `out/variants/V93T10.diff`
  (`a2cb1c85…`).
- `git log --oneline ca1d8cfa..HEAD` was empty at the start and before staging.
- `runregistrykit list` at the start: 98 runs, 96 finished, 2 `unknown` (pre-registry), nothing `running` or stale. No
  Ray was started, no store was opened, no run was registered, nothing was written under `var/`, and nothing ran on
  the Mac Studio.

**Commit:** the code commit of this log (one commit, not pushed), on `ca1d8cfa`.

**Result:**
- **§1 as amended is in `AdaptiveLevin/levin_quadrature.py`:** the Clenshaw–Curtis rule returns ${\rm CC}_{4N-3}$,
  declares $|{\rm CC}_{8N-7} - {\rm CC}_{4N-3}|$ plus $\kappa s_{\rm own}$ on its estimate and on both floors, with its
  floor, node term, declared term and `resasc` on the $(4N-3)$-point grid; `SIX_PI` is `CC_GATE_PHASE_VARIATION =
  10.0 * np.pi`; the constants `_CC_OWN_NOISE_KAPPA = 4.0`, `_CC_STANDARD_CHOP_TOL = 2.0**-52`,
  `_CC_OWN_NOISE_TAIL_FALLBACK = 8`, `_CC_GATE_FORMER_PHASE_VARIATION = 6.0 * np.pi`; the result dict gains
  `num_cc_solves`, `num_cc_gate_route_changes`, `num_cc_noise_set` and (deviation 2) `num_cc_resampled`. The variant's
  knot machinery and its `adaptive_levin_sincos` wrapper do not ship. **It equals V93T10 bit for bit**, the counts
  equal to the variant's, on every call compared on the laptop: 35 driver calls, 14 single Clenshaw–Curtis solves and
  4,320 LiouvilleGreen draw calls (Verification 3 and 4).
- **§3:** a new module `AdaptiveLevin/tests/test_levin_cc_estimate.py`, 10 tests: (a) one (four integrands at two
  orders, as subtests), (b) three, (c) three, (d), (e), (f). **Every breakage caught**, (i)–(vi), each by the test §3
  names for it.
- **Existing assertions:** 30 tests in 11 modules changed, and comments in a twelfth, listed below with why; the harness of most of them now runs
  07c's Clenshaw–Curtis rule through `cc_rule_07c()` (deviation 4), their assertions unchanged. **One test's call goes
  from converged to unconverged** (`test_levin_trust_yardstick` test 2's noisy(A=512) cell), honest at both codes; none
  the other way.
- **The seven roots:** AdaptiveLevin 141 → 151, the rise equal to the 10 tests added; ComputeTargets 781 (the wall-clock flake did not fire), CosmologyModels 39, LiouvilleGreen 156 (skipped 1), ObjectFactories 280, RunDrivers 127 and tools 64, each equal to its baseline.
- **The LiouvilleGreen draws:** no test newly failing (80 of 80 units pass under the parent, the commit and V93T10); the 42 calls in which the fix does not act are bit-identical to the parent; it acts in 4,278 of 4,320; every call equals V93T10's bit for bit, counts included. **73 calls converged at the parent are unconverged at the commit, and 35 the reverse** (07c's 26 newly unconverged calls all converged again; Observations 4). The three-Bessel results where the fix acts go from 654 to 547 honest, their errors unchanged and their declarations smaller (Observations 2).

## What shipped

### §1, the code (`AdaptiveLevin/levin_quadrature.py`, line numbers at the commit; 6,805 lines)

1. **§1.1, the rule.** `_adaptive_levin_subregion_cc` (line 3555) samples the integrand on the $(2N-1)$-point grid,
   reuses those samples as the $(4N-3)$-point grid's even nodes and evaluates its odd nodes, and the $(4N-3)$-point
   samples as the $(8N-7)$-point grid's even nodes (nesting checked by `np.array_equal(grid[::2], coarse)`, as the
   variant checks it; a grid that does not nest is sampled afresh and counted, which no call did). It returns
   ${\rm CC}_{4N-3}$; `abserr_direct` is $|{\rm CC}_{8N-7} - {\rm CC}_{4N-3}| + \kappa s_{\rm own}$; `round_off_err` and
   `round_off_err_less_node` are `_cc_roundoff_floor` on the $(4N-3)$-point grid, each plus $\kappa s_{\rm own}$; the node
   term ($E$, the node bound), `_declared_endpoint_phase_err`'s term (from the $(4N-3)$-point $f_{\rm scale}$) and
   `resasc` are on that grid. $\theta'$ on the $(2N-1)$-point grid is formed as before (`theta_prime_on_fine_grid` with
   $M = N$) and is the coarse sample of the $(4N-3)$-point grid's (with $M = 2N-1$). ${\rm CC}_N$ and ${\rm CC}_{2N-1}$
   are no longer formed. Every evaluation, and every floating-point operation that reaches an output, is in V93T10's
   order and association (the floor's node term is `abs(E) + node_bound`, as the variant has it, not 07c's
   `abs(E) + R_g + R_2 + R_c`).
2. **§1.1, the noise term:** `_cc_own_noise(g, w)` (line 3535) $= \sigma_{\rm own}\sqrt{(n-1)/2}\sqrt{\sum w_j^2}$, with
   `_cc_chebyshev_coefficients` (line 3470; the variant's `_a2_coeffs`, its cosine rows cached by $n$ in
   `_cc_coefficient_cosines`, line 3458, each row formed as the variant forms it) and `_cc_standard_chop` (line 3492;
   the variant's `_a2_standard_chop`, transcribed statement for statement). Times `_CC_OWN_NOISE_KAPPA`.
3. **§1.2, the constants** (lines 295–370): `CC_GATE_PHASE_VARIATION = 10.0 * np.pi` under a comment giving §9.4.3's
   basis, margin and the four places it does not reach; `_CC_GATE_FORMER_PHASE_VARIATION = 6.0 * np.pi`, commented as
   read only by the route-change count; `_CC_OWN_NOISE_KAPPA = 4.0`, `_CC_STANDARD_CHOP_TOL = 2.0**-52` and
   `_CC_OWN_NOISE_TAIL_FALLBACK = 8` under one comment with $\kappa$'s basis (`kappa93.md`) and where it does not reach.
4. **§1.2, the gate** (line 3151): `if phase_span < CC_GATE_PHASE_VARIATION:`; the solve's dict is marked
   `cc_gate_route_changed = phase_span >= _CC_GATE_FORMER_PHASE_VARIATION`. Its one reader in code; Rule A reads none.
5. **§1.3:** `converged` and every Levin-route declaration unchanged; the round-off exit, the window's count, the chase
   stop's stopped leaves and floor-limited leaves read the new declarations through the dict as before.
6. **§1.4, the counts:** the Clenshaw–Curtis dict carries `cc_noise_set` ($\kappa s_{\rm own} > \max({\rm estimate},
   {\rm floor}_{4N-3})$, the floor before the noise term) and `cc_num_resampled`; `_adaptive_levin`'s `_solve` (line
   4923) counts, for every solve whose dict is `is_direct`, `num_cc_solves`, `num_cc_gate_route_changes`,
   `num_cc_noise_set` and `num_cc_resampled` (initialised at line 4880), returned as ints beside 07c's counts (line
   6154). `adaptive_levin_sincos`'s body is unchanged.
7. **§1.5 and §1.2, the docstrings and comments rewritten** (line numbers at the commit):
   - the module docstring's gate item (lines 106–134) and its Rule A sentence (line 169, "the gate's 6 pi of the
     time");
   - `SIX_PI`'s comment block, as `CC_GATE_PHASE_VARIATION`'s (lines 295–339);
   - the order-sweep comment at 279 ("the Levin/fallback boundary of the time, 6 pi");
   - `_LEVIN_DIRECT_SOLVE_PHASE_SPAN`'s comment (line 616, "see CC_GATE_PHASE_VARIATION");
   - 07c's chase comment (lines 758–760, a sentence on 07d's rule added) and the window-count comment (lines
     774–777, what the count reads);
   - `_DQK21_SATURATION_FACTOR`'s comment (lines 814–823);
   - `_cc_node_term_parts`'s and `_cc_roundoff_floor`'s docstrings ("the returned rule's weights", "n samples");
   - `_cc_leaf_unresolved`'s docstring (lines 1306–1313);
   - `used_interval`'s comments on abserr (around line 1636), the value (line 1682) and `resasc` (line 1712), and the
     `resasc` and `abserr_fallback` accessors;
   - `_levin_leaf`'s docstring ("CC_{4N-3} on the Clenshaw-Curtis fallback");
   - `build_Levin_data`'s docstring (V's rounding "against the gate's 10 pi"; "see CC_GATE_PHASE_VARIATION's comment");
   - `theta_prime_on_fine_grid`'s docstring, made generic in $M$ and naming its two calls;
   - `_adaptive_levin_subregion_impl`'s docstring (the gate paragraph, `force_direct`) and the gate's comment
     (formerly line 3033, "SIX_PI is unchanged, and unmeasured");
   - `_adaptive_levin_subregion_cc`'s docstring and its comments (node reuse, floor, `resasc`, noise, estimate);
   - `_nodal_guard`'s docstring (Rule A, "the gate's 6 pi of the time");
   - `_oriented_cc`'s docstring ("the nested Clenshaw-Curtis pair" → the rule, with its estimate);
   - `_locate_stationary_points`'s Rule A sentence (lines 4088–4094, 6 pi of the time);
   - `_adaptive_levin`'s docstring: Rule A in item 1, the guard's "below CC_GATE_PHASE_VARIATION" in item 4, and a
     paragraph on the rule and the four counts (lines 4575–4589); the leaf comment at the fallback branch
     ("|CC_{8N-7} - CC_{4N-3}| plus its noise term"); the trust check's docstring ("CC_{4N-3} on the fallback");
   - `adaptive_levin_sincos`'s docstring: the fallback paragraph, the four counts (line 6728), and Rule A twice
     ("the C2 gate's 6 pi of the time").
   Comments that say what 6π did at the time (07's Rule A, 07a's estimate, the order sweep) keep "6 pi" as the number
   then in force; those that describe the gate now name `CC_GATE_PHASE_VARIATION`. The phrase "nested pair" is kept
   where it describes the rule generically, since the estimate is still a nested pair.

### §3, the tests (`AdaptiveLevin/tests/test_levin_cc_estimate.py`, new, 10 tests)

All without Ray or a datastore. The module also holds `load_parent_07d()` (`ca1d8cfa`'s module by `git show`, as the
other modules read their parents) and `cc_rule_07c()`, a context manager that patches this module's
`_adaptive_levin_subregion_cc` with the parent's and `CC_GATE_PHASE_VARIATION` with the parent's `SIX_PI`: 07c's code
path, bit-identical to the parent on all 35 driver calls compared (Verification 3). The pseudo-noise is 07c's `noise()`
(a hash of the double's bits, imported from `test_levin_chase_stop`). References: mpmath at 40 digits split at the
integrand's breaks, the 30-digit difference as their error.

| §3 | test | what it asserts |
|---|---|---|
| (a) | `TestEstimate.test_a_estimate_directly` (subtests: N = 24 and 16 × smooth, kink, tent, noise) | below |
| (b) | `TestNoiseTerm` b1, b2, b3 | below |
| (c) | `TestThreshold` c1, c2, c3 | below |
| (d) | `TestNotTouched.test_d_untouched_call_bit_identical` | below |
| (e) | `TestWindowCount.test_e_count_reads_the_returned_estimate` | below |
| (f) | `TestCounts.test_f_counts` | below |

- **(a)** On `_adaptive_levin_subregion_cc` directly, $\theta = 3x$ on $[0, 1]$, at $N = 24$ and $N = 16$:
  smooth $(e^{-x}, \cos x)$; a kink $(|x - 0.37|, 0)$; a tent of half-width half the local spacing at an odd node of
  the $(8N-7)$-point grid, between two $(4N-3)$-point nodes, added to $e^{-x}$ (derivative jumps neither
  ${\rm CC}_{2N-1}$ nor ${\rm CC}_{4N-3}$ sees, the shape of an error the two share); and $\cos x + \eta n(x)$,
  $\eta = 10^{-9}$. On each the value equals ${\rm CC}_{4N-3}$ computed in the test (Trefethen's `clencurt` weights,
  written out, at the module's nodes) within $10^{-14}$; on the kink it differs from ${\rm CC}_{2N-1}$ by more than
  $10^{-6}$; on the noise case the noise's shift of ${\rm CC}_{4N-3}$ is positive and at most $\eta$ (measured
  $5.1\times10^{-11}$); and $\max(\text{abserr\_direct}, \text{round\_off\_err}) \ge |{\rm value} - {\rm ref}| +
  \epsilon_{\rm ref}$. Measured at $N = 24$: kink true $1.06\times10^{-5}$ against $1.41\times10^{-4}$; tent
  $4.26\times10^{-3}$ against $8.53\times10^{-3}$; noise $1.0\times10^{-12}$ against $2.2\times10^{-10}$.
- **(b1)** $\theta = 0$, $f = (0, p)$, $p$ of degree 10: $\kappa s_{\rm own} > 0$, at most the floor without it
  ($3.2\times10^{-15}$ against $1.1\times10^{-14}$ at $N = 24$), `cc_noise_set` false.
- **(b2)** $\theta = 0$, $f = (0, e^{-x} + \eta n(x))$, $\eta = 10^{-9}, 10^{-12}$: $s_{\rm own}/(\sigma\sqrt{\sum w^2})$
  with $\sigma = \eta/\sqrt3$ in $[0.5, 2]$ (basis: Part A′'s 5th percentile 0.838 and median 1.01 on 1,089 group
  leaves, and a single draw's scatter of about 11%; measured 0.773, 0.935 at $N = 24$ and 0.814, 1.004 at $N = 16$);
  `cc_noise_set` at $\eta = 10^{-9}$.
- **(b3)** On (a)'s four integrands, (b1)'s polynomial and a scatter case, both orders: `abserr_direct`,
  `phase_err_roundoff` and `phase_err_less_node` each equal the same solve's at $\kappa = 0$ plus $4 s_{\rm own}$, bit
  for bit, $s_{\rm own}$ recorded from `_cc_own_noise` through the module attribute and the 4 a literal (§1.2); the
  value is unchanged by $\kappa$; `cc_noise_set` is $4 s_{\rm own} > \max({\rm estimate}, {\rm floor})$ at $\kappa = 0$.
- **(c1)** `CC_GATE_PHASE_VARIATION == 10π`; a region of $V = 0.999 \times 10\pi$ is direct, one of
  $1.001 \times 10\pi$ is Levin (`_adaptive_levin_subregion`). **(c2)** Calls with root $V = 8\pi$ and $9.99\pi$
  converge in one region with 7 Clenshaw–Curtis solves and `num_cc_gate_route_changes` 1; $V = 5\pi$, 0. **(c3)**
  Rule A: on $\theta = 16\pi(x - 1/2)^2$ ($V = 8\pi$, a root at 1/2) the located points, windows and skipped phases are
  the same with the gate patched to $6\pi$ and to $100\pi$.
- **(d)** 07c's (f1), $\theta = 10^3 x$: no Clenshaw–Curtis solve (four Levin leaves of $79.6\pi$, their quarters
  $19.9\pi$), no window, no guard, all four 07d counts 0, and bit-identical to `ca1d8cfa` in value, abserr,
  `converged`, `num_regions`, `max_depth` and 22 region fields.
- **(e)** $\theta = 10^3(x - 1/2)^2$, $f = (e^{-x}, \cos x)(1 + \eta n)$, $\eta = 0$ and $10^{-13}$, rtol $10^{-8}$,
  `depth_max` 12: every `_cc_leaf_unresolved(abserr, resasc)` call the window's count makes is fed a pair
  (`abserr_direct`, `resasc`) some Clenshaw–Curtis solve returned in the call, that `resasc` within $10^{-10}$ of the
  $(4N-3)$-point resasc the test computes on the solve's span and away from the $(2N-1)$-point one.
- **(f)** On (e)'s calls: `num_cc_solves` equals the calls to `_adaptive_levin_subregion_cc` (wrapper);
  `num_cc_gate_route_changes` those not under `_oriented_cc` with $V \ge 6\pi$; `num_cc_noise_set` those with
  $4 s_{\rm own} > \max({\rm estimate}, {\rm floor})$, each re-solved at $\kappa = 0$; `num_cc_resampled` 0; at
  $\eta = 10^{-13}$ (142 solves, 16 route changes, 61 set) all positive and the set count below the solves.

### Existing assertions changed (§3, "Existing assertions")

The fix acts on every Clenshaw–Curtis solve, so 30 existing tests in 11 modules failed at the commit before these
changes (87 failures and 3 errors, counting subtests). Each is listed with what changed and why. Two kinds of change:
- **P1, the new code:** a test of the rule's own floor, or of a property of the call at the shipped code, stays on the
  new code; a number it pins is re-measured and updated, the old value kept in a comment.
- **P2, 07c's code path:** a test that certifies a mechanism other than the Clenshaw–Curtis rule (the chase stop, the
  trust yardstick, the signed node sum, the round-off exit's premise, the window's stop, the rate rules, the slow-leaf
  trust test) on a construction made of 07c's rule, or that pins the hex of an older module or variant, runs its
  commit-side call inside `cc_rule_07c()`, its assertions unchanged. Where it compares with a parent or variant it
  loads, only the commit's call is patched. The path is bit-identical to `ca1d8cfa` (Verification 3), so the test
  still certifies the mechanism; the new code's numbers on the same call are given beside it.
Every changed test's docstring gains a paragraph beginning "Since prompts/levin-phase-floor/07d-…".

**`test_levin_phase_variation.py`** (§1.2's list):
1. Tests 2 and 3 (lines 169, 171, 202, 203 of the tree) compare 07a's variation $V$ and 07's estimate with $6\pi$:
   they test 07a's variation estimate at the number $6\pi$, not the gate, so they now read `SIX_PI_07A = 6.0 * math.pi`,
   a literal with a comment saying why. Both still pass on the new code: test 2's $V = 18.47$ is below $10\pi$ too, and
   test 3's bump has $V = 10\pi$ exactly over $[0, 1]$, not below the new gate, so $[0,1]$ still goes to the Levin solve
   and its halves ($5\pi$) to the rule. The module docstring names the new constant.

**`test_levin_quadrature.py`** (§1.2's list; comments only, no assertion changes):
2. Line 111 (the two-phase basis "comfortably above SIX_PI"): describes the gate; now names `CC_GATE_PHASE_VARIATION`
   and records that this file's smallest span with that basis is 36.75 rad, measured at the commit (above $10\pi$; the
   fallback is still not exercised by it).
3. Line 305 (`GRZIntegral(10)`'s assertion message, "below SIX_PI ~= 18.8 rad"): the gate; now names
   `CC_GATE_PHASE_VARIATION` ≈ 31.4 rad. The assertion (every region a fallback region) is unchanged and passes.
4. Line 314 (06c's comment on `GRZIntegral(100)`, "below the SIX_PI gate (102 leaves …)"): a measurement at 6π; now "the
   gate's 6 pi of the time", with 07d's 10 leaves added.
5. Line 747 ("phase span >> SIX_PI"): the gate; now `CC_GATE_PHASE_VARIATION`.

**`test_levin_chase_stop.py`** (07c's module; P2 for all but (f1), which makes no Clenshaw–Curtis solve and is
unchanged):
6. `_b_call()` (feeding (b), (c1), (c2), (d), (e)) runs both its commit call and its parent path (stop limit infinite)
   inside `cc_rule_07c()`; `_nu_B()`'s three leaf solves too, since they are compared bit for bit with the 07c-path
   regions; (c3) and (f2) run their call inside it. Why: they certify 07c's stop, unchanged by 07d, on constructions of
   07c's rule; at the new code the chase in (b) does not arise (0 stops, 3 noise-level bisections; counts 158, 18, 158,
   0) and (f2)'s counted bisection does not occur. New code, against the clean reference: (b) converged, 26 regions,
   abserr $6.79\times10^{-13}$, error $9.1\times10^{-15}$; (c3) converged, 11 regions (32 at 07c), abserr
   $5.29\times10^{-13}$, error $8.3\times10^{-15}$; (f2) converged, 4 regions, error $8.3\times10^{-17}$. No `converged`
   change. (b)'s and (c3)'s new abserrs are below the module's `NOISE_BOUND` ($1.774\times10^{-12}$), the worst-case
   margin its honesty assertion adds (Observations 3).

**`test_levin_trust_yardstick.py`** (P2 for tests 2–5; test 1 unchanged):
7. Test 2 (`test_bit_identical_to_the_variant`): the commit's two calls per cell run inside `cc_rule_07c()`; the key
   set admits `CC_KEYS_07D` (the four counts), left out of the compared records as 07c's are; **two assertions added
   per cell** on the 07c path: `num_cc_gate_route_changes == 0` and `num_cc_noise_set == 0`. Why: it certifies the
   yardstick by bit identity with 06l's variant and 06g's parent, which carry 07c's rule.
8. Tests 3, 4, 5 (`test_trusted_on_cancelling_node_errors`, `test_levin_parent_keeps_cc_halves_out`,
   `test_levin_over_levin_far_above_noise_stays_untrusted`): `_leaf()` and `leaf_solves()` run the commit inside
   `cc_rule_07c()`. Why: their leaves are made of 07c's Clenshaw–Curtis halves; at the new code test 3's leaf is never
   made, test 4's is a floor-limited Clenshaw–Curtis leaf (r = 0.199), and test 5's is untrusted (r = 197).
   Test 2's cells at the new code: three make no Clenshaw–Curtis solve and are identical; sqrt_fallback 5 → 4 regions,
   runge_fallback 11 → 5, sinc_12b 17 → 10, noisy(A=128) 17 → 20, far(3e5) 107 → 25, each value within 0.021 of the
   sum of the two abserrs. **noisy(A=512, k=60) goes from converged to unconverged** (abserr $9.08\times10^{-13}$ →
   $2.16\times10^{-12}$ against a request of $10^{-12}$; 81 regions; counts 331, 55, 40, 0). The test has no reference;
   one was computed for the noise-free integrand (mpmath, 25 against 35 digits agreeing to $1.4\times10^{-23}$):
   honest at both codes, error $1.05\times10^{-15}$ (0.001 of abserr) at 07c and $4.99\times10^{-15}$ (0.002) at 07d.

**`test_levin_end_nodes.py`** (P1):
9. Test 4 (`test_node_term_bounds_node_error`, 43 subtests failing): `N_VALUE = 2 * N_FINE - 1` added and the true
   node error taken on the $(4N-3)$-point grid, the grid the rule's node term is now formed on. Assertions unchanged.
   The 43 failures were the grid mismatch, not the bound (e.g. "T3 level 0 leaf" compared the 47-point rule's node
   error, $2.109\times10^{-17}$, with the 93-point rule's $\phi_{\rm node}$, $2.095\times10^{-17}$). On 122 cells the
   largest true error / $\phi_{\rm node}$ is 0.992 (0.997 at the parent on its grid), every cell honest.
10. `test_supplied_theta_prime_within_eta`: θ′ compared at the $(4N-3)$-point grid's nodes (the old grid paired
    values with the wrong nodes, worst 860 u); now worst 0.998 u against the unchanged bound 4 u.

**`test_levin_node_sum.py`** (P2):
11. Tests 4, 5, 6 (`test_signed_node_sum`, `test_node_limited_stop`, `test_floor_limited_keeps_own_node_error`): the
    commit's call inside `cc_rule_07c()`; test 4's comparison with 06g's parent stays outside. Why: constructions of
    07c's rule (238 fallback leaves; a node-limited stop on 5 leaves; a floor-limited root leaf) certifying the signed
    node sum. New code: GRZ(1000) unconverged, 34 regions (parent unconverged, 238); test 5 unconverged, 1 phase-limited
    leaf, not node-limited (parent unconverged, node-limited), honest; test 6 unconverged, 2 leaves (parent
    unconverged, 1), honest. No `converged` change.

**`test_levin_cc_floor.py`** (P2 for one branch):
12. Test 6 (`test_nothing_else_moves`): the branch asserting bit identity on problems with no fallback leaf at the
    parent runs both its commit call and the in-process 06c parent (`parent_floor()`) inside `cc_rule_07c()`;
    `_parent_cc()` gained a `rule=` keyword (default as before) so the in-process parent wraps the rule in force. The
    honesty branch stays on the new code. Why: at $10\pi$ `input_validation_order_clamp` has fallback leaves, so the
    branch's premise no longer held, and the in-process parent wrapped the new rule. New code: every problem keeps the
    parent's `converged` and is honest; `input_validation_order_clamp` is converged, 6 regions, its abserr
    $1.7\times10^{-14}$ → $1.13\times10^{-8}$ (Observations 6).

**`test_levin_roundoff_exit.py`** (P2):
13. Test 4 (`test_nothing_else_moves`): both calls, with and without the premise bypass, inside `cc_rule_07c()`. Why:
    "bit-identical across the bypass" was a property of 07c's rule; at the new code GRZ(10), GRZ(100), GRZ(1000) and
    `input_validation_order_clamp` differ across it (e.g. GRZ(10) abserr $1.1\times10^{-14}$ against
    $1.55\times10^{-8}$ under the bypass). New code against the parent on every problem: `converged` unchanged, every
    one honest; regions SincIntegral 17 → 10, GRZ(10) 12 → 2, GRZ(100) 100 → 10, GRZ(1000) 238 → 34,
    `input_validation_order_clamp` 1 → 6, `stationary_phase` 513 → 231.
14. `test_exit_against_its_premise_fallback_branch`: the body inside `cc_rule_07c()`. Why: (a) and (b) are
    constructions of 07c's rule (2 and 4 fallback regions); at the new code each is one leaf at depth 0. New code: (a)
    converged, honest; (b) unconverged, honest (both as at the parent).

**`test_levin_rate_rules.py`** (07b's module):
15. (b) `test_b_leaf_beside_a_window`, P2 for its "rule refusing nothing" call only (it pins 07a's hex): inside
    `cc_rule_07c()`. The main call is unchanged: converged, 16 regions, honest at 0.00444. At the new code the
    refuse-nothing call is still 7.62 times its abserr from the reference, the carrier present.
16. (d) `test_d_refused_at_depth_max`, P2 for the parent hex, P1 for the last assertion: the parent-path call inside
    `cc_rule_07c()`; the "same declaration with the rule refusing nothing" assertion now compares with a new-code call
    refusing nothing (asserted converged), whose abserr equals the main call's, `0x1.8075b9a053779p-33` (07c's
    `…aba2113acp-33`). Unconverged at both codes, 2 untrusted leaves.
17. (e2) `test_e2_clenshaw_curtis_leaves_untested`, P1 and P2 as two subtests: the non-hex assertions (rule never
    called, both counts 0, every leaf direct, the trusted wide leaf [0, 3]) at the new code and on the 07c path; the
    parent hex on the 07c path only. New code: 2 regions (6), converged at both.

**`test_levin_stationary.py`:**
18. Test 10b (`test_10b_untrusted_leaves_at_noise_stop`), P2, the call inside `cc_rule_07c()`. Why: it certifies the
    window's stop at the integrand's noise, a construction of 07c's rule; at the new code the noise term declares the
    noise, the window does not stop (count 4), and the call converges in 70 regions (145 at 07c), abserr
    $3.58\times10^{-14}$ ($1.435\times10^{-12}$). Honest against the unquantised closed form ($9.4\times10^{-16}$, 0.026
    of abserr), but below the quantisation's bound $2^{-40}$ that its honesty assertion adds (Observations 3).
19. Test 14b (`test_14b_unresolved`), P1 (§3 (e)'s concern): `resasc` recomputed on the $(4N-3)$-point grid; the peak
    moved from node 13 of the $(2N-1)$-point grid, width $10^{-3}$, to node 13 (odd) of the $(8N-7)$-point grid, width
    $5\times10^{-3}$, so the returned rule misses it while the estimate sees it. At the old peak the noise term reads the
    peak as a plateau and abserr exceeds resasc (1.70×), so the case no longer showed the factor 200 at work. New: the
    smooth leaf resolved (200 abserr $7.0\times10^{-13}$ of resasc), the peak unresolved (200 abserr/resasc 120,
    abserr/resasc 0.60).
20. Test 14d (`test_14d_value_test_against_the_request`), P1: `stopped_after_split` 22 → 49 and `num_regions` 32 → 55,
    old values in comments; the rest unchanged. Unconverged and honest at both codes (0.0206 and 0.0126 of abserr).
    Tests 13 (b) and (c), which route on $V$ near $6\pi$, test the location and pass unchanged: it reads no threshold.

**`test_levin_leaf_values.py`** (P2):
21. `test_slow_leaf_untrusted_and_refined`: the commit's call and the `trust_test()` loop inside `cc_rule_07c()`, the
    06g parent outside. Why: 06g's slow leaf ($r = 0.9107$) is formed from 07c's rule; at the new code its halves go to
    the rule and $r = 1.028$. New code: converged, 10 regions (17), error $1.4\times10^{-15}$.
22. `test_noise_leaf_trusted`: the call inside `cc_rule_07c()` (at the new code no region has $1/4 \le r < 1$ with
    $\delta_1$ within its floor; 3 at the parent).

**`test_levin_global_budget.py`:**
23. Test 12b (`test_trust_check_before_resolution`), P1 and P2: at the new code converged, honest, every region trusted
    with $r < 1$, and the phase-limited count 0 (was 1); the original assertions, the count 1 among them, on the 07c path.

No other test changed. Calls whose `converged` changes, parent → commit, among the existing tests' calls: only item 8's
noisy(A=512) (converged → unconverged, honest at both).

### Records (`docs/levin-phase-floor/07d/code/`)

- `compare_v93t10_07d.py`, `out/compare_v93t10_07d.{txt,json}`: the commit against V93T10 and the parent on 35 driver
  calls and 14 single solves, and 07c's code path (`cc_rule_07c()`) against the parent.
- `breakages_07d.py`, `out/breakages_summary.txt`, `out/breakages/`.
- `lg_draws_07d.py` (07c's harness copied, changed where marked "prompt 07d": the four counts recorded, V93T10's read
  from its module dict `_A2_C`, and left out of the result digest; three trees, the parent, the commit and V93T10; the
  report by whether the fix acts, the converged changes listed with 07a's three calls and 07c's 26 named),
  `out/parent/`, `out/commit/`, `out/V93T10/`, `out/lg_draws_report.{txt,json}`.
- Each script is `black`-formatted before its recorded run, and its outputs record its SHA-256; no `CLAUDE.md` change
  is needed.

## Predictions and verdicts

### Part B's acceptance (§7 item 2)

| Rule | Number | Verdict |
|---|---|---|
| §1 as amended | §1.1–§1.5; equal to V93T10 with equal counts on 35 driver calls, 14 single solves and 4,320 draw calls | holds |
| every test of §3 passing | the seven roots (Verification 1) | holds |
| every breakage caught | (i)–(vi), each by the test §3 names (Verification 2) | holds |
| every changed existing assertion listed | 30 tests in 11 modules, and comments in a twelfth (above) | holds |
| the seven roots rising only by the tests added | AdaptiveLevin +10 = tests added; the other six equal | holds |
| the LiouvilleGreen draws with no new failure | 0 of 80 units newly failing; 42 of 42 untouched calls bit-identical | holds |

### §5's and §6's predictions, copied before they run — not yet scored (Part C)

*Copied word for word from the prompt, §5 and §6, their headings demoted two levels.*

#### 5. Rehearsal R33 — on the Mac Studio, after the code commit is pushed

As 07c §5, with:
- records under `var/levin-phase-floor-07d/rehearsal/`;
- slug `r33-07d`, `--prompt 07d`, `--job-name levin-phase-floor-07d`, and the fingerprint `1b332ffa…`.

**The parent rehearsal is R32** (`levin-phase-floor-07c-r32-07c-20261119T140417`,
`var/datastores/levin-phase-floor-07c-r32*`, kept). It was built at `cc3368ce`, whose code is
`67b5af74`'s.
1. **S0′ — R32 is the parent.** `git diff --stat 67b5af74 <code commit's parent> -- . ':!docs'
   ':!prompts'` lists nothing the build imports. **Holds** if so; if not, the agent stops and asks.
2. **The lock:** `acquire`, `set-pid` at the launch, and `release` after the readers, each a shell
   command of its own.
3. **R33 against R32,** by `tools.store_value_diff` with `--json` and 05's `classify_non_qsi.py` (job
   names normalised in a copy):
   - **outside `QuadSourceIntegral`,** every value table identical, every other difference of the four
     volatile kinds, and no row in one store only. If §9 changes an integrand provider, it states what
     moves here instead;
   - **inside it,** every moved row tabulated, with the rows whose `total_converged` changes in each
     direction.
4. **S3 — consistency.** Every row with finite `total_abserr` in both has
   $|{\rm total}_{33} - {\rm total}_{32}| \le {\rm total\_abserr}_{32} + {\rm total\_abserr}_{33}$, or is
   one §9 predicts to move further, listed. A row whose new declaration is smaller and fails this is
   reported with its calls' route changes and estimates.
5. **S1, S2 and S4** are re-scored on R33 as in 07c §5.
6. **The readers** run on R33 and exit 0, and the store is byte-identical before and after. **The lock**
   is released, and `ray-processes` lists nothing. **R32 and R33 are kept.**

#### 6. The census — on the Mac Studio, Ray-free, on A3 v2's scratch copy

07c's census harness (`docs/levin-phase-floor/07c/census/scripts/`) is copied, changed where marked
"prompt 07d", and re-run at the code commit. The "before" is 07c's census
(`levin-phase-floor-07c-census-20261119T201419`, checkpoint `33d94ce2…`). Diagnostics are as §9 sets
them, and record on every call whether the change acted, and where.

**Gate Z0, before the run:**
- **The parent's code is 07c's:** as S0′.
- **The harness reproduces 07c's census at the parent,** bit for bit, on 06k's 100 K0 units: one
  registered run, slug `k0`.
- **The instrument changes nothing:** a unit where the change acts and one where it does not, at the new
  code, with and without it, are bit-identical.
- **The new code is deterministic:** two runs of one unit where the change acts are bit-identical.
- **The projection** comes from K0 at the new code. Beyond 48 hours is a stop, with what would be cut.

**Predictions, fixed before the run** (README §5 rule 3), copied unchanged into `state/predictions.md`
with §9's additions:
- **Z1 — nothing else moves.** Every call in which the change does not act (§9 names the counts) equals
  07c's call bit for bit in value, abserr, `converged`, `num_regions` and maximum depth. Every row all of
  whose calls are such equals 07c's in `total` and `total_abserr`. **Holds** if true at both rtols.
- **Z2 — no Levin leaf is returned across a stationary point** (07a's W2).
- **Z3 — every phase with something to locate is located** (07a's W3), at the threshold Rule A reads.
- **Z4 — honest where it is referenced.** Every call referenced in 07b (U4's 190), in 07c's Part A (R4,
  R5 and R2-28) and in this prompt's Part A ((Rt) and (Ra)) is scored from the census against its
  reference by 07's rule. **Holds** if every one is honest or unconverged. R2-28 is reported against its
  own issue, as §9 predicts it.
- **Z5 — the gate is consistent, and the guard is a backstop** (07a's W5), at the new threshold.
- **Z6 — 06b's 24 are honest,** against $S_{10}$ by 06b's rule, or unconverged.
- **Z7 — the pairs:** $r_g$ over all 2,484 pairs. Every pair has $r_g \le 1$, or a member unconverged.
- **Z8 — the code is the chosen variant:** every census call Part A replayed equals the variant's replay
  record bit for bit in value, abserr, `converged`, regions and maximum depth.

**Descriptive, not scored:**
- where the change acts, by class, band, $k$ and rtol;
- regions whose route changed, in each direction;
- `converged` changes against 07c's and against 07b's, in each direction;
- rows at depth 20;
- the abserr ratio against 07c's;
- values moved past the sum of the two codes' abserrs (**the sum**, not each separately);
- window stops and chase stops against 07c's;
- regions and cost against 07c's.

### §9.4.5's additions to §5's and §6's predictions, copied before they run — not yet scored (Part C)

*Copied word for word from the prompt, §9.4.5.*

#### 9.4.5 Additions to §5's and §6's predictions

- **Z1** counts a call untouched if `num_cc_solves` is 0. It compares value, abserr, `converged`,
  `num_regions` and the maximum depth. On Part A′'s 4,220 replays, 595 are untouched and every one is
  bit-identical to the tree.
- **Z4, on the user's ruling.** R2-28 is predicted converged and dishonest at 1.10 of its abserr at
  both rtols, as its census calls `c-1e-7-0-13757-5` and `c-1e-8-0-13757-5`. Its error is the
  blind spot's, which is 08's. **Z4 holds** if every other referenced call is honest or unconverged.
  `R:a:W2-0-12977-5` is predicted honest at 0.0498. R2-28 is reported with its ratio against
  `[07c-r2-28-trusted-levin-leaf-hides-a-common-error]`.
- **Z5** is scored at `CC_GATE_PHASE_VARIATION`. **Z3** is unchanged: Rule A reads no threshold.
- **Z6 and Z7 are predicted to hold.** No call Part A′ replayed is newly converged and dishonest under
  V93T10. The calls not replayed rest on that.
- **Z8 — the code is V93T10.** Every census call that Part A′ replayed under V93T10 (4,205 distinct
  calls; 4,220 records at both rtols) equals its replay record bit for bit in value, abserr,
  `converged`, regions and maximum depth. **Holds** if all do.
- **§5, R33 against R32.**
  - **Outside `QuadSourceIntegral`,** nothing moves (no provider changes).
  - **Inside it,** every row with a call that makes a Clenshaw–Curtis solve may move.
  - **S3** is scored as written. A row past its bound is reported with its calls' three counts.
- **Descriptive, added:**
  - the three counts by class, band, $k$ and rtol;
  - route changes in each direction (only Levin → Clenshaw–Curtis is possible);
  - `converged` changes against 07c's in each direction, with the calls listed;
  - chase stops, stopped leaves and their declarations, against 07c's;
  - window stops against 07c's.

**Part B's review** replays these calls from copies of their captures at the code commit, each equal
to V93T10's replay record in `float.hex`:

| call | why | value | abserr | regions | depth | converged |
|---|---|---|---|---|---|---|
| R2-28 (`c-1e-7-0-13757-5`) | the blind spot | `-0x1.654fbbd97572cp-96` | `0x1.1c6301b316e32p-122` | 92 | 12 | yes |
| `R:a:W2-0-12977-5` | route-changed; the call $12\pi$ broke | `-0x1.f3e1629c3c030p-80` | `0x1.d4c8e49cd7d44p-104` | 30 | 8 | yes |
| `d:h-1e-7-0-12851\|a20` | (Ra), route-changed | `0x1.a32bb1c43505ep-39` | `0x1.9e1a8b9a9eda6p-76` | 7 | 3 | yes |
| `d:h-1e-7-0-12661\|a11` | untouched, and equal to 07c's census | `0x1.e15ce4f5719b9p-46` | `0x1.4bd325169a900p-82` | 8 | 3 | yes |

The second and third are also scored against their references.

## Deviations from the prompt

1. **`IMPLEMENTATION CHOICE`** — the names §1 leaves to the agent: `_cc_own_noise`, `_cc_chebyshev_coefficients`,
   `_cc_coefficient_cosines` and `_cc_standard_chop` for the variant's `_a2_s_own`, `_a2_coeffs` and
   `_a2_standard_chop`; the dict keys `cc_noise_set`, `cc_num_resampled` and `cc_gate_route_changed` through which a
   solve reports to `_adaptive_levin`'s counts; `cc_rule_07c()` and `load_parent_07d()` in the test module. The
   coefficient sums are formed as the variant forms them, row by row, the rows' cosines cached by $n$; `standardChop`'s
   `max(int(np.argmin(cc)) + 1 - 1, 1)` is written `max(int(np.argmin(cc)), 1)`, the same integer.
2. **`IMPLEMENTATION CHOICE`** — a fourth count, `num_cc_resampled`. §1.1 says a span whose grids do not nest is
   "resampled and counted" (the variant's `num_a2_resampled`), and §1.4 lists three counts. The count is kept, beside
   the three, so that §1.1's count exists; it is 0 on every call run here, as in all of Part A′'s replays. The resampled
   branch is the variant's: it samples the finer grid afresh and, as the variant does, still hands the
   $(2N-1)$-point grid's θ′ to the $(4N-3)$-point node term (Observations 5).
3. **`IMPLEMENTATION CHOICE`** — the counts are carried by the solve's dict and summed in `_adaptive_levin`'s `_solve`,
   for every solve whose dict is `is_direct`, in place of the variant's module counter. Every Clenshaw–Curtis solve
   of a call goes through `_solve` (one per call to `_adaptive_levin_subregion_cc`, which (f) checks). The variant also
   counted the re-solves `_write_progress_data` makes under `emit_diagnostics=True`, which production never sets and
   which change no value; the commit does not count them.
4. **`IMPLEMENTATION CHOICE`** — the existing tests that certify mechanisms other than the rule on constructions of
   07c's rule run their commit-side calls on 07c's code path, `cc_rule_07c()` (the parent's
   `_adaptive_levin_subregion_cc` patched in and the gate at $6\pi$), their assertions unchanged, as 07c's tests ran the
   parent's path by patching `_LEVIN_CHASE_STOP_LIMIT`; the path is checked bit-identical to the parent's module on all
   35 compared calls. §3 says each changed assertion is listed with why; it does not say how. The alternative,
   re-pinning every number to the new code, would have turned those tests into records of 07d's numbers, certifying
   nothing about the mechanisms, whose constructions mostly no longer arise at the new code. The new code's numbers on
   each such call are in the list above.
5. **`IMPLEMENTATION CHOICE`** — the existing-test changes were made by three forked agents of this session, one per
   group of modules, under one written policy (P1/P2 above), each module run and its numbers measured by the fork; the
   agent reviewed the diffs, re-ran the seven roots, and added a blank line to one docstring (test 14d).
6. **`IMPLEMENTATION CHOICE`** — the non-finite checks. The variant checks $f$ and θ′ only on the $(2N-1)$-point grid,
   as the tree does; the commit also raises the tree's `ValueError` (with its warning) on a non-finite $f$ sample of
   the $(4N-3)$- or $(8N-7)$-point grid or a non-finite θ′ on the $(4N-3)$-point grid, where the variant would return
   a non-finite value or declaration. No finite input is affected.
7. **`IMPLEMENTATION CHOICE`** — (a)'s "derivative jump" is two cases: a kink, which tells ${\rm CC}_{4N-3}$ from
   ${\rm CC}_{2N-1}$ (breakage (iv)); and a tent at an odd node of the $(8N-7)$-point grid, which neither
   ${\rm CC}_{2N-1}$ nor ${\rm CC}_{4N-3}$ samples, the only shape on which the restored pair understates (breakage
   (iii)): on the kink $|{\rm CC}_{4N-3} - {\rm CC}_{2N-1}|$ overstates ${\rm CC}_{4N-3}$'s error, so it cannot catch
   (iii). The tent stands for the error ${\rm CC}_{2N-1}$ and ${\rm CC}_{4N-3}$ share, which Part A′ found on 19 leaves.
8. **`IMPLEMENTATION CHOICE`** — (a)'s noise case scores against the noise-free integral with the noise's shift in the
   returned value, and checks that shift against the stated bound $\eta\,(b - a)$ separately. Adding the worst-case
   bound to the true error, as 07c's module does with its `NOISE_BOUND`, would compare the declaration with a number
   ($10^{-9}$) the rule's answer is not off by ($10^{-12}$).
9. **`IMPLEMENTATION CHOICE`** — (b)'s "exactly $\kappa s_{\rm own}$" is tested by comparison with the same solve at
   $\kappa = 0$ (patched), with $s_{\rm own}$ recorded from `_cc_own_noise` through its module attribute and $\kappa$ the
   literal 4 of §1.2, so that breakage (i) fails it. (b)'s band is $[0.5, 2]$, with its basis in the module.
10. **`STRUCTURALLY REQUIRED`** — the breakages are reverted by writing the module's original bytes back, not by
    `git checkout -- <file>`, which would have restored `ca1d8cfa`'s module and discarded the change under test; after
    each the module's SHA-256 (`90e2cd3a…`) and `git status --short` are checked equal to their values at the start.
11. **`IMPLEMENTATION CHOICE`** — records beside the draws' harness: the comparison with V93T10 and the parent, and the
    breakages, as 07c kept them.
12. **`IMPLEMENTATION CHOICE`** — §1.5's list is extended to every comment that names the fallback's estimate, its grid
    or the gate (item 7 of §1 above), as §1 item 6 asks; "nested pair" is kept where it describes the rule generically.

The review replays of §9.4.5 (R2-28, `R:a:W2-0-12977-5`, `d:h-1e-7-0-12851|a20`, `d:h-1e-7-0-12661|a11`) need their
captures, which are on the Mac Studio; they are the orchestrator's, as 07c's were.

## Verification performed

1. **The seven roots at the commit, serially** (the commit's working tree; `levin_quadrature.py` SHA-256 `90e2cd3a…`):

| root | baseline (dispatch SHA) | at the commit |
|---|---|---|
| AdaptiveLevin | 141 OK | 151 OK (+10, the tests added) |
| ComputeTargets | 781, 1 failure (the wall-clock flake) | 781 OK (the flake did not fire) |
| CosmologyModels | 39 OK | 39 OK |
| LiouvilleGreen | 156 OK, skipped 1 | 156 OK, skipped 1 |
| ObjectFactories | 280 OK | 280 OK |
| RunDrivers | 127 OK | 127 OK |
| tools | 64 OK | 64 OK |

   Only the log and the board changed after that.

2. **The breakages** (`breakages_07d.py`, `out/breakages_summary.txt`, `out/breakages/`), each applied as text
   replacements, `test_levin_cc_estimate` run, the module's bytes written back (SHA-256 `90e2cd3a…` and
   `git status --short` as at the start, after each):

| breakage | caught by (tests failing) | named in §3 |
|---|---|---|
| (i) $\kappa = 0$ | (a) kink at both orders, noise at N = 16; (b1), (b2) at $\eta = 10^{-9}$, (b3) all twelve; (f) at $\eta = 10^{-13}$ | (a)'s noise case or (b) |
| (ii) noise term in `abserr_direct` only | (b3), all twelve | (b) |
| (iii) the pair $|{\rm CC}_{4N-3} - {\rm CC}_{2N-1}|$ as the estimate | (a) tent at both orders; also (c2) at $9.99\pi$ | (a)'s jump case |
| (iv) ${\rm CC}_{2N-1}$ returned | (a) kink and noise at both orders; also (c2) at $9.99\pi$, (e) at $\eta = 10^{-13}$ | (a) |
| (v) the gate at $6\pi$ | (c1), (c2) at $8\pi$ and $9.99\pi$; also (f) at $\eta = 10^{-13}$ | (c) |
| (vi) the window's count fed $|{\rm CC}_{2N-1} - {\rm CC}_N|$ and the $(2N-1)$-point `resasc` | (e) at both $\eta$ | (e) |

   (vi) puts the $(2N-1)$-point `resasc` in the solve's dict and the old pair in a key of its own, which the count
   reads from the parent's halves' solves. A first run of the script applied (vi) through a leaf attribute that is
   `None` outside `build_p_sample`, and (e) and (f) raised rather than failed; the script was corrected and the
   recorded run is the second, all six re-run.

3. **The comparison with V93T10** (`compare_v93t10_07d.py`): 35 driver calls (the new module's (c2) three, (c3), (d),
   (e)/(f) two; 07c's chase calls (b), (c3), (f1), (f2); 07b's rate-rule calls (b)–(e2); the nine yardstick cells; the
   seven leaf-value cells; the round-off exit's test 1; the global budget's tests 1 and 3): **35 of 35 identical to
   V93T10** in every result key but `elapsed` and the counts and in every region field by field, with V93T10's counts
   (`num_a2_cc93`, `num_a2_route_changed`, `num_a2_resampled`) equal to the commit's; the fix acts in 22; **the 13 where
   it does not are identical to the parent**; **07c's code path (`cc_rule_07c()`) identical to the parent on 35 of
   35**. One `converged` change, the yardstick's noisy(A=512) cell (above). 14 single Clenshaw–Curtis solves of (a) and
   (b), at both orders: every key of the commit's dict but the three 07d flags identical to V93T10's
   `_adaptive_levin_subregion_cc`. The four review calls of §9.4.5 need captures on the Mac Studio, and are the
   orchestrator's.

4. **The LiouvilleGreen draws:** `lg_draws_07d.py` over seeds 0–19, the four default tests of
   `test_3bessel_analytic.py`, 80 units under each of the three trees, `--jobs 6` (`out/lg_draws_report.txt`):
   - test outcomes, parent → commit and parent → V93T10: pass → pass 80, **newly failing 0**;
   - 4,320 Levin calls paired, none unpaired. **The fix does not act in 42: all bit-identical to the parent.** It acts
     in 4,278 (114,871 Clenshaw–Curtis solves; 10,804 gate route changes in 4,242 calls; the noise term sets 26,107
     solves in 4,256 calls; 0 resampled); every one moves; `converged` (parent, commit): (True, True) 3,686,
     (False, False) 484, **(True, False) 73, (False, True) 35** (Observations 4; every one listed in the report);
     abserr commit/parent min $3.3\times10^{-7}$, median 0.199, max $7.3\times10^3$; regions 140,896 → 96,191; no call at
     depth 20 under either; chase stops 4,819 → 4,005;
   - 07a's three calls: seed 3 `test_YJJ` call 4 and seed 8 `test_JJJ` call 4 unconverged at both; seed 17 `test_JJJ`
     call 11 unconverged → converged. 07c's 26: unconverged at the parent, **converged at the commit, all 26**;
   - **V93T10 against the commit: 4,320 of 4,320 identical**, counts equal;
   - the three-Bessel results where the fix acts (1,080): honest 654 → 547, converged 857 → 884, converged and
     dishonest 422 → 533 (Observations 2); none where it does not act;
   - wall summed over units: parent 1,180.1 s, commit 964.8 s, V93T10 1,078.8 s (not comparable run to run: the laptop
     carried other work).

5. **black** on every Python file changed or added; `--check` clean on `AdaptiveLevin/` and
   `docs/levin-phase-floor/07d/code/`. `git log --oneline ca1d8cfa..HEAD` empty before staging; `git status --short`
   lists only the dispatch's paths.

## Observations not acted on

1. **On a kink, $|{\rm CC}_{8N-7} - {\rm CC}_{4N-3}|$ alone understates ${\rm CC}_{4N-3}$'s error, and the noise term
   carries the declaration.** (a)'s kink $|x - 0.37|\sin 3x$ at $N = 24$: true error $1.06\times10^{-5}$, the pair alone
   $6.07\times10^{-6}$ (0.57), the declaration $1.41\times10^{-4}$; at $c = 0.618$ the pair is 0.78 of the error, and at
   $N = 16$, $c = 0.37$, 0.79. Where the error falls as $h^2$ the pair is about $3/4$ of it. The declaration bounds the
   error because `standardChop` finds no plateau in the slowly decaying coefficients and the last eight's rms enters as
   "noise". Part A′ found pair93 understating on no settled leaf of production's integrands, whose spline knots jump in
   a higher derivative; a jump in a value or a first derivative inside a production leaf would rest on the same
   accident. Not acted on: §1 is fixed.
2. **The three-Bessel results' honesty falls in the LiouvilleGreen draws, with their errors unchanged.** Over the 1,080
   results the fix acts in: honest 654 → 547, converged 857 → 884, converged and dishonest 422 → 533, the worst
   $|v - {\rm analytic}|/{\rm abserr}$ $4.9\times10^6$ → $2.1\times10^9$. 119 results become dishonest (all converged;
   114 of them `test_YJJ_log_singularity`, 117 `quad_YJJ`) and 12 honest. On the 119 the error against the analytic
   value is the parent's to within 0.998–1.006; their abserr fell to a median 0.0012 of the parent's (at most 0.86). So
   the value did not get worse: the Levin calls' declarations shrank (median 0.199 over all acted calls), and an error
   of the result that the Levin calls do not carry, which the larger declarations covered by accident, is now exposed.
   Where that error comes from (the evaluator's other parts, or an error the Levin calls share) was not measured. Not a
   §8 stop: no test fails, and the code equals V93T10.
3. **Worst-case noise margins in two existing tests no longer fit under the new declarations.** 07c's
   `test_levin_chase_stop` adds `NOISE_BOUND` ($1.774\times10^{-12}$) to the true error, and `test_levin_stationary`
   10b adds the quantisation's $2^{-40}$; at the new code (b), (c3) and 10b declare $6.8\times10^{-13}$,
   $5.3\times10^{-13}$ and $3.6\times10^{-14}$, below those margins, while their errors against the clean references
   are 0.013, 0.016 and 0.026 of abserr. The margins bound the noise's shift of the integral; whether each call is
   honest against its noisy integrand's own integral is not measured. Those tests run on 07c's path (P2).
4. **`converged` in the LiouvilleGreen draws:** 108 calls change, 73 converged → unconverged and 35 unconverged →
   converged. The 73 are all at rtol $10^{-10}$, each a budget miss (abserr 1.009–4.82 times its request, median 1.19;
   abserr 1.07–9.5 times the parent's), no untrusted leaf, maximum depth 7. Of the 35, 26 are 07c's 26 (all converged
   again) and one is 07a's seed 17 `test_JJJ` call 11; 07a's other two stay unconverged. Regions 140,896 → 96,191;
   chase stops 4,819 → 4,005, stopped leaves 38,080 → 28,084; no call at depth 20 under either.
5. **The resampled branch** (grids that do not nest, which no span has been found to have) is the variant's: it samples
   $f$ and the basis afresh on the finer grid but still takes the $(2N-1)$-point grid's θ′ as the even nodes of the
   $(4N-3)$-point grid's θ′. Kept for bit identity with V93T10; unreachable on every call run.
6. **A declaration can rise by orders of magnitude.** `input_validation_order_clamp` (test_levin_quadrature) goes from
   one Levin region at abserr $1.7\times10^{-14}$ to six fallback leaves at $1.13\times10^{-8}$, converged and honest
   (error $8.9\times10^{-16}$), the noise term set on 6 solves; in the draws the abserr ratio reaches $7.3\times10^3$.
   The noise term reads a slowly decaying coefficient tail as noise (as in item 1), where an interpolant has not
   converged; this is overstatement, cost and the budget, not honesty.
7. **The census harness reads `SIX_PI`.** 07c's `docs/levin-phase-floor/07c/census/scripts/worker.py` line 2510
   (`SIX_07A = float(LQ.SIX_PI)`) reads a name the module no longer has; Part C's copy must read
   `CC_GATE_PHASE_VARIATION` or the literal, as it intends ("prompt 07d").
8. **Repeated phase evaluations.** Where no `theta_deriv` is supplied and `theta_mod_2pi` is, θ′ on the
   $(2N-1)$-point grid evaluates the raw phase at its $2N-1$ nodes and θ′ on the $(4N-3)$-point grid evaluates it again
   at all $4N-3$, the $2N-1$ shared nodes twice. Kept for bit identity; `QuadSourceIntegral` supplies `theta_deriv`.

## State handed on

- The code commit (not pushed). Part C (§5 R33, §6 the census) runs on the Mac Studio after it is pushed; §9.4.5's
  review replays are the orchestrator's.
- No issue moves in this commit (§4); `docs/OPEN_ISSUES.md` is not touched.
- The scratch worktrees are removed and pruned; nothing is running.
