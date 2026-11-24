# Prompt 07d — the $6\pi$ gate, the Clenshaw–Curtis estimate and the integrand's noise

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-11-21, on the user's decisions:
- **At 07b's records commit** (the board's Decisions, **07b Part C**), offered (a) a sharper
  Clenshaw–Curtis estimate measured in 07c, with the threshold in a 07d before 08; (b) one prompt for
  all; (c) 07c alone: *"Go with (a); make the records commit, then write 07c"*. So the threshold is this
  prompt's, after 07c and before 08.
- **On 2026-11-21, after `handover`'s record of B2's $10^8$ cell** (`74ebb54f`). The user asked what is on
  the critical path before 08. The orchestrator answered 07d. Its scope proposal: the Clenshaw–Curtis
  estimate (`[06d-cc-nested-estimate-measures-the-coarse-rule]`, which 07c measured but did not fix)
  and the noise floor (`[07c-integrand-noise-above-the-cc-floor]`) are in scope, because the threshold is
  chosen against the estimate on both sides of it. The references for
  `[07c-chase-stop-acts-on-unreferenced-analytic-calls]` go in its sample, and
  `[07c-stopped-leaf-declaration-overstates]` is left to 08. The user: *"Yes, write 07d on that
  scope"*.
- **Amended 2026-11-22 (§9.1)** at Part A's stop: a second measurement, Part A′, first.
- **Amended 2026-11-24 (§9.4)** at Part A′'s stop: the fix is variant V93T10, ${\rm CC}_{93}$
  returned with $|{\rm CC}_{185} - {\rm CC}_{93}|$ plus a noise term and the gate at $10\pi$. §1 and §3
  are written. The user: *"Go with V93T10, and put the blind spot in 08"*.

**Where it runs:** in three parts, as 07b and 07c did.
- **Part A, §2, the measurement:** on the Mac Studio, Ray-free, with no lock, in a worktree at the
  dispatch SHA, whose code is 07c's (`67b5af74`). It measures the integrand's noise, the
  Clenshaw–Curtis estimate against the true error, and both routes against the phase variation. It
  scores candidate estimates and thresholds, and takes two new reference samples. **It ends at a
  planned stop.** There the user chooses the estimate, its noise term and the threshold with the tables
  in front of them. The orchestrator writes the choice into §1, §3 and §9 as an amendment, committed and
  pushed before Part B.
- **Part A′, §9.2, a second measurement,** added at Part A's stop on the user's word (§9.1): on the
  Mac Studio, Ray-free, in a fresh worktree at the amendment's SHA, whose code is still 07c's. It ends
  at a planned stop, where the user makes the choice Part A's stop was for.
- **Part B, §1, §3 and §4, the code:** on the laptop, by a fresh-context subagent, in the main checkout
  at `HEAD`. One commit.
- **Part C, §5 and §6:** on the Mac Studio, after the code commit is pushed. **§5 is rehearsal R33,**
  under the lock, against 07c's R32. **§6 is the census,** Ray-free, on 06's certified scratch copy of
  A3 v2, with 07c's census as the "before". **This census is 08's "before".**

The records of Parts A, A′ and C are committed on the laptop afterwards. Part A's and Part A′'s go in
with §9's amendments, and Part C's in 07d's records commit.

**Closes, in its records commit, on the user's word:**
- **`[06d-cc-nested-estimate-measures-the-coarse-rule]`**, if:
  - the chosen estimate bounds $\text{CC}_{2N-1}$'s true error on every scored leaf of §2.3's sample,
    within its stated margin;
  - and §6's predictions hold.
- **`[07c-integrand-noise-above-the-cc-floor]`**, if:
  - the chosen declaration carries the noise §2.2 measures;
  - and §6's predictions hold.
- **`[07-c2-gate-threshold-was-never-measured]`**, if:
  - a threshold is chosen on §2.4's measurement, whether or not it is $6\pi$;
  - and §6's predictions hold.
- **`[07c-chase-stop-acts-on-unreferenced-analytic-calls]`**, if every referenced analytic call of
  §2.6's (Ra) that the stop acts in is honest or unconverged:
  - at 07c's code;
  - and in §6's census.

  Otherwise it is narrowed by the measurement.

**Narrows, by its measurements:** any of the four that does not close.

**Recommended model:** **Opus**, for every agent.

**Read first:**
1. **The board's §3 entries,** with every narrowing:
   - this prompt's four: `[07-c2-gate-threshold-was-never-measured]`,
     `[06d-cc-nested-estimate-measures-the-coarse-rule]`, `[07c-integrand-noise-above-the-cc-floor]`,
     `[07c-chase-stop-acts-on-unreferenced-analytic-calls]`;
   - beside them: `[07c-stopped-leaf-declaration-overstates]`,
     `[06e-cc-declared-phase-term-is-a-crude-proxy]` and
     `[07-cc-interior-nodes-shift-with-a-rounded-midpoint]`;
   - the Decisions rows **07b Part C**, **07c Part C** and **07d**.
2. **07c's prompt,** [`07c-chase-below-a-refused-leaf.md`](07c-chase-below-a-refused-leaf.md): §0.2,
   §2.3 and §9; and 07a's §0.2 and §1.1, on the phase variation $V$ the gate reads.
3. **`AdaptiveLevin/levin_quadrature.py`:**
   - **the constants, with their comments:** `SIX_PI`, `_LEVIN_DIRECT_SOLVE_PHASE_SPAN`,
     `_DQK21_SATURATION_FACTOR`, `_LEVIN_WINDOW_IROFF1_LIMIT`,
     `_LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO`, and `_LEVIN_CHASE_NOISE_FACTOR` with
     `_LEVIN_CHASE_STOP_LIMIT`;
   - **the Clenshaw–Curtis route:** `_adaptive_levin_subregion_impl` (the gate and `force_direct`),
     `_adaptive_levin_subregion_cc`, `_oriented_cc`, `_cc_roundoff_floor`, `_cc_node_term_parts`,
     `_declared_endpoint_phase_err` and `_cc_leaf_unresolved`;
   - **the Levin route's declarations:** `_roundoff_floor`, `_levin_G0_G1`, and
     `_Basis_SinCos.theta_prime_on_fine_grid`;
   - **what reads `SIX_PI` besides the gate:** `_locate_stationary_points` (Rule A);
   - **in `_adaptive_levin`:** `_check_leaf`, the round-off exit (`_roundoff_exit_premise_holds`), the
     window's count (`_window_bisection_counts`) and the chase stop.
4. **07c's Part A records,** `docs/levin-phase-floor/07c/measurement/`:
   - `log.md`, deviations 5, 8, 9, 10, 11 and 15;
   - `issues.md`, items 1–5;
   - `out/d2/` (`d2_table.md`, `d2_regimes.json`, `reproduce.json`);
   - the scripts `leafcc_07c.py`, `leafcc_specs_07c.py`, `leafref_07c.py`, `d2_population_07c.py`,
     `d2_regimes_07c.py`, `d2_table_07c.py`, `draw_refs_07c.py`, `score_07c.py` and
     `variant_loader.py`;
   - its references (`out/census_refs_07c*.json`).
5. **07c's census records,** `docs/levin-phase-floor/07c/census/`: `log.md`, `out/scores/summary.json`
   and `scripts/`; and the Mac Studio review, `docs/levin-phase-floor/07c/review/`.
6. **The evidence on the gate:**
   - 07a's census, `VERIFICATION.md` §7a: `1-34599`, a Levin solve just below $6\pi$ that was
     dishonest at 07;
   - 07a's three LiouvilleGreen draw calls, `docs/levin-phase-floor/07a/code/out/lg_draws_changes.txt`;
   - the 2026-09 audit, `docs/adaptive-levin-audit-2026-09.md` §2.2.

**Gate (Part A):**
- 07c's records (`d3b973d7`), the closures (`c1a91d73`) and this prompt's commit are on
  `origin/handover-remedial`.
- **The laptop:** the main checkout is clean. `runregistrykit list` shows nothing `running` or stale on
  either machine.
- **The code is 07c's:** `git diff --stat 67b5af74 <dispatch SHA> -- AdaptiveLevin ComputeTargets
  LiouvilleGreen tools` is empty.
- **On the Mac Studio:**
  - 07c's census checkpoint is `33d94ce2…`;
  - its run directory, `levin-phase-floor-07c-census-20261119T201419`, and
    `var/levin-phase-floor-07c/census/out/stop3/compare_census_calls.calls.jsonl` match the SHA-256 in
    07c's `census/KEPT_ON_MAC_STUDIO.txt`;
  - 07c Part A's captures and readings match its `measurement/KEPT_ON_MAC_STUDIO.txt`;
  - A3 v2's scratch copy matches its certification;
  - **the user has said the Mac Studio is free** of other projects' long runs (orchestrator note §0).

---

## 0. What this prompt is

### 0.1 The gate, and the pressures on it

**The gate.** It sends a region whose phase variation $V$ is below `SIX_PI` to the nested
Clenshaw–Curtis rule. The Levin solve takes every other region.
- $V$ is computed from the raw phase since 07a.
- The number $6\pi$ itself was never measured. The user, when the issue was opened: *"I just made it up
  by guessing a value that wouldn't be "too" oscillatory that CC doesn't handle it, but also wouldn't
  require refinement down to a single 2pi phase change."*

**What the records hold on both sides of it:**
- **Above the gate, Clenshaw–Curtis is accurate, and its estimate forces refinement.**
  - The audit measured the fine rule accurate to $3\times10^{-14}$ at $6\pi$ and $3\times10^{-9}$ at
    $10\pi$, at the order then in use.
  - The estimate $|{\rm CC}_{2N-1} - {\rm CC}_N|$ is the coarse rule's error, not the fine rule's.
- **Just below it, at tight tolerances, Clenshaw–Curtis's summed declarations exceed the request.** In
  07a's three LiouvilleGreen draw calls, one region with $V$ just under $6\pi$ went to Clenshaw–Curtis
  and the call lost convergence. Nothing in those calls was dishonest.
- **The Levin solve just below $6\pi$ was dishonest once.** `1-34599` at 07 was converged at 4.03× its
  declared error; 07a sent its region to Clenshaw–Curtis and it became honest. How many own-class calls
  are like it was never measured. 07b's references for three route-changed calls were honest.
- **Below the gate the Levin solve is badly conditioned.** Its eq. (151) floor grows as $1/G_0$, so
  lowering the threshold is not free either.

### 0.2 What 07c measured of the estimate

The fallback returns ${\rm CC}_{2N-1}$ and declares $\max(|{\rm CC}_{2N-1} - {\rm CC}_N|, \text{floor})$,
at the call's order $N$: 24 for `QuadSourceIntegral`, so ${\rm CC}_{47}$ and ${\rm CC}_{24}$ (07c Part A's
deviation 5). On a sample of 25 leaves per stratum (07c's D2, `out/d2/d2_table.md`):
- **The declaration falls below ${\rm CC}_{47}$'s true error** on:
  - 3, 11, 7 and 5 of 25 gate leaves, by the quarters of $V/6\pi$;
  - 5 of 25 chase leaves;
  - 18 of 25 window pieces.
- **No candidate tried bounds the true error on every leaf.** The minimum ratios to it are:
  - the pair, 0.0199;
  - a third nested rule, 0.00408;
  - the sum of the last four Chebyshev coefficients, 0.199;
  - the largest of them doubled, 0.118;
  - QUADPACK's `dqk21` transform, $2\times10^{-5}$.
- **Only 11–19 of 25 leaves a stratum were resolved,** that is, the reference's stated error below the
  true error. The rest were reported, not scored.
- **Two regimes, by D2's own classing** (true error within $3\sigma\sqrt{\sum w^2}$ of the noise, or
  not):
  - 22,700 of 22,876 measured chase leaves are noise-dominated;
  - on the 176 truncation-dominated chase leaves every candidate understates (the pair's median 0.23).
    **Why is not known.** An estimate from a leaf's own samples cannot see an error that ${\rm CC}_N$
    and ${\rm CC}_{2N-1}$ share. A derivative discontinuity at a spline knot inside the leaf would do it,
    and so would the reference's own failure. Neither has been tested.
- **And it still overstates near $6\pi$.** In 07c's test (b), two stopped Clenshaw–Curtis leaves declare
  $2.1\times10^{-10}$ and $2.6\times10^{-10}$ against a call's true error of $1.4\times10^{-14}$.

### 0.3 The noise

The integrand as evaluated carries a per-sample scatter of about $3$–$6\times10^{-12}$ of $|g|$ on the
chase's leaves (07c Part A, `issues.md` item 1).
- `_cc_roundoff_floor` models only the rounding of the sum, of the argument and of the nodes.
- No `theta_abserr` is declared for these integrands.
- On window pieces, $s_{\rm CC}/\text{floor}$ has median 484.
- On R31's record, every interpolatory rule disagreed with every other above the leaf's floor. 07c's
  leaf reference needed a least-squares fit to average the scatter (its deviation 8).

**Where the scatter comes from is not measured.** Candidates include:
- the amplitude tables' stored values, interpolated by splines;
- the phase;
- the products that form $g$.

"Noise" is a statistical description, and the scatter may not be independent from node to node:
- a spline through tabulated values is smooth between knots;
- a leaf spanning many knots samples a function whose higher derivatives jump at each one;
- a leaf inside one knot interval may show no scatter at all.

Which of these holds decides what a floor term can be: a declared bound from the integrand's provider,
or an estimate from the leaf's own samples.

### 0.4 Why the three are one prompt, and in this order

The threshold is chosen by comparing the two routes' declarations with their true errors on each side
of it. Clenshaw–Curtis's declaration is the estimate plus the floor, and both are known to fail: the
estimate understates on part of every stratum, and the floor misses the noise. A threshold chosen on
them would rest on numbers known to be wrong. So §2 measures in this order:
1. **The noise** (§2.2);
2. **the estimate, with a noise term** (§2.3);
3. **then the threshold, under the tree's estimate and under each estimate that qualifies** (§2.4).

Every candidate is scored against references whose stated error is itself measured. A leaf whose
reference cannot decide is reported, not scored.

### 0.5 The analytic calls

07c's chase stop acts in 82,172 of 289,482 census calls, 57,659 of them analytic, where its constants
were chosen on group calls and almost nothing is referenced (07c §9.3,
`[07c-chase-stop-acts-on-unreferenced-analytic-calls]`). Many of its stopped leaves are Clenshaw–Curtis
leaves, whose declaration is this prompt's estimate. §2.6's (Ra) takes the references that issue asks
for, scored at 07c's code and under every variant.

### 0.6 What it does not do

- **A stopped leaf's $r$-based declaration** (`[07c-stopped-leaf-declaration-overstates]`) is 08's,
  with what `total_converged` means. A change to the estimate changes a stopped Clenshaw–Curtis leaf's
  residual. That is reported (§2.7 table 5); it is not this prompt's to fix.
- **The depth cap and `rtol`** (G2's L2 as reopened, and Q5) are 08's.
- **The Levin route's own declarations** are unchanged: eq. (151), the trust check, the rate rules, the
  chase stop and its constants. The same goes for `[07c-r2-28-trusted-levin-leaf-hides-a-common-error]`
  and `[06m-levin-floor-misses-node-rounding-far-from-the-origin]`. §2.4 measures the Levin solve's
  honesty against $V$; it does not change how the solve declares.
- **The least-squares switch at `_LEVIN_DIRECT_SOLVE_PHASE_SPAN` ($20\pi$)** is unchanged.
- **The window's stop rule is unchanged.** A change to a window piece's estimate changes its count,
  which is reported.
- **No integrand provider changes in Part A.** If the chosen noise term needs a declared quantity from
  `ComputeTargets/` or `LiouvilleGreen/`, §9 names the files, and the user rules on that scope at the
  stop.

## 1. The code (Part B)

**Written at §2's stop, as §9's amendment,** from the user's choice. Whatever is chosen, §9 states:
1. **What changes,** in `AdaptiveLevin/levin_quadrature.py` unless §9 says otherwise, with its reference
   implementation (Part A's variant) and its diff against the tree:
   - **the estimate:** what the fallback declares and from which samples, at what cost in evaluations;
   - **the noise term:** its form, and where its input comes from (the leaf's samples, or a quantity the
     caller declares);
   - **the threshold:** its value. If it is not $6\pi$, the constant is renamed for what it is, and
     every reader of it is listed: the gate, Rule A, and any comment that names $6\pi$.
2. **Which leaves it touches:**
   - every Clenshaw–Curtis leaf whose declaration changes, by route: gate leaves, window pieces, guarded
     leaves at the cap, and leaves the chase stop returns;
   - every region whose route changes;
   - so which calls are predicted bit-identical to 07c's (§6 Z1).
3. **Its basis:** §2's measurement, with every constant named and commented, its margin stated, and the
   classes of the census its basis does not reach.
4. **What it does to the declarations and to `converged`.** No leaf declares less than §2's
   measurement shows it can carry, beyond the stated margin. §9 states what follows for:
   - the round-off exit (a leaf at its new floor);
   - the window's count (`_cc_leaf_unresolved` reads the estimate);
   - the chase stop's stopped leaves;
   - floor-limited leaves.
5. **Diagnostics:** counts in the result dict, so that §6 can tell a call the change acts in from one it
   does not. At least: regions whose route changed against the tree's threshold, and Clenshaw–Curtis
   leaves whose declaration the noise term or the new estimate set.
6. **Docstrings and comments** that describe the gate, the fallback's estimate or its floor, rewritten
   to say what is now done and why. The agent lists each with its line.

**As amended at Part A′'s stop (§9.4), the fix is variant V93T10.** The user: *"Go with V93T10, and
put the blind spot in 08"*. Every change is in `AdaptiveLevin/levin_quadrature.py`; no integrand
provider changes. The reference implementation is Part A′'s variant,
`docs/levin-phase-floor/07d/measurement-a2/scripts/variants/levin_quadrature_a2_V93T10.py` (SHA-256
`f85976fc…`), with its diff against the tree, `out/variants/V93T10.diff` (`a2cb1c85…`). Every line
the diff adds is marked `Part A'`. The names below replace the variant's, and its module-level
counter `_A2_C` is replaced by counts carried in the result dicts. The variant's knot machinery
(`_a2_break`, `_a2_knots_in`, `_A2_KNOTS*`, `_A2_GATE_KNOTS`) is inert in V93T10 and is not carried.
Nor is its wrapper of `adaptive_levin_sincos`: the counts are added in `_adaptive_levin`'s result, as
07c's were, and `adaptive_levin_sincos`'s body is unchanged.

### 1.1 The fallback: ${\rm CC}_{4N-3}$ returned, with a nested estimate and a noise term

In `_adaptive_levin_subregion_cc`, with $N$ = `chebyshev_order` ($2N-1 = 47$, $4N-3 = 93$ and
$8N-7 = 185$ at production's $N = 24$):
- **The samples.** The integrand is sampled on the $(8N-7)$-point Lobatto grid of the span, nested: the
  $(2N-1)$-point samples are reused for the $(4N-3)$-point rule, and those for the $(8N-7)$-point rule,
  as the variant does by `grid[::2]`. A span on which the grids do not nest is resampled and counted
  (the variant's `num_a2_resampled`, 0 in all of Part A′'s replays). **Cost:** $8N-7$ evaluations a
  solve, against $2N-1$ at the tree.
- **The value** is ${\rm CC}_{4N-3}$.
- **The noise term** is $\kappa\,s_{\rm own}$, with $s_{\rm own} = \sigma_{\rm own}\sqrt{(n-1)/2}\,
  \sqrt{\sum_j w_j^2}$, $n = 4N-3$ and $w$ the ${\rm CC}_{4N-3}$ weights. $\sigma_{\rm own}$ is the rms
  of the $(4N-3)$-point interpolant's Chebyshev coefficients from the cut that Chebfun's `standardChop`
  finds at tolerance $2^{-52}$, or of its last 8 coefficients if it finds none. It is the variant's
  `_a2_s_own`, `_a2_coeffs` and `_a2_standard_chop`, transcribed from 07d Part A's.
- **The estimate** is $|{\rm CC}_{8N-7} - {\rm CC}_{4N-3}|$ (pair93). `abserr_direct` is the estimate
  plus the noise term.
- **The floors.** `round_off_err` and `round_off_err_less_node` are `_cc_roundoff_floor` on the
  $(4N-3)$-point grid, each plus the noise term. Adding it to the estimate alone understates on 4 of
  Part A's scored leaves (Part A's deviation 9), so it goes on both.
- **On the $(4N-3)$-point grid as well:** the node term ($E$ and the node bound),
  `_declared_endpoint_phase_err`'s term and `resasc`.
- So the leaf declares $\max(|{\rm CC}_{8N-7} - {\rm CC}_{4N-3}|,\ \text{floor}_{4N-3}) +
  \kappa s_{\rm own}$, with the declared phase term as before. ${\rm CC}_N$ and ${\rm CC}_{2N-1}$
  enter no value and no error.
- **Through every caller:** gate leaves, window pieces and guarded leaves at the cap (`force_direct`,
  through `_oriented_cc`, whose reversal of the span is unchanged), and every Clenshaw–Curtis
  sub-solve of a trust check.

**The code reproduces V93T10's outputs bit for bit** (§9.4.5, Z8). A change that would alter any
output relative to V93T10's replays is a stop.

### 1.2 The threshold and the constants

Each is a module constant with a comment stating its basis, its margin and where the basis does not
reach (§9.4.3).
- **`SIX_PI` becomes `CC_GATE_PHASE_VARIATION = 10.0 * np.pi`.** Its one reader in code is the gate in
  `_adaptive_levin_subregion_impl`. Rule A reads no threshold since 07a (§9.1 item 7).
  - **The comments that name it** are at lines 161, 271, 287–296, 542, 2622, 3033, 3368, 3667,
    3806–3808, 4327, 6264 and 6435 of the tree (`d141a80b…`). One that says what 6π did at the time
    (07's Rule A, the 07a variation estimate, the order-sweep measurement at 271) keeps "6π" as the
    number then in force. One that describes the gate now names the new constant.
  - **The tests that name it:** `test_levin_phase_variation.py` lines 169, 171, 202 and 203, and the
    comments of `test_levin_quadrature.py` at lines 111, 305, 314 and 747. The agent reads each and
    says whether it tests the gate (it then reads the new constant) or 07a's variation estimate at
    the number 6π (it then keeps $6\pi$ as a literal, with a comment saying why). Each is listed.
- **`_CC_OWN_NOISE_KAPPA = 4.0`**, the variant's `_A2_KAPPA`.
- **`_CC_STANDARD_CHOP_TOL = 2.0 ** -52`** and **`_CC_OWN_NOISE_TAIL_FALLBACK = 8`**, the variant's
  literals.
- **`_CC_GATE_FORMER_PHASE_VARIATION = 6.0 * np.pi`**, read only by §1.4's route-change count. It is
  commented as such.

### 1.3 What it does to `converged` and to the declarations

The tree's rule stands: a call is converged only if its budget is met, no returned leaf is untrusted,
and no guarded leaf is at the depth cap. The Levin route's declarations, the trust test, the rate
rules, the chase stop and its constants, the windows and their stop rule, and
`_LEVIN_DIRECT_SOLVE_PHASE_SPAN` are unchanged (§0.6). What moves, on Part A′'s 4,220 replays
(`out/tables/t7_scores.md`):
- **The round-off exit** compares a leaf's residual with its new floor, which carries the noise term.
- **The window's count:** `_cc_leaf_unresolved` reads the new `abserr` and the $(4N-3)$-point
  `resasc`. 208 calls have different phase-limited window leaves under every V93 variant, V93 at $6\pi$
  included.
- **The chase stop's stopped leaves** declare as 07c §1.1 says, from the new Clenshaw–Curtis
  declarations:
  - stops fall from 1,644 to 1,546;
  - stopped leaves from 22,822 to 21,192;
  - the sum of their declarations from $4.16\times10^{-19}$ to $9.75\times10^{-21}$.

  The stopped leaf's declaration is `[07c-stopped-leaf-declaration-overstates]`'s, and 08's.
- **Floor-limited leaves** are limited by the new floor.
- **`converged`:** none lost and 7 gained, so 9 calls unconverged become 2.

### 1.4 Diagnostics

The result dict gains three counts over the whole run of the call:
- **`num_cc_solves`:** every Clenshaw–Curtis solve, by any caller (§1.1). This is the variant's
  `num_a2_cc93`.
- **`num_cc_gate_route_changes`:** solves sent to Clenshaw–Curtis by the gate with
  $6\pi \le V <$ `CC_GATE_PHASE_VARIATION`, which the tree would have sent to the Levin solve. This is
  the variant's `num_a2_route_changed`.
- **`num_cc_noise_set`:** Clenshaw–Curtis solves whose $\kappa s_{\rm own}$ exceeds
  $\max(|{\rm CC}_{8N-7} - {\rm CC}_{4N-3}|, \text{floor}_{4N-3})$. It is not in the variant, and it
  changes no value.

**The fix acts in a call iff `num_cc_solves` is positive.** On Part A′'s 4,220 replays, exactly the
3,625 calls with a Clenshaw–Curtis solve differ from the tree. The 595 without one are bit-identical.

### 1.5 Docstrings and comments

Rewritten to say what is now done and why, each listed by the agent with its line:
- `SIX_PI`'s comment block, as `CC_GATE_PHASE_VARIATION`'s, with §9.4.3's basis;
- the gate's comment in `_adaptive_levin_subregion_impl` (line 3033: "SIX_PI is unchanged, and
  unmeasured");
- `_adaptive_levin_subregion_cc`'s docstring and its node-reuse comment;
- `_oriented_cc`'s docstring where it says "the nested Clenshaw-Curtis pair";
- the region record's comment at line 1554 ("the nested-pair estimate $|{\rm CC}_{2N-1}-{\rm CC}_N|$");
- `_DQK21_SATURATION_FACTOR`'s comment (line 742) and the comments at lines 1628 and 3543, on what
  `_cc_leaf_unresolved` reads;
- `_adaptive_levin`'s and `adaptive_levin_sincos`'s docstrings, on the three new counts;
- the module docstring's account of the fallback.

## 2. The measurement — Part A, on the Mac Studio, a planned stop

**Run at the dispatch SHA, whose code is 07c's.** Nothing in the tree changes.
- **Variants** are copies of `levin_quadrature.py` in the agent's `scripts/variants/`, loaded by
  `variant_loader.py`. Each has its diff against the tree saved.
- **Records** go in `var/levin-phase-floor-07d/measurement/`, in `scripts/`, `state/` and `out/`. Every
  output carries the SHA-256 of its script and of the module or variant it ran.
- **Every seed** is written to `state/` before its draw, and every selection is saved before any solve
  or reading.

### 2.1 The calls

1. **A stratified call sample from 07c's census,** drawn so that §2.3's and §2.4's leaf strata can each
   be filled.
   - It covers: group and analytic calls; both rtols; every $k$ band; calls with gate leaves in each
     quarter of $V/6\pi$; calls with Levin leaves at $V$ in $[6\pi, 8\pi)$, $[8\pi, 12\pi)$ and
     $[12\pi, 20\pi)$; calls with window pieces; calls where the chase stop acted; and calls with guarded
     leaves at the cap, if the census has any.
   - Captured from A3 v2's scratch copy by 07a's `make_capture.py`. **This removes 07c's restriction**
     (its deviation 9) of the leaf sample to calls already captured.
2. **07c Part A's 454 calls,** with their captures and references, as the regression set. Every call
   there that is honest at 07c must stay honest, or become unconverged.
3. **R2-28 and R31's record,** each with its reference.
4. **(Rt)'s and (Ra)'s calls** (§2.6), captured as item 1.
5. **Named leaf cases:**
   - 07a's three LiouvilleGreen draw calls (seed 3 `test_YJJ` call 4, seed 8 `test_JJJ` call 4, seed 17
     `test_JJJ` call 11), run from the tests' own construction;
   - the two near-$6\pi$ stopped leaves of 07c's test (b) (07c's Part B log, Observations 4);
   - `1-34599`'s region from 07a's census.

**Every replay at 07c's code reproduces 07c's census record bit for bit:** value, abserr, `converged`,
regions, depth, and the chase stop's three counts. One that does not is a stop. R31's record reproduces
its store values.

### 2.2 The noise (D-N)

On a leaf sample drawn as §2.3's, before any candidate estimate is formed:
- **Its size.** The per-sample scatter $\sigma$ against $|g|$, by two methods: the residual of a
  Chebyshev least-squares fit on a dense grid (07c's `leafcc_07c.py`, at two degrees), and the plateau
  of the fine interpolant's Chebyshev coefficients (where they stop decaying). Each is reported, with
  where the two disagree.
- **Its source.** The same measurement on each component separately:
  - each amplitude $f_i$;
  - the phase as the basis receives it (`theta_mod_2pi`, and the raw phase);
  - and their product.
- **Its structure.** Whether the scatter is independent from node to node:
  - its autocorrelation along the leaf;
  - the spread of ${\rm CC}_{2N-1}$ over grids shifted by fractions of a node spacing, against
    $\sigma\sqrt{\sum w_j^2}$.
- **Against the tables.** The number of amplitude-table knots inside each leaf, from the objects'
  tables (`tools.filon_reference.production_edges`' reading of them), and the scatter against it.
- **Against what is declared.** Whether any declared quantity predicts the scatter: `theta_abserr`
  where supplied, or the tables' own tolerances where the objects record them.

**The table:** $\sigma/|g|$ by class (group label or analytic block, $k$ band, route, knots inside the
leaf), with its source and structure.

### 2.3 The estimate (D-E)

**A leaf sample** from §2.1 item 1's calls:
- **drawn by stratum,** in a seeded order fixed before any leaf is solved;
- **filled leaf by leaf** until the stratum holds **at least 40 scored leaves** or is exhausted;
- **every drawn leaf reported,** scored or not.

**The strata:**
- gate leaves by quarter of $V/6\pi$;
- leaves the chase stop returned;
- window pieces;
- guarded leaves at the cap, if any;
- each of these on group and on analytic calls, at both rtols.

A zero count of understatements on $n$ scored leaves bounds the stratum's rate at about $3/n$ at 95%
confidence. That is the margin a qualifying estimate carries, and §9 states it.

**For each leaf**, by 07c's `leafcc_07c.py` method or a stated extension of it:
- ${\rm CC}_N$, ${\rm CC}_{2N-1}$ and the floor, as the driver forms them;
- **a reference,** with its stated error, and the true error $|{\rm CC}_{2N-1} - \text{ref}|$;
- **each candidate's declaration** $D$ (estimate plus floor), and **its verdict:**
  - **bounded** if $D \ge |{\rm CC}_{2N-1} - \text{ref}| + \epsilon_{\rm ref}$;
  - **understates** if $D < |{\rm CC}_{2N-1} - \text{ref}| - \epsilon_{\rm ref}$;
  - **undecided** otherwise.

  A leaf is **scored** if its verdict is not undecided for the tree's declaration.

**Why the pair understates, before any candidate is scored.** On every leaf where the tree's declaration
understates, in this sample and among 07c's 176 truncation-dominated chase leaves, name the cause by
measurement:
- amplitude-table knots inside the leaf, and the true error's change when the leaf is split at them;
- an error common to ${\rm CC}_N$ and ${\rm CC}_{2N-1}$;
- noise;
- or the reference's own failure, checked by a second reference rule.

If a class of leaves has an error that no estimate from the leaf's own samples can see, the agent
reports it, with its share of the census, and does not hide it in a margin.

**Candidates**, each with its basis and its cost in evaluations:
- **the tree's:** the pair and the floor;
- **the pair, or a coefficient tail, plus a noise term** from §2.2. The noise term is either estimated
  from the leaf's samples or declared by the caller; both forms are scored where §2.2 supports both;
- **a coefficient tail read with its plateau,** the decaying part read as truncation and the plateau as
  noise. The basis is Chebfun's `standardChop` (Aurentz and Trefethen 2017); the agent checks that
  reference's actual rule before adapting it;
- **a third nested rule ${\rm CC}_{4N-3}$ with a noise term,** at $2N-2$ more evaluations;
- others at the agent's choice, each with its basis.

**The table:** for each candidate and stratum:
- the counts bounded, understates and undecided;
- $D/\text{true error}$: its minimum, median and maximum;
- how often the estimate sits at or below the floor;
- the extra evaluations.

**A candidate qualifies** if it understates on no scored leaf of any stratum it applies to. Its
overstatement (the median and maximum ratio) is reported beside it. Overstatement is cost and the
upper pressure on the threshold; it is never a reason by itself to accept an understating estimate.

### 2.4 The threshold (D-T)

**On leaves with $V$ in bands across and above the gate,** each band drawn and filled as §2.3's strata:
$V/6\pi$ in quarters below 1, then $[6\pi, 8\pi)$, $[8\pi, 10\pi)$, $[10\pi, 12\pi)$, $[12\pi, 16\pi)$ and
$[16\pi, 20\pi)$. Each leaf is solved **both ways** on its own span:
- **by Clenshaw–Curtis** at the call's order, with the tree's declaration and each qualifying
  candidate's;
- **by the Levin solve** (`_adaptive_levin_subregion` without the gate), with its own declaration.

Each route is scored against the leaf's reference, as §2.3 scores. Leaves the tree sends to Levin get
Clenshaw–Curtis as a counterfactual, and leaves the tree sends to Clenshaw–Curtis get the Levin solve.
`1-34599`'s region is among them.

**The table**, per band and route:
- the counts bounded, understates and undecided;
- the ratio's minimum, median and maximum;
- the true error against the request a leaf of that width would be given in its call;
- the evaluations.

**For each threshold $T$** on a grid holding at least $4\pi$, $6\pi$, $8\pi$, $10\pi$, $12\pi$ and
$16\pi$, the agent states:
- whether the route each side of $T$ is bounded on every scored leaf of its bands, under the tree's
  estimate and under each qualifying candidate;
- **what Rule A does at $T$:** the phases it would skip or locate that it does not at $6\pi$, counted on
  §2.1's calls.

~~Rule A reads `SIX_PI` by name. A variant that moves the threshold moves Rule A with it, unless it says
otherwise and gives its basis.~~ **Corrected at §2's stop (§9.1 item 7):** at 07c's code Rule A reads no
threshold. 07a replaced it, and `_locate_stationary_points` locates every phase whose rate is not
identically zero. A threshold variant does not move it.

### 2.5 The variants

**At most eight.** Each is a copy of the module with its diff and SHA-256 saved. Each states which
leaves and routes it changes, and shows a call it does not touch bit-identical to the tree.
**Each constant has a stated basis in §2.2–§2.4, with its margin.** Where that basis does not reach a
class of the census, the variant says so.
- **(E) A qualifying estimate with its noise term,** in two scopes:
  - **(E-gate):** on gate leaves only;
  - **(E-all):** on every Clenshaw–Curtis leaf, window pieces, guarded leaves and stopped leaves
    included. A window piece's estimate feeds its window's count, so (E-all) reports every window
    whose stop changes against 07c's.
- **(T) A threshold,** at each value §2.4 finds both routes bounded on, under the estimate it was
  bounded with.
- **(ET) Combinations,** the qualifying estimates with the thresholds bounded under them.
- If no estimate qualifies, **(E) is run with the best candidate, flagged as having no bound basis** (as
  07c's (E) was), so that the tables show what it would do.

Every variant is run on every call of §2.1, at both rtols.

### 2.6 The references

Read once, by 07's V4 method as 07c's Part A read them:
- `tools/filon_reference.py`, FCC at two resolutions;
- QAWO where projected at 30 minutes or less;
- `fccsplit` where a segment holds a stationary point;
- Route A for analytic blocks where it applies.

**The rule for the reference and its stated error is fixed in `state/` before any reading.** A call is
scored if the reference's stated error is at most min(abserr)/100 over the codes and variants scored;
otherwise it is reported, not scored.
- **(Rt) Route-changed calls:** about 30 own-class calls whose route changes under any (T) or (ET)
  variant, drawn with a recorded seed. The strata are the direction (Clenshaw–Curtis to Levin, or the
  reverse), group or analytic, $k$ and rtol.
- **(Ra) Analytic calls the chase stop acts in:** about 30, drawn with a recorded seed from 07c's census
  (`num_chase_stops` positive). The strata are three properties of the stopped leaf carrying most of
  the call's abserr: its route (Clenshaw–Curtis or Levin), its trust at the stop, and its share of the
  abserr. Each is scored at 07c's code from 07c's census, and under every variant.

**The projection comes first,** by 07c's model with its $\gamma D_f$ term for every segment holding a
stationary point, read with 07c's measured margin (its `issues.md` item 9). If Part A projects beyond 36
hours in all, that is a stop. It is reported before anything long runs, with what would be dropped to
fit.

### 2.7 The tables, fixed before the run

1. **(D-N):** §2.2's table.
2. **(D-E):** §2.3's table, and the cause found on every leaf where the tree's declaration understates.
3. **(D-T):** §2.4's table by band and route, and each threshold's verdict.
4. **Honesty:** for the tree and each variant, every referenced call of §2.1 (07c Part A's, R2-28, R31's
   record, (Rt) and (Ra)), scored by 07's rule: converged and honest, converged and dishonest, or
   unconverged (with $|v - {\rm ref}|/{\rm abserr}$).
5. **Convergence, reach and cost.** For each variant, on all of §2.1's calls, against the tree:
   - `converged` changes in each direction;
   - regions and wall;
   - regions whose route changed;
   - window stops changed;
   - the chase stop's counts, and its stopped leaves' declarations;
   - Rule A's changed skips;
   - the calls it touches.

**A variant qualifies** if:
- on table 4, no referenced call is converged and dishonest that is not so at the tree;
- its estimate qualifies on table 2;
- and its threshold is bounded on both sides on table 3.

(Ra)'s verdicts at 07c's code are reported beside the table, not folded into it.

### 2.8 Stop and report

At the end the agent reports, and stops:
- §2.7's tables;
- which variants qualify;
- what D-N, D-E and D-T found;
- the causes of understatement;
- (Ra)'s verdicts.

**It recommends nothing.** The user chooses at the stop, and the orchestrator writes §1, §3 and §9.

## 3. Tests (Part B)

Written with §1 at §2's stop (§9). Whatever the fix, they include:
1. **The estimate, directly,** on constructed integrands with a known ${\rm CC}_{2N-1}$ error:
   - smooth;
   - with a derivative jump inside the span;
   - with a deterministic pseudo-noise of stated size added to the samples.

   The declaration is at or above the true error on each, within its margin.
2. **The noise term,** if it reads a declared quantity: absent, it changes nothing; present, it enters
   as §9 says.
3. **The threshold:** a region just below and just above it goes to the route §9 states; Rule A follows
   as §9 states.
4. **A call the fix must not touch** (no Clenshaw–Curtis leaf and no route change): bit-identical at the
   parent and the commit, its counts zero.
5. **Deliberate breakages,** each caught by the test named for it, re-applied and reverted. At least:
   - the estimate returned without its noise term;
   - the pair restored as the estimate;
   - the threshold set back to $6\pi$, if it moves;
   - the window's count fed the old estimate.
6. **Every changed existing assertion listed,** with why. A test that goes from converged to
   unconverged, or back, is listed with its honesty at both codes against its own expected answer. It is
   reported, not a stop.
7. **The LiouvilleGreen draws** (07c's `docs/levin-phase-floor/07c/code/lg_draws_07c.py`, run at the
   parent and the commit):
   - no test newly failing;
   - every call the fix should not touch bit-identical;
   - every `converged` change listed, with 07a's three calls and 07c's 26 newly unconverged calls named.

**If an acceptance test cannot pass without a change outside §1 as amended,** the agent stops and asks
(§8).

**As amended for V93T10 (§9.4).** In `AdaptiveLevin/tests/`, in a module the agent names. Item 2's
noise term reads no declared quantity: it is read from the leaf's own samples, so (b) replaces it.
- **(a) The estimate, directly (item 1),** on `_adaptive_levin_subregion_cc`, on constructed
  integrands with a closed-form or independently computed integral, at production's $N = 24$ and at one
  other order:
  - smooth;
  - with a derivative jump inside the span;
  - with a deterministic pseudo-noise of stated size $\sigma$ added to the samples, the answer's error
    from it bounded and stated.

  On each, the value equals ${\rm CC}_{4N-3}$ computed independently in the test, and the declaration
  $\max(\text{abserr\_direct},\ \text{round\_off\_err})$ is at or above the true error.
- **(b) The noise term.**
  - On a polynomial integrand that ${\rm CC}_{4N-3}$ integrates exactly, $\kappa s_{\rm own}$ is at
    the rounding level and moves no declaration by more than that.
  - With the pseudo-noise of (a), $s_{\rm own}/(\sigma\sqrt{\sum w_j^2})$ lies in a band the agent
    states with its basis. Part A′'s $\sigma_{\rm own,93}/\hat\sigma$ has 5th percentile 0.838 and median
    1.01 on group leaves.
  - The noise term is in `abserr_direct`, `round_off_err` and `round_off_err_less_node`, each by
    exactly $\kappa s_{\rm own}$.
- **(c) The threshold (item 3).** Regions with $V$ just below and just above
  `CC_GATE_PHASE_VARIATION` go to Clenshaw–Curtis and to the Levin solve. One with
  $6\pi \le V < 10\pi$ is counted in `num_cc_gate_route_changes`; one below $6\pi$ is not.
- **(d) A call the fix must not touch (item 4):** no Clenshaw–Curtis solve anywhere, including every
  trust check's quarters (so every checked leaf has $V \ge 40\pi$), no window and no guard.
  Bit-identical at the parent and the commit, with `num_cc_solves` 0.
- **(e) The window's count.** For a window piece, `_cc_leaf_unresolved` reads the `abserr` and the
  $(4N-3)$-point `resasc` that the solve returned.
- **(f) The counts.** On a call with Clenshaw–Curtis solves, `num_cc_solves` equals the number of
  calls to `_adaptive_levin_subregion_cc`, counted by the test through a wrapper. `num_cc_noise_set`
  counts as §1.4 defines it.

**Breakages,** each recorded with the test that catches it, then reverted:
- **(i)** the noise term removed ($\kappa = 0$): (a)'s pseudo-noise case or (b) must fail;
- **(ii)** the noise term added to `abserr_direct` only, not to the floors: (b) must fail;
- **(iii)** the pair restored as the estimate, $|{\rm CC}_{4N-3} - {\rm CC}_{2N-1}|$: (a)'s jump case
  must fail;
- **(iv)** ${\rm CC}_{2N-1}$ returned in place of ${\rm CC}_{4N-3}$: (a) must fail;
- **(v)** the threshold set back to $6\pi$: (c) must fail;
- **(vi)** the window's count fed the old estimate and the $(2N-1)$-point `resasc`: (e) must fail.

**Existing assertions.** The fix acts on every Clenshaw–Curtis solve, so many existing tests' numbers
move. Each changed assertion is listed with why. A test that goes from converged to unconverged, or
back, is listed with its honesty at both codes against the test's own expected answer; it is
reported, not a stop.

**The LiouvilleGreen draws (item 7)** are 07c's harness, copied to
`docs/levin-phase-floor/07d/code/lg_draws_07d.py`, which records the three counts. Every call with
`num_cc_solves` 0 is bit-identical at the parent and the commit.

## 4. Records of the code (Part B)

- **The log** `logs/07d-six-pi-gate-and-cc-estimate.md`, in README §5.1's form, with §5's and §6's
  predictions (and §9's additions) copied in, marked not yet scored.
- **The board's 07d row.**
- **The LiouvilleGreen draws'** harness and outputs, in `docs/levin-phase-floor/07d/code/`.

**No issue moves in the code commit.** They move in 07d's records commit, after Part C.

## 5. Rehearsal R33 — on the Mac Studio, after the code commit is pushed

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

## 6. The census — on the Mac Studio, Ray-free, on A3 v2's scratch copy

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

## 7. Acceptance

- **Part A:** §2.7's tables complete and its stop reached, with every replay reproducing 07c's census.
- **Part B:** §1 as amended; every test of §3 passing and every breakage caught; the seven roots rising
  only by the tests added; the LiouvilleGreen draws with no new failure.
- **Part C:** §5's S0′–S4 and §6's Z0 hold; Z1–Z8 scored, each a verdict against its rule.

## 8. Stop conditions — stop and ask

- A replay at 07c's code that does not reproduce its census record.
- A capture that does not check against 07c's census.
- Part A's projection beyond 36 hours, before the rest runs.
- **In Part A,** a class of leaves whose error no estimate from its own samples can see, if it carries
  a share of any referenced call's error above that call's abserr: report before scoring the variants.
- **In Part B:**
  - an acceptance test that cannot pass without a change outside §1 as amended;
  - a LiouvilleGreen test failing at the commit that passes at the parent;
  - an output of the code differing from the chosen variant's replay record on a call the review
    replays.
- **In Part C:**
  - S0′ failing;
  - the census's projection beyond 48 hours;
  - a call Z1 says is untouched differing from 07c's, or Z8 failing, before anything else is scored;
  - a lock held.

## 9. Amendments at §2's stop

*Written by the orchestrator at §2.8's stop, from the user's words, quoted.*

### 9.1 At Part A's stop, 2026-11-22: a second measurement first

**The user's words.** At the stop, after the orchestrator's account of the results:
- *"We are going to want to do that measurement, but first some more things to consider:"*
  - *"if we are going to estimate the CC quadrature on the CC93 rule, we should check whether it is
    better to return that value rather than CC47"*;
  - *"can we determine the location of what you're calling the "knots" so that we bisect the intervals at
    a knot rather than elsewhere? is that a better strategy?"*;
  - *"if the region is small enough that bisection is not gaining much, it seems better to pass the
    whole thing to CC. Does this argue that the 6pi gate is in fact somewhat too small?"*
- On the orchestrator's five-item scope: *"I'd like all 5 in the amendment, unless there is a reason
  not to."*

So no estimate or threshold is chosen yet. §1 and §3 stay unwritten until Part A′'s stop.

**What Part A found, and what the orchestrator's checks added.** Part A's records are in
`docs/levin-phase-floor/07d/measurement/`, with the orchestrator's check scripts and their outputs in
its `orchestrator_check/`.
1. **D-E stands.** nest3N4 — $|{\rm CC}_{4N-3} - {\rm CC}_{2N-1}|$ plus $4 s_{\rm own}$, where
   $s_{\rm own} = \sigma_{\rm own}\sqrt{\sum w_j^2}$ and $\sigma_{\rm own}$ is read from the plateau of
   the leaf's own 47 Chebyshev coefficients (`state/candidates_07d.json`) — understates on no scored
   leaf. The tree's pair understates on 114 of 928: 68 by knots, 46 by noise.
2. **The leaf reference understates its error on knotted leaves.** The orchestrator's composite
   Gauss–Legendre reference, split at the knots and taken at 8 to 256 panels a piece, disagrees with
   Part A's v3 reference by more than $\epsilon_{\rm ref}$ on leaves holding one listed knot, by up to
   100× on band leaves. On `band_10_12|group|1e-7` #3 it moves by $4\times10^{-41}$ between 16 and 64
   panels, far above the noise, and settles by 256. v3's 32-panel check cannot see such structure. What
   the structure is has not been established.
3. **D-T's Levin route is not the tree's.** As §2.4 said, the Levin solve ran without the gate, and
   **so did its halves and quarters** (`leafcc_07d.py`). In the tree, `_check_leaf` solves them through
   the gate. A $6.4\pi$ leaf's halves ($3.2\pi$) and quarters ($1.6\pi$) go to Clenshaw–Curtis. So
   table 3's residual is the tree's only where the halves reach $6\pi$ (about $V \ge 12\pi$), and its
   trust check only where the quarters do (about $V \ge 24\pi$).
   The fault is §2.4's wording, not the agent's.
4. **Table 3's `levin_tc` column leaves out the floor.** The tree declares
   $\max(\delta_1/(1-r), \text{phase\_err}) + \text{truncation}$ on a trusted Levin leaf (`total_err`).
   The column used $\delta_1/(1-r) + \text{truncation}$.
5. **Against the orchestrator's reference, 14 of the 37 trusted, or $r<1$, band leaves that table 3's
   `levin_tc` counts as understating still understate** (rows, some of them one leaf drawn in two
   strata). Items 3 and 4 account for all of them:
   - **Six analytic leaves, $11.6\pi$ to $18.6\pi$:** each is bounded by its floor, at 7 to 31 times its
     true error.
   - **Three group leaves, at $6.4\pi$ to $7.5\pi$, with 5 to 7 knots:** here the Levin halves and
     quarters share one error. On the first, $V_P$ is out by $1.45\times10^{-35}$, the halves by
     $7.8\times10^{-36}$ and the quarters by $7.9\times10^{-36}$, so $r = 0.011$. ${\rm CC}_{47}$ on
     the whole leaf is out by $5.4\times10^{-37}$. The tree would not solve these halves by Levin.

   So D-T does not show the tree's Levin route understating, and it does not show that it is honest.
   At the call level, table 4 at $6\pi$ has 410 calls honest; the 4 dishonest are R2-28's.
6. **The orchestrator's reading of Part A's leaf records**, against v3's references
   (`orchestrator_check/mycc93.*` and `mylevknots.*`):
   - **${\rm CC}_{93}$ is the better value.** On leaves where the reference can tell, it is closer than
     ${\rm CC}_{47}$ on 448 against 14 group leaves with 5+ knots, 144 against 19 with 1–4, 96 against
     31 with none, and 299 against 22 analytic. The median gain is 16×, 2.5× and 1.7× on those group
     classes, and $1.6\times10^2$ to $4\times10^5$ on analytic leaves.
   - **nest3N4 bounds ${\rm CC}_{93}$'s error with no understatement, and loosely.** The median
     $D/\text{true}$ is 6 and 9 on group leaves with no knots and with 1–4, $1.8\times10^3$ with 5+, and
     $4\times10^3$ to $3\times10^{11}$ on analytic leaves. That looseness is the likely cause of ET20's
     cost: 1.68× the regions, 2.67× the wall and 74 calls unconverged.
   - **On group leaves, the knots decide between the routes:**

     | knots in leaf | leaves | $V_P$ worse than ${\rm CC}_{93}$ | $V_P$ better |
     |---|---|---|---|
     | none | 20 | 0 | 6 |
     | 1–4 | 209 | 110 | 13 |
     | 5+ | 335 | 332 | 1 |

     Above the gate the median is 5 to 18 knots a group leaf by band (maximum 183).
   - **On analytic leaves, $V_P$ beats ${\rm CC}_{47}$ above $6\pi$.** ${\rm CC}_{47}$ meets the
     leaf's share of the request on only 10 of 84 leaves at $16$–$20\pi$. ${\rm CC}_{93}$ meets it on
     every leaf to $20\pi$, within about 2× of $V_P$.
7. **Rule A reads no threshold** at 07c's code (Part A's deviation 2). §2.4 is corrected in place, and
   §2.7's Rule A column is dropped.

### 9.2 Part A′ — the measurement

**Run at the amendment's SHA, whose code is 07c's,** in a new worktree, with records in
`var/levin-phase-floor-07d/measurement-a2/`. Otherwise it runs as §2:
- variants in `scripts/variants/`;
- every output carrying the SHA-256 of its script and module;
- every seed written before its draw, and every selection before any solve.

**Part A's records are its inputs, read-only:** its captures, selections, leaf records, call references
and variant replays. Nothing in `var/levin-phase-floor-07d/measurement/` is written. **No call is
re-captured, and no call reference is re-read,** except as item 6 says.

**0. The leaf references, revised before anything is scored.** v4:
- **The method:** composite Gauss–Legendre on the leaf split at its knots (`production_edges`; the whole
  leaf where none are known), at a stated order.
- **Its stated error:** the panel count a piece is doubled from 8 until two successive doublings
  agree within 3× their combined noise, at most 512 a piece. $\epsilon_{\rm ref}$ is the last
  doubling's change plus twice the noise term. A leaf that does not settle is reported, not scored.
- **The rule is fixed in `state/` before any reading.**

Every leaf of Part A's D-E and D-T samples is re-referenced by v4, and so are item 3's new band and
item 1's leaves. Part A's tables 2 and 3 are re-scored on v4, with every verdict that changes from v3
tabulated, and nest3N4's qualification re-stated. v3's numbers are kept beside v4's, not replaced.

**1. The threshold, with the Levin route as the tree runs it (the corrected D-T).** For each $T$ on the
grid $4\pi, 6\pi, 8\pi, 10\pi, 12\pi, 16\pi, 20\pi$, on every band leaf, Part A's and item 3's:
- **On the Levin side ($V \ge T$),** the leaf is solved as `_check_leaf` would check it under a gate at
  $T$:
  - $V_P$ by the Levin solve, and its halves and quarters each by the route the gate at $T$ sends them;
  - $\delta_1$, $\delta_2$, $r$, the noise yardstick (`_trust_noise_yardstick`) and the rate rules
    (`_leaf_rate_rule`), each by the tree's own function;
  - the tree's trust verdict, and on a trusted leaf the value it returns ($Q$) with the declaration
    `total_err` forms: $\max(\delta_1/(1-r), \text{phase\_err}) + \text{truncation}$.

  **A trusted leaf is scored.** An untrusted one is counted, with $r$ and $\delta_1$, as a leaf the tree
  would bisect, and is not scored as a declaration. The depth-cap form is reported beside, not scored.
- **The reproduction is checked against the tree** before it is used. An instrumented variant at
  $6\pi$ records every check on Part A's replayed calls ($r$, $\delta_1$, $\delta_2$, $\nu_B$, trusted).
  The reproduction equals it bit for bit on at least 200 checked leaves, group and analytic.
- **On the Clenshaw–Curtis side ($V < T$),** ${\rm CC}_{47}$ under the tree's declaration and nest3N4,
  and ${\rm CC}_{93}$ under each declaration of item 2.
- **The Clenshaw–Curtis sub-solves inside a Levin check** take the value the variant returns:
  ${\rm CC}_{47}$ under the tree's fallback, ${\rm CC}_{93}$ under item 2's.
- **For each $T$, the verdict:** whether each side is bounded on every scored leaf of its bands, by
  call kind (group or analytic), under the tree's fallback and under each qualifying item-2 declaration.
  **A side with no scored leaf is "vacuous", not "bounded"**, and is never a reason to choose $T$.

**2. ${\rm CC}_{93}$ as the returned value.** On every leaf of Part A's D-E and D-T samples, and item 3's:
- ${\rm CC}_{47}$'s and ${\rm CC}_{93}$'s true errors against v4, with which is closer where v4 can tell.
- **Candidate declarations of ${\rm CC}_{93}$'s own error,** each with its basis and its cost in
  evaluations beyond the 93:
  - nest3N4 as it stands (no cost);
  - $|{\rm CC}_{185} - {\rm CC}_{93}|$ plus $\kappa s_{\rm own,93}$ (92 evaluations; Part A recorded
    ${\rm CC}_{185}$ on every leaf);
  - a coefficient tail of the 93-point interpolant read with its plateau, as Part A's chopN4 (no cost);
  - others at the agent's choice.

  $s_{\rm own,93}$ is read from the 93 samples as Part A read $s_{\rm own}$ from 47. $\kappa$ is set
  by D-N's measurement, as Part A set it, with its basis.
- **Scored and qualified as §2.3,** against ${\rm CC}_{93}$'s true error, by stratum.
- **Each candidate's overstatement,** median and maximum $D/\text{true}$, beside its verdict.

**3. The routes by call kind and knots.** Items 1 and 2's tables are given split by group and
analytic, and by knots in the leaf (none, 1–4, 5+), with:
- $V_P$'s, $Q$'s, ${\rm CC}_{47}$'s and ${\rm CC}_{93}$'s true errors against the leaf's share of
  the request;
- which route is the more accurate, where v4 can tell.

**A new band, $[20\pi, 32\pi)$,** is drawn and filled as §2.4's, with a seed written first, so that
$T = 20\pi$ is tested on both sides.

The agent states whether a gate that reads the knots as well as $V$ is supported, and on what. **No
production code knows the knots.** `production_edges` is test tooling, which walks the integrand's
closures and fits `QuadSourceIntegral`'s layout only. A gate or a bisection that reads them needs the
integrand provider to declare them. §0.6's last bullet applies: the user rules on that scope at the
stop.

**4. The knots by type.** `production_edges` merges five kinds of knot:
- each phase spline's knots;
- the Green's function amplitude spline's knots;
- the cumulative tables' nodes;
- their nearest-node switch midpoints;
- their declared breaks.

For each kind, on the group leaves of items 1 and 2:
- **its count a leaf,** by band;
- **its continuity, measured:** the jump at the knot in the value and in the first three derivatives of
  each ingredient it belongs to, from one-sided evaluations or fits, against the noise;
- **the error it carries:** the change in ${\rm CC}_{47}$'s, ${\rm CC}_{93}$'s and $V_P$'s true error
  when the leaf is split at that kind's knots alone.

**And a bisection that uses them,** run as a variant (item 6's (K)):
- a leaf holding a knot in the middle half of its span is bisected at the knot nearest its midpoint;
- every other leaf is bisected at its midpoint, as the tree does;
- the knots are supplied by the harness, from `production_edges`, restricted to the kinds the
  measurement above finds to carry error, with that restriction's basis;
- on calls where `production_edges` does not apply, the variant bisects as the tree does.

**5. The cost of each route, at each $T$.**
- **Per leaf, by band:** evaluations and wall for:
  - Clenshaw–Curtis, at 47, 93 and 185 points;
  - the Levin solve alone;
  - the Levin leaf as the tree checks it (its halves and quarters by their routes at $T$).
- **Per call:** item 6's table, from the variants.

**6. The variants, at most eight,** each run on every call of §2.1 at both rtols, as §2.5:
- **(V93):** ${\rm CC}_{93}$ returned on every Clenshaw–Curtis leaf, with item 2's best qualifying
  declaration, gate at $6\pi$. If none qualifies, the best candidate is used, flagged as having no
  bound basis.
- **(V93-T):** (V93) at each $T$ item 1 finds bounded on both sides under it, at most three values.
- **(K):** item 4's bisection on (V93), gate at $6\pi$.
- **(KT):** (K) at the best (V93-T) threshold, if any.
- **(G):** a gate that reads the knots, if item 3 supports one, with its basis. The harness supplies
  the knots.
- **Others at the agent's choice,** up to eight in all, each with its basis.

**New call references** only where a variant's route-changed calls fall in an (Rt) stratum with no call:
- at most 30, drawn with a seed written first;
- read by §2.6's rule, projected first.

Part A's Ea, Eg, ET12, ET20, Ep and T20 replays are carried over for comparison, not re-run.

**The projection comes first:** item 0's references, item 3's new band and item 6's runs. Beyond 36
hours in all is a stop, reported with what would be dropped to fit.

### 9.3 Part A′'s tables and stop

**The tables, fixed before the run:**
1. **v4 against v3:** every leaf whose verdict changes, by table and stratum.
2. **(D-E on v4):** §2.3's table re-scored, for ${\rm CC}_{47}$'s candidates.
3. **(${\rm CC}_{93}$):** item 2's table, and ${\rm CC}_{93}$ against ${\rm CC}_{47}$.
4. **(D-T as the tree runs it):** item 1's table, by band, route, call kind, knots and $T$, with each
   $T$'s verdict.
5. **(Knots):** item 4's table by kind.
6. **(Cost):** item 5's per-leaf table.
7. **Honesty and convergence:** §2.7's tables 4 and 5 for item 6's variants, beside Part A's.

**A variant qualifies** if:
- on table 7, no referenced call is converged and dishonest that is not so at the tree;
- its declaration qualifies on table 2 or 3, against the value it returns;
- its threshold is bounded, not vacuous, on both sides on table 4.

**Stop and report,** as §2.8: the tables, which variants qualify, what each item found, and the
causes of any understatement. **It recommends nothing.** The user chooses at the stop, and the
orchestrator writes §1, §3 and §9.4.

**Stop conditions,** beside §8's:
- the reproduction of `_check_leaf` failing its check against the tree;
- more than a tenth of a stratum's leaves left unsettled by v4.

### 9.4 At Part A′'s stop, 2026-11-24: the fix is V93T10

*Written by the orchestrator on 2026-11-24, at Part A′'s stop and after its addendum, from the user's
words, quoted.* Part A′'s records, with the addendum's, are in
`docs/levin-phase-floor/07d/measurement-a2/`. The tables of record are `out/tables/` (t1–t7,
`kappa93.md`, `understate_ref2.md`) and the addendum's `out/tables_add.md`, `tables_add_supp.md`,
`dissect_b.md` and `edge_vp_add.json`. What stays on the Mac Studio is listed by SHA-256: every
variant's replays in `docs/levin-phase-floor/07d/measurement-a2/KEPT_REPLAYS_ON_MAC_STUDIO/`, and the
files over 2 MB in its `KEPT_ON_MAC_STUDIO.txt`. The four review calls' V93T10 replay records are copied
(§9.4.5). The orchestrator's checks are in its `orchestrator_check/`.

#### 9.4.1 The rulings

- **At Part A′'s stop (2026-11-23).** V93T8 and V93T10 qualify under §9.3; V93, V93tail, V93T12, K,
  KT and G do not. The user: *"I can't see a reason to believe that the `8pi`, `10pi` and `12pi`
  windows will be always safe, while `6pi`, `16pi` and `20pi` have understated errors. I think probably
  this is just an artefact of which references we are looking at. Unless you can propose a reason to
  think otherwise, my guess is that we have to accept a small number of understated errors, but
  hopefully these don't impact the science outputs too significantly. Can you propose a rationale for
  `V93T8` vs `V93T10`? It seems unnecessary to include both. My guess is `V93T10` is everything is
  otherwise equal, since there seems some merit in sending slightly larger regions to CC. For now I'm
  happy to do a short run to settle the near-edge leaves."* Part A′'s agent ran the addendum: (A) the
  near-edge leaves against a 48-point Gauss–Legendre reference and a smooth-phase reference, and (B)
  the dissection of V93T12's new dishonest call.
- **After the addendum.** Asked *"does that now mean `12pi` is an even better choice than `10pi`?"*,
  the orchestrator answered no (§9.4.3). The user: *"Go with V93T10, and put the blind spot in 08"*.

#### 9.4.2 What Part A′ measured

**Checks.** The orchestrator's §A′.3 checks (`orchestrator_check/`) agree with the agent:
- on four leaves, its own 32-point Gauss–Legendre reference against v4 and the agent's 48-point one;
- the D-T reproduction at $8\pi$ against table 4;
- the ${\rm CC}_{93}$ candidates' understatement counts;
- the knot counts and (K)'s bisection points;
- table 7, re-scored by its own reader.

Every Part A′ run exited 0.

**The references.** v4 understates its own error where $\hat\sigma$ is large: it settles one doubling
early. On the four leaves that made table 4 fail at $4\pi$, $6\pi$, $16\pi$ and $20\pi$, v4 and the
48-point reference disagree by 1.17–1.28 $\epsilon_{\rm v4}$. Against the 48-point reference all four
are bounded or undecided at every $T$.

**The estimate.** pair93 understates on no settled leaf. Under its own 93-point floor it is bounded on
3,636 of 3,642 and undecided on 6. Its median $D/\text{true}$ is 5.85 on the Clenshaw–Curtis strata's
group leaves and 34.8 on the band's. The tree's ${\rm CC}_{47}$ declaration understates on 74 of 899
leaves: 55 by knots, 19 by an error common to ${\rm CC}_{24}$ and ${\rm CC}_{47}$.

**The threshold (table 4, under the ${\rm CC}_{93}$ fallback and pair93, at $10\pi$):**

| side | leaves scored | understate |
|---|---|---|
| group, Levin | 474 | 0 |
| group, Clenshaw–Curtis | 1,040 | 0 |
| analytic, Levin | 118 | 0 |
| analytic, Clenshaw–Curtis | 575 | 0 |

**The variants (table 7; 417 referenced and scored calls; 4,220 replayed):**

| variant | honest / dishonest / unconverged | qualifies | regions / tree | wall / tree |
|---|---|---|---|---|
| tree | 410 / 4 / 3 | — | 1 | 1 |
| V93T8 | 412 / 4 / 1 | yes | 0.921 | 2.23 |
| **V93T10** | **412 / 4 / 1** | **yes** | **0.907** | **2.32** |
| V93, V93tail | 412 / 4 / 1 | no: (c) at $6\pi$, on v4's error | 0.912, 0.992 | 2.03, 1.52 |
| V93T12 | 411 / 5 / 1 | no: (a), `R:a:W2-0-12977-5` at 2.79 | 0.888 | 2.27 |
| K, KT | 413 / 3 / 1 | no: (a), three calls | 0.789, 0.791 | 1.57, 1.68 |
| G | 416 / 0 / 1 | no: (c) at $6\pi$ | 4.519 | 10.35 |

The four dishonest calls under the tree and the V93 family are R2-28's, at 1.14 under the tree and
1.10 under V93T10. Under V93T10, 37,574 regions in 3,158 calls change route.

**The addendum.**
- **(A) The near-edge leaves.** Under the ${\rm CC}_{93}$ fallback, the returned leaves are bounded,
  undecided or understating as follows:

  | $T$ | bounded | undecided | understate |
  |---|---|---|---|
  | $6\pi$ | 53 | 14 | 0 |
  | $8\pi$ | 41 | 2 | 0 |
  | $10\pi$ | 42 | 3 | 0 |
  | $12\pi$ | 21 | 1 | 0 |

  Under the ${\rm CC}_{47}$ fallback, `band_8_10|group|1e-8#23` understates at 0.51.

  **On 49 trusted near-edge checks** with $D/\text{true}$ in $[0.95, 1.05]$:
  - the returned $Q$ carries the halves' error, and $D \approx \delta_1$ follows it by construction;
  - $V_P$ is about 35× more accurate than $Q$.
- **(B) `R:a:W2-0-12977-5` under V93T12.** A region with $V = 59.4\pi$ at depth 4 is trusted once one
  of its quarters ($V = 10.26\pi$) goes to ${\rm CC}_{93}$. It declares $4.46\times10^{-32}$ against a
  true error of $3.62\times10^{-31}$.
  - $V_P$, the halves and the three Levin quarters share an error of $-3.6$ to
    $-4.0\times10^{-31}$.
  - At $6$–$10\pi$, that quarter's own Levin error ($+2.13\times10^{-31}$) made the check refuse the
    region.
  - Three Levin regions at $662$–$1{,}530\pi$ in the same call understate under every code, about
    $10^5$ below the call's budget.

#### 9.4.3 The basis, and where it does not reach

- **`CC_GATE_PHASE_VARIATION` $= 10\pi$.** It is bounded on both sides of table 4 for both call kinds,
  as the table above shows, and it qualifies on table 7.
  - **Between the two that qualify,** $10\pi$ sends larger regions to Clenshaw–Curtis, which the user
    preferred all else equal. The measured costs are within 4% of each other (wall 2.32 against
    2.23).
  - **Against $12\pi$, which is not better:** raising the gate changes which sub-solves of a large
    region's trust check go to Clenshaw–Curtis. That can remove the dissent that made the check refuse
    a region whose Levin solves share an error. It happened twice at $12\pi$:
    - the W2 call (§9.4.2 (B));
    - `band_20_32v|group|1e-7#43` ($V = 21.7\pi$, visited), refused at $6$–$10\pi$ with $r = 3.83$,
      trusted at $12\pi$ with $r = 0.135$ once its halves go to ${\rm CC}_{93}$, and 0.76 of its true
      error.

    Neither happens at $10\pi$. This is two events, so the evidence is weak, but it is against
    $12\pi$.
- **`_CC_OWN_NOISE_KAPPA` $= 4$.** It is the smallest of $\{3, 4, 5, 6, 8\}$ with $\kappa$ times the
  5th percentile of $\sigma_{\rm own,93}/\hat\sigma$ on chopped group leaves at least 2.8, Part A's rule
  (`kappa93.md`). That percentile is 0.838 over 1,089 group leaves, with median 1.01. On 909 analytic
  leaves it is 1.05, with median 1.81.
- **The estimate.** pair93 has no understatement on any settled leaf, at a median overstatement of 5.85
  on the Clenshaw–Curtis strata's group leaves. It costs $4N-4$ evaluations a solve beyond
  ${\rm CC}_{4N-3}$.
- **Where the basis does not reach,** on the user's reading (*"we have to accept a small number of
  understated errors"*):
  - **The trust check's blind spot.** When $V_P$, the halves and the quarters share an error, the check
    passes and the leaf declares about $\delta_1$, below its error. $Q$ is then returned where $V_P$ is
    the better value. The fix does not change the Levin route's declarations (§0.6). The blind spot is
    08's (§9.4.6), and R2-28 stays dishonest under V93T10 (§9.4.5).
  - **The near-edge leaves.** There, $D \approx$ the true error by construction, and whether
    $D \ge$ the true error rests on the $(1-r)$ factor. 3 of the 45 returned at $10\pi$ are
    undecided.
  - **`band_20_32v|group|1e-7#79`,** visited, not returned, understates at $8\pi$ under the
    ${\rm CC}_{47}$ fallback only, at 0.91. Under ${\rm CC}_{93}$ it is untrusted ($r = 7.3$).
  - **High-$V$ Levin leaves** ($662$–$1{,}530\pi$) on one call understate under every code, about $10^5$
    below budget.
  - **The knots.** No production code knows them, and V93T10 reads none. Its estimate bounds
    ${\rm CC}_{93}$'s error on knotted leaves as measured. The knot readers miss the sine component of
    three-Bessel calls (706 leaves).

#### 9.4.4 The census's cost

07c's census took 43,080 s (12.0 h). On Part A′'s 4,220 calls V93T10 takes 0.907 of the tree's
regions and 2.32 of its wall, since each Clenshaw–Curtis solve evaluates $8N-7$ points against
$2N-1$. If the census scales as the sample does, it takes about 28 h. **§6's harness projects it** from
Z0's K0 run at the new code. Beyond 48 hours is a stop, as §6 says.

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

#### 9.4.6 The issues

- **On the user's word** (*"put the blind spot in 08"*),
  `[07c-r2-28-trusted-levin-leaf-hides-a-common-error]` is narrowed and **assigned to 08**. Part A′'s
  evidence:
  - the W2 region of §9.4.2 (B);
  - `#43` at $12\pi$;
  - the 49 near-edge checks on which $Q$ carries the halves' error while $V_P$ is about 35× better.

  The addendum's `issues.md` items 1 and 2 are this issue's.
- **`[07-c2-gate-threshold-was-never-measured]`, `[06d-cc-nested-estimate-measures-the-coarse-rule]`
  and `[07c-integrand-noise-above-the-cc-floor]`** close or narrow in 07d's records commit, on the
  user's word, as the header says.
- **The others** were put to the user at this amendment, and opened with it. The user: *"I am happy
  with your proposed rulings, so open them in this commit"*.
  - **Opened, ten:**
    - from Part A: `[07d-levin-own-declaration-understates-below-20pi]` (item 1),
      `[07d-cc-fallback-does-not-split-at-table-knots]` (item 5),
      `[07d-analytic-phase-steps-at-its-2pi-wraps]` (item 6) and
      `[07d-route-a-qawo-cost-scales-with-k-and-span]` (item 7);
    - from Part A′: `[07d-f0-steps-where-the-nearest-table-node-switches]` (item 1),
      `[07d-knot-readers-miss-the-three-bessel-sine-component]` (item 2),
      `[07d-leaf-reference-understates-its-own-error]` (item 3, with the addendum's item 5 and the
      orchestrator's v4 observation), `[07d-levin-declaration-at-making-understates-on-knotted-leaves]`
      (item 6) and `[07d-trusted-q-beats-cc93-on-knotted-group-leaves]` (item 7, **assigned to 08**
      with the blind spot);
    - from the addendum: `[07d-high-v-levin-leaves-understate-far-below-budget]` (item 4, with the
      orchestrator's high-$V$ observation).
  - **Folded into existing issues as evidence:**
    - Part A's item 2 and Part A′'s item 4 into `[06d-cc-nested-estimate-measures-the-coarse-rule]`;
    - Part A's items 3 and 4 into `[07c-integrand-noise-above-the-cc-floor]`;
    - Part A′'s item 5 and the addendum's item 3 into R2-28's.
  - **Dropped:** Part A's item 8 (a method note), Part A's item 9 (corrected in §9.1) and Part A′'s
    item 8 (variants not chosen).

#### 9.4.7 Stops added (into §8)

- **In Part B:** an output of the code differing from V93T10's replay record on a call the review
  replays.
- **In Part C:** Z8 failing, before anything else is scored.
