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

**Where it runs:** in three parts, as 07b and 07c did.
- **Part A, §2, the measurement:** on the Mac Studio, Ray-free, with no lock, in a worktree at the
  dispatch SHA, whose code is 07c's (`67b5af74`). It measures the integrand's noise, the
  Clenshaw–Curtis estimate against the true error, and both routes against the phase variation. It
  scores candidate estimates and thresholds, and takes two new reference samples. **It ends at a
  planned stop.** There the user chooses the estimate, its noise term and the threshold with the tables
  in front of them. The orchestrator writes the choice into §1, §3 and §9 as an amendment, committed and
  pushed before Part B.
- **Part B, §1, §3 and §4, the code:** on the laptop, by a fresh-context subagent, in the main checkout
  at `HEAD`. One commit.
- **Part C, §5 and §6:** on the Mac Studio, after the code commit is pushed. **§5 is rehearsal R33,**
  under the lock, against 07c's R32. **§6 is the census,** Ray-free, on 06's certified scratch copy of
  A3 v2, with 07c's census as the "before". **This census is 08's "before".**

The records of Parts A and C are committed on the laptop afterwards. Part A's go in with §9's
amendment, and Part C's in 07d's records commit.

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

Rule A reads `SIX_PI` by name. A variant that moves the threshold moves Rule A with it, unless it says
otherwise and gives its basis.

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

*Written by the orchestrator at §2.8's stop, from the user's words, quoted.* Empty until then.
