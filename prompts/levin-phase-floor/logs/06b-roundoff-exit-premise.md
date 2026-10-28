# levin-phase-floor 06b — the round-off-limited exit checks its premise

> **§1–§3 only** (the code, the tests and these records), written by the executing agent on the
> laptop. §4 (rehearsals R15 and R16) and §5 (the census) run later on the Mac Studio. The
> orchestrator appends their record to this log.

**Subject:** prompt 06b §1–§3, on the laptop, in the main checkout
`/Users/ds283/Documents/Code/SecondaryGWKit`, branch `handover-remedial`.
- **Dispatch SHA:** `da5c24d`. Nothing landed on the branch while the work ran
  (`git log --oneline da5c24d..HEAD` was empty before staging). The stash
  (`stash@{0}`, 07 Part A's §1) was not touched, and `var/levin-phase-floor-07/` was not read.
- **The gate.** `git status --short` was empty at dispatch. `runregistrykit list` showed nothing
  `running` or stale: 61 runs, 59 finished, 2 `unknown` (pre-registry). The seven roots'
  baselines are the orchestrator's, taken at `93d5cb1`, whose code is `da5c24d`'s (deviation 1).
- No Ray was started, no run was registered or launched, no store was opened, and nothing was
  written under `var/`.

**Commit:** `4ddc3c2`, the code commit, pushed 2026-10-28 after the orchestrator's review passed all eight checks (orchestrator note `prompt-06b.md` §A.3, recorded in its §B.5).

**Result:**
- **§1.** `_adaptive_levin`'s round-off-limited exit fires only if the region's two halves' floors
  sum to at least $\rho = \tfrac12$ of its own (`_LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO`), checked
  after every existing condition holds, on both branches. On the Levin branch the halves are
  `dataL`/`dataR`; on the fallback branch they are solved on demand and, if the region is then
  bisected, carried forward as the children's estimates. A test-only keyword
  `_test_only_bypass_roundoff_premise` restores the old test.
- **§2.** `AdaptiveLevin/tests/test_levin_roundoff_exit.py`, 6 methods, all passing: tests 1–4 as
  the prompt writes them, plus two small contract tests. Test 1's parent accepts
  $\theta = 100e^x$ on $[0, 20]$ at depth 0 through the exit with a relative error of
  $7.5\times10^{-7}$ against `rtol` $10^{-7}$; the new code bisects once and converges at
  $1.7\times10^{-9}$. Test 3 was built (no fallback for breakage (iii) needed).
- **The breakages.** (i) is caught by test 1 (and test 3); (ii) by test 2 (and tests 1 and 3);
  (iii) by test 3 alone. Each was reverted, and the file's SHA-256 checked back.
- **Test 5.** 160 units (20 seeds × 2 modes × 4 tests): every test passes at the parent and under
  the new code. 1,026 of 4,320 Levin calls had a phase-limited region at the parent and 133 under
  the new code; every call that moved is one of the 1,026, and every other call is bit-identical.
  **One call reaches `depth_max` under the new code** (seed 15, `test_JJJ`, J110, group 3: depth
  5 → 20, 23 → 2,295 solves, `abserr` $2.636\times10^{-10} \to 2.634\times10^{-10}$, unconverged
  in both). It is a finding, opened as `[06b-premise-check-can-chase-a-flat-residual-to-depth-max]`.
- **The seven roots.** Six equal the baseline; `AdaptiveLevin` rose by 6, the new module's 6
  methods.
- **No stop condition of §7 arose.**

## What shipped

**`AdaptiveLevin/levin_quadrature.py`.**
- **`_LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO = 0.5`**, after `_LEVIN_ROUNDOFF_G0_NOISE_FLOOR`, with
  §0.2's argument in its comment: why the premise can fail (the $G_1/G_0$ term, roughly
  square-rooted by halving when $\theta'$ is exponential), and why $\tfrac12$ — genuine round-off
  does not fall under bisection and grows where $k^2$ dominates; the halves' sum is
  $(1 + 2\sqrt R)/(1 + 2R)$ of the parent's for $R = G_1/G_0$ with $G_1 > k^2$, 0.64 at $R = 3$
  and 0.35 at $R = 10$; $\rho = 1$ would never fire while $R > 1$.
- **`_roundoff_exit_premise_holds(phase_err, phase_err_L, phase_err_R)`**, after
  `_declared_endpoint_phase_err`: `phase_err_L + phase_err_R >= rho * phase_err`. Its docstring
  states §1 item 6: a half's `+inf` makes the sum `+inf`, the premise holds and the exit may fire,
  today's behaviour, since a finite parent floor is still required.
- **`_adaptive_levin(..., _test_only_bypass_roundoff_premise=False)`.** A private keyword, last in
  the signature, documented as test-only in a comment at the top of the body; read only by the two
  `if phase_limited and not _test_only_bypass_roundoff_premise:` tests. `adaptive_levin_sincos`
  does not expose it (asserted by `test_bypass_is_not_exposed`).
- **The Levin branch.** After `phase_limited = phase_limited and not resolved`:
  `phase_limited = _roundoff_exit_premise_holds(phase_err, dataL["phase_err"], dataR["phase_err"])`
  (each read as `.get("phase_err", 0.0) or 0.0`, the pattern the parent's own read uses). No
  evaluation is added.
- **The fallback branch.** After its `phase_limited = phase_limited and not resolved`, and only if
  that holds: both halves are solved with `_adaptive_levin_subregion`, exactly the call that
  would produce each half when popped as a region, with the run's shared `vectorize_cache`;
  `num_evaluations += 2`, and their solve counts enter `num_solves_*`. Then the premise is
  checked. If the region is bisected, the halves are carried forward as the children's
  `estimate`s, as the Levin branch carries `dataL`/`dataR`, so neither half is solved twice. A
  `LinAlgError` in a half is logged and re-raised, as the Levin branch does for `dataL`/`dataR`.
- **Comments and docstrings that state the premise:** the module docstring (a paragraph after the
  round-off bullet); `_roundoff_floor`'s docstring (a paragraph after "cannot itself freeze a
  region that genuine further bisection could still improve", saying a finite floor could, until
  06b); the comment above each `phase_limited` test (the Levin branch's old *"the children
  inherit essentially the same round-off floor"* is replaced by what is now checked); the fallback
  branch's opening comment (its children are no longer always freshly evaluated);
  `used_interval.phase_limited`; the result dict's `"phase_limited"` comment; and
  `adaptive_levin_sincos`'s `"phase_limited"` entry (deviation 6).
- **Not changed:** eq. (151), `_levin_G0_G1`, `_declared_endpoint_phase_err`, what `"phase_err"`
  holds, `resolved`, `_local_atol`, `DEFAULT_LEVIN_MAX_DEPTH`, `rtol`, the C2 gate, the result
  dict's keys, and every caller. Among existing code files, the commit touches
  `levin_quadrature.py` alone.

**`AdaptiveLevin/tests/test_levin_roundoff_exit.py`** (new; no Ray, no datastore). References are
closed forms evaluated by `mpmath`; each reference's own error is estimated as its 30-digit value's
difference from its 40-digit value, and asserted below $10^{-6}$ of what is tested against it
(deviation 8). Every accuracy assertion checks both that the value lies within its `abserr` of the
reference and that `abserr` is at least the true error.
1. **`test_exit_against_its_premise_levin_branch`** (test 1). $\theta = 100e^x$, `theta_deriv`
   supplied, $x \in [0, 20]$, $f = (1, 0)$, `atol` $10^{-15}$, `rtol` $10^{-7}$, order 16. $\theta'$
   spans $e^{20} = 4.85\times10^8$ (8.7 decades), no stationary point. Reference
   ${\rm Si}(100e^{20}) - {\rm Si}(100) = 0.008570859924939634$ (own error $5\times10^{-32}$).
   - **The parent** (bypass; and the parent commit's own module, `parent_demo.txt`, bit for bit):
     value `0.008570866361715342`, abserr `9.703303918195807e-07`, **1 region, 1 phase-limited**,
     depth 0, 3 solves, converged False. Floor $9.70\times10^{-7}$, step-(4) residual
     $6.42\times10^{-9}$. True error $6.44\times10^{-9}$, **relative $7.51\times10^{-7}$** against
     `rtol` $10^{-7}$. These are recorded in the test's docstring.
   - The halves' floors are $2.20\times10^{-11}$ each, $4.5\times10^{-5}$ of the parent's
     together (asserted under $10^{-3}$).
   - **The new code:** value `0.008570859939533229`, abserr $4.41\times10^{-11}$, 2 regions, 0
     phase-limited, depth 1, 7 solves, converged True; true error $1.46\times10^{-11}$ (relative
     $1.70\times10^{-9}$). Asserted: converged, no phase-limited region, within tolerance, honest.
2. **`test_exit_fires_where_premise_holds`** (test 2), two problems, each asserted: at least one
   phase-limited region, `max_depth < depth_max`, bit-identical to the bypassed call (value,
   abserr, evaluations, region count, phase-limited count, max depth, every field of every region),
   $G_1/G_0 < 3$ on every region (from `theta_deriv` on each region's own Chebyshev grid), honest.
   - **(a)** $\int_{1/3}^{7/3} e^{-x}\sin(10^6x)\,dx$ with `theta_deriv`, `atol` $10^{-18}$, `rtol`
     $10^{-13}$, order 12 — `test_roundoff_floor_independent_of_theta_presentation`'s problem.
     $G_1/G_0 = 1$; halves/parent 0.684 (the falling $\|f\|$). 1 region, phase-limited, at
     depth 0 under both.
   - **(b)** $\int_0^2 \sin(10^6e^{x/2})\,dx = 2({\rm Si}(10^6e) - {\rm Si}(10^6))$, `atol`
     $10^{-24}$, `rtol` $10^{-16}$, order 16. $G_1/G_0 = e$; halves/parent 0.668. 1
     region, phase-limited at depth 0 under both (deviation 7).
3. **`test_exit_against_its_premise_fallback_branch`** (test 3). $\theta = 10^{-6}e^x$ on
   $[0, 6]$, $f = (e^x, 0)$, `atol` $10^{-20}$, `rtol` $10^{-10}$; total phase $4\times10^{-4}$ rad,
   so every region is a fallback region; $\theta' > 0$. Reference $(\cos A - \cos Ae^6)/A$.
   - **The parent:** 1 direct region at depth 0, phase-limited: nested pair $2.2\times10^{-11}$
     (relerr $2.7\times10^{-10}$), floor $1.03\times10^{-5}$; value exact to the last bit.
   - **The new code:** the halves' floors are $5.14\times10^{-7}$ each (0.0996 of the parent's
     together), so it bisects: 2 direct regions at depth 1, both resolved, 3 solves, abserr
     $1.03\times10^{-6}$; true error $1.4\times10^{-17}$. Asserted: depth ≥ 1 everywhere, every
     region direct, none phase-limited, honest, abserr below half the parent's.
4. **`test_nothing_else_moves`** (test 4). 27 problems: every problem `test_levin_quadrature.py`
   drives `_adaptive_levin` with, restated with that file's arguments (the file is not edited;
   its m = 4 basis class is imported from it). Two raise `ValueError` under both; 25 are driven.
   All 25 are asserted bit-identical (deviation 9). 8 have a phase-limited region at the parent;
   they are tabulated below.
- **`test_premise_holds_on_infinite_half_floor`** ($+\infty$ halves; $\rho = 0.5$, inclusive) and
  **`test_bypass_is_not_exposed`** (deviation 10).

**`docs/levin-phase-floor/06b/calibration/`.** Every output's header records the SHA-256 of its
script, of `prov.py`, of `levin_quadrature.py` (`1ba4635e…` in all four) and of each input, the
`HEAD` (`da5c24d`) and the dirty tree's `git status --short`. Every script was formatted with
`black` before the run that is kept.
- `scripts/prov.py` — the shared provenance header.
- `scripts/parent_demo.py` → `out/parent_demo.txt`. Reads the parent's
  `AdaptiveLevin/levin_quadrature.py` with `git show da5c24d:…` into a temporary directory, imports
  it beside the working tree's, and runs every problem of the new test module three ways: the
  parent, the bypass, the new code. **Parent and bypass agree bit for bit on all 29 driven problems
  and raise alike on the two others.** Test 1's parent numbers come from here.
- `scripts/lg_seeded_draws.py` → `out/lg_seeded_draws.jsonl` (`run`, 160 units, one per line,
  with every Levin call's summary and the test's printed output) and
  `out/lg_seeded_draws_report.txt` (`report`). Wall time 64.5 s on 8 processes.
- `scripts/lg_seed15_trail.py` → `out/lg_seed15_trail.txt`. The one call that reaches
  `depth_max`, rebuilt from `random.seed(15)` and followed down three ancestries.

### Test 4: the problems with a phase-limited region at the parent

Each is bit-identical between the parent, the bypass and the new code (`parent_demo.txt`).

| problem | value | abserr | regions | phase-limited | depth | solves |
|---|---|---|---|---|---|---|
| `roundoff_floor_raw(1e6)` | `-3.3472684104646384e-07` | `2.1495939317214157e-16` | 1 | 1 | 0 | 3 |
| `roundoff_floor_reduced(1e6)` | `-3.347268410408778e-07` | `2.1495939317213671e-16` | 1 | 1 | 0 | 3 |
| `roundoff_floor_raw(1e12)` | `-2.9701896642525725e-13` | `2.149593931721415e-16` | 1 | 1 | 0 | 3 |
| `roundoff_floor_reduced(1e12)` | `-2.970231621698786e-13` | `2.1495939317213671e-16` | 1 | 1 | 0 | 3 |
| `theta_abserr_scalar` | `-3.3472684104646384e-07` | `8.350000312578765e-15` | 1 | 1 | 0 | 3 |
| `theta_abserr_callable` | `-3.3472684104646384e-07` | `8.350000312578765e-15` | 1 | 1 | 0 | 3 |
| `theta_mod_2pi_path_only` | `-3.3472684104087496e-07` | `2.1495939317214157e-16` | 1 | 1 | 0 | 3 |
| `theta_deriv_path_only` | `-3.3472684104646665e-07` | `2.1495939317213671e-16` | 1 | 1 | 0 | 3 |

All eight have constant $\theta'$: $G_1/G_0 = 1$ and the premise holds.

### Test 5: the LiouvilleGreen draws

Per seed, the calls whose phase-limited regions changed, from `lg_seeded_draws_report.txt` (which
lists each call, its group, span, both results and both sets of phase-limited regions). The four
counts are `test_YJJ_log_singularity` / `test_JJJ` / `test_YJJ` / `test_abserr_bounds_truth`. "pl"
is calls with a phase-limited region, parent → new. "F→T" is calls whose `converged` went False to
True; no call went True to False. The ratio is new/parent `abserr`, median and range.

| seed | calls changed | by test | pl | F→T | largest depth | at the cap | abserr ratio median [range] |
|---|---|---|---|---|---|---|---|
| 0 | 50 | 4/13/7/26 | 50 → 7 | 6 | 9 → 9 | 0 | 0.999 [3.7e-4, 1] |
| 1 | 43 | 4/6/7/26 | 43 → 5 | 6 | 9 → 9 | 0 | 0.999 [3.0e-4, 1] |
| 2 | 59 | 14/14/5/26 | 59 → 6 | 21 | 9 → 9 | 0 | 0.38 [2.6e-4, 1] |
| 3 | 47 | 4/11/6/26 | 47 → 6 | 6 | 10 → 10 | 0 | 0.999 [4.8e-4, 1] |
| 4 | 56 | 6/16/8/26 | 56 → 10 | 9 | 9 → 9 | 0 | 0.999 [5.1e-4, 1] |
| 5 | 49 | 4/12/7/26 | 49 → 8 | 6 | 9 → 9 | 0 | 0.999 [4.0e-4, 1] |
| 6 | 57 | 14/11/6/26 | 57 → 4 | 18 | 9 → 9 | 0 | 0.054 [3.9e-4, 1] |
| 7 | 48 | 4/12/6/26 | 48 → 5 | 7 | 10 → 10 | 0 | 0.999 [4.1e-4, 1] |
| 8 | 50 | 4/14/6/26 | 50 → 5 | 8 | 9 → 9 | 0 | 0.999 [3.5e-4, 1] |
| 9 | 47 | 5/12/4/26 | 47 → 9 | 6 | 9 → 9 | 0 | 0.999 [5.0e-4, 1] |
| 10 | 49 | 5/12/6/26 | 49 → 6 | 6 | 9 → 9 | 0 | 0.999 [4.5e-4, 1] |
| 11 | 52 | 6/13/7/26 | 52 → 9 | 9 | 9 → 9 | 0 | 0.999 [3.9e-4, 1] |
| 12 | 53 | 4/17/6/26 | 53 → 7 | 6 | 9 → 9 | 0 | 0.999 [4.2e-4, 1] |
| 13 | 50 | 4/13/7/26 | 50 → 9 | 6 | 9 → 9 | 0 | 0.999 [3.6e-4, 1] |
| 14 | 51 | 6/12/7/26 | 51 → 5 | 8 | 10 → 10 | 0 | 0.999 [3.4e-4, 1] |
| 15 | 60 | 18/12/4/26 | 60 → 9 | 24 | 10 → **20** | **1** | 0.269 [2.7e-4, 1] |
| 16 | 52 | 5/13/8/26 | 52 → 7 | 5 | 9 → 9 | 0 | 0.999 [6.5e-4, 1] |
| 17 | 48 | 4/11/7/26 | 48 → 5 | 6 | 9 → 9 | 0 | 0.999 [3.5e-4, 1] |
| 18 | 52 | 4/15/7/26 | 52 → 5 | 7 | 9 → 9 | 0 | 0.999 [4.2e-4, 1] |
| 19 | 53 | 6/15/6/26 | 53 → 6 | 7 | 9 → 9 | 0 | 0.999 [4.0e-4, 1] |

- **Outcomes:** all 80 (seed, test) pairs pass at the parent and under the new code. No stop.
- **Totals:** 4,320 Levin calls; 1,026 with a phase-limited region at the parent, 133 under the
  new code; 1,026 moved, and they are exactly the 1,026. **Every call with no phase-limited region
  at the parent is bit-identical** in value, abserr, evaluations and every region.
- **Largest printed relative error against the closed form,** over all seeds, parent / new:
  `test_JJJ` $5.61\times10^{-10}$ / $9.29\times10^{-12}$; `test_YJJ` $1.61\times10^{-8}$ /
  $1.61\times10^{-8}$; `test_YJJ_log_singularity` $2.93\times10^{-4}$ / $2.93\times10^{-4}$.
- `test_abserr_bounds_truth` draws nothing, so its 26 changed calls are the same at every seed.
- **The seed-15 call at the cap** (`lg_seed15_trail.txt`). `test_JJJ`'s second draw, J110 at
  $(k, q, s) = (0.8743, 4.9331, 0.1827)$, group 3, `rtol` $10^{-10}$, order 12, span
  $\log x \in [2.046, 27.631]$.
  - **Parent:** 6 regions, depth 5, 23 solves. $[8.4426, 14.8387]$ is accepted at depth 2
    through the exit: residual $3.51\times10^{-15}$ (relerr $1.93\times10^{-10}$), floor
    $2.20\times10^{-13}$ — against its premise, halves/own $= 0.042$.
  - **New:** 602 regions, depth 20, 2,295 solves; 56 Clenshaw–Curtis regions at depth 20, all in
    $[10.6797, 10.6904]$. Value moves by $1.7\times10^{-15}$ ($5\times10^{-11}$ relative).
    `abserr` $2.6361\times10^{-10} \to 2.6340\times10^{-10}$, both set by the floor of
    $[14.84, 27.63]$, which is accepted resolved at depth 1 under both. Converged False under both.
  - **Along the ancestry of $\log x = 10.6797$** the step-(4) residual stays at
    $1.2$–$2.2\times10^{-15}$ from depth 2 to 17 while the eq. (151) floor falls from
    $2.2\times10^{-13}$ to $7.8\times10^{-20}$. The premise ratio passes $\tfrac12$ at depth 4,
    where the floor ($5.0\times10^{-16}$) is already below the residual, so the exit can no
    longer apply. At depth 18–19 the fallback's nested pair is $2.3\times10^{-9}$ and
    $9.1\times10^{-14}$; at 20 it is resolved. On the ancestries of 10.685 and 12.0 the same flat
    residual collapses to $\sim10^{-21}$ at depth 13 and 16, once the region no longer contains
    some point. So the residual is set by something localised that eq. (151) does not model; what
    it is is not diagnosed.

## Predictions and verdicts

§1–§3 score nothing. Prompt 06b §5's predictions are copied below unchanged, before §5 runs, as §3
asks. They are **not yet scored**; §5 scores them on the Mac Studio.

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

**Scored on the Mac Studio, 2026-10-28** (the census agent, re-scored by the Mac Studio
orchestrator from both checkpoints; run `levin-phase-floor-06b-census-20261028T161212`, 18,560 of
18,560 units, none over budget):

| Rule | The numbers | Verdict |
|---|---|---|
| **W0** (gate) | the parent's code is 06a's; `low-G+r-1`, `ctrl-low-1` at `1e-7` and `0-1971` at `1e-8` reproduce 06a's units bit for bit; two runs of one $b$ unit bit-identical | **holds** |
| **W1** | 36,112 of 36,112 calls and 6,777 of 6,777 rows equal 06a's at `1e-7`; 35,605 of 35,605 and 6,177 of 6,177 at `1e-8`. The analytic-block calls read by the premise counter, as the user accepted; its cross-check on group calls shows 0 disagreements | **holds** |
| **W2** | 24 calls, none dropped ($\|S_9 - S_{10}\| \le 0.0066$ `abserr`$_b$). $\|{\rm value}_b - S_{10}\| \le$ `abserr`$_b$ on 7; it fails on 17, median ratio 1.79, maximum 19.7 (`b-1e-7-3-34455` call 1). $b$'s values are closer to the reference than 06a's (4 of 24 outside their request against 23 of 24), but its declared error understates its true error | **fails** |
| **W3** | 44 regions accepted phase-limited at `1e-7`, all in analytic-block calls; each one's halves' floor sum is 0.5045–0.8908 of its own | **holds** |
| **W4** | three moved analytic-block calls at `1e-8` reach depth 20: `2-1704` Y1 (1,136 regions), `2-13964` J1 (31,885 regions, 159 s), `3-34987` J2 (14,293 regions); none over budget | **fails** |

The rehearsals of §4 hold: R15 and R16 both give `1b332ffa…`; outside `QuadSourceIntegral` all 7
value tables are identical; inside it the 782 rows with no phase-limited flag in R15 are
bit-identical, and the 82 others are tabulated.

## Deviations from the prompt

1. **The baselines are the orchestrator's.** **IMPLEMENTATION CHOICE.** They were taken at
   `93d5cb1`, whose code is `da5c24d`'s, and passed in the dispatch; they were not re-run before
   the change.
2. **The fallback branch's halves are full subregion solves, carried forward if the region is
   bisected.** **IMPLEMENTATION CHOICE** (§0.2: "The mechanism is the agent's choice").
   - Each half is `_adaptive_levin_subregion(half)`, the call that makes a region's own
     `"phase_err"`, on the half's own samples. A floor-only helper would sample $f$ and $\theta'$
     at the same nodes, so it would save only the basis evaluation and the value, and then the
     children would be solved again when popped.
   - So the halves' floors are the floors each half would carry as a region. A half that the
     total-variation gate sends to Levin carries the Levin branch's floor; this was not seen in
     any test.
   - The two solves count in `evaluations`/`num_subregion_solves` and `num_solves_*` once, where
     they are made, and not again when a carried half is popped — the Levin branch's accounting.
3. **On the fallback branch the premise is checked at `depth_max` too.** **IMPLEMENTATION
   CHOICE.** A region there is accepted whatever the premise says, so the two solves only decide
   its `phase_limited` flag. Without them a region accepted at the cap would be flagged
   phase-limited on a floor bisection would lower, which is what the flag must no longer mean. On
   the Levin branch the check is free.
4. **"After the existing conditions" includes `not resolved`.** **IMPLEMENTATION CHOICE.** The
   premise is checked after `phase_limited = phase_limited and not resolved`, so a resolved region
   costs nothing extra; that is the reading of §0.2's "only when the exit would otherwise fire".
5. **A helper function, `_roundoff_exit_premise_holds`, beside the constant.** **IMPLEMENTATION
   CHOICE.** §0.3 names "one new constant". The two branches share the one inequality, and test
   `test_premise_holds_on_infinite_half_floor` checks it directly (§1 item 6, and $\rho$ inclusive).
6. **Docstrings and comments beyond §1 item 5's three.** **IMPLEMENTATION CHOICE**, under §0.3's
   "the comments and docstrings that state the premise": `used_interval.phase_limited` ("Subdividing
   such a region cannot improve it"), the result dict's `"phase_limited"` comment,
   `adaptive_levin_sincos`'s `"phase_limited"` entry, and the fallback branch's opening comment,
   which said a bisected fallback region's children are always freshly evaluated (deviation 2
   makes that false).
7. **Test 2 runs two problems, the second at `rtol` $10^{-16}$.** **IMPLEMENTATION CHOICE.** The
   existing problem (a) has $G_1/G_0 = 1$ exactly, so its halves/parent (0.684) comes from $\|f\|$
   alone. (b) has $G_1/G_0 = e$, halves/parent 0.668, so it tests that $\rho$ lets a modest ratio
   through. Its
   floor ($6.4\times10^{-16}$) binds only if the step-(4) residual ($8.5\times10^{-22}$, relerr
   $7.4\times10^{-16}$) is not already resolved, hence `atol` $10^{-24}$ and `rtol` $10^{-16}$,
   below double precision; at `rtol` $10^{-13}$ and $10^{-15}$, and at order 12, it resolved
   instead (failed attempts, scratch only).
8. **The reference value is the 40-digit evaluation; the 30-digit one estimates its error.**
   **IMPLEMENTATION CHOICE.** §2 asks for "`mpmath` at 30 digits … Report its own error estimate".
   The references are closed forms (Si, cos, exp) evaluated by `mpmath`; the difference of the two
   precisions is the estimate (at most $1.5\times10^{-26}$), asserted below $10^{-6}$ of the
   larger of the true error and `abserr`.
9. **Test 4 asserts bit-identity where the parent has a phase-limited region too.**
   **IMPLEMENTATION CHOICE.** §2 asks only that those be tabulated. All eight are bit-identical
   (constant $\theta'$), so the stronger assertion holds at this commit and is kept; they are also
   tabulated above. Test 4 restates `test_levin_quadrature.py`'s driver problems in the new
   module, since that file may not be edited; problems that do not drive `_adaptive_levin` (the
   `_roundoff_floor` and CC-weight unit tests, `test_mode_filter_gates_on_endpoint_not_mean`'s
   direct call to `_adaptive_levin_subregion_impl`) are not restated. `test_converged_flag`'s first
   call is `SinIntegral`'s, and is run once.
10. **Two tests beyond §2's six:** `test_premise_holds_on_infinite_half_floor` and
    `test_bypass_is_not_exposed`. **IMPLEMENTATION CHOICE** — §1 items 4 and 6 state contracts that
    a test can hold.
11. **Test 5 runs every default test of the module at every seed, including the seed-free
    `test_abserr_bounds_truth`; `test_YJJ_log_scaling` is not run.** **IMPLEMENTATION CHOICE.**
    The latter is skipped by default (`THREE_BESSEL_DIAGNOSTIC_PLOTS` unset) and asserts nothing.
    The harness also checks, beyond §2, that every call with no phase-limited region at the parent
    is bit-identical.
12. **Calibration outputs beyond test 5's:** `parent_demo.py` (the parent-behaviour demonstration
    the dispatch allows: the bypass against the parent commit's own module) and
    `lg_seed15_trail.py` (the diagnosis of the call at the cap, part of test 5's record), with a
    shared `prov.py`. **IMPLEMENTATION CHOICE.**
13. **The seeded-draws run was made twice; the first is not kept.** **IMPLEMENTATION CHOICE**
    (`CLAUDE.md` rule 7). A per-seed summary section was added to the report after the first run,
    which changed the script's hash; the run and its report were both re-made by the final script,
    not re-stamped. The two runs' totals were identical (1,026 / 133 / 1,026). A timing probe at
    seed 10 (8 units, from a scratchpad driver importing the harness) ran before them, and is not
    kept: every unit took under 5 s there, and at most 7.5 s in the kept run, so the harness is a
    65-second job, and README §5 rule 7 (long runs registered and detached) does not apply.
14. **Test problems were found by search.** **IMPLEMENTATION CHOICE**, failed attempts recorded.
    Test 1: $\theta = Ae^x$ on $[0, L]$ at $(A, L, {\rm rtol}) = (100, 20, 10^{-7})$, chosen; $(10,
    20)$ is resolved at depth 2 under both; $(1000, 20)$ is accepted *resolved* at depth 0 with the
    floor as its `abserr` (observation 1); $(100, 19)$ and $(100, 21)$ behave like the choice.
    Test 3: $(A, L) = (10^{-6}, 6)$, chosen; $(10^{-6}, 4)$, $(10^{-5}, 4)$, $(10^{-4}, 4)$ and
    $(10^{-3}, 3)$ are resolved at depth 0. Test 2 (b): deviation 7; $f = (e^{-x}, 0)$ with the
    same phase fails the premise at depth 0 (the falling $\|f\|$ halves the sum) and fires at
    depth 1.
15. **A diagnostic trail was first run at Chebyshev order 16, not the three-Bessel order 12.**
    **UNINTENDED DRIFT**, in a scratch script only. Its residuals disagreed with the driver's; the
    order was found and the kept `lg_seed15_trail.py` uses the call's own `chebyshev_order`, and
    reproduces the driver's depth-2 numbers exactly ($3.507\times10^{-15}$,
    $2.196\times10^{-13}$).
16. **Two issues opened** (§3, and `docs/OPEN_ISSUES.md`, 113 → 115):
    `[06b-resolved-region-keeps-a-floor-bisection-would-cut]` and
    `[06b-premise-check-can-chase-a-flat-residual-to-depth-max]`. **IMPLEMENTATION CHOICE**
    (`CLAUDE.md` rule 4). The header issue does not move.

## Verification performed

- **The new module**, `PYTHONPATH=. ./venv/bin/python -m unittest AdaptiveLevin.tests.test_levin_roundoff_exit`:
  6 tests, OK.
- **The breakages**, each applied with `sed`, the new module run, then reverted from a backup and
  the SHA-256 checked back. They were run twice: first on the file before one docstring paragraph
  was re-wrapped (`22fddbf5…`), then again on the final file (`1ba4635e…`), with the same results:
  - **(i)** premise check removed on both branches (`if False and phase_limited …` at both
    tests): **test 1 fails** (`converged` False), test 3 fails (a phase-limited region remains).
  - **(ii)** exit removed altogether (`phase_limited = False` in place of both
    `phase_limited and not resolved`): **test 2 fails** (`0 not greater than or equal to 1 : test
    2a`); tests 1 and 3 fail on their parent assertions.
  - **(iii)** premise on the Levin branch only (the fallback's test disabled): **test 3 fails**,
    alone.
- **Test 4 against the parent's own module** (`parent_demo.txt`): parent == bypass bit for bit on
  all 29 driven problems; both `nan` problems raise `ValueError` under all three.
- **Test 5** (`lg_seeded_draws_report.txt`): 160/160 units, none missing; no stop; no
  non-phase-limited call moved.
- **The seven roots, after the change** (the commands of `CLAUDE.md`, run in sequence; `uptime`
  beside each):

  | Root | Baseline (`93d5cb1`) | After | Result | `uptime` load averages, before |
  |---|---|---|---|---|
  | AdaptiveLevin | 32 | **38** | OK | 17.35 17.89 15.88 |
  | ComputeTargets | 780 | 780 | OK (133.4 s) | 17.35 17.89 15.88 |
  | CosmologyModels | 39 | 39 | OK | 10.72 14.92 14.97 |
  | LiouvilleGreen | 148 | 148 | OK (skipped=1) | 10.66 14.84 14.94 |
  | ObjectFactories | 280 | 280 | OK | 9.28 14.19 14.70 |
  | RunDrivers | 127 | 127 | OK | 9.86 14.07 14.65 |
  | tools | 58 | 58 | OK | 9.31 13.81 14.55 |

  AdaptiveLevin rose by 6, the new module's 6 tests. The `test_wall_time_per_object` flake did not
  fire. The machine was loaded by other work throughout (load averages 8–18 on 10 cores).
- **`black`** on `levin_quadrature.py`, the new test module and the calibration scripts; `black
  --check AdaptiveLevin/` clean.
- **Scope.** `git status --short` before staging: `levin_quadrature.py` modified;
  `AdaptiveLevin/tests/test_levin_roundoff_exit.py` and `docs/levin-phase-floor/06b/` new; then
  this log, the board and `docs/OPEN_ISSUES.md`. No existing test file was edited.

## Observations not acted on

1. **A resolved region keeps a floor that bisection would cut.** The premise is checked only on
   the exit. A region accepted as *resolved* still carries its eq. (151) floor into `abserr`, even
   where $G_1/G_0$ makes it orders of magnitude above its halves'. $\theta = 1000e^x$ on $[0, 20]$
   at `rtol` $10^{-7}$ is accepted resolved at depth 0 (relerr $3.7\times10^{-8}$) and declares
   `abserr` $9.7\times10^{-7}$ against a true error of $2.1\times10^{-11}$, `converged` False. The
   seed-15 call's `abserr` is the same pattern ($[14.84, 27.63]$ at depth 1, accepted by `atol`,
   floor $2.63\times10^{-10}$). §0.3 keeps `resolved` unchanged, so it is opened as
   `[06b-resolved-region-keeps-a-floor-bisection-would-cut]`.
2. **The premise check can chase a residual that eq. (151) does not model to `depth_max`.** Test
   5's seed-15 call, above: opened as
   `[06b-premise-check-can-chase-a-flat-residual-to-depth-max]`. It bears on §5's W4.
3. **`test_levin_quadrature.py`'s stationary-phase problem returns `abserr = inf`** under the
   parent and the new code alike (an interior stationary point; its test asserts `abserr >
   true_err`, which `inf` satisfies). This is `[04-roundoff-floor-can-be-infinite]` and 07's
   ground; nothing new.
4. **The halves' floors' sum depends on $\|f\|$ as well as on $G_1/G_0$.** With constant
   $\theta'$ and an $f$ that falls by $e^{-1}$ across a region, the halves' sum is
   $(1 + e^{-1})/2 \approx 0.68$ of the parent's; with $G_1/G_0 = e$ as well, it is 0.46 and the
   premise fails (deviation 14). Bisection does lower that floor — the parent's uses
   $\max|f|$ over the whole region — so this is the premise working as written, and costs about
   one level.

## State handed on

- **The code commit** is made and not pushed. §4 (R15 at its parent `da5c24d`, R16 at it) and §5
  (the census, 06a's harness at the code commit) run on the Mac Studio after the push. The board's
  06b row says the code landed and §4–§5 are pending.
- **For §4:** a call with no phase-limited region at the parent is bit-identical, so a
  `QuadSourceIntegral` row with no phase-limited flag in R15 must be bit-identical in R16. The
  fallback branch adds two solves to the evaluation counters where its premise is checked, so
  `evaluations`/`num_subregion_solves` differ only in calls that had a phase-limited fallback
  region at the parent.
- **For §5's W4:** expect possible cap hits. On the laptop's draws one call of 1,026 that moved
  reached depth 20.
- **For 07's rebase:** the bypass keyword is the last parameter of `_adaptive_levin`; the two
  `phase_limited` tests now end with the premise block, after `and not resolved`. 07's rule (ii)
  (the exit barred in window pieces and guarded regions) composes with it.
- `docs/levin-phase-floor/06b/calibration/` holds the scripts and outputs; nothing is kept on the
  Mac Studio from this part.

# The Mac Studio record

Appended by the orchestrator in 06b's records commit, as the two agents wrote them. Paths in them
are relative to [`docs/levin-phase-floor/06b/rehearsal/`](../../../docs/levin-phase-floor/06b/rehearsal/)
and [`docs/levin-phase-floor/06b/census/`](../../../docs/levin-phase-floor/06b/census/). The Mac
Studio orchestrator's review is [`docs/levin-phase-floor/06b/review.md`](../../../docs/levin-phase-floor/06b/review.md).

**The review** passed checks 1–9 and stopped on check 10 alone: the census log has no **What
shipped** section, and its deviation 19 (no over-budget pass) is unclassified. The user accepted
both as they stand: *"If the only issue is that the agent didn't say whether anything went over
budget, then I don't care. Just accept it."* The laptop re-checked one moved R15/R16 row and one
unmoved census unit (prompt-06b.md §B.6).

## §4 — rehearsals R15 and R16 on the Mac Studio

**Prompt:** [`06b-roundoff-exit-premise.md`](../06b-roundoff-exit-premise.md) §4, under the
orchestrator's Part B dispatch (`92bc62f`) and its addendum B.5.
**Host:** mps026298, the Mac Studio.
- **R15** ran in the worktree `SecondaryGWKit-lpf06b-da5c24d`, detached at `da5c24d`. That is the
  code commit `4ddc3c2`'s parent.
- **R16**, the comparison and the readers ran in the worktree `SecondaryGWKit-lpf06b-92bc62f`,
  detached at `92bc62f`. Its code is `4ddc3c2`'s: `git diff --stat 4ddc3c2 92bc62f` outside
  `docs/` and `prompts/` is empty.
- **The main checkout** stayed at `28ebaf0` and clean. It ran no tree code. It held the venv, the
  stores, the registry and these records.

**Model:** Claude Opus 5.5, the rehearsal subagent of the Mac Studio orchestrator, in three turns:
1. the gate, the lock and R15's launch;
2. R15's checks and R16's launch;
3. R16's checks, the comparison, the tabulation, the readers, the release and these records.

**Date:** 2026-10-28, 13:52–14:12 UTC.
**Commit:** none. The records are committed on the laptop, in 06b's records commit.
**Result:** COMPLETE WITH DEVIATIONS. No deviation stopped the work or changed a computation, and
no §7 stop condition fired.
- **The fingerprints.** R15 and R16 both ended `done`, exit 0, with fingerprint `1b332ffa…`.
  Each equals R8's class by class, and R16's `classes` object equals R15's.
- **Outside `QuadSourceIntegral`:**
  - every value table is bit-identical: 7 tables, 360,462 rows;
  - every other difference is a timing, a label, `parents_digest` or the residual-memo counters;
  - 0 differences are unclassified, and no row is in one store only.
- **Inside `QuadSourceIntegral`:**
  - 82 of 864 rows carry a phase-limited flag in R15;
  - all 782 rows without one are bit-identical in every value field and every work counter;
  - the 82 are tabulated;
  - R16 carries **no** phase-limited flag on any row.
- **The readers** exited 0, and R16 is byte-identical before and after them.
- **The lock** is released, and `ray-processes` lists nothing.

The measurements are in `verification.md`, written for `docs/levin-phase-floor/VERIFICATION.md`
§6b, "The round-off-limited exit".

### What shipped

Nothing in the repository: no code and no commit. Everything was written on the Mac Studio, in the
main checkout.
- **The records directory** is `var/levin-phase-floor-06b/rehearsal/`:
  - `out/` holds every command (`commands.log`) and every output;
  - `scripts/` holds:
    - 05's `classify_non_qsi.py`, unchanged;
    - 06's `run_readers_worktree.sh`, unchanged;
    - this agent's `tabulate_qsi_phase_limited.py` (§4.3's per-row tabulation), as run in
      attempt 2;
    - `rec.sh`, the recorder of read-only checks;
  - `state/` holds the lock transcript, the launch records, the store hashes, attempt 1's script
    and `progress.md`;
  - `extract/` is the readers' output, 180 MB, and is not for copying back;
  - `log.md`, `verification.md` and `issues.md`.
- **The stores:**
  - `var/datastores/levin-phase-floor-06b-r15.sqlite` and `-r16.sqlite`, each with 4 shards;
  - **they are kept** (§4.6).
- **The runs:**
  - `var/runs/levin-phase-floor-06b-r15-parent-20261028T135349/` and
    `…-r16-premise-20261028T135745/`;
  - the launch logs `var/runs/levin-phase-floor-06b-r15.launch.log` and `-r16.launch.log`.

### Predictions and verdicts

§4 states acceptance rules, not pre-registered predictions. Each is scored here as written.

| rule (§4, §6.2) | number | verdict |
|---|---|---|
| R15 `done` with `1b332ffa…` | `done`, exit 0, digest `1b332ffa948a3dbd…`, classes equal R8's | **holds** |
| R16 `done` with `1b332ffa…` | `done`, exit 0, digest `1b332ffa948a3dbd…`, classes equal R8's and R15's | **holds** |
| Outside `QuadSourceIntegral`, every value table identical | 7 value tables, 360,462 rows identical, 0 differ | **holds** |
| …every other difference of the four volatile kinds | 0 unclassified (`classify_non_qsi.py`, exit 0) | **holds** |
| …no row in one store only | 0 only-in-A, only-in-B or unmatched, in every class | **holds** |
| Inside, a row with no phase-limited flag in R15 is bit-identical in `total`, `total_abserr`, the analytic columns and every group record | 0 of 782 differ, in that set, in every value field and in every work counter | **holds** |
| Inside, every other row is tabulated | 82 rows, in `out/qsi_tabulation.txt` and `out/qsi_rows.csv` | **done** |
| The readers exit 0, and the store is byte-identical | 6 of 6 exit 0; 5 files identical by SHA-256, mtime and size | **holds** |
| The lock is released, and `ray-processes` lists nothing | released, exit 0; `status` free; `ray-processes` none | **holds** |

### Where the phase-limited flags are stored (§4.3)

The flags were established from the code at `92bc62f` (`ComputeTargets/QuadSourceIntegral.py`,
`ObjectFactories/QuadSourceIntegral.py`; the factory at `da5c24d` is the same) and from the stores.
They all live in the shard table `QuadSourceIntegral`.

| flag | column | encoding | where it is set |
|---|---|---|---|
| (a) `total_phase_limited` | `total_phase_limited` | SQLAlchemy `Boolean`, nullable. Stored as SQLite `integer` 0/1: `typeof` is `integer` on all 864 rows of both stores | `evaluate_QuadSource_integral`: the OR of every group's flag, over every Levin sub-interval |
| (b) each group record's `phase_limited` | `metadata`, a JSON string (`json.dumps`, read with `json.loads`), at `WKB_Levin.subintervals[i].groups[j].phase_limited` | JSON `true`/`false` | `phase_group_Levin_integral`: each phase group's `adaptive_levin_sincos` flag |
| (c) each analytic block's `phase_limited` | `metadata`, at `analytic.{0pt5,2pt5}.{J,Y}{1..4}_data.phase_limited` when that three-Bessel integral is all Levin, or at `analytic.{0pt5,2pt5}.Levin.{J,Y}{1..4}_data.phase_limited` when it is split between quad and Levin | JSON `true`/`false` | `_three_bessel_Levin`'s per-block record (via `_three_bessel_integrals` and `analytic_integral`) |

- **Two more stored keys** are ORs of (b): `WKB_Levin.phase_limited`, and
  `WKB_Levin.subintervals[i].phase_limited`. The script asserts that they are on every row, and
  that `total_phase_limited` equals the OR of (b). No assertion fired.
- **No other** metadata leaf named `phase_limited` exists in either store.
- **The ORs of (c),** `J_phase_limited` and `Y_phase_limited`, are computed in
  `_three_bessel_Levin` but **stored nowhere**. So (c) has to be read block by block.
- **"Phase-limited in R15"** is (a) or any (b) or any (c), on R15's row.

### Deviations from the prompt

1. **The lock transcript** is `state/lock.log`; 06's was `out/lock.log`. The brief puts lock
   transcripts in `state/`. `IMPLEMENTATION CHOICE`.
2. **Extra `ray-processes` looks:** one before `acquire`, one after each build, and one after the
   readers. They are read-only and not in §4.1's list. Each was a command of its own, appended by
   `>>`. `IMPLEMENTATION CHOICE`.
3. **The drivers ran with both `PYTHONDONTWRITEBYTECODE=1` and `-B`.** R13 used the variable
   alone; the brief asks for either. No effect on any value. `IMPLEMENTATION CHOICE`.
4. **`scripts/rec.sh`** records each read-only check: command, cwd, UTC start and end, output and
   exit code, in `out/NAME.txt`. It was used for checks only, never for a lock command or a
   launch. `IMPLEMENTATION CHOICE`.
5. **R16's purpose string** adds "QuadSourceIntegral alone is expected to differ, only in rows with
   a phase-limited flag in R15". That is §0.3 and §4.3's expectation, stated beside the brief's
   required content (host, tree SHA, `4ddc3c2`, `1b332ffa…`). `IMPLEMENTATION CHOICE`.
6. **The tabulation, attempt 1, failed.** It counted store-bookkeeping columns as values: `serial`,
   every `*_serial` (the row's own and its parents' serials, which each store assigns for itself)
   and `timestamp`. It reported 782 "value violations", all in those columns. Rows are paired by
   physical key, so these columns are not values. They are like `parents_digest` of
   `[05-store-value-diff-compares-volatile-columns]`, which `store_value_diff` itself never
   compares.
   - **The edit:** an `identity` kind for these columns, which are then not compared.
   - **The records:** the diff is `out/tabulate_qsi_phase_limited.attempt1_to_2.diff`. Attempt 1's
     script is `state/tabulate_qsi_phase_limited.attempt1.py`, and its output
     `out/qsi_tabulation.attempt1.txt`, `.stderr` and `out/qsi_rows.attempt1.csv`.
   - Attempt 2's flagged-row table is identical to attempt 1's. Only the verdict lines changed.

   `UNINTENDED DRIFT` (the script's first classification was wrong), corrected before any result
   was used.
7. **`run_readers.sh` cannot run from the code worktree as written.** It `cd`s to its own tree and
   calls `./venv/bin/python`, which the worktree lacks. 06's `run_readers_worktree.sh` was used,
   copied unchanged (`cmp` identical, `ec96cf09…`). Its diff against `run_readers.sh` equals 06's
   (`out/run_readers_worktree.diff`). `STRUCTURALLY REQUIRED`.
8. **One `rec.sh` check failed to parse.** Its first form, `readers_check`, used bash process
   substitution, which `sh -c` rejects (exit 2). Nothing ran. The second form overwrote its
   output file. The failure is recorded in `out/commands.log`. `UNINTENDED DRIFT`, with no effect.
9. **Two inline read-only looks were not saved:**
   - the tool JSON's `QuadSourceIntegral` keys and counts. `out/qsi_tabulation.txt`'s cross-check
     restates what was used;
   - the readers' file-name patterns against 06's, and the grep of `random.sample` in the readers.

   `IMPLEMENTATION CHOICE`.
10. **The scripts are left as run, not formatted by `black`.** `black --check` would reformat
    `tabulate_qsi_phase_limited.py` and `classify_non_qsi.py` (`out/black_check.txt`). They are
    records, byte-identical to the run, as 05's and 06's rehearsal scripts are. Committing them
    under `docs/levin-phase-floor/06b/rehearsal/scripts/` needs that path in `CLAUDE.md`'s
    formatting exemptions. `IMPLEMENTATION CHOICE`, for the laptop.

### Verification performed

**Gate** (`out/00-gate-trees.txt`, `out/01-gate-registry-lock.txt`, `out/02-gate-launcher.txt`):
- **The trees:**
  - the main checkout is `handover-remedial` at `28ebaf0` and clean;
  - the two worktrees are at `da5c24d` and `92bc62f`, both clean, ignored files included;
  - `4ddc3c2^` is `da5c24d`;
  - `da5c24d..92bc62f` on code paths touches only `AdaptiveLevin/levin_quadrature.py` and the new
    test module.
- **The registry and the lock:** the registry held 25 runs, all finished. The lock was free, with
  no old-location lock.
- **The host:** `auth_token` is mode 600, and 109 GiB were free.
- **The tools by SHA-256:** the launcher is `5b984c75…` in all three trees. `store_value_diff.py`
  is `7997af85…` in both worktrees, and `run_readers.sh` is `7c63a5ba…`.

**The builds** (`out/r15_launch_check.txt`, `out/r15_end_check.txt`, `out/r16_launch_check.txt`,
`out/r16_end_check.txt`):
- **The launch check.** The manifests carry the right `git_head`, with `git_dirty` false. Each
  driver's cwd (`lsof`) is its worktree.
- **The end.** No Traceback in either launch log, `stdout.log` or `stderr.log`, and no
  `!! WARNING` in either launch log. Each driver exited 0.
- **The worktrees** were clean after each build, with nothing newer than the pre-R15 marker.

**The stores:**
- R15's five files are unchanged from the end of its build to the end of the session;
- both stores' ten files are identical before and after the comparison, and at the end;
- R16's five are identical before and after the readers;
- there is never a journal, `-wal` or `-shm` file after a build.

The hashes are in `state/r15_store_sha256_after_build.txt`, `state/store_sha256_before_diff.txt`,
`state/store_sha256_after_diff.txt`, `state/r16_store_sha256_{before,after}_reads.txt` and
`state/stores_sha256_final.txt`.

**The comparison** (`out/store_value_diff.json`, `51b9b94a…`, 40.3 MB; `out/classify_non_qsi.txt`;
`out/qsi_tabulation.txt`; `out/qsi_set_check.txt`):
- **The tool** exited 1 in 18 s. `QuadSourceIntegral` is the only class with a value-like
  difference.
- **The cross-checks.** The tabulation's pairing by $(k, q, r, z_{\rm resp})$ gives the tool's key
  set, and `total`'s `float.hex` agrees with the tool on all 864 rows: 0 mismatches. The metadata
  key sets are equal on every row.
- **Rows whose `total` moved** are exactly the 26 rows with a group flag (b) in R15.
- **Rows whose `analytic_rad` moved** are exactly the 70 rows with an analytic-block flag (c) in
  R15.

**Work counters.** These are treated as counters, not values: the columns `WKB_Levin_num_regions`,
`WKB_Levin_evaluations`, `WKB_Levin_simple_regions`, `WKB_Levin_SVD_errors`,
`WKB_Levin_order_changes`, `WKB_Levin_chebyshev_min_order`, `WKB_Levin_max_depth`,
`numeric_quad_compute_steps` and `numeric_quad_RHS_evaluations`, and the metadata leaves `regions`,
`simple_regions` and `evaluations`.
- **Why.** Each counts what the integrator did, not a number it computed. The addendum expects
  evaluation counts to move only where halves are now solved.
- **They were compared anyway:** 0 of the 782 unflagged rows differ in any of them.

**The readers** (`out/readers.txt`, `out/readers_check.txt`):
- all six exit 0, each with the store identical before and after, no journal, and no Ray process
  left;
- 0 Tracebacks;
- each prints one `!! WARNING: number of shards read from database (=4) does not match specified
  number of shards (=10)`, as 06's did.

**The lock** (`state/lock.log`): ten commands, each a shell command of its own, every one exit 0:
1. look;
2. acquire;
3. set-pid 96498 (R15);
4. look;
5. set-pid 97360 (R16);
6. look;
7. look;
8. release;
9. `status` (free, no Ray processes);
10. look.

### Observations not acted on

1. **R16 carries no phase-limited flag at all.** On the rehearsal scope the premise check turns
   off every phase-limited acceptance: 48 group flags on 26 rows, and 143 analytic-block flags on
   70 rows, all go to 0.
   - **The test's problem against production.** §2 test 2 shows the exit still fires on a
     constructed problem with $G_1/G_0 < 3$. No production row here exercises that case.
   - **What it bears on:** §5's descriptive "how often the exit fired", and W3.
   - It is not scored here.
2. **In the 26 rows whose total moved:**
   - `WKB_Levin_max_depth` is unchanged in every one;
   - Levin regions rose from 530 to 612, and evaluations from 1314 to 1638;
   - `total_abserr` fell in 24, by factors of 0.242–0.985, and rose in 2, by 1.002 and 1.012;
   - 7 went from unconverged to converged, and 19 stayed unconverged;
   - $|\Delta{\rm total}|$ is at most 0.36 × `total_abserr`$_{R15}$ and at most 0.597 ×
     `total_abserr`$_{R16}$.
   - In the 70 analytic-flagged rows, $|\Delta$`analytic_rad`$|$ is at most 0.748 × R15's analytic
     `abserr` (median 0.058).
   - Honesty is §5 W2's question, not §4's. These numbers are descriptive.
3. **`_three_bessel_Levin` computes `J_phase_limited` and `Y_phase_limited` and stores neither.**
   A reader of `analytic_rad` has to walk the per-block metadata (flag (c)) to learn that an
   analytic block was accepted on its floor. This is not a defect of 06b's.
4. **The readers' output file counts differ from 06's run on R14:**
   - `extract_Gk_data`: 468 against 396;
   - `extract_GkSource_data`: 180 against 176;
   - `extract_QuadSourceIntegral_data`: 76 against 70.

   `extract_Gk_data.py`, `extract_GkSource_data.py` and `extract_QuadSourceIntegral_data.py` each
   draw their subsample with unseeded `random.sample`, so the counts vary from run to run. Not
   investigated further.

### State handed on

- **Stores, kept:**
  - `var/datastores/levin-phase-floor-06b-r15{,-shard000[0-3]}.sqlite`;
  - `var/datastores/levin-phase-floor-06b-r16{,-shard000[0-3]}.sqlite`;
  - the final hashes are in `state/stores_sha256_final.txt`.
- **Runs:** `levin-phase-floor-06b-r15-parent-20261028T135349` and
  `levin-phase-floor-06b-r16-premise-20261028T135745`, both `done`. The registry holds 27 runs,
  all finished.
- **The lock is free,** and `ray-processes --ray-temp-dir /tmp/ray-sgk` lists nothing. The census
  agent may launch.
- **The trees.** Both worktrees are clean and unwritten. The main checkout is at `28ebaf0` and
  clean.
- **For the laptop:**
  - copy `var/levin-phase-floor-06b/rehearsal/` back without `extract/` (180 MB);
  - `out/store_value_diff.json` is 40.3 MB, and the orchestrator decides whether to commit it;
  - add `docs/levin-phase-floor/06b/rehearsal/scripts/` to `CLAUDE.md`'s `black` exemptions
    (deviation 10).

## §5 — the census on the Mac Studio

**Subject:** prompt 06b §5 on the Mac Studio (mps026298), Ray-free, no lock. 06a's census re-run
at the code commit `4ddc3c2` against 06a's run as the "before".
- **Trees:** $b$ is the worktree `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06b-92bc62f`,
  detached at Part B's dispatch SHA `92bc62f`, whose code is `4ddc3c2`'s. $p$ is the worktree
  `…-lpf06b-da5c24d` at the code commit's parent `da5c24d`, used for Gate W0's reproduction only.
- **Run:** `levin-phase-floor-06b-census-20261028T161212`. Launched 16:12:12Z (sh 8931, driver 8935);
  the last unit was recorded at 17:02:45Z, and the run ended `done`, exit 0, by 17:03:32Z. That is
  about 51 min, against 06a's 45.5. All 18,560 units are `done`: none over budget, none in error.
- **Commit:** none here; the laptop commits this directory.

**Result:**
- **W0 holds**, all three clauses.
- **W1 holds.** At both rtols, every group call and every row the rule reaches equals 06a's bit
  for bit.
- **W2 fails.** 17 of the 24 sampled calls have $|{\rm value}_b - {\rm ref}| > $ `abserr`$_b$, by up
  to 19.7×. b's values are closer to the reference than 06a's: 4 of 24 are outside their requested
  tolerance under b, against 23 of 24 under 06a. But b's declared error understates its true error.
- **W3 holds.** 44 regions are accepted as phase-limited at 1e-7, all in analytic-block calls. Each
  one's halves' floor sum is at least 0.504 of its own floor.
- **W4 fails.** Three analytic-block calls at 1e-8 that moved reach depth 20, after the premise
  failed. No unit is over budget.
- **The store:** the scratch copy is byte-identical to its certification before, during and after,
  and gives `0e32f24a…`. No worker initialised Ray.

### Before the launch (in order; every command and its output in `out/cmd/`)

1. **Harness copied** (`00`): 06a's 39 scripts were copied from the code worktree's
   `docs/levin-phase-floor/06a/scripts/`, which equals (`diff -r`) `var/levin-phase-floor-06a/scripts/`.
   They went into `scripts/`, and unchanged into `state/scripts_06a_reference/`. The diff against
   06a is saved at preparation and at the end: `out/scripts_diff_against_06a_at_prep.txt` and
   `…_final.txt`.
2. **Rows** (`01`): 06a's `state/rows.json` was copied; its sha256 `aa523d28…` is equal.
3. **The store** (`02`–`05`, `20`, `21`): the scratch copy's five files equal
   `copy_files_certified.txt` (EQUAL), with no journal, `-wal` or `-shm` file. `fingerprint_copy.py`
   gives `0e32f24a…`, problems none, at $b$ and at $p$.
4. **W2's capture list** (`06`): `state/w2_capture.json`, sha256 `217723d1…`. 06a had 1,213
   phase-limited calls at 1e-7; 975 of them, on 529 units, have no sign change of θ′ in 06a's scan.
5. **Development runs** (`07`–`10`, `state/dev/`): smoke1 and smoke2 at $b$, smoke3 at $p$, and
   `w2_reference.py` on two captured calls.
6. **Gate W0** (`12`–`15`): **holds.**
   - **Clause 1** (`15`). `e05c34f..da5c24d` outside `docs/` and `prompts/` changes 8 files:
     - `CLAUDE.md`;
     - three whose ASTs without docstrings equal `e05c34f`'s: `ComputeTargets/primitive_phase.py`,
       `main.py` and `ComputeTargets/tests/test_primitive_phase.py`;
     - four test modules whose ASTs differ: `test_g_phase_reanchor`, `test_phase_groups`,
       `test_quadsource_integral` and `test_tk_source_functions`. None of them is among the 102
       parent-tree modules the harness imported in W0's parent workers
       (`state/W0/p_tree_modules.json`). The only test modules imported are `ComputeTargets.tests`
       and `ComputeTargets.tests.test_main_plumbing`.
   - **Clause 2** (`14`). At $p$, three units are identical to 06a's in every computed value
     (06a's C0 (d) definition) and on C0 (c)'s terms. Their dispatcher counts are equal too
     (descriptive).

     | Row | rtol | Call value/abserr pairs | Regions (× 5 arrays) | Total |
     |---|---|---|---|---|
     | `low-G+r-1` (2-38302) | 1e-7 | 14 | 9 | `0x1.5e6b3999d1ab2p-101` |
     | `ctrl-low-1` (2-2095) | 1e-7 | 14 | 8 | `-0x1.e3272e29a4c5dp-102` |
     | `0-1971` | 1e-8 | 10 | 38 | `0x1.6b3058aae2397p-66` |

   - **Clause 3** (`14`). `b-1e-7-1-34599` was run in two fresh workers. The row is fixed by rule:
     the capture-list unit with the most captured calls. The two runs are identical in every
     computed value, including the analytic blocks, W3 and the captured calls' values.
   - No worker initialised Ray.
7. **Driver smoke test** (`16`, `state/smoke/`): 2 rows × 2 rtols, 4 units done.
8. **Predictions** (`18`, `19`): `state/predictions.md`, sha256
   `8df517969e4973aa222200c0e6ba5d35b796cca1cab323ab44fe17a4476f6e4e`, mtime
   2026-10-28T14:13:55.8Z, before the launch at 16:12:12Z. Part A is the prompt's lines 296–319
   byte for byte. Part B is W2's selection rule and the readings. The file is unchanged since
   (`41`).
9. **Pre-launch** (`22`–`24`): every script equals those W0 and the smoke test ran. The trees are
   clean, the registry held 27 runs, all finished, and the lock was free.
10. **Launch** (`25`) and **one check** (`26`): `running`, driver 8935, sh 8931, stderr empty.

### Rulings recorded before the launch

- **The analytic-block reading (W1's row rule and W4)**, as written in Part B of
  `state/predictions.md`. For an analytic block's call, "phase-limited in 06a" is read from b's
  premise counter, and "moved" is a failed premise check. The user's ruling, relayed by the
  coordinator on 2026-10-28, verbatim: *"For 06b, that reading is reasonable. I don't think it's
  worth spending a lot of time worrying about it."* `state/predictions.md` was not edited for it;
  its Part B already holds the reading. The reading's cross-check holds: on the group calls,
  "phase-limited in 06a" and "b asked the premise" agree on all 74,650 calls (`37`).
- **Deviation 17** below: see the coordinator's note there.

### After the run

- **Status** (`27`): `done`, 18,560 of 18,560 units, exit 0, stderr empty.
  - Every record is at `92bc62f` with empty status, and carries `run_census.py` `5d47334e…`.
  - Every unit's modules are 102 from its own tree, none from $p$ or the main checkout.
  - Every worker reported `ray_initialized` false, with no foreign module.
  - `/tmp/ray-sgk` held 67 entries before W0 and 67 after the run. This is descriptive only:
    R15 and R16 used it in the same period.
- **Over-budget pass:** none was needed, since no unit is over budget.
- **The store** (`28`–`32`): the copy's five files equal the certification after the run and after
  the fingerprint, with no journal file. `fingerprint_copy.py` at $b$ gives `0e32f24a…`, problems
  none.
- **Checkpoint:** `checkpoint.jsonl`, sha256
  `f934ea245c0086529fb9372a9ad457d4d8668114ce346dca5440404c766646ad`. The run directory is 1.8 GB:
  16 region pickles, 5 input pickles (975 captured calls), the events and the worker logs. It is
  kept in full.

### Predictions and verdicts

`state/predictions.md` (sha256 `8df51796…`, mtime 14:13:55Z, before the launch at 16:12:12Z)
holds §5's W1–W4 unchanged. They are copied here unchanged from the prompt:

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

| | Number | Verdict |
|---|---|---|
| **W0** | see "Before the launch", 6 | holds |
| **W1** | At 1e-7: 36,112 group calls not phase-limited in 06a, all 36,112 equal in value, abserr, regions and depth; 6,777 rows all of whose calls are such, all equal in `total` and `total_abserr`. At 1e-8: 35,605 of 35,605 calls and 6,177 of 6,177 rows equal. A further 1,918 rows at 1e-7 (2,358 at 1e-8) are excluded only by an analytic block's call that was phase-limited in 06a. (`37`) | **holds** |
| **W2** | 24 drawn (12 low, 12 mid, no shortfall; `state/w2_selection.json`). Every call's re-run from its reloaded inputs equals b's call bit for bit. $\|S_9 - S_{10}\|/$`abserr`$_b \le$ 0.0066 on all 24, so none is dropped. $\|{\rm value}_b - S_{10}\| \le$ `abserr`$_b$ on 7; it fails on 17, with err/abserr up to **19.7** (median of all 24: 1.79). (`35`, `37`) | **fails** |
| **W3** | 44 regions accepted phase-limited at 1e-7, in 27 units. All 44 are in analytic-block calls, all Levin, at depth 4–6. Each recomputed own floor equals the stored `phase_err` bit for bit. Halves' sum / own floor ranges from 0.504 to 0.891, so 0 violate the rule, re-scored from the hex floors as well. There are 0 recomputation errors, and the region arrays hold the same 44. (`37`, `out/w3_regions.csv`) | **holds** |
| **W4** | Moved calls: 1e-7, 1,213 group and 12,633 analytic, largest depth 14 and 18. 1e-8: 1,720 group and 17,722 analytic, largest depth 14 and **20**. Three moved calls at 1e-8 reach depth 20, all analytic blocks after the premise failed: `b-1e-8-2-1704` call 16 (Y1, 1,136 regions); `b-1e-8-2-13964` call 5 (J1, 31,885 regions, 127,539 evaluations, unit wall 159 s); `b-1e-8-3-34987` call 9 (J2, 14,293 regions, 69 s). 0 units over budget. (`37`) | **fails** |

**W2, examined further** (descriptive, after the score; `38`, `39`):
- **The reference is sound on the three calls cross-checked:** #21, the worst; #1; and #4.
  - $S_{11}$ agrees with $S_{10}$ to within 0.0018 of `abserr`$_b$.
  - On #1 the adaptive driver at `rtol` 1e-11 (215 regions) agrees with $S_{10}$ to 0.0016 of
    `abserr`$_b$.
  - On #21 and #4 the driver at `rtol` 1e-11 returns b's value exactly, with b's regions (4 and 2),
    each accepted as resolved. Its error against $S_{10}$ stays at 19.7 and 5.7 `abserr`$_b$. The
    error b's value carries is shared by those regions and their halves, so the step-(4) residual
    does not see it, and the regions' eq. (151) floors are below it.
- **Against each call's own request,** max(atol, rtol|value|):
  - b's error exceeds it on 4 of 24, by at most 3.42×;
  - 06a's exceeds it on 23 of 24.

  b's values are better, and its declared errors are smaller than its true errors on 17 of 24.
  06a's declared errors were within the truth on 23 of 24 (`err_06a/abserr_06a`: median 0.068,
  max 1.13). b's `abserr` fell to between 1/10 and 1/16,500 of 06a's.

### Descriptive items (`out/cmd/37`, unless noted)

- **How often the exit fired.**
  - **At 1e-7:** phase-limited group calls fell from 1,213 on 585 rows under 06a to 0 under b.
    `total_phase_limited` rows fell from 585 to 0. 32 analytic-block calls are phase-limited under
    b, on 27 rows.
  - **At 1e-8:** phase-limited group calls fell from 1,720 on 745 rows to 0. 800 analytic-block
    calls are phase-limited under b, on 442 rows.
  - **06a's that would have failed the premise:**
    - at the region level, matching 06a's region arrays to b's by start and end: all 1,213 (1e-7)
      and all 1,720 (1e-8) of 06a's phase-limited group-call regions fail it and are bisected under
      b;
    - by b's premise counter: on the group calls, 1,217 checks at 1e-7 and 1,733 at 1e-8, all
      failed;
    - on 06a's phase-limited analytic-block calls (by the reading): 12,633 at 1e-7 and 17,722 at
      1e-8, every one with a failed premise; 13,031 of 13,075 checks and 23,544 of 24,989 failed.
- **Depth**, row depth over group calls as 06a defines it:
  - at 1e-7 the largest is 12 under 06a and 14 under b; rows at ≥ 11 rise from 50 to 80;
  - at 1e-8 the largest is 13 under 06a and 14 under b;
  - including the analytic blocks, b's largest is 18 at 1e-7 and 20 at 1e-8 (3 rows; W4).

  Full histograms are in `37`.
- **The sample's errors before:** `err_06a/abserr_06a` has median 0.068, max 1.13 (1 of 24
  outside). `err_06a/request` exceeds 1 on 23 of 24 (`39`).
- **Totals** (585 rows moved at 1e-7, 745 at 1e-8):
  - Median $|\Delta{\rm total}|$/`total_abserr`$_{06a}$ by (band, k) is 0.02–0.17. Rows moving by
    more than `total_abserr`$_{06a}$: 32 at 1e-7 and 36 at 1e-8.
  - In every one of them `WKB_Levin` moved, half with `analytic_rad` too. Up to 2.2×10⁴ of
    06a's error, relatively up to 6.9×10⁻³.
  - The largest moves are group calls with an interior stationary point that 06a accepted
    phase-limited at depth 0, and b bisects to depth 11–12. Of the group calls moving more than
    their 06a abserr, 30 of 38 at 1e-7 (35 of 47 at 1e-8) have a sign change of θ′.
  - `total_abserr`$_b$/`total_abserr`$_{06a}$ on the moved rows has median 0.36 at 1e-7 and 0.22
    at 1e-8.
  - `out/moved_calls.csv` and `out/census_rows_06b.csv.gz` hold the details.
- **Stationary-point rows** (06a's 720; 07's, not scored): their totals moved on 323 rows at 1e-7
  and 370 at 1e-8. Row depth is unchanged on 690 (686) and up by 2–6 on the rest. Their
  phase-limited group calls go to 0.
- **Convergence:**
  - `total_converged` false on 2,578 rows under 06a and 2,349 under b at 1e-7; 2,638 and 2,388 at
    1e-8.
  - Unconverged group calls: 6,964 and 5,788 at 1e-7; 7,647 and 6,186 at 1e-8.
  - `convergence_kinds.py` (06a's changed copy, `--tree b`) at 1e-7: bound only 4,018, resolution
    1,770, phase-limited 0. Under 06a: 4,014 / 1,737 / 1,213. At 1e-8: 3,784 / 2,402 / 0, against
    3,745 / 2,182 / 1,720. No row is capped (`40`, `out/convergence_kinds_b.txt`).
- **Cost:**
  - Wall time 51 min against 06a's 45.5.
  - The Levin timed sections sum to 2,477 s at 1e-7 and 3,375 s at 1e-8, against 2,399 s and
    2,760 s.
  - The largest unit is 2.4 s at 1e-7 and 159 s at 1e-8 (the depth-20 call), against 0.8 s and
    1.0 s.
  - Cost is not an argument (README §5 rule 4).

### Deviations from the prompt (numbered and classified)

1. **`prov.py`:** trees `p` and `b`, 06a's run and records, and `WIDER` one level deeper.
   STRUCTURALLY REQUIRED.
2. **`worker.py`, module evidence:** the tree evidence names `AdaptiveLevin.levin_quadrature` and
   the premise check; each unit records its modules by root (`modules`); `bye` lists the tree's
   modules. STRUCTURALLY REQUIRED (the brief; W0 clause 1).
3. **`worker.py`, per-call records:**
   - per call, `n_phase_limited_regions`;
   - the code's premise check, counted by a pass-through at `LQ._roundoff_exit_premise_holds`
     during the timed section, at b only;
   - the analytic blocks' calls summarised (`other_calls`) and their region arrays pickled
     (`other_regions`).

   IMPLEMENTATION CHOICE: 06a recorded no per-call result for the analytic blocks, and W1's row
   rule and W4 need them.
4. **`worker.py`:** W3's recomputed floors. STRUCTURALLY REQUIRED.
5. **`worker.py` and new `make_capture.py`:** W2's inputs, captured for all 975 candidate calls.
   STRUCTURALLY REQUIRED; capturing every candidate is an IMPLEMENTATION CHOICE.
6. **`run_census.py`:** `--tree`, unit ids `<tree>-…`, and `w3` and `capture` in each 1e-7 unit.
   STRUCTURALLY REQUIRED.
7. **`launch.py`:** prompt `06b`, purpose and scope. STRUCTURALLY REQUIRED.
8. **`harness.py`:** `job_name="levin-phase-floor-06b"`, as the brief asks. datastorekit 0.5.0
   stores it and reads it nowhere (`11`). IMPLEMENTATION CHOICE.
9. **`rows.json` copied from 06a** (same sha256). A3 v2 itself is neither read nor hashed; only the
   copy is checked. IMPLEMENTATION CHOICE.
10. **New `w0.py` and `w0_clause1.py`.** IMPLEMENTATION CHOICE throughout:
    - the determinism row is fixed by rule;
    - there is one parent worker per unit;
    - clause 1's import list is taken from W0's parent workers (the reproduction run), not from a
      separate dry run.
11. **`out/cmd/13`: the first W0 check stopped** (ModuleNotFoundError) at the clause-3 capture
    comparison, which unpickles records that need the tree on the path. Clause 2 had already
    printed "holds". It was re-run with `PYTHONPATH` set to b as `14`; nothing else was re-run.
    UNINTENDED DRIFT.
12. **New `w2_reference.py`.** STRUCTURALLY REQUIRED (the prompt allows it). Its method choices are
    IMPLEMENTATION CHOICEs: pieces made by the driver's own bisection, each piece evaluated with
    `_adaptive_levin_subregion`, `math.fsum`, and the re-run check.
13. **Development runs** `07`–`10` and the driver smoke test `16`. IMPLEMENTATION CHOICE.
14. **`out/cmd/17` is a mistaken command:** it wrote the prompt's lines to `/tmp/__x`, deleted the
    file, and checked nothing. `check_predictions.py` (`18`) replaced it. UNINTENDED DRIFT.
15. **`state/predictions.md` was edited once, before the launch:** its header named `out/cmd/17` as
    the byte check, and now names "18 and 19". Part A is unchanged (`19`). UNINTENDED DRIFT.
16. **New `check_predictions.py` and `scripts_equal.py`.** IMPLEMENTATION CHOICE.
17. **The brief says no census process imports `ray`.** 06a's harness and the production modules
    import the `ray` package. Ray is stood in and never initialised: every worker's
    `ray_initialized` is false. STRUCTURALLY REQUIRED reading.
    **The coordinator, 2026-10-28, before the launch:** the brief's wording "no census process
    imports ray" was stricter than the binding rule, which is that no Ray session appears (none
    started by the census, none under /tmp/ray-sgk from it). The laptop classifies this deviation
    as arising from the brief.
18. **Scripts are not `black`-formatted:** they are records of the run. IMPLEMENTATION CHOICE.
19. **No over-budget pass:** no unit was over budget. Recorded for completeness; it is not a
    departure from the prompt.
20. **New `analyse06b.py`.** Its first `select` (`33`) stopped with a KeyError, because it parsed
    the rtol's `-` in a unit id as a separator. That was fixed with `rowkey()`, and the select was
    re-run as `34`. The selection was written only by `34`. UNINTENDED DRIFT.
21. **The first `report` (`36`) stopped** while writing the per-row file, on rows with no group
    call, after printing every score. That was fixed, and two descriptive blocks were added (which
    component of a row's total moved, and the W4 calls' detail). It was re-run as `37`; the scores
    are the same. UNINTENDED DRIFT for the stop; the additions are IMPLEMENTATION CHOICE.
22. **New `w2_crosscheck.py` (`38`) and `w2_request.py` (`39`):** descriptive checks of W2's
    reference and of the errors against each call's request, run after W2 was scored. They are
    not part of the rule. IMPLEMENTATION CHOICE.
23. **`convergence_kinds_06a.py` run with `--tree b --capped-run <this run> --capped-tree b`**
    (`40`). `out/convergence_kinds_b.txt` is its output without the per-record input lines.
    IMPLEMENTATION CHOICE.
24. **`out/cmd/41` ended with exit 1** from GNU `stat` (BSD flags; 06a's deviation 17 again). Every
    other line of it is recorded. UNINTENDED DRIFT.

Deviations 11, 14, 15, 20, 21 and 24 are the UNINTENDED DRIFT. None of them touched a store, a tree,
or any record outside `var/levin-phase-floor-06b/census/` and the run directory.

### Verification performed
- **The trees:** $b$ at `92bc62f`, `status --short --ignored` empty at W0, at the launch, on all
  18,560 records and at the end (`41`). $p$ at `da5c24d`, likewise. The main checkout is at
  `28ebaf0`, status empty, throughout.
- **The copy:** byte-identical to `copy_files_certified.txt` before W0, before the launch, after the
  run and after the fingerprint. `0e32f24a…` before and after.
- **The run:** 18,560 records, every unit `done`, every pickle record matching its sha256 (read in
  `37` and `40`), and `stderr.log` empty.
- **No Ray:** every census worker reported `ray_initialized` false.
- **06a's records were only read:** its checkpoint is still `c13685b4…` (`33`).

### Observations not acted on

All are in `issues.md`:
- **C1:** W2's failure. b's `abserr` understates its error on calls the premise bisects.
- **C2:** W4's failure. Three analytic-block calls chase to depth 20 at 1e-8.
- **C3:** 06a's stationary-point calls move under b by up to 6.2×10⁴ of their 06a abserr (2.5×10⁴ at 1e-7);
  they are 07's.
- **C4:** the analytic blocks are where the exit fired most, and 06a's census did not record them.

### State handed on
- **Nothing is running.** The run is `done`. No lock was taken. Nothing was written in either
  worktree or the main checkout outside `var/levin-phase-floor-06b/census/` and the run directory.
- **The scratch copy is unchanged** (`0e32f24a…`) and is still kept for the user to delete.
- **The run directory** `var/runs/levin-phase-floor-06b-census-20261028T161212/` (1.8 GB) is kept
  in full: checkpoint, region pickles and captured inputs, for 07's census.
- **For the orchestrator:** copy `log.md`, `verification.md`, `issues.md`, `scripts/`, `out/` and
  `state/` to `docs/levin-phase-floor/06b/census/`. Leave out `out/cmd/40` (3.6 MB), and the
  pickles under `state/dev/`, `state/W0/` and `state/smoke/`, which hold about 50 MB of captured
  inputs.
