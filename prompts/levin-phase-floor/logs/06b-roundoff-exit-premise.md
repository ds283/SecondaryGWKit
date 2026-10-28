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

*(Not yet scored. A note for §5, not a score: on the laptop's LiouvilleGreen draws, 1 of the 1,026
calls that moved reached `depth_max`, which is W4's pattern — `[06b-premise-check-can-chase-a-flat-residual-to-depth-max]`.)*

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
