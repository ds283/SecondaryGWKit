# levin-phase-floor 06c — global error control in the Levin driver

> **§1–§3 only** (the code, the tests and these records), written by the executing agent on the
> laptop. §4 (rehearsals R17 and R18) and §5 (the census) run later on the Mac Studio. The
> orchestrator appends their record to this log.

**Subject:** prompt 06c §1–§3, on the laptop, in the main checkout
`/Users/ds283/Documents/Code/SecondaryGWKit`, branch `handover-remedial`.
- **Dispatch SHA:** `3f2a36d`. Five orchestrator commits landed on the branch while the work ran,
  each amending this prompt (its §8): `8b08b8d` (the stall rule amended to (b2), after the first
  stop), `cf7913b` (the stall rule withdrawn, after the second), `9486375` (the trust check one
  level down, after review of the first code commit), `63b7721` (a noise level for the check, after
  review of the second) and `fef401c` (the noise level withdrawn, after the third stop and the (b′)
  measurement). `git log --oneline 3f2a36d..HEAD` before staging shows those five and nothing else.
  The stashes were not touched, and `var/levin-phase-floor-07/` was not read.
- **Two earlier code commits, never pushed.** `79ae8c7`, made at `cf7913b` (the global budget
  without the trust check), and `c85f753`, made at `9486375` (`79ae8c7` plus the check). Each passed
  its review; each time the orchestrator saved it as a patch, reset the branch and amended the
  prompt. **This commit re-lands `c85f753`'s content unchanged in code** (`git cherry-pick
  --no-commit c85f753`; `git diff c85f753 -- AdaptiveLevin ComputeTargets RunDrivers
  docs/levin-phase-floor/06c` is empty), with these records updated; its parent is `fef401c`.
- **The gate.** `git status --short` was empty at dispatch. `runregistrykit list` showed nothing
  `running` or stale on the laptop (61 runs: 59 finished, 2 `unknown`, pre-registry). The seven
  roots were taken at `3f2a36d` before any change (Verification).
- No Ray was started, no run was registered, no store was opened, and nothing was written under
  `var/`.

**Commit:** the code commit (this log's own; hash in the board's 06c row once pushed).

**Result:**
- **§1.** `_adaptive_levin` keeps every leaf of its partition with its value and its error
  (`used_interval.total_err`), and bisects the refinable leaf with the largest error until
  $\sum e_i \le \max(\texttt{atol}, \texttt{rtol}|\sum v_i|)$; a leaf is not refinable at
  `depth_max` or when floor-limited (06b's premise). **Before it may stop, each leaf's estimate is
  checked one level down** (§8, the third amendment): trusted leaves ($r < 1$) declare
  $\max(\epsilon_P, \Delta_1)/(1-r)$; untrusted ones are refined first, whatever the budget, and one
  left at `depth_max` declares $\max(\epsilon_P, \Delta_1+\Delta_2)$ and makes the call unconverged.
  `_local_atol` and per-region acceptance are gone. No stall rule: one was specified, amended once
  and withdrawn after two stops.
- **§2.** `AdaptiveLevin/tests/test_levin_global_budget.py`, 15 methods, all passing: tests 1–8,
  tests 12a–12e, a NaN-ordering test and an exact-running-sum test. The parent misses on tests 1,
  2 and 3; the new code converges, within the request, honest, on all three. **Test 12a** (the
  Gaussian bump that was dishonest under `79ae8c7`) is honest — by a margin of $1.8\times10^{-4}$
  relative. **12e:** none of the 25 problems `test_levin_quadrature.py` drives is honest at the
  parent and dishonest under the new code.
- **The breakages** (i)–(iv) are each caught by the test the prompt names: (i) by test 1, (ii) by
  test 4, (iii) by test 3, (iv) by test 12a.
- **Existing tests.** Four assertions changed, each of a kind §2 test 9 allows, with one
  comment-only change; listed below.
- **Test 10.** 160 units: every test passes at the parent and under the new code; no stop. Of
  4,320 Levin calls, converged 1,207 → 3,254 (2,052 False → True, **5** True → False); **230
  calls reach `depth_max`** (1 at the parent); 241 leaves returned untrusted, in 19 calls, none
  converged; solves 133,092 → 5,936,892 (×45), wall time 65 s (06b) → 1,031 s. **12e's table:** at
  the integral level `test_JJJ` and `test_YJJ` are honest on all 140 cases under both;
  `test_YJJ_log_singularity` is dishonest on 259 of 800 cases at the parent (44 flagged converged)
  and 235 under the new code (228 flagged converged), and **5 cases honest at the parent are
  dishonest under the new code, all 5 flagged converged** — a finding for the user, not a stop.
- **The seven roots.** Six equal the baseline; `AdaptiveLevin` rose by 15, the new module's 15
  methods. All OK.
- **Three stops, two review findings and five amendments** on the way (below, and the prompt's
  §8). The last two: a noise level for the check, $\nu = 8uM$, could not reach 12f's residual (the
  third stop); a phase-scaled level (b′), measured as a scratch variant, overstated the noise by
  $10^2$–$10^6$ and made five converged calls unconverged, while barely moving item 1. The user then
  withdrew the noise level: **06c ships `c85f753`'s code, with no noise rule**, and this commit
  re-lands it unchanged.
- `[06c-budget-can-stop-on-a-pre-asymptotic-residual]`, opened from `79ae8c7`'s finding, is
  **narrowed, not closed**: the check makes 12a honest, but its widening is an extrapolation, not
  a bound; item 1's near-singular cases are input conditioning; the noise question waits for after
  07.

## What shipped

**`AdaptiveLevin/levin_quadrature.py`.**
- **`_adaptive_levin`, rewritten as one budget for the call** (§1 items 1–7):
  - **The leaves.** `_levin_leaf` (replacing `_levin_interval`): interval, depth, sequence number,
    value, its `used_interval` (`region`, whose `total_err` is $e_i$), whether it is a fallback
    leaf, its halves' solves when in hand (`dataL`/`dataR`), its grandchildren's solves once
    checked, whether it has been checked, a heap version, its own solve only if `build_p_sample`,
    and the three histories for the depth-18 log. A leaf is solved, and its error known, when it is
    made: a Levin leaf has its two halves for the step-(4) residual then (solved, or carried from
    its parent's check); a fallback leaf's error is its nested pair.
  - **The order.** A heap of the refinable leaves keyed `(untrusted first, -e_i, start, seq,
    version)`: an untrusted leaf before every trusted one, then the largest error, ties by the
    smaller start and creation order, so the order is total and deterministic. When a leaf's error
    or trust changes it is pushed again under a new version and the stale entry is skipped. A leaf
    at `depth_max`, or floor-limited, is never a candidate. **NaN rule:** `_error_for_ordering()`
    reads a NaN error as $+\infty$, in the order and in the sums; unit-tested
    (`test_nan_error_is_ordered_as_infinite`), no path known to produce one.
  - **The sums.** Running sums of the finite errors and of the values are held exactly
    (`_ExactSum`, Shewchuk's partials), with infinite errors counted separately. When the running
    test passes it is confirmed on the returned sums (`_sum_leaves_ascending()`: plain addition in
    ascending x) before anything else happens. A DEBUG line at the end of each call reports the
    budget test against the returned sums, rejections, and the trust check's counts.
  - **The trust check** (§8, the third amendment), `_check_leaf()`: from the leaf's halves (solved
    on demand on the fallback branch) and its four grandchildren,
    $\Delta_1 = |V_P - (V_L+V_R)|$, $\Delta_2 = |(V_L+V_R) - ((V_{LL}+V_{LR})+(V_{RL}+V_{RR}))|$,
    $r = \Delta_2/\Delta_1$ (0 if both are 0; $+\infty$ if only $\Delta_1$ is, or either is not
    finite). Trusted ($r < 1$): declared resolution error $\max(\epsilon_P, \Delta_1)/(1-r)$.
    Untrusted: refinable whatever the budget, chosen first; at `depth_max` it declares
    $\max(\epsilon_P, \Delta_1+\Delta_2)$. Taken lazily: when the budget passes and is confirmed,
    every leaf not yet checked and not floor-limited is checked, in ascending x, and the loop
    continues; it stops only when the budget passes with every leaf checked or floor-limited and
    no refinable leaf untrusted. The grandchildren become the children's halves if the leaf is
    bisected. Values are unchanged. A leaf later found floor-limited is exempt: its declared error
    reverts to its own estimate (its $\Delta$s stay recorded).
  - **Bisection.** A Levin leaf's children are its `dataL`/`dataR`; a fallback leaf's are its halves,
    solved when it is chosen or checked. Each solve is counted once, where it is made.
  - **Floor-limited and capped leaves** as `79ae8c7`: floor-limited only when chosen (premise from
    the halves; the bypass skips it), and never chosen again.
  - **Removed:** `_local_atol` and every use; `resolved` as an acceptance test. `relerr` stays, a
    diagnostic, its denominator floored at `atol`.
  - **Unchanged around the loop:** the progress notification ($V$, leaves, refinable leaves),
    `_write_progress_data` (handed the refinable leaves, largest error first), the depth-18 log,
    the two health warnings, the returned `regions` sorted by start. The not-converged warning names
    leaves at `depth_max` (and how many untrusted) and floor-limited ones.
- **`used_interval`** gains `delta1`, `delta2`, `convergence_ratio`, `trusted` (None if the check was
  not taken, as for a floor-limited region) and `declared_resolution` (the widened or capped
  resolution error, which `total_err` uses in place of `abserr` when set); `abserr`,
  `abserr_resolution` and `abserr_fallback` stay the raw estimates. `__str__` shows `r` and
  "(UNTRUSTED)".
- **Result:** one new key, `num_untrusted_regions`; `converged` is the budget **and** no untrusted
  leaf returned.
- **Docstrings and comments** (§1 item 8): the module docstring (global error control and the
  check; the round-off bullet's exit paragraph; the fallback bullet), the comment on
  `_LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO`, `_roundoff_floor()`'s paragraph on an infinite floor,
  `used_interval`, `_adaptive_levin`'s docstring (the budget, the order, refinable, the check, why
  the per-region scheme failed, the stall rule withdrawn), the loop's comments, the result dict's
  comments, `_notify_progress`, and `adaptive_levin_sincos`'s docstring.
- **Not changed:** eq. (151), `_levin_G0_G1`, `_declared_endpoint_phase_err`, what `"phase_err"`
  holds, 06b's premise and $\rho$, the bypass's reach, the subregion solvers, the C2 gate, the
  p-mode gate, `abserr_truncation`, `DEFAULT_LEVIN_MAX_DEPTH`, the Chebyshev order, the input
  validation, the existing result keys.

**`AdaptiveLevin/tests/test_levin_global_budget.py`** (new; no Ray, no datastore). References are
closed forms in mpmath; each reference's own error (30 against 40 digits) is asserted below
$10^{-6}$ of what is tested against it. The closed forms of tests 1, 5 and 6 agree with subdivided
`mpmath.quad` to $3.8\times10^{-24}$ or better (`parent_demo.txt`). Parent numbers are in the
docstrings with `3f2a36d`; the module does not run the parent. "Request" is
$\max(\texttt{atol}, \texttt{rtol}|I|)$.

| test | problem | parent (`3f2a36d`) | new |
|---|---|---|---|
| 1 cancellation, fallback | 07's family (R), $x_s = 0.7236$, $c = 10^4$, $f = (-\cos 3x, \cos 3x)$, `rtol` $10^{-7}$ | 8 regions, depth 5; $\sum\|v_i\| = 137\|I\|$; **unconverged** at true error **0.72** × request: **a miss** | converged, honest, within the request (`parent_demo.txt`) |
| 2 cancellation, Levin | $f = (1/x, 0)$, $\theta = 100.5\pi x$ on $[1, 100.0049751\ldots]$, order 16, `rtol` $10^{-8}$ | 7 Levin regions; $\sum\|v_i\| = 310\|I\|$; **unconverged** at true **0.9987** × request: **a miss**, narrowly | converged on Levin leaves only, honest |
| 3 inflated floor | $\theta = 1000e^x$ on $[0, 20]$, `rtol` $10^{-7}$ | 1 region; abserr $9.703\times10^{-7}$ (the floor), **unconverged**, true $2.09\times10^{-11}$ = 0.37 × request: **a miss** | converged, abserr ≤ `rtol`$\|I\|$, honest |
| 4 floor still stops | 06b's test 2 (a), (b) | 1 region, phase-limited, depth 0 | the same |
| 5 unattainable | $\|x-1/3\|\sin 100x$, `depth_max` 6 | unconverged | returns unconverged; every leaf at the cap or floor-limited; the kink's leaf at depth 6; honest |
| 6 infinite floor | $\theta = 10^4(x-x^2)$, no `theta_deriv` | **abserr $+\infty$** | abserr finite, no infinite leaf, honest |
| 7 total near zero | $\tanh(30x)\cos(20x)$ on $[-1,1]$ | depth 0 | abserr ≤ `atol`, converged, value within `atol` (depth 0, by symmetry) |
| 8 determinism | tests 1 and 3, twice | — | bit-identical |
| 12a bump | $e^{-400(x-1.1)^2}\sin(3\times10^4x)$, `atol` $10^{-10}$ | 4 regions, $-3.437\times10^{-12}$, abserr $1.888\times10^{-11}$, honest | 5 leaves, value $-8.5626\times10^{-11}$ (as `79ae8c7`), **abserr $8.5641\times10^{-11}$** (`79ae8c7`: $8.219\times10^{-11}$) against true $8.5626\times10^{-11}$: **honest**, converged; the leaf $[0.75, 1.5]$ has $r = 0.0403$ |
| 12b before resolution | SincIntegral, `rtol` $10^{-10}$, order 12 | converged, honest | converged, honest; every leaf trusted, $r$ from 0.0013 to 0.911 |
| 12c stationary-point line | family (R), $x_s = 0.382$, $c = 10^5$, `rtol` $10^{-7}$ | converged, true $1.07\times10^{-12}$ | converged, true $1.07\times10^{-12}$, abserr $1.38\times10^{-12}$, 12 leaves |
| 12d untrusted at the cap | spike $e^{-((x-0.4)/10^{-3})^2}\cos(0.1x)$, `depth_max` 0 | 1 region, value $8.2\times10^{-9}$, abserr $8.2\times10^{-9}$, **converged, dishonest** (true $1.77\times10^{-3}$) | the root untrusted, $r = 2560$, declares $\Delta_1+\Delta_2 = 4.81\times10^{-3}$, flagged, **unconverged**, honest |
| 12e nothing honest becomes dishonest | the 25 driven problems with a reference | every one honest | every one honest (asserted) |

Also `test_nan_error_is_ordered_as_infinite` and `test_running_sum_is_exact` (§1 items 2 and 3).

**12b, $r$ along SincIntegral's leftmost line** (each level's own solves, `parent_demo.txt`): depth 0
0.847, 1 0.532, 2 0.309, 3 0.126, 4 0.047, 5 0.041, 6 0.0036, 7 0.0022, 8 0.143, 9 0.093 — every
value below 1, so the line the stall rule broke is trusted at every level, and its returned leaves
are widened by at most $1/(1-0.911)$ (the leaf $[1.387, 1.773]$).

**Existing tests changed** (§2 test 9; before and after measured with the parent module and the
final driver):

| test | assertion before | after | numbers, parent → new |
|---|---|---|---|
| `test_levin_quadrature.test_GRZIntegral` | λ = 100: some returned region takes the Levin rule | removed for λ = 100, with a comment; λ = 10 and 1000 unchanged | 24 regions, some Levin → 102 leaves, all fallback, all floor-limited. The integral is $\approx 10^{-17}$, so the budget is `atol`. Which rule the regions take: allowed |
| `test_levin_quadrature.test_mode_filter_truncation_visible_in_abserr` | `tight["num_regions"] == 1` | removed, with a comment; the loose call's `== 1` and every other assertion unchanged | the tight call (`rtol` $10^{-8}$): 1 region → **1,279 leaves**, depth 15, 3 → 10,231 solves, converged and honest (true $1.1\times10^{-16} \to 4.0\times10^{-17}$). Its root's residual ($2.1\times10^{-16}$) is round-off noise far above eq. (151)'s floor ($2\times10^{-20}$), so it is not floor-limited, and the noise's $r$ (1.96 at the root) reads as untrusted (Observations 3). A region count: allowed |
| `test_levin_roundoff_exit.test_exit_against_its_premise_fallback_branch` (06b's test 3) | `num_phase_limited_regions == 0` | every region flagged phase-limited satisfies 06b's premise, from its own halves | 2 regions at depth 1, none phase-limited (06b) → 4 leaves at depth 2, all phase-limited, abserr $4.59\times10^{-7}$, true error 0. Depth, honesty and abserr-halved assertions unchanged. Allowed by name |
| `test_levin_roundoff_exit.test_nothing_else_moves` (06b's test 4) | bit-identity across the bypass, evaluation count included | the same without the evaluation count; evaluations asserted no smaller without the bypass | only the count differs, on GRZ(100) (203 → 407) and the stationary-phase oracle (193,893 → 218,735). Allowed |
| `test_levin_quadrature.test_converged_flag` | comment on the length share | comment only | no assertion changed |

**Callers, comment-only** (§0.3), each checked by AST with docstrings stripped (comments are not in
the AST) against `3f2a36d`, identical: `ComputeTargets/QuadSourceIntegral.py`'s `atol_sub` comment,
and `RunDrivers/quadsource_atol_sweep.py`'s module docstring on `local_atol`.

**`docs/levin-phase-floor/06c/calibration/`.** Every output's header records the SHA-256 of its
script, of `prov.py`, of `levin_quadrature.py` and of each input, the `HEAD`, the dirty tree's
`git status --short`, and the parent module's SHA-256 (`1ba4635e…`, 06b's driver).
- `scripts/prov.py` — 06b's, retitled, plus `load_parent()` (the parent from
  `git show 3f2a36d:…`, imported under another name) and `DebugCapture` (the driver's DEBUG line:
  budget-test disagreements, rejections, trust-check counts).
- `scripts/parent_demo.py` → `out/parent_demo.txt`: tests 1–7 and 12a–12d under both codes, the
  reference cross-checks, 12b's leftmost-line ratios, and tests 11 and 12e.
- `scripts/lg_seeded_draws.py` → `out/lg_seeded_draws.jsonl`, `out/lg_seeded_draws_report.txt`:
  test 10, with 12e's table.

### Tests 11 and 12e: the 25 driven problems

Parent → new (`parent_demo.txt`). Converged 13 → 13, the same 13; **honest 25 → 25**, none honest at
the parent and dishonest under the new code. Problems where nothing but the solve count moved
(single-region problems now spend 4 more solves on the check) are omitted.

| problem | converged | regions | depth | solves | trust check: checks / found untrusted |
|---|---|---|---|---|---|
| SincIntegral (and vectorized_sinc) | T → T | 10 → 9 | 8 → 8 | 39 → 71 | 9 / 0 |
| GRZIntegral(10) | T → T | 8 → 11 | 3 → 4 | 15 → 87 | 14 / 10 (7 then floor-limited) |
| GRZIntegral(100) | F → F | 24 → 102 | 6 → 7 | 51 → 407 | 0 / 0 |
| GRZIntegral(1000) | F → F | 4 → 382 | 2 → 11 | 15 → 1,527 | 0 / 0 |
| converged_flag_bump | T → T | 4 → 5 | 3 → 3 | 15 → 39 | 7 / 2 |
| stationary_phase | F → F | 20 → **96,941** | 10 → **20** | 67 → **218,735** | 0 / 0 (budget never met) |
| mode_filter_tight | T → T | 1 → **1,279** | 0 → 15 | 3 → **10,231** | 2,541 / 528 |
| SinIntegral, CosIntegral | — | — | — | +4 or 0 | the root found untrusted, then floor-limited (phase-limited 0 → 1) |

The running test and the returned budget never disagree, and the confirmation never rejected a
passing running test. The stationary-phase oracle (`atol` = `rtol` = $10^{-10}$, a request of
$6.9\times10^{-14}$ below what the floors near $x = \tfrac12$ allow) is refined until every leaf is at
`depth_max` (84,514) or floor-limited (12,427); its budget is never met, so the check is never taken
there. The AdaptiveLevin suite's wall time rises from ~0.1 s to 109–116 s.

### Test 10: the LiouvilleGreen draws

`lg_seeded_draws_report.txt` (from `lg_seeded_draws.jsonl`; wall time 1,031 s on 8 processes). The
parent is `3f2a36d`'s own module (`1ba4635e…`, 06b's driver); the new code's is `4161004e…`.

- **Outcomes:** all 80 (seed, test) pairs pass under both. **No stop.**
- **Calls:** 4,320 under each; none bit-identical. Converged (parent, new): T,T 1,202; T,F **5**;
  F,T 2,052; F,F 1,061. Phase-limited calls 133 → 3,652 (every leaf the budget needed and found
  floor-limited is flagged, and the trust check's untrusted leaves are chosen first, which finds
  more of them floor-limited).
- **The five True → False:**

  | test, seed, call | group, `rtol` | parent | new |
  |---|---|---|---|
  | `test_YJJ` 6, #7 | 4, $10^{-10}$ | 25 regions, abserr $8.14\times10^{-13}$ | 53 leaves, all floor-limited, abserr $1.84\times10^{-12}$ (floors above the request, as under `79ae8c7`) |
  | `test_YJJ_log_singularity` 0, #113 | 2, $10^{-8}$ | 8 regions, $2.95\times10^{-10}$ | 48 leaves, depth 20, **1 untrusted** |
  | `test_YJJ_log_singularity` 3, #89 | 2, $10^{-8}$ | 8, $1.96\times10^{-10}$ | 87, depth 20, **2 untrusted** |
  | `test_YJJ_log_singularity` 7, #143 | 4, $10^{-8}$ | 10, $1.27\times10^{-10}$ | 112, depth 20, **14 untrusted** |
  | `test_YJJ_log_singularity` 7, #151 | 4, $10^{-8}$ | 10, $1.27\times10^{-10}$ | 123, depth 20, **13 untrusted** |

  The values move by $2.5\times10^{-15}$ to $1.3\times10^{-13}$. Four now report unconverged because
  leaves at the cap fail the check: honest flags where the parent's were not supported.
- **Cost:** regions 48,231 → 1,774,775; solves 133,092 → 5,936,892. The check took 102,113 checks in
  36,356 passes, found 48,340 leaves untrusted (refined), and returned 241 untrusted (in 19 calls,
  none converged, all at depth 20). 995,198 leaves were returned unchecked — capped leaves of calls
  whose budget was never met (deviation 29).
- **`depth_max`:** 230 calls reach depth 20 under the new code (80 `test_abserr_bounds_truth`, 4 per
  seed; 66 `test_JJJ`; 16 `test_YJJ`; 68 `test_YJJ_log_singularity`), against 1 at the parent (seed
  15's `test_JJJ` call 6). 226 are unconverged under both; the other 4 are four of the True → False.
- **Per seed** (in the report): every seed's largest depth is 20 under the new code; converged
  counts go from 46–81 to 149–171 of 216.
- **Largest printed relative error against the closed form,** parent / new: `test_JJJ`
  $9.29\times10^{-12}$ / $1.71\times10^{-11}$; `test_YJJ` $1.61\times10^{-8}$ / $3.39\times10^{-8}$;
  `test_YJJ_log_singularity` $2.93\times10^{-4}$ both. The per-region scheme over-delivered on
  cancelling sums; the budget delivers what is asked.
- **12e, honesty at the integral level** (each printed case's true error against its reported
  abserr; the report lists every case dishonest under either code):

  | test | cases | dishonest at the parent (flagged converged) | dishonest under the new code (flagged converged) | honest → dishonest |
  |---|---|---|---|---|
  | `test_JJJ` | 100 | 0 | 0 | 0 |
  | `test_YJJ` | 40 | 0 | 0 | 0 |
  | `test_YJJ_log_singularity` | 800 | 259 (44) | 235 (228) | **5, all flagged converged** |

  The five are all $(Y_0, J_2, J_2)$ at $\epsilon = 10^{-4}$ or $10^{-5}$ from the singularity (seed 1
  #26, seed 8 #28, seed 14 #27, seed 18 #26, seed 19 #27), true/reported 1.06–2.2 (for example
  seed 8: true $1.68\times10^{-9}$ against $7.54\times10^{-10}$; the parent declared
  $1.53\times10^{-9}$ against a true $1.47\times10^{-9}$). The test's comments put this band's
  error in near-singular conditioning of the integral and its reference, not in the quadrature,
  and "reported abserr" there is the sum of four Levin calls' errors; no single call can be checked
  against a reference here. 29 cases go the other way, dishonest → honest. **A finding for the user,
  not a stop** (§8, 12e). The test passes under both codes; `test_abserr_bounds_truth` bounds on 7
  of 7 under both (true/reported $3.3\times10^{-6}$–$1.1\times10^{-3}$ at the parent,
  $4.3\times10^{-5}$–$5.0\times10^{-2}$ under the new code).
- The running test and the returned budget never disagree, and the confirmation never rejected a
  passing running test.

## Predictions and verdicts

§1–§3 score nothing. Prompt 06c §4's Y1 and §5's X1–X5 are copied below unchanged, before §4 and
§5 run, as §3 asks. They are **not yet scored**. (X6, added by the first amendment, was withdrawn
with the stall rule by the second.) §5 also gains a descriptive item on the trust check (the third
amendment).

**§4, Y1 — not yet scored:**

   - **Y1 — consistency.** **Rule:** every row with finite `total_abserr` in both stores has
     $|{\rm total}_{18} - {\rm total}_{17}| \le {\rm total\_abserr}_{17} + {\rm
     total\_abserr}_{18}$. Rows with an infinite `total_abserr` in either are counted separately.
     **Holds** if no row violates it. A violation is a finding for the user, not by itself a stop.

**§5, X1–X5 — not yet scored:**

**Predictions, fixed before the run** (README §5 rule 3). Copy them into the log unchanged.
"Moved" means the call's value, abserr, region count or maximum depth differs from 06b's.
- **X1 — honest where it moved.**
  - **The sample:** 24 calls at `1e-7` that moved, with no sign change of $\theta'$ in 06a's scan
    (`spscan.py`): 12 that 06b reported unconverged with no region at `depth_max`, none
    phase-limited and a finite abserr (or all such, if fewer), and the rest stratified by band
    (low, mid) and by $k$. The selection rule is fixed in `state/predictions.md` before the run.
  - **The reference:** W2's, Levin forced to $2^9$ and to $2^{10}$ equal pieces, with each piece's
    own floor reported. The two must agree to within a hundredth of $g$'s `abserr`; a call where
    they do not is reported, and dropped from the rule.
  - **The rule:** $|{\rm value}_g - {\rm ref}| \le$ `abserr`$_g$ for every call left in the sample.
- **X2 — budget misses converge.**
  - **The rule:** every Levin call that 06b reported unconverged, with no region at `depth_max`, no
    phase-limited region and a finite abserr, is `converged` under $g$, outside 06a's stationary-
    point rows.
  - **Holds** if true at both rtols.
- **X3 — no infinite declared error.**
  - **The rule:** no call under $g$ declares an infinite abserr unless one of its leaves is at
    `depth_max`.
  - **Holds** if true at both rtols.
- **X4 — the rows agree.**
  - **The rule:** outside 06a's 720 stationary-point rows, every row with finite `total_abserr`
    under both has $|{\rm total}_g - {\rm total}_{06b}| \le {\rm total\_abserr}_{06b} + {\rm
    total\_abserr}_g$.
  - **Holds** if true at both rtols.
- **X5 — the cap.**
  - **The rule:** at both rtols, no call that moved reaches `depth_max`. An over-budget unit counts
    as false.
  - A failure is a finding for the user, not a fault. It bears on G2's L2 (Decisions, **G2 L2,
    decided**).

## The history: three stops, two review findings, five amendments

Each stop was reported before any commit, with the driver built, saved as a patch, and the
checkout restored to clean. The prompt's §8 records the user's decisions.

**First stop: the stall rule as dispatched** (two failed splits in a row, $\rho = \tfrac12$, an
infinite child counting as a failure). On the 25 driven problems:
- **SincIntegral** (and its vectorised twin): the parent converges (10 regions, depth 8). Along the
  leftmost line $(e_L+e_R)/e_P$ = 0.846, 0.532, 0.309, 0.126, 0.047, 0.041, 0.004, 0.002, 0.194,
  so [1, 25.75] stalled at depth 2: converged False, value 0.008672631630100134 against
  0.008666078496905705, true error $6.55\times10^{-6}$ against a declared $4.85\times10^{-6}$:
  **dishonest**.
- **The stationary-phase oracle:** $x = \tfrac12$ is a node of every leaf touching it, with an
  infinite floor; it stalled at depth 2, relative error 1.90. With no stall rule: relative error
  $8.2\times10^{-12}$, 96,941 leaves.
- $\rho$ = 0.75, 0.9 or 1.0 removed the SincIntegral stall but not the oracle's; an infinite parent
  never counting as failed fixed the oracle but not SincIntegral at $\rho = \tfrac12$.

**Variant (b),** on the user's request ("Can we try (b)"), three failed splits in a row at
$\rho = \tfrac12$, measured from a scratch copy of the driver (checked bit-identical to the patched
driver at run length 2 on all 25 problems): (b1) fixed SincIntegral but the oracle still stalled
(relative error 0.44); (b2), with an infinite parent never failing, stalled nowhere the parent
converges (oracle relative error $3.0\times10^{-11}$); three-in-a-row stalls SincIntegral at every
$\rho \le 0.30$. The user adopted (b2) (`8b08b8d`).

**Second stop: (b2),** building test 1. On 07's family (R) with cancelling amplitudes (a 50-call
scan: $x_s \in \{0.382, 0.7236\}$, $c \in \{10^4,\ldots,10^8\}$, `rtol` $\in \{10^{-6},\ldots,10^{-10}\}$,
`theta_deriv` supplied), Levin leaves across the stationary point have estimates that **rise** under
bisection until they reach the fallback (ratios 2.06, 1577, 0.961, … at $x_s = 0.382$, $c = 10^5$).
At $c = 10^5$, `rtol` $10^{-7}$, both $x_s$, the parent converges (true $1.07\times10^{-12}$,
$1.53\times10^{-12}$); (b2) stalled two leaves at depth 3, true errors $4.27\times10^{-3}$ (declared
$4.11\times10^{-3}$, **dishonest**) and $4.82\times10^{-5}$. At every $c \ge 10^5$ (b2) stalled on the
stationary-point line. With no stall rule the budget reproduced the parent exactly on the converged
cases. The user withdrew the rule ("(A). Let's remeasure the stall question after 07", `cf7913b`).

**The review finding: `79ae8c7`.** The global budget without a stall rule was committed as
`79ae8c7` (10 new tests; the 20 draws; 38 → 48). Its records showed it **dishonest where the parent
was honest**:
- `test_converged_flag`'s Gaussian bump: 3 leaves, value $-8.563\times10^{-11}$, abserr
  $8.219\times10^{-11}$, `converged` — the leaf $[0.75, 1.5]$'s residual $8.2\times10^{-11}$ under the
  whole budget, its halves carrying a further $3.4\times10^{-12}$ in the same direction. The parent:
  $-3.4\times10^{-12}$, abserr $1.9\times10^{-11}$, honest. Opened as
  `[06c-budget-can-stop-on-a-pre-asymptotic-residual]`.
- In the draws' `test_YJJ_log_singularity`, the dishonest printed cases flagged `converged` rose from
  44 (of 259 dishonest, parent) to 221 (of 221).
- Also from `79ae8c7`'s draws: converged calls 1,207 → 3,279 of 4,320 (one True → False); 208 calls
  at depth 20 (1 at the parent); solves ×40; `test_JJJ` and `test_YJJ` honest at the integral level
  on all 140 cases under both.

The user chose a check one level down, folded in before the push ("Yes, please do that. We want the
Part B census to be measuring the final scheme.", `9486375`). `c85f753` was `79ae8c7` plus it.

**The second review finding: `c85f753`.** Its review passed; its records (all above) showed two
things the user took up:
1. **Item 1.** In `test_YJJ_log_singularity`, 5 cases honest at the parent and dishonest under
   `c85f753`, all flagged `converged`; over the test's 800 cases, dishonest **and** `converged` rose
   from 44 to 228.
2. **Round-off read as non-convergence.** The tight mode-filter call: a residual of
   $2.1\times10^{-16}$ far above eq. (151)'s floor of $2\times10^{-20}$, $r = 1.96$, refined from 1
   region to 1,279 leaves.

The user chose a noise level for the check, $\nu = 8uM$ (`63b7721`, the fourth amendment), "and see
what that does to item 1".

**Third stop: $\nu = 8uM$ cannot reach 12f's residual** (before any code was written). With
$M = |V_P| + |V_L| + |V_R| + |V_{LL}| + |V_{LR}| + |V_{RL}| + |V_{RR}|$ and $u = 2^{-53}$, measured on
`c85f753`'s own solves for $e^{-x}\sin(10^6x)$ on $[1, 1.0003]$, `rtol` $10^{-8}$:

| depth (root's left line) | $V_P$ | $M$ | $\nu = 8uM$ | $\Delta_1$ | $\Delta_2$ | $\max(\Delta_1,\Delta_2)/\nu$ |
|---|---|---|---|---|---|---|
| 0 | $4.81\times10^{-7}$ | $1.44\times10^{-6}$ | $1.28\times10^{-21}$ | $2.07\times10^{-16}$ | $4.06\times10^{-16}$ | $3.2\times10^5$ |
| 1 | $1.96\times10^{-7}$ | $5.87\times10^{-7}$ | $5.2\times10^{-22}$ | $2.35\times10^{-16}$ | $9.89\times10^{-16}$ | $1.9\times10^6$ |
| 2 | $7.69\times10^{-8}$ | $2.31\times10^{-7}$ | $2.0\times10^{-22}$ | $5.54\times10^{-16}$ | $6.41\times10^{-16}$ | $3.1\times10^6$ |
| 3–7 | | | $1.3$–$2.7\times10^{-21}$ | | | $2.5\times10^4$–$1.3\times10^5$ |

Over the 1,279 leaves `c85f753` returns there, none has $\max(\Delta_1,\Delta_2) \le \nu$ (the ratio
runs from 38 to $7.7\times10^5$, median $1.25\times10^4$). The differences are rounding of the
**phase argument** ($u|\theta||V| \approx 5.3\times10^{-17}$ at $|\theta| \approx 10^6$), not of the
values ($u|V| \approx 5\times10^{-23}$).

**The (b′) measurement** (on the user's word, "Yes, measure (b') first"; a scratch variant of
`c85f753`, nothing committed): $\nu = 8u\,A\,P$, $A$ the largest argument handed to sin/cos over the
seven solves (`theta_mod_2pi` if supplied, else the raw $\theta$; the endpoints on the Levin branch,
every fine node on the fallback), $P$ the summed endpoint magnitudes ($\sum_i |p_i(a)| + |p_i(b)|$
on the Levin branch, $\sum_k |w_kf(x_k)|$ on the fallback), over the seven solves; a leaf with
$\max(\Delta_1,\Delta_2) \le \nu$ noise-limited, trusted, not refinable, declaring
$\max(\epsilon_P, \Delta_1, \Delta_2, \nu)$.
- **12f's call:** 1 leaf (noise-limited), 7 solves, converged, honest (true $1.1\times10^{-16}$,
  abserr $4.6\times10^{-15}$); along the root's line $\nu = 1.4$–$17.7\times10^{-15}$ against
  $\max(\Delta) = 3.3\times10^{-17}$–$9.9\times10^{-16}$.
- **12a–12e hold** (12a abserr $8.5641\times10^{-11}$ against $8.5626$, unchanged; 12c converged,
  honest).
- **Five converged calls become unconverged**, convergence assertions §2 test 9 does not allow to
  change; $\nu$ overstates the true error by $10^2$–$10^6$:

  | call | declared $\nu$ | budget | true error |
  |---|---|---|---|
  | test 2 (8 leaves) | $1.5$–$1.75\times10^{-14}$ each, abserr $1.33\times10^{-13}$ | $1.00\times10^{-13}$ | $1.2\times10^{-16}$ |
  | test 3 ($A = 4.85\times10^{11}$ on $[10, 20]$) | $6.25\times10^{-11}$ per leaf | $5.63\times10^{-11}$ | $1.1\times10^{-14}$ |
  | SinIntegral (`test_converged_flag`) | $6.2\times10^{-10}$ | $5.6\times10^{-11}$ | $8.9\times10^{-16}$ |
  | `lstsq_path` | $2.7\times10^{-14}$ | $2.0\times10^{-14}$ | $4.5\times10^{-17}$ |
  | `direct_solve_path` | $1.2\times10^{-14}$ | $6.7\times10^{-15}$ | $1.8\times10^{-16}$ |

  $A$ (the largest argument) multiplies $P$ summed over solves whose arguments differ: in test 3 the
  large $|\theta|$ is at $x = 20$, where the $p$ terms are tiny.
- **On the 25 driven problems:** none honest under `c85f753` is dishonest under the variant.
- **The 20 draws** (variant only; parent and `c85f753` from `c85f753`'s record): all 80 units pass.

  | | converged calls | depth-20 calls | solves | noise-limited leaves | untrusted returned |
  |---|---|---|---|---|---|
  | parent | 1,207 | 1 | 133,092 | — | — |
  | `c85f753` | 3,254 | 230 | 5,936,892 | — | 241 |
  | (b′) | 3,266 | 217 | 5,757,700 | 17,352 | 109 |

- **Item 1,** `test_YJJ_log_singularity`, 800 printed cases:

  | | dishonest | dishonest and converged | converged | worst true/declared | worst among converged |
  |---|---|---|---|---|---|
  | parent | 259 | **44** | 68 | $8.11\times10^5$ | $3.73\times10^5$ |
  | `c85f753` | 235 | **228** | 449 | $5.46\times10^5$ | $5.46\times10^5$ |
  | (b′) | 226 | **225** | 460 | $2.61\times10^5$ | $2.61\times10^5$ |

  The worst cases are all $(Y_0, J_2, J_2)$ at $\epsilon = 10^{-10}$. Parent → (b′): 3 honest →
  dishonest (seed 8 #28, seed 14 #27, seed 19 #27, ratios 2.36, 1.08, 1.33, all converged), 36 the
  other way; `c85f753` → (b′): 3 honest → dishonest (seed 2 #28, seed 13 #29, seed 18 #28), 12 the
  other way. `test_JJJ` and `test_YJJ` honest on all 140 cases under all three. So item 1's
  dishonesty is not round-off: it is the test's near-singular conditioning of the integral and its
  reference, an input error no noise rule reaches.

**The user's decision** ("Yes, please go ahead with (ii)", `fef401c`, the fifth amendment): 06c ships
`c85f753`'s code with **no noise rule**. Round-off read as non-convergence costs work, not honesty;
item 1 is recorded as input conditioning, which §5's X1 measures on A3 v2; the noise question waits
for after 07. In production both callers supply `theta_mod_2pi`, so arguments are at most $2\pi$.

## Deviations from the prompt

1. **The baselines were re-taken at the dispatch SHA,** and equal the orchestrator's.
   **IMPLEMENTATION CHOICE.**
2. **After each stop the checkout was restored to clean and the work kept as a patch** in the
   session's scratchpad (`06c-driver-wip.patch`, `fed7097a…`; `06c-b2-wip.patch`, `5ff7fd4e…`).
   **IMPLEMENTATION CHOICE.**
3. **The first stop's variants were measured by substituting `_split_failed` at run time** in
   scratch scripts. **IMPLEMENTATION CHOICE.**
4. **The first stop's variant test runner disabled logging,** so `test_input_validation` failed in
   those runs only. **UNINTENDED DRIFT**, confined to scratch scripts, corrected in the (b) runs.
5. **Variant (b) was measured from a scratch copy of the driver** loaded in place of the module, so
   the checkout stayed clean; checked bit-identical at run length 2. **STRUCTURALLY REQUIRED.**
6. **Three-in-a-row was measured with a run-length counter, and (b2) built with three flags per
   leaf,** checked bit-identical on all 25 problems. **IMPLEMENTATION CHOICE.** Both gone.
7. **The variant (b) runs left logging on, output captured.** **IMPLEMENTATION CHOICE.**
8. **The $\rho$ scan for three-in-a-row used a 0.01 grid.** **IMPLEMENTATION CHOICE.**
9. **The stall rule, `used_interval.stalled`, `num_stalled_regions` and their counters were built
   and then removed** (the second amendment); one sentence in `_adaptive_levin`'s docstring records
   the withdrawal. **STRUCTURALLY REQUIRED.**
10. **A DEBUG summary per call** (leaves, bisections, floor-limited, at `depth_max`, the trust
    check's counts, the budget test against the returned sums, rejections), silent by default; how
    §1 item 3's running-test record and the check's counts are measured. **IMPLEMENTATION CHOICE.**
11. **The running test is confirmed on the returned sums before the loop acts on it.**
    **IMPLEMENTATION CHOICE.**
12. **Floor-limited is tested only on a chosen leaf, and a capped leaf is never chosen,** so a capped
    leaf is never flagged phase-limited (06b checked the premise at `depth_max` too, its deviation
    3). **IMPLEMENTATION CHOICE.**
13. **The comment in `_write_progress_data`'s per-region plot filter** now says it is not the
    driver's test. **IMPLEMENTATION CHOICE.**
14. **`RunDrivers/quadsource_atol_sweep.py`'s module docstring was corrected,** a launcher rather
    than a direct caller, because §1 item 8 asks for every comment describing `local_atol`; AST
    checked. **IMPLEMENTATION CHOICE.**
15. **A comment-only change in `test_converged_flag`,** beyond test 9's assertion kinds, because §1
    item 8 asks for it. **IMPLEMENTATION CHOICE.**
16. **Test 2's parent miss is narrow** (true 0.9987 of the request). Attempts: GRZ with λ near 4n
    and `theta_deriv` (resolved at depth 0); $f = 1/x$ with $\cos\omega = \cos\omega L = 0$ over
    $\omega/\pi \in \{30.5, \ldots, 1000.5\}$, $L \approx 20$–100, orders 12 and 16, `rtol`
    $10^{-6}$–$10^{-11}$ (mostly the parent's flag right and its value outside the request; one other
    strict miss where the new code did not converge); a Gaussian amplitude (strict misses, but
    every region fallback). **IMPLEMENTATION CHOICE.**
17. **Test 1 uses `atol` $10^{-18}$, a fixed constant,** not 07's $10^{-14}|I|$
    ($5.8\times10^{-19}$). **IMPLEMENTATION CHOICE.**
18. **Test 4 imports 06b's test 2 problems** and checks them under the new code only.
    **IMPLEMENTATION CHOICE.**
19. **Test 6 uses $c = 10^4$, not the oracle's $10^6$** (96,941 leaves, 21 s, at $10^6$).
    **IMPLEMENTATION CHOICE.**
20. **Test 7's odd integrand resolves at depth 0 under both codes,** so it tests the budget at
    $V \approx 0$, not refinement towards it. **IMPLEMENTATION CHOICE.**
21. **Test 11 is descriptive in `parent_demo.txt`, not an assertion,** as the amended prompt makes
    it. **IMPLEMENTATION CHOICE.**
22. **The draws ran as a background process of this session, not registered and detached,** though
    they take more than ten minutes (README §5 rule 7): the dispatch forbids writing under `var/`.
    **STRUCTURALLY REQUIRED.**
23. **`parent_demo.py` is re-run after the last edit to its inputs,** so its recorded hashes match
    the files committed (`CLAUDE.md` rule 7). **IMPLEMENTATION CHOICE.**
24. **The second amendment's commit was quoted as `1aa3a6b`; it is `cf7913b`,** as the orchestrator
    confirmed. **IMPLEMENTATION CHOICE.**
25. **An issue opened**, `[06c-budget-can-stop-on-a-pre-asymptotic-residual]` (board §3,
    `docs/OPEN_ISSUES.md`), and narrowed by this commit. **IMPLEMENTATION CHOICE** (`CLAUDE.md`
    rule 4).
26. **`79ae8c7`'s draws were made twice;** its first record (112 MB, every phase-limited region of
    every call) was too large to commit, and the harness now keeps only the count. **IMPLEMENTATION
    CHOICE.** Both of those runs are superseded by this commit's run.
27. **This commit is rebuilt from `79ae8c7` by `git cherry-pick --no-commit`.** The board conflicted
    (the third amendment had edited the 06c row); it was resolved to `9486375`'s version, and this
    commit's §3 entry and 06c row were then written into it by hand, not merged from `79ae8c7`.
    **STRUCTURALLY REQUIRED** (the branch was reset under the work).
28. **`used_interval` gains `declared_resolution` besides §8's $\Delta_1$, $\Delta_2$, $r$ and
    `trusted`,** and `total_err` uses it; `abserr`, `abserr_resolution` and `abserr_fallback` stay the
    raw estimates, so the result's breakdown keys are unwidened. **IMPLEMENTATION CHOICE.**
29. **A leaf at `depth_max` is checked only at a budget pass,** as §8 rule 4 states ("every leaf
    checked, floor-limited, or at `depth_max`"); a call whose budget is never met returns its capped
    leaves unchecked (`trusted` None), as the stationary-phase oracle does. Checking them at the end
    too would widen those calls' declared errors. **IMPLEMENTATION CHOICE**; Observations 2.
30. **Changing a leaf's error or trust re-pushes it on the heap under a version,** stale entries
    skipped. **IMPLEMENTATION CHOICE.**
31. **A floor-limited leaf found after it was checked reverts to its own estimate** (`trusted` and
    `declared_resolution` cleared, $\Delta$s kept as a record). §8 rule 3 exempts floor-limited
    leaves; this is how a prior check is undone. **IMPLEMENTATION CHOICE.**
32. **`converged` requires no untrusted leaf returned, and the DEBUG line compares the running test
    with the returned budget alone** (so an untrusted capped leaf is not counted as a
    disagreement). **IMPLEMENTATION CHOICE.**
33. **Breakage (ii) was run on test 4 alone:** with floor-limited leaves left refinable the whole
    module exceeded a 1,500 s timeout (every floor-bound leaf chased to depth 20), and test 4 alone
    took 179 s. **IMPLEMENTATION CHOICE.**
34. **Breakage (iv) makes every checked leaf trusted with no declared error** (the check's
    $r < 1$ branch replaced by "trusted, unwidened"); it is caught by 12a, and also by 12d and 12e.
    **IMPLEMENTATION CHOICE.**
35. **The driver's docstrings were finished after the breakages ran;** its AST without docstrings
    is identical to the breakage-tested file (`b4ef46ed…` → `4161004e…`). **IMPLEMENTATION CHOICE.**
36. **12d's spike sits at $x_0 = 0.4$.** At $x_0$ = 0.137, 0.4 and 0.613 the root is untrusted and
    honest; at $x_0$ = 0.2 and 0.3 it is untrusted but its cap declaration $\Delta_1+\Delta_2$
    ($1.03\times10^{-3}$) understates the true error ($1.77\times10^{-3}$) (Observations 1).
    **IMPLEMENTATION CHOICE**, recorded.
37. **12e's references use the problems' spans as doubles** (1/3, 7/3, 1.0003, 1.1 exactly as
    floats), and assert honesty on all 25 since every one is honest at the parent.
    **IMPLEMENTATION CHOICE.**
38. **12c sets `atol` = $5.1\times10^{-19}$,** about $10^{-14}|I|$ as 07 §2 sets it.
    **IMPLEMENTATION CHOICE.**
39. **Test 10's honesty table is at the integral level** (each printed three-Bessel case against
    its closed form, the test's own `converged`), since no single Levin call has a reference.
    **IMPLEMENTATION CHOICE.**
40. **The third stop's premise was measured before any code was written,** from `c85f753`'s module
    read with `git show` into the scratchpad, the checkout untouched. **IMPLEMENTATION CHOICE.**
41. **The (b′) variant was a scratchpad copy of `c85f753`'s module,** loaded in place of
    `AdaptiveLevin.levin_quadrature` to run `c85f753`'s own test module, and under its own name for
    the comparisons. **IMPLEMENTATION CHOICE.**
42. **(b′)'s $A$ was taken per solve, re-evaluating the arguments** through `theta_mod_2pi` or
    `raw_theta` (two endpoints on the Levin branch, every fine node on the fallback; extra
    evaluations, measurement only); $A = 0$ for the $m = 4$ test basis, which has no $\theta$, so the
    rule was inert there. **IMPLEMENTATION CHOICE.**
43. **(b′)'s $P$ on the Levin branch included the modes the p-gate drops.** **IMPLEMENTATION
    CHOICE.**
44. **(b′)'s noise test was taken inside the check, before $r$,** at the check's lazy timing; a
    noise-limited leaf trusted and not refinable, as a floor-limited one. **IMPLEMENTATION CHOICE.**
45. **12a–12e under (b′) were checked by running `c85f753`'s own test module** (all 15 tests)
    against the variant. **IMPLEMENTATION CHOICE.**
46. **Only the variant's 80 draw units were run;** the parent's and `c85f753`'s came from
    `c85f753`'s committed record (`git show`). The harness was copied into the scratchpad and wrote
    only there. **IMPLEMENTATION CHOICE.**
47. **Item 1 was analysed at the integral level,** as test 10's table. **IMPLEMENTATION CHOICE.**
48. **The (b′) draws ran 1,318 s as an unregistered background process** (nothing may go under
    `var/`). **STRUCTURALLY REQUIRED.**
49. **This commit re-lands `c85f753`'s content by `git cherry-pick --no-commit c85f753`** onto
    `fef401c`; the board and index merged without conflict, keeping `fef401c`'s Decisions rows and
    this prompt's 06c row and §3 entry. The code, tests and calibration records are byte-identical
    to `c85f753`'s. **STRUCTURALLY REQUIRED** (the branch was reset under the work).
50. **The tests, breakages and draws are not re-run;** they carry over from `c85f753`, whose code
    this commit holds unchanged (deviation 49). Only the seven roots are re-taken. **IMPLEMENTATION
    CHOICE**, as the orchestrator directed.
51. **The (b′) scratch files are not committed** (the variant, its scripts, its draws and item 1's
    800-row table stay in the session's scratchpad); this log carries their numbers. **IMPLEMENTATION
    CHOICE** (only the dispatched paths may be committed).

## Verification performed

- **This commit's code is `c85f753`'s, byte for byte** (`git diff c85f753 -- AdaptiveLevin
  ComputeTargets RunDrivers docs/levin-phase-floor/06c` empty). The new module, the breakages, the
  existing tests, `parent_demo.txt` and the draws below were run on `c85f753`'s code and carry over;
  the seven roots were re-taken on this commit's tree, 2026-10-29 (load averages before
  each 5.7–16.9):

  | Root | Baseline (`3f2a36d`) | This commit | Result |
  |---|---|---|---|
  | AdaptiveLevin | 38 | **53** | OK (118.3 s) |
  | ComputeTargets | 780 | 780 | OK (291.9 s) |
  | CosmologyModels | 39 | 39 | OK |
  | LiouvilleGreen | 148 | 148 | OK, skipped 1 (237.3 s) |
  | ObjectFactories | 280 | 280 | OK |
  | RunDrivers | 127 | 127 | OK |
  | tools | 58 | 58 | OK |

  The same counts as `c85f753`'s run below. The `test_wall_time_per_object` flake did not fire.
- **The new module,** `PYTHONPATH=. ./venv/bin/python -m unittest AdaptiveLevin.tests.test_levin_global_budget`:
  15 tests, OK.
- **The in-tree driver without the check** (before it was added) was `79ae8c7`'s, re-applied by
  `git cherry-pick --no-commit`; `79ae8c7`'s own verification (its breakages, its draws, 38 → 48) is
  superseded by what follows, all re-run on the final driver.
- **The breakages,** each applied by a scratch script that copies the module aside, edits it by
  exact string replacement and restores the copy; the module's SHA-256 was `b4ef46ed…` before,
  between and after (its docstrings were finished afterwards, deviation 35):
  - **(i)** after a leaf is chosen, `if region.relerr < rtol: continue`: **test 1 fails**, and tests
    2, 3, 5, 6, 12a, 12c.
  - **(ii)** `if False and floor_limited:`: **test 4 fails** (`0 not greater than or equal to 1 :
    test 4 (06b 2a)`), run alone in 179 s (deviation 33).
  - **(iii)** the leaf error read as the residual alone and the confirmation skipped: **test 3
    fails**, and tests 2 and 6.
  - **(iv)** every checked leaf trusted, unwidened: **test 12a fails** (`-8.5626e-11 is not within
    abserr 8.219e-11`), and 12d (`True is not false`) and 12e (`converged_flag_bump`).
- **The existing tests** under the final driver: AdaptiveLevin 53 OK, LiouvilleGreen 148 OK
  (skipped 1).
- **Callers:** `ComputeTargets/QuadSourceIntegral.py` and `RunDrivers/quadsource_atol_sweep.py`, AST
  with docstrings stripped identical to `3f2a36d`'s.
- **`parent_demo.txt`** run on the final driver and inputs; **`lg_seeded_draws`** run once on the
  final driver (`4161004e…`, recorded in its header), and its report written from it. The parser of
  printed cases was checked against `79ae8c7`'s draws record (259 / 221 dishonest
  `test_YJJ_log_singularity` cases, as measured then).
- **The seven roots, after the change** (the commands of `CLAUDE.md`, run in sequence; the machine
  heavily loaded by other work):

  | Root | Baseline (`3f2a36d`) | After | Result | load averages before |
  |---|---|---|---|---|
  | AdaptiveLevin | 38 | **53** | OK (116.4 s; 0.1 s before) | 63.90 72.67 56.06 |
  | ComputeTargets | 780 | 780 | OK (306.0 s, under load; 141 s before) | 55.98 66.02 55.51 |
  | CosmologyModels | 39 | 39 | OK | 52.41 51.25 50.69 |
  | LiouvilleGreen | 148 | 148 | OK, skipped 1 (442.8 s; 26.2 s before) | 52.41 51.25 50.69 |
  | ObjectFactories | 280 | 280 | OK | 20.34 24.19 36.03 |
  | RunDrivers | 127 | 127 | OK | 18.58 23.52 35.51 |
  | tools | 58 | 58 | OK | 18.10 23.24 35.27 |

  AdaptiveLevin rose by 15, the new module's. The `test_wall_time_per_object` flake did not fire.
- **`black`** on `levin_quadrature.py`, the three test modules, the two callers and the calibration
  scripts; `black --check` clean on all of them.
- **Scope.** `git status --short` before staging lists only the dispatched paths.

## Observations not acted on

1. **The check's widening is an extrapolation, not a bound, and can be outrun.** On 12a it covers
   the true error by $1.8\times10^{-4}$ relative ($8.5641$ against $8.5626\times10^{-11}$): enough, by
   luck of a geometric tail. On 12d-like spikes at $x_0$ = 0.2 and 0.3 the root is correctly found
   untrusted, but its cap declaration $\Delta_1+\Delta_2$ understates the error by 1.7×: the
   grandchildren's nodes catch only part of the spike. And it cannot reach an error common to a leaf
   and its descendants (`[06b-moved-calls-declare-less-than-their-error]`). In the draws, 5
   near-singular `test_YJJ_log_singularity` cases are honest at the parent and dishonest under the
   new code (true/reported 1.06–2.2, all flagged converged), against 29 the other way. So
   `[06c-budget-can-stop-on-a-pre-asymptotic-residual]` is narrowed, not closed.
2. **A call whose budget is never met returns its capped leaves unchecked** (deviation 29): the
   check is never taken on the stationary-phase oracle's 84,514 capped leaves. Its abserr there is
   unwidened. §5's descriptive item on the check counts returned unchecked leaves.
3. **Round-off noise reads as untrusted** where the residual sits above eq. (151)'s floor: the tight
   mode-filter call (residual $2\times10^{-16}$, floor $2\times10^{-20}$, $r = 1.96$) refines from 1
   region to 1,279 before its noise fits the budget. Honest and converged, at ×3,400 the solves. The
   floor does not model the solve's own round-off at this level; the check makes that visible as
   cost.
4. **A request that cannot be met is chased to exhaustion** (the stationary-phase oracle: 20 →
   96,941 leaves; the draws' depth-20 calls below). The withdrawn stall rule's ground, re-measured
   after 07 under `[06b-premise-check-can-chase-a-flat-residual-to-depth-max]`.
5. **`relerr` in the depth-18 history log divides by zero** when a history residual is exactly 0
   (a `RuntimeWarning` seen on one draw). Unchanged from the parent; diagnostic only.
6. **The family (R) scan shows the stationary-point weakness of both schemes** at large $c$
   (parent true errors up to 315 × the request, unconverged). 07's.

## State handed on

- **The code commit** is made and not pushed; its parent is `fef401c`. §4 (R17 at that parent,
  whose code is `3f2a36d`'s; R18 at the code commit) and §5 run on the Mac Studio after the push.
  The board's 06c row says the code landed and §4–§5 are pending.
- **For §4 and §5:** every call can move. Expect much higher cost where a request cannot be met
  (exhaustive refinement) and where round-off noise sits above the floor (Observations 3): on the
  laptop's draws solves rose ×45 and 230 of 4,320 calls reached depth 20. 06b's census budget of 300 s per unit may be exceeded; an
  over-budget unit counts against X5 as the prompt says.
- **For §5's X1:** honesty is the question; the check narrows but does not close it (Observations
  1), and item 1's near-singular cases show input conditioning that no error estimate reaches. §5's new descriptive item (the check's $r$ distribution, untrusted leaves, the widening's
  effect, its solves) has what it needs in each leaf's `used_interval` and in
  `num_untrusted_regions`.
- **For 07's rebase:** `_adaptive_levin`'s loop is new; 07's window pieces and guard enter as
  leaves, and its rule (ii) (the exit barred in window pieces and guarded regions) becomes "never
  floor-limited" there. The bypass keyword stays last.
- `docs/levin-phase-floor/06c/calibration/` holds the scripts and outputs; nothing is kept on the
  Mac Studio from this part.
