# levin-phase-floor 06m — the signed trust yardstick in the driver

> **Part A, §1–§3 only** (the code, its tests and these records), written by the executing agent
> on the laptop. Part B, §4 (rehearsals R25 and R26) and §5 (the census, Gates M0 and M), runs
> later on the Mac Studio; the orchestrator appends its record to this log.

**Subject:** prompt 06m §1–§3, on the laptop, in the main checkout
`/Users/ds283/Documents/Code/SecondaryGWKit`, branch `handover-remedial`.
- **Dispatch SHA:** `5ff8a5e4` (pushed to `origin/handover-remedial`). The parent module is
  `git show 5ff8a5e4:AdaptiveLevin/levin_quadrature.py`, 06g's code (`3ae6f3a4`).
  `git log --oneline 5ff8a5e4..HEAD` was empty at the start and before staging.
- **The gate** held: 06l's records commit `3076ba0e` is on `origin/handover-remedial`;
  `git status --short` was empty at the start; `runregistrykit list` showed 79 runs, 77 finished
  and 2 `unknown` (pre-registry), nothing `running` or stale; `git diff 3ae6f3a4 HEAD --
  AdaptiveLevin/` is empty. The seven roots at the dispatch SHA are the orchestrator's (Verification).
- No Ray was started, no run was registered, no store was opened, and nothing was written under
  `var/`. `var/levin-phase-floor-07/` was not read. The stashes were not touched, but `git stash list`
  was run once, read-only (deviation 9).

**Commit:** the code commit of this log (one commit, not pushed).

**Result:** route B's yardstick is in the driver. `_check_leaf`'s noise clause compares $\Delta_1$
with $\nu_B$ from a new pure function, `_trust_noise_yardstick`, in 06l's arithmetic and order;
nothing else in the driver changes. **The commit is 06l's variant bit for bit** on test 2's nine
cells (every result key but `elapsed`, every returned region's fields) and on all 4,320 Levin calls
of the twenty LiouvilleGreen draws (every call field and a digest of the whole result). Tests 1–5
pass, the five breakages are caught, every existing test passes with **no existing assertion
changed**, and the seven roots rise only by the nine tests added (AdaptiveLevin 82 → 91). No issue
opened. One observation is worth the orchestrator's eye (observation 1: on a constructed integrand
at $x \approx 3\times10^5$, Levin leaves carry $\Delta_1$ up to $3.8\times10^3$ times their eq. (151)
floor).

## What shipped

**`AdaptiveLevin/levin_quadrature.py`.**
1. **`_trust_noise_yardstick(phi_P, parent_is_direct, data_L, data_R)`**, module level, after
   `_solve_floor_in_budget`. Pure; returns `float(nu)`:
   - $\nu \leftarrow \phi_P$;
   - for each half in the order L, R that is not Clenshaw–Curtis (`is_direct` false or absent),
     $\nu \leftarrow \nu + \phi_h$, `phase_err` with `None` read as 0.0;
   - if the parent is Clenshaw–Curtis and a half is: for each Clenshaw–Curtis half in the order
     L, R, a `RuntimeError` if it lacks `phase_err_less_node` or `node_error_signed` (absent or
     `None`), else $\nu \leftarrow \nu + \phi^-_h$; then $\nu \leftarrow \nu + |S|$, $S$ the one
     half's $E$ or $E_L + E_R$.

   Its docstring states the formula, the four cases as a table, the modification on a Levin parent,
   the basis (06l, `VERIFICATION.md` §6l), the order and why it is fixed (bit identity with 06l's
   census; floating-point addition is not associative), and the guard.
2. **`_check_leaf`'s trust test** calls it with `region.phase_err`, `leaf.is_direct`, `dataL` and
   `dataR`, on every checked leaf before the test (as the variant does, deviation 3), and compares
   $\Delta_1$ with the result: trusted iff $r < 1/4$, or $r < 1$ and $\Delta_1 \le \nu_B$. Nothing
   else in `_check_leaf` changes.
3. **Comments and docstrings.** The comment above `_LEVIN_TRUST_MAX_RATIO` (the clause now reads
   $\Delta_1 \le \nu_B$; why $\phi_P$ alone was too small, 06j's 75 first-population leaves; why the
   halves' node errors are combined signed, 06k's route 1 and 06l; why a Levin parent's
   Clenshaw–Curtis halves are kept out, $1.1\times10^4$ under route 1 against 2.85 under B; what B
   does not reach, the 12 leaves and no margin; that $1/4$ is unchanged). `_check_leaf`'s docstring
   and `_adaptive_levin`'s where they state the trust test. Also two more places that restate the
   test, `used_interval`'s comment on `_trusted` and its `trusted` docstring (deviation 1).

The diff is 134 insertions and 11 deletions, all in this file; the only executable change is the
new function and the three lines of the trust test.

**`AdaptiveLevin/tests/test_levin_trust_yardstick.py`** (new, 9 tests). No Ray, no datastore. The
parent is `git show 5ff8a5e4:…`, executed as `levin_quadrature_parent_5ff8a5e`; 06l's variant is
executed from its committed path as `levin_quadrature_routeB_06l` after its SHA-256 is checked equal
to `5aa6178a…`. No stand-in is used: every leaf case is on a real integrand (deviation 4).

| Test | Where | What it asserts | Key numbers (`docs/levin-phase-floor/06m/calibration/out/cells.txt`) |
|---|---|---|---|
| 1 | `TestYardstickExactly` (5 tests) | Nine constructed cases, each bit for bit (`float.hex`) against the formula written out in §0.1's order, $\nu_B \le \nu_1$ in each: Levin/Levin+Levin; Levin/CC+Levin and Levin/Levin+CC (the CC half adds nothing); Levin/CC+CC ($\nu_B = \phi_P$); CC/CC+CC with $E$ of opposite signs and with both negative; CC/CC+Levin and CC/Levin+CC; CC/Levin+Levin. One CC/CC case where adding $\lvert E_L+E_R\rvert$ before the $\phi^-$ terms changes the last bit (asserted to differ, and the code to give §0.1's). The signed sum below the absolute sum. A Levin half's `None` floor read as 0.0. The guard raises on a CC parent whose CC half lacks $\phi^-$ or $E$ (absent or `None`, L or R), and not on a Levin parent | order case: §0.1's `0x1.41a327af3434ep-31`, $\lvert S\rvert$ first `…434fp-31` |
| 2 | `TestCodeIsTheVariants.test_bit_identical_to_the_variant` | Nine cells through `adaptive_levin_sincos` under the commit, the variant and the parent. Every result key the commit returns, `elapsed` aside, is the variant's bit for bit, and so is every returned region's span, depth, type, value, `value_kind`, `value_own`, every error field, `phase_limited`, `trusted`, `delta1`, `delta2` and `convergence_ratio`. Where the variant records no firing, the commit is the parent's bit for bit. Across the cells: at least one firing on a CC parent and one on a Levin parent; fallback and Levin leaves; depth $\ge 2$; the variant withdraws no trust | Cells and firings below |
| 3 | `TestLeafCases.test_trusted_on_cancelling_node_errors` | `far(3e5)`, leaf [300003.640625, 300003.65625] depth 8: CC parent, CC halves; its solves recomputed afresh reproduce the driver's $\Delta_1$ to the last bit. $E_L E_R < 0$, both $> 2\lvert E_P\rvert$, $\lvert E_L+E_R\rvert < 0.1(\lvert E_L\rvert+\lvert E_R\rvert)$; $\lvert r-\tfrac12\rvert < 0.1$; $\phi_P < \Delta_1 \le \nu_B$. Commit: trusted, returned, `quarters`, call converged. Parent: untrusted, same $r$, $\Delta_1$, $\phi_P$, not returned (refined) | $r = 0.4190$, $\Delta_1 = 7.163\times10^{-10}$, $\phi_P = 5.629\times10^{-10}$, $\nu_B = 9.145\times10^{-10}$ ($\nu_1 = 4.403\times10^{-9}$); $E_P = 4.92\times10^{-10}$, $E_L = -1.969\times10^{-9}$, $E_R = 1.744\times10^{-9}$, $\lvert E_L+E_R\rvert/(\lvert E_L\rvert+\lvert E_R\rvert) = 0.060$. Call: 107 regions (commit), 146 (parent), both converged |
| 4 | `…test_levin_parent_keeps_cc_halves_out` | `noisy(A=512)`, leaf [22.89453125, 23.125] depth 8: Levin parent, CC halves; $\nu_B = \phi_P$ exactly; $\lvert r-\tfrac12\rvert < 0.1$; $\nu_B < \Delta_1 \le \nu_1$ (route 1 would trust it). Untrusted under both codes | $r = 0.4856$, $\Delta_1 = 6.484\times10^{-16}$, $\phi_P = \nu_B = 1.645\times10^{-16}$, $\nu_1 = 2.854\times10^{-14}$ (the halves' floors 93× and 79× $\phi_P$) |
| 5 | `…test_levin_over_levin_far_above_noise_stays_untrusted` | `far(3e5)`, leaf [300002.0, 300002.5] depth 3: Levin parent, Levin halves; $\nu_B = \nu_1$; $1/4 \le r < 1$; $\Delta_1 > 10^3\,\nu_B$. Untrusted under both codes, refined under both | $r = 0.6330$, $\Delta_1 = 1.159\times10^{-10}$, $\nu_B = \nu_1 = 8.11\times10^{-14}$, $\Delta_1/\nu_B = 1.43\times10^3$ (06l's f02: $4.7\times10^6$; deviation 6) |

**Test 2's cells** (`test2_cells()`):

| Cell | Leaves (commit) | Variant's firings (CC parent / Levin parent) | Commit = variant | Commit = parent |
|---|---|---|---|---|
| `exp_sin_1e3` (06g's) | 1 Levin, depth 0 | 0 / 0 | yes | yes |
| `si_1000_exp` (06g's) | 3 Levin, depth 2 | 0 / 0 | yes | yes |
| `gaussian_bump` (06g's) | 10 Levin, depth 5 | 0 / 0 | yes | yes |
| `sqrt_fallback`, rtol 1e-10 (06g's) | 5 fallback, depth 4 | 0 / 0 | yes | yes |
| `runge_fallback` (06g's) | 11 fallback, depth 5 | 0 / 0 | yes | yes |
| `sinc_12b` (06c's 12b) | 17 (8 fallback), depth 12 | 0 / 0 | yes | yes |
| `noisy(A=128,k=60)`, atol 1e-12 | 17 Levin, depth 6 (parent: 25, 7 fallback, depth 10) | **0 / 1** | yes | no (fires) |
| `noisy(A=512,k=60)`, atol 1e-12 | 114 (87 fallback), depth 10 (parent 149) | **3 / 2** | yes | no (fires) |
| `far(3e5)`, rtol 1e-6 | 107 (106 fallback), depth 8 (parent 146) | **40 / 0** | yes | no (fires) |

`noisy(A, k)` is $\int_1^{60} [((A + 1/x) - A)\sin\theta + ((A + \cos x) - A)\cos\theta]\,dx$ with
$\theta = kx + x^2$, rtol $10^{-30}$, `depth_max` 16: the amplitudes carry an absolute rounding
noise of about ${\rm ulp}(A)$ at every node, 06j's second population. `far(3e5)` is
$\int_{x_0}^{x_0+4} [10^3\cos(0.7(x-x_0))\sin\theta + e^{-(x-x_0)}\cos\theta]\,dx$ at
$x_0 = 3\times10^5$, $\theta = 200(x-x_0) + 3(x-x_0)^2$, atol $10^{-30}$, rtol $10^{-6}$,
`depth_max` 12: the nodes' rounding, ${\rm ulp}(3\times10^5) \approx 6\times10^{-11}$, sets the
Clenshaw–Curtis node error $E$, 06j's first population. Every call converges under both codes.

**Calibration** (`docs/levin-phase-floor/06m/calibration/`):
- **`scripts/prov.py`, `scripts/lg_seeded_draws.py`, `scripts/lg_compare_06m.py`,
  `scripts/variant_loader.py`**: 06l's LiouvilleGreen harness, copied from
  `docs/levin-phase-floor/06l/scripts/calibration/` (and `variant_loader.py` from
  `docs/levin-phase-floor/06l/scripts/`, unchanged), each change marked "prompt 06m". The diff and
  both sides' SHA-256 are `out/scripts_diff.txt` (deviation 7).
- **`scripts/cells.py`** (new) → `out/cells.txt`: test 2's cells under the three codes with every
  firing, and tests 3–5's leaves; the source of every number the tests quote.
- **`scripts/breakages.py`** (new; 06g's `breakages.py` the pattern) → `out/breakages.txt`.
- **`out/lg_draws/`**: `lg_seeded_draws.jsonl` (the run, 240 units, 21 MB), `run_stdout.txt`,
  `compare.txt`/`compare.json` (`lg_compare_06m.py`) and `lg_seeded_draws_report.txt` (06g's report,
  parent against commit).

## Predictions and verdicts

§4's and §5's rules, copied before they run, **not yet scored** (Part B, on the Mac Studio).

- **§4, R1** — copied verbatim, **not yet scored**:

  > - **R1 — consistency.** **Rule:** every row outside 06a's 720 stationary-point rows with finite
  >   `total_abserr` in both stores has $|{\rm total}_{26} - {\rm total}_{25}| \le {\rm
  >   total\_abserr}_{25} + {\rm total\_abserr}_{26}$, and the same on the oracle. Rows with an infinite
  >   error in either are counted separately; the stationary-point rows are tabulated, not scored.
  >   **Holds** if no row violates either. A violation is a finding for the user, not by itself a stop.

- **§5, Gate M0** — copied verbatim, **not yet scored**:

  > **Gate M0 — K0 first.** At $m$ the harness reproduces route B's census on 06k's 100 K0 units
  > (`docs/levin-phase-floor/06l/out/k0/units.json`), bit for bit by the comparison above. It is one
  > registered run with slug `k0`. **Holds** if all 100 are equal. A failure is a stop: the code is not
  > what 06l measured.

- **§5, Gate M** — copied verbatim, **not yet scored**:

  > **Gate M — the census is route B's.** **Rule:** all 18,560 units are `done`, and each equals route
  > B's census by the comparison above. **Holds** if all are equal. A failure is a stop, with the first
  > differing units and fields reported.

Part A's own acceptance (§6 item 1) is not a prediction; it is met (Verification): every test
passing and the five breakages caught; no existing assertion changed; the LiouvilleGreen draws equal
to the variant's bit for bit, with no new failure; the seven roots rising only by the nine tests
added.

## Deviations from the prompt

1. **Two more comments that restate the trust test were updated** (`IMPLEMENTATION CHOICE`). §1
   item 3 names three places. `used_interval.__init__`'s comment on `_trusted` and the `trusted`
   property's docstring also stated "delta1 at most phase_err", which the change made false; both now
   name the yardstick and date the two rules. Comments only; the module's executable change is §1's.
2. **`_trust_noise_yardstick` returns `float(nu)`** (`IMPLEMENTATION CHOICE`). The prompt says it
   "returns a float"; the variant's $\nu_B$ is whatever type the additions give (a NumPy float64 where
   `phase_err` is one). The conversion is exact; tests 2 and 6 find the commit bit-identical to the
   variant.
3. **The yardstick is formed on every checked leaf, before the trust test** (`IMPLEMENTATION
   CHOICE`), not only where $1/4 \le r < 1$, as the variant forms it. So the guard raises exactly
   where the variant's would, and the call sequence is the variant's.
4. **No stand-in; two constructed integrand families** (`IMPLEMENTATION CHOICE`). None of 06g's
   test cells fires route B, and a scratch search over about a hundred clean integrands and requests
   (smooth and oscillatory, Levin and fallback, tight `atol`/`rtol`; scratch scripts outside the
   repository, not kept) found none either: B fires only on noise-level leaves. The families
   `noisy(A, k)` (amplitude noise of ${\rm ulp}(A)$, the census's second population) and `far(3e5)`
   (node rounding at large $x$, its first population) give real firings on both parent kinds, and
   real leaves for tests 3–5, so the stand-in the prompt allows was not needed.
5. **Test 2's cells** (`IMPLEMENTATION CHOICE`): nine, six of 06g's (`test_levin_leaf_values`'s
   cells, copied into the module rather than imported, with `sqrt_fallback` at rtol $10^{-10}$ only)
   and the three firing cells. The variant's firings: Levin parent 3 (one on `noisy(128)`, two on
   `noisy(512)`), Clenshaw–Curtis parent 43 (three on `noisy(512)`, forty on `far(3e5)`).
6. **Tests 3 and 5's leaves against the prompt's models** (`IMPLEMENTATION CHOICE`). Test 3's $r$ is
   0.419 ("near 1/2" is asserted as $|r - \tfrac12| < 0.1$); every Clenshaw–Curtis firing on
   `far(3e5)` sits at $r \approx 0.42$, and this one has the strongest cancellation among them
   (0.060). Its halves' $E$ are 4.0× and 3.5× $E_P$, not the census's $200$–$1.6\times10^6$×. Test 5's
   leaf has $\Delta_1/\nu_B = 1.43\times10^3$, not f02's $4.7\times10^6$; it is the same kind (a
   Levin-over-Levin leaf far above every yardstick at $r < 1$), on a cell of this prompt's own,
   asserted as $\Delta_1 > 10^3\,\nu_B$.
7. **The LiouvilleGreen harness, changed where marked "prompt 06m"** (`IMPLEMENTATION CHOICE`; the
   diff is `out/scripts_diff.txt`):
   - modes `parent` (the dispatch commit's module, `git show` into a temporary file outside the
     repository, installed by `variant_loader.install` as 06l installs a variant), `commit` (the
     tree's module) and `routeB` (06l's variant, SHA-256 checked); 06l's were `tree`, `identity`,
     `routeB`;
   - each unit's working directory a temporary directory outside the repository (06l:
     `var/…/state/lg_cwd`);
   - each call record also keeps `result_sha256` (every result key but `elapsed`, `regions` and
     `route_06l`, floats as hex), `regions_full_sha256` (every returned region's value, trust,
     differences and every error field) and `result_keys`, so that the comparison with the variant
     covers the driver's whole result;
   - `prov.py`: `REPO` the repository the copy sits in, `OUT` `calibration/out/lg_draws/`, the parent
     the dispatch SHA;
   - `lg_compare_06m.py` (renamed from `lg_compare_06l.py`): base mode `parent`, the commit and the
     variant each against it, and a section comparing the commit with the variant bit for bit;
   - 06g's `report` reads mode `commit` where it read `new`, with a header line saying so.
8. **Breakages (ii) and (iii) as built** (`IMPLEMENTATION CHOICE`). (ii) adds each Clenshaw–Curtis
   half's $\phi^-$ on a Levin parent and nothing else (not its $|E|$), the prompt's wording. (iii)
   replaces `nu = nu + abs(node_sum)` by `nu = nu + 0.0 * 0.0`. Each breakage runs the whole test
   module, so every catching test is recorded, not only the required ones.
9. **`git stash list` was run once** (`UNINTENDED DRIFT`), in a command checking for other
   sessions' commits on the branch. The dispatch says never to run `git stash` in any form. It is read-only; it printed a count
   (4) and changed nothing. No other `git stash` command was run.

## Verification performed

**The seven roots**, each by its own command (`CLAUDE.md`), after the change:

| Root | At `5ff8a5e4` (orchestrator) | After | |
|---|---|---|---|
| AdaptiveLevin | 82 | **91** | +9, the new module |
| ComputeTargets | 780 | 780 | OK (the wall-clock flake did not occur) |
| CosmologyModels | 39 | 39 | OK |
| LiouvilleGreen | 156 (1 skipped) | 156 (1 skipped) | OK |
| ObjectFactories | 280 | 280 | OK |
| RunDrivers | 127 | 127 | OK |
| tools | 64 | 64 | OK |

**Test 7, existing tests:** every test in the seven roots passes; **no existing assertion was
changed**. In particular `test_levin_leaf_values.TestSlowLeaves.test_slow_leaf_untrusted_and_refined`,
which asserts every checked leaf's trust equal to 06g's restated test ($\Delta_1 \le \phi_P$), passes:
route B never fires on its cell (observation 3).

**Tests 1–5:** 9 tests, OK (`PYTHONPATH=. ./venv/bin/python -m unittest
AdaptiveLevin.tests.test_levin_trust_yardstick`). Test 2 found the commit and the variant
bit-identical on all nine cells, and the commit and the parent bit-identical on the six where the
variant never fires. Test 2 is a §7 stop if it finds a difference; it found none.

**The breakages** (`scripts/breakages.py` → `out/breakages.txt`), each applied to
`_trust_noise_yardstick` by exact replacement, the whole test module run, then reverted; the module's
SHA-256 (`beaf0e23…`) equal to the clean file's after each revert and at the end:

| Breakage | Required | Failing tests | |
|---|---|---|---|
| (i) $\lvert E_L\rvert+\lvert E_R\rvert$ in place of $\lvert E_L+E_R\rvert$ | 1 | 1 (3 of its 5) | caught |
| (ii) a CC half's $\phi^-$ counted on a Levin parent | 1, 4 | 1, 2, 4 | caught; test 4 on its first assertion, $\nu_B = \phi_P$ (the broken yardstick is $2.83\times10^{-14} \ge \Delta_1$, so the leaf would be trusted too); test 1's guard test errors (`KeyError`) where a Levin parent's CC half lacks $\phi^-$ |
| (iii) the $\lvert\sum E\rvert$ term dropped | 1, 3 | 1, 2, 3 | caught; test 3's leaf $\Delta_1 = 7.163\times10^{-10} > \nu = 6.901\times10^{-10}$ |
| (iv) a CC half's full $\phi_h$ for $\phi^-_h$ | 1 | 1 | caught |
| (v) $\lvert\sum E\rvert$ before the $\phi^-$ terms | 1 | 1 | caught; the order case gives `…434fp-31` for `…434ep-31` |

Under (v), **test 2's comparison with the variant passes**: on its nine cells the reordered sum moves
no leaf's trust, so every result stays the variant's. The order is held by test 1, and on the census
by Gates M0 and M. Under (i) and (iv) test 2 also passes, for the same reason.

**Test 6, the LiouvilleGreen draws** (`out/lg_draws/`): `test_3bessel_analytic.py`, seeds 0–19, its
four default tests, three modes, 240 units, 8 processes, 381 s, all `pass`.
- **No test fails under the commit that passes under the parent:** 80 of 80 units pass under all
  three.
- **The commit equals the variant bit for bit:** on all 80 units (outcome, printed cases,
  three-Bessel results and the whole captured stdout) and all 4,320 Levin calls, every call field
  but the variant's `route_06l` (value, abserr, converged, counts, `node_*` sums, `regions_sha256`,
  `regions_full_sha256`, `result_sha256`), with the same result keys less `route_06l`. 0 differences.
- **Under the variant 122 calls fire** (06l: 122), 222 times on a Clenshaw–Curtis parent and 68 on
  a Levin parent; none withdraws. Against the parent, 121 calls change value or abserr, under the
  commit and the variant alike.
- **Honesty and convergence, parent → commit:** no convergence change (3,828 converged calls, 492
  not, under both); no call at `depth_max` under either; no node-limited call; no case moves between
  honest and dishonest. Near-singular `test_YJJ_log_singularity` cases ($\epsilon \le 10^{-4}$):
  560, 432 dishonest and converged under the parent, the commit and the variant alike (worst ratio
  $7.97\times10^5$), the true error changing on 3. Every other case: 520, none dishonest, 284
  converged, the true error changing on 13. Regions 73,464 → 73,013, solves 504,212 → 500,608. The
  largest value move against the two errors combined is 0.0033.

The draws were run twice: the first run's `lg_seeded_draws.py` was edited afterwards (its `report`
mode only), so the run was recomputed with the final script rather than re-stamped (`CLAUDE.md`
rule 7). Both runs gave the same comparison. The comparison script's first version compared
`result_keys` as a list including the variant's `route_06l` and so reported every call different on
that key alone; it was corrected to compare the key sets less `route_06l` before the recorded run.

**Records:** `black --check` clean on both changed Python files outside `calibration/scripts/`.
SHA-256 of the committed module `beaf0e23aabc…`, of the test module `7b5ddd335df6…`.

## Observations not acted on

1. **On `far(3e5)`, Levin leaves carry $\Delta_1$ far above their eq. (151) floor.** Test 5's leaf
   has $\Delta_1/\phi_P = 3.8\times10^3$ at $r = 0.63$, and the Levin leaf [300001.875, 300002.0] at
   depth 5 has $\Delta_1 = 4.6\times10^{-9}$ against $\phi_P = 3.4\times10^{-14}$ ($1.3\times10^5$×) at
   $r = 0.49$. The phase there moves by $\theta'\,{\rm ulp}(x) \approx 1.2\times10^{-8}$ rad across a
   node's rounding, and eq. (151) has no node term (the Clenshaw–Curtis floor has had one since 06d).
   The trust check refuses these leaves and refines them into fallback leaves, so the call stays
   honest by construction there; the floor is what understates. This is a constructed integrand,
   far from any span this campaign has examined, and I do not know whether A3 v2 has Levin leaves at
   $|x|\,\theta'$ this large. **No issue opened**: for the orchestrator whether it is one, perhaps
   beside 06e's end-node work.
2. **B fires only on noise-level leaves.** A scratch search over about a hundred clean integrands
   (deviation 4) found no firing at any request; the census's firings come from rounding in the
   integrand's inputs or in the nodes, which the two families here reproduce.
3. **`test_levin_leaf_values.trust_test()` restates 06g's rule** ($\Delta_1 \le \phi_P$), and
   `test_slow_leaf_untrusted_and_refined` asserts every checked leaf's trust equal to it. It passes
   because route B never fires on its cell; on a cell where B fires it would fail. Not changed (§2
   test 7 allows a change only where an assertion moves). Its docstring and the helper now describe
   the rule before 06m.
4. **The near-singular `test_YJJ_log_singularity` cases**, 432 of 560 dishonest and flagged
   converged, worst $7.97\times10^5$, are unchanged by 06m and identical under all three codes, as
   06g and 06l found.
5. **`out/lg_draws/lg_seeded_draws.jsonl` is 21 MB**, three times 06g's 6.8 MB record (three modes,
   and the result keys per call). It is committed as 06g's was; it could be dropped from the commit if
   the size matters more than the record.
6. **`calibration/scripts/` is left unformatted**, as the dispatch says (the copies stay diffable
   against 06l's), but `CLAUDE.md`'s list of record script directories exempt from `black --check`
   does not name it; `black --check` on it and on 06g's `calibration/scripts/` reports 10 files
   between them. Not changed (`CLAUDE.md` is outside this prompt's paths).

## State handed on

- **The code commit is local, not pushed.** Part B (§4 R25 at the code commit's parent and R26 at
  the code commit, each in a worktree; §5 the census at a tree whose code is the code commit's, Gates
  M0 and M) is dispatched from it once pushed. R1, M0 and M are above, not yet scored.
- **The header issue does not move here.** `[06g-slow-leaves-at-half-cost-convergence-outside-stationary-points]`
  moves only in 06m's records commit, after Part B, on the user's word; the narrower issue
  `[06m-calls-flagged-unconverged-within-their-request]` is proposed then, not now.
- The board's 06m row records Part A. `docs/OPEN_ISSUES.md` and the board's §3 are unchanged (no
  issue opened).
- Nothing was written under `var/`; no store, run or lock was touched.
