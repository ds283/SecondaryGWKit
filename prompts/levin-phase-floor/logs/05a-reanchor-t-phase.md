# levin-phase-floor 05a — rows 1 and 2 for $T$: the node-anchored remainder, against $2\pi$

> **§1–§4 only** (the keyword, the fixtures, the tests and these records), written by the
> executing agent on the laptop. §5 (rehearsals R11 and R12) and §6 (the re-measurement) run later
> on the Mac Studio. The orchestrator appends their record to this log.

**Subject:** prompt 05a §1–§4, on the laptop, in the main checkout
`/Users/ds283/Documents/Code/SecondaryGWKit`, branch `handover-remedial`.
- **Dispatch SHA:** `c4eaefd`. It adds only an orchestrator note to `a2ad0db`, which wrote the
  prompt. Nothing landed on the branch between the dispatch and the commit
  (`git log --oneline c4eaefd..HEAD` empty before staging).
- **The gate.** `git status --short` was empty. `runregistrykit list` showed nothing `running` or
  stale: 60 runs, 58 finished, 2 `unknown` (pre-registry). The seven roots' baselines are the
  orchestrator's, taken at `a2ad0db` (deviation 1).
- No Ray was started, no run was launched, no store was opened, and nothing was written under
  `var/`. The tests build no store.

**Commit:** this one commit (§4), not pushed. Its hash is not written here; the board's 05a row
carries none until the records commit.

**Result:**
- **§1.** `TkSourceFunctions(..., exact_remainder=True)`, passed to the `PrimitivePhase` of
  `_build_WKB`. Production never passes it, so $T$'s remainder is now node-anchored and reduced
  against $2\pi$, exactly as $G$'s since 05. No fallback. The module docstring, the comment on
  `TK_PHASE_SIGN` and `primitive_phase.py`'s "Who uses which" say so. No code in
  `primitive_phase.py` changed.
- **§2.** Every fixture that builds a `TkSourceFunctions` on a `ClosedFormPrimitive` `cs_tau`
  passes `exact_remainder=False`: nine sites (listed under What shipped). One more line, the
  signature of `test_quadsource_integral`'s `capture`, had to accept the keyword (deviation 3).
  05's `test_transfer_function_remainders_are_unchanged` is replaced by test 4(b). No other
  existing assertion changed.
- **§3.** `ComputeTargets/tests/test_t_phase_reanchor.py`, 10 methods, all passing, and 4(b) in
  `test_g_phase_reanchor.py`.
  - Test 2: every error within 0.19–0.29 of $4\varepsilon(k|p_z|+2\pi)$, $z_{\rm init}$ on and off a
    node, $k$ up to $3\times10^8$, $|\theta_T|$ up to $2.4\times10^{12}$ rad.
  - Test 3: σ_θ sampled in $z$ falls from $0.242\,\varepsilon|\theta_T|$ to
    $4.8\times10^{-8}\,\varepsilon|\theta_T|$ in the scored window, at a table node. Midway between
    nodes the construction's own residual $\varepsilon k|p_z|$ leaves $2.8\times10^{-4}$ (unscored).
- **The breakages.** All four were caught by the test the prompt names, and each was reverted.
- **The seven roots.** Six equal the baseline; `ComputeTargets` rose by 10, the methods added
  (the replaced test is renamed in place). (Verification, below.)
- **No stop condition of §8 arose.** Two findings bear on §6's T1 before it runs: σ_θ on a window
  that its 65 samples do not resolve is not a measurement, and between table nodes $T$'s
  node-anchored remainder keeps up to $\sim10^{-2}\,\varepsilon|\theta_T|$ (Observations 1 and 2).

## What shipped

**`ComputeTargets/TkSourceFunctions.py` (§1).**
- The constructor keyword `exact_remainder: bool = True`, documented in its docstring, held as
  `self._exact_remainder` and passed to `_build_WKB`'s `PrimitivePhase(...)`. Nothing switches it
  off: a `cs_tau` without node accessors makes `PrimitivePhase` raise its `ValueError`.
- The module docstring's "Phase" section gains a paragraph: what the remainder now is, that
  `z_init`'s end partial is non-zero, that production never passes the keyword and closed-form
  fixtures pass `False`, and that the abscissa's own rounding remains.
- The comment above `TK_PHASE_SIGN` says the same in four lines.
- `raw_theta`, `theta_deriv`, the amplitude, the regions and every check are untouched.

**`ComputeTargets/primitive_phase.py`.** The module docstring's "Who uses which" only, rewritten:
both production factors use the node-anchored remainder, $G$ since 05 (G1) and $T$ since 05a
(the user's decision of 2026-10-26); $T$'s anchor carries a non-zero end partial; closed-form
fixtures keep the default; neither mode touches the abscissa's rounding or the stored samples'
noise. No code changed (breakage (iv) was reverted, `cmp`-checked).

**The fixture sites (§2)**, each `exact_remainder=False` at a `TkSourceFunctions(...)` on a
`ClosedFormPrimitive` `cs_tau` (line numbers after the change):

| file | line | in |
|---|---|---|
| `ComputeTargets/tests/test_tk_source_functions.py` | 433 | `Fixture.exact_functions` |
| ″ | 468 | `Fixture.LG_functions` |
| ″ | 617 | `AnalyticRadiationFixture.functions` |
| ″ | 978 | `TestConsistencyChecks.test_nonzero_cos_coeff_raises` |
| ″ | 1313 | `TestPrimitivePhaseConsumer.test_inconsistent_friction_sample_raises` (its `build`) |
| ″ | 1359 | `TestPrimitivePhaseConsumer.test_sign_convention_is_checked` |
| ″ | 1404 | `TestPrimitivePhaseConsumer.test_missing_background_tables_are_refused_by_name` |
| `ComputeTargets/tests/test_phase_groups.py` | 658 | `matched_LG_functions` |
| `ComputeTargets/tests/test_quadsource_integral.py` | 426 | `Case.__init__`, the `exact=False` branch's `Tk_builder` |

and one line that is not a construction: `test_quadsource_integral.py:297`, `exact_Tk_inputs`'
`capture(model, k, Tk_numeric, Tk_WKB, exact_remainder=True)` (deviation 3).

`test_quadsource_parent_main.py` and `test_primitive_phase.py` name the class only in comments
(`:361`, `:480`); neither builds one. No other test module constructs a `TkSourceFunctions`. The
one test call of `evaluate_QuadSource_integral` (`test_quadsource_integral.py:526`) passes
`Tk_functions_builder`. The
scripts under `docs/` that build one are records, and were not edited.

**`ComputeTargets/tests/test_g_phase_reanchor.py` (§2).** Only the replaced test.
`TestNothingElseMoves.test_transfer_function_remainders_are_unchanged` became
`test_transfer_function_keyword_off_is_WKB_mod_2pi_of_raw_theta`, prompt 05a §3 test 4(b). Its
docstring says it replaces 05's test and why. It captures `_TableCsTauFixture`'s stored samples,
builds the `TkSourceFunctions` on its table `cs_tau` with `exact_remainder=False`, and keeps 05's
check unchanged: at all 173 WKB samples and the 172 midpoints, `theta_mod_2pi` equals
`WKB_mod_2pi(raw_theta)[1]` in `float.hex`.

**`ComputeTargets/tests/test_t_phase_reanchor.py` (§3), new: 10 methods.**
- **`_LCDMTk`, the background of tests 2–5.** 05's flat ΛCDM with radiation in 1/Mpc and its grid
  (100 nodes per decade, 1e12 to 0.1), imported from `test_g_phase_reanchor`.
  - `cs_tau` is a real `CumulativeTable` of $c_s/H$ at `CS_TAU_GAUSS_ORDER`, shifted by
    $c_s\tau_{\rm init}$ as `compute_background` shifts it, with $c_s^2 = 1/3$.
  - `friction_F` is a closed form; the stored friction samples are its own `delta`, bit-equal.
  - `z_init` is the node at $10^9$ ("on a node"), or 0.3 of the way in $\log(1+z)$ to the next
    node up ("off a node", $z_{\rm init} = 1.00693\times10^9$, $p_R = -1.843\times10^{-6}$ Mpc).
  - The stored phase is `WKB_mod_2pi(k cs_tau.delta(z, z_init) + phi(u))` at every grid node from
    `z_init` down to 0.1, with $\phi(u) = -0.09 + 0.02\sin u$. The numeric row holds 20 nodes
    above `z_init`, the switch node and 6 below.
- **Imported, not copied:** from `test_g_phase_reanchor`, `_TableCsTauFixture`,
  `_ProductionPhaseGk`, `_reference_remainder` and its mpmath context, the background and `_Value`;
  `exact_Tk_inputs` from `test_quadsource_integral`; `Case` and its helpers from
  `test_phase_groups`.

| test (§3) | method(s) |
|---|---|
| 1 | `TestProductionConfiguration.test_table_cs_tau_with_no_keyword_is_node_anchored`, `.test_closed_form_cs_tau_with_no_keyword_raises`, `.test_closed_form_cs_tau_with_keyword_off_builds` |
| 2 | `TestExactness.test_remainder_against_two_pi` (8 subtests: 4 $k$ × `z_init` on and off a node) |
| 3 | `TestNoJitter.test_sigma_theta_off_and_on` |
| 4 | (a) `TestNothingElseMoves.test_raw_theta_and_theta_deriv_are_unchanged`; (b) in `test_g_phase_reanchor` (above); (c) `TestNothingElseMoves.test_G_phase_from_build_phase_is_still_node_anchored` |
| 5 | `TestTWKB.test_T_WKB_moves_by_at_most_the_old_error` (8 subtests) |
| 6 | `TestEndToEnd.test_evaluate_sum_against_the_direct_product`, `.test_composed_remainder_moves_by_at_most_the_T_factors_old_errors` |

## Predictions and verdicts

§1–§4 score nothing. Prompt 05a §6's predictions, copied unchanged before §6 runs, as §4 asks.
They are **not yet scored**; §6 scores them on the Mac Studio. Observations 1 and 2 below bear on
how T1 can be measured.

**Predictions, fixed before the run** (README §5 rule 3). Copy them into the log unchanged.
- **T1 — the floor is there, and goes.**
  - **The rule:** in every probe window with $\varepsilon|\theta_T| \ge 10^{-11}$ rad, sampled in
    $z$: $\sigma_\theta \ge 0.1\,\varepsilon|\theta_T|$ under $c$, and
    $\sigma_\theta \le \max(10^{-3}\varepsilon|\theta_T|,\ 10^{-13}\ {\rm rad})$ under $t$.
  - **Holds** if true in every such window.
  - **The $u$-sampled $\sigma_\theta$ under $t$** is not scored. Wherever it exceeds
    $0.1\,\varepsilon|\theta_T|$, report it against $\varepsilon\,\tfrac{z}{1+z}|d\theta_T/du|$, and
    propose an issue for the abscissa term in the log's "Observations not acted on".
- **T2 — the cap does not return.**
  - **The rule:** on every row (all 39), under $t$ at `1e-7`, no group has a region at depth ≥ 18.
  - **Holds** if true on every row; an over-budget unit counts as false.
- **T3 — $T$'s errors were within the declared error.**
  - **The rule:** on every row (all 39), $|{\rm total}_t - {\rm total}_c| \le$ `total_abserr`$_c$,
    at `1e-7`.
  - A failure measures what $T$'s two errors cost a total beyond what it declared. That is a
    finding about the rows before this prompt, not a fault in it.
- **T4 — the declared error is honest after the change.**
  - **The rule:** on every one of the 31 depth-20 rows with both $t$ units in budget,
    $|{\rm total}_t(10^{-7}) - {\rm total}_t(10^{-8})| \le$ `total_abserr`$_t(10^{-7})$.
  - 05's L3, the same rule under $c$, held on 29 of 31.

| | Verdict |
|---|---|
| T1 | not yet scored (§6) |
| T2 | not yet scored (§6) |
| T3 | not yet scored (§6) |
| T4 | not yet scored (§6) |

## Deviations from the prompt

1. **The baselines are the orchestrator's.** **IMPLEMENTATION CHOICE.** The gate asks for the
   seven roots' counts at the dispatch `HEAD`. The orchestrator took them at `a2ad0db`, whose code
   is the dispatch SHA's (11:34–11:38 GMT), and passed them in the dispatch. They were not re-run
   before the change. The run after it is compared with them.
2. **Tests 2–5 run on a fixture of their own, `_LCDMTk`.** **IMPLEMENTATION CHOICE.**
   - §3 asks for a real `TablePrimitive` "as in `_TableCsTauFixture`", imported if it can be.
     It is imported, and used by tests 1, 4(b) and 6.
   - It cannot serve tests 2–5. Its background has $H_0 = 1$ and its WKB phase stops at
     $x = 10^3$ rad, so $k|\Delta_{\tau_s}| \ge 10^9$ (test 3) and $k$ up to $3\times10^8$/Mpc
     with production's magnitudes (test 2) are out of its reach. Its table starts at `z_init`,
     so `z_init` cannot be off a node.
   - `_LCDMTk` is built from 05's background pieces, imported, and reaches
     $|\theta_T| = 2.4\times10^{12}$ rad.
3. **`exact_Tk_inputs`' `capture` accepts the keyword.** **STRUCTURALLY REQUIRED.**
   - `test_quadsource_integral.py:301` patches `test_tk_source_functions.TkSourceFunctions` with
     `capture(model, k, Tk_numeric, Tk_WKB)` to collect a fixture's stored samples.
     `Fixture.exact_functions` now passes `exact_remainder=False`, so without the change the
     capture raises `TypeError`.
   - The signature gains `exact_remainder=True`, mirroring the real one, and nothing else changes.
     It is not an `exact_remainder=False` at a construction, so it is recorded here.
4. **Test 3's window: where it sits, a resolvability guard, and a failed first attempt.**
   **IMPLEMENTATION CHOICE.**
   - **Where.** §3 leaves the window's place to the agent ("choose it and record $z$ and the
     phase"). It is centred on the table node nearest $z = 0.12$, at $k = 2\times10^5$/Mpc, with
     `z_init` off a node (production's shape): $z = 0.12022644346174131$,
     $\theta_T = -1573308901.376984$ rad.
   - **Why a node.** The construction's own residual is the rounding of $v = -k p_z + \phi$, about
     $\varepsilon k|p_z|$. For $G$ that was negligible beside $\varepsilon k\eta_R$. For $T$, half a
     table interval of $\tau_s$ is about 1% of $|d\theta_T/du|/k$, so midway between nodes
     $\varepsilon k|p_z|$ is not negligible. That window is measured too, unscored:
     $2.8\times10^{-4}\,\varepsilon|\theta_T|$, over the scored bound of $10^{-4}$. At the node it
     vanishes.
   - **The guard.** The test asserts that the largest step of the unwrapped samples is below
     $\pi/2$ (it is 0.908 rad), and that $k|\Delta| \ge 10^9$ and one node holds the window.
   - **Why a guard, and the failed attempt.** The first prototype (in the scratchpad, not
     committed) put windows at $k = 3\times10^6$ and $3\times10^8$ at $z = 10^4$ down to $0.3$. At
     $z \le 100$ it gave σ_θ "on" of 0.01–0.2 $\varepsilon|\theta_T|$, even though the remainder
     there agreed with the mpmath reference to $8\times10^{-14}$ rad.
     - The phase moved ~1300 rad across the window, ~20 rad a step. `numpy.unwrap` folds each
       step by whole multiples of $2\pi$ *per index*.
     - The sampled doubles are not equally spaced to within their rounding ($\sim\varepsilon z$),
       so a fold that is linear in the index is not smooth in $z$. That is jitter of
       $\sim6\pi\,\delta z/h \approx 10^{-7}$ rad, of the order of $\varepsilon|\theta_T|$ itself.
     - A window is measurable only where $10^{-6}\,\tfrac{z}{1+z}|d\theta_T/du| \ll 64\pi$. With
       $k|\Delta| \ge 10^9$ that means small $z$ and modest $k$, hence the choice.
   - **The threshold** takes 05's strict reading: "at most $10^{-3}$ of that" is
     $10^{-3}\times0.1\,\varepsilon|\theta_T|$.
5. **Test 4(b) replaces 05's test in place, renamed, with local imports.** **IMPLEMENTATION
   CHOICE.**
   - §2: "Replace it with §3 test 4(b), and say so in its place." The method is renamed to say
     what it now tests, and its docstring says it replaces 05's test, citing §2.
   - `Fixture.exact_functions` now passes `exact_remainder=False`. So `_TableCsTauFixture`'s own
     `exact_functions()` would give a keyword-off object *by way of the fixture*, and 05's body
     would pass unchanged, for the wrong reason. The test instead captures the stored samples and
     passes the keyword itself.
   - The `patch` and `test_tk_source_functions` imports are local to the method, so that nothing
     else in the module changes.
6. **Three of the nine fixture sites build an object that raises before its phase is built.**
   **IMPLEMENTATION CHOICE.**
   - `test_nonzero_cos_coeff_raises`, `test_sign_convention_is_checked` and
     `test_missing_background_tables_are_refused_by_name` expect a `RuntimeError` from a check that
     runs before `PrimitivePhase`. They pass with or without the keyword.
   - They are on a `ClosedFormPrimitive` `cs_tau` (§2's criterion), so they pass it: each then
     tests its own check whatever the order of checks in the constructor. Without it, a reordering
     would surface `PrimitivePhase`'s `ValueError` instead, which is not a `RuntimeError`.
   - In the last of them, one subcase has `cs_tau = None` (the missing table). The call site is
     shared by both subcases.
7. **Test 5's bound.** **IMPLEMENTATION CHOICE.**
   - "Plus $10^{-14}$ relative" is read as $10^{-14}|M|$, relative to the amplitude:
     $|\Delta T_{\rm WKB}| \le |M|\,(4\varepsilon k|\Delta_{\tau_s}| + 0.18\,\varepsilon|\theta_T| +
     10^{-14})$.
   - $\Delta_{\tau_s}$ and $\theta_T$ are evaluated at $\log(1+z)$ with `x_is_log=True`, as `T_WKB`
     evaluates its phase. $|M|$ is asserted bit-identical between the two objects.
   - It runs at 202 redshifts (200 log-uniform and the WKB region's two ends) × 4 $k$ × 2 anchors.
8. **Test 6's bound.** **IMPLEMENTATION CHOICE.** "The sum of the two $T$ factors' old errors" is
   test 5's expression for each of $T_q$ and $T_r$ at the point, plus $10^{-14}$ rad. $G$ is the
   same `_ProductionPhaseGk` object in both group sets, so only $T$ moves.
9. **Test 2's abscissae** (left open by §3). **IMPLEMENTATION CHOICE.** 200 log-uniform in $u$
   from the WKB bottom up to `z_init` (above the switch node $\phi$ is clamped, within its
   cushion), plus every 25th stored sample, `z_init`, the switch node and the WKB bottom, passed
   as $z$.
10. **Test 1 is three methods, and covers both fixtures.** **IMPLEMENTATION CHOICE.** The
    production-configuration method checks `_TableCsTauFixture` and `_LCDMTk` at both anchors.
    Test 4(a) runs at $k = 3\times10^6$ and $3\times10^8$, both anchors, 241 abscissae each.
11. **`TkSourceFunctions` holds the keyword as `self._exact_remainder`**, with no property of its
    own. **IMPLEMENTATION CHOICE.** `phase.exact_remainder` already exposes it, and `_build_WKB`,
    called from the constructor, reads the attribute.

12. **A stray scratch file under `/tmp`.** **UNINTENDED DRIFT.** While checking that the
    predictions were copied byte for byte, one shell command redirected an intermediate `sed`
    output to `/tmp/x` (1546 bytes, the copied predictions) instead of the session scratchpad. It
    was deleted at once. Nothing read it; it touched nothing in the repository or under `var/`.
    The copy itself was checked with `diff` against the prompt's text: identical but for the
    trailing blank line.

Deviation 12 is the only `UNINTENDED DRIFT`.

## Verification performed

**The seven roots.** The commands of `CLAUDE.md`, run serially after the change.

| root | baseline at `a2ad0db` (orchestrator, 11:34–11:38 GMT) | after the change | change |
|---|---|---|---|
| AdaptiveLevin | 32 OK | 32 OK | equal |
| ComputeTargets | 754 OK | 764 OK | +10: the 10 methods of `test_t_phase_reanchor`, plus test 4(b), less the one test it replaces |
| CosmologyModels | 39 OK | 39 OK | equal |
| LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) | equal |
| ObjectFactories | 280 OK | 280 OK | equal |
| RunDrivers | 127 OK | 127 OK | equal |
| tools | 55 OK | 55 OK | equal |

- `uptime` before the run: `11:57  up 21 days, 10:42, 2 users, load averages: 4.62 4.10 7.34`
  (11:57:48 UTC); after it: `12:01  up 21 days, 10:46, 2 users, load averages: 3.41 3.55 6.28`
  (12:01:47 UTC). `ComputeTargets` took 164.8 s.
- The known flake, `test_tk_wkb_phase … test_wall_time_per_object`, did not fire, so it was not
  re-run alone.

**§3, measured** (each test prints its numbers):

| test | number |
|---|---|
| 1 | table `cs_tau`, no keyword: `exact_remainder` true (`_TableCsTauFixture`, `_LCDMTk` both anchors). `ClosedFormPrimitive`, no keyword: `ValueError` "… ClosedFormPrimitive lacks end_partial, node_limbs". Keyword off: builds |
| 2 | worst err/bound on a node 0.289, 0.239, 0.252, 0.260 and off a node 0.223, 0.194, 0.258, 0.249, at $k$ = 1e5, 3e6, 1e8, 3e8. $k p_R$ off a node: −0.1843, −5.529, −184.3, −552.9 rad. Max $\lvert\theta_T\rvert$ 7.9e8 to 2.4e12 rad; 241–242 points and 211–218 node constants per case |
| 3 | see below |
| 4 | (a) `raw_theta` and `theta_deriv` (both forms) bit-identical at 964 abscissae; (b) 345 of 345 remainders bit-identical to `WKB_mod_2pi(raw_theta)[1]`, keyword off; (c) `_build_phase` gives `exact_remainder` true, sign −1 |
| 5 | worst $\lvert\Delta T\rvert$/bound 0.27–0.39 over the 8 cases; 195–202 of 202 points moved |
| 6 | Oracle 1 max relative residual $4.62\times10^{-16}$ (bound $10^{-12}$); composed remainder on against off at worst $2.01\times10^{-13}$ rad, 0.242 of its bound, 781 of 800 points moved |

**Test 3**, $k = 2\times10^5$/Mpc, `z_init` off a node, window of relative width $10^{-6}$ in $z$,
65 doubles. $\varepsilon|\theta_T| = 3.4934\times10^{-7}$ rad at the node.

| window | $z$ | $\theta_T$ (rad) | sampled in $z$: off | on | sampled in $u$: off | on | $\varepsilon\tfrac{z}{1+z}\lvert d\theta_T/du\rvert$ |
|---|---|---|---|---|---|---|---|
| **at a node (scored)** | 0.12022644346174131 | −1573308901.376984 | 8.4469e-8 rad = **0.2418** $\varepsilon\lvert\theta_T\rvert$ | 1.6739e-14 rad = **4.79e-8** | 8.8175e-8 = 0.2524 | 1.8764e-9 = 0.0054 | 1.2910e-8 = 0.0370 |
| midway between nodes (unscored) | 0.12162578626949438 | −1572632406.9830265 | 8.9750e-8 = 0.2570 | 9.8017e-11 = 2.81e-4 | 8.4091e-8 = 0.2408 | 2.0315e-9 = 0.0058 | 1.3051e-8 = 0.0374 |

- The scored assertions: off $\ge 0.1\,\varepsilon|\theta_T|$ (0.2418) and on
  $\le 10^{-4}\,\varepsilon|\theta_T|$ ($4.79\times10^{-8}$). The largest unwrapped step is 0.908
  rad.
- Sampled in $u$, as production calls the phase, the keyword takes σ_θ from 0.25 to
  $0.005\,\varepsilon|\theta_T|$ at $z \approx 0.12$. What is left is the abscissa term, as §0.1
  predicts: $z/(1+z) = 0.107$ here, so it is small.

**The deliberate breakages.** Each was made in the working tree, run, recorded, and reverted by
copying back a saved original, checked with `cmp`. `grep -rn BREAKAGE ComputeTargets/` finds
nothing.

- **(i) The keyword's default `False`.** The edit, at `TkSourceFunctions.__init__`:
  `exact_remainder: bool = False`.
  - **Caught by test 1:** `test_table_cs_tau_with_no_keyword_is_node_anchored`
    (`AssertionError: False is not true`) and `test_closed_form_cs_tau_with_no_keyword_raises`
    (`AssertionError: ValueError not raised`).
  - **Also caught** by test 2 in all 8 subtests (up to $3.1\times10^{-4}$ rad at $k = 3\times10^8$,
    e.g. `54960377201.04252 not less than or equal to 1.0`), test 3
    (`8.446862610955506e-08 not less than or equal to 3.493447534312874e-11`), test 4(a)'s
    `assertTrue(on.exact_remainder)` in 4 subtests, and test 6's precondition.
    `FAILED (failures=16)`.
- **(ii) A silent fallback to `False`** when `cs_tau` lacks the accessors. The edit:
  `self._exact_remainder = bool(exact_remainder) and all(callable(getattr(model.functions.cs_tau, name, None)) for name in ("end_partial", "node_limbs"))`.
  - **Caught by test 1's raise**, `test_closed_form_cs_tau_with_no_keyword_raises`:
    `AssertionError: ValueError not raised`. Nothing else failed (`FAILED (failures=1)`), as
    expected: on a table the fallback never fires.
- **(iii) `_build_phase` turns the node-anchored remainder off for $G$.** The edit, in
  `GkSourcePolicyData._build_phase`: `exact_remainder=False`.
  - **Caught by test 4(c)**, `test_G_phase_from_build_phase_is_still_node_anchored`
    (`AssertionError: False is not true`), and by test 6's precondition. `FAILED (failures=2)`.
  - 05's `test_build_phase_turns_the_node_anchored_remainder_on` fails too
    (`FAILED (failures=1)` of 12).
- **(iv) $p_R \to 0$ in `PrimitivePhase`'s anchor.** The edit, at `primitive_phase.py:329`:
  `self._anchor_exact = Fraction(hi_R) + Fraction(lo_R)`.
  - **Caught by test 2 off a node**, in all 4 subtests, by exactly $k p_R$ reduced:
    0.1843 rad at $k = 10^5$, 0.754 rad at $3\times10^6$ ($-5.529 + 2\pi$), 2.095 at $10^8$ and
    0.0026 at $3\times10^8$. For example
    `AssertionError: 33026552756483.41 not less than or equal to 1.0 : (316.22776601683796, False, 0.1843076367314538)`.
  - On a node, all 4 pass, since $p_R = 0$ there.
  - **Also caught** by test 5 off a node, in all 4 subtests. `FAILED (failures=8)`.

**Other checks.**
- **The existing suites of the touched modules** (`test_tk_source_functions`, `test_phase_groups`,
  `test_quadsource_integral`, `test_g_phase_reanchor`, `test_primitive_phase`,
  `test_quadsource_parent_main`): 137 tests, OK, with the fixture change and before the new
  module was written.
- **`black`** on the seven `.py` files changed; only the new module was reformatted.
- **`git diff --stat 120fdfe HEAD -- . ':!docs' ':!prompts'`**, before the commit, shows only
  `CLAUDE.md`: 05's code is this commit's parent's.

## Observations not acted on

1. **For §6's T1: a probe window its 65 samples do not resolve measures `numpy.unwrap`, not the
   phase.**
   - σ_θ needs every step between samples well below $\pi$. A window of relative width $10^{-6}$
     moves $\theta_T$ by about $10^{-6}\tfrac{z}{1+z}|d\theta_T/du|$. Where $|\theta_T| \gtrsim
     10^9$ that is hundreds to thousands of rad at $z \gtrsim 1$. Examples, on `_LCDMTk`:
     1330 rad at $k = 3\times10^6$, $z = 100$; 4300 rad at $z = 1$.
   - Then `unwrap` folds by a whole number of cycles per index. Rounding of the sampled doubles
     turns that into jitter of order $\varepsilon|\theta_T|$, whatever the remainder.
   - Measured at $k = 3\times10^6$ and $3\times10^8$, $z = 100$ to $0.3$, on node-centred windows:
     σ_θ "on" was 0.01–0.11 $\varepsilon|\theta_T|$ in $z$, against a remainder exact to
     $8\times10^{-14}$ rad.
   - **Bearing:** T1's under-$t$ clause, $\le 10^{-3}\varepsilon|\theta_T|$, would fail on such
     windows for a reason that is not the phase. A guard on the largest unwrapped step, or a window
     narrow enough to resolve, would separate the two. **Proposed issue**, for the orchestrator to
     open or not: `[05a-unresolved-probe-window-measures-the-unwrap]`.
2. **For §6's T1 and for the construction: between nodes, $T$'s node-anchored remainder keeps
   $\varepsilon k|p_z|$, up to $\sim10^{-2}\,\varepsilon|\theta_T|$.**
   - $v = -k p_z + \phi$ is formed in double. $|p_z|$ is up to half a table interval of $\tau_s$,
     about $0.0115\,|d\theta_T/du|/k$ on a 100-per-decade grid.
   - For $G$ it was negligible, because $|\theta_G| \approx k\eta_R$ dwarfs $k|p_z|$ wherever the
     cliff was. For $T$, $|\theta_T| \approx k\tau_s(z)$ is of the order of $|d\theta_T/du|$, so the
     ratio is a fixed fraction.
   - Measured on resolved windows, sampled in $z$: midway between nodes $7.5\times10^{-3}\,
     \varepsilon|\theta_T|$ at $z \approx 10^4$ ($k = 3\times10^6$) and $2.8\times10^{-4}$ at
     $z \approx 0.12$ ($k = 2\times10^5$). At a node: $5\times10^{-8}$ to $1\times10^{-6}$.
   - **Bearing on T1:** T1's windows sit at each sub-interval's low-$z$ end. If that end is a
     table node and the window is at it, T1's under-$t$ clause sees the node value. If not, it can
     see up to $\sim10^{-2}$, above its $10^{-3}$.
   - **Bearing on production:** it is below the abscissa term wherever the latter is not
     suppressed. $\varepsilon\tfrac{z}{1+z}|d\theta_T/du|$ is $\sim0.5$–$0.9\,\varepsilon|\theta_T|$
     at $z \gg 1$ and 0.037 at $z = 0.12$. So it is not what production sees. Removing it would mean
     a finer anchor than the table node, which no prompt has asked for.
3. **The `handover_continuity` metadata of `QuadSourceIntegral` moves.** It is computed from
   `TkSourceFunctions.handover_continuity`, which takes $\sin$ and $\cos$ of `theta_mod_2pi` at the
   switch and overlap nodes, and is stored in the row's metadata (`QuadSourceIntegral.py:417`). So
   §5's comparison should expect differences in `QuadSourceIntegral.metadata` there, inside the
   class whose values are meant to move.
4. **Three comments and docstrings are now stale, outside this commit's allowed lines.**
   - `primitive_phase.py:324-325`, the constructor's comment on the anchor: "(zero when the
     anchor is a node, as in production)". In production that is now true for $G$ only. The
     dispatch allows only the module docstring's "Who uses which" in that file.
   - `test_g_phase_reanchor.py`'s module docstring (lines 11 and 20–22) still says
     "`TkSourceFunctions` does not" and that a `TkSourceFunctions` phase built as production builds
     it keeps its remainders. The dispatch allows only the replaced test in that file.
   - `_TableCsTauFixture`'s docstring says its `TkSourceFunctions` "builds its phase exactly as
     production does". Through `exact_functions()` it now gets `exact_remainder=False`; the tests
     that need production's configuration capture its samples instead (deviation 5).

   A follow-up commit could correct all three; none changes behaviour.
5. **`test_t_phase_reanchor` imports from three other test modules** (`test_g_phase_reanchor`,
   `test_phase_groups` and `test_quadsource_integral`), as §3 asks ("import that fixture rather
   than copying it"). A rename of their private helpers would break it at import.

## State handed on

- **One commit**, not pushed. It holds:
  - `ComputeTargets/TkSourceFunctions.py`;
  - `ComputeTargets/primitive_phase.py` (the docstring's "Who uses which" only);
  - `ComputeTargets/tests/test_t_phase_reanchor.py` (new);
  - `ComputeTargets/tests/test_g_phase_reanchor.py` (the one replaced test);
  - `ComputeTargets/tests/test_tk_source_functions.py`, `test_phase_groups.py` and
    `test_quadsource_integral.py` (the fixture sites, and deviation 3's line);
  - this log, and the board's 05a row.
- `docs/OPEN_ISSUES.md` is untouched: no issue was opened. Observation 1 proposes one.
- **The header's issues do not move in this commit** (§4): `[05-t-remainder-keeps-the-anchoring-floor]`
  and `[04-g-phase-is-reduced-against-the-double-two-pi]`. They move in 05a's records commit,
  after §5 and §6.
- **For §5:** R11 at this commit's parent, R12 at this commit. `QuadSourceIntegral`'s totals and
  its `metadata` (`handover_continuity`, observation 3) are expected to move; nothing else is.
- **For §6:** observations 1 and 2, before T1's probe is written.
