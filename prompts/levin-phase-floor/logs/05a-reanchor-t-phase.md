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
*Orchestrator, after the review:* the commit is **`1fc349c`**. It passed all nine checks of the
review on the laptop and was pushed. The follow-ups are `bd38e50` (observation 4's three comments,
on the user's word) and `2db3f9e` (T1 amended by prompt §9, on the user's decision, after
observations 1 and 2).

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

**T1 above was replaced before the run by prompt §9** (the user's decision, 2026-10-26, after
observations 1 and 2 below; added by the orchestrator after the review of `1fc349c`). No §6 number
existed. The amended T1, copied unchanged from prompt §9, is scored in its place; the probe window
is narrowed until it is resolved, as §9 states:
- **The rule:** in every resolved probe window with $\varepsilon|\theta_T| \ge 10^{-11}$ rad,
  sampled in $z$: $\sigma_\theta \ge 0.1\,\varepsilon|\theta_T|$ under $c$, and
  $\sigma_\theta \le \max\big(10^{-3}\varepsilon|\theta_T|,\ \varepsilon(k\max|p_z| + 2\pi),\ 10^{-13}\ {\rm rad}\big)$
  under $t$.
- **Holds** if true in every such window. **Fails** if false in any. **Undecided** if none is
  false but some window with $\varepsilon|\theta_T| \ge 10^{-11}$ rad is unresolved.
- **Also report, unscored,** in how many windows the term $\varepsilon(k\max|p_z| + 2\pi)$ is the
  largest of the three, and $\sigma_\theta/(\varepsilon|\theta_T|)$ under $t$ in those windows.

| | Verdict (scored on the Mac Studio, §6 below; re-scored there by its orchestrator) |
|---|---|
| **Gate T0** | **holds**: (a) `CLAUDE.md` alone; (b) $c$ equals 05's $c$ bit for bit on both rows (and on all 39); (c) two $t$ runs identical |
| **T1** (as amended by §9) | **holds**, by the true step, the reading of record (the user's decision; see below): 70 of 70 windows, 0 unresolved; $\sigma_c$ 0.198–0.374 $\varepsilon\lvert\theta_T\rvert$, $\sigma_t$ at most $1.31\times10^{-5}\,\varepsilon\lvert\theta_T\rvert$ (0.0131 of its bound). Read literally, as `predictions.md` first fixed it, it fails in 36 of 70, all aliased windows |
| **T2** | **holds**: 39 of 39; the largest depth under $t$ at `1e-7` is 11 |
| **T3** | **holds**: 39 of 39; the largest $\lvert t-c\rvert/$`total_abserr`$_c$ is $4.66\times10^{-6}$ (`low-G-1`) |
| **T4** | **fails**: 29 of 31, false on `mid-G-r-3` (1.29) and `mid-G-3` (1.70), the rows and ratios of 05's L3 under $c$; holds on the 11 rows capped under B1 (largest 0.166) |

**T1's reading of record, and when it was decided** (the laptop orchestrator, in the records
commit). The re-measurement agent fixed both readings of §9's step rule in `predictions.md` at
12:30:03Z, before the run: the literal (folded) step to be scored, the true step to be reported.
It asked which §9 meant. Both were scored at 12:35:49Z. The question reached the laptop while the
run was reported as going, and the laptop told the user that no T1 number existed. That was wrong.
The user chose the true step without having seen either number. The ruling reached the Mac Studio
at about 13:23Z. Once the laptop had corrected the record of the timing, the user confirmed the
true step as T1's verdict of record. Prompt §9's wording is corrected in the records commit to say
so; `predictions.md` and the literal reading's 36 failures stay as they were.

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
     open or not: `[05a-unresolved-probe-window-measures-the-unwrap]`. *(Orchestrator: not opened. The user
     amended T1 before §6 instead (prompt §9, `2db3f9e`): an unresolved window is narrowed until it
     resolves, and one that cannot be is undecided.)*
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

   A follow-up commit could correct all three; none changes behaviour. *(Orchestrator: corrected in `bd38e50`, on
   the user's word, comments and docstrings only.)*
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

---

# The Mac Studio record: §5 and §6

> **Appended by the laptop orchestrator** in 05a's records commit (orchestrator note §B.4). Part B
> ran on the Mac Studio on 2026-10-26, from the dispatch SHA `149c393` (main checkout, code
> `1fc349c` up to `bd38e50`'s comments) and the worktree `SecondaryGWKit-lpf05a-c4eaefd` (detached
> at `c4eaefd`, the code commit's parent). Its orchestrator dispatched two agents, whose logs follow
> as they wrote them. Its review passed all ten checks (`docs/levin-phase-floor/05a/review.md`).
>
> A path `scripts/…`, `out/…` or `state/…` below is under `var/levin-phase-floor-05a/rehearsal/` or
> `…/wider/` there, and is committed at `docs/levin-phase-floor/05a/rehearsal/` or `…/wider/`. Not
> copied back: the run directories' checkpoints and unit pickles, the readers' `extract/` (177 MB),
> the comparison's full `store_value_diff.json` (41 MB) and `.txt` (10 MB), `qsi_rows.csv`, and the
> pickles under `wider/state/`. They stay in the Mac Studio's `var/`.
>
> **One question, answered by the user on 2026-10-26:** which reading of §9's step rule is T1's.
> The user chose the true step. The timing is under Predictions and verdicts above.
>
> **The issues the agents proposed.** X2 is opened as
> `[05a-t-phase-abscissa-rounding-remains]`. X1 is not opened: the user's ruling, and §9's wording
> corrected in the records commit, settle it. The rehearsal agent's four recurrences, and X3–X7, are
> recorded on the board where they bear.

## §5 — rehearsals R11 and R12 on the Mac Studio

**Prompt:** [`05a-reanchor-t-phase.md`](../05a-reanchor-t-phase.md) §5, under the orchestrator's
Part B dispatch, its §B.1 and §B.5 (§B.5 governing).
**Host:** mps026298, the Mac Studio.
- The main checkout was at the dispatch SHA `149c393`. Its code is the code commit `1fc349c`'s, up
  to `bd38e50`'s comment-only changes.
- The worktree `SecondaryGWKit-lpf05a-c4eaefd` was detached at `c4eaefd`, the code commit's parent.

**Model:** Claude Opus 5.5, as the Mac Studio orchestrator's rehearsal subagent, in three turns:
- the gate, the lock and R11's launch;
- R11's checks and R12's launch;
- R12's checks, the comparison, the classification and tabulation, the readers, the release and
  these records.

**Date:** 2026-10-26, 12:20–12:33 UTC.
**Commit:** none. The records are committed on the laptop in 05a's records commit.
**Result:** COMPLETE WITH DEVIATIONS. None of the deviations stopped the work, and none changed a
computation.
- R11 and R12 both ended `done` with exit 0 and fingerprint `1b332ffa…`. Each is equal to R8's
  class by class, and R12's `classes` object equals R11's.
- Every value table of every class is bit-identical between them: 7 tables, 360,462 rows.
- Every row-column difference outside `QuadSourceIntegral` is one of the four volatile kinds. 0 are
  unclassified, and no row is in one store only.
- `QuadSourceIntegral` differs:
  - `total` moved in 424 of 864 rows, all through `WKB_Levin`;
  - $\lvert\Delta{\rm total}\rvert/\lvert{\rm total}\rvert$ is at most 2.9e-11;
  - $\lvert\Delta{\rm total}\rvert/$`total_abserr` is at most 8.8e-5;
  - no depth, region count or convergence flag changed;
  - the `handover_continuity` metadata moved in every row.
- The readers exited 0, and the stores were left byte-identical.
- The lock was released.

The measurements are in `verification.md`, written for `docs/levin-phase-floor/VERIFICATION.md`
§5a "Layer 1(a) for $T$".

### What shipped

Nothing in the repository: no code, and no commit. Everything was written on the Mac Studio, in
the main checkout:
- **The records directory** `var/levin-phase-floor-05a/rehearsal/`, holding:
  - `out/`;
  - `scripts/`: 05's `classify_non_qsi.py` and `tabulate_qsi.py`, unchanged;
  - `state/`;
  - `extract/`, the readers' output, 177 MB, not for copying back;
  - `log.md`, `verification.md` and `issues.md`.
- **The stores** `var/datastores/levin-phase-floor-05a-r11*.sqlite` and `-r12*.sqlite`, five files
  each.
- **The runs** `levin-phase-floor-05a-r11-parent-20261026T122041` and
  `levin-phase-floor-05a-r12-reanchor-t-20261026T122434`, with their launch logs
  `var/runs/levin-phase-floor-05a-r{11,12}.launch.log`.

The stores are kept, as prompt §5.7 says.

### Predictions and verdicts

§5 fixes no scored predictions; §6 holds them. The acceptance checks of §7.2, and the dispatch's
own:

| check | number | verdict |
|---|---|---|
| R11 ends `done` with `1b332ffa…` | `done`, exit 0, `1b332ffa…`, 21/21 class digests = R8's | holds |
| R12 ends `done` with `1b332ffa…` | `done`, exit 0, `1b332ffa…`, 21/21 = R8's; `classes` object = R11's | holds |
| every value table outside `QuadSourceIntegral` identical | all 7 value tables bit-identical (360,462 rows) | holds |
| every other difference there of the four kinds; no row in one store only | 0 unclassified; 0 only-A, only-B or unmatched in any class | holds |
| `QuadSourceIntegral` differences tabulated | 424/864 `total`s; max rel 2.9e-11; max \|Δ\|/abserr 8.8e-5 | done |
| the readers exit 0 | 6 × exit 0; store byte-identical | holds |
| the lock is released | `release` exit 0; `status` free; `ray-processes` none | holds |
| (dispatch) R11 ≠ R12 in `QuadSourceIntegral` | not value-identical, so each tree ran its own code | holds |

### Deviations from the prompt

#### 1. R11 ran from a worktree, on the main checkout's venv — STRUCTURALLY REQUIRED

Prompt §5.2 calls for a worktree, and the dispatch gave it. The worktree has no venv, so R11 used
the main checkout's `venv/bin/python` by absolute path, with its cwd and `PYTHONPATH` set to the
worktree. Its `--database` and launch log were absolute paths into the main checkout. As for any
worktree run, the registry recorded it under the main checkout's `var/runs/`.

The evidence that R11 ran the worktree's code:
- the manifest records `git_head` `c4eaefda…`, `git_dirty` false, and the worktree's launcher;
- `lsof` showed that the driver's cwd, and that of the `Datastore` actor 22808, was the worktree;
- R11 and R12 differ in `QuadSourceIntegral`.

#### 2. `PYTHONDONTWRITEBYTECODE=1` on R11 only — IMPLEMENTATION CHOICE

It keeps `__pycache__` out of the worktree, which must not be written, and it changes no
computation. This follows R9 in 05. R12 used R10's launch form. After both runs the worktree's
`git status --short --ignored` was empty, and `find -newer state/r11_launch_marker` listed
nothing.

#### 3. R12's `--database` and launch log were given as absolute paths — IMPLEMENTATION CHOICE

R10 in 05 gave both as paths relative to the main checkout. The dispatch asked for absolute paths
for both stores. Both forms name the same files from the main checkout's cwd. The manifest's
`results` is relative either way.

#### 4. A shared `--job-name levin-phase-floor-05a` — IMPLEMENTATION CHOICE

This is the dispatch's name, as 05 shared `levin-phase-floor-05`. The job name reaches only
`label`, which differs anyway by its timestamp.

#### 5. A text form of the comparison, with `--max-rows 1000000` — IMPLEMENTATION CHOICE

This was run beside the `--json` form that §5.4 asks for, as 05 did. The JSON carries every row
whatever that setting is.

#### 6. `/usr/bin/time -p` wrapped the comparison — IMPLEMENTATION CHOICE

Its three lines (`real`, `user`, `sys`) land in `out/store_value_diff.{json,txt}.stderr`, which
hold nothing else: the tool wrote nothing to stderr. 05's files have the same form.

#### 7. Two inline read-only Python summaries, not saved as scripts — IMPLEMENTATION CHOICE

The per-class summary `out/store_value_diff_summary.txt` reads only the JSON; its text is in
`out/commands.log` verbatim. `out/qsi_extra_counts.txt` reads only `out/qsi_rows.csv` and
`out/classify_non_qsi.txt`. Neither opens a store. 05's summary was made the same way. 05's other
two scripts, `gkwkb_reuse_rows.py` and `parent_serials.py`, were not re-run; §5 does not ask for
them.

#### 8. The two copied scripts' docstrings still name R9 and R10 — IMPLEMENTATION CHOICE

They are copied **unchanged**, as §5.4 says, and their SHA-256 equal 05's (`26690e61…`,
`b7508351…`). Both take their stores and JSON as arguments, and neither prints a store's name: the
outputs say A and B. **Here A = R11 (parent `c4eaefd`) and B = R12 (code `1fc349c`).** Neither
needed a change to read R11 and R12, and both exited 0 on their first run.

#### 9. The gate record lists this agent's own directory — UNINTENDED DRIFT

The command that wrote `out/gate.txt` began with `mkdir -p` of `var/levin-phase-floor-05a/rehearsal/`.
So its `ls var/levin-phase-floor-05a` shows `rehearsal`, and cannot show the directory's absence
before the run. The other gate lines are unaffected:
- `ls var/datastores | grep levin-phase-floor-05a` found nothing;
- the registry held 16 runs, all finished.

The orchestrator's §B.5 recorded at 12:12 GMT that the directory did not exist. Prompt §6's
`wider/` was not there at the gate either.

#### 10. The stores' fingerprints were not re-taken — IMPLEMENTATION CHOICE

Both fingerprints are the launcher's, taken at the end of each build and read from `status.json`.
- R11's five files are byte-identical from just after its build (12:24) to the end.
- R12's files were first hashed at 12:28, after its fingerprint and before any read. They are
  identical to the end.

### Verification performed

| check | result | where |
|---|---|---|
| gate | both trees at their SHAs and clean; 16 runs finished; lock free; no 05a store; the launcher `5b984c75…` in both trees | `out/gate.txt` |
| R11 | `done`, exit 0, pid 22684, `1b332ffa…`, `c4eaefda…`, `git_dirty` false; 0 Tracebacks; `stderr.log` empty | `out/r11_end_check.txt`, `out/r11_status.json` |
| R12 | `done`, exit 0, pid 24897, `1b332ffa…`, `149c3938…`, `git_dirty` false; 0 Tracebacks; `stderr.log` empty | `out/r12_end_check.txt`, `out/r12_status.json` |
| worktree unwritten | `status --ignored` empty; `find -newer` empty, after both runs and at the end | `out/r11_end_check.txt`, `out/r12_end_check.txt`, `out/final_check.txt` |
| scripts | the two copies, the tool and `run_readers.sh` all have 05's SHA-256 | `out/script_sha256.txt` |
| value tables | 7 tables, 360,462 rows, all bit-identical | `out/store_value_diff_summary.txt` |
| non-QSI row columns | 4 kinds, 0 unclassified | `out/classify_non_qsi.txt` |
| QSI tabulation | pairing = the tool's identities; 0 `total` mismatches against the tool | `out/qsi_tabulation.txt`, `out/qsi_rows.csv` |
| readers | 6 × exit 0; lookups equal 05's; store identical | `out/readers.txt` |
| stores | 10 files identical from before the comparison to the end | `state/store_sha256_before_diff.txt`, `state/stores_sha256_final.txt` |
| lock | acquire 0; set-pid 0 ×2; ray-processes 0 ×3; release 0; status 0 | `out/lock.log` |

### Observations not acted on

- **The change is small on this store.** 05's change to $G$ moved 542 `total`s by up to 4.0e-5
  relative and 28× `total_abserr`, and changed depths and regions. This change to $T$ moves 424
  `total`s by at most 2.9e-11 relative and 8.8e-5 × `total_abserr`. It changes no depth, region
  count or convergence flag.
  - By depth, the median $\lvert\Delta\rvert/$`total_abserr` is 2e-9 to 1.1e-8 at depths 2–8, and
    1.5e-6 to 1.7e-5 at depths 10–14.
  - This is recorded, not judged. §6 scores the effect on A3 v2's rows.
- **`handover_continuity` moves in every row,** even where nothing else does. Its keys
  `partition.handover.T{q,r}.handover_continuity[i].d{T,Tprime}_rel` differ in all 864 rows, as
  §B.5 expected. In 292 rows they are the only non-wall-clock keys that differ: the 291 with no
  WKB part, and one at depth 0. This is a diagnostic, not a value that enters `total`.
- **26 rows have `total_abserr` = inf in both stores,** at depths 0–8. 12 of their `total`s differ,
  by 1.1e-16 to 1.0e-15 relative. The tabulation prints their ratio to `total_abserr` as 0. This is the open issue
  `[04-roundoff-floor-can-be-infinite]`, and it is not this prompt's.
- **The GkWKB residual-memo counters** differ in 31 rows here and in 25 rows in 05's pair. This is
  further circumstantial support for 05's P2: the counters depend on scheduling. No same-code pair
  exists to prove it.
- **The readers' file counts differ from 05's** (Gk 387 against 450, GkSource 162 against 142, QSI
  73 against 74). The lookups are equal. Each reader draws an unseeded `random.sample`: the open
  issue `[10-the-readers-draw-their-work-lists-unseeded]`.
- **`[05-store-value-diff-compares-volatile-columns]`** recurs exactly as 05 found it. It is
  already open.
- **Prompt §6's run `levin-phase-floor-05a-wider-20261026T123047`** started at 12:30:47 and was
  `done` (109/109) by the final check. It is not this agent's, and it was not touched.
- **The size of `out/`.** `store_value_diff.json` is 41 MB and `.txt` 10 MB, as in 05. The
  orchestrator decides whether to copy them back whole. `store_value_diff_summary.txt`,
  `classify_non_qsi.txt` and `qsi_rows.csv` carry what the records cite.

### State handed on

- The lock is free: `status` reports the lock directory does not exist, and there are no Ray
  processes of `/tmp/ray-sgk`. `~/.raypoolkit/` is empty.
- The registry holds 19 runs, all finished: R11 and R12 `done`, and §6's `wider` `done`.
- **R11's store** has primary `85538005…` and **R12's** has primary `94dae1cc…`. Both are in
  `var/datastores/`, both carry fingerprint `1b332ffa…`, and neither has a sidecar. They are kept.
- The main checkout is at `149c393` and clean. The worktree is at `c4eaefd`, clean and unwritten.
- The records' links assume they are copied to `docs/levin-phase-floor/05a/rehearsal/`.

## §6 — the wider re-measurement, with §9's T1 (Mac Studio)

**Subject:** prompt 05a §6, with T1 as amended by §9. It was run on the Mac Studio (mps026298) by
the re-measurement agent, Ray-free and with no lock. It ran beside the R11/R12 agent, which held
the lock for its own builds.
- **Trees:**
  - $c$ is the code commit's parent `c4eaefd`, whose code is 05's $c$. It is in the git worktree
    `/Users/ds283/Documents/Code/SecondaryGWKit-lpf05a-c4eaefd` (detached).
  - $t$ is the code commit `1fc349c`, in the main checkout at Part B's dispatch SHA `149c393`. Its
    code is `1fc349c`'s up to `bd38e50`'s comments.
- **Run:** `levin-phase-floor-05a-wider-20261026T123047`. It ended `done`, 109/109 units, exit 0,
  12:30:47Z–12:32:25Z.

**Commit:** none here. The orchestrator commits the records in 05a's records commit.

**Result:**
- **Gate T0 holds,** on all three clauses.
- **T1, as amended by §9 and scored on the reading fixed before the run, fails.** It fails in 36
  of 70 scored windows, all on the $t$ clause. Those 36 are exactly the windows the literal halving
  rule accepts while their samples step 4.9–352 rad each: aliased windows, where $\sigma_\theta$
  measures `numpy.unwrap` (issues X1).
  - In the 34 other windows, $\sigma_t \le 1.3\times10^{-5}\,\varepsilon|\theta_T|$.
  - $\sigma_c$ is 0.196–0.374 $\varepsilon|\theta_T|$ in all 70.
  - **Read by the true step** (unscored; both readings kept), T1 holds in all 70.
  - The question of which reading is the rule's went to the user after the launch. It is not yet
    answered.
- **Added after the user's decision (relayed about 13:23Z; deviation 16):** T1 is scored by the
  true step. **T1, scored by the true step: holds**, in 70 of 70 scored windows.
  - $\sigma_c$ is 0.198–0.374 $\varepsilon|\theta_T|$.
  - $\sigma_t$ is $4.4\times10^{-10}$–$1.31\times10^{-5}\,\varepsilon|\theta_T|$, at most 0.013 of
    its bound.
  - No window is unresolved, and no re-probe was needed: every window, the 21 below
    $10^{-11}$ included, is resolved within the kept levels.
  - The literal reading, which fails in 36 windows as above, now sits beside it, unscored.
- **T2 holds** (39 of 39). The largest depth under $t$ at `1e-7` is 11.
- **T3 holds** (39 of 39). The largest $|t-c|$/`total_abserr`$_c$ is $4.7\times10^{-6}$.
- **T4 fails** (29 of 31): `mid-G-r-3` at 1.29 and `mid-G-3` at 1.70. These are the same rows and
  ratios as 05's L3 under $c$. T4 holds on all 11 rows capped under B1, largest 0.166.
- **No unit went over budget.** The largest $t$ wall/budget is 0.861 (`t-1e-8-R3`).
- **The store is byte-identical:** fingerprint `0e32f24a…`, `problems: none`.
- **No region count, depth or convergence flag changes** in any of the 219 group calls between
  $c$ and $t$.

### Predictions and verdicts

Prompt 05a §6's predictions are copied unchanged below. §6's T1 is marked "replaced before the run
by §9", with §9's T1 after it. `state/predictions.md` holds the same text, quoted word for word
with §9's T1 in place of §6's, plus its "Readings fixed before the run". It was written at
12:30:03Z: after gate T0 held (12:28:33Z) and before the launch (12:30:47Z).

**Predictions, fixed before the run** (README §5 rule 3). Copy them into the log unchanged.
- **T1 — the floor is there, and goes.** *(Replaced before the run by §9, 2026-10-26.)*
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

**T1, amended** (prompt §9, replacing §6's T1 before the run):
- **The rule:** in every resolved probe window with $\varepsilon|\theta_T| \ge 10^{-11}$ rad,
  sampled in $z$: $\sigma_\theta \ge 0.1\,\varepsilon|\theta_T|$ under $c$, and
  $\sigma_\theta \le \max\big(10^{-3}\varepsilon|\theta_T|,\ \varepsilon(k\max|p_z| + 2\pi),\ 10^{-13}\ {\rm rad}\big)$
  under $t$.
- **Holds** if true in every such window. **Fails** if false in any. **Undecided** if none is
  false but some window with $\varepsilon|\theta_T| \ge 10^{-11}$ rad is unresolved.
- **Also report, unscored,** in how many windows the term $\varepsilon(k\max|p_z| + 2\pi)$ is the
  largest of the three, and $\sigma_\theta/(\varepsilon|\theta_T|)$ under $t$ in those windows.

**Descriptive, not scored:**
- $|{\rm total}_t - {\rm total}_c|/|{\rm total}_c|$ and $/$`total_abserr`$_c$ on every row;
- regions and maximum depth per group, $c$ against $t$, with the `Gqr` groups and the low-$z$ end
  of each sub-interval shown apart;
- `total_converged` under $c$ and $t$;
- **the kinds of non-convergence** under $t$ at both tolerances, by the orchestrator's
  `prompts/levin-phase-floor/measure/convergence_kinds.py`, copied unchanged, recorded by SHA-256,
  and run as `convergence_kinds.py <this run> --tree t --capped-run <05's run>`. It reads the
  `checkpoint.jsonl` and `unit_*.pkl` layout of 05's run, which the harness keeps. The same
  measurement under 05's $c$ is `prompts/levin-phase-floor/measure/out/convergence_kinds.txt`.

**Readings fixed before the run** (`state/predictions.md`):
- The "largest step of the unwrapped samples" is `max|diff(numpy.unwrap(θ))|` of the 65 $z$
  samples, as test 3's guard computes it.
- The window is $[z_{\min}, z_{\min}(1+w)]$.
- $\varepsilon = 2^{-52}$.
- T2 and T3 are taken over the 39 rows and T4 over the 31 stored at depth 20. The 11 rows capped
  under B1 are reported apart.
- An over-budget $t$@`1e-7` unit makes T3 undecided on its row.

**Verdicts** (`out/wider_05a.txt`; T1 re-scored independently from `state/probe_windows.json`,
`out/cmd/24_rescore_t1.txt`):

| | Number | Verdict |
|---|---|---|
| **T0 (a)** | `git diff --stat 120fdfe c4eaefd -- . ':!docs' ':!prompts'`: `CLAUDE.md` alone (5+, 3−) | holds |
| **T0 (b)** | $c$@1e-7 against 05's `c-1e-7-*` on `low-G+r-1` and `ctrl-low-1`, by `==` and `float.hex`: `total`, `total_abserr` and all 7 calls' value and abserr, 16 doubles per row, identical. So are the whole result summary (68 floats), the call summaries, `qs_check`, the dispatcher's counts and flags, and every region array of 05's pickles | holds |
| **T0 (c)** | two in-process $t$ runs on the same two rows: result, calls, regions, probe summaries and every probe sample (31,200 and 31,070 values) bit-identical | holds |
| **T1 (§9), as written** | 91 windows on 37 rows, 70 with $\varepsilon\lvert\theta_T\rvert \ge 10^{-11}$, 0 unresolved. True in 34, false in 36: the $c$ clause holds in all 70 ($\sigma_c$ 0.196–0.374 $\varepsilon\lvert\theta\rvert$); the $t$ clause fails in 36, $\sigma_t$ 0.016–0.23 $\varepsilon\lvert\theta\rvert$, up to 232× the bound. The 36 are exactly the scored windows whose true step per sample at the level used is ≥ π/2 (4.9–352 rad); elsewhere $\sigma_t \le 1.3\times10^{-5}\,\varepsilon\lvert\theta\rvert$ | **fails** |
| (T1 by the true step, unscored) | the same rule at the first level where the true step is also below π/2 in both trees: true in 70 of 70; $\sigma_t$ $4.4\times10^{-10}$–$1.31\times10^{-5}\,\varepsilon\lvert\theta\rvert$ (≤ 0.013 of the bound); $\sigma_c$ 0.198–0.374 | (holds) |
| **T1 (§9), scored by the true step** *(added after the user's decision, deviation 16; this is now T1's scored verdict)* | windows resolved by the true step alone, $\lvert d\theta_T/dz\rvert(b-a)/64 < \pi/2$ under both $c$ and $t$, halving further where needed; every other clause of §9 unchanged. 91 windows, 70 scored, 0 unresolved, true in 70. $\sigma_c$ 0.198–0.374 $\varepsilon\lvert\theta_T\rvert$; $\sigma_t$ $4.4\times10^{-10}$–$1.31\times10^{-5}\,\varepsilon\lvert\theta_T\rvert$, largest $\sigma_t$/bound 0.0131. The largest unwrapped step at the levels used is 1.44 rad, so these windows also pass the literal rule. Levels used: $n$ = 0 (23), 2 (12), 3 (7), 5 (8), 6 (7), 7 (2), 8 (11). These are the same levels, and therefore the same numbers, as the unscored row above. No re-probe was needed (`out/cmd/26_T1_true_step_scored.txt`) | **holds** |
| (T1 by the literal step, now unscored) | the "as written" row above, kept beside the scored verdict | (fails) |
| **T2** | 39 of 39; no group under $t$@1e-7 has a region at depth ≥ 18; largest depth 11 (and 11 on the 11 rows capped under B1) | holds |
| **T3** | 39 of 39; largest $\lvert t-c\rvert$/`total_abserr`$_c$ $4.66\times10^{-6}$ (`low-G-1`); largest relative $3.6\times10^{-12}$ (R4); bit-identical totals on 13 rows | holds |
| **T4** | 29 of 31, none excluded; false on `mid-G-r-3` (1.29) and `mid-G-3` (1.70), neither capped under B1; true on all 11 capped (largest 0.166). Every ratio equals 05's L3 under $c$ to the digits printed | **fails** |

**T1's unscored items:**
- $\varepsilon(k\max|p_z| + 2\pi)$ is the largest of the three terms in **0 of 70** windows, so
  there are no $\sigma_t$ ratios to report for it. `1e-3 ε|θ|` is largest in 69 windows and
  `1e-13` in 1. $k\max|p_z| \le 9.8\times10^{-7}|\theta_T|$, with one table node per window
  (issues X5).
- **Sampled in $u$ under $t$, at the true-step level:**
  - 14 windows exceed $0.1\,\varepsilon|\theta_T|$, at 0.10–0.22, where the abscissa term is
    0.69–0.98 $\varepsilon|\theta_T|$;
  - over all 70 windows, $\sigma_u$ is 0.14–0.29 of the abscissa term;
  - under $c$ it is 0.21–0.40 $\varepsilon|\theta_T|$;
  - at $z < 1$ (6 windows) it is 0.0025–0.0096 under $t$.
  - At the literal level, 27 windows exceed 0.1, the aliased ones among them.
  - The issue the rule asks for is proposed under observation 2.

**Descriptive:**
- **$|t-c|$:** relative at most $3.6\times10^{-12}$, and at most $4.7\times10^{-6}$ of
  `total_abserr`$_c$; 13 rows are bit-identical (T3's table).
- **Groups:** all 219 group calls are the same sequence (regime, label, `x_span` bit for bit), with
  the same region count, maximum depth and convergence flag under $c$ and $t$.
  - That includes the 140 `Gqr` groups: 264 regions, max depth 11, in both.
  - The lowest-10 % (in $\log(1+z)$) region counts and depths of every group are the same.
  - Per group in `out/wider_05a.txt`.
- **Cost:** low band 288 → 288 regions (8.0 → 4.6 s, load-dominated); mid band 321 → 321;
  R1–R7 250 → 250; the 11 capped rows 426 → 426 at 1e-7 and 690 at 1e-8.
- **`total_converged`:** unchanged between $c$ and $t$ on every row.
  - Depth-20 rows: 21 of 31 unconverged at 1e-7, 19 at 1e-8.
  - Controls: all converged.
  - 3 rows have an infinite `total_abserr` in both trees: `low-G+r-2`, `low-G-r-1`, `low-Gqr-3`.
- **Kinds of non-convergence** under $t$ (`out/convergence_kinds_t.txt`, script SHA-256
  `49c83a72…`):
  - 1e-7: resolution 18, bound only 32, phase-limited 17 (capped rows: 12, 21, 10);
  - 1e-8: 21, 24, 27 (capped rows: 17, 13, 18).
  - These are identical to 05's $c$ tallies.

### What shipped

Nothing is committed. Everything is under `var/levin-phase-floor-05a/wider/`. The checkpoints and
the pickles, which hold every probe sample of every level, are in the run directory (21 MB).

**Scripts** (`scripts/`). 05's harness, copied; the diff against 05's copy is
`out/scripts_diff_against_05.txt` (at the launch) and `…_final.txt` (with the analysis).
- **Changed:**
  - `prov.py`: the trees and paths;
  - `worker.py`: trees, provenance, the `build_partition` capture, the probe call;
  - `run_wider.py` and `launch.py`: trees, unit sets, prompt `05a`;
  - `rec.sh`: also writes `out/commands.log`;
  - `analyse.py`: rewritten for T1–T4.
- **New:** `probe.py` (the phase probe), `t0.py` (gate T0, from 05's `l0.py`) and `rescore_t1.py`
  (re-scores T1 from `state/probe_windows.json` alone).
- **Added after the user's ruling:** `t1_true_step_only.py`, which scores T1 by the true step
  alone from `state/probe_windows.json` (`out/cmd/26_T1_true_step_scored.txt`). Beside it, in
  `state/`, is `predictions_note_T1_reading.md`. The scripts diff including it is
  `out/scripts_diff_against_05_after_ruling.txt`.
- **Unchanged:** `harness.py`, `rebuild.py`, `dispatch.py` and `hash_store.py`. `convergence_kinds.py`
  is byte-identical to `prompts/levin-phase-floor/measure/`. `draw_sample.py` and `l0.py` were
  copied and are unused.

**Outputs** (`out/`):
- `commands.log` and `cmd/00…25`: every command with its output (01 is unused);
- `T0.txt`;
- `wider_05a.txt`: every row, every group, every probe window under both readings, and the scores;
- `convergence_kinds_t.txt`;
- `fingerprint_05a_wider.txt`;
- `status_final.json`;
- the two scripts diffs.

**State** (`state/`):
- `sample.json` (05's, unchanged) and `sample.md`;
- `predictions.md`, the three `prompt_*.raw` and `predictions_attempt1_wrong_window_lines.md`;
- `T0/` and `smoke/`;
- `store_files_{before,prelaunch,after,after_fingerprint}.txt`;
- `launch.txt` and `launch_check.txt`;
- `analysis.json`;
- `probe_windows.json` (4 MB): every window's ladders under both trees, and the samples as
  `float.hex` at both readings' levels.

### Deviations from the prompt

1. **The $c$ tree runs from a worktree.** **STRUCTURALLY REQUIRED.** The prompt runs two trees,
   and the orchestrator made the worktree.
   - Code is loaded through `PYTHONPATH` with `-B` and `PYTHONDONTWRITEBYTECODE=1`.
   - Nothing was written there: `git status --short --ignored` is empty at the end, although R11
     shared the worktree.
   - The manifest's `git_head` is `149c393` (the $t$ tree). Every checkpoint payload records its
     own tree's `HEAD`, its status and its modules' files.
2. **05's trees `p`/`c` became `c`/`t`, with 05a's paths and unit sets.** **STRUCTURALLY REQUIRED**,
   as §6 allows. The changes are in `prov.py`, `run_wider.py` (budget from $c$), `launch.py`
   (prompt `05a`), and the docstrings of `worker.py` and `rec.sh`. Each change is marked "prompt
   05a"; 05's docstrings are kept below a note.
3. **The scripts live in `var/levin-phase-floor-05a/wider/scripts/`,** not `…/05a/scripts/`.
   **IMPLEMENTATION CHOICE** (the orchestrator's), mirroring 05's `wider/` layout.
4. **How the probe is built.** **IMPLEMENTATION CHOICE**, where §6 and §9 leave the method open.
   - It is a new `probe.py`.
   - `worker.py` wraps `ComputeTargets.QuadSourceIntegral.build_partition` at its module
     attribute, to keep the unit's own $T$ objects.
   - The probe runs in each tree's 1e-7 unit after `wall_s` is taken, outside the budget, with the
     dispatcher uninstalled. T0 (b) shows the unit's results are unchanged by it.
   - Each tree computes **every** halving level until its 65 $z$ samples stop being distinct. The
     joint level is chosen afterwards, in the analysis, as the first level at which both trees
     pass.
   - The window is $[z_{\min}, z_{\min}(1+w)]$. The step that drives the halving is the
     $z$-sampled one.
   - $\max|p_z|$ is recorded under both trees; the $t$ value is the one scored.
   - $\sigma_u$ is None where the 65 $u$ samples are not distinct, which happened at no level used.
5. **The halving rule is read literally,** with the true-step reading reported apart and unscored.
   **IMPLEMENTATION CHOICE**, fixed in `state/predictions.md` before the run after T0's $c$ windows
   showed the two differ.
   - The point was raised in the launch report and relayed to the user; it is unanswered.
   - All levels are kept, so either reading can be scored (issues X1).
6. **Gate T0 uses a new script, `t0.py`, derived from 05's `l0.py`.** **STRUCTURALLY REQUIRED**:
   T0's clauses are not L0's.
   - T0 (b) compares more than the clause names (the whole result and the region arrays).
   - T0 (c) includes the probe.
   - Both rows are 05's L0 rows, `low-G+r-1` and `ctrl-low-1`.
7. **The worker records more than 05's did:** `TkSourceFunctions`' file and its keyword's default,
   and each factor's `phase.exact_remainder`. **IMPLEMENTATION CHOICE.** The dispatch asked for
   both.
8. **`rec.sh` also appends every record to `out/commands.log`.** **IMPLEMENTATION CHOICE.**
9. **`state/predictions.md` was written twice before the launch.** **UNINTENDED DRIFT.**
   - The first write (12:29:30Z) quoted §9's window from line 391 instead of 388. It is kept as
     `state/predictions_attempt1_wrong_window_lines.md`; a working copy passed through the
     session scratchpad on the way.
   - The second write (12:30:03Z) fixed the range and one internal reference.
   - Both writes were before the launch, and the quoted predictions are identical in both.
10. **The smoke test** (`low-G+r-1`, `ctrl-NULL-1`, all three sets) ran after the predictions were
    written and before the launch, as 05's did. **IMPLEMENTATION CHOICE.** It needed no script
    change, so T0 was not re-run.
11. **`analyse.py` was rewritten, after the run.** **STRUCTURALLY REQUIRED**: 05's L1–L3 are not
    05a's rules. The "low-$z$ end of each sub-interval" (descriptive) is read as the regions
    starting in the lowest 10 % of the group's sub-interval in $\log(1+z)$, which is an
    **IMPLEMENTATION CHOICE**.
12. **`rescore_t1.py`, an independent re-score of T1 from the saved windows alone.**
    **IMPLEMENTATION CHOICE.** It agrees with `analyse.py` on every window and both readings.
    `analyse.py` also asserts that $\sigma$ recomputed from the saved samples equals the probe's.
13. **`convergence_kinds.py` was run from the main checkout with `-I`, from this directory's
    unchanged copy.** **IMPLEMENTATION CHOICE.**
    - Its header says "capped: tree p … in <05's run>", which is 05's $p$ (B1's code), as §6
      intends.
    - `out/convergence_kinds_t.txt` drops the per-pickle hash lines; the full output is in
      `out/cmd/22`.
14. **The work ran beside R11/R12 and other load,** with a load average of 10–35 during the setup.
    **IMPLEMENTATION CHOICE**, which the prompt allows ("It may run beside §5's builds"). Wall times
    and budgets carry that load; the closest any unit came to its budget is 0.861.
15. **05's `draw_sample.py` and `l0.py` were copied unchanged and are unused.** **IMPLEMENTATION
    CHOICE**, so that the copy is 05's harness whole.

16. **T1's scored verdict is taken by the true step, by the user's decision after the run.**
    **STRUCTURALLY REQUIRED** (the user's ruling on the rule's reading; added after the run).
    - **The ruling.** Relayed to this session at about 13:23Z from the laptop session, in answer
      to "Which reading of §9's resolution rule should be T1's scored verdict?", the user chose
      "True step (Recommended)". That option read: *"T1 is scored on windows whose true step
      |dθ/dz|(b−a)/64 is below π/2, narrowing further where it isn't. This is what §9 meant by
      'resolved'. The literal unwrapped-step reading is reported beside it, unscored. Recorded as
      an amendment of §9's wording, made before any T1 number exists."*
    - **What the files show of the timing.** `predictions.md` fixed both readings at 12:30:03Z,
      the literal one as scored. Both were first scored at 12:35:49Z (`out/wider_05a.txt`). The
      ruling reached this session at about 13:23Z. When the user made it is for the laptop
      session to record. This is set out in `state/predictions_note_T1_reading.md`.
    - **How it was applied.** `predictions.md` and its first copy are left as written.
      `scripts/t1_true_step_only.py` is new; it reads `state/probe_windows.json` only.
      - It takes the first level at which the true step alone is below $\pi/2$ in both trees.
      - That level equals `analyse.py`'s true-step level in all 91 windows, so the saved samples
        are the windows' own.
      - It re-computes $\sigma$ from those samples and scores §9's rule.
      - No window needed a re-probe, and none was run.
      - The literal reading stays reported, unscored.

Deviation 9 is the only `UNINTENDED DRIFT`.

### Verification performed

- **The trees.**
  - The main checkout was at `149c393` with an empty status, and the worktree at `c4eaefd` with an
    empty status. This holds at the start, before the launch, on every checkpoint record and at the
    end (`out/cmd/00`, `15`, `25`).
  - The worktree's `git status --short --ignored` is empty at the start and at the end.
- **Module provenance.** Each worker checked and recorded the files of `ComputeTargets`,
  `primitive_phase`, `QuadSource`, `QuadSourceIntegral`, `GkSourcePolicyData`,
  `TkSourceFunctions`, `StoreFixtures`, `main.py` and the original `theta_mod_2pi`. All of them come
  from its own tree. `TkSourceFunctions`' keyword is absent under $c$ and defaults to True under
  $t$.
- **Which phase each tree used.**
  - $c$: $G$ 98,194 `theta_mod_2pi` calls with `exact_remainder=True`, and $T$ 59,906 with `False`.
  - $t$: $G$ 268,136 with True, and $T$ 167,562 with True.
  - Every unit's $T$ factors report `exact_remainder` False under $c$ and True under $t$, and so do
    all 91 probe windows.
  - The $t$ counts equal 05's $c$ counts.
- **The sample.** Its SHA-256 is `73aa7876…`, and it is `cmp`-identical to 05's. It equals the
  `sample_sha256` in 05's run header, checked before any row ran.
- **The store.**
  - All six files' SHA-256, size and mtime are identical before, before the launch, after the run
    and after the fingerprint. No journal, `-wal` or `-shm` file appeared.
  - `store fingerprint`, without `--write`, gives `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`,
    `problems: none`, and "matches the fingerprint taken … at b90752d".
- **The run.**
  - `done`, 109/109, in the planned order, exit 0, no problems, none over budget, no errors and no
    probe errors.
  - Every record carries `run_wider.py`'s SHA-256 `af656ad4…`.
  - Ray was never initialised, and no lock was taken.
- **`QuadSource`.** It is recomputed in every unit, equals the stored row at its nodes, and adds
  B1's 3 nodes.
- **The probe windows.** $c$ and $t$ give the same 91 windows, starting at the same double. Their
  `raw_theta` and `theta_deriv` at that point are bit-identical, as §0.4 requires.
- **05's records were only read:** 05's harness is `diff`-identical in its `var/` and `docs/`
  copies at the end.

### Observations not acted on

All are in `issues.md`:
1. **X1:** §9's halving rule, as written, admits aliased windows. T1 fails in exactly those 36, on
   the $t$ clause, and holds in all 70 by the true step. Proposed tag:
   `[05a-t1-halving-rule-admits-aliased-windows]`. The user's ruling is pending.
   *Added after the ruling:* the user chose the true step (deviation 16). T1 is scored by it and
   holds, 70 of 70. The literal reading is reported beside it, unscored. The laptop amends §9's
   wording in the records commit. What remains of X1 is the record that the wording as first
   written admitted aliased windows; whether that is still opened as an issue is the laptop's to
   decide.
2. **X2:** the abscissa term remains in $T$'s $u$-sampled phase.
   - $\sigma_u$ under $t$ is 0.10–0.22 $\varepsilon|\theta_T|$ in 14 windows at $z \gg 1$, against
     abscissa terms of 0.69–0.98; it is 0.14–0.29 of the term everywhere.
   - Proposed issue, as T1's $u$-clause asks: `[05a-t-phase-abscissa-rounding-remains]`.
3. **X3:** T4 fails on the same two rows as 05's L3, with the same ratios (05's W5, untouched by
   $T$).
4. **X4:** $T$'s two errors change no region, depth or convergence flag on the 39 rows, and move
   the totals by at most $4.7\times10^{-6}$ of the declared error.
5. **X5:** §9's $p_z$ term never binds. The windows sit at or near table nodes, with
   $k\max|p_z| \le 10^{-6}|\theta_T|$.
6. **X6:** 15 scipy `IntegrationWarning`s, as in 05.
7. **X7:** the kinds of non-convergence under $t$ are identical to 05's under $c$.

### State handed on

- **Nothing is running.** The run is `done`, no lock was taken, and nothing was written in either
  tree.
- **The store is unchanged**, fingerprint `0e32f24a…`.
- **The worktree** `SecondaryGWKit-lpf05a-c4eaefd` is untouched by this work. It is the
  orchestrator's to remove once R11's records are taken.
- **For the orchestrator:**
  - copy `out/`, `scripts/`, `state/`, `log.md`, `verification.md` and `issues.md` to
    `docs/levin-phase-floor/05a/wider/`. `state/T0/*/*.pkl` and `state/smoke/*.pkl` may be left
    out; `state/probe_windows.json` (4 MB) is what T1's re-score needs;
  - append this section to `logs/05a-reanchor-t-phase.md`;
  - put `verification.md` in `VERIFICATION.md` §5a. Its links assume the `05a/wider/` location.
- **For the user:** X1, the reading of §9's halving rule. Under the literal reading, T1 fails.
  Under the true-step reading, it holds in every window. Both are scored from the same saved
  samples.
  - *Added after the ruling:* answered: the true step (deviation 16). T1's scored verdict holds.
  - The laptop amends §9's wording in the records commit, and records when the user decided.
  - `state/predictions_note_T1_reading.md` records what the files here show.
