# levin-phase-floor 06 — the producers' stored phase: store the small part, reduce the large one exactly

> **§1–§4 only** (the code, the fixtures, the tests and these records), written by the executing
> agent on the laptop. §5 (rehearsals R13 and R14) and §6 (the re-measurement) run later on the
> Mac Studio. The orchestrator appends their record to this log.

**Subject:** prompt 06 §1–§4, on the laptop, in the main checkout
`/Users/ds283/Documents/Code/SecondaryGWKit`, branch `handover-remedial`.
- **Dispatch SHA:** first `2006de7`; on the resume, `9f66622`, which amends the prompt only
  (§2, §0.2, §1.4 and §9 point 6). Nothing landed on the branch between the resume's dispatch and
  the commit (`git log --oneline 9f66622..HEAD` was empty before staging).
- **The gate.** `git status --short` was empty at both dispatches. `runregistrykit list` showed
  nothing `running` or stale: 60 runs, 58 finished, 2 `unknown` (pre-registry). The seven roots'
  baselines are the orchestrator's, taken at `2006de7`, whose code is `9f66622`'s (deviation 1).
- No Ray was started, no run was launched, no production store was opened, and nothing was written
  under `var/`. The tests build stores only in `tempfile` directories.

**The first dispatch stopped** before any code was written, at §8 ("an existing assertion has to
change outside §2's exception"). §1.1 makes the phase function's payload name new things, and
`test_gk_wkb_phase.py:920` (`TestPayloadContract.test_keys_and_stage_data`) asserts the payload's
exact key set. The stop report also gave eight further findings. The user ruled on all of them, and
the prompt was amended at `9f66622` (§9 point 6). The tree was clean when the work resumed.

**Commit:** this one commit (§4), not pushed. Its hash is not written here; the board's 06 row
carries none until the records commit.

**Result:**
- **§1.1.** `WKB_phase_function` forms each sample's phase exactly. The leading table's four node
  limbs and two end partials are used, with `k` multiplied in exactly and `rho.delta` added
  exactly. The phase is reduced exactly against 2pi. The payload gains `theta_exact_sample` (the
  exact phase, a `Fraction`) and `rho_delta_sample`.
- **§1.2.** Both producers' `store()` add `deltaTheta` to the exact phase and reduce the sum
  exactly. They store `theta_residual = deltaTheta - rho.delta`, formed from those two small terms,
  and compute `G_WKB` and `T_WKB` from the exact remainder. `apply_phase_offset` has no production
  caller left.
- **§1.3.** `theta_residual` is a nullable `Float(64)` column on `GkWKBValue`, `TkWKBValue` and
  `GkSourceValue`, with its write and its read on every path.
- **§1.4.** `assemble_GkSource_values` carries `r` through.
  - `_build_phase` forms `phi_G = r + k tau.delta(z_s, z_init)`, unwraps it by whole cycles and
    checks the unwrap against the stored rectifier shifts. A disagreement warns in one line.
  - `TkSourceFunctions` takes `phi_T = r`, and `_check_phase_sign` puts the phase back together
    to check the sign.
  - `build_phi_samples` is kept, with its docstring corrected.
  - The integer helper is new and sits beside `PrimitivePhase`'s reducer: `exact_leading_phase`
    and `reduce_exactly_mod_2pi`. `PrimitivePhase` is untouched.
- **§1.5.** `tools/store_value_diff.py` needed extending: `open_read_only` refuses a store whose
  files lack a declared column. It now reads such a store without the column, reports the column
  as present in one store only, and compares every other column as before.
- **§2.** Every stand-in that feeds a consumer carries the residual (list below). Three existing
  assertions changed, all named by §2: the payload key set and the two source greps. A fourth,
  `test_residual_part_alone`, changed under §2's class exception, with its bound unchanged.
- **§3.** `ComputeTargets/tests/test_stored_phase_residual.py` has 13 methods, all passing. The
  numbers:
  - test 1: every stored remainder is within 0.50 ulp(2pi) of the exact one, in 16 cases;
  - test 3: phi's window RMS is at most 1.8e-10 ulp(theta) for `T` and 3.3e-17 for `G`. The old
    construction gave 0.37–0.39 ulp(theta) at the median for `T` and 3.9 for `G`.
  - test 4: two wraps, and the unwrap equals the rectifier's shifts at all 51 samples.
- **The breakages.** All five were caught by the test the prompt names, and each was reverted.
- **The seven roots.** Five equal the baseline. `ComputeTargets` rose by 13 (the new module) and
  `tools` by 3 (the new tool test).
- **No stop condition of §8 arose on the resume.**

## What shipped

**`ComputeTargets/primitive_phase.py`.**
- **New at the end of the module, beside `_remainder_against_two_pi`:**
  - `exact_leading_phase(k, leading, z_a, z_b, *, sign, small_terms)` gives
    `sign k leading.delta(z_a, z_b) + sum(small_terms)` exactly, as a `Fraction`. It reads
    `delta_parts` and `node_limbs`, which spends the integrand evaluations of one `delta` call.
    Every double is taken as `n / 2^s` and summed in integers.
  - `reduce_exactly_mod_2pi(theta)` gives `(div, mod)`, with `div = ceil(theta/2pi)` exact and
    `mod` the exact remainder in `(-2pi, 0]`, rounded once. It works in integers, against
    `floor(2pi 2^bits)` truncated from one cached `mpmath` constant. The working scale puts the
    truncation below 2^-120 rad. A remainder that rounds to `-TWO_PI` is stored as 0.0 with
    `div - 1` (deviation 3).
- **`build_phi_samples`** is kept. Its docstring now says that no production consumer calls it,
  why, and what it is kept for. Nothing else in the file changed: `PrimitivePhase`, its reducer
  and the module docstring are untouched (observation 1).

**`Quadrature/integrators/WKB_phase_function.py`.**
- The per-sample loop forms `theta_exact = exact_leading_phase(k, leading, z_init, z, sign=-1,
  small_terms=(-rho_delta,))` and reduces it with `reduce_exactly_mod_2pi`.
- The payload gains `theta_exact_sample` and `rho_delta_sample`; the zero-length branch gives
  `[Fraction(0)]` and `[0.0]`.
- The module docstring has a paragraph on the exact phase and the residual, and the function's
  docstring says how each payload key is formed. `rho_delta` and the metadata are unchanged.

**`ComputeTargets/GkWKBIntegration.py`, `ComputeTargets/TkWKBIntegration.py`.**
- **STEP 3 of `store()`:** `reduce_exactly_mod_2pi(theta_exact + Fraction(deltaTheta))` per
  sample, and `theta_residual = deltaTheta - rho_delta`. The `apply_phase_offset` import is
  replaced by `reduce_exactly_mod_2pi`.
- **The value objects** take `theta_residual=None` as a keyword, with a property.
- `G_WKB` and `T_WKB` use the new remainder through the code already there.

**`ComputeTargets/GkSource.py`.** `_WKBData` gains `theta_residual`, `GkSourceValue` takes it, and
`assemble_GkSource_values` passes `WKB.theta_residual` through. The rectifier is unchanged.

**`ComputeTargets/GkSourcePolicyData.py`.** `_build_phase` builds `phi` with a new
`_phi_samples`:
- it forms `r + k tau.delta(z_s, z_init)`. The second term is 0.0 when `z_init` is null (pure WKB)
  and is a double of the span's size otherwise;
- it unwraps by `floor((value - previous)/TWO_PI + 0.5)` cycles in ascending `z_s`;
- it compares those cycles with `raw_theta_div_2pi - theta_div_2pi` relative to the first sample.
  Any disagreement prints one `!! WARNING (GkSourcePolicyData, <label>)` line naming `k`, `z_R`,
  the number of samples and up to eight of them, and `phi` keeps its own unwrap;
- a value with no residual raises `RuntimeError`, with no fallback.

The docstring of `_build_phase` says all this. The `build_phi_samples` import is gone.

**`ComputeTargets/TkSourceFunctions.py`.**
- `_build_WKB` takes `phi = [v.theta_residual]`, and raises if any is missing.
- `_check_phase_sign(z_points, data, phi_points)` puts the phase back together to check the sign.
  - At every sample it compares the stored `theta_mod_2pi` with
    `WKB_mod_2pi(TK_PHASE_SIGN k cs_tau.delta(z, z_init) + phi)[1]` by cyclic distance.
  - The tolerance is `16 eps (k|cs_tau.delta| + |phi|) + 1e-12` rad, and it raises otherwise.
  - The sum is formed only to compare remainders; phi is never formed from it.
- The module docstring ("Phase", and the input protocol) says so.

**`LiouvilleGreen/WKBtools.py`**, docstrings and comments only: `WKB_mod_2pi`'s header,
`wrap_theta`, `apply_phase_offset` (no production caller since 06) and `shift_theta_sample`. Its
AST with docstrings stripped equals `9f66622`'s (Verification).

**`ObjectFactories/GkWKBIntegration.py`, `ObjectFactories/TkWKBIntegration.py`,
`ObjectFactories/GkSource.py`.**
- Each value table gains `sqla.Column("theta_residual", sqla.Float(64), nullable=True)`.
- Each path that writes a value row writes the column: the parent's `store()` and the value
  factory's `build()` insert.
- Each path that reads one selects it and passes it on: the parent's `build()`, and the value
  factory's `_build_impl_serial`, `_build_impl_model` and `read_batch`.
- No `inventory_spec()` changed, and `test_fingerprint_format_guard` passes unchanged.

**`tools/store_value_diff.py` (§1.5).**
- `_absent_columns(primary, factories)` probes `open_read_only`. On a `StoreSchemaMismatch` whose
  only differences are declared columns the files lack, it notes them and reopens without them.
  Any other difference is re-raised, which is exit 2 as before.
- `_WithoutColumns` wraps a factory and drops those columns from `register()`.
- `_TableRead` records the columns it read. `_compare_tables` compares the columns common to both
  stores and reports `columns_only_in_a` and `columns_only_in_b`; either makes the class not
  value-identical. The text report names each column. The module docstring has a paragraph on it.

**`tools/tests/test_store_value_diff_absent_column.py`, new: 3 methods.**
- **(t6a)** A store without `GkWKBValue.theta_residual` against a full one, both ways round. Exit
  1; the column is in one store only; every value row is matched and identical on its other
  columns; every other class is identical; both stores' files are unchanged.
- **(t6b)** Beside the absent column, a one-ulp `theta_mod_2pi` change is still reported.
- **(t6c)** An undeclared column is still a refusal, exit 2.

**`ComputeTargets/tests/test_stored_phase_residual.py` (§3), new: 13 methods.**
- **The background** is `test_gk_wkb_phase`'s LambdaCDM stand-in, with its three
  `TablePrimitive` tables on the production source grid.
- **The producers run as in production:** `WKB_phase_function._function`, then the real
  `store()` with `ray` stood in by `test_tk_wkb_phase._FakeRay`.
- **Each test-1 object** is numerically seeded, with initial data that give `deltaTheta = 2.0`
  rad and `B = 1`. Its `z_init` is the 5th residual-table node below the top of the WKB region
  (taken as `k L(z) >= 100`), or 37% of the way in `log(1+z)` to the next node down. It is
  sampled at every grid node below `z_init`: 938–1309 samples.
- **The mpmath reference** (60 digits) is formed from the same limbs, partials, `rho.delta` and
  `deltaTheta`, independently of the helpers.

| test (§3) | method(s) |
|---|---|
| 1 | `TestStoredRemainderIsExact.test_remainder_and_cycle_count_against_2pi` (16 subtests: 2 sectors × 2 anchors × 4 k), `.test_the_helper_normalises_a_remainder_that_rounds_to_minus_two_pi`, `.test_the_helper_against_mpmath` |
| 2 | `TestResidualIsTheSmallPart.test_residual_and_its_reconstruction` (16 subtests) |
| 3 | `TestPhiIsNoiseFree.test_phi_T_and_phi_G` |
| 4 | `TestTheTwoRepairsAgree.test_unwrap_matches_the_rectifier_and_phi_is_continuous`, `.test_a_disagreement_warns_and_the_build_goes_on` |
| 5 | `TestTheConsumers.test_raw_theta_at_the_samples`, `.test_a_value_without_the_residual_is_refused` |
| 6 | (a) `TestNothingElseMoves.test_a_primitive_phase_is_its_own_formulas_and_reduces_as_the_producers_do`; (b) `.test_rho_and_the_background_tables`; (b), (c) `.test_the_numeric_producers_and_QuadSource_import_nothing_this_prompt_changed` |
| 7 | `TestEndToEnd.test_evaluate_sum_against_the_direct_product` |

**The stand-in sites (§2)**, line numbers after the change.

| file | line | in | residual |
|---|---|---|---|
| `ComputeTargets/tests/test_tk_source_functions.py` | 157–179 | `FakeWKBValue` gains `theta_residual=None`; new helper `closed_form_residual` | — |
| ″ | 448 | `Fixture.exact_functions` (Bessel) | double, θ − k·cs_tau.δ; \|θ\| ≤ 995 rad, so at most ~ε\|θ\| ≈ 2.2e-13 rad |
| ″ | 493 | `Fixture.LG_functions` (`solve_ivp`) | double; \|θ\| ≤ ~1e3, ≈ 2.2e-13 rad |
| ″ | 649 | `AnalyticRadiationFixture.stored_values` | double; \|θ\| ≤ 1.0e6, ≈ 2.2e-10 rad |
| ″ | 1016 | `TestConsistencyChecks.test_nonzero_cos_coeff_raises` (Bessel) | double; ≈ 2.2e-13 rad |
| ″ | 1362 | `TestPrimitivePhaseConsumer.test_inconsistent_friction_sample_raises` (`solve_ivp`) | double; ≈ 2.2e-13 rad |
| ″ | 1417 | `TestPrimitivePhaseConsumer.test_sign_convention_is_checked` (`solve_ivp`, wrong sign) | double, **in its own convention**, θ' + k·cs_tau.δ (`sign=-1`); ≈ 2.2e-13 rad |
| ″ | 1473 | `TestPrimitivePhaseConsumer.test_missing_background_tables_are_refused_by_name` (`solve_ivp`) | double, from `f.model`'s `cs_tau` (one subcase's model has none); ≈ 2.2e-13 rad |
| `ComputeTargets/tests/test_phase_groups.py` | 72, 651 | `matched_LG_functions` (`solve_ivp`) | double; \|θ\| ≤ ~1e3, ≈ 2.2e-13 rad (deviation 8) |
| `ComputeTargets/tests/test_g_phase_reanchor.py` | 645–661 | `_Value` gains `theta_residual`, and `raw_theta_div_2pi = div` and `z_init = None` (deviation 9) | — |
| ″ | 688 | `_ProductionPhaseGk` (Bessel `G`) | double, θ − GK_PHASE_SIGN·k·τ.δ(z, z_resp); \|θ\| ≤ 2.2e3, ≈ 4.9e-13 rad (deviation 8) |
| `ComputeTargets/tests/test_t_phase_reanchor.py` | 261, 272 | `_LCDMTk.inputs` | small term: `phi_true(u)` itself |
| ″ | 583 | `TestNothingElseMoves.test_G_phase_from_build_phase_is_still_node_anchored` | small term: the 0.01 rad offset |
| `ComputeTargets/tests/test_gk_source_policy.py` | 35, 130–176 | `FakeGkSource` (closed-form linear phase) | double, θ − GK_PHASE_SIGN·k·τ.δ; \|θ\| ≤ 9.2e4, ≈ 2e-11 rad; plus `raw_theta_div_2pi`, `z_init`, and denser WKB samples (deviations 8–10) |

**Not stand-ins of a consumer's input, and not changed:**
- `test_quadsource_integral.py` reaches `TkSourceFunctions` only through
  `test_tk_source_functions`' fixtures. Its `Tk_builder` and `capture` pass the values on whole.
- `test_gk_source_primitive_phase.py` and `test_primitive_phase.py` test `build_phi_samples` and
  a copied rectifier directly.
- `test_residual_table_reuse.py`, `test_gk_wkb_phase.py` and `test_tk_wkb_phase.py` drive the
  producers themselves.
- `ObjectFactories/tests/test_read_path_mechanics.py`, `test_value_reads_parent_key.py` and
  `test_gksource_parent_set.py` insert rows to exercise read mechanics. Their rows leave the
  nullable column NULL, and no consumer reads them (deviation 11).

**Existing assertions changed (§2):**

| file:line | test | old → new | bound |
|---|---|---|---|
| `test_gk_wkb_phase.py:887-904` | `TestPayloadContract.REQUIRED_KEYS` (used at `:924`, unchanged) | 10 keys → the same 10 plus `theta_exact_sample` and `rho_delta_sample` | — (§2's named exception) |
| `test_gk_wkb_phase.py:1154` | `test_gk_store_has_no_sign_fix_or_rebase` | `assertIn("apply_phase_offset", source)` → `assertIn("reduce_exactly_mod_2pi", source)`; the `assertNotIn` lines unchanged | — (§2's named exception) |
| `test_tk_wkb_phase.py:943` | `test_tk_wkb_integration_greps_are_empty` | the same | — (§2's named exception) |
| `test_tk_wkb_phase.py:441-463` | `TestRadiationPhase.test_residual_part_alone` | it pins the stored phase through `div * TWO_PI + mod`, which 06 moves. Now: `div * 2pi + mod` and `k cs_tau.delta` from the limbs and partials, in mpmath. Measured: **6.33e-13 rad** at `9f66622` → **4.58e-16 rad** now (and 1.94e-12 rad, failing, had the double reconstruction been kept) | 1e-12 rad, unchanged; within it (deviation 12) |

Neither producer's source names `apply_phase_offset` any more, not even in a comment. The old
assertion would now fail.

## Predictions and verdicts

§1–§4 score nothing. Prompt 06 §6's predictions are copied below unchanged, as amended by §9
point 4 (S1, amended in place before dispatch), before §6 runs, as §4 asks. They are **not yet
scored**; §6 scores them on the Mac Studio.

**Predictions, fixed before the run** (README §5 rule 3). Copy them into the log unchanged.
- **S1 — $\varphi_G$'s noise goes.**
  - **The rule:** in every **pure-WKB** probe window of $G$ on R1–R7,
    $\sigma_s \le \max(10^{-3}\,\sigma_{t_5},\ 10^{-6}F)$.
  - **Holds** if true in every pure-WKB window. **Undecided** if there is none.
  - Numerically seeded windows are not scored: their $\sigma_s$ is reported beside the window RMS
    of $\Delta\theta$ alone (the probe, above).
  - 04 measured $\sigma$ at 0 at the median, quantised, and 0.46–0.78 $F$ at the maximum, for
    production's $\varphi_G$.
- **S2 — $\varphi_T$'s too.** S1's rule for every probe window of a $T$ factor. $T$ has one
  object per $k$, so its $\Delta\theta$ is a constant and every $T$ window is scored.
- **S3 — the cap does not return.** On every row (all 39), under $s$ at `1e-7`, no group has a
  region at depth ≥ 18. An over-budget unit counts as false.
- **S4 — the stored noise was within the declared error.**
  - **The rule:** on every row (all 39), $|{\rm total}_s - {\rm total}_{t_5}| \le$
    `total_abserr`$_{t_5}$, at `1e-7`.
  - A failure measures what the stored noise cost a total beyond what it declared. That is a
    finding about the rows before this prompt, not a fault in it.
- **S5 — the declared error is honest after the change.**
  - **The rule:** on every one of the 31 depth-20 rows with both $s$ units in budget,
    $|{\rm total}_s(10^{-7}) - {\rm total}_s(10^{-8})| \le$ `total_abserr`$_s(10^{-7})$.
  - 05a's T4, the same rule under $t_5$, held on 29 of 31.

**Descriptive, not scored:**
- $|{\rm total}_s - {\rm total}_{t_5}|$, relative and against `total_abserr`$_{t_5}$, on every row;
- regions and maximum depth per group, $t_5$ against $s$;
- `total_converged` under both trees;
- `theta_deriv`'s change at the Levin windows: $\varphi'$ was the term 04 and
  `phase-representation` 02 found worse than omitting $\varphi$;
- `GkSourcePolicyData`'s `Levin_z` under both trees;
- `convergence_kinds.py` under $s$, as 05a ran it.

| | Verdict |
|---|---|
| S1 | not yet scored (§6) |
| S2 | not yet scored (§6) |
| S3 | not yet scored (§6) |
| S4 | not yet scored (§6) |
| S5 | not yet scored (§6) |

## Deviations from the prompt

1. **The baselines are the orchestrator's.** **IMPLEMENTATION CHOICE.** They were taken at
   `2006de7`, whose code is `9f66622`'s, and passed in the dispatch. They were not re-run before
   the change.
2. **phi_G is formed, and the agreement check run, in `_build_phase`, not in
   `assemble_GkSource_values`.** **STRUCTURALLY REQUIRED**, by the user's decision on the first
   stop (prompt §9 point 6; §1.4 as amended).
   - `assemble_GkSource_values(k_exit, z_response, z_sample, numeric_data, WKB_data)` has no `tau`
     table, so it cannot form `r + k tau.delta(z_s, z_init)`. It only carries `r`.
   - The check uses the rectifier's shifts as the stored rows record them,
     `raw_theta_div_2pi - theta_div_2pi`, relative to the first sample of `_build_phase`'s data.
3. **The normalised cycle count is `div - 1`, where §9 point 6 and the dispatch say "raised by
   one".** **STRUCTURALLY REQUIRED.**
   - A remainder in `(-2pi, -TWO_PI]` is `rem = -2pi + d` with `0 < d <= 2pi - TWO_PI`. So
     `div 2pi + rem = (div - 1) 2pi + d`: storing 0.0 needs `div - 1`.
   - For the producers' phases, which are negative, that raises the count's magnitude by one, which
     is how the wording is read. The helper's docstring says so.
   - It is tested on three rationals whose remainder rounds to `-TWO_PI`, a positive one and a
     negative one among them.
   - The first version raised `div` by one. My own check on a tiny positive phase found it before
     any test was written (deviation 15).
4. **The payload's two new keys.** **IMPLEMENTATION CHOICE** (§1.1 leaves their form open).
   - `theta_exact_sample` is the exact phase `-(k L.delta + rho.delta)` as a `fractions.Fraction`,
     and the producer adds `Fraction(deltaTheta)` to it. That is "enough for the producer to add
     deltaTheta exactly", in its plainest form.
   - `rho_delta_sample` is `rho.delta(z_init, z)`, the double the phase function always formed.
   - The existing `(div, mod)` keys are now the exact reduction of `theta_exact_sample`.
5. **The exact arithmetic is integer, beside `PrimitivePhase`'s mpmath reducer, and not factored
   out of it.** **IMPLEMENTATION CHOICE**, as the user accepted on the first stop.
   - `PrimitivePhase` and `_remainder_against_two_pi` are untouched.
   - Test 6(a) shows that the two reducers give the same constants, bit for bit, on 301 node
     rationals. `test_the_helper_against_mpmath` shows the integer reducer exact against mpmath on
     2000 rationals.
   - The cost is about 2 µs a sample for `exact_leading_phase`'s arithmetic beyond its `delta_parts`
     call, plus 0.9 µs for the reduction (measured at 1300 samples). An earlier version with a list
     of tuples took about 3 µs; it was rewritten inline before any wall-time run.
6. **`exact_leading_phase` reads `delta_parts`, never `delta` as well.** **IMPLEMENTATION CHOICE.**
   - This spends the same integrand evaluations per sample as the old `leading.delta(z_init, z)`.
     So the metadata's `lead_partials` and `lead_evals`, and `test_gk_wkb_phase`'s exact counts of
     them, are unchanged.
7. **`_check_phase_sign`'s new form.** **IMPLEMENTATION CHOICE**, as the user accepted (item 4).
   - Its signature changes from `(z_points, theta_points, phi_points)` to
     `(z_points, data, phi_points)`. It is private, and its one caller is `_build_WKB`.
   - `test_sign_convention_is_checked`, whose stand-in is now of the opposite convention with its
     residual formed in its own convention (`closed_form_residual(..., sign=-1)`), still raises. With
     the right sign every fixture's stored remainder matches.
8. **Three closed-form stand-ins outside the dispatch's list use the double form.**
   **STRUCTURALLY REQUIRED.**
   - The dispatch lists the Bessel and `solve_ivp` fixtures of `test_tk_source_functions.py` and
     `AnalyticRadiationFixture`, and says every other stand-in uses the small-term construction.
     Three others also have a phase given only in closed form, with no separate small terms to form
     `r` from:
     - `test_phase_groups.matched_LG_functions` (a `solve_ivp` phase);
     - `test_g_phase_reanchor._ProductionPhaseGk` (the Bessel `G` phase);
     - `test_gk_source_policy.FakeGkSource` (`theta = rate log(1+z)`).
   - A small-term route for the `solve_ivp` phase (solving for `r` directly) would make the stored
     `theta` and `r` disagree at the ODE's tolerance. That is above `_check_phase_sign`'s rounding
     tolerance.
   - Each uses `theta - sign k L.delta` in double. The bound is in the table above.
9. **The `G` stand-ins also gain `raw_theta_div_2pi` and `z_init`.** **STRUCTURALLY REQUIRED.**
   - `_build_phase` now reads both, as every `GkSourceValue` holds them: `z_init` for the span term
     and `raw_theta_div_2pi` for the agreement check.
   - `test_g_phase_reanchor._Value` and `test_gk_source_policy.FakeGkSource` model pure-WKB values
     with no wrap. So `z_init = None` and `raw_theta_div_2pi = theta_div_2pi`. Neither touches an
     assertion.
10. **`test_gk_source_policy.FakeGkSource` samples its WKB values more densely.** **STRUCTURALLY
    REQUIRED.** This is the one stand-in change that is not a value's own fields. It is flagged
    here for the review.
    - Its phase, `rate log(1+z)` with `k = 1`, lives entirely in phi. At `rate = 5e3` it moves by
      ~1180 rad between its 40 samples. The new unwrap presumes, as a producer's residual
      guarantees, that phi moves by less than pi between neighbours. Without the change
      `test_a_threshold_crossing_still_reports_the_redshift` got `Levin_z = None`, and the
      agreement check warned.
    - The fix: the WKB values are sampled at `max(n, ceil(rate * span / (pi/2)) + 1)` points, on a
      grid holding the original `n`, at most pi/2 a step. The background table stays on the
      original `n` nodes; the samples between them are off its grid.
    - The constructor's signature and every assertion are unchanged. The three tests take 0.27 s.
    - **A failed first attempt** put the table on all ~29,000 samples. `CumulativeTable`'s
      accumulation is quadratic, so it took 24 s, and it was replaced.
11. **The `ObjectFactories` row fixtures are not given the residual.** **IMPLEMENTATION CHOICE.**
    They exercise read mechanics on rows inserted directly. The column is nullable, so they read
    it back as NULL. No consumer reads them.
12. **`test_residual_part_alone`'s computation changed under §2's class exception.**
    **STRUCTURALLY REQUIRED.**
    - It reconstructed the stored phase as `div * TWO_PI + mod` and added `k cs_tau.delta` in
      double. Its `1e-12` threshold, by its own docstring, sits at the reconstruction's floor.
    - The old stored phase shared the double's roundings, and they cancelled. The exact stored
      phase does not share them: the double reconstruction gives 1.94e-12 rad, over the bound.
    - It now reconstructs as `div * 2pi + mod` and adds the exact `k cs_tau.delta` from the limbs
      and partials, in mpmath at 60 digits. That gives 4.58e-16 rad (6.33e-13 at `9f66622`), with
      the bound unchanged. The asymptote assertions are untouched.
13. **Test geometry left open by §3.** **IMPLEMENTATION CHOICE.**
    - **Test 1's anchor.** "On a node" is the 5th residual-table node below the top of the WKB
      region, `k L(z) >= 100`. "Off" is 37% of the way in `log(1+z)` to the next node down. The
      initial data give `deltaTheta = 2.0` and `B = 1`.
    - **Test 1's first anchor**, at the residual table's own top, failed `WKB_phase_function`'s
      WKB-criterion guard (1.5 and 270 at `k = 3e8`). That attempt was replaced.
    - **Test 3, `T`:** the test-1 objects at `k = 3e8`, read by a real `TkSourceFunctions` on a
      numeric stand-in holding the switch node and the overlap. 261 windows have `|theta| >= 1e11`.
    - **Test 3, `G`:** a `GkSource` at `z_R = 0.1` over 160 pure-WKB objects at `k = 3e8`,
      assembled by `assemble_GkSource_values._function`; 152 windows.
    - **Test 4:** numerically seeded objects at `k = 3e8`. The sources have `k tau(z_s)` in
      25–80, with `z_init` two grid nodes below each source. `deltaTheta` falls 0.35 rad a source
      from 3.0, which gives two wraps.
    - **Test 4's two failed attempts.** The first, at `k = 1e5` with `z_init` off the grid, had a
      window RMS of 0.040 ulp(theta). The second, at `k = 1e5` on the grid, had 0.037. That is
      04's degree-4 fit's own floor on a smooth ~3 rad span term, about 1e-8 rad, above 1e-3
      ulp(theta) at `|theta| ~ 1.4e9`. Test 3's bound applies at `|theta| >= 1e11`, so test 4 was
      moved to `k = 3e8` (`|theta| ~ 4e12`), where it is 1.8e-5 ulp.
    - **Test 5's bound** for `T` is `4 eps (k |cs_tau.delta(z, z_init)| + |phi(z)|)`, and for
      `G` the same with `tau.delta(z_s, z_R)`.
14. **Test 6(b) and 6(c) are an import-closure check plus two bit-identity checks.**
    **IMPLEMENTATION CHOICE.**
    - **"By the diff":** `git diff --stat 9f66622` over `cumulative_table.py`, `BackgroundModel.py`,
      `phase_residual.py`, `GkNumericIntegration.py`, `TkNumericIntegration.py`, `QuadSource.py`,
      `QuadSourceIntegral.py`, `phase_groups.py`, `AdaptiveLevin/`, `StoreFixtures/`, `config/`,
      `main.py` and `ComputeTargets/__init__.py` is empty.
    - **The two bit-identity checks:** the payload's `rho.delta` is the residual table's own,
      bit for bit, and `CumulativeTable.delta` is still `node_term + (p_b - p_a)`.
    - **The import closure:** the numeric producers, `QuadSource`, `BackgroundModel`,
      `phase_residual` and `cumulative_table` import, through this repository, no module whose
      code 06 changed.
15. **Two of my own code errors, caught before any test of them ran, and fixed.** **UNINTENDED
    DRIFT.**
    - The normalisation's cycle count went the wrong way (deviation 3).
    - `_phi_samples` first accumulated the unwrap's cycles a second time (`unwrapped += cycles`)
      when `cycles` was already the running total. It was found on reading, before any run.
16. **Three failed attempts in writing the new test module.** **IMPLEMENTATION CHOICE** (fixes
    to the agent's own test code).
    - `import ComputeTargets.GkWKBIntegration as …` bound the class that `ComputeTargets`
      re-exports, so `mock.patch.object(…, "ray")` failed. The module now comes from `sys.modules`,
      as `test_tk_wkb_phase` does.
    - The negative normalisation case was first built at `2^-200`, beyond the 60-digit
      reference's precision. It now uses `2^-120`.
    - Test 4's `|theta|` first read the rectified cycle count, whose first block is zero, so every
      ulp was tiny. It now reads the producer's `raw_theta_div_2pi`.
17. **The new tool test is a module of its own** (`tools/tests/test_store_value_diff_absent_column.py`),
    rather than methods added to `test_store_value_diff.py`. **IMPLEMENTATION CHOICE.** The dispatch
    allows "a test in `tools/tests/`", and a separate module leaves 05's module untouched.
18. **`GkSource.values_as_DataFrame` gains no `theta_residual` column.** **IMPLEMENTATION
    CHOICE.** It is a diagnostic frame, and §1.3 asks only for the table column and the value
    object.

Deviation 15 is the only `UNINTENDED DRIFT`.

## Verification performed

**The seven roots.** The commands of `CLAUDE.md`, run serially after the change.

| root | baseline at `2006de7` (orchestrator) | after the change | change |
|---|---|---|---|
| AdaptiveLevin | 32 OK | 32 OK | equal |
| ComputeTargets | 764 OK | 777 OK | +13: the 13 methods of `test_stored_phase_residual` |
| CosmologyModels | 39 OK | 39 OK | equal |
| LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) | equal |
| ObjectFactories | 280 OK | 280 OK | equal |
| RunDrivers | 127 OK | 127 OK | equal |
| tools | 55 OK | 58 OK | +3: the 3 methods of `test_store_value_diff_absent_column` |

- `uptime` before the run: `16:52  up 21 days, 15:37, 2 users, load averages: 4.83 5.44 5.25`
  (16:52:38 UTC); after it: `16:56  up 21 days, 15:41, 2 users, load averages: 11.55 6.87 5.79`
  (16:56:11 UTC). `ComputeTargets` took 144.4 s.
- **The known flake**, `test_tk_wkb_phase … test_wall_time_per_object`, did not fire. In that run
  it measured 0.0513 s cold (limit 0.06) and 0.0207 s warm (limit 0.05), against 0.0487 and
  0.0173 at `2006de7` (load ~6). Neither limit moved. `test_gk_wkb_phase`'s cost test measured
  0.0014 s (limit 0.05).

**§3, measured** (each test prints its numbers):

| test | number |
|---|---|
| 1 | in all 16 cases (2 sectors × 2 anchors × 4 k; 938–1309 samples; `|theta|` up to 4.1e12 for `G` and 1.9e11 for `T`), `|mod − exact|` at most 0.496–0.500 ulp(2pi), every cycle count exact; the helper exact against mpmath on 2000 rationals up to 5e12; 3 normalisation cases stored as `(div − 1, 0.0)` |
| 2 | `theta_residual` against `deltaTheta − rho.delta`: worst 0.124–0.125 of `4 eps(|deltaTheta| + |rho.delta|)`; the exact large term added back reproduces the stored remainder to 0.169–0.185 of `2 ulp(2pi) + 4 eps(…)` |
| 3 | phi_T, z_init on a node: 261 windows, window RMS / ulp(theta) median 3.73e-11, max 1.49e-10; off a node: median 4.03e-11, max 1.81e-10. phi_G at z_R = 0.1: 152 windows, median 5.7e-18, max 3.3e-17. Bound 1e-3. **Unscored, the old construction on the same stored values:** phi_T median 0.369 and 0.393, max 0.656 and 0.720; phi_G median 3.85, max 94.3 |
| 4 | 51 sources, two wraps; the unwrap's cycles `{0, 1, 2}` equal the rectifier's shifts at all 51 samples; no warning; window RMS of phi_G at most 1.8e-5 ulp(theta). A rectifier record with no shifts gives one `!! WARNING` line naming k, z_R and the samples, and the same phi, bit for bit |
| 5 | raw_theta against the exact phase: `T` worst 0.31–0.43 of its bound over 8 cases (phi_T equal to the stored residuals in `float.hex`); `G`, 160 sources, 0.194. A value without the residual raises in both consumers |
| 6 | (a) 301 abscissae bit-identical to `PrimitivePhase`'s own formulas (raw_theta, theta_deriv, theta_mod_2pi on and off); 301 node constants identical under both reducers; (b) rho.delta and the table's delta bit-identical; (b), (c) import closures disjoint from the changed modules |
| 7 | Oracle 1, `G q r`, w = 1/3: max relative residual 4.61e-16 (bound 1e-12) |

**The deliberate breakages.** Each was made in the working tree from a saved copy, run, recorded,
and reverted by copying the saved original back, checked with `cmp`. `grep -rn BREAKAGE` over the
changed packages finds nothing.

- **(i) The producer reduces against `TWO_PI` again.**
  - The edit, in `reduce_exactly_mod_2pi`: `two_pi = Fraction(TWO_PI).numerator << (work - …)`,
    in place of `_two_pi_scaled(work)`.
  - **Caught by test 1** in all 16 subtests, `FAILED (failures=18)`. The error is
    `|theta|/2pi · (2pi − TWO_PI)`: 5.35e-8 rad at `G`, k = 1e5, up to 1.61e-4 rad at k = 3e8.
    For example: `AssertionError: 0.00016054241986894753 not less than or equal to
    1.7763568394002505e-15`.
  - Also by the helper's two unit tests.
- **(ii) `theta_residual` formed as `theta + k L.delta` in double**, in both producers.
  - **Caught by test 2** in all 16 subtests, at 7.6e6 to 2.7e11 times its bound.
  - **Also by test 3:** phi_T's window RMS 0.615 ulp(theta). `FAILED (failures=17)`.
- **(iii) `_build_phase` back on `build_phi_samples` over `div * TWO_PI + mod`.**
  - **Caught by test 3:** `AssertionError: 94.3480553883097 not less than or equal to 0.001 :
    phi_G, z_R = 0.1`.
- **(iv) The phi_G unwrap removed** (`cycles = 0`).
  - **Caught by test 4:** `test_unwrap_matches_the_rectifier_and_phi_is_continuous` fails on its
    no-warning assertion. The warning says the unwrap disagrees with the rectifier at 33 of 51
    samples.
  - `test_a_disagreement_warns_and_the_build_goes_on` fails too (`0 != 1`). `FAILED (failures=2)`.
- **(v) `TkSourceFunctions` builds phi_T the old way** (`build_phi_samples` over
  `div * TWO_PI + mod`).
  - **Caught by test 3:** `AssertionError: 0.6562351654792864 not less than or equal to 0.001 :
    phi_T, z_init on a node`.
- **And the tool extension removed** (`absent = {}`), beyond §3's list.
  - (t6a) fails in both subtests (`2 != 1`, the store refused for lacking `theta_residual`), and
    (t6b) errors with `StoreSchemaMismatch`.

**Other checks.**
- **`LiouvilleGreen/WKBtools.py` is docstrings and comments only:** its AST with docstrings
  stripped equals `git show 9f66622:LiouvilleGreen/WKBtools.py`'s.
- **The untouched paths:** `git diff --stat 9f66622` over the paths the dispatch forbids is empty
  (deviation 14).
- **The test modules touched for stand-ins**, before the new module was written:
  `test_tk_source_functions` (22), `test_phase_groups`, `test_quadsource_integral`,
  `test_gk_source_policy`, `test_g_phase_reanchor` and `test_t_phase_reanchor` (76 together, with
  the policy module's 3), `test_gk_wkb_phase` and `test_tk_wkb_phase` (38). All OK.
- **`black`:** all 21 `.py` files changed or added are clean under `--check`.
  `test_g_phase_reanchor.py` was reformatted, in its stand-in lines only.

## Observations not acted on

1. **`primitive_phase.py`'s module docstring is now stale in two places.**
   - Lines 27–32 say that phi "is supplied by the caller, which recovers it from the stored
     samples as `phi(z_i) = theta_stored(z_i) − sign k leading.delta(z_i, z_anchor)`".
   - The end of "Who uses which" says that neither mode touches "the noise in the stored samples
     of phi (`[02-consumer-phi-below-the-storage-granularity]`)".
   - Both consumers now take phi from the stored residual. The dispatch allows only the helper
     and `build_phi_samples`' docstring in that file, so they are left. A follow-up commit could
     correct both, docstring only. No issue is opened.
2. **The rectifier is fragile where phi_G is not.** On test 3's pure-WKB source at k = 3e8, the
   old construction's phi_G reached 94 ulp(theta) in its worst window. That is far above the
   quantisation 04 measured (0.46–0.78 F), and it shows the old rectifier misreading
   quantised samples. It no longer touches phi; its `theta_div_2pi` output is still stored, and
   is now only checked against.
3. **The agreement check may warn in R14.** As the stop report said:
   - the rectifier changes a block only where raw theta rises;
   - a numerically seeded source with `k Delta tau >~ pi` between neighbours, or a downward wrap
     of `deltaTheta`, can defeat it.
   
   The fixtures here never warn. A warning in R14 is §8's stop; the build is not killed by it.
4. **The cold wall-time test's margin is now about 8 ms.** It measured 0.0513 s against 0.06 s,
   at load ~5. At `2006de7` it was 0.0487 s at load ~6. The warm one is far inside its limit.
   Under heavy load the cold limit could be reached, which is the same flake class as
   `test_wall_time_per_object`'s known one.
5. **`theta_div_2pi` is now a count against 2pi, but some readers rebuild theta against
   `TWO_PI`.** These are `GkWKBValue.theta`, `TkWKBValue.theta`, `GkSourceValue.WKB.theta` and
   `raw_theta`, and the extract readers.
   - The rebuilt phase is therefore off by `div · (2pi − TWO_PI) ≈ 0.04 eps |theta|` (1.6e-4 rad at
     4e12), on top of its own ulp.
   - No consumer of a phase uses it, and the rectifier's comparisons are unaffected at that size.
     It is a diagnostic's accuracy, recorded for §5's readers.

## State handed on

- **One commit**, not pushed. It holds:
  - `Quadrature/integrators/WKB_phase_function.py`;
  - `ComputeTargets/{GkWKBIntegration,TkWKBIntegration,GkSource,GkSourcePolicyData,TkSourceFunctions,primitive_phase}.py`;
  - `LiouvilleGreen/WKBtools.py`;
  - `ObjectFactories/{GkWKBIntegration,TkWKBIntegration,GkSource}.py`;
  - `tools/store_value_diff.py` and `tools/tests/test_store_value_diff_absent_column.py`;
  - `ComputeTargets/tests/test_stored_phase_residual.py` (new);
  - the stand-in and assertion changes in `ComputeTargets/tests/{test_tk_source_functions,test_phase_groups,test_g_phase_reanchor,test_t_phase_reanchor,test_gk_source_policy,test_gk_wkb_phase,test_tk_wkb_phase}.py`;
  - this log, and the board's 06 row and header.
- `docs/OPEN_ISSUES.md` is untouched: no issue was opened.
- **The header's issue, `[02-consumer-phi-below-the-storage-granularity]`, does not move in this
  commit** (§4). It moves in 06's records commit, after §5 and §6.
- **For §5:** R13 at this commit's parent and R14 at this commit, compared at this commit with
  `PYTHONPATH=. ./venv/bin/python -m tools.store_value_diff R13.sqlite R14.sqlite --json`.
  - The tool reads R13 without `theta_residual`. In each of `GkWKBValue`, `TkWKBValue` and
    `GkSourceValue` it reports `theta_residual` under `columns_only_in_b`, and compares every
    other column as before.
  - The three owning classes are then "not value-identical" for that reason alone, beside their
    value changes.
  - Look for `!! WARNING (GkSourcePolicyData` in R14's output: any such line is §8's stop.
- **For §6:** the re-run producers need `WKB_phase_function`'s two new payload keys. `store()`
  reads them.
