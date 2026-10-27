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

**Commit:** `dc9d28c` (§4). It passed the orchestrator's nine checks on the laptop, with
breakages (ii) and (iii) re-applied there, and was pushed (orchestrator note §A.3, §B.5).

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

**Scored on the Mac Studio, 2026-10-26** (run `levin-phase-floor-06-wider-20261026T223634`; the
§6 record below, and the Mac Studio review's check 9, which re-scored each from the saved windows
and checkpoints and matched). Added by the laptop orchestrator in the records commit; the table
above is as the code's agent wrote it.

| | Verdict | Numbers |
|---|---|---|
| S1, reading (I) of record | **holds** | 6,700 of 6,700 pure-WKB $G$ windows on R1–R7; $\sigma_{t_5}$ exactly 0 in all; $\sigma_s \le 8.8\times10^{-14}F$ |
| (S1, reading (L), unscored) | undecided | no window: A3 v2 stores no null `z_init` |
| S2 | **fails** | true in 10,747 of 12,862 $T$ windows; the 2,115 false are all at $z \ge 1.75\times10^7$, where $\sigma_s/\sigma_{t_5}$ = 0.952–1.04 |
| S3 | **holds** | 39 of 39; largest depth 11 |
| S4 | **fails** | 34 of 39; `low-G-3` 10.8, `ctrl-low-1` 3.90, `ctrl-low-2` 3.57, `mid-G-2` 1.62, `R2` 1.49 × `total_abserr`$_{t_5}$ |
| S5 | **fails** | 28 of 31; `mid-G-3` 1.57, `mid-G-r-3` 1.32, `mid-G-1` 1.31 |

**The pure-WKB reading** was ruled by the user, "Clearly (I).", with no S1 number in existence
(the §6 record, deviation 17). **The header issue** was closed by the user on 2026-10-27 although
S2 fails: S2's false windows lie where there was no storage noise to remove (W4).

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

# The Mac Studio record: §5 and §6

> **Appended by the laptop orchestrator** in 06's records commit (orchestrator note §B.4). Part B
> ran on the Mac Studio on 2026-10-26, from two detached worktrees, while the main checkout stayed
> at `28ebaf0`: `SecondaryGWKit-lpf06-9f66622` (the code commit's parent: R13 and $t_5$) and
> `SecondaryGWKit-lpf06-f5cee4c` (Part B's dispatch SHA, code `dc9d28c`: R14 and $s$). Its
> orchestrator dispatched two agents, whose logs follow as they wrote them. Its review passed all
> ten checks on the user's ruling on check 4 (`docs/levin-phase-floor/06/review.md`).
>
> A path `scripts/…`, `out/…` or `state/…` below is under `var/levin-phase-floor-06/rehearsal/` or
> `…/wider/` there, and is committed at `docs/levin-phase-floor/06/rehearsal/` or `…/wider/`. Not
> copied back: every pickle, every file over 2 MB, every CSV, the comparison's 1.09 GB
> `store_value_diff.json`, the readers' `extract/` (176 MB) and the scratch copy of A3 v2. They
> stay in the Mac Studio's `var/`, listed with their SHA-256 in
> `docs/levin-phase-floor/06/KEPT_ON_MAC_STUDIO.md`.
>
> **Three questions, each answered by the user on 2026-10-26:**
> - **How $s$ reads A3 v2** (the re-measurement's stop before U0): route (B), *"We have to be
>   practical and pragmatic. The repo rules are there to prevent cheating, not to get in the way
>   of honest work."* A certified scratch copy with the three nullable columns added.
> - **S1's pure-WKB reading:** *"Clearly (I)."*, `z_init == z_source`, ruled with no S1 number in
>   existence. The prompt's "`z_init` null" was the laptop's error, from a comment at
>   `GkSource.py:34` never checked against the store.
> - **Check 4:** *"Check 4 passes: the classifier's 185 UNCLASSIFIED all lie inside the five
>   classes prompt §5.4 expects to change, with 0 outside, so the review passes."* 05's classifier
>   exempts `QuadSourceIntegral` alone, and the orchestrator note asked for "no `UNCLASSIFIED`".
>
> **P1, investigated after the review.** Two rehearsal totals at $z_{\rm response} = 250$ moved by
> 1.8e4× and 484× R13's declared error. At the user's request a laptop subagent rebuilt both rows
> in-process under each tree and integrated the one group that differed against an independent
> reference. R13's totals are right and R14's are wrong: the group has an interior stationary point,
> which a single Levin region misses with an error estimate about $10^4$ too small, and R13's
> noisier phase forced the bisection that found it by luck. 06's integrand is the more accurate of
> the two. [`docs/levin-phase-floor/06/p1/report.md`](../../../docs/levin-phase-floor/06/p1/report.md);
> opened as `[06-levin-accepts-a-region-across-a-stationary-point]`.
>
> **The issues the agents proposed.** P1 is opened as above; P2 as
> `[06-ray-task-output-does-not-reach-the-launch-log]`; W4 as
> `[06-s2-window-metric-floors-on-phi-t-structure]`. The rest (W1–W3, W5–W12, and the rehearsal
> agent's recurrences) are recorded on the board under "06's other findings". W3's open point is
> settled: `CumulativeTable.delta(z, z)` is exactly 0.0 by construction (its docstring), so a
> pure-WKB value's second term in `_phi_samples` is exactly zero.
>
> **The laptop's re-checks** (orchestrator note §B.4), read-only over SSH, in
> `docs/levin-phase-floor/06/laptop_rechecks/`:
> - on the changed `GkWKBValue` of largest $|\theta|$ (766,842,538 rad), each store holds exactly
>   one row with the tabulated (cycle count, remainder); `G_WKB` and R14's `theta_residual` read as
>   tabulated; $|\Delta{\rm mod}| = 0.237\,\varepsilon|\theta|$; the R13/R14 files hash unchanged;
> - on `mid-G-1`, a row neither U0 nor the Mac Studio review used, the $t_5$ unit at `1e-7` equals
>   05a's $t$ unit in total, `total_abserr` and all five calls, by bit pattern.

## §5 — rehearsals R13 and R14 on the Mac Studio

**Prompt:** [`06-producers-store-the-residual.md`](../06-producers-store-the-residual.md) §5, with
its §9, under the orchestrator's Part B dispatch (`f5cee4c`), its §B.1 and §B.5. §B.5 governs.
**Host:** mps026298, the Mac Studio.
- **R13** ran in the worktree `SecondaryGWKit-lpf06-9f66622`, detached at `9f66622`, the code
  commit's parent.
- **R14**, the comparison and the readers ran in the worktree `SecondaryGWKit-lpf06-f5cee4c`,
  detached at `f5cee4c`, whose code is the code commit `dc9d28c`'s.
- **The main checkout** stayed at `28ebaf0` and clean. It ran nothing, and was only the home of the
  venv, the stores, the registry and these records.

**Model:** Claude Opus 5.5, as the Mac Studio orchestrator's rehearsal subagent, in three turns:
- the gate, the lock and R13's launch;
- R13's checks and R14's launch;
- R14's checks, the comparison, the tabulations, the readers, the release and these records.

**Date:** 2026-10-26, 21:55–22:15 UTC.
**Commit:** none. The records are committed on the laptop in 06's records commit.
**Result:** COMPLETE WITH DEVIATIONS. None of the deviations stopped the work, and none changed a
computation.
- **The builds.** R13 and R14 both ended `done` with exit 0 and fingerprint `1b332ffa…`. Each
  equals R8's class by class, and R14's `classes` object equals R13's.
- **The warning.** R14 printed no `!! WARNING (GkSourcePolicyData` line (0, in its launch log,
  `stdout.log`, `stderr.log` and Ray's worker logs), and `_check_phase_sign` did not raise.
- **Outside the five classes of §0.3:**
  - every value table is bit-identical: 4 tables, 40,958 rows;
  - every other difference is of the volatile kinds: 0 unclassified;
  - no row and no column is in one store only.
- **Inside them:**
  - `theta_residual` is in R14 only;
  - `theta_mod_2pi` moved in 146,251 $G$ and 2,162 $T$ rows, by a median of 0.3–0.4
    $\varepsilon|\theta|$;
  - **no `theta_div_2pi` changed;**
  - `G_WKB` and `T_WKB` moved by the remainder's change alone;
  - `GkSourcePolicyData` moved only in `Levin_z_dtheta_dlogz`;
  - 554 of 864 `QuadSourceIntegral` totals moved. Two of them, at $z_{\rm resp} = 250$, moved by
    1.8e4 and 484 × R13's `total_abserr`, with their depth halved.
- **The readers** exited 0, and the stores were left byte-identical.
- **The lock** was released.

The measurements are in `verification.md`, written for `docs/levin-phase-floor/VERIFICATION.md`
§6 "The stored phase".

### What shipped

Nothing in the repository: no code, and no commit. Everything was written on the Mac Studio, in
the main checkout:
- **The records directory** `var/levin-phase-floor-06/rehearsal/`, holding:
  - `out/`;
  - `scripts/`:
    - 05's `classify_non_qsi.py` and `tabulate_qsi.py`, unchanged;
    - this agent's `tabulate_phase_classes.py` (§5.4's per-class tabulation),
      `examine_outliers.py`, `summarise_diff.py`, `qsi_extra_counts.py`, `peek_structure.py` and
      `run_readers_worktree.sh`;
  - `state/`;
  - `extract/`, the readers' output, 176 MB, not for copying back;
  - `log.md`, `verification.md` and `issues.md`.
- **The stores** `var/datastores/levin-phase-floor-06-r13*.sqlite` and `-r14*.sqlite`, five files
  each.
- **The runs** `levin-phase-floor-06-r13-parent-20261026T215550` and
  `levin-phase-floor-06-r14-residual-20261026T215945`, with their launch logs
  `var/runs/levin-phase-floor-06-r1{3,4}.launch.log`.

The stores are kept, as prompt §5.7 says.

### Predictions and verdicts

§5 fixes no scored predictions; §6 holds them. These are the acceptance checks of §7.2, and the
dispatch's own:

| check | number | verdict |
|---|---|---|
| R13 ends `done` with `1b332ffa…` | `done`, exit 0, `1b332ffa…`, 21/21 class digests = R8's | holds |
| R14 ends `done` with `1b332ffa…` | `done`, exit 0, `1b332ffa…`, 21/21 = R8's; `classes` object = R13's | holds |
| outside the five classes, every value table identical | 4 tables, 40,958 rows, bit-identical | holds |
| outside them, every other difference of the four kinds; no row in one store only | 0 unclassified outside; 0 only-A, only-B or unmatched in any class; no column in one store only outside | holds |
| inside them, the differences tabulated (§5.4's six items) | `out/phase_classes_tabulation.txt`, `out/qsi_tabulation.txt` | done |
| `theta_div_2pi` changes: none, or one cycle within 4ε\|θ\| of 0 or −2π | 0 changes in all three value tables; 0 `raw_theta_div_2pi` changes | holds |
| the readers exit 0, store byte-identical | 6 × exit 0; 5 files identical | holds |
| the lock is released | `release` exit 0; `status` free; `ray-processes` none | holds |
| (dispatch) R13 ≠ R14 in `GkWKBIntegration`'s values | 146,251 of 147,642 `GkWKBValue` rows differ | holds |
| (dispatch, §8) no `!! WARNING (GkSourcePolicyData` line in R14 | 0 (launch log, stdout, stderr, 24 Ray worker logs) | holds |
| (dispatch) `_check_phase_sign` does not raise in R14 | no Traceback, `Error` or `Exception` anywhere; `done` exit 0 | holds |

### Deviations from the prompt

#### 1. Both builds ran from worktrees, on the main checkout's venv — STRUCTURALLY REQUIRED

Prompt §5.2 puts R13 in a worktree, and the dispatch puts R14 in one too, so that the main checkout
does not move.
- **The python.** Neither worktree has a venv, so each build used the main checkout's
  `venv/bin/python` by absolute path, with cwd and `PYTHONPATH` set to its worktree.
- **The paths.** The `--database` and the launch log were absolute paths into the main checkout.
  The registry recorded both runs under the main checkout's `var/runs/`.

The evidence that each ran its own tree:
- the manifests record `git_head` `9f66622d…` and `f5cee4c7…`, both with `git_dirty` false;
- `lsof` showed each driver's cwd, and its `Datastore` actor's cwd (97834 and 133), as its own
  worktree;
- the two stores differ in `GkWKBValue`, which the parent cannot write differently from itself.

#### 2. `PYTHONDONTWRITEBYTECODE=1` on both builds, and `-B` on the in-process tools — IMPLEMENTATION CHOICE

These keep `__pycache__` out of the worktrees, which must not be written, and change no
computation. This is R11's form from 05a. Every check of the worktrees found them clean: `git
status --short --ignored` was empty, and `find -newer` the pre-R13 marker listed nothing, after
each build, after the readers and at the end.

#### 3. The comparison ran with the code worktree as its cwd — STRUCTURALLY REQUIRED

`python -m` puts the cwd first on `sys.path`. The code worktree's `tools/store_value_diff.py`
(`7997af85…`) is the only one that reads a store lacking `theta_residual`. The main checkout's is
05's `5c92d8e3…`, so run from there `-m tools.store_value_diff` would have imported it.
- The dispatch's form names `PYTHONPATH=<code worktree>`. The cwd made it bind.
- The store paths were absolute.

#### 4. No text form of the comparison — IMPLEMENTATION CHOICE

05 and 05a also ran the text form with `--max-rows 1000000`. Here the JSON alone is 1.09 GB, against
41 MB in 05a, because 146,251 value rows differ in each of three tables. §5.4 asks for `--json`
only. Every summary was taken from the JSON, or from the tool's own reader.

#### 5. The `!! WARNING` search was extended to Ray's worker logs — IMPLEMENTATION CHOICE

The dispatch asks for a `grep` of R14's launch log and stdout. In both builds the launch log holds
the `Datastore` actor's output only, and no line from any task. `_build_phase` and
`_check_phase_sign` run in Ray tasks, so their prints land first in
`/tmp/ray-sgk/session_*/logs/worker-*.out`.
- The same patterns were searched there too, for both sessions: `!! WARNING (GkSourcePolicyData`,
  any `!! WARNING`, `Traceback`, `_check_phase_sign`, `error|exception`.
- Every count is 0 ([`out/ray_worker_logs_check.txt`](out/ray_worker_logs_check.txt)).
- These logs are read-only, and are kept by Ray, not by this record. The count in the launch log
  and stdout is 0 as well.

#### 6. `classify_non_qsi.py`, copied unchanged, exits 1 with 185 "UNCLASSIFIED" — IMPLEMENTATION CHOICE in reading it

05 wrote the script for a change to `QuadSourceIntegral` alone, so it exempts only that class.
§5.4 says it sorts the differences outside the five classes of §0.3. All 185 problems it reports
are inside them:
- 182 are `GkSourcePolicyData` rows whose `metadata` key `Levin_z_dtheta_dlogz` moved;
- 3 are the value tables of `TkWKBIntegration`, `GkWKBIntegration` and `GkSource`.

Outside the five classes it reports 0. The script was not edited. Its output was read class by
class, and `summarise_diff.py` checks the outside/inside split from the JSON independently.
- **Why this was not a stop.** The dispatch lists "an `UNCLASSIFIED` difference" among its stops,
  and §8 words the stop as "outside those classes".
- I read the dispatch's stop as §8's. Every one of the 185 is a change §0.3 expects and §5.4
  tabulates.
- If the orchestrator reads it otherwise, this is the point to ask at. Nothing after it wrote to a
  store.

#### 7. The readers ran through a three-line copy of `run_readers.sh` — STRUCTURALLY REQUIRED

`run_readers.sh` changes directory to its own tree, then calls `./venv/bin/python`.
- **From the code worktree** there is no venv.
- **From the main checkout** it would run the main checkout's code (`28ebaf0`, without the column)
  against R14, not the code commit's.

`scripts/run_readers_worktree.sh` takes the tree and the python from the environment
(`READERS_TREE`, `READERS_PYTHON`) and sets `PYTHONDONTWRITEBYTECODE=1`. Nothing else differs:
[`out/run_readers_worktree.diff`](out/run_readers_worktree.diff), 3 lines. The readers'
`--database` and `--output` were absolute.

#### 8. More read-only scripts than the one the dispatch names — IMPLEMENTATION CHOICE

The dispatch names one new script, for §5.4's per-class tabulation: `tabulate_phase_classes.py`.
Four more were written, each saved in `scripts/` with its SHA-256 in
[`out/script_sha256.txt`](out/script_sha256.txt), and each stamps its own hash and its inputs':
- `summarise_diff.py` is 05a's inline summary made a script, with the inside/outside split and the
  columns in one store only;
- `peek_structure.py` is a first look at the JSON's shape;
- `examine_outliers.py` covers the rows above 4 ε|θ| and above A|Δθ|;
- `qsi_extra_counts.py` gives counts from `qsi_rows.csv`.

None writes a store.

There were also two exploratory runs, inline in Python, that printed the tool reader's column lists
and key shapes. Their output was not saved, and no number in these records comes from them; their
form is in `out/commands.log`.

#### 9. `examine_outliers.py` failed once — UNINTENDED DRIFT

The first run exited 1 with a `ValueError`. The CSV holds k as a `float.hex` string, and the script
parsed it with `float()`. The script was corrected to use `float.fromhex` and re-run, and it exited
0.
- The failed run's output is kept, as `out/examine_outliers.attempt1.{txt,stderr}`.
- It had printed only the header lines before failing. No number came from it.

#### 10. The per-class tabulation pairs rows with the tool's own reader — IMPLEMENTATION CHOICE

`tabulate_phase_classes.py` calls the code worktree's `tools.store_value_diff.read_store`, with
`config.datastore`'s factories. It does not use raw SQLite, as `tabulate_qsi.py` does.
- **What that gives.** The pairing is exactly the tool's identity matching, and every column of
  both sides is in hand, not only the differing ones the JSON carries. The JSON omits
  `theta_div_2pi` and `theta_residual` because they do not differ.
- **The check.** Its differing-row counts per column equal the JSON's for all three tables.

#### 11. The scale ε|θ| was taken from R14's own stored cycle count, per table — IMPLEMENTATION CHOICE

§5.4 forms $|\theta|$ "from the stored cycle count".
- **For the producer tables** that is the producer's own count, anchored at its `z_init`.
- **For `GkSourceValue`** it is the rectified count, so the same remainder is measured against a
  larger $|\theta|$ there.

Both are reported. The 54 producer rows above 4 ε|θ| are re-expressed against `GkSource`'s
$|\theta|$ in `out/examine_outliers.txt`.

#### 12. The fingerprints were not re-taken — IMPLEMENTATION CHOICE

Both are the launcher's, taken at the end of each build and read from `status.json`, as in 05a.
- R13's five files are byte-identical from just after its build (21:59) to the end.
- R14's five were first hashed at 22:04, after its fingerprint and before any read, and are
  identical to the end.

#### 13. `/usr/bin/time` wrapped the comparison and four scripts — IMPLEMENTATION CHOICE

- **`-p`** wrapped the comparison, `classify_non_qsi.py` and `tabulate_qsi.py`.
- **`-l`**, which adds the peak memory, wrapped `peek_structure.py` and
  `tabulate_phase_classes.py`. The peaks were 3.5 and 4.6 GB.

Its lines are the whole of each `.stderr` file beside these outputs: no script wrote anything else
to stderr.

#### 14. The gate's `ls var/levin-phase-floor-06` shows `rehearsal` — IMPLEMENTATION CHOICE

I created the records directory a moment before the gate re-check, so that the gate's output
could be written into it. There was no `var/levin-phase-floor-06/` before that, and no
`levin-phase-floor-06-*` store: the gate's `grep` exited 1.

### Verification performed

| check | result | where |
|---|---|---|
| gate | three trees at their SHAs and clean; `dc9d28c^` = `9f66622`; `dc9d28c..f5cee4c` touches only `prompts/`; 21 runs finished; lock free; no 06 store; launcher `5b984c75…` in all trees | `out/gate.txt` |
| R13 | `done`, exit 0, pid 97740, `1b332ffa…`, `9f66622d…`, `git_dirty` false; 0 Tracebacks; `stderr.log` empty | `out/r13_end_check.txt`, `out/r13_status.json` |
| R14 | `done`, exit 0, pid 99938, `1b332ffa…`, `f5cee4c7…`, `git_dirty` false; 0 Tracebacks; `stderr.log` empty | `out/r14_end_check.txt`, `out/r14_status.json` |
| warning / sign check | 0 in launch logs, stdout, stderr and 24 worker logs per build | `out/r14_end_check.txt`, `out/ray_worker_logs_check.txt` |
| worktrees unwritten | `status --ignored` empty; `find -newer` empty, after each build, after the readers, and at the end | `out/r13_end_check.txt`, `out/r14_end_check.txt`, `out/final_check.txt` |
| scripts | the two copies equal 05's (`cmp`); `run_readers.sh` and the launcher equal 05's; the tool is `dc9d28c`'s | `out/script_sha256.txt` |
| value tables | outside: 4 tables, 40,958 rows, identical; inside: 3 tables differ, `theta_residual` in R14 only | `out/store_value_diff_summary.txt` |
| row columns outside | 0 unclassified; every kind timing or `label` | `out/classify_non_qsi.txt` |
| per-class tabulation | the counts equal the JSON's; 0 cycle changes | `out/phase_classes_tabulation.txt`, `out/examine_outliers.txt` |
| QSI tabulation | pairing = the tool's identities; 0 `total` mismatches against the tool | `out/qsi_tabulation.txt`, `out/qsi_extra_counts.txt` |
| readers | 6 × exit 0; lookups equal 05a's; R14 identical | `out/readers.txt`, `state/r14_store_sha256_{before,after}_reads.txt` |
| stores | 10 files identical from before the comparison to the end | `state/store_sha256_before_diff.txt`, `state/store_sha256_after_diff.txt`, `state/stores_sha256_final.txt` |
| lock | acquire 0; set-pid 0 ×2; ray-processes 0 ×4; release 0; status 0 | `out/lock.log` |

### Observations not acted on

- **Two `QuadSourceIntegral` rows moved far beyond R13's declared error.**
  - **Where:** at $z_{\rm resp} = 250$, at $(k, q, r) = (3.162\times10^5)^3$ and
    $(10^5, 3.162\times10^5, 3.162\times10^5)$.
  - **How far:** `total` moved by 1.6e-3 and 2.8e-5 relative. Against `total_abserr`$_A$ that is
    1.8e4 and 484×; against `total_abserr`$_B$, 1.3e4 and 555×.
  - **What else changed:** depth 12 → 6 and 14 → 6, and the region count was roughly halved. Both
    rows are unconverged in both stores.
  - **Elsewhere:** 2 more rows exceed 1 × `total_abserr`$_A$ (1.25 and 1.02), and 55 exceed 0.1.
  - This is the rehearsal scope's counterpart of §6's S4. It is recorded here, not judged, and is
    proposed in `issues.md` for the board to weigh.
- **ε|θ| understates the old error near the producer's anchor.**
  - Every producer row above 4 ε|θ| (52 $G$, 2 $T$; max 272) has $|\theta| < 1$ rad and
    $|\Delta| \le$ one ulp of $2\pi$.
  - The old sum cancelled terms of size $k\tau$, so its error there is set by them, not by
    $|\theta|$.
  - The 99th percentile overall is 1.2–1.4 ε|θ|.
- **`G_WKB` and `T_WKB`.** Every change is within $A|\Delta\theta_{\rm mod}| + 4\varepsilon A$, the
  remainder's change plus $G$'s own rounding. The relative changes up to 3.6e-4 are at zeros of the
  oscillation.
- **`Levin_z` did not move in any row.** Only the diagnostic `Levin_z_dtheta_dlogz` moved, in 182 of
  291 `GkSourcePolicyData` rows, by at most 2.6e-6 relative.
- **`total_phase_limited` flipped in 13 rows** (4 0→1, 9 1→0). No `total_converged` flag changed.
- **The GkWKB residual-memo counters** differ in 25 rows. That is 05's P2 again: 25 in 05 and 31 in
  05a.
- **26 rows have `total_abserr` = inf in both stores.** 9 of their `total`s differ. This is
  `[04-roundoff-floor-can-be-infinite]`.
- **The readers' file counts differ from 05a's** (Gk 396 against 387, GkSource 176 against 162,
  QSI 70 against 73), and the lookups are equal. This is
  `[10-the-readers-draw-their-work-lists-unseeded]`.
- **Each reader logs one `!! WARNING`** about the number of shards read against the number
  specified. It is not the §8 warning, and no reader failed.
- **The size of `out/`.** `store_value_diff.json` is 1.09 GB, against 41 MB in 05a. The orchestrator
  decides whether to copy it back. The summaries, `phase_classes_rows.csv` (297,493 rows) and
  `qsi_rows.csv` carry what the records cite.
- **`[05-store-value-diff-compares-volatile-columns]`** recurs exactly as 05 found it. It is already
  open.
- **Prompt §6's re-measurement** ran beside these builds under its own agent. It was not touched.

### State handed on

- **The lock is free.** `status` reports that the lock directory does not exist, and there are no
  Ray processes of `/tmp/ray-sgk`. `~/.raypoolkit/` is empty.
- **The registry** holds 23 runs, all finished. R13 and R14 are `done`.
- **The stores.** R13's primary is `766839d6…` and R14's is `fff6cf7f…`. Both are in
  `var/datastores/`, both carry fingerprint `1b332ffa…`, and neither has a sidecar. They are kept.
- **The trees.** The main checkout is at `28ebaf0` and clean. Both worktrees are at their SHAs,
  clean and unwritten.
- **The links.** The records' links assume they are copied to `docs/levin-phase-floor/06/rehearsal/`.

## §6 — the re-measurement (Mac Studio)

**Subject:** prompt 06 §6 on the Mac Studio (mps026298), Ray-free, no lock, by the re-measurement
agent. **Run:** `levin-phase-floor-06-wider-20261026T223634`, launched 22:36:34Z (sh 19695,
driver 19700). It ended `done`, 109/109, exit 0, at 23:23:16Z. **Commit:** none here; the orchestrator commits
the records in 06's records commit.

**Result:** U0 holds (all four clauses). **S1 holds** (reading (I), of record, the user's
ruling; 6,700 of 6,700 pure-WKB windows), **S2 fails** (2,115 of 12,862 T windows, all at
z ≥ 1.75e7, where `sigma_s ≈ sigma_t5`), **S3 holds** (39 of 39, depth ≤ 11), **S4 fails** (34 of
39), **S5 fails** (28 of 31). No unit went over budget. A3 v2 is byte-identical, fingerprint
`0e32f24a…`.

**Stop and the user's ruling.** `s` cannot open A3 v2 (`StoreSchemaMismatch`: its factories
declare `theta_residual`, which A3 v2's value tables lack; `out/cmd/06`). The user chose route (B)
(relayed 2026-10-26): an unregistered scratch copy of A3 v2 with the three nullable columns added,
all NULL, certified, opened read-only by `s`; the copy is kept until the records are committed and
deleted on the user's word.

### Gate U0 (before the launch): holds
- (a) `git diff --stat 1fc349c 9f66622 -- . ':!docs' ':!prompts'`: `CLAUDE.md`,
  `primitive_phase.py`, `test_g_phase_reanchor.py`; both files' ASTs without docstrings equal
  `1fc349c`'s (`out/U0_a.txt`). **Holds.**
- (b) t5@1e-7 on `low-G+r-1` and `ctrl-low-1` equals 05a's `t-1e-7-*` bit for bit: total, abserr,
  every call's value and abserr, the whole result summary, calls, `qs_check`, dispatcher, stored
  summary, region arrays (`out/U0.txt`). **Holds.**
- (c) the producers re-run at the parent on A3 v2 (both rows; 1,608 + 1,548 `GkWKBIntegration`,
  241,146 values; 4 `TkWKBIntegration`, 5,172 values): `theta_div_2pi`, `theta_mod_2pi`, `G_WKB`,
  `T_WKB` and every coefficient bit-identical to A3 v2. Beyond the clause: the re-assembled
  `GkSource` matches in every stored column but one `analytic_G_rad` (`0.0` vs `-0.0`), the
  recomputed `GkSourcePolicyData` equals the stored one, and each re-run unit's result, calls,
  regions are bit-identical to the stored-input t5 unit. Only the diagnostic `analytic_*_w` and
  some `analytic_G_rad` values differ (issues W2). **Holds.**
- (d) two in-process `s` runs per row, on the certified copy: result, calls, `qs_check`,
  dispatcher, regions, the re-run's record, the policy, the probe windows and samples, every
  `deltaTheta`: bit-identical. **Holds.**

### Predictions and verdicts

Prompt 06 §6's predictions, copied unchanged (as amended by §9 point 4 before dispatch).
`state/predictions.md` holds the same text, verified byte for byte, written at 22:32:53Z, after U0
held (22:32:12Z) and before the launch (22:36:34Z), with its readings and, added at 22:36:09Z
before the launch, the pure-WKB tag read two ways.

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

**The pure-WKB tag, and the user's ruling.** A3 v2 stores a pure-WKB object's `z_init` as its
`z_source`, never null (9,552 of 9,552; `out/cmd/28`), so §6's "(`GkSource`'s `z_init` null)" tags
no window. Reading (I), `z_init == z_source` exactly, and reading (L), null, were fixed before the
launch. **The user ruled "Clearly (I)."** (relayed through the laptop session). The question was
relayed at 22:40 UTC, when no `s` unit on R1–R7 had checkpointed; when the ruling reached the
orchestrator at 22:44:56 UTC the checkpoint held 49 units, none an `s` unit on R1–R7. The ruling
was made with no S1 number in existence; this agent read nothing from the run between the launch
check and the ruling.

**Verdicts** (`out/cmd/39_analyse06.txt`; windows in `state/probe_windows.json`):

| | Number | Verdict |
|---|---|---|
| **U0 (a)–(d)** | see "Verification performed" | holds |
| **S1, reading (I), of record** | 6,700 pure-WKB G windows on R1–R7, all matched across trees, tag agreeing in t5, s and the re-run's parents; true in 6,700. `sigma_t5` is exactly 0 in every one; `sigma_s` ≤ 8.8e-14 F (median 5.7e-16 F), at most 8.8e-8 of the bound | **holds** |
| (S1, reading (L), unscored) | 0 windows | (undecided) |
| **S2** | 12,862 T windows (6,071 T_q, 6,791 T_r), all matched; true in 10,747, false in 2,115. Every false window is at z ≥ 1.75e7 (133 in [1e6, 1e8), 1,089 in [1e8, 1e10), 875 in [1e10, 1e12), 18 above), `\|theta_T\|` 0.05–1.5e3 rad; there `sigma_s/sigma_t5` = 0.952–1.04, `sigma` 1.7e-12–1.7e-10 rad = 5–1.5e7 `eps\|theta\|`, largest `sigma_s`/bound 102. The true windows: `sigma_s/sigma_t5` median 3.8e-5 | **fails** |
| **S3** | 39 of 39; largest depth 11 (11 also on the 11 rows capped under B1); no unit over budget | **holds** |
| **S4** | 34 of 39, none undecided; false on `low-G-3` 10.8, `ctrl-low-1` 3.90, `ctrl-low-2` 3.57, `mid-G-2` 1.62, `R2` 1.49 (× `total_abserr_t5`; relative 4.0e-6, 2.5e-2, 5.4e-5, 5.7e-7, 1.7e-5) | **fails** |
| **S5** | 28 of 31, none excluded; false on `mid-G-3` 1.57, `mid-G-r-3` 1.32, `mid-G-1` 1.31; true on all 11 capped under B1, largest 0.823 | **fails** |

**Numerically seeded G windows (reported, not scored):** 3,594 windows (all at z ≥ 4.2e10);
`sigma_s` 5.5e-7–1.2e-3 F (median 2.1e-4 F) against `sigma_t5` 0–0.78 F (median 0.27 F); the window
RMS of the re-run `deltaTheta` alone is 2.8e-5 F at the median (56 windows above 1e-3 rad, from
`deltaTheta` jumps that are not whole cycles); `sigma_s`/that RMS median 7.8. Per row in
`out/cmd/39`.

**Descriptive:**
- `|total_s - total_t5|`: relative median 1.3e-6, largest 2.5e-2 (`ctrl-low-1`); against
  `total_abserr_t5` median 0.29, largest 10.8 (`low-G-3`); bit-identical on `ctrl-NULL-1`,
  `ctrl-NULL-2`. Three rows have an infinite `total_abserr_t5` (`low-G+r-2`, `low-G-r-1`, `low-Gqr-3`).
- Regions: 901 → 742 (s@1e-7), 1,053 (s@1e-8); depth-20 rows 859 → 702; controls 42 → 40; capped
  under B1 426 → 336 (558 at 1e-8); R1–R7 250 → 200 (332). 48 of 219 group calls change regions,
  depth or convergence; the group sequence (regime, label) is the same in every row. Per row and
  group in `out/cmd/39`.
- `total_converged`: unconverged t5 21, s@1e-7 20 (`mid-G-r-3` converges), s@1e-8 19; controls all
  converged.
- `theta_deriv` (log) at the window centres: T relative change ≤ 4.6e-14; G pure-WKB windows
  median 1.4e-16, max 7.4e-8; G seeded windows median 4.3e-4, p90 0.07, max 4.9 (`out/cmd/43`).
- `GkSourcePolicyData`: `Levin_z` changes on `low-G+r-1` only (`0x1.56a9ce9670b63p+46` →
  `0x1.4edc7c892d90fp+46`); type, quality, crossover unchanged on all 39.
- `convergence_kinds.py` under s (SHA-256 `49c83a72…`, `out/convergence_kinds_s.txt`): 1e-7
  resolution 14, bound only 32, phase-limited 23 (capped rows 9, 23, 12); 1e-8 20, 25, 32 (16, 15,
  20). 05a's t: 18, 32, 17 and 21, 24, 27.
- The re-run values against A3 v2's (39 s@1e-7 units; 3,960,030 G, 73,015 T values): 17 G cycle
  changes, each a single cycle with the new remainder within `4 eps|theta|` of 0 or `-2 pi`; no T
  cycle change; re-run time 1,443 s over the 39 s@1e-7 units (in `rebuild_s`, outside the budgets).

### What shipped

Nothing is committed. Under `var/levin-phase-floor-06/wider/`:
- **`scripts/`**: 05a's harness, changed as deviations 8–16 say; new `rerun.py`,
  `phase_probe06.py`, `u0.py`, `u0_clause_a.py`, `make_copy.py`, `certify_copy.py`,
  `fingerprint_copy.py`, `pure_wkb_zinit.py`, `explore_store.py`, `analyse06.py`,
  `characterise_s2.py`; unchanged and unused: `probe.py`, `analyse.py`, `t0.py`, `l0.py`,
  `draw_sample.py`, `rescore_t1.py`, `t1_true_step_only.py`; `convergence_kinds.py` byte-identical
  to `prompts/levin-phase-floor/measure/`. Diffs against 05a's copy:
  `out/scripts_diff_against_05a_at_launch.txt` and `…_final.txt`.
- **`out/`**: `commands.log` and `cmd/00`–`43` (every command with its output), `U0_a.txt`,
  `U0.txt`, `convergence_kinds_s.txt`, the scripts diffs.
- **`state/`**: `sample.json` (05's, `73aa7876…`), `predictions.md`,
  `predictions_as_first_written.md`, the two `prompt_*.raw`, `U0/`, `smoke/`, `dev/`,
  `scripts_05a_reference/`, `store_files_{before,at_stop,prelaunch,after,after_fingerprint}.txt`,
  `copy_files_{certified,prelaunch,after,after_fingerprint}.txt`, `analysis.json`,
  `probe_windows.json` (every matched window under both trees, with its samples, `z_init`s and
  `deltaTheta`s, for re-scoring S1 and S2), and **`a3v2_copy/`, the scratch copy (408 MB), kept for
  the user to delete**.
- The run directory holds the checkpoints and the unit pickles (86 MB).

### Deviations
1. **Both trees run from worktrees.** STRUCTURALLY REQUIRED.
2. **`prov.py` reads git with `--no-optional-locks`.** IMPLEMENTATION CHOICE. `out/cmd/01` ran a
   plain `git status` before it: UNINTENDED DRIFT (no write seen; worktrees clean throughout).
3. **`rec.sh`'s comments name 06.** IMPLEMENTATION CHOICE.
4. **A compound command with an `rm` was refused by a safety check**, ran nothing, was re-issued
   without it. UNINTENDED DRIFT.
5. **`explore_store.py`**, read-only sizing (`out/cmd/05`). IMPLEMENTATION CHOICE.
6. **Development runs before U0** (`out/cmd/06`, `07`, `17`; `state/dev/`). IMPLEMENTATION CHOICE.
7. **`state/scripts_05a_reference/`**, 05a's harness unchanged, to diff against. IMPLEMENTATION CHOICE.
8. **Route (B): `s` reads a certified scratch copy of A3 v2.** STRUCTURALLY REQUIRED (the user's
   decision). `make_copy.py` (`out/cmd/10`): five files copied byte for byte (SHA-256 equal), then
   `ALTER TABLE … ADD COLUMN theta_residual FLOAT` on the three value tables of each shard.
   `certify_copy.py`: 3,411,578 rows, every one identical by bit pattern, schema identical but for
   the added column, `theta_residual` NULL in every row. Its first run (`out/cmd/11`) said NOT
   CERTIFIED because of the script's own check of the `CREATE` sql (SQLite inserts the column before
   the constraints): UNINTENDED DRIFT, fixed, re-run `out/cmd/12` CERTIFIED. Fingerprint: `python -m
   runregistrykit store fingerprint` loads the main checkout's fingerprinter, whose factories cannot
   read the copy (`out/cmd/13`); `fingerprint_copy.py` calls the `s` tree's same fingerprinter object
   (`out/cmd/14` failed on the `taken` argument's type, my error: UNINTENDED DRIFT; `out/cmd/15`): **`0e32f24a…`,
   problems none**. IMPLEMENTATION CHOICE. `hash_store.py` takes an optional directory.
9. **`harness.py`: four more in-process stand-ins** (`WKB_phase_function.remote` at both producer
   modules, `assemble_GkSource_values.remote`, `apply_GkSource_policy.remote`). STRUCTURALLY REQUIRED.
10. **`rebuild.py`: `rerun=`, `use_store_dir`, the solver map and redshifts in the context, the
    policy summary.** STRUCTURALLY REQUIRED.
11. **`rerun.py`, new: how the producers are re-run.** IMPLEMENTATION CHOICE where §6 leaves the
    method open: integration rows through the factory with `_do_not_populate`, cross-checked against
    the row read by `sqlite3` `mode=ro`; `z_sample` from the stored value rows' redshifts, descending;
    each mode's T rows on its own shard (from `shard_keys`); read-backs formed as the factories form
    them (stored serials, values descending, JSON metadata); `deltaTheta` captured by wrapping
    `atan2` at the producer module for one `store()` call; every re-run value compared with the
    stored one. The `GkSourcePolicyData` is recomputed from the re-run `GkSource` under `s`.
12. **`phase_probe06.py`, new; 05a's `probe.py` copied unchanged and unused.** STRUCTURALLY REQUIRED.
13. **`worker.py`:** trees; `rerun`; a pass-through `sys.stdout` counter of `!! WARNING
    (GkSourcePolicyData` lines; more module evidence; `build_partition`'s arguments kept for the
    probe. Trees/rerun/probe STRUCTURALLY REQUIRED; counter and evidence IMPLEMENTATION CHOICE.
14. **`run_wider.py`, `launch.py`:** trees t5/s, unit sets, `rerun` on every `s` unit, the probe on
    R1–R7 at 1e-7 only, prompt `06`. STRUCTURALLY REQUIRED.
15. **The producer re-run is outside the budget** (in `rebuild_s`, as the stored reads are for t5);
    the budget covers the integral's compute, as in 05/05a. IMPLEMENTATION CHOICE.
16. **U0 by new scripts `u0_clause_a.py` and `u0.py`** (from 05a's `t0.py`); U0 units carry the
    probe; clause (c) run on both gate rows and extended to the whole chain. IMPLEMENTATION CHOICE.
    The first t5 U0 run (`out/cmd/20`) errored in `rerun.py` (T rows read from k's shard, not the
    mode's); fixed, re-run (`out/cmd/21`) into the same ledger, the check reading each unit's last
    record. UNINTENDED DRIFT.
17. **`predictions.md` gained a section after it was written, before the launch:** the pure-WKB
    tag read two ways. A3 v2 stores a pure-WKB object's `z_init` as its `z_source`, never null
    (9,552 of 9,552; `out/cmd/28`), so the prompt's "(`z_init` null)" tags no window. Reading (I),
    `z_init == z_source`, is scored pending the user's ruling; (L) is reported. Found in the smoke
    test, after R7's G window RMS ranges had been seen (no S1 number). The first version is kept as
    `state/predictions_as_first_written.md`. IMPLEMENTATION CHOICE, for the user.
    - **The user's ruling (relayed through the laptop session):** "Clearly (I)." Reading (I),
      `z_init == z_source` exactly, is S1's verdict of record; reading (L) stays reported beside
      it, unscored. The ruling was made while no S1 number existed. The question was relayed at
      22:40 UTC, when no `s` unit on R1–R7 had checkpointed. When the ruling reached the
      orchestrator at 22:44:56 UTC, the checkpoint held 49 units, none of them an `s` unit on R1–R7
      (the orchestrator's account). This agent read nothing from the run between the launch check
      and the ruling.
18. **A driver smoke test** (`ctrl-NULL-1`, `R7`, all sets; `state/smoke/`) after the predictions,
    before the launch. IMPLEMENTATION CHOICE.

19. **The analysis is a new `analyse06.py`** (05a's `analyse.py` copied unchanged and unused), with
    a new read-only `characterise_s2.py` and an inline read-only summary of `theta_deriv` by window
    tag (`out/cmd/43`). IMPLEMENTATION CHOICE.
20. **`convergence_kinds.py` was run from the main checkout with `-I`** from this directory's
    unchanged copy, `--tree s --capped-run <05's run>`, as 05a ran it; `out/convergence_kinds_s.txt`
    drops the per-pickle hash lines (full output `out/cmd/41`). IMPLEMENTATION CHOICE.

Deviations 2 (first command), 4, 8 (the first certification and the first fingerprint call) and
16 (the first t5 U0 run) are the UNINTENDED DRIFT; none touched a store, a tree or a record outside `wider/` and the run directory.

### Verification performed
- **The trees.** t5 at `9f66622` and s at `f5cee4c`, both `git status --short --ignored` empty at
  the start, at the stop, before the launch, on every checkpoint record and at the end; the main
  checkout at `28ebaf0`, status empty, and no tree ran from it. Every record carries `run_wider.py`'s
  SHA-256 `962e12c7…` and its tree's `HEAD` and status; every worker recorded its modules' files,
  all from its own tree, the producers' exact reduction absent under t5 and present under s, and
  which store it read. Ray was never initialised and no lock was taken.
- **U0.** (a) `out/U0_a.txt`. (b)–(d) `out/U0.txt`: (b) 16 + 16 named doubles, 68-float result
  summaries, region arrays, all bit-identical to 05a's units; (c) the clause's four columns and
  every coefficient bit-identical on 241,146 G and 5,172 T values, and the whole chain reproduces
  the stored-input unit bit for bit; (d) both rows' two `s` runs identical in every compared field,
  8,438 and 8,218 probe samples and 1,608 and 1,548 `deltaTheta`. The scripts the run used are the
  ones U0 ran on (`out/cmd/31`).
- **The sample** is `cmp`-identical to 05's, SHA-256 `73aa7876…`, checked before any row ran.
- **The run.** `done`, 109/109 in the planned order, exit 0, problems none, no unit over budget,
  no error, no probe error, no `!! WARNING (GkSourcePolicyData` line (counted per unit, and 0 by
  `grep` of `stdout.log`). `stderr.log` holds 15 scipy `IntegrationWarning`s, as 05 and 05a.
- **The copy** (route B): certified before use (rows, schema, NULLs, fingerprint `0e32f24a…`);
  its five files identical in SHA-256, size and mtime from certification to the end; re-fingerprinted
  after the run, `0e32f24a…`.
- **A3 v2.** Its six files' SHA-256, size and mtime identical before, at the stop, before the
  launch, after the run and after the fingerprint; no journal, `-wal` or `-shm`. `store fingerprint`
  (no `--write`): `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`, problems none,
  "matches the fingerprint taken … at b90752d".
- **05's and 05a's records were only read**: 05a's harness is `diff`-identical in `var/` and
  `docs/`, and its run directory holds 122 files.

### Observations not acted on
All in `issues.md`: W1 (no post-06 SHA can read a pre-06 store; route B), W2 (A3 v2's analytic
columns predate `1c19998`), W3 (§6's pure-WKB premise), W4 (S2 badly posed where `|theta_T|` is
small), W5 (S4's five rows), W6 (S5's three rows, `mid-G-1` new), W7 (more phase-limited calls),
W8 (seeded windows), W9 (re-run against stored), W10 (`Levin_z` on one row), W11 (scipy
warnings), W12 (`store fingerprint` and worktrees).

### State handed on
- **Nothing is running.** The run is `done`; no lock was taken; nothing was written in either
  worktree or the main checkout outside `var/levin-phase-floor-06/wider/` and the run directory.
- **A3 v2 is unchanged**, fingerprint `0e32f24a…`.
- **The scratch copy `state/a3v2_copy/` is kept**, for the user to delete once the records are
  committed. It need not be copied into `docs/`.
- **For the orchestrator:** copy `out/`, `scripts/`, `state/` (without `a3v2_copy/`; the U0, smoke
  and dev pickles may be left out), `log.md`, `verification.md`, `issues.md` to
  `docs/levin-phase-floor/06/wider/`; append this section to `logs/06-producers-store-the-residual.md`;
  put `verification.md` in `VERIFICATION.md` §6. Its links assume `06/wider/`.
- **For the user:** S2 fails as written in the windows at high z where `phi_T`'s own structure, not
  storage noise, sets `sigma` (W4); S4 and S5 fail as findings (W5, W6).
