# The QCD reference fixture — what depends on it, and what would move it

**Campaign:** [`../../prompts/qcd-background-audit/README.md`](../../prompts/qcd-background-audit/README.md)
· **Prompt:** [`02-regenerable-qcd-references.md`](../../prompts/qcd-background-audit/02-regenerable-qcd-references.md)
**Regenerator:** [`generate_qcd_references.py`](generate_qcd_references.py)
**Verified on:** `qcd-background-audit` at the commit this prompt made (see the log for the SHA).

This is the map the next four prompts (04, 05, 06, 07) consume. It answers, for every test in the
tree that depends on a QCD background number: what it asserts, what it is scored against, and
**which kind of change would move it** — a `T(z)` node value (prompt 04), the representation's
shape (05, 06), the break-point set (07), or none of the three (the assertion is structural, or
scored against something the campaign does not touch).

---

## 1. The regenerator

`generate_qcd_references.py` rebuilds **only** `payload["models"]["QCDModel"]` of
`ComputeTargets/tests/wkb_reference_data.json`, reduced from
`docs/gktk-remedial/generate_references.py` (that file is untouched; this script imports its
support module `docs/gktk-remedial/reference_lib.py`, which is generic quadrature infrastructure
shared across campaigns, not campaign-specific state).

**Command:**

```bash
PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/generate_qcd_references.py --dry-run
PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/generate_qcd_references.py
```

**What "no change" means.** The comparison is restricted to `SCIENCE_KEYS` — `tau_minus_top`,
`cs_tau_minus_top`, `friction_F_minus_top`, `rho_G`, `rho_T`, `rho_anchor_z`,
`primitives_at_rho_anchor`, `short_baseline`, `reference_floor`, `grid`, `z_top`, `checkpoints` —
deliberately excluding `method` (free text, only ever appended to) and the two timing fields
(`build_time_seconds`, `generation_time_seconds`), which differ on every run whether or not the
representation moved. On this tree the script makes **no write at all** when nothing has moved —
not even to refresh a timestamp — so `git status --porcelain` on the JSON stays empty whether or
not `--dry-run` was passed. That is deliberate: a prompt that regenerates the fixture without
moving the representation must leave no diff.

**Reproduction, measured on this tree:**

```
$ PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/generate_qcd_references.py --dry-run
   QCDModel: compute_background + splines took 0.604 s
   reference quadrature took 111.683 s
** QCD block science-content comparison against the shipped file:
  tau_minus_top: unchanged
  cs_tau_minus_top: unchanged
  friction_F_minus_top: unchanged
  rho_G: unchanged
  rho_T: unchanged
  rho_anchor_z: unchanged
  primitives_at_rho_anchor: unchanged
  short_baseline: unchanged
  reference_floor: unchanged
  grid: unchanged
  z_top: unchanged
  checkpoints: unchanged
No change: the QCD block's science content (12 keys) is bit-identical to the shipped file.
Total wall time 112.297 s (build 0.604 s, reference quadrature 111.683 s).
```

`git status --porcelain ComputeTargets/tests/wkb_reference_data.json` is empty before and after.
Exit code `0` (dry-run reports "no change" as `0`, not the ≥1 the prompt's §2 item 2 describes for
a *mismatch* — see the log's deviation note: the prompt anticipated only two outcomes, reproduces
or does not, and did not specify the exit code for the "reproduces" case; `0` was chosen because
nothing is wrong).

**A second, uncovered generator.** The JSON also carries a top-level `convergence` block, written
by a **different** script, `docs/gktk-remedial/residual_convergence.py` (`GkTk-remedial` prompt
02, not this campaign's). It holds QCD-specific numbers this script does **not** touch:
`convergence.geometry.QCDModel` (node/interval counts, `branch_boundaries`, `cs2_transition`,
`T_spline_knots_in_range`) and `convergence.models.QCDModel.<scheme>.<quantity>` (per-Gauss-order
convergence figures, `json_vs_reference_max_rel`, `decision.N_tau`/`N_cs_tau`/`N_F`/`N_rho`,
`decision.rho_adaptive_fallback_required`). Five tests below read this block directly (marked
"JSON: convergence block" in the table). **This is squarely what prompt 08 exists to re-take** —
its charter is exactly "re-derive `integration_break_points`... and re-measure the per-sector
break-point policy," which is what `residual_convergence.py` computes — but it is worth recording
explicitly here: **prompts 04–07 must not assume this script regenerates the `convergence` block.**
It does not, and neither the acceptance test above nor `--dry-run` says anything about it.

## 2. The two floor issues, re-measured

Both are **re-measured, not closed**, here — quoted from the `convergence` block this prompt's
generator does not itself write, since they are already the output of `residual_convergence.py`
and this prompt changes no number:

- **`[02-qcd-reference-floor]`.** The QCD `tau`/`cs_tau` references are themselves accurate only
  to the JSON's own quadrature floor: `convergence.models.QCDModel["branch+knots"]["tau"]["json_vs_reference_max_rel"]
  = 1.8785409360313507e-14` and the `cs_tau` equivalent `= 1.8866533733829712e-14` (both at Gauss
  order 8/12; the friction floor is `0.0` at order 8). `test_qcd_nodes_against_adaptive_reference`
  and `test_qcd_checkpoints` (below) assert `<= 3 * floor`, i.e. `<= 5.636e-14` / `5.660e-14`, read
  from this same JSON field at test time rather than hardcoded — so a future re-run of
  `residual_convergence.py` that moves the floor moves the threshold with it automatically.
- **`[03-qcd-short-baseline-reference-endpoint-rounding]`.** Not a field in either JSON block: it
  is the ulp(u)/W_baseline rounding of the *rounded* `log1p(z)` endpoints the short-baseline
  references integrate between, ~1e-13 relative on the 37% fractions near z=1e6
  (`test_background_tau.py`'s own docstring; GkTk-remedial log 03 measured the shipped table
  agreeing with an exact-endpoint `quad` to ≤8.8e-16). This script's own short-baseline block
  (`short_baseline`, computed the same way as the shipped one) reproduces the shipped values
  exactly on this tree (see §1's "unchanged"), so it carries the identical rounding and the figure
  is unchanged by this prompt. `test_qcd_short_baselines` asserts README §6's absolute `1e-13`
  threshold, not the JSON's internal self-agreement, per the existing issue note.

## 3. Verified file list

```bash
grep -rln --include="*.py" "QCD\|qcd" ComputeTargets/tests/ CosmologyModels/tests/ LiouvilleGreen/tests/
```

returns, on this tree: the eleven `ComputeTargets/tests/` modules the prompt names
(`test_background_tau.py`, `test_background_cs_tau_friction.py`, `test_background_derivatives.py`,
`test_gk_wkb_phase.py`, `test_tk_wkb_phase.py`, `test_phase_residual.py`,
`test_residual_table_reuse.py`, `test_omega_eff_split.py`, `test_numeric_break_points.py`,
`test_numeric_break_point_key.py`, `test_wkb_reference.py`) plus `wkb_reference.py` itself (the
harness, not a test module) and, in `CosmologyModels/tests/`, prompt 01's two new modules
(`T_z_reference.py`, `test_T_z_representation.py`). `LiouvilleGreen/tests/` has no hit. The list
matches the prompt's; no module is missing and none was invented.

`test_background_derivatives.py` mentions "QCD_Cosmology" once, in its module docstring, but every
fixture it builds is a `LambdaCDM` proxy — **zero of its five tests touch a QCD number.** It is
listed for completeness and carries no row below.

## 4. The map

One row per test method that asserts something depending on a QCD background number (a value
computed through `QCD_Cosmology`/`QCDModel`, however indirectly). Tests that build a `QCDModel`
only as one of several models in a **self-consistency** check (the same quantity computed two
ways from the *same* model instance, cancelling per README §0.2) are included and marked
accordingly — they are exactly the kind of test the campaign's §0.2 says proves nothing about the
representation's accuracy, but they still exercise QCD code paths and are worth knowing about.

Legend for "scored against": **JSON** = `wkb_reference_data.json` `models.QCDModel` block (this
prompt's regenerator); **JSON(conv)** = the separate `convergence` block (§1); **closed form** =
a value derived algebraically/physically, not cached; **self** = another quantity computed from
the same model in the same test (common-mode, README §0.2). "Moved by" lists every one of
{**node** = `T(z)` node value (prompt 04), **shape** = representation shape (05/06), **breaks** =
break-point set (07), **none**} that applies.

### `ComputeTargets/tests/test_background_tau.py`

| Test | Asserts | Tolerance | Scored against | Moved by |
|---|---|---|---|---|
| `test_payload_shape` | payload key lengths/types for both LambdaCDM and QCD payloads | exact/structural | self | none |
| `test_tau_is_a_callable_returning_a_float` | `tau` monotonic, `TablePrimitive`, sign of `.delta` | structural | self | none |
| `test_tau_init_is_the_radiation_era_closed_form` | `tau(z_init)` equals the closed-form radiation-era IC | exact equality | closed form | node, shape (mild: `rho(z_init)` at $z\sim2\times10^{16}$ is deep radiation-domination, but not exempt from the T(z) spline) |
| `test_qcd_nodes_against_adaptive_reference` | `tau.delta(z_top,z)` at checkpoints | `<= 3 * floor`, floor read live from `convergence["branch+knots"]["tau"]["json_vs_reference_max_rel"] = 1.879e-14` (threshold 5.636e-14) | JSON + JSON(conv) | node, shape, breaks |
| `test_qcd_short_baselines` | one-interval and 37%-fraction `tau.delta` | `<= 1e-13` | JSON | node, shape, breaks |
| `test_qcd_break_points` | `len(integration_break_points) == T_spline_knots_in_range + len(branch_boundaries)` (**today: ~407**), break points strictly increasing and in range, break points align with `convergence.geometry.QCDModel.branch_boundaries` to `1e-9` | exact count + `1e-9` alignment | closed form (`integration_break_points`) + JSON(conv) geometry | **breaks** — this is G1 itself. **This assertion's expected count must be rewritten by prompt 07**: once `BREAK_POINT_ALL` collapses to 3, `T_spline_knots_in_range` goes to 0 and `expected` becomes 3, not ~407. Flagged for prompt 07, not fixed here (out of this prompt's scope). |
| `test_persisted_pair_round_trips_exactly_in_Mpc_units` | `Mpc` round-trip, `tau_lo` below ulp of `tau` | structural | self | none |
| `test_reconstruction_from_values_is_bit_for_bit` | rebuilt table == built table, on- and off-grid, `break_points` included | exact equality (self-consistency) | self | none (immune by construction; the *break_points array itself* used inside is `breaks`-sensitive, but the test's own pass/fail is not) |

`test_build_cost` and `test_delta_throughput` print QCD numbers but assert only on the LambdaCDM
side; `test_oracle_improvement_note` is LambdaCDM-only. Not rows.

### `ComputeTargets/tests/test_background_cs_tau_friction.py`

| Test | Asserts | Tolerance | Scored against | Moved by |
|---|---|---|---|---|
| `test_payload_shape` | `cs_tau`/`friction_F` payload shapes for both models | structural | self | none |
| `test_qcd_checkpoints` | `cs_tau` checkpoint error; `friction_F` checkpoint error | `cs_tau <= 3 * floor` (floor `1.887e-14`, threshold `5.660e-14`); `friction <= FRICTION_REL_TOL = 1e-13` | JSON + JSON(conv) | node, shape, breaks |
| `test_qcd_short_baselines_including_the_transitions` | `cs_tau`/`friction_F` across JSON short baselines **and** the production intervals holding the two Hubble branch boundaries, the `EOS_T_LO` kink and the steepest $c_s^2$ transition (indices read from `convergence.geometry.QCDModel`) | `cs_tau <= 1e-13`; `friction <= 2e-14` absolute | JSON + JSON(conv) geometry | node, shape, breaks (the interval *indices* themselves are `convergence`-block state, §1) |
| `test_persisted_values_round_trip_exactly_in_Mpc_units` | `Mpc` round-trip; `cs_tau_lo` below ulp | structural | self | none |
| `test_reconstruction_from_values` | rebuilt `cs_tau`/`friction_F` tables == built tables | exact / near-ulp | self | none |

`test_build_cost` prints only. `test_model_functions_still_constructs_with_thirteen_positional_arguments`
and `test_background_model_value_keeps_constructing` build generic stand-ins with no cosmology at
all — not QCD-dependent despite living in this file. `TestFrictionODEComparison` is LambdaCDM-only.

### `ComputeTargets/tests/test_gk_wkb_phase.py`

| Test | Asserts | Tolerance | Scored against | Moved by |
|---|---|---|---|---|
| `TestRealBackground.test_qcd_k_3e8` | Green's-function phase at JSON checkpoints from the 3-e-fold anchor, $k=3\times10^8$ | `<= 5e-3` rad (`QCD_TOL`) | JSON (`tau_minus_top`, `rho_G`, `rho_anchor_z`) | node, shape, breaks |

Every other class in this 1,166-line file (`TestRadiationControl`, `TestOffGridAnchor`,
`TestStoreAlgebra`, `TestCrossObjectConsistency`, `TestPayloadContract`, `TestCost`,
`TestApplyPhaseOffset`, `TestSourceHygiene`) is built on `RadiationModel`/`LambdaCDMModel` stand-ins
only; verified by reading each `setUpClass` and the one other literal "QCD" hit
(`test_residual_is_exactly_zero_in_radiation`, which is a radiation-only test whose name coincides
with a grep hit elsewhere in its docstring).

### `ComputeTargets/tests/test_tk_wkb_phase.py`

| Test | Asserts | Tolerance | Scored against | Moved by |
|---|---|---|---|---|
| `TestRealBackground.test_qcd_k_3e8` | transfer-function phase and friction at JSON checkpoints, $k=3\times10^8$ | phase `<= 5e-3` rad (`QCD_PHASE_TOL`); friction `<= 1e-12` relative (`FRICTION_REL_TOL`) | JSON (`tau_minus_top`→`cs_tau`-equivalent, `rho_T`, `rho_anchor_z`) | node, shape, breaks |

Same structure as the Gk file: every other class is Radiation/LambdaCDM-only.

### `ComputeTargets/tests/test_phase_residual.py`

| Test | Asserts | Tolerance | Scored against | Moved by |
|---|---|---|---|---|
| `TestAgainstReferences.test_rho_at_the_reference_checkpoints` | `rho_G`/`rho_T` at JSON checkpoints, both production models, all 3 k | `<= 1e-7` rad absolute (`RHO_REFERENCE_ABS_TOL`) | JSON | node, shape, breaks (QCD rows only; LambdaCDM rows in the same test are unaffected) |
| `TestSizeSanity.test_rho_G_magnitudes` | `\|rho_G\|` on QCD at $k=3\times10^8$ inside a physically-motivated window | `5e-4 <= \|rho_G\| <= 3e-3` | closed-form magnitude window | node, shape (wide margin; low risk) |
| `TestSizeSanity.test_rho_T_magnitudes` | `rho_T` on both models, all k, inside a window | `-0.12 <= rho_T <= -0.06` | closed-form magnitude window | node, shape (wide margin) |
| `TestCost.test_integrand_evaluations_and_wall_time` | QCD table evaluations vs. LambdaCDM's baseline | `baseline < evals <= 1.30 * baseline` (`COST_BREAK_POINT_FACTOR`) | self (cost, not accuracy) | **breaks** — this margin is measured on today's ~404-knot break-point set; expect it to change substantially (most likely improve, i.e. get closer to `baseline`) once G1 lands. Re-check in prompt 07/08. |
| `TestGuards.test_no_adaptive_fallback_was_required` | `RHO_GAUSS_ORDER == convergence.decision.N_rho`; `not convergence.decision.rho_adaptive_fallback_required` | exact | JSON(conv) `decision` | shape, breaks — this is precisely what prompt 08 re-measures |

### `ComputeTargets/tests/test_residual_table_reuse.py`

| Test | Asserts | Tolerance | Scored against | Moved by |
|---|---|---|---|---|
| `TestKeyDiscriminates.test_different_models_do_not_share_through_the_store_id` | `len(qcd_table) != len(lambdacdm_table)` (different store ids give different cached tables) | structural (`assertNotEqual`) | self | breaks (node count differs because QCD declares break points and LambdaCDM declares none; still true after G1 collapses to 3, low risk) |
| `TestAnchoringIsFree.test_phase_is_unchanged_on_all_three_models` | shared-table phase == per-object-table phase, Gk sector, on Radiation/LambdaCDM/QCD | `<= 1e-9` rad (`ANCHORING_TOL`) | self (README §0.2 common-mode) | **none** — by construction this cancels; exactly the trap the campaign exists to see past |
| `TestAnchoringIsFree.test_phase_is_unchanged_in_the_transfer_function_sector` | as above, Tk sector | `<= 1e-9` rad | self | none |
| `TestAmortisedCost.test_qcd` | amortised residual-table evaluations per object | `< build_evaluations / 10` | self (cost) | breaks (build evaluation count is break-point-set sensitive) |

`TestRadiationControlSurvives.test_rho_G_is_bit_exactly_zero_through_the_shared_table` is
radiation-only despite living in a file full of QCD references elsewhere; not a row.

### `ComputeTargets/tests/test_omega_eff_split.py`

| Test | Asserts | Tolerance | Scored against | Moved by |
|---|---|---|---|---|
| `test_Gk_omegaEff_sq_is_bit_identical_to_the_pre_refactor_expression` | two algebraic forms of `Gk_omegaEff_sq` agree, for `model in (Radiation, LambdaCDM, QCD)` | exact (`assertEqual`) | self | none |
| `test_Tk_omegaEff_sq_is_bit_identical_to_the_pre_refactor_expression` | as above, `Tk_omegaEff_sq` | exact | self | none |
| `test_Gk_split_sums_to_the_return_value` | `leading + correction == Gk_omegaEff_sq` to re-association rounding | `<= REASSOCIATION_ULPS` | self | none |
| `test_Tk_split_sums_to_the_return_value` | as above, Tk | ulp-bounded | self | none |
| `test_Gk_correction_does_not_depend_on_k` | correction term is k-independent | exact | self | none |
| `test_Tk_correction_does_not_depend_on_k` | as above, Tk | exact | self | none |

All six build `QCDModel(grid)` as one of three models in `_Shared.models`, but every assertion
compares two ways of computing the *same* quantity from the *same* model instance — textbook
common-mode cancellation (README §0.2). None can see a representation change.

### `ComputeTargets/tests/test_numeric_break_points.py`

| Test | Asserts | Tolerance | Scored against | Moved by |
|---|---|---|---|---|
| `TestDeclaration.test_qcd_discontinuities_are_a_subset_of_the_breaks` | `discontinuity_temperatures_GeV ⊆ break_temperatures_GeV`, strictly fewer | structural | closed form (`QCD_EOS`) | none (equation of state, out of campaign scope, §0.5) |
| `TestDeclaration.test_qcd_declares_exactly_the_temperatures_at_which_it_jumps` | measured G/Gs/w jumps == declared discontinuities == `{T_LO, T_120_MEV, T_HI}` | `JUMP_THRESHOLD` | closed form (`QCD_EOS`) | none |
| `TestDeclaration.test_hubble_jumps_at_the_declared_crossings_and_not_at_the_kink` | `H(z)` steps by the expected relative size at `T_LO`/`T_120_MEV`, is continuous at `EOS_T_LO` | `delta=0.05` relative match; `< 1e-8` for the non-jump | closed form (`Hubble` either side of the crossing) | shape (weakly: a segmented representation must still reproduce these jump sizes exactly, audit §2 — a good post-06 regression check) |
| `TestDeclaration.test_kind_selects_knots_or_jumps` | `len(BREAK_POINT_DISCONTINUITY) == 2`; **`len(BREAK_POINT_ALL) > 100`** | exact / `> 100` | closed form (`_cosmology_break_points`) | **breaks — WILL FAIL after prompt 07.** `len(every) > 100` hard-codes today's ~404-knot artefact; once G1 collapses `BREAK_POINT_ALL` to the 3 genuine crossings this assertion is false on its face and prompt 07 must rewrite it (from `> 100` to `== 3`, or equivalent). Flagged here, not fixed. |
| `TestQCDReferenceConvergence.test_the_branch_crossing_is_inside_the_range` | exactly one declared discontinuity inside the k=4.97e7 test range, at the `T_120_MEV` crossing | `places=6` | closed form | none (EOS crossing location, not T(z) representation) |
| `TestQCDReferenceConvergence.test_split_converges_where_unsplit_does_not` | unsplit reference-convergence drift `> 1e-6`; split drift `< ACCEPTANCE_DRIFT = 3.4e-8` | as stated | self (tightened-vs-reference internal convergence, on real `QCD_Cosmology`) | **node, shape, breaks — this *is* `GkTk-remedial` prompt 19's measurement, re-taken by this campaign's prompt 08.** Today's figure (board: 8.72e-09 worst of 50 k, with the 404 knots) is explicitly flagged in README §2 (f) as having been taken against a defective background; prompt 08 must re-take it post-07. |

`TestSmoothCosmologiesAreUnchanged`, `TestSplitAtDeclaredJump`, `TestStopModeAcrossSegments`,
`TestPerSectorPolicy`, `TestManyBreakPoints` all use synthetic fixtures (`_JumpCosmology`,
`_ManyBreakCosmology`, etc.), not `QCD_Cosmology`/`QCDModel` — not rows. **One observation**:
`TestManyBreakPoints.test_boundaries_closer_than_the_standoff_collapse`'s docstring states, as
fact, "the T(z) spline's knots are 2.85e-02 apart in log(1+z), and the closest a declared
temperature crossing comes to a knot is 3.4e-03" — true of today's representation, and it will
become a stale description once prompt 07 removes the knots. Not asserted (it is prose, not a
threshold), so it does not fail, but it is worth a line in prompt 07's log alongside
`[19-cosmologymodels-docstrings-predate-per-sector-policy]`, which is the same class of drift.

### `ComputeTargets/tests/test_numeric_break_point_key.py`

No row: every QCD mention is in a docstring or in `TestOneDeclarationPerSector`/`TestTheKeyField`/
`TestOldSchemaFailsLoudly`, all of which compile SQL `WHERE` clauses or read `ast` against
`TkNumericIntegration.BREAK_POINT_KIND`/`GkNumericIntegration.BREAK_POINT_KIND` — schema wiring,
not a QCD background number. Relevant to §7 D5 (prompt 08): if `BREAK_POINT_KIND` changes, these
tests' fixtures do not need new QCD numbers, but the campaign should re-read this file before
prompt 08 touches the constant.

### `ComputeTargets/tests/test_wkb_reference.py`

| Test | Asserts | Tolerance | Scored against | Moved by |
|---|---|---|---|---|
| `TestReferenceJSON.test_json_is_complete` | schema shape (keys, lengths, checkpoint spread, sign conventions) for `model_key in MODEL_KEYS` including `"QCDModel"` | structural | JSON (schema only) | none — the schema is unchanged by any of prompts 04–07 |
| `TestStandInModels.test_epsilon_in_the_radiation_era` | `\|epsilon_QCD(1e10) - 2\| <= QCD_EPSILON_TOLERANCE` | `6e-2` | closed-form physical expectation | node, shape (mild; margin is generous relative to the ~1e-8 scale of T1) |

`test_radiation_closed_forms`, `test_baselines_block`, `test_reference_module_does_not_import_mpmath`,
`test_production_grid_shape`, `test_radiation_model_is_self_consistent`, and everything in
`TestErrorDefinitions` do not touch a QCD number (the grid-shape test uses only the
LambdaCDM-derived `z_init`/grid, even though `self.qcd` is built alongside it in `setUpClass`).

### `ComputeTargets/tests/test_background_derivatives.py`

No rows (§3).

### `CosmologyModels/tests/test_T_z_representation.py` (prompt 01's guard)

All seven tests are scored against `CosmologyModels/tests/T_z_reference.py` (`accurate_T`,
`jump_locations`, `entropy_factor` — the rtol=1e-14 defining-equation oracle, never the JSON and
never the shipped `_solve_T_z`). They are listed as one group because prompt 01's own log is
their record; repeating its thresholds here would risk disagreeing with it.

| Test | Moved by |
|---|---|
| `test_T_z_matches_the_defining_equation` | node |
| `test_the_node_solve_converges` | node |
| `test_conformal_time_matches_the_exact_background` | **node, shape, breaks — this is T1's guard, the whole point of the campaign** (README §0.2, §5 note 1). Measures `3.4605051e-08` today; prompt 06 tightens its threshold from `4.0e-8` to `1e-15`. |
| `test_the_branch_joins_are_where_the_fixture_puts_them` | none (pins the EOS fixture itself, §0.5) |
| `test_T_z_is_a_step_at_the_lowest_crossing` | shape (this is the fact prompt 06 must reproduce, not remove) |
| `test_a_segment_edge_bisected_and_one_root_found_disagree` | shape (pins the bisection-vs-root-find trap, README §2 (b)) |
| `test_a_constant_gs_equation_of_state_is_an_exact_ramp` | shape (the `PureRadiationEOS` exactness check, README §2 (g)) |

## 5. Tally

45 test methods found to depend on a QCD background number (structural or accuracy), across the
eleven `ComputeTargets/tests/` modules named in the prompt plus `CosmologyModels/tests/test_T_z_representation.py`.
A test can appear in more than one "moved by" bucket.

Counts below are **rows touching the category at all**, so a row that (for example) is scored
against both the JSON and its `convergence` block counts once in each — the four "scored against"
counts therefore do not sum to 45, but every row is counted in at least one of them (checked
directly against the per-file tables above), and likewise for the four "moved by" counts.
"Closed form" includes the seven `CosmologyModels/tests/test_T_z_representation.py` tests scored
against `T_z_reference.py`'s `rtol=1e-14` root-solved oracle — an independent reference, not a
literal algebraic formula, but neither the JSON fixture nor a same-model self-comparison.

| Category | Count | Note |
|---|---|---|
| Scored against `wkb_reference_data.json` `models.QCDModel` (JSON) | 8 | `test_qcd_nodes_against_adaptive_reference`, `test_qcd_short_baselines`, `test_qcd_checkpoints`, `test_qcd_short_baselines_including_the_transitions`, `TestRealBackground.test_qcd_k_3e8` (×2, Gk and Tk), `test_rho_at_the_reference_checkpoints`, `test_json_is_complete` |
| Scored against the separate `convergence` block (JSON(conv), §1) | 5 | `test_qcd_nodes_against_adaptive_reference`, `test_qcd_checkpoints`, `test_qcd_short_baselines_including_the_transitions`, `test_qcd_break_points`, `test_no_adaptive_fallback_was_required` |
| Scored against a closed form (incl. the `T_z_reference.py` oracle) | 17 | `test_tau_init_is_the_radiation_era_closed_form`, `test_qcd_break_points` (partly), `test_rho_G_magnitudes`, `test_rho_T_magnitudes`, `test_epsilon_in_the_radiation_era`, five tests in `test_numeric_break_points.py` (`test_qcd_discontinuities_are_a_subset_of_the_breaks`, `test_qcd_declares_exactly_the_temperatures_at_which_it_jumps`, `test_hubble_jumps_at_the_declared_crossings_and_not_at_the_kink`, `test_kind_selects_knots_or_jumps`, `test_the_branch_crossing_is_inside_the_range`), and all seven `test_T_z_representation.py` tests |
| Scored against self (common-mode / structural) | 19 | payload shape, round-trip, reconstruction, `test_omega_eff_split.py`'s six, `TestAnchoringIsFree`'s two, cost/count tests, `TestKeyDiscriminates`, `test_split_converges_where_unsplit_does_not` |
| **Moved by node** (prompt 04) | 15 | `test_tau_init_is_the_radiation_era_closed_form`, `test_qcd_nodes_against_adaptive_reference`, `test_qcd_short_baselines`, `test_qcd_checkpoints`, `test_qcd_short_baselines_including_the_transitions`, `TestRealBackground.test_qcd_k_3e8` (×2), `test_rho_at_the_reference_checkpoints`, `test_rho_G_magnitudes`, `test_rho_T_magnitudes`, `test_split_converges_where_unsplit_does_not`, `test_epsilon_in_the_radiation_era`, `test_T_z_matches_the_defining_equation`, `test_the_node_solve_converges`, `test_conformal_time_matches_the_exact_background` |
| **Moved by shape** (05/06) | 18 | the 15 node-moved tests above except `test_T_z_matches_the_defining_equation`/`test_the_node_solve_converges` (node-only), plus `test_no_adaptive_fallback_was_required`, `test_hubble_jumps_at_the_declared_crossings_and_not_at_the_kink`, `test_T_z_is_a_step_at_the_lowest_crossing`, `test_a_segment_edge_bisected_and_one_root_found_disagree`, `test_a_constant_gs_equation_of_state_is_an_exact_ramp` |
| **Moved by breaks** (07) | 15 | `test_qcd_nodes_against_adaptive_reference`, `test_qcd_short_baselines`, `test_qcd_break_points`, `test_qcd_checkpoints`, `test_qcd_short_baselines_including_the_transitions`, `TestRealBackground.test_qcd_k_3e8` (×2), `test_rho_at_the_reference_checkpoints`, `test_integrand_evaluations_and_wall_time`, `test_no_adaptive_fallback_was_required`, `test_different_models_do_not_share_through_the_store_id`, `test_qcd` (amortised cost), `test_kind_selects_knots_or_jumps` (**will need editing**), `test_split_converges_where_unsplit_does_not` (**prompt 08's re-measurement**), `test_conformal_time_matches_the_exact_background` |
| **Moved by none** | 20 | structural/self-consistency tests and the untouched-EOS-declaration tests |

**The two assertions that need editing, not merely re-measuring, once their prompt lands:**

1. `test_numeric_break_points.py::TestDeclaration::test_kind_selects_knots_or_jumps` —
   `self.assertGreater(len(every), 100)` is true only because of the artefact G1 removes; prompt
   07 must change it (to `== 3`, or whatever the collapsed count is measured to be).
2. `test_background_tau.py::TestBackgroundTau::test_qcd_break_points` — its `expected` count is
   `T_spline_knots_in_range + len(branch_boundaries)`, i.e. ~407 today; once `T_spline_knots_in_range`
   goes to 0 (prompt 07), `expected` becomes just `len(branch_boundaries)` — the same code
   continues to work *if* `convergence.geometry.QCDModel.T_spline_knots_in_range` is re-measured to
   `0` by whatever re-runs `residual_convergence.py`, but if that block is not regenerated in step,
   this test reads a stale `T_spline_knots_in_range` and gets the wrong `expected`. **Prompt 07
   must either regenerate the `convergence` block or note explicitly that this test's `expected`
   needs a different source.**
