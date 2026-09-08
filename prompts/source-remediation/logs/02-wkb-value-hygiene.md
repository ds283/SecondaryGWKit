# Log 02 — WKB value, policy and label hygiene (B1, B2, B3, B4, A6, B9, B10)

**Prompt:** prompts/source-remediation/02-wkb-value-hygiene.md
**Commit:** (this commit) — Fix WKB value, policy and label hygiene slips
**Model:** Claude Sonnet 5
**Date:** 2026-09-08
**Result:** COMPLETE WITH DEVIATIONS

## What shipped

- **B1** `ComputeTargets/TkWKBIntegration.py:632,636` — `analytic_T_w`/`analytic_Tprime_w`
  accessors returned `self._analytic_T_rad`/`self._analytic_Tprime_rad`. Changed to return
  `self._analytic_T_w`/`self._analytic_Tprime_w`, which the constructor already stores correctly.
- **B2** `ComputeTargets/GkWKBIntegration.py:583,588` — same slip; `analytic_G_w`/
  `analytic_Gprime_w` now return `self._analytic_G_w`/`self._analytic_Gprime_w` instead of the
  `_rad` members.
- **B3** `ComputeTargets/TkWKBIntegration.py:356` and `ComputeTargets/GkWKBIntegration.py:312`
  (pre-flight `WKB_criterion_init` warnings) — added `fabs(...)` around the numerator
  (`d_ln_omega_WKB_init` / `d_ln_omega_init`), matching every other evaluation of this criterion
  in the codebase.
- **B4** `ComputeTargets/TkWKBIntegration.py:115` and `ComputeTargets/GkWKBIntegration.py:81` —
  renamed `self._init_efolds_suph = None` to `self._init_efolds_subh = None` in the
  payload-is-`None` branch, matching the property, `store()`, and the deserialisation branch in
  both files.
- **A6** `ComputeTargets/GkSourcePolicyData.py:364` — `"WKB_minimal": numeric_clearance > 0.0`
  changed to `"WKB_minimal": WKB_clearance > 0.0`.
- **B9** `ComputeTargets/GkSourcePolicyData.py:662-671` — left the two theta-spline constructions
  as they are (single chunk in `_classify_Levin`, `chunk_logstep=125` in `_create_functions`) and
  added a comment at `:662` cross-referencing `_classify_Levin` (now `:170` area) explaining why
  they differ deliberately. See "Deviations" below for the reasoning.
- **B10** `ComputeTargets/QuadSource.py:304` — `ZSplineWrapper(..., "T_k", ...)` relabelled to
  `ZSplineWrapper(..., "quadratic source", ...)`.

No physical number changes in this commit. Stored diagnostic columns known-bad in existing
datastores (unaffected by this fix, since it only corrects code going forward): the
`analytic_T_w`/`analytic_Tprime_w` columns of every existing `TkWKBValue` row and the
`analytic_G_w`/`analytic_Gprime_w` columns of every existing `GkWKBValue` row (and, via
`GkSource.assemble_GkSource_values`, the corresponding `GkSourceValue` columns in the WKB-only
region) hold radiation-oracle values, not the `wPerturbations`-based oracle. `crossover_z`/
`Levin_z` for any `(k, z_response)` that fell through to the `WKB_minimal` band under the old,
buggy test may differ once recomputed (see A6 deviation below).

## Deviations from the prompt

### B9 — IMPLEMENTATION CHOICE: left the chunking inconsistency as-is

The prompt allows either leaving the two constructions as-is (with a cross-reference comment) or
making them consistent, provided a measurement shows the two choices for `Levin_z` differ by more
than one grid point on a synthetic phase.

I inspected `LiouvilleGreen/phase_spline.py::theta_deriv` rather than running the suggested
synthetic-phase script: `theta_deriv` builds a local scipy spline from only the points inside the
matched chunk, so the derivative used by the Levin threshold test already only ever samples the
data local to the query point. `phase_spline` splines are piecewise (via `make_interp_spline`
internally), not global polynomials, so a single-chunk spline built over the whole domain is not
materially less locally accurate than a chunked one away from chunk boundaries — the risk the
`_classify_Levin` comment (`:159-161`) is guarding against is specifically an artificial kink in
the derivative exactly at an automatically-chosen chunk boundary, which a single chunk removes by
construction. The `_create_functions` use (`:662` onward) is evaluated mod 2π after many
oscillation cycles, where the `theta_div_2pi` rebasing that chunking exists for (per the class
docstring) is the dominant precision concern, not the derivative.

Given the two call sites solve genuinely different problems (a smooth-derivative threshold test
vs. a many-cycle mod-2π evaluation), I judged the existing split deliberate and correct, and
picked "leave as-is + comment" over building and running the synthetic-phase falsification script
the prompt describes as the bar for merging them. **This is a judgement call, not a measurement**,
and a later reader who wants the numeric confirmation should build the `θ = -α(1+z)^β` fixture the
prompt describes and compare `Levin_z` between `chunk_logstep=None` and `chunk_logstep=125` before
disagreeing.

### A6 — IMPLEMENTATION CHOICE: no code change beyond the one-line fix; reachability left to prompt 12

The prompt says the audit could not determine whether any shipped configuration falls through the
first nine `CLASSIFICATION_BANDS` to reach `"WKB_minimal"`, and that prompt 12 measures this. No
deviation was needed here — the fix is exactly the one line the prompt specifies — but noting
explicitly that no attempt was made to check reachability in this prompt, consistent with the
campaign's per-prompt scope.

No other deviations. All seven items are a one-to-three-line change (B9's comment is the largest,
~6 lines), none changes a physical number, and none touches a file outside the prompt's list.

## Verification performed

- `PYTHONPATH=. ./venv/bin/python -c "import ComputeTargets"` — succeeded, no output (success).
- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s AdaptiveLevin/tests -t .` — `Ran 32
  tests in 0.042s`, `OK`.
- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t .` — `Ran 5
  tests in 0.015s`, `OK`.
- `grep -rn "_init_efolds_suph"` across the repository confirms the only remaining occurrences of
  that exact spelling are in `TkNumericIntegration.py` and `GkNumericIntegration.py`, which are a
  different, self-consistently-spelled attribute (`init_efolds_suph`, "e-folds to super-horizon")
  unrelated to the WKB files' `_init_efolds_subh` typo, and are out of this prompt's file list —
  left untouched.
- B1/B2: ran a short script constructing a `TkWKBValue` and a `GkWKBValue` with distinct `_rad`
  and `_w` constructor arguments and asserting the accessors return the `_w` values:
  ```
  TkWKBValue: analytic_T_w = 333.0 analytic_Tprime_w = 444.0
  GkWKBValue: analytic_G_w = 777.0 analytic_Gprime_w = 888.0
  B1/B2 verification PASSED
  ```
  (constructed with `analytic_T_rad=111.0, analytic_Tprime_rad=222.0, analytic_T_w=333.0,
  analytic_Tprime_w=444.0` and the analogous Gk values; script also asserts the `_rad` accessors
  still return the `_rad` values.)
- B3: reran the audit's `docs/spec-code-audit/scripts/TK_08_criterion_sign.py`, which prints
  `d ln omega_eff/dz` and the (no-`fabs`) criterion at representative sub-horizon points. Sample
  row at $w=1/3$, $k=10^4$, $z=1$: `d_ln_omega_dz = -1`, criterion (no fabs) `= -0.00069282`,
  confirming the sign is negative throughout the sub-horizon regime the audit measured, so the
  warning was unreachable before the `fabs` fix and is reachable (in principle, should the
  criterion exceed 1 in magnitude) after it.

## Observations not acted on

- None beyond what the prompt already flags (B9 reachability, A6 reachability — both explicitly
  deferred to prompt 12 by the campaign design).

## State handed to the next prompt

- No new names, protocols, or thresholds are introduced by this prompt. Prompt 03
  (`ComputeTargets/BackgroundModel.py`) is unaffected by anything here.
- Existing `TkWKBValue`/`GkWKBValue` rows retain their known-bad `analytic_*_w` columns (B1/B2 is
  a code-only fix, not a migration); anything reading those columns before a rebuild is still
  reading radiation values.
- A6 may change `crossover_z`/`Levin_z` for any `(k, z_response)` that reaches the `WKB_minimal`
  band; whether any shipped configuration does so is still open, per the prompt, and is measured
  in prompt 12.
