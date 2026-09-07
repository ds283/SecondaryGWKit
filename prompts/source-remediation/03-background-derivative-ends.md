# Prompt 03 — Remove the grid-end bias in `BackgroundModel._build_derivative` (A7)

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Audit:** §0.2 A7; `TK-report.md` TK-5 (measurement tables), §3 note 5
**Depends on:** nothing (ordering only: after 02)
**Recommended model:** Opus (the fix is small but has to be chosen by measurement, and the
persisted `BackgroundModel` value table must not change shape)
**Files you may touch:** `ComputeTargets/BackgroundModel.py`, new
`ComputeTargets/tests/__init__.py`, new `ComputeTargets/tests/test_background_derivatives.py`, plus
the log and the status board.

---

## Character of this commit

An accuracy fix to how spline-derived background derivatives are built, for cosmologies that do
not supply analytic derivatives (`LambdaCDM_GenericEOS`, `QCD_Cosmology`). `LambdaCDM` supplies
`d_lnH_dz`, `d2_lnH_dz2`, `d3_lnH_dz3`, `d_wPerturbations_dz`, `d2_wPerturbations_dz2`
analytically and is unaffected; that also makes it the reference for the measurement below.

**No schema change.** The `BackgroundModelValue` table keeps its columns; only the numbers in the
outermost few rows change.

## The defect

`ComputeTargets/BackgroundModel.py:120-149`, `_build_derivative`: when the cosmology lacks an
analytic method, fit a not-a-knot cubic `make_interp_spline` in $\log(1+z)$ through the sampled
values on the production grid, differentiate it, and divide by $1+z$. Second and third derivatives
are built by applying this to the *previous* derivative's samples, a stack up to three deep
(`d3_lnH_dz3`, `d2_wPerturbations_dz2`).

Audit TK-5 measured, on `main.py`'s actual grid ($z\in[0.1,10^{12}]$, 100 per decade, $n=1300$),
running the identical code path on a `LambdaCDM` background and comparing with its exact
derivatives:

| quantity | depth | $z=0.1$ (end) | 2nd | 3rd | interior median |
|---|---|---|---|---|---|
| ε | 1 | 1.3e-05 | 3.3e-06 | 8.1e-07 | 6.8e-13 |
| ε′ | 2 | 9.4e-04 | 2.0e-06 | 6.8e-05 | 1.4e-06 |
| ε″ | 3 | **3.0e-01** | 5.3e-02 | 1.3e-02 | 6.3e-05 |
| w″ | 2 | **3.8e-01** | 1.0e-03 | 2.6e-02 | 1.8e-08 |

Propagated into `Tk_omegaEff_sq` / `Tk_d_ln_omegaEff_dz` the impact is ≤9e-7 / ≤6e-5 and confined
to the outermost 2–3 points. So this is a bounded defect — fix it because the QCD models will be
the production case and because `_build_T_z_spline` (`LambdaCDM_GenericEOS.py:180-197`) already
pads its own grid by 5 % for exactly this reason.

## What to do

1. **Reproduce the measurement first.** `docs/spec-code-audit/scripts/TK_06_spline_end_bias.py`
   and `TK_07_omegaEff_spline_impact.py` are the audit's scripts; run them and confirm the table
   above at HEAD. Their `FakeModel`/grid construction is what your test should reuse.
2. **Choose the remedy by measurement.** Candidates, in the order to try:
   - **(a) Pad the fitting grid.** Extend the sample used *inside* `_build_derivative` beyond both
     ends of `z_sample` (e.g. 5 % in $\log(1+z)$ at each end, or a fixed number of extra
     log-spaced points), evaluate `f_to_diff` there, fit, differentiate, then evaluate only on the
     production grid. Works for `f_to_diff` callers directly. For `sample_to_diff` callers (the
     stacked derivatives) the padded samples have to come from the padded evaluation of the
     *previous* level, so the padding must be carried through the stack — restructure the helper to
     work on a padded grid throughout and truncate once at the end.
   - **(b) Differentiate the padded first-level spline analytically to higher order** instead of
     re-splining samples (`make_interp_spline(...).derivative(n)`), which avoids the stack
     entirely for `d2_lnH_dz2`/`d3_lnH_dz3`. Cubic splines have a discontinuous third derivative,
     so for the third level you may need a higher spline degree (`k=5`) or (a). Measure.
   - **(c) A finer private fitting grid** (e.g. 4× the production density) if (a)/(b) leave the
     end error more than 10× the interior.
   Pick the simplest one that brings every end-point error within **10× the interior median** for
   all five quantities; report the after-table in the same format. If none does, report the best
   and open a §3 issue.
3. **Preserve behaviour when analytic methods exist.** The `hasattr(cosmology, attr)` branch must
   be untouched; assert in the test that a `LambdaCDM` model gives bit-identical `BackgroundModel`
   values before and after.
4. **Test.** `ComputeTargets/tests/test_background_derivatives.py`: build the spline path on a
   `LambdaCDM` background (forcing the spline branch — e.g. wrap the cosmology in a small proxy
   that hides the analytic attributes) and assert the end-point relative errors against the exact
   derivatives are within the threshold you achieved, with the interior median also asserted. Keep
   the test under ~10 s.

## Verification

- Before/after tables (quantity × position) in the log, from the audit scripts and your test.
- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .` passes.
- State explicitly whether `BackgroundModel.compute()`'s runtime changed measurably (it runs once
  per model; a 2× slowdown is acceptable, 20× is not).

## Log and commit

Log to `logs/03-background-derivative-ends.md`; record the remedy chosen as an IMPLEMENTATION
CHOICE with the measured alternatives. Board: row 03, item A7. One commit; body gives the before
and after end-point errors for ε″ and w″ and names the affected cosmology classes.
