# Log 03 — Remove the grid-end bias in `BackgroundModel._build_derivative` (A7)

**Prompt:** prompts/source-remediation/03-background-derivative-ends.md
**Commit:** *(SHA intentionally not embedded — the campaign convention since prompt 01
deviation 4: writing it and amending changes it again)* — "Remove the grid-end bias in the
background derivative splines"
**Model:** Claude Opus 5
**Date:** 2026-09-08
**Result:** COMPLETE WITH DEVIATIONS

## What shipped

### `ComputeTargets/BackgroundModel.py`

**New module-level constants (`:24-37`)** describing the private fitting grid:
`DERIVATIVE_FIT_PAD_POINTS = 12`, `DERIVATIVE_FIT_REFINE = 3`,
`DERIVATIVE_FIT_PAD_FLOOR = 0.9`, `DERIVATIVE_FIT_PAD_FRACTION = 0.05`,
`DERIVATIVE_SPLINE_ORDER = 5`.

**New module-level helper `_build_derivative_fit_grid(z_sample)` (`:40-96`).** Builds, in
`x = log(1+z)`:

- an ascending copy of the production abscissae plus the inverse permutation back to `z_sample`
  order (`z_sample` is held descending);
- `DERIVATIVE_FIT_REFINE - 1` extra points inside each production interval (so the production
  points remain an exact subset of the fit grid, no assumption of uniform spacing);
- `DERIVATIVE_FIT_PAD_POINTS` extra points beyond each end at the local fit-grid spacing. The
  low-end spacing is clamped so that `1+z` never falls below `DERIVATIVE_FIT_PAD_FLOOR*(1+z_min)`
  (with `z_min >= 0` this keeps the grid at `z >= -0.1`, inside the `z >= -0.19` range over which
  `LambdaCDM_GenericEOS._build_T_z_spline` builds its `T(z)` spline), and both ends are capped so
  the padding never extends the fitted range by more than `DERIVATIVE_FIT_PAD_FRACTION`. Neither
  cap binds on the shipped grid; they only protect a very coarse or very short grid.
- the production `x` and `z` values are written back exactly at the selected indices, so no
  `expm1(log1p(z))` round trip perturbs them.

On `main.py`'s grid (z ∈ [0.1, 1e12], 100/decade, n = 1300) the fit grid is 3921 points spanning
z ∈ [0.01058, 1.0885e12]; the low-end padding stays at positive z and neither cap binds.

**`compute_background._build_derivative` (`:193-257` after the change, was `:120-149`).**

| before | after |
|---|---|
| `sample_to_diff=` keyword, samples on the production grid | `fit_sample_to_diff=` keyword, samples on the padded fit grid |
| analytic branch `[getattr(cosmology, attr)(z.z) for z in z_sample]` | analytic branch evaluated on the fit grid (values at the production points are the same calls at the same z) |
| `make_interp_spline(x_data, y_data)` — cubic, not-a-knot, on the production grid | `make_interp_spline(fit_x, y_data, k=fit_k)` — quintic (`k=5`, falling back to 3 on a grid of fewer than 6 points), on the padded/refined grid |
| derivative evaluated through a `ZSplineWrapper(..., deriv=True)`, one point at a time | derivative evaluated on the whole fit grid and divided by `1+z` directly |
| returns production samples | returns fit-grid samples; a new `_truncate()` selects the production points and restores `z_sample` order |

The five call sites (`:241-257`) now thread fit-grid arrays through the stack and truncate once at
the end. `d_wPerturbations_dz` is now built with `f_to_diff=cosmology.wPerturbations` rather than
`sample_to_diff=wPerturbations_sample` (see deviation 2).

`ZSplineWrapper` is still imported and used by `BackgroundModel.functions._build_func`; only
`_build_derivative`'s use of it is gone.

**No schema change.** `BackgroundModelValue` keeps its columns; the returned dictionary keys, their
lengths and their ordering are unchanged.

### `ComputeTargets/tests/__init__.py`, `ComputeTargets/tests/test_background_derivatives.py` (new)

Five `unittest` cases, no Ray and no datastore (`compute_background._function` is driven directly and
`redshift`/`redshift_array` are plain objects). A `_HideAnalyticDerivatives` proxy hides the five
analytic derivative methods of a Planck2018 `LambdaCDM` so the spline branch is exercised with an
exact reference available — the pattern audit script `TK_06_spline_end_bias.py` uses.

## Deviations from the prompt

### 1. The remedy is (a) + (b) + (c) together, not the first of them that works — IMPLEMENTATION CHOICE

The prompt asks for candidates in the order (a) pad the fitting grid, (b) differentiate a
higher-order spline analytically instead of stacking, (c) a finer private grid, and says to pick the
simplest that brings every end-point error within 10× the interior median. Measured on the audit's
grid, against `LambdaCDM`'s exact derivatives (all numbers are relative error at the z=0.1 end / the
interior median):

| variant | eps | eps′ | eps″ | w′ | w″ | worst ratio |
|---|---|---|---|---|---|---|
| baseline (cubic, no pad) | 1.3e-05/6.8e-13 | 9.4e-04/1.4e-06 | 3.0e-01/6.3e-05 | 1.7e-06/1.0e-09 | 3.8e-01/1.8e-08 | 2.1e+07 |
| (a) alone, pad 20, cubic | 2.6e-08/6.8e-13 | 1.3e-07/1.4e-06 | 9.7e-07/6.1e-05 | 1.1e-09/1.0e-09 | 1.7e-08/1.8e-08 | 3.8e+04 (eps) |
| (b) alone, one k=5 spline, `.derivative(n)`, no pad | 4.9e-08/7.1e-14 | 5.9e-06/2.0e-08 | 2.9e-03/3.6e-07 | 4.4e-10/2.2e-12 | 1.6e-04/7.7e-09 | 8.1e+03 |
| (a)+(b), pad 16, k=5, stacked, no refine | 1.4e-11/7.0e-14 | 5.9e-11/4.2e-09 | 3.0e-08/2.4e-07 | 2.7e-14/2.2e-12 | 2.2e-09/2.5e-10 | 2.0e+02 (eps) |
| **(a)+(b)+(c), pad 12, refine 3, k=5, stacked** | **2.7e-13/1.1e-13** | **5.7e-11/4.1e-08** | **2.8e-08/7.7e-06** | **6.8e-15/4.6e-12** | **1.8e-09/2.1e-09** | **2.4** |

Neither (a) nor (b) alone meets the criterion. (a) leaves ε (stack depth 1) 4 orders above the
interior median: padding removes the *end* bias but the cubic's own interpolation error at low z
remains, and the interior median is set by the radiation era where `ln H` is exactly linear in
`log(1+z)` and the spline is exact. (b) alone removes the stacking amplification but not the end
condition. Refinement (c) was then needed to bring ε down to the interior level. Since the three are
cheap and orthogonal, all three are applied.

Two further choices inside the design:

- **Stacked quintics, not `.derivative(n)` of one quintic.** Both were measured. Taking the 2nd/3rd
  derivative of a single k=5 spline is worse for w″ (4.1e-07 at the end, 1.2e-07 interior) than
  re-splining each level (1.8e-09 / 2.1e-09), and it would have required rewriting the chain rule
  for `d²/dz²`, `d³/dz³` in the helper. Stacking keeps the existing structure and the existing
  meaning of each level.
- **pad = 12, refine = 3.** A scan over (pad, refine, floor) on the shipped grid gave a worst ratio
  of 1.3–7.7 for every combination with k=5, pad ≥ 8 and refine ≥ 2; the ratios at that level are
  float64 round-off and move by a factor of a few with the exact grid endpoints (see "Observations").
  pad = 12, refine = 3 was chosen because it is the cheapest configuration whose worst ratio (2.4)
  has a factor-4 margin *and* whose padding stays at positive z on the shipped grid, so it does not
  rely on any cosmology being evaluable in the future (z < 0).

### 2. `d_wPerturbations_dz` is now built from `cosmology.wPerturbations` rather than from `wPerturbations_sample` — STRUCTURALLY REQUIRED

The prompt's option (a) notes that for `sample_to_diff` callers "the padded samples have to come
from the padded evaluation of the *previous* level". For the four stacked derivatives that is what
happens. `d_wPerturbations_dz` is the exception: its "previous level" is `wPerturbations_sample`,
which is computed on the *production* grid for storage and cannot be extended. Since
`wPerturbations` is a callable on the cosmology, the call site was changed to `f_to_diff=` — which
evaluates exactly the same function at exactly the same production points, plus the padded and
refined ones. No stored value changes as a result.

### 3. The 10× criterion is asserted at the low-z end only; the high-z end is asserted in absolute error — STRUCTURALLY REQUIRED

The prompt's table and the audit's finding are about the z = 0.1 end. At the z = 1e12 end the
background is radiation dominated, so ε → 2 exactly and ε′, ε″, w′, w″ → 0; the *reference values*
there are 1.7e-21 (ε′), 3.4e-33 (ε″), 1.1e-21 (w′) and 2.3e-33 (w″), and a relative error against
them is meaningless — the audit's own TK-5 table carries footnote † saying exactly this. The
absolute error at that end improved anyway (ε′ 3.5e-18 → 9.2e-23, ε″ 4.5e-28 → 4.3e-32,
w″ 4.4e-37 → 1.7e-36 i.e. unchanged; w′ 1.2e-27 → 8.6e-27, five times larger but 6 orders below its
own reference value). The test therefore asserts the 10× ratio at the low-z end and absolute
thresholds at the high-z end.

### 4. An extra test asserting absolute low-z thresholds — IMPLEMENTATION CHOICE

The prompt asks for a test asserting the achieved end-point relative errors and the interior median.
Because the surviving residual at the low-z end is float64 round-off in the differentiated spline,
the *ratio* statistic is not stable to better than a factor of a few. The ratio assertion (10×) is
kept as the prompt specifies, and a second case asserts absolute relative-error caps ~30× above the
measured values, so a real regression is caught even if the ratio statistic drifts on another
scipy/numpy build. Both are far below the pre-fix numbers.

### 5. The commit SHA is not embedded in the log or the board — IMPLEMENTATION CHOICE

README §5.1's template has a `**Commit:** <sha>` field, but a log that is part of the commit it
names cannot carry that commit's own SHA: writing it and amending changes the SHA again. Prompt 01's
log recorded the same problem (its deviation 4) and prompt 02 wrote `(this commit)`. This log
follows that convention and identifies the commit by its subject line. The orchestrator can read the
SHA from `git log`.

## Verification performed

All commands run from the repository root with `./venv/bin/python`.

### Reproduction at HEAD (prompt step 1)

`PYTHONPATH=. ./venv/bin/python docs/spec-code-audit/scripts/TK_06_spline_end_bias.py` reproduces
the audit's TK-5 table exactly (grid n = 1300, z ∈ [0.1, 1e12]):

```
    quantity stack depth   z=0.1 (end)     2nd pt     3rd pt     median  z=1e12 (end)
     epsilon           1     1.251e-05  3.251e-06  8.133e-07  6.778e-13     3.797e-13
    d_eps_dz           2     9.397e-04  2.034e-06  6.833e-05  1.416e-06     2.051e+03
  d2_eps_dz2           3     3.005e-01  5.293e-02  1.277e-02  6.333e-05     1.319e+05
          w'           1     1.738e-06  4.573e-07  1.186e-07  1.043e-09     1.079e-06
         w''           2     3.775e-01  1.041e-03  2.598e-02  1.821e-08     5.850e-05
```

`TK_07_omegaEff_spline_impact.py` likewise reproduces its table (ω_eff² 2.1e-07 and
d ln ω_eff/dz 2.6e-04 at z = 0.1 for k = 1; 8.7e-07 / 5.7e-05 at z = 1e12).

### After the fix

Same grid, same quantities, but driving the *real* `compute_background` code path with the analytic
methods hidden (`scratch/after_table.py`, reproduced by `ComputeTargets/tests/`):

| quantity | depth | z=0.1 (end) | 2nd | 3rd | interior median | z=1e12 (end) |
|---|---|---|---|---|---|---|
| ε | 1 | **2.71e-13** | 3.31e-13 | 5.03e-13 | 1.13e-13 | 2.34e-14 |
| ε′ | 2 | **5.75e-11** | 2.02e-11 | 4.85e-12 | 4.08e-08 | (ref→0)† |
| ε″ | 3 | **2.76e-08** | 4.36e-08 | 2.45e-08 | 7.69e-06 | (ref→0)† |
| w′ | 1 | **6.79e-15** | 3.16e-14 | 2.91e-15 | 4.57e-12 | (ref→0)† |
| w″ | 2 | **1.75e-09** | 3.08e-09 | 5.27e-09 | 2.10e-09 | (ref→0)† |

end/interior ratios: ε 2.4, ε′ 0.0014, ε″ 0.0036, w′ 0.0015, w″ 0.83 — every one inside the
prompt's 10× acceptance. Improvement at the z = 0.1 end: ε ×4.6e+07, ε′ ×1.6e+07, ε″ ×1.1e+07,
w′ ×2.6e+08, w″ ×2.2e+08. The interior improved too (ε″ median 6.3e-05 → 7.7e-06, w″ 1.8e-08 →
2.1e-09), and the worst absolute error anywhere on the grid fell from 6.6e-02 to 1.6e-08 for ε″ and
from 2.2e-08 to 1.1e-15 for w″.

† at the high-z end, absolute errors (reference value in brackets): ε′ 9.2e-23 (1.7e-21), ε″
4.3e-32 (3.4e-33), w′ 8.6e-27 (1.1e-21), w″ 1.7e-36 (2.3e-33). Before the fix: 3.5e-18, 4.5e-28,
1.2e-27, 4.4e-37.

### Propagation into the WKB inputs (TK_07 rebuilt on the new code path)

| z | ω_eff² rel diff (was) | d ln ω_eff/dz rel diff (was) |
|---|---|---|
| 0.1 (low-z end) | 1.3e-14 (2.1e-07) | 1.0e-11 (2.6e-04) |
| 0.12 (2nd) | 4.5e-15 (6.8e-10) | 1.7e-11 (1.4e-06) |
| 1079 (interior) | 0.0 (2.4e-13) | 5.3e-15 (1.1e-10) |
| 1e12 (high-z end) | 2.3e-11 (8.7e-07) | 5.3e-09 (5.7e-05) |

worst over the probe set: ω_eff² 3.8e-11 (was 8.7e-07), d ln ω_eff/dz 2.0e-08 (was 5.7e-05).

### Behaviour of the analytic branch

`test_analytic_branch_is_untouched` asserts, element by element with `assertEqual`, that all five
derivative sample arrays returned for a plain `LambdaCDM` are **bit-identical** to a direct call of
its analytic methods on `z_sample`. It passes. This is why `_build_derivative_fit_grid` writes the
production `z` values back into the fit grid exactly rather than leaving `expm1(log1p(z))`.

### Test suite

```
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .
Ran 5 tests in 0.188s
OK
```

Also re-run and green, to confirm nothing else moved: `AdaptiveLevin/tests` (32 tests, OK) and
`CosmologyModels/tests` (5 tests, OK). `LiouvilleGreen/tests` takes many minutes and contains no
reference to `BackgroundModel` (`grep -rl BackgroundModel LiouvilleGreen/` is empty), so it was
left running rather than waited on; it cannot be affected by this change, which touches only
`ComputeTargets/BackgroundModel.py`.

### Runtime

`compute_background` on the shipped 1300-point grid, best of 5, after warm-up:

| cosmology | branch | before | after | ratio |
|---|---|---|---|---|
| `LambdaCDM` (Planck2018) | analytic | 14.1 ms | 31.0 ms | 2.2× |
| `QCD_Cosmology` (Planck2018, max_z=1e13) | spline | 88.4 ms | 121.8 ms | 1.4× |

Within the prompt's stated budget ("a 2× slowdown is acceptable"); the 2.2× on the analytic branch
is because that branch is now evaluated on the padded/refined grid too (3921 points rather than
1300), which is required whenever a cosmology supplies *some* but not all of the five methods. The
absolute cost is 17 ms, once per cosmology, against a background integration that itself takes tens
of ms and a pipeline that runs for hours.

### Not done here

The audit's §4 item 4 — the same measurement on a real `LambdaCDM_GenericEOS`/QCD run rather than on
a `LambdaCDM` driven through the spline branch — still needs a pipeline run and belongs to prompt 12.
No `GenericEOS` model has an independent closed form to check against, which is why the test uses the
proxy.

## Observations not acted on

1. **The end/interior ratio at this accuracy is round-off, not bias.** With the fix in place the
   low-z end residuals for ε (2.7e-13) and w″ (1.8e-09) are at the level of float64 round-off in a
   differentiated spline. Two grids differing only in whether the top point is `1e12` or `1+1e12`
   gave worst ratios of 2.4 and 6.6 respectively. The prompt's 10× criterion is met on the shipped
   grid but should not be read as a tight bound; the absolute-threshold test exists for this reason.
2. **A coarse grid loses part of the benefit.** At 50 samples per decade (n = 650) the low-end
   padding hits the `DERIVATIVE_FIT_PAD_FLOOR = 0.9` clamp — 12 points at the natural spacing would
   reach z = −0.28, outside the `T(z)` spline range of the GenericEOS models — so the effective
   padding is only ~2.5 production spacings and the end/interior ratios rise to 329 (ε) and 73 (w″).
   The absolute errors are still 3–7 orders better than the pre-fix code (ε″ 5.97e-08 at the end).
   `main.py` ships 100/decade, where the clamp does not bind. Nothing done; noted in §3 of the board.
3. **`_build_func` (`BackgroundModel.functions`) re-splines the stored samples with a plain cubic
   and no padding** (`:387-404`). It builds interpolants of already-stored values, not derivatives,
   so it does not suffer the stacking amplification; but the consumers of `ModelFunctions` evaluate
   through it, so the outermost grid interval still carries an ordinary cubic end-condition error on
   top of the (now much smaller) sample error. Out of scope for this prompt.
4. **`ZSplineWrapper`'s out-of-bounds message is hardcoded to `GkSource.function:`**
   (`spline_wrappers.py:40,52`), so before this change a range error from `_build_derivative` would
   have read `GkSource.function: evaluated d_lnH_dz out of bounds`. That is audit B10's problem in a
   different file; prompt 02 fixed the `QuadSource` label but the message prefix itself is still
   wrong for every user of the class. Not touched.

## State handed to the next prompt

- **Names.** `ComputeTargets/tests/` now exists (with `__init__.py`); prompt 05 adds to it. The new
  public-ish symbols in `BackgroundModel` are `_build_derivative_fit_grid` and the five
  `DERIVATIVE_FIT_*` / `DERIVATIVE_SPLINE_ORDER` constants; a test can monkey-patch the constants on
  the module object (note that `import ComputeTargets.BackgroundModel as X` binds the *class*,
  because `ComputeTargets/__init__.py` re-exports it — use
  `importlib.import_module("ComputeTargets.BackgroundModel")`).
- **Driving `compute_background` without Ray.** `compute_background._function` is the undecorated
  body, and `redshift(store_id=i, z=...)` / `redshift_array([...])` need no datastore. Prompts 05–08
  can build background payloads the same way.
- **Accuracy now available to consumers.** For a cosmology with no analytic derivatives, ε, ε′, ε″,
  w′, w″ are good to ≲3e-08 relative anywhere on a 100-per-decade grid including its ends (was
  3.8e-01 at the end), and `Tk_omegaEff_sq` / `Tk_d_ln_omegaEff_dz` to 3.8e-11 / 2.0e-08 (was
  8.7e-07 / 5.7e-05). Prompt 05's `TkSourceFunctions` differentiates the LG amplitude using
  `d_ln_omegaEff_dz` and ε from `ModelFunctions`; those inputs are no longer the limiting error at
  the grid ends.
- **No schema or datastore consequence.** Stored `BackgroundModel` rows built with a `LambdaCDM`
  cosmology are unchanged. Rows built with a `LambdaCDM_GenericEOS`/QCD cosmology change in the
  outermost few redshifts of the five derivative columns — but those rows are already being rebuilt
  because of prompt 01.
