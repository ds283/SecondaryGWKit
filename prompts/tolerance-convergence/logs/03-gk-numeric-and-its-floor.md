# Log 03 — `GkNumericIntegration`, and the floor that decides whether it matters

**Prompt:** `prompts/tolerance-convergence/03-gk-numeric-and-its-floor.md`
**Commit:** *(this prompt's own commit)* — "Sweep the Gk numeric tolerance matrix against its floor"
**Model:** Opus 5
**Date:** 2026-09-17
**Result:** COMPLETE WITH DEVIATIONS — `GkNumericIntegration` is characterised over the version-2
production grids on all three models in **both** axes, at all fifty production wavenumbers, with the
reference converged **50 / 50 on each** and the radiation column calibrated against
`compute_analytic_G` to a **ratio of 1.00**. **README §2 (d)'s prior is confirmed in this sector**:
four decades of `atol` move the maximum by ≤1.2 %, four decades of `rtol` by ×13,300, and the
corners agree to three significant figures. **The consumer-spline floor is freshly measured and
dominates** — 1.6e-04 to 1.9e-04 of the envelope three e-folds inside the horizon and up to
**9.35e-03** at four, against 2.6e-07 for the solver — so **the target is `unchanged`** (§6.1
rule 4), and D1's compute question in this sector is "spend nothing", which is the reverse of what
README §7 D1 expected.

**Two of the prompt's own premises did not hold, and both are handed back rather than worked
around.** §2.2's claim that the outermost $z_{\rm source}$ is the least favourable is **false** at
all nine probes (by ≤×1.45), so the sweep characterises the sector rather than bounding it — that
is §2.2's and §9's stop, and deviation 8 says why the sweep was nevertheless completed. §5's claim
that the consumer splines the **response** grid is **false**: it splines `z_source` on the
**source** grid, twelve times finer (deviation 1). Two §3 issues opened; nothing repaired.

## What shipped

Three files, **none of them production code**.

### `docs/tolerance-convergence/gk_numeric_sweep.py` — new, the generator

Run from the repository root; emits the tables of `GK-NUMERIC-SWEEP.md` on stdout and a progress
log on stderr. Public entry point `main()`; the pieces a later prompt may want are
`Subject` (model + cosmology + version-2 grid + response grid + object count),
`sweep_subject`, `z_source_bound_check`, `floor_probe`, `measure_floor_window`,
`floor_for_subject`, `phase_per_interval`, `predicted_floor_profile` and `object_count`.

### `docs/tolerance-convergence/GK-NUMERIC-SWEEP.md` — new, the measurement document

Every table in it is that script's stdout, pasted verbatim under §§2–9, with the prose of §0, §1,
§10 and §11 around it (acceptance 6).

Four *strings* in the script were corrected after the 1,820 s run rather than the run repeated: the
section headings were promoted from `###` to `##` so that the generated blocks sit under the
document's own numbering, the §8 caption's "~15 rad, five times past the Nyquist limit" became
"several times the Nyquist limit" (the measured figures are 7.0 and 13.2 rad, not 15), the runtime
footer's probe count was corrected from 32 to the ladder's 96, and §6 gained the closing paragraph
that states the stop. The pasted text carries the corrected strings, so the document and a re-run
agree; **no table, and no number in any table, is affected by any of the four**, and the script is
what produces them.

### `ComputeTargets/tests/convergence_reference.py` — **additive only**

One new function, `gk_geometry_at_source(cosmology, k_inv_Mpc, grid, z_source)`, inserted between
`gk_geometry` and `_numeric_run`. It is `gk_geometry` with the source redshift supplied by the
caller instead of fixed at five e-folds outside the horizon, which is what §2.2's check of the
`z_source` bound needs. **No existing line of that module changed** (prompt §2.3).

### Board, index, log

`prompts/tolerance-convergence/IMPLEMENTATION_STATE.md` — row 03, item **T4**, two new §3 issues;
`docs/OPEN_ISSUES.md` — two rows added, count and date corrected. Same commit (`CLAUDE.md`).

## Deviations from the prompt

### 1. The consumer's spline runs over `z_source`, not over the response grid — **STRUCTURALLY REQUIRED**

Prompt §5 says the floor "is a property of **the response grid's density**" and directs the
measurement to "sample $G$ on the response grid as the policy does". **The tree says otherwise.**
`GkSource` is keyed on `z_response`, and its `z_sample` is
`z_source_sample.truncate(z_response, keep="higher")` (`main.py:2288-2291`) — the **source** grid
above that response redshift. `GkSourcePolicyData._create_functions` then builds
`make_interp_spline` over `log(1 + v.z_source.z)` for the `has_numeric` values in that sample
(`:667-678`), so the spline's knots are **source** redshifts and its spacing is the source grid's.
The rest of §5's description — `numeric_smallest_z` to `z_sample.max`, `MIN_SPLINE_DATA_POINTS = 5`,
`ZSplineWrapper(..., log_z=True)` — matches exactly.

The difference is not cosmetic. The response grid is `PRODUCTION_RESPONSE_SPARSENESS = 12` times
coarser, so a cubic interpolation error quoted on it is wrong by about $12^4$; §8 of the document
measures the gap in the only currency that still means something there (radians of $G$'s
oscillation per lattice interval), because on the response lattice the samples alias and no
interpolation error is defined at all.

**What was done:** the floor was measured on the axis the code uses, and the response lattice is
reported beside it as §8, labelled as not being a floor. Nothing was "reconciled by adjusting the
method until the figures agree" (§5's own prohibition).

### 2. The floor probe calls `numeric_with_phase_cut._function` directly — **STRUCTURALLY REQUIRED**

`G(z_source, z_response)` at one response redshift for many source redshifts is one
`GkNumericIntegration` solve per source redshift, each read at a single sample. `gk_run` cannot do
it: `_numeric_run` passes `mode="stop"` unconditionally, and in that mode
`numeric_with_phase_cut` raises at `:710` unless the integration terminates on the phase-cut
event — right for a production work item, wrong for a two-point probe that ends above the search
window. The probe therefore calls the undecorated `_function` with `mode=None` and every other
argument the production call site passes (`GkNumericIntegration.compute`, including this sector's
`BREAK_POINT_DISCONTINUITY`), reusing `convergence_reference`'s own `_Proxy` / `_KExit` stand-ins,
`UNITS` and `PRODUCTION_DELTA_LOGZ`. Fixing this inside the facility would have meant changing an
existing line of `_numeric_run`, which §2.3 forbids.

### 3. The candidate `rtol` axis stops at `1e-10`, not at SciPy's clamp — **IMPLEMENTATION CHOICE**

The reference has to be tighter than every candidate, and SciPy clamps `rtol` at
`SCIPY_RTOL_FLOOR = 2.22e-14`, so the campaign's reference pair is `(1e-18, 1e-12)` with `1e-13`
as the only effective tightening left (`GkTk-remedial` prompt 17; `convergence_reference`'s own
comment). A candidate at `rtol = 1e-11` or tighter would sit within one decade of the reference's
own `rtol` and could not be scored against it. The axis therefore runs `1e-6 … 1e-10`, **two
decades either side of production**, which is the span §4 asks for on the `atol` axis and the same
span here. The alternative — quoting a `rtol = 1e-12` candidate against a reference at the same
`rtol` — measures nothing, and §9 makes that a stop rather than a caveat.

### 4. Cells at or below ten times the reference's own drift are reported as bounds — **IMPLEMENTATION CHOICE**

`reference_drift` requires the smallest difference the measurement *intends to report* and
evaluates `drift <= that / 10` itself. Taken as "the smallest number anywhere in the tables" that
criterion is failed on `QCDModel` by the tightest few matrix cells, whose error is genuinely of
the same order as the reference's drift — which is a statement about the reference's resolution,
not a discovery about the candidates. So each `(model, k)` marks a candidate **resolved** when its
maximum exceeds `CRITERION_RATIO x drift`, and the criterion is restated against the smallest
resolved cell; the unresolved cells are counted, their settings named, and they are quoted as
bounds. If no cell at a wavenumber were resolved the script raises, which is §9's stop.

The alternative considered was to leave the matrix as it was and record every `(model, k)` on
`QCDModel` as "not converged", which is true of nothing that matters: the **production** setting —
the only cell any conclusion here rests on — stands three to four orders above the drift at every
wavenumber on every model, and that factor is reported in its own column of §3 precisely so the
reader can check it rather than take the verdict on trust.

### 5. The floor is a profile, not a number — **IMPLEMENTATION CHOICE**

§5 asks for "the floor … per model and over the fifty wavenumbers, with the same
maximum/second/median treatment as §4". A single number would have been misleading: $G$'s
dependence on $z_{\rm source}$ oscillates at $\mathrm{d}\theta/\mathrm{d}u = k(1+z)/H$, which runs
from $e^4$ at the bottom of the numeric region to well under one outside the horizon, so the
interpolation error falls by five orders across the region the consumer splines. It is reported as
a ladder of six positions in e-folds inside the horizon, with max and median over the fifty
wavenumbers at each rung, plus the worst rung per model in the §4 shape. The maximum over the
ladder is the headline; the rung three e-folds inside the horizon is the one comparable with
review §10.1's inherited figure.

### 6. `RadiationModel`'s anchor had to be chosen — **IMPLEMENTATION CHOICE**

Prompt 02a named the two *production* anchors, `PRODUCTION_Z_INIT_LAMBDACDM` and
`PRODUCTION_Z_INIT_QCD`. `RadiationModel` is a control and has no anchor in the record. It is
given one built exactly as those two are — the outermost source redshift of the earliest-exiting
wavenumber, `horizon_exit_z(model, 3e8/Mpc, -5)` — so that the control is measured on its own
version-2 construction rather than on LambdaCDM's `z_init`, which on an exact-radiation background
sits five orders above where any production wavenumber needs the grid to start. The value and the
resulting grid's digest are in §2.1 of the document, as board standing note 18 requires.

### 7. The response grid is winnowed without `protect` — **IMPLEMENTATION CHOICE**

`main.py:1073` winnows the source grid with `protect=z_protected_sample`, the declared break and
feature redshifts, so that they survive into the response grid. `BuiltSourceGrid` does not carry
the protected list, and adding it would have meant changing an existing line of
`convergence_reference.py`. The response lattice is used here for exactly two things — choosing the
response redshift the floor is probed at (the nearest node to five e-folds inside the horizon) and
the leading-order spacing profile of §8 — and neither is sensitive to a handful of extra nodes at
the declared crossings. The matrix of §5, which is the measurement, does not use it at all: it goes
through `gk_geometry`, which winnows inside the facility exactly as prompt 01 left it.

### 8. The sweep was completed although §2.2's premise failed — **IMPLEMENTATION CHOICE, and it is a stop handed back**

§2.2 says that if the outermost source redshift is not the least favourable, the prompt should
"stop and say so **before** completing the sweep". It is not, and this log and the document say so
in those words, at the top of both. The sweep was nevertheless completed, for three reasons stated
here so that the user can overrule them:

1. **The failure is of the word "bound", not of the design.** The maximum envelope-relative error
   is *flat* in $z_{\rm source}$ — it has no trend at all — and the largest excess of any interior
   source redshift over the outermost is a factor of 1.45. What varies monotonically is the median
   and the evaluation count, both of which the document reports.
2. **Nothing the sweep concludes turns on a factor of 1.5.** The floor dominates the production
   solver error by between two and four orders depending on where in the numeric region it is read.
3. **Stopping with no measurement would have left the campaign with neither the matrix nor the
   floor**, and §2.2's own alternative — widening the sweep over the second axis — is forbidden in
   the same paragraph ("do not quietly widen the sweep to compensate"). It was not widened.

The premise is recorded as `[03-outermost-z-source-is-not-the-least-favourable]` on the board and
in `docs/OPEN_ISSUES.md`, and the hand-back asks the user whether a bound over the
$(k, z_{\rm source})$ plane is wanted — which is a much larger measurement — or whether the
characterisation is enough and §2.2's and `gk_geometry`'s wording should say "representative".

## Verification performed

All figures below are on the **version-2** source grid at **each cosmology's own** production
anchor (README §5 rule 6; board standing note 18), under `BREAK_POINT_DISCONTINUITY`, at all fifty
production wavenumbers on all three models, envelope-relative. The full tables are
[`docs/tolerance-convergence/GK-NUMERIC-SWEEP.md`](../../../docs/tolerance-convergence/GK-NUMERIC-SWEEP.md);
the run took 1,820 s.

### The grids, reproduced

| model | anchor `z_init` | samples | digest | matches |
|---|---|---|---|---|
| RadiationModel | `44523947729.772957` | 2306 | `3bef2c06` | *new* — the control has no anchor in the record (deviation 6) |
| LambdaCDMModel | `20636395964161516` | **1778** | **`60a3205a`** | the published LambdaCDM grid |
| QCDModel | `33003344446051300` | **2034** | **`21ffc126`** | prompt 02a's QCD-at-its-own-anchor grid |

### The drift verdict, per model (acceptance 1)

| model | worst drift | at k | median | not converged | least headroom | production cell above the drift | cells at or below it |
|---|---|---|---|---|---|---|---|
| RadiationModel | 2.06e-11 | 4.972e+07 | 1.69e-11 | **0 / 50** | 10.7 | **×1.1e+04** | 0 / 650 |
| LambdaCDMModel | 2.08e-11 | 4.972e+07 | 1.33e-11 | **0 / 50** | 9.53 | **×8.8e+03** | 0 / 650 |
| QCDModel | 2.98e-10 | 3.092e+06 | 3.66e-11 | **0 / 50** | 1.04 | **×399** | 10 / 650 |

Reference `(1e-18, 1e-12)`, tightened to `(1e-19, 1e-13)`; `rtol_step_is_effective` holds for both
(SciPy's clamp is 2.220446049250313e-14). The ten `QCDModel` cells at or below the drift are the
three tightest settings, `(1e-12, 1e-10)`, `(1e-10, 1e-10)` and `(1e-08, 1e-10)`, and are quoted as
bounds (deviation 4).

### The radiation anchor calibration (prompt §3)

`compute_analytic_G` through `radiation_anchors` / `anchor_error`, beside the self-convergence
figure at the same setting:

| setting | self-convergence, max over grid | vs exact G | ratio |
|---|---|---|---|
| `(1e-10, 1e-06)` | 3.21e-05 | 3.21e-05 | **1.00** |
| `(1e-10, 1e-08)` | 2.60e-07 | 2.60e-07 | **1.00** |
| `(1e-10, 1e-10)` | 2.42e-09 | 2.44e-09 | 1.01 |
| `(1e-12, 1e-08)` | 2.61e-07 | 2.61e-07 | **1.00** |

The reference itself sits **2.28e-11** of the envelope from the exact `G` at its worst wavenumber,
which is the floor under the second column. **The self-convergence machinery is measuring the
same thing the oracle is**, to the third significant figure, at every setting — the calibration
prompt 17 did not have.

### `atol` is inert; `rtol` is the lever (prompt §6 question 1)

Maximum envelope-relative error over the fifty wavenumbers:

| axis | span | Radiation | LambdaCDM | QCD |
|---|---|---|---|---|
| `atol`, at `rtol = 1e-8` | `1e-8 → 1e-12` | 2.62e-07 → 2.61e-07 | 2.49e-07 → 2.49e-07 | 2.65e-07 → 2.62e-07 |
| `rtol`, at `atol = 1e-10` | `1e-6 → 1e-10` | 3.21e-05 → 2.42e-09 | 2.89e-05 → 2.37e-09 | 3.22e-05 → 3.73e-09 |

Four decades of `atol` move the answer by at most **1.2 %**; four decades of `rtol` move it by a
factor of **13,300**, which is 10.1 per decade. The corners close it: at `rtol = 1e-6`,
`(1e-08, 1e-06)` and `(1e-12, 1e-06)` agree to **three significant figures** on all three models
(3.21e-05 / 2.89e-05 / 3.22e-05, identical in both columns); at `rtol = 1e-10` they agree to 1 % on
Radiation and LambdaCDM, and differ by 3.46e-09 against 5.13e-09 on QCD, which is within a few
times that model's reference drift and is one of the ten unresolved cells.

The magnitude argument behind it, measured directly on `LambdaCDMModel`'s production response grid:
`max|G| = 9.538e+14` and the smallest non-zero sample 2.121e+12 at `k = 1e5`/Mpc;
`max|G| = 2.899e+18` and smallest 6.42e+15 at `3e8`/Mpc. `atol = 1e-10` is therefore between
**1e-22 and 1e-29** of the quantity it bounds. README §2 (e) puts $|G|\sim10^{10}$; the tree is two
to nine orders above that, so the argument is stronger than the README states, not weaker.

### The floor, freshly measured (prompt §5, acceptance 4)

Envelope-relative, on the **source** axis (deviation 1), at a response redshift five e-folds inside
the horizon, per rung of the ladder in e-folds inside the horizon of $z_{\rm source}$:

| model | statistic | +4 | +3 | +2 | +1 | 0 | −2 |
|---|---|---|---|---|---|---|---|
| RadiationModel | max over k | 4.57e-04 | 1.64e-04 | 3.39e-06 | 3.08e-08 | 4.93e-08 | 2.24e-08 |
| RadiationModel | median | 8.26e-05 | 1.92e-06 | 7.69e-08 | 9.35e-10 | 5.14e-10 | 1.08e-09 |
| LambdaCDMModel | max over k | **9.35e-03** | 1.73e-04 | 3.63e-06 | 3.26e-08 | 1.01e-07 | 3.01e-08 |
| LambdaCDMModel | median | 4.88e-03 | 1.14e-04 | 2.17e-06 | 2.00e-08 | 1.16e-08 | 1.23e-08 |
| QCDModel | max over k | **7.79e-03** | 1.86e-04 | 3.61e-06 | 2.20e-06 | 1.96e-07 | 3.90e-08 |
| QCDModel | median | 3.71e-03 | 1.14e-04 | 1.24e-06 | 1.45e-08 | 9.41e-09 | 1.18e-08 |

**Its own uncertainty is 0.00 %**: the worst window at three wavenumbers per model, re-taken at
`(1e-19, 1e-13)` instead of `(1e-18, 1e-12)`, moves in no printed digit (nine of nine cases). On the
control the probe that supplies the spline's nodes is itself **9.8e-13 to 5.6e-12** of the window
envelope from `compute_analytic_G`, so the number measured is the spline's error and not the
probe's.

**The inherited figure, recorded beside it** (§6.1 rule 5). Review §10.1
(`docs/gk-wkb-review-fable-2026-09-09.md` §10.1) puts the consumer's cubic spline of the numeric $G$
at **1e-5 to 1e-4 of the value near the hand-over**, "the larger error by two orders". Measured here
at the hand-over rung, three e-folds inside the horizon: **1.14e-04 to 1.86e-04** of the envelope,
and relative to the value at an antinode within a factor of two of that. **The inherited range is
confirmed at the top of its band, and "two orders" is an understatement**: the dominance is ×631 to
×710 at the hand-over and ×1,760 to ×37,700 at the bottom of the numeric region. It is also
*positional*, and that is new: outside about one e-fold inside the horizon the spline error falls to
1e-08–1e-07 and the solver becomes the larger of the two, by 7 to 12 times.

**No figure anywhere in this prompt is below the freshly measured floor** (prompt §9, README §2 (f)).
The smallest solver error reported at any setting is 2.37e-09 on LambdaCDM at
`(1e-10, 1e-10)`; the floor's *smallest* rung on that model is 1.23e-08 median / 3.01e-08 max. That
is the one place where a solver figure sits below a floor figure — and it is the intended reading,
not an arithmetic error: it says the outer region of the numeric grid is *better* resolved by the
solver than by the consumer's spline, which is exactly why §6.1 rule 3 does not license loosening
(document §0.1, question 3).

### The $z_{\rm source}$ bound (prompt §2.2, acceptance 3) — **it fails**

Three wavenumbers per model, seven source redshifts from −5 to +3 e-folds, production setting, each
against its own converged reference (drift 8.3e-12 to 5.1e-10):

| model | k = 1e5 | k = 1e7 | k = 3e8 |
|---|---|---|---|
| RadiationModel | worst at **+0**, ×1.45 | worst at **+3**, ×1.07 | worst at **+3**, ×1.05 |
| LambdaCDMModel | worst at **−1**, ×1.09 | worst at **+2**, ×1.38 | worst at **−1**, ×1.25 |
| QCDModel | worst at **−1**, ×1.01 | worst at **−1**, ×1.05 | worst at **+0**, ×1.43 |

**The outermost source redshift is not the least favourable at any of the nine probes.** The
maximum is *flat* in $z_{\rm source}$ — 1.09e-07 to 2.39e-07 across the whole table, with no trend —
while the median rises monotonically with e-folds inside the horizon (3.09e-09 → 1.03e-08 at
k = 1e5 on the control) and the evaluation count falls (12,752 → 11,081). Deviation 8 and
`[03-outermost-z-source-is-not-the-least-favourable]`.

### Cost (prompt §6 question 4, acceptance 5)

| model | objects / model | `(1e-10, 1e-7)` | `(1e-10, 1e-8)` | `(1e-10, 1e-9)` |
|---|---|---|---|---|
| RadiationModel | 58,350 | 9,262 (−27.3 %) | **12,744** | 16,700 (+31.0 %) |
| LambdaCDMModel | 29,290 | 9,308 (−27.4 %) | **12,815** | 16,770 (+30.9 %) |
| QCDModel | 38,105 | 9,649 (−27.2 %) | **13,258** | 17,306 (+30.5 %) |

Median right-hand-side evaluations per object over the fifty wavenumbers; times the object count
that is **0.540 / 0.744 / 0.974 ×10⁹** per model on Radiation, **0.273 / 0.375 / 0.491** on
LambdaCDM and **0.368 / 0.505 / 0.659** on QCD. Counts, not wall time (README §2 (i)). The object
count is one per $(k, z_{\rm source})$ with $z_{\rm source}$ outside `z_exit_subh_e4`
(`main.py:1884`), summed over fifty wavenumbers on the version-2 grid.

**No recommendation here moves a sector's total cost at all** — the recommendation is `unchanged` —
so the factor-of-two stop of README §4.3 is not approached.

### Suites and the additive-change acceptance (acceptance 7, 8)

- **`ComputeTargets`: 491 tests, OK, 186 s.** The board header records 484 (note 16, at prompt 02)
  and prompt 02a added seven, which is 491; this prompt adds none. **The wall-clock flake of note 16
  (`test_tk_wkb_phase.TestCost.test_wall_time_per_object`) did not fire on this run** — all 491
  passed.
- **`CosmologyModels`: 39 tests, OK, 0.67 s**, unchanged from the `bc6dc97` re-anchor.
- **`docs/gktk-remedial/tk_numeric_atol_sweep.py` reproduces bit for bit** (acceptance 7). The
  script was run twice on this tree — once with `HEAD`'s `convergence_reference.py` restored in
  place, once with this prompt's — and the two 294-line outputs are **identical except for the two
  lines that print the wall clock about themselves**. `git diff --numstat` on that module is
  **33 insertions, 0 deletions**, so no existing line changed and prompt 01's published figures
  cannot have moved.
- **`black`:** `ComputeTargets/tests/convergence_reference.py` and
  `docs/tolerance-convergence/gk_numeric_sweep.py` are both clean under `--check`. (Repo-wide,
  `black --check .` reports 54 files it would reformat; the identical 54 are reported at `HEAD`
  with this prompt's work stashed, so it is pre-existing and none of it is this prompt's.)

## Observations not acted on

1. **The QCD reference's resolution in this sector is ~3e-10 of the envelope, and that is a
   property of SciPy's `rtol` clamp, not of the tree.** The campaign's reference pair is
   `(1e-18, 1e-12)` and the only effective tightening left is `rtol → 1e-13`
   (`SCIPY_RTOL_FLOOR = 2.22e-14`). On `QCDModel` that step moves the reference by up to 2.98e-10
   of the envelope, against 1.3e-11 to 2.1e-11 on the two smooth models — twenty times worse, and
   it is concentrated at the wavenumbers whose response grid crosses the QCD transition. The
   consequence is that **no measurement of this sector on QCD can resolve a candidate error below
   about 3e-09**, which is where the three tightest matrix cells sit. It costs this prompt nothing
   (the production cell is 399 times above it) and it will cost prompt 04 nothing, but a later
   prompt that wants a decade more resolution on QCD has no tolerance left to buy it with and would
   need a different reference construction. Not opened as an issue: it is a documented property of
   the integrator and the campaign's own convention, recorded here so that the next prompt does not
   rediscover it as a failure.

2. **`gk_geometry`'s docstring states the claim deviation 8 falsifies** — "one source redshift per k
   is taken here, the outermost, which is the longest and therefore the least favourable run"
   (`ComputeTargets/tests/convergence_reference.py`). Correcting it would change an existing line of
   a module this prompt may only add to (§2.3), so it is left alone and recorded in
   `[03-outermost-z-source-is-not-the-least-favourable]` instead.

3. **`main.py:1884`'s comment explains why `z_source` may go as deep as four e-folds inside the
   horizon in terms of the *WKB* region's coverage**, and the measurement here says the numeric
   representation is at its worst exactly there — 9.4e-03 of the envelope on LambdaCDM. The two are
   the same trade seen from opposite ends, and neither the hand-over campaign's issues nor this
   campaign's have it written down as one. Folded into
   `[03-numeric-g-consumer-spline-is-the-dominant-error-near-the-hand-over]` rather than opened
   separately.

4. **README §2 (c) and §7 D1's "~65,000 objects per model" is a version-0 figure.** On the
   version-2 grid at each cosmology's own anchor it is **29,290** on LambdaCDM and **38,105** on
   QCD. The conclusion §2 (c) draws from it is untouched — the sector is still three orders larger
   than `TkNumericIntegration`'s fifty — so this is recorded rather than opened; prompt 06
   reconciles the README, as it already must for §0.1's "only one actually uses it" (board §1).

## State handed to the next prompt

### For prompt 05 and README §7 **D1** — what the user is being asked to accept

**`GkNumericIntegration` keeps `(atol, rtol) = (1e-10, 1e-8)`.** Decoupled, that is
`DEFAULT_GK_NUMERIC_ABS_TOLERANCE = 1e-10` and `DEFAULT_GK_NUMERIC_REL_TOLERANCE = 1e-8`, and the
two halves rest on different kinds of evidence, which the shipped comments must distinguish.

**The five provenance fields (README §1.2, §5 rule 9)** are in
[`GK-NUMERIC-SWEEP.md`](../../../docs/tolerance-convergence/GK-NUMERIC-SWEEP.md) §10.1 and §10.2,
in the shape `docs/TOLERANCE-PROVENANCE.md` will want, and are summarised here:

| | `rtol = 1e-8` | `atol = 1e-10` |
|---|---|---|
| **keys** | `GkNumericIntegration` alone; 29,290 / 38,105 / 58,350 objects per model (LambdaCDM / QCD / Radiation), one per $(k, z_{\rm source})$ outside `z_exit_subh_e4` | the same |
| **chosen by** | this document §5.2, §7: v2 grid at each own anchor, `BREAK_POINT_DISCONTINUITY`, 50 k × 3 models, envelope-relative max **2.60e-07 / 2.48e-07 / 2.62e-07**, reference `(1e-18, 1e-12)` drift **≤2.1e-11** (Rad, LCDM) and **≤3.0e-10** (QCD), converged 50/50 on each | **nothing**: inert across `1e-8 → 1e-12` to within 1.2 %, because $\lvert G\rvert$ is 2.1e+12–2.9e+18 in `Mpc_units`. The comment must say *inert and unchosen*, not invent a justification (README §1.2's closing rule) |
| **competing floor** | the consumer's `numeric_Gk` spline: **1.6e-04 / 1.7e-04 / 1.9e-04** of the envelope 3 e-folds inside the horizon, **4.6e-04 / 9.4e-03 / 7.8e-03** at 4, **1e-08–1e-07** outside 1. Floor uncertainty 0.00 %; confirms review §10.1 | the same, and it never binds |
| **cost** | 12,744 / 12,815 / 13,258 RHS evaluations per object; ×objects = 0.744 / 0.375 / 0.505 ×10⁹ per model. One decade tighter **+31 %**, one looser **−27 %** | none measurable (12,728 → 12,773 across four decades on the control) |
| **citation** | `prompts/tolerance-convergence` prompt 03, this log, 2026-09-17 | the same |

**The decision has no compute cost either way**, which inverts README §7 D1's expectation: D1 was
written believing one decade of `rtol` in this sector was the campaign's whole compute question, and
the measurement says the money should not be spent because the consumer that reads the answer is
three orders coarser than the answer.

### For prompt 03a

- **The axes separate cleanly in this sector**, which is what §3.3a says the $T_k$ re-take should be
  taken knowing: `atol` inert over four decades, `rtol` exactly one decade of error per decade of
  setting, and the corners agreeing to three significant figures. The prior of README §2 (d) is
  **confirmed** in the sector where it had never been tested.
- **The reference pair `(1e-18, 1e-12)` with `(1e-19, 1e-13)` as the tightening converges at all
  fifty wavenumbers on all three models in this sector**, and the candidate axis must stop two
  decades above the reference's `rtol` for the drift to mean anything (deviation 3).
- **`gk_geometry_at_source(cosmology, k_inv_Mpc, grid, z_source)`** is in
  `ComputeTargets/tests/convergence_reference.py` if 03a wants the same second-axis check for
  `TkNumericIntegration` — though that target has only one object per $k$ and no second axis.
- **The outermost source redshift is not a bound** (deviation 8): a `Tk` sweep has no equivalent
  exposure, since `tk_geometry` has no free source redshift, but the wording "least favourable"
  should not be borrowed.

### For prompt 04

- **The floor at the bottom of the numeric region is the source grid's density**, and the density
  criterion that sets it is sized for the phase-residual spline
  (`SOURCE_GRID_CONSUMER_TARGET_RAD`), not for $G$. §8 of the document gives
  `h dtheta/du` on both lattices; **T7** owns where the criterion should apply.
- **`RadiationModel`'s version-2 anchor and grid**, should prompt 04 want the control on the same
  construction: `horizon_exit_z(model, 3e8/Mpc, -5) = 44523947729.772957`, **2306 samples**, digest
  **`3bef2c06`**.

### For prompt 06 and `docs/TOLERANCE-PROVENANCE.md`

Both entries above, verbatim from §10.1 and §10.2 of the measurement document, plus the two §3
issues this prompt opens. Note that §10.2 is the campaign's first instance of README §1.2's
"where the provenance of an existing constant cannot be established, the note says so in those
words rather than inventing one" being the *right* answer for a constant this campaign has just
measured, rather than for one it inherited.

