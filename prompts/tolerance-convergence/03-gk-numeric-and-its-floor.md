# Prompt 03 — `GkNumericIntegration`, and the floor that decides whether it matters

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Establishes:** board item **T4**, and the freshly measured consumer-spline floor **T4** is scored
against · **Informs:** **D1**, the decoupled tolerance pairs — it recommends, it does not decide
**Depends on:** prompts 01, 02 and 02a. Prompt 01's `convergence_reference.py` is the only way this
prompt is allowed to measure; prompt 02's inventory is where its target came from; prompt 02a is why
it can measure QCD at QCD's own anchor.
**Recommended model:** **Opus** — the measurement is mechanical, the design of it is not. This
prompt's result is most likely to be "tighten nothing", and a prompt that cannot tell that outcome
from a failed measurement will report the wrong one.

**Files you may create or touch:**
`docs/tolerance-convergence/GK-NUMERIC-SWEEP.md` — new, the measurement document;
`docs/tolerance-convergence/gk_numeric_sweep.py` — new, the script that regenerates every table in
it;
`ComputeTargets/tests/convergence_reference.py` — **additive only**, see §2.3;
plus this campaign's log, board and `docs/OPEN_ISSUES.md`.

**Do not touch:** `config/defaults.py`; `main.py`; any file under `ComputeTargets/` other than the
additive block of §2.3; any `Datastore/` file; `QuadSourceIntegral.py`, `QuadSource.py`,
`phase_groups.py` or `AdaptiveLevin/` (README §0.4). **Do not change a parameter** (README §5
rule 8) — this prompt reads constants and measures.

**Read first:** README §2 (d), (e), (f) and (i); §3.3's first bullet; **§6.1's target rule in
full** — it is the whole of §6 of this prompt; §5 rules 5, 6 and 9; board items **T4** and
`[01-convergence-block-has-a-separate-generator]`; the module docstring of
`ComputeTargets/tests/convergence_reference.py`; `gk_geometry` (`:817`) and `gk_run` (`:916`).

---

## 1. Why this prompt exists

**The $G_k$ numeric sector has never been swept.** It carries **~65,000 objects per model**
(`main.py:1770-1791`, a version-0 count), which is three orders more than any other sector this
campaign touches, so it is where the compute decision actually lives (README §2 (c)). The only
measurement in the record is review §10.1: two models, four source redshifts, and a step along the
**diagonal** `(1e-10, 1e-8) → (1e-13, 1e-11)`. A diagonal cannot separate `atol` from `rtol`, so
"the error is set by `rtol`" is an *interpretation* of a two-variable step there, not a measurement
(README §2 (d)). `GkTk-remedial` prompt 17 did separate the axes — but in the **other** sector.

**And the expected answer is that nothing should be tightened.** $|G| \sim 10^{10}$ at the
hand-over in `Mpc_units`, so an absolute floor of `1e-10` should never bind (README §2 (e)); and the
consumer's cubic spline of the numeric $G$ is claimed to be "the larger error by two orders" (review
§10.1). If both hold, §6.1 rule 4 applies and the target is **`unchanged`**, written in that word,
with the factor by which the floor dominates. **That is a result and this prompt must be able to
deliver it as one.** The failure mode to guard against is a prompt that treats "no change
recommended" as having found nothing and goes looking for a number to move.

**The floor is inherited from a review paragraph, and §3.3 requires this prompt to re-measure it.**
`1e-5`–`1e-4` of the value near the hand-over is the figure every argument above rests on, and no
one has taken it on this tree. §6.1 rule 1 says measure the dominating floor first; rule 5 says a
prompt may supersede an inherited floor and must record both.

## 2. What is measured, and on what

### 2.1 The sector, the geometry, the grid

- **Error measure:** envelope-relative, README §6's definition, unchanged. Take it through
  `sector_errors("Gk", …)` and `sector_error_measure`; do not write your own.
- **Geometry:** `gk_geometry` — the **response** grid, `winnow(PRODUCTION_RESPONSE_SPARSENESS=12)`
  of the source grid, cut to the source redshift above and to `0.85 z_e6` below, with the
  `(z_e3, z_e6)` stop window. This is `main.py`'s `build_Gk_numeric_work` geometry and you must not
  reconstruct it by hand.
- **Break-point policy:** `BREAK_POINT_DISCONTINUITY`, which for *this* sector is both the module
  default and what the production call site passes. Say so in the document; the reader coming from
  prompt 17 will assume `BREAK_POINT_ALL`, which is the $T_k$ sector's.
- **Grid:** `SOURCE_GRID_V2`, and **name the anchor** — `PRODUCTION_Z_INIT_LAMBDACDM` for
  LambdaCDM, `PRODUCTION_Z_INIT_QCD` for QCD (prompt 02a; board standing note 18). A figure without
  its grid generation is not comparable (§5 rule 6) and, since 02a, one without its anchor is not
  either.
- **Models:** `RadiationModel`, `LambdaCDMModel`, `QCDModel`. **All fifty production wavenumbers**,
  `PRODUCTION_K_GRID_INV_MPC`. §6.1 rule 2 forbids scoring a target at a representative $k$, and
  prompt 17 established that the distribution has a tail three orders above its centre.

### 2.2 The $z_{\rm source}$ axis is a bound, not a sample — and you must check that it is

`GkNumericIntegration` is one object per $(k, z_{\rm source})$, and that second axis is where the
~65,000 comes from. `gk_geometry` takes **one** source redshift per $k$ — the outermost, five
e-folds outside the horizon — on the stated grounds that it is the longest run and therefore the
least favourable.

**Treat that as a claim to be tested, not a sampling decision.** If it holds, the whole sector is
bounded by fifty runs per model and the cost figure of §6.4 is conservative rather than
approximate, which is a stronger result than any sub-sample would give. So:

- take the sweep at the outermost $z_{\rm source}$, as the facility builds it; and
- at **three** wavenumbers spanning the production range, on each model, re-take the production
  setting at a spread of interior source redshifts and show whether the outermost is in fact the
  worst. Report the spread.

If the outermost is **not** the worst anywhere, stop and say so before completing the sweep: the
bound is false, the sector is not characterised by fifty runs, and the shape of this prompt changes.
Do not quietly widen the sweep to compensate.

### 2.3 The one additive change you may make to the facility

§2.2's check needs a geometry at a caller-supplied source redshift, and `gk_geometry` fixes it.
You may **add** a helper to `ComputeTargets/tests/convergence_reference.py` — a new function beside
`gk_geometry`, or an optional keyword that defaults to today's behaviour. **No existing line of that
file may change**, and prompt 01's published figures must reproduce bit-identically afterwards
(prompt 01's own acceptance). If you cannot do it additively, stop and say so rather than editing.

This is not a precedent for editing the facility generally. It is one helper, for one check that
this prompt's central claim rests on.

## 3. The reference, and the rule that no figure travels without its drift

**On `RadiationModel` you are not self-converging: you have the truth.** `analytic_G` and
`analytic_Gprime` are closed forms, and `rho_G == 0` identically in exact radiation, which makes the
Green's-function residual a pure quadrature-error measurement with no reference to build. Score the
radiation column against the anchors through `anchor_error`, and use it to calibrate what the
self-convergence machinery reports on the other two models. This is the check prompt 17 did not have
and the reason its QCD figures were wrong.

**On `LambdaCDMModel` and `QCDModel`**, build the reference with `converged_reference` and obtain
its drift through `reference_drift`, which will not return a number without the verdict attached.
`CRITERION_RATIO = 10.0`. **No error is reported anywhere in this prompt's document or log without
its reference's drift beside it** (§5 rule 5), and no conclusion is drawn from a signal that does
not exceed it.

**If the reference fails to converge at any $(model, k)$, that is a stop.** Continuing past it is
exactly the error `GkTk-remedial` prompt 17 made, and it is the reason this campaign exists. Report
which $(model, k)$ and what the drift was; do not drop the wavenumber and do not loosen the
criterion.

## 4. The matrix, staged

The charter says a matrix in `atol` and `rtol`, and the reason is §2 (d): the record contains one
clean single-axis measurement in the other sector and one diagonal in this one. **A diagonal is not
an acceptable substitute here however tempting the runtime.**

**The anchor** is production: `atol = DEFAULT_ABS_TOLERANCE = 1e-10`,
`rtol = DEFAULT_REL_TOLERANCE = 1e-8`.

**Stage 1 — the axes, separately, at full grid coverage.** Hold one and move the other by decades
through the anchor, all fifty $k$, all three models. Two constraints fix the ends:

- `rtol` is clamped from below by SciPy at `100 eps = 2.220446049250313e-14`
  (`SCIPY_RTOL_FLOOR`, `numeric_with_phase_cut.py:62`). A step across that floor measures nothing,
  and `TolerancePair.rtol_step_is_effective` will tell you so. Do not report one.
- `atol` should be inert in this sector by the magnitude argument of §2 (e). **Test it.** Move it
  both ways far enough that inertness is a measurement — if two decades either side of `1e-10`
  change no error at any $k$ on any model, say that, with the figures.

**Stage 2 — refine, and take the interaction corners.** Around the loosest setting on each axis that
first clears the floor of §5, refine by half-decades if a decade is too coarse to resolve where the
crossing is. Then take the off-axis corners needed to answer whether the axes interact at all: if
`atol` is inert on its own axis, a corner where it is moved *together* with a tightened `rtol` is
what makes "inert" a statement about the matrix rather than about a line through it.

**Report, per $(model, k, atol, rtol)$:** the **maximum**, **second-largest** and **median**
envelope-relative error, **the location of the maximum** ($z$, and where it sits relative to the
hand-over and horizon crossing), and the **evaluation count**. Per $(model, atol, rtol)$: the
distribution over the fifty wavenumbers, not just its centre.

## 5. The floor — measure it, do not inherit it

**The consumer is a specific spline.** `GkSourcePolicyData.py:325-336` selects the `has_numeric`
samples between `numeric_smallest_z` and `z_sample.max`, requires at least
`MIN_SPLINE_DATA_POINTS = 5` of them, builds `make_interp_spline` over them and wraps it in
`ZSplineWrapper` with `log_z=True`. That interpolation error is the floor, and it is a property of
**the response grid's density**, not of `atol` or `rtol` — which is precisely why it can dominate a
solver error by orders and why tightening the solver would buy nothing.

**Method.** With the solver set tight enough that its own error is negligible against what you are
about to measure — demonstrate that, do not assert it — sample $G$ on the response grid as the
policy does, build the consumer's spline exactly as the lines above build it, and score the spline
against the tight solution **between** the grid nodes, which is where the consumer reads it and
where a spline's error lives. Report the floor with its own uncertainty (§6.1 rule 1), per model and
over the fifty wavenumbers, with the same maximum/second/median treatment as §4.

**Record both figures.** The inherited `1e-5`–`1e-4` of the value near the hand-over, with its
citation, and yours beside it (§6.1 rule 5). If they disagree, say so plainly and do not reconcile
them by adjusting your method until they agree.

**A measured accuracy below your own freshly measured floor is a stop** (README §2 (f), §6.1
rule 5). It is an arithmetic error in the measurement and never a discovery.

## 6. The questions this prompt must answer

Answer each in the document, in these words, with the measurement beside it:

1. **Is README §2 (d)'s prior right in the $G_k$ sector?** Is the error set by `rtol`, with `atol`
   inert — and is that true across the matrix, or only along a line?
2. **What is the floor, freshly measured**, and by what factor does it dominate the solver error at
   the production setting `(1e-10, 1e-8)`?
3. **Apply §6.1's target rule.** The target is the **loosest** setting whose **maximum** error over
   the whole grid on all three models is at or below the floor — or it is **`unchanged`**, in that
   word, with the dominating factor recorded (rule 4). Tightest-is-safest is an error of the same
   kind as missing the floor (rule 3).
4. **What does it cost**, at the recommended setting and **one step either side** (§1.2), in
   right-hand-side evaluations **times ~65,000 objects per model**? Counts, never wall time
   (§2 (i)); this machine's elapsed times overstate by up to 53 %.
5. **What would the decoupled pair be**, with the evidence for each half? **Recommend; do not
   decide** — D1 is the user's and prompt 05 may not start until they accept.

## 7. Out of scope, and where each piece went

- **`TkNumericIntegration` and `wavenumber_exit_time`** — board items **T5** and **T6**, and they
  are **prompt 03a's**, not yours. The charter was split on 2026-09-17 so that the sector carrying
  the compute decision gets its own commit and its own review. Do not measure them, do not
  characterise them, and do not recommend for them.
- **Every parameter's value.** Prompt 05, after D1. You recommend with evidence and stop.
- **The source grid, the band, `RESIDUAL_WKB_REGION_MARGIN`, `_solve_horizon_exit` and the grid
  digest.** Prompt 02a's stops stand unchanged; none of them is loosened for this prompt. If the
  sweep appears to need one of them moved, that is a finding for §3 of the board, not a change.
- **`find_phase_extremum`'s `xtol=1e-6, rtol=1e-4`** (`LiouvilleGreen/integration_tools.py:92`),
  which fixes the numeric stop point. It belongs to the hand-over campaign
  (`[11-stop-point-root-tolerance]`) and is not retuned here.

## 8. Acceptance

1. **Coverage.** All fifty production wavenumbers, three models, at the production setting, each
   figure carrying its reference's drift verdict, its grid generation and its anchor.
2. **The matrix, not a diagonal.** Both axes moved independently at full grid coverage, plus the
   interaction corners of §4; the `rtol` floor respected and no step reported across it.
3. **The $z_{\rm source}$ bound checked** at three wavenumbers per model, per §2.2, with the result
   stated either way.
4. **The floor freshly measured**, per model, with its own uncertainty, and the inherited figure
   recorded beside it with its citation.
5. **§6.1's target rule applied**, with the target — or the word `unchanged` and the dominating
   factor — and the cost at that setting and one step either side, in evaluations times objects.
6. **`gk_numeric_sweep.py` regenerates every table** in `GK-NUMERIC-SWEEP.md`. A table in the
   document that the script cannot reproduce is an unfinished prompt.
7. **Prompt 01's published figures are unmoved** and `docs/gktk-remedial/tk_numeric_atol_sweep.py`
   still reproduces, if you touched `convergence_reference.py` at all.
8. **Suites green** at the board header's counts, allowing for `[computetargets-suite-flake]`'s
   wall-clock assertion.
9. **The five provenance fields** (§1.2, §5 rule 9) for anything you recommend, in the log's "State
   handed to the next prompt". A number shipped without them is an unfinished prompt.

## 9. Stop conditions

Stop and report rather than working around any of these:

- **A reported accuracy below the floor you have just measured** (§2 (f), §6.1 rule 5).
- **The reference does not converge** at any $(model, k)$ — prompt 17's error, and not a wavenumber
  to drop.
- **The outermost $z_{\rm source}$ is not the least favourable** anywhere (§2.2). The bound is the
  premise of the whole sweep.
- **Your recommendation would move the sector's total cost by more than a factor of two** (§4.3).
  That is a compute-budget decision for the user, not a measurement outcome.
- **You need to change a parameter, or edit a file outside the list above** — including any
  non-additive change to `convergence_reference.py`.
- **You cannot state the grid generation *and* the anchor** for any figure you intend to publish.

## 10. The log

`logs/03-gk-numeric-and-its-floor.md`, on `GkTk-remedial` §5.1's template, classifying every
deviation. Beyond the template it must carry:

- the drift verdict table, per model, with the radiation anchor calibration beside the
  self-convergence figures;
- the floor, freshly measured, with its uncertainty and the inherited figure it supersedes or
  confirms;
- the answers to §6's five questions, each in one paragraph, with the number in it;
- the $z_{\rm source}$ bound check of §2.2, stated either way;
- and, under **"State handed to the next prompt"**, the five provenance fields of §1.2 for the
  recommended pair, and — for **prompt 05** and **D1** — what the user is being asked to accept,
  in the form a decision can be taken from rather than a summary of the measurement.
