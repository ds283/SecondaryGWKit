# Prompt 18 — Honour the cosmology's declared discontinuities in the numeric ODE

**Campaign:** [`README.md`](README.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Covers:** §3 `[17-qcd-reference-not-converged]`
**Review sections:** §10.1 (the $G_k$ numeric run's accuracy and cost, and its statement that the
error is set by `rtol`); §12.5. Also `docs/gktk-remedial/RESIDUAL-CONVERGENCE.md` §2–§3 (prompt 02's
measurement of the jumps).
**Design facts:** README §2 (h) — the numeric region is sound and its diagnostic is a live warning;
§2 (d) — the floors are not targets.
**Depends on:** 02 (which built the break-point protocol and measured the jumps), 11 and 16 (which
own the current shape of `numeric_with_phase_cut`), 12, 17 (which found this).
**Recommended model:** Opus — a shared production integrator that both $G_k$ and $T_k$ run through,
and one API decision.
**Files you may touch:** `Quadrature/integrators/numeric_with_phase_cut.py`;
`CosmologyModels/GenericEOS/GenericEOS.py`, `QCD_EOS.py`, `LambdaCDM_GenericEOS.py` (the
declaration, §2.2 — and *only* the declaration); `ComputeTargets/BackgroundModel.py`'s
`_cosmology_break_points` if §2.3 requires it; a new
`ComputeTargets/tests/test_numeric_break_points.py`; extensions to
`docs/gktk-remedial/tk_numeric_atol_sweep.py` and a new additive section of
`docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md`; plus the log, the status board and
`docs/OPEN_ISSUES.md`.
**Do not touch:** `config/defaults.py` — **no tolerance changes in this prompt**; `atol` stays at
`1e-10`, `DEFAULT_TK_NUMERIC_ABS_TOLERANCE` at `1e-13` and `rtol` at `1e-8`, whatever you measure.
The `rtol` question prompt 17 raised is a separate, pipeline-wide decision and is not yours. Also:
the stop window and the $\sqrt{z_{e3}z_{e4}}$ limit, `store()`'s $(B,\delta)$ algebra, the
`GkWKBValue`/`TkWKBValue` schemas, the `GkSource` rectifier, the cumulative-table quadrature path
(§2.2 must not change what the tables split on), and everything README §5 rule 8 lists.

Read first: README §2 (d), (h), §5, §6; `IMPLEMENTATION_STATE.md`'s
`[17-qcd-reference-not-converged]` and `[12-tk-numeric-atol-largest-k-excursion]` entries in §3;
`docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md` §2, §4 and §8; `logs/02-qcd-residual-convergence.md`
and `logs/17-tk-numeric-atol-k-sweep.md` — the "State handed to the next prompt" sections of both.

---

## 1. What is wrong

Prompt 17 could not demonstrate its reference converged on `QCDModel`. At four of the fifty
production wavenumbers — $k \in \{1.58\times10^7,\ 4.97\times10^7,\ 5.86\times10^7,\
2.55\times10^8\}$ — two runs of the same integrator at `(atol, rtol) = (1e-18, 1e-12)` and one
decade tighter differ by $1.6\times10^{-6}$ to $6.2\times10^{-6}$ of the envelope, against a
smallest reported candidate difference of $3.45\times10^{-7}$. Prompt 17 §2.1's criterion is that
the drift be at most a tenth of that, so it fails there by more than an order of magnitude, and at
those wavenumbers the sweep is partly measuring its own reference. Prompt 17 reported anyway, with
the drift carried as a per-$k$ column and conclusions drawn only where the signal stood 57×–175×
above it; the orchestrator stopped on it and the user accepted the mitigation and asked for this
prompt.

**The cause is not the tolerance.** `QCD_Cosmology`'s $H(z)$ genuinely *jumps* — by $4.4\times10^{-4}$
relative at `T_LO` and $1.0\times10^{-4}$ at `T_120_MEV` — because `QCD_EOS`'s $G(T)$ and $G_s(T)$
switch between the Saikawa–Shirai fit and asymptotic constants and the pieces do not join
(`QCD_EOS.py:161`, measured in `RESIDUAL-CONVERGENCE.md` §2). DOP853 estimates its local error from
the difference of two embedded formulas, which assumes the right-hand side is smooth enough over a
step for both Taylor expansions to hold. Across a jump that estimate is meaningless and the method
falls to first order: error $\sim h$ rather than $h^8$, so a decade of tolerance shrinks $h$ by
$10^{1/8}\approx1.33$ and buys about a quarter. Worse, *where the step lands relative to the jump*
changes discontinuously with the tolerance, so the drift is not even monotone — prompt 17 measured
`(1e-19, 1e-13)` moving the reference by $1.03\times10^{-6}$ at $k=10^8$ while `(1e-20, 1e-14)`
moved it by $7.6\times10^{-9}$ — and SciPy clamps `rtol` at $100\varepsilon = 2.22\times10^{-14}$,
so there is nowhere further to go.

**The remedy already exists in the API and is simply not wired to the ODE.** Prompt 02 built a
declaration protocol: `GenericEOS.break_temperatures_GeV` returns `()` by default ("a smooth
equation of state has none"), `QCD_EOS` overrides it, `LambdaCDM_GenericEOS.integration_break_points`
converts to redshifts, and `ComputeTargets/BackgroundModel._cosmology_break_points` duck-types the
consumer side so that a cosmology declaring nothing is treated as smooth. **No consumer knows
anything structural about any equation of state**, which is the property that has to survive this
prompt: a general-purpose calculator must work for any equation of state, and must not hard-code
the implementation details of one. The cumulative tables split their Gauss panels at the declared
points. [`numeric_with_phase_cut`](../../Quadrature/integrators/numeric_with_phase_cut.py) does
not: it makes a single `solve_ivp(..., method="DOP853")` call over the whole range (`:225`), and
both `GkNumericIntegration` and `TkNumericIntegration` run through it.

## 2. What to build

### 2.1 The obstacle: 404 of the 407 declared points are the wrong ones

`integration_break_points` returns **both** kinds of non-smoothness, and deliberately so: the
interior knots of the $T(z)$ spline, where quantities built from $T(z)$ are only $C^2$, **and** the
crossings of the equation of state's branch temperatures, where $H$ jumps. Its docstring says why —
for a fixed-order Gauss–Legendre panel the knots are "the load-bearing half"
(`LambdaCDM_GenericEOS.py:263`). On the production range that is **404 spline knots and 3
temperature crossings** (`ComputeTargets/tests/wkb_reference_data.json`,
`convergence.geometry.QCDModel`).

For an *adaptive* ODE the split is the other way round. A $C^2$ point does not break the embedded
error estimator — the step controller absorbs it, at worst paying a few extra steps — whereas a
*jump* in the right-hand side invalidates it outright. Restarting `solve_ivp` at all 407 points
would pay 408 startup transients per object to fix three, and `TkNumericIntegration` is one object
per $k$ while `GkNumericIntegration` is ~65,000 per model (README §6), so the cost is not
notional.

**So the declaration has to distinguish a jump from a kink, and each consumer must ask for what it
needs.** That is the one design decision in this prompt and it is deliberately generic: the
equation of state is the thing that knows whether its pieces join, so it is the thing that says so.

### 2.2 The declaration

Extend the protocol so that a non-smooth point carries its *kind*. The shape is yours — one
defensible option is a parallel `discontinuity_temperatures_GeV` on `GenericEOS` (default `()`,
documented as a subset of `break_temperatures_GeV`) with a matching `kind=` argument or companion
method on `integration_break_points`; another is to have the existing accessors return
`(value, kind)` pairs. Whichever you pick:

- **`GenericEOS`'s default must remain "smooth"**, so that an equation of state written by someone
  who has never read this campaign gets the current behaviour and no split.
- **The quadrature path must be unchanged.** `CumulativeTable`'s panels still split at *every*
  declared point, jump and kink alike. Prompt 02's convergence result depends on that and
  `test_background_tau.test_qcd_break_points` asserts the count; if that test's expectation has to
  change, you have gone wrong.
- **`QCD_EOS` declares which of its four break temperatures are jumps.** `QCD_EOS.py:161` already
  records the distinction in prose: $G$ and $G_s$ switch pieces at `T_LO`, `T_120_MEV` and `T_HI`
  and do not join continuously, while `w` clamps at `EOS_T_LO`, so $c_s^2$ is continuous there and
  only its slope is not. Confirm each by measurement — evaluate $H$ either side of each declared
  temperature and quote the relative jump — rather than transcribing the docstring.
- Nothing in `Quadrature/` or `ComputeTargets/` may name a temperature, a model or an equation of
  state.

### 2.3 The segmented integration

In `numeric_with_phase_cut`, obtain the declared discontinuities inside
$(z_{\min}, z_{\text{init}})$ and, if there are any, integrate the segments between them in
sequence instead of making one call, carrying the final state of each segment into the next as the
initial condition of the following one. If there are none — every current test stand-in, every
`LambdaCDM` cosmology, `RadiationModel` — take **exactly the present code path**, so that the
existing numbers are reproduced bit for bit.

`_cosmology_break_points` lives in `ComputeTargets/BackgroundModel.py` and `numeric_with_phase_cut`
in `Quadrature/`. Check the import direction before using it; if it would create a cycle, move the
helper somewhere neutral rather than duplicating it, and say so in the log.

What must survive the change, all of it currently load-bearing:

1. **`t_eval`.** The returned samples must still be exactly the requested `z_sample`, in order, and
   the equality guard at `:330` (`"returned sample points that differ from those requested"`) must
   still hold. Distribute `t_eval` across the segments; do not round a break point onto a sample.
2. **`mode="stop"`.** The stop event and its dense-output root-find (`find_phase_extremum`) must
   still work when the extremum falls in any segment — including the case where the event fires
   before the last segment, which must then terminate the whole integration and not merely that
   segment. README §2 (h): the stop point is a maximum and nothing depends on which extremum.
3. **The supervisor's accounting.** `RHS_evaluations`, `compute_steps` (`sol.nfev`), the RHS timing
   statistics, and `has_unresolved_osc` / `unresolved_z` / `unresolved_efolds_subh` must aggregate
   over segments, not report the last one. Preserving the warning is a campaign stop condition
   (§2 (h)).
4. **Failure reporting.** `sol.success` is checked per call; a failure in segment $j$ must name the
   segment and the redshift, not silently return a short solution.

### 2.4 Say why, in the code

The user asked for this explicitly. The module docstring of `numeric_with_phase_cut` (or a comment
block immediately above the segmented driver) must explain, in prose a later reader can act on:

- why an adaptive Runge–Kutta method cannot be trusted across a discontinuity in the right-hand
  side — the embedded error estimator, the order collapse, the non-monotone refinement;
- that the integrator learns about discontinuities only by *asking the cosmology*, that a cosmology
  declaring none is treated as smooth, and that no equation-of-state knowledge lives here;
- **how the convergence test that detects the failure works**: two runs of the same integrator at
  tolerances a decade apart, the difference between them being an estimate of the distance to the
  converged answer, valid only where refinement is monotone — which is exactly what a jump
  destroys. Point at `docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md` §4 for the measurement.

Keep it proportionate to the surrounding file: this is a paragraph or three, not an essay.

## 3. What must be measured

Extend `docs/gktk-remedial/tk_numeric_atol_sweep.py` and add a new top-level section to
`docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md` — **additive**, as §3 of prompt 17 requires: the
existing sections were correct for the tree they were taken on and are not to be rewritten.

1. **The §2.1 convergence test, re-run on `QCDModel` at all 50 wavenumbers.** This is the
   acceptance test of the prompt. Report the drift per $k$ before and after, in the same
   envelope-relative measure, and the four named wavenumbers explicitly.
2. **The same test for $G_k$, on all three models — it has never been done.** Prompt 12's $G_k$
   measurement was candidate-against-candidate (`atol` 1e-10 vs 1e-13) on the radiation control
   alone, and review §10.1 measured $G_k$ on radiation and LambdaCDM against the radiation oracle.
   No converged-reference drift figure exists for $G_k$ on any model, and `GkNumericIntegration`
   runs through the same driver, so the same failure is expected on QCD and must be either
   demonstrated and fixed or shown not to occur. Use `GkNumericIntegration`'s production geometry,
   not $T_k$'s.
3. **A regression, on both smooth models.** `RadiationModel` and `LambdaCDMModel` declare no
   discontinuities, so every number in the existing document must reproduce **exactly** — not
   nearly. Quote at least prompt 17's two control figures (2.534e-6 at $k=10^6$; 2.56e-4 at
   $x=10.78$, $k=3\times10^8$) and confirm the RHS-evaluation counts 7403 and 8483 are unchanged.
4. **The cost, in RHS evaluations** (§5 note 14 — counts, not wall time): per object and summed
   over the grid, QCD only, before and after. Splitting costs startup steps; say how much.
5. **Whether anything else on QCD moves.** The split changes computed values on that model. Say by
   how much, for both sectors, at the production tolerances — this is what tells prompt 13 whether
   a QCD datastore built before this commit is still usable.

### 3.1 The finding you must report and must not act on

`solver_serial` is **not** part of the `GkNumericIntegration` lookup key: the query filters on
`wavenumber_exit_serial`, `model_serial`, `atol_serial` and `rtol_serial` only
(`Datastore/SQL/ObjectFactories/GkNumericIntegration.py:221-227`), and the solver label is stored
but never matched. So rows computed before and after this prompt are **indistinguishable by key on
QCD while holding different values**, exactly the hazard prompt 12's tolerance change avoided by
moving a key. Establish whether the same is true of `TkNumericIntegration`, state it in the log and
open it as a §3 issue. **Do not change the key, the label or the factory** — that is a datastore
decision with its own migration, and it is the user's.

## 4. Verification and acceptance

- **The acceptance test.** After the change, prompt 17 §2.1's criterion holds on `QCDModel` at all
  50 wavenumbers: the reference drift is at most a tenth of the smallest candidate difference the
  sweep reports there, i.e. **$\le3.4\times10^{-8}$** of the envelope. If it does not — if the
  jumps were not the whole cause, or the knots matter after all — **stop and say so with the
  measurement**; do not tighten a tolerance to reach it, and do not split at the 404 knots to reach
  it without saying what it costs and asking.
- The suite passes: `discover -s ComputeTargets/tests -t .`, **299 before, plus the new
  `test_numeric_break_points.py` cases**; none removed, none changed in expectation. In particular
  `test_background_tau.test_qcd_break_points` and `test_cumulative_table.py` are untouched and
  still pass — the quadrature split is not what this prompt changes.
- The new test must not need Ray or a datastore (README §5 rule 7): call the remote through its
  undecorated `_function`, use prompt 01's stand-ins. It must cover at least: a cosmology declaring
  no discontinuities takes the single-call path and reproduces a known result exactly; a stand-in
  declaring a synthetic jump is split at it and beats the unsplit run against an independently
  known answer; `t_eval` equality survives; `mode="stop"` still finds its extremum when the event
  lies in an interior segment.
- `black --check` clean on every file touched.
- `git diff HEAD~1 --stat` touches nothing outside the allowed list — in particular **not**
  `config/defaults.py`.

## 5. Log and commit

Close `[17-qcd-reference-not-converged]` only if the acceptance test of §4 passes; otherwise narrow
it with the measurement. Either way update `docs/OPEN_ISSUES.md` in the same commit, and open the
§3.1 datastore-key issue there as well (the count moves). Update board row 18 and M22.

"State handed to the next prompt": whether prompt 13 may use a QCD datastore built before this
commit, and if not, what has to be regenerated; the $G_k$ drift figures, which are new; the cost in
RHS evaluations on QCD; and the shape the declaration ended up with, since it is now part of the
`CosmologyModels` API that any future equation of state must satisfy.

Commit subject, or something equally specific:
`Split the numeric ODE at the cosmology's declared discontinuities`.
