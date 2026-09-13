# Log 18 — Honour the cosmology's declared discontinuities in the numeric ODE

**Prompt:** prompts/GkTk-remedial/18-numeric-ode-break-points.md
**Commit:** *(this commit)* — Split the numeric ODE at the cosmology's declared discontinuities
**Model:** Claude Opus 5
**Date:** 2026-09-13
**Result:** COMPLETE WITH DEVIATIONS — **the §4 acceptance test is missed at 3 of the 50 QCD $T_k$
wavenumbers** (worst drift 1.97e-07 against the 3.4e-08 criterion, median 5.12e-09, down from 23
offenders and 6.17e-06 before). All four wavenumbers §4 names are fixed, by 347x–5764x; $G_k$ is
shown never to have had the failure on any model; the two smooth models reproduce exactly. The
residue is the $T(z)$ spline's $C^2$ knots, which **would** close it (1.97e-07 → 4.65e-09) at a
measured **+219 % / +155 %** of the production right-hand-side evaluations — the decision prompt
§4 reserves for the user.

## What shipped

### The declaration (`CosmologyModels/`)

`CosmologyModels/GenericEOS/GenericEOS.py`

* New module constants `BREAK_POINT_ALL = "all"`, `BREAK_POINT_DISCONTINUITY = "discontinuity"`,
  `BREAK_POINT_KINDS`, with a comment block saying why a quadrature needs one and an adaptive ODE
  the other.
* New `GenericEOSBase.discontinuity_temperatures_GeV` property, default `()` — "a smooth equation
  of state has none", so an equation of state written by someone who has never read this campaign
  gets today's behaviour and no split. Documented as a subset of `break_temperatures_GeV`.

`CosmologyModels/GenericEOS/QCD_EOS.py`

* `QCD_EOS.discontinuity_temperatures_GeV` returns `(T_LO, T_120_MEV, T_HI)` — three of the four
  break temperatures. The docstring carries the **measurement**, not the prose (see Verification).

`CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py:250` → `integration_break_points(z_lo, z_hi,
kind=BREAK_POINT_ALL)`

* `kind=BREAK_POINT_ALL` (the default) is byte-for-byte the old behaviour: the $T(z)$ spline knots
  plus every `break_temperatures_GeV` crossing. **The quadrature path is untouched** —
  `test_background_tau.test_qcd_break_points` and `test_cumulative_table.py` are unedited and pass.
* `kind=BREAK_POINT_DISCONTINUITY` returns only the `discontinuity_temperatures_GeV` crossings —
  no knots.
* An unknown `kind` raises `ValueError`; an equation of state declaring a discontinuity that is
  not also a break raises `RuntimeError` (the quadrature would then be split less finely than the
  ODE, which is backwards).

### The consumer side (`ComputeTargets/BackgroundModel.py`)

* `_cosmology_break_points(cosmology, z_lo, z_hi, kind=BREAK_POINT_ALL)` — one new keyword,
  passed straight through; still duck-typed, so a cosmology with no `integration_break_points` is
  treated as smooth. Every existing caller (the three `CumulativeTable` builds, `phase_residual`,
  two test modules) takes the default and is unchanged.
* The two constants are re-exported here, so that `Quadrature/` can name the kind it wants without
  importing an equation-of-state module. **Nothing in `Quadrature/` or `ComputeTargets/` names a
  temperature, a model or an equation of state.**

### The segmented integration (`Quadrature/integrators/numeric_with_phase_cut.py`)

New module docstring (§2.4 of the prompt): why an adaptive Runge–Kutta cannot be trusted across a
discontinuous right-hand side (embedded estimator, order collapse from eight to one, non-monotone
refinement); that the integrator learns about discontinuities only by asking the cosmology and
that one declaring none is treated as smooth; and how the two-tolerance convergence test that
detected the failure works and why a jump destroys its premise, pointing at
`TK-NUMERIC-ATOL-SWEEP.md` §4 and §9.

New public symbol:

* `declared_discontinuities_in_z(model, z_lo, z_hi) -> List[float]` — the declared *jump*
  redshifts strictly inside the range, descending. `expm1` of the declaration's `u = log(1+z)`,
  which is the lossy direction (`CLAUDE.md`) but is used only as a limit of integration.

New module-private machinery:

* `BREAK_POINT_STANDOFF = 1.0e-12` and `_standoff_boundary(z)` — see deviation 1.
* `_SegmentedDenseOutput` — a `sol(z) -> state` callable assembled from one `OdeSolution` per
  segment, which is the whole protocol `find_phase_extremum` uses, so the stop-mode root-find can
  search a window that straddles a boundary.
* `_SegmentedSolution` — the `t`, `y`, `nfev`, `status`, `sol` a SciPy `OdeResult` would carry.
* `_solve_segmented(...)` — the driver.

`numeric_with_phase_cut` itself gains four lines: `break_z = declared_discontinuities_in_z(model,
z_min, z_init.z)` before the supervisor block, and `if len(break_z) == 0:` around the **unmodified**
`solve_ivp(...)` call, with `_solve_segmented(...)` in the `else`. A cosmology declaring nothing
therefore executes exactly the pre-existing statement.

What `_solve_segmented` preserves, point by point against prompt §2.3:

1. **`t_eval`.** Segment *j* takes the requested samples with `z_end < s <= z_start`; the lowest
   segment also takes one sitting exactly on `z_min`. Each non-final segment additionally asks for
   its own lower boundary as an output point purely to read the state off for the next segment's
   initial condition, and that point is then dropped — so no break point is ever rounded on to a
   sample and the returned grid is exactly the grid requested. The `:330` equality guard is
   untouched and still holds.
2. **`mode="stop"`.** The event list and `dense_output` are passed to every segment. A terminal
   event (`sol.status == 1`) breaks out of the segment loop, so it terminates the whole
   integration and not merely that segment, and `status` is carried to the existing
   `sol.status != 1` check. The dense output is the composite above.
3. **The supervisor's accounting.** One `NumericIntegrationSupervisor` spans every segment, so
   `RHS_evaluations` and the timing statistics aggregate by construction; `compute_steps` sums
   `sol.nfev` over segments. `scan_sample_grid_for_unresolved_osc` runs once, after the loop, on
   the concatenated sample grid, so `has_unresolved_osc` / `unresolved_z` /
   `unresolved_efolds_subh` are unaffected by the split.
4. **Failure reporting.** Each segment's `sol.success` is checked, and a failure raises naming the
   segment index, the segment's redshift range and the redshift reached. A non-final segment that
   does not return the state at its lower boundary raises as well.

One latent crash was fixed in the same function while measuring the all-knots variant: SciPy
leaves `sol.y` as an empty **list** rather than an empty array when a solve with an explicit
`t_eval` returns no output points at all, which happens whenever a terminal event fires inside a
segment above every one of that segment's requested points. `segment_y` is normalised before
slicing. Covered by `test_event_fires_in_a_segment_that_returns_no_output_points`.

### Tests

New `ComputeTargets/tests/test_numeric_break_points.py`, 21 cases, no Ray and no datastore
(`numeric_with_phase_cut._function`, prompt 01's stand-ins):

* `TestDeclaration` — the base default is smooth; `QCD_EOS`'s discontinuities are a proper subset
  of its breaks; **which** temperatures jump is measured from `G`, `Gs` and `w` rather than read
  off the docstring; `H(z)` steps at the two declared crossings in range and is continuous at
  `EOS_T_LO`; `kind=` selects knots-plus-temperatures or temperatures-only; an unknown kind raises.
* `TestSmoothCosmologiesAreUnchanged` — `RadiationModel` and `LambdaCDMModel` declare nothing;
  a LambdaCDM $G_k$ run reproduces a direct `solve_ivp` call with the same arguments **bit for
  bit** at every sample.
* `TestSplitAtDeclaredJump` — a synthetic piecewise-frequency oscillator with a closed-form
  solution: only the declared *jump* reaches the ODE and the declared *kink* does not; the split
  run matches the closed form and converges monotonely under refinement; the returned grid is
  exactly the requested grid, including a sample sitting exactly on the break; the accounting is
  the whole run's, not the last segment's; the unresolved-oscillation flag is unchanged.
* `TestStopModeAcrossSegments` — the event in an interior segment terminates everything; the
  returned stop state matches the closed form through the composite dense output; a degenerate
  segment returning no output points is handled; the extremum found with a window straddling a
  boundary is the one found with a window inside a single segment.
* `TestQCDReferenceConvergence` — the real claim on the real cosmology: at $k = 4.97\times10^7$
  the split run's reference-convergence drift is below the criterion and the unsplit run's is
  above 1e-6.

### The measurement

`docs/gktk-remedial/tk_numeric_atol_sweep.py` gains a **second entry point**, `--break-points`
(`main_break_points`), below a banner; nothing above it is modified. It reuses prompt 17's
stand-ins, geometry, error definition and control, and adds `gk_geometry` / `run_gk` (the
`GkNumericIntegration` production geometry: response grid, one source redshift per $k$, five
e-folds outside the horizon), `_SmoothCosmology` / `_UnsplitModel` (the "before" column measured
*after* the change, by hiding `integration_break_points` — real physics, historic code path),
`break_point_sweep`, and the two follow-up experiments of §9.6 and §9.7.

`docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md` gains **§9**, additive; §§1–8 are untouched.

## Deviations from the prompt

### 1. The segment boundary stands off the declared crossing — `STRUCTURALLY REQUIRED`

**What the prompt assumed.** §2.3: "obtain the declared discontinuities inside
$(z_{\min}, z_{\text{init}})$ and … integrate the segments between them in sequence".

**What was actually there.** Splitting *at* the crossing fixes three of §4's four named
wavenumbers and leaves the fourth exactly where it was: $k = 4.972\times10^7$ stays at
**2.99e-06** against 2.758e-06 unsplit. The cause is that an explicit Runge–Kutta evaluates a
stage at the far end of every step, so a segment that ends exactly on the crossing evaluates its
last stage exactly there — and which branch of `QCD_EOS` answers at that point is decided by the
rounding of the cosmology's own internal `T(z)` lookup, a coin flip. When it lands on the far
branch, the final step of the departing segment is a straddling step again, at the controller's
full step size, and the split buys nothing.

**What was done instead.** The boundary is placed `BREAK_POINT_STANDOFF = 1e-12` relative in
$(1+z)$ on the **near** side — higher $z$, the side the departing segment lives on, these
integrations always running downwards. Measured, §9.6: at $k = 4.972\times10^7$ the drift goes
2.99e-06 → **6.48e-09**; displacing the boundary to the far side instead breaks the two
neighbours symmetrically (4.60e-09 → 5.84e-07 at $k = 1.584\times10^7$, 7.94e-09 → 8.50e-08 at
$5.855\times10^7$), which is what identifies the mechanism rather than merely correlating with it.
A displacement of $10^{-9}$ works equally well, so the constant is not delicate; $10^{-12}$ is four
orders above the cosmology's own evaluation noise and fourteen below the integration range in
$u = \log(1+z)$. The sliver of the far side swept by the arriving segment is $10^{-12}$ of a
range over which the jump is $10^{-4}$, i.e. $10^{-16}$ relative — below the representation floor.

### 2. The synthetic fixture does not demonstrate the accuracy improvement — `STRUCTURALLY REQUIRED`

**What the prompt assumed.** §4: "a stand-in declaring a synthetic jump is split at it and beats
the unsplit run against an independently known answer".

**What was actually there.** It does not, and could not be tuned into doing so. A synthetic
piecewise-frequency oscillator $y'' = -\omega(z)^2 y$ with a closed-form solution was swept over
frequency ratios from $1.0001$ to $10$, grids from 20 to 800 points, and tolerances from
$(10^{-10}, 10^{-8})$ to $(10^{-16}, 10^{-13})$: split and unsplit agree within a factor of two
throughout, and at half the settings the unsplit run is the better of the two. The reason is that
DOP853's controller *detects* a jump of that size — it rejects the straddling step and grinds the
step down until the jump is resolved, which costs steps, not accuracy. The production failure
needs a jump small enough to slip past the controller (1.04e-04 relative in $H$) and consequential
enough to matter once it has, which the synthetic could not be made to exhibit. A first-order
exponential variant was tried too and is worse: its error is dominated by the $e^{30}$
amplification of everything.

**What was done instead.** The synthetic fixture keeps every *structural* obligation — the
segment partition, `t_eval` exactness, `mode="stop"` across segments, the aggregated accounting,
agreement with the closed form, and monotone convergence under refinement — and the accuracy claim
is made where it is real, in `TestQCDReferenceConvergence`, on `QCD_Cosmology` at
$k = 4.97\times10^7$: split 6.5e-09 against unsplit 2.8e-06. The module docstring records the
negative result in full, so nobody repeats the search. Claiming the improvement on a fixture that
does not show it would have been worse than measuring it where it is.

### 3. `integration_break_points` now requires its third parameter — `IMPLEMENTATION CHOICE`

`_cosmology_break_points` always calls `method(z_lo, z_hi, kind=kind)`. A cosmology that
duck-typed the old two-argument signature would now raise `TypeError`. The alternatives were a
`try`/`except TypeError` fallback, or `inspect.signature` introspection. Neither was taken:
`LambdaCDM_GenericEOS` is the only implementation in the tree, the prompt states the declaration
is now part of the `CosmologyModels` API that any future equation of state must satisfy, and a
silent fallback would make a cosmology that *has* discontinuities but an old signature integrate
unsplit without saying so — which is precisely the failure this prompt exists to remove.

### 4. Where the `kind` vocabulary lives — `IMPLEMENTATION CHOICE`

The prompt left the shape open ("a parallel `discontinuity_temperatures_GeV` … with a matching
`kind=` argument or companion method … another is to have the existing accessors return
`(value, kind)` pairs"). Chosen: the parallel property plus a `kind=` keyword, with the two names
as constants in `CosmologyModels/GenericEOS/GenericEOS.py` (the authority on what they mean) and
re-exported from `ComputeTargets/BackgroundModel.py` beside `_cosmology_break_points`.

`(value, kind)` pairs were rejected because every existing caller of `break_temperatures_GeV` and
`integration_break_points` would have had to be changed to unpack them, including the quadrature
path the prompt forbids disturbing. A companion method (`integration_discontinuity_points`) was
rejected because it would duplicate the range handling and the root-solve, and because a future
third kind would then need a third method. The re-export exists so that `Quadrature/` names the
kind without importing an equation-of-state module — prompt §2.2's last bullet.

## Verification performed

Everything below was run; nothing is reasoned. Commands are from the repository root.

### The acceptance test (prompt §4) — **missed at 3 of 50**

`PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/tk_numeric_atol_sweep.py --break-points`,
712 s, Python 3.12.14 / NumPy 2.2.4 / SciPy 1.15.2. Reference `(1e-18, 1e-12)` against
`(1e-19, 1e-13)`, envelope-relative, per $(model, sector, k)$ over all 50 production wavenumbers.

| sector | model | worst drift, split | worst, unsplit | median | $k$ above 3.4e-08 | met? |
|---|---|---|---|---|---|---|
| $T_k$ | RadiationModel | 4.21e-11 | — | 1.92e-11 | 0 | yes |
| $T_k$ | LambdaCDMModel | 5.7e-11 | — | 3.76e-11 | 0 | yes |
| $T_k$ | **QCDModel** | **1.97e-07** @ $k=4.287\times10^6$ | 6.17e-06 | 5.12e-09 | **3** | **no** |
| $G_k$ | RadiationModel | 1.94e-11 | — | 1.35e-11 | 0 | yes |
| $G_k$ | LambdaCDMModel | 2.1e-11 | — | 1.39e-11 | 0 | yes |
| $G_k$ | QCDModel | 8.41e-09 | 8.89e-09 | 2.05e-09 | 0 | yes |

The three QCD $T_k$ wavenumbers still above the criterion are $k = 8.366\times10^5$ (6.08e-08),
$4.287\times10^6$ (1.97e-07) and $4.223\times10^7$ (3.46e-08) — 1.8x to 5.8x the criterion,
against 23 wavenumbers and up to 180x before. Each is a genuine non-convergence and not noise in
the drift estimate: repeating with `(1e-20, 1e-14)`, `(1e-17, 1e-11)` and `(3e-19, 3e-13)` gives
5.57e-08 / 6.16e-08 / 5.66e-08, 1.96e-07 / 1.84e-07 / 2.03e-07 and 3.51e-08 / 1.91e-08 / 3.49e-08.

### §4's four named wavenumbers — all fixed

| $k$ [1/Mpc] | drift before | after | improvement |
|---|---|---|---|
| 1.584e+07 | 1.60e-06 | 4.60e-09 | 347x |
| 4.972e+07 | 2.76e-06 | 6.48e-09 | 426x |
| 5.855e+07 | 6.17e-06 | 7.94e-09 | 777x |
| 2.548e+08 | 5.29e-06 | 9.18e-10 | 5764x |

### $G_k$ (prompt §3 item 2) — the failure does not occur

No converged-reference drift figure for `GkNumericIntegration` existed on any model. It does now,
for all three, at all 50 wavenumbers: 1.94e-11 (Radiation), 2.1e-11 (LambdaCDM), 8.41e-09 (QCD),
none above the criterion. On QCD the split and unsplit columns agree at essentially every
wavenumber (8.41e-09 against 8.89e-09 at the worst), so the same failure was **shown not to
occur** for $G_k$ rather than demonstrated and fixed. That is consistent with review §12.5's
statement that `atol` never binds for $G_k$ because $|G|$ is enormous in these units: the
Green's-function run's error is set by `rtol` on a quantity that does not decay, so a 1.04e-04
perturbation of one step is not visible against it.

### The regression on the smooth models (prompt §3 item 3) — exact

* Prompt 12's control, against the exact $T$: **2.534e-06** at $x = 28.22$ in **7403** RHS
  evaluations ($k = 10^6$), and **2.56e-04** at $x = 10.78$ in **8483** ($k = 3\times10^8$) — the
  same figures and the same two evaluation counts §3 of the document reports, 0.02 % and 0.01 %
  from prompt 12's.
* §4's reference-convergence row reproduces to the digit: worst drift **4.21e-11 at $k = 3\times10^8$**
  (Radiation) and **5.7e-11 at $k = 1.561\times10^8$** (LambdaCDM), medians 1.92e-11 and 3.76e-11.
* Both models report **1 segment** at every wavenumber, which is the single-`solve_ivp` statement.
* `test_lambdacdm_reproduces_a_direct_solve_ivp_call_exactly` asserts bit-for-bit equality of
  every returned value and derivative against an independently issued `solve_ivp` call.

### Cost (prompt §3 item 4), QCD only, production tolerances

| sector | per object, unsplit | per object, split | grid total, unsplit | split | change |
|---|---|---|---|---|---|
| $T_k$ | 9709 | 9843 | 485467 | 492158 | **+1.38 %** |
| $G_k$ | 13268 | 13320 | 663403 | 666011 | **+0.39 %** |

One extra restart costs about 130 right-hand-side evaluations in the $T_k$ sector and about 50 in
the $G_k$ sector, against ~10k and ~13k per object.

### What else moves on QCD (prompt §3 item 5)

Split against unsplit at the production tolerances (`atol` 1e-13 for $T_k$, 1e-10 for $G_k$,
`rtol` 1e-8), envelope-relative:

| sector | worst shift | at $k$ | median | smallest | median solver error at the same tolerance |
|---|---|---|---|---|---|
| $T_k$ | **2.82e-04** | 1.584e+07 | 8.53e-07 | 4.52e-08 | 8.35e-07 |
| $G_k$ | **1.77e-06** | 2.197e+07 | 1.92e-07 | 1.96e-09 | 9.37e-07 |

**A `QCDModel` datastore built before this commit is not reusable.** The worst $T_k$ shift,
2.82e-04 of the envelope, is two orders above README §6's 3e-06 row and 300x the median solver
error at the same tolerance; the $G_k$ shift is smaller but still above the solver error at 20
wavenumbers. `LambdaCDMModel` and `RadiationModel` are bit-identical and their rows are unaffected.

### The declaration, measured rather than transcribed (prompt §2.2)

`G`, `Gs` and `w` evaluated at $T(1 \pm 10^{-12})$ across each of `QCD_EOS`'s four break
temperatures:

| temperature | $|\Delta G|/G$ | $|\Delta G_s|/G_s$ | $|\Delta w|/w$ | verdict |
|---|---|---|---|---|---|
| `T_LO` = 1e-5 GeV | 8.876e-04 | 2.284e-03 | 0 (clamped) | **jump** |
| `EOS_T_LO` = 2e-3 GeV | 1.746e-14 | 1.745e-14 | 1.331e-15 | continuous |
| `T_120_MEV` = 0.12 GeV | 2.075e-04 | 3.744e-04 | 7.457e-04 | **jump** |
| `T_HI` = 1e16 GeV | 1.454e-02 | 1.395e-02 | 2.340e-03 | **jump** |

At `EOS_T_LO` the differences scale linearly with the probe separation (1.75e-12 at
$10^{-10}$), which is what identifies them as rounding of a continuous function rather than a
step. One level up, in the quantity the right-hand side reads: $H(z)$ steps by **4.437e-04** at
$z = 4.25344\times10^{7}$ (`T_LO`) and **1.038e-04** at $z = 8.64355\times10^{11}$ (`T_120_MEV`),
and by 9.27e-11 at `EOS_T_LO`'s $z = 1.18721\times10^{10}$ — i.e. not at all. `T_HI` does not
cross inside the production redshift range. This confirms `QCD_EOS.py:161`'s prose and
`RESIDUAL-CONVERGENCE.md` §2's 4.4e-4 / 1.0e-4 exactly. `T_120_MEV` is the only declared
discontinuity inside any production numeric range, so every QCD object is split into **2
segments**.

### The suite

```
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .
```

299 before, **320 after** (21 new cases in `test_numeric_break_points.py`), none removed and none
changed in expectation. The one failure is the known, machine-load-sensitive
`test_tk_wkb_phase.TestCost.test_wall_time_per_object`, which measures a best-of-N cold wall time
against a hard-coded 0.06 s limit and came in at 0.0681 s; it fails on the clean tree at `4e7b3fc`
in 2 of 3 runs, is unrelated to this prompt, is not in the prompt's file list and was left alone. `test_background_tau.test_qcd_break_points` and `test_cumulative_table.py`
are unedited and pass — the quadrature split is not what this prompt changed.
`test_numeric_phase_cut.TestBitIdentity`, whose constants were captured before this prompt on
`RadiationModel`, passes unchanged, which is a second bit-identity statement about the
no-declaration path.

`./venv/bin/python -m black --check` is clean on every file touched.
`git diff HEAD~1 --stat` touches nothing outside the prompt's list; in particular **not**
`config/defaults.py` — `atol` is still 1e-10, `DEFAULT_TK_NUMERIC_ABS_TOLERANCE` still 1e-13 and
`rtol` still 1e-8.

## Observations not acted on

### The $C^2$ spline knots would close the acceptance gap, at 2.5x–3.2x the cost (§9.7)

Prompt §4 forbids splitting at the 404 knots "to reach it without saying what it costs and
asking". Measured, so that the user can decide:

| $k$ [1/Mpc] | drift, jumps only | drift, jumps + knots |
|---|---|---|
| 8.366e+05 | 6.08e-08 | **6.89e-10** |
| 4.287e+06 | 1.97e-07 | **4.65e-09** |
| 4.223e+07 | 3.46e-08 | **2.30e-09** |

All three fall below the 3.4e-08 criterion, so splitting at every declared break point would make
the acceptance test pass at all 50 wavenumbers. The cost, at the **production** tolerances over
every fifth wavenumber of the grid:

| sector | jumps only | jumps + knots | change |
|---|---|---|---|
| $T_k$ | 98,389 | 313,698 | **+218.8 %** |
| $G_k$ | 132,874 | 338,526 | **+154.8 %** |

That is 2.5x to 3.2x the right-hand-side evaluations of every QCD numeric object — and
`GkNumericIntegration` is one object per $(k, z_{\rm source})$, ~65,000 per model. **The question
for the user:** is closing a 1.97e-07 residual at three of fifty wavenumbers worth tripling the
cost of the QCD numeric stage in both sectors? No code change was made either way; the machinery
to do it is one argument (`kind=BREAK_POINT_ALL` in `declared_discontinuities_in_z`).

Note that an earlier measurement of the same comparison, taken *before* the standoff of deviation 1
was added, showed the all-knots split making $k = 4.972\times10^7$ **worse** (7.06e-06). That was
the boundary-placement effect, not the knots; with the standoff it is 6.48e-09 either way.

### `solver_serial` is not part of either numeric lookup key (prompt §3.1) — reported, not acted on

Confirmed for both sectors by reading the factories:

* `Datastore/SQL/ObjectFactories/GkNumericIntegration.py:221-227` filters on `validated`,
  `wavenumber_exit_serial`, `model_serial`, `atol_serial`, `rtol_serial` (and `z_source_serial`
  when supplied). `solver_serial` is selected (`:204`) and joined for its label, never matched.
* `Datastore/SQL/ObjectFactories/TkNumericIntegration.py:225-231` is **the same**: the same five
  filters (and `z_init_serial`), with `solver_serial` selected at `:209`, written at `:417` and
  joined at `:220` — never matched.

So rows computed before and after this commit are indistinguishable by key on `QCDModel` while
holding different values, which §9.4 measures at up to 2.82e-04 of the envelope. This is the
hazard prompt 12's tolerance change avoided by moving a key. **Not acted on:** the key, the label
and the factories are untouched, per §3.1. Opened as `[18-numeric-solver-not-in-lookup-key]`.

### The solver label is unchanged

`"solve_ivp+DOP853-stepping0"` is returned whether or not the run was split. Distinguishing them
would be the natural way to make the datastore hazard above visible, but the label is the string
`main.py` registers and `store()` looks up (`GkNumericIntegration.py:489`-style), so changing it
is a datastore decision with its own migration — the same decision §3.1 reserves. Recorded in the
same issue.

### The two smooth models' $G_k$ drift is now on record

1.94e-11 (Radiation) and 2.1e-11 (LambdaCDM) worst over the grid. Review §10.1 measured $G_k$
against the radiation oracle, not against a converged run of itself; these are the first
convergence figures for that sector and are three orders below the smallest candidate difference
any sweep reports. Nothing follows from them here; prompt 13 may use them.

## State handed to the next prompt

**Prompt 13 may not use a `QCDModel` datastore built before this commit.** The split changes
computed values on that model in both numeric sectors, and `solver_serial` is not part of either
lookup key, so pre-split and post-split rows are indistinguishable by key. What has to be
regenerated: every `TkNumericIntegration` and `GkNumericIntegration` row on `QCDModel`, and
everything downstream of them — `TkWKBIntegration` and `GkWKBIntegration` take their initial data
from the numeric stop point, so the whole QCD chain follows. `LambdaCDMModel` and `RadiationModel`
rows are **bit-identical** and need no regeneration: those cosmologies declare no discontinuities
and take the single-`solve_ivp` path unchanged. Nothing in the schema or the payload keys moved.

**The $G_k$ drift figures, which are new.** Worst reference-convergence drift over the 50
production wavenumbers, `(1e-18, 1e-12)` against `(1e-19, 1e-13)`, envelope-relative, on
`GkNumericIntegration`'s own production geometry: **1.94e-11** (RadiationModel), **2.1e-11**
(LambdaCDMModel), **8.41e-09** (QCDModel); medians 1.35e-11, 1.39e-11, 2.05e-09; none above the
3.4e-08 criterion, before or after the split. The QCD failure that motivated this prompt is a
$T_k$ phenomenon only.

**The cost, in RHS evaluations, on QCD.** Per object at the production tolerances: $T_k$
9709 → **9843** (+1.38 %), $G_k$ 13268 → **13320** (+0.39 %); grid totals 485,467 → 492,158 and
663,403 → 666,011. One restart, because `T_120_MEV` is the only declared discontinuity inside any
production numeric range; `T_LO`'s crossing at $z = 4.25\times10^7$ is below every numeric grid and
`T_HI` is above the model's range.

**The shape the declaration ended up with**, now part of the `CosmologyModels` API:

```python
# CosmologyModels/GenericEOS/GenericEOS.py
BREAK_POINT_ALL = "all"                       # every non-smooth point: what a Gauss panel needs
BREAK_POINT_DISCONTINUITY = "discontinuity"   # the subset that jumps: what an adaptive ODE needs
BREAK_POINT_KINDS = (BREAK_POINT_ALL, BREAK_POINT_DISCONTINUITY)

class GenericEOSBase:
    @property
    def break_temperatures_GeV(self) -> tuple: ...          # unchanged, default ()
    @property
    def discontinuity_temperatures_GeV(self) -> tuple:      # NEW, default ()
        """A subset of break_temperatures_GeV: where the pieces do not join."""

# CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py
def integration_break_points(self, z_lo, z_hi, kind=BREAK_POINT_ALL) -> np.ndarray: ...

# ComputeTargets/BackgroundModel.py  (re-exports the two constants)
def _cosmology_break_points(cosmology, z_lo, z_hi, kind=BREAK_POINT_ALL) -> np.ndarray: ...

# Quadrature/integrators/numeric_with_phase_cut.py
def declared_discontinuities_in_z(model, z_lo, z_hi) -> List[float]: ...
BREAK_POINT_STANDOFF = 1.0e-12   # relative in (1+z), on the near side of each crossing
```

Any future equation of state must satisfy it by doing nothing: both properties default to `()`,
and a cosmology that declares nothing integrates in one call exactly as before. The one
non-obvious obligation is on a future *consumer*: a segment boundary must stand off the crossing
on the side the departing integration lives on, or an explicit Runge–Kutta's final stage evaluates
the far branch and the split buys nothing — deviation 1, measured in §9.6.

**`[17-qcd-reference-not-converged]` is narrowed, not closed.** 23 QCD $T_k$ wavenumbers above the
criterion become 3, worst 6.17e-06 becomes 1.97e-07, and $G_k$ is shown never to have had the
failure. What remains is the $T(z)$ spline's $C^2$ knots, which would close it at +219 % / +155 %
of the production evaluations — the user's call, per prompt §4.

**`[18-numeric-solver-not-in-lookup-key]` is opened**: `solver_serial` is stored but not matched
by either numeric factory's query, so a change of solver behaviour is invisible to the datastore.
